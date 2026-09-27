"""Fourier-diffraction sampling of complex apertures (Fraunhofer model).

Fixed contract (see README.md for the full specification):

* Input aperture ``A`` is an N x N square complex-amplitude array, given in a
  *centered* layout, 1 <= N <= 32, sample pitch ``dx > 0``.
* Far field (Fraunhofer), forward DFT convention

      S[k, l] = sum_{m,n} a[m,n] exp(-i 2 pi (k m + l n) / M)

  with the physical field

      U = dx**2 / (wavelength * z) * S

  and observation-plane sample pitch ``du = wavelength * z / (M * dx)``.
* Optional centered zero padding to M x M, N <= M <= 64.  The observation
  pitch is always recomputed from M (padding changes the physical pitch).
* Even-grid ``fftshift`` convention: shifted physical coordinate of index p is

      x[p] = (p - M // 2) * du

  so index ``M // 2`` is exactly at coordinate 0 and the negative extent is
  one sample larger than the positive extent on even grids.

No third-party numerical library is used; the DFT is a self-written
mixed-radix Cooley-Tukey transform with an O(n**2) fallback for prime lengths.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

# Hard interface bounds fixed by the task.
MAX_APERTURE = 32
MAX_PADDED = 64


@dataclass(frozen=True)
class PixelEnergyResult:
    """Adaptive area-integrated detector-pixel energies.

    Attributes
    ----------
    energy:
        M x M integrated energy
        ``integral integral |U(x, y)|**2 dx dy`` over each detector pixel.
    relative_error:
        Latest accepted relative change for each pixel.  The absolute error
        estimate is ``relative_error[i][j] * max(|energy_new|,
        |energy_previous|)``.
    subdivisions:
        Number of midpoint subdivisions per axis used for each final estimate.
    converged:
        Per-pixel convergence flag.
    all_converged:
        True only when every pixel met the requested tolerance.
    tolerance:
        Requested relative tolerance.
    absolute_tolerance:
        Requested absolute energy tolerance.
    min_subdivisions, max_subdivisions:
        Refinement bounds used by the integrator.
    """

    energy: list[list[float]]
    relative_error: list[list[float]]
    subdivisions: list[list[int]]
    converged: list[list[bool]]
    all_converged: bool
    tolerance: float
    absolute_tolerance: float
    min_subdivisions: int
    max_subdivisions: int


@dataclass(frozen=True)
class RegionEnergyResult:
    """Adaptive integral over a rectangular union of detector pixels."""

    energy: float
    relative_error: float
    subdivisions: int
    converged: bool
    pixel_energy_sum: float
    consistency_error: float
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    tolerance: float
    absolute_tolerance: float
    min_subdivisions: int
    max_subdivisions: int


@dataclass(frozen=True)
class FraunhoferResult:
    """Far-field result.

    Attributes
    ----------
    intensity:
        ``|U|**2`` as a list of M lists of M floats, in fftshifted (physical,
        centered) layout: ``intensity[i][j]`` belongs to coordinate
        ``(x[i], y[j])``.
    field:
        Shifted complex field ``U`` with the same layout.
    x, y:
        Physical coordinates (metres) of axis-0 / axis-1 samples, length M.
    du:
        Observation-plane sample pitch in metres.
    center:
        Index ``(M // 2, M // 2)`` of the zero coordinate (DC pixel).
    input_power:
        Independently meaningful incident power proxy ``dx**2 * sum |A|**2``.
    output_power:
        ``du**2 * sum |U|**2``; Parseval conservation makes it equal to
        ``input_power`` up to floating-point round-off.
    power_error:
        Signed *relative* power discrepancy
        ``(output_power - input_power) / input_power``.  It is ``0.0`` for a
        strictly zero input (where a ratio is undefined).
    n:
        Input aperture side length.
    m:
        (Possibly padded) transform side length.
    aperture:
        Parsed centered input aperture, retained for continuous-area
        integration.
    dx, wavelength, z:
        Physical propagation parameters used to generate this result.
    """

    intensity: list[list[float]]
    field: list[list[complex]]
    x: list[float]
    y: list[float]
    du: float
    center: tuple[int, int]
    input_power: float
    output_power: float
    power_error: float
    n: int
    m: int
    aperture: list[list[complex]]
    dx: float
    wavelength: float
    z: float


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _strict_int(value, name: str, *, lo: int | None = None, hi: int | None = None) -> int:
    """Reject booleans and non-ints; bool must not silently count as int."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}: {value!r}")
    if lo is not None and value < lo:
        raise ValueError(f"{name} must be >= {lo}, got {value}")
    if hi is not None and value > hi:
        raise ValueError(f"{name} must be <= {hi}, got {value}")
    return value


def _finite_real(value, name: str, *, positive: bool = False, nonneg: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}: {value!r}")
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and f <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    if nonneg and f < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return f


def _as_complex(value, where: str) -> complex:
    if isinstance(value, bool):
        raise TypeError(f"{where} must be a complex amplitude, got bool")
    if isinstance(value, complex):
        c = value
    elif isinstance(value, (int, float)):
        c = complex(value)
    else:
        raise TypeError(f"{where} must be a numeric complex amplitude, got {type(value).__name__}")
    if math.isnan(c.real) or math.isnan(c.imag) or math.isinf(c.real) or math.isinf(c.imag):
        raise ValueError(f"{where} must be finite, got {value!r}")
    return c


def _parse_aperture(aperture) -> list[list[complex]]:
    if isinstance(aperture, (str, bytes)) or not hasattr(aperture, "__iter__"):
        raise TypeError("aperture must be a 2-D array-like of complex amplitudes")
    rows: list[list[complex]] = []
    width: int | None = None
    for i, row in enumerate(aperture):
        if isinstance(row, (str, bytes)) or not hasattr(row, "__iter__"):
            raise TypeError(f"aperture[{i}] must be an iterable row of complex amplitudes")
        parsed: list[complex] = []
        for j, v in enumerate(row):
            parsed.append(_as_complex(v, f"aperture[{i}][{j}]"))
        if not parsed:
            raise ValueError(f"aperture[{i}] is empty")
        if width is None:
            width = len(parsed)
        elif len(parsed) != width:
            raise ValueError(
                f"aperture is not rectangular: row 0 has {width} elements, "
                f"row {i} has {len(parsed)}"
            )
        rows.append(parsed)
    if not rows:
        raise ValueError("aperture must contain at least one row")
    n = len(rows)
    if n != width:
        raise ValueError(f"aperture must be square (N x N), got shape {n} x {width}")
    if n > MAX_APERTURE:
        raise ValueError(f"aperture side length must be <= {MAX_APERTURE}, got {n}")
    return rows


# ---------------------------------------------------------------------------
# Self-written DFT (no FFT package)
# ---------------------------------------------------------------------------


def _smallest_prime_factor(n: int) -> int:
    if n % 2 == 0:
        return 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return d
        d += 2
    return n  # n is prime


def _dft1(a: list[complex], sign: int) -> list[complex]:
    """1-D DFT.  sign = -1 forward (e^{-2pi i}), +1 inverse sign."""
    n = len(a)
    if n == 1:
        return [a[0]]
    p = _smallest_prime_factor(n)
    if p == n:
        # Prime length: direct O(n**2) sum with precomputed roots.
        roots = [cmath.exp(sign * 2j * math.pi * k / n) for k in range(n)]
        out: list[complex] = []
        for k in range(n):
            s = 0j
            for m, v in enumerate(a):
                s += v * roots[(k * m) % n]
            out.append(s)
        return out
    # General Cooley-Tukey: n = p * q, decimate a[r + j*p], j = 0..q-1.
    q = n // p
    parts = [_dft1([a[r + j * p] for j in range(q)], sign) for r in range(p)]
    roots = [cmath.exp(sign * 2j * math.pi * k / n) for k in range(n)]
    out = [0j] * n
    for k in range(n):
        s = 0j
        kq = k % q
        for r in range(p):
            s += roots[(k * r) % n] * parts[r][kq]
        out[k] = s
    return out


def _dft2(a: list[list[complex]]) -> list[list[complex]]:
    m = len(a)
    transformed_rows = [_dft1(row, -1) for row in a]
    out = [[0j] * m for _ in range(m)]
    for j in range(m):
        col = [transformed_rows[i][j] for i in range(m)]
        spec = _dft1(col, -1)
        for i in range(m):
            out[i][j] = spec[i]
    return out


def _validate_refinement(
    min_subdivisions: int,
    max_subdivisions: int,
    *,
    name: str = "subdivisions",
) -> tuple[int, int]:
    lo = _strict_int(min_subdivisions, f"min_{name}", lo=1, hi=64)
    hi = _strict_int(max_subdivisions, f"max_{name}", lo=1, hi=64)
    if lo & (lo - 1):
        raise ValueError(f"min_{name} must be a power of two, got {lo}")
    if hi & (hi - 1):
        raise ValueError(f"max_{name} must be a power of two, got {hi}")
    if hi < lo:
        raise ValueError(
            f"max_{name} must be >= min_{name}, got {hi} < {lo}"
        )
    return lo, hi


def _pixel_midpoint_energies(
    aperture: list[list[complex]],
    transform_size: int,
    dx: float,
    wavelength: float,
    z: float,
    subdivisions: int,
) -> list[list[float]]:
    """Composite midpoint energy for all detector pixels at one refinement."""
    n = len(aperture)
    m = transform_size
    q = subdivisions
    # A 2*q*m transform gives output samples on a half-pixel-spaced grid, so
    # every pixel's q midpoint points can be selected without phase shifting.
    fine_m = 2 * q * m
    source = [[0j] * fine_m for _ in range(fine_m)]
    for i in range(n):
        si = (i - n // 2) % fine_m
        for j in range(n):
            sj = (j - n // 2) % fine_m
            source[si][sj] = aperture[i][j]

    spectrum = _dft2(source)
    field_scale = dx * dx / (wavelength * z)
    du = wavelength * z / (m * dx)
    cell_area = (du / q) ** 2
    energies = [[0.0] * m for _ in range(m)]

    for i in range(m):
        for j in range(m):
            total = 0.0
            for ti in range(q):
                ri = (
                    2 * q * (i - m // 2) + 2 * ti - q + 1
                ) % fine_m
                for tj in range(q):
                    rj = (
                        2 * q * (j - m // 2) + 2 * tj - q + 1
                    ) % fine_m
                    value = field_scale * spectrum[ri][rj]
                    total += value.real * value.real + value.imag * value.imag
            energies[i][j] = total * cell_area
    return energies


def _relative_change(previous: list[list[float]], current: list[list[float]]) -> list[list[float]]:
    out = [[0.0] * len(current) for _ in current]
    for i, row in enumerate(current):
        for j, value in enumerate(row):
            scale = max(abs(value), abs(previous[i][j]))
            out[i][j] = abs(value - previous[i][j]) / scale if scale else 0.0
    return out


def integrate_pixel_energies(
    result: FraunhoferResult,
    tolerance: float = 1e-8,
    absolute_tolerance: float = 0.0,
    min_subdivisions: int = 4,
    max_subdivisions: int = 32,
) -> PixelEnergyResult:
    """Integrate ``|U|**2`` over finite detector-pixel areas.

    The composite 2-D midpoint rule is refined as 1, 2, 4, ... samples per
    pixel axis.  A pixel is accepted only after two successive refinements
    satisfy both the relative and absolute error estimates; the first checked
    level is ``min_subdivisions``.
    """
    if not isinstance(result, FraunhoferResult):
        raise TypeError(
            "result must be a FraunhoferResult, got "
            f"{type(result).__name__}"
        )
    tolerance = _finite_real(tolerance, "tolerance", nonneg=True)
    absolute_tolerance = _finite_real(
        absolute_tolerance, "absolute_tolerance", nonneg=True
    )
    lo, hi = _validate_refinement(min_subdivisions, max_subdivisions)

    m = result.m
    previous = [[0.0] * m for _ in range(m)]
    current = [[0.0] * m for _ in range(m)]
    errors = [[0.0] * m for _ in range(m)]
    last_errors = [[float("inf")] * m for _ in range(m)]
    used = [[0] * m for _ in range(m)]
    accepted = [[False] * m for _ in range(m)]
    stable = [[False] * m for _ in range(m)]
    q = 1

    while q <= hi:
        current = _pixel_midpoint_energies(
            result.aperture,
            m,
            result.dx,
            result.wavelength,
            result.z,
            q,
        )
        if q >= lo:
            new_errors = _relative_change(previous, current)
            for i in range(m):
                for j in range(m):
                    absolute_change = abs(current[i][j] - previous[i][j])
                    passes = (
                        new_errors[i][j] <= tolerance
                        and absolute_change <= absolute_tolerance
                    ) if absolute_tolerance > 0.0 else new_errors[i][j] <= tolerance
                    if passes and stable[i][j]:
                        accepted[i][j] = True
                        errors[i][j] = new_errors[i][j]
                        used[i][j] = q
                    stable[i][j] = passes
                    last_errors[i][j] = new_errors[i][j]
        previous = current
        if all(all(row) for row in accepted):
            break
        q *= 2

    for i in range(m):
        for j in range(m):
            if not accepted[i][j]:
                errors[i][j] = last_errors[i][j]
                used[i][j] = hi

    return PixelEnergyResult(
        energy=current,
        relative_error=errors,
        subdivisions=used,
        converged=accepted,
        all_converged=all(all(row) for row in accepted),
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        min_subdivisions=lo,
        max_subdivisions=hi,
    )


def integrate_region_energy(
    result: FraunhoferResult,
    row_start: int,
    row_end: int,
    col_start: int,
    col_end: int,
    tolerance: float = 1e-8,
    absolute_tolerance: float = 0.0,
    min_subdivisions: int = 4,
    max_subdivisions: int = 16,
) -> RegionEnergyResult:
    """Integrate one rectangular, half-open union ``[r0, r1) x [c0, c1)``.

    The region estimate uses the same physical midpoint samples as the pixel
    integrator.  ``pixel_energy_sum`` independently adds the selected final
    per-pixel values, making detector tiling checkable against the integral
    over the combined area.
    """
    if not isinstance(result, FraunhoferResult):
        raise TypeError(
            "result must be a FraunhoferResult, got "
            f"{type(result).__name__}"
        )
    row_start = _strict_int(row_start, "row_start", lo=0, hi=result.m)
    row_end = _strict_int(row_end, "row_end", lo=0, hi=result.m)
    col_start = _strict_int(col_start, "col_start", lo=0, hi=result.m)
    col_end = _strict_int(col_end, "col_end", lo=0, hi=result.m)
    if row_end <= row_start or col_end <= col_start:
        raise ValueError("region must contain at least one detector pixel")
    tolerance = _finite_real(tolerance, "tolerance", nonneg=True)
    absolute_tolerance = _finite_real(
        absolute_tolerance, "absolute_tolerance", nonneg=True
    )
    lo, hi = _validate_refinement(min_subdivisions, max_subdivisions)

    previous_total = 0.0
    current_total = 0.0
    final_grid: list[list[float]] = []
    relative_error = float("inf")
    used = hi
    converged = False
    was_stable = False
    q = 1

    while q <= hi:
        grid = _pixel_midpoint_energies(
            result.aperture,
            result.m,
            result.dx,
            result.wavelength,
            result.z,
            q,
        )
        selected = [
            grid[i][j]
            for i in range(row_start, row_end)
            for j in range(col_start, col_end)
        ]
        current_total = sum(selected)
        final_grid = grid

        if q >= lo:
            scale = max(abs(current_total), abs(previous_total))
            relative_error = (
                abs(current_total - previous_total) / scale if scale else 0.0
            )
            absolute_change = abs(current_total - previous_total)
            passes = (
                relative_error <= tolerance
                and absolute_change <= absolute_tolerance
            ) if absolute_tolerance > 0.0 else relative_error <= tolerance
            if passes and was_stable:
                converged = True
                used = q
            was_stable = passes

        previous_total = current_total
        if converged:
            break
        q *= 2

    tiled_sum = math.fsum(
        final_grid[i][j]
        for i in range(row_start, row_end)
        for j in range(col_start, col_end)
    )
    consistency_scale = max(abs(current_total), abs(tiled_sum), 1.0)
    consistency_error = abs(current_total - tiled_sum) / consistency_scale

    return RegionEnergyResult(
        energy=current_total,
        relative_error=relative_error,
        subdivisions=used,
        converged=converged,
        pixel_energy_sum=tiled_sum,
        consistency_error=consistency_error,
        row_start=row_start,
        row_end=row_end,
        col_start=col_start,
        col_end=col_end,
        tolerance=tolerance,
        absolute_tolerance=absolute_tolerance,
        min_subdivisions=lo,
        max_subdivisions=hi,
    )


# ---------------------------------------------------------------------------
# Deterministic aperture generators
# ---------------------------------------------------------------------------


def circular_aperture(n: int, radius: float, dx: float) -> list[list[float]]:
    """Deterministic centered circular aperture of ones.

    Pixel ``(i, j)`` is included when
    ``hypot((i - n//2) * dx, (j - n//2) * dx) <= radius`` (boundary included).
    """
    n = _strict_int(n, "n", lo=1, hi=MAX_APERTURE)
    radius = _finite_real(radius, "radius", nonneg=True)
    dx = _finite_real(dx, "dx", positive=True)
    c = n // 2
    grid = [[0.0] * n for _ in range(n)]
    r2 = radius * radius
    for i in range(n):
        for j in range(n):
            d2 = ((i - c) * dx) ** 2 + ((j - c) * dx) ** 2
            if d2 <= r2:
                grid[i][j] = 1.0
    return grid


def rectangular_aperture(n: int, width: float, height: float, dx: float) -> list[list[float]]:
    """Deterministic centered rectangular aperture of ones.

    Pixel ``(i, j)`` is included when
    ``|i - n//2| * dx <= width/2`` and ``|j - n//2| * dx <= height/2``.
    Width is along axis 0, height along axis 1; boundaries are included.
    """
    n = _strict_int(n, "n", lo=1, hi=MAX_APERTURE)
    width = _finite_real(width, "width", nonneg=True)
    height = _finite_real(height, "height", nonneg=True)
    dx = _finite_real(dx, "dx", positive=True)
    c = n // 2
    hx, hy = width / 2.0, height / 2.0
    grid = [[0.0] * n for _ in range(n)]
    for i in range(n):
        if abs((i - c) * dx) > hx:
            continue
        for j in range(n):
            if abs((j - c) * dx) <= hy:
                grid[i][j] = 1.0
    return grid


# ---------------------------------------------------------------------------
# Fraunhofer propagation
# ---------------------------------------------------------------------------


def fraunhofer(aperture, dx, wavelength, z, pad: int | None = None) -> FraunhoferResult:
    """Compute the sampled Fraunhofer diffraction field of a complex aperture.

    Parameters
    ----------
    aperture:
        N x N iterable of iterables of ``int | float | complex``, 1 <= N <= 32,
        given in centered layout.  NaN / infinite elements are rejected.
    dx:
        Aperture-plane sample pitch in metres (finite, > 0).
    wavelength:
        Optical wavelength in metres (finite, > 0).
    z:
        Propagation distance in metres (finite, > 0).
    pad:
        Optional transform side length M.  ``None`` uses M = N; otherwise
        ``N <= M <= 64``.  Padding is centered and zeros only; the returned
        observation pitch ``du`` always corresponds to M.

    Returns
    -------
    FraunhoferResult

    Raises
    ------
    TypeError
        Wrong parameter types; booleans are rejected wherever an int or a
        number is expected.
    ValueError
        NaN / infinity, non-positive physical parameters, bad shapes or bounds.
        The offending parameter name (or ``aperture[i][j]``) is in the message.
    """
    a_centered = _parse_aperture(aperture)
    n = len(a_centered)
    dx = _finite_real(dx, "dx", positive=True)
    wavelength = _finite_real(wavelength, "wavelength", positive=True)
    z = _finite_real(z, "z", positive=True)
    if pad is None:
        m = n
    else:
        m = _strict_int(pad, "pad", lo=n, hi=MAX_PADDED)

    # Embed the centered aperture into an M x M centered zero array and move to
    # unshifted (DFT) indices: unshifted index of shifted index i is
    # (i - n//2) mod m.  Offset o = m//2 - n//2 lines coordinates up for both
    # even and odd n, m.
    unshifted = [[0j] * m for _ in range(m)]
    for i in range(n):
        mi = (i - n // 2) % m
        for j in range(n):
            mj = (j - n // 2) % m
            unshifted[mi][mj] = a_centered[i][j]

    spectrum = _dft2(unshifted)

    scale = dx * dx / (wavelength * z)
    du = wavelength * z / (m * dx)

    # fftshift: shifted index p = (k + m//2) mod m.
    field: list[list[complex]] = [[0j] * m for _ in range(m)]
    intensity: list[list[float]] = [[0.0] * m for _ in range(m)]
    output_power = 0.0
    for k in range(m):
        pi = (k + m // 2) % m
        for l in range(m):
            pj = (l + m // 2) % m
            u = scale * spectrum[k][l]
            field[pi][pj] = u
            inten = u.real * u.real + u.imag * u.imag
            intensity[pi][pj] = inten
            output_power += inten

    input_power = 0.0
    for row in a_centered:
        for v in row:
            input_power += v.real * v.real + v.imag * v.imag
    input_power *= dx * dx
    output_power *= du * du

    if input_power == 0.0:
        power_error = 0.0
    else:
        power_error = (output_power - input_power) / input_power

    coords = [(p - m // 2) * du for p in range(m)]
    return FraunhoferResult(
        intensity=intensity,
        field=field,
        x=coords,
        y=list(coords),
        du=du,
        center=(m // 2, m // 2),
        input_power=input_power,
        output_power=output_power,
        power_error=power_error,
        n=n,
        m=m,
        aperture=a_centered,
        dx=dx,
        wavelength=wavelength,
        z=z,
    )
