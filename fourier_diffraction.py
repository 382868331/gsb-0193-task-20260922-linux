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
MAX_MODES = 3
MAX_WAVELENGTHS = 3


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


@dataclass(frozen=True)
class CoherentMode:
    """One coherent mode: an aperture illuminated at one wavelength.

    Fields of *different* modes are mutually incoherent: they are never added
    as complex amplitudes.  Modes sharing a wavelength are still treated as
    incoherent (e.g. orthogonal spatial modes); only the weighted intensities
    are summed.

    Attributes
    ----------
    aperture:
        N x N iterable of iterables of ``int | float | complex``, 1 <= N <= 32,
        centered layout.  All modes in one :func:`fraunhofer_modes` call must
        share the same N (and the common ``dx``).
    wavelength:
        Optical wavelength in metres (finite, > 0).  At most
        ``MAX_WAVELENGTHS`` distinct values are allowed per call.
    weight:
        Non-negative finite real intensity weight ``w`` of the mode.  The mode
        contributes ``w * |U|**2`` to the summed intensity grid.
    """

    aperture: list
    wavelength: float
    weight: float = 1.0


@dataclass(frozen=True)
class MultiModeResult:
    """Incoherent superposition of independently propagated coherent modes.

    Attributes
    ----------
    intensity:
        M x M ``float`` grid, ``sum_q weight_q * |U_q|**2``, in the same
        centered (fftshifted) layout as :attr:`FraunhoferResult.intensity`.
        There is deliberately **no** ``field`` attribute: fields of different
        wavelengths (and incoherent modes in general) must not be added as
        complex amplitudes, so no cross-mode complex field exists here.
    fx, fy:
        Length-M spatial-frequency coordinates in cycles/metre,
        ``(p - M//2) / (M*dx)``.  These are the wavelength-independent common
        axis of the sum; mode ``q``'s physical coordinate at pixel ``p`` is
        ``mode_results[q].x[p] == wavelength_q * z * fx[p]``.
    center:
        Index ``(M // 2, M // 2)`` of the zero coordinate.
    weights:
        The validated non-negative mode weights, in input order.
    wavelengths:
        The validated mode wavelengths in metres, in input order.
    input_power:
        ``sum_q weight_q * dx**2 * sum |A_q|**2``.
    output_power:
        ``sum_q weight_q * du_q**2 * sum |U_q|**2``.  Each per-mode term obeys
        Parseval, so this equals ``input_power`` up to round-off.
    power_error:
        Signed relative discrepancy ``(output - input) / input``; ``0.0`` when
        the total weighted input power is zero.
    mode_results:
        The individual :class:`FraunhoferResult` of every mode, in input
        order.  Per-mode ``field`` / physical ``x`` / ``y`` / ``du`` live here.
    n, m:
        Common input aperture side length / transform side length.
    """

    intensity: list[list[float]]
    fx: list[float]
    fy: list[float]
    center: tuple[int, int]
    weights: list[float]
    wavelengths: list[float]
    input_power: float
    output_power: float
    power_error: float
    mode_results: list[FraunhoferResult]
    n: int
    m: int


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
    )


# ---------------------------------------------------------------------------
# Multi-wavelength / multi-mode incoherent propagation
# ---------------------------------------------------------------------------


def fraunhofer_modes(modes, dx, z, pad: int | None = None) -> MultiModeResult:
    """Propagate several coherent modes independently and sum intensities.

    Each :class:`CoherentMode` is propagated on its own with :func:`fraunhofer`
    (its own wavelength, hence its own ``du`` and physical coordinates), and
    only the weighted *intensities* are added on the common M x M pixel grid::

        I[p, q] = sum_s weight_s * |U_s[p, q]|**2

    Complex amplitudes are **never** added across modes or wavelengths: light
    of distinct wavelengths, and incoherent modes in general, does not
    interfere.  The returned object therefore exposes no summed ``field``;
    the per-mode complex fields remain available in ``mode_results``.

    All modes share ``dx``, ``z``, the aperture side length N and ``pad``.
    Between 1 and ``MAX_MODES`` modes are allowed, with at most
    ``MAX_WAVELENGTHS`` distinct wavelengths.  Every input is validated
    *before* any propagation runs, so a rejected call performs no computation
    and leaves no partially updated state (the function is pure).

    Parameters
    ----------
    modes:
        Iterable of 1..3 :class:`CoherentMode` instances with identical N.
    dx:
        Common aperture-plane sample pitch in metres (finite, > 0).
    z:
        Common propagation distance in metres (finite, > 0).
    pad:
        Optional common transform side length M, ``N <= M <= 64``; ``None``
        uses M = N.

    Returns
    -------
    MultiModeResult

    Raises
    ------
    TypeError
        ``modes`` is not an iterable of :class:`CoherentMode`, a weight is not
        real, or ``pad`` is not a strict int (booleans rejected).
    ValueError
        Wrong mode count (outside 1..3), more than 3 distinct wavelengths,
        negative / non-finite / NaN weights or wavelengths, mismatched
        aperture sizes, or any underlying :func:`fraunhofer` shape/bounds
        violation.  Messages locate the offending entry, e.g.
        ``modes[1].weight`` or ``modes[0].aperture[i][j]``.
    """
    if isinstance(modes, (str, bytes)) or not hasattr(modes, "__iter__"):
        raise TypeError(
            "modes must be an iterable of CoherentMode, "
            f"got {type(modes).__name__}"
        )
    mode_list = list(modes)
    if not 1 <= len(mode_list) <= MAX_MODES:
        raise ValueError(
            f"modes must contain between 1 and {MAX_MODES} modes, "
            f"got {len(mode_list)}"
        )
    for idx, mode in enumerate(mode_list):
        if isinstance(mode, bool) or not isinstance(mode, CoherentMode):
            raise TypeError(
                f"modes[{idx}] must be a CoherentMode, got {type(mode).__name__}"
            )

    # Validate and materialize every aperture first so generator inputs cannot
    # be exhausted halfway through, and so all checks precede any propagation.
    grids: list[list[list[complex]]] = []
    n_common: int | None = None
    for idx, mode in enumerate(mode_list):
        grid = _parse_aperture_named(mode.aperture, f"modes[{idx}].aperture")
        n = len(grid)
        if n_common is None:
            n_common = n
        elif n != n_common:
            raise ValueError(
                f"modes[{idx}].aperture side length {n} does not match "
                f"modes[0].aperture side length {n_common}"
            )
        grids.append(grid)
    assert n_common is not None
    n = n_common

    weights: list[float] = []
    wavelengths: list[float] = []
    for idx, mode in enumerate(mode_list):
        weights.append(_finite_real(mode.weight, f"modes[{idx}].weight", nonneg=True))
        wavelengths.append(
            _finite_real(mode.wavelength, f"modes[{idx}].wavelength", positive=True)
        )
    distinct_wavelengths = {w for w in wavelengths}
    if len(distinct_wavelengths) > MAX_WAVELENGTHS:
        raise ValueError(
            f"at most {MAX_WAVELENGTHS} distinct wavelengths are allowed, "
            f"got {len(distinct_wavelengths)}"
        )

    dx = _finite_real(dx, "dx", positive=True)
    z = _finite_real(z, "z", positive=True)
    if pad is None:
        m = n
    else:
        m = _strict_int(pad, "pad", lo=n, hi=MAX_PADDED)

    # All inputs are valid from here on; propagate the modes independently.
    mode_results: list[FraunhoferResult] = []
    for grid, wavelength in zip(grids, wavelengths):
        mode_results.append(fraunhofer(grid, dx, wavelength, z, pad=m))

    intensity: list[list[float]] = [[0.0] * m for _ in range(m)]
    input_power = 0.0
    output_power = 0.0
    for weight, result in zip(weights, mode_results):
        for i in range(m):
            row = intensity[i]
            src = result.intensity
            for j in range(m):
                row[j] += weight * src[i][j]
        input_power += weight * result.input_power
        output_power += weight * result.output_power

    if input_power == 0.0:
        power_error = 0.0
    else:
        power_error = (output_power - input_power) / input_power

    fx = [(p - m // 2) / (m * dx) for p in range(m)]
    return MultiModeResult(
        intensity=intensity,
        fx=fx,
        fy=list(fx),
        center=(m // 2, m // 2),
        weights=weights,
        wavelengths=wavelengths,
        input_power=input_power,
        output_power=output_power,
        power_error=power_error,
        mode_results=mode_results,
        n=n,
        m=m,
    )


def _parse_aperture_named(aperture, where: str) -> list[list[complex]]:
    """Parse an aperture, prefixing every element-level error with ``where``.

    Structural (iterability / rectangularity / squareness / size) failures are
    re-raised with the same prefix; per-element failures from the shared
    validator already carry the full path.
    """
    if isinstance(aperture, (str, bytes)) or not hasattr(aperture, "__iter__"):
        raise TypeError(f"{where} must be a 2-D array-like of complex amplitudes")
    try:
        return _parse_aperture(aperture)
    except (TypeError, ValueError) as exc:
        message = str(exc)
        if message.startswith("aperture"):
            raise type(exc)(f"{where}{message[len('aperture'):]}") from None
        raise type(exc)(f"{where}: {message}") from None
