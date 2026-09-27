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
# Continuous field and adaptive pixel-energy integration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntegrationResult:
    """Energy of the far-field intensity integrated over a rectangular region.

    Attributes
    ----------
    energy:
        Integral of ``|U(x, y)|**2`` over ``region`` (the power a detector
        pixel covering that area would receive).
    error_estimate:
        Conservative absolute error estimate: sum over all accepted cells of
        ``|Q_cell - sum(Q_subcells)|`` (no Richardson rescaling, so it errs on
        the safe side).
    converged:
        ``True`` only if the adaptive refinement met the requested tolerance
        everywhere.  ``False`` means the refinement budget (``max_depth`` or
        ``max_evaluations``) was exhausted first; ``energy`` is still the best
        available estimate, but the requested accuracy is *not* guaranteed.
        Budget exhaustion is reported this way -- it is never silently
        treated as success.
    evaluations:
        Number of continuous-field evaluations actually used.
    depth:
        Deepest subdivision level reached (0 = the whole region as one cell).
    region:
        ``(x0, x1, y0, y1)`` in metres, the exact integrated rectangle.
    """

    energy: float
    error_estimate: float
    converged: bool
    evaluations: int
    depth: int
    region: tuple[float, float, float, float]


def _continuous_field(a: list[list[complex]], dx: float, wavelength: float,
                      z: float, x: float, y: float) -> complex:
    """Continuous Fraunhofer field U(x, y), consistent with the DFT bins.

    ``U(x, y) = dx**2/(wavelength*z) * sum_{m,n} a[m,n]
                * exp(-i 2 pi (x (m - n0) + y (n - n0)) dx / (wavelength z))``

    with ``n0 = len(a) // 2``: aperture pixel ``(m, n)`` of the centered
    input layout sits at physical coordinate
    ``((m - n0) * dx, (n - n0) * dx)``.

    At unshifted bin coordinates ``x = k*du, y = l*du`` this reproduces the
    DFT spectrum exactly (same sign convention as :func:`fraunhofer`).
    Phasors are advanced by repeated multiplication (N <= 32, so the
    accumulated round-off stays at the 1e-14 level).
    """
    n = len(a)
    n0 = n // 2
    factor = 2.0 * math.pi * dx / (wavelength * z)
    bx = cmath.exp(-1j * factor * x)
    by = cmath.exp(-1j * factor * y)
    total = 0j
    ex = 1.0 + 0.0j
    for m in range(n):
        row = a[m]
        inner = 0j
        ey = 1.0 + 0.0j
        for k in range(n):
            v = row[k]
            if v:
                inner += v * ey
            ey *= by
        total += inner * ex
        ex *= bx
    # Relocate phasor powers to centered offsets (m - n0, n - n0): dividing
    # by bx**n0 * by**n0 places pixel (m, n) at its physical coordinate.
    total /= (bx ** n0) * (by ** n0)
    scale = dx * dx / (wavelength * z)
    return scale * total


# 3-point Gauss-Legendre nodes/weights on [-1, 1].  The nodes are irrational
# multiples of the cell half-width, so sample points of a cell never coincide
# with those of its subcells or with cell centres/corners: a handful of
# collocated samples cannot accidentally agree on a rapidly varying
# integrand and fake convergence.
_GL3 = (
    (-math.sqrt(3.0 / 5.0), 5.0 / 9.0),
    (0.0, 8.0 / 9.0),
    (math.sqrt(3.0 / 5.0), 5.0 / 9.0),
)


def _cell_energy(field_at, x0: float, x1: float, y0: float, y1: float) -> float:
    """3x3 tensor Gauss-Legendre estimate of the intensity integral."""
    hx = 0.5 * (x1 - x0)
    hy = 0.5 * (y1 - y0)
    mx = 0.5 * (x0 + x1)
    my = 0.5 * (y0 + y1)
    total = 0.0
    for tx, wx in _GL3:
        x = mx + hx * tx
        for ty, wy in _GL3:
            u = field_at(x, my + hy * ty)
            total += wx * wy * (u.real * u.real + u.imag * u.imag)
    return total * hx * hy


def _validate_integration_controls(rtol, atol, min_depth, max_depth,
                                   max_evaluations) -> tuple[float, float, int, int, int]:
    rtol = _finite_real(rtol, "rtol", positive=True)
    atol = _finite_real(atol, "atol", nonneg=True)
    min_depth = _strict_int(min_depth, "min_depth", lo=0)
    max_depth = _strict_int(max_depth, "max_depth", lo=0)
    if min_depth > max_depth:
        raise ValueError(f"min_depth must be <= max_depth, got {min_depth} > {max_depth}")
    max_evaluations = _strict_int(max_evaluations, "max_evaluations", lo=9)
    return rtol, atol, min_depth, max_depth, max_evaluations


def integrate_intensity(aperture, dx, wavelength, z, x0, x1, y0, y1,
                        rtol: float = 1e-6, atol: float = 0.0,
                        min_depth: int = 2, max_depth: int = 12,
                        max_evaluations: int = 200000) -> IntegrationResult:
    """Integrate the far-field intensity over the rectangle [x0, x1] x [y0, y1].

    The integrand is the *continuous* Fraunhofer intensity (direct aperture
    sum, see :func:`_continuous_field`), not an interpolation of the sampled
    grid.  The rectangle is refined by adaptive quadtree subdivision of 3x3
    Gauss-Legendre cells.

    Convergence contract (guards against premature stops):

    * Every cell is subdivided at least ``min_depth`` times before its
      acceptance test is even consulted, so a smooth-looking handful of
      samples in a fast bright/dark oscillation cannot terminate the
      refinement early.
    * A cell is accepted only when ``|Q_cell - sum(Q_4 subcells)|`` is within
      the cell's area-proportional share of the global tolerance
      ``T = max(atol, rtol * Q_whole_region)``.
    * Gauss nodes never coincide between refinement levels, so parent and
      child estimates are genuinely independent sample sets.

    ``converged`` in the result tells whether the requested tolerance was
    actually reached; exhausting ``max_depth`` / ``max_evaluations`` yields
    ``converged=False`` instead of a silent low-accuracy answer.

    Raises
    ------
    TypeError / ValueError
        Same input contract as :func:`fraunhofer`; additionally the region
        bounds must be finite reals with ``x0 < x1`` and ``y0 < y1``.
    """
    a_centered = _parse_aperture(aperture)
    dx = _finite_real(dx, "dx", positive=True)
    wavelength = _finite_real(wavelength, "wavelength", positive=True)
    z = _finite_real(z, "z", positive=True)
    x0 = _finite_real(x0, "x0")
    x1 = _finite_real(x1, "x1")
    y0 = _finite_real(y0, "y0")
    y1 = _finite_real(y1, "y1")
    if not x0 < x1:
        raise ValueError(f"x0 must be < x1, got x0={x0!r}, x1={x1!r}")
    if not y0 < y1:
        raise ValueError(f"y0 must be < y1, got y0={y0!r}, y1={y1!r}")
    rtol, atol, min_depth, max_depth, max_evaluations = _validate_integration_controls(
        rtol, atol, min_depth, max_depth, max_evaluations
    )

    def field_at(x: float, y: float) -> complex:
        return _continuous_field(a_centered, dx, wavelength, z, x, y)

    root_area = (x1 - x0) * (y1 - y0)
    root_q = _cell_energy(field_at, x0, x1, y0, y1)
    evaluations = 9
    tolerance = max(atol, rtol * root_q)

    energy = 0.0
    error_estimate = 0.0
    converged = True
    depth_reached = 0
    # Stack entries: (x0, x1, y0, y1, depth, cell estimate).
    stack = [(x0, x1, y0, y1, 0, root_q)]
    while stack:
        cx0, cx1, cy0, cy1, depth, q_cell = stack.pop()
        if depth > depth_reached:
            depth_reached = depth
        hx = 0.5 * (cx1 - cx0)
        hy = 0.5 * (cy1 - cy0)
        mx = cx0 + hx
        my = cy0 + hy

        sub = None
        diff = math.inf
        if depth >= min_depth and depth > 0:
            sub = [
                _cell_energy(field_at, cx0, mx, cy0, my),
                _cell_energy(field_at, mx, cx1, cy0, my),
                _cell_energy(field_at, cx0, mx, my, cy1),
                _cell_energy(field_at, mx, cx1, my, cy1),
            ]
            evaluations += 36
            q_sub = sub[0] + sub[1] + sub[2] + sub[3]
            diff = abs(q_cell - q_sub)
            cell_share = tolerance * ((cx1 - cx0) * (cy1 - cy0) / root_area)
            if diff <= cell_share:
                energy += q_sub
                error_estimate += diff
                continue

        can_subdivide = depth < max_depth and evaluations + 36 <= max_evaluations
        if not can_subdivide:
            # Budget exhausted: keep the best estimate but report failure.
            if sub is None:
                sub = [
                    _cell_energy(field_at, cx0, mx, cy0, my),
                    _cell_energy(field_at, mx, cx1, cy0, my),
                    _cell_energy(field_at, cx0, mx, my, cy1),
                    _cell_energy(field_at, mx, cx1, my, cy1),
                ]
                evaluations += 36
                q_sub = sub[0] + sub[1] + sub[2] + sub[3]
                diff = abs(q_cell - q_sub)
            energy += q_sub
            error_estimate += diff
            converged = False
            continue

        if sub is None:
            sub = [
                _cell_energy(field_at, cx0, mx, cy0, my),
                _cell_energy(field_at, mx, cx1, cy0, my),
                _cell_energy(field_at, cx0, mx, my, cy1),
                _cell_energy(field_at, mx, cx1, my, cy1),
            ]
            evaluations += 36
        stack.append((cx0, mx, cy0, my, depth + 1, sub[0]))
        stack.append((mx, cx1, cy0, my, depth + 1, sub[1]))
        stack.append((cx0, mx, my, cy1, depth + 1, sub[2]))
        stack.append((mx, cx1, my, cy1, depth + 1, sub[3]))

    if error_estimate > tolerance:
        converged = False
    return IntegrationResult(
        energy=energy,
        error_estimate=error_estimate,
        converged=converged,
        evaluations=evaluations,
        depth=depth_reached,
        region=(x0, x1, y0, y1),
    )


def pixel_energy(aperture, dx, wavelength, z, i: int, j: int,
                 pad: int | None = None, rtol: float = 1e-6, atol: float = 0.0,
                 min_depth: int = 2, max_depth: int = 12,
                 max_evaluations: int = 200000) -> IntegrationResult:
    """Energy received by the detector pixel centred on sample ``(i, j)``.

    The pixel is the ``du x du`` square centred at the shifted sample
    coordinate ``((i - M//2) * du, (j - M//2) * du)`` with
    ``du = wavelength * z / (M * dx)`` and ``M = pad or N`` -- the same grid
    convention as :func:`fraunhofer`.  The intensity is integrated over the
    pixel area (see :func:`integrate_intensity`); it is *not* approximated by
    ``intensity[i][j] * du**2``.

    ``i`` and ``j`` must be strict ints with ``0 <= i, j < M`` (booleans are
    rejected).  All other parameters follow :func:`integrate_intensity`.
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
    i = _strict_int(i, "i", lo=0, hi=m - 1)
    j = _strict_int(j, "j", lo=0, hi=m - 1)

    du = wavelength * z / (m * dx)
    xc = (i - m // 2) * du
    yc = (j - m // 2) * du
    half = 0.5 * du
    return integrate_intensity(
        a_centered, dx, wavelength, z,
        xc - half, xc + half, yc - half, yc + half,
        rtol=rtol, atol=atol, min_depth=min_depth,
        max_depth=max_depth, max_evaluations=max_evaluations,
    )
