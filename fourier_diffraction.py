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

In addition to the single-wavelength :func:`fraunhofer` entry,
:func:`fraunhofer_modes` propagates up to 3 wavelength groups of up to 3
coherent modes each.  Every mode is propagated independently with its own
wavelength, aperture and non-negative weight, and the detector grid receives
the *intensity* sum ``sum_m weight_m * |U_m|**2``.  Complex amplitudes are
never added across wavelengths (or across modes); only intensities stack.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

# Hard interface bounds fixed by the task.
MAX_APERTURE = 32
MAX_PADDED = 64
MAX_WAVELENGTH_GROUPS = 3
MAX_MODES_PER_WAVELENGTH = 3


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
class ModeComponent:
    """One independently propagated coherent mode.

    Attributes
    ----------
    wavelength:
        This mode's wavelength in metres.  Modes sharing a wavelength are
        grouped by the caller; the library never combines their fields.
    weight:
        Non-negative linear intensity weight (0.0 is allowed and yields a
        zero-strength component).
    result:
        The ordinary single-mode :class:`FraunhoferResult` of this mode.
    weighted_power:
        ``weight * result.output_power``; the contribution of this mode to the
        incoherent detector power.
    """

    wavelength: float
    weight: float
    result: FraunhoferResult
    weighted_power: float


@dataclass(frozen=True)
class MultimodeResult:
    """Incoherent superposition of independently propagated modes.

    All components live on the same DFT-bin grid (same ``m`` and ``pad``), so
    the detector arrays are directly index-aligned even though the physical
    pitch ``du`` differs between wavelength groups.

    Attributes
    ----------
    intensity:
        M x M ``float`` detector grid:
        ``intensity[i][j] = sum_g sum_t weight[g][t] * |U[g][t][i][j]|**2``.
        Complex amplitudes are never summed across wavelengths or modes.
    components:
        Flat tuple of all :class:`ModeComponent`, ordered as the input groups
        (group by group, mode by mode).  Each retains its own wavelength,
        weight, field, intensity, coordinates, powers and power error.
    wavelengths:
        Tuple of the distinct group wavelengths, in input order.
    weights:
        Tuple of tuples of weights aligned with the input groups.
    total_weight:
        ``sum`` of all weights (informational; intensities are *not* divided by
        it — weights are plain linear coefficients).
    detector_power:
        ``sum`` over modes of ``weight * du_m**2 * sum |U_m|**2``.  Note that
        each mode is weighted on the detector grid in its own physical units;
        this is the Parseval-consistent power budget of the superposition.
    m:
        Common (possibly padded) transform side length of every component.
    pad:
        Common transform-side argument actually used (``m`` when ``pad=None``).
    center:
        Zero-coordinate index ``(m // 2, m // 2)`` (identical for all
        components).
    """

    intensity: list[list[float]]
    components: tuple[ModeComponent, ...]
    wavelengths: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    total_weight: float
    detector_power: float
    m: int
    pad: int
    center: tuple[int, int]


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


def _propagate(
    a_centered: list[list[complex]], n: int, dx: float, wavelength: float, z: float, m: int
) -> FraunhoferResult:
    """Core Fraunhofer propagation on already-validated finite inputs."""
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

    return _propagate(a_centered, n, dx, wavelength, z, m)


# ---------------------------------------------------------------------------
# Finite-bandwidth, multi-wavelength incoherent mode propagation
# ---------------------------------------------------------------------------


_GROUP_KEYS = frozenset({"wavelength", "modes"})
_MODE_KEYS = frozenset({"aperture", "weight"})


def _is_real_number(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
    )


def _row_like(value) -> bool:
    """True for a non-string iterable (an aperture row must be such)."""
    return not isinstance(value, (str, bytes)) and hasattr(value, "__iter__")


def _looks_like_aperture(value) -> bool:
    """True if value has the shape of a 2-D aperture.

    Non-empty iterable whose first row is iterable and whose first element is
    a scalar (a number), not a nested sequence.
    """
    if not _row_like(value):
        return False
    for row in value:
        if not _row_like(row):
            return False
        for elem in row:
            return not _row_like(elem)
        return False
    return False


def _parse_mode(mode, where: str):
    """Return (aperture_array_like, weight_or_None) for one mode specification.

    Accepted forms::

        {"aperture": A, "weight": w}   # weight optional, defaults to 1.0
        (A,) or (A, w)
        A                              # bare 2-D aperture, weight 1.0

    List/tuple forms are disambiguated by content: a 2-item sequence is a
    ``(aperture, weight)`` wrapper only when its second item is a scalar real
    number; a 1-item sequence is a wrapper only when its item itself looks like
    a 2-D aperture.  Everything else is taken as a bare aperture grid.
    """
    if isinstance(mode, dict):
        unknown = set(mode) - _MODE_KEYS
        if unknown:
            raise ValueError(
                f"{where}: unknown mode key(s) {sorted(unknown)!r}; "
                f"allowed keys are 'aperture' and 'weight'"
            )
        if "aperture" not in mode:
            raise ValueError(f"{where}: mode mapping is missing required key 'aperture'")
        return mode["aperture"], mode.get("weight", 1.0)

    if isinstance(mode, (tuple, list)):
        if len(mode) == 2 and _is_real_number(mode[1]):
            return mode[0], mode[1]
        if len(mode) == 2 and not _row_like(mode[1]):
            raise TypeError(
                f"{where} weight must be a real number, got {type(mode[1]).__name__}: "
                f"{mode[1]!r}"
            )
        if len(mode) == 1 and not _looks_like_aperture(mode) and _looks_like_aperture(mode[0]):
            return mode[0], 1.0
        if len(mode) >= 3 and _looks_like_aperture(mode[0]) and all(
            _is_real_number(x) for x in mode[1:]
        ):
            raise ValueError(
                f"{where}: mode sequence must have 1 or 2 items (aperture[, weight]), "
                f"got {len(mode)}"
            )

    # Bare aperture specification: any other list/tuple (including 1x1 and
    # N-row grids with N > 2) is validated as a 2-D aperture grid.
    return mode, 1.0


def _parse_group(group, g: int):
    """Return (wavelength, [(aperture, weight), ...]) for one wavelength group."""
    where = f"groups[{g}]"
    if isinstance(group, dict):
        unknown = set(group) - _GROUP_KEYS
        if unknown:
            raise ValueError(
                f"{where}: unknown group key(s) {sorted(unknown)!r}; "
                f"allowed keys are 'wavelength' and 'modes'"
            )
        if "wavelength" not in group:
            raise ValueError(f"{where}: group mapping is missing required key 'wavelength'")
        if "modes" not in group:
            raise ValueError(f"{where}: group mapping is missing required key 'modes'")
        wavelength_raw, modes_raw = group["wavelength"], group["modes"]
    elif isinstance(group, (tuple, list)) and len(group) == 2:
        wavelength_raw, modes_raw = group
    else:
        raise TypeError(
            f"{where}: each wavelength group must be a mapping with keys "
            f"'wavelength'/'modes', or a (wavelength, modes) pair"
        )

    wavelength = _finite_real(wavelength_raw, f"{where} wavelength", positive=True)

    if isinstance(modes_raw, (str, bytes)) or not hasattr(modes_raw, "__iter__"):
        raise TypeError(f"{where} modes must be an iterable of mode specifications")
    mode_specs = list(modes_raw)
    if not 1 <= len(mode_specs) <= MAX_MODES_PER_WAVELENGTH:
        raise ValueError(
            f"{where}: each wavelength must carry 1..{MAX_MODES_PER_WAVELENGTH} modes, "
            f"got {len(mode_specs)}"
        )

    parsed_modes = []
    for t, mode_raw in enumerate(mode_specs):
        mwhere = f"{where}.modes[{t}]"
        aperture_raw, weight_raw = _parse_mode(mode_raw, mwhere)
        weight = _finite_real(weight_raw, f"{mwhere} weight", nonneg=True)
        try:
            aperture = _parse_aperture(aperture_raw)
        except (TypeError, ValueError) as exc:
            raise type(exc)(f"{mwhere} aperture: {exc}") from None
        parsed_modes.append((aperture, weight))
    return wavelength, parsed_modes


def fraunhofer_modes(groups, dx, z, pad: int | None = None) -> MultimodeResult:
    """Incoherent multi-wavelength / multi-mode Fraunhofer propagation.

    Each mode is propagated independently through the ordinary Fraunhofer
    model.  The returned detector grid holds the *intensity* superposition::

        I[i][j] = sum_g sum_t weight[g][t] * |U[g][t][i][j]|**2

    Complex amplitudes are never added across wavelengths (nor across modes):
    different wavelengths do not interfere, so only intensities stack.

    Parameters
    ----------
    groups:
        Iterable of 1..3 wavelength groups.  Each group is either a mapping
        ``{"wavelength": lam, "modes": [mode, ...]}`` or a pair
        ``(lam, [mode, ...])``; each wavelength carries 1..3 modes.  Group
        wavelengths must be finite positive reals and pairwise distinct.

        Each mode is one of::

            {"aperture": A, "weight": w}   # weight optional (default 1.0)
            (A,) or (A, w)
            A                              # bare N x N aperture, weight 1.0

        ``A`` follows the same rules as in :func:`fraunhofer` (square
        1..32 x 1..32, finite complex elements).  ``w`` is a finite,
        non-negative real linear intensity weight (zero is allowed); booleans
        are rejected.  Weights are *not* normalized: ``w=2`` literally doubles
        that mode's intensity contribution.
    dx:
        Aperture-plane sample pitch in metres, shared by all modes (> 0).
    z:
        Propagation distance in metres, shared by all modes (> 0).
    pad:
        Optional common transform side length M.  ``None`` selects
        ``M = max(N_m)`` so every mode is embedded into the same M x M
        DFT-bin grid; otherwise ``max(N_m) <= M <= 64``.  Each wavelength
        still gets its own physical pitch ``du = wavelength*z/(M*dx)``,
        available on its component result.

    Returns
    -------
    MultimodeResult
        Combined intensity grid plus one :class:`ModeComponent` per mode,
        each retaining its own field, coordinates and power accounting.

    Raises
    ------
    TypeError / ValueError
        For any malformed group/mode specification, non-finite or negative
        weights, non-positive wavelengths, duplicate group wavelengths,
        illegal apertures, or ``pad`` out of range.  All inputs are fully
        validated before any propagation; the call is a pure function and
        never leaves a partially computed state.
    """
    if isinstance(groups, dict):
        raise TypeError(
            "groups must be an iterable of wavelength groups, got a single mapping; "
            "wrap it in a list: [{'wavelength': ..., 'modes': [...]}]"
        )
    if isinstance(groups, (str, bytes)) or not hasattr(groups, "__iter__"):
        raise TypeError("groups must be an iterable of wavelength groups")
    group_list = list(groups)
    if not 1 <= len(group_list) <= MAX_WAVELENGTH_GROUPS:
        raise ValueError(
            f"groups must contain 1..{MAX_WAVELENGTH_GROUPS} wavelength groups, "
            f"got {len(group_list)}"
        )

    plan = []
    seen_wavelengths: set[float] = set()
    max_n = 0
    for g, group_raw in enumerate(group_list):
        wavelength, parsed_modes = _parse_group(group_raw, g)
        if wavelength in seen_wavelengths:
            raise ValueError(
                f"groups[{g}]: duplicate wavelength {wavelength!r}; "
                f"each wavelength may appear in only one group (put its modes together)"
            )
        seen_wavelengths.add(wavelength)
        for aperture, _weight in parsed_modes:
            max_n = max(max_n, len(aperture))
        plan.append((wavelength, parsed_modes))

    dx = _finite_real(dx, "dx", positive=True)
    z = _finite_real(z, "z", positive=True)
    if pad is None:
        m = max_n
    else:
        m = _strict_int(pad, "pad", lo=max_n, hi=MAX_PADDED)

    total: list[list[float]] = [[0.0] * m for _ in range(m)]
    components: list[ModeComponent] = []
    wavelengths: list[float] = []
    weights: list[tuple[float, ...]] = []
    total_weight = 0.0
    detector_power = 0.0

    for g, (wavelength, parsed_modes) in enumerate(plan):
        wavelengths.append(wavelength)
        group_weights: list[float] = []
        for t, (aperture, weight) in enumerate(parsed_modes):
            n = len(aperture)
            r = _propagate(aperture, n, dx, wavelength, z, m)
            for i in range(m):
                row = total[i]
                src = r.intensity[i]
                if weight == 0.0:
                    continue
                for j in range(m):
                    row[j] += weight * src[j]
            wp = weight * r.output_power
            detector_power += wp
            total_weight += weight
            group_weights.append(weight)
            components.append(
                ModeComponent(
                    wavelength=wavelength,
                    weight=weight,
                    result=r,
                    weighted_power=wp,
                )
            )
        weights.append(tuple(group_weights))

    return MultimodeResult(
        intensity=total,
        components=tuple(components),
        wavelengths=tuple(wavelengths),
        weights=tuple(weights),
        total_weight=total_weight,
        detector_power=detector_power,
        m=m,
        pad=m,
        center=(m // 2, m // 2),
    )
