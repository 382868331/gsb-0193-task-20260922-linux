"""Fraunhofer (far-field) diffraction propagation on a sampled aperture.

Model (fixed by the task contract)::

    U[p, q] = dx**2 / (wavelength * z) * DFT2(a_padded)[p, q]

with observation-plane sampling

    dx_out = wavelength * z / (M * dx)

where ``N x N`` is the input aperture (N <= 32) and ``M x M`` is the
optionally zero-padded DFT canvas (M <= 64, M >= N).  Zero padding
refines the observation-plane sampling, so ``dx_out`` is always reported
for the padded size and never silently reused from the unpadded case.

Even-grid / fftshift convention
-------------------------------
The raw DFT uses corner indexing (frequency 0 at index 0).  Results are
returned after the standard ``fftshift`` (``out[i] = raw[(i - M//2)
mod M]`` per axis): output row/column ``i`` carries raw frequency
``i - M//2`` (mod M), so DC is at index ``M//2`` with physical
coordinate 0.  For even M the integer frequency grid runs
``-M/2 ... M/2 - 1`` (negative Nyquist edge included, the duplicated
positive edge omitted); for odd M it is the symmetric
``-(M-1)/2 ... (M-1)/2``.  Physical observation coordinates are

    x[j] = (j - M//2) * dx_out       (columns)
    y[i] = (i - M//2) * dx_out       (rows)

The aperture is embedded with its first sample at canvas index 0;
centering the *physical* aperture instead would only multiply the field
by a linear phase and leaves every returned intensity unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .dft import dft2_raw, fftshift_2d
from .validation import (
    DiffractionError,
    checked_complex,
    checked_int,
    checked_real,
)

__all__ = ["FraunhoferResult", "fraunhofer_propagate"]

MAX_APERTURE_N = 32
MAX_PADDED_M = 64


@dataclass(frozen=True)
class FraunhoferResult:
    """Shifted Fraunhofer-plane result.

    Attributes:
        intensity: ``M x M`` list of ``|U|^2`` after fftshift.
        field: ``M x M`` complex field ``U`` after fftshift.
        x: observation-plane coordinates of columns (length M).
        y: observation-plane coordinates of rows (length M).
        pixel_size: observation-plane sample spacing ``dx_out``.
        center: ``(row, column)`` index of DC, always ``(M//2, M//2)``.
        input_power: ``dx^2 * sum |a|^2`` over the input aperture.
        output_power: ``dx_out^2 * sum |U|^2`` over the observation plane.
        relative_power_error: ``(output_power - input_power) / input_power``.
        input_size: N of the input aperture.
        padded_size: M of the DFT canvas (M >= N).
    """

    intensity: list[list[float]]
    field: list[list[complex]]
    x: list[float]
    y: list[float]
    pixel_size: float
    center: tuple[int, int]
    input_power: float
    output_power: float
    relative_power_error: float
    input_size: int
    padded_size: int

    def peak(self) -> tuple[int, int, float]:
        """Return ``(row, column, intensity)`` of the brightest sample."""
        best_r = best_c = 0
        best_v = -1.0
        for r, row in enumerate(self.intensity):
            for c, value in enumerate(row):
                if value > best_v:
                    best_v = value
                    best_r, best_c = r, c
        return best_r, best_c, best_v


def _validate_aperture(aperture) -> tuple[int, list[list[complex]]]:
    if not isinstance(aperture, (list, tuple)) or len(aperture) == 0:
        raise DiffractionError(
            "aperture", "expected a non-empty 2-D sequence of samples"
        )
    rows: list[list[complex]] = []
    n = len(aperture)
    if n > MAX_APERTURE_N:
        raise DiffractionError(
            "aperture",
            f"aperture has {n} rows; at most {MAX_APERTURE_N} are accepted",
        )
    for i, row in enumerate(aperture):
        name = f"aperture[{i}]"
        if not isinstance(row, (list, tuple)):
            raise DiffractionError(
                name, f"each row must be a sequence, got {type(row).__name__}"
            )
        if len(row) != n:
            raise DiffractionError(
                name,
                f"non-square aperture: row {i} has length {len(row)}, expected {n}",
            )
        converted: list[complex] = []
        for j, sample in enumerate(row):
            converted.append(checked_complex(sample, f"aperture[{i}][{j}]"))
        rows.append(converted)
    return n, rows


def fraunhofer_propagate(aperture, dx, wavelength, z, *, pad_to=None):
    """Propagate a sampled complex aperture to the Fraunhofer plane.

    Args:
        aperture: square ``N x N`` ``list[list[complex | float | int]]``
            with ``1 <= N <= 32``.  Booleans, NaN and infinite samples
            are rejected.
        dx: positive finite input-plane pixel spacing (same units as
            ``wavelength`` and ``z``).
        wavelength: positive finite optical wavelength.
        z: positive finite propagation distance.
        pad_to: optional canvas size M with ``N <= M <= 64``.  Defaults
            to N (no padding).  Must be an int (not bool).

    Returns:
        FraunhoferResult with shifted intensity/field, physical
        coordinates, DC index and the independently summed power error.

    Raises:
        DiffractionError: on any contract violation, with the offending
            parameter name attached.
    """
    n, field_in = _validate_aperture(aperture)
    dx = checked_real(dx, "dx", positive=True)
    wavelength = checked_real(wavelength, "wavelength", positive=True)
    z = checked_real(z, "z", positive=True)
    if pad_to is None:
        m = n
    else:
        m = checked_int(pad_to, "pad_to", minimum=n, maximum=MAX_PADDED_M)

    # Embed the N x N aperture at the corner of an M x M zero canvas.
    canvas = [[0.0j for _ in range(m)] for _ in range(m)]
    for i in range(n):
        src = field_in[i]
        dst = canvas[i]
        for j in range(n):
            dst[j] = src[j]

    spectrum = dft2_raw(canvas)

    scale = dx * dx / (wavelength * z)
    dx_out = wavelength * z / (m * dx)
    half = m // 2
    coords = [(i - half) * dx_out for i in range(m)]

    shifted = fftshift_2d(spectrum, m)
    out_field = [[scale * v for v in row] for row in shifted]
    intensity = [
        [(v.real * v.real + v.imag * v.imag) for v in row] for row in out_field
    ]

    input_power = 0.0
    for row in field_in:
        for v in row:
            input_power += v.real * v.real + v.imag * v.imag
    input_power *= dx * dx

    output_sum = 0.0
    for row in intensity:
        output_sum += math.fsum(row)
    output_power = output_sum * dx_out * dx_out

    if input_power > 0.0:
        rel_err = (output_power - input_power) / input_power
    else:
        rel_err = output_power  # identically zero for an all-zero aperture

    return FraunhoferResult(
        intensity=intensity,
        field=out_field,
        x=coords,
        y=list(coords),
        pixel_size=dx_out,
        center=(half, half),
        input_power=input_power,
        output_power=output_power,
        relative_power_error=rel_err,
        input_size=n,
        padded_size=m,
    )
