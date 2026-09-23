"""Fixed-input demonstration for fourier_diffraction.

Run:  python demo.py

Performs a real Fraunhofer computation (deterministic circular aperture,
N=32 zero-padded to a 64x64 canvas) and prints the far-field main peak,
its physical coordinates, the Parseval-style power check and a one-line
cross section.  Then it exercises one genuine rejection boundary (an
out-of-contract zero-pad size) and prints the resulting error.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fourier_diffraction import (
    DiffractionError,
    circular_aperture,
    fraunhofer_propagate,
)


def main() -> int:
    # --- Fixed, deterministic physical setup -----------------------------
    n = 32            # input aperture grid (<= 32 contract bound)
    dx = 10.0e-6      # pixel spacing, m
    wavelength = 0.5e-6
    z = 0.25          # propagation distance, m
    radius = 50.0e-6  # physical aperture radius, m

    aperture = circular_aperture(n, dx, radius)
    open_pixels = sum(1 for row in aperture for v in row if v != 0.0j)

    print("=== Fraunhofer diffraction demo (real computation) ===")
    print(f"input: {n}x{n} circular aperture, {open_pixels} open pixels")
    print(f"dx={dx:.3e} m, lambda={wavelength:.3e} m, z={z:.3f} m")

    result = fraunhofer_propagate(aperture, dx, wavelength, z, pad_to=64)
    pr, pc, peak = result.peak()
    print(f"canvas: {result.padded_size}x{result.padded_size} "
          f"(zero-padded), observation spacing dx_out="
          f"{result.pixel_size:.6e} m")
    print(f"DC center index (fftshift convention): {result.center}")
    print(f"main peak at shifted index ({pr}, {pc}), "
          f"physical (x={result.x[pc]:.6e} m, y={result.y[pr]:.6e} m)")
    print(f"peak intensity = {peak:.6e}")
    print(f"input power  = {result.input_power:.10e}")
    print(f"output power = {result.output_power:.10e}")
    print(f"relative power error = {result.relative_power_error:+.3e}")

    # Normalized cross section through the peak row (13 samples around DC).
    h = result.center[0]
    line = [result.intensity[h][j] / peak for j in range(h - 6, h + 7)]
    print("normalized central cross section (13 samples):")
    print("  " + " ".join(f"{v:6.3f}" for v in line))
    assert (pr, pc) == result.center, "circular aperture peak must be at DC"

    # --- Genuine rejection boundary --------------------------------------
    print()
    print("=== Rejection boundary (actually triggered) ===")
    try:
        fraunhofer_propagate(aperture, dx, wavelength, z, pad_to=65)
    except DiffractionError as exc:
        print(f"pad_to=65 rejected as required: {exc}")
    else:
        print("ERROR: pad_to=65 was not rejected")
        return 1

    # A second, distinct boundary: NaN physical input is refused.
    try:
        fraunhofer_propagate(aperture, dx, float("nan"), z)
    except DiffractionError as exc:
        print(f"NaN wavelength rejected as required: {exc}")
    else:
        print("ERROR: NaN wavelength was not rejected")
        return 1

    print("demo completed normally")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
