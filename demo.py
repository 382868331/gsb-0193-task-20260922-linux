"""Fixed-input demonstration of fourier_diffraction.

Runs one real Fraunhofer calculation (16x16 circular aperture, evaluated on a
32x32 detector grid), integrates finite detector-pixel areas, and triggers one
genuine input-rejection boundary.
"""

import time

import fourier_diffraction as fd


def main() -> None:
    # ---- Fixed physical parameters ------------------------------------------------
    n = 16
    dx = 10.0e-6          # 10 micrometre aperture-plane pitch
    wavelength = 0.5e-6   # 500 nm
    z = 0.10              # 10 cm propagation
    radius = 6.0 * dx     # circle spanning 13 pixels across

    aperture = fd.circular_aperture(n, radius, dx)
    lit = int(sum(sum(row) for row in aperture))

    t0 = time.perf_counter()
    result = fd.fraunhofer(aperture, dx, wavelength, z, pad=32)
    elapsed = time.perf_counter() - t0

    # Locate the main peak from the returned intensity grid.
    peak_i = peak_j = 0
    peak = -1.0
    for i, row in enumerate(result.intensity):
        for j, v in enumerate(row):
            if v > peak:
                peak, peak_i, peak_j = v, i, j

    ci, cj = result.center
    # Independently normalize the peak for an honest relative-sidelobe report.
    second = 0.0
    for i, row in enumerate(result.intensity):
        for j, v in enumerate(row):
            if (i, j) != (peak_i, peak_j) and v > second:
                second = v

    print("=== Fraunhofer diffraction demo (real computation) ===")
    print(f"input aperture : {n}x{n} circular, radius = {radius:.2e} m, {lit} lit pixels")
    print(f"transform grid : {result.m}x{result.m} (centered zero pad)")
    print(f"dx / du        : {dx:.3e} m / {result.du:.3e} m (pitch updated for padding)")
    print(f"field extent   : x in [{result.x[0]:.3e}, {result.x[-1]:.3e}] m")
    print(f"main peak      : intensity = {peak:.6e}")
    print(f"peak location  : index ({peak_i}, {peak_j}), "
          f"coord ({result.x[peak_i]:+.3e}, {result.y[peak_j]:+.3e}) m")
    print(f"center index   : {result.center}, coordinate there = "
          f"({result.x[ci]:+.1e}, {result.y[cj]:+.1e}) m")
    print(f"strongest other bin / peak = {second / peak:.4%} (off-axis content is real)")
    print(f"input power    : {result.input_power:.12e}")
    print(f"output power   : {result.output_power:.12e}")
    print(f"relative power error (Parseval check) : {result.power_error:+.3e}")
    print(f"elapsed        : {elapsed*1000:.1f} ms")

    t0 = time.perf_counter()
    energies = fd.integrate_pixel_energies(
        result,
        tolerance=5e-2,
        min_subdivisions=2,
        max_subdivisions=4,
    )
    block = fd.integrate_region_energy(
        result,
        ci,
        ci + 2,
        cj,
        cj + 2,
        tolerance=5e-2,
        min_subdivisions=2,
        max_subdivisions=4,
    )
    energy_elapsed = time.perf_counter() - t0
    tiled = sum(
        energies.energy[i][j]
        for i in range(ci, ci + 2)
        for j in range(cj, cj + 2)
    )
    center_area_energy = energies.energy[ci][cj]
    center_point_proxy = result.intensity[ci][cj] * result.du**2
    print()
    print("=== Finite-area detector energy ===")
    print(f"center pixel energy            : {center_area_energy:.6e} J proxy")
    print(f"center-point * area (not used) : {center_point_proxy:.6e} J proxy")
    print(f"center relative refinement     : {energies.relative_error[ci][cj]:.3e}")
    print(f"center converged               : {energies.converged[ci][cj]}")
    print(f"2x2 tiled energy               : {tiled:.6e}")
    print(f"2x2 direct region energy       : {block.energy:.6e}")
    print(f"tile/region consistency error  : {block.consistency_error:.3e}")
    print(f"region converged               : {block.converged}")
    print(f"integration elapsed            : {energy_elapsed*1000:.1f} ms")

    # ---- A boundary that really triggers -------------------------------------------
    print()
    print("=== Input rejection boundary (actually raised below) ===")
    try:
        # Booleans must never be silently accepted as integer grid sizes.
        fd.fraunhofer(aperture, dx, wavelength, z, pad=True)
    except TypeError as exc:
        print(f"TypeError raised as specified: {exc}")
    else:
        raise AssertionError("expected TypeError for pad=True was not raised")


if __name__ == "__main__":
    main()
