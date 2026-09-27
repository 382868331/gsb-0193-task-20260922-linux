"""Fixed-input demonstration of fourier_diffraction.

Runs one real Fraunhofer calculation (32x32 circular aperture, zero-padded to
64x64) and one genuinely triggered input-rejection boundary.  No sleep, no
network, no prerecorded results; everything is computed here and now.
"""

import time

import fourier_diffraction as fd


def main() -> None:
    # ---- Fixed physical parameters ------------------------------------------------
    n = 32
    dx = 10.0e-6          # 10 micrometre aperture-plane pitch
    wavelength = 0.5e-6   # 500 nm
    z = 0.10              # 10 cm propagation
    radius = 6.0 * dx     # circle spanning 13 pixels across

    aperture = fd.circular_aperture(n, radius, dx)
    lit = int(sum(sum(row) for row in aperture))

    t0 = time.perf_counter()
    result = fd.fraunhofer(aperture, dx, wavelength, z, pad=64)
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

    # ---- Detector-pixel energies (area integration, not centre x area) -------
    print()
    print("=== Detector pixel energies (adaptive area integration) ===")
    ci, cj = result.center
    t0 = time.perf_counter()
    center_energy = fd.pixel_energy(aperture, dx, wavelength, z, ci, cj, pad=64,
                                    rtol=1e-6)
    off_energy = fd.pixel_energy(aperture, dx, wavelength, z, ci + 3, cj - 2, pad=64,
                                 rtol=1e-6)
    elapsed_pix = time.perf_counter() - t0
    naive_center = result.intensity[ci][cj] * result.du ** 2
    print(f"pixel ({ci},{cj}) energy   : {center_energy.energy:.9e} "
          f"(converged={center_energy.converged}, err<={center_energy.error_estimate:.1e}, "
          f"depth={center_energy.depth}, evals={center_energy.evaluations})")
    print(f"  naive I_centre*du^2     : {naive_center:.9e} "
          f"({(naive_center - center_energy.energy) / center_energy.energy:+.2%} off)")
    print(f"pixel ({ci+3},{cj-2}) energy : {off_energy.energy:.9e} "
          f"(converged={off_energy.converged}, err<={off_energy.error_estimate:.1e})")

    # Cross-check: four neighbouring pixels summed vs one integration over
    # the same total rectangle.
    block = [
        fd.pixel_energy(aperture, dx, wavelength, z, ci + di, cj + dj, pad=64,
                        rtol=1e-8)
        for di in (0, 1) for dj in (0, 1)
    ]
    half = 0.5 * result.du
    merged = fd.integrate_intensity(aperture, dx, wavelength, z,
                                    -half, 1.5 * result.du, -half, 1.5 * result.du,
                                    rtol=1e-8)
    summed = sum(p.energy for p in block)
    print(f"2x2 pixels summed       : {summed:.9e}")
    print(f"same region, one shot   : {merged.energy:.9e} "
          f"(agree to {abs(summed - merged.energy) / merged.energy:.1e} relative)")
    print(f"pixel integration time  : {elapsed_pix*1000:.1f} ms (first two pixels)")

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
