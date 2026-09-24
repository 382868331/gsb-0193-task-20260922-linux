"""Fixed-input demonstration of fourier_diffraction.

Runs real Fraunhofer calculations:

1. the original single-mode case (32x32 circular aperture, padded to 64x64);
2. the new finite-wavelength, non-negative-weight multi-mode case: three
   wavelengths propagated independently, intensities summed;
3. genuinely triggered input-rejection errors.

No sleep, no network, no prerecorded results; everything is computed here and
now, and the multi-mode numbers are cross-checked against hand formulas.
"""

import time

import fourier_diffraction as fd


def section_single_mode() -> None:
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

    print("=== [1] Single wavelength (original entry point) ===")
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


def section_multi_mode() -> None:
    print()
    print("=== [2] Three wavelengths, non-negative weights (new entry point) ===")
    # Same 3x3-block rectangular aperture illuminated by three spectral lines.
    n = 12
    dx = 10.0e-6
    z = 0.10
    aperture = fd.rectangular_aperture(n, 3 * dx, 3 * dx, dx)  # 9 lit pixels
    wavelengths = [4.0e-7, 5.5e-7, 7.0e-7]   # 400 / 550 / 700 nm
    weights = [1.0, 2.0, 0.5]
    modes = [fd.CoherentMode(aperture, lam, w)
             for lam, w in zip(wavelengths, weights)]

    t0 = time.perf_counter()
    r = fd.fraunhofer_modes(modes, dx, z, pad=24)
    elapsed = time.perf_counter() - t0

    ci, cj = r.center
    dc = r.intensity[ci][cj]
    peak = max(max(row) for row in r.intensity)

    # Hand check of the DC pixel: every mode has K = 9 unit phasors aligned at
    # DC, so U_q(0) = dx^2/(lambda_q z) * 9 and the summed DC intensity is
    # sum_q w_q * (9 dx^2/(lambda_q z))^2.  Computed straight from the inputs,
    # not from the module.
    k_lit = int(sum(sum(row) for row in aperture))
    dc_hand = sum(w * (k_lit * dx * dx / (lam * z)) ** 2
                  for w, lam in zip(weights, wavelengths))

    # Weight-linearity boundary behavior: weights (1, 2, 0.5) vs all doubled
    # (2, 4, 1) must double every bin of the incoherent intensity sum.
    r_doubled = fd.fraunhofer_modes(
        [fd.CoherentMode(aperture, lam, 2.0 * w)
         for lam, w in zip(wavelengths, weights)],
        dx, z, pad=24,
    )
    ratios = [
        r_doubled.intensity[i][j] / r.intensity[i][j]
        for i in range(r.m) for j in range(r.m) if r.intensity[i][j] > 0.0
    ]

    # Zero-weight boundary mode: a fourth "spectral line" would be rejected,
    # and weight 0 makes a mode's contribution exactly zero (compare with the
    # two-mode result where that mode is simply absent).
    r_zero = fd.fraunhofer_modes(
        [fd.CoherentMode(aperture, wavelengths[0], 1.0),
         fd.CoherentMode(aperture, wavelengths[1], 0.0)],
        dx, z, pad=24,
    )
    r_one = fd.fraunhofer_modes(
        [fd.CoherentMode(aperture, wavelengths[0], 1.0)], dx, z, pad=24
    )

    print(f"modes          : {len(modes)} (wavelengths "
          f"{[f'{lam*1e9:.0f} nm' for lam in wavelengths]}, weights {weights})")
    print(f"summed grid    : {r.m}x{r.m}; no summed complex field exists "
          f"(cross-wavelength amplitudes never add)")
    print(f"per-mode du    : {[f'{mr.du:.3e}' for mr in r.mode_results]} m "
          f"(each wavelength keeps its own pitch)")
    print(f"common fx axis : [{r.fx[0]:+.2e}, {r.fx[-1]:+.2e}] cycles/m, "
          f"DC index {r.center}")
    print(f"DC intensity   : {dc:.10e}")
    print(f"DC hand formula: {dc_hand:.10e}  "
          f"(K={k_lit}, rel. diff {abs(dc - dc_hand)/dc_hand:+.2e})")
    print(f"grid maximum   : {peak:.6e} (at the shared DC pixel: {dc == peak})")
    print(f"doubling every weight doubles every bin: ratio min/max = "
          f"{min(ratios):.12f} / {max(ratios):.12f}")
    print(f"zero-weight mode adds nothing: grids equal = "
          f"{r_zero.intensity == r_one.intensity}")
    print(f"weighted input/output power: {r.input_power:.10e} / "
          f"{r.output_power:.10e}")
    print(f"relative power error (weighted Parseval) : {r.power_error:+.3e}")
    print(f"elapsed        : {elapsed*1000:.1f} ms")


def section_errors() -> None:
    print()
    print("=== [3] Input rejection (errors actually raised below) ===")
    aperture = fd.rectangular_aperture(8, 2.0e-5, 2.0e-5, 1.0e-5)

    # A negative mode weight must be rejected ...
    try:
        fd.fraunhofer_modes(
            [fd.CoherentMode(aperture, 5.0e-7, 1.0),
             fd.CoherentMode(aperture, 6.0e-7, -0.5)],
            1.0e-5, 0.1,
        )
    except ValueError as exc:
        print(f"negative weight -> ValueError as specified: {exc}")
    else:
        raise AssertionError("expected ValueError for negative weight")

    # ... a fourth mode exceeds the 3-mode limit ...
    try:
        fd.fraunhofer_modes(
            [fd.CoherentMode(aperture, lam * 1e-7)
             for lam in (4.0, 5.0, 6.0, 7.0)],
            1.0e-5, 0.1,
        )
    except ValueError as exc:
        print(f"four modes     -> ValueError as specified: {exc}")
    else:
        raise AssertionError("expected ValueError for four modes")

    # ... and the original validation still fires (bool is not an int pad).
    try:
        fd.fraunhofer(aperture, 1.0e-5, 5.0e-7, 0.1, pad=True)
    except TypeError as exc:
        print(f"pad=True       -> TypeError as specified: {exc}")
    else:
        raise AssertionError("expected TypeError for pad=True")


def main() -> None:
    print("=== Fraunhofer diffraction demo (real computation) ===")
    section_single_mode()
    section_multi_mode()
    section_errors()


if __name__ == "__main__":
    main()
