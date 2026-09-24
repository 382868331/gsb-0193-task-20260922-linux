"""Fixed-input demonstration of fourier_diffraction.

Part 1: one real single-wavelength Fraunhofer calculation (32x32 circular
aperture, zero-padded to 64x64) and one genuinely triggered input rejection.

Part 2: finite-bandwidth, multi-wavelength incoherent mode propagation via
``fraunhofer_modes`` (3 wavelength groups, up to 2 modes each, non-negative
weights), an independent hand-computed intensity check for centered single
pixels, and two genuinely triggered specification errors.

No sleep, no network, no prerecorded results; everything is computed here and
now and finishes in well under a second.
"""

import time

import fourier_diffraction as fd


def single_wavelength_demo() -> None:
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

    print("=== [1] Single-wavelength Fraunhofer demo (real computation) ===")
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

    # ---- A boundary that really triggers -------------------------------------------
    print()
    print("--- input rejection boundary (actually raised below) ---")
    try:
        # Booleans must never be silently accepted as integer grid sizes.
        fd.fraunhofer(aperture, dx, wavelength, z, pad=True)
    except TypeError as exc:
        print(f"TypeError raised as specified: {exc}")
    else:
        raise AssertionError("expected TypeError for pad=True was not raised")


def multimode_demo() -> None:
    print()
    print("=== [2] Multi-wavelength / multi-mode incoherent demo ===")
    # Three wavelength groups (the contractual maximum), 1-2 modes each.
    # Every mode is propagated independently; only weighted intensities stack.
    dx = 10.0e-6
    z = 0.10
    m = 32
    groups = [
        {"wavelength": 0.45e-6, "modes": [
            {"aperture": fd.circular_aperture(24, 4.0 * dx, dx), "weight": 1.0},
            {"aperture": fd.rectangular_aperture(16, 3.0 * dx, 2.0 * dx, dx),
             "weight": 0.4},
        ]},
        {"wavelength": 0.55e-6, "modes": [
            {"aperture": fd.circular_aperture(20, 2.5 * dx, dx), "weight": 0.7},
        ]},
        {"wavelength": 0.65e-6, "modes": [
            {"aperture": fd.rectangular_aperture(20, 2.0 * dx, 5.0 * dx, dx),
             "weight": 0.25},
            {"aperture": fd.circular_aperture(12, 1.5 * dx, dx), "weight": 0.0},
        ]},
    ]

    t0 = time.perf_counter()
    r = fd.fraunhofer_modes(groups, dx, z, pad=m)
    elapsed = time.perf_counter() - t0

    peak, peak_i, peak_j = 0.0, 0, 0
    for i, row in enumerate(r.intensity):
        for j, v in enumerate(row):
            if v > peak:
                peak, peak_i, peak_j = v, i, j

    print(f"wavelengths    : {[f'{w*1e9:.0f} nm' for w in r.wavelengths]}, "
          f"{len(r.components)} modes total")
    print(f"common grid    : {r.m}x{r.m} DFT bins, center {r.center}")
    print("per-mode detail (independent propagation; own physical pitch):")
    for k, comp in enumerate(r.components):
        print(
            f"  mode {k}: lam={comp.wavelength*1e9:4.0f} nm  w={comp.weight:4.2f}  "
            f"du={comp.result.du*1e6:6.3f} um  "
            f"power_err={comp.result.power_error:+.1e}  "
            f"weighted_power={comp.weighted_power:.6e}"
        )
    print(f"sum of weights : {r.total_weight:.3f} (informational; no normalization)")
    print(f"detector power : {r.detector_power:.6e} = sum of weighted powers above")
    print(f"combined peak  : intensity = {peak:.6e} at bin ({peak_i}, {peak_j})")
    print(f"elapsed        : {elapsed*1000:.1f} ms")

    # ---- Independent hand check: centered unit pixels --------------------------------
    # A centered unit pixel has DFT exactly 1 in every bin, so each mode gives
    # constant intensity (dx^2/(lam*z))^2 and the detector reads the plain
    # weighted sum.  This arithmetic is written out independently of fd.
    print()
    print("--- hand-computed check (centered single pixels, 3 wavelengths) ---")
    wl_w = [(0.4e-6, 1.0), (0.5e-6, 2.0), (0.8e-6, 0.25)]
    check_groups = [
        {"wavelength": wl, "modes": [{"aperture": [[1.0]], "weight": w}]}
        for wl, w in wl_w
    ]
    check = fd.fraunhofer_modes(check_groups, dx, z, pad=8)
    hand_value = 0.0
    for wl, w in wl_w:
        hand_value += w * (dx * dx / (wl * z)) ** 2
    worst = max(abs(v - hand_value) for row in check.intensity for v in row)
    print(f"hand constant  : sum_g w_g*(dx^2/(lam_g*z))^2 = {hand_value:.10e}")
    print(f"detector range : [{min(min(row) for row in check.intensity):.10e}, "
          f"{max(max(row) for row in check.intensity):.10e}]")
    print(f"max deviation  : {worst:.2e} (exact up to floating-point round-off)")
    assert worst <= 1e-20 * hand_value, "hand-computed mode sum mismatch"

    # ---- Specification errors that really trigger ------------------------------------
    print()
    print("--- multimode rejection boundaries (actually raised below) ---")
    try:
        # Negative weights are forbidden: intensities must not be subtracted.
        fd.fraunhofer_modes(
            [{"wavelength": 0.5e-6,
              "modes": [{"aperture": [[1.0]], "weight": -0.5}]}],
            dx, z,
        )
    except ValueError as exc:
        print(f"ValueError for negative weight: {exc}")
    else:
        raise AssertionError("expected ValueError for negative weight was not raised")

    try:
        # A 4th wavelength group exceeds the contractual 3-group bound.
        fd.fraunhofer_modes(
            [(wl, [[[1.0]]]) for wl in (0.4e-6, 0.5e-6, 0.6e-6, 0.7e-6)],
            dx, z,
        )
    except ValueError as exc:
        print(f"ValueError for 4 wavelength groups: {exc}")
    else:
        raise AssertionError("expected ValueError for 4 groups was not raised")


def main() -> None:
    single_wavelength_demo()
    multimode_demo()


if __name__ == "__main__":
    main()
