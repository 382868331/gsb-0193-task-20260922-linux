"""Fixed-input demonstration of fourier_diffraction + inverse_aperture.

Part 1 runs one real Fraunhofer calculation (32x32 circular aperture,
zero-padded to 64x64).  Part 2 synthesises complex-amplitude observations
with that forward model, reconstructs the aperture on a fixed support mask
with the matrix-free inverse solver, and re-projects the result onto the
observation coordinates.  Part 3 triggers genuine input rejections.  No
sleep, no network, no prerecorded results; everything is computed here and
now.
"""

import cmath
import time

import fourier_diffraction as fd
import inverse_aperture as ia


def forward_demo() -> None:
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


def inverse_demo() -> None:
    # ---- Synthetic ground truth on a fixed rectangular support --------------------
    n = 16
    dx = 10.0e-6
    wavelength = 0.5e-6
    z = 0.10
    support = fd.rectangular_aperture(n, 8 * dx, 10 * dx, dx)  # floats 0.0/1.0
    mask = [[v == 1.0 for v in row] for row in support]
    n_unknowns = sum(sum(row) for row in mask)
    c = n // 2
    true_aperture = [
        [
            (cmath.exp(1j * (0.35 * (i - c) ** 2 - 0.5 * (j - c)))
             * (1.0 + 0.2 * ((i - c) % 3)) if mask[i][j] else 0j)
            for j in range(n)
        ]
        for i in range(n)
    ]

    # ---- Synthetic complex-amplitude observations from the forward model ----------
    m = 32
    fwd = fd.fraunhofer(true_aperture, dx, wavelength, z, pad=m)
    # Incomplete sampling: every second grid point in each axis (256 of 1024),
    # plus 25 off-grid coordinates halfway between grid samples (evaluated
    # with the module's own forward operator on the true aperture).
    observations = [
        (fwd.x[p], fwd.y[q], fwd.field[p][q])
        for p in range(0, m, 2)
        for q in range(0, m, 2)
    ]
    true_coeffs = [true_aperture[i][j] for i in range(n) for j in range(n) if mask[i][j]]
    half = fwd.du / 2.0
    off_grid = [
        (fwd.x[2 * kx] + half, fwd.y[2 * ky + 1] + half, 0j)
        for kx in range(5)
        for ky in range(5)
    ]
    off_vals = ia.forward_operator(true_coeffs, mask, off_grid, dx, wavelength, z)
    observations += [(x, y, v) for (x, y, _), v in zip(off_grid, off_vals)]

    t0 = time.perf_counter()
    res = ia.invert_aperture(observations, mask, dx, wavelength, z, 1e-6, tol=1e-10)
    elapsed = time.perf_counter() - t0

    # Re-project the recovered aperture onto the observation coordinates.
    coeffs = [res.aperture[i][j] for i in range(n) for j in range(n) if mask[i][j]]
    predicted = ia.forward_operator(coeffs, mask, observations, dx, wavelength, z)
    observed = [v for _, _, v in observations]
    max_reproj = max(abs(p - b) for p, b in zip(predicted, observed))
    max_obs = max(abs(b) for b in observed)

    true_flat = [v for row in true_aperture for v in row]
    got_flat = [v for row in res.aperture for v in row]
    diff = max(abs(g - t) for g, t in zip(got_flat, true_flat))
    true_max = max(abs(v) for v in true_flat)

    print()
    print("=== Inverse problem demo (recover + re-project, real computation) ===")
    print(f"support        : {n}x{n} mask, {n_unknowns} unknown complex pixels")
    print(f"observations   : {res.n_observations} complex amplitudes "
          f"(incomplete: every 2nd point of a {m}x{m} grid + 25 off-grid)")
    print(f"lambda         : {res.lam:.1e} (strictly positive Tikhonov weight)")
    print(f"stop reason    : {res.stop_reason} after {res.iterations} CG iterations "
          f"(converged={res.converged})")
    print(f"true residual  : ||F a - b|| = {res.residual_norm:.6e}")
    print(f"objective      : ||F a - b||^2 + lam ||a||^2 = {res.objective:.6e}")
    print(f"normal eq res  : ||F*F a + lam a - F* b|| = {res.normal_residual:.6e}")
    print(f"re-projection  : max |F a - b| = {max_reproj:.3e} "
          f"(max |b| = {max_obs:.3e})")
    print(f"distance to the synthetic true aperture: {diff:.3e} "
          f"(max |true| = {true_max:.3e})")
    print(f"elapsed        : {elapsed*1000:.1f} ms")

    # ---- The regularised solution is not "the" true aperture ----------------------
    res2 = ia.invert_aperture(observations, mask, dx, wavelength, z, 1e-2, tol=1e-10)
    got2 = [v for row in res2.aperture for v in row]
    diff2 = max(abs(g - t) for g, t in zip(got2, true_flat))
    energy1 = sum(abs(v) ** 2 for v in got_flat)
    energy2 = sum(abs(v) ** 2 for v in got2)
    print()
    print("regularisation makes the objective's minimiser unique, but it is NOT")
    print("the unique true aperture: changing lambda changes the solution.")
    print(f"  lam=1e-6: residual={res.residual_norm:.3e}  energy={energy1:.3e}  "
          f"dist-to-truth={diff:.3e}")
    print(f"  lam=1e-2: residual={res2.residual_norm:.3e}  energy={energy2:.3e}  "
          f"dist-to-truth={diff2:.3e}")
    print("  (larger lambda -> smaller aperture energy, larger data residual)")


def boundary_demo() -> None:
    print()
    print("=== Input rejection boundaries (actually raised below) ===")
    aperture = fd.circular_aperture(8, 3.0, 1.0)
    try:
        # Booleans must never be silently accepted as integer grid sizes.
        fd.fraunhofer(aperture, 1e-5, 5e-7, 0.1, pad=True)
    except TypeError as exc:
        print(f"TypeError raised as specified: {exc}")
    else:
        raise AssertionError("expected TypeError for pad=True was not raised")
    try:
        # The regularisation weight must be strictly positive.
        ia.invert_aperture([(0.0, 0.0, 1 + 1j)], [[True]], 1e-5, 5e-7, 0.1, 0.0)
    except ValueError as exc:
        print(f"ValueError raised as specified: {exc}")
    else:
        raise AssertionError("expected ValueError for lam=0.0 was not raised")


def main() -> None:
    forward_demo()
    inverse_demo()
    boundary_demo()


if __name__ == "__main__":
    main()
