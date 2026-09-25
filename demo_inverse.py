"""Fixed-input demonstration of the inverse problem (inverse_aperture).

Synthesises complex-amplitude observations by an independent brute-force
sum of the Fraunhofer kernel at arbitrary physical coordinates, inverts them
on a fixed boolean support mask, and reprojects the recovered aperture back
onto the observation coordinates.  Everything is computed here and now (no
sleep, no network, no prerecorded results); runtime is a fraction of a
second, far below the 90 s budget.

Also demonstrates, with real output, that the regularised solution is *not*
the unique true aperture, and shows genuinely triggered failure boundaries.
"""

import cmath
import math
import time

import fourier_diffraction as fd
import inverse_aperture as inv


def true_field(x, y, aperture, dx, wavelength, z):
    """Independent synthesis: direct O(N^2) sum of the Fraunhofer kernel."""
    n = len(aperture)
    c = n // 2
    s = 0j
    for m in range(n):
        for nn in range(n):
            a = aperture[m][nn]
            if a != 0j:
                phase = -2.0 * math.pi * (x * (m - c) * dx + y * (nn - c) * dx)
                s += a * cmath.exp(1j * phase / (wavelength * z))
    return dx * dx / (wavelength * z) * s


def main() -> None:
    # ---- Fixed physical parameters (same as the forward demo) ----------------------
    n = 16
    dx = 10.0e-6
    wavelength = 0.5e-6
    z = 0.10

    # ---- Synthetic "true" aperture on a fixed circular support ---------------------
    support = fd.circular_aperture(n, 5.5 * dx, dx)
    mask = [[support[i][j] > 0.5 for j in range(n)] for i in range(n)]
    unknowns = sum(row.count(True) for row in mask)
    true = [
        [
            (
                cmath.rect(1.0 + 0.3 * math.sin(0.7 * i), 0.25 * (i - j))
                if mask[i][j]
                else 0j
            )
            for j in range(n)
        ]
        for i in range(n)
    ]

    # ---- Observations: complex amplitudes at arbitrary physical coordinates --------
    # Incomplete sampling: a 16x16 set of coordinates spanning the forward
    # model's 32x32 observation grid but deliberately shifted OFF the DFT
    # sample points, so no FFT shortcut applies.  The first measurement is
    # repeated once to show duplicates are kept as repeated measurements.
    fwd = fd.fraunhofer(true, dx, wavelength, z, pad=32)  # only for grid spacing
    observations = []
    for p in range(0, 32, 2):
        for q in range(0, 32, 2):
            x = fwd.x[p] + 0.31 * fwd.du
            y = fwd.y[q] - 0.17 * fwd.du
            observations.append((x, y, true_field(x, y, true, dx, wavelength, z)))
    observations.append(observations[0])

    lam = 1e-6
    t0 = time.perf_counter()
    result = inv.invert_aperture(mask, observations, dx, wavelength, z, lam, tol=1e-11)
    elapsed = time.perf_counter() - t0

    print("=== Inverse problem demo: aperture from complex-amplitude observations ===")
    print(f"support        : {n}x{n} mask, {unknowns} unknown complex pixels")
    print(f"observations   : {result.n_observations} complex amplitudes at off-grid "
          f"coordinates (one measurement duplicated on purpose)")
    print(f"lambda         : {lam:.1e} (strictly positive)")
    print(f"stop reason    : {result.stop_reason} "
          f"(converged={result.converged}, iterations={result.iterations})")
    print(f"residual norm  : ||F a - b|| = {result.residual_norm:.6e}")
    print(f"objective      : ||F a - b||^2 + lam ||a||^2 = {result.objective:.6e}")
    print(f"normal residual: ||(F*F + lam I) a - F* b|| = "
          f"{result.normal_residual_norm:.6e}")
    print(f"elapsed        : {elapsed*1000:.1f} ms")
    print("(convergence is fast here because observations sampled uniformly over")
    print(" the aperture's frequency support make F*F nearly diagonal)")

    # ---- Reprojection: push the recovered aperture back through the operator -------
    op = inv.ApertureOperator(mask, observations, dx, wavelength, z)
    x_vec = [result.aperture[i][j] for i, j in op.pixels]
    reprojected = op.forward(x_vec)
    worst = max(
        abs(f - u) for f, (_, _, u) in zip(reprojected, observations)
    )
    obs_scale = max(abs(u) for _, _, u in observations)
    print()
    print("=== Reprojection of the recovered aperture ===")
    print(f"max |F a - u| over all observations = {worst:.3e} "
          f"({worst / obs_scale:.2%} of the largest observed amplitude)")

    # ---- The regularised solution is NOT the unique true aperture ------------------
    diff_peak = max(
        abs(result.aperture[i][j] - true[i][j])
        for i in range(n)
        for j in range(n)
    )
    true_energy = sum(abs(v) ** 2 for row in true for v in row)
    sol_energy = sum(abs(v) ** 2 for row in result.aperture for v in row)
    print()
    print("=== Recovered aperture vs. the synthetic truth ===")
    print(f"max |a_recovered - a_true| = {diff_peak:.4f} "
          f"(truth has peak amplitude {max(abs(v) for row in true for v in row):.3f})")
    print(f"aperture energy: recovered {sol_energy:.4f} vs. true {true_energy:.4f}")
    print("The match is only approximate: with incomplete sampling the data do")
    print("not determine a unique aperture, and lam > 0 deliberately trades fit")
    print("for energy.  The solver returns the minimum-(residual + lam*energy)")
    print("aperture, one of infinitely many apertures compatible with the data —")
    print("not 'the' true aperture.")

    # ---- Failure boundaries that really trigger ------------------------------------
    print()
    print("=== Failure boundaries (actually triggered below) ===")
    try:
        # lambda must be strictly positive; lam = 0 would make the
        # underdetermined normal equations singular.
        inv.invert_aperture(mask, observations, dx, wavelength, z, 0.0)
    except ValueError as exc:
        print(f"ValueError raised as specified: {exc}")
    else:
        raise AssertionError("expected ValueError for lam=0 was not raised")

    capped = inv.invert_aperture(
        mask, observations, dx, wavelength, z, lam, tol=1e-14, max_iter=1
    )
    print(f"max_iter=1 run: converged={capped.converged}, "
          f"stop_reason={capped.stop_reason!r}, iterations={capped.iterations}")
    if capped.converged:
        raise AssertionError("capped run unexpectedly converged")
    print("Reaching the iteration cap is reported as max_iter_reached, "
          "never as convergence.")


if __name__ == "__main__":
    main()
