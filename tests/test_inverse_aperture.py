"""Required tests for inverse_aperture.

Expected values come from references implemented directly in this file: an
explicit dense Fourier matrix with a brute-force complex Gaussian elimination
for the normal equations, and hand-computed inner products.  The module under
test never generates its own expected values; consistency with the forward
model (fourier_diffraction) appears as a supplementary check only.
"""

import cmath
import math
import unittest

import fourier_diffraction as fd
import inverse_aperture as inv


# ---------------------------------------------------------------------------
# Independent references (must not call the module's operator or solver)
# ---------------------------------------------------------------------------


def dense_matrix(mask, observations, dx, wavelength, z):
    """Explicit dense Fourier matrix, built term by term from the definition."""
    n = len(mask)
    c = n // 2
    scale = dx * dx / (wavelength * z)
    pixels = [(i, j) for i in range(n) for j in range(n) if mask[i][j]]
    rows = []
    for x, y, _u in observations:
        row = []
        for m, nn in pixels:
            phase = -2.0 * math.pi * (x * (m - c) * dx + y * (nn - c) * dx) / (wavelength * z)
            row.append(scale * cmath.exp(1j * phase))
        rows.append(row)
    return rows, pixels


def matvec(a, x):
    return [sum(aij * xj for aij, xj in zip(row, x)) for row in a]


def solve_dense(a_rows, b, lam):
    """Solve (A* A + lam I) x = A* b by brute-force Gaussian elimination."""
    k = len(a_rows[0]) if a_rows else 0
    p = len(a_rows)
    # Normal equations, accumulated element by element.
    h = [[0j] * k for _ in range(k)]
    c = [0j] * k
    for i in range(p):
        row = a_rows[i]
        for s in range(k):
            c[s] += row[s].conjugate() * b[i]
            for t in range(k):
                h[s][t] += row[s].conjugate() * row[t]
    for s in range(k):
        h[s][s] += lam
    # Gaussian elimination with partial pivoting (complex arithmetic).
    aug = [row[:] + [c[i]] for i, row in enumerate(h)]
    for col in range(k):
        piv = max(range(col, k), key=lambda r: abs(aug[r][col]))
        aug[col], aug[piv] = aug[piv], aug[col]
        pivot = aug[col][col]
        for r in range(k):
            if r == col:
                continue
            factor = aug[r][col] / pivot
            if factor == 0j:
                continue
            for cc in range(col, k + 1):
                aug[r][cc] -= factor * aug[col][cc]
    return [aug[i][k] / aug[i][i] for i in range(k)]


def inner(u, v):
    return sum((a.conjugate() * b for a, b in zip(u, v)), 0j)


def det_mask(n, lit_mod=3):
    """Deterministic boolean mask with a non-trivial shape."""
    return [[(i * i + j * 3 + i * j) % lit_mod != 0 for j in range(n)] for i in range(n)]


def det_observations(count, dx, wavelength, z, n, seed_shift=0.0):
    """Deterministic, non-grid-aligned observation coordinates and amplitudes."""
    obs = []
    for i in range(count):
        x = ((i * 7 % 11) - 5) * 0.37 * wavelength * z / (n * dx) + seed_shift
        y = ((i * 5 % 9) - 4) * 0.41 * wavelength * z / (n * dx) - 2 * seed_shift
        u = cmath.rect(0.5 + 0.1 * (i % 4), 0.3 * i)
        obs.append((x, y, u))
    return obs


DX, WAVELENGTH, Z = 1e-5, 5e-7, 0.1


# ---------------------------------------------------------------------------
# Adjoint consistency (complex inner product)
# ---------------------------------------------------------------------------


class TestAdjointConsistency(unittest.TestCase):
    def test_adjoint_is_conjugate_transpose(self):
        n = 6
        mask = det_mask(n)
        obs = det_observations(9, DX, WAVELENGTH, Z, n)
        op = inv.ApertureOperator(mask, obs, DX, WAVELENGTH, Z)

        x = [cmath.rect(1.0 + 0.1 * t, 0.77 * t) for t in range(op.n_unknowns)]
        r = [cmath.rect(0.3 + 0.2 * i, -0.53 * i) for i in range(op.n_observations)]

        lhs = inner(op.forward(x), r)   # <F x, r>
        rhs = inner(x, op.adjoint(r))   # <x, F* r>
        scale = max(1.0, abs(lhs))
        self.assertLess(abs(lhs - rhs), 1e-12 * scale)

    def test_adjoint_matches_explicit_dense_matrix(self):
        n = 4
        mask = det_mask(n)
        obs = det_observations(7, DX, WAVELENGTH, Z, n)
        op = inv.ApertureOperator(mask, obs, DX, WAVELENGTH, Z)
        a_rows, _ = dense_matrix(mask, obs, DX, WAVELENGTH, Z)

        x = [cmath.rect(0.4 + 0.05 * t, 1.1 * t) for t in range(op.n_unknowns)]
        got_f = op.forward(x)
        want_f = matvec(a_rows, x)
        for g, w in zip(got_f, want_f):
            self.assertLess(abs(g - w), 1e-12 * max(1.0, abs(w)))

        r = [cmath.rect(0.7, 0.9 * i) for i in range(op.n_observations)]
        got_a = op.adjoint(r)
        want_a = [
            sum(a_rows[i][t].conjugate() * r[i] for i in range(len(obs)))
            for t in range(op.n_unknowns)
        ]
        for g, w in zip(got_a, want_a):
            self.assertLess(abs(g - w), 1e-12 * max(1.0, abs(w)))


# ---------------------------------------------------------------------------
# Consistency with the original forward propagation (supplementary)
# ---------------------------------------------------------------------------


class TestForwardModelConsistency(unittest.TestCase):
    def test_operator_reproduces_fraunhofer_grid_samples(self):
        n, m = 8, 16
        aperture = [
            [cmath.rect(0.5 + 0.1 * ((i + j) % 3), 0.4 * (i - j)) for j in range(n)]
            for i in range(n)
        ]
        mask = [[True] * n for _ in range(n)]
        result = fd.fraunhofer(aperture, DX, WAVELENGTH, Z, pad=m)

        obs = []
        for p in range(m):
            for q in range(m):
                obs.append((result.x[p], result.y[q], 0j))
        op = inv.ApertureOperator(mask, obs, DX, WAVELENGTH, Z)
        x = [aperture[i][j] for i in range(n) for j in range(n)]
        got = op.forward(x)
        want = [result.field[p][q] for p in range(m) for q in range(m)]
        peak = max(abs(v) for v in want)
        for g, w in zip(got, want):
            self.assertLess(abs(g - w), 1e-10 * peak)


# ---------------------------------------------------------------------------
# Small-aperture comparison against an explicit dense solve
# ---------------------------------------------------------------------------


class TestExplicitDenseComparison(unittest.TestCase):
    def test_cg_matches_dense_normal_equation_solve(self):
        n = 4
        mask = det_mask(n)
        obs = det_observations(11, DX, WAVELENGTH, Z, n, seed_shift=1e-4)
        lam = 1e-6

        result = inv.invert_aperture(mask, obs, DX, WAVELENGTH, Z, lam, tol=1e-13)
        self.assertTrue(result.converged, msg=result.stop_reason)

        a_rows, pixels = dense_matrix(mask, obs, DX, WAVELENGTH, Z)
        want = solve_dense(a_rows, [u for _, _, u in obs], lam)
        got = [result.aperture[i][j] for i, j in pixels]
        scale = max(1.0, max(abs(v) for v in want))
        for g, w in zip(got, want):
            self.assertLess(abs(g - w), 1e-8 * scale)

        # Pixels outside the mask stay exactly zero.
        for i in range(n):
            for j in range(n):
                if not mask[i][j]:
                    self.assertEqual(result.aperture[i][j], 0j)

    def test_returned_quantities_are_self_consistent(self):
        n = 5
        mask = det_mask(n)
        obs = det_observations(13, DX, WAVELENGTH, Z, n)
        lam = 3e-7
        result = inv.invert_aperture(mask, obs, DX, WAVELENGTH, Z, lam, tol=1e-12)
        self.assertTrue(result.converged, msg=result.stop_reason)

        # residual == F a - b, recomputed by hand from the dense matrix.
        a_rows, pixels = dense_matrix(mask, obs, DX, WAVELENGTH, Z)
        x = [result.aperture[i][j] for i, j in pixels]
        want_residual = [f - u for f, (_, _, u) in zip(matvec(a_rows, x), obs)]
        for g, w in zip(result.residual, want_residual):
            self.assertLess(abs(g - w), 1e-12 * max(1.0, abs(w)))

        want_rnorm = math.sqrt(sum(abs(v) ** 2 for v in want_residual))
        self.assertAlmostEqual(result.residual_norm, want_rnorm, delta=1e-12 * max(1.0, want_rnorm))
        want_obj = want_rnorm**2 + lam * sum(abs(v) ** 2 for v in x)
        self.assertAlmostEqual(result.objective, want_obj, delta=1e-10 * max(1.0, abs(want_obj)))

        # Normal-equation residual at the reported solution is genuinely small.
        self.assertLess(result.normal_residual_norm, 1e-8 * max(1.0, result.residual_norm))


# ---------------------------------------------------------------------------
# Edge cases: incomplete sampling, zero observations, empty mask, lambda
# ---------------------------------------------------------------------------


class TestEdgeCases(unittest.TestCase):
    def test_incomplete_sampling_still_solves(self):
        # Fewer observations than unknowns: lam > 0 keeps the system positive
        # definite; the minimum-norm regularised solution must still satisfy
        # the normal equations.
        n = 5
        mask = [[True] * n for _ in range(n)]  # 25 unknowns
        obs = det_observations(6, DX, WAVELENGTH, Z, n)  # 6 measurements
        lam = 1e-6
        result = inv.invert_aperture(mask, obs, DX, WAVELENGTH, Z, lam, tol=1e-12)
        self.assertTrue(result.converged, msg=result.stop_reason)
        self.assertEqual(result.n_unknowns, 25)
        self.assertEqual(result.n_observations, 6)
        self.assertLess(result.normal_residual_norm, 1e-8)

        a_rows, _ = dense_matrix(mask, obs, DX, WAVELENGTH, Z)
        want = solve_dense(a_rows, [u for _, _, u in obs], lam)
        got = [result.aperture[i][j] for i in range(n) for j in range(n)]
        scale = max(1.0, max(abs(v) for v in want))
        for g, w in zip(got, want):
            self.assertLess(abs(g - w), 1e-8 * scale)

    def test_zero_observations_gives_zero_aperture(self):
        mask = det_mask(4)
        result = inv.invert_aperture(mask, [], DX, WAVELENGTH, Z, 1e-6)
        self.assertTrue(result.converged)
        self.assertEqual(result.stop_reason, "no_observations")
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.residual, [])
        self.assertEqual(result.residual_norm, 0.0)
        self.assertEqual(result.objective, 0.0)
        self.assertEqual(result.normal_residual_norm, 0.0)
        for row in result.aperture:
            for v in row:
                self.assertEqual(v, 0j)

    def test_empty_mask_gives_zero_aperture_and_full_residual(self):
        n = 4
        mask = [[False] * n for _ in range(n)]
        obs = det_observations(5, DX, WAVELENGTH, Z, n)
        result = inv.invert_aperture(mask, obs, DX, WAVELENGTH, Z, 1e-6)
        self.assertTrue(result.converged)
        self.assertEqual(result.stop_reason, "empty_support")
        self.assertEqual(result.n_unknowns, 0)
        for row in result.aperture:
            for v in row:
                self.assertEqual(v, 0j)
        # With no degrees of freedom the residual is exactly -b.
        for r, (_, _, u) in zip(result.residual, obs):
            self.assertEqual(r, -u)
        want = math.sqrt(sum(abs(u) ** 2 for _, _, u in obs))
        self.assertAlmostEqual(result.residual_norm, want, delta=1e-15 * want)
        self.assertAlmostEqual(result.objective, want * want, delta=1e-15 * want * want)

    def test_lambda_trades_residual_against_energy(self):
        n = 5
        mask = det_mask(n)
        obs = det_observations(10, DX, WAVELENGTH, Z, n)
        small = inv.invert_aperture(mask, obs, DX, WAVELENGTH, Z, 1e-9, tol=1e-12)
        large = inv.invert_aperture(mask, obs, DX, WAVELENGTH, Z, 1e-2, tol=1e-12)
        self.assertTrue(small.converged and large.converged)

        def energy(res):
            return sum(abs(v) ** 2 for row in res.aperture for v in row)

        # Larger lambda must shrink the aperture energy and grow the data
        # residual; each solution must beat the other solution's objective
        # under its own lambda (optimality of the reported minimiser).
        self.assertLess(energy(large), energy(small))
        self.assertGreater(large.residual_norm, small.residual_norm)
        obj_small_at_large = large.residual_norm**2 + 1e-9 * energy(large)
        obj_large_at_small = small.residual_norm**2 + 1e-2 * energy(small)
        self.assertLess(small.objective, obj_small_at_large + 1e-15)
        self.assertLess(large.objective, obj_large_at_small + 1e-15)

    def test_duplicate_observations_are_kept_as_repeated_measurements(self):
        n = 4
        mask = det_mask(n)
        base = det_observations(5, DX, WAVELENGTH, Z, n)
        doubled = base + base[:2]  # repeat the first two measurements
        lam = 1e-6
        r1 = inv.invert_aperture(mask, base, DX, WAVELENGTH, Z, lam, tol=1e-12)
        r2 = inv.invert_aperture(mask, doubled, DX, WAVELENGTH, Z, lam, tol=1e-12)
        self.assertTrue(r1.converged and r2.converged)

        # Duplicates are not deduplicated: residual keeps one entry per row.
        self.assertEqual(len(r2.residual), len(doubled))
        self.assertEqual(r2.n_observations, len(doubled))
        # The repeated rows reappear verbatim in the residual.
        self.assertEqual(r2.residual[5], r2.residual[0])
        self.assertEqual(r2.residual[6], r2.residual[1])
        # And the weighting actually changes the minimiser.
        diff = max(
            abs(a - b)
            for row_a, row_b in zip(r1.aperture, r2.aperture)
            for a, b in zip(row_a, row_b)
        )
        self.assertGreater(diff, 1e-12)

    def test_max_iter_cap_is_not_reported_as_convergence(self):
        n = 5
        mask = det_mask(n)
        obs = det_observations(10, DX, WAVELENGTH, Z, n)
        result = inv.invert_aperture(
            mask, obs, DX, WAVELENGTH, Z, 1e-9, tol=1e-14, max_iter=1
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.stop_reason, "max_iter_reached")
        self.assertEqual(result.iterations, 1)
        # Residual/objective are still honest values at the capped iterate.
        self.assertGreater(result.normal_residual_norm, 0.0)


# ---------------------------------------------------------------------------
# Recovery round-trip (supplementary; uses the forward model for synthesis)
# ---------------------------------------------------------------------------


class TestRecovery(unittest.TestCase):
    def test_full_sampling_recovers_aperture(self):
        n = 6
        mask = det_mask(n)
        true = [
            [cmath.rect(0.8, 0.31 * (i - j)) if mask[i][j] else 0j for j in range(n)]
            for i in range(n)
        ]
        # Synthesise observations with the *forward* module on its own grid.
        fwd = fd.fraunhofer(true, DX, WAVELENGTH, Z, pad=16)
        obs = [
            (fwd.x[p], fwd.y[q], fwd.field[p][q])
            for p in range(16)
            for q in range(16)
        ]
        result = inv.invert_aperture(mask, obs, DX, WAVELENGTH, Z, 1e-12, tol=1e-11)
        self.assertTrue(result.converged, msg=result.stop_reason)
        peak = max(abs(v) for row in true for v in row)
        for i in range(n):
            for j in range(n):
                self.assertLess(abs(result.aperture[i][j] - true[i][j]), 1e-4 * peak)


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------


class TestErrors(unittest.TestCase):
    def setUp(self):
        self.mask = det_mask(4)
        self.obs = det_observations(3, DX, WAVELENGTH, Z, 4)

    def test_lambda_must_be_strictly_positive(self):
        for bad in (0.0, -1e-12, float("nan"), float("inf")):
            with self.subTest(lam=bad):
                with self.assertRaises(ValueError) as cm:
                    inv.invert_aperture(self.mask, self.obs, DX, WAVELENGTH, Z, bad)
                self.assertIn("lam", str(cm.exception))
        with self.assertRaises(TypeError):
            inv.invert_aperture(self.mask, self.obs, DX, WAVELENGTH, Z, True)

    def test_mask_must_be_strict_bool_square(self):
        with self.assertRaises(TypeError):
            inv.invert_aperture([[1, 0], [0, 1]], self.obs, DX, WAVELENGTH, Z, 1e-6)
        with self.assertRaises(ValueError):
            inv.invert_aperture([[True] * 3] * 2, self.obs, DX, WAVELENGTH, Z, 1e-6)
        with self.assertRaises(ValueError):
            inv.invert_aperture([], self.obs, DX, WAVELENGTH, Z, 1e-6)
        with self.assertRaises(ValueError):
            inv.invert_aperture(
                [[True] * 33] * 33, self.obs, DX, WAVELENGTH, Z, 1e-6
            )

    def test_observation_validation(self):
        with self.assertRaises(ValueError) as cm:
            inv.invert_aperture(self.mask, [(0.0, 0.0)], DX, WAVELENGTH, Z, 1e-6)
        self.assertIn("observations[0]", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            inv.invert_aperture(
                self.mask, [(float("nan"), 0.0, 1.0)], DX, WAVELENGTH, Z, 1e-6
            )
        self.assertIn("observations[0]", str(cm.exception))
        with self.assertRaises(ValueError):
            inv.invert_aperture(
                self.mask, [(0.0, 0.0, complex(float("inf"), 0.0))],
                DX, WAVELENGTH, Z, 1e-6,
            )
        with self.assertRaises(TypeError):
            inv.invert_aperture(self.mask, [(0.0, 0.0, "1+0j")], DX, WAVELENGTH, Z, 1e-6)

    def test_solver_parameter_validation(self):
        with self.assertRaises(TypeError):
            inv.invert_aperture(
                self.mask, self.obs, DX, WAVELENGTH, Z, 1e-6, max_iter=True
            )
        with self.assertRaises(ValueError):
            inv.invert_aperture(
                self.mask, self.obs, DX, WAVELENGTH, Z, 1e-6, max_iter=0
            )
        with self.assertRaises(ValueError):
            inv.invert_aperture(
                self.mask, self.obs, DX, WAVELENGTH, Z, 1e-6, tol=0.0
            )
        with self.assertRaises(ValueError):
            inv.invert_aperture(self.mask, self.obs, 0.0, WAVELENGTH, Z, 1e-6)


if __name__ == "__main__":
    unittest.main()
