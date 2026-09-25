"""Required tests for inverse_aperture.

The independent reference here is an *explicit dense* complex normal-equation
solve (kernel matrix built term by term from the documented contract formula,
Gaussian elimination with partial pivoting) implemented in this file.  The
module under test never contributes to its own expected values; the
round-trip against fourier_diffraction.fraunhofer is a supplementary
consistency check.
"""

import cmath
import math
import unittest

import fourier_diffraction as fd
import inverse_aperture as ia


# ---------------------------------------------------------------------------
# Independent explicit references (no use of the module under test)
# ---------------------------------------------------------------------------


def pixel_centres(n, dx):
    """Centred physical pixel coordinates xi_i = (i - n//2) * dx."""
    c = n // 2
    return [(i - c) * dx for i in range(n)]


def explicit_kernel(pixels, obs, n, dx, wavelength, z):
    """Dense kernel matrix K[o][p] from the documented contract formula."""
    scale = dx * dx / (wavelength * z)
    inv_lz = 1.0 / (wavelength * z)
    xi = pixel_centres(n, dx)
    rows = []
    for x, y, _ in obs:
        row = []
        for i, j in pixels:
            phase = 2.0 * math.pi * (x * xi[i] + y * xi[j]) * inv_lz
            row.append(scale * cmath.exp(-1j * phase))
        rows.append(row)
    return rows


def explicit_solve(pixels, obs, n, dx, wavelength, z, lam):
    """Dense normal-equation solve (K* K + lam I) a = K* b.

    Gaussian elimination with partial pivoting on the explicit Hermitian
    system; completely independent of the module's matrix-free CG.
    """
    p = len(pixels)
    k_mat = explicit_kernel(pixels, obs, n, dx, wavelength, z)
    b = [v for _, _, v in obs]
    h = [[0j] * p for _ in range(p)]
    g = [0j] * p
    for row, bv in zip(k_mat, b):
        for q in range(p):
            g[q] += row[q].conjugate() * bv
            for r in range(p):
                h[q][r] += row[q].conjugate() * row[r]
    for q in range(p):
        h[q][q] += lam

    # Solve H a = g by Gaussian elimination with partial pivoting.
    m_aug = [h[q][:] + [g[q]] for q in range(p)]
    for col in range(p):
        piv = max(range(col, p), key=lambda r: abs(m_aug[r][col]))
        m_aug[col], m_aug[piv] = m_aug[piv], m_aug[col]
        pivot = m_aug[col][col]
        for r in range(col + 1, p):
            f = m_aug[r][col] / pivot
            if f != 0j:
                for c in range(col, p + 1):
                    m_aug[r][c] -= f * m_aug[col][c]
    a = [0j] * p
    for r in range(p - 1, -1, -1):
        s = m_aug[r][p]
        for c in range(r + 1, p):
            s -= m_aug[r][c] * a[c]
        a[r] = s / m_aug[r][r]
    return a


def inner_first(u, v):
    """Complex inner product linear in the first argument: sum u * conj(v)."""
    return sum(a * b.conjugate() for a, b in zip(u, v))


def norm(v):
    return math.sqrt(sum(abs(x) ** 2 for x in v))


# Deterministic pseudo-random stream (no external packages).
def det_complex(k):
    return cmath.rect(1.0 + 0.25 * ((k * 7) % 5), 0.613 * k + 0.1 * (k % 3))


# ---------------------------------------------------------------------------
# Adjoint consistency (complex inner product)
# ---------------------------------------------------------------------------


class TestAdjointConsistency(unittest.TestCase):
    def test_adjoint_matches_complex_inner_product(self):
        # <F a, b> == <a, F* b> with <u, v> = sum u * conj(v).
        cases = [
            (4, [(0, 0), (1, 2), (3, 3)], [(1e-3, -2e-3, 1 + 2j), (0.0, 4e-4, 0.5j)]),
            (5, [(i, j) for i in range(5) for j in range(5) if (i + j) % 2 == 0],
             [(1e-4, 1e-4, 1.0), (1e-4, 1e-4, 2.0), (-3e-4, 2e-4, -1j)]),  # duplicate coords kept
            (3, [(2, 2)], []),  # zero observations: both sides are zero
        ]
        dx, wavelength, z = 1e-5, 5e-7, 0.1
        for n, pix, obs in cases:
            with self.subTest(n=n, n_pix=len(pix), n_obs=len(obs)):
                mask = [[(i, j) in pix for j in range(n)] for i in range(n)]
                a = [det_complex(k) for k in range(len(pix))]
                bvals = [det_complex(100 + k) for k in range(len(obs))]
                fa = ia.forward_operator(a, mask, obs, dx, wavelength, z)
                adj_b = ia.adjoint_operator(bvals, mask, obs, dx, wavelength, z)
                lhs = inner_first(fa, bvals)
                rhs = inner_first(a, adj_b)
                scale = max(1.0, abs(lhs), abs(rhs))
                self.assertLess(abs(lhs - rhs), 1e-12 * scale)


# ---------------------------------------------------------------------------
# Small-aperture comparison against the independent explicit dense solve
# ---------------------------------------------------------------------------


class TestExplicitMatrixComparison(unittest.TestCase):
    def test_cg_matches_explicit_dense_solve(self):
        n = 4
        dx, wavelength, z = 2e-5, 6e-7, 0.05
        pixels = [(0, 1), (1, 1), (1, 2), (2, 2), (3, 0), (3, 3)]
        mask = [[(i, j) in pixels for j in range(n)] for i in range(n)]
        obs = [
            (1.3e-4, -2.1e-4, 0.3 + 0.1j),
            (-4.2e-4, 0.0, -0.2 + 0.4j),
            (0.0, 5.5e-4, 0.1 - 0.2j),
            (7.7e-4, 7.7e-4, -0.5 - 0.1j),
            (-1.1e-4, 3.3e-4, 0.05 + 0.6j),
        ]
        lam = 0.37
        res = ia.invert_aperture(obs, mask, dx, wavelength, z, lam, tol=1e-13)
        self.assertTrue(res.converged)
        self.assertEqual(res.stop_reason, "converged")

        want = explicit_solve(pixels, obs, n, dx, wavelength, z, lam)
        got = [res.aperture[i][j] for i, j in pixels]
        scale = max(1.0, max(abs(v) for v in want))
        for g, w in zip(got, want):
            self.assertLess(abs(g - w), 1e-9 * scale)

        # Reported diagnostics must match a direct explicit recomputation.
        k_mat = explicit_kernel(pixels, obs, n, dx, wavelength, z)
        b = [v for _, _, v in obs]
        resid = [sum(row[p] * got[p] for p in range(len(pixels))) - bv
                 for row, bv in zip(k_mat, b)]
        self.assertLess(abs(norm(resid) - res.residual_norm), 1e-9 * max(1.0, norm(resid)))
        energy = sum(abs(v) ** 2 for v in got)
        want_obj = sum(abs(v) ** 2 for v in resid) + lam * energy
        self.assertLess(abs(res.objective - want_obj), 1e-9 * max(1.0, abs(want_obj)))
        # Normal-equation residual of the explicit solution is also tiny.
        self.assertLess(res.normal_residual, 1e-9 * max(1.0, norm(g := [sum(
            row[p].conjugate() * bv for row, bv in zip(k_mat, b)) for p in range(len(pixels))])))

    def test_aperture_zero_outside_mask(self):
        n = 3
        mask = [[True, False, True], [False, False, False], [True, False, False]]
        obs = [(1e-4, 2e-4, 1 + 1j), (-3e-4, 0.0, 0.5)]
        res = ia.invert_aperture(obs, mask, 1e-5, 5e-7, 0.1, 0.1)
        for i in range(n):
            for j in range(n):
                if not mask[i][j]:
                    self.assertEqual(res.aperture[i][j], 0j)


# ---------------------------------------------------------------------------
# Incomplete sampling, zero observations, empty mask, lambda variation
# ---------------------------------------------------------------------------


class TestProblemShapes(unittest.TestCase):
    def test_incomplete_sampling_underdetermined(self):
        # 2 observations, 6 unknowns: only regularisation makes this unique.
        n = 4
        pixels = [(0, 0), (0, 3), (1, 1), (2, 2), (3, 1), (3, 3)]
        mask = [[(i, j) in pixels for j in range(n)] for i in range(n)]
        obs = [(2e-4, -1e-4, 0.4 - 0.2j), (-5e-4, 6e-4, -0.1 + 0.3j)]
        lam = 0.5
        res = ia.invert_aperture(obs, mask, 1e-5, 5e-7, 0.1, lam, tol=1e-12)
        self.assertTrue(res.converged)
        want = explicit_solve(pixels, obs, n, 1e-5, 5e-7, 0.1, lam)
        got = [res.aperture[i][j] for i, j in pixels]
        scale = max(1.0, max(abs(v) for v in want))
        for g, w in zip(got, want):
            self.assertLess(abs(g - w), 1e-8 * scale)
        # lam * I keeps the normal equations positive definite: residual small.
        self.assertLess(res.normal_residual, 1e-8)

    def test_zero_observations_gives_zero_aperture(self):
        mask = [[True, True], [False, True]]
        res = ia.invert_aperture([], mask, 1e-5, 5e-7, 0.1, 0.25)
        self.assertTrue(res.converged)
        self.assertEqual(res.stop_reason, "converged")
        self.assertEqual(res.iterations, 0)
        self.assertEqual(res.residual, [])
        self.assertEqual(res.residual_norm, 0.0)
        self.assertEqual(res.objective, 0.0)
        self.assertEqual(res.normal_residual, 0.0)
        self.assertEqual(res.n_observations, 0)
        self.assertEqual(res.n_unknowns, 3)
        for row in res.aperture:
            for v in row:
                self.assertEqual(v, 0j)

    def test_empty_mask_gives_zero_aperture(self):
        mask = [[False, False], [False, False]]
        obs = [(1e-4, 0.0, 3 + 4j), (0.0, -2e-4, -1.0)]
        res = ia.invert_aperture(obs, mask, 1e-5, 5e-7, 0.1, 1.0)
        self.assertTrue(res.converged)
        self.assertEqual(res.n_unknowns, 0)
        for row in res.aperture:
            for v in row:
                self.assertEqual(v, 0j)
        # No degrees of freedom: residual is exactly -b, objective is ||b||^2.
        self.assertEqual(res.residual, [-(3 + 4j), 1.0 + 0j])
        self.assertAlmostEqual(res.objective, 25.0 + 1.0, delta=1e-12)
        self.assertEqual(res.normal_residual, 0.0)

    def test_lambda_shifts_residual_energy_tradeoff(self):
        n = 4
        pixels = [(0, 0), (1, 1), (2, 2), (3, 3), (1, 2)]
        mask = [[(i, j) in pixels for j in range(n)] for i in range(n)]
        obs = [
            (1e-4, 2e-4, 0.5 + 0.2j),
            (-3e-4, 1e-4, -0.3 + 0.1j),
            (2e-4, -4e-4, 0.2 - 0.4j),
        ]
        args = (obs, mask, 1e-5, 5e-7, 0.1)
        small = ia.invert_aperture(*args, 1e-4)
        large = ia.invert_aperture(*args, 1e2)
        self.assertTrue(small.converged and large.converged)
        energy_small = sum(abs(v) ** 2 for row in small.aperture for v in row)
        energy_large = sum(abs(v) ** 2 for row in large.aperture for v in row)
        # Larger lam suppresses aperture energy and pays with data residual.
        self.assertLess(energy_large, energy_small)
        self.assertGreater(large.residual_norm, small.residual_norm)
        self.assertNotEqual(small.objective, large.objective)


# ---------------------------------------------------------------------------
# Duplicate observations and iteration-cap semantics
# ---------------------------------------------------------------------------


class TestDuplicatesAndStopping(unittest.TestCase):
    def test_duplicate_observations_kept_as_repeated_measurements(self):
        n = 3
        pixels = [(0, 0), (1, 1), (2, 2)]
        mask = [[(i, j) in pixels for j in range(n)] for i in range(n)]
        base = [(-2e-4, 3e-4, -0.2 + 0.1j)]
        dup = [(1e-4, 2e-4, 0.5 - 0.3j)] * 3  # same triple three times
        lam = 0.2
        args = (mask, 1e-5, 5e-7, 0.1)
        res_dup = ia.invert_aperture(base + dup, *args, lam, tol=1e-13)
        res_single = ia.invert_aperture(base + dup[:1], *args, lam, tol=1e-13)

        # Duplicates change the answer (they add weight, not get deduplicated).
        got_dup = [res_dup.aperture[i][j] for i, j in pixels]
        got_single = [res_single.aperture[i][j] for i, j in pixels]
        self.assertGreater(
            max(abs(a - b) for a, b in zip(got_dup, got_single)), 1e-6
        )
        # And the duplicated problem matches the explicit dense solve that
        # simply contains the repeated row three times.
        want = explicit_solve(pixels, base + dup, n, 1e-5, 5e-7, 0.1, lam)
        scale = max(1.0, max(abs(v) for v in want))
        for g, w in zip(got_dup, want):
            self.assertLess(abs(g - w), 1e-9 * scale)
        self.assertEqual(res_dup.n_observations, 4)
        self.assertEqual(len(res_dup.residual), 4)

    def test_iteration_cap_is_not_convergence(self):
        n = 4
        pixels = [(i, j) for i in range(n) for j in range(n)]  # 16 unknowns
        mask = [[True] * n for _ in range(n)]
        # Non-degenerate data so one CG step cannot solve the system.
        obs = [
            (1e-4 * (k - 2), 7e-5 * (k % 3 - 1), det_complex(k))
            for k in range(5)
        ]
        res = ia.invert_aperture(obs, mask, 1e-5, 5e-7, 0.1, 1e-3,
                                 tol=1e-12, max_iterations=1)
        self.assertEqual(res.stop_reason, "max_iterations")
        self.assertFalse(res.converged)
        self.assertEqual(res.iterations, 1)
        # Diagnostics are still honest values for the returned iterate.
        self.assertGreater(res.normal_residual, 0.0)
        # With enough iterations the same problem converges.
        full = ia.invert_aperture(obs, mask, 1e-5, 5e-7, 0.1, 1e-3, tol=1e-12)
        self.assertTrue(full.converged)
        self.assertLess(full.normal_residual, 1e-10)


# ---------------------------------------------------------------------------
# Consistency with the original Fraunhofer forward propagation
# ---------------------------------------------------------------------------


class TestFraunhoferConsistency(unittest.TestCase):
    def test_recover_aperture_from_grid_observations(self):
        # Synthesise observations with the ORIGINAL forward propagation on a
        # padded grid, then invert from those physical coordinates.
        n, m = 8, 16
        dx, wavelength, z = 1e-5, 5e-7, 0.1
        mask = [[True] * n for _ in range(n)]
        true_aperture = [
            [det_complex(10 * i + j) if (i + j) % 3 else 0.5 - 0.25j for j in range(n)]
            for i in range(n)
        ]
        fwd = fd.fraunhofer(true_aperture, dx, wavelength, z, pad=m)
        obs = [
            (fwd.x[p], fwd.y[q], fwd.field[p][q])
            for p in range(m)
            for q in range(m)
        ]
        lam = 1e-12
        res = ia.invert_aperture(obs, mask, dx, wavelength, z, lam, tol=1e-12)
        self.assertTrue(res.converged)
        true_flat = [v for row in true_aperture for v in row]
        got_flat = [v for row in res.aperture for v in row]
        scale = max(abs(v) for v in true_flat)
        err = max(abs(g - t) for g, t in zip(got_flat, true_flat))
        self.assertLess(err, 1e-6 * scale)
        # Re-projection through the public forward operator matches the
        # observed field to the same order.
        pred = ia.forward_operator(got_flat, mask, obs, dx, wavelength, z)
        b = [v for _, _, v in obs]
        self.assertLess(norm([p_ - b_ for p_, b_ in zip(pred, b)]),
                        1e-6 * norm(b))


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------


class TestInverseErrors(unittest.TestCase):
    def setUp(self):
        self.mask = [[True, False], [False, True]]
        self.obs = [(1e-4, 2e-4, 1 + 1j)]

    def test_lambda_must_be_strictly_positive(self):
        for bad in (0.0, -1e-9, -2.0):
            with self.subTest(lam=bad):
                with self.assertRaises(ValueError) as cm:
                    ia.invert_aperture(self.obs, self.mask, 1e-5, 5e-7, 0.1, bad)
                self.assertIn("lam", str(cm.exception))
        with self.assertRaises(ValueError):
            ia.invert_aperture(self.obs, self.mask, 1e-5, 5e-7, 0.1, float("nan"))
        with self.assertRaises(TypeError):
            ia.invert_aperture(self.obs, self.mask, 1e-5, 5e-7, 0.1, True)

    def test_mask_validation(self):
        with self.assertRaises(TypeError) as cm:  # non-bool element
            ia.invert_aperture(self.obs, [[1, 0], [0, 1]], 1e-5, 5e-7, 0.1, 1.0)
        self.assertIn("mask[0][0]", str(cm.exception))
        with self.assertRaises(ValueError):  # not square
            ia.invert_aperture(self.obs, [[True, True, True]], 1e-5, 5e-7, 0.1, 1.0)
        with self.assertRaises(ValueError):  # ragged
            ia.invert_aperture(self.obs, [[True], [True, False]], 1e-5, 5e-7, 0.1, 1.0)
        with self.assertRaises(ValueError):  # too large
            ia.invert_aperture(self.obs, [[True] * 33] * 33, 1e-5, 5e-7, 0.1, 1.0)
        with self.assertRaises(TypeError):
            ia.invert_aperture(self.obs, 1.5, 1e-5, 5e-7, 0.1, 1.0)

    def test_observation_validation(self):
        with self.assertRaises(ValueError) as cm:  # wrong arity
            ia.invert_aperture([(1e-4, 2e-4)], self.mask, 1e-5, 5e-7, 0.1, 1.0)
        self.assertIn("observations[0]", str(cm.exception))
        with self.assertRaises(ValueError):  # NaN coordinate
            ia.invert_aperture([(float("nan"), 0.0, 1j)], self.mask, 1e-5, 5e-7, 0.1, 1.0)
        with self.assertRaises(TypeError):  # bool coordinate
            ia.invert_aperture([(True, 0.0, 1j)], self.mask, 1e-5, 5e-7, 0.1, 1.0)
        with self.assertRaises(ValueError):  # non-finite value
            ia.invert_aperture([(0.0, 0.0, complex(float("inf"), 0))],
                               self.mask, 1e-5, 5e-7, 0.1, 1.0)
        with self.assertRaises(TypeError):
            ia.invert_aperture("not-observations", self.mask, 1e-5, 5e-7, 0.1, 1.0)

    def test_solver_parameter_validation(self):
        with self.assertRaises(ValueError):
            ia.invert_aperture(self.obs, self.mask, 1e-5, 5e-7, 0.1, 1.0, tol=0.0)
        with self.assertRaises(ValueError):
            ia.invert_aperture(self.obs, self.mask, 1e-5, 5e-7, 0.1, 1.0, max_iterations=0)
        with self.assertRaises(TypeError):
            ia.invert_aperture(self.obs, self.mask, 1e-5, 5e-7, 0.1, 1.0, max_iterations=True)
        with self.assertRaises(ValueError):  # physical params still validated
            ia.invert_aperture(self.obs, self.mask, -1e-5, 5e-7, 0.1, 1.0)

    def test_operator_argument_validation(self):
        with self.assertRaises(ValueError) as cm:  # wrong coefficient count
            ia.forward_operator([1j], self.mask, self.obs, 1e-5, 5e-7, 0.1)
        self.assertIn("coefficients", str(cm.exception))
        with self.assertRaises(ValueError):  # wrong value count
            ia.adjoint_operator([1j, 2j], self.mask, self.obs, 1e-5, 5e-7, 0.1)


if __name__ == "__main__":
    unittest.main()
