"""Required tests for fourier_diffraction.

Every numerical expected value below is produced by an *independent*
reference implemented directly in this file (direct geometric-series sums,
brute-force O(n^2) DFT, hand-counted Parseval sums).  The module under test is
never used to generate its own expected values; round-trip checks appear only
as supplementary consistency tests.
"""

import cmath
import math
import unittest

import fourier_diffraction as fd


# ---------------------------------------------------------------------------
# Independent references (must not import or call the core DFT)
# ---------------------------------------------------------------------------


def direct_dft1(a, sign=-1):
    """Brute-force O(n^2) DFT, independent of the module's transform."""
    n = len(a)
    out = []
    for k in range(n):
        s = 0j
        for m, v in enumerate(a):
            s += v * cmath.exp(sign * 2j * math.pi * k * m / n)
        out.append(s)
    return out


def analytic_rect_spectrum(k, l, m0, m1, n0, n1, n):
    """Separable DFT of a centered block of ones.

    Unshifted aperture rows are m in [m0, m1] and columns n in [n0, n1]
    (inclusive, integer contiguous, allowed to be negative); DFT size N.
    Closed-form finite geometric series, no DFT routine involved.
    """

    def axis(q, lo, count):
        if q % n == 0:
            return complex(count)
        w = cmath.exp(-2j * math.pi * q / n)
        return (w ** lo) * (1.0 - w ** count) / (1.0 - w)

    km = m1 - m0 + 1
    kn = n1 - n0 + 1
    return axis(k, m0, km) * axis(l, n0, kn)


def parseval_sum(grid, dx):
    """dx**2 * sum |A|**2 computed element by element from the raw inputs."""
    total = 0.0
    for row in grid:
        for v in row:
            total += float(v.real) ** 2 + float(v.imag) ** 2
    return total * dx * dx


def shifted_index(k, m):
    return (k + m // 2) % m


def reference_field(aperture, dx, wavelength, z, x, y):
    """Independent continuous Fraunhofer field: plain double sum with one
    cmath.exp call per aperture pixel (no phasor power iteration)."""
    n = len(aperture)
    n0 = n // 2
    total = 0j
    for mi in range(n):
        for ni in range(n):
            v = aperture[mi][ni]
            if v:
                phase = -2.0 * math.pi * (
                    x * (mi - n0) + y * (ni - n0)
                ) * dx / (wavelength * z)
                total += v * cmath.exp(1j * phase)
    return total * dx * dx / (wavelength * z)


# 5-point Gauss-Legendre on [-1, 1], literal nodes/weights.
_REF_GL5_T = (0.0, -0.5384693101056831, 0.5384693101056831,
              -0.9061798459386640, 0.9061798459386640)
_REF_GL5_W = (0.5688888888888889, 0.4786286704993665, 0.4786286704993665,
              0.2369268850561891, 0.2369268850561891)


def reference_integral(aperture, dx, wavelength, z, x0, x1, y0, y1, cells=6):
    """Independent fixed-grid 5x5 Gauss-Legendre quadrature: uniform split
    into cells x cells squares, no adaptivity, different order and nodes
    than the 3-point rule used by the module under test."""
    hx = (x1 - x0) / cells
    hy = (y1 - y0) / cells
    total = 0.0
    for cx in range(cells):
        for cy in range(cells):
            mx = x0 + (cx + 0.5) * hx
            my = y0 + (cy + 0.5) * hy
            acc = 0.0
            for tx, wx in zip(_REF_GL5_T, _REF_GL5_W):
                for ty, wy in zip(_REF_GL5_T, _REF_GL5_W):
                    u = reference_field(aperture, dx, wavelength, z,
                                        mx + 0.5 * hx * tx, my + 0.5 * hy * ty)
                    acc += wx * wy * abs(u) ** 2
            total += acc * 0.25 * hx * hy
    return total


# ---------------------------------------------------------------------------
# Core transform tests
# ---------------------------------------------------------------------------


class TestDFT(unittest.TestCase):
    def test_dft1_matches_direct_definition(self):
        # length 1, a prime, a pure power of two and a mixed-radix composite
        for n in (1, 2, 3, 5, 7, 8, 12, 16, 31, 32, 48, 64):
            a = [cmath.rect(1.0, 0.37 * m) + 0.5j * (m % 3) for m in range(n)]
            got = fd._dft1(a, -1)
            want = direct_dft1(a, -1)
            self.assertEqual(len(got), n)
            scale = max(1.0, max(abs(v) for v in want))
            for g, w in zip(got, want):
                self.assertLess(abs(g - w), 1e-11 * scale, msg=f"n={n}")

    def test_inverse_sign_recovers_input(self):
        # Supplementary round-trip consistency only.
        for n in (1, 6, 16, 32):
            a = [complex((m % 4) - 1, (m % 5) - 2) for m in range(n)]
            spec = fd._dft1(a, -1)
            back = [v / n for v in fd._dft1(spec, +1)]
            for x, y in zip(a, back):
                self.assertLess(abs(x - y), 1e-12 * max(1.0, abs(x)))


class TestSinglePixel(unittest.TestCase):
    def test_centered_single_pixel_is_constant_intensity(self):
        n = 16
        dx, wavelength, z = 1e-5, 5e-7, 0.1
        a = [[0j] * n for _ in range(n)]
        a[n // 2][n // 2] = 1.0
        r = fd.fraunhofer(a, dx, wavelength, z)

        scale = dx * dx / (wavelength * z)
        self.assertEqual(r.m, n)
        for row in r.intensity:
            for v in row:
                # DFT of a centered unit delta is exactly 1 in every bin.
                self.assertAlmostEqual(v, scale * scale, delta=1e-300)
        # Field is real and identical everywhere (zero phase).
        first = r.field[0][0]
        for row in r.field:
            for v in row:
                self.assertAlmostEqual(v.real, scale, delta=1e-15)
                self.assertEqual(v.imag, 0.0)
        self.assertEqual(first, scale)

    def test_off_center_single_pixel_is_constant_magnitude_linear_phase(self):
        n = 12
        a = [[0j] * n for _ in range(n)]
        a[2][5] = 1.0  # unshifted offset m = 2 - 6 = -4, n_off = 5 - 6 = -1
        r = fd.fraunhofer(a, 2e-6, 6e-7, 0.05)
        scale = (2e-6 ** 2) / (6e-7 * 0.05)
        mags = [math.sqrt(v) for row in r.intensity for v in row]
        self.assertTrue(all(abs(mag - abs(scale)) < 1e-14 for mag in mags))

        # Independent linear-phase reference in shifted layout.
        for p in range(n):
            for q in range(n):
                k = (p - n // 2) % n
                l = (q - n // 2) % n
                want = scale * cmath.exp(-2j * math.pi * (k * -4 + l * -1) / n)
                self.assertLess(abs(r.field[p][q] - want), 1e-14)


class TestRectangularAnalytic(unittest.TestCase):
    def test_rectangular_field_matches_discrete_analytic_formula(self):
        n = 16
        dx, wavelength, z = 1e-5, 5e-7, 0.1
        c = n // 2
        # Even x extent (4) and odd y extent (5), deliberately asymmetric.
        i0, i1 = c - 2, c + 1
        j0, j1 = c - 2, c + 2
        a = [[0.0] * n for _ in range(n)]
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                a[i][j] = 1.0

        r = fd.fraunhofer(a, dx, wavelength, z)
        scale = dx * dx / (wavelength * z)
        m0, m1 = i0 - c, i1 - c
        n0, n1 = j0 - c, j1 - c

        max_amp = 0.0
        for k in range(n):
            for l in range(n):
                max_amp = max(max_amp, abs(scale * analytic_rect_spectrum(k, l, m0, m1, n0, n1, n)))
        for k in range(n):
            for l in range(n):
                p, q = shifted_index(k, n), shifted_index(l, n)
                want = scale * analytic_rect_spectrum(k, l, m0, m1, n0, n1, n)
                self.assertLess(
                    abs(r.field[p][q] - want),
                    1e-12 * max_amp,
                    msg=f"k={k},l={l}",
                )
                self.assertLess(
                    abs(r.intensity[p][q] - abs(want) ** 2),
                    1e-12 * max_amp ** 2,
                )

        # DC bin: all K*L phasors align.
        self.assertAlmostEqual(r.field[c][c], scale * 4 * 5, delta=1e-13)

    def test_generator_matches_hand_built_block(self):
        # Width 4*dx and height 6*dx must light exactly the expected pixels,
        # boundary included, on both even and odd grids.
        for n in (15, 16):
            dx = 1.0
            c = n // 2
            grid = fd.rectangular_aperture(n, 4.0, 6.0, dx)
            for i in range(n):
                for j in range(n):
                    expect = abs(i - c) <= 2 and abs(j - c) <= 3
                    self.assertEqual(grid[i][j], 1.0 if expect else 0.0, msg=(n, i, j))


class TestCircularAperture(unittest.TestCase):
    def test_deterministic_and_symmetric(self):
        g1 = fd.circular_aperture(32, 10.5, 1.0)
        g2 = fd.circular_aperture(32, 10.5, 1.0)
        self.assertEqual(g1, g2)
        c = 16
        for i in range(32):
            for j in range(32):
                inside = math.hypot(i - c, j - c) <= 10.5
                self.assertEqual(g1[i][j], 1.0 if inside else 0.0)
                self.assertEqual(g1[i][j], g1[j][i])

    def test_radius_zero_is_single_center_pixel(self):
        g = fd.circular_aperture(8, 0.0, 1.0)
        self.assertEqual(sum(sum(row) for row in g), 1.0)
        self.assertEqual(g[4][4], 1.0)


# ---------------------------------------------------------------------------
# Parseval, zero padding, phase ramp
# ---------------------------------------------------------------------------


class TestParseval(unittest.TestCase):
    def test_power_conservation_independent_check(self):
        for n in (1, 7, 16, 31, 32):
            with self.subTest(n=n):
                # Deterministic complex aperture without using the library.
                a = [
                    [complex((i * 5 + j) % 3 - 1, (i + 2 * j) % 4 - 1.5) for j in range(n)]
                    for i in range(n)
                ]
                dx = 1.0 + n * 1e-3
                r = fd.fraunhofer(a, dx, 0.5, 1.0)
                expected_power = parseval_sum(a, dx)
                self.assertAlmostEqual(r.input_power, expected_power, delta=1e-9 * expected_power)
                self.assertLess(abs(r.power_error), 1e-12, msg=f"n={n}")
                # output_power must also equal the independently summed input.
                self.assertLess(abs(r.output_power - expected_power) / expected_power, 1e-12)

    def test_zero_aperture_power_error_is_explicit_zero(self):
        r = fd.fraunhofer([[0.0, 0.0], [0.0, 0.0]], 1.0, 1.0, 1.0)
        self.assertEqual(r.input_power, 0.0)
        self.assertEqual(r.output_power, 0.0)
        self.assertEqual(r.power_error, 0.0)
        self.assertEqual(max(max(row) for row in r.intensity), 0.0)


class TestZeroPadding(unittest.TestCase):
    def test_padded_grid(self):
        n = 16
        m = 64
        dx, wavelength, z = 1e-5, 5e-7, 0.1
        a = fd.rectangular_aperture(n, 4 * dx, 6 * dx, dx)
        r0 = fd.fraunhofer(a, dx, wavelength, z)
        r1 = fd.fraunhofer(a, dx, wavelength, z, pad=m)

        self.assertEqual(r0.m, n)
        self.assertEqual(r1.m, m)
        # Pitch must be updated for the padded grid.
        self.assertAlmostEqual(r0.du, wavelength * z / (n * dx), delta=1e-18)
        self.assertAlmostEqual(r1.du, wavelength * z / (m * dx), delta=1e-18)
        self.assertAlmostEqual(r1.du, r0.du * n / m, delta=1e-20)

        # Independently computed power, unchanged by padding.
        expected = parseval_sum(a, dx)
        self.assertLess(abs(r0.output_power - expected) / expected, 1e-12)
        self.assertLess(abs(r1.output_power - expected) / expected, 1e-12)
        self.assertLess(abs(r1.power_error), 1e-12)

        # Original DFT bins must reappear at the interpolated padded bins:
        # shifted coordinate (p - n/2) du_n equals (4p - m/2) du_m.
        scale = dx * dx / (wavelength * z)
        for p in (0, 3, 8, 12, 15):
            for q in (1, 7, 15):
                self.assertLess(
                    abs(r1.field[4 * p][4 * q] - r0.field[p][q]),
                    1e-12 * scale,
                )
                self.assertAlmostEqual(r1.x[4 * p], r0.x[p], delta=1e-18)


class TestPhaseRampShift(unittest.TestCase):
    def test_phase_ramp_translates_intensity_pattern(self):
        n = 16
        c = n // 2
        base = [[0.0] * n for _ in range(n)]
        for i in range(c - 1, c + 2):
            for j in range(c - 2, c + 3):
                base[i][j] = 1.0
        s0, s1 = 2, -3
        ramped = [
            [
                base[i][j] * cmath.exp(2j * math.pi * (s0 * (i - c) + s1 * (j - c)) / n)
                for j in range(n)
            ]
            for i in range(n)
        ]
        r0 = fd.fraunhofer(base, 1e-5, 5e-7, 0.1)
        r1 = fd.fraunhofer(ramped, 1e-5, 5e-7, 0.1)

        # Shift theorem: intensity pattern moves by (s0, s1) shifted pixels,
        # circularly; magnitudes are unchanged.
        peak = max(max(row) for row in r0.intensity)
        for p in range(n):
            for q in range(n):
                self.assertAlmostEqual(
                    r1.intensity[p][q],
                    r0.intensity[(p - s0) % n][(q - s1) % n],
                    delta=1e-12 * peak,
                )
        # The amplitude itself: shift theorem gives B[k, l] = A[k-s0, l-s1],
        # evaluated here from the independent analytic formula (mod N).
        k, l = 3, -2
        p, q = shifted_index(k, n), shifted_index(l, n)
        scale = 1e-5 ** 2 / (5e-7 * 0.1)
        want = scale * analytic_rect_spectrum(
            (k - s0) % n, (l - s1) % n, -1, 1, -2, 2, n
        )
        self.assertLess(abs(r1.field[p][q] - want), 1e-12 * abs(scale * 20))


# ---------------------------------------------------------------------------
# Coordinates, center index and even-grid fftshift convention
# ---------------------------------------------------------------------------


class TestGridConvention(unittest.TestCase):
    def test_even_grid_convention(self):
        m = 16
        r = fd.fraunhofer([[1.0]], 1e-5, 5e-7, 0.1, pad=m)
        self.assertEqual(r.center, (8, 8))
        self.assertEqual(r.x[8], 0.0)
        self.assertEqual(r.x[7], -r.du)
        self.assertAlmostEqual(r.x[15], 7 * r.du, delta=1e-20)
        self.assertAlmostEqual(r.x[0], -8 * r.du, delta=1e-20)
        self.assertEqual(r.x, r.y)

    def test_odd_grid_convention(self):
        n = 15
        r = fd.fraunhofer([[1.0]], 1e-5, 5e-7, 0.1, pad=n)
        self.assertEqual(r.center, (7, 7))
        self.assertEqual(r.x[7], 0.0)
        self.assertAlmostEqual(r.x[6], -r.du, delta=1e-20)
        self.assertAlmostEqual(r.x[8], r.du, delta=1e-20)
        self.assertAlmostEqual(r.x[0], -7 * r.du, delta=1e-20)
        self.assertAlmostEqual(r.x[14], 7 * r.du, delta=1e-20)

    def test_result_is_frozen(self):
        r = fd.fraunhofer([[1.0]], 1.0, 1.0, 1.0)
        with self.assertRaises(Exception):
            r.du = 2.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Error semantics
# ---------------------------------------------------------------------------


class TestErrors(unittest.TestCase):
    def assertLocalized(self, exc, name):
        self.assertIn(name, str(exc))

    def test_reject_nan_infinity_numeric(self):
        a = [[1.0]]
        for kwargs, name in [
            (dict(dx=float("nan"), wavelength=1.0, z=1.0), "dx"),
            (dict(dx=1.0, wavelength=float("inf"), z=1.0), "wavelength"),
            (dict(dx=1.0, wavelength=1.0, z=float("-inf")), "z"),
        ]:
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as cm:
                    fd.fraunhofer(a, **kwargs)
                self.assertLocalized(cm.exception, name)

    def test_reject_nan_complex_element(self):
        with self.assertRaises(ValueError) as cm:
            fd.fraunhofer([[1.0, complex(float("nan"), 0.0)]], 1.0, 1.0, 1.0)
        self.assertIn("aperture[0][1]", str(cm.exception))

    def test_reject_nonpositive_physical_inputs(self):
        cases = [
            (0.0, 1.0, 1.0, "dx"),
            (-1e-9, 1.0, 1.0, "dx"),
            (1.0, 0.0, 1.0, "wavelength"),
            (1.0, -5e-7, 1.0, "wavelength"),
            (1.0, 1.0, 0.0, "z"),
        ]
        for dx, wavelength, z, name in cases:
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as cm:
                    fd.fraunhofer([[1.0]], dx, wavelength, z)
                self.assertIn(name, str(cm.exception))

    def test_bool_rejected_as_int_or_number(self):
        with self.assertRaises(TypeError):  # pad must be a strict int
            fd.fraunhofer([[1.0]], 1.0, 1.0, 1.0, pad=True)
        with self.assertRaises(TypeError):
            fd.fraunhofer([[1.0]], True, 1.0, 1.0)
        with self.assertRaises(TypeError):
            fd.circular_aperture(True, 1.0, 1.0)
        with self.assertRaises(TypeError):
            fd.rectangular_aperture(False, 1.0, 1.0, 1.0)

    def test_strict_int_bounds(self):
        with self.assertRaises(TypeError):
            fd.circular_aperture(16.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            fd.circular_aperture(0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            fd.circular_aperture(33, 1.0, 1.0)
        with self.assertRaises(ValueError):
            fd.fraunhofer([[1.0]], 1.0, 1.0, 1.0, pad=65)
        with self.assertRaises(ValueError):
            fd.fraunhofer([[1.0] * 4] * 4, 1.0, 1.0, 1.0, pad=2)

    def test_shape_and_type_errors(self):
        with self.assertRaises(ValueError):
            fd.fraunhofer([[1.0] * 33] * 33, 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            fd.fraunhofer([[1.0, 0.0], [0.0, 0.0, 0.0]], 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            fd.fraunhofer([], 1.0, 1.0, 1.0)
        with self.assertRaises(TypeError):
            fd.fraunhofer([["1.0"]], 1.0, 1.0, 1.0)
        with self.assertRaises(TypeError):
            fd.fraunhofer(1.5, 1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Pixel-energy integration
# ---------------------------------------------------------------------------


class TestContinuousField(unittest.TestCase):
    def test_matches_dft_bins(self):
        # Consistency check: the continuous field sampled at unshifted bin
        # coordinates must reproduce the DFT-based field of fraunhofer().
        n = 12
        dx, wavelength, z = 1e-5, 5e-7, 0.1
        a = [[complex((i + j) % 3 - 1, (i * j) % 2) for j in range(n)] for i in range(n)]
        r = fd.fraunhofer(a, dx, wavelength, z)
        scale = max(abs(v) for row in r.field for v in row)
        for k in range(n):
            for l in range(n):
                u = fd._continuous_field(a, dx, wavelength, z, k * r.du, l * r.du)
                p, q = shifted_index(k, n), shifted_index(l, n)
                self.assertLess(abs(u - r.field[p][q]), 1e-10 * scale)

    def test_matches_independent_direct_sum(self):
        # Independent reference: plain cmath.exp double sum in this file.
        n = 8
        dx, wavelength, z = 2e-5, 6e-7, 0.07
        a = fd.rectangular_aperture(n, 5 * dx, 3 * dx, dx)
        r = fd.fraunhofer(a, dx, wavelength, z)
        for x, y in [(0.0, 0.0), (1.3 * r.du, -2.7 * r.du), (-4.2 * r.du, 0.9 * r.du)]:
            got = fd._continuous_field(a, dx, wavelength, z, x, y)
            want = reference_field(a, dx, wavelength, z, x, y)
            self.assertLess(abs(got - want), 1e-12 * max(1.0, abs(want)))


class TestIntegrateIntensity(unittest.TestCase):
    def test_constant_intensity_region_is_exact(self):
        # Single centered pixel aperture: intensity is exactly scale**2
        # everywhere, so the integral is scale**2 * area analytically.
        n = 8
        dx, wavelength, z = 1e-5, 5e-7, 0.1
        a = [[0j] * n for _ in range(n)]
        a[n // 2][n // 2] = 1.0
        scale = dx * dx / (wavelength * z)
        x0, x1, y0, y1 = -3.7e-4, 5.2e-4, -1.1e-4, 2.9e-4
        r = fd.integrate_intensity(a, dx, wavelength, z, x0, x1, y0, y1)
        want = scale * scale * (x1 - x0) * (y1 - y0)
        self.assertTrue(r.converged)
        self.assertLess(abs(r.energy - want), 1e-12 * want)
        self.assertEqual(r.region, (x0, x1, y0, y1))

    def test_matches_independent_fixed_quadrature(self):
        # Speckled complex aperture; module result vs the independent
        # fixed-grid 5-point Gauss reference in this file.
        n = 8
        dx, wavelength, z = 1.5e-5, 5.5e-7, 0.08
        a = [[complex((i * 3 + j) % 3 - 1, (i + 2 * j) % 4 - 1.5) for j in range(n)]
             for i in range(n)]
        r0 = fd.fraunhofer(a, dx, wavelength, z)
        x0, x1 = -2.3 * r0.du, 1.7 * r0.du
        y0, y1 = -1.2 * r0.du, 2.6 * r0.du
        r = fd.integrate_intensity(a, dx, wavelength, z, x0, x1, y0, y1, rtol=1e-8)
        want = reference_integral(a, dx, wavelength, z, x0, x1, y0, y1, cells=8)
        self.assertTrue(r.converged)
        self.assertLess(abs(r.energy - want), 1e-7 * want)
        self.assertLessEqual(r.error_estimate, max(1e-8 * r.energy, 1e-300) * 4)

    def test_min_depth_forces_subdivision(self):
        # Even for an exactly constant integrand the refinement must go at
        # least min_depth levels deep: no early stop on identical samples.
        n = 4
        a = [[0j] * n for _ in range(n)]
        a[n // 2][n // 2] = 1.0
        r = fd.integrate_intensity(a, 1e-5, 5e-7, 0.1, -1e-4, 1e-4, -1e-4, 1e-4,
                                   min_depth=3)
        self.assertGreaterEqual(r.depth, 3)
        self.assertGreaterEqual(r.evaluations, 9 * (1 + 4 + 16 + 64))

    def test_budget_exhaustion_is_reported_not_silent(self):
        n = 8
        dx, wavelength, z = 1.5e-5, 5.5e-7, 0.08
        a = [[complex((i + j) % 2, (i * j) % 3 - 1) for j in range(n)] for i in range(n)]
        r0 = fd.fraunhofer(a, dx, wavelength, z)
        # Absurdly tight tolerance with a tiny depth budget cannot converge.
        r = fd.integrate_intensity(a, dx, wavelength, z,
                                   -3 * r0.du, 3 * r0.du, -3 * r0.du, 3 * r0.du,
                                   rtol=1e-15, max_depth=3)
        self.assertFalse(r.converged)
        self.assertEqual(r.depth, 3)
        # A sane request on the same region does converge: failure above is
        # the budget, not the integrand.
        r2 = fd.integrate_intensity(a, dx, wavelength, z,
                                    -3 * r0.du, 3 * r0.du, -3 * r0.du, 3 * r0.du,
                                    rtol=1e-6)
        self.assertTrue(r2.converged)

    def test_additivity_neighbouring_regions(self):
        # Energy of neighbouring pixels summed must match one integration
        # over the same total rectangle (cross-check, tight tolerance).
        n = 8
        dx, wavelength, z = 1.5e-5, 5.5e-7, 0.08
        a = fd.circular_aperture(n, 3 * dx, dx)
        r0 = fd.fraunhofer(a, dx, wavelength, z, pad=16)
        du = r0.du
        ci, cj = r0.center
        parts = [
            fd.pixel_energy(a, dx, wavelength, z, ci + di, cj + dj, pad=16, rtol=1e-9)
            for di in (0, 1) for dj in (0, 1)
        ]
        for p in parts:
            self.assertTrue(p.converged)
        summed = sum(p.energy for p in parts)
        whole = fd.integrate_intensity(a, dx, wavelength, z,
                                       -0.5 * du, 1.5 * du, -0.5 * du, 1.5 * du,
                                       rtol=1e-9)
        self.assertTrue(whole.converged)
        self.assertLess(abs(summed - whole.energy), 1e-7 * whole.energy)
        # And the merged region matches the independent reference.
        want = reference_integral(a, dx, wavelength, z,
                                  -0.5 * du, 1.5 * du, -0.5 * du, 1.5 * du, cells=8)
        self.assertLess(abs(whole.energy - want), 1e-7 * want)


class TestPixelEnergy(unittest.TestCase):
    def test_region_matches_detector_pixel(self):
        n = 8
        dx, wavelength, z = 1e-5, 5e-7, 0.1
        a = fd.rectangular_aperture(n, 4 * dx, 4 * dx, dx)
        r0 = fd.fraunhofer(a, dx, wavelength, z, pad=16)
        i, j = 9, 6
        r = fd.pixel_energy(a, dx, wavelength, z, i, j, pad=16)
        du = r0.du
        x0, x1, y0, y1 = r.region
        self.assertAlmostEqual(x0, (i - 8) * du - 0.5 * du, delta=1e-20)
        self.assertAlmostEqual(x1, (i - 8) * du + 0.5 * du, delta=1e-20)
        self.assertAlmostEqual(y0, (j - 8) * du - 0.5 * du, delta=1e-20)
        self.assertAlmostEqual(y1, (j - 8) * du + 0.5 * du, delta=1e-20)
        self.assertTrue(r.converged)
        # Constant-integrand sanity: energy is du**2 * mean intensity > 0.
        self.assertGreater(r.energy, 0.0)

    def test_center_pixel_beats_center_times_area(self):
        # For a peaked pattern the integrated pixel energy differs from the
        # naive intensity*du**2 estimate, and the integral is the smaller
        # one at the peak (intensity is convex there).
        n = 16
        dx, wavelength, z = 1e-5, 5e-7, 0.1
        a = fd.circular_aperture(n, 5 * dx, dx)
        r0 = fd.fraunhofer(a, dx, wavelength, z)
        ci, cj = r0.center
        r = fd.pixel_energy(a, dx, wavelength, z, ci, cj, rtol=1e-8)
        naive = r0.intensity[ci][cj] * r0.du * r0.du
        self.assertTrue(r.converged)
        self.assertLess(r.energy, naive)
        self.assertLess(abs(r.energy - naive) / naive, 0.2)

    def test_energy_matches_integrate_intensity_same_region(self):
        n = 8
        dx, wavelength, z = 1e-5, 5e-7, 0.1
        a = fd.circular_aperture(n, 3 * dx, dx)
        du = wavelength * z / (n * dx)
        i, j = 5, 2
        r_pix = fd.pixel_energy(a, dx, wavelength, z, i, j, rtol=1e-9)
        r_int = fd.integrate_intensity(
            a, dx, wavelength, z,
            (i - n // 2 - 0.5) * du, (i - n // 2 + 0.5) * du,
            (j - n // 2 - 0.5) * du, (j - n // 2 + 0.5) * du,
            rtol=1e-9,
        )
        self.assertAlmostEqual(r_pix.energy, r_int.energy, delta=1e-12 * r_int.energy)


class TestIntegrationErrors(unittest.TestCase):
    def test_bad_region(self):
        a = [[1.0]]
        with self.assertRaises(ValueError):
            fd.integrate_intensity(a, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0)  # x0 == x1
        with self.assertRaises(ValueError):
            fd.integrate_intensity(a, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 1.0)  # x0 > x1
        with self.assertRaises(ValueError) as cm:
            fd.integrate_intensity(a, 1.0, 1.0, 1.0, 0.0, 1.0, 3.0, 2.0)  # y0 > y1
        self.assertIn("y0", str(cm.exception))
        with self.assertRaises(ValueError):
            fd.integrate_intensity(a, 1.0, 1.0, 1.0, float("nan"), 1.0, 0.0, 1.0)
        with self.assertRaises(TypeError):
            fd.integrate_intensity(a, 1.0, 1.0, 1.0, True, 1.0, 0.0, 1.0)

    def test_bad_controls(self):
        a = [[1.0]]
        with self.assertRaises(ValueError):
            fd.integrate_intensity(a, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, rtol=0.0)
        with self.assertRaises(ValueError):
            fd.integrate_intensity(a, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, atol=-1e-9)
        with self.assertRaises(ValueError):
            fd.integrate_intensity(a, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                                   min_depth=5, max_depth=3)
        with self.assertRaises(TypeError):
            fd.integrate_intensity(a, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                                   max_depth=True)
        with self.assertRaises(TypeError):
            fd.integrate_intensity(a, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                                   max_evaluations=100.5)

    def test_bad_pixel_index(self):
        a = [[1.0, 0.0], [0.0, 0.0]]
        with self.assertRaises(TypeError):
            fd.pixel_energy(a, 1.0, 1.0, 1.0, True, 0)
        with self.assertRaises(TypeError):
            fd.pixel_energy(a, 1.0, 1.0, 1.0, 0, 1.5)
        with self.assertRaises(ValueError) as cm:
            fd.pixel_energy(a, 1.0, 1.0, 1.0, 2, 0)
        self.assertIn("i", str(cm.exception))
        with self.assertRaises(ValueError):
            fd.pixel_energy(a, 1.0, 1.0, 1.0, 0, -1)
        with self.assertRaises(ValueError):
            fd.pixel_energy(a, 1.0, 1.0, 1.0, 0, 0, pad=4, max_evaluations=8)


if __name__ == "__main__":
    unittest.main()
