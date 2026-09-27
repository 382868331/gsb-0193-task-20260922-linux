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


class TestPixelEnergyIntegration(unittest.TestCase):
    def test_single_pixel_aperture_integrates_constant_area(self):
        r = fd.fraunhofer([[1.0]], 1.0, 1.0, 1.0, pad=8)
        energies = fd.integrate_pixel_energies(
            r, tolerance=1e-12, min_subdivisions=2, max_subdivisions=8
        )
        expected = r.du * r.du
        self.assertTrue(energies.all_converged)
        self.assertTrue(max(max(row) for row in energies.subdivisions) <= 8)
        for row in energies.energy:
            for value in row:
                self.assertAlmostEqual(value, expected, delta=1e-18)

    def test_area_integral_matches_independent_sinc_reference(self):
        n, m = 2, 8
        dx, wavelength, z = 1.0, 1.0, 1.0
        aperture = [[0.0] * n for _ in range(n)]
        aperture[0][n // 2] = 1.0
        aperture[1][n // 2] = 1.0
        r = fd.fraunhofer(aperture, dx, wavelength, z, pad=32)
        result = fd.integrate_pixel_energies(
            r, tolerance=1e-3, min_subdivisions=4, max_subdivisions=8
        )

        field_scale = dx * dx / (wavelength * z)
        argument = math.pi * dx * r.du / (wavelength * z)
        expected_center = (
            2.0
            * field_scale**2
            * r.du**2
            * (1.0 + math.sin(argument) / argument)
        )
        ci, cj = r.center
        self.assertTrue(result.converged[ci][cj])
        self.assertLess(
            abs(result.energy[ci][cj] - expected_center) / expected_center,
            2e-5,
        )
        self.assertNotAlmostEqual(
            result.energy[ci][cj],
            r.intensity[ci][cj] * r.du**2,
            delta=1e-7 * expected_center,
        )

    def test_region_matches_sum_of_tiled_pixels_at_same_refinement(self):
        r = fd.fraunhofer(
            fd.rectangular_aperture(8, 3.0, 5.0, 1.0),
            1.0,
            1.0,
            1.0,
            pad=32,
        )
        pixels = fd.integrate_pixel_energies(
            r, tolerance=1e-2, min_subdivisions=4, max_subdivisions=8
        )
        region = fd.integrate_region_energy(
            r, 14, 18, 14, 19, tolerance=1e-2,
            min_subdivisions=4, max_subdivisions=8,
        )
        tiled = math.fsum(
            pixels.energy[i][j]
            for i in range(14, 18)
            for j in range(14, 19)
        )
        self.assertTrue(
            all(
                pixels.converged[i][j]
                for i in range(14, 18)
                for j in range(14, 19)
            )
        )
        self.assertTrue(region.converged)
        self.assertLess(region.consistency_error, 1e-14)
        self.assertLess(abs(region.energy - tiled), 1e-14 * max(tiled, 1.0))

    def test_unconverged_result_is_explicit(self):
        r = fd.fraunhofer(
            fd.rectangular_aperture(12, 5.0, 7.0, 1.0),
            1.0,
            1.0,
            1.0,
            pad=16,
        )
        result = fd.integrate_pixel_energies(
            r, tolerance=1e-14, min_subdivisions=2, max_subdivisions=2
        )
        self.assertFalse(result.all_converged)
        self.assertTrue(any(not value for row in result.converged for value in row))

    def test_reject_invalid_refinement_and_region(self):
        r = fd.fraunhofer([[1.0]], 1.0, 1.0, 1.0, pad=4)
        with self.assertRaises(ValueError):
            fd.integrate_pixel_energies(r, min_subdivisions=3, max_subdivisions=8)
        with self.assertRaises(ValueError):
            fd.integrate_region_energy(r, 1, 1, 0, 2)
        with self.assertRaises(ValueError):
            fd.integrate_region_energy(r, 0, 5, 0, 2)


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


if __name__ == "__main__":
    unittest.main()
