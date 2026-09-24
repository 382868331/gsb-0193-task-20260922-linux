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
# Independent reference for the full single-mode Fraunhofer intensity
# (brute-force DFT + physics scaling + fftshift; never calls the core)
# ---------------------------------------------------------------------------


def direct_fraunhofer_intensity(grid, dx, wavelength, z, m):
    n = len(grid)
    emb = [[0j] * m for _ in range(m)]
    for i in range(n):
        mi = (i - n // 2) % m
        for j in range(n):
            emb[mi][(j - n // 2) % m] = complex(grid[i][j])
    rows = [direct_dft1(row, -1) for row in emb]
    scale = dx * dx / (wavelength * z)
    out = [[0.0] * m for _ in range(m)]
    for k in range(m):
        for l in range(m):
            s = 0j
            for i in range(m):
                s += rows[i][l] * cmath.exp(-2j * math.pi * k * i / m)
            u = scale * s
            out[(k + m // 2) % m][(l + m // 2) % m] = u.real ** 2 + u.imag ** 2
    return out


# ---------------------------------------------------------------------------
# Multi-mode / multi-wavelength incoherent propagation
# ---------------------------------------------------------------------------


class TestMultiModeNormal(unittest.TestCase):
    def test_single_mode_degenerates_to_single_entry(self):
        # Explicitly required degeneracy: one mode, weight 1 is the old call.
        n = 8
        a = [
            [complex((i * 3 + j) % 2, (i + j) % 3 - 1) for j in range(n)]
            for i in range(n)
        ]
        dx, wavelength, z, pad = 1.7e-6, 5.3e-7, 0.08, 16
        single = fd.fraunhofer(a, dx, wavelength, z, pad=pad)
        multi = fd.fraunhofer_modes(
            [fd.CoherentMode(a, wavelength)], dx, z, pad=pad
        )
        self.assertEqual(multi.n, n)
        self.assertEqual(multi.m, pad)
        self.assertEqual(multi.weights, [1.0])
        self.assertEqual(multi.wavelengths, [wavelength])
        for i in range(pad):
            for j in range(pad):
                self.assertEqual(multi.intensity[i][j], single.intensity[i][j])
        # The per-mode result is the ordinary result, field included.
        self.assertEqual(multi.mode_results[0].field, single.field)
        self.assertAlmostEqual(multi.power_error, single.power_error, delta=1e-18)
        # No summed complex field can exist by construction.
        self.assertFalse(hasattr(multi, "field"))

    def test_hand_computed_single_pixel_two_wavelengths(self):
        # Centered single pixels: each mode's field is the constant
        # U_q = dx^2/(lambda_q z) * amplitude_q in every bin, so the summed
        # intensity per bin is the hand-summed constant below.
        dx, z = 1e-5, 0.1
        lam1, lam2 = 4.0e-7, 7.0e-7
        w1, w2 = 2.0, 0.5
        amp1, amp2 = 3.0, 2.0j
        m = 6
        r = fd.fraunhofer_modes(
            [
                fd.CoherentMode([[amp1]], lam1, w1),
                fd.CoherentMode([[amp2]], lam2, w2),
            ],
            dx, z, pad=m,
        )
        s1 = dx * dx / (lam1 * z) * abs(amp1)
        s2 = dx * dx / (lam2 * z) * abs(amp2)
        expected = w1 * s1 * s1 + w2 * s2 * s2
        for row in r.intensity:
            for v in row:
                self.assertAlmostEqual(v, expected, delta=1e-12 * expected)

        # Weighted Parseval, hand summed.
        pin = (
            w1 * dx * dx * abs(amp1) ** 2
            + w2 * dx * dx * abs(amp2) ** 2
        )
        self.assertLess(abs(r.input_power - pin) / pin, 1e-15)
        self.assertLess(abs(r.output_power - pin) / pin, 1e-12)
        self.assertLess(abs(r.power_error), 1e-12)

    def test_weights_are_linear_independent_reference(self):
        # 3 modes, distinct wavelengths, arbitrary complex 4x4 apertures,
        # padding to 5, checked against the brute-force reference.
        n, m = 4, 5
        dx, z = 2.3e-6, 0.05
        lams = [4.0e-7, 5.5e-7, 7.0e-7]
        weights = [0.5, 1.0, 2.25]
        apertures = [
            [[complex((i + q) % 3 - 1, (i * j + q) % 4 - 1.5) for j in range(n)]
             for i in range(n)]
            for q in range(3)
        ]
        modes = [fd.CoherentMode(a, lam, w)
                 for a, lam, w in zip(apertures, lams, weights)]
        r = fd.fraunhofer_modes(modes, dx, z, pad=m)

        expected = [[0.0] * m for _ in range(m)]
        pin = 0.0
        for a, lam, w in zip(apertures, lams, weights):
            ref = direct_fraunhofer_intensity(a, dx, lam, z, m)
            for i in range(m):
                for j in range(m):
                    expected[i][j] += w * ref[i][j]
            pin += w * parseval_sum(a, dx)

        scale = max(max(row) for row in expected)
        for i in range(m):
            for j in range(m):
                self.assertLess(
                    abs(r.intensity[i][j] - expected[i][j]),
                    1e-12 * scale,
                    msg=(i, j),
                )
        self.assertLess(abs(r.output_power - pin) / pin, 1e-12)
        self.assertLess(abs(r.power_error), 1e-12)

        # Linearity in the weights: doubling every weight doubles every bin
        # and the total powers, with no change to per-mode fields.
        r2 = fd.fraunhofer_modes(
            [fd.CoherentMode(a, lam, 2.0 * w)
             for a, lam, w in zip(apertures, lams, weights)],
            dx, z, pad=m,
        )
        for i in range(m):
            for j in range(m):
                self.assertAlmostEqual(
                    r2.intensity[i][j], 2.0 * r.intensity[i][j],
                    delta=1e-12 * scale,
                )
        self.assertAlmostEqual(r2.input_power, 2.0 * r.input_power,
                               delta=1e-12 * r2.input_power)

    def test_intensities_add_even_when_fields_would_interfere(self):
        # Two off-center single pixels at different wavelengths carry linear
        # phases.  Adding complex amplitudes would produce bin-dependent
        # interference; the incoherent intensity sum must stay bin-independent.
        n = 5
        a1 = [[0j] * n for _ in range(n)]
        a2 = [[0j] * n for _ in range(n)]
        a1[0][1] = 1.0
        a2[4][1] = 1.0
        dx, z = 1e-5, 0.1
        lam1, lam2 = 4.5e-7, 6.5e-7
        r = fd.fraunhofer_modes(
            [fd.CoherentMode(a1, lam1), fd.CoherentMode(a2, lam2)], dx, z
        )
        s1 = dx * dx / (lam1 * z)
        s2 = dx * dx / (lam2 * z)
        expected = s1 * s1 + s2 * s2
        flats = [v for row in r.intensity for v in row]
        self.assertLess(max(flats) - min(flats), 1e-12 * expected)
        for v in flats:
            self.assertLess(abs(v - expected), 1e-12 * expected)

        # Demonstrate with the returned per-mode fields that the forbidden
        # cross-mode complex addition would in fact oscillate with the bin.
        u1 = [r.mode_results[0].field[i][1] for i in range(n)]
        u2 = [r.mode_results[1].field[i][1] for i in range(n)]
        cross = [abs(p + q) ** 2 for p, q in zip(u1, u2)]
        self.assertGreater(max(cross) - min(cross), 1e-6 * expected)

    def test_common_frequency_axis_and_per_wavelength_coordinates(self):
        n, m = 3, 8
        dx, z = 1e-5, 0.1
        lam1, lam2 = 4.0e-7, 6.0e-7
        a1 = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        a2 = [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        r = fd.fraunhofer_modes(
            [fd.CoherentMode(a1, lam1), fd.CoherentMode(a2, lam2)],
            dx, z, pad=m,
        )
        self.assertEqual(r.center, (m // 2, m // 2))
        for p in range(m):
            fx = (p - m // 2) / (m * dx)
            self.assertAlmostEqual(r.fx[p], fx, delta=1e-20)
            self.assertAlmostEqual(r.fy[p], fx, delta=1e-20)
            # mode q physical coordinate = lambda_q * z * fx
            self.assertAlmostEqual(r.mode_results[0].x[p], lam1 * z * fx,
                                   delta=1e-14 * abs(lam1 * z * fx) + 1e-30)
            self.assertAlmostEqual(r.mode_results[1].x[p], lam2 * z * fx,
                                   delta=1e-14 * abs(lam2 * z * fx) + 1e-30)
        # du scales linearly with the wavelength.
        self.assertAlmostEqual(
            r.mode_results[1].du / r.mode_results[0].du, lam2 / lam1,
            delta=1e-14,
        )

    def test_three_modes_three_wavelengths_and_generator_input(self):
        n = 6
        aps = [
            fd.rectangular_aperture(n, 2.0, 4.0, 1.0),
            fd.circular_aperture(n, 1.5, 1.0),
            [[complex((i + j) % 2, 0.0) for j in range(n)] for i in range(n)],
        ]
        lams = [4.0e-7, 5.0e-7, 6.0e-7]
        r = fd.fraunhofer_modes(
            (fd.CoherentMode(a, lam) for a, lam in zip(aps, lams)),
            dx=1.0, z=1.0,
        )
        self.assertEqual(len(r.mode_results), 3)
        self.assertEqual(len({w for w in r.wavelengths}), 3)
        self.assertEqual(r.m, n)

    def test_zero_weight_and_zero_aperture(self):
        # A zero-weight mode contributes nothing; compare with a hand value.
        dx, z, lam = 1e-5, 0.1, 5e-7
        a = [[1.0]]
        r = fd.fraunhofer_modes(
            [fd.CoherentMode(a, lam, 0.0), fd.CoherentMode(a, lam, 3.0)],
            dx, z, pad=4,
        )
        s = dx * dx / (lam * z)
        expected_zw = 3.0 * s * s
        for row in r.intensity:
            for v in row:
                self.assertAlmostEqual(v, expected_zw, delta=1e-14 * expected_zw)
        # Everything zero: explicit 0.0 power error, like the single entry.
        r0 = fd.fraunhofer_modes(
            [fd.CoherentMode([[0.0, 0.0], [0.0, 0.0]], lam, 0.0)], 1.0, 1.0
        )
        self.assertEqual(r0.input_power, 0.0)
        self.assertEqual(r0.output_power, 0.0)
        self.assertEqual(r0.power_error, 0.0)
        self.assertEqual(max(max(row) for row in r0.intensity), 0.0)

    def test_results_are_frozen(self):
        r = fd.fraunhofer_modes([fd.CoherentMode([[1.0]], 1.0)], 1.0, 1.0)
        with self.assertRaises(Exception):
            r.intensity = []  # type: ignore[misc]
        mode = fd.CoherentMode([[1.0]], 1.0)
        with self.assertRaises(Exception):
            mode.weight = 2.0  # type: ignore[misc]


class TestMultiModeErrors(unittest.TestCase):
    def test_mode_count_bounds(self):
        a = [[1.0]]
        for count in (0, 4):
            with self.subTest(count=count):
                with self.assertRaises(ValueError) as cm:
                    fd.fraunhofer_modes(
                        [fd.CoherentMode(a, 5e-7) for _ in range(count)],
                        1.0, 1.0,
                    )
                self.assertIn("modes", str(cm.exception))

    def test_modes_container_and_element_types(self):
        with self.assertRaises(TypeError):
            fd.fraunhofer_modes(3, 1.0, 1.0)
        with self.assertRaises(TypeError) as cm:
            fd.fraunhofer_modes([("ap", 5e-7)], 1.0, 1.0)
        self.assertIn("modes[0]", str(cm.exception))
        with self.assertRaises(TypeError):
            fd.fraunhofer_modes([True], 1.0, 1.0)

    def test_weight_validation_is_localized(self):
        a = [[1.0]]
        for bad, kind in [(-0.1, ValueError), (float("nan"), ValueError),
                          (float("inf"), ValueError), ("1.0", TypeError),
                          (True, TypeError)]:
            with self.subTest(bad=bad):
                with self.assertRaises(kind) as cm:
                    fd.fraunhofer_modes(
                        [fd.CoherentMode(a, 5e-7, 1.0),
                         fd.CoherentMode(a, 6e-7, bad)],
                        1.0, 1.0,
                    )
                self.assertIn("modes[1].weight", str(cm.exception))

    def test_wavelength_validation_is_localized(self):
        a = [[1.0]]
        for bad, kind in [(0.0, ValueError), (-5e-7, ValueError),
                          (float("nan"), ValueError), (1j, TypeError)]:
            with self.subTest(bad=bad):
                with self.assertRaises(kind) as cm:
                    fd.fraunhofer_modes(
                        [fd.CoherentMode(a, 5e-7), fd.CoherentMode(a, bad)],
                        1.0, 1.0,
                    )
                self.assertIn("modes[1].wavelength", str(cm.exception))

    def test_aperture_sizes_must_match(self):
        with self.assertRaises(ValueError) as cm:
            fd.fraunhofer_modes(
                [fd.CoherentMode([[1.0]], 5e-7),
                 fd.CoherentMode([[1.0, 0.0], [0.0, 1.0]], 6e-7)],
                1.0, 1.0,
            )
        self.assertIn("modes[1].aperture", str(cm.exception))

    def test_bad_aperture_inside_mode_is_localized(self):
        # NaN element keeps the full element path.
        with self.assertRaises(ValueError) as cm:
            fd.fraunhofer_modes(
                [fd.CoherentMode([[complex(float("nan"), 0.0)]], 5e-7)],
                1.0, 1.0,
            )
        self.assertIn("modes[0].aperture[0][0]", str(cm.exception))
        # Non-square structure keeps the mode index.
        with self.assertRaises(ValueError) as cm:
            fd.fraunhofer_modes(
                [fd.CoherentMode([[1.0, 0.0], [0.0]], 5e-7)], 1.0, 1.0
            )
        self.assertIn("modes[0].aperture", str(cm.exception))
        # Non-numeric element.
        with self.assertRaises(TypeError) as cm:
            fd.fraunhofer_modes(
                [fd.CoherentMode([["x"]], 5e-7)], 1.0, 1.0
            )
        self.assertIn("modes[0].aperture[0][0]", str(cm.exception))

    def test_common_physical_and_pad_validation(self):
        a = [[1.0]]
        with self.assertRaises(ValueError):
            fd.fraunhofer_modes([fd.CoherentMode(a, 5e-7)], 0.0, 1.0)
        with self.assertRaises(ValueError):
            fd.fraunhofer_modes([fd.CoherentMode(a, 5e-7)], 1.0, -1.0)
        with self.assertRaises(TypeError):
            fd.fraunhofer_modes([fd.CoherentMode(a, 5e-7)], 1.0, 1.0, pad=False)
        with self.assertRaises(ValueError):
            fd.fraunhofer_modes(
                [fd.CoherentMode([[1.0, 0.0], [0.0, 1.0]], 5e-7)],
                1.0, 1.0, pad=1,
            )
        with self.assertRaises(ValueError):
            fd.fraunhofer_modes([fd.CoherentMode(a, 5e-7)], 1.0, 1.0, pad=65)

    def test_failure_leaves_no_state(self):
        # Pure function: a rejected call changes nothing; a valid retry after
        # a failure returns the deterministic result unchanged.
        good = fd.CoherentMode([[1.0]], 5e-7)
        r1 = fd.fraunhofer_modes([good], 1.0, 1.0)
        with self.assertRaises(ValueError):
            fd.fraunhofer_modes(
                [good, fd.CoherentMode([[1.0]], 6e-7, -1.0)], 1.0, 1.0
            )
        r2 = fd.fraunhofer_modes([good], 1.0, 1.0)
        self.assertEqual(r1.intensity, r2.intensity)


if __name__ == "__main__":
    unittest.main()
