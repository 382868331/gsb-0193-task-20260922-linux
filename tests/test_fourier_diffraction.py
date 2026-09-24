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
# Multi-wavelength / multi-mode incoherent propagation
# ---------------------------------------------------------------------------


def independent_mode_intensity(grid, dx, wavelength, z, m):
    """Brute-force reference for one mode (no call into the core propagation).

    Embeds the centered grid into an M x M array, transforms rows and columns
    with the file-local direct O(n^2) DFT, applies the Fraunhofer scale and the
    fftshift mapping, and returns (intensity M x M, du, output_power).
    """
    n = len(grid)
    a = [[0j] * m for _ in range(m)]
    for i in range(n):
        mi = (i - n // 2) % m
        for j in range(n):
            mj = (j - n // 2) % m
            a[mi][mj] = complex(grid[i][j])
    rows = [direct_dft1(row, -1) for row in a]
    spec = [[0j] * m for _ in range(m)]
    for j in range(m):
        col = [rows[i][j] for i in range(m)]
        col_spec = direct_dft1(col, -1)
        for i in range(m):
            spec[i][j] = col_spec[i]
    scale = dx * dx / (wavelength * z)
    du = wavelength * z / (m * dx)
    out = [[0.0] * m for _ in range(m)]
    total = 0.0
    for k in range(m):
        pi = (k + m // 2) % m
        for l in range(m):
            pj = (l + m // 2) % m
            inten = abs(scale * spec[k][l]) ** 2
            out[pi][pj] = inten
            total += inten
    return out, du, total * du * du


def hand_grid(m):
    return [[0.0] * m for _ in range(m)]


class TestMultimodeDegradation(unittest.TestCase):
    def test_single_mode_matches_single_entry_exactly(self):
        n = 12
        dx, wavelength, z = 1e-5, 5e-7, 0.1
        a = [
            [complex((i * 3 + j) % 4 - 1.5, (i + j) % 3 - 1) for j in range(n)]
            for i in range(n)
        ]
        single = fd.fraunhofer(a, dx, wavelength, z, pad=16)
        multi = fd.fraunhofer_modes(
            [{"wavelength": wavelength, "modes": [{"aperture": a, "weight": 1.0}]}],
            dx, z, pad=16,
        )
        self.assertEqual(multi.m, 16)
        self.assertEqual(multi.pad, 16)
        self.assertEqual(len(multi.components), 1)
        comp = multi.components[0]
        self.assertEqual(comp.wavelength, wavelength)
        self.assertEqual(comp.weight, 1.0)
        self.assertEqual(comp.weighted_power, comp.result.output_power)
        for i in range(16):
            for j in range(16):
                self.assertEqual(multi.intensity[i][j], single.intensity[i][j])
                self.assertEqual(comp.result.intensity[i][j], single.intensity[i][j])
                self.assertEqual(comp.result.field[i][j], single.field[i][j])
        self.assertEqual(multi.total_weight, 1.0)
        self.assertLess(abs(comp.result.power_error), 1e-12)
        self.assertLess(abs(multi.detector_power - single.output_power),
                        1e-12 * single.output_power)

    def test_default_weight_and_bare_aperture_forms(self):
        a = fd.rectangular_aperture(6, 2.0, 2.0, 1.0)
        forms = [
            [{"wavelength": 0.5, "modes": [[a]]}],                      # dict, bare grid
            [{"wavelength": 0.5, "modes": [{"aperture": a}]}],         # dict, default w
            [(0.5, [a])],                                              # pair, bare grid
            [(0.5, [(a,)])],                                           # 1-tuple wrapped
            [(0.5, [(a, 1.0)])],                                       # explicit weight
        ]
        ref = fd.fraunhofer(a, 1.0, 0.5, 1.0)
        for groups in forms:
            with self.subTest(groups=groups):
                r = fd.fraunhofer_modes(groups, 1.0, 1.0)
                self.assertEqual(r.components[0].weight, 1.0)
                for i in range(6):
                    for j in range(6):
                        self.assertEqual(r.intensity[i][j], ref.intensity[i][j])

    def test_result_is_frozen(self):
        r = fd.fraunhofer_modes([(0.5, [[[1.0]]])], 1.0, 1.0)
        with self.assertRaises(Exception):
            r.m = 5  # type: ignore[misc]
        with self.assertRaises(Exception):
            r.intensity = []  # type: ignore[misc]


class TestMultimodeStacking(unittest.TestCase):
    def test_intensity_sum_against_independent_reference(self):
        # 3 wavelength groups (boundary count), 3/1/2 modes, mixed aperture
        # sizes, common padded grid.  Expected grid assembled solely from the
        # brute-force reference.
        dx, z, m = 1.0, 2.0, 12
        specs = [
            (0.4, [
                (fd.rectangular_aperture(7, 3.0, 2.0, dx), 1.0),
                (fd.circular_aperture(5, 1.5, dx), 0.375),
                ([[complex(1, -1), 0.0], [0.0, 1.0]], 2.5),
            ]),
            (0.6, [
                (fd.rectangular_aperture(9, 4.0, 4.0, dx), 0.9),
            ]),
            (0.8, [
                (fd.circular_aperture(3, 1.0, dx), 0.0),
                ([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], 1.5),
            ]),
        ]
        groups = [
            {"wavelength": wl, "modes": [{"aperture": ap, "weight": w} for ap, w in modes]}
            for wl, modes in specs
        ]
        r = fd.fraunhofer_modes(groups, dx, z, pad=m)

        expected = hand_grid(m)
        expected_detector_power = 0.0
        comp_index = 0
        for gi, (wl, modes) in enumerate(specs):
            self.assertEqual(r.wavelengths[gi], wl)
            self.assertEqual(r.weights[gi], tuple(w for _, w in modes))
            for ap, w in modes:
                ref_i, ref_du, ref_power = independent_mode_intensity(ap, dx, wl, z, m)
                comp = r.components[comp_index]
                self.assertEqual(comp.wavelength, wl)
                self.assertEqual(comp.weight, w)
                self.assertAlmostEqual(comp.result.du, ref_du, delta=1e-20)
                self.assertEqual(comp.result.m, m)
                self.assertLess(abs(comp.result.power_error), 1e-12)
                self.assertAlmostEqual(comp.weighted_power, w * ref_power,
                                       delta=1e-9 * max(1.0, w * ref_power))
                expected_detector_power += w * ref_power
                for i in range(m):
                    for j in range(m):
                        expected[i][j] += w * ref_i[i][j]
                comp_index += 1

        self.assertEqual(comp_index, 6)
        peak = max(max(row) for row in expected)
        for i in range(m):
            for j in range(m):
                self.assertAlmostEqual(
                    r.intensity[i][j], expected[i][j],
                    delta=1e-10 * max(1.0, peak),
                    msg=f"({i},{j})",
                )
        self.assertAlmostEqual(r.detector_power, expected_detector_power,
                               delta=1e-9 * expected_detector_power)
        self.assertAlmostEqual(r.total_weight, 1.0 + 0.375 + 2.5 + 0.9 + 0.0 + 1.5,
                               delta=1e-12)

    def test_hand_computed_centered_pixels_constant_intensity(self):
        # A single centered unit pixel has DFT exactly 1 in every bin, so
        # U == dx^2/(wl*z) and I == scale^2 everywhere.  The multi-wavelength
        # detector grid must be the plain weighted sum of those constants.
        dx, z, m = 1.0, 1.0, 8
        wl_w = [(0.4, 1.0), (0.5, 2.0), (0.8, 0.25)]
        groups = [(wl, [([[1.0]], w)]) for wl, w in wl_w]
        r = fd.fraunhofer_modes(groups, dx, z, pad=m)
        hand = sum(w * (dx * dx / (wl * z)) ** 2 for wl, w in wl_w)
        for i in range(m):
            for j in range(m):
                self.assertAlmostEqual(r.intensity[i][j], hand, delta=1e-300)
        # Each component grid is itself a constant equal to its scale squared.
        for comp, (wl, w) in zip(r.components, wl_w):
            scale2 = (dx * dx / (wl * z)) ** 2
            for i in range(m):
                for j in range(m):
                    self.assertAlmostEqual(comp.result.intensity[i][j], scale2,
                                           delta=1e-300)

    def test_fields_never_cross_added_even_same_wavelength(self):
        # Two equal-weight modes at the SAME wavelength with opposite uniform
        # fields (+s and -s everywhere).  Coherent field addition would cancel
        # to zero; incoherent intensity stacking must give 2*s^2 at every bin.
        m = 6
        r = fd.fraunhofer_modes(
            [(0.5, [([[1.0]], 1.0), ([[ -1.0]], 1.0)])], 1.0, 1.0, pad=m
        )
        s2 = (1.0 / (0.5 * 1.0)) ** 2
        for i in range(m):
            for j in range(m):
                self.assertAlmostEqual(r.intensity[i][j], 2.0 * s2, delta=1e-20)
        self.assertEqual(len(r.components), 2)

    def test_weight_linearity_and_additivity(self):
        a = fd.rectangular_aperture(8, 4.0, 2.0, 1.0)
        base = fd.fraunhofer_modes([(0.5, [(a, 1.0)])], 1.0, 1.0)
        peak = max(max(row) for row in base.intensity)

        # Exact linearity: w multiplies intensity everywhere.
        for w in (0.0, 0.5, 3.0):
            rw = fd.fraunhofer_modes([(0.5, [(a, w)])], 1.0, 1.0)
            for i in range(8):
                for j in range(8):
                    self.assertAlmostEqual(rw.intensity[i][j], w * base.intensity[i][j],
                                           delta=1e-12 * max(1.0, peak))

        # Additivity of weights across separate modes (same aperture).
        r12 = fd.fraunhofer_modes([(0.5, [(a, 1.0), (a, 2.0)])], 1.0, 1.0)
        r3 = fd.fraunhofer_modes([(0.5, [(a, 3.0)])], 1.0, 1.0)
        for i in range(8):
            for j in range(8):
                self.assertAlmostEqual(r12.intensity[i][j], r3.intensity[i][j],
                                       delta=1e-12 * max(1.0, peak))
        self.assertEqual(r12.total_weight, 3.0)
        self.assertEqual(len(r12.components), 2)

    def test_zero_weight_component_is_zero_but_kept(self):
        a = fd.rectangular_aperture(5, 2.0, 2.0, 1.0)
        r = fd.fraunhofer_modes(
            [{"wavelength": 0.5, "modes": [{"aperture": a, "weight": 0.0}]}], 1.0, 1.0
        )
        self.assertEqual(max(max(row) for row in r.intensity), 0.0)
        self.assertEqual(r.detector_power, 0.0)
        self.assertEqual(r.total_weight, 0.0)
        self.assertEqual(len(r.components), 1)
        # The mode was still propagated: its unweighted field/power survive.
        self.assertGreater(r.components[0].result.output_power, 0.0)
        self.assertEqual(r.components[0].weighted_power, 0.0)


class TestMultimodeGridAndBounds(unittest.TestCase):
    def test_default_grid_is_max_aperture_side(self):
        r = fd.fraunhofer_modes(
            [(0.5, [fd.rectangular_aperture(6, 2.0, 2.0, 1.0)]),
             (0.7, [[[1.0, 0.0], [0.0, 1.0]]])],
            1.0, 1.0,
        )
        self.assertEqual(r.m, 6)
        self.assertEqual(r.pad, 6)
        self.assertEqual(r.center, (3, 3))
        for comp in r.components:
            self.assertEqual(comp.result.m, 6)

    def test_three_by_three_boundary_count_accepted(self):
        a = [[1.0]]
        wls = (0.4, 0.5, 0.6)
        groups = [
            (wls[g], [(a, 1.0), (a, 0.5), (a, 0.25)])
            for g in range(3)
        ]
        r = fd.fraunhofer_modes(groups, 1.0, 1.0, pad=4)
        self.assertEqual(len(r.components), 9)
        self.assertEqual(r.wavelengths, wls)
        self.assertEqual(r.m, 4)

    def test_per_wavelength_pitch_is_independent(self):
        r = fd.fraunhofer_modes(
            [(0.4, [[[1.0]]]), (0.8, [[[1.0]]])], 1.0, 1.0, pad=8
        )
        self.assertAlmostEqual(r.components[0].result.du, 0.4 / 8, delta=1e-18)
        self.assertAlmostEqual(r.components[1].result.du, 0.8 / 8, delta=1e-18)
        # Shared result is index-aligned; physical coordinate arrays differ.
        self.assertNotEqual(r.components[0].result.x, r.components[1].result.x)

    def test_pad_bounds_against_largest_mode(self):
        a = fd.rectangular_aperture(10, 2.0, 2.0, 1.0)
        with self.assertRaises(ValueError) as cm:
            fd.fraunhofer_modes([(0.5, [a])], 1.0, 1.0, pad=8)
        self.assertIn("pad", str(cm.exception))
        with self.assertRaises(ValueError):
            fd.fraunhofer_modes([(0.5, [a])], 1.0, 1.0, pad=65)
        with self.assertRaises(TypeError):
            fd.fraunhofer_modes([(0.5, [a])], 1.0, 1.0, pad=16.0)
        with self.assertRaises(TypeError):
            fd.fraunhofer_modes([(0.5, [a])], 1.0, 1.0, pad=False)


class TestMultimodeErrors(unittest.TestCase):
    def setUp(self):
        self.a = fd.rectangular_aperture(4, 2.0, 2.0, 1.0)

    def check_raises(self, groups, dx=1.0, z=1.0, pad=None,
                     exc=(TypeError, ValueError), contains=None):
        with self.assertRaises(exc) as cm:
            fd.fraunhofer_modes(groups, dx, z, pad=pad)
        if contains is not None:
            self.assertIn(contains, str(cm.exception))

    def test_group_count_and_shape(self):
        self.check_raises("not-iterable", exc=TypeError, contains="groups")
        self.check_raises(123, exc=TypeError)
        self.check_raises({"wavelength": 0.5, "modes": [[[1.0]]]},
                          exc=TypeError, contains="wrap it in a list")
        self.check_raises([], contains="1..3")
        self.check_raises(
            [(0.4, [[[1.0]]]), (0.5, [[[1.0]]]), (0.6, [[[1.0]]]), (0.7, [[[1.0]]])],
            contains="1..3",
        )
        self.check_raises([0.5], exc=TypeError, contains="groups[0]")
        self.check_raises([(0.5,)], exc=TypeError)
        self.check_raises([(0.5, [[[1.0]]], 1.0)], exc=TypeError)
        self.check_raises([{}], contains="'wavelength'")
        self.check_raises([{"wavelength": 0.5}], contains="'modes'")
        self.check_raises([{"wavelength": 0.5, "modes": [[[1.0]]], "x": 1}],
                          contains="unknown group key")

    def test_mode_count_and_shape(self):
        self.check_raises([(0.5, [])], contains="1..3 modes")
        four = [(self.a, 1.0)] * 4
        self.check_raises([(0.5, four)], contains="1..3 modes")
        self.check_raises([{"wavelength": 0.5, "modes": "ab"}], exc=TypeError)
        self.check_raises([{"wavelength": 0.5, "modes": [{}]}],
                          contains="'aperture'")
        self.check_raises(
            [{"wavelength": 0.5, "modes": [{"aperture": self.a, "weight": 1, "q": 2}]}],
            contains="unknown mode key",
        )
        self.check_raises([(0.5, [(self.a, 1.0, 2.0)])],
                          contains="1 or 2 items")

    def test_weights(self):
        self.check_raises([(0.5, [(self.a, -0.01)])], contains="weight")
        self.check_raises([(0.5, [(self.a, float("nan"))])],
                          exc=ValueError, contains="weight")
        self.check_raises([(0.5, [(self.a, float("inf"))])],
                          exc=ValueError, contains="weight")
        self.check_raises([(0.5, [(self.a, True)])], exc=TypeError, contains="weight")
        self.check_raises([(0.5, [(self.a, "1.0")])], exc=TypeError)
        self.check_raises(
            [{"wavelength": 0.5, "modes": [{"aperture": self.a, "weight": None}]}],
            exc=TypeError, contains="weight",
        )

    def test_wavelengths_and_physical_inputs(self):
        self.check_raises([(0.0, [(self.a, 1.0)])], contains="wavelength")
        self.check_raises([(-0.5, [(self.a, 1.0)])], contains="wavelength")
        self.check_raises([(float("nan"), [(self.a, 1.0)])], exc=ValueError)
        self.check_raises([(True, [(self.a, 1.0)])], exc=TypeError)
        self.check_raises([(0.5, [(self.a, 1.0)])], dx=0.0, contains="dx")
        self.check_raises([(0.5, [(self.a, 1.0)])], z=-1.0, contains="z")
        self.check_raises(
            [(0.5, [(self.a, 1.0)]), (0.50, [(self.a, 1.0)])],
            contains="duplicate wavelength",
        )
        # Distinct float values that differ only by round-off are still distinct.
        r = fd.fraunhofer_modes(
            [(0.5, [(self.a, 1.0)]), (0.5 + 1e-16, [(self.a, 1.0)])], 1.0, 1.0
        )
        self.assertEqual(len(r.wavelengths), 2)

    def test_bad_aperture_is_localized_inside_mode(self):
        # Non-square grid: mode path reports the offending aperture location.
        self.check_raises(
            [(0.5, [([[1.0, 0.0], [0.0, 0.0, 0.0]], 1.0)])],
            exc=ValueError, contains="modes[0] aperture",
        )
        # Non-numeric element keeps the precise element coordinate.
        self.check_raises(
            [(0.5, [[[["1.0"]]]])],
            exc=TypeError, contains="aperture[0][0]",
        )
        # Oversized aperture is rejected even before pad validation.
        big = [[1.0] * 33] * 33
        self.check_raises([(0.5, [big])], exc=ValueError)

    def test_failure_leaves_no_partial_state(self):
        # Pure-function call: a raising call must not affect an earlier result
        # and the arguments themselves stay untouched.
        good = fd.fraunhofer_modes([(0.5, [(self.a, 1.0)])], 1.0, 1.0)
        snapshot = [row[:] for row in good.intensity]
        with self.assertRaises(ValueError):
            fd.fraunhofer_modes(
                [(0.5, [(self.a, 1.0)]), (0.6, [(self.a, -1.0)])], 1.0, 1.0
            )
        self.assertEqual(good.intensity, snapshot)


if __name__ == "__main__":
    unittest.main()
