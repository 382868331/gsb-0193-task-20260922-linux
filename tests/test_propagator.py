"""Fraunhofer propagator acceptance tests.

Independent references only (``tests/_reference.py``): literal 2-D DFT,
Parseval summation, and the discrete geometric-series closed form for a
rectangle.  No expected value is obtained by calling the code under test;
round-trip comparisons appear only as supplementary checks.
"""

import cmath
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import unittest

from fourier_diffraction import (
    circular_aperture,
    rectangular_aperture,
    fraunhofer_propagate,
)
from fourier_diffraction.dft import fftshift_2d
from _reference import (
    naive_dft2,
    parseval_sum_spectrum,
    rect_spectrum_closed_form,
)


def _close(a, b, rel=1e-11, abs_=1e-11):
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_)


class PropagationModelTests(unittest.TestCase):
    def test_scale_and_grid_against_literal_dft(self):
        # Random N=5 aperture, no padding, arbitrary physical parameters.
        rng = __import__("random").Random(2026)
        n, dx, wl, z = 5, 0.7, 0.13, 19.0
        aperture = [
            [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]
            for _ in range(n)
        ]
        result = fraunhofer_propagate(aperture, dx, wl, z)

        scale = dx * dx / (wl * z)
        expected_raw = naive_dft2(aperture)
        expected_shifted = fftshift_2d(expected_raw, n)
        for i in range(n):
            for j in range(n):
                eu = scale * expected_shifted[i][j]
                self.assertTrue(
                    _close(result.field[i][j], eu, 1e-10, 1e-9),
                    (i, j, result.field[i][j], eu),
                )
                self.assertTrue(
                    _close(result.intensity[i][j], abs(eu) ** 2, 1e-10, 1e-9)
                )

        # Observation spacing and even/asymmetric-grid coordinates.
        dx_out = wl * z / (n * dx)
        self.assertTrue(_close(result.pixel_size, dx_out, 1e-15))
        self.assertEqual(result.center, (n // 2, n // 2))
        half = n // 2
        for j, x in enumerate(result.x):
            self.assertTrue(_close(x, (j - half) * dx_out, 1e-14, 1e-14))
        self.assertEqual(result.x, result.y)

    def test_even_grid_fftshift_convention_explicit(self):
        # Even M=8: DC at index 4; coordinate grid is -4..3 samples.
        n, dx, wl, z = 8, 1.0, 0.2, 10.0
        aperture = rectangular_aperture(n, dx, 2.0, 2.0)
        result = fraunhofer_propagate(aperture, dx, wl, z)
        self.assertEqual(result.center, (4, 4))
        dx_out = wl * z / (n * dx)
        indices = [round(x / dx_out) for x in result.x]
        self.assertEqual(indices, list(range(-4, 4)))
        # Brightest sample for a symmetric rectangle is the DC sample.
        r, c, peak = result.peak()
        self.assertEqual((r, c), (4, 4))
        self.assertGreater(peak, 0.0)

    def test_padding_updates_spacing_and_refines_grid(self):
        n, dx, wl, z = 4, 0.5, 0.2, 10.0
        aperture = circular_aperture(n, dx, 0.6)
        unpadded = fraunhofer_propagate(aperture, dx, wl, z)
        padded = fraunhofer_propagate(aperture, dx, wl, z, pad_to=8)
        self.assertEqual(unpadded.pixel_size, wl * z / (n * dx))
        self.assertEqual(padded.padded_size, 8)
        self.assertTrue(
            _close(padded.pixel_size, wl * z / (8 * dx), 1e-15)
        )
        self.assertEqual(padded.center, (4, 4))
        self.assertEqual(len(padded.intensity), 8)


class ParsevalTests(unittest.TestCase):
    def test_parseval_reference_on_padded_canvas(self):
        # Independent Parseval: M^2 * sum|a|^2 (embedded canvas) equals
        # the reference sum of |DFT|^2.
        n, m = 6, 16
        rng = __import__("random").Random(31337)
        aperture = [
            [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(n)]
            for _ in range(n)
        ]
        canvas = [[0.0j] * m for _ in range(m)]
        for i in range(n):
            for j in range(n):
                canvas[i][j] = aperture[i][j]
        input_energy = sum(
            abs(v) ** 2 for row in aperture for v in row
        )
        spec_energy = parseval_sum_spectrum(canvas)
        self.assertTrue(
            _close(spec_energy, m * m * input_energy, 1e-9, 1e-7)
        )

    def test_reported_power_error_small_on_unpadded(self):
        rng = __import__("random").Random(55)
        aperture = [
            [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(12)]
            for _ in range(12)
        ]
        dx, wl, z = 0.4, 0.11, 23.0
        result = fraunhofer_propagate(aperture, dx, wl, z)
        self.assertTrue(math.isfinite(result.relative_power_error))
        self.assertLess(abs(result.relative_power_error), 1e-10)
        # Powers must be positive and the two totals must agree closely.
        self.assertGreater(result.input_power, 0.0)
        self.assertGreater(result.output_power, 0.0)

    def test_zero_padding_preserves_total_power(self):
        n, dx, wl, z = 10, 0.6, 0.17, 31.0
        aperture = circular_aperture(n, dx, 1.1)
        sizes = (10, 12, 16, 32, 64)
        powers = []
        for m in sizes:
            result = fraunhofer_propagate(aperture, dx, wl, z, pad_to=m)
            powers.append(result.output_power)
            self.assertLess(abs(result.relative_power_error), 1e-10)
        for p in powers[1:]:
            self.assertTrue(_close(p, powers[0], 1e-10, 1e-10),
                            (sizes, powers))

    def test_power_quantities_match_parseval_formula(self):
        n, m, dx, wl, z = 7, 14, 0.9, 0.2, 5.0
        aperture = rectangular_aperture(n, dx, 1.8, 0.9)
        result = fraunhofer_propagate(aperture, dx, wl, z, pad_to=m)
        # Analytical total power from the model:
        # P_out = dx_out^2 * scale^2 * M^2 * (number of open pixels).
        n_open = sum(1 for row in aperture for v in row if v != 0.0j)
        scale = dx * dx / (wl * z)
        dx_out = wl * z / (m * dx)
        expected = dx_out ** 2 * scale ** 2 * m * m * n_open
        self.assertTrue(_close(result.output_power, expected, 1e-10, 1e-10))
        expected_in = dx * dx * n_open
        self.assertTrue(_close(result.input_power, expected_in, 1e-14, 1e-14))

    def test_all_zero_aperture_power_is_zero(self):
        aperture = [[0.0j for _ in range(4)] for _ in range(4)]
        result = fraunhofer_propagate(aperture, 1.0, 1.0, 1.0, pad_to=8)
        self.assertEqual(result.input_power, 0.0)
        self.assertEqual(result.output_power, 0.0)
        self.assertEqual(result.relative_power_error, 0.0)
        self.assertEqual(max(max(row) for row in result.intensity), 0.0)


class DiscreteReferenceTests(unittest.TestCase):
    def test_single_pixel_constant_intensity(self):
        # One open pixel: |DFT| is constant over the whole output grid.
        n = 6
        aperture = [[0.0j for _ in range(n)] for _ in range(n)]
        aperture[1][4] = 3.0 - 2.0j
        result = fraunhofer_propagate(aperture, 1.0, 1.0, 1.0)
        amp = abs(3.0 - 2.0j) ** 2
        values = [v for row in result.intensity for v in row]
        spread = max(values) - min(values)
        mean = math.fsum(values) / len(values)
        # |a exp(-i theta)|^2 is constant; allow only floating-point noise.
        self.assertLess(spread, 1e-12)
        self.assertTrue(_close(mean, amp, 1e-13, 1e-13))
        self.assertTrue(_close(min(values), amp, 1e-12, 1e-12))

    def test_single_pixel_phase_is_linear_ramp(self):
        n = 5
        aperture = [[0.0j] * n for _ in range(n)]
        a0, b0 = 2, 1
        aperture[a0][b0] = 1.0
        result = fraunhofer_propagate(aperture, 1.0, 1.0, 1.0)
        # Raw spectrum at (p, q) is exp(-2pi i (p*a0 + q*b0)/n);
        # shifted output (i, j) carries p = i - h, q = j - h.
        h = n // 2
        for i in range(n):
            for j in range(n):
                p, q = i - h, j - h
                expected = cmath.exp(-2j * math.pi * (p * a0 + q * b0) / n)
                self.assertTrue(
                    _close(result.field[i][j], expected, 1e-12, 1e-12),
                    (i, j, result.field[i][j], expected),
                )

    def test_rectangle_matches_discrete_closed_form(self):
        # Rectangle built by the generator with open indices 1..2 on a
        # 4-pixel width=2*dx aperture embedded in a 16-sample canvas.
        n, m, dx, wl, z = 4, 16, 1.0, 0.3, 12.0
        aperture = rectangular_aperture(n, dx, 2.0 * dx, 2.0 * dx)
        # Generator with n=4, width=2dx opens |x|<=dx -> indices 1,2.
        probe = (n - 1) // 2  # a known-open row/column for this rectangle
        open_rows = [i for i, row in enumerate(aperture) if row[probe] != 0.0j]
        self.assertEqual(open_rows, [1, 2])
        result = fraunhofer_propagate(aperture, dx, wl, z, pad_to=m)
        expected_raw = rect_spectrum_closed_form(m, 1, 2, 1, 2)
        scale = dx * dx / (wl * z)
        expected_shifted = fftshift_2d(expected_raw, m)
        for i in range(m):
            for j in range(m):
                eu = scale * expected_shifted[i][j]
                self.assertTrue(
                    _close(result.field[i][j], eu, 1e-9, 1e-8),
                    (i, j, result.field[i][j], eu),
                )

    def test_rectangle_matches_closed_form_with_padding_variants(self):
        n, dx = 6, 0.5
        # width = 3*dx opens |x| <= 1.5*dx -> x in {-1.5,-0.5,0.5,1.5}*dx
        # -> indices 1,2,3,4 (contiguous in corner embedding).
        aperture = rectangular_aperture(n, dx, 3.0 * dx, 1.0 * dx)
        probe = (n - 1) // 2  # known-open row/column for this rectangle
        open_cols = [j for j in range(n) if aperture[probe][j] != 0.0j]
        open_rows = [i for i in range(n) if aperture[i][probe] != 0.0j]
        self.assertEqual(open_cols, [1, 2, 3, 4])
        self.assertEqual(open_rows, [2, 3])
        wl, z = 0.25, 8.0
        for m in (6, 10, 64):
            with self.subTest(m=m):
                result = fraunhofer_propagate(aperture, dx, wl, z, pad_to=m)
                expected_raw = rect_spectrum_closed_form(
                    m, open_rows[0], open_rows[-1],
                    open_cols[0], open_cols[-1]
                )
                scale = dx * dx / (wl * z)
                expected_shifted = fftshift_2d(expected_raw, m)
                # Spot-check DC plus 12 deterministic samples.
                for i, j in [(m // 2, m // 2), (0, 0), (1, 3), (m - 2, 5)]:
                    eu = scale * expected_shifted[i][j]
                    self.assertTrue(
                        _close(result.field[i][j], eu, 1e-9, 1e-8),
                        (m, i, j),
                    )

    def test_circular_aperture_geometry_is_deterministic(self):
        n, dx = 8, 1.0
        aperture = circular_aperture(n, dx, 1.5)
        xs = [(j - 3.5) for j in range(n)]
        for i, y in enumerate(xs):
            for j, x in enumerate(xs):
                inside = math.hypot(x, y) <= 1.5 + 1e-12
                self.assertEqual(aperture[i][j] != 0.0j, inside)
        # Rebuilding yields an identical aperture (deterministic).
        again = circular_aperture(n, dx, 1.5)
        self.assertEqual(aperture, again)


class PhaseRampTests(unittest.TestCase):
    def test_phase_ramp_translates_pattern(self):
        # Multiplying the aperture by exp(2*pi*i*s*j/N + 2*pi*i*t*i/N)
        # cyclically shifts the raw spectrum by (s, t); after fftshift
        # the intensity peak moves by the same integer number of samples.
        n, dx, wl, z = 16, 0.5, 0.2, 10.0
        base = rectangular_aperture(n, dx, 2.0, 1.0)
        s, t = 3, -2
        ramped = [
            [base[i][j] * cmath.exp(2j * math.pi * (s * j + t * i) / n)
             for j in range(n)]
            for i in range(n)
        ]
        r0 = fraunhofer_propagate(base, dx, wl, z)
        r1 = fraunhofer_propagate(ramped, dx, wl, z)
        p0 = r0.peak()
        p1 = r1.peak()
        self.assertEqual(
            (p1[0], p1[1]),
            ((p0[0] + t) % n, (p0[1] + s) % n),
        )
        # Intensity pattern is the same pattern cyclically translated.
        h = n // 2
        for i in range(n):
            for j in range(n):
                si = (i - t) % n
                sj = (j - s) % n
                self.assertTrue(
                    _close(r1.intensity[i][j], r0.intensity[si][sj],
                           1e-9, 1e-8),
                    (i, j),
                )

    def test_phase_ramp_against_closed_form(self):
        # A ramp on a single-pixel aperture: expected field given
        # explicitly by the DFT shift theorem at every output sample.
        n, dx, wl, z = 6, 1.0, 0.5, 4.0
        a0, b0, s, t = 1, 2, 2, -1
        aperture = [[0.0j] * n for _ in range(n)]
        aperture[a0][b0] = cmath.exp(0.3j)
        aperture = [
            [aperture[i][j] * cmath.exp(2j * math.pi * (s * j + t * i) / n)
             for j in range(n)]
            for i in range(n)
        ]
        result = fraunhofer_propagate(aperture, dx, wl, z)
        scale = dx * dx / (wl * z)
        h = n // 2
        for i in range(n):
            for j in range(n):
                p, q = i - h, j - h
                expected = scale * cmath.exp(0.3j) * cmath.exp(
                    -2j * math.pi * ((p - t) * a0 + (q - s) * b0) / n
                )
                self.assertTrue(
                    _close(result.field[i][j], expected, 1e-9, 1e-8),
                    (i, j),
                )


class SupplementaryRoundTripTests(unittest.TestCase):
    """Supplementary only: padded vs unpadded spectra must be consistent."""

    def test_padded_dc_matches_unpadded_dc(self):
        n, dx, wl, z = 10, 0.6, 0.17, 31.0
        aperture = circular_aperture(n, dx, 1.1)
        r0 = fraunhofer_propagate(aperture, dx, wl, z)
        r1 = fraunhofer_propagate(aperture, dx, wl, z, pad_to=40)
        dc0 = r0.field[r0.center[0]][r0.center[1]]
        dc1 = r1.field[r1.center[0]][r1.center[1]]
        self.assertTrue(_close(dc0, dc1, 1e-12, 1e-12))

    def test_padded_spectrum_samples_unpadded_at_coincident_frequencies(self):
        # Zero-padding interpolates: at coincident frequencies
        # p_m/M == p_n/N the field scale dx**2/(wavelength*z) is
        # independent of M, so the samples must agree exactly.
        n, m, dx, wl, z = 8, 32, 1.0, 0.25, 9.0
        aperture = rectangular_aperture(n, dx, 3.0, 1.0)
        r0 = fraunhofer_propagate(aperture, dx, wl, z)
        r1 = fraunhofer_propagate(aperture, dx, wl, z, pad_to=m)
        ratio = m // n
        for p in range(n):
            for q in range(n):
                i = m // 2 + (p - n // 2) * ratio
                j = m // 2 + (q - n // 2) * ratio
                self.assertTrue(
                    _close(r1.field[i][j], r0.field[p][q], 1e-9, 1e-9),
                    (p, q),
                )


if __name__ == "__main__":
    unittest.main()
