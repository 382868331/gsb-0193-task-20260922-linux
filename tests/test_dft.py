"""Tests for the self-implemented DFT core.

Expected values come only from the independent literal-sum reference in
``tests/_reference.py``; the package core is never used to generate its
own expectations.
"""

import cmath
import math
import os
import random
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import unittest

from fourier_diffraction.dft import dft1d, dft2_raw, fftshift_2d
from _reference import naive_dft1, naive_dft2


def _close(a, b, rel=1e-12, abs_=1e-12):
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_)


class Dft1dTests(unittest.TestCase):
    def test_dc_single_element(self):
        self._assert_close(dft1d([3.0 + 4.0j]), [3.0 + 4.0j])

    def test_known_two_point(self):
        got = dft1d([1.0j, 2.0j])
        self._assert_close(got, [3.0j, -1.0j])

    def test_impulse_is_flat_spectrum(self):
        for n in (2, 8, 32, 64):
            with self.subTest(n=n):
                x = [0.0j] * n
                x[3 % n] = 2.0 - 1.0j
                spec = dft1d(x)
                self.assertEqual(len(spec), n)
                for k, v in enumerate(spec):
                    expected = (2.0 - 1.0j) * cmath.exp(
                        -2j * math.pi * k * (3 % n) / n
                    )
                    self.assertTrue(_close(v, expected), (n, k, v, expected))

    def test_matches_naive_all_lengths_1_to_16(self):
        rng = random.Random(20260923)
        for n in range(1, 17):
            x = [
                complex(rng.uniform(-2, 2), rng.uniform(-2, 2))
                for _ in range(n)
            ]
            with self.subTest(n=n):
                self._assert_close(dft1d(x), naive_dft1(x))

    def test_matches_naive_at_contract_sizes(self):
        rng = random.Random(4242)
        for n in (31, 32, 33, 48, 63, 64):
            x = [
                complex(rng.uniform(-1, 1), rng.uniform(-1, 1))
                for _ in range(n)
            ]
            with self.subTest(n=n):
                self._assert_close(dft1d(x), naive_dft1(x))

    def _assert_close(self, got, expected, rel=1e-11, abs_=1e-10):
        self.assertEqual(len(got), len(expected))
        for k, (a, b) in enumerate(zip(got, expected)):
            self.assertTrue(
                _close(a, b, rel, abs_),
                f"index {k}: got {a!r}, expected {b!r}",
            )


class Dft2dTests(unittest.TestCase):
    def test_matches_direct_2d_sum(self):
        rng = random.Random(7)
        for m in (1, 2, 3, 4, 5, 7, 8, 11, 16):
            canvas = [
                [complex(rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5))
                 for _ in range(m)]
                for _ in range(m)
            ]
            with self.subTest(m=m):
                got = dft2_raw(canvas)
                expected = naive_dft2(canvas)
                for p in range(m):
                    for q in range(m):
                        self.assertTrue(
                            _close(got[p][q], expected[p][q], 1e-10, 1e-9),
                            f"m={m} ({p},{q}): {got[p][q]!r} vs {expected[p][q]!r}",
                        )

    def test_dc_term_is_total_sum(self):
        rng = random.Random(99)
        m = 12
        canvas = [
            [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(m)]
            for _ in range(m)
        ]
        spec = dft2_raw(canvas)
        total = 0.0j
        for row in canvas:
            total += sum(row)
        self.assertTrue(_close(spec[0][0], total, 1e-12, 1e-12))


class FftShiftTests(unittest.TestCase):
    def test_even_grid_dc_at_half_index(self):
        m = 8
        raw = [[0.0j] * m for _ in range(m)]
        raw[0][0] = 1.0j  # DC in raw corner indexing
        shifted = fftshift_2d(raw, m)
        self.assertTrue(_close(shifted[m // 2][m // 2], 1.0j))
        for i in range(m):
            for j in range(m):
                if (i, j) != (m // 2, m // 2):
                    self.assertEqual(shifted[i][j], 0.0j)

    def test_even_grid_origin_wraps_to_negative_edge(self):
        # Raw mode k=1 must appear at shifted index m//2 + 1, and raw
        # mode k=m-1 == -1 at shifted index m//2 - 1.
        m = 6
        raw = [[0.0j] * m for _ in range(m)]
        raw[1][m - 1] = 5.0
        shifted = fftshift_2d(raw, m)
        self.assertEqual(shifted[m // 2 + 1][m // 2 - 1], 5.0)

    def test_shift_is_involution(self):
        m = 64
        rng = random.Random(11)
        raw = [
            [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(m)]
            for _ in range(m)
        ]
        twice = fftshift_2d(fftshift_2d(raw, m), m)
        for i in range(m):
            for j in range(m):
                self.assertEqual(twice[i][j], raw[i][j])

    def test_odd_grid_is_symmetric_about_center(self):
        # Odd M: fftshift must map raw mode k to index k + M//2 (mod M);
        # the grid -h..h must be symmetric and DC at M//2.
        m = 5
        h = m // 2
        raw = [[0.0j] * m for _ in range(m)]
        raw[0][0] = 1.0 + 0.0j   # DC
        raw[1][m - 1] = 7.0j     # modes (k_row=+1, k_col=-1)
        shifted = fftshift_2d(raw, m)
        self.assertEqual(shifted[h][h], 1.0 + 0.0j)
        self.assertEqual(shifted[h + 1][h - 1], 7.0j)
        # Nothing else is non-zero.
        for i in range(m):
            for j in range(m):
                if (i, j) not in {(h, h), (h + 1, h - 1)}:
                    self.assertEqual(shifted[i][j], 0.0j)

    def test_odd_grid_matches_literal_roll_for_all_modes(self):
        m = 7
        h = m // 2
        rng = random.Random(123)
        raw = [
            [complex(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(m)]
            for _ in range(m)
        ]
        shifted = fftshift_2d(raw, m)
        for i in range(m):
            for j in range(m):
                self.assertEqual(
                    shifted[i][j], raw[(i - h) % m][(j - h) % m]
                )


if __name__ == "__main__":
    unittest.main()
