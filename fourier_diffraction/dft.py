"""Self-contained discrete Fourier transform core.

No FFT package (numpy, scipy, ...) is imported anywhere in this project.
The transform is computed by the package itself in two ways:

* iterative radix-2 Cooley-Tukey for power-of-two lengths (the common
  case, including 32 and 64);
* a direct O(n^2) DFT for any other length.

Both paths implement the same convention as a standard unnormalized FFT::

    X[k] = sum_{n=0}^{N-1} x[n] * exp(-2*pi*i*k*n/N)
"""

from __future__ import annotations

import cmath
import math

__all__ = ["dft1d", "dft2_raw", "fftshift_2d"]

_TWIDDLE_CACHE: dict[int, list[complex]] = {}


def _twiddles(n: int) -> list[complex]:
    """exp(-2*pi*i*k/n) for k in 0..n/2-1 (cached, deterministic)."""
    table = _TWIDDLE_CACHE.get(n)
    if table is None:
        table = [
            cmath.exp(-2j * math.pi * k / n) for k in range(n // 2)
        ]
        _TWIDDLE_CACHE[n] = table
    return table


def _radix2(x: list[complex]) -> list[complex]:
    """Iterative in-place radix-2 DFT; returns a new list."""
    a = list(x)
    n = len(a)
    # Bit-reversal permutation.
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        half = length >> 1
        step = n // length
        full = _twiddles(n)
        # Twiddles for this stage: exp(-2*pi*i*k/length).
        tw = [full[k * step] for k in range(half)]
        for start in range(0, n, length):
            for k in range(half):
                u = a[start + k]
                v = a[start + k + half] * tw[k]
                a[start + k] = u + v
                a[start + k + half] = u - v
        length <<= 1
    return a


def _naive_dft(x: list[complex]) -> list[complex]:
    """Direct O(n^2) DFT, used for lengths that are not powers of two."""
    n = len(x)
    out: list[complex] = []
    for k in range(n):
        acc = 0.0j
        # Accumulate in natural order; lengths here are at most 64.
        for m, sample in enumerate(x):
            acc += sample * cmath.exp(-2j * math.pi * k * m / n)
        out.append(acc)
    return out


def dft1d(x: list[complex]) -> list[complex]:
    """Unnormalized DFT of a 1-D complex sequence (self-implemented)."""
    n = len(x)
    if n == 1:
        return [complex(x[0])]
    if n & (n - 1) == 0:
        return _radix2(x)
    return _naive_dft(x)


def dft2_raw(canvas: list[list[complex]]):
    """2-D DFT of a square ``M x M`` canvas via separable 1-D DFTs.

    Returns the *raw* (unshifted) transform ``X[k_row, k_col]``.
    """
    m = len(canvas)
    # Transform every row.
    rows = [dft1d(row) for row in canvas]
    # Transform every column of the row-transformed matrix.
    result = [[0.0j] * m for _ in range(m)]
    column = [0.0j] * m
    for col in range(m):
        for row in range(m):
            column[row] = rows[row][col]
        spec = dft1d(column)
        for row in range(m):
            result[row][col] = spec[row]
    return result


def fftshift_2d(raw: list[list[complex]], m: int) -> list[list[complex]]:
    """Cyclic shift by ``(-m//2, -m//2)`` (the standard fftshift).

    Output index ``i`` holds raw frequency ``k = i - m//2`` (mod m), so
    DC (k = 0) sits at index ``m//2`` and index 0 holds the most negative
    frequency.  For even m the sampled grid is ``-m/2 ... m/2 - 1``; for
    odd m the symmetric grid is ``-(m-1)/2 ... (m-1)/2``.
    """
    h = m // 2
    shifted = [[0.0j] * m for _ in range(m)]
    for i in range(m):
        src_row = raw[(i - h) % m]
        dst = shifted[i]
        for j in range(m):
            dst[j] = src_row[(j - h) % m]
    return shifted
