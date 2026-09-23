"""Independent references for the test suite.

Nothing in this module imports the package's DFT core or propagator.
Expected values are produced from first principles with ``cmath`` /
``math`` only:

* :func:`naive_dft1` -- direct O(n^2) DFT sum;
* :func:`naive_dft2` -- direct 2-D sum for small canvases;
* :func:`parseval_sum_spectrum` -- independent total spectral energy
  via separable row transforms with a shared twiddle table;
* :func:`rect_spectrum_closed_form` -- exact discrete geometric-series
  closed form for a rectangular (contiguous-index) aperture.
"""

from __future__ import annotations

import cmath
import math

_TWIDDLE: dict[int, list[complex]] = {}


def twiddle_table(m: int) -> list[complex]:
    """E[k] = exp(-2*pi*i*k/m), shared by every row/column of size m."""
    table = _TWIDDLE.get(m)
    if table is None:
        table = [cmath.exp(-2j * math.pi * k / m) for k in range(m)]
        _TWIDDLE[m] = table
    return table


def naive_dft1(x: list[complex]) -> list[complex]:
    """Direct, literal DFT definition: X[k] = sum_n x[n] exp(-2pi i kn/N)."""
    n = len(x)
    e = twiddle_table(n)
    out = []
    for k in range(n):
        acc = 0.0j
        for m, sample in enumerate(x):
            acc += sample * e[(k * m) % n]
        out.append(acc)
    return out


def naive_dft2(canvas: list[list[complex]]) -> list[list[complex]]:
    """Direct O(M^4) 2-D DFT. Intended for M <= 16."""
    m = len(canvas)
    e = twiddle_table(m)
    out = [[0.0j] * m for _ in range(m)]
    for p in range(m):
        ep = [e[(p * i) % m] for i in range(m)]
        row_out = out[p]
        for q in range(m):
            eq = [e[(q * j) % m] for j in range(m)]
            acc = 0.0j
            for i in range(m):
                epi = ep[i]
                src = canvas[i]
                for j in range(m):
                    acc += src[j] * epi * eq[j]
            row_out[q] = acc
    return out


def parseval_sum_spectrum(canvas: list[list[complex]]) -> float:
    """Independent sum of |DFT2(canvas)|^2.

    Uses Parseval per dimension with literal 1-D DFTs: after transforming
    every row, summing |Y|^2 over all rows and frequencies and
    multiplying by M equals the full 2-D spectral energy.
    """
    m = len(canvas)
    total = 0.0
    for row in canvas:
        spec = naive_dft1(row)
        total += math.fsum(v.real * v.real + v.imag * v.imag for v in spec)
    return m * total


def _geom_series(k: int, lo: int, hi: int, m: int) -> complex:
    """sum_{n=lo}^{hi} exp(-2*pi*i*k*n/M), exact geometric-series form."""
    length = hi - lo + 1
    k_mod = k % m
    if k_mod == 0:
        return complex(length, 0.0)
    # exp(-i*pi*k*(lo+hi)/M) * sin(pi*k*L/M) / sin(pi*k/M)
    phase = cmath.exp(-1j * math.pi * k * (lo + hi) / m)
    amplitude = math.sin(math.pi * k * length / m) / math.sin(
        math.pi * k / m
    )
    return phase * amplitude


def rect_spectrum_closed_form(
    m: int, i0: int, i1: int, j0: int, j1: int
) -> list[list[complex]]:
    """Raw (unshifted) DFT of a rectangle of unit samples on an M canvas.

    Open rows are the contiguous range ``i0..i1`` and open columns
    ``j0..j1`` (canvas indices); everything else is zero.
    """
    out = [[0.0j] * m for _ in range(m)]
    for p in range(m):
        row = out[p]
        si = _geom_series(p, i0, i1, m)
        for q in range(m):
            row[q] = si * _geom_series(q, j0, j1, m)
    return out
