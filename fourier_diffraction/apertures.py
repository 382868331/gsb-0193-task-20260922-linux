"""Deterministic aperture generation.

All generators return a fresh ``list[list[complex]]`` of shape ``N x N``
with purely real, unit transmission inside the aperture and zero outside.
Generation uses the *same* pixel lattice as the Fraunhofer propagator:
samples at ``x[i] = (i - (N - 1) / 2) * dx``, i.e. zero-centered for even
grids (no pixel sits on the origin).
"""

from __future__ import annotations

import math

from .validation import DiffractionError, checked_int, checked_real

__all__ = ["circular_aperture", "rectangular_aperture", "grid_coordinates"]

MAX_APERTURE_N = 32
MIN_APERTURE_N = 1


def grid_coordinates(n: int, dx: float):
    """Return ``(xs, ys)``: the physical pixel coordinates (lists of float).

    Coordinates are centered between the two middle pixels for even ``n``:
    ``x[i] = (i - (n - 1) / 2) * dx``.  ``xs == ys`` for the square grid.
    """
    n = checked_int(n, "n", minimum=MIN_APERTURE_N, maximum=MAX_APERTURE_N)
    dx = checked_real(dx, "dx", positive=True)
    offset = (n - 1) / 2.0
    coords = [(i - offset) * dx for i in range(n)]
    return coords, list(coords)


def _empty(n: int):
    return [[0.0j for _ in range(n)] for _ in range(n)]


def circular_aperture(n: int, dx: float, radius: float):
    """Binary circular aperture of physical ``radius``.

    Pixel (i, j) is open iff ``sqrt(x_i**2 + y_j**2) <= radius``
    (boundary inclusive), with ``i`` the y/row index and ``j`` the
    x/column index.
    """
    n = checked_int(n, "n", minimum=MIN_APERTURE_N, maximum=MAX_APERTURE_N)
    dx = checked_real(dx, "dx", positive=True)
    radius = checked_real(radius, "radius", positive=True)
    xs, ys = grid_coordinates(n, dx)
    field = _empty(n)
    r2 = radius * radius
    for i, y in enumerate(ys):
        row = field[i]
        for j, x in enumerate(xs):
            if x * x + y * y <= r2:
                row[j] = 1.0 + 0.0j
    return field


def rectangular_aperture(n: int, dx: float, width: float, height: float):
    """Binary rectangular aperture of physical ``width`` (x) and ``height`` (y).

    Open iff ``|x_i| <= width / 2`` and ``|y_j| <= height / 2``
    (boundaries inclusive).
    """
    n = checked_int(n, "n", minimum=MIN_APERTURE_N, maximum=MAX_APERTURE_N)
    dx = checked_real(dx, "dx", positive=True)
    width = checked_real(width, "width", positive=True)
    height = checked_real(height, "height", positive=True)
    xs, ys = grid_coordinates(n, dx)
    field = _empty(n)
    hx, hy = width / 2.0, height / 2.0
    for i, y in enumerate(ys):
        row = field[i]
        if abs(y) <= hy:
            for j, x in enumerate(xs):
                if abs(x) <= hx:
                    row[j] = 1.0 + 0.0j
    return field
