"""Inverse problem for the Fraunhofer forward model of fourier_diffraction.

Given complex-amplitude observations at specified physical observation-plane
coordinates (not intensities) and a fixed boolean aperture-support mask, find
the complex aperture ``a`` supported on the mask that minimises

    J(a) = sum_o |F a - b|_o**2  +  lam * sum_p |a_p|**2 ,   lam > 0

where the forward operator ``F`` reuses the pixel centres, phase convention
and physical scale of :func:`fourier_diffraction.fraunhofer`:

    (F a)(x, y) = dx**2 / (wavelength * z)
                  * sum_{(i,j) in support} a[i,j]
                    * exp(-i 2 pi (x * xi_i + y * eta_j) / (wavelength * z))

with centred pixel coordinates ``xi_i = (i - n//2) * dx``,
``eta_j = (j - n//2) * dx``.  On the DFT grid ``x = k * du``,
``du = wavelength * z / (M * dx)`` this is exactly the module's forward
model, so observations may be taken from :func:`fourier_diffraction.fraunhofer`
output coordinates directly.

The normal equations

    (F* F + lam * I) a = F* b

are solved by a matrix-free conjugate-gradient iteration: the forward
operator and its conjugate transpose are applied through separable
per-axis exponential factors, and no dense Fourier matrix is ever formed.
Reaching the iteration cap is reported as ``stop_reason="max_iterations"``
and is *not* convergence.

Duplicate observations are kept as repeated measurements (a repeated triple
weighs its residual term accordingly); nothing is deduplicated.

Only the Python standard library is used.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass

from fourier_diffraction import (
    MAX_APERTURE,
    _as_complex,
    _finite_real,
    _strict_int,
)

# Defensive interface bound: memory per observation is O(N) complex factors.
MAX_OBSERVATIONS = 1_000_000

_DEFAULT_TOL = 1e-10


@dataclass(frozen=True)
class InverseResult:
    """Result of :func:`invert_aperture`.

    Attributes
    ----------
    aperture:
        N x N complex aperture; exactly ``0j`` outside the support mask.
    residual:
        True data residual ``F a - b`` as a list of complex values, one per
        observation, in observation order (duplicates included).
    residual_norm:
        Euclidean norm of ``residual``.
    objective:
        ``residual_norm**2 + lam * sum |a|**2`` of the returned aperture,
        recomputed directly from the returned quantities.
    normal_residual:
        Directly recomputed normal-equation residual norm
        ``||F* F a + lam a - F* b||`` (not the CG recursive estimate).
    stop_reason:
        ``"converged"`` when the relative normal-equation residual reached
        ``tol``; ``"max_iterations"`` when the iteration cap was hit first.
        ``"max_iterations"`` is NOT a converged result.
    iterations:
        Number of CG iterations actually performed.
    converged:
        ``stop_reason == "converged"``.
    n_observations:
        Number of observations (duplicates counted individually).
    n_unknowns:
        Number of support pixels (True cells of the mask).
    lam:
        The regularisation parameter actually used.
    """

    aperture: list[list[complex]]
    residual: list[complex]
    residual_norm: float
    objective: float
    normal_residual: float
    stop_reason: str
    iterations: int
    converged: bool
    n_observations: int
    n_unknowns: int
    lam: float


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _parse_mask(mask) -> tuple[int, list[tuple[int, int]]]:
    """Validate the boolean support mask; return (n, support pixels)."""
    if isinstance(mask, (str, bytes)) or not hasattr(mask, "__iter__"):
        raise TypeError("mask must be a 2-D array-like of booleans")
    rows: list[list[bool]] = []
    width: int | None = None
    for i, row in enumerate(mask):
        if isinstance(row, (str, bytes)) or not hasattr(row, "__iter__"):
            raise TypeError(f"mask[{i}] must be an iterable row of booleans")
        parsed: list[bool] = []
        for j, v in enumerate(row):
            if not isinstance(v, bool):
                raise TypeError(
                    f"mask[{i}][{j}] must be a bool, got {type(v).__name__}: {v!r}"
                )
            parsed.append(v)
        if not parsed:
            raise ValueError(f"mask[{i}] is empty")
        if width is None:
            width = len(parsed)
        elif len(parsed) != width:
            raise ValueError(
                f"mask is not rectangular: row 0 has {width} elements, "
                f"row {i} has {len(parsed)}"
            )
        rows.append(parsed)
    if not rows:
        raise ValueError("mask must contain at least one row")
    n = len(rows)
    if n != width:
        raise ValueError(f"mask must be square (N x N), got shape {n} x {width}")
    if n > MAX_APERTURE:
        raise ValueError(f"mask side length must be <= {MAX_APERTURE}, got {n}")
    pixels = [(i, j) for i in range(n) for j in range(n) if rows[i][j]]
    return n, pixels


def _parse_observations(observations) -> list[tuple[float, float, complex]]:
    """Validate observations; duplicates are kept as repeated measurements."""
    if isinstance(observations, (str, bytes)) or not hasattr(observations, "__iter__"):
        raise TypeError("observations must be an iterable of (x, y, value) triples")
    parsed: list[tuple[float, float, complex]] = []
    for idx, item in enumerate(observations):
        where = f"observations[{idx}]"
        if isinstance(item, (str, bytes)) or not hasattr(item, "__iter__"):
            raise TypeError(f"{where} must be an (x, y, value) triple")
        parts = list(item)
        if len(parts) != 3:
            raise ValueError(
                f"{where} must have exactly 3 elements (x, y, value), got {len(parts)}"
            )
        x = _finite_real(parts[0], f"{where} x")
        y = _finite_real(parts[1], f"{where} y")
        v = _as_complex(parts[2], f"{where} value")
        parsed.append((x, y, v))
    if len(parsed) > MAX_OBSERVATIONS:
        raise ValueError(
            f"observations count must be <= {MAX_OBSERVATIONS}, got {len(parsed)}"
        )
    return parsed


def _parse_coefficients(coefficients, expected: int, name: str) -> list[complex]:
    if isinstance(coefficients, (str, bytes)) or not hasattr(coefficients, "__iter__"):
        raise TypeError(f"{name} must be an iterable of complex amplitudes")
    coeffs = [_as_complex(v, f"{name}[{k}]") for k, v in enumerate(coefficients)]
    if len(coeffs) != expected:
        raise ValueError(
            f"{name} must have exactly {expected} elements (one per support "
            f"pixel, row-major), got {len(coeffs)}"
        )
    return coeffs


# ---------------------------------------------------------------------------
# Matrix-free forward / adjoint operators
# ---------------------------------------------------------------------------


class _Setup:
    """Parsed problem geometry plus separable per-axis exponential factors.

    For each observation o, ``row_exp[o][i] = exp(-i 2 pi x_o xi_i / (lam z))``
    and ``col_exp[o][j] = exp(-i 2 pi y_o eta_j / (lam z))``; the kernel
    factorises as ``row_exp[o][i] * col_exp[o][j]``, so applying either
    operator costs O(n_observations * n_unknowns) complex multiplies and no
    dense (n_observations x n_unknowns) Fourier matrix is ever formed.
    """

    __slots__ = ("n", "pixels", "obs", "scale", "row_exp", "col_exp",
                 "row_exp_conj", "col_exp_conj")

    def __init__(self, mask, observations, dx, wavelength, z):
        self.n, self.pixels = _parse_mask(mask)
        self.obs = _parse_observations(observations)
        dx = _finite_real(dx, "dx", positive=True)
        wavelength = _finite_real(wavelength, "wavelength", positive=True)
        z = _finite_real(z, "z", positive=True)
        self.scale = dx * dx / (wavelength * z)
        inv_lz = 1.0 / (wavelength * z)
        c = self.n // 2
        xi = [(i - c) * dx for i in range(self.n)]
        self.row_exp: list[list[complex]] = []
        self.col_exp: list[list[complex]] = []
        self.row_exp_conj: list[list[complex]] = []
        self.col_exp_conj: list[list[complex]] = []
        for x, y, _ in self.obs:
            rx = [cmath.exp(-2j * math.pi * x * v * inv_lz) for v in xi]
            ry = [cmath.exp(-2j * math.pi * y * v * inv_lz) for v in xi]
            self.row_exp.append(rx)
            self.col_exp.append(ry)
            self.row_exp_conj.append([v.conjugate() for v in rx])
            self.col_exp_conj.append([v.conjugate() for v in ry])


def _forward_core(coeffs: list[complex], s: _Setup) -> list[complex]:
    """(F a)(x_o, y_o) for every observation o, in observation order."""
    out: list[complex] = []
    pixels = s.pixels
    for rx, ry in zip(s.row_exp, s.col_exp):
        acc = 0j
        for coef, (i, j) in zip(coeffs, pixels):
            acc += coef * rx[i] * ry[j]
        out.append(s.scale * acc)
    return out


def _adjoint_core(values: list[complex], s: _Setup) -> list[complex]:
    """F* values: conjugate transpose of the forward operator.

    Satisfies the complex inner-product identity
    ``sum_o (F a)_o * conj(b_o) == sum_p a_p * conj((F* b)_p)``.
    """
    g = [0j] * len(s.pixels)
    pixels = s.pixels
    for (crx, cry), v in zip(zip(s.row_exp_conj, s.col_exp_conj), values):
        t = s.scale * v
        for p, (i, j) in enumerate(pixels):
            g[p] += t * crx[i] * cry[j]
    return g


def forward_operator(coefficients, mask, observations, dx, wavelength, z) -> list[complex]:
    """Apply the forward operator: predicted complex amplitudes at the
    observation coordinates.

    ``coefficients`` holds one complex amplitude per True cell of ``mask``
    in row-major order.  ``observations`` uses the same ``(x, y, value)``
    triple contract as :func:`invert_aperture`; only the coordinates are
    used here.  Returns one complex value per observation, in order.
    """
    s = _Setup(mask, observations, dx, wavelength, z)
    coeffs = _parse_coefficients(coefficients, len(s.pixels), "coefficients")
    return _forward_core(coeffs, s)


def adjoint_operator(values, mask, observations, dx, wavelength, z) -> list[complex]:
    """Apply the conjugate transpose of the forward operator.

    ``values`` holds one complex value per observation (in order); returns
    one complex amplitude per support pixel (row-major mask order).
    """
    s = _Setup(mask, observations, dx, wavelength, z)
    vals = _parse_coefficients(values, len(s.obs), "values")
    return _adjoint_core(vals, s)


# ---------------------------------------------------------------------------
# Matrix-free conjugate gradient on the normal equations
# ---------------------------------------------------------------------------


def _norm_sq(v: list[complex]) -> float:
    return sum(x.real * x.real + x.imag * x.imag for x in v)


def _cg_solve(apply_h, g: list[complex], tol: float, max_iterations: int):
    """Solve H a = g for Hermitian positive-definite H (matrix-free).

    Returns (a, iterations, stop_reason) where stop_reason is "converged"
    when ||g - H a|| <= tol * ||g|| and "max_iterations" otherwise; hitting
    the cap is never reported as convergence.
    """
    dim = len(g)
    gnorm = math.sqrt(_norm_sq(g))
    if dim == 0 or gnorm == 0.0:
        # H a = 0 with H positive definite (lam > 0) has the unique solution 0.
        return [0j] * dim, 0, "converged"
    a = [0j] * dim
    r = list(g)  # r = g - H a
    p = list(g)
    rs = gnorm * gnorm
    threshold = tol * gnorm
    iterations = 0
    while iterations < max_iterations:
        hp = apply_h(p)
        php = sum(pc.conjugate() * h for pc, h in zip(p, hp)).real
        alpha = rs / php
        for k in range(dim):
            a[k] += alpha * p[k]
            r[k] -= alpha * hp[k]
        rs_new = _norm_sq(r)
        iterations += 1
        if math.sqrt(rs_new) <= threshold:
            return a, iterations, "converged"
        beta = rs_new / rs
        for k in range(dim):
            p[k] = r[k] + beta * p[k]
        rs = rs_new
    return a, iterations, "max_iterations"


# ---------------------------------------------------------------------------
# Inverse entry point
# ---------------------------------------------------------------------------


def invert_aperture(
    observations,
    mask,
    dx,
    wavelength,
    z,
    lam,
    *,
    tol: float = _DEFAULT_TOL,
    max_iterations: int | None = None,
) -> InverseResult:
    """Reconstruct a complex aperture from complex-amplitude observations.

    Parameters
    ----------
    observations:
        Iterable of ``(x, y, value)`` triples: physical observation-plane
        coordinates in metres and the observed complex amplitude there.
        May be empty.  Duplicates are kept as repeated measurements.
    mask:
        N x N square iterable of strict ``bool`` (1 <= N <= 32); True cells
        form the fixed aperture support.  An all-False mask is allowed and
        yields the zero aperture.
    dx, wavelength, z:
        Same physical parameters (finite, > 0, metres) as
        :func:`fourier_diffraction.fraunhofer`.
    lam:
        Tikhonov weight on the aperture energy, finite and strictly > 0.
    tol:
        Relative normal-equation residual tolerance (finite, > 0).
    max_iterations:
        CG iteration cap, strict int >= 1; ``None`` uses
        ``max(32, min(4096, 2 * n_unknowns))``.  Reaching the cap yields
        ``stop_reason="max_iterations"`` and ``converged=False``.

    Returns
    -------
    InverseResult
    """
    lam = _finite_real(lam, "lam", positive=True)
    tol = _finite_real(tol, "tol", positive=True)
    s = _Setup(mask, observations, dx, wavelength, z)
    n_unknowns = len(s.pixels)
    if max_iterations is None:
        max_iterations = max(32, min(4096, 2 * n_unknowns))
    else:
        max_iterations = _strict_int(max_iterations, "max_iterations", lo=1)

    b = [v for _, _, v in s.obs]

    def apply_h(a: list[complex]) -> list[complex]:
        ha = _adjoint_core(_forward_core(a, s), s)
        return [h + lam * ai for h, ai in zip(ha, a)]

    g = _adjoint_core(b, s)
    coeffs, iterations, stop_reason = _cg_solve(apply_h, g, tol, max_iterations)

    # Honestly recomputed diagnostics from the returned solution.
    residual = [f - bi for f, bi in zip(_forward_core(coeffs, s), b)]
    residual_norm = math.sqrt(_norm_sq(residual))
    energy = _norm_sq(coeffs)
    objective = residual_norm * residual_norm + lam * energy
    normal_residual = math.sqrt(_norm_sq([h - gi for h, gi in zip(apply_h(coeffs), g)]))

    aperture = [[0j] * s.n for _ in range(s.n)]
    for coef, (i, j) in zip(coeffs, s.pixels):
        aperture[i][j] = coef

    return InverseResult(
        aperture=aperture,
        residual=residual,
        residual_norm=residual_norm,
        objective=objective,
        normal_residual=normal_residual,
        stop_reason=stop_reason,
        iterations=iterations,
        converged=stop_reason == "converged",
        n_observations=len(s.obs),
        n_unknowns=n_unknowns,
        lam=lam,
    )
