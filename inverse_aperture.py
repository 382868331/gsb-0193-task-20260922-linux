"""Inverse problem on top of the Fraunhofer forward model.

Given *complex-amplitude* observations ``u_i`` at specified physical
observation-plane coordinates ``(x_i, y_i)`` (not just intensities) and a
fixed boolean support mask, find the complex aperture ``a`` supported on the
mask that minimises

    || F a - b ||**2 + lam * ||a||**2,        lam > 0 strictly,

where ``F`` is the Fraunhofer forward operator of ``fourier_diffraction``
evaluated at the observation coordinates:

    (F a)(x, y) = dx**2 / (wavelength * z)
                  * sum_p a_p exp(-i 2 pi (x * xi_p + y * eta_p) / (wavelength * z))

with pixel centres ``xi_p = (m - n//2) * dx``, ``eta_p = (j - n//2) * dx`` —
exactly the centred-layout pixel centres, phase convention and physical scale
of the forward model, so on the DFT grid ``x = k * du`` the operator
reproduces ``fraunhofer(...)`` field samples bin for bin (this is tested).

The normal equations

    (F* F + lam I) a = F* b

are solved by a matrix-free conjugate-gradient iteration: the forward
operator and its conjugate transpose are applied as functions (separable
per-axis phasor tables of size n_obs x N are cached; the dense
n_obs x n_unknown Fourier matrix is never formed).  Reaching the iteration
cap is reported as ``converged=False`` — it is never treated as convergence.

Duplicate observations are kept as repeated measurements (they contribute
multiple times to the residual and the objective); nothing is deduplicated.

Standard library only.
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

__all__ = ["ApertureOperator", "InverseResult", "invert_aperture"]

DEFAULT_MAX_ITER = 1000
DEFAULT_TOL = 1e-10


# ---------------------------------------------------------------------------
# Validation helpers (same error semantics as the forward module)
# ---------------------------------------------------------------------------


def _parse_mask(mask) -> list[list[bool]]:
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
    return rows


def _parse_observations(observations) -> list[tuple[float, float, complex]]:
    if isinstance(observations, (str, bytes)) or not hasattr(observations, "__iter__"):
        raise TypeError("observations must be an iterable of (x, y, amplitude) triples")
    parsed: list[tuple[float, float, complex]] = []
    for i, entry in enumerate(observations):
        if isinstance(entry, (str, bytes)) or not hasattr(entry, "__iter__"):
            raise TypeError(f"observations[{i}] must be an (x, y, amplitude) triple")
        items = list(entry)
        if len(items) != 3:
            raise ValueError(
                f"observations[{i}] must have exactly 3 elements (x, y, amplitude), "
                f"got {len(items)}"
            )
        x = _finite_real(items[0], f"observations[{i}] x")
        y = _finite_real(items[1], f"observations[{i}] y")
        u = _as_complex(items[2], f"observations[{i}] amplitude")
        # Duplicates are meaningful repeated measurements: kept as-is.
        parsed.append((x, y, u))
    return parsed


# ---------------------------------------------------------------------------
# Matrix-free forward / adjoint operator
# ---------------------------------------------------------------------------


class ApertureOperator:
    """Fraunhofer forward operator restricted to a boolean support mask.

    The unknown vector has one complex entry per ``True`` mask pixel, in
    row-major order.  ``forward`` maps it to complex field samples at the
    observation coordinates; ``adjoint`` is its conjugate transpose with
    respect to the standard complex inner products
    ``<u, v> = sum conj(u_k) v_k`` on both spaces.

    Separable per-axis phasor tables (``n_obs x N`` each) are precomputed;
    the dense ``n_obs x n_unknown`` matrix is never formed.
    """

    def __init__(self, mask, observations, dx, wavelength, z):
        self.mask = _parse_mask(mask)
        self.n = len(self.mask)
        self.dx = _finite_real(dx, "dx", positive=True)
        self.wavelength = _finite_real(wavelength, "wavelength", positive=True)
        self.z = _finite_real(z, "z", positive=True)
        self.observations = _parse_observations(observations)

        self.pixels: list[tuple[int, int]] = [
            (i, j)
            for i in range(self.n)
            for j in range(self.n)
            if self.mask[i][j]
        ]
        self.scale = self.dx * self.dx / (self.wavelength * self.z)

        # Pixel-centre coordinates, identical to the forward model's centred
        # layout: centre index n//2 sits at coordinate 0.
        c = self.n // 2
        centres = [(p - c) * self.dx for p in range(self.n)]

        # Separable phasors: phase(x, y; m, n) = px[i][m] * py[i][n].
        lz = self.wavelength * self.z
        self._px: list[list[complex]] = [
            [cmath.exp(-2j * math.pi * x * xm / lz) for xm in centres]
            for x, _, _ in self.observations
        ]
        self._py: list[list[complex]] = [
            [cmath.exp(-2j * math.pi * y * yn / lz) for yn in centres]
            for _, y, _ in self.observations
        ]

    @property
    def n_unknowns(self) -> int:
        return len(self.pixels)

    @property
    def n_observations(self) -> int:
        return len(self.observations)

    def forward(self, x: list[complex]) -> list[complex]:
        """Apply F: mask-supported aperture vector -> observation samples."""
        if len(x) != self.n_unknowns:
            raise ValueError(
                f"x must have {self.n_unknowns} entries, got {len(x)}"
            )
        out: list[complex] = []
        for i in range(self.n_observations):
            px_i = self._px[i]
            py_i = self._py[i]
            s = 0j
            for t, (m, n) in enumerate(self.pixels):
                s += x[t] * px_i[m] * py_i[n]
            out.append(self.scale * s)
        return out

    def adjoint(self, r: list[complex]) -> list[complex]:
        """Apply F* (conjugate transpose): observation vector -> aperture."""
        if len(r) != self.n_observations:
            raise ValueError(
                f"r must have {self.n_observations} entries, got {len(r)}"
            )
        out = [0j] * self.n_unknowns
        for i in range(self.n_observations):
            ri = r[i]
            if ri == 0j:
                continue
            px_i = self._px[i]
            py_i = self._py[i]
            for t, (m, n) in enumerate(self.pixels):
                out[t] += ri * (px_i[m] * py_i[n]).conjugate()
        return [self.scale * v for v in out]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InverseResult:
    """Result of the regularised aperture inversion.

    Attributes
    ----------
    aperture:
        N x N complex aperture; pixels outside the mask are exactly ``0j``.
    residual:
        True residual ``F a - b`` at every observation, in input order
        (duplicated observations appear duplicated here).
    residual_norm:
        ``||F a - b||``.
    objective:
        ``||F a - b||**2 + lam * ||a||**2`` evaluated at the returned
        aperture (recomputed explicitly, not from CG recurrences).
    normal_residual_norm:
        ``||(F* F + lam I) a - F* b||`` recomputed explicitly at the final
        iterate.
    iterations:
        Number of CG iterations actually performed.
    converged:
        True only when a stopping criterion was met.  Reaching the iteration
        cap gives ``converged=False`` with ``stop_reason="max_iter_reached"``.
    stop_reason:
        One of ``"converged"``, ``"max_iter_reached"``, ``"zero_rhs"``,
        ``"no_observations"``, ``"empty_support"``.
    lam, n, n_observations, n_unknowns:
        Echoes of the problem dimensions and regularisation weight.
    """

    aperture: list[list[complex]]
    residual: list[complex]
    residual_norm: float
    objective: float
    normal_residual_norm: float
    iterations: int
    converged: bool
    stop_reason: str
    lam: float
    n: int
    n_observations: int
    n_unknowns: int


# ---------------------------------------------------------------------------
# Matrix-free CG on the normal equations
# ---------------------------------------------------------------------------


def _dot(u: list[complex], v: list[complex]) -> complex:
    """Standard complex inner product ``sum conj(u_k) v_k``."""
    return sum((a.conjugate() * b for a, b in zip(u, v)), 0j)


def invert_aperture(
    mask,
    observations,
    dx,
    wavelength,
    z,
    lam,
    *,
    tol: float = DEFAULT_TOL,
    max_iter: int | None = None,
) -> InverseResult:
    """Solve ``min_a ||F a - b||^2 + lam ||a||^2`` over the mask support.

    Parameters
    ----------
    mask:
        N x N booleans (1 <= N <= 32); ``True`` pixels are the unknowns.
        An all-``False`` mask is legal and yields the zero aperture.
    observations:
        Iterable of ``(x, y, amplitude)`` triples: physical coordinates in
        metres and the observed *complex* amplitude there.  May be empty;
        duplicates are kept as repeated measurements.
    dx, wavelength, z:
        Same physical parameters (metres, finite, > 0) as ``fraunhofer``.
    lam:
        Tikhonov weight on the aperture energy; must be finite and
        **strictly** positive.
    tol:
        Relative stopping tolerance on the normal-equation residual:
        stop when ``||r|| <= tol * ||F* b||``.
    max_iter:
        CG iteration cap (strict int >= 1); ``None`` uses 1000.  Hitting the
        cap is *not* convergence.

    Returns
    -------
    InverseResult
    """
    op = ApertureOperator(mask, observations, dx, wavelength, z)
    lam = _finite_real(lam, "lam", positive=True)
    tol = _finite_real(tol, "tol", positive=True)
    if max_iter is None:
        max_iter = DEFAULT_MAX_ITER
    max_iter = _strict_int(max_iter, "max_iter", lo=1)

    k = op.n_unknowns
    p = op.n_observations
    b = [u for _, _, u in op.observations]

    def finish(x: list[complex], iterations: int, converged: bool, reason: str):
        # Everything below is recomputed explicitly from the final iterate.
        residual = [fi - bi for fi, bi in zip(op.forward(x), b)]
        residual_norm = math.sqrt(sum(v.real * v.real + v.imag * v.imag for v in residual))
        energy = sum(v.real * v.real + v.imag * v.imag for v in x)
        objective = residual_norm * residual_norm + lam * energy
        # Normal-equation residual (F*F + lam I) x - F* b, explicit.
        nr = _normal_residual(op, lam, x, b)
        normal_residual_norm = math.sqrt(_dot(nr, nr).real)
        aperture = [[0j] * op.n for _ in range(op.n)]
        for t, (i, j) in enumerate(op.pixels):
            aperture[i][j] = x[t]
        return InverseResult(
            aperture=aperture,
            residual=residual,
            residual_norm=residual_norm,
            objective=objective,
            normal_residual_norm=normal_residual_norm,
            iterations=iterations,
            converged=converged,
            stop_reason=reason,
            lam=lam,
            n=op.n,
            n_observations=p,
            n_unknowns=k,
        )

    if k == 0:
        return finish([], 0, True, "empty_support")
    if p == 0:
        return finish([0j] * k, 0, True, "no_observations")

    c = op.adjoint(b)
    norm_c = math.sqrt(_dot(c, c).real)
    if norm_c == 0.0:
        # b is orthogonal to the range of F: x = 0 solves the normal
        # equations exactly.
        return finish([0j] * k, 0, True, "zero_rhs")

    def apply_h(v: list[complex]) -> list[complex]:
        Av = op.forward(v)
        AtAv = op.adjoint(Av)
        return [a + lam * vi for a, vi in zip(AtAv, v)]

    x = [0j] * k
    r = list(c)  # normal-equation residual c - H x, with x = 0
    d = list(c)
    rr = _dot(r, r).real
    threshold = tol * norm_c

    iterations = 0
    converged = False
    for it in range(1, max_iter + 1):
        iterations = it
        hd = apply_h(d)
        denom = _dot(d, hd).real
        if denom == 0.0:
            # H is positive definite for lam > 0, so this can only be
            # round-off on an already-converged direction.
            break
        alpha = rr / denom
        x = [xi + alpha * di for xi, di in zip(x, d)]
        r = [ri - alpha * hdi for ri, hdi in zip(r, hd)]
        rr_new = _dot(r, r).real
        if math.sqrt(rr_new) <= threshold:
            rr = rr_new
            converged = True
            break
        beta = rr_new / rr
        d = [ri + beta * di for ri, di in zip(r, d)]
        rr = rr_new

    reason = "converged" if converged else "max_iter_reached"
    return finish(x, iterations, converged, reason)


def _normal_residual(
    op: ApertureOperator, lam: float, x: list[complex], b: list[complex]
) -> list[complex]:
    """Explicit (F*F + lam I) x - F* b, used only for honest reporting."""
    ax = op.forward(x)
    atax = op.adjoint(ax)
    atb = op.adjoint(b)
    return [h + lam * xi - ci for h, xi, ci in zip(atax, x, atb)]
