r"""6-basis :math:`O(e)` epicyclic SVD residual validation filter.

Checks whether candidate high-order polynomial trajectories stay within
:math:`2 x_{\max} e_{\max}^2 c` of the linearized eccentric epicyclic family
:math:`\{1, t, \sin\omega t, \cos\omega t, \sin 2\omega t, \cos 2\omega t\}`
over some physical orbital frequency :math:`\omega \in [\omega_{\min}, \omega_{\max}]`,
using rank-revealing thin SVD orthonormalization (avoiding normal-equation
ill-conditioning when :math:`\omega \Delta T \ll 1`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import attrs
import numpy as np
from numba import njit

from pyloki.utils.misc import C_VAL

if TYPE_CHECKING:
    from numpy import typing as npt


@attrs.frozen(auto_attribs=True, kw_only=True)
class ValidatorConfig:
    """Configuration for the :math:`O(e)` epicyclic trajectory validator."""

    enabled: bool = False
    omega_min: float = 1e-4
    omega_max: float = 1e-2
    x_max: float = 5.0
    ecc_max: float = 0.25
    v_max: float | None = None
    excursion_sigma: float = 3.2
    f_search_bounds: tuple[float, float] | None = None
    deriv_bounds: tuple[float, ...] | None = None
    n_colloc: int = 20
    svd_rcond: float = 1e-10
    min_level: int = 2

    @property
    def max_residual_m(self) -> float:
        """Allowed epicyclic residual bound ``2 * x_max * ecc_max^2 * c`` (meters)."""
        return 2.0 * self.x_max * (self.ecc_max**2) * C_VAL


@njit(cache=True)
def epicyclic_min_residual_batch(
    z_batch: np.ndarray,
    projectors: np.ndarray,
) -> np.ndarray:
    """Compute ``min_omega ||P_perp(omega) z||_inf`` for a batch of trajectories.

    Parameters
    ----------
    z_batch : np.ndarray
        Shape ``(n_batch, M)`` trajectory values at ``M`` collocation times.
    projectors : np.ndarray
        Shape ``(n_omega, M, M)`` orthogonal residual projectors ``I - Q Q^T``.
    """
    n_batch, m_pts = z_batch.shape
    n_om = projectors.shape[0]
    min_res = np.empty(n_batch, dtype=np.float64)

    for b in range(n_batch):
        best = np.inf
        for w in range(n_om):
            max_abs = 0.0
            for i in range(m_pts):
                acc = 0.0
                for j in range(m_pts):
                    acc += projectors[w, i, j] * z_batch[b, j]
                val = abs(acc)
                if val > max_abs:
                    max_abs = val
                    if max_abs >= best:
                        break
            best = min(best, max_abs)
        min_res[b] = best
    return min_res


@attrs.frozen(auto_attribs=True, kw_only=True)
class EpicyclicProjector:
    """Precomputed rank-revealing SVD residual projectors for a tree span."""

    t_colloc: npt.NDArray[np.float64]
    omega_grid: npt.NDArray[np.float64]
    projectors: npt.NDArray[np.float64]
    config: ValidatorConfig

    @classmethod
    def build(
        cls,
        t_start: float,
        t_end: float,
        t_seg: float,
        config: ValidatorConfig,
        *,
        max_omega_points: int | None = None,
    ) -> EpicyclicProjector:
        """Build interior collocation grid and thin-SVD residual projectors."""
        lo = t_start + 1.5 * t_seg
        hi = t_end - 0.5 * t_seg
        if hi <= lo:
            lo, hi = t_start, t_end
        t_colloc = np.linspace(lo, hi, config.n_colloc, dtype=np.float64)
        t_centered = t_colloc - float(np.mean(t_colloc))
        span = max(float(t_colloc[-1] - t_colloc[0]), 1e-6)
        t_scaled = t_centered / (0.5 * span)

        d_om = max((config.ecc_max**2) / span, 1e-12)
        if config.omega_max <= config.omega_min:
            omega_grid = np.array([config.omega_min], dtype=np.float64)
        else:
            n_om = max(2, math.ceil((config.omega_max - config.omega_min) / d_om) + 1)
            if max_omega_points is not None:
                n_om = min(max_omega_points, n_om)
            omega_grid = np.linspace(
                config.omega_min,
                config.omega_max,
                n_om,
                dtype=np.float64,
            )

        m_pts = t_colloc.size
        eye_m = np.eye(m_pts, dtype=np.float64)
        projectors = np.empty((omega_grid.size, m_pts, m_pts), dtype=np.float64)
        for idx, om in enumerate(omega_grid):
            wt = float(om) * t_centered
            design = np.column_stack(
                [
                    np.ones(m_pts, dtype=np.float64),
                    t_scaled,
                    np.sin(wt),
                    np.cos(wt),
                    np.sin(2.0 * wt),
                    np.cos(2.0 * wt),
                ],
            )
            u, s, _ = np.linalg.svd(design, full_matrices=False)
            cutoff = config.svd_rcond * float(s[0]) if s.size > 0 else 0.0
            rank = int(np.sum(s > cutoff))
            q = u[:, :rank]
            projectors[idx] = eye_m - (q @ q.T)

        return cls(
            t_colloc=t_colloc,
            omega_grid=omega_grid,
            projectors=np.ascontiguousarray(projectors, dtype=np.float64),
            config=config,
        )

    def filter_trajectories(
        self,
        z_batch: npt.ArrayLike,
        *,
        v_batch: npt.ArrayLike | None = None,
        f0_batch: npt.ArrayLike | None = None,
        deriv_batches: tuple[npt.ArrayLike, ...] | None = None,
    ) -> npt.NDArray[np.bool_]:
        """Return boolean mask of trajectories passing epicyclic & physical bounds."""
        z = np.ascontiguousarray(z_batch, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.t_colloc.size:
            msg = f"z_batch must have shape (N, {self.t_colloc.size}), got {z.shape}"
            raise ValueError(msg)
        n_batch = z.shape[0]
        valid = np.ones(n_batch, dtype=np.bool_)
        if n_batch == 0:
            return valid

        # 1. Cheap O(M) excursion pre-filter before projector matrix multiplies
        if self.config.excursion_sigma > 0.0:
            med = np.median(z, axis=1)
            std = np.std(z, axis=1)
            max_exc = np.max(z, axis=1) - med
            min_exc = med - np.min(z, axis=1)
            thr_exc = self.config.excursion_sigma * std
            # Constant / near-zero trajectories (std == 0) always pass
            non_flat = std > 1e-15
            valid &= (~non_flat) | ((max_exc <= thr_exc) & (min_exc <= thr_exc))

        # 2. Velocity excursion & search-frequency band overlap checks
        if v_batch is not None:
            v = np.asarray(v_batch, dtype=np.float64)
            v_min = np.min(v, axis=1)
            v_max = np.max(v, axis=1)
            if self.config.v_max is not None:
                valid &= (v_max - v_min) <= (2.0 * self.config.v_max)
            if self.config.f_search_bounds is not None and f0_batch is not None:
                f0 = np.asarray(f0_batch, dtype=np.float64)
                f_lo, f_hi = self.config.f_search_bounds
                f_traj_lo = f0 * (1.0 - v_max / C_VAL)
                f_traj_hi = f0 * (1.0 - v_min / C_VAL)
                valid &= (f_traj_hi >= f_lo) & (f_traj_lo <= f_hi)

        # 3. Higher-derivative (r >= 2) infinity-norm bounds
        if deriv_batches is not None and self.config.deriv_bounds is not None:
            for d_arr, bound in zip(
                deriv_batches,
                self.config.deriv_bounds,
                strict=False,
            ):
                if bound > 0.0:
                    d_vals = np.asarray(d_arr, dtype=np.float64)
                    valid &= np.max(np.abs(d_vals), axis=1) <= bound

        # 4. Run O(n_omega * M^2) SVD projector only on surviving candidates
        surv_idx = np.flatnonzero(valid)
        if surv_idx.size > 0:
            residuals = epicyclic_min_residual_batch(
                np.ascontiguousarray(z[surv_idx], dtype=np.float64),
                self.projectors,
            )
            valid[surv_idx] = residuals <= self.config.max_residual_m
        return valid


def evaluate_taylor_leaves_on_grid(
    leaves_batch: np.ndarray,
    t_ref: float,
    t_colloc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, ...]]:
    """Evaluate Taylor leaves ``(N, n_order+2, 2)`` on ``t_colloc``.

    Returns ``(z_batch, v_batch, f0_batch, deriv_batches)`` where
    ``deriv_batches[r - 2]`` is the ``r``-th time derivative of ``z(t)`` for
    ``r >= 2``.
    """
    n_batch, n_slots, _ = leaves_batch.shape
    n_order = n_slots - 2
    dt = np.asarray(t_colloc, dtype=np.float64) - float(t_ref)
    m_pts = dt.size
    z = np.zeros((n_batch, m_pts), dtype=np.float64)
    v = np.zeros((n_batch, m_pts), dtype=np.float64)
    f0 = np.asarray(leaves_batch[:, n_order - 1, 0], dtype=np.float64)

    # Descending Taylor derivatives: slot (n_order - k) holds d_k for k = 2..n_order
    for k in range(2, n_order + 1):
        dk = leaves_batch[:, n_order - k, 0][:, None]
        z += dk * (dt[None, :] ** k) / math.factorial(k)
        v += dk * (dt[None, :] ** (k - 1)) / math.factorial(k - 1)

    derivs: list[np.ndarray] = []
    for r in range(2, n_order + 1):
        dr = np.zeros((n_batch, m_pts), dtype=np.float64)
        for k in range(r, n_order + 1):
            dk = leaves_batch[:, n_order - k, 0][:, None]
            dr += dk * (dt[None, :] ** (k - r)) / math.factorial(k - r)
        derivs.append(dr)
    return z, v, f0, tuple(derivs)
