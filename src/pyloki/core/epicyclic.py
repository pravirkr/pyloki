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
    n_colloc: int = 20
    svd_rcond: float = 1e-10

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
        max_omega_points: int = 256,
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
            n_om = min(
                max_omega_points,
                max(2, math.ceil((config.omega_max - config.omega_min) / d_om) + 1),
            )
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
    ) -> npt.NDArray[np.bool_]:
        """Return boolean mask of trajectories passing the epicyclic residual bound."""
        z = np.ascontiguousarray(z_batch, dtype=np.float64)
        if z.ndim != 2 or z.shape[1] != self.t_colloc.size:
            msg = f"z_batch must have shape (N, {self.t_colloc.size}), got {z.shape}"
            raise ValueError(msg)
        residuals = epicyclic_min_residual_batch(z, self.projectors)
        valid = residuals <= self.config.max_residual_m
        if v_batch is not None and self.config.v_max is not None:
            v = np.asarray(v_batch, dtype=np.float64)
            v_span = np.max(v, axis=1) - np.min(v, axis=1)
            valid &= v_span <= (2.0 * self.config.v_max)
        return valid
