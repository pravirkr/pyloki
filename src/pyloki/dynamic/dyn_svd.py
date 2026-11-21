"""Empirical SVD + manifold novelty-cloud orbital pruning runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from pyloki.core import common, svd_kepler
from pyloki.detection import scoring
from pyloki.utils.snail import MiddleOutScheme

if TYPE_CHECKING:
    from pyloki.config import PulsarSearchConfig


@dataclass(frozen=True, kw_only=True)
class SVDPruningResult:
    """Surviving empirical-SVD orbital leaves, scores, and integrated folds."""

    coords: np.ndarray
    f0: np.ndarray
    scores: np.ndarray
    folds: np.ndarray
    n_levels: int
    ref_seg: int


class SVDPruningRunner:
    """Pruning runner operating over a precomputed ``SVDOrbitalBasis`` chain.

    Follows the same ``seed -> branch -> resolve -> load -> shift_add -> score``
    lifecycle as polynomial EP pruning, expanding leaves along the empirical
    Keplerian novelty cloud rather than an axis-aligned polynomial hypercube.
    """

    def __init__(self, basis: svd_kepler.SVDOrbitalBasis) -> None:
        self.basis = basis

    def execute(
        self,
        dyp_fold: np.ndarray,
        param_arr: list[np.ndarray],
        param_grid_count_init: np.ndarray,
        param_limits: np.ndarray,
        cfg: PulsarSearchConfig,
        *,
        t_seg: float,
        threshold_scheme: np.ndarray | None = None,
        ref_seg: int = 0,
        max_sugg: int = 2**14,
    ) -> SVDPruningResult:
        """Run empirical-SVD pruning across all levels in ``self.basis``."""
        n_levels = len(self.basis.levels)
        nsegments = int(dyp_fold.shape[0])
        if n_levels == 0 or nsegments == 0:
            return SVDPruningResult(
                coords=np.empty((0, 0), dtype=np.float64),
                f0=np.empty(0, dtype=np.float64),
                scores=np.empty(0, dtype=np.float32),
                folds=np.empty((0, 2, cfg.nbins), dtype=np.float32),
                n_levels=0,
                ref_seg=int(ref_seg),
            )

        eff_ref = int(ref_seg)
        scheme = MiddleOutScheme(
            nsegments=nsegments,
            ref_idx=eff_ref,
            tsegment=float(t_seg),
            stride=1,
        )
        t_ref = float(scheme.get_coord(0)[0])

        # Seed from P-FFA grid at ref_seg (all (a, f) seed points)
        lvl0 = self.basis.levels[0]
        rank0 = int(lvl0.v_phase.shape[0])
        f_grid = np.asarray(param_arr[-1], dtype=np.float64)
        n_acc = int(dyp_fold.shape[1])
        n_freq = int(dyp_fold.shape[2])
        n_seed = n_acc * n_freq

        cur_folds = np.ascontiguousarray(
            dyp_fold[eff_ref].reshape(n_seed, 2, cfg.nbins),
            dtype=np.float32,
        )
        cur_f0 = np.ascontiguousarray(
            np.tile(f_grid[:n_freq], n_acc),
            dtype=np.float64,
        )
        cur_coords = np.zeros((n_seed, rank0), dtype=np.float64)

        score_mode_id = getattr(cfg, "score_mode_id", 0)
        score_n_harmonics = getattr(cfg, "score_n_harmonics", 16)
        kadane_biases = getattr(
            cfg,
            "kadane_biases",
            scoring.generate_kadane_biases(cfg.nbins),
        )
        cur_scores = scoring.score_batch_ev_njit(
            cur_folds,
            cfg.score_widths,
            score_mode_id,
            score_n_harmonics,
            kadane_biases,
        )
        p_limits_2d = np.ascontiguousarray(param_limits[-2:], dtype=np.float64)
        p_counts_2d = np.ascontiguousarray(param_grid_count_init, dtype=np.int64)

        max_lvl = min(n_levels, nsegments)
        levels_done = 1
        for lvl in range(1, max_lvl):
            seg_idx = scheme.get_segment_idx(lvl)
            t_eval = float(scheme.get_segment_coord(lvl)[0])
            lvl_cur = self.basis.levels[lvl]

            if lvl_cur.trans_matrix is None or lvl_cur.novelties is None:
                msg = f"SVDBasisLevel {lvl} is missing trans_matrix or novelties"
                raise ValueError(msg)
            # Preserve distinct seed f0 tracks during lattice deduplication
            f0_groups = np.rint(cur_f0 * 1e6).astype(np.int64)
            child_coords, child_origins = svd_kepler.svd_branch_batch(
                cur_coords,
                lvl_cur.trans_matrix,
                lvl_cur.novelties,
                lvl_cur.delta,
                f0_groups,
            )
            child_f0 = np.ascontiguousarray(cur_f0[child_origins], dtype=np.float64)
            if child_coords.shape[0] == 0:
                cur_coords = child_coords
                cur_f0 = child_f0
                cur_folds = np.empty((0, 2, cfg.nbins), dtype=np.float32)
                cur_scores = np.empty(0, dtype=np.float32)
                break

            pidx, pshift = svd_kepler.svd_resolve_to_pffa_batch(
                child_coords,
                child_f0,
                t_ref,
                t_eval,
                lvl_cur.t_axis,
                lvl_cur.v_phase,
                lvl_cur.v_vel,
                lvl_cur.v_acc,
                lvl_cur.tau_tol,
                p_counts_2d,
                p_limits_2d,
                cfg.nbins,
            )
            fold_seg = dyp_fold[seg_idx]
            loaded = np.ascontiguousarray(
                fold_seg[pidx[:, 0], pidx[:, 1]],
                dtype=np.float32,
            )
            combined = common.shift_add_batch(loaded, pshift, cur_folds, child_origins)
            scores = scoring.score_batch_ev_njit(
                combined,
                cfg.score_widths,
                score_mode_id,
                score_n_harmonics,
                kadane_biases,
            )

            thr = (
                float(threshold_scheme[lvl - 1])
                if threshold_scheme is not None and lvl - 1 < len(threshold_scheme)
                else -np.inf
            )
            keep = np.flatnonzero(scores >= thr)
            if keep.size == 0:
                cur_coords = np.empty((0, child_coords.shape[1]), dtype=np.float64)
                cur_f0 = np.empty(0, dtype=np.float64)
                cur_folds = np.empty((0, 2, cfg.nbins), dtype=np.float32)
                cur_scores = np.empty(0, dtype=np.float32)
                break
            if keep.size > max_sugg:
                top = np.argsort(scores[keep])[::-1][:max_sugg]
                keep = keep[top]

            cur_coords = np.ascontiguousarray(child_coords[keep], dtype=np.float64)
            cur_f0 = np.ascontiguousarray(child_f0[keep], dtype=np.float64)
            cur_folds = np.ascontiguousarray(combined[keep], dtype=np.float32)
            cur_scores = np.ascontiguousarray(scores[keep], dtype=np.float32)
            levels_done = lvl + 1

        if cur_scores.size > 0:
            order = np.argsort(cur_scores)[::-1]
            cur_coords = cur_coords[order]
            cur_f0 = cur_f0[order]
            cur_folds = cur_folds[order]
            cur_scores = cur_scores[order]

        return SVDPruningResult(
            coords=cur_coords,
            f0=cur_f0,
            scores=cur_scores,
            folds=cur_folds,
            n_levels=levels_done,
            ref_seg=eff_ref,
        )
