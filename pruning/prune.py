from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange
from rich.progress import track

from pruning import kernels, utils
from pruning.base import PruningDPFunctions

if TYPE_CHECKING:
    from pruning.ffa import DynamicProgramming


@njit(cache=True, fastmath=True)
def pruning_iteration(
    suggestion: kernels.SuggestionStruct,
    fold_segment: np.ndarray,
    prune_funcs: PruningDPFunctions,
    threshold: float,
    prune_level: int,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
    seg_cur: int,
    seg_ref: int,
    max_sugg: int = 2**17,
) -> tuple[kernels.SuggestionStruct, kernels.PruneStats]:
    suggestion_new = suggestion.get_new(max_sugg)
    n_leaves_total = 0
    n_leaves_physical = 0
    n_branches = suggestion.size
    iparam = 0
    tcheby = coord_prev[1] * 2

    trans_matrix = prune_funcs.get_trans_matrix(coord_cur, coord_prev)
    physical_validation_iter = (prune_level % 4 == 1) and (prune_level > 16)
    physical_validation_iter = False
    if physical_validation_iter:
        validation_params = prune_funcs.get_validation_params(tcheby)

    for isuggest in range(n_branches):
        leaves_arr = prune_funcs.branch(suggestion.param_sets[isuggest], prune_level)
        n_leaves_total += len(leaves_arr)
        if physical_validation_iter:
            leaves_arr = prune_funcs.validate_physical(
                leaves_arr,
                tcheby,
                tcheby,
                validation_params,
            )
        n_leaves_physical += len(leaves_arr)

        for ileaf in range(len(leaves_arr)):
            leaf = leaves_arr[ileaf]
            param_idx, phase_shift = prune_funcs.resolve(leaf, seg_cur, seg_ref)
            partial_res = prune_funcs.shift(fold_segment[param_idx], phase_shift)
            combined_res = prune_funcs.add(suggestion.folds[isuggest], partial_res)
            score = prune_funcs.score(combined_res)

            if score >= threshold:
                suggestion_new.param_sets[iparam] = prune_funcs.transform_coords(
                    leaf,
                    coord_cur,
                    trans_matrix,
                )
                suggestion_new.folds[iparam] = combined_res
                suggestion_new.scores[iparam] = score
                suggestion_new.backtracks[iparam] = np.array(
                    [isuggest, *list(param_idx), phase_shift],
                )

                iparam += 1
                if iparam == max_sugg:
                    threshold = np.median(suggestion_new.scores)
                    suggestion_new = suggestion_new.apply_threshold(threshold)
                    iparam = suggestion_new.actual_size

    suggestion_new = suggestion_new.trim_empty(iparam)
    stats = kernels.PruneStats(
        n_branches,
        n_leaves_total,
        n_leaves_physical,
        suggestion_new.size,
    )
    return suggestion_new, stats


class Pruning:
    """Pruning algorithm for the dynamic programming algorithm.

    Time is linearly advancing. The algorithm starts with a reference segment
    and iteratively prunes the parameter space based on the scores of the
    suggestions.

    Parameters
    ----------
    dyp : DynamicProgramming
        _description_
    threshold_scheme : np.ndarray
        _description_
    max_sugg : int, optional
        _description_, by default 2**17
    """

    def __init__(
        self,
        dyp: DynamicProgramming,
        threshold_scheme: np.ndarray,
        max_sugg: int = 2**17,
    ) -> None:
        self._dyp = dyp
        if dyp.fold.ndim > 5:
            msg = "Pruning only supports initial data with up to 2D parameter."
            raise ValueError(msg)
        self._prune_funcs = PruningDPFunctions(
            dyp.cfg,
            dyp.param_arr,
            dyp.dparams,
            dyp.chunk_duration,
        )
        self._threshold_scheme = threshold_scheme
        self._max_sugg = max_sugg

        self.logger = utils.get_logger("Pruning")

    @property
    def max_sugg(self) -> int:
        return self._max_sugg

    @property
    def dyp(self) -> DynamicProgramming:
        return self._dyp

    @property
    def prune_level(self) -> int:
        return self._prune_level

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def prune_funcs(self) -> PruningDPFunctions:
        return self._prune_funcs

    @property
    def segment_scheme(self) -> np.ndarray:
        return self._segment_scheme

    @property
    def seg_ref(self) -> int:
        return self.segment_scheme[0]

    @property
    def threshold_scheme(self) -> np.ndarray:
        return self._threshold_scheme

    @property
    def suggestion(self) -> kernels.SuggestionStruct:
        return self._suggestion

    @property
    def t_ref(self) -> float:
        return self._t_ref

    @property
    def backtrack_arr(self) -> np.ndarray:
        return self._backtrack_arr

    @property
    def best_intermediate_arr(self) -> np.ndarray:
        return self._best_intermediate_arr

    def initialize(self, seg_ref: int = 12) -> None:
        """Initialize the pruning algorithm.

        Parameters
        ----------
        seg_ref : int, optional
            The reference segment to start the pruning algorithm, by default 12

        Notes
        -----
        Reference time for the parameters will be the middle of the reference segment.
        """
        tstart = time.time()
        self._segment_scheme = utils.snail_access_scheme(
            self.dyp.nchunks,
            ref_idx=seg_ref,
        )
        self._complete = False
        self._prune_level = 0
        self.logger.info(f"Initializing pruning with ref segment: {self.seg_ref}")
        # Initialize the suggestions with the first segment
        fold_segment = self.prune_funcs.load(self.dyp.fold, self.seg_ref)
        self._suggestion = self.prune_funcs.suggest(fold_segment)
        self._t_ref = (self.seg_ref + 0.5) * self.prune_funcs.tchunk_ffa
        self._backtrack_arr = np.zeros(
            (self.dyp.fold.shape[0], self.max_sugg, len(self.dyp.fold.shape[1:])),
        )
        self._best_intermediate_arr = np.empty(
            (self.dyp.fold.shape[0], 3),
            dtype=object,
        )
        self.logger.info(f"Initialization time: {time.time() - tstart}")

    def prune_enumeration(
        self,
        snr_lim: float,
        seg_ref_list: np.ndarray | None = None,
        *,
        lazy: bool = True,
    ) -> list[tuple[int, kernels.SuggestionStruct]]:
        res = []
        if seg_ref_list is None:
            seg_ref_list = np.arange(0, self.dyp.nchunks, self.dyp.nchunks // 16)
        for seg_ref in seg_ref_list:
            self.initialize(seg_ref=seg_ref)
            for _ in track(range(self.dyp.nchunks - 1), description="Pruning"):
                self.prune_iter()
            if self.suggestion.size > 0 and np.max(self.suggestion.scores) > snr_lim:
                res.append((seg_ref, self.suggestion))
                if lazy:
                    return res
        return res

    def prune_iter(self) -> None:
        if self.is_complete:
            return
        self._prune_level += 1
        scheme_till_now = self.segment_scheme[: self.prune_level + 1]
        idx_cur = self.segment_scheme[self.prune_level]
        ref_cur = (np.min(scheme_till_now) + np.max(scheme_till_now) + 1) / 2
        ref_prev = (np.min(scheme_till_now[:-1]) + np.max(scheme_till_now[:-1]) + 1) / 2
        scale_cur = ref_cur - np.min(scheme_till_now)
        scale_prev = ref_prev - np.min(scheme_till_now[:-1])
        fold_segment = self.prune_funcs.load(self.dyp.fold, seg_cur)
        threshold = self.threshold_scheme[self.prune_level]
        suggestion, pstats = pruning_iteration(
            self.suggestion,
            fold_segment,
            self.prune_funcs,
            threshold,
            self.prune_level,
            ref_cur,
            ref_prev,
            scale_cur,
            scale_prev,
            seg_cur,
            self.seg_ref,
            self.load_func,
            self.max_sugg,
        )
        log_str = (
            f"level: {self.prune_level}, seg_cur: {seg_cur}, "
            f"lb_leaves= {pstats.lb_leaves:.2f}, "
            f"branch_frac= {pstats.branch_frac_tot:.2f}, "
        )
        if suggestion.size == 0:
            self._complete = True
            self._suggestion = suggestion
            self.logger.info(log_str)
            self.logger.info(f"Pruning complete at level: {self.prune_level}")
            return

        log_str += (
            f"score thresh: {threshold:.2f}, max: {suggestion.scores.max():.2f}, "
            f"min: {suggestion.scores.min():.2f}, P(surv): {pstats.surv_frac:.2f}"
        )
        self.logger.info(log_str)
        # Records to track the numercal stability of the algorithm
        self._best_intermediate_arr[self.prune_level] = suggestion.get_best()
        self._backtrack_arr[self.prune_level, : suggestion.size] = suggestion.backtracks
        self._suggestion = suggestion

    def generate_branching_pattern(self, n_iters: int, isuggest: int = 0) -> np.ndarray:
        branching_pattern = []
        leaf_param_sets = self.suggestion.param_sets
        for ii in range(1, n_iters + 1):
            leaves_arr = self.prune_funcs.branch(leaf_param_sets[isuggest], ii)
            branching_pattern.append(len(leaves_arr))
            leaf_param_sets = leaves_arr
        return np.array(branching_pattern)
