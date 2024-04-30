from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from rich.progress import track

from pruning import kernels, utils
from pruning.base import PruningDPFunctions

if TYPE_CHECKING:
    from typing import Callable

    from pruning.ffa import DynamicProgramming


@njit(cache=True)
def load_folds_seg_1d(fold_in: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """fold_in shape: (nfreqs, 2, nbins)."""
    return fold_in[param_idx[0]]


@njit(cache=True)
def load_folds_seg_2d(fold_in: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """fold_in shape: (naccels, nfreqs, 2, nbins)."""
    return fold_in[param_idx[0], param_idx[1]]


@njit(cache=True)
def load_folds_seg_3d(fold_in: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """fold_in shape: (njerks, naccels, nfreqs, 2, nbins)."""
    return fold_in[param_idx[0], param_idx[1], param_idx[2]]


@njit(cache=True)
def load_folds_seg_4d(fold_in: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """fold_in shape: (nsnap, njerks, naccels, nfreqs, 2, nbins)."""
    return fold_in[param_idx[0], param_idx[1], param_idx[2], param_idx[3]]


@njit
def pruning_iteration(
    suggestion: kernels.SuggestionStruct,
    fold_segment: np.ndarray,
    prune_funcs: PruningDPFunctions,
    threshold: float,
    idx_distance: int,
    prune_level: int,
    load_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    max_sugg: int = 2**17,
) -> tuple[kernels.SuggestionStruct, kernels.PruneStats]:
    suggestion_new = suggestion.get_new(max_sugg)
    n_leaves_total = 0
    iparam = 0

    for isuggest in range(suggestion.size):
        leaves_arr = prune_funcs.branch(suggestion.param_sets[isuggest], prune_level)
        n_leaves = len(leaves_arr)

        for ileaf in range(n_leaves):
            leaf_param_set = leaves_arr[ileaf]
            param_idx, phase_shift = prune_funcs.resolve(leaf_param_set, idx_distance)
            partial_res = prune_funcs.shift(
                load_func(fold_segment, param_idx),
                phase_shift,
            )
            combined_res = prune_funcs.add(suggestion.folds[isuggest], partial_res)
            score = prune_funcs.score_func(combined_res)

            if score >= threshold:
                suggestion_new.param_sets[iparam] = leaf_param_set
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
        n_leaves_total += n_leaves

    suggestion_new = suggestion_new.trim_empty(iparam)
    stats = kernels.PruneStats(
        suggestion.size,
        n_leaves_total,
        n_leaves_total,
        suggestion_new.size,
    )
    return suggestion_new, stats


class Pruning:
    def __init__(
        self,
        dyp: DynamicProgramming,
        threshold_scheme: np.ndarray,
        max_sugg: int = 2**17,
    ) -> None:
        self._dyp = dyp
        self._prune_funcs = PruningDPFunctions(
            dyp.cfg,
            dyp.param_arr,
            dyp.dparams,
            dyp.chunk_duration,
        )
        self._load_func = self._set_load_func(dyp.cfg.nparams)
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
    def load_func(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return self._load_func

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
        seg_cur = self.segment_scheme[self.prune_level]
        fold_segment = self.prune_funcs.load(self.dyp.fold, seg_cur)
        idx_distance = seg_cur - self.seg_ref
        threshold = self.threshold_scheme[self.prune_level]
        suggestion, pstats = pruning_iteration(
            self.suggestion,
            fold_segment,
            self.prune_funcs,
            threshold,
            idx_distance,
            self.prune_level,
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

    def _set_load_func(
        self,
        nparams: int,
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        nparams_to_load_func = {
            1: load_folds_seg_1d,
            2: load_folds_seg_2d,
            3: load_folds_seg_3d,
            4: load_folds_seg_4d,
        }
        return nparams_to_load_func[nparams]
