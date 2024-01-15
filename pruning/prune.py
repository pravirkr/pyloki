from __future__ import annotations
from rich.progress import track
from numba import njit
import numpy as np

from pruning.ffa import DynamicProgramming
from pruning import kernels, utils, base


@njit
def pruning_iteration(
    fold_segment: np.ndarray,
    prune_funcns: base.PruningAccelDPFunctions,
    suggestion: kernels.SuggestionStruct,
    threshold: float,
    idx_distance: int,
    max_sugg: int = 131072,
) -> tuple[kernels.SuggestionStruct, int]:
    suggestion_new = suggestion.init_new(max_sugg)
    n_leaves_total = 0
    iparam = 0

    for isuggest in range(suggestion.size):
        leaves_arr = prune_funcns.branch(suggestion.param_sets[isuggest], idx_distance)
        n_leaves_total += len(leaves_arr)

        for _, leaf_params in enumerate(leaves_arr):
            data_pos, phase_shift = prune_funcns.resolve(leaf_params, idx_distance)
            partial_res = prune_funcns.shift(fold_segment[data_pos], phase_shift)
            combined_res = prune_funcns.add(suggestion.data[isuggest], partial_res)
            score = prune_funcns.score_func(combined_res)

            if score >= threshold:
                suggestion_new.param_sets[iparam] = leaf_params
                suggestion_new.data[iparam] = combined_res
                suggestion_new.scores[iparam] = score
                suggestion_new.backtracks[iparam] = np.array(
                    [isuggest] + list(data_pos) + [phase_shift]
                )

                iparam += 1
                if iparam == max_sugg:
                    threshold = np.median(suggestion_new.scores)
                    suggestion_new = suggestion_new.apply_threshold(threshold)
                    iparam = suggestion_new.actual_size

    suggestion_new = suggestion_new.trim_empty(iparam)
    return suggestion_new, n_leaves_total


class Pruning(object):
    def __init__(
        self,
        dyp: DynamicProgramming,
        threshold_scheme: np.ndarray,
        max_sugg: int = 131072,
    ) -> None:
        self._max_sugg = max_sugg
        self._dyp = dyp
        self._prune_funcns = self._set_prune_funcns(dyp)

        self._threshold_scheme = threshold_scheme

        self._iter_num = None
        self._complete = False
        self.nleaves_arr = []

    @property
    def max_sugg(self) -> int:
        return self._max_sugg

    @property
    def dyp(self) -> DynamicProgramming:
        return self._dyp

    @property
    def iter_num(self) -> int:
        return self._iter_num

    @property
    def is_complete(self) -> bool:
        return self._complete

    @property
    def fold(self) -> np.ndarray:
        return self.dyp.fold

    @property
    def prune_funcns(self) -> base.PruningAccelDPFunctions:
        return self._prune_funcns

    @property
    def segment_access_scheme(self) -> np.ndarray:
        return self._segment_access_scheme

    @property
    def param_ref_ind(self) -> int:
        return self.segment_access_scheme[0]

    @property
    def threshold_scheme(self) -> np.ndarray:
        return self._threshold_scheme

    def initialize(self, ref_ind: int = 12) -> None:
        # Should allow to perform starting position outer enumeration (for agressive pruning)
        # Should also allow for "middle-out" enumeration scheme (more efficient)
        self._segment_access_scheme = utils.snail_access_scheme(
            self.dyp.nchunks, ref_ind=ref_ind
        )
        self._complete = False

        iter_num = 0
        isegment = self.segment_access_scheme[iter_num]
        fold_segment = self.prune_funcns.load(self.fold, isegment)
        self.suggestion = self.prune_funcns.suggest(fold_segment)
        self.ref_time_new_block = (isegment + 0.5) * self.prune_funcns.tchunk_current

        self.best_intermediate_scores = []
        self.best_intermediate_params = []
        self.best_intermediate_folds = []
        self.backtrack_arr = np.zeros(
            (self.fold.shape[0], self.max_sugg, len(self.fold.shape[1:]))
        )
        self._iter_num = iter_num

    def prune_iter(self):
        if self.is_complete:
            return
        self._iter_num += 1
        isegment = self.segment_access_scheme[self.iter_num]
        fold_segment = self.prune_funcns.load(self.fold, isegment)
        indexing_distance = isegment - self.param_ref_ind
        suggestion, n_leaves_total = pruning_iteration(
            fold_segment,
            self.prune_funcns,
            self.suggestion,
            self.threshold_scheme[self.iter_num],
            indexing_distance,
            max_sugg=self.max_sugg,
        )
        log_str = (
            f"iter: {self.iter_num}, iseg: {isegment}, "
            f"lb leaves= {np.log2(n_leaves_total):.2f}, branch frac= {n_leaves_total / self.suggestion.size:.1f}, "
        )

        self.nleaves_arr.append(n_leaves_total)
        self.suggestion = suggestion
        self.ref_time_new_block = (isegment + 0.5) * self.prune_funcns.tchunk_current

        if self.suggestion.size == 0:
            self._complete = True
            return
        else:
            log_str += f"score max: {self.suggestion.scores.max():.2f}, min: {self.suggestion.scores.min():.2f}, "
            log_str += f"P(surv): {self.suggestion.size / n_leaves_total:.2f}"
            # With high SNR, these records allow to trace the numerical stability of the pruning algorithm.
            best_ind = np.argmax(self.suggestion.scores)
            self.best_intermediate_scores.append(self.suggestion.scores[best_ind])
            self.best_intermediate_params.append(self.suggestion.param_sets[best_ind])
            self.best_intermediate_folds.append(self.suggestion.data[best_ind])
            self.backtrack_arr[self.iter_num, : suggestion.size] = suggestion.backtracks
        print(log_str)

    def prune_enumeration(self, snr_lim, ref_inds=None, lazy=True):
        res = []
        if ref_inds is None:
            ref_inds = np.arange(0, self.dyp.nchunks, self.dyp.nchunks // 16)
        for ref_ind in ref_inds:
            self.initialize(ref_ind=ref_ind)
            for _ in track(range(self.dyp.nchunks - 1), description="[red]Pruning..."):
                self.prune_iter()
            if self.suggestion.size > 0 and np.max(self.suggestion.scores) > snr_lim:
                res.append((ref_ind, self.suggestion))
                if lazy:
                    return res
        return res

    def generate_branching_pattern(self, n_iters: int, isuggest: int = 0) -> np.ndarray:
        branching_pattern = []
        leaf_param_sets = self.suggestion.param_sets
        for ii in range(1, n_iters + 1):
            leaves_arr = self.prune_funcns.branch(leaf_param_sets[isuggest], ii)
            branching_pattern.append(len(leaves_arr))
            leaf_param_sets = leaves_arr
        return np.array(branching_pattern)

    def _set_prune_funcns(self, dyp) -> base.PruningAccelDPFunctions:
        nparams_to_dp_funcns = {
            2: base.PruningAccelDPFunctions,
            3: base.PruningJerkDPFunctions,
        }
        return nparams_to_dp_funcns[dyp.params.nparams](
            dyp.params, dyp.param_arr, dyp.dparams, dyp.chunk_duration
        )
