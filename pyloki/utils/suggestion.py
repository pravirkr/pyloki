# ruff: noqa: ARG001, ANN001

from __future__ import annotations

from typing import Self

import numpy as np
from numba import njit, types
from numba.experimental import structref
from numba.extending import overload, overload_method

from pyloki.utils import psr_utils


@structref.register
class SuggestionStructTemplate(types.StructRef):
    pass


class SuggestionStruct(structref.StructRefProxy):
    """A struct to hold suggestions for pruning.

    Parameters
    ----------
    param_sets : np.ndarray
        Array of parameter sets with shape (nsuggestions, nparams, 2)
    folds : np.ndarray
        Array of folded profiles with shape (nsuggestions, ..., 2, nbins)
    scores : np.ndarray
        Array of scores for each suggestion (nsuggestions,)
    backtracks : np.ndarray
        Array of backtracks for each suggestion (nsuggestions, 2 + nparams)
    """

    def __new__(
        cls,
        param_sets: np.ndarray,
        folds: np.ndarray,
        scores: np.ndarray,
        backtracks: np.ndarray,
    ) -> Self:
        """Create a new instance of SuggestionStruct."""
        return suggestion_struct_init(param_sets, folds, scores, backtracks)

    @property
    @njit(cache=True, fastmath=True)
    def param_sets(self) -> np.ndarray:
        return self.param_sets

    @property
    @njit(cache=True, fastmath=True)
    def folds(self) -> np.ndarray:
        return self.folds

    @property
    @njit(cache=True, fastmath=True)
    def scores(self) -> np.ndarray:
        return self.scores

    @property
    @njit(cache=True, fastmath=True)
    def backtracks(self) -> np.ndarray:
        return self.backtracks

    @property
    @njit(cache=True, fastmath=True)
    def valid_size(self) -> int:
        """Get the valid size of the suggestion struct, beyond which is garbage."""
        return self.valid_size

    @valid_size.setter
    @njit(cache=True, fastmath=True)
    def valid_size(self, value: int) -> None:
        self.valid_size = value

    @property
    @njit(cache=True, fastmath=True)
    def size(self) -> int:
        return self.size

    @property
    @njit(cache=True, fastmath=True)
    def nparams(self) -> int:
        return self.param_sets.shape[1]

    @property
    @njit(cache=True, fastmath=True)
    def size_lb(self) -> float:
        return np.log2(self.valid_size) if self.valid_size > 0 else 0.0

    @property
    @njit(cache=True, fastmath=True)
    def score_max(self) -> float:
        return np.max(self.scores[: self.valid_size]) if self.valid_size > 0 else 0.0

    @property
    @njit(cache=True, fastmath=True)
    def score_min(self) -> float:
        return np.min(self.scores[: self.valid_size]) if self.valid_size > 0 else 0.0

    def get_new(self, max_sugg: int) -> SuggestionStruct:
        return get_new_func(self, max_sugg)

    def get_best(self) -> tuple[np.ndarray, np.ndarray, float]:
        return get_best_func(self)

    def get_transformed(self, delta_t: float) -> np.ndarray:
        """Transform the search parameters to some given t_ref.

        Parameters
        ----------
        delta_t : float
            Time shift to apply to the search parameters.

        Returns
        -------
        np.ndarray
            Array of transformed search parameters (nsuggestions, nparams, 2)
        """
        return get_transformed_func(self, delta_t)

    def add(
        self,
        param_set: np.ndarray,
        fold: np.ndarray,
        score: float,
        backtrack: np.ndarray,
    ) -> bool:
        """Add a suggestion to the struct if there is space."""
        return add_func(self, param_set, fold, score, backtrack)

    def trim_threshold(self) -> float:
        """Trim the suggestions to the threshold."""
        return trim_threshold_func(self)

    def trim_empty(self) -> SuggestionStruct:
        """Return only the valid portion of the struct, excluding garbage data."""
        return trim_empty_func(self)

    def trim_repeats(self) -> None:
        """Trim repeated suggestions."""
        return trim_repeats_func(self)

    def trim_repeats_threshold(self) -> float:
        """Trim repeated suggestions and up to the threshold."""
        return trim_repeats_threshold_func(self)

    def _keep(self, indices: np.ndarray) -> None:
        """Optimized in-place update."""
        return keep_func(self, indices)


fields_suggestion_struct = [
    ("param_sets", types.f8[:, :, :]),
    ("folds", types.f4[:, :, :]),
    ("scores", types.f4[:]),
    ("backtracks", types.i4[:, :]),
    ("valid_size", types.int64),
    ("size", types.int64),
]

structref.define_boxing(SuggestionStructTemplate, SuggestionStruct)
SuggestionStructType = SuggestionStructTemplate(fields_suggestion_struct)


@njit(cache=True, fastmath=True)
def suggestion_struct_init(
    param_sets: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    backtracks: np.ndarray,
) -> SuggestionStruct:
    self = structref.new(SuggestionStructType)
    self.param_sets = param_sets
    self.folds = folds
    self.scores = scores
    self.backtracks = backtracks
    self.valid_size = param_sets.shape[0]
    self.size = param_sets.shape[0]
    return self


@njit(cache=True, fastmath=True)
def get_new_func(self: SuggestionStruct, max_sugg: int) -> SuggestionStruct:
    param_sets = np.empty(
        (max_sugg, *self.param_sets.shape[1:]),
        dtype=self.param_sets.dtype,
    )
    folds = np.empty((max_sugg, *self.folds.shape[1:]), dtype=self.folds.dtype)
    scores = np.empty(max_sugg, dtype=self.scores.dtype)
    backtracks = np.empty(
        (max_sugg, self.backtracks.shape[1]),
        dtype=self.backtracks.dtype,
    )
    sugg_new = SuggestionStruct(param_sets, folds, scores, backtracks)
    sugg_new.valid_size = 0
    return sugg_new


@njit(cache=True, fastmath=True)
def get_best_func(self: SuggestionStruct) -> tuple[np.ndarray, np.ndarray, float]:
    idx_max = np.argmax(self.scores[: self.valid_size])
    return (
        self.param_sets[idx_max].copy(),
        self.folds[idx_max].copy(),
        self.scores[idx_max],
    )


@njit(cache=True, fastmath=True)
def get_transformed_func(self: SuggestionStruct, delta_t: float) -> np.ndarray:
    # Exclude last two rows
    return psr_utils.shift_params_batch(self.param_sets[:, :-2, :], delta_t)


@njit(cache=True, fastmath=True)
def add_func(
    self: SuggestionStruct,
    param_set: np.ndarray,
    fold: np.ndarray,
    score: float,
    backtrack: np.ndarray,
) -> bool:
    pos = self.valid_size
    if pos >= self.size:
        return False
    self.param_sets[pos] = param_set
    self.folds[pos] = fold
    self.scores[pos] = score
    self.backtracks[pos] = backtrack
    self.valid_size += 1
    return True


@njit(cache=True, fastmath=True)
def trim_threshold_func(self: SuggestionStruct) -> float:
    threshold = np.median(self.scores[: self.valid_size])
    idx = self.scores[: self.valid_size] >= threshold
    self._keep(idx)
    return threshold


@njit(cache=True, fastmath=True)
def trim_empty_func(self: SuggestionStruct) -> SuggestionStruct:
    return SuggestionStruct(
        self.param_sets[: self.valid_size],
        self.folds[: self.valid_size],
        self.scores[: self.valid_size],
        self.backtracks[: self.valid_size],
    )


@njit(cache=True, fastmath=True)
def trim_repeats_func(self: SuggestionStruct) -> None:
    idx = get_unique_indices_scores(
        self.param_sets[: self.valid_size],
        self.scores[: self.valid_size],
    )
    idx_bool = np.zeros(self.valid_size, dtype=np.bool_)
    idx_bool[idx] = True
    self._keep(idx_bool)


@njit(cache=True, fastmath=True)
def trim_repeats_threshold_func(self: SuggestionStruct) -> float:
    threshold = np.median(self.scores[: self.valid_size])
    idx = get_unique_indices_scores(
        self.param_sets[: self.valid_size],
        self.scores[: self.valid_size],
    )
    idx_bool = np.zeros(self.valid_size, dtype=np.bool_)
    idx_bool[idx] = True
    idx_threshold = self.scores[: self.valid_size] >= threshold
    idx_bool = np.logical_and(idx_bool, idx_threshold)
    self._keep(idx_bool)
    return threshold


@njit(cache=True, fastmath=True)
def keep_func(self: SuggestionStruct, indices: np.ndarray) -> None:
    count = int(np.sum(indices))
    if count == 0:
        self.valid_size = 0
        return
    idx_valid = np.where(indices)[0]
    # move valid data to the front
    self.param_sets[:count] = self.param_sets[idx_valid]
    self.folds[:count] = self.folds[idx_valid]
    self.scores[:count] = self.scores[idx_valid]
    self.backtracks[:count] = self.backtracks[idx_valid]
    # update valid size
    self.valid_size = count


@overload(SuggestionStruct)
def overload_sugg_construct(
    param_sets: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    backtracks: np.ndarray,
) -> types.FunctionType:
    def impl(
        param_sets: np.ndarray,
        folds: np.ndarray,
        scores: np.ndarray,
        backtracks: np.ndarray,
    ) -> SuggestionStruct:
        return suggestion_struct_init(param_sets, folds, scores, backtracks)

    return impl


@overload_method(SuggestionStructTemplate, "get_new")
def ol_get_new_func(self: SuggestionStruct, max_sugg: int) -> types.FunctionType:
    def impl(self: SuggestionStruct, max_sugg: int) -> SuggestionStruct:
        return get_new_func(self, max_sugg)

    return impl


@overload_method(SuggestionStructTemplate, "get_best")
def ol_get_best_func(self: SuggestionStruct) -> types.FunctionType:
    def impl(self: SuggestionStruct) -> tuple[np.ndarray, np.ndarray, float]:
        return get_best_func(self)

    return impl


@overload_method(SuggestionStructTemplate, "get_transformed")
def ol_get_transformed_func(
    self: SuggestionStruct,
    delta_t: float,
) -> types.FunctionType:
    def impl(self: SuggestionStruct, delta_t: float) -> np.ndarray:
        return get_transformed_func(self, delta_t)

    return impl


@overload_method(SuggestionStructTemplate, "add")
def ol_add_func(
    self: SuggestionStruct,
    param_set: np.ndarray,
    fold: np.ndarray,
    score: float,
    backtrack: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: SuggestionStruct,
        param_set: np.ndarray,
        fold: np.ndarray,
        score: float,
        backtrack: np.ndarray,
    ) -> bool:
        return add_func(self, param_set, fold, score, backtrack)

    return impl


@overload_method(SuggestionStructTemplate, "trim_threshold")
def ol_trim_threshold_func(self: SuggestionStruct) -> types.FunctionType:
    def impl(self: SuggestionStruct) -> float:
        return trim_threshold_func(self)

    return impl


@overload_method(SuggestionStructTemplate, "trim_empty")
def ol_trim_empty_func(self: SuggestionStruct) -> types.FunctionType:
    def impl(self: SuggestionStruct) -> SuggestionStruct:
        return trim_empty_func(self)

    return impl


@overload_method(SuggestionStructTemplate, "trim_repeats")
def ol_trim_repeats_func(self: SuggestionStruct) -> types.FunctionType:
    def impl(self: SuggestionStruct) -> None:
        return trim_repeats_func(self)

    return impl


@overload_method(SuggestionStructTemplate, "trim_repeats_threshold")
def ol_trim_repeats_threshold_func(self: SuggestionStruct) -> types.FunctionType:
    def impl(self: SuggestionStruct) -> float:
        return trim_repeats_threshold_func(self)

    return impl


@overload_method(SuggestionStructTemplate, "_keep")
def ol_keep_func(self: SuggestionStruct, indices: np.ndarray) -> types.FunctionType:
    def impl(self: SuggestionStruct, indices: np.ndarray) -> None:
        return keep_func(self, indices)

    return impl


@njit(cache=True, fastmath=True)
def get_unique_indices(params: np.ndarray) -> np.ndarray:
    nparams = params.shape[0]
    unique_dict = {}
    unique_indices = np.empty(nparams, dtype=np.int64)
    count = 0
    for ii in range(nparams):
        key = int(np.round(params[ii][-1:, 0][0] * 10**9))
        if key not in unique_dict:
            unique_dict[key] = True
            unique_indices[count] = ii
            count += 1

    return unique_indices[:count]


@njit(cache=True, fastmath=True)
def get_unique_indices_scores(params: np.ndarray, scores: np.ndarray) -> np.ndarray:
    nparams = params.shape[0]
    unique_dict: dict[int, bool] = {}
    scores_dict: dict[int, float] = {}
    count_dict: dict[int, int] = {}
    unique_indices = np.empty(nparams, dtype=np.int64)
    count = 0
    for ii in range(nparams):
        key = int(np.sum(params[ii][-2:, 0] * 10**9) + 0.5)
        if unique_dict.get(key, False):
            if scores[ii] > scores_dict[key]:
                scores_dict[key] = scores[ii]
                count = count_dict[key]
                unique_indices[count] = ii
        else:
            unique_dict[key] = True
            scores_dict[key] = scores[ii]
            count_dict[key] = count
            unique_indices[count] = ii
            count += 1
    return unique_indices[:count]
