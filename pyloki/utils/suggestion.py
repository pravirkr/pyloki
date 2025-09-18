# ruff: noqa: ARG001

from __future__ import annotations

from typing import Self

import numpy as np
from numba import njit, types
from numba.experimental import structref
from numba.extending import overload, overload_method

from pyloki.utils import transforms


@structref.register
class SuggestionStructTemplate(types.StructRef):
    pass


@structref.register
class SuggestionStructComplexTemplate(types.StructRef):
    pass


class SuggestionStruct(structref.StructRefProxy):
    """A struct to hold suggestions for pruning.

    Parameters
    ----------
    param_sets : np.ndarray
        Array of parameter sets. Shape: (nsuggestions, nparams + 2, 2)
    folds : np.ndarray
        Array of folded profiles. Shape: (nsuggestions, 2, nbins)
    scores : np.ndarray
        Array of scores. Shape: (nsuggestions,)
    backtracks : np.ndarray
        Array of backtracks. Shape: (nsuggestions, nparams + 2)
    mode : str
        The mode (basis) of the nparams in the param_sets.
        - "taylor" : Taylor basis (snap, jerk, accel, freq, period)
        - "circular" : Circular basis (omega, freq, x_cos_phi, x_sin_phi)
        - "chebyshev" : Chebyshev basis (..., freq)

    Notes
    -----
    The last row rows of the param_sets is reserved.
    - row (-1) : f0, _
    """

    def __new__(
        cls,
        param_sets: np.ndarray,
        folds: np.ndarray,
        scores: np.ndarray,
        backtracks: np.ndarray,
        mode: str = "taylor",
    ) -> Self:
        """Create a new instance of SuggestionStruct."""
        return suggestion_struct_init(param_sets, folds, scores, backtracks, mode)

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
    def mode(self) -> str:
        return self.mode

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
        return self.nparams

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

    def get_transformed(
        self,
        coord_mid: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> np.ndarray:
        """Transform the search parameters to some given t_ref.

        Parameters
        ----------
        coord_mid : tuple[float, float]
            Final representation coordinate.
        coord_init : tuple[float, float]
            Initial representation coordinate.

        Returns
        -------
        np.ndarray
            Array of transformed search parameters (nsuggestions, nparams, 2)
        """
        return get_transformed_func(self, coord_mid, coord_init)

    def add(
        self,
        param_set: np.ndarray,
        fold: np.ndarray,
        score: float,
        backtrack: np.ndarray,
    ) -> bool:
        """Add a suggestion to the struct if there is space."""
        return add_func(self, param_set, fold, score, backtrack)

    def add_batch(
        self,
        param_sets_batch: np.ndarray,
        folds_batch: np.ndarray,
        scores_batch: np.ndarray,
        backtracks_batch: np.ndarray,
        current_threshold: float,
    ) -> float:
        """Add a batch of suggestions to the struct.

        If the buffer is full, it will be trimmed and the threshold will be updated.

        """
        return add_batch_func(
            self,
            param_sets_batch,
            folds_batch,
            scores_batch,
            backtracks_batch,
            current_threshold,
        )

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


class SuggestionStructComplex(structref.StructRefProxy):
    """A struct to hold suggestions for pruning.

    This is the same as SuggestionStruct, but with complex folded profiles.
    """

    def __new__(
        cls,
        param_sets: np.ndarray,
        folds: np.ndarray,
        scores: np.ndarray,
        backtracks: np.ndarray,
        mode: str = "taylor",
    ) -> Self:
        """Create a new instance of SuggestionStruct."""
        return suggestion_struct_complex_init(
            param_sets,
            folds,
            scores,
            backtracks,
            mode,
        )

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
    def mode(self) -> str:
        return self.mode

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
        return self.nparams

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
        return get_new_func_complex(self, max_sugg)

    def get_best(self) -> tuple[np.ndarray, np.ndarray, float]:
        return get_best_func(self)

    def get_transformed(
        self,
        coord_mid: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> np.ndarray:
        return get_transformed_func(self, coord_mid, coord_init)

    def add(
        self,
        param_set: np.ndarray,
        fold: np.ndarray,
        score: float,
        backtrack: np.ndarray,
    ) -> bool:
        return add_func(self, param_set, fold, score, backtrack)

    def add_batch(
        self,
        param_sets_batch: np.ndarray,
        folds_batch: np.ndarray,
        scores_batch: np.ndarray,
        backtracks_batch: np.ndarray,
        current_threshold: float,
    ) -> float:
        return add_batch_func(
            self,
            param_sets_batch,
            folds_batch,
            scores_batch,
            backtracks_batch,
            current_threshold,
        )

    def trim_threshold(self) -> float:
        """Trim the suggestions to the threshold."""
        return trim_threshold_func(self)

    def trim_empty(self) -> SuggestionStruct:
        """Return only the valid portion of the struct, excluding garbage data."""
        return trim_empty_func_complex(self)

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
    ("param_sets", types.f8[:, :, ::1]),
    ("folds", types.f4[:, :, ::1]),
    ("scores", types.f4[:]),
    ("backtracks", types.i4[:, ::1]),
    ("mode", types.unicode_type),
    ("valid_size", types.int64),
    ("size", types.int64),
    ("nparams", types.int64),
]

structref.define_boxing(SuggestionStructTemplate, SuggestionStruct)
SuggestionStructType = SuggestionStructTemplate(fields_suggestion_struct)

fields_suggestion_struct_complex = [
    ("param_sets", types.f8[:, :, ::1]),
    ("folds", types.c8[:, :, ::1]),
    ("scores", types.f4[:]),
    ("backtracks", types.i4[:, ::1]),
    ("mode", types.unicode_type),
    ("valid_size", types.int64),
    ("size", types.int64),
    ("nparams", types.int64),
]

structref.define_boxing(SuggestionStructComplexTemplate, SuggestionStructComplex)
SuggestionStructComplexType = SuggestionStructComplexTemplate(
    fields_suggestion_struct_complex,
)


@njit(cache=True, fastmath=True)
def suggestion_struct_init(
    param_sets: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    backtracks: np.ndarray,
    mode: str = "taylor",
) -> SuggestionStruct:
    self = structref.new(SuggestionStructType)
    self.param_sets = param_sets
    self.folds = folds
    self.scores = scores
    self.backtracks = backtracks
    self.mode = mode
    self.valid_size = param_sets.shape[0]
    self.size = param_sets.shape[0]
    self.nparams = param_sets.shape[1] - 2
    return self


@njit(cache=True, fastmath=True)
def suggestion_struct_complex_init(
    param_sets: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    backtracks: np.ndarray,
    mode: str = "taylor",
) -> SuggestionStructComplex:
    self = structref.new(SuggestionStructComplexType)
    self.param_sets = param_sets
    self.folds = folds
    self.scores = scores
    self.backtracks = backtracks
    self.mode = mode
    self.valid_size = param_sets.shape[0]
    self.size = param_sets.shape[0]
    self.nparams = param_sets.shape[1] - 2
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
    sugg_new = SuggestionStruct(param_sets, folds, scores, backtracks, self.mode)
    sugg_new.valid_size = 0
    return sugg_new


@njit(cache=True, fastmath=True)
def get_new_func_complex(
    self: SuggestionStructComplex,
    max_sugg: int,
) -> SuggestionStructComplex:
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
    sugg_new = SuggestionStructComplex(param_sets, folds, scores, backtracks, self.mode)
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
def get_transformed_func(
    self: SuggestionStruct,
    coord_mid: tuple[float, float],
    coord_init: tuple[float, float],
) -> np.ndarray:
    if self.mode == "taylor":
        return transforms.taylor_report_batch(self.param_sets)
    if self.mode == "chebyshev":
        return transforms.chebyshev_report_batch(self.param_sets, coord_mid)
    if self.mode == "taylor_fixed":
        delta_t = coord_mid[0] - coord_init[0]
        if self.nparams < 4:
            return transforms.taylor_fixed_report_batch(self.param_sets, delta_t)
        if self.nparams == 4:
            return transforms.taylor_fixed_circular_report_batch(
                self.param_sets,
                delta_t,
            )
        msg = "For Taylor fixed suggestion struct, nparams must be less than 4."
        raise ValueError(msg)
    msg = "mode must be in ['taylor', 'chebyshev', 'taylor_fixed']."
    raise ValueError(msg)


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
def add_batch_func(
    self: SuggestionStruct,
    param_sets_batch: np.ndarray,
    folds_batch: np.ndarray,
    scores_batch: np.ndarray,
    backtracks_batch: np.ndarray,
    current_threshold: float,
) -> float:
    num_to_add = len(scores_batch)
    if num_to_add == 0:
        return current_threshold
    effective_threshold = current_threshold

    # Start with all candidates
    mask = scores_batch >= effective_threshold
    idxs = np.where(mask)[0]

    while len(idxs) > 0:
        space_left = self.size - self.valid_size
        if space_left == 0:
            # Buffer is full, try to trim
            new_threshold_from_trim = self.trim_threshold()
            effective_threshold = max(effective_threshold, new_threshold_from_trim)
            # Re-filter after new threshold
            mask = scores_batch >= effective_threshold
            idxs = np.where(mask)[0]
            continue  # Try again with new threshold

        n_to_add = min(len(idxs), space_left)
        pos = self.valid_size
        # Batched assignment
        self.param_sets[pos : pos + n_to_add] = param_sets_batch[idxs[:n_to_add]]
        self.folds[pos : pos + n_to_add] = folds_batch[idxs[:n_to_add]]
        self.scores[pos : pos + n_to_add] = scores_batch[idxs[:n_to_add]]
        self.backtracks[pos : pos + n_to_add] = backtracks_batch[idxs[:n_to_add]]
        self.valid_size += n_to_add

        # Remove added candidates from idxs
        idxs = idxs[n_to_add:]
    return effective_threshold


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
        self.mode,
    )


@njit(cache=True, fastmath=True)
def trim_empty_func_complex(self: SuggestionStructComplex) -> SuggestionStructComplex:
    return SuggestionStructComplex(
        self.param_sets[: self.valid_size],
        self.folds[: self.valid_size],
        self.scores[: self.valid_size],
        self.backtracks[: self.valid_size],
        self.mode,
    )


@njit(cache=True, fastmath=True)
def trim_repeats_func(self: SuggestionStruct) -> None:
    idx = get_unique_indices_scores(
        self.param_sets[: self.valid_size, : self.nparams, 0],
        self.scores[: self.valid_size],
    )
    idx_bool = np.zeros(self.valid_size, dtype=np.bool_)
    idx_bool[idx] = True
    self._keep(idx_bool)


@njit(cache=True, fastmath=True)
def trim_repeats_threshold_func(self: SuggestionStruct) -> float:
    threshold = np.median(self.scores[: self.valid_size])
    idx = get_unique_indices_scores(
        self.param_sets[: self.valid_size, : self.nparams, 0],
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
    mode: str,
) -> types.FunctionType:
    def impl(
        param_sets: np.ndarray,
        folds: np.ndarray,
        scores: np.ndarray,
        backtracks: np.ndarray,
        mode: str,
    ) -> SuggestionStruct:
        return suggestion_struct_init(param_sets, folds, scores, backtracks, mode)

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
    coord_mid: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: SuggestionStruct,
        coord_mid: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> np.ndarray:
        return get_transformed_func(self, coord_mid, coord_init)

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


@overload_method(SuggestionStructTemplate, "add_batch")
def ol_add_batch_func(
    self: SuggestionStruct,
    param_sets_batch: np.ndarray,
    folds_batch: np.ndarray,
    scores_batch: np.ndarray,
    backtracks_batch: np.ndarray,
    current_threshold: float,
) -> types.FunctionType:
    def impl(
        self: SuggestionStruct,
        param_sets_batch: np.ndarray,
        folds_batch: np.ndarray,
        scores_batch: np.ndarray,
        backtracks_batch: np.ndarray,
        current_threshold: float,
    ) -> float:
        return add_batch_func(
            self,
            param_sets_batch,
            folds_batch,
            scores_batch,
            backtracks_batch,
            current_threshold,
        )

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


@overload(SuggestionStructComplex)
def overload_sugg_construct_complex(
    param_sets: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    backtracks: np.ndarray,
    mode: str,
) -> types.FunctionType:
    def impl(
        param_sets: np.ndarray,
        folds: np.ndarray,
        scores: np.ndarray,
        backtracks: np.ndarray,
        mode: str,
    ) -> SuggestionStructComplex:
        return suggestion_struct_complex_init(
            param_sets,
            folds,
            scores,
            backtracks,
            mode,
        )

    return impl


@overload_method(SuggestionStructComplexTemplate, "get_new")
def ol_get_new_func_complex(
    self: SuggestionStructComplex,
    max_sugg: int,
) -> types.FunctionType:
    def impl(self: SuggestionStructComplex, max_sugg: int) -> SuggestionStructComplex:
        return get_new_func_complex(self, max_sugg)

    return impl


@overload_method(SuggestionStructComplexTemplate, "get_best")
def ol_get_best_func_complex(
    self: SuggestionStructComplex,
) -> types.FunctionType:
    def impl(self: SuggestionStructComplex) -> tuple[np.ndarray, np.ndarray, float]:
        return get_best_func(self)

    return impl


@overload_method(SuggestionStructComplexTemplate, "get_transformed")
def ol_get_transformed_func_complex(
    self: SuggestionStructComplex,
    coord_mid: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: SuggestionStructComplex,
        coord_mid: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> np.ndarray:
        return get_transformed_func(self, coord_mid, coord_init)

    return impl


@overload_method(SuggestionStructComplexTemplate, "add")
def ol_add_func_complex(
    self: SuggestionStructComplex,
    param_set: np.ndarray,
    fold: np.ndarray,
    score: float,
    backtrack: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: SuggestionStructComplex,
        param_set: np.ndarray,
        fold: np.ndarray,
        score: float,
        backtrack: np.ndarray,
    ) -> bool:
        return add_func(self, param_set, fold, score, backtrack)

    return impl


@overload_method(SuggestionStructComplexTemplate, "add_batch")
def ol_add_batch_func_complex(
    self: SuggestionStructComplex,
    param_sets_batch: np.ndarray,
    folds_batch: np.ndarray,
    scores_batch: np.ndarray,
    backtracks_batch: np.ndarray,
    current_threshold: float,
) -> types.FunctionType:
    def impl(
        self: SuggestionStructComplex,
        param_sets_batch: np.ndarray,
        folds_batch: np.ndarray,
        scores_batch: np.ndarray,
        backtracks_batch: np.ndarray,
        current_threshold: float,
    ) -> float:
        return add_batch_func(
            self,
            param_sets_batch,
            folds_batch,
            scores_batch,
            backtracks_batch,
            current_threshold,
        )

    return impl


@overload_method(SuggestionStructComplexTemplate, "trim_threshold")
def ol_trim_threshold_func_complex(
    self: SuggestionStructComplex,
) -> types.FunctionType:
    def impl(self: SuggestionStructComplex) -> float:
        return trim_threshold_func(self)

    return impl


@overload_method(SuggestionStructComplexTemplate, "trim_empty")
def ol_trim_empty_func_complex(
    self: SuggestionStructComplex,
) -> types.FunctionType:
    def impl(self: SuggestionStructComplex) -> SuggestionStructComplex:
        return trim_empty_func_complex(self)

    return impl


@overload_method(SuggestionStructComplexTemplate, "trim_repeats")
def ol_trim_repeats_func_complex(
    self: SuggestionStructComplex,
) -> types.FunctionType:
    def impl(self: SuggestionStructComplex) -> None:
        return trim_repeats_func(self)

    return impl


@overload_method(SuggestionStructComplexTemplate, "trim_repeats_threshold")
def ol_trim_repeats_threshold_func_complex(
    self: SuggestionStructComplex,
) -> types.FunctionType:
    def impl(self: SuggestionStructComplex) -> float:
        return trim_repeats_threshold_func(self)

    return impl


@overload_method(SuggestionStructComplexTemplate, "_keep")
def ol_keep_func_complex(
    self: SuggestionStructComplex,
    indices: np.ndarray,
) -> types.FunctionType:
    def impl(self: SuggestionStructComplex, indices: np.ndarray) -> None:
        return keep_func(self, indices)

    return impl


@njit(cache=True, fastmath=True)
def get_unique_indices(param_sets: np.ndarray) -> np.ndarray:
    nparams = param_sets.shape[0]
    unique_dict = {}
    unique_indices = np.empty(nparams, dtype=np.int64)
    count = 0
    for ii in range(nparams):
        key = int(np.round(param_sets[ii][-1:, 0][0] * 10**9))
        if key not in unique_dict:
            unique_dict[key] = True
            unique_indices[count] = ii
            count += 1

    return unique_indices[:count]


@njit(cache=True, fastmath=True)
def get_unique_indices_scores(param_sets: np.ndarray, scores: np.ndarray) -> np.ndarray:
    nparams = param_sets.shape[0]
    unique_dict: dict[int, bool] = {}
    scores_dict: dict[int, float] = {}
    count_dict: dict[int, int] = {}
    unique_indices = np.empty(nparams, dtype=np.int64)
    count = 0
    for ii in range(nparams):
        key = int(np.sum(param_sets[ii][-2:, 0] * 10**9) + 0.5)
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
