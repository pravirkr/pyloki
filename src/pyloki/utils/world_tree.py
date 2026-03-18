# ruff: noqa: ARG001

from __future__ import annotations

import math
from typing import Self

import numpy as np
from numba import njit, types
from numba.experimental import structref
from numba.extending import overload, overload_method


@structref.register
class WorldTreeTemplate(types.StructRef):
    pass


@structref.register
class WorldTreeComplexTemplate(types.StructRef):
    pass


class WorldTree(structref.StructRefProxy):
    """A struct to hold candidates for pruning.

    Parameters
    ----------
    leaves : np.ndarray
        Array of leaves. Shape: (n_leaves, nparams + 2, 2)
    folds : np.ndarray
        Array of folded profiles. Shape: (n_leaves, 2, nbins)
    scores : np.ndarray
        Array of scores. Shape: (n_leaves,)
    backtracks : np.ndarray
        Array of backtracks. Shape: (n_leaves, nparams + 2)

    Notes
    -----
    The last row rows of the leaves is reserved.
    - row (-1) : f0, _
    """

    def __new__(
        cls,
        leaves: np.ndarray,
        folds: np.ndarray,
        scores: np.ndarray,
        backtracks: np.ndarray,
    ) -> Self:
        """Create a new instance of WorldTree."""
        return world_tree_init(leaves, folds, scores, backtracks)

    @property
    @njit(cache=True, fastmath=True)
    def leaves(self) -> np.ndarray:
        return self.leaves

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
        """Get the valid size of the world tree, beyond which is garbage."""
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

    def get_new(self, max_sugg: int) -> WorldTree:
        return get_new_func(self, max_sugg)

    def get_best(self) -> tuple[np.ndarray, np.ndarray, float]:
        return get_best_func(self)

    def add(
        self,
        leaf: np.ndarray,
        fold: np.ndarray,
        score: float,
        backtrack: np.ndarray,
    ) -> bool:
        """Add a candidate leaf to the world tree if there is space."""
        return add_func(self, leaf, fold, score, backtrack)

    def add_batch(
        self,
        leaves_batch: np.ndarray,
        folds_batch: np.ndarray,
        scores_batch: np.ndarray,
        backtracks_batch: np.ndarray,
        current_threshold: float,
    ) -> float:
        """Add a batch of candidate leaves to the world tree.

        If the buffer is full, it will be trimmed and the threshold will be updated.

        """
        return add_batch_func(
            self,
            leaves_batch,
            folds_batch,
            scores_batch,
            backtracks_batch,
            current_threshold,
        )

    def prune_on_overload(
        self,
        scores_batch: np.ndarray,
        current_threshold: float,
    ) -> float:
        """Prune the candidates on overload."""
        return prune_on_overload_func(self, scores_batch, current_threshold)

    def trim_empty(self) -> WorldTree:
        """Return only the valid portion of the struct, excluding garbage data."""
        return trim_empty_func(self)

    def trim_repeats(self) -> None:
        """Trim repeated candidates."""
        return trim_repeats_func(self)

    def trim_repeats_threshold(self) -> float:
        """Trim repeated candidates and up to the threshold."""
        return trim_repeats_threshold_func(self)

    def _keep(self, indices: np.ndarray) -> None:
        """Optimized in-place update."""
        return keep_func(self, indices)


class WorldTreeComplex(structref.StructRefProxy):
    """A struct to hold candidates for pruning.

    This is the same as WorldTree, but with complex folded profiles.
    """

    def __new__(
        cls,
        leaves: np.ndarray,
        folds: np.ndarray,
        scores: np.ndarray,
        backtracks: np.ndarray,
    ) -> Self:
        """Create a new instance of WorldTree."""
        return world_tree_complex_init(
            leaves,
            folds,
            scores,
            backtracks,
        )

    @property
    @njit(cache=True, fastmath=True)
    def leaves(self) -> np.ndarray:
        return self.leaves

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
        """Get the valid size of the world tree, beyond which is garbage."""
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

    def get_new(self, max_sugg: int) -> WorldTree:
        return get_new_func_complex(self, max_sugg)

    def get_best(self) -> tuple[np.ndarray, np.ndarray, float]:
        return get_best_func(self)

    def add(
        self,
        leaf: np.ndarray,
        fold: np.ndarray,
        score: float,
        backtrack: np.ndarray,
    ) -> bool:
        return add_func(self, leaf, fold, score, backtrack)

    def add_batch(
        self,
        leaves_batch: np.ndarray,
        folds_batch: np.ndarray,
        scores_batch: np.ndarray,
        backtracks_batch: np.ndarray,
        current_threshold: float,
    ) -> float:
        return add_batch_func(
            self,
            leaves_batch,
            folds_batch,
            scores_batch,
            backtracks_batch,
            current_threshold,
        )

    def prune_on_overload(
        self,
        scores_batch: np.ndarray,
        current_threshold: float,
    ) -> float:
        """Prune the candidates on overload."""
        return prune_on_overload_func(self, scores_batch, current_threshold)

    def trim_empty(self) -> WorldTree:
        """Return only the valid portion of the struct, excluding garbage data."""
        return trim_empty_func_complex(self)

    def trim_repeats(self) -> None:
        """Trim repeated candidates."""
        return trim_repeats_func(self)

    def trim_repeats_threshold(self) -> float:
        """Trim repeated candidates and up to the threshold."""
        return trim_repeats_threshold_func(self)

    def _keep(self, indices: np.ndarray) -> None:
        """Optimized in-place update."""
        return keep_func(self, indices)


fields_world_tree = [
    ("leaves", types.f8[:, :, ::1]),
    ("folds", types.f4[:, :, ::1]),
    ("scores", types.f4[:]),
    ("backtracks", types.i4[:, ::1]),
    ("valid_size", types.int64),
    ("size", types.int64),
    ("nparams", types.int64),
]

structref.define_boxing(WorldTreeTemplate, WorldTree)
WorldTreeType = WorldTreeTemplate(fields_world_tree)

fields_world_tree_complex = [
    ("leaves", types.f8[:, :, ::1]),
    ("folds", types.c8[:, :, ::1]),
    ("scores", types.f4[:]),
    ("backtracks", types.i4[:, ::1]),
    ("valid_size", types.int64),
    ("size", types.int64),
    ("nparams", types.int64),
]

structref.define_boxing(WorldTreeComplexTemplate, WorldTreeComplex)
WorldTreeComplexType = WorldTreeComplexTemplate(
    fields_world_tree_complex,
)


@njit(cache=True, fastmath=True)
def world_tree_init(
    leaves: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    backtracks: np.ndarray,
) -> WorldTree:
    self = structref.new(WorldTreeType)
    self.leaves = leaves
    self.folds = folds
    self.scores = scores
    self.backtracks = backtracks
    self.valid_size = leaves.shape[0]
    self.size = leaves.shape[0]
    self.nparams = leaves.shape[1] - 2
    return self


@njit(cache=True, fastmath=True)
def world_tree_complex_init(
    leaves: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    backtracks: np.ndarray,
) -> WorldTreeComplex:
    self = structref.new(WorldTreeComplexType)
    self.leaves = leaves
    self.folds = folds
    self.scores = scores
    self.backtracks = backtracks
    self.valid_size = leaves.shape[0]
    self.size = leaves.shape[0]
    self.nparams = leaves.shape[1] - 2
    return self


@njit(cache=True, fastmath=True)
def get_new_func(self: WorldTree, max_sugg: int) -> WorldTree:
    leaves = np.empty(
        (max_sugg, *self.leaves.shape[1:]),
        dtype=self.leaves.dtype,
    )
    folds = np.empty((max_sugg, *self.folds.shape[1:]), dtype=self.folds.dtype)
    scores = np.empty(max_sugg, dtype=self.scores.dtype)
    backtracks = np.empty(
        (max_sugg, self.backtracks.shape[1]),
        dtype=self.backtracks.dtype,
    )
    tree_new = WorldTree(leaves, folds, scores, backtracks)
    tree_new.valid_size = 0
    return tree_new


@njit(cache=True, fastmath=True)
def get_new_func_complex(
    self: WorldTreeComplex,
    max_sugg: int,
) -> WorldTreeComplex:
    leaves = np.empty(
        (max_sugg, *self.leaves.shape[1:]),
        dtype=self.leaves.dtype,
    )
    folds = np.empty((max_sugg, *self.folds.shape[1:]), dtype=self.folds.dtype)
    scores = np.empty(max_sugg, dtype=self.scores.dtype)
    backtracks = np.empty(
        (max_sugg, self.backtracks.shape[1]),
        dtype=self.backtracks.dtype,
    )
    tree_new = WorldTreeComplex(leaves, folds, scores, backtracks)
    tree_new.valid_size = 0
    return tree_new


@njit(cache=True, fastmath=True)
def get_best_func(self: WorldTree) -> tuple[np.ndarray, np.ndarray, float]:
    idx_max = np.argmax(self.scores[: self.valid_size])
    return (
        self.leaves[idx_max].copy(),
        self.folds[idx_max].copy(),
        self.scores[idx_max],
    )


@njit(cache=True, fastmath=True)
def add_func(
    self: WorldTree,
    leaf: np.ndarray,
    fold: np.ndarray,
    score: float,
    backtrack: np.ndarray,
) -> bool:
    pos = self.valid_size
    if pos >= self.size:
        return False
    self.leaves[pos] = leaf
    self.folds[pos] = fold
    self.scores[pos] = score
    self.backtracks[pos] = backtrack
    self.valid_size += 1
    return True


@njit(cache=True, fastmath=True)
def add_batch_func(
    self: WorldTree,
    leaves_batch: np.ndarray,
    folds_batch: np.ndarray,
    scores_batch: np.ndarray,
    backtracks_batch: np.ndarray,
    current_threshold: float,
) -> float:
    num_to_add = len(scores_batch)
    if num_to_add == 0:
        return current_threshold

    space_left = self.size - self.valid_size
    # If there is enough space, add all (fast path)
    if num_to_add <= space_left:
        pos = self.valid_size
        self.leaves[pos : pos + num_to_add] = leaves_batch
        self.folds[pos : pos + num_to_add] = folds_batch
        self.scores[pos : pos + num_to_add] = scores_batch
        self.backtracks[pos : pos + num_to_add] = backtracks_batch
        self.valid_size += num_to_add
        return current_threshold

    # Slow path: Overflow & Pruning (Median + Top-K Strategy)
    effective_threshold = self.prune_on_overload(scores_batch, current_threshold)
    idxs = np.arange(num_to_add)
    pending_idxs = idxs[scores_batch >= effective_threshold]
    space_left = self.size - self.valid_size
    n_to_add = min(len(pending_idxs), space_left)
    if n_to_add == 0:
        return effective_threshold
    # Batched assignment
    pos = self.valid_size
    self.leaves[pos : pos + n_to_add] = leaves_batch[pending_idxs]
    self.folds[pos : pos + n_to_add] = folds_batch[pending_idxs]
    self.scores[pos : pos + n_to_add] = scores_batch[pending_idxs]
    self.backtracks[pos : pos + n_to_add] = backtracks_batch[pending_idxs]
    self.valid_size += n_to_add
    return effective_threshold


@njit(cache=True, fastmath=True)
def prune_on_overload_func(
    self: WorldTree,
    scores_batch: np.ndarray,
    current_threshold: float,
) -> float:
    num_to_add = len(scores_batch)
    if num_to_add == 0:
        return current_threshold
    # Compute threshold from Top-K and Median
    total_candidates = self.valid_size + num_to_add
    scratch_scores = np.empty(total_candidates, dtype=self.scores.dtype)
    scratch_scores[: self.valid_size] = self.scores[: self.valid_size]
    scratch_scores[self.valid_size :] = scores_batch
    k_idx = -self.size
    mid_idx = -(total_candidates // 2) - 1
    part = np.partition(scratch_scores, [k_idx, mid_idx])
    kth_val = part[k_idx]
    mid_val = part[mid_idx]
    topk_threshold = math.nextafter(float(kth_val), float("inf"))
    median_threshold = math.nextafter(float(mid_val), float("inf"))
    effective_threshold = max(current_threshold, topk_threshold, median_threshold)
    current_scores = self.scores[: self.valid_size]
    idx = current_scores >= effective_threshold
    self._keep(idx)
    return effective_threshold


@njit(cache=True, fastmath=True)
def trim_empty_func(self: WorldTree) -> WorldTree:
    return WorldTree(
        self.leaves[: self.valid_size],
        self.folds[: self.valid_size],
        self.scores[: self.valid_size],
        self.backtracks[: self.valid_size],
    )


@njit(cache=True, fastmath=True)
def trim_empty_func_complex(self: WorldTreeComplex) -> WorldTreeComplex:
    return WorldTreeComplex(
        self.leaves[: self.valid_size],
        self.folds[: self.valid_size],
        self.scores[: self.valid_size],
        self.backtracks[: self.valid_size],
    )


@njit(cache=True, fastmath=True)
def trim_repeats_func(self: WorldTree) -> None:
    idx = get_unique_indices_scores(
        self.leaves[: self.valid_size, : self.nparams, 0],
        self.scores[: self.valid_size],
    )
    idx_bool = np.zeros(self.valid_size, dtype=np.bool_)
    idx_bool[idx] = True
    self._keep(idx_bool)


@njit(cache=True, fastmath=True)
def trim_repeats_threshold_func(self: WorldTree) -> float:
    threshold = np.median(self.scores[: self.valid_size])
    idx = get_unique_indices_scores(
        self.leaves[: self.valid_size, : self.nparams, 0],
        self.scores[: self.valid_size],
    )
    idx_bool = np.zeros(self.valid_size, dtype=np.bool_)
    idx_bool[idx] = True
    idx_threshold = self.scores[: self.valid_size] >= threshold
    idx_bool = np.logical_and(idx_bool, idx_threshold)
    self._keep(idx_bool)
    return threshold


@njit(cache=True, fastmath=True)
def keep_func(self: WorldTree, indices: np.ndarray) -> None:
    count = int(np.sum(indices))
    if count == 0:
        self.valid_size = 0
        return
    idx_valid = np.where(indices)[0]
    # move valid data to the front
    self.leaves[:count] = self.leaves[idx_valid]
    self.folds[:count] = self.folds[idx_valid]
    self.scores[:count] = self.scores[idx_valid]
    self.backtracks[:count] = self.backtracks[idx_valid]
    # update valid size
    self.valid_size = count


@overload(WorldTree)
def overload_world_tree_construct(
    leaves: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    backtracks: np.ndarray,
) -> types.FunctionType:
    def impl(
        leaves: np.ndarray,
        folds: np.ndarray,
        scores: np.ndarray,
        backtracks: np.ndarray,
    ) -> WorldTree:
        return world_tree_init(leaves, folds, scores, backtracks)

    return impl


@overload_method(WorldTreeTemplate, "get_new")
def ol_get_new_func(self: WorldTree, max_sugg: int) -> types.FunctionType:
    def impl(self: WorldTree, max_sugg: int) -> WorldTree:
        return get_new_func(self, max_sugg)

    return impl


@overload_method(WorldTreeTemplate, "get_best")
def ol_get_best_func(self: WorldTree) -> types.FunctionType:
    def impl(self: WorldTree) -> tuple[np.ndarray, np.ndarray, float]:
        return get_best_func(self)

    return impl


@overload_method(WorldTreeTemplate, "add")
def ol_add_func(
    self: WorldTree,
    leaf: np.ndarray,
    fold: np.ndarray,
    score: float,
    backtrack: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: WorldTree,
        leaf: np.ndarray,
        fold: np.ndarray,
        score: float,
        backtrack: np.ndarray,
    ) -> bool:
        return add_func(self, leaf, fold, score, backtrack)

    return impl


@overload_method(WorldTreeTemplate, "add_batch")
def ol_add_batch_func(
    self: WorldTree,
    leaves_batch: np.ndarray,
    folds_batch: np.ndarray,
    scores_batch: np.ndarray,
    backtracks_batch: np.ndarray,
    current_threshold: float,
) -> types.FunctionType:
    def impl(
        self: WorldTree,
        leaves_batch: np.ndarray,
        folds_batch: np.ndarray,
        scores_batch: np.ndarray,
        backtracks_batch: np.ndarray,
        current_threshold: float,
    ) -> float:
        return add_batch_func(
            self,
            leaves_batch,
            folds_batch,
            scores_batch,
            backtracks_batch,
            current_threshold,
        )

    return impl


@overload_method(WorldTreeTemplate, "prune_on_overload")
def ol_prune_on_overload_func(
    self: WorldTree,
    scores_batch: np.ndarray,
    current_threshold: float,
) -> types.FunctionType:
    def impl(
        self: WorldTree,
        scores_batch: np.ndarray,
        current_threshold: float,
    ) -> float:
        return prune_on_overload_func(self, scores_batch, current_threshold)

    return impl


@overload_method(WorldTreeTemplate, "trim_empty")
def ol_trim_empty_func(self: WorldTree) -> types.FunctionType:
    def impl(self: WorldTree) -> WorldTree:
        return trim_empty_func(self)

    return impl


@overload_method(WorldTreeTemplate, "trim_repeats")
def ol_trim_repeats_func(self: WorldTree) -> types.FunctionType:
    def impl(self: WorldTree) -> None:
        return trim_repeats_func(self)

    return impl


@overload_method(WorldTreeTemplate, "trim_repeats_threshold")
def ol_trim_repeats_threshold_func(self: WorldTree) -> types.FunctionType:
    def impl(self: WorldTree) -> float:
        return trim_repeats_threshold_func(self)

    return impl


@overload_method(WorldTreeTemplate, "_keep")
def ol_keep_func(self: WorldTree, indices: np.ndarray) -> types.FunctionType:
    def impl(self: WorldTree, indices: np.ndarray) -> None:
        return keep_func(self, indices)

    return impl


@overload(WorldTreeComplex)
def overload_world_tree_construct_complex(
    leaves: np.ndarray,
    folds: np.ndarray,
    scores: np.ndarray,
    backtracks: np.ndarray,
) -> types.FunctionType:
    def impl(
        leaves: np.ndarray,
        folds: np.ndarray,
        scores: np.ndarray,
        backtracks: np.ndarray,
    ) -> WorldTreeComplex:
        return world_tree_complex_init(leaves, folds, scores, backtracks)

    return impl


@overload_method(WorldTreeComplexTemplate, "get_new")
def ol_get_new_func_complex(
    self: WorldTreeComplex,
    max_sugg: int,
) -> types.FunctionType:
    def impl(self: WorldTreeComplex, max_sugg: int) -> WorldTreeComplex:
        return get_new_func_complex(self, max_sugg)

    return impl


@overload_method(WorldTreeComplexTemplate, "get_best")
def ol_get_best_func_complex(
    self: WorldTreeComplex,
) -> types.FunctionType:
    def impl(self: WorldTreeComplex) -> tuple[np.ndarray, np.ndarray, float]:
        return get_best_func(self)

    return impl


@overload_method(WorldTreeComplexTemplate, "add")
def ol_add_func_complex(
    self: WorldTreeComplex,
    leaf: np.ndarray,
    fold: np.ndarray,
    score: float,
    backtrack: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: WorldTreeComplex,
        leaf: np.ndarray,
        fold: np.ndarray,
        score: float,
        backtrack: np.ndarray,
    ) -> bool:
        return add_func(self, leaf, fold, score, backtrack)

    return impl


@overload_method(WorldTreeComplexTemplate, "add_batch")
def ol_add_batch_func_complex(
    self: WorldTreeComplex,
    leaves_batch: np.ndarray,
    folds_batch: np.ndarray,
    scores_batch: np.ndarray,
    backtracks_batch: np.ndarray,
    current_threshold: float,
) -> types.FunctionType:
    def impl(
        self: WorldTreeComplex,
        leaves_batch: np.ndarray,
        folds_batch: np.ndarray,
        scores_batch: np.ndarray,
        backtracks_batch: np.ndarray,
        current_threshold: float,
    ) -> float:
        return add_batch_func(
            self,
            leaves_batch,
            folds_batch,
            scores_batch,
            backtracks_batch,
            current_threshold,
        )

    return impl


@overload_method(WorldTreeComplexTemplate, "prune_on_overload")
def ol_prune_on_overload_func_complex(
    self: WorldTreeComplex,
    scores_batch: np.ndarray,
    current_threshold: float,
) -> types.FunctionType:
    def impl(
        self: WorldTreeComplex,
        scores_batch: np.ndarray,
        current_threshold: float,
    ) -> float:
        return prune_on_overload_func(self, scores_batch, current_threshold)

    return impl


@overload_method(WorldTreeComplexTemplate, "trim_empty")
def ol_trim_empty_func_complex(
    self: WorldTreeComplex,
) -> types.FunctionType:
    def impl(self: WorldTreeComplex) -> WorldTreeComplex:
        return trim_empty_func_complex(self)

    return impl


@overload_method(WorldTreeComplexTemplate, "trim_repeats")
def ol_trim_repeats_func_complex(
    self: WorldTreeComplex,
) -> types.FunctionType:
    def impl(self: WorldTreeComplex) -> None:
        return trim_repeats_func(self)

    return impl


@overload_method(WorldTreeComplexTemplate, "trim_repeats_threshold")
def ol_trim_repeats_threshold_func_complex(
    self: WorldTreeComplex,
) -> types.FunctionType:
    def impl(self: WorldTreeComplex) -> float:
        return trim_repeats_threshold_func(self)

    return impl


@overload_method(WorldTreeComplexTemplate, "_keep")
def ol_keep_func_complex(
    self: WorldTreeComplex,
    indices: np.ndarray,
) -> types.FunctionType:
    def impl(self: WorldTreeComplex, indices: np.ndarray) -> None:
        return keep_func(self, indices)

    return impl


@njit(cache=True, fastmath=True)
def get_unique_indices(leaves: np.ndarray) -> np.ndarray:
    nparams = leaves.shape[0]
    unique_dict = {}
    unique_indices = np.empty(nparams, dtype=np.int64)
    count = 0
    for ii in range(nparams):
        key = int(np.round(leaves[ii][-1:, 0][0] * 10**9))
        if key not in unique_dict:
            unique_dict[key] = True
            unique_indices[count] = ii
            count += 1

    return unique_indices[:count]


@njit(cache=True, fastmath=True)
def get_unique_indices_scores(leaves: np.ndarray, scores: np.ndarray) -> np.ndarray:
    nparams = leaves.shape[0]
    unique_dict: dict[int, bool] = {}
    scores_dict: dict[int, float] = {}
    count_dict: dict[int, int] = {}
    unique_indices = np.empty(nparams, dtype=np.int64)
    count = 0
    for ii in range(nparams):
        key = int(np.sum(leaves[ii][-2:, 0] * 10**9) + 0.5)
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
