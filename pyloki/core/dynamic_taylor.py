# ruff: noqa: ARG001

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from numba import njit, typed, types
from numba.experimental import structref
from numba.extending import overload_method

from pyloki.core import common, taylor
from pyloki.detection import scoring

if TYPE_CHECKING:
    from pyloki.config import PulsarSearchConfig
    from pyloki.utils.suggestion import SuggestionStruct, SuggestionStructComplex


@structref.register
class PruneTaylorDPFunctsTemplate(types.StructRef):
    pass


@structref.register
class PruneTaylorComplexDPFunctsTemplate(types.StructRef):
    pass


class PruneTaylorDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        param_arr: list[np.ndarray],
        dparams: np.ndarray,
        tseg_ffa: float,
        cfg: PulsarSearchConfig,
    ) -> Self:
        """Create a new instance of PruneTaylorDPFuncts."""
        return prune_taylor_dp_functs_init(
            param_arr,
            dparams,
            tseg_ffa,
            cfg.nbins,
            cfg.tol_bins,
            cfg.param_limits,
            cfg.bseg_brute,
            cfg.score_widths,
            cfg.prune_poly_order,
            cfg.branch_max,
        )

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    def resolve(
        self,
        leaf_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaf_batch, coord_add, coord_init)

    def branch(
        self,
        param_set_batch: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, param_set_batch, coord_cur)

    def suggest(
        self,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStruct:
        return suggest_func(self, fold_segment, coord_init)

    def score(self, batch_combined_res: np.ndarray) -> np.ndarray:
        return score_func(self, batch_combined_res)

    def pack(self, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    def shift_add(
        self,
        segment_batch: np.ndarray,
        shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return shift_add_func(
            self,
            segment_batch,
            shift_batch,
            folds,
            isuggest_batch,
        )

    def transform(
        self,
        leaf: np.ndarray,
        coord_cur: tuple[float, float],
        trans_matrix: np.ndarray,
    ) -> np.ndarray:
        return transform_func(self, leaf, coord_cur, trans_matrix)

    def get_transform_matrix(
        self,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_cur, coord_prev)

    def validate(
        self,
        leaves: np.ndarray,
        coord_valid: tuple[float, float],
        validation_params: tuple[np.ndarray, np.ndarray, float],
    ) -> np.ndarray:
        return validate_func(self, leaves, coord_valid, validation_params)

    def get_validation_params(
        self,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)


class PruneTaylorComplexDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        param_arr: list[np.ndarray],
        dparams: np.ndarray,
        tseg_ffa: float,
        cfg: PulsarSearchConfig,
    ) -> Self:
        """Create a new instance of PruneTaylorDPFuncts."""
        return prune_taylor_complex_dp_functs_init(
            param_arr,
            dparams,
            tseg_ffa,
            cfg.nbins,
            cfg.tol_bins,
            cfg.param_limits,
            cfg.bseg_brute,
            cfg.score_widths,
            cfg.prune_poly_order,
            cfg.branch_max,
        )

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    def resolve(
        self,
        leaf_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_complex_func(self, leaf_batch, coord_add, coord_init)

    def branch(
        self,
        param_set_batch: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, param_set_batch, coord_cur)

    def suggest(
        self,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStructComplex:
        return suggest_complex_func(self, fold_segment, coord_init)

    def score(self, batch_combined_res: np.ndarray) -> np.ndarray:
        return score_complex_func(self, batch_combined_res)

    def pack(self, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    def shift_add(
        self,
        segment_batch: np.ndarray,
        shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return shift_add_complex_func(
            self,
            segment_batch,
            shift_batch,
            folds,
            isuggest_batch,
        )

    def transform(
        self,
        leaf: np.ndarray,
        coord_cur: tuple[float, float],
        trans_matrix: np.ndarray,
    ) -> np.ndarray:
        return transform_func(self, leaf, coord_cur, trans_matrix)

    def get_transform_matrix(
        self,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_cur, coord_prev)

    def validate(
        self,
        leaves: np.ndarray,
        coord_valid: tuple[float, float],
        validation_params: tuple[np.ndarray, np.ndarray, float],
    ) -> np.ndarray:
        return validate_func(self, leaves, coord_valid, validation_params)

    def get_validation_params(
        self,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)


fields_prune_taylor_dp_funcs = [
    ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
    ("dparams", types.f8[:]),
    ("tseg_ffa", types.f8),
    ("nbins", types.int64),
    ("tol_bins", types.f8),
    ("param_limits", types.ListType(types.Tuple([types.f8, types.f8]))),
    ("bseg_brute", types.int64),
    ("score_widths", types.i8[::1]),
    ("poly_order", types.i8),
    ("branch_max", types.i8),
]

structref.define_boxing(PruneTaylorDPFunctsTemplate, PruneTaylorDPFuncts)
PruneTaylorDPFunctsType = PruneTaylorDPFunctsTemplate(fields_prune_taylor_dp_funcs)

structref.define_boxing(PruneTaylorComplexDPFunctsTemplate, PruneTaylorComplexDPFuncts)
PruneTaylorComplexDPFunctsType = PruneTaylorComplexDPFunctsTemplate(
    fields_prune_taylor_dp_funcs,
)


@njit(cache=True, fastmath=True)
def prune_taylor_dp_functs_init(
    param_arr: list[np.ndarray],
    dparams: np.ndarray,
    tseg_ffa: float,
    nbins: int,
    tol_bins: float,
    param_limits: list[tuple[float, float]],
    bseg_brute: int,
    score_widths: np.ndarray,
    poly_order: int,
    branch_max: int,
) -> PruneTaylorDPFuncts:
    """Initialize the PruneTaylorDPFuncts object."""
    self = structref.new(PruneTaylorDPFunctsType)
    self.param_arr = typed.List(param_arr)
    self.dparams = dparams
    self.tseg_ffa = tseg_ffa
    self.nbins = nbins
    self.tol_bins = tol_bins
    self.param_limits = typed.List(param_limits)
    self.bseg_brute = bseg_brute
    self.score_widths = score_widths
    self.poly_order = poly_order
    self.branch_max = branch_max
    return self


@njit(cache=True, fastmath=True)
def prune_taylor_complex_dp_functs_init(
    param_arr: list[np.ndarray],
    dparams: np.ndarray,
    tseg_ffa: float,
    nbins: int,
    tol_bins: float,
    param_limits: list[tuple[float, float]],
    bseg_brute: int,
    score_widths: np.ndarray,
    poly_order: int,
    branch_max: int,
) -> PruneTaylorComplexDPFuncts:
    """Initialize the PruneTaylorComplexDPFuncts object."""
    self = structref.new(PruneTaylorComplexDPFunctsType)
    self.param_arr = typed.List(param_arr)
    self.dparams = dparams
    self.tseg_ffa = tseg_ffa
    self.nbins = nbins
    self.tol_bins = tol_bins
    self.param_limits = typed.List(param_limits)
    self.bseg_brute = bseg_brute
    self.score_widths = score_widths
    self.poly_order = poly_order
    self.branch_max = branch_max
    return self


@njit(cache=True, fastmath=True)
def load_func(self: PruneTaylorDPFuncts, fold: np.ndarray, seg_idx: int) -> np.ndarray:
    return fold[seg_idx]


@njit(cache=True, fastmath=True)
def resolve_func(
    self: PruneTaylorDPFuncts,
    leaf_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if self.poly_order == 4:
        param_idx_batch, relative_phase_batch = taylor.poly_taylor_resolve_snap_batch(
            leaf_batch,
            coord_add,
            coord_init,
            self.param_arr,
            self.nbins,
        )
        relative_phase_batch_int = np.round(relative_phase_batch).astype(np.int32)
        return param_idx_batch, relative_phase_batch_int
    param_idx_batch, relative_phase_batch = taylor.poly_taylor_resolve_batch(
        leaf_batch,
        coord_add,
        coord_init,
        self.param_arr,
        self.nbins,
    )
    relative_phase_batch_int = np.round(relative_phase_batch).astype(np.int32)
    return param_idx_batch, relative_phase_batch_int


@njit(cache=True, fastmath=True)
def resolve_complex_func(
    self: PruneTaylorComplexDPFuncts,
    leaf_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if self.poly_order == 4:
        return taylor.poly_taylor_resolve_snap_batch(
            leaf_batch,
            coord_add,
            coord_init,
            self.param_arr,
            self.nbins,
        )
    return taylor.poly_taylor_resolve_batch(
        leaf_batch,
        coord_add,
        coord_init,
        self.param_arr,
        self.nbins,
    )


@njit(cache=True, fastmath=True)
def branch_func(
    self: PruneTaylorDPFuncts,
    param_set_batch: np.ndarray,
    coord_cur: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return taylor.poly_taylor_branch_batch(
        param_set_batch,
        coord_cur,
        self.nbins,
        self.tol_bins,
        self.poly_order,
        self.param_limits,
        self.branch_max,
    )


@njit(cache=True, fastmath=True)
def suggest_func(
    self: PruneTaylorDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> SuggestionStruct:
    return taylor.poly_taylor_suggest(
        fold_segment,
        coord_init,
        self.param_arr,
        self.dparams,
        self.poly_order,
        self.score_widths,
    )


@njit(cache=True, fastmath=True)
def suggest_complex_func(
    self: PruneTaylorComplexDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> SuggestionStructComplex:
    return taylor.poly_taylor_suggest_complex(
        fold_segment,
        coord_init,
        self.param_arr,
        self.dparams,
        self.poly_order,
        self.score_widths,
    )


@njit(cache=True, fastmath=True)
def score_func(
    self: PruneTaylorDPFuncts,
    combined_res_batch: np.ndarray,
) -> np.ndarray:
    return scoring.snr_score_batch_func(combined_res_batch, self.score_widths)


@njit(cache=True, fastmath=True)
def score_complex_func(
    self: PruneTaylorComplexDPFuncts,
    combined_res_batch: np.ndarray,
) -> np.ndarray:
    return scoring.snr_score_batch_func_complex(combined_res_batch, self.score_widths)


@njit(cache=True, fastmath=True)
def pack_func(self: PruneTaylorDPFuncts, data: np.ndarray) -> np.ndarray:
    return common.pack(data)


@njit(cache=True, fastmath=True)
def shift_add_func(
    self: PruneTaylorDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    return common.shift_add_batch(segment_batch, shift_batch, folds, isuggest_batch)


@njit(cache=True, fastmath=True)
def shift_add_complex_func(
    self: PruneTaylorComplexDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    n_batch, n_comps, n_cols = segment_batch.shape
    res = np.empty((n_batch, n_comps, n_cols), dtype=segment_batch.dtype)
    k = np.arange(n_cols)
    fold_bins = self.nbins
    for irow in range(n_batch):
        shift = shift_batch[irow]
        fold_row = folds[isuggest_batch[irow]]
        phase = np.exp(-2j * np.pi * k * shift / fold_bins)
        res[irow, 0] = (segment_batch[irow, 0] * phase) + fold_row[0]
        res[irow, 1] = (segment_batch[irow, 1] * phase) + fold_row[1]
    return res


@njit(cache=True, fastmath=True)
def transform_func(
    self: PruneTaylorDPFuncts,
    leaf: np.ndarray,
    coord_cur: tuple[float, float],
    trans_matrix: np.ndarray,
) -> np.ndarray:
    return leaf


@njit(cache=True, fastmath=True)
def get_transform_matrix_func(
    self: PruneTaylorDPFuncts,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> np.ndarray:
    return common.get_trans_matrix(coord_cur, coord_prev)


@njit(cache=True, fastmath=True)
def validate_func(
    self: PruneTaylorDPFuncts,
    leaves: np.ndarray,
    coord_valid: tuple[float, float],
    validation_params: tuple[np.ndarray, np.ndarray, float],
) -> np.ndarray:
    return leaves


@njit(cache=True, fastmath=True)
def get_validation_params_func(
    self: PruneTaylorDPFuncts,
    coord_add: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    return common.get_validation_params(coord_add)


@overload_method(PruneTaylorDPFunctsTemplate, "load")
def ol_load_func(
    self: PruneTaylorDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> types.FunctionType:
    def impl(self: PruneTaylorDPFuncts, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "resolve")
def ol_resolve_func(
    self: PruneTaylorDPFuncts,
    leaf_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        leaf_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaf_batch, coord_add, coord_init)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "branch")
def ol_branch_func(
    self: PruneTaylorDPFuncts,
    param_set_batch: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        param_set_batch: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, param_set_batch, coord_cur)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "suggest")
def ol_suggest_func(
    self: PruneTaylorDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStruct:
        return suggest_func(self, fold_segment, coord_init)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "score")
def ol_score_func(
    self: PruneTaylorDPFuncts,
    combined_res_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        combined_res_batch: np.ndarray,
    ) -> np.ndarray:
        return score_func(self, combined_res_batch)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "pack")
def ol_pack_func(self: PruneTaylorDPFuncts, data: np.ndarray) -> types.FunctionType:
    def impl(self: PruneTaylorDPFuncts, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "shift_add")
def ol_shift_add_func(
    self: PruneTaylorDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        segment_batch: np.ndarray,
        shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return shift_add_func(self, segment_batch, shift_batch, folds, isuggest_batch)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "transform")
def ol_transform_func(
    self: PruneTaylorDPFuncts,
    leaf: np.ndarray,
    coord_cur: tuple[float, float],
    trans_matrix: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        leaf: np.ndarray,
        coord_cur: tuple[float, float],
        trans_matrix: np.ndarray,
    ) -> np.ndarray:
        return transform_func(self, leaf, coord_cur, trans_matrix)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_func(
    self: PruneTaylorDPFuncts,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_cur, coord_prev)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "validate")
def ol_validate_func(
    self: PruneTaylorDPFuncts,
    leaves: np.ndarray,
    coord_valid: tuple[float, float],
    validation_params: tuple[np.ndarray, np.ndarray, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        leaves: np.ndarray,
        coord_valid: tuple[float, float],
        validation_params: tuple[np.ndarray, np.ndarray, float],
    ) -> np.ndarray:
        return validate_func(self, leaves, coord_valid, validation_params)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "get_validation_params")
def ol_get_validation_params_func(
    self: PruneTaylorDPFuncts,
    coord_add: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "load")
def ol_load_complex_func(
    self: PruneTaylorComplexDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        fold: np.ndarray,
        seg_idx: int,
    ) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "resolve")
def ol_resolve_complex_func(
    self: PruneTaylorComplexDPFuncts,
    leaf_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        leaf_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_complex_func(self, leaf_batch, coord_add, coord_init)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "branch")
def ol_branch_complex_func(
    self: PruneTaylorComplexDPFuncts,
    param_set_batch: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        param_set_batch: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, param_set_batch, coord_cur)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "suggest")
def ol_suggest_complex_func(
    self: PruneTaylorComplexDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStructComplex:
        return suggest_complex_func(self, fold_segment, coord_init)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "score")
def ol_score_complex_func(
    self: PruneTaylorComplexDPFuncts,
    combined_res_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        combined_res_batch: np.ndarray,
    ) -> np.ndarray:
        return score_complex_func(self, combined_res_batch)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "pack")
def ol_pack_complex_func(
    self: PruneTaylorComplexDPFuncts,
    data: np.ndarray,
) -> types.FunctionType:
    def impl(self: PruneTaylorComplexDPFuncts, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "shift_add")
def ol_shift_add_complex_func(
    self: PruneTaylorComplexDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        segment_batch: np.ndarray,
        shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return shift_add_complex_func(
            self,
            segment_batch,
            shift_batch,
            folds,
            isuggest_batch,
        )

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "transform")
def ol_transform_complex_func(
    self: PruneTaylorComplexDPFuncts,
    leaf: np.ndarray,
    coord_cur: tuple[float, float],
    trans_matrix: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        leaf: np.ndarray,
        coord_cur: tuple[float, float],
        trans_matrix: np.ndarray,
    ) -> np.ndarray:
        return transform_func(self, leaf, coord_cur, trans_matrix)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_complex_func(
    self: PruneTaylorComplexDPFuncts,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_cur, coord_prev)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "validate")
def ol_validate_complex_func(
    self: PruneTaylorComplexDPFuncts,
    leaves: np.ndarray,
    coord_valid: tuple[float, float],
    validation_params: tuple[np.ndarray, np.ndarray, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        leaves: np.ndarray,
        coord_valid: tuple[float, float],
        validation_params: tuple[np.ndarray, np.ndarray, float],
    ) -> np.ndarray:
        return validate_func(self, leaves, coord_valid, validation_params)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "get_validation_params")
def ol_get_validation_params_complex_func(
    self: PruneTaylorComplexDPFuncts,
    coord_add: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    return impl
