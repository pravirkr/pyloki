# ruff: noqa: ARG001

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from numba import njit, typed, types
from numba.experimental import structref
from numba.extending import overload_method

from pyloki.core import common, taylor, taylor_fixed
from pyloki.detection import scoring

if TYPE_CHECKING:
    import numpy as np

    from pyloki.config import PulsarSearchConfig
    from pyloki.utils.suggestion import SuggestionStruct, SuggestionStructComplex


@structref.register
class PruneTaylorFixedDPFunctsTemplate(types.StructRef):
    pass


@structref.register
class PruneTaylorFixedComplexDPFunctsTemplate(types.StructRef):
    pass


class PruneTaylorFixedDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        param_arr: list[np.ndarray],
        dparams: np.ndarray,
        tseg_ffa: float,
        cfg: PulsarSearchConfig,
    ) -> Self:
        """Create a new instance of PruneTaylorFixedDPFuncts."""
        return prune_taylor_dp_functs_init(
            param_arr,
            dparams,
            tseg_ffa,
            cfg.nbins,
            cfg.tol_bins,
            cfg.param_limits,
            cfg.p_orb_min,
            cfg.bseg_brute,
            cfg.score_widths,
            cfg.prune_poly_order,
            cfg.branch_max,
            cfg.snap_threshold,
            cfg.use_conservative_grid,
        )

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    def resolve(
        self,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    def branch(
        self,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
        coord_cur_fixed: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev, coord_cur_fixed)

    def suggest(
        self,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStruct:
        return suggest_func(self, fold_segment, coord_init)

    def score(self, combined_res_batch: np.ndarray) -> np.ndarray:
        return score_func(self, combined_res_batch)

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
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    def get_transform_matrix(
        self,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    def validate(
        self,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

    def get_validation_params(
        self,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)


class PruneTaylorFixedComplexDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        param_arr: list[np.ndarray],
        dparams: np.ndarray,
        tseg_ffa: float,
        cfg: PulsarSearchConfig,
    ) -> Self:
        """Create a new instance of PruneTaylorFixedComplexDPFuncts."""
        return prune_taylor_complex_dp_functs_init(
            param_arr,
            dparams,
            tseg_ffa,
            cfg.nbins,
            cfg.tol_bins,
            cfg.param_limits,
            cfg.p_orb_min,
            cfg.bseg_brute,
            cfg.score_widths,
            cfg.prune_poly_order,
            cfg.branch_max,
            cfg.snap_threshold,
            cfg.use_conservative_grid,
        )

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    def resolve(
        self,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    def branch(
        self,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
        coord_cur_fixed: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev, coord_cur_fixed)

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
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    def get_transform_matrix(
        self,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    def validate(
        self,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

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
    ("p_orb_min", types.f8),
    ("bseg_brute", types.int64),
    ("score_widths", types.i8[::1]),
    ("poly_order", types.i8),
    ("branch_max", types.i8),
    ("snap_threshold", types.f8),
    ("grid_conservative", types.bool_),
]

structref.define_boxing(PruneTaylorFixedDPFunctsTemplate, PruneTaylorFixedDPFuncts)
PruneTaylorFixedDPFunctsType = PruneTaylorFixedDPFunctsTemplate(
    fields_prune_taylor_dp_funcs,
)

structref.define_boxing(
    PruneTaylorFixedComplexDPFunctsTemplate,
    PruneTaylorFixedComplexDPFuncts,
)
PruneTaylorFixedComplexDPFunctsType = PruneTaylorFixedComplexDPFunctsTemplate(
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
    p_orb_min: float,
    bseg_brute: int,
    score_widths: np.ndarray,
    poly_order: int,
    branch_max: int,
    snap_threshold: float,
    grid_conservative: bool,
) -> PruneTaylorFixedDPFuncts:
    """Initialize the PruneTaylorFixedDPFuncts object."""
    self = structref.new(PruneTaylorFixedDPFunctsType)
    self.param_arr = typed.List(param_arr)
    self.dparams = dparams
    self.tseg_ffa = tseg_ffa
    self.nbins = nbins
    self.tol_bins = tol_bins
    self.param_limits = typed.List(param_limits)
    self.p_orb_min = p_orb_min
    self.bseg_brute = bseg_brute
    self.score_widths = score_widths
    self.poly_order = poly_order
    self.branch_max = branch_max
    self.snap_threshold = snap_threshold
    self.grid_conservative = grid_conservative
    return self


@njit(cache=True, fastmath=True)
def prune_taylor_complex_dp_functs_init(
    param_arr: list[np.ndarray],
    dparams: np.ndarray,
    tseg_ffa: float,
    nbins: int,
    tol_bins: float,
    param_limits: list[tuple[float, float]],
    p_orb_min: float,
    bseg_brute: int,
    score_widths: np.ndarray,
    poly_order: int,
    branch_max: int,
    snap_threshold: float,
    grid_conservative: bool,
) -> PruneTaylorFixedComplexDPFuncts:
    """Initialize the PruneTaylorFixedComplexDPFuncts object."""
    self = structref.new(PruneTaylorFixedComplexDPFunctsType)
    self.param_arr = typed.List(param_arr)
    self.dparams = dparams
    self.tseg_ffa = tseg_ffa
    self.nbins = nbins
    self.tol_bins = tol_bins
    self.param_limits = typed.List(param_limits)
    self.p_orb_min = p_orb_min
    self.bseg_brute = bseg_brute
    self.score_widths = score_widths
    self.poly_order = poly_order
    self.branch_max = branch_max
    self.snap_threshold = snap_threshold
    self.grid_conservative = grid_conservative
    return self


@njit(cache=True, fastmath=True)
def load_func(
    self: PruneTaylorFixedDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> np.ndarray:
    return fold[seg_idx]


@njit(cache=True, fastmath=True)
def resolve_func(
    self: PruneTaylorFixedDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if self.poly_order == 4:
        return taylor_fixed.poly_taylor_fixed_resolve_circular_batch(
            leaves_batch,
            coord_add,
            coord_cur,
            coord_init,
            self.param_arr,
            self.nbins,
        )
    return taylor_fixed.poly_taylor_fixed_resolve_batch(
        leaves_batch,
        coord_add,
        coord_cur,
        coord_init,
        self.param_arr,
        self.nbins,
    )


@njit(cache=True, fastmath=True)
def branch_func(
    self: PruneTaylorFixedDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
    coord_cur_fixed: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return taylor_fixed.poly_taylor_fixed_branch_batch(
        leaves_batch,
        coord_cur_fixed,
        self.nbins,
        self.tol_bins,
        self.poly_order,
        self.param_limits,
        self.branch_max,
    )


@njit(cache=True, fastmath=True)
def suggest_func(
    self: PruneTaylorFixedDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> SuggestionStruct:
    return taylor_fixed.poly_taylor_fixed_suggest(
        fold_segment,
        coord_init,
        self.param_arr,
        self.dparams,
        self.poly_order,
        self.score_widths,
    )


@njit(cache=True, fastmath=True)
def suggest_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> SuggestionStructComplex:
    return taylor_fixed.poly_taylor_fixed_suggest_complex(
        fold_segment,
        coord_init,
        self.param_arr,
        self.dparams,
        self.poly_order,
        self.score_widths,
    )


@njit(cache=True, fastmath=True)
def score_func(
    self: PruneTaylorFixedDPFuncts,
    combined_res_batch: np.ndarray,
) -> np.ndarray:
    return scoring.snr_score_batch_func(combined_res_batch, self.score_widths)


@njit(cache=True, fastmath=True)
def score_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    combined_res_batch: np.ndarray,
) -> np.ndarray:
    return scoring.snr_score_batch_func_complex(combined_res_batch, self.score_widths)


@njit(cache=True, fastmath=True)
def pack_func(self: PruneTaylorFixedDPFuncts, data: np.ndarray) -> np.ndarray:
    return common.pack(data)


@njit(cache=True, fastmath=True)
def shift_add_func(
    self: PruneTaylorFixedDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    return common.shift_add_batch(segment_batch, shift_batch, folds, isuggest_batch)


@njit(cache=True, fastmath=True)
def shift_add_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    return common.shift_add_complex_batch(
        segment_batch,
        shift_batch,
        folds,
        isuggest_batch,
    )


@njit(cache=True, fastmath=True)
def transform_func(
    self: PruneTaylorFixedDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> np.ndarray:
    return leaves_batch


@njit(cache=True, fastmath=True)
def get_transform_matrix_func(
    self: PruneTaylorFixedDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> np.ndarray:
    return common.get_trans_matrix(coord_next, coord_prev)


@njit(cache=True, fastmath=True)
def validate_func(
    self: PruneTaylorFixedDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if self.poly_order == 4:
        return taylor.poly_taylor_validate_batch(
            leaves_batch,
            leaves_origins,
            self.p_orb_min,
            self.snap_threshold,
        )
    return leaves_batch, leaves_origins


@njit(cache=True, fastmath=True)
def get_validation_params_func(
    self: PruneTaylorFixedDPFuncts,
    coord_add: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    return common.get_validation_params(coord_add)


@overload_method(PruneTaylorFixedDPFunctsTemplate, "load")
def ol_load_func(
    self: PruneTaylorFixedDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedDPFuncts,
        fold: np.ndarray,
        seg_idx: int,
    ) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    return impl


@overload_method(PruneTaylorFixedDPFunctsTemplate, "resolve")
def ol_resolve_func(
    self: PruneTaylorFixedDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedDPFuncts,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    return impl


@overload_method(PruneTaylorFixedDPFunctsTemplate, "branch")
def ol_branch_func(
    self: PruneTaylorFixedDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
    coord_cur_fixed: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedDPFuncts,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
        coord_cur_fixed: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev, coord_cur_fixed)

    return impl


@overload_method(PruneTaylorFixedDPFunctsTemplate, "suggest")
def ol_suggest_func(
    self: PruneTaylorFixedDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedDPFuncts,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStruct:
        return suggest_func(self, fold_segment, coord_init)

    return impl


@overload_method(PruneTaylorFixedDPFunctsTemplate, "score")
def ol_score_func(
    self: PruneTaylorFixedDPFuncts,
    combined_res_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedDPFuncts,
        combined_res_batch: np.ndarray,
    ) -> np.ndarray:
        return score_func(self, combined_res_batch)

    return impl


@overload_method(PruneTaylorFixedDPFunctsTemplate, "pack")
def ol_pack_func(
    self: PruneTaylorFixedDPFuncts,
    data: np.ndarray,
) -> types.FunctionType:
    def impl(self: PruneTaylorFixedDPFuncts, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    return impl


@overload_method(PruneTaylorFixedDPFunctsTemplate, "shift_add")
def ol_shift_add_func(
    self: PruneTaylorFixedDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedDPFuncts,
        segment_batch: np.ndarray,
        shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return shift_add_func(self, segment_batch, shift_batch, folds, isuggest_batch)

    return impl


@overload_method(PruneTaylorFixedDPFunctsTemplate, "transform")
def ol_transform_func(
    self: PruneTaylorFixedDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedDPFuncts,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    return impl


@overload_method(PruneTaylorFixedDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_func(
    self: PruneTaylorFixedDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedDPFuncts,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    return impl


@overload_method(PruneTaylorFixedDPFunctsTemplate, "validate")
def ol_validate_func(
    self: PruneTaylorFixedDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedDPFuncts,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

    return impl


@overload_method(PruneTaylorFixedDPFunctsTemplate, "get_validation_params")
def ol_get_validation_params_func(
    self: PruneTaylorFixedDPFuncts,
    coord_add: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedDPFuncts,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    return impl


@overload_method(PruneTaylorFixedComplexDPFunctsTemplate, "load")
def ol_load_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedComplexDPFuncts,
        fold: np.ndarray,
        seg_idx: int,
    ) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    return impl


@overload_method(PruneTaylorFixedComplexDPFunctsTemplate, "resolve")
def ol_resolve_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    return impl


@overload_method(PruneTaylorFixedComplexDPFunctsTemplate, "branch")
def ol_branch_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
    coord_cur_fixed: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
        coord_cur_fixed: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev, coord_cur_fixed)

    return impl


@overload_method(PruneTaylorFixedComplexDPFunctsTemplate, "suggest")
def ol_suggest_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedComplexDPFuncts,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStructComplex:
        return suggest_complex_func(self, fold_segment, coord_init)

    return impl


@overload_method(PruneTaylorFixedComplexDPFunctsTemplate, "score")
def ol_score_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    combined_res_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedComplexDPFuncts,
        combined_res_batch: np.ndarray,
    ) -> np.ndarray:
        return score_complex_func(self, combined_res_batch)

    return impl


@overload_method(PruneTaylorFixedComplexDPFunctsTemplate, "pack")
def ol_pack_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    data: np.ndarray,
) -> types.FunctionType:
    def impl(self: PruneTaylorFixedComplexDPFuncts, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    return impl


@overload_method(PruneTaylorFixedComplexDPFunctsTemplate, "shift_add")
def ol_shift_add_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedComplexDPFuncts,
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


@overload_method(PruneTaylorFixedComplexDPFunctsTemplate, "transform")
def ol_transform_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    return impl


@overload_method(PruneTaylorFixedComplexDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedComplexDPFuncts,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    return impl


@overload_method(PruneTaylorFixedComplexDPFunctsTemplate, "validate")
def ol_validate_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedComplexDPFuncts,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

    return impl


@overload_method(PruneTaylorFixedComplexDPFunctsTemplate, "get_validation_params")
def ol_get_validation_params_complex_func(
    self: PruneTaylorFixedComplexDPFuncts,
    coord_add: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorFixedComplexDPFuncts,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    return impl
