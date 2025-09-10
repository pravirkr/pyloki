# ruff: noqa: ARG001

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from numba import njit, typed, types
from numba.experimental import structref
from numba.extending import overload_method

from pyloki.core import common, taylor
from pyloki.detection import scoring

if TYPE_CHECKING:
    import numpy as np

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
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

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


class PruneTaylorComplexDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        param_arr: list[np.ndarray],
        dparams: np.ndarray,
        tseg_ffa: float,
        cfg: PulsarSearchConfig,
    ) -> Self:
        """Create a new instance of PruneTaylorComplexDPFuncts."""
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
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

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
    p_orb_min: float,
    bseg_brute: int,
    score_widths: np.ndarray,
    poly_order: int,
    branch_max: int,
    snap_threshold: float,
    grid_conservative: bool,
) -> PruneTaylorDPFuncts:
    """Initialize the PruneTaylorDPFuncts object."""
    self = structref.new(PruneTaylorDPFunctsType)
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
) -> PruneTaylorComplexDPFuncts:
    """Initialize the PruneTaylorComplexDPFuncts object."""
    self = structref.new(PruneTaylorComplexDPFunctsType)
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
def load_func(self: PruneTaylorDPFuncts, fold: np.ndarray, seg_idx: int) -> np.ndarray:
    return fold[seg_idx]


@njit(cache=True, fastmath=True)
def resolve_func(
    self: PruneTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if self.poly_order == 4:
        return taylor.poly_taylor_resolve_circular_batch(
            leaves_batch,
            coord_add,
            coord_cur,
            coord_init,
            self.param_arr,
            self.nbins,
        )
    return taylor.poly_taylor_resolve_batch(
        leaves_batch,
        coord_add,
        coord_cur,
        coord_init,
        self.param_arr,
        self.nbins,
    )


@njit(cache=True, fastmath=True)
def branch_func(
    self: PruneTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return taylor.poly_taylor_branch_batch(
        leaves_batch,
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
def score_func(self: PruneTaylorDPFuncts, combined_res_batch: np.ndarray) -> np.ndarray:
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
    return common.shift_add_complex_batch(
        segment_batch,
        shift_batch,
        folds,
        isuggest_batch,
    )


@njit(cache=True, fastmath=True)
def transform_func(
    self: PruneTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> np.ndarray:
    if self.poly_order == 4:
        return taylor.poly_taylor_transform_circular_batch(
            leaves_batch,
            coord_next,
            coord_cur,
            self.grid_conservative,
        )
    return taylor.poly_taylor_transform_batch(
        leaves_batch,
        coord_next,
        coord_cur,
        self.grid_conservative,
    )


@njit(cache=True, fastmath=True)
def get_transform_matrix_func(
    self: PruneTaylorDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> np.ndarray:
    return common.get_trans_matrix(coord_next, coord_prev)


@njit(cache=True, fastmath=True)
def validate_func(
    self: PruneTaylorDPFuncts,
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
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "branch")
def ol_branch_func(
    self: PruneTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_next, coord_prev)

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
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_func(
    self: PruneTaylorDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "validate")
def ol_validate_func(
    self: PruneTaylorDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

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
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "branch")
def ol_branch_complex_func(
    self: PruneTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_next, coord_prev)

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
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_complex_func(
    self: PruneTaylorComplexDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    return impl


@overload_method(PruneTaylorComplexDPFunctsTemplate, "validate")
def ol_validate_complex_func(
    self: PruneTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

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
