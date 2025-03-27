# ruff: noqa: ARG001

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from numba import njit, typed, types
from numba.experimental import structref
from numba.extending import overload_method

from pyloki.core import basic, defaults
from pyloki.detection import scoring

if TYPE_CHECKING:
    import numpy as np

    from pyloki.config import PulsarSearchConfig
    from pyloki.utils.suggestion import SuggestionStruct


@structref.register
class PruneTaylorDPFunctsTemplate(types.StructRef):
    pass


class PruneTaylorDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        cfg: PulsarSearchConfig,
        param_arr: list[np.ndarray],
        dparams: np.ndarray,
        tseg_ffa: float,
        poly_order: int = 3,
    ) -> Self:
        """Create a new instance of PruneTaylorDPFuncts."""
        return prune_taylor_dp_functs_init(
            cfg.nbins,
            cfg.tol_bins,
            cfg.param_limits,
            cfg.bseg_brute,
            cfg.score_widths,
            param_arr,
            dparams,
            tseg_ffa,
            poly_order,
        )

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    def resolve(
        self,
        leaf: np.ndarray,
        coord_add: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, int]:
        return resolve_func(self, leaf, coord_add, coord_init)

    def branch(
        self,
        param_set: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return branch_func(self, param_set, coord_cur)

    def suggest(
        self,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStruct:
        return suggest_func(self, fold_segment, coord_init)

    def score(self, combined_res: np.ndarray) -> float:
        return score_func(self, combined_res)

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return add_func(self, data0, data1)

    def pack(self, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return shift_func(self, data, phase_shift)

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
    ("nbins", types.int64),
    ("tol_bins", types.f8),
    ("param_limits", types.ListType(types.Tuple([types.f8, types.f8]))),
    ("bseg_brute", types.int64),
    ("score_widths", types.i8[::1]),
    ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
    ("dparams", types.f8[:]),
    ("tseg_ffa", types.f8),
    ("poly_order", types.i8),
]

structref.define_boxing(PruneTaylorDPFunctsTemplate, PruneTaylorDPFuncts)
PruneTaylorDPFunctsType = PruneTaylorDPFunctsTemplate(fields_prune_taylor_dp_funcs)


@njit(cache=True, fastmath=True)
def prune_taylor_dp_functs_init(
    nbins: int,
    tol_bins: float,
    param_limits: list[tuple[float, float]],
    bseg_brute: int,
    score_widths: np.ndarray,
    param_arr: list[np.ndarray],
    dparams: np.ndarray,
    tseg_ffa: float,
    poly_order: int,
) -> PruneTaylorDPFuncts:
    self = structref.new(PruneTaylorDPFunctsType)
    self.nbins = nbins
    self.tol_bins = tol_bins
    self.param_limits = typed.List(param_limits)
    self.bseg_brute = bseg_brute
    self.score_widths = score_widths
    self.param_arr = typed.List(param_arr)
    self.dparams = dparams
    self.tseg_ffa = tseg_ffa
    self.poly_order = poly_order
    return self


@njit(cache=True, fastmath=True)
def load_func(self: PruneTaylorDPFuncts, fold: np.ndarray, seg_idx: int) -> np.ndarray:
    return fold[seg_idx]


@njit(cache=True, fastmath=True)
def resolve_func(
    self: PruneTaylorDPFuncts,
    leaf: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
) -> tuple[np.ndarray, int]:
    return basic.poly_taylor_resolve(
        leaf,
        self.param_arr,
        coord_add,
        coord_init,
        self.nbins,
    )


@njit(cache=True, fastmath=True)
def branch_func(
    self: PruneTaylorDPFuncts,
    param_set: np.ndarray,
    coord_cur: tuple[float, float],
) -> np.ndarray:
    return basic.poly_taylor_branch(
        param_set,
        coord_cur,
        self.nbins,
        self.tol_bins,
        self.poly_order,
        self.param_limits,
    )


@njit(cache=True, fastmath=True)
def suggest_func(
    self: PruneTaylorDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> SuggestionStruct:
    return basic.poly_taylor_suggest(
        fold_segment,
        coord_init,
        self.param_arr,
        self.dparams,
        self.poly_order,
        self.score_widths,
    )


@njit(cache=True, fastmath=True)
def score_func(self: PruneTaylorDPFuncts, combined_res: np.ndarray) -> float:
    return scoring.snr_score_func(combined_res, self.score_widths)


@njit(cache=True, fastmath=True)
def add_func(
    self: PruneTaylorDPFuncts,
    data0: np.ndarray,
    data1: np.ndarray,
) -> np.ndarray:
    return defaults.add(data0, data1)


@njit(cache=True, fastmath=True)
def pack_func(self: PruneTaylorDPFuncts, data: np.ndarray) -> np.ndarray:
    return defaults.pack(data)


@njit(cache=True, fastmath=True)
def shift_func(
    self: PruneTaylorDPFuncts,
    data: np.ndarray,
    phase_shift: int,
) -> np.ndarray:
    return defaults.shift(data, phase_shift)


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
    return defaults.get_trans_matrix(coord_cur, coord_prev)


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
    return defaults.get_validation_params(coord_add)


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
    leaf: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        leaf: np.ndarray,
        coord_add: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, int]:
        return resolve_func(self, leaf, coord_add, coord_init)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "branch")
def ol_branch_func(
    self: PruneTaylorDPFuncts,
    param_set: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        param_set: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return branch_func(self, param_set, coord_cur)

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
    combined_res: np.ndarray,
) -> types.FunctionType:
    def impl(self: PruneTaylorDPFuncts, combined_res: np.ndarray) -> float:
        return score_func(self, combined_res)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "add")
def ol_add_func(
    self: PruneTaylorDPFuncts,
    data0: np.ndarray,
    data1: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        data0: np.ndarray,
        data1: np.ndarray,
    ) -> np.ndarray:
        return add_func(self, data0, data1)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "pack")
def ol_pack_func(self: PruneTaylorDPFuncts, data: np.ndarray) -> types.FunctionType:
    def impl(self: PruneTaylorDPFuncts, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    return impl


@overload_method(PruneTaylorDPFunctsTemplate, "shift")
def ol_shift_func(
    self: PruneTaylorDPFuncts,
    data: np.ndarray,
    phase_shift: int,
) -> types.FunctionType:
    def impl(
        self: PruneTaylorDPFuncts,
        data: np.ndarray,
        phase_shift: int,
    ) -> np.ndarray:
        return shift_func(self, data, phase_shift)

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
