# ruff: noqa: ARG001

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from numba import njit, typed, types
from numba.experimental import structref
from numba.extending import overload_method

from pyloki.core import circular, dynamic_taylor

if TYPE_CHECKING:
    import numpy as np

    from pyloki.config import PulsarSearchConfig
    from pyloki.utils.suggestion import SuggestionStruct


@structref.register
class PruneCircularDPFunctsTemplate(types.StructRef):
    pass


class PruneCircularDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        cfg: PulsarSearchConfig,
        param_arr: list[np.ndarray],
        dparams: np.ndarray,
        tseg_ffa: float,
        poly_order: int = 3,
        branch_max: int = 16,
    ) -> Self:
        """Create a new instance of PruneCircularDPFuncts."""
        return prune_circular_dp_functs_init(
            cfg.nbins,
            cfg.tol_bins,
            cfg.param_limits,
            cfg.bseg_brute,
            cfg.score_widths,
            param_arr,
            dparams,
            tseg_ffa,
            poly_order,
            branch_max,
        )

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return dynamic_taylor.load_func(self, fold, seg_idx)

    def resolve_batch(
        self,
        leaf_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_batch_func(self, leaf_batch, coord_add, coord_init)

    def branch_batch(
        self,
        param_set_batch: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_batch_func(self, param_set_batch, coord_cur)

    def suggest(
        self,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStruct:
        return suggest_func(self, fold_segment, coord_init)

    def score_batch(self, batch_combined_res: np.ndarray) -> np.ndarray:
        return dynamic_taylor.score_batch_func(self, batch_combined_res)

    def shift_add_batch(
        self,
        segment_batch: np.ndarray,
        phase_shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return dynamic_taylor.shift_add_batch_func(
            self,
            segment_batch,
            phase_shift_batch,
            folds,
            isuggest_batch,
        )

    def transform(
        self,
        leaf: np.ndarray,
        coord_cur: tuple[float, float],
        trans_matrix: np.ndarray,
    ) -> np.ndarray:
        return dynamic_taylor.transform_func(self, leaf, coord_cur, trans_matrix)

    def get_transform_matrix(
        self,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return dynamic_taylor.get_transform_matrix_func(self, coord_cur, coord_prev)

    def validate(
        self,
        leaves: np.ndarray,
        coord_valid: tuple[float, float],
        validation_params: tuple[np.ndarray, np.ndarray, float],
    ) -> np.ndarray:
        return dynamic_taylor.validate_func(
            self,
            leaves,
            coord_valid,
            validation_params,
        )

    def get_validation_params(
        self,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return dynamic_taylor.get_validation_params_func(self, coord_add)


fields_prune_circular_dp_funcs = [
    ("nbins", types.int64),
    ("tol_bins", types.f8),
    ("param_limits", types.ListType(types.Tuple([types.f8, types.f8]))),
    ("bseg_brute", types.int64),
    ("score_widths", types.i8[::1]),
    ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
    ("dparams", types.f8[:]),
    ("tseg_ffa", types.f8),
    ("poly_order", types.i8),
    ("branch_max", types.i8),
]

structref.define_boxing(PruneCircularDPFunctsTemplate, PruneCircularDPFuncts)
PruneCircularDPFunctsType = PruneCircularDPFunctsTemplate(
    fields_prune_circular_dp_funcs,
)


@njit(cache=True, fastmath=True)
def prune_circular_dp_functs_init(
    nbins: int,
    tol_bins: float,
    param_limits: list[tuple[float, float]],
    bseg_brute: int,
    score_widths: np.ndarray,
    param_arr: list[np.ndarray],
    dparams: np.ndarray,
    tseg_ffa: float,
    poly_order: int,
    branch_max: int,
) -> PruneCircularDPFuncts:
    self = structref.new(PruneCircularDPFunctsType)
    self.nbins = nbins
    self.tol_bins = tol_bins
    self.param_limits = typed.List(param_limits)
    self.bseg_brute = bseg_brute
    self.score_widths = score_widths
    self.param_arr = typed.List(param_arr)
    self.dparams = dparams
    self.tseg_ffa = tseg_ffa
    self.poly_order = poly_order
    self.branch_max = branch_max
    return self


@njit(cache=True, fastmath=True)
def resolve_batch_func(
    self: PruneCircularDPFuncts,
    leaf_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return circular.poly_circular_resolve_batch(
        leaf_batch,
        coord_add,
        coord_init,
        self.param_arr,
        self.nbins,
    )


@njit(cache=True, fastmath=True)
def branch_batch_func(
    self: PruneCircularDPFuncts,
    param_set_batch: np.ndarray,
    coord_cur: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return circular.poly_circular_branch_batch(
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
    self: PruneCircularDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> np.ndarray:
    return fold_segment


@overload_method(PruneCircularDPFunctsTemplate, "load")
def ol_load_func(
    self: PruneCircularDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> types.FunctionType:
    def impl(self: PruneCircularDPFuncts, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return dynamic_taylor.load_func(self, fold, seg_idx)

    return impl


@overload_method(PruneCircularDPFunctsTemplate, "resolve_batch")
def ol_resolve_batch_func(
    self: PruneCircularDPFuncts,
    leaf_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircularDPFuncts,
        leaf_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return dynamic_taylor.resolve_batch_func(
            self,
            leaf_batch,
            coord_add,
            coord_init,
        )

    return impl


@overload_method(PruneCircularDPFunctsTemplate, "branch_batch")
def ol_branch_batch_func(
    self: PruneCircularDPFuncts,
    param_set_batch: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircularDPFuncts,
        param_set_batch: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return dynamic_taylor.branch_batch_func(self, param_set_batch, coord_cur)

    return impl


@overload_method(PruneCircularDPFunctsTemplate, "suggest")
def ol_suggest_func(
    self: PruneCircularDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircularDPFuncts,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStruct:
        return suggest_func(self, fold_segment, coord_init)

    return impl


@overload_method(PruneCircularDPFunctsTemplate, "score_batch")
def ol_score_batch_func(
    self: PruneCircularDPFuncts,
    combined_res_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneCircularDPFuncts,
        combined_res_batch: np.ndarray,
    ) -> np.ndarray:
        return dynamic_taylor.score_batch_func(self, combined_res_batch)

    return impl


@overload_method(PruneCircularDPFunctsTemplate, "shift_add_batch")
def ol_shift_add_batch_func(
    self: PruneCircularDPFuncts,
    segment_batch: np.ndarray,
    phase_shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneCircularDPFuncts,
        segment_batch: np.ndarray,
        phase_shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return dynamic_taylor.shift_add_batch_func(
            self,
            segment_batch,
            phase_shift_batch,
            folds,
            isuggest_batch,
        )

    return impl


@overload_method(PruneCircularDPFunctsTemplate, "transform")
def ol_transform_func(
    self: PruneCircularDPFuncts,
    leaf: np.ndarray,
    coord_cur: tuple[float, float],
    trans_matrix: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneCircularDPFuncts,
        leaf: np.ndarray,
        coord_cur: tuple[float, float],
        trans_matrix: np.ndarray,
    ) -> np.ndarray:
        return dynamic_taylor.transform_func(self, leaf, coord_cur, trans_matrix)

    return impl


@overload_method(PruneCircularDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_func(
    self: PruneCircularDPFuncts,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircularDPFuncts,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return dynamic_taylor.get_transform_matrix_func(self, coord_cur, coord_prev)

    return impl


@overload_method(PruneCircularDPFunctsTemplate, "validate")
def ol_validate_func(
    self: PruneCircularDPFuncts,
    leaves: np.ndarray,
    coord_valid: tuple[float, float],
    validation_params: tuple[np.ndarray, np.ndarray, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircularDPFuncts,
        leaves: np.ndarray,
        coord_valid: tuple[float, float],
        validation_params: tuple[np.ndarray, np.ndarray, float],
    ) -> np.ndarray:
        return dynamic_taylor.validate_func(
            self,
            leaves,
            coord_valid,
            validation_params,
        )

    return impl


@overload_method(PruneCircularDPFunctsTemplate, "get_validation_params")
def ol_get_validation_params_func(
    self: PruneCircularDPFuncts,
    coord_add: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircularDPFuncts,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return dynamic_taylor.get_validation_params_func(self, coord_add)

    return impl
