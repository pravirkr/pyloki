# ruff: noqa: ARG001

from __future__ import annotations

from typing import TYPE_CHECKING, Self, cast

from numba import njit, typed, types
from numba.experimental import structref
from numba.extending import overload_method

from pyloki.core import circular, common, taylor
from pyloki.detection import scoring

if TYPE_CHECKING:
    import numpy as np

    from pyloki.config import PulsarSearchConfig
    from pyloki.utils.suggestion import SuggestionStruct, SuggestionStructComplex


@structref.register
class PruneCircTaylorDPFunctsTemplate(types.StructRef):
    pass


@structref.register
class PruneCircTaylorComplexDPFunctsTemplate(types.StructRef):
    pass


class PruneCircTaylorDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        param_arr: list[np.ndarray],
        dparams: np.ndarray,
        tseg_ffa: float,
        cfg: PulsarSearchConfig,
        use_moving_grid: bool = True,
    ) -> Self:
        """Create a new instance of PruneCircTaylorDPFuncts."""
        return prune_circ_taylor_dp_functs_init(
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
            cfg.use_conservative_tile,
            cfg.p_orb_min,
            cfg.x_mass_const,
            cfg.minimum_snap_cells,
            use_moving_grid,
        )

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    def suggest(
        self,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStruct:
        return suggest_func(self, fold_segment, coord_init)

    def branch(
        self,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

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

    def resolve(
        self,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

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

    def score(self, combined_res_batch: np.ndarray) -> np.ndarray:
        return score_func(self, combined_res_batch)

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

    def pack(self, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)


class PruneCircTaylorComplexDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        param_arr: list[np.ndarray],
        dparams: np.ndarray,
        tseg_ffa: float,
        cfg: PulsarSearchConfig,
        use_moving_grid: bool = True,
    ) -> Self:
        """Create a new instance of PruneCircTaylorComplexDPFuncts."""
        return prune_circ_taylor_complex_dp_functs_init(
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
            cfg.use_conservative_tile,
            cfg.p_orb_min,
            cfg.x_mass_const,
            cfg.minimum_snap_cells,
            use_moving_grid,
        )

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    def suggest(
        self,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStructComplex:
        return suggest_complex_func(self, fold_segment, coord_init)

    def branch(
        self,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

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

    def resolve(
        self,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

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

    def score(self, batch_combined_res: np.ndarray) -> np.ndarray:
        return score_complex_func(self, batch_combined_res)

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

    def pack(self, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)


fields_prune_circ_taylor_dp_funcs = [
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
    ("use_conservative_tile", types.bool_),
    ("p_orb_min", types.f8),
    ("x_mass_const", types.f8),
    ("minimum_snap_cells", types.f8),
    ("use_moving_grid", types.bool_),
]

structref.define_boxing(PruneCircTaylorDPFunctsTemplate, PruneCircTaylorDPFuncts)
PruneCircTaylorDPFunctsType = PruneCircTaylorDPFunctsTemplate(
    fields_prune_circ_taylor_dp_funcs,
)

structref.define_boxing(
    PruneCircTaylorComplexDPFunctsTemplate,
    PruneCircTaylorComplexDPFuncts,
)
PruneCircTaylorComplexDPFunctsType = PruneCircTaylorComplexDPFunctsTemplate(
    fields_prune_circ_taylor_dp_funcs,
)


@njit(cache=True, fastmath=True)
def prune_circ_taylor_dp_functs_init(
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
    use_conservative_tile: bool,
    p_orb_min: float,
    x_mass_const: float,
    minimum_snap_cells: float,
    use_moving_grid: bool,
) -> PruneCircTaylorDPFuncts:
    """Initialize the PruneCircTaylorDPFuncts object."""
    self = cast("PruneCircTaylorDPFuncts", structref.new(PruneCircTaylorDPFunctsType))
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
    self.use_conservative_tile = use_conservative_tile
    self.p_orb_min = p_orb_min
    self.x_mass_const = x_mass_const
    self.minimum_snap_cells = minimum_snap_cells
    self.use_moving_grid = use_moving_grid
    if self.poly_order != 5:
        msg = (
            f"Poly order {self.poly_order} is not supported for circular orbit pruning"
        )
        raise ValueError(msg)
    return self


@njit(cache=True, fastmath=True)
def prune_circ_taylor_complex_dp_functs_init(
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
    use_conservative_tile: bool,
    p_orb_min: float,
    x_mass_const: float,
    minimum_snap_cells: float,
    use_moving_grid: bool,
) -> PruneCircTaylorComplexDPFuncts:
    """Initialize the PruneCircTaylorComplexDPFuncts object."""
    self = cast(
        "PruneCircTaylorComplexDPFuncts",
        structref.new(PruneCircTaylorComplexDPFunctsType),
    )
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
    self.use_conservative_tile = use_conservative_tile
    self.p_orb_min = p_orb_min
    self.x_mass_const = x_mass_const
    self.minimum_snap_cells = minimum_snap_cells
    self.use_moving_grid = use_moving_grid
    if self.poly_order != 5:
        msg = (
            f"Poly order {self.poly_order} is not supported for circular orbit pruning"
        )
        raise ValueError(msg)
    return self


@njit(cache=True, fastmath=True)
def load_func(
    self: PruneCircTaylorDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> np.ndarray:
    return fold[seg_idx]


@njit(cache=True, fastmath=True)
def suggest_func(
    self: PruneCircTaylorDPFuncts,
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
    self: PruneCircTaylorComplexDPFuncts,
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
def branch_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    # Pass coord_cur for moving grid, coord_cur_fixed for fixed grid
    return circular.circ_taylor_branch_batch(
        leaves_batch,
        coord_cur,
        self.nbins,
        self.tol_bins,
        self.poly_order,
        self.param_limits,
        self.branch_max,
    )


@njit(cache=True, fastmath=True)
def validate_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return circular.circ_taylor_validate_batch(
        leaves_batch,
        leaves_origins,
        self.p_orb_min,
        self.x_mass_const,
        self.minimum_snap_cells,
    )


@njit(cache=True, fastmath=True)
def get_validation_params_func(
    self: PruneCircTaylorDPFuncts,
    coord_add: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    return common.get_validation_params(coord_add)


@njit(cache=True, fastmath=True)
def resolve_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    if self.use_moving_grid:
        return circular.circ_taylor_resolve_batch(
            leaves_batch,
            coord_add,
            coord_cur,
            coord_init,
            self.param_arr,
            self.nbins,
            self.minimum_snap_cells,
        )
    return circular.circ_taylor_fixed_resolve_batch(
        leaves_batch,
        coord_add,
        coord_cur,
        coord_init,
        self.param_arr,
        self.nbins,
        self.minimum_snap_cells,
    )


@njit(cache=True, fastmath=True)
def shift_add_func(
    self: PruneCircTaylorDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    return common.shift_add_batch(segment_batch, shift_batch, folds, isuggest_batch)


@njit(cache=True, fastmath=True)
def shift_add_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
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
def score_func(
    self: PruneCircTaylorDPFuncts,
    combined_res_batch: np.ndarray,
) -> np.ndarray:
    return scoring.snr_score_batch_func(combined_res_batch, self.score_widths)


@njit(cache=True, fastmath=True)
def score_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    combined_res_batch: np.ndarray,
) -> np.ndarray:
    return scoring.snr_score_batch_func_complex(combined_res_batch, self.score_widths)


@njit(cache=True, fastmath=True)
def transform_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> np.ndarray:
    if self.use_moving_grid:
        return circular.circ_taylor_transform_batch(
            leaves_batch,
            coord_next,
            coord_cur,
            self.use_conservative_tile,
            self.minimum_snap_cells,
        )
    return leaves_batch


@njit(cache=True, fastmath=True)
def get_transform_matrix_func(
    self: PruneCircTaylorDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> np.ndarray:
    return common.get_trans_matrix(coord_next, coord_prev)


@njit(cache=True, fastmath=True)
def pack_func(self: PruneCircTaylorDPFuncts, data: np.ndarray) -> np.ndarray:
    return common.pack(data)


@overload_method(PruneCircTaylorDPFunctsTemplate, "load")
def ol_load_func(
    self: PruneCircTaylorDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        fold: np.ndarray,
        seg_idx: int,
    ) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorDPFunctsTemplate, "suggest")
def ol_suggest_func(
    self: PruneCircTaylorDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStruct:
        return suggest_func(self, fold_segment, coord_init)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorDPFunctsTemplate, "branch")
def ol_branch_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorDPFunctsTemplate, "validate")
def ol_validate_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorDPFunctsTemplate, "get_validation_params")
def ol_get_validation_params_func(
    self: PruneCircTaylorDPFuncts,
    coord_add: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorDPFunctsTemplate, "resolve")
def ol_resolve_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorDPFunctsTemplate, "shift_add")
def ol_shift_add_func(
    self: PruneCircTaylorDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        segment_batch: np.ndarray,
        shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return shift_add_func(self, segment_batch, shift_batch, folds, isuggest_batch)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorDPFunctsTemplate, "score")
def ol_score_func(
    self: PruneCircTaylorDPFuncts,
    combined_res_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        combined_res_batch: np.ndarray,
    ) -> np.ndarray:
        return score_func(self, combined_res_batch)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorDPFunctsTemplate, "transform")
def ol_transform_func(
    self: PruneCircTaylorDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_func(
    self: PruneCircTaylorDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorDPFuncts,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorDPFunctsTemplate, "pack")
def ol_pack_func(self: PruneCircTaylorDPFuncts, data: np.ndarray) -> types.FunctionType:
    def impl(self: PruneCircTaylorDPFuncts, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "load")
def ol_load_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        fold: np.ndarray,
        seg_idx: int,
    ) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "suggest")
def ol_suggest_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStructComplex:
        return suggest_complex_func(self, fold_segment, coord_init)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "branch")
def ol_branch_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "validate")
def ol_validate_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "get_validation_params")
def ol_get_validation_params_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    coord_add: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "resolve")
def ol_resolve_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "shift_add")
def ol_shift_add_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
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

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "score")
def ol_score_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    combined_res_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        combined_res_batch: np.ndarray,
    ) -> np.ndarray:
        return score_complex_func(self, combined_res_batch)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "transform")
def ol_transform_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneCircTaylorComplexDPFuncts,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    return cast("types.FunctionType", impl)


@overload_method(PruneCircTaylorComplexDPFunctsTemplate, "pack")
def ol_pack_complex_func(
    self: PruneCircTaylorComplexDPFuncts,
    data: np.ndarray,
) -> types.FunctionType:
    def impl(self: PruneCircTaylorComplexDPFuncts, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    return cast("types.FunctionType", impl)
