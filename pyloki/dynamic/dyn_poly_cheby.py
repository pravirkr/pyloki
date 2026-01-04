# ruff: noqa: ARG001

from __future__ import annotations

from typing import TYPE_CHECKING, Self

from numba import njit, typed, types
from numba.experimental import structref
from numba.extending import overload_method

from pyloki.core import chebyshev, common
from pyloki.detection import scoring

if TYPE_CHECKING:
    import numpy as np

    from pyloki.config import PulsarSearchConfig
    from pyloki.utils.suggestion import SuggestionStruct, SuggestionStructComplex


@structref.register
class PrunePolyChebyshevDPFunctsTemplate(types.StructRef):
    pass


@structref.register
class PrunePolyChebyshevComplexDPFunctsTemplate(types.StructRef):
    pass


class PrunePolyChebyshevDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        param_arr: types.ListType[types.Array],
        dparams: np.ndarray,
        tseg_ffa: float,
        cfg: PulsarSearchConfig,
        use_moving_grid: bool = True,
    ) -> Self:
        """Create a new instance of PrunePolyChebyshevDPFuncts."""
        return prune_chebyshev_dp_functs_init(
            param_arr,
            dparams,
            tseg_ffa,
            cfg.nbins,
            cfg.eta,
            cfg.param_limits,
            cfg.bseg_brute,
            cfg.score_widths,
            cfg.prune_poly_order,
            cfg.branch_max,
            cfg.use_conservative_grid,
            use_moving_grid,
        )

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        """Load the data for the given segment index from the folded data structure.

        Parameters
        ----------
        fold : np.ndarray
            The folded data structure to load from.
        seg_idx : int
            Segment index to load from the fold.

        Returns
        -------
        np.ndarray
            The data for the given index.

        Notes
        -----
        Future implementations may include:
        - Simply accessing the data from the fold.
        - Calibration (RFI removal + fold_e, fold_v generation) of the data structure.
        - Compute the data structure live (using dynamic programming).
        - Save the calculated data structure to prevent excessive computation.
        - Remove pulsars with known ephemeris to keep the suggestion counts low.
        - Implement it as a class, and pass its loading function here.

        """
        return load_func(self, fold, seg_idx)

    def suggest(
        self,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStruct:
        """Generate an initial suggestion struct for the starting segment.

        Parameters
        ----------
        fold_segment : np.ndarray
            The folded data segment to generate the suggestion for.
        coord_init : tuple[float, float]
            The coordinate of the starting segment.

        Returns
        -------
        common.SuggestionStruct
            The initial suggestion struct for the segment.
        """
        return suggest_func(self, fold_segment, coord_init)

    def branch(
        self,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Branch the current parameter set into the finer grid of parameters (leaves).

        Parameters
        ----------
        leaves_batch : np.ndarray
            The current parameter set to branch.
        coord_cur : tuple[float, float]
            The current coordinate.
        coord_prev : tuple[float, float]
            The previous coordinate.

        Returns
        -------
        np.ndarray
            The branched parameter set.
        """
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

    def validate(
        self,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate which of the leaves are physical.

        Parameters
        ----------
        leaves : np.ndarray
            Set of leaves (parameter sets) to validate.
        coord_cur : tuple[float, float]
            The current coordinate.

        Returns
        -------
        np.ndarray
            Boolean mask indicating which of the leaves are physical.

        Notes
        -----
        - The validation_params are pre-computed to reduce computation.
        - This function should filter out leafs with unphysical derivatives.
        - pruning scans only functions that are physical at position (t/2).
        But same bounds apply everywhere.
        """
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

    def get_validation_params(
        self,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Prepare the validation parameters for the epicyclic validation.

        Parameters
        ----------
        coord_add : float
            The Chebyshev time to prepare the validation parameters for.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, float]
            The validation parameters for the epicyclic validation.
        """
        return get_validation_params_func(self, coord_add)

    def resolve(
        self,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve the leaf parameters to find the closest param index and phase shift.

        Parameters
        ----------
        leaves_batch : np.ndarray
            Current leaf parameter set.
        coord_add : tuple[float, float]
            Time coordinate of the added segment.
        coord_cur : tuple[float, float]
            Time coordinate of the current segment.
        coord_init : tuple[float, float]
            Time coordinate of the starting segment.

        Returns
        -------
        tuple[np.ndarray, int]
            The resolved parameter index and the relative phase shift.
        """
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
        """Calculate the statistical detection score of the combined fold.

        Parameters
        ----------
        combined_res_batch : np.ndarray
            The combined fold to calculate the score for (fold_e, fold_v).

        Returns
        -------
        float
            The statistical detection score of the combined fold.

        Notes
        -----
        - Score units should be log(P(D|pulsar(theta)) / P(D|noise)).
        - Maybe use it to keep track of a family of scores (profile width, etc).
        """
        return score_func(self, combined_res_batch)

    def transform(
        self,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        """Transform the leaf parameters to the new coordinate system.

        Parameters
        ----------
        leaves_batch : np.ndarray
            Current leaf parameter set.
        coord_next : tuple[float, float]
            The new coordinate to transform the leaf to.
        coord_cur : tuple[float, float]
            The current coordinate.

        Returns
        -------
        np.ndarray
            The transformed leaf parameter set.
        """
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    def get_transform_matrix(
        self,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        """Get the transformation matrix between two coordinate systems."""
        return get_transform_matrix_func(self, coord_next, coord_prev)

    def pack(self, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)


class PrunePolyChebyshevComplexDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        param_arr: types.ListType[types.Array],
        dparams: np.ndarray,
        tseg_ffa: float,
        cfg: PulsarSearchConfig,
    ) -> Self:
        """Create a new instance of PrunePolyChebyshevComplexDPFuncts."""
        return prune_chebyshev_complex_dp_functs_init(
            param_arr,
            dparams,
            tseg_ffa,
            cfg.nbins,
            cfg.eta,
            cfg.param_limits,
            cfg.bseg_brute,
            cfg.score_widths,
            cfg.prune_poly_order,
            cfg.branch_max,
            cfg.use_conservative_grid,
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

    def score(self, combined_res_batch: np.ndarray) -> np.ndarray:
        return score_complex_func(self, combined_res_batch)

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
        """Get the transformation matrix between two coordinate systems."""
        return get_transform_matrix_func(self, coord_next, coord_prev)

    def pack(self, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)


fields_prune_chebyshev_dp_funcs = [
    ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
    ("dparams", types.f8[:]),
    ("tseg_ffa", types.f8),
    ("nbins", types.int64),
    ("eta", types.f8),
    ("param_limits", types.ListType(types.Tuple([types.f8, types.f8]))),
    ("bseg_brute", types.int64),
    ("score_widths", types.i8[::1]),
    ("poly_order", types.i8),
    ("branch_max", types.i8),
    ("use_conservative_tile", types.bool_),
    ("use_moving_grid", types.bool_),
]

structref.define_boxing(PrunePolyChebyshevDPFunctsTemplate, PrunePolyChebyshevDPFuncts)
PrunePolyChebyshevDPFunctsType = PrunePolyChebyshevDPFunctsTemplate(
    fields_prune_chebyshev_dp_funcs,
)

structref.define_boxing(
    PrunePolyChebyshevComplexDPFunctsTemplate,
    PrunePolyChebyshevComplexDPFuncts,
)
PrunePolyChebyshevComplexDPFunctsType = PrunePolyChebyshevComplexDPFunctsTemplate(
    fields_prune_chebyshev_dp_funcs,
)


@njit(cache=True, fastmath=True)
def prune_chebyshev_dp_functs_init(
    param_arr: list[np.ndarray],
    dparams: np.ndarray,
    tseg_ffa: float,
    nbins: int,
    eta: float,
    param_limits: list[tuple[float, float]],
    bseg_brute: int,
    score_widths: np.ndarray,
    poly_order: int,
    branch_max: int,
    use_conservative_tile: bool,
    use_moving_grid: bool,
) -> PrunePolyChebyshevDPFuncts:
    """Initialize the PrunePolyChebyshevDPFuncts struct."""
    self = structref.new(PrunePolyChebyshevDPFunctsType)
    self.param_arr = typed.List(param_arr)
    self.dparams = dparams
    self.tseg_ffa = tseg_ffa
    self.nbins = nbins
    self.eta = eta
    self.param_limits = typed.List(param_limits)
    self.bseg_brute = bseg_brute
    self.score_widths = score_widths
    self.poly_order = poly_order
    self.branch_max = branch_max
    self.use_conservative_tile = use_conservative_tile
    self.use_moving_grid = use_moving_grid
    return self


@njit(cache=True, fastmath=True)
def prune_chebyshev_complex_dp_functs_init(
    param_arr: list[np.ndarray],
    dparams: np.ndarray,
    tseg_ffa: float,
    nbins: int,
    eta: float,
    param_limits: list[tuple[float, float]],
    bseg_brute: int,
    score_widths: np.ndarray,
    poly_order: int,
    branch_max: int,
    use_conservative_tile: bool,
    use_moving_grid: bool,
) -> PrunePolyChebyshevComplexDPFuncts:
    """Initialize the PrunePolyChebyshevComplexDPFuncts struct."""
    self = structref.new(PrunePolyChebyshevComplexDPFunctsType)
    self.param_arr = typed.List(param_arr)
    self.dparams = dparams
    self.tseg_ffa = tseg_ffa
    self.nbins = nbins
    self.eta = eta
    self.param_limits = typed.List(param_limits)
    self.bseg_brute = bseg_brute
    self.score_widths = score_widths
    self.poly_order = poly_order
    self.branch_max = branch_max
    self.use_conservative_tile = use_conservative_tile
    self.use_moving_grid = use_moving_grid
    return self


@njit(cache=True, fastmath=True)
def load_func(
    self: PrunePolyChebyshevDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> np.ndarray:
    return fold[seg_idx]


@njit(cache=True, fastmath=True)
def suggest_func(
    self: PrunePolyChebyshevDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> SuggestionStruct:
    return chebyshev.poly_chebyshev_suggest(
        fold_segment,
        coord_init,
        self.param_arr,
        self.dparams,
        self.poly_order,
        self.score_widths,
    )


@njit(cache=True, fastmath=True)
def suggest_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> SuggestionStructComplex:
    return chebyshev.poly_chebyshev_suggest_complex(
        fold_segment,
        coord_init,
        self.param_arr,
        self.dparams,
        self.poly_order,
        self.score_widths,
    )


@njit(cache=True, fastmath=True)
def branch_func(
    self: PrunePolyChebyshevDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    # Pass coord_cur for moving grid, coord_cur_fixed for fixed grid
    # Pass coord_prev for moving grid, coord_prev_fixed for fixed grid
    return chebyshev.poly_chebyshev_branch_batch(
        leaves_batch,
        coord_cur,
        coord_prev,
        self.nbins,
        self.eta,
        self.poly_order,
        self.param_limits,
        self.branch_max,
        self.use_conservative_tile,
    )


@njit(cache=True, fastmath=True)
def validate_func(
    self: PrunePolyChebyshevDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    return leaves_batch, leaves_origins


@njit(cache=True, fastmath=True)
def get_validation_params_func(
    self: PrunePolyChebyshevDPFuncts,
    coord_add: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    return common.get_validation_params(coord_add)


@njit(cache=True, fastmath=True)
def resolve_func(
    self: PrunePolyChebyshevDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    # Pass coord_cur for moving grid, coord_cur_fixed for fixed grid
    if self.use_moving_grid:
        return chebyshev.poly_chebyshev_resolve_batch(
            leaves_batch,
            coord_add,
            coord_cur,
            coord_init,
            self.param_arr,
            self.nbins,
        )
    return chebyshev.poly_chebyshev_fixed_resolve_batch(
        leaves_batch,
        coord_add,
        coord_cur,
        coord_init,
        self.param_arr,
        self.nbins,
    )


@njit(cache=True, fastmath=True)
def shift_add_func(
    self: PrunePolyChebyshevDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    return common.shift_add_batch(segment_batch, shift_batch, folds, isuggest_batch)


@njit(cache=True, fastmath=True)
def shift_add_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
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
    self: PrunePolyChebyshevDPFuncts,
    combined_res_batch: np.ndarray,
) -> float:
    return scoring.snr_score_batch_func(combined_res_batch, self.score_widths)


@njit(cache=True, fastmath=True)
def score_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    combined_res_batch: np.ndarray,
) -> float:
    return scoring.snr_score_batch_func_complex(combined_res_batch, self.score_widths)


@njit(cache=True, fastmath=True)
def transform_func(
    self: PrunePolyChebyshevDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> np.ndarray:
    if self.use_moving_grid:
        return chebyshev.poly_chebyshev_transform_batch(
            leaves_batch,
            coord_next,
            coord_cur,
            self.use_conservative_tile,
        )
    return leaves_batch


@njit(cache=True, fastmath=True)
def get_transform_matrix_func(
    self: PrunePolyChebyshevDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> np.ndarray:
    return common.get_trans_matrix(coord_next, coord_prev)


@njit(cache=True, fastmath=True)
def pack_func(self: PrunePolyChebyshevDPFuncts, data: np.ndarray) -> np.ndarray:
    return common.pack(data)


@overload_method(PrunePolyChebyshevDPFunctsTemplate, "load")
def ol_load_func(
    self: PrunePolyChebyshevDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevDPFuncts,
        fold: np.ndarray,
        seg_idx: int,
    ) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    return impl


@overload_method(PrunePolyChebyshevDPFunctsTemplate, "suggest")
def ol_suggest_func(
    self: PrunePolyChebyshevDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevDPFuncts,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStruct:
        return suggest_func(self, fold_segment, coord_init)

    return impl


@overload_method(PrunePolyChebyshevDPFunctsTemplate, "branch")
def ol_branch_func(
    self: PrunePolyChebyshevDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevDPFuncts,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

    return impl


@overload_method(PrunePolyChebyshevDPFunctsTemplate, "validate")
def ol_validate_func(
    self: PrunePolyChebyshevDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevDPFuncts,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

    return impl


@overload_method(PrunePolyChebyshevDPFunctsTemplate, "get_validation_params")
def ol_get_validation_params_func(
    self: PrunePolyChebyshevDPFuncts,
    coord_add: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevDPFuncts,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    return impl


@overload_method(PrunePolyChebyshevDPFunctsTemplate, "resolve")
def ol_resolve_func(
    self: PrunePolyChebyshevDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevDPFuncts,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    return impl


@overload_method(PrunePolyChebyshevDPFunctsTemplate, "shift_add")
def ol_shift_add_func(
    self: PrunePolyChebyshevDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevDPFuncts,
        segment_batch: np.ndarray,
        shift_batch: np.ndarray,
        folds: np.ndarray,
        isuggest_batch: np.ndarray,
    ) -> np.ndarray:
        return shift_add_func(self, segment_batch, shift_batch, folds, isuggest_batch)

    return impl


@overload_method(PrunePolyChebyshevDPFunctsTemplate, "score")
def ol_score_func(
    self: PrunePolyChebyshevDPFuncts,
    combined_res_batch: np.ndarray,
) -> types.FunctionType:
    def impl(self: PrunePolyChebyshevDPFuncts, combined_res_batch: np.ndarray) -> float:
        return score_func(self, combined_res_batch)

    return impl


@overload_method(PrunePolyChebyshevDPFunctsTemplate, "transform")
def ol_transform_func(
    self: PrunePolyChebyshevDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevDPFuncts,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    return impl


@overload_method(PrunePolyChebyshevDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_func(
    self: PrunePolyChebyshevDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevDPFuncts,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    return impl


@overload_method(PrunePolyChebyshevDPFunctsTemplate, "pack")
def ol_pack_func(
    self: PrunePolyChebyshevDPFuncts,
    data: np.ndarray,
) -> types.FunctionType:
    def impl(self: PrunePolyChebyshevDPFuncts, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    return impl


@overload_method(PrunePolyChebyshevComplexDPFunctsTemplate, "load")
def ol_load_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevComplexDPFuncts,
        fold: np.ndarray,
        seg_idx: int,
    ) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    return impl


@overload_method(PrunePolyChebyshevComplexDPFunctsTemplate, "suggest")
def ol_suggest_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevComplexDPFuncts,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStructComplex:
        return suggest_complex_func(self, fold_segment, coord_init)

    return impl


@overload_method(PrunePolyChebyshevComplexDPFunctsTemplate, "branch")
def ol_branch_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return branch_func(self, leaves_batch, coord_cur, coord_prev)

    return impl


@overload_method(PrunePolyChebyshevComplexDPFunctsTemplate, "validate")
def ol_validate_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevComplexDPFuncts,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(self, leaves_batch, leaves_origins, coord_cur)

    return impl


@overload_method(PrunePolyChebyshevComplexDPFunctsTemplate, "get_validation_params")
def ol_get_validation_params_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    coord_add: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevComplexDPFuncts,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    return impl


@overload_method(PrunePolyChebyshevComplexDPFunctsTemplate, "resolve")
def ol_resolve_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_add: tuple[float, float],
    coord_cur: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_add: tuple[float, float],
        coord_cur: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return resolve_func(self, leaves_batch, coord_add, coord_cur, coord_init)

    return impl


@overload_method(PrunePolyChebyshevComplexDPFunctsTemplate, "shift_add")
def ol_shift_add_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevComplexDPFuncts,
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


@overload_method(PrunePolyChebyshevComplexDPFunctsTemplate, "score")
def ol_score_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    combined_res_batch: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevComplexDPFuncts,
        combined_res_batch: np.ndarray,
    ) -> np.ndarray:
        return score_complex_func(self, combined_res_batch)

    return impl


@overload_method(PrunePolyChebyshevComplexDPFunctsTemplate, "transform")
def ol_transform_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    leaves_batch: np.ndarray,
    coord_next: tuple[float, float],
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevComplexDPFuncts,
        leaves_batch: np.ndarray,
        coord_next: tuple[float, float],
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return transform_func(self, leaves_batch, coord_next, coord_cur)

    return impl


@overload_method(PrunePolyChebyshevComplexDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    coord_next: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PrunePolyChebyshevComplexDPFuncts,
        coord_next: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_next, coord_prev)

    return impl


@overload_method(PrunePolyChebyshevComplexDPFunctsTemplate, "pack")
def ol_pack_complex_func(
    self: PrunePolyChebyshevComplexDPFuncts,
    data: np.ndarray,
) -> types.FunctionType:
    def impl(self: PrunePolyChebyshevComplexDPFuncts, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    return impl
