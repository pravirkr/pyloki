# ruff: noqa: ARG001

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from numba import njit, typed, types
from numba.experimental import structref
from numba.extending import overload_method

from pyloki.core import chebyshev as cheby
from pyloki.core import common
from pyloki.detection import scoring
from pyloki.utils import maths

if TYPE_CHECKING:
    from pyloki.config import PulsarSearchConfig
    from pyloki.utils.suggestion import SuggestionStruct


@structref.register
class PruneChebyshevDPFunctsTemplate(types.StructRef):
    pass


class PruneChebyshevDPFuncts(structref.StructRefProxy):
    def __new__(
        cls,
        param_arr: types.ListType[types.Array],
        dparams: np.ndarray,
        tseg_ffa: float,
        cfg: PulsarSearchConfig,
    ) -> Self:
        """Create a new instance of PruneChebyshevDPFuncts."""
        return prune_chebyshev_dp_functs_init(
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
            cfg.use_fft_shifts,
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

    def resolve(
        self,
        leaf: np.ndarray,
        coord_add: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Resolve the leaf parameters to find the closest param index and phase shift.

        Parameters
        ----------
        leaf : np.ndarray
            Current leaf parameter set.
        coord_add : tuple[float, float]
            Time coordinate of the added segment.
        coord_init : tuple[float, float]
            Time coordinate of the starting segment.

        Returns
        -------
        tuple[np.ndarray, int]
            The resolved parameter index and the relative phase shift.
        """
        return resolve_func(self, leaf, coord_add, coord_init)

    def branch(
        self,
        param_set: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        """Branch the current parameter set into the finer grid of parameters (leaves).

        Parameters
        ----------
        param_set : np.ndarray
            The current parameter set to branch.

        Returns
        -------
        np.ndarray
            The branched parameter set.
        """
        return branch_func(self, param_set, coord_cur)

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

    def score(self, combined_res: np.ndarray) -> np.ndarray:
        """Calculate the statistical detection score of the combined fold.

        Parameters
        ----------
        combined_res : np.ndarray
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
        """Transform the leaf parameters to the new coordinate system.

        Parameters
        ----------
        leaf : np.ndarray
            Current leaf parameter set.
        coord_cur : tuple[float, float]
            The new coordinate to transform the leaf to.
        trans_matrix : np.ndarray
            The transformation matrix to apply.

        Returns
        -------
        np.ndarray
            The transformed leaf parameter set.
        """
        return transform_func(self, leaf, coord_cur, trans_matrix)

    def get_transform_matrix(
        self,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        """Get the transformation matrix between two coordinate systems."""
        return get_transform_matrix_func(self, coord_cur, coord_prev)

    def validate(
        self,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_valid: tuple[float, float],
        validation_params: tuple[np.ndarray, np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate which of the leaves are physical.

        Parameters
        ----------
        leaves : np.ndarray
            Set of leaves (parameter sets) to validate.
        coord_valid : np.ndarray
            hacky way to pass min and max
        validation_params : np.ndarray
            Pre-computed validation parameters for the physical validation.

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
        return validate_func(
            self,
            leaves_batch,
            leaves_origins,
            coord_valid,
            validation_params,
        )

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


fields_prune_chebyshev_dp_funcs = [
    ("nbins", types.int64),
    ("tol_bins", types.f8),
    ("param_limits", types.ListType(types.Tuple([types.f8, types.f8]))),
    ("bseg_brute", types.int64),
    ("score_widths", types.i8[::1]),
    ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
    ("dparams", types.f8[:]),
    ("tseg_ffa", types.f8),
    ("poly_order", types.i8),
    ("n_derivs", types.i8),
    ("cheb_table", types.f8[:, :]),
]

structref.define_boxing(PruneChebyshevDPFunctsTemplate, PruneChebyshevDPFuncts)
PruneChebyshevDPFunctsType = PruneChebyshevDPFunctsTemplate(
    fields_prune_chebyshev_dp_funcs,
)


@njit(cache=True, fastmath=True)
def prune_chebyshev_dp_functs_init(
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
    use_fft_shifts: bool,
) -> PruneChebyshevDPFuncts:
    """Initialize the PruneChebyshevDPFuncts struct."""
    self = structref.new(PruneChebyshevDPFunctsType)
    self.param_arr = param_arr
    self.dparams = dparams
    self.tseg_ffa = tseg_ffa
    self.nbins = nbins
    self.tol_bins = tol_bins
    self.param_limits = typed.List(param_limits)
    self.bseg_brute = bseg_brute
    self.score_widths = score_widths
    self.poly_order = poly_order
    self.branch_max = branch_max
    self.use_fft_shifts = use_fft_shifts
    n_derivs = 2  # fix later
    self.n_derivs = n_derivs
    self.cheb_table = maths.gen_chebyshev_polys_table(poly_order, n_derivs)
    return self


@njit(cache=True, fastmath=True)
def load_func(
    self: PruneChebyshevDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> np.ndarray:
    return fold[seg_idx]


@njit(cache=True, fastmath=True)
def resolve_func(
    self: PruneChebyshevDPFuncts,
    leaf: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
) -> tuple[np.ndarray, int]:
    return cheby.poly_chebyshev_resolve(
        leaf,
        self.param_arr,
        coord_add,
        coord_init,
        self.nbins,
        self.cheb_table,
    )


@njit(cache=True, fastmath=True)
def branch_func(
    self: PruneChebyshevDPFuncts,
    param_set: np.ndarray,
    coord_cur: tuple[float, float],
) -> np.ndarray:
    return cheby.poly_chebyshev_branch(
        param_set,
        self.tol_bins,
        self.tsamp,
        self.poly_order,
    )


@njit(cache=True, fastmath=True)
def suggest_func(
    self: PruneChebyshevDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> SuggestionStruct:
    dparams = np.array(
        [self.dparams[-1], self.dparams[-2]]  # df, da
        + [
            2 * self.cfg.deriv_bounds[k] / maths.fact(k)
            for k in range(3, self.poly_order + 1)
        ],
    )
    return cheby.poly_chebyshev_suggest(
        fold_segment,
        self.param_arr,
        dparams,
        self.poly_order,
        coord_init,
        self.score,
    )


@njit(cache=True, fastmath=True)
def score_func(self: PruneChebyshevDPFuncts, combined_res: np.ndarray) -> float:
    return scoring.snr_score_func(combined_res, self.score_widths)


@njit(cache=True, fastmath=True)
def add_func(
    self: PruneChebyshevDPFuncts,
    data0: np.ndarray,
    data1: np.ndarray,
) -> np.ndarray:
    return common.add(data0, data1)


@njit(cache=True, fastmath=True)
def pack_func(self: PruneChebyshevDPFuncts, data: np.ndarray) -> np.ndarray:
    return common.pack(data)


@njit(cache=True, fastmath=True)
def shift_func(
    self: PruneChebyshevDPFuncts,
    data: np.ndarray,
    phase_shift: int,
) -> np.ndarray:
    return common.shift(data, phase_shift)


@njit(cache=True, fastmath=True)
def transform_func(
    self: PruneChebyshevDPFuncts,
    leaf: np.ndarray,
    coord_cur: tuple[float, float],
    trans_matrix: np.ndarray,
) -> np.ndarray:
    return cheby.poly_chebyshev_transform(leaf, coord_cur, trans_matrix)


@njit(cache=True, fastmath=True)
def get_transform_matrix_func(
    self: PruneChebyshevDPFuncts,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> np.ndarray:
    return cheby.poly_chebyshev_transform_matrix(
        coord_cur,
        coord_prev,
        self.poly_order,
    )


@njit(cache=True, fastmath=True)
def validate_func(
    self: PruneChebyshevDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_valid: tuple[float, float],
    validation_params: tuple[np.ndarray, np.ndarray, float],
) -> tuple[np.ndarray, np.ndarray]:
    t_ref_add = coord_valid[1]
    t_ref_init = coord_valid[0]
    return cheby.poly_chebyshev_validate(
        leaves_batch,
        t_ref_add,
        t_ref_init,
        validation_params,
        self.tseg_ffa,
        self.deriv_bounds,
        self.n_valid,
        self.cheb_table,
        self.period_bounds,
    ), leaves_origins


@njit(cache=True, fastmath=True)
def get_validation_params_func(
    self: PruneChebyshevDPFuncts,
    coord_add: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, float]:
    t_ref = coord_add[1] - coord_add[0]
    return cheby.prepare_epicyclic_validation_params(
        t_ref,
        self.tseg_ffa,
        self.num_validation,
        self.omega_bounds,
        self.x_max,
        self.ecc_max,
    )


@overload_method(PruneChebyshevDPFunctsTemplate, "load")
def ol_load_func(
    self: PruneChebyshevDPFuncts,
    fold: np.ndarray,
    seg_idx: int,
) -> types.FunctionType:
    def impl(
        self: PruneChebyshevDPFuncts,
        fold: np.ndarray,
        seg_idx: int,
    ) -> np.ndarray:
        return load_func(self, fold, seg_idx)

    return impl


@overload_method(PruneChebyshevDPFunctsTemplate, "resolve")
def ol_resolve_func(
    self: PruneChebyshevDPFuncts,
    leaf: np.ndarray,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneChebyshevDPFuncts,
        leaf: np.ndarray,
        coord_add: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, int]:
        return resolve_func(self, leaf, coord_add, coord_init)

    return impl


@overload_method(PruneChebyshevDPFunctsTemplate, "branch")
def ol_branch_func(
    self: PruneChebyshevDPFuncts,
    param_set: np.ndarray,
    coord_cur: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneChebyshevDPFuncts,
        param_set: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        return branch_func(self, param_set, coord_cur)

    return impl


@overload_method(PruneChebyshevDPFunctsTemplate, "suggest")
def ol_suggest_func(
    self: PruneChebyshevDPFuncts,
    fold_segment: np.ndarray,
    coord_init: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneChebyshevDPFuncts,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> SuggestionStruct:
        return suggest_func(self, fold_segment, coord_init)

    return impl


@overload_method(PruneChebyshevDPFunctsTemplate, "score")
def ol_score_func(
    self: PruneChebyshevDPFuncts,
    combined_res: np.ndarray,
) -> types.FunctionType:
    def impl(self: PruneChebyshevDPFuncts, combined_res: np.ndarray) -> float:
        return score_func(self, combined_res)

    return impl


@overload_method(PruneChebyshevDPFunctsTemplate, "add")
def ol_add_func(
    self: PruneChebyshevDPFuncts,
    data0: np.ndarray,
    data1: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneChebyshevDPFuncts,
        data0: np.ndarray,
        data1: np.ndarray,
    ) -> np.ndarray:
        return add_func(self, data0, data1)

    return impl


@overload_method(PruneChebyshevDPFunctsTemplate, "pack")
def ol_pack_func(self: PruneChebyshevDPFuncts, data: np.ndarray) -> types.FunctionType:
    def impl(self: PruneChebyshevDPFuncts, data: np.ndarray) -> np.ndarray:
        return pack_func(self, data)

    return impl


@overload_method(PruneChebyshevDPFunctsTemplate, "shift")
def ol_shift_func(
    self: PruneChebyshevDPFuncts,
    data: np.ndarray,
    phase_shift: int,
) -> types.FunctionType:
    def impl(
        self: PruneChebyshevDPFuncts,
        data: np.ndarray,
        phase_shift: int,
    ) -> np.ndarray:
        return shift_func(self, data, phase_shift)

    return impl


@overload_method(PruneChebyshevDPFunctsTemplate, "transform")
def ol_transform_func(
    self: PruneChebyshevDPFuncts,
    leaf: np.ndarray,
    coord_cur: tuple[float, float],
    trans_matrix: np.ndarray,
) -> types.FunctionType:
    def impl(
        self: PruneChebyshevDPFuncts,
        leaf: np.ndarray,
        coord_cur: tuple[float, float],
        trans_matrix: np.ndarray,
    ) -> np.ndarray:
        return transform_func(self, leaf, coord_cur, trans_matrix)

    return impl


@overload_method(PruneChebyshevDPFunctsTemplate, "get_transform_matrix")
def ol_get_transform_matrix_func(
    self: PruneChebyshevDPFuncts,
    coord_cur: tuple[float, float],
    coord_prev: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneChebyshevDPFuncts,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return get_transform_matrix_func(self, coord_cur, coord_prev)

    return impl


@overload_method(PruneChebyshevDPFunctsTemplate, "validate")
def ol_validate_func(
    self: PruneChebyshevDPFuncts,
    leaves_batch: np.ndarray,
    leaves_origins: np.ndarray,
    coord_valid: tuple[float, float],
    validation_params: tuple[np.ndarray, np.ndarray, float],
) -> types.FunctionType:
    def impl(
        self: PruneChebyshevDPFuncts,
        leaves_batch: np.ndarray,
        leaves_origins: np.ndarray,
        coord_valid: tuple[float, float],
        validation_params: tuple[np.ndarray, np.ndarray, float],
    ) -> tuple[np.ndarray, np.ndarray]:
        return validate_func(
            self,
            leaves_batch,
            leaves_origins,
            coord_valid,
            validation_params,
        )

    return impl


@overload_method(PruneChebyshevDPFunctsTemplate, "get_validation_params")
def ol_get_validation_params_func(
    self: PruneChebyshevDPFuncts,
    coord_add: tuple[float, float],
) -> types.FunctionType:
    def impl(
        self: PruneChebyshevDPFuncts,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return get_validation_params_func(self, coord_add)

    return impl
