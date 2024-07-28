from __future__ import annotations

import numpy as np
from numba import typed, types
from numba.experimental import jitclass

from pruning import chebyshev, defaults, kernels, scores


@jitclass(
    spec=[
        ("nsamps", types.i8),
        ("tsamp", types.f8),
        ("nbins", types.i8),
        ("tolerance", types.f8),
        ("param_limits", types.ListType(types.Tuple([types.f8, types.f8]))),
        ("segment_len", types.i8),
    ],
)
class SearchConfig:
    """Class to hold the configuration for the polynomial search.

    Parameters
    ----------
    nsamps : int
        Number of samples in the time series
    tsamp : float
        Sampling time of the time series
    nbins : int
        Number of bins in the folded time series
    tolerance : float
        Tolerance (in bins) for the polynomial search
        (in units of number of time bins across the pulsar ducy)
    param_limits : types.ListType
        List of tuples with the min and max values for each search parameter
    segment_len : int, optional
        Length of the segment to use for the initial search bruefold search.

    Notes
    -----
    The parameter limits are assumed to be in the order: ..., acceleration, period.
    If segment_len is not provided i.e = 0, it is calculated automatically to be
    optimal for the search.
    """

    def __init__(
        self,
        nsamps: int,
        tsamp: float,
        nbins: int,
        tolerance: float,
        param_limits: types.ListType[types.Tuple],
        segment_len: int = 0,
    ) -> None:
        self.nsamps = nsamps
        self.tsamp = tsamp
        self.nbins = nbins
        self.tolerance = tolerance
        self.param_limits = param_limits
        self._check_params()

        if segment_len == 0:
            self.segment_len = self._segment_len_default()
        else:
            self.segment_len = segment_len

    @property
    def tsegment(self) -> float:
        return self.segment_len * self.tsamp

    @property
    def nparams(self) -> int:
        """int: Number of parameters in the search. Assumed to be non-changing."""
        return len(self.param_limits)

    @property
    def f_min(self) -> float:
        return self.param_limits[-1][0]

    @property
    def f_max(self) -> float:
        return self.param_limits[-1][1]

    def freq_step(self, tobs: float) -> float:
        return kernels.freq_step(tobs, self.nbins, self.f_max, self.tolerance)

    def deriv_step(self, tobs: float, deriv: int) -> float:
        t_ref = tobs / 2
        return kernels.param_step(tobs, self.tsamp, deriv, self.tolerance, t_ref=t_ref)

    def get_dparams(self, ffa_level: int) -> np.ndarray:
        tsegment_cur = 2**ffa_level * self.tsegment
        dparams = np.zeros(self.nparams, dtype=np.float64)
        dparams[-1] = self.freq_step(tsegment_cur)
        for deriv in range(2, self.nparams + 1):
            dparams[-deriv] = self.deriv_step(tsegment_cur, deriv)
        return dparams

    def get_updated_dparams(self, param_steps: types.ListType[types.f8]) -> np.ndarray:
        if len(param_steps) != self.nparams:
            msg = f"param_steps must have length {self.nparams}, got {len(param_steps)}"
            raise ValueError(msg)
        dparams = np.zeros_like(self.dparams)
        for iparam in range(self.nparams):
            if iparam == self.nparams - 1:
                dparams[iparam] = param_steps[iparam]
            dparams[iparam] = min(self.dparams[iparam], param_steps[iparam])
        return dparams

    def get_param_arr(self, param_steps: np.ndarray) -> types.ListType[types.Array]:
        if len(param_steps) != self.nparams:
            msg = f"param_steps must have length {self.nparams}, got {len(param_steps)}"
            raise ValueError(msg)
        return typed.List(
            [
                kernels.range_param(*self.param_limits[iparam], param_steps[iparam])
                for iparam in range(self.nparams)
            ],
        )

    def _segment_len_default(self) -> int:
        init_levels = 1 if self.nparams == 1 else 5
        levels = int(np.log2(self.nsamps * self.tsamp * self.f_max) - init_levels)
        return int(self.nsamps / 2**levels)

    def _check_params(self) -> None:
        if self.nparams < 1 or self.nparams > 4:
            msg = f"param_limits must have 1-4 elements, got {len(self.param_limits)}"
            raise ValueError(msg)
        for _, (val_min, val_max) in enumerate(self.param_limits):
            if val_min >= val_max:
                msg = f"param_limits must have min < max, got {self.param_limits}"
                raise ValueError(msg)


@jitclass(spec=[("cfg", SearchConfig.class_type.instance_type)])  # type: ignore [attr-defined]
class FFASearchDPFunctions:
    def __init__(self, cfg: SearchConfig) -> None:
        self.cfg = cfg

    def init(
        self,
        ts_e: np.ndarray,
        ts_v: np.ndarray,
        param_arr: types.ListType[types.Array],
    ) -> np.ndarray:
        return defaults.ffa_init(
            ts_e,
            ts_v,
            param_arr,
            self.cfg.segment_len,
            self.cfg.nbins,
            self.cfg.tsamp,
        )

    def step(self, ffa_level: int) -> np.ndarray:
        return self.cfg.get_dparams(ffa_level)

    def resolve(
        self,
        pset_cur: np.ndarray,
        parr_prev: np.ndarray,
        ffa_level: int,
        latter: int,
    ) -> tuple[np.ndarray, int]:
        return defaults.ffa_resolve(
            pset_cur,
            parr_prev,
            ffa_level,
            latter,
            self.cfg.tsegment,
            self.cfg.nbins,
        )

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return defaults.add(data0, data1)

    def pack(self, data: np.ndarray, ffa_level: int) -> np.ndarray:
        return defaults.pack(data, ffa_level)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)


@jitclass(
    spec=[
        ("cfg", SearchConfig.class_type.instance_type),  # type: ignore [attr-defined]
        ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
        ("dparams", types.f8[:]),
        ("tsegment_ffa", types.f8),
    ],
)
class PruningDPFunctions:
    def __init__(
        self,
        cfg: SearchConfig,
        param_arr: types.ListType[types.Array],
        dparams: np.ndarray,
        tsegment_ffa: float,
    ) -> None:
        self.cfg = cfg
        self.param_arr = param_arr
        self.dparams = dparams
        self.tsegment_ffa = tsegment_ffa

    def load(self, fold: np.ndarray, index: int) -> np.ndarray:
        return fold[index]

    def resolve(
        self,
        leaf: np.ndarray,
        seg_cur: int,
        seg_ref: int,
    ) -> tuple[tuple[int, int], int]:
        t_ref_cur = (seg_cur + 1 / 2) * self.tsegment_ffa
        t_ref_init = (seg_ref + 1 / 2) * self.tsegment_ffa
        return defaults.poly_taylor_resolve(
            leaf,
            self.param_arr,
            t_ref_cur,
            t_ref_init,
            self.cfg.nbins,
        )

    def branch(self, param_set: np.ndarray, prune_level: int) -> np.ndarray:
        tchunk_curr = self.tsegment_ffa * (prune_level + 1)
        return defaults.branch2leaves(
            param_set,
            tchunk_curr,
            self.cfg.tolerance,
            self.cfg.tsamp,
            self.cfg.nbins,
        )

    def suggest(self, fold_segment: np.ndarray) -> kernels.SuggestionStruct:
        return defaults.suggestion_struct(
            fold_segment,
            self.param_arr,
            self.dparams,
            self.score,
        )

    def score(self, combined_res: np.ndarray) -> float:
        return scores.snr_score_func(combined_res)

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return defaults.add(data0, data1)

    def pack(self, x: np.ndarray) -> np.ndarray:
        return defaults.pack(x)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)

    def validate_physical(
        self,
        leaves: np.ndarray,
        tcheby: float,
        tzero: float,
        validation_params: np.ndarray,
    ) -> np.ndarray:
        return defaults.validate_physical(leaves, tcheby, tzero, validation_params)

    def get_trans_matrix(
        self,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return defaults.get_trans_matrix(coord_cur, coord_prev)

    def transform_coords(
        self,
        leaf: np.ndarray,
        coord_ref: tuple[float, float],  # noqa: ARG002
        trans_matrix: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray:
        return leaf


@jitclass(
    spec=[
        ("cfg", SearchConfig.class_type.instance_type),  # type: ignore [attr-defined]
        ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
        ("dparams", types.f8[:]),
        ("tsegment_ffa", types.f8),
        ("poly_order", types.i8),
        ("n_validation_derivs", types.i8),
        ("_cheb_table", types.f8[:, :]),
    ],
)
class PruningChebychevDPFunctions:
    def __init__(
        self,
        cfg: SearchConfig,
        param_arr: types.ListType[types.Array],
        dparams: np.ndarray,
        tsegment_ffa: float,
        poly_order: int,
        n_validation_derivs: int,
    ) -> None:
        self.cfg = cfg
        self.param_arr = param_arr
        # / dparam = np.array([[self.ds, self.dj, self.da, self.dp]])
        self.dparams = dparams
        self.tsegment_ffa = tsegment_ffa
        self._cheb_table = self._compute_cheb_table(poly_order, n_validation_derivs)

    @property
    def cheb_table(self) -> np.ndarray:
        return self._cheb_table

    def load(self, fold: np.ndarray, index: int) -> np.ndarray:
        """Load the data for the given index from the fold.

        Parameters
        ----------
        fold : np.ndarray
            The folded data structure to load from.
        index : int
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
        return fold[index]

    def resolve(
        self,
        leaf: np.ndarray,
        seg_cur: int,
        seg_ref: int,
    ) -> tuple[tuple[int, int], int]:
        t_ref_cur = (seg_cur + 1 / 2) * self.tsegment_ffa
        t_ref_init = (seg_ref + 1 / 2) * self.tsegment_ffa
        return chebyshev.poly_chebychev_resolve(
            leaf,
            self.param_arr,
            t_ref_cur,
            t_ref_init,
            self.cfg.nbins,
            self.cheb_table,
        )

    def branch(self, param_set: np.ndarray) -> np.ndarray:
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
        return chebyshev.poly_chebychev_branch2leaves(
            param_set,
            self.poly_order,
            self.cfg.tolerance,
            self.cfg.tsamp,
        )

    def suggest(
        self,
        fold_segment: np.ndarray,
        ref_index: int,
    ) -> kernels.SuggestionStruct:
        """Generate an initial suggestion struct for the starting segment.

        Parameters
        ----------
        fold_segment : np.ndarray
            The folded data segment to generate the suggestion for.
        ref_index : int
            The reference index of the segment.

        Returns
        -------
        kernels.SuggestionStruct
            The initial suggestion struct for the segment.
        """
        t_ref = self.tsegment_ffa * (ref_index + 1 / 2)
        scale = self.tsegment_ffa / 2
        return chebyshev.suggestion_struct_chebyshev(
            fold_segment,
            self.param_arr,
            self.dparams,
            self.poly_order,
            t_ref,
            scale,
            self.score,
        )

    def score(self, combined_res: np.ndarray) -> float:
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
        return scores.snr_score_func(combined_res)

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return defaults.add(data0, data1)

    def pack(self, x: np.ndarray) -> np.ndarray:
        return defaults.pack(x)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)

    def get_validation_params(
        self,
        tcheby: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Prepare the validation parameters for the epicyclic validation.

        Parameters
        ----------
        tcheby : float
            The Chebyshev time to prepare the validation parameters for.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, float]
            The validation parameters for the epicyclic validation.
        """
        return chebyshev.prepare_epicyclic_validation_params(
            tcheby,
            self.tsegment_ffa,
            self.num_validation,
            self.omega_bounds,
            self.x_max,
            self.ecc_max,
        )

    def validate_physical(
        self,
        leaves: np.ndarray,
        tcheby: float,
        tzero: float,
        validation_params: np.ndarray,
    ) -> np.ndarray:
        """Validate which of the leaves are physical.

        Parameters
        ----------
        leaves : np.ndarray
            Set of leaves (parameter sets) to validate.
        indices_arr : np.ndarray
            Segment access scheme till now.
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
        return chebyshev.poly_chebychev_physical_validation(
            leaves,
            tcheby,
            tzero,
            validation_params,
            self.tsegment_ffa,
            self.derivative_bounds,
            self.num_validation,
            self.cheb_table,
            self.period_bounds,
        )

    def get_trans_matrix(
        self,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return chebyshev.poly_chebychev_coord_trans_matrix(
            coord_cur,
            coord_prev,
            self.poly_order,
        )

    def transform_coords(
        self,
        leaf: np.ndarray,
        coord_ref: tuple[float, float],
        trans_matrix: np.ndarray,
    ) -> np.ndarray:
        return chebyshev.poly_chebychev_coord_trans(leaf, coord_ref, trans_matrix)
