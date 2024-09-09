from __future__ import annotations

import numpy as np
from numba import njit, typed, types
from numba.experimental import jitclass

from pyloki.config import PulsarSearchConfig
from pyloki.core import common, defaults
from pyloki.detection import scoring
from pyloki.utils import math, np_utils
from pyloki.utils.misc import C_VAL


@njit(cache=True, fastmath=True)
def ffa_init(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    param_arr: types.ListType,
    bseg_brute: int,
    nbins: int,
    tsamp: float,
) -> np.ndarray:
    freq_arr = param_arr[-1]
    t_ref = bseg_brute * tsamp / 2
    return common.brutefold_start(
        ts_e,
        ts_v,
        freq_arr,
        bseg_brute,
        nbins,
        tsamp,
        t_ref,
    )


@njit(cache=True, fastmath=True)
def shift_params(param_vec: np.ndarray, tj_minus_ti: float) -> np.ndarray:
    """
    Shift the parameters to a new reference time.

    Parameters
    ----------
    param_vec : np.ndarray
        Parameter vector [..., a, v, d]
    tj_minus_ti : float
        Reference time to shift the parameters to. t_ref = t_j - t_i

    Returns
    -------
    np.ndarray
        Parameters at the new reference time.
    """
    nparams = len(param_vec)
    powers = np.tril(np.arange(nparams)[:, np.newaxis] - np.arange(nparams))
    # Calculate the transformation matrix (taylor coefficients)
    coeffs = tj_minus_ti**powers / math.fact(powers) * np.tril(np.ones_like(powers))
    return np.dot(coeffs, param_vec)


@njit(cache=True, fastmath=True)
def ffa_resolve(
    pset_cur: np.ndarray,
    parr_prev: np.ndarray,
    ffa_level: int,
    latter: int,
    tseg_brute: float,
    nbins: int,
) -> tuple[np.ndarray, int]:
    """
    Resolve the parameters of the current iter among the previous iter parameters.

    Parameters
    ----------
    pset_cur : np.ndarray
        Parameter set of the current iteration to resolve.
    parr_prev : np.ndarray
        Parameter array of the previous iteration.
    ffa_level : int
        Current FFA level (same level as pset_cur).
    latter : int
        Switch for the two halves of the previous iteration segments (0 or 1).
    tseg_brute : float
        Duration of the brute force segment.
    nbins : int
        Number of bins in the data.

    Returns
    -------
    tuple[np.ndarray, int]
        The resolved parameter set index and the relative phase shift.
    """
    nparams = len(pset_cur)
    t_ref_prev = (latter - 0.5) * 2 ** (ffa_level - 1) * tseg_brute
    if nparams == 1:
        pset_prev, delay_rel = pset_cur, 0
    else:
        kvec_cur = np.zeros(nparams + 1, dtype=np.float64)
        kvec_cur[:-2] = pset_cur[:-1]  # till acceleration
        kvec_prev = shift_params(kvec_cur, t_ref_prev)
        pset_prev = kvec_prev[:-1]
        pset_prev[-1] = pset_cur[-1] * (1 + kvec_prev[-2] / C_VAL)
        delay_rel = -kvec_prev[-1] / C_VAL
    relative_phase = common.get_phase_idx_helper(
        t_ref_prev,
        pset_prev[-1],
        nbins,
        delay_rel,
    )
    pindex_prev = np.empty(nparams, dtype=np.int64)
    for ip in range(nparams):
        pindex_prev[ip] = np_utils.find_nearest_sorted_idx(parr_prev[ip], pset_prev[ip])
    return pindex_prev, relative_phase


@njit(cache=True, fastmath=True)
def poly_taylor_resolve(
    leaf: np.ndarray,
    param_arr: types.ListType,
    coord_add: tuple[float, float],
    coord_init: tuple[float, float],
    nbins: int,
) -> tuple[np.ndarray, int]:
    """
    Resolve the leaf parameters to find the closest param index and phase shift.

    Parameters
    ----------
    leaf : np.ndarray
        The leaf parameter set.
    param_arr : types.ListType
        Parameter array containing the parameter values for the current segment.
    coord_add : tuple[float, float]
        The coordinates of the added segment (level cur).
    coord_init : tuple[float, float]
        The coordinates for the starting segment (level 0).
    nbins : int
        Number of bins in the folded profile.

    Returns
    -------
    tuple[np.ndarray, int]
        The resolved parameter index and the relative phase shift.

    Notes
    -----
    leaf is referenced to t_ref_init, so we need to shift it to t_ref_cur to get the
    resolved parameters index and phase shift.

    """
    nparams = len(param_arr)
    # distance between the current segment reference time and the global reference time
    tpoly = coord_add[0] - coord_init[0]

    kvec_cur = np.zeros(nparams + 1, dtype=np.float64)
    kvec_cur[:-2] = leaf[:-3, 0]  # till acceleration
    kvec_new = shift_params(kvec_cur, tpoly)

    f0 = leaf[-3, 0]

    new_a = kvec_new[-3]
    new_f = f0 * (1 + kvec_new[-2] / C_VAL)
    delay = kvec_new[-1] / C_VAL

    relative_phase = common.get_phase_idx(tpoly, f0, nbins, delay)
    idx_a = np_utils.find_nearest_sorted_idx(param_arr[-2], new_a)
    idx_f = np_utils.find_nearest_sorted_idx(param_arr[-1], new_f)
    index_prev = np.empty(nparams, dtype=np.int64)
    index_prev[-1] = idx_f
    index_prev[-2] = idx_a
    return index_prev, relative_phase


@njit(cache=True, fastmath=True)
def poly_taylor_step(
    nparams: int,
    tobs: float,
    freq: float,
    tsamp: float,
    tol: int,
) -> np.ndarray:
    dparam_opt = np.zeros(nparams)
    dparam_opt[-1] = common.freq_step_approx(tobs, freq, tsamp, tol)
    for deriv in range(2, nparams + 1):
        dparam_opt[-deriv] = common.param_step(tobs, tsamp, deriv, tol, t_ref=tobs / 2)
    return dparam_opt


@njit(cache=True, fastmath=True)
def split_taylor_params(
    param_cur: np.ndarray,
    dparam_cur: np.ndarray,
    dparam_opt: np.ndarray,
    tseg_cur: float,
    tol: float,
    tsamp: float,
    param_limits: types.ListType[types.Tuple[float, float]],
) -> np.ndarray:
    nparams = len(param_cur)
    leaf_params = typed.List.empty_list(types.float64[:])
    leaf_dparams = np.empty(nparams, dtype=np.float64)

    effective_tol = tol * tsamp
    for i in range(nparams):
        deriv = nparams - i
        shift = (
            (dparam_cur[i] - dparam_opt[i]) / 2 * (tseg_cur) ** deriv / math.fact(deriv)
        )
        if shift > effective_tol:
            leaf_param, dparam_act = common.branch_param(
                param_cur[i],
                dparam_cur[i],
                dparam_opt[i],
                param_limits[i][0],
                param_limits[i][1],
            )
        else:
            leaf_param, dparam_act = np.array([param_cur[i]]), dparam_cur[i]
        leaf_dparams[i] = dparam_act
        leaf_params.append(leaf_param)
    return common.get_leaves(leaf_params, leaf_dparams)


@njit(cache=True, fastmath=True)
def poly_taylor_branch2leaves(
    param_set: np.ndarray,
    coord_cur: tuple[float, float],
    tol: float,
    tsamp: float,
    poly_order: int,
    param_limits: types.ListType[types.Tuple[float, float]],
) -> np.ndarray:
    """
    Branch a parameter set to leaves.

    Parameters
    ----------
    param_set : np.ndarray
        Parameter set to branch. Shape: (nparams + 3, 2)
    tol : float
        Tolerance for the parameter step size in bins.
    tsamp : float
        Sampling time.

    Returns
    -------
    np.ndarray
        Array of leaf parameter sets.
    """
    _, scale_cur = coord_cur
    param_cur = param_set[0:-2, 0]
    dparam_cur = param_set[0:-2, 1]
    f0, _ = param_set[-2]
    t0, scale = param_set[-1]
    nparams = len(param_cur)

    duration = 2 * scale_cur
    dparam_opt = poly_taylor_step(nparams, duration, param_cur[-1], tsamp, tol)
    leafs_taylor = split_taylor_params(
        param_cur,
        dparam_cur,
        dparam_opt,
        duration,
        tol,
        tsamp,
        param_limits,
    )
    leaves = np.zeros((len(leafs_taylor), poly_order + 2, 2))
    leaves[:, :-2] = leafs_taylor
    leaves[:, -2, 0] = f0
    leaves[:, -1, 0] = t0
    leaves[:, -1, 1] = scale
    return leaves


@njit(cache=True, fastmath=True)
def poly_taylor_leaves(
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    coord_init: tuple[float, float],
) -> np.ndarray:
    """
    Generate the leaf parameter sets for Taylor polynomials.

    Parameters
    ----------
    param_arr : types.ListType
        Parameter array for each dimension; only (acceleration, period).
    dparams : np.ndarray
        Parameter step sizes for each dimension. Shape is (poly_order,).
    poly_order : int
        The order of the Taylor polynomial.
    coord_init : tuple[float, float]
        The coordinates for the starting segment (level 0).
        - coord_init[0] -> t0 (reference time)
        - coord_init[1] -> scale (duration of the segment)

    Returns
    -------
    np.ndarray
        The leaf parameter sets.

    """
    t0, scale = coord_init
    leafs_taylor = common.get_leaves(param_arr, dparams)
    leaves = np.zeros((len(leafs_taylor), poly_order + 2, 2))
    leaves[:, :-2] = leafs_taylor
    leaves[:, -2, 0] = leafs_taylor[:, -1, 0]  # f0
    leaves[:, -1, 0] = t0
    leaves[:, -1, 1] = scale
    return leaves


@njit(cache=True, fastmath=True)
def poly_taylor_suggestion_struct(
    fold_segment: np.ndarray,
    param_arr: types.ListType,
    dparams: np.ndarray,
    poly_order: int,
    coord_init: tuple[float, float],
    score_func: types.FunctionType,
) -> common.SuggestionStruct:
    """
    Generate a suggestion struct from a fold segment.

    Parameters
    ----------
    fold_segment : np.ndarray
        The fold segment to generate suggestions for. The shape of the array is
        (n_accel, n_period, 2, n_bins). Parameter dimensions are first two.
    param_arr : types.ListType
        Parameter array containing the parameter values for each dimension.
    dparams : np.ndarray
        Parameter step sizes for each dimension in a 1D array.
    poly_order : int
        The order of the Taylor polynomial.
    coord_init : tuple[float, float]
        The coordinates for the starting segment (level 0).
    score_func : _type_
        Function to score the folded data.

    Returns
    -------
    common.SuggestionStruct
        Suggestion struct
    """
    n_param_sets = np.prod(np.array([len(arr) for arr in param_arr]))
    # \n_param_sets = n_accel * n_period
    # \param_sets_shape = [n_param_sets, 2]
    param_sets = poly_taylor_leaves(param_arr, dparams, poly_order, coord_init)
    data = fold_segment.reshape((n_param_sets, *fold_segment.shape[-2:]))
    scores = np.zeros(n_param_sets)
    for iparam in range(n_param_sets):
        scores[iparam] = score_func(data[iparam])
    backtracks = np.zeros((n_param_sets, 2 + len(param_arr)))
    return common.SuggestionStruct(param_sets, data, scores, backtracks)


@njit(cache=True, fastmath=True)
def generate_branching_pattern(
    param_arr: types.ListType,
    dparams: np.ndarray,
    tchunk_ffa: float,
    nsegments: int,
    tol: float,
    tsamp: float,
    isuggest: int,
) -> np.ndarray:
    leaf_param_sets = common.get_leaves(param_arr, dparams)
    branching_pattern = []
    for prune_level in range(1, nsegments):
        tchunk_cur = tchunk_ffa * (prune_level + 1)  # noqa: F841
        leaves_arr = poly_taylor_branch2leaves(
            leaf_param_sets[isuggest],
            tol,
            tsamp,
        )
        branching_pattern.append(len(leaves_arr))
        leaf_param_sets = leaves_arr
    return np.array(branching_pattern)


@jitclass(spec=[("cfg", PulsarSearchConfig.class_type.instance_type)])
class FFASearchDPFunctions:
    """
    A container class for the functions used in the FFA search.

    Parameters
    ----------
    cfg : PulsarSearchConfig
        Configuration object for the search.
    """

    def __init__(self, cfg: PulsarSearchConfig) -> None:
        self.cfg = cfg

    def init(
        self,
        ts_e: np.ndarray,
        ts_v: np.ndarray,
        param_arr: types.ListType[types.Array],
    ) -> np.ndarray:
        return ffa_init(
            ts_e,
            ts_v,
            param_arr,
            self.cfg.bseg_brute,
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
        return ffa_resolve(
            pset_cur,
            parr_prev,
            ffa_level,
            latter,
            self.cfg.tseg_brute,
            self.cfg.nbins,
        )

    def add(self, data_tail: np.ndarray, data_head: np.ndarray) -> np.ndarray:
        return defaults.add(data_tail, data_head)

    def pack(self, data: np.ndarray, ffa_level: int) -> np.ndarray:  # noqa: ARG002
        return defaults.pack(data)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)


@jitclass(
    spec=[
        ("cfg", PulsarSearchConfig.class_type.instance_type),
        ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
        ("dparams", types.f8[:]),
        ("tseg_ffa", types.f8),
        ("poly_order", types.i8),
    ],
)
class PruningTaylorDPFunctions:
    def __init__(
        self,
        cfg: PulsarSearchConfig,
        param_arr: types.ListType[types.Array],
        dparams: np.ndarray,
        tseg_ffa: float,
        poly_order: int = 3,
    ) -> None:
        self.cfg = cfg
        self.param_arr = param_arr
        self.dparams = dparams
        self.tseg_ffa = tseg_ffa
        self.poly_order = poly_order

    def load(self, fold: np.ndarray, seg_idx: int) -> np.ndarray:
        return fold[seg_idx]

    def resolve(
        self,
        leaf: np.ndarray,
        coord_add: tuple[float, float],
        coord_init: tuple[float, float],
    ) -> tuple[np.ndarray, int]:
        coord_add = (coord_add[0] * self.tseg_ffa, coord_add[1] * self.tseg_ffa)
        coord_init = (coord_init[0] * self.tseg_ffa, coord_init[1] * self.tseg_ffa)
        return poly_taylor_resolve(
            leaf,
            self.param_arr,
            coord_add,
            coord_init,
            self.cfg.nbins,
        )

    def branch(
        self,
        param_set: np.ndarray,
        coord_cur: tuple[float, float],
    ) -> np.ndarray:
        coord_cur = (coord_cur[0] * self.tseg_ffa, coord_cur[1] * self.tseg_ffa)
        return poly_taylor_branch2leaves(
            param_set,
            coord_cur,
            self.cfg.tol,
            self.cfg.tsamp,
            self.poly_order,
            self.cfg.param_limits,
        )

    def suggest(
        self,
        fold_segment: np.ndarray,
        coord_init: tuple[float, float],
    ) -> common.SuggestionStruct:
        coord_init = (coord_init[0] * self.tseg_ffa, coord_init[1] * self.tseg_ffa)
        return poly_taylor_suggestion_struct(
            fold_segment,
            self.param_arr,
            self.dparams,
            self.poly_order,
            coord_init,
            self.score,
        )

    def score(self, combined_res: np.ndarray) -> float:
        return scoring.snr_score_func(combined_res)

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return defaults.add(data0, data1)

    def pack(self, x: np.ndarray) -> np.ndarray:
        return defaults.pack(x)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)

    def validate(
        self,
        leaves: np.ndarray,
        coord_valid: tuple[float, float],  # noqa: ARG002
        validation_params: tuple[np.ndarray, np.ndarray, float],  # noqa: ARG002
    ) -> np.ndarray:
        return leaves

    def transform(
        self,
        leaf: np.ndarray,
        coord_ref: tuple[float, float],  # noqa: ARG002
        trans_matrix: np.ndarray,  # noqa: ARG002
    ) -> np.ndarray:
        return leaf

    def get_transform_matrix(
        self,
        coord_cur: tuple[float, float],
        coord_prev: tuple[float, float],
    ) -> np.ndarray:
        return defaults.get_trans_matrix(coord_cur, coord_prev)

    def get_validation_params(
        self,
        coord_add: tuple[float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        return defaults.get_validation_params(coord_add)
