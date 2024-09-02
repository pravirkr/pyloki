from __future__ import annotations

import numpy as np
from numba import njit, typed, types
from numba.experimental import jitclass

from pruning import scores
from pruning.config import PulsarSearchConfig
from pruning.core import common, defaults
from pruning.utils import math, np_utils
from pruning.utils.misc import C_VAL


@njit(cache=True, fastmath=True)
def ffa_init(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    param_arr: types.ListType,
    segment_len: int,
    nbins: int,
    tsamp: float,
) -> np.ndarray:
    freq_arr = param_arr[-1]
    t_ref = segment_len * tsamp / 2
    return common.brutefold_start(
        ts_e,
        ts_v,
        freq_arr,
        segment_len,
        nbins,
        tsamp,
        t_ref,
    )


@njit(cache=True, fastmath=True)
def shift_params(param_vec: np.ndarray, tj_minus_ti: float) -> np.ndarray:
    """Shift the parameters to a new reference time.

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
    tchunk_init: float,
    nbins: int,
) -> tuple[np.ndarray, float]:
    """Resolve the parameters of the current iter among the previous iter parameters.

    Parameters
    ----------
    pset_cur : np.ndarray
        Parameter set of the current iteration to resolve.
    parr_prev : np.ndarray
        Parameter array of the previous iteration.
    ffa_level : int
        Current FFA level.
    latter : int
        Switch for the two halves of the previous iteration segments (0 or 1).
    tchunk_init : float
        Initial chunk duration.
    nbins : int
        Number of bins in the data.

    Returns
    -------
    tuple[np.ndarray, float]
        The resolved parameter set index and the relative phase shift.
    """
    nparams = len(pset_cur)
    t_ref_prev = (latter - 0.5) * 2 ** (ffa_level - 1) * tchunk_init
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
    t_ref_cur: float,
    t_ref_init: float,
    nbins: int,
) -> tuple[tuple[int, int], int]:
    """Resolve the leaf parameters to find the closest param index and phase shift.

    Parameters
    ----------
    leaf : np.ndarray
        The leaf parameter set.
    param_arr : types.ListType
        Parameter array containing the parameter values for the current segment.
    t_ref_cur : float
        The reference time for the current segment.
    t_ref_init : float
        The reference time for the initial segment (pruning level 0).
    nbins : int
        Number of bins in the folded profile.

    Returns
    -------
    tuple[tuple[int, int], int]
        The resolved parameter index and the relative phase shift.

    Notes
    -----
    leaf is referenced to t_ref_init, so we need to shift it to t_ref_cur to get the
    resolved parameters index and phase shift.

    """
    nparams = len(leaf)
    # distance between the current segment reference time and the global reference time
    tpoly = t_ref_cur - t_ref_init

    kvec_cur = np.zeros(nparams + 1, dtype=np.float64)
    kvec_cur[:-2] = leaf[:-1, 0]  # till acceleration
    kvec_new = shift_params(kvec_cur, tpoly)

    old_f = leaf[-1, 0]

    new_a = kvec_new[-3]
    new_f = old_f * (1 + kvec_new[-2] / C_VAL)
    delay = kvec_new[-1] / C_VAL

    relative_phase = common.get_phase_idx(tpoly, old_f, nbins, delay)
    prev_index_a = np_utils.find_nearest_sorted_idx(param_arr[-2], new_a)
    prev_index_f = np_utils.find_nearest_sorted_idx(param_arr[-1], new_f)
    return (prev_index_a, prev_index_f), relative_phase


@njit(cache=True, fastmath=True)
def split_params(
    param_cur: np.ndarray,
    dparam_cur: np.ndarray,
    dparam_opt: np.ndarray,
    tol_time: float,
    tchunk_cur: float,
) -> np.ndarray:
    nparams = len(param_cur)
    leaf_params = typed.List.empty_list(types.float64[:])
    leaf_dparams = np.empty(nparams, dtype=np.float64)
    for iparam in range(nparams):
        deriv = nparams - iparam
        shift = (
            (dparam_cur[iparam] - dparam_opt[iparam])
            / 2
            * (tchunk_cur) ** deriv
            / math.fact(deriv)
        )
        if shift > tol_time:
            leaf_param, dparam_act = common.branch_param(
                param_cur[iparam],
                dparam_cur[iparam],
                dparam_opt[iparam],
            )
        else:
            leaf_param, dparam_act = np.array([param_cur[iparam]]), dparam_cur[iparam]
        leaf_dparams[iparam] = dparam_act
        leaf_params.append(leaf_param)
    return common.get_leaves(leaf_params, leaf_dparams)


@njit(cache=True, fastmath=True)
def branch2leaves(
    param_set: np.ndarray,
    tchunk_cur: float,
    tolerance: float,
    tsamp: float,
    nbins: int,
    t_ref: float,
) -> np.ndarray:
    """Branch a parameter set to leaves.

    Parameters
    ----------
    param_set : np.ndarray
        Parameter set to branch. Shape: (nparams, 2)
    tchunk_cur : float
        Total chunk duration at the current pruning level.
    tolerance : float
        Tolerance for the parameter step size in bins.
    tsamp : float
        Sampling time.
    nbins : int
        Number of bins in the folded profile.

    Returns
    -------
    np.ndarray
        Array of leaf parameter sets.
    """
    nparams, _ = param_set.shape
    param_cur = param_set[:, 0]
    dparam_cur = param_set[:, 1]
    dparam_opt = np.zeros(nparams)
    for iparam in range(nparams):
        deriv = nparams - iparam
        if deriv == 1:
            dparam_opt_p = common.freq_step(
                tchunk_cur,
                nbins,
                param_cur[iparam],
                tolerance,
            )
            # for param_cur = freq, tchunk is number of period jumps
            tchunk_cur *= param_cur[iparam]
        else:
            dparam_opt_p = common.param_step(
                tchunk_cur,
                tsamp,
                deriv,
                tolerance,
                t_ref=t_ref,
            )
        dparam_opt[iparam] = dparam_opt_p
    tol_time = tolerance * tsamp
    return split_params(param_cur, dparam_cur, dparam_opt, tol_time, tchunk_cur)


@njit(cache=True, fastmath=True)
def suggestion_struct(
    fold_segment: np.ndarray,
    param_arr: types.ListType,
    dparams: np.ndarray,
    score_func: types.FunctionType,
) -> common.SuggestionStruct:
    """Generate a suggestion struct from a fold segment.

    Parameters
    ----------
    fold_segment : np.ndarray
        The fold segment to generate suggestions for. The shape of the array is
        (n_accel, n_period, 2, n_bins). Parameter dimensions are first two.
    param_arr : types.ListType
        Parameter array containing the parameter values for each dimension.
    dparams : np.ndarray
        Parameter step sizes for each dimension in a 1D array.
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
    param_sets = common.get_leaves(param_arr, dparams)
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
    tol_bins: float,
    tsamp: float,
    nbins: int,
    isuggest: int,
) -> np.ndarray:
    leaf_param_sets = common.get_leaves(param_arr, dparams)
    branching_pattern = []
    for prune_level in range(1, nsegments):
        tchunk_cur = tchunk_ffa * (prune_level + 1)
        leaves_arr = branch2leaves(
            leaf_param_sets[isuggest],
            tchunk_cur,
            tol_bins,
            tsamp,
            nbins,
            tchunk_cur / 2,
        )
        branching_pattern.append(len(leaves_arr))
        leaf_param_sets = leaves_arr
    return np.array(branching_pattern)


@jitclass(spec=[("cfg", PulsarSearchConfig.class_type.instance_type)])  # type: ignore [attr-defined]
class FFASearchDPFunctions:
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
        return ffa_resolve(
            pset_cur,
            parr_prev,
            ffa_level,
            latter,
            self.cfg.tsegment,
            self.cfg.nbins,
        )

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return defaults.add(data0, data1)

    def pack(self, data: np.ndarray, ffa_level: int) -> np.ndarray:  # noqa: ARG002
        return defaults.pack(data)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)


@jitclass(
    spec=[
        ("cfg", PulsarSearchConfig.class_type.instance_type),  # type: ignore [attr-defined]
        ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
        ("dparams", types.f8[:]),
        ("tsegment_ffa", types.f8),
    ],
)
class PruningDPFunctions:
    def __init__(
        self,
        cfg: PulsarSearchConfig,
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
        return poly_taylor_resolve(
            leaf,
            self.param_arr,
            t_ref_cur,
            t_ref_init,
            self.cfg.nbins,
        )

    def branch(self, param_set: np.ndarray, prune_level: int) -> np.ndarray:
        tchunk_curr = self.tsegment_ffa * (prune_level + 1)
        return branch2leaves(
            param_set,
            tchunk_curr,
            self.cfg.tolerance,
            self.cfg.tsamp,
            self.cfg.nbins,
        )

    def suggest(self, fold_segment: np.ndarray) -> common.SuggestionStruct:
        return suggestion_struct(
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
