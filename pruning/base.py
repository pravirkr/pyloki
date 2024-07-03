from __future__ import annotations

import numpy as np
from numba import typed, types
from numba.experimental import jitclass

from pruning import defaults, kernels, scores


@jitclass(
    spec=[
        ("nsamps", types.i8),
        ("dt", types.f8),
        ("tol_bins", types.f8),
        ("nbins", types.i8),
        ("param_limits", types.ListType(types.Tuple([types.f8, types.f8]))),
        ("chunk_len", types.i8),
        ("_param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
        ("_dparams", types.f8[:]),
    ],
)
class SearchConfig:
    """Class to hold the configuration for the polynomial search.

    Parameters
    ----------
    nsamps : int
        Number of samples in the time series
    dt : float
        Sampling time of the time series
    tol_bins : float
        Tolerance parameter for the polynomial search
        (in units of number of time bins across the pulsar ducy)
    nbins : int
        Number of bins in the folded time series
    param_limits : types.ListType
        List of tuples with the min and max values for each search parameter
    chunk_len : int, optional
        Length of the chunks to be used in the search, by default 0

    Notes
    -----
    The parameter limits are assumed to be in the order: ..., acceleration, period.
    If chunk_len is not provided i.e = 0, it is calculated automatically to be
    optimal for the search.
    """

    def __init__(
        self,
        nsamps: int,
        dt: float,
        tol_bins: float,
        nbins: int,
        param_limits: types.ListType[types.Tuple],
        chunk_len: int = 0,
    ) -> None:
        self.nsamps = nsamps
        self.dt = dt
        self.tol_bins = tol_bins
        self.nbins = nbins
        self.param_limits = param_limits
        self._check_params()

        if chunk_len == 0:
            self.chunk_len = self._optimal_chunk_len()
        else:
            self.chunk_len = chunk_len

        self._param_arr = self._init_param_arr()
        self._dparams = self._init_dparams()

    @property
    def param_arr(self) -> types.ListType[types.Array]:
        return self._param_arr

    @property
    def dparams(self) -> np.ndarray:
        return self._dparams

    @property
    def tchunk(self) -> float:
        return self.chunk_len * self.dt

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
        return kernels.freq_step(tobs, self.nbins, self.f_max, self.tol_bins)

    def deriv_step(self, tobs: float, deriv: int) -> float:
        t_ref = tobs / 2
        return kernels.param_step(tobs, self.dt, deriv, self.tol_bins, t_ref=t_ref)

    def ffa_step(self, ffa_level: int) -> types.ListType[types.f8]:
        tchunk_cur = 2**ffa_level * self.tchunk
        step_freq = self.freq_step(tchunk_cur)
        step_list = typed.List([step_freq])
        for deriv in range(2, self.nparams + 1):
            step_param = self.deriv_step(tchunk_cur, deriv)
            step_list.insert(0, step_param)
        return step_list

    def get_updated_param_arr(
        self,
        param_steps: types.ListType[types.f8],
    ) -> types.ListType[types.Array]:
        if len(param_steps) != self.nparams:
            msg = f"param_steps must have length {self.nparams}, got {len(param_steps)}"
            raise ValueError(msg)
        return typed.List(
            [
                kernels.range_param(*self.param_limits[iparam], param_steps[iparam])
                for iparam in range(self.nparams)
            ],
        )

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

    def _init_param_arr(self) -> types.ListType[types.Array]:
        df = self.freq_step(self.tchunk)
        freqs_ar = np.arange(self.f_min, self.f_max, df)
        param_arr = typed.List([freqs_ar])
        for iparam in range(self.nparams - 1):
            param_val = np.array([sum(self.param_limits[iparam]) / 2], dtype=np.float64)
            param_arr.insert(0, param_val)
        return param_arr

    def _init_dparams(self) -> np.ndarray:
        dparams = np.zeros(self.nparams, dtype=np.float64)
        for iparam in range(self.nparams):
            dparams[iparam] = (
                self.param_limits[iparam][1] - self.param_limits[iparam][0]
            )
        return dparams

    def _optimal_chunk_len(self) -> int:
        init_levels = 1 if self.nparams == 1 else 5
        levels = int(np.log2(self.nsamps * self.dt * self.f_max) - init_levels)
        return int(self.nsamps / 2**levels)

    def _check_params(self) -> None:
        if self.nparams > 4:
            msg = f"Number of parameters > 4 not supported yet, got {self.nparams}"
            raise ValueError(msg)
        for _, (val_min, val_max) in enumerate(self.param_limits):
            if val_min >= val_max:
                msg = f"param_limits must have min < max, got {self.param_limits}"
                raise ValueError(msg)


@jitclass(spec=[("cfg", SearchConfig.class_type.instance_type)])  # type: ignore [attr-defined]
class FFASearchDPFunctions:
    def __init__(self, cfg: SearchConfig) -> None:
        self.cfg = cfg

    def init(self, ts_e: np.ndarray, ts_v: np.ndarray) -> np.ndarray:
        return defaults.ffa_init(
            ts_e,
            ts_v,
            self.cfg.param_arr,
            self.cfg.chunk_len,
            self.cfg.nbins,
            self.cfg.dt,
        )

    def step(self, ffa_level: int) -> types.ListType[types.f8]:
        return self.cfg.ffa_step(ffa_level)

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
            self.cfg.tchunk,
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
        ("tchunk_ffa", types.f8),
    ],
)
class PruningDPFunctions:
    def __init__(
        self,
        cfg: SearchConfig,
        param_arr: types.ListType[types.Array],
        dparams: np.ndarray,
        tchunk_ffa: float,
    ) -> None:
        self.cfg = cfg
        self.param_arr = param_arr
        self.dparams = dparams
        self.tchunk_ffa = tchunk_ffa

    def load(self, fold: np.ndarray, index: int) -> np.ndarray:
        return fold[index]

    def resolve(self, param_set: np.ndarray, idx_dist: int) -> tuple[np.ndarray, int]:
        t_ref_prev = idx_dist * self.tchunk_ffa
        return defaults.prune_resolve(
            param_set,
            self.param_arr,
            t_ref_prev,
            self.cfg.nbins,
        )

    def branch(self, param_set: np.ndarray, prune_level: int) -> np.ndarray:
        tchunk_curr = self.tchunk_ffa * (prune_level + 1)
        return defaults.branch2leaves(
            param_set,
            tchunk_curr,
            self.cfg.tol_bins,
            self.cfg.dt,
            self.cfg.nbins,
        )

    def suggest(self, fold_segment: np.ndarray) -> kernels.SuggestionStruct:
        return defaults.suggestion_struct(
            fold_segment,
            self.param_arr,
            self.dparams,
            self.score_func,
        )

    def score_func(self, combined_res: np.ndarray) -> float:
        return scores.snr_score_func(combined_res)

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return defaults.add(data0, data1)

    def pack(self, x: np.ndarray) -> np.ndarray:
        return defaults.pack(x)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)

    def validate_physical(self, leaves_arr: np.ndarray) -> np.ndarray:
        return defaults.validate_physical(leaves_arr)

    def get_trans_matrix(self, leaf_param_set: np.ndarray) -> np.ndarray:
        return defaults.get

    def transform_coords(
        self,
        leaf_param_set: np.ndarray,
        trans_matrix: np.ndarray,
        coord_ref: np.ndarray,
    ) -> np.ndarray:
        return defaults.transform_coords
