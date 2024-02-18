from __future__ import annotations
from numba import types, typed
from numba.experimental import jitclass
import numpy as np

from pruning import defaults, kernels, scores


@jitclass(
    spec=[
        ("nsamps", types.i8),
        ("dt", types.f8),
        ("tol", types.f8),
        ("nbins", types.i8),
        ("param_limits", types.ListType(types.Tuple([types.f8, types.f8]))),
        ("chunk_len", types.i8),
        ("_param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
        ("_dparams", types.f8[:]),
    ]
)
class SearchParams(object):
    """Class to hold the parameters for the polynomial search.

    Parameters
    ----------
    nsamps : int
        Number of samples in the time series
    dt : float
        Sampling time of the time series
    tol : float
        Tolerance parameter for the polynomial search
        (in units of number of time bins across the pulsar ducy)
    nbins : int
        Number of bins in the folded time series
    param_limits : types.ListType
        List of tuples with the min and max values for each parameter
    chunk_len : int, optional
        Length of the chunks to be used in the search, by default 0

    Notes
    ------
    The parameter limits are assumed to be in the order: acceleration, period.
    If chunk_len is not provided i.e = 0, it is calculated automatically to be
    optimal for the search.
    """

    def __init__(
        self,
        nsamps: int,
        dt: float,
        tol: float,
        nbins: int,
        param_limits: types.ListType,
        chunk_len: int = 0,
    ) -> None:
        self.nsamps = nsamps
        self.dt = dt
        self.tol = tol
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
        """int: Number of parameters in the search. Assumed to be non-chnaging"""
        return len(self.param_limits)

    @property
    def p_min(self) -> float:
        return self.param_limits[-1][0]

    @property
    def p_max(self) -> float:
        return self.param_limits[-1][1]

    def period_step(self, tobs: float) -> float:
        return kernels.period_step(tobs, self.dt, self.p_min, self.tol)

    def period_step_init(self, tobs: float) -> float:
        return kernels.period_step(tobs, self.dt, self.p_min, self.tol)
        # return kernels.period_step_init(tobs, self.nbins, self.p_min, self.tol)

    def deriv_step(self, tobs: float, deriv: int) -> float:
        t_ref = tobs / 2
        return kernels.param_step(tobs, self.dt, deriv, self.tol, t_ref=t_ref)

    def ffa_step(self, ffa_level: int) -> types.ListType[types.f8]:
        tchunk_cur = 2**ffa_level * self.tchunk
        step_period = self.period_step(tchunk_cur)
        step_arr = typed.List([step_period])
        for deriv in range(2, self.nparams + 1):
            step_param = self.deriv_step(tchunk_cur, deriv)
            step_arr.insert(0, step_param)
        return step_arr

    def get_updated_param_arr(
        self, param_steps: types.ListType
    ) -> types.ListType[types.Array]:
        if len(param_steps) != self.nparams:
            raise ValueError(
                f"param_steps must have length {self.nparams}, got {len(param_steps)}"
            )
        return typed.List(
            [
                kernels.range_param(*self.param_limits[iparam], param_steps[iparam])
                for iparam in range(self.nparams)
            ]
        )

    def get_updated_dparams(self, param_steps: types.ListType) -> np.ndarray:
        if len(param_steps) != self.nparams:
            raise ValueError(
                f"param_steps must have length {self.nparams}, got {len(param_steps)}"
            )
        dparams = np.zeros_like(self.dparams)
        for iparam in range(self.nparams):
            if iparam == self.nparams - 1:
                dparams[iparam] = param_steps[iparam]
            dparams[iparam] = min(self.dparams[iparam], param_steps[iparam])
        return dparams

    def _init_param_arr(self) -> types.ListType[types.Array]:
        dp = self.period_step_init(self.tchunk)
        period_arr = np.arange(self.p_min, self.p_max, dp)
        param_arr = typed.List([period_arr])
        for iparam in range(self.nparams - 1):
            param_val = np.array([sum(self.param_limits[iparam]) / 2], dtype=np.float64)
            param_arr.insert(0, param_val)
        return param_arr

    def _init_dparams(self) -> np.ndarray:
        dparams = np.zeros(self.nparams, dtype=np.float64)
        for iparam in range(self.nparams):
            dparams[iparam] = self.param_limits[iparam][1] - self.param_limits[iparam][0]
        return dparams

    def _optimal_chunk_len(self) -> int:
        if self.nparams == 1:
            init_levels = 1
        else:
            init_levels = 5
        levels = int(np.log2(self.nsamps * self.dt / self.p_max) - init_levels)
        return int(self.nsamps / 2**levels)

    def _check_params(self):
        if self.nparams > 4:
            raise ValueError(f"param_limits with len > {self.nparams} not supported yet.")
        for ival, (val_min, val_max) in enumerate(self.param_limits):
            if val_min >= val_max:
                raise ValueError(
                    f"param_limits[{ival}] must have min < max, got {self.param_limits}"
                )


@jitclass(spec=[("params", SearchParams.class_type.instance_type)])
class PeriodSearchDPFunctions(object):
    def __init__(self, params: SearchParams) -> None:
        self.params = params

    def init(self, ts_e: np.ndarray, ts_v: np.ndarray) -> np.ndarray:
        period_arr = self.params.param_arr[-1]
        return kernels.fold_brute_start(
            ts_e,
            ts_v,
            period_arr,
            self.params.chunk_len,
            self.params.nbins,
            self.params.dt,
        )

    def step(self, ffa_level: int) -> types.ListType[types.f8]:
        tchunk_cur = 2**ffa_level * self.params.tchunk
        step_period = self.params.period_step(tchunk_cur)
        return typed.List([step_period])

    def resolve(
        self, param_set: np.ndarray, param_arr: np.ndarray, ffa_level: int, latter: int
    ) -> tuple[tuple[int, ...], int]:
        param_idx_prev, phase_rel = defaults.ffa_resolve(
            param_set, param_arr, ffa_level, latter, self.params.tchunk, self.params.nbins
        )
        (p_idx_prev,) = param_idx_prev
        return (p_idx_prev,), phase_rel

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return defaults.add(data0, data1)

    def pack(self, data: np.ndarray, ffa_level: int) -> np.ndarray:
        return defaults.pack(data, ffa_level)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)


@jitclass(spec=[("params", SearchParams.class_type.instance_type)])
class AccelSearchDPFunctions(object):
    def __init__(self, params: SearchParams) -> None:
        self.params = params

    def init(self, ts_e: np.ndarray, ts_v: np.ndarray) -> np.ndarray:
        accel_arr, period_arr = self.params.param_arr
        fold = kernels.fold_brute_start(
            ts_e,
            ts_v,
            period_arr,
            self.params.chunk_len,
            self.params.nbins,
            self.params.dt,
        )
        # rotating the phases to be centered in the middle of the segment
        for iperiod, period in enumerate(period_arr):
            tmiddle = self.params.tchunk / 2
            n_roll = kernels.get_phase_idx(tmiddle, period, self.params.nbins)
            fold[:, iperiod] = kernels.nb_roll3d(fold[:, iperiod], -n_roll)
        # TODO: Correct for the accel in case it is not 0.
        return fold.reshape(fold.shape[:1] + (len(accel_arr),) + fold.shape[1:])

    def step(self, ffa_level: int) -> types.ListType[types.f8]:
        tchunk_cur = 2**ffa_level * self.params.tchunk
        step_period = self.params.period_step(tchunk_cur)
        step_accel = self.params.deriv_step(tchunk_cur, 2)
        return typed.List([step_accel, step_period])

    def resolve(
        self, param_set: np.ndarray, param_arr: np.ndarray, ffa_level: int, latter: int
    ) -> tuple[tuple[int, int], int]:
        param_idx_prev, phase_rel = defaults.ffa_resolve(
            param_set, param_arr, ffa_level, latter, self.params.tchunk, self.params.nbins
        )
        a_idx_prev, p_idx_prev = param_idx_prev
        return (a_idx_prev, p_idx_prev), phase_rel

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return defaults.add(data0, data1)

    def pack(self, data: np.ndarray, ffa_level: int) -> np.ndarray:
        return defaults.pack(data, ffa_level)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)


@jitclass(spec=[("params", SearchParams.class_type.instance_type)])
class JerkSearchDPFunctions(object):
    def __init__(self, params: SearchParams) -> None:
        self.params = params

    def init(self, ts_e: np.ndarray, ts_v: np.ndarray) -> np.ndarray:
        jerk_arr, accel_arr, period_arr = self.params.param_arr
        fold = kernels.fold_brute_start(
            ts_e,
            ts_v,
            period_arr,
            self.params.chunk_len,
            self.params.nbins,
            self.params.dt,
        )
        # rotating the phases to be centered in the middle of the segment
        for iperiod, period in enumerate(period_arr):
            tmiddle = self.params.tchunk / 2
            n_roll = kernels.get_phase_idx(tmiddle, period, self.params.nbins)
            fold[:, iperiod] = kernels.nb_roll3d(fold[:, iperiod], -n_roll)
        # TODO: Correct for the accel in case it is not 0.
        return fold.reshape(
            fold.shape[:1] + (len(jerk_arr), len(accel_arr)) + fold.shape[1:]
        )

    def step(self, ffa_level: int) -> types.ListType[types.f8]:
        tchunk_cur = 2**ffa_level * self.params.tchunk
        step_period = self.params.period_step(tchunk_cur)
        step_accel = self.params.deriv_step(tchunk_cur, 2)
        step_jerk = self.params.deriv_step(tchunk_cur, 3)
        return typed.List([step_jerk, step_accel, step_period])

    def resolve(
        self, param_set: np.ndarray, param_arr: np.ndarray, ffa_level: int, latter: int
    ) -> tuple[tuple[int, int], int]:
        param_idx_prev, phase_rel = defaults.ffa_resolve(
            param_set, param_arr, ffa_level, latter, self.params.tchunk, self.params.nbins
        )
        j_idx_prev, a_idx_prev, p_idx_prev = param_idx_prev
        return (j_idx_prev, a_idx_prev, p_idx_prev), phase_rel

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return defaults.add(data0, data1)

    def pack(self, data: np.ndarray, ffa_level: int) -> np.ndarray:
        return defaults.pack(data, ffa_level)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)


@jitclass(spec=[("params", SearchParams.class_type.instance_type)])
class SnapSearchDPFunctions(object):
    def __init__(self, params: SearchParams) -> None:
        self.params = params

    def init(self, ts_e: np.ndarray, ts_v: np.ndarray) -> np.ndarray:
        snap_arr, jerk_arr, accel_arr, period_arr = self.params.param_arr
        fold = kernels.fold_brute_start(
            ts_e,
            ts_v,
            period_arr,
            self.params.chunk_len,
            self.params.nbins,
            self.params.dt,
        )
        # rotating the phases to be centered in the middle of the segment
        for iperiod, period in enumerate(period_arr):
            tmiddle = self.params.tchunk / 2
            n_roll = kernels.get_phase_idx(tmiddle, period, self.params.nbins)
            fold[:, iperiod] = kernels.nb_roll3d(fold[:, iperiod], -n_roll)
        # TODO: Correct for the accel in case it is not 0.
        return fold.reshape(
            fold.shape[:1]
            + (len(snap_arr), len(jerk_arr), len(accel_arr))
            + fold.shape[1:]
        )

    def step(self, ffa_level: int) -> types.ListType[types.f8]:
        tchunk_cur = 2**ffa_level * self.params.tchunk
        step_period = self.params.period_step(tchunk_cur)
        step_accel = self.params.deriv_step(tchunk_cur, 2)
        step_jerk = self.params.deriv_step(tchunk_cur, 3)
        step_snap = self.params.deriv_step(tchunk_cur, 4)
        return typed.List([step_snap, step_jerk, step_accel, step_period])

    def resolve(
        self, param_set: np.ndarray, param_arr: np.ndarray, ffa_level: int, latter: int
    ) -> tuple[tuple[int, int], int]:
        param_idx_prev, phase_rel = defaults.ffa_resolve(
            param_set, param_arr, ffa_level, latter, self.params.tchunk, self.params.nbins
        )
        s_idx_prev, j_idx_prev, a_idx_prev, p_idx_prev = param_idx_prev
        return (s_idx_prev, j_idx_prev, a_idx_prev, p_idx_prev), phase_rel

    def add(self, data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
        return defaults.add(data0, data1)

    def pack(self, data: np.ndarray, ffa_level: int) -> np.ndarray:
        return defaults.pack(data, ffa_level)

    def shift(self, data: np.ndarray, phase_shift: int) -> np.ndarray:
        return defaults.shift(data, phase_shift)


@jitclass(
    spec=[
        ("params", SearchParams.class_type.instance_type),
        ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
        ("dparams", types.f8[:]),
        ("tchunk_current", types.f8),
    ]
)
class PruningAccelDPFunctions(object):
    def __init__(
        self,
        params: SearchParams,
        param_arr: types.ListType,
        dparams: np.ndarray,
        tchunk_current: float,
    ) -> None:
        self.params = params
        self.param_arr = param_arr
        self.dparams = dparams
        self.tchunk_current = tchunk_current

    def load(self, fold, index) -> np.ndarray:
        return fold[index]

    def resolve(
        self, param_set: np.ndarray, idx_distance: int
    ) -> tuple[tuple[int, int], float]:
        return defaults.prune_resolve_accel(
            param_set,
            idx_distance,
            self.param_arr,
            self.tchunk_current,
            self.params.nbins,
        )

    def branch(self, sug_params_cur: np.ndarray, idx_distance: int) -> np.ndarray:
        return defaults.branch2leaves(
            sug_params_cur,
            idx_distance,
            self.tchunk_current,
            self.params.tol,
            self.params.dt,
        )

    def suggest(self, fold_segment: np.ndarray) -> kernels.SuggestionStruct:
        return defaults.suggestion_struct(
            fold_segment, self.param_arr, self.dparams, self.score_func
        )

    def score_func(self, combined_res):
        return scores.snr_score_func(combined_res)

    def physical_validation(self):
        return defaults.identity_func

    def validation_params(self, params):
        return defaults.prepare_param_validation(params)

    def add(self, data0, data1):
        return defaults.add(data0, data1)

    def pack(self, x):
        return defaults.pack(x)

    def shift(self, data, phase_shift):
        return defaults.shift(data, phase_shift)

    def aggregate_stats(self, scores):
        return defaults.aggregate_stats(scores)

    def coord_transform(self, a, b, c):
        return defaults.coord_trans_params(a, b, c)

    def coord_transform_matrix(self, data_access_scheme):
        return defaults.prepare_coordinate_trans(data_access_scheme)

    def get_phase(self, sug_params):
        return defaults.get_phase(sug_params)


@jitclass(
    spec=[
        ("params", SearchParams.class_type.instance_type),
        ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
        ("dparams", types.f8[:]),
        ("tchunk_current", types.f8),
    ]
)
class PruningJerkDPFunctions(object):
    def __init__(
        self,
        params: SearchParams,
        param_arr: types.ListType,
        dparams: np.ndarray,
        tchunk_current: float,
    ) -> None:
        self.params = params
        self.param_arr = param_arr
        self.dparams = dparams
        self.tchunk_current = tchunk_current

    def load(self, fold, index) -> np.ndarray:
        return fold[index]

    def resolve(
        self, param_set: np.ndarray, idx_distance: int
    ) -> tuple[tuple[int, int], float]:
        return defaults.prune_resolve_jerk(
            param_set,
            idx_distance,
            self.param_arr,
            self.tchunk_current,
            self.params.nbins,
        )

    def branch(self, sug_params_cur: np.ndarray, idx_distance: int) -> np.ndarray:
        return defaults.branch2leaves(
            sug_params_cur,
            idx_distance,
            self.tchunk_current,
            self.params.tol,
            self.params.dt,
        )

    def suggest(self, fold_segment: np.ndarray) -> kernels.SuggestionStruct:
        return defaults.suggestion_struct(
            fold_segment, self.param_arr, self.dparams, self.score_func
        )

    def score_func(self, combined_res):
        return scores.snr_score_func(combined_res)

    def physical_validation(self):
        return defaults.identity_func

    def validation_params(self, params):
        return defaults.prepare_param_validation(params)

    def add(self, data0, data1):
        return defaults.add(data0, data1)

    def pack(self, x):
        return defaults.pack(x)

    def shift(self, data, phase_shift):
        return defaults.shift(data, phase_shift)

    def aggregate_stats(self, scores):
        return defaults.aggregate_stats(scores)

    def coord_transform(self, a, b, c):
        return defaults.coord_trans_params(a, b, c)

    def coord_transform_matrix(self, data_access_scheme):
        return defaults.prepare_coordinate_trans(data_access_scheme)

    def get_phase(self, sug_params):
        return defaults.get_phase(sug_params)


@jitclass(
    spec=[
        ("params", SearchParams.class_type.instance_type),
        ("param_arr", types.ListType(types.Array(types.f8, 1, "C"))),
        ("dparams", types.f8[:]),
        ("tchunk_current", types.f8),
    ]
)
class PruningSnapDPFunctions(object):
    def __init__(
        self,
        params: SearchParams,
        param_arr: types.ListType,
        dparams: np.ndarray,
        tchunk_current: float,
    ) -> None:
        self.params = params
        self.param_arr = param_arr
        self.dparams = dparams
        self.tchunk_current = tchunk_current

    def load(self, fold, index) -> np.ndarray:
        return fold[index]

    def resolve(
        self, param_set: np.ndarray, idx_distance: int
    ) -> tuple[tuple[int, int], float]:
        return defaults.prune_resolve_snap(
            param_set,
            idx_distance,
            self.param_arr,
            self.tchunk_current,
            self.params.nbins,
        )

    def branch(self, sug_params_cur: np.ndarray, idx_distance: int) -> np.ndarray:
        return defaults.branch2leaves(
            sug_params_cur,
            idx_distance,
            self.tchunk_current,
            self.params.tol,
            self.params.dt,
        )

    def suggest(self, fold_segment: np.ndarray) -> kernels.SuggestionStruct:
        return defaults.suggestion_struct(
            fold_segment, self.param_arr, self.dparams, self.score_func
        )

    def score_func(self, combined_res):
        return scores.snr_score_func(combined_res)

    def physical_validation(self):
        return defaults.identity_func

    def validation_params(self, params):
        return defaults.prepare_param_validation(params)

    def add(self, data0, data1):
        return defaults.add(data0, data1)

    def pack(self, x):
        return defaults.pack(x)

    def shift(self, data, phase_shift):
        return defaults.shift(data, phase_shift)

    def aggregate_stats(self, scores):
        return defaults.aggregate_stats(scores)

    def coord_transform(self, a, b, c):
        return
