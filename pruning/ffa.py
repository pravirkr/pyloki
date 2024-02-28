from __future__ import annotations
from numba import njit, types
import numpy as np
import time

from pruning import base, utils
from pruning.timeseries import TSData


@njit(cache=True)
def unify_fold(
    fold_in: np.ndarray,
    param_arr: types.ListType[types.Array],
    fold_out: np.ndarray,
    param_cart_new: np.ndarray,
    ffa_level: int,
    dp_funcns: base.PeriodSearchDPFunctions,
):
    for iparam_set in range(len(param_cart_new)):
        param_set = param_cart_new[iparam_set]
        param_idx0, phase_shift0 = dp_funcns.resolve(param_set, param_arr, ffa_level, 0)
        param_idx1, phase_shift1 = dp_funcns.resolve(param_set, param_arr, ffa_level, 1)
        for ipair in range(fold_out.shape[0]):
            fold0 = dp_funcns.shift(fold_in[ipair * 2][param_idx0], phase_shift0)
            fold1 = dp_funcns.shift(fold_in[ipair * 2 + 1][param_idx1], phase_shift1)
            fold_out[ipair][iparam_set] = dp_funcns.add(fold0, fold1)


class DynamicProgramming(object):
    def __init__(self, ts_data: TSData, params: base.SearchParams, data_type=np.float32):
        self.ts_data = ts_data
        self.data_type = data_type
        self._params = params
        self._dp_funcns = self._set_dp_funcns(params)

        self._fold = None
        self._ffa_level = None
        self._param_steps = None
        self._param_arr = None
        self._dparams = None
        self._chunk_duration = None

        self.time_init = 0
        self.time_step = 0
        self.time_cart = 0
        self.time_fold = 0
        self.time_total = 0

    @property
    def params(self) -> base.SearchParams:
        return self._params

    @property
    def dp_funcns(self) -> base.PeriodSearchDPFunctions:
        return self._dp_funcns

    @property
    def fold(self) -> np.ndarray:
        return self._fold

    @property
    def nchunks(self) -> int:
        return self._fold.shape[0]

    @property
    def nbins(self) -> int:
        return self._fold.shape[-1]

    @property
    def ffa_level(self) -> int:
        return self._ffa_level

    @property
    def param_steps(self) -> types.ListType[types.f8]:
        return self._param_steps

    @property
    def param_arr(self) -> types.ListType[types.Array]:
        return self._param_arr

    @property
    def nparam_vol(self) -> int:
        return np.prod(list(map(len, self.param_arr)))

    @property
    def dparams(self) -> np.ndarray:
        return self._dparams

    @property
    def chunk_duration(self):
        return self._chunk_duration

    def initialize(self):
        tstart = time.time()
        self._ffa_level = 0
        print("Initiating data structure...")
        fold = self.dp_funcns.init(self.ts_data.ts_e, self.ts_data.ts_v)
        fold = self.dp_funcns.pack(fold, self.ffa_level)
        print(f"fold dimensions: {fold.shape}")

        self._fold = fold.astype(self.data_type)
        self._param_arr = self.params.param_arr
        self._dparams = self.params.dparams
        self._chunk_duration = self.params.tchunk
        self.time_init += time.time() - tstart
        print(f"initialization time: {self.time_init}")

    def ffa_iter(self):
        tstart = time.time()
        self._ffa_level += 1
        param_steps = self.dp_funcns.step(self.ffa_level)
        print(f"param steps: {param_steps}")
        param_arr_new = self.params.get_updated_param_arr(param_steps)
        dparams_new = self.params.get_updated_dparams(param_steps)
        self.time_step += time.time() - tstart
        tstart = time.time()
        param_cart_new = utils.cartesian_prod_st(param_arr_new)
        self.time_cart += time.time() - tstart
        fold_new = np.zeros(
            (self.nchunks // 2, len(param_cart_new), *self.fold.shape[-2:]),
            self.fold.dtype,
        )
        tstart = time.time()
        unify_fold(
            self.fold,
            self.param_arr,
            fold_new,
            param_cart_new,
            self.ffa_level,
            self.dp_funcns,
        )
        self.time_fold += time.time() - tstart
        self._fold = fold_new.reshape(
            (self.nchunks // 2, *list(map(len, param_arr_new)), *self.fold.shape[-2:])
        )
        self._param_steps = param_steps
        self._param_arr = param_arr_new
        self._dparams = dparams_new
        self._chunk_duration *= 2

    def ffa_iter_dry(self):
        self._ffa_level += 1
        param_steps = self.dp_funcns.step(self.ffa_level)
        print(f"param steps: {param_steps}")
        param_arr_new = self.params.get_updated_param_arr(param_steps)
        dparams_new = self.params.get_updated_dparams(param_steps)
        complexity = np.prod(list(map(len, param_arr_new)))
        self._param_steps = param_steps
        self._param_arr = param_arr_new
        self._dparams = dparams_new
        self._chunk_duration *= 2
        return complexity

    def do_iterations(self, n_iters="max"):
        tstart = time.time()
        if n_iters == "max":
            n_iters = int(np.log2(len(self.fold)))
        elif n_iters == "prune":
            n_iters = int(np.log2(len(self.fold))) - 7
        for _ in range(n_iters):
            print(f"performing iteration: {self.ffa_level + 1}")
            self.ffa_iter()
            print(f"fold dimensions: {self.fold.shape}")
            print(f"elapsed time: {time.time() - tstart}")
        self.time_total += time.time() - tstart

    def do_iterations_dry(self, n_iters="max"):
        if n_iters == "max":
            n_iters = int(np.log2(len(self.fold)))
        elif n_iters == "prune":
            n_iters = int(np.log2(len(self.fold))) - 7
        complexity = []
        for _ in range(n_iters):
            print(f"performing iteration: {self.ffa_level + 1}")
            complexity.append(self.ffa_iter_dry())
        return complexity

    def _set_dp_funcns(self, params):
        nparams_to_dp_funcns = {
            1: base.FreqSearchDPFunctions,
            2: base.AccelSearchDPFunctions,
            3: base.JerkSearchDPFunctions,
            4: base.SnapSearchDPFunctions,
        }
        return nparams_to_dp_funcns[params.nparams](params)
