from __future__ import annotations
from numba import njit, types
import numpy as np
import time

from pruning import base, utils
from pruning.timeseries import TSData


@njit
def unify_fold(
    fold: np.ndarray,
    param_cart_new: np.ndarray,
    param_arr: types.ListType[types.Array],
    ffa_level: int,
    dp_funcns: base.PeriodSearchDPFunctions,
):
    nchunks = fold.shape[0]
    outdata = np.zeros((nchunks // 2, len(param_cart_new), *fold.shape[-2:]), fold.dtype)
    for iparam_set, param_set in enumerate(param_cart_new):
        params_ind0, phase_shift0 = dp_funcns.resolve(
            param_set, param_arr, ffa_level - 1, 0
        )
        params_ind1, phase_shift1 = dp_funcns.resolve(
            param_set, param_arr, ffa_level - 1, 1
        )
        for ipair in range(outdata.shape[0]):
            fold0 = dp_funcns.shift(fold[ipair * 2][params_ind0], phase_shift0)
            fold1 = dp_funcns.shift(fold[ipair * 2 + 1][params_ind1], phase_shift1)
            outdata[ipair][iparam_set] = dp_funcns.add(fold0, fold1)
    return outdata


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
        print(f"initialization time: {time.time() - tstart}")

    def ffa_iter(self):
        self._ffa_level += 1
        param_steps = self.dp_funcns.step(self.ffa_level)
        print(f"param steps: {param_steps}")
        param_arr_new = self.params.get_updated_param_arr(param_steps)
        dparams_new = self.params.get_updated_dparams(param_steps)
        param_cart_new = utils.cartesian_prod_st(param_arr_new)
        fold_new = unify_fold(
            self.fold, param_cart_new, self.param_arr, self.ffa_level, self.dp_funcns
        )
        self._fold = fold_new.reshape(
            (self.nchunks // 2, *list(map(len, param_arr_new)), *self.fold.shape[-2:])
        )
        self._param_steps = param_steps
        self._param_arr = param_arr_new
        self._dparams = dparams_new
        self._chunk_duration *= 2

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

    def _set_dp_funcns(self, params):
        nparams_to_dp_funcns = {
            1: base.PeriodSearchDPFunctions,
            2: base.AccelSearchDPFunctions,
            3: base.JerkSearchDPFunctions,
        }
        return nparams_to_dp_funcns[params.nparams](params)
