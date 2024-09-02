from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange, types

from pyloki.core import FFASearchDPFunctions, PulsarSearchConfig
from pyloki.utils import np_utils
from pyloki.utils.misc import get_logger

if TYPE_CHECKING:
    from typing import Callable

    from numpy import typing as npt

    from pyloki.io.timeseries import TimeSeries


logger = get_logger(__name__)


@njit(cache=True, fastmath=True)
def load_folds_1d(fold_in: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """fold_in shape: (nsegments, nfreqs, 2, nbins)."""
    return fold_in[iseg, param_idx[0]]


@njit(cache=True, fastmath=True)
def load_folds_2d(fold_in: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """fold_in shape: (nsegments, naccels, nfreqs, 2, nbins)."""
    return fold_in[iseg, param_idx[0], param_idx[1]]


@njit(cache=True, fastmath=True)
def load_folds_3d(fold_in: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """fold_in shape: (nsegments, njerks, naccels, nfreqs, 2, nbins)."""
    return fold_in[iseg, param_idx[0], param_idx[1], param_idx[2]]


@njit(cache=True, fastmath=True)
def load_folds_4d(fold_in: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """fold_in shape: (nsegments, nsnap, njerks, naccels, nfreqs, 2, nbins)."""
    return fold_in[iseg, param_idx[0], param_idx[1], param_idx[2], param_idx[3]]


@njit(cache=True, fastmath=True, parallel=True)
def unify_fold(
    fold_in: np.ndarray,
    p_arr_prev: types.ListType[types.Array],
    fold_out: np.ndarray,
    p_cart: np.ndarray,
    ffa_level: int,
    dp_funcs: FFASearchDPFunctions,
    load_func: Callable[[np.ndarray, int, np.ndarray], np.ndarray],
) -> None:
    for iparam_set in prange(len(p_cart)):
        p_set = p_cart[iparam_set]
        p_idx0, phase_shift0 = dp_funcs.resolve(p_set, p_arr_prev, ffa_level, 0)
        p_idx1, phase_shift1 = dp_funcs.resolve(p_set, p_arr_prev, ffa_level, 1)
        for ipair in range(fold_out.shape[0]):
            fold0 = dp_funcs.shift(load_func(fold_in, ipair * 2, p_idx0), phase_shift0)
            fold1 = dp_funcs.shift(
                load_func(fold_in, ipair * 2 + 1, p_idx1),
                phase_shift1,
            )
            fold_out[ipair, iparam_set] = dp_funcs.add(fold0, fold1)


class DynamicProgramming:
    def __init__(
        self,
        ts_data: TimeSeries,
        cfg: PulsarSearchConfig,
        data_type: npt.DTypeLike = np.float32,
    ) -> None:
        self.ts_data = ts_data
        self.data_type = data_type
        self._cfg = cfg
        self._dp_funcs = FFASearchDPFunctions(cfg)
        self._load_func = self._set_load_func(cfg.nparams)

        self.time_init = 0.0
        self.time_step = 0.0
        self.time_cart = 0.0
        self.time_fold = 0.0
        self.time_total = 0.0

    @property
    def cfg(self) -> PulsarSearchConfig:
        return self._cfg

    @property
    def dp_funcs(self) -> FFASearchDPFunctions:
        return self._dp_funcs

    @property
    def load_func(self) -> Callable[[np.ndarray, int, np.ndarray], np.ndarray]:
        return self._load_func

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
    def param_steps(self) -> types.ListType:
        return self._param_steps

    @property
    def param_arr(self) -> types.ListType[types.Array]:
        return self._param_arr

    @property
    def nparam_vol(self) -> int:
        if self.param_arr is None:
            return 0
        return int(np.prod(list(map(len, self.param_arr))))

    @property
    def dparams(self) -> np.ndarray:
        return self._dparams

    @property
    def chunk_duration(self) -> float:
        return self._chunk_duration

    def initialize(self) -> None:
        tstart = time.time()
        self._ffa_level = 0
        logger.info("Initializing data structure...")
        param_steps = self.dp_funcs.step(self.ffa_level)
        logger.info(f"param steps: {param_steps}")
        param_arr = self.cfg.get_param_arr(param_steps)
        # Check if the param_arr is correctly set
        self._check_param_arr(param_arr)
        fold = self.dp_funcs.init(self.ts_data.ts_e, self.ts_data.ts_v, param_arr)
        # Reshape fold to have the correct number of dimensions
        fold = np.expand_dims(fold, axis=list(range(1, self.cfg.nparams)))
        # Pack the fold to the correct level
        fold = self.dp_funcs.pack(fold, self.ffa_level)
        logger.info(f"fold dimensions: {fold.shape}")

        self._fold = fold.astype(self.data_type)
        self._param_arr = param_arr
        self._dparams = param_steps
        self._chunk_duration = self.cfg.tsegment
        self.time_init += time.time() - tstart
        logger.info(f"Initialization time: {self.time_init}")

    def ffa_iter(self) -> None:
        tstart = time.time()
        self._ffa_level += 1
        param_steps = self.dp_funcs.step(self.ffa_level)
        logger.info(f"param steps: {param_steps}")
        param_arr_new = self.cfg.get_param_arr(param_steps)
        self.time_step += time.time() - tstart
        tstart = time.time()
        param_cart_new = np_utils.cartesian_prod_st(param_arr_new)
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
            self.dp_funcs,
            self.load_func,
        )
        self.time_fold += time.time() - tstart
        self._fold = fold_new.reshape(
            (self.nchunks // 2, *list(map(len, param_arr_new)), *self.fold.shape[-2:]),
        )
        self._param_steps = param_steps
        self._param_arr = param_arr_new
        self._chunk_duration *= 2

    def ffa_iter_dry(self) -> int:
        self._ffa_level += 1
        param_steps = self.dp_funcs.step(self.ffa_level)
        logger.info(f"param steps: {param_steps}")
        param_arr_new = self.cfg.get_param_arr(param_steps)
        complexity = int(np.prod(list(map(len, param_arr_new))))
        self._param_steps = param_steps
        self._param_arr = param_arr_new
        self._chunk_duration *= 2
        return complexity

    def execute(self, n_iters: str | int = "max") -> None:
        tstart = time.time()
        if n_iters == "max":
            n_iters = int(np.log2(len(self.fold)))
        elif n_iters == "prune":
            n_iters = int(np.log2(len(self.fold))) - 7
        for _ in range(int(n_iters)):
            self.ffa_iter()
            logger.info(
                f"iteration: {self.ffa_level}, fold dimensions: {self.fold.shape}",
            )
        self.time_total += time.time() - tstart

    def do_iterations_dry(self, n_iters: str | int = "max") -> list[int]:
        if n_iters == "max":
            n_iters = int(np.log2(len(self.fold)))
        elif n_iters == "prune":
            n_iters = int(np.log2(len(self.fold))) - 7
        complexity = []
        for _ in range(int(n_iters)):
            logger.info(f"performing iteration: {self.ffa_level + 1}")
            complexity.append(self.ffa_iter_dry())
        return complexity

    def get_fold_norm(
        self,
        iseg: int = 0,
        param_idx: tuple | None = None,
    ) -> np.ndarray:
        if param_idx is None:
            fold = self.fold[iseg]
        else:
            fold = self.load_func(self.fold, iseg, np.array(param_idx))
        return fold[..., 0, :] / np.sqrt(fold[..., 1, :])

    def _set_load_func(
        self,
        nparams: int,
    ) -> Callable[[np.ndarray, int, np.ndarray], np.ndarray]:
        nparams_to_load_func = {
            1: load_folds_1d,
            2: load_folds_2d,
            3: load_folds_3d,
            4: load_folds_4d,
        }
        return nparams_to_load_func[nparams]

    def _check_param_arr(self, param_arr: types.ListType[types.Array]) -> None:
        if self.cfg.nparams > 1:
            for iparam in range(self.cfg.nparams - 1):
                nvals = len(param_arr[iparam])
                if nvals > 1:
                    msg = f"param_arr has {nvals} values, should have only one"
                    raise ValueError(msg)
