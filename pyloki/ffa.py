from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, prange, types

from pyloki.core import FFASearchDPFunctions, set_ffa_load_func
from pyloki.utils import np_utils
from pyloki.utils.misc import get_logger
from pyloki.utils.timing import Timer

if TYPE_CHECKING:
    from typing import Callable

    from numpy import typing as npt

    from pyloki.config import PulsarSearchConfig
    from pyloki.io.timeseries import TimeSeries


logger = get_logger(__name__)


@Timer(name="unify_fold")
@njit(fastmath=True, parallel=True)
def unify_fold(
    fold_in: np.ndarray,
    param_arr_prev: types.ListType[types.Array],
    fold_out: np.ndarray,
    param_cart_cur: np.ndarray,
    ffa_level: int,
    dp_funcs: FFASearchDPFunctions,
    load_func: Callable[[np.ndarray, int, np.ndarray], np.ndarray],
) -> None:
    """Unify the fold by combining the two folds from the previous level.

    Parameters
    ----------
    fold_in : np.ndarray
        Input fold structure from the previous level.
    param_arr_prev : types.ListType[types.Array]
        Parameter array from the previous level.
    fold_out : np.ndarray
        Output fold structure for the current level.
    param_cart_cur : np.ndarray
        Cartesian product of the parameter array for the current level.
    ffa_level : int
        Current level of the FFA search.
    dp_funcs : FFASearchDPFunctions
        A container for the dynamic programming functions.
    load_func : Callable[[np.ndarray, int, np.ndarray], np.ndarray]
        A function to load the fold from the input structure.
    """
    for iparam_set in prange(len(param_cart_cur)):
        p_set = param_cart_cur[iparam_set]

        # Resolve parameters for tail and head
        p_idx_tail, phase_shift_tail = dp_funcs.resolve(
            p_set,
            param_arr_prev,
            ffa_level,
            0,
        )
        p_idx_head, phase_shift_head = dp_funcs.resolve(
            p_set,
            param_arr_prev,
            ffa_level,
            1,
        )
        for ipair in range(fold_out.shape[0]):
            fold_tail = dp_funcs.shift(
                load_func(fold_in, ipair * 2, p_idx_tail),
                phase_shift_tail,
            )
            fold_head = dp_funcs.shift(
                load_func(fold_in, ipair * 2 + 1, p_idx_head),
                phase_shift_head,
            )
            fold_out[ipair, iparam_set] = dp_funcs.add(fold_tail, fold_head)


class DynamicProgramming:
    """Dynamic Programming class for the FFA search.

    Parameters
    ----------
    ts_data : TimeSeries
        Input time series data.
    cfg : PulsarSearchConfig
        A configuration object for the FFA search.
    data_type : npt.DTypeLike, optional
        Data type for the FFA search, by default np.float32.
    """

    def __init__(
        self,
        ts_data: TimeSeries,
        cfg: PulsarSearchConfig,
        data_type: npt.DTypeLike = np.float32,
    ) -> None:
        self._ts_data = ts_data
        self._cfg = cfg
        self._data_type = data_type
        self._dp_funcs = FFASearchDPFunctions(cfg)
        self._load_func = set_ffa_load_func(cfg.nparams)

    @property
    def ts_data(self) -> TimeSeries:
        return self._ts_data

    @property
    def cfg(self) -> PulsarSearchConfig:
        return self._cfg

    @property
    def data_type(self) -> npt.DTypeLike:
        return self._data_type

    @property
    def dp_funcs(self) -> FFASearchDPFunctions:
        return self._dp_funcs

    @property
    def load_func(self) -> Callable[[np.ndarray, int, np.ndarray], np.ndarray]:
        return self._load_func

    @property
    def fold(self) -> np.ndarray:
        """obj:`~numpy.ndarray`: Fold structure for the FFA search."""
        return self._fold

    @property
    def nsegments(self) -> int:
        """obj:`int`: Number of segments in the fold structure."""
        return self._fold.shape[0]

    @property
    def nbins(self) -> int:
        """obj:`int`: Number of bins in the fold structure."""
        return self._fold.shape[-1]

    @property
    def ffa_level(self) -> int:
        """:obj:`int`: Current level of the Dynamic Programming search."""
        return self._ffa_level

    @property
    def dparams(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: Paramater step sizes at the current FFA level."""
        return self._dparams

    @property
    def dparams_limited(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: Paramater step sizes at the current FFA level."""
        return self._dparams_limited

    @property
    def param_arr(self) -> types.ListType[types.Array]:
        """:obj:`list[ndarray]`: Parameter array at the current FFA level."""
        return self._param_arr

    @property
    def nparam_vol(self) -> int:
        """:obj:`int`: Total complexity (parameter volume) at the current FFA level."""
        return int(np.prod([len(p) for p in self.param_arr])) if self.param_arr else 0

    @property
    def tseg(self) -> float:
        """:obj:`float`: Segment duration at the current FFA level."""
        return self._tseg

    @Timer(name="ffa_initialize", logger=logger.info)
    def initialize(self) -> None:
        """Initialize the data structure for the FFA search."""
        logger.info("Initializing data structure...")
        self._ffa_level = 0
        dparams = self.dp_funcs.step(self.ffa_level)
        logger.info(f"param steps: {dparams}")
        param_arr = self.cfg.get_param_arr(dparams)
        self._check_init_param_arr(param_arr)

        # Initialize the fold structure and reshape to correct dimensions
        fold = self.dp_funcs.init(self.ts_data.ts_e, self.ts_data.ts_v, param_arr)
        fold = np.expand_dims(fold, axis=list(range(1, self.cfg.nparams)))
        fold = self.dp_funcs.pack(fold, self.ffa_level)
        logger.info(f"fold dimensions: {fold.shape}")

        self._fold = fold.astype(self.data_type)
        self._param_arr = param_arr
        self._dparams = dparams
        self._dparams_limited = self.cfg.get_dparams_limited(self.ffa_level)
        self._tseg = self.cfg.tseg_brute

    @Timer(name="ffa_execute", logger=logger.info)
    def execute(self) -> None:
        """Execute the FFA search."""
        n_iters = self.cfg.niters_ffa
        for _ in range(int(n_iters)):
            self._execute_iter()
            logger.info(f"i_iter: {self.ffa_level}, fold dims: {self.fold.shape}")

    def _execute_iter(self) -> None:
        """Execute a single iteration of the FFA search."""
        self._ffa_level += 1
        dparams = self.dp_funcs.step(self.ffa_level)
        logger.info(f"param steps: {dparams}")
        param_arr_cur = self.cfg.get_param_arr(dparams)

        param_cart_cur = np_utils.cartesian_prod_st(param_arr_cur)
        fold_cur = np.zeros(
            (self.nsegments // 2, len(param_cart_cur), *self.fold.shape[-2:]),
            self.fold.dtype,
        )
        unify_fold(
            self.fold,
            self.param_arr,
            fold_cur,
            param_cart_cur,
            self.ffa_level,
            self.dp_funcs,
            self.load_func,
        )
        self._fold = fold_cur.reshape(
            (
                self.nsegments // 2,
                *list(map(len, param_arr_cur)),
                *self.fold.shape[-2:],
            ),
        )
        self._param_arr = param_arr_cur
        self._dparams = dparams
        self._dparams_limited = self.cfg.get_dparams_limited(self.ffa_level)
        self._tseg *= 2

    def get_fold_norm(
        self,
        iseg: int = 0,
        param_idx: tuple | None = None,
    ) -> np.ndarray:
        """Get the normalized fold for the given segment and parameter index.

        Parameters
        ----------
        iseg : int, optional
            Segment index, by default 0.
        param_idx : tuple | None, optional
            Parameter index, by default None.

        Returns
        -------
        np.ndarray
            Normalized fold for the given segment and parameter index.
        """
        if param_idx is None:
            fold = self.fold[iseg]
        else:
            fold = self.load_func(self.fold, iseg, np.array(param_idx))
        return fold[..., 0, :] / np.sqrt(fold[..., 1, :])

    def get_complexity(self) -> list[int]:
        """Get the complexity (number of parameters to search) per ffa level.

        Returns
        -------
        list[int]
            List with the complexity per FFA level.
        """
        complexity = []
        for ffa_level in range(self.cfg.niters_ffa + 1):
            dparams = self.dp_funcs.step(ffa_level)
            complexity.append(
                int(np.prod([len(p) for p in self.cfg.get_param_arr(dparams)])),
            )
        return complexity

    def _check_init_param_arr(self, param_arr: types.ListType[types.Array]) -> None:
        if self.cfg.nparams > 1:
            for iparam in range(self.cfg.nparams - 1):
                nvals = len(param_arr[iparam])
                if nvals > 1:
                    msg = f"param_arr has {nvals} values, should have only one"
                    raise ValueError(msg)
