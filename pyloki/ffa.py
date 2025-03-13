from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pyloki.core import FFASearchDPFuncts, set_ffa_load_func, unify_fold
from pyloki.utils import np_utils
from pyloki.utils.misc import PicklableStructRefWrapper, get_logger, track_progress
from pyloki.utils.timing import Timer

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import typing as npt

    from pyloki.config import PulsarSearchConfig
    from pyloki.io.timeseries import TimeSeries


logger = get_logger(__name__)


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
        self._dp_funcs = PicklableStructRefWrapper[FFASearchDPFuncts](
            FFASearchDPFuncts,
            cfg,
        )
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
    def dp_funcs(self) -> FFASearchDPFuncts:
        return self._dp_funcs.get_instance()

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
    def param_arr(self) -> list[np.ndarray]:
        """:obj:`list[ndarray]`: Parameter array at the current FFA level."""
        return self._param_arr

    @property
    def param_arr_dict(self) -> dict[str, np.ndarray]:
        """:obj:`dict`: Dictionary of parameter arrays at the current FFA level."""
        param_names = self.cfg.param_names
        return {f"{param_names[i]}": p for i, p in enumerate(self.param_arr)}

    @property
    def nparam_vol(self) -> int:
        """:obj:`int`: Total complexity (parameter volume) at the current FFA level."""
        return int(np.prod([len(p) for p in self.param_arr])) if self.param_arr else 0

    @property
    def leaves_lb(self) -> int:
        """:obj:`int`: Number of leaves at the current FFA level."""
        return np.round(np.log2(self.nparam_vol), 2)

    @property
    def tseg(self) -> float:
        """:obj:`float`: Segment duration at the current FFA level."""
        return self._tseg

    @Timer(name="ffa_initialize", logger=logger.info)
    def initialize(self) -> None:
        """Initialize the data structure for the FFA search."""
        self._ffa_level = 0
        dparams = self.cfg.get_dparams(self.ffa_level)
        logger.info(f"FFA initialize: Grid sizes: {dparams}")
        param_arr = self.cfg.get_param_arr(dparams)
        self._check_init_param_arr(param_arr)

        # Initialize the fold structure and reshape to correct dimensions
        fold = self.dp_funcs.init(self.ts_data.ts_e, self.ts_data.ts_v, param_arr)
        fold = np.expand_dims(fold, axis=list(range(1, self.cfg.nparams)))
        fold = self.dp_funcs.pack(fold, self.ffa_level)

        self._fold = fold.astype(self.data_type)
        self._param_arr = param_arr
        self._dparams = dparams
        self._dparams_limited = self.cfg.get_dparams_limited(self.ffa_level)
        self._tseg = self.cfg.tseg_brute
        logger.info(
            f"ffa level: {self.ffa_level:2d}, leaves: {self.leaves_lb:.2f}, "
            f"fold dims: {self.fold.shape}",
        )

    @Timer(name="ffa_execute", logger=logger.info)
    def execute(self) -> None:
        """Execute the FFA search."""
        n_iters = self.cfg.niters_ffa
        for _ in track_progress(
            range(n_iters),
            description="Computing FFA",
            total=n_iters,
            get_leaves=lambda: self.leaves_lb,
        ):
            self._execute_iter()
            logger.info(
                f"ffa level: {self.ffa_level:2d}, leaves: {self.leaves_lb:.2f}, "
                f"fold dims: {self.fold.shape}",
            )
        logger.info(f"FFA complete: Grid sizes: {self.dparams}")

    def _execute_iter(self) -> None:
        """Execute a single iteration of the FFA search."""
        self._ffa_level += 1
        dparams = self.cfg.get_dparams(self.ffa_level)
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
            dparams = self.cfg.get_dparams(ffa_level)
            complexity.append(
                int(np.prod([len(p) for p in self.cfg.get_param_arr(dparams)])),
            )
        return complexity

    def _check_init_param_arr(self, param_arr: list[np.ndarray]) -> None:
        if self.cfg.nparams > 1:
            for iparam in range(self.cfg.nparams - 1):
                nvals = len(param_arr[iparam])
                if nvals > 1:
                    msg = f"param_arr has {nvals} values, should have only one"
                    raise ValueError(msg)
