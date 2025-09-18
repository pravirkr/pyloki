from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import typed

from pyloki.core import (
    FFATaylorComplexDPFuncts,
    FFATaylorDPFuncts,
    set_ffa_load_func,
    unify_fold,
)
from pyloki.core.fold import ffa_taylor_resolve
from pyloki.utils import np_utils
from pyloki.utils.misc import (
    PicklableStructRefWrapper,
    get_logger,
    quiet_logger,
    track_progress,
)
from pyloki.utils.timing import Timer

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy import typing as npt

    from pyloki.config import PulsarSearchConfig
    from pyloki.io.timeseries import TimeSeries

DP_FUNCS_TYPE = FFATaylorDPFuncts | FFATaylorComplexDPFuncts

logger = get_logger(__name__)

ffacoord_dtype = np.dtype(
    [
        ("i_tail", np.uint32),
        ("shift_tail", np.float32),
        ("i_head", np.uint32),
        ("shift_head", np.float32),
    ],
)

ffacoordfreq_dtype = np.dtype(
    [
        ("idx", np.uint32),
        ("shift", np.float32),
    ],
)


class FFFPlan:
    """FFA Plan class for the FFA search.

    Parameters
    ----------
    ts_data : TimeSeries
        Input time series data.
    cfg : PulsarSearchConfig
        A configuration object for the FFA search.
    """

    def __init__(self, cfg: PulsarSearchConfig) -> None:
        self._cfg = cfg
        self.configure_plan()

    @property
    def cfg(self) -> PulsarSearchConfig:
        """Get the configuration object for the FFA search."""
        return self._cfg

    def configure_plan(self) -> None:
        """Configure the FFA plan based on the provided configuration."""
        levels = self.cfg.niters_ffa + 1
        self.n_params = self.cfg.nparams
        self.n_levels = levels
        self.segment_lens = np.empty(levels, dtype=np.float32)
        self.nsegments = np.empty(levels, dtype=np.int32)
        self.tsegments = np.empty(levels, dtype=np.float32)
        self.ncoords = np.empty(levels, dtype=np.int32)
        self.ncoords_lb = np.empty(levels, dtype=np.float32)
        self.dparams = np.empty((levels, self.n_params), dtype=np.float32)
        self.dparams_limited = np.empty((levels, self.n_params), dtype=np.float32)
        self.params = []
        for i_level in range(levels):
            segment_len = self.cfg.tseg_brute * (2**i_level)
            tsegment = segment_len * self.cfg.tsamp
            nsegments_cur = self.cfg.nsamps // segment_len
            dparam_arr = self.cfg.get_dparams(i_level)
            dparam_arr_lim = self.cfg.get_dparams(i_level)
            param_arr = self.cfg.get_param_arr(dparam_arr)
            self.segment_lens[i_level] = segment_len
            self.nsegments[i_level] = nsegments_cur
            self.tsegments[i_level] = tsegment
            self.dparams[i_level, :] = dparam_arr
            self.dparams_limited[i_level, :] = dparam_arr_lim
            self.ncoords[i_level] = np.prod([len(p) for p in param_arr])
            self.ncoords_lb[i_level] = np.round(np.log2(self.ncoords[i_level]), 2)
            self.params.append(param_arr)

    def resolve_coordinates(self) -> list[np.ndarray]:
        """Resolve the coordinates for each level of the FFA search.

        Returns
        -------
        list[np.ndarray]
            List of parameter arrays for each level.
        """
        coords = []
        for i_level in range(self.n_levels):
            param_arr = self.params[i_level]
            param_arr_prev = self.params[i_level - 1]
            param_cart_cur = np_utils.cartesian_prod_st(param_arr)
            if self.n_params == 1:
                # Special case for single parameter
                coords_cur = np.zeros(len(param_cart_cur), dtype=ffacoordfreq_dtype)
                for iparam_set in range(len(param_cart_cur)):
                    p_set = param_cart_cur[iparam_set]
                    p_idx, shift = ffa_taylor_resolve(
                        p_set,
                        param_arr_prev,
                        i_level,
                        1,
                        self.cfg.tseg_brute,
                        self.cfg.nbins,
                    )
                    # Flatten the p_idx
                    coords_cur[iparam_set] = (p_idx[0], shift)
            else:
                lengths = [len(a) for a in param_arr_prev]
                param_strides_prev = np.cumprod([1, *lengths[::-1][:-1]])[::-1]
                coords_cur = np.zeros(len(param_cart_cur), dtype=ffacoord_dtype)
                for iparam_set in range(len(param_cart_cur)):
                    p_set = param_cart_cur[iparam_set]
                    # Resolve parameters for tail and head
                    p_idx_tail, shift_tail = ffa_taylor_resolve(
                        p_set,
                        param_arr_prev,
                        i_level,
                        0,
                        self.cfg.tseg_brute,
                        self.cfg.nbins,
                    )
                    p_idx_head, shift_head = ffa_taylor_resolve(
                        p_set,
                        param_arr_prev,
                        i_level,
                        1,
                        self.cfg.tseg_brute,
                        self.cfg.nbins,
                    )
                    # Flatten the p_idx
                    p_idx_tail_flat = sum(
                        i * s
                        for i, s in zip(p_idx_tail, param_strides_prev, strict=False)
                    )
                    p_idx_head_flat = sum(
                        i * s
                        for i, s in zip(p_idx_head, param_strides_prev, strict=False)
                    )
                    # Store the coordinates
                    coords_cur[iparam_set] = (
                        p_idx_tail_flat,
                        shift_tail,
                        p_idx_head_flat,
                        shift_head,
                    )
            coords.append(coords_cur)
        return coords


class DynamicProgramming:
    """Dynamic Programming class for the FFA search.

    Parameters
    ----------
    ts_data : TimeSeries
        Input time series data.
    cfg : PulsarSearchConfig
        A configuration object for the FFA search.
    """

    def __init__(self, ts_data: TimeSeries, cfg: PulsarSearchConfig) -> None:
        self._ts_data = ts_data
        self._cfg = cfg
        if cfg.use_fft_shifts:
            self._data_type = np.complex64
            self._dp_funcs = PicklableStructRefWrapper[DP_FUNCS_TYPE](
                FFATaylorComplexDPFuncts,
                cfg,
            )
        else:
            self._data_type = np.float32  # type: ignore[assignment]
            self._dp_funcs = PicklableStructRefWrapper[DP_FUNCS_TYPE](
                FFATaylorDPFuncts,
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
    def dp_funcs(self) -> DP_FUNCS_TYPE:
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
        param_arr_t = typed.List(param_arr)
        fold = self.dp_funcs.init(self.ts_data.ts_e, self.ts_data.ts_v, param_arr_t)
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
    def execute(self, *, show_progress: bool = True) -> None:
        """Execute the FFA search."""
        n_iters = self.cfg.niters_ffa
        for _ in track_progress(
            range(n_iters),
            description="Computing FFA",
            total=n_iters,
            get_leaves=lambda: self.leaves_lb,
            show_progress=show_progress,
        ):
            self._execute_iter()
            logger.info(
                f"ffa level: {self.ffa_level:2d}, leaves: {self.leaves_lb:5.2f}, "
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
        param_arr_t = typed.List(self.param_arr)
        unify_fold(
            self.fold,
            param_arr_t,
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
        if self.cfg.use_fft_shifts:
            logger.info("Using FFT for phase shifts: Inverse FFTing the fold")
            logger.info(f"Fold shape: {self.fold.shape}")
            fold_t = np.fft.irfft(self.fold).astype(np.float32)
        else:
            fold_t = self.fold
        if param_idx is None:
            fold = fold_t[iseg]
        else:
            fold = self.load_func(fold_t, iseg, np.array(param_idx))
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


def compute_ffa(
    tseries: TimeSeries,
    search_cfg: PulsarSearchConfig,
    *,
    quiet: bool = False,
    show_progress: bool = False,
) -> np.ndarray:
    """Compute the FFA for a given configuration.

    Parameters
    ----------
    tseries : TimeSeries
        Input time series data.
    search_cfg : PulsarSearchConfig
        A configuration object for the FFA search.
    quiet : bool, optional
        If True, suppresses logging output, by default False.
    show_progress : bool, optional
        Whether to show progress bar, by default False.

    Returns
    -------
    DynamicProgramming
        The dynamic programming object with the FFA results.
    """
    with quiet_logger(quiet=quiet):
        dyp = DynamicProgramming(tseries, search_cfg)
        dyp.initialize()
        dyp.execute(show_progress=show_progress)
    return dyp.fold
