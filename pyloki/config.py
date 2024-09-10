from __future__ import annotations

import numpy as np
from numba import typed, types
from numba.experimental import jitclass

from pyloki.core import common
from pyloki.utils import math


@jitclass(
    spec=[
        ("nsamps", types.i8),
        ("tsamp", types.f8),
        ("nbins", types.i8),
        ("tol", types.f8),
        ("param_limits", types.ListType(types.Tuple([types.f8, types.f8]))),
        ("bseg_brute", types.i8),
        ("bseg_ffa", types.i8),
        ("prune_poly_order", types.i8),
        ("prune_n_derivs", types.i8),
    ],
)
class PulsarSearchConfig:
    """
    Class to hold the configuration for the polynomial search.

    Parameters
    ----------
    nsamps : int
        Number of samples in the time series to be searched
    tsamp : float
        Sampling time of the time series (in seconds)
    nbins : int
        Number of bins to keep in the folded profile
    tol : float
        Tolerance (in bins) for the polynomial search
        (in units of bins across the pulsar ducy)
    param_limits : types.ListType[types.Tuple[float, float]]
        List of tuples with the min and max values for each search parameter
    bseg_brute : int, optional
        Segment length (in bins) at the end of brute-fold stage. If not
        provided (i.e = 0), a default value is used.
    bseg_ffa : int, optional
        Segment length (in bins) at the end of FFA stage. If not
        provided (i.e = 0), a default value is used.
    prune_poly_order : int, optional
        Maximum polynomial order to use in the pruning stage, by default 3
    prune_n_derivs : int, optional
        Number of derivatives to validate in the pruning stage, by default 3

    Notes
    -----
    The parameter limits are assumed to be in the order: ..., jerk, accel, freq.
    """

    def __init__(
        self,
        nsamps: int,
        tsamp: float,
        nbins: int,
        tol: float,
        param_limits: types.ListType[types.Tuple[float, float]],
        bseg_brute: int = 0,
        bseg_ffa: int = 0,
        prune_poly_order: int = 3,
        prune_n_derivs: int = 3,
    ) -> None:
        self.nsamps = nsamps
        self.tsamp = tsamp
        self.nbins = nbins
        self.tol = tol
        self.param_limits = param_limits
        self.bseg_brute = bseg_brute if bseg_brute != 0 else self._bseg_brute_default()
        self.bseg_ffa = bseg_ffa if bseg_ffa != 0 else self._bseg_ffa_default()
        self.prune_poly_order = prune_poly_order
        self.prune_n_derivs = prune_n_derivs
        self._validate_params()

    @property
    def tseg_brute(self) -> float:
        """:obj:`float`: Segment length (in seconds) at the end of brute-fold stage."""
        return self.bseg_brute * self.tsamp

    @property
    def tseg_ffa(self) -> float:
        """:obj:`float`: Segment length (in seconds) at the end of FFA stage."""
        return self.bseg_ffa * self.tsamp

    @property
    def niters_ffa(self) -> int:
        """:obj:`int`: Number of FFA iterations to perform."""
        return int(np.log2(self.bseg_ffa / self.bseg_brute))

    @property
    def nparams(self) -> int:
        """:obj:`int`: Number of parameters in the search."""
        return len(self.param_limits)

    @property
    def f_min(self) -> float:
        """:obj:`float`: Minimum frequency value to search."""
        return self.param_limits[-1][0]

    @property
    def f_max(self) -> float:
        """:obj:`float`: Maximum frequency value to search."""
        return self.param_limits[-1][1]

    def get_dparams(self, ffa_level: int) -> np.ndarray:
        """
        Get the parameter step sizes for the given FFA level.

        Parameters
        ----------
        ffa_level : int
            FFA level for which to compute the parameter steps

        Returns
        -------
        np.ndarray
            Array with the parameter step sizes
        """
        tseg_cur = 2**ffa_level * self.tseg_brute
        dparams = np.zeros(self.nparams, dtype=np.float64)
        dparams[-1] = self._freq_step(tseg_cur)
        for deriv in range(2, self.nparams + 1):
            dparams[-deriv] = self._deriv_step(tseg_cur, deriv)
        return dparams

    def get_dparams_limited(self, ffa_level: int) -> np.ndarray:
        dparams = self.get_dparams(ffa_level)
        dparams_lim = np.zeros_like(dparams)
        for iparam in range(self.nparams):
            if iparam == self.nparams - 1:
                dparams_lim[iparam] = dparams[iparam]
            dparam_diff = self.param_limits[iparam][1] - self.param_limits[iparam][0]
            dparams_lim[iparam] = min(dparam_diff, dparams[iparam])
        return dparams_lim

    def get_param_arr(self, dparams: np.ndarray) -> types.ListType[types.Array]:
        """
        Get the parameter ranges for the given parameter steps.

        Parameters
        ----------
        dparams : np.ndarray
            Array with the parameter step sizes

        Returns
        -------
        types.ListType[types.Array]
            List of arrays with the parameter ranges

        Raises
        ------
        ValueError
            If the length of `dparams` is not equal to the number of parameters
        """
        if len(dparams) != self.nparams:
            msg = f"dparams must have length {self.nparams}, got {len(dparams)}"
            raise ValueError(msg)
        return typed.List(
            [
                common.range_param(*self.param_limits[iparam], dparams[iparam])
                for iparam in range(self.nparams)
            ],
        )

    def _freq_step(self, tobs: float) -> float:
        return common.freq_step(tobs, self.nbins, self.f_max, self.tol)

    def _deriv_step(self, tobs: float, deriv: int) -> float:
        t_ref = tobs / 2
        return common.param_step(tobs, self.tsamp, deriv, self.tol, t_ref=t_ref)

    def _bseg_brute_default(self) -> int:
        init_levels = 1 if self.nparams == 1 else 5
        levels = int(np.log2(self.nsamps * self.tsamp * self.f_max) - init_levels)
        return int(self.nsamps / 2**levels)

    def _bseg_ffa_default(self) -> int:
        return self.nsamps

    def _validate_params(self) -> None:
        if self.nsamps <= 0 or not math.is_power_of_two(self.nsamps):
            msg = f"nsamps must be a power of two and > 0, got {self.nsamps}"
            raise ValueError(msg)
        if not math.is_power_of_two(self.bseg_brute):
            msg = f"bseg_brute must be a power of two, got {self.bseg_brute}"
            raise ValueError(msg)
        if self.bseg_brute > self.nsamps:
            msg = f"bseg_brute must be less than nsamps, got {self.bseg_brute}"
            raise ValueError(msg)
        if not math.is_power_of_two(self.bseg_ffa):
            msg = f"bseg_ffa must be a power of two, got {self.bseg_ffa}"
            raise ValueError(msg)
        if self.bseg_ffa > self.nsamps:
            msg = f"bseg_ffa must be less than nsamps, got {self.bseg_ffa}"
            raise ValueError(msg)
        if self.bseg_ffa <= self.bseg_brute:
            msg = f"bseg_ffa must be greater than bseg_brute, got {self.bseg_ffa}"
            raise ValueError(msg)

        if self.nparams < 1 or self.nparams > 4:
            msg = f"param_limits must have 1-4 elements, got {self.nparams}"
            raise ValueError(msg)
        for _, (val_min, val_max) in enumerate(self.param_limits):
            if val_min >= val_max:
                msg = f"param_limits must have min < max, got {self.param_limits}"
                raise ValueError(msg)
