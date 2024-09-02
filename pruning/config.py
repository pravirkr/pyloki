from __future__ import annotations

import numpy as np
from numba import typed, types
from numba.experimental import jitclass

from pruning.core import common


@jitclass(
    spec=[
        ("nsamps", types.i8),
        ("tsamp", types.f8),
        ("nbins", types.i8),
        ("tolerance", types.f8),
        ("param_limits", types.ListType(types.Tuple([types.f8, types.f8]))),
        ("segment_len", types.i8),
    ],
)
class PulsarSearchConfig:
    """Class to hold the configuration for the polynomial search.

    Parameters
    ----------
    nsamps : int
        Number of samples in the time series
    tsamp : float
        Sampling time of the time series
    nbins : int
        Number of bins in the folded time series
    tolerance : float
        Tolerance (in bins) for the polynomial search
        (in units of number of time bins across the pulsar ducy)
    param_limits : types.ListType
        List of tuples with the min and max values for each search parameter
    segment_len : int, optional
        Length of the segment to use for the initial search bruefold search.

    Notes
    -----
    The parameter limits are assumed to be in the order: ..., acceleration, period.
    If segment_len is not provided i.e = 0, it is calculated automatically to be
    optimal for the search.
    """

    def __init__(
        self,
        nsamps: int,
        tsamp: float,
        nbins: int,
        tolerance: float,
        param_limits: types.ListType[types.Tuple],
        segment_len: int = 0,
    ) -> None:
        self.nsamps = nsamps
        self.tsamp = tsamp
        self.nbins = nbins
        self.tolerance = tolerance
        self.param_limits = param_limits
        self._check_params()

        if segment_len == 0:
            self.segment_len = self._segment_len_default()
        else:
            self.segment_len = segment_len

    @property
    def tsegment(self) -> float:
        return self.segment_len * self.tsamp

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
        return common.freq_step(tobs, self.nbins, self.f_max, self.tolerance)

    def deriv_step(self, tobs: float, deriv: int) -> float:
        t_ref = tobs / 2
        return common.param_step(tobs, self.tsamp, deriv, self.tolerance, t_ref=t_ref)

    def get_dparams(self, ffa_level: int) -> np.ndarray:
        tsegment_cur = 2**ffa_level * self.tsegment
        dparams = np.zeros(self.nparams, dtype=np.float64)
        dparams[-1] = self.freq_step(tsegment_cur)
        for deriv in range(2, self.nparams + 1):
            dparams[-deriv] = self.deriv_step(tsegment_cur, deriv)
        return dparams

    def get_updated_dparams(self, param_steps: types.ListType) -> np.ndarray:
        if len(param_steps) != self.nparams:
            msg = f"param_steps must have length {self.nparams}, got {len(param_steps)}"
            raise ValueError(msg)
        dparams = np.zeros_like(self.dparams)
        for iparam in range(self.nparams):
            if iparam == self.nparams - 1:
                dparams[iparam] = param_steps[iparam]
            dparams[iparam] = min(self.dparams[iparam], param_steps[iparam])
        return dparams

    def get_param_arr(self, param_steps: np.ndarray) -> types.ListType[types.Array]:
        if len(param_steps) != self.nparams:
            msg = f"param_steps must have length {self.nparams}, got {len(param_steps)}"
            raise ValueError(msg)
        return typed.List(
            [
                common.range_param(*self.param_limits[iparam], param_steps[iparam])
                for iparam in range(self.nparams)
            ],
        )

    def _segment_len_default(self) -> int:
        init_levels = 1 if self.nparams == 1 else 5
        levels = int(np.log2(self.nsamps * self.tsamp * self.f_max) - init_levels)
        return int(self.nsamps / 2**levels)

    def _check_params(self) -> None:
        if self.nparams < 1 or self.nparams > 4:
            msg = f"param_limits must have 1-4 elements, got {len(self.param_limits)}"
            raise ValueError(msg)
        for _, (val_min, val_max) in enumerate(self.param_limits):
            if val_min >= val_max:
                msg = f"param_limits must have min < max, got {self.param_limits}"
                raise ValueError(msg)


