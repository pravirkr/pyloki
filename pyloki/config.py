from __future__ import annotations

import itertools

import numpy as np
from numba import typed, types

from pyloki import kepler
from pyloki.utils import math, psr_utils
from pyloki.utils.misc import C_VAL


class ParamLimits:
    """Class to hold the search parameter limits/bounds.

    Parameters
    ----------
    limits : types.ListType[types.Tuple[float, float]]
        List of tuples with the min and max values for each search parameter.
        Should be in the order: ..., jerk, accel, freq.
        Paramaters are defined at t=t_c (center of the observation).
    """

    def __init__(self, limits: types.ListType[types.Tuple[float, float]]) -> None:
        self.limits = limits

    def get_cheby_limits(
        self,
        tobs: float,
    ) -> types.ListType[types.Tuple[float, float]]:
        """Get the corresponding Chebyshev coefficient bounds.

        The order is reversed from the Taylor coefficients.

        Parameters
        ----------
        tobs : float
            Total observation time (in seconds).

        Returns
        -------
        types.ListType[types.Tuple[float, float]]
            List of tuples with the Chebyshev coefficient bounds.
            Should be in the order: [freq, alpha_0, alpha_1, ..., alpha_n], where
            n is the highest polynomial order in the search.
        """
        t_s = tobs / 2
        # Get kinematic terms only (ignore freq), in increasing derivative order
        d_limits = self.limits[:-1][::-1]
        d_limits = [(0, 0), (0, 0), *d_limits]
        d_corners = np.array(list(itertools.product(*list(d_limits))))
        alpha_corners = np.vstack(
            [math.taylor_to_cheby(d_vec, t_s) for d_vec in d_corners],
        )
        alpha_bounds = list(
            zip(
                np.min(alpha_corners, axis=0),
                np.max(alpha_corners, axis=0),
                strict=False,
            ),
        )
        return typed.List([self.limits[-1], *alpha_bounds])

    @classmethod
    def from_taylor(
        cls,
        freq: tuple[float, float],
        accel: tuple[float, float] | None = None,
        jerk: tuple[float, float] | None = None,
        snap: tuple[float, float] | None = None,
    ) -> ParamLimits:
        """Generate search parameter limits from Taylor series kinematic parameters.

        Parameters
        ----------
        freq : tuple[float, float]
            Frequency range to search (min, max).
        accel : tuple[float, float] | None, optional
            Acceleration range to search (min, max), by default None
        jerk : tuple[float, float] | None, optional
            Jerk range to search (min, max), by default None
        snap : tuple[float, float] | None, optional
            Snap range to search (min, max), by default None

        Returns
        -------
        ParamLimits
            Object with the search parameter limits.
        """
        default_limit = (0.0, 0.0)
        all_params = [snap, jerk, accel, freq]
        last_non_none = next(
            i for i, param in enumerate(all_params) if param is not None
        )
        out = [
            param if param is not None else default_limit
            for param in all_params[last_non_none:]
        ]
        out = [(float(min_val), float(max_val)) for min_val, max_val in out]

        return cls(typed.List(out))

    @classmethod
    def from_circular(
        cls,
        freq: float,
        poly_order: int,
        p_orb_min: float,
        m_c: float,
        m_p: float = 1.4,
    ) -> ParamLimits:
        """Generate search parameter limits from circular orbit parameters.

        Parameters
        ----------
        freq : float
            Expected frequency of the orbit (in Hz).
        poly_order : int
            Highest polynomial order to include in the search.
        p_orb_min : float
            Minimum orbital period (in seconds).
        m_c : float
            Mass of the companion (in solar masses).
        m_p : float, optional
            Mass of the pulsar (in solar masses), by default 1.4.

        Returns
        -------
        ParamLimits
            Object with the search parameter limits.
        """
        omega_orb_max = 2 * np.pi / p_orb_min
        # x_orb = Projected orbital radius, a * sin(i) / c (in light-sec).
        x_orb = 0.005 * ((m_p + m_c) * p_orb_min**2) ** (1 / 3) * m_c / (m_p + m_c)
        max_derivs = x_orb * C_VAL * omega_orb_max ** np.arange(poly_order + 1)
        bounds = [(-d, d) for d in max_derivs[2:][::-1]]
        freq_shift = max_derivs[1] / C_VAL
        bounds.append((freq * (1 - freq_shift), freq * (1 + freq_shift)))
        return cls(typed.List(bounds))

    @classmethod
    def from_upper(
        cls,
        true_params: list[float],
        d_range: tuple[float, float],
        t_obs: float,
    ) -> ParamLimits:
        """Generate search parameter limits from upper bounds.

        Parameters
        ----------
        true_params : list[float]
            True values of the search parameters.
        d_range : tuple[float, float]
            Range of the upper parameter to search (min, max).
        t_obs : float
            Total observation time (in seconds).

        Returns
        -------
        ParamLimits
            Object with the search parameter limits.
        """
        nparams = len(true_params)
        dvec = np.zeros(nparams + 1, dtype=np.float64)
        dvec[1:-2] = true_params[1:-1]  # till acceleration
        dvec[0] = d_range[0]
        dvec_min_up = psr_utils.shift_params(dvec, t_obs / 2)
        dvec_min_low = psr_utils.shift_params(dvec, -t_obs / 2)
        dvec[0] = d_range[1]
        dvec_max_up = psr_utils.shift_params(dvec, t_obs / 2)
        dvec_max_low = psr_utils.shift_params(dvec, -t_obs / 2)
        dvec_bound_low = np.minimum(dvec_min_low, dvec_max_low)
        dvec_bound_up = np.maximum(dvec_min_up, dvec_max_up)
        bounds_d = [
            (low, up) for low, up in zip(dvec_bound_low, dvec_bound_up, strict=False)
        ]
        bounds = bounds_d[:-2]
        freq_shift = dvec_bound_up[-2] / C_VAL
        bounds.append(
            (true_params[-1] * (1 - freq_shift), true_params[-1] * (1 + freq_shift)),
        )
        return cls(typed.List(bounds))

    @classmethod
    def from_keplerian(
        cls,
        freq: tuple[float, float],
        poly_order: int,
        p_orb_min: float,
        ecc_max: float,
        tobs: float,
        m_c: float,
        m_p: float = 1.4,
    ) -> ParamLimits:
        """Generate search parameter limits from Keplerian orbit parameters.

        Parameters
        ----------
        freq : tuple[float, float]
            Frequency range to search (min, max).
        poly_order : int
            Highest polynomial order to include in the search.
        p_orb_min : float
            Minimum orbital period (in seconds).
        ecc_max : float
            Maximum eccentricity of the orbit.
        tobs : float
            Total observation time (in seconds).
        m_c : float
            Mass of the companion (in solar masses).
        m_p : float, optional
            Mass of the pulsar (in solar masses), by default 1.4.

        Returns
        -------
        ParamLimits
            Object with the search parameter limits.
        """
        out = typed.List([(float(freq[0]), float(freq[1]))])
        omega_orb_max = 2 * np.pi / p_orb_min
        # x_orb = Projected orbital radius, a * sin(i) / c (in light-sec).
        x_orb = 0.005 * ((m_p + m_c) * p_orb_min**2) ** (1 / 3) * m_c / (m_p + m_c)
        n_rad = tobs * omega_orb_max
        bounds = kepler.find_max_deriv_bounds(
            x_orb,
            n_rad,
            ecc_max,
            poly_order + 1,
            p_orb_min,
        )
        for bound in bounds:
            out.insert(0, (-bound, bound))
        return cls(out)


class PulsarSearchConfig:
    """Class to hold the configuration for the polynomial search.

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
    ducy_max : float, optional
        Maximum duty cycle to search for, by default 0.2.
    wtsp : float, optional
        Spacing factor between consecutive boxcar widths, by default 1.5.
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
        ducy_max: float = 0.2,
        wtsp: float = 1.5,
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
        self.ducy_max = ducy_max
        self.wtsp = wtsp
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
    def param_names(self) -> list[str]:
        """:obj:`list[str]`: Names of the search parameters."""
        default_names = ["snap", "jerk", "accel", "freq"]
        return default_names[-self.nparams :]

    @property
    def f_min(self) -> float:
        """:obj:`float`: Minimum frequency value to search."""
        return self.param_limits[-1][0]

    @property
    def f_max(self) -> float:
        """:obj:`float`: Maximum frequency value to search."""
        return self.param_limits[-1][1]

    def get_dparams_f(self, ffa_level: int) -> np.ndarray:
        """Get the step sizes for frequency and its derivatives.

        Parameters
        ----------
        ffa_level : int
            FFA level for which to compute the parameter steps.

        Returns
        -------
        np.ndarray
            Array with the step sizes in decreasing derivative order.
        """
        tseg_cur = 2**ffa_level * self.tseg_brute
        t_ref = 0 if self.nparams == 1 else tseg_cur / 2
        return psr_utils.poly_taylor_step_f(
            self.nparams,
            tseg_cur,
            self.nbins,
            self.tol,
            t_ref=t_ref,
        )

    def get_dparams(self, ffa_level: int) -> np.ndarray:
        """Get the step sizes for the search parameters.

        Parameters
        ----------
        ffa_level : int
            FFA level for which to compute the parameter steps.

        Returns
        -------
        np.ndarray
            Array with the parameter step sizes.
        """
        tseg_cur = 2**ffa_level * self.tseg_brute
        t_ref = 0 if self.nparams == 1 else tseg_cur / 2
        return psr_utils.poly_taylor_step_d(
            self.nparams,
            tseg_cur,
            self.nbins,
            self.tol,
            self.f_max,
            t_ref=t_ref,
        )

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
        """Get the parameter ranges for the given parameter steps.

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
                psr_utils.range_param(*self.param_limits[iparam], dparams[iparam])
                for iparam in range(self.nparams)
            ],
        )

    def _bseg_brute_default(self) -> int:
        init_levels = 1 if self.nparams == 1 else 5
        levels = int(np.log2(self.nsamps * self.tsamp * self.f_min) - init_levels)
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

        if self.nparams < 1:  # or self.nparams > 4:
            msg = f"param_limits must have 1-4 elements, got {self.nparams}"
            raise ValueError(msg)
        for _, (val_min, val_max) in enumerate(self.param_limits):
            if val_min >= val_max:
                msg = f"param_limits must have min < max, got {self.param_limits}"
                raise ValueError(msg)
