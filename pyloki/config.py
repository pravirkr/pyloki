# ruff: noqa: ARG001, ARG002

from __future__ import annotations

import itertools

import attrs
import numpy as np

from pyloki import kepler
from pyloki.core import (
    generate_bp_chebyshev,
    generate_bp_chebyshev_approx,
    generate_bp_chebyshev_fixed,
    generate_bp_taylor,
    generate_bp_taylor_approx,
    generate_bp_taylor_circular,
    generate_bp_taylor_fixed,
    generate_bp_taylor_fixed_circular,
)
from pyloki.detection.scoring import generate_box_width_trials
from pyloki.utils import maths, psr_utils, transforms
from pyloki.utils.misc import C_VAL


def _is_power_of_two(
    instance: PulsarSearchConfig,
    attribute: attrs.Attribute,
    value: int,
) -> None:
    if value == 0:
        return
    if not maths.is_power_of_two(value):
        msg = f"'{attribute.name}' must be a power of 2: {value}"
        raise ValueError(msg)


class ParamLimits:
    """Class to hold the search parameter limits/bounds.

    Parameters
    ----------
    limits : types.ListType[types.Tuple[float, float]]
        List of tuples with the min and max values for each search parameter.
        Should be in the order: ..., jerk, accel, freq.
        Paramaters are defined at t=t_c (center of the observation).
    """

    def __init__(self, limits: list[tuple[float, float]]) -> None:
        self.limits = limits

    def get_cheby_limits(
        self,
        tobs: float,
    ) -> list[tuple[float, float]]:
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
            [transforms.taylor_to_cheby(d_vec, t_s) for d_vec in d_corners],
        )
        alpha_bounds = list(
            zip(
                np.min(alpha_corners, axis=0),
                np.max(alpha_corners, axis=0),
                strict=False,
            ),
        )
        return [self.limits[-1], *alpha_bounds]

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

        return cls(out)

    @classmethod
    def from_circular(
        cls,
        freq: float,
        p_orb_min: float,
        m_c: float,
        m_p: float = 1.4,
        poly_order: int = 5,
    ) -> ParamLimits:
        """Generate search parameter limits from circular orbit parameters.

        Parameters
        ----------
        freq : float
            Expected intrinsic spin frequency of the orbit (in Hz).
        p_orb_min : float
            Minimum orbital period to cover (in seconds).
        m_c : float
            Companion mass (in solar masses).
        m_p : float, optional
            Pulsar mass (in solar masses), by default 1.4.
        poly_order : int, optional
            Order of the polynomial to use for the search, by default 5.

        Returns
        -------
        ParamLimits
            Object with the search parameter limits.
        """
        poly_order = max(poly_order, 2)
        omega_orb_max = 2 * np.pi / p_orb_min
        # x_orb = Projected orbital radius, a * sin(i) / c (in light-sec).
        x_orb = 0.005 * ((m_p + m_c) * p_orb_min**2) ** (1 / 3) * m_c / (m_p + m_c)
        max_derivs = x_orb * C_VAL * omega_orb_max ** np.arange(poly_order + 1)
        bounds = [(-d, d) for d in max_derivs[2:][::-1]]
        freq_shift = max_derivs[1] / C_VAL
        bounds.append((freq * (1 - freq_shift), freq * (1 + freq_shift)))
        return cls(bounds)

    @classmethod
    def from_circular_dynamic(
        cls,
        freq: float,
        p_orb_min: float,
        m_c: float,
        t_drift: float,
        m_p: float = 1.4,
        poly_order: int = 4,
    ) -> ParamLimits:
        """Generate dynamic search parameter limits from circular orbit.

        This method first calculates the maximum physical amplitudes for each
        parameter (s, j, a, v), then projects these worst-case values to the
        edge of the observation window (t_obs/2) to find the maximum possible
        drift. This provides the necessary "headroom" for circular orbit searches.
        """
        poly_order = max(poly_order, 2)
        omega_orb_max = 2 * np.pi / p_orb_min
        # x_orb = Projected orbital radius, a * sin(i) / c (in light-sec).
        x_orb = 0.005 * ((m_p + m_c) * p_orb_min**2) ** (1 / 3) * m_c / (m_p + m_c)
        max_derivs = x_orb * C_VAL * omega_orb_max ** np.arange(poly_order + 1)
        drifted_max_values_d = transforms.shift_taylor_params(
            max_derivs[::-1],
            t_drift / 2.0,
        )
        bounds = [(-d, d) for d in drifted_max_values_d[:-1]]
        freq_shift = drifted_max_values_d[-1] / C_VAL
        bounds.append((freq * (1 - freq_shift), freq * (1 + freq_shift)))

        return cls(bounds)

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
        dvec_min_up = transforms.shift_taylor_params(dvec, t_obs / 2)
        dvec_min_low = transforms.shift_taylor_params(dvec, -t_obs / 2)
        dvec[0] = d_range[1]
        dvec_max_up = transforms.shift_taylor_params(dvec, t_obs / 2)
        dvec_max_low = transforms.shift_taylor_params(dvec, -t_obs / 2)
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
        return cls(bounds)

    @classmethod
    def from_keplerian(
        cls,
        freq: tuple[float, float],
        p_orb_min: float,
        ecc_max: float,
        tobs: float,
        m_c: float,
        m_p: float = 1.4,
        poly_order: int = 4,
    ) -> ParamLimits:
        """Generate search parameter limits from Keplerian orbit parameters.

        Parameters
        ----------
        freq : tuple[float, float]
            Frequency range to search (min, max).
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
        poly_order : int, optional
            Highest polynomial order to include in the search, by default 4.

        Returns
        -------
        ParamLimits
            Object with the search parameter limits.
        """
        poly_order = max(poly_order, 2)
        out = [(float(freq[0]), float(freq[1]))]
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


@attrs.frozen(auto_attribs=True, kw_only=True)
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

    nsamps: int = attrs.field(
        validator=[
            attrs.validators.instance_of((int, np.integer)),
            attrs.validators.gt(0),
            _is_power_of_two,
        ],
    )
    tsamp: float = attrs.field(validator=attrs.validators.gt(0))
    nbins: int = attrs.field(
        validator=[
            attrs.validators.instance_of((int, np.integer)),
            attrs.validators.gt(0),
        ],
    )
    tol_bins: float = attrs.field(validator=attrs.validators.gt(0))
    param_limits: list[tuple[float, float]] = attrs.field()
    ducy_max: float = attrs.field(default=0.2, validator=attrs.validators.gt(0))
    wtsp: float = attrs.field(default=1.5, validator=attrs.validators.gt(0))
    prune_poly_order: int = attrs.field(default=3, validator=attrs.validators.gt(0))
    p_orb_min: float = attrs.field(default=0)
    bseg_brute: int = attrs.field(
        default=0,
        validator=[attrs.validators.instance_of((int, np.integer)), _is_power_of_two],
    )
    bseg_ffa: int = attrs.field(
        default=0,
        validator=[attrs.validators.instance_of((int, np.integer)), _is_power_of_two],
    )
    use_fft_shifts: bool = attrs.field(default=True)
    branch_max: int = attrs.field(default=16, validator=attrs.validators.gt(10))
    snap_threshold: float = attrs.field(default=5, validator=attrs.validators.gt(0))
    use_conservative_grid: bool = attrs.field(default=False)

    @param_limits.validator
    def _param_limits_validator(
        self,
        attribute: attrs.Attribute,
        value: list[tuple[float, float]],
    ) -> None:
        if len(value) < 1:  # or len(value) > 4:
            msg = f"param_limits must have 1-5 elements, got {len(value)}"
            raise ValueError(msg)
        for _, (val_min, val_max) in enumerate(value):
            if not isinstance(val_min, int | float) or not isinstance(
                val_max,
                int | float,
            ):
                msg = f"param_limits must be tuples of numbers, got {value}"
                raise TypeError(msg)
            if val_min >= val_max:
                msg = f"param_limits must have min < max, got {value}"
                raise ValueError(msg)

    def __attrs_post_init__(self) -> None:
        if self.bseg_brute == 0:
            object.__setattr__(self, "bseg_brute", self._bseg_brute_default())
        if self.bseg_ffa == 0:
            object.__setattr__(self, "bseg_ffa", self._bseg_ffa_default())
        if self.bseg_brute > self.nsamps:
            msg = f"bseg_brute ({self.bseg_brute}) must be < nsamps ({self.nsamps})"
            raise ValueError(msg)
        if self.bseg_ffa > self.nsamps:
            msg = f"bseg_ffa ({self.bseg_ffa}) must be <= nsamps ({self.nsamps})"
            raise ValueError(msg)
        if self.bseg_ffa <= self.bseg_brute:
            msg = f"bseg_ffa ({self.bseg_ffa}) must be > bseg_brute ({self.bseg_brute})"
            raise ValueError(msg)
        if self.prune_poly_order == 5 and self.p_orb_min == 0:
            msg = "p_orb_min must be provided for a circular orbit search"
            raise ValueError(msg)

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
        default_names = ["crackle", "snap", "jerk", "accel", "freq"]
        return default_names[-self.nparams :]

    @property
    def f_min(self) -> float:
        """:obj:`float`: Minimum frequency value to search."""
        return self.param_limits[-1][0]

    @property
    def f_max(self) -> float:
        """:obj:`float`: Maximum frequency value to search."""
        return self.param_limits[-1][1]

    @property
    def score_widths(self) -> np.ndarray:
        """Get the boxcar widths for the scoring stage."""
        return generate_box_width_trials(
            self.nbins,
            ducy_max=self.ducy_max,
            wtsp=self.wtsp,
        )

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
            self.tol_bins,
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
        return psr_utils.poly_taylor_step_d_f(
            self.nparams,
            tseg_cur,
            self.nbins,
            self.tol_bins,
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

    def get_param_arr(self, dparams: np.ndarray) -> list[np.ndarray]:
        """Get the parameter ranges for the given parameter steps.

        Parameters
        ----------
        dparams : np.ndarray
            Array with the parameter step sizes

        Returns
        -------
        list[np.ndarray]
            List of arrays with the parameter ranges

        Raises
        ------
        ValueError
            If the length of `dparams` is not equal to the number of parameters
        """
        if len(dparams) != self.nparams:
            msg = f"dparams must have length {self.nparams}, got {len(dparams)}"
            raise ValueError(msg)
        return [
            psr_utils.range_param(*self.param_limits[iparam], dparams[iparam])
            for iparam in range(self.nparams)
        ]

    def generate_branching_pattern_approx(
        self,
        kind: str = "taylor",
        ref_seg: int = 0,
        isuggest: int = 0,
    ) -> np.ndarray:
        """Generate the approximate branching pattern for the pruning search.

        This is a simplified version of the branching pattern that only tracks the
        worst-case branching factor.

        Returns
        -------
        np.ndarray
            Branching pattern for the pruning search.
        """
        nsegments_ffa = int(np.ceil(self.nsamps / self.bseg_ffa))
        dparams = self.get_dparams(self.niters_ffa)
        param_arr = self.get_param_arr(dparams)
        dparams_lim = self.get_dparams_limited(self.niters_ffa)
        if kind == "taylor":
            return generate_bp_taylor_approx(
                param_arr,
                dparams_lim,
                self.param_limits,
                self.tseg_ffa,
                nsegments_ffa,
                self.nbins,
                self.tol_bins,
                ref_seg,
                isuggest,
                self.use_conservative_grid,
            )
        if kind == "chebyshev":
            return generate_bp_chebyshev_approx(
                param_arr,
                dparams_lim,
                self.param_limits,
                self.tseg_ffa,
                nsegments_ffa,
                self.nbins,
                self.tol_bins,
                ref_seg,
                isuggest,
                self.use_conservative_grid,
            )
        msg = f"Invalid kind: {kind}"
        raise ValueError(msg)

    def generate_branching_pattern(
        self,
        kind: str = "taylor",
        ref_seg: int = 0,
    ) -> np.ndarray:
        """Generate the exact branching pattern for the pruning search.

        This tracks the exact number of branches per node to compute the average
        branching factor.

        Parameters
        ----------
        kind : str
            The kind of branching pattern to generate.
        ref_seg : int
            The reference segment to generate the branching pattern for.

        Returns
        -------
        np.ndarray
            The branching pattern for the pruning search.
        """
        nsegments_ffa = int(np.ceil(self.nsamps / self.bseg_ffa))
        dparams = self.get_dparams(self.niters_ffa)
        param_arr = self.get_param_arr(dparams)
        dparams_lim = self.get_dparams_limited(self.niters_ffa)
        if kind == "taylor":
            return generate_bp_taylor(
                param_arr,
                dparams_lim,
                self.param_limits,
                self.tseg_ffa,
                nsegments_ffa,
                self.nbins,
                self.tol_bins,
                ref_seg,
                self.use_conservative_grid,
            )
        if kind == "chebyshev":
            return generate_bp_chebyshev(
                param_arr,
                dparams_lim,
                self.param_limits,
                self.tseg_ffa,
                nsegments_ffa,
                self.nbins,
                self.tol_bins,
                ref_seg,
                self.use_conservative_grid,
            )
        if kind == "taylor_fixed":
            return generate_bp_taylor_fixed(
                param_arr,
                dparams_lim,
                self.param_limits,
                self.tseg_ffa,
                nsegments_ffa,
                self.nbins,
                self.tol_bins,
                ref_seg,
            )
        if kind == "chebyshev_fixed":
            return generate_bp_chebyshev_fixed(
                param_arr,
                dparams_lim,
                self.param_limits,
                self.tseg_ffa,
                nsegments_ffa,
                self.nbins,
                self.tol_bins,
                ref_seg,
                self.use_conservative_grid,
            )
        msg = f"Invalid kind: {kind}"
        raise ValueError(msg)

    def generate_branching_pattern_circular(
        self,
        kind: str = "taylor",
        ref_seg: int = 0,
    ) -> np.ndarray:
        """Generate the exact branching pattern for the circular pruning search.

        This tracks the exact number of branches per node to compute the average
        branching factor.

        Parameters
        ----------
        kind : str
            The kind of branching pattern to generate.
        ref_seg : int
            The reference segment to generate the branching pattern for.

        Returns
        -------
        np.ndarray
            The branching pattern for the pruning search.
        """
        nsegments_ffa = int(np.ceil(self.nsamps / self.bseg_ffa))
        dparams = self.get_dparams(self.niters_ffa)
        param_arr = self.get_param_arr(dparams)
        dparams_lim = self.get_dparams_limited(self.niters_ffa)
        if kind == "taylor":
            return generate_bp_taylor_circular(
                param_arr,
                dparams_lim,
                self.param_limits,
                self.tseg_ffa,
                nsegments_ffa,
                self.nbins,
                self.tol_bins,
                ref_seg,
                self.use_conservative_grid,
            )
        if kind == "taylor_fixed":
            return generate_bp_taylor_fixed_circular(
                param_arr,
                dparams_lim,
                self.param_limits,
                self.tseg_ffa,
                nsegments_ffa,
                self.nbins,
                self.tol_bins,
                ref_seg,
            )
        msg = f"Invalid kind: {kind}"
        raise ValueError(msg)

    def _bseg_brute_default(self) -> int:
        init_levels = 1 if self.nparams == 1 else 5
        levels = int(np.log2(self.nsamps * self.tsamp * self.f_min) - init_levels)
        return int(self.nsamps / 2**levels)

    def _bseg_ffa_default(self) -> int:
        return self.nsamps
