from __future__ import annotations

from typing import Callable

import attrs
import numpy as np
from numba import njit
from scipy import optimize

from pruning import math, utils


def semi_major_axis(mass: float, period: float) -> float:
    omega = 2 * np.pi / period
    return (utils.m_sun_val * utils.g_val * mass / omega**2) ** (1 / 3) / utils.c_val


def mass_function(dv_over_c: float, period: float) -> float:
    av = dv_over_c * utils.c_val / 2
    omega = 2 * np.pi / period
    return av**3 / omega / utils.g_val


@njit
def keplerian_nu(
    t_arr: np.ndarray,
    p_orb: float,
    ecc: float,
    phi: float,
    n_iters: int = 8,
) -> np.ndarray:
    """Solves the kepler equation and returns the true anomaly Nu.

    Parameters
    ----------
    t_arr : np.ndarray
        Array of time values for the keplerian orbit to be evaluated on.
    p_orb : float
        Orbital period, in the same units as t_arr
    ecc : float
        Eccentricity, between 0 and 1
    phi : float
        Phase, in the same units as t_arr
    n_iters : int, optional
        Number of iterations, by default 8

    Returns
    -------
    np.ndarray
        Array of the Keplerian true anomaly Nu, in radians at the given times.
    """
    t_arr = t_arr % p_orb
    m_arr = 2 * np.pi * (t_arr / p_orb) + phi
    e_arr = m_arr[:]
    for _ in range(n_iters):
        e_arr = e_arr + (m_arr + ecc * np.sin(e_arr) - e_arr) / (
            1 - ecc * np.cos(e_arr)
        )
    return 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(e_arr / 2))


@njit
def keplerian_z(
    t_arr: np.ndarray,
    p_orb: float,
    ecc: float,
    phi: float,
    a: float,
    om: float,
    inc: float,
) -> np.ndarray:
    """Calculate the z coordinate of a keplerian orbit at given times.

    Parameters
    ----------
    t_arr : np.ndarray
        Array of time values for the keplerian orbit to be evaluated on.
    p_orb : float
        Orbital period, in the same units as t_arr
    ecc : float
        Orbital eccentricity, between 0 and 1
    phi : float
        Orbital Phase, in the same units as t_arr
    a : float
        Semi-major axis, in arbitrary units; determines the output length units
    om : float
        Longitude of the ascending node, in radians
    inc : float
        Inclination, in radians

    Returns
    -------
    np.ndarray
        Array of the z coordinate of the keplerian orbit at the given times.
    """
    ecc = abs(ecc)
    nu = keplerian_nu(t_arr, p_orb, ecc, phi)
    r = a * (1 - ecc**2) / (1 + ecc * np.cos(nu))
    return r * np.sin(nu + om) * np.sin(inc)


def keplerian_z_derivatives(
    t: np.ndarray,
    phi: float,
    p: float,
    a: float,
    inc: float,
    ecc: float,
    om: float,
    eps: float = 1e-5,
) -> np.ndarray:
    effective_eps = eps * p / (2 * np.pi)
    pos_arr = keplerian_z(t, phi, p, a, inc, ecc, om)
    pos_arr_plus = keplerian_z(t + effective_eps / 2, phi, p, a, inc, ecc, om)
    pos_arr_minus = keplerian_z(t - effective_eps / 2, phi, p, a, inc, ecc, om)
    vel_arr = (pos_arr_plus - pos_arr_minus) / effective_eps
    acc_arr = (pos_arr_minus + pos_arr_plus - 2 * pos_arr) / ((effective_eps / 2) ** 2)
    return np.array((pos_arr, vel_arr, acc_arr))


def find_derivative_connections(
    x_orb: float,
    n_rad: float,
    ecc_orb: float,
    deg: int,
) -> list[np.ndarray]:
    t_arr = np.arange(-n_rad / 2, n_rad / 2, n_rad / 100)
    fits = []
    errs = []
    for phi in np.arange(0, 2 * np.pi, 0.05):
        for om in np.arange(0, 2 * np.pi, 0.05):
            z_arr = keplerian_z(t_arr, np.pi * 2, ecc_orb, phi, x_orb, om, np.pi / 2)
            fit = np.polyfit(t_arr, z_arr, deg, full=True)
            errs.append(fit[1])
            fits.append(fit[0][::-1])
    return fits


def find_max_deriv_bounds(
    x_orb: float,
    n_rad: float,
    ecc_orb: float,
    deg: int,
    omega: int = 1,
) -> np.ndarray:
    t_arr = np.arange(-n_rad / 2, n_rad / 2, n_rad / 100)
    fits = []
    errs = []
    for phi in np.arange(0, 2 * np.pi, 0.1):
        for om in np.arange(0, 2 * np.pi, 0.1):
            z_arr = keplerian_z(t_arr, np.pi * 2, ecc_orb, phi, x_orb, om, np.pi / 2)
            fit = np.polyfit(t_arr, z_arr, deg, full=True)
            errs.append(fit[1])
            fits.append(fit[0][::-1])
    factors = np.array([omega**i * math.fact(i) for i in range(deg + 1)])
    return np.max(fits, 0) * utils.c_val * factors


def keplerian_derivative_bounds(
    deriv_index: int,
    x_max: float,
    omega_max: float,
    ecc_max: float,
) -> float:
    return (
        x_max
        * omega_max**deriv_index
        * sum([ecc_max**k * (k + 1) ** deriv_index for k in range(100)])
        * utils.c_val
    )


@attrs.define(auto_attribs=True, slots=True, kw_only=True)
class KeplerianOrbit:
    semi_major_axis: float
    eccentricity: float
    omega_ecc: float
    phi: float
    orbital_period: float
    inclination: float = attrs.field(default=np.pi / 2)

    def solve_keplerian_phases(
        self,
        t_arr: np.ndarray,
        recovered_phases: np.ndarray,
        initial_guess: str = "default",
    ) -> optimize.OptimizeResult:
        def log_likelihood_fun(
            all_params: np.ndarray,
            t_arr: np.ndarray,
            recovered_phases: np.ndarray,
            *,
            ret_model_phases: bool = False,
        ) -> float:
            phi_orb, p_orb, x_orb, ecc_orb, omega_ecc_orb, const, lin = all_params
            kep_phases = (
                np.array(
                    keplerian_z(
                        t_arr,
                        phi_orb,
                        p_orb,
                        x_orb * utils.c_val,
                        np.pi / 2,
                        ecc_orb,
                        omega_ecc_orb,
                    ),
                )
                / utils.c_val
            )
            model_phases = kep_phases + const + lin * t_arr
            if ret_model_phases:
                return model_phases
            return np.sqrt(
                np.mean(np.abs(model_phases - recovered_phases)[25:220] ** 2),
            )

        def fun_to_opt(all_params: np.ndarray) -> float:
            return log_likelihood_fun(all_params, t_arr, recovered_phases)

        return optimize.minimize(
            fun_to_opt,
            initial_guess,
            method="Nelder-Mead",
            options={"maxfev": 10000},
        )


def build_data(
    x_max: float,
    ecc_max: float,
    omega_max: float,
    pol_deg: int = 10,
    n_rad: float = 2 * np.pi,
    om_ratio: float = 0.7,
    x_ratio: float = 0.7,
) -> np.ndarray:
    fits = []
    # generating ecc fits
    for e in np.arange(0, ecc_max, 0.05):
        scaled_fits = find_derivative_connections(1, n_rad, e, pol_deg)
        fits += [
            x_max * f * np.array([omega_max**i for i in range(pol_deg + 1)])
            for f in scaled_fits
        ]

    # expanding to different omegas
    fit_collection = []
    for om in np.linspace(om_ratio, 1, 10):
        fit_collection += [
            f * np.array([om**i for i in range(pol_deg + 1)]) for f in fits
        ]
    fits_arr = np.array(fit_collection)

    # expanding to different x's
    fit_collection = []
    for x in np.linspace(x_ratio, 1, 10):
        fit_collection += [f * x for f in fits_arr]

    return np.array(fit_collection)


def predictor_generator_func(
    dic: dict,
    coord_function: Callable[[np.ndarray], tuple],
) -> Callable[[np.ndarray, int, int], bool]:
    @njit
    def predictor_func(derivatives: np.ndarray, ind_start: int, ind_end: int) -> bool:
        ranges = dic.get(coord_function(derivatives), False)
        if ranges is False:
            return False
        for i in range(ind_start, ind_end + 1):
            if derivatives[i] < ranges[i][0] or derivatives[i] > ranges[i][1]:
                return False

        return True

    return predictor_func


def test_prediction_power(
    ptg: PredictionTableGenerator,
    predictor_func: Callable[[np.ndarray, int, int], bool],
    n_trials: int = 1000,
) -> int:
    count = 0
    mins = np.min(ptg.data, 0)
    maxs = np.max(ptg.data, 0)
    rng = np.random.default_rng()
    for _ in range(n_trials):
        coord = np.array([rng.uniform(mins[j], maxs[j]) for j in [2, 3, 4, 5, 6]])
        if predictor_func(coord, 5, 6):
            count += 1

    return count


# Should implement two separate logics:
# First logic -> an exhaustive 3 table to predict f_2,f_3,f_4 -> f_5,f_6
# Second logic -> an exhaustive 3 table to predict "Om"^2 = -f_4/f_2, "A^2"
#               = (f_2/"Om"^2)^2 + (f_3/"Om"^3)^2,
#               then use atan2(f2,f3/Om), f5/Om^5/A, f6/Om^6/A as table coordinates.
# Alternative second logic -> an exhaustive 3 table from:
# \               f(3)/f(2) / (-f(4)/f(2))^1/2, f(5)/f(2) / (-f(4)/f(2))^3/2,
# \               f(6)/f(2) / (-f(4)/f(2))^2
class PredictionTableGenerator:
    def __init__(
        self,
        x_max: float,
        ecc_max: float,
        omega_max: float,
        pol_deg: int = 10,
        n_rad: float = 2 * np.pi,
        om_ratio: float = 0.7,
        x_ratio: float = 0.7,
    ) -> None:
        self.data = build_data(
            x_max,
            ecc_max,
            omega_max,
            pol_deg=pol_deg,
            n_rad=n_rad,
            om_ratio=om_ratio,
            x_ratio=x_ratio,
        )

    @property
    def std_vals(self) -> np.ndarray:
        return np.std(self.data, axis=0)

    # other mode is "23456"
    def generate_range_table(self, d_arr: np.ndarray, mode: str = "23456") -> tuple:
        if mode == "23456":
            coord_function = self.get_table_coords_function_23456(d_arr)
        else:
            coord_function = self.get_table_coords_function(d_arr)
        dic: dict[tuple, list[tuple]] = {}
        for d in self.data:
            dic[coord_function(d)] = [*dic.get(coord_function(d), []), tuple(d)]

        for key in dic:
            ar = np.array(dic[key])
            dic[key] = list(map(tuple, zip(np.min(ar, 0), np.max(ar, 0))))

        return dic, coord_function

    def get_table_coords_function(
        self,
        d_arr: np.ndarray,
    ) -> Callable[[np.ndarray], tuple]:
        d_eff = np.max([d_arr, self.std_vals[: len(d_arr)] / 30], 0)

        @njit
        def coord_function(derivatives: np.ndarray) -> tuple:
            return (
                int(derivatives[2] / d_eff[2]),
                int(derivatives[3] / d_eff[3]),
                int(derivatives[4] / d_eff[4]),
            )

        return coord_function
