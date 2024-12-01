from __future__ import annotations

from typing import Callable

import attrs
import numpy as np
from astropy import constants
from numba import njit
from scipy import optimize

from pyloki.utils import math
from pyloki.utils.misc import C_VAL


def semi_major_axis(mass: float, p_orb: float) -> float:
    """
    Calculate the semi-major axis of an orbit.

    Parameters
    ----------
    mass : float
        Mass of the central body in solar masses.
    p_orb : float
        Orbital period in seconds.

    Returns
    -------
    float
        Semi-major axis of the orbit in light-seconds.
    """
    omega = 2 * np.pi / p_orb
    a = (mass * constants.M_sun * constants.G / omega**2) ** (1 / 3) / constants.c
    return float(a.value)


def mass_function(radial_velocity_ratio: float, p_orb: float) -> float:
    r"""
    Calculate the mass function of a binary system.

    .. math::
        f = \frac{(M_2 \sin(i))^3}{(M_1 + M_2)^2} = \frac{K^3 P}{2 \pi G}

    Parameters
    ----------
    radial_velocity_ratio : float
        The ratio of the radial velocity semi-amplitude of \( K \) to c.
    p_orb : float
        Orbital period in seconds.

    Returns
    -------
    float
        Mass function of the system in solar masses.
    """
    av = radial_velocity_ratio * constants.c / 2
    omega = 2 * np.pi / p_orb
    f = av**3 / (omega * constants.G)
    return float(f.value)


@njit(cache=True, fastmath=True)
def keplerian_nu(
    t_arr: np.ndarray,
    p_orb: float,
    ecc: float,
    phi: float,
    n_iters: int = 8,
) -> np.ndarray:
    """
    Solves the kepler equation and returns the true anomaly Nu.

    Parameters
    ----------
    t_arr : np.ndarray
        Array of time values for the keplerian orbit to be evaluated.
    p_orb : float
        Orbital period, in the same units as t_arr
    ecc : float
        Eccentricity, between 0 and 1
    phi : float
        Phase, in the same units as t_arr
    n_iters : int, optional
        Number of iterations to solve the kepler equation, by default 8

    Returns
    -------
    np.ndarray
        Array of the Keplerian true anomaly Nu, in radians at the given times.
    """
    t_arr = t_arr % p_orb
    m_arr = 2 * np.pi * (t_arr / p_orb) + phi
    e_arr = m_arr.copy()
    for _ in range(n_iters):
        e_arr += (m_arr + ecc * np.sin(e_arr) - e_arr) / (1 - ecc * np.cos(e_arr))
    return 2 * np.arctan(np.sqrt((1 + ecc) / (1 - ecc)) * np.tan(e_arr / 2))


@njit(cache=True, fastmath=True)
def keplerian_z(
    t_arr: np.ndarray,
    p_orb: float,
    ecc: float,
    phi: float,
    a: float,
    om: float,
    inc: float,
) -> np.ndarray:
    """
    Calculate the z coordinate of a keplerian orbit at given times.

    Parameters
    ----------
    t_arr : np.ndarray
        Array of time values for the keplerian orbit to be evaluated.
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
    t_arr: np.ndarray,
    p_orb: float,
    ecc: float,
    phi: float,
    a: float,
    om: float,
    inc: float,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Calculate the position, velocity and accel of a keplerian orbit at given times.

    Parameters
    ----------
    t_arr : np.ndarray
        Array of time values for the keplerian orbit to be evaluated.
    p_orb : float
        Orbital period, in the same units as t_arr.
    ecc : float
        Orbital eccentricity, between 0 and 1.
    phi : float
        Orbital Phase, in the same units as t_arr.
    a : float
        Semi-major axis, in arbitrary units.
    om : float
        Longitude of the ascending node, in radians.
    inc : float
        Inclination, in radians.
    eps : float, optional
        Finite difference step size for the derivatives, by default 1e-5.

    Returns
    -------
    np.ndarray
        Array of the position, velocity and acceleration of the keplerian orbit.
    """
    effective_eps = eps * p_orb / (2 * np.pi)
    pos_arr = keplerian_z(t_arr, p_orb, ecc, phi, a, om, inc)
    pos_arr_plus = keplerian_z(t_arr + effective_eps / 2, p_orb, ecc, phi, a, om, inc)
    pos_arr_minus = keplerian_z(t_arr - effective_eps / 2, p_orb, ecc, phi, a, om, inc)
    vel_arr = (pos_arr_plus - pos_arr_minus) / effective_eps
    acc_arr = (pos_arr_minus + pos_arr_plus - 2 * pos_arr) / ((effective_eps / 2) ** 2)
    return np.array((pos_arr, vel_arr, acc_arr))


def find_derivative_connections(
    a: float,
    n_rad: float,
    ecc: float,
    deg: int,
    res: float = 0.05,
) -> list[np.ndarray]:
    """Find polynomial fits for various orbital configurations.

    Parameters
    ----------
    a : float
        Semi-major axis of the orbit.
    n_rad : float
        Number of radians to sample the orbit over.
    ecc : float
        Orbital eccentricity.
    deg : int
        Degree of the polynomial fit.
    res : float, optional
        Resolution of the grid search, by default 0.05

    Returns
    -------
    list[np.ndarray]
        List of polynomial coefficients for the fits.
    """
    t_arr = np.linspace(-n_rad / 2, n_rad / 2, 100)
    fits = []
    errs = []
    for phi in np.arange(0, 2 * np.pi, res):
        for om in np.arange(0, 2 * np.pi, res):
            z_arr = keplerian_z(t_arr, 2 * np.pi, ecc, phi, a, om, np.pi / 2)
            fit = np.polyfit(t_arr, z_arr, deg, full=True)
            errs.append(fit[1])
            fits.append(fit[0][::-1])
    return fits


def find_max_deriv_bounds(
    a: float,
    n_rad: float,
    ecc: float,
    deg: int,
    p_orb: float = 2 * np.pi,
) -> np.ndarray:
    """Find the maximum bounds for the derivatives of a keplerian orbit.

    Parameters
    ----------
    a : float
        Semi-major axis of the orbit.
    n_rad : float
        Number of radians to sample the orbit over.
    ecc : float
        Orbital eccentricity.
    deg : int
        Degree of the polynomial fit.
    p_orb : float, optional
        Orbital period, by default 2 * np.pi

    Returns
    -------
    np.ndarray
        Array of the maximum bounds for the derivatives.
    """
    omega = 2 * np.pi / p_orb
    fits = find_derivative_connections(a, n_rad, ecc, deg, res=0.1)
    factors = np.array([omega**i * math.fact(i) for i in range(deg + 1)])
    return np.max(fits, axis=0) * C_VAL * factors


def keplerian_derivative_bounds(
    ideriv: int,
    a_max: float,
    omega_max: float,
    ecc_max: float,
) -> float:
    """
    Calculate the maximum bounds for a specific derivative of a keplerian orbit.

    Parameters
    ----------
    ideriv : int
        Index (order) of the derivative.
        0 for position, 1 for velocity, 2 for acceleration, etc.
    a_max : float
        Maximum semi-major axis of the orbit.
    omega_max : float
        Maximum angular frequency of the orbit in radians per second.
    ecc_max : float
        Maximum eccentricity of the orbit.

    Returns
    -------
    float
        The maximum bound for the derivative.

    Notes
    -----
    The function uses a series approximation that converges for all valid
    eccentricities (0 <= ecc < 1). The series is truncated at 100 terms, which
    should provide sufficient accuracy for most applications.
    """
    return (
        a_max
        * omega_max**ideriv
        * sum([ecc_max**k * (k + 1) ** ideriv for k in range(100)])
        * C_VAL
    )


def simulate_keplerian_orbit(  # noqa: PLR0913
    t_arr: np.ndarray,
    n_orbits: int,
    rng: np.random.Generator,
    p_pul_min: float,
    delta_p: float,
    p_dot_pul_max: float,
    p_orb_range: tuple[float, float],
    ecc_range: tuple[float, float],
    a_range: tuple[float, float],
    phi_range: tuple[float, float] = (0, 2 * np.pi),
    om_range: tuple[float, float] = (0, 2 * np.pi),
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Simulate examples of keplerian orbits with random parameters.

    Parameters
    ----------
    t_arr : np.ndarray
        Array of time values for the keplerian orbit to be evaluated.
    n_orbits : int
        Number of orbits to simulate.
    rng : np.random.Generator
        Random number generator.
    p_pul_min : float
        Minimum pulsar period.
    delta_p : float
        Maximum pulsar period derivative.
    p_dot_pul_max : float
        Maximum pulsar period derivative.
    p_orb_range : tuple[float, float]
        Range of orbital periods values (min, max).
    ecc_range : tuple[float, float]
        Range of eccentricity values (min, max).
    a_range : tuple[float, float]
        Range of semi-major axis values (min, max).
    phi_range : tuple[float, float], optional
        Range of orbital phase values (min, max), by default (0, 2 * np.pi).
    om_range : tuple[float, float], optional
        Range of longitude of the ascending node (min, max), by default (0, 2 * np.pi).
    eps : float, optional
        Finite difference step size for the derivatives, by default 1e-5.

    Returns
    -------
    np.ndarray
        Array of simulated orbits.
    """
    t_middle = (t_arr[0] + t_arr[-1]) / 2

    # Generate random parameters for all examples at once
    p_orb = rng.uniform(*p_orb_range, n_orbits)
    ecc = rng.uniform(*ecc_range, n_orbits)
    phi = rng.uniform(*phi_range, n_orbits)
    a = rng.uniform(*a_range, n_orbits) * C_VAL
    om = rng.uniform(*om_range, n_orbits)
    inc = np.pi / 2
    v = rng.uniform(-0.5, 0.5, n_orbits) * (delta_p / p_pul_min) * C_VAL
    a = rng.uniform(-0.5, 0.5, n_orbits) * (p_dot_pul_max / p_pul_min) * C_VAL

    # Pre-compute common terms
    t_diff = t_arr - t_middle

    examples = []
    for i in range(n_orbits):
        # linear part responsible for the dp inside a single period bin.
        lin_part_phase = t_diff * v[i]
        # spin down part is responsible for the overall p_dot of the pulsar
        spin_down_phase = a[i] * t_diff**2 / 2
        spin_down_vel = a[i] * t_diff
        # Calculate the orbital part
        orbital_part = keplerian_z_derivatives(
            t_arr,
            p_orb[i],
            ecc[i],
            phi[i],
            a[i],
            om[i],
            inc,
            eps=eps,
        )
        orbital_part[0] += lin_part_phase + spin_down_phase
        orbital_part[1] += v[i] + spin_down_vel
        orbital_part[2] += a[i]
        examples.append(orbital_part)
    return np.array(examples)


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
                        x_orb * C_VAL,
                        np.pi / 2,
                        ecc_orb,
                        omega_ecc_orb,
                    ),
                )
                / C_VAL
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


@attrs.define(auto_attribs=True, slots=True, kw_only=True)
class KeplerianParamLimits:
    p_pul: tuple[float, float]
    p_orb: tuple[float, float]
    x_orb: tuple[float, float]
    ecc: tuple[float, float]
    phi_plus_om: tuple[float, float]
    phi_minus_om: tuple[float, float]

    @property
    def omega_orb(self) -> tuple[float, float]:
        return 2 * np.pi / self.p_orb[1], 2 * np.pi / self.p_orb[0]

    def generate_grid(self, tol: float, tsamp: float, tobs: float) -> tuple:
        tol_time = tol * tsamp
        d_p_pul = 0.5 * tol_time / (tobs / self.p_pul[0])
        d_e = tol_time / self.x_orb[1]
        d_x = tol_time / (1 + self.ecc[1])
        d_phi_plus_om_ecc = 1.0 / (self.x_orb[1] / tol_time)
        d_omega_orb = d_phi_plus_om_ecc / tobs
        d_phi_minus_om_ecc = tol_time / self.x_orb[1] / self.ecc[1]
        periods = np.arange(*self.p_pul, d_p_pul)
        eccs = np.arange(*self.ecc, d_e)
        x_orbs = np.arange(*self.x_orb, d_x)
        omega_orbs = np.arange(*self.omega_orb, d_omega_orb)
        phi_plus_oms = np.arange(*self.phi_plus_om, d_phi_plus_om_ecc)
        phi_minus_oms = np.arange(*self.phi_minus_om, d_phi_minus_om_ecc)
        grid = np.meshgrid(
            periods,
            eccs,
            x_orbs,
            omega_orbs,
            phi_plus_oms,
            phi_minus_oms,
            indexing="ij",
        )
        # Calculate derived parameters
        cur_p_orbs = 2 * np.pi / grid[3]
        cur_phis = (grid[4] + grid[5]) / 2.0
        cur_omega_eccs = (grid[4] - grid[5]) / 2.0
        return np.stack(
            [grid[0], cur_p_orbs, grid[2], grid[1], cur_omega_eccs, cur_phis],
            axis=-1,
        )
