from __future__ import annotations

from typing import TYPE_CHECKING

import attrs
import numpy as np
from astropy import constants
from numba import njit
from scipy import optimize

from pyloki.utils import maths
from pyloki.utils.misc import C_VAL

if TYPE_CHECKING:
    from collections.abc import Callable


def semi_major_axis(mass: float, p_orb: float) -> float:
    """Calculate the semi-major axis of an orbit.

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
    a = (mass * constants.M_sun * constants.G / omega**2) ** (1 / 3) / constants.c  # ty: ignore[unresolved-attribute]
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
    av = radial_velocity_ratio * constants.c / 2  # ty: ignore[unresolved-attribute]
    omega = 2 * np.pi / p_orb
    f = av**3 / (omega * constants.G)  # ty: ignore[unresolved-attribute]
    return float(f.value)


@njit(cache=True, fastmath=True)
def keplerian_nu(
    t_arr: np.ndarray,
    p_orb: float,
    ecc: float,
    phi: float,
    n_iters: int = 8,
) -> np.ndarray:
    """Solves the kepler equation and calculates the true anomaly Nu at given times.

    Parameters
    ----------
    t_arr : np.ndarray
        Array of time values for the keplerian orbit to be evaluated.
    p_orb : float
        Orbital period, in the same units as t_arr.
    ecc : float
        Orbital eccentricity (0 <= ecc < 1).
    phi : float
        Orbital phase (mean anomaly at t=0), in the same units as t_arr.
    n_iters : int, optional
        Number of iterations for Newton's method to solve Kepler's equation,
        by default 8.

    Returns
    -------
    np.ndarray
        Array of the Keplerian true anomaly Nu, in radians at the given times.
    """
    t_norm = (t_arr % p_orb) / p_orb
    # mean anomaly
    m_arr = 2 * np.pi * t_norm + phi
    # Solve Kepler's Equation: M = E - e*sin(E) for E (eccentric anomaly)
    # Initial guess for E is M (A better guess could be M + e*sin(M))
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
    aop: float,
    inc: float,
) -> np.ndarray:
    """Calculate the projected z-coordinate of an orbiting body at given times.

    Parameters
    ----------
    t_arr : np.ndarray
        Array of time values for the keplerian orbit to be evaluated.
    p_orb : float
        Orbital period, in the same units as t_arr.
    ecc : float
        Orbital eccentricity (0 <= ecc < 1).
    phi : float
        Orbital phase (mean anomaly at t=0), in the same units as t_arr.
    a : float
        Semi-major axis, in arbitrary units; determines the output length units.
    aop : float
        Argument of periastron, in radians.
    inc : float
        Inclination of the orbit, in radians.

    Returns
    -------
    np.ndarray
        Array of the z coordinate of the orbiting body at the given times.
    """
    ecc = abs(ecc)
    nu = keplerian_nu(t_arr, p_orb, ecc, phi)
    r = a * (1 - ecc**2) / (1 + ecc * np.cos(nu))
    return r * np.sin(nu + aop) * np.sin(inc)


def keplerian_z_derivatives(
    t_arr: np.ndarray,
    p_orb: float,
    ecc: float,
    phi: float,
    a: float,
    aop: float,
    inc: float,
    eps: float = 1e-5,
) -> np.ndarray:
    """Calculate the position derivatives of an orbiting body at given times.

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
    aop : float
        Argument of periastron, in radians.
    inc : float
        Inclination of the orbit, in radians.
    eps : float, optional
        Finite difference step size for the derivatives, by default 1e-5.

    Returns
    -------
    np.ndarray
        Array of the position, velocity and acceleration of the orbiting body.
    """
    effective_eps = eps * p_orb / (2 * np.pi)
    pos_arr = keplerian_z(t_arr, p_orb, ecc, phi, a, aop, inc)
    pos_arr_plus = keplerian_z(t_arr + effective_eps / 2, p_orb, ecc, phi, a, aop, inc)
    pos_arr_minus = keplerian_z(t_arr - effective_eps / 2, p_orb, ecc, phi, a, aop, inc)
    vel_arr = (pos_arr_plus - pos_arr_minus) / effective_eps
    acc_arr = (pos_arr_minus + pos_arr_plus - 2 * pos_arr) / ((effective_eps / 2) ** 2)
    return np.array((pos_arr, vel_arr, acc_arr))


@njit(cache=True, fastmath=True)
def compute_nu_derivatives(e_arr: np.ndarray, ecc: float, p_orb: float) -> tuple:
    n = 2.0 * np.pi / p_orb
    one_minus_e_cos_e_arr = 1.0 - ecc * np.cos(e_arr)
    sqrt_factor = np.sqrt(1.0 - ecc**2)
    dnu_dt = n * sqrt_factor / (one_minus_e_cos_e_arr**2)
    d2nu_dt2 = (2.0 * n**2 * sqrt_factor * ecc * np.sin(e_arr)) / (
        one_minus_e_cos_e_arr**4
    )
    return dnu_dt, d2nu_dt2


def keplerian_z_derivatives_exact(
    t_arr: np.ndarray,
    p_orb: float,
    ecc: float,
    phi: float,
    a: float,
    aop: float,
    inc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact analytical computation of z-coordinate and its derivatives..

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
    aop : float
        Argument of periastron, in radians.
    inc : float
        Inclination of the orbit, in radians.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Array of the position, velocity and acceleration of the orbiting body.
    """
    ecc = abs(ecc)
    nu = keplerian_nu(t_arr, p_orb, ecc, phi)
    # TODO: Need to finish this part.
    e_arr = nu  # Use a attr class for this.
    dnu_dt, d2nu_dt2 = compute_nu_derivatives(e_arr, ecc, p_orb)
    one_plus_e_cos_nu = 1.0 + ecc * np.cos(nu)
    r = a * (1.0 - ecc**2) / one_plus_e_cos_nu
    dr_dnu = a * (1.0 - ecc**2) * ecc * np.sin(nu) / (one_plus_e_cos_nu**2)
    z = r * np.sin(nu + aop) * np.sin(inc)
    dz_dnu = dr_dnu * np.sin(nu + aop) + r * np.cos(nu + aop)
    dz_dt = dz_dnu * np.sin(inc) * dnu_dt
    d2r_dnu2 = (
        a
        * (1.0 - ecc**2)
        * ecc
        / (one_plus_e_cos_nu**3)
        * (np.cos(nu) * one_plus_e_cos_nu + 2.0 * ecc * np.sin(nu) ** 2)
    )

    d2z_dnu2 = (
        d2r_dnu2 * np.sin(nu + aop)
        + 2.0 * dr_dnu * np.cos(nu + aop)
        - r * np.sin(nu + aop)
    )
    d2z_dt2 = (d2z_dnu2 * (dnu_dt**2) + dz_dnu * d2nu_dt2) * np.sin(inc)
    return z, dz_dt, d2z_dt2


def find_derivative_connections(
    a: float,
    n_rad: float,
    ecc: float,
    poly_degree: int,
    n_samples: int = 100,
    res: float = 0.05,
) -> list[np.ndarray]:
    """Generate polynomial coefficients for Keplerian orbits by fitting z(t).

    Derivatives here refer to the polynomial coefficients c_k from sum(c_k * t^k).

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
    t_arr = np.linspace(-n_rad / 2, n_rad / 2, n_samples)
    p_orb = 2 * np.pi  # Assumed period for normalized time axis
    inc = np.pi / 2  # Assumed inclination
    fits = []
    errs = []
    for phi_val in np.arange(0, 2 * np.pi, res):
        for aop_val in np.arange(0, 2 * np.pi, res):
            z_arr = keplerian_z(t_arr, p_orb, ecc, phi_val, a, aop_val, inc)
            fit = np.polyfit(t_arr, z_arr, poly_degree, full=True)
            errs.append(fit[1])
            fits.append(fit[0][::-1])  # Reverse to ascending power order
    return fits


def find_max_deriv_bounds(
    a: float,
    n_rad: float,
    ecc: float,
    deg: int,
    p_orb_min: float = 2 * np.pi,
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
    p_orb_min : float, optional
        Minimum orbital period (in seconds), by default 2 * np.pi

    Returns
    -------
    np.ndarray
        Array of the maximum bounds for the derivatives.
    """
    omega = 2 * np.pi / p_orb_min
    fits = find_derivative_connections(a, n_rad, ecc, deg, res=0.1)
    factors = np.array([omega**i * maths.fact(i) for i in range(deg + 1)])
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


def simulate_keplerian_orbit(
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
    """Simulate examples of keplerian orbits with random parameters.

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
    p_orb: float
    inc: float = attrs.field(default=np.pi / 2)

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
            phi_orb, p_orb, x_orb, ecc_orb, aop_orb, const, lin = all_params
            kep_phases = (
                np.array(
                    keplerian_z(
                        t_arr,
                        phi_orb,
                        p_orb,
                        ecc_orb,
                        x_orb * C_VAL,
                        aop_orb,
                        self.inc,
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


class PredictionTableGenerator:
    def __init__(
        self,
        x_orb_max: float,
        ecc_max: float,
        omega_max: float,
        poly_deg: int = 10,
        n_rad: float = 2 * np.pi,
        om_ratio_min: float = 0.7,
        x_ratio_min: float = 0.7,
    ) -> None:
        self.data = self._build_table_data(
            x_orb_max,
            ecc_max,
            omega_max,
            poly_deg=poly_deg,
            n_rad=n_rad,
            om_ratio_min=om_ratio_min,
            x_ratio_min=x_ratio_min,
        )
        self.poly_deg = poly_deg

    @property
    def std_vals(self) -> np.ndarray:
        return np.std(self.data, axis=0)

    def generate_range_table(
        self,
        d_arr: np.ndarray,
        key_indices: tuple[int, ...] = (2, 3, 4),
    ) -> tuple[dict[tuple, list[tuple]], Callable[[np.ndarray], tuple]]:
        """Generate a lookup table (dictionary) for all coefficients."""
        coord_function = self.get_table_coords_function(d_arr, key_indices)
        dic: dict[tuple, list[tuple]] = {}
        for d in self.data:
            dic[coord_function(d)] = [*dic.get(coord_function(d), []), tuple(d)]

        for key, value in dic.items():
            ar = np.array(value)
            dic[key] = list(map(tuple, zip(np.min(ar, 0), np.max(ar, 0), strict=False)))

        return dic, coord_function

    def get_table_coords_function(
        self,
        discretization_bins: np.ndarray,
        key_indices: tuple[int, ...] = (2, 3, 4),
    ) -> Callable[[np.ndarray], tuple]:
        """Get a function that converts derivative vector to a discrete coordinate key.

        Parameters
        ----------
        discretization_bins : np.ndarray
            Array of bin sizes for each coefficient used in the key.
        key_indices : tuple[int, ...], optional
            Tuple of indices of coefficients to use for forming the key,
            by default (2, 3, 4).

        Returns
        -------
        Callable[[np.ndarray], tuple]
            Function that converts a derivative vector to a discrete coordinate key.
        """
        if len(discretization_bins) != len(key_indices):
            msg = "Length of discretization_bins must match length of key_indices."
            raise ValueError(msg)

        effective_bins = np.max(
            [discretization_bins, self.std_vals[: len(discretization_bins)] / 30],
            0,
        )

        if key_indices == (2, 3, 4):

            @njit
            def coord_function_fixed(derivatives: np.ndarray) -> tuple:
                return (
                    int(derivatives[2] / effective_bins[0]),
                    int(derivatives[3] / effective_bins[1]),
                    int(derivatives[4] / effective_bins[2]),
                )

            return coord_function_fixed

        def coord_function_generic(derivatives: np.ndarray) -> tuple:
            return tuple(int(derivatives[i] / effective_bins[i]) for i in key_indices)

        return coord_function_generic

    def _build_table_data(
        self,
        x_orb_max: float,
        ecc_max: float,
        omega_max: float,
        poly_deg: int = 10,
        n_rad: float = 2 * np.pi,
        om_ratio_min: float = 0.7,
        x_ratio_min: float = 0.7,
        ecc_res: float = 0.05,
    ) -> np.ndarray:
        """Build a dataset of polynomial coefficients from various Keplerian orbits."""
        ecc_vals = np.arange(0, ecc_max, ecc_res)
        base_coeffs = [
            find_derivative_connections(1, n_rad, float(ecc), poly_deg)
            for ecc in ecc_vals
        ]
        base_coeffs_arr = np.vstack(
            [np.array(f) for sublist in base_coeffs for f in sublist],
        )
        # Apply omega scaling
        omega_powers = omega_max ** np.arange(poly_deg + 1)
        fits = x_orb_max * base_coeffs_arr * omega_powers  # shape: (n_fits, poly_deg+1)
        # Expand to different omegas (frequency) scaling
        om_ratios = np.linspace(om_ratio_min, 1, 10)
        om_factors = om_ratios[:, None] ** np.arange(poly_deg + 1)
        # \Shape (10, n_fits, poly_deg+1)
        fits = fits[None, :, :] * om_factors[:, None, :]
        fits = fits.reshape(-1, poly_deg + 1)

        # Expand to different x's (amplitude) scaling
        x_ratios = np.linspace(x_ratio_min, 1, 10)
        # \Shape: (10, n_fits*10, poly_deg+1)
        fits = fits[None, :, :] * x_ratios[:, None, None]
        return fits.reshape(-1, poly_deg + 1)


@attrs.frozen(auto_attribs=True, kw_only=True)
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

    def generate_grid(self, tol: float, tsamp: float, tobs: float) -> np.ndarray:
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
