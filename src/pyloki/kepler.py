from __future__ import annotations

import math
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
    a = (mass * constants.M_sun * constants.G / omega**2) ** (1 / 3) / constants.c
    return float(a.value)


def mass_function(
    radial_velocity_ratio: float,
    p_orb: float,
    ecc: float = 0.0,
) -> float:
    r"""
    Calculate the mass function of a binary system.

    .. math::
        f = \frac{(M_2 \sin(i))^3}{(M_1 + M_2)^2}
          = \frac{K^3 P (1 - e^2)^{3/2}}{2 \pi G M_\odot}

    Parameters
    ----------
    radial_velocity_ratio : float
        The ratio of the radial velocity semi-amplitude :math:`K` to :math:`c`.
    p_orb : float
        Orbital period in seconds.
    ecc : float, optional
        Orbital eccentricity (``0 <= ecc < 1``), by default 0.0.

    Returns
    -------
    float
        Mass function of the system in solar masses.
    """
    if p_orb <= 0.0:
        msg = "p_orb must be positive."
        raise ValueError(msg)
    if not (0.0 <= ecc < 1.0):
        msg = "ecc must satisfy 0 <= ecc < 1."
        raise ValueError(msg)
    av = radial_velocity_ratio * constants.c
    omega = 2.0 * np.pi / p_orb
    f = (av**3 / (omega * constants.G)) * ((1.0 - ecc**2) ** 1.5) / constants.M_sun
    return float(f.value)


@njit(cache=True, fastmath=True)
def keplerian_ecc_anomaly(
    t_arr: np.ndarray,
    p_orb: float,
    ecc: float,
    phi: float,
    n_iters: int = 8,
) -> np.ndarray:
    """Solve Kepler's equation for the eccentric anomaly E at given times.

    Parameters
    ----------
    t_arr : np.ndarray
        Array of time values for the Keplerian orbit to be evaluated.
    p_orb : float
        Orbital period, in the same units as t_arr.
    ecc : float
        Orbital eccentricity (0 <= ecc < 1).
    phi : float
        Orbital phase (mean anomaly at t=0), in radians.
    n_iters : int, optional
        Number of Newton iterations to solve Kepler's equation, by default 8.

    Returns
    -------
    np.ndarray
        Array of eccentric anomalies E, in radians.
    """
    t_norm = (t_arr % p_orb) / p_orb
    m_arr = 2.0 * np.pi * t_norm + phi
    e_arr = m_arr + ecc * np.sin(m_arr)
    for _ in range(n_iters):
        e_arr += (m_arr + ecc * np.sin(e_arr) - e_arr) / (1.0 - ecc * np.cos(e_arr))
    return e_arr


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
        Orbital phase (mean anomaly at t=0), in radians.
    n_iters : int, optional
        Number of iterations for Newton's method to solve Kepler's equation,
        by default 8.

    Returns
    -------
    np.ndarray
        Array of the Keplerian true anomaly Nu, in radians at the given times.
    """
    e_arr = keplerian_ecc_anomaly(t_arr, p_orb, ecc, phi, n_iters=n_iters)
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(0.5 * e_arr),
        np.sqrt(1.0 - ecc) * np.cos(0.5 * e_arr),
    )


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
        Orbital phase (mean anomaly at t=0), in radians.
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
        Orbital phase (mean anomaly at t=0), in radians.
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
def compute_nu_derivatives(
    e_arr: np.ndarray,
    ecc: float,
    p_orb: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = 2.0 * np.pi / p_orb
    one_minus_e_cos_e_arr = 1.0 - ecc * np.cos(e_arr)
    sqrt_factor = np.sqrt(1.0 - ecc**2)
    dnu_dt = n * sqrt_factor / (one_minus_e_cos_e_arr**2)
    d2nu_dt2 = -(2.0 * n**2 * sqrt_factor * ecc * np.sin(e_arr)) / (
        one_minus_e_cos_e_arr**4
    )
    return dnu_dt, d2nu_dt2


@njit(cache=True, fastmath=True)
def keplerian_z_derivatives_exact(
    t_arr: np.ndarray,
    p_orb: float,
    ecc: float,
    phi: float,
    a: float,
    aop: float,
    inc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact analytical computation of z-coordinate and its derivatives.

    Parameters
    ----------
    t_arr : np.ndarray
        Array of time values for the keplerian orbit to be evaluated.
    p_orb : float
        Orbital period, in the same units as t_arr.
    ecc : float
        Orbital eccentricity, between 0 and 1.
    phi : float
        Orbital phase (mean anomaly at t=0), in radians.
    a : float
        Semi-major axis, in arbitrary units.
    aop : float
        Argument of periastron, in radians.
    inc : float
        Inclination of the orbit, in radians.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Tuple of position, velocity and acceleration arrays of the orbiting body.
    """
    ecc = abs(ecc)
    e_arr = keplerian_ecc_anomaly(t_arr, p_orb, ecc, phi)
    nu = 2.0 * np.arctan2(
        np.sqrt(1.0 + ecc) * np.sin(0.5 * e_arr),
        np.sqrt(1.0 - ecc) * np.cos(0.5 * e_arr),
    )
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


@njit(cache=True, fastmath=True)
def poly_projection_matrix(t_arr: np.ndarray, deg: int) -> np.ndarray:
    n = t_arr.size
    m = deg + 1
    v_mat = np.empty((n, m), dtype=np.float64)
    v_mat[:, 0] = 1.0
    for k in range(1, m):
        v_mat[:, k] = v_mat[:, k - 1] * t_arr

    # Column scaling for numerical stability
    scale_vec = np.empty(m, dtype=np.float64)

    for k in range(m):
        s = 0.0
        for i in range(n):
            s += v_mat[i, k] * v_mat[i, k]

        s = np.sqrt(s)
        if s == 0.0:
            s = 1.0

        scale_vec[k] = s

        for i in range(n):
            v_mat[i, k] /= s

    p_mat_scaled = np.linalg.inv(v_mat.T @ v_mat) @ v_mat.T
    p_mat = np.empty_like(p_mat_scaled)
    for k in range(m):
        for i in range(n):
            p_mat[k, i] = p_mat_scaled[k, i] / scale_vec[k]

    return p_mat


@njit(cache=True, fastmath=True)
def mat_vec_product(p_mat: np.ndarray, y_vec: np.ndarray) -> np.ndarray:
    m, n = p_mat.shape
    out = np.zeros(m, dtype=np.float64)
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += p_mat[i, j] * y_vec[j]
        out[i] = s
    return out


@njit(cache=True, fastmath=True)
def polyfit_error(t_arr: np.ndarray, z_arr: np.ndarray, coeffs: np.ndarray) -> float:
    err = 0.0

    for i in range(t_arr.size):
        pred = 0.0

        for k in range(coeffs.size - 1, -1, -1):
            pred = pred * t_arr[i] + coeffs[k]

        r = z_arr[i] - pred
        err += r * r

    return err


@njit(cache=True, fastmath=True)
def find_derivative_connections(
    a_sin_i_orb: float,
    n_rad: float,
    ecc_max: float,
    poly_order: int,
    n_samples: int = 100,
    res: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate polynomial coefficients for Keplerian orbits by fitting z(t).

    Derivatives here refer to the polynomial coefficients c_k from sum(c_k * t^k).

    Parameters
    ----------
    a_sin_i_orb : float
        Projected semi-major axis of the orbit.
    n_rad : float
        Number of radians to sample the orbit over.
    ecc_max : float
        Orbital eccentricity.
    poly_order : int
        Degree of the polynomial fit.
    n_samples : int, optional
        Number of orbit samples used for the fit, by default 100.
    res : float, optional
        Resolution of the grid search, by default 0.05.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Polynomial coefficient fits and corresponding sum-of-squared residuals.
    """
    t_arr = np.linspace(-n_rad / 2, n_rad / 2, n_samples)
    p_orb = 2 * np.pi  # Assumed period for normalized time axis
    inc = np.pi / 2  # Assumed inclination
    p_mat = poly_projection_matrix(t_arr, poly_order)

    # dense near periastron
    phi_grid_1 = np.linspace(0.0, 0.3, 200)
    phi_grid_2 = np.linspace(0.3, 2.0 * np.pi, 100)
    phi_grid = np.empty(phi_grid_1.size + phi_grid_2.size)
    phi_grid[: phi_grid_1.size] = phi_grid_1
    phi_grid[phi_grid_1.size :] = phi_grid_2

    n_phi = phi_grid_1.size + phi_grid_2.size
    n_aop = int(np.ceil((2.0 * np.pi) / res))

    n_total = n_phi * n_aop
    fits = np.empty((n_total, poly_order + 1), dtype=np.float64)
    errs = np.empty(n_total, dtype=np.float64)

    row = 0
    for i in range(n_phi):
        phi_val = phi_grid[i]
        for j in range(n_aop):
            aop_val = j * res
            if aop_val >= 2.0 * np.pi:
                continue
            z_arr = keplerian_z(
                t_arr,
                p_orb,
                ecc_max,
                phi_val,
                a_sin_i_orb,
                aop_val,
                inc,
            )
            coeffs = mat_vec_product(p_mat, z_arr)
            err = polyfit_error(t_arr, z_arr, coeffs)

            fits[row, :] = coeffs
            errs[row] = err

            row += 1
    return fits[:row], errs[:row]


def find_max_deriv_bounds(
    a_sin_i_orb: float,
    n_rad: float,
    p_orb_min: float,
    ecc_max: float,
    poly_order: int,
    res: float = 0.1,
) -> np.ndarray:
    """Find the maximum bounds for the derivatives of a keplerian orbit.

    Parameters
    ----------
    a_sin_i_orb : float
        Projected semi-major axis of the orbit (in light-seconds).
    n_rad : float
        Number of radians to sample the orbit over.
    p_orb_min : float
        Minimum orbital period (in seconds).
    ecc_max : float
        Maximum orbital eccentricity.
    poly_order : int
        Degree of the polynomial fit. The function returns poly_order + 1
        values, one for each derivative order 0 through poly_order.
    res : float, optional
        Resolution of the grid search, by default 0.1.

    Returns
    -------
    np.ndarray
        Array of length poly_order + 1 with the maximum absolute bounds
        for each derivative order (0 = position, 1 = velocity, ...).
    """
    omega_orb_max = 2 * np.pi / p_orb_min
    coeffs, _ = find_derivative_connections(
        a_sin_i_orb,
        n_rad,
        ecc_max,
        poly_order,
        res=res,
    )
    factors = np.array(
        [omega_orb_max**i * maths.fact(i) for i in range(poly_order + 1)],
    )
    return np.max(np.abs(coeffs), axis=0) * C_VAL * factors


def keplerian_derivative_bounds(
    ideriv: int,
    a_max: float,
    omega_max: float,
    ecc_max: float,
) -> float:
    r"""
    Calculate the maximum bounds for a specific derivative of a keplerian orbit.

    Parameters
    ----------
    ideriv : int
        Index (order) of the derivative.
        0 for position, 1 for velocity, 2 for acceleration, etc.
    a_max : float
        Maximum projected semi-major axis of the orbit (in light-seconds).
    omega_max : float
        Maximum angular frequency of the orbit in radians per second.
    ecc_max : float
        Maximum eccentricity of the orbit (``0 <= ecc_max < 1``).

    Returns
    -------
    float
        The maximum bound for the derivative.

    Notes
    -----
    Evaluates the exact closed-form infinite series
    :math:`\\sum_{k=0}^{\\infty} (k + 1)^m e^k` via Eulerian polynomials for
    ``0 <= ecc_max < 1`` without truncation error. For exact high-eccentricity
    bounds above the Laplace limit (:math:`e \\approx 0.6627`), prefer
    :func:`find_max_deriv_bounds`.
    """
    if ideriv < 0:
        msg = "ideriv must be non-negative."
        raise ValueError(msg)
    if not (0.0 <= ecc_max < 1.0):
        msg = "ecc_max must satisfy 0 <= ecc_max < 1."
        raise ValueError(msg)
    if ideriv == 0:
        series_sum = 1.0 / (1.0 - ecc_max)
    else:
        eulerian = np.ones(1, dtype=np.float64)
        for r in range(2, ideriv + 1):
            nxt = np.empty(r, dtype=np.float64)
            nxt[0] = 1.0
            nxt[-1] = 1.0
            for j in range(1, r - 1):
                nxt[j] = (r - j) * eulerian[j - 1] + (j + 1) * eulerian[j]
            eulerian = nxt
        poly_val = 0.0
        for j in range(eulerian.size - 1, -1, -1):
            poly_val = poly_val * ecc_max + eulerian[j]
        series_sum = poly_val / ((1.0 - ecc_max) ** (ideriv + 1))
    return float(a_max * (omega_max**ideriv) * series_sum * C_VAL)


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
        Period grid step (in seconds).
    p_dot_pul_max : float
        Maximum pulsar period derivative.
    p_orb_range : tuple[float, float]
        Range of orbital periods values (min, max).
    ecc_range : tuple[float, float]
        Range of eccentricity values (min, max).
    a_range : tuple[float, float]
        Range of projected semi-major axis values (min, max) in light-seconds.
    phi_range : tuple[float, float], optional
        Range of orbital phase values (min, max) in radians, by default (0, 2 * np.pi).
    om_range : tuple[float, float], optional
        Range of argument of periastron values (min, max) in radians,
        by default (0, 2 * np.pi).
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
    a_orb = rng.uniform(*a_range, n_orbits) * C_VAL
    om = rng.uniform(*om_range, n_orbits)
    inc = np.pi / 2
    v = rng.uniform(-0.5, 0.5, n_orbits) * (delta_p / p_pul_min) * C_VAL
    a_spin = rng.uniform(-0.5, 0.5, n_orbits) * (p_dot_pul_max / p_pul_min) * C_VAL

    # Pre-compute common terms
    t_diff = t_arr - t_middle

    examples = []
    for i in range(n_orbits):
        # linear part responsible for the dp inside a single period bin.
        lin_part_phase = t_diff * v[i]
        # spin down part is responsible for the overall p_dot of the pulsar
        spin_down_phase = a_spin[i] * t_diff**2 / 2
        spin_down_vel = a_spin[i] * t_diff
        # Calculate the orbital part
        orbital_part = keplerian_z_derivatives(
            t_arr,
            p_orb[i],
            ecc[i],
            phi[i],
            a_orb[i],
            om[i],
            inc,
            eps=eps,
        )
        orbital_part[0] += lin_part_phase + spin_down_phase
        orbital_part[1] += v[i] + spin_down_vel
        orbital_part[2] += a_spin[i]
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
        initial_guess: np.ndarray | list[float] | str = "default",
        fit_mask: np.ndarray | slice | None = None,
    ) -> optimize.OptimizeResult:
        t_arr = np.asarray(t_arr, dtype=np.float64)
        recovered_phases = np.asarray(recovered_phases, dtype=np.float64)
        if t_arr.ndim != 1 or t_arr.shape != recovered_phases.shape or t_arr.size == 0:
            msg = (
                "t_arr and recovered_phases must be non-empty 1D arrays "
                "of equal length."
            )
            raise ValueError(msg)

        mask = slice(None) if fit_mask is None else fit_mask
        if recovered_phases[mask].size == 0:
            msg = "fit_mask selects zero samples."
            raise ValueError(msg)

        t_centered = t_arr - np.mean(t_arr)

        def eval_model_phases(all_params: np.ndarray) -> np.ndarray:
            phi_orb, p_orb, x_orb, ecc_orb, aop_orb, const, lin = all_params
            kep_phases = (
                keplerian_z(
                    t_arr,
                    p_orb,
                    ecc_orb,
                    phi_orb,
                    x_orb * C_VAL,
                    aop_orb,
                    np.pi / 2,
                )
                / C_VAL
            )
            return kep_phases + const + lin * t_centered

        def fun_to_opt(all_params: np.ndarray) -> float:
            _, p_orb, x_orb, ecc_orb, _, _, _ = all_params
            if p_orb <= 0.0 or x_orb < 0.0 or not (0.0 <= ecc_orb < 1.0):
                return float(np.inf)
            model_phases = eval_model_phases(all_params)
            residuals = (model_phases - recovered_phases)[mask]
            return float(np.sqrt(np.mean(residuals * residuals)))

        if isinstance(initial_guess, str):
            if initial_guess != "default":
                msg = f"Unsupported initial_guess: {initial_guess!r}"
                raise ValueError(msg)
            x_init = self.semi_major_axis * float(np.sin(self.inc))
            kep_init = (
                keplerian_z(
                    t_arr,
                    self.p_orb,
                    self.eccentricity,
                    self.phi,
                    x_init * C_VAL,
                    self.omega_ecc,
                    np.pi / 2,
                )
                / C_VAL
            )
            design = np.column_stack([np.ones_like(t_centered), t_centered])
            const_init, lin_init = np.linalg.lstsq(
                design[mask],
                (recovered_phases - kep_init)[mask],
                rcond=None,
            )[0]
            x0 = np.array(
                [
                    self.phi,
                    self.p_orb,
                    x_init,
                    self.eccentricity,
                    self.omega_ecc,
                    const_init,
                    lin_init,
                ],
                dtype=np.float64,
            )
        else:
            x0 = np.asarray(initial_guess, dtype=np.float64)
            if x0.shape != (7,):
                msg = f"initial_guess must have shape (7,), got {x0.shape}."
                raise ValueError(msg)

        return optimize.minimize(
            fun_to_opt,
            x0,
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
        buckets: dict[tuple, list[np.ndarray]] = {}
        for d in self.data:
            buckets.setdefault(coord_function(d), []).append(d)

        dic: dict[tuple, list[tuple]] = {}
        for key, rows in buckets.items():
            ar = np.asarray(rows)
            dic[key] = list(
                map(tuple, zip(np.min(ar, axis=0), np.max(ar, axis=0), strict=True)),
            )

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
        if not all(0 <= idx <= self.poly_deg for idx in key_indices):
            msg = f"key_indices must be in [0, {self.poly_deg}], got {key_indices}."
            raise ValueError(msg)

        bins_arr = np.asarray(discretization_bins, dtype=np.float64)
        key_idx_arr = np.asarray(key_indices, dtype=np.intp)
        effective_bins = np.maximum(
            bins_arr,
            self.std_vals[key_idx_arr] / 30.0,
        )
        if np.any(effective_bins <= 0.0):
            msg = "Effective discretization bin sizes must be positive."
            raise ValueError(msg)

        if key_indices == (2, 3, 4):
            b0 = float(effective_bins[0])
            b1 = float(effective_bins[1])
            b2 = float(effective_bins[2])

            @njit
            def coord_function_fixed(derivatives: np.ndarray) -> tuple[int, int, int]:
                return (
                    math.floor(derivatives[2] / b0),
                    math.floor(derivatives[3] / b1),
                    math.floor(derivatives[4] / b2),
                )

            return coord_function_fixed

        key_idx_tuple = tuple(int(k) for k in key_indices)
        eff_bins_tuple = tuple(float(b) for b in effective_bins)

        def coord_function_generic(derivatives: np.ndarray) -> tuple[int, ...]:
            return tuple(
                math.floor(derivatives[idx] / eff_bins_tuple[pos])
                for pos, idx in enumerate(key_idx_tuple)
            )

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
        """Build a dataset of kinematic Taylor derivatives from Keplerian orbits."""
        if x_orb_max <= 0.0 or omega_max <= 0.0:
            msg = "x_orb_max and omega_max must be positive."
            raise ValueError(msg)
        if not (0.0 <= ecc_max < 1.0):
            msg = "ecc_max must satisfy 0 <= ecc_max < 1."
            raise ValueError(msg)
        if ecc_res <= 0.0:
            msg = "ecc_res must be positive."
            raise ValueError(msg)

        if ecc_max == 0.0:
            ecc_vals = np.array([0.0], dtype=np.float64)
        else:
            grid_ecc = np.arange(0.0, ecc_max, ecc_res, dtype=np.float64)
            if grid_ecc.size == 0 or grid_ecc[-1] < ecc_max - 1e-12:
                ecc_vals = np.append(grid_ecc, ecc_max)
            else:
                ecc_vals = grid_ecc

        base_coeffs_arr = np.vstack(
            [
                find_derivative_connections(1.0, n_rad, float(ecc), poly_deg)[0]
                for ecc in ecc_vals
            ],
        )
        # Scale polynomial coeffs c_k to Taylor derivatives d_k = k! c x w^k c_k
        deriv_factors = np.array(
            [(omega_max**i) * maths.fact(i) for i in range(poly_deg + 1)],
            dtype=np.float64,
        )
        fits = x_orb_max * C_VAL * base_coeffs_arr * deriv_factors
        # Expand to different omegas (frequency) scaling
        om_ratios = np.linspace(om_ratio_min, 1, 10)
        om_factors = om_ratios[:, None] ** np.arange(poly_deg + 1)
        fits = fits[None, :, :] * om_factors[:, None, :]
        fits = fits.reshape(-1, poly_deg + 1)

        # Expand to different x's (amplitude) scaling
        x_ratios = np.linspace(x_ratio_min, 1, 10)
        fits = fits[None, :, :] * x_ratios[:, None, None]
        return fits.reshape(-1, poly_deg + 1)


def _grid_axis(bounds: tuple[float, float], step: float) -> np.ndarray:
    """Build a non-empty 1D grid over [min, max] allowing equal-bound pins."""
    vmin, vmax = bounds
    if vmin > vmax:
        msg = f"Invalid bounds: min ({vmin}) exceeds max ({vmax})."
        raise ValueError(msg)
    if vmin == vmax:
        return np.array([vmin], dtype=np.float64)
    return np.arange(vmin, vmax, step, dtype=np.float64)


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
        if tol <= 0.0 or tsamp <= 0.0 or tobs <= 0.0:
            msg = "tol, tsamp, and tobs must be positive."
            raise ValueError(msg)
        if self.p_pul[0] <= 0.0 or self.p_orb[0] <= 0.0 or self.x_orb[1] <= 0.0:
            msg = "p_pul[0], p_orb[0], and x_orb[1] must be positive."
            raise ValueError(msg)
        if self.ecc[0] < 0.0 or self.ecc[1] >= 1.0:
            msg = "ecc bounds must satisfy 0 <= ecc_min <= ecc_max < 1."
            raise ValueError(msg)

        tol_time = tol * tsamp
        d_p_pul = 0.5 * tol_time / (tobs / self.p_pul[0])
        d_e = tol_time / self.x_orb[1]
        d_x = tol_time / (1.0 + self.ecc[1])
        d_phi_plus_om_ecc = tol_time / self.x_orb[1]
        d_omega_orb = d_phi_plus_om_ecc / tobs

        periods = _grid_axis(self.p_pul, d_p_pul)
        eccs = _grid_axis(self.ecc, d_e)
        x_orbs = _grid_axis(self.x_orb, d_x)
        omega_orbs = _grid_axis(self.omega_orb, d_omega_orb)
        phi_plus_oms = _grid_axis(self.phi_plus_om, d_phi_plus_om_ecc)
        if self.ecc[1] == 0.0:
            if self.phi_minus_om[0] > self.phi_minus_om[1]:
                msg = (
                    f"Invalid bounds: min ({self.phi_minus_om[0]}) exceeds "
                    f"max ({self.phi_minus_om[1]})."
                )
                raise ValueError(msg)
            phi_minus_oms = np.array([self.phi_minus_om[0]], dtype=np.float64)
        else:
            d_phi_minus_om_ecc = tol_time / self.x_orb[1] / self.ecc[1]
            phi_minus_oms = _grid_axis(self.phi_minus_om, d_phi_minus_om_ecc)

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
