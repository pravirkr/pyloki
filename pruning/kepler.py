import numpy as np
from numba import njit

from pruning import utils


def semi_major_axis(mass: float, period: float) -> float:
    omega = 2 * np.pi / period
    return (utils.M_sun * utils.G * mass / omega**2) ** (1 / 3.0) / utils.C


def mass_function(dv_over_c: float, period: float) -> float:
    av = dv_over_c * utils.C / 2
    omega = 2 * np.pi / (period)
    return av**3 / omega / utils.G


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
        Semi-major axis, in arbitrary units
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
    pos_arr_plus = keplerian_z(t + effective_eps / 2.0, phi, p, a, inc, ecc, om)
    pos_arr_minus = keplerian_z(t - effective_eps / 2.0, phi, p, a, inc, ecc, om)
    vel_arr = (pos_arr_plus - pos_arr_minus) / effective_eps
    acc_arr = (pos_arr_minus + pos_arr_plus - 2 * pos_arr) / (
        (effective_eps / 2.0) ** 2
    )
    return np.array((pos_arr, vel_arr, acc_arr))

