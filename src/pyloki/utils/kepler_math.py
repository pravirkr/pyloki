"""Low-level stateless Numba Keplerian anomaly and line-of-sight delay kernels.

Lives in ``pyloki.utils`` so both ``pyloki.core`` and top-level ``pyloki.kepler``
can share the exact Kepler solver without violating package layering.
"""

from __future__ import annotations

import numpy as np
from numba import njit


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
    """Solve Kepler's equation and calculate the true anomaly nu at given times.

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
        Array of the Keplerian true anomaly nu, in radians at the given times.
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
