from __future__ import annotations

from abc import ABC, abstractmethod

import attrs
import numpy as np

from pyloki import kepler
from pyloki.utils import maths
from pyloki.utils.misc import C_VAL


@attrs.define(kw_only=True)
class Modulating(ABC):
    """Base class for time-modulating models."""

    # Optional "reference epoch" convention: phase origin occurs at t_ref.
    @abstractmethod
    def generate(self, t_arr: np.ndarray, t_ref: float = 0.0) -> np.ndarray:
        """Return proper-time array t_proper given barycentric t_arr (seconds)."""


@attrs.define(auto_attribs=True, kw_only=True)
class DerivativeModulating(Modulating):
    """Taylor model around t_ref (up to snap).

    Coefficients are displacement derivatives d^(k)(t_ref) in meters.
    """

    shift: float = 0
    vel: float = 0
    acc: float = 0
    jerk: float = 0
    snap: float = 0

    def generate(self, t_arr: np.ndarray, t_ref: float = 0) -> np.ndarray:
        dt = t_arr - t_ref
        delay = (
            self.snap * (dt**4) / 24.0
            + self.jerk * (dt**3) / 6.0
            + self.acc * (dt**2) / 2.0
            + self.vel * dt
            + self.shift
        )
        return t_arr - delay / C_VAL

    def to_circular(self) -> dict[str, float]:
        eps = 1e-30
        if abs(self.acc) < eps and abs(self.snap) < eps:
            msg = "Degenerate phase: cannot recover omega from (d2,d3,d4)."
            raise ValueError(msg)
        if self.acc * self.snap >= 0.0:
            msg = "Incompatible with circular orbit: require d2*d4<0."
            raise ValueError(msg)

        omega = np.sqrt(-self.snap / self.acc)
        p_orb = 2.0 * np.pi / omega
        # x components (seconds / light-seconds)
        x_sin_nu = -self.acc / (C_VAL * omega**2)
        x_cos_nu = -self.jerk / (C_VAL * omega**3)
        x_orb = np.hypot(x_sin_nu, x_cos_nu)
        psi = np.atan2(x_sin_nu, x_cos_nu)  # phase at t_ref

        return {"p_orb": p_orb, "psi": psi, "x_orb": x_orb}


@attrs.define(auto_attribs=True, kw_only=True)
class DerivativeSeriesModulating(Modulating):
    """Kinematic Taylor model around t_ref (N>4).

    Coefficients are displacement derivatives d^(k)(t_ref) in meters.
    """

    coeffs: np.ndarray  # shape (N+1,)

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    def generate(self, t_arr: np.ndarray, t_ref: float = 0) -> np.ndarray:
        k = np.arange(self.order + 1)
        norm_coeffs = self.coeffs / maths.fact(k)
        powers = (t_arr - t_ref)[None, :] ** k[:, None]
        delay = np.sum(norm_coeffs[:, None] * powers, axis=0)
        return t_arr - delay / C_VAL

    def to_circular(self) -> dict[str, float]:
        if self.order < 4:
            msg = "Need at least d2,d3,d4 to recover circular parameters."
            raise ValueError(msg)
        d2, d3, d4 = self.coeffs[2], self.coeffs[3], self.coeffs[4]
        eps = 1e-30
        if abs(d2) < eps and abs(d4) < eps:
            msg = "Degenerate phase: cannot recover omega from (d2,d3,d4)."
            raise ValueError(msg)
        if d2 * d4 >= 0.0:
            msg = "Incompatible with circular orbit: require d2*d4<0."
            raise ValueError(msg)

        omega = np.sqrt(-d4 / d2)
        p_orb = 2.0 * np.pi / omega
        # x components (seconds / light-seconds)
        x_sin_nu = -d2 / (C_VAL * omega**2)
        x_cos_nu = -d3 / (C_VAL * omega**3)
        x_orb = np.hypot(x_sin_nu, x_cos_nu)
        psi = np.arctan2(x_sin_nu, x_cos_nu)  # phase at t_ref

        return {"p_orb": p_orb, "psi": psi, "x_orb": x_orb}


@attrs.define(auto_attribs=True, kw_only=True)
class CircularModulating(Modulating):
    """Non-relativistic circular Keplerian model.

    One must supply (P_orb, psi) and either:
      - x: projected semi-major axis in light-seconds (seconds), or
      - masses (m_p, m_c) [+ sin_i] to compute x.
    """

    p_orb: float
    psi: float
    x_orb: float | None = None
    m_c: float | None = None
    m_p: float = 1.4
    sin_i: float = 1.0

    def __attrs_post_init__(self) -> None:
        # x_orb = Projected orbital radius, a * sin(i) / c (in light-sec).
        if self.x_orb is None and self.m_c is not None:
            a = 0.005 * ((self.m_p + self.m_c) * self.p_orb**2) ** (1 / 3)
            self.x_orb = a * (self.m_c / (self.m_p + self.m_c)) * self.sin_i

    def generate(self, t_arr: np.ndarray, t_ref: float = 0) -> np.ndarray:
        omega = 2 * np.pi / self.p_orb
        delay = self.x_orb * np.sin(omega * (t_arr - t_ref) + self.psi)
        return t_arr - delay

    def to_derivatives(self) -> dict[str, float]:
        omega = 2.0 * np.pi / self.p_orb
        # At t_ref, nu=psi
        x_sin_nu = self.x_orb * np.sin(self.psi)
        x_cos_nu = self.x_orb * np.cos(self.psi)
        d0 = C_VAL * x_sin_nu
        d1 = C_VAL * x_cos_nu * omega
        d2 = -C_VAL * x_sin_nu * omega**2
        d3 = -C_VAL * x_cos_nu * omega**3
        d4 = -d2 * omega**2

        return {"shift": d0, "vel": d1, "acc": d2, "jerk": d3, "snap": d4}

    def to_derivatives_series(self, n: int) -> dict[str, np.ndarray]:
        if n < 0:
            msg = "n must be >= 0."
            raise ValueError(msg)
        omega = 2.0 * np.pi / self.p_orb
        x_sin_nu = self.x_orb * np.sin(self.psi)
        x_cos_nu = self.x_orb * np.cos(self.psi)
        d_arr = np.empty(n + 1, dtype=np.float64)
        d_arr[0] = C_VAL * x_sin_nu
        if n >= 1:
            d_arr[1] = C_VAL * x_cos_nu * omega
        if n >= 2:
            d_arr[2] = -C_VAL * x_sin_nu * omega**2
        if n >= 3:
            d_arr[3] = -C_VAL * x_cos_nu * omega**3
        if n >= 4:
            d_arr[4] = C_VAL * x_sin_nu * omega**4
            d2 = d_arr[2]
            d3 = d_arr[3]
            d4 = d_arr[4]
            ratio = d4 / d2 if d2 != 0 else 0.0
            for k in range(5, n + 1):
                if k % 2 == 0:
                    d_arr[k] = (ratio ** ((k - 2) // 2)) * d2
                else:
                    d_arr[k] = (ratio ** ((k - 3) // 2)) * d3
        return {"coeffs": d_arr}


@attrs.define(auto_attribs=True, kw_only=True)
class CircularT0Modulating(Modulating):
    """A circular modulating function with a reference epoch."""

    a: float  # projected semi-major axis (in seconds)
    p_orb: float  # orbital period (in seconds)
    t0: float  # time of periastron or reference epoch (in seconds)

    def generate(self, t_arr: np.ndarray, t_ref: float = 0) -> np.ndarray:
        omega = 2 * np.pi / self.p_orb
        phi = 2 * np.pi * ((t_ref - self.t0) % self.p_orb) / self.p_orb
        delay = self.a * np.sin(omega * (t_arr - t_ref) + phi)
        return t_arr - delay / C_VAL


@attrs.define(auto_attribs=True, kw_only=True)
class KeplerianModulating(Modulating):
    """A Keplerian modulating function."""

    p_orb: float
    ecc: float
    phi: float
    a: float
    om: float
    inc: float

    def generate(self, t_arr: np.ndarray, t_ref: float = 0) -> np.ndarray:
        delay = kepler.keplerian_z(
            t_arr - t_ref,
            self.p_orb,
            self.ecc,
            self.phi,
            self.a,
            self.om,
            self.inc,
        )
        return t_arr - delay / C_VAL


type_to_mods: dict[str, type[Modulating]] = {
    "derivative": DerivativeModulating,
    "derivative_series": DerivativeSeriesModulating,
    "circular": CircularModulating,
    "circular_t0": CircularT0Modulating,
    "keplerian": KeplerianModulating,
}
