import attrs
import numpy as np

from pyloki import kepler
from pyloki.utils.misc import C_VAL


@attrs.frozen(auto_attribs=True, kw_only=True)
class DerivativeModulating:
    """A derivative modulating function."""

    shift: float = 0
    vel: float = 0
    acc: float = 0
    jerk: float = 0
    snap: float = 0

    def generate(self, t_arr: np.ndarray, t_ref: float = 0) -> np.ndarray:
        delay = (
            ((t_arr - t_ref) ** 4 / 24 * self.snap)
            + ((t_arr - t_ref) ** 3 / 6 * self.jerk)
            + ((t_arr - t_ref) ** 2 / 2 * self.acc)
            + ((t_arr - t_ref) * self.vel)
            + self.shift
        )
        return t_arr + delay / C_VAL


@attrs.frozen(auto_attribs=True, kw_only=True)
class CircularModulating:
    """A circular modulating function."""

    a: float
    p_orb: float
    phi: float

    def generate(self, t_arr: np.ndarray, t_ref: float = 0) -> np.ndarray:
        omega = 2 * np.pi / self.p_orb
        delay = self.a * np.sin(omega * (t_arr - t_ref) + self.phi)
        return t_arr + delay / C_VAL


@attrs.frozen(auto_attribs=True, kw_only=True)
class CircularT0Modulating:
    """A circular modulating function with a reference epoch."""

    a: float  # projected semi-major axis (in seconds)
    p_orb: float  # orbital period (in seconds)
    t0: float  # time of periastron or reference epoch (in seconds)

    def generate(self, t_arr: np.ndarray, t_ref: float = 0) -> np.ndarray:
        omega = 2 * np.pi / self.p_orb
        phi = 2 * np.pi * ((t_ref - self.t0) % self.p_orb) / self.p_orb
        delay = self.a * np.sin(omega * (t_arr - t_ref) + phi)
        return t_arr + delay / C_VAL


@attrs.frozen(auto_attribs=True, kw_only=True)
class KeplerianModulating:
    """A Keplerian modulating function."""

    p_orb: float
    ecc: float
    phi: float
    a: float
    om: float
    inc: float

    def generate(self, t: np.ndarray) -> np.ndarray:
        delay = kepler.keplerian_z(
            t,
            self.p_orb,
            self.ecc,
            self.phi,
            self.a,
            self.om,
            self.inc,
        )
        return t + delay / C_VAL


type_to_mods = {
    "derivative": DerivativeModulating,
    "circular": CircularModulating,
    "circular_t0": CircularT0Modulating,
    "keplerian": KeplerianModulating,
}
