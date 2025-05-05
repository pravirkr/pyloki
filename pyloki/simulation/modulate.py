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

    amplitude: float
    p_orb: float
    phi: float

    def generate(self, t: np.ndarray) -> np.ndarray:
        omega = 2 * np.pi / self.p_orb
        delay = self.amplitude * np.sin(omega * t + self.phi)
        return t + delay / C_VAL


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
    "keplerian": KeplerianModulating,
}
