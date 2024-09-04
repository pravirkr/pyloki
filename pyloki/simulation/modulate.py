import numpy as np
from astropy import constants
from numba import types
from numba.experimental import jitclass

from pyloki import kepler


@jitclass(
    spec=[
        ("shift", types.f8),
        ("vel", types.f8),
        ("acc", types.f8),
        ("jerk", types.f8),
        ("snap", types.f8),
    ],
)
class DerivativeModulating:
    def __init__(
        self,
        shift: float = 0,
        vel: float = 0,
        acc: float = 0,
        jerk: float = 0,
        snap: float = 0,
    ) -> None:
        self.shift = shift
        self.vel = vel
        self.acc = acc
        self.jerk = jerk
        self.snap = snap

    def generate(self, t_arr: np.ndarray, t_ref: float = 0) -> np.ndarray:
        delay = (
            ((t_arr - t_ref) ** 4 / 24 * self.snap)
            + ((t_arr - t_ref) ** 3 / 6 * self.jerk)
            + ((t_arr - t_ref) ** 2 / 2 * self.acc)
            + ((t_arr - t_ref) * self.vel)
            + self.shift
        )
        return t_arr + delay / constants.c


@jitclass(spec=[("amplitude", types.f8), ("period", types.f8), ("phi", types.f8)])
class CircularModulating:
    def __init__(self, amplitude: float, period: float, phi: float) -> None:
        self.amplitude = amplitude
        self.period = period
        self.phi = phi

    def generate(self, t: np.ndarray) -> np.ndarray:
        delay = self.amplitude * np.sin(2 * np.pi / self.period * t + self.phi)
        return t + delay / constants.c


@jitclass(
    spec=[
        ("p_orb", types.f8),
        ("ecc", types.f8),
        ("phi", types.f8),
        ("amp", types.f8),
        ("om", types.f8),
        ("inc", types.f8),
    ],
)
class KeplerianModulating:
    def __init__(
        self,
        p_orb: float,
        ecc: float,
        phi: float,
        amp: float,
        om: float,
        inc: float,
    ) -> None:
        self.p_orb = p_orb
        self.ecc = ecc
        self.phi = phi
        self.amp = amp
        self.om = om
        self.inc = inc

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
        return t + delay / constants.c


type_to_mods = {
    "derivative": DerivativeModulating,
    "circular": CircularModulating,
    "keplerian": KeplerianModulating,
}

