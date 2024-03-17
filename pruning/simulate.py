import numpy as np
from numba import types
from numba.experimental import jitclass
from scipy import stats

from pruning import utils


@jitclass(
    spec=[
        ("shift", types.f8),
        ("vel", types.f8),
        ("acc", types.f8),
        ("jerk", types.f8),
        ("snap", types.f8),
    ]
)
class DerivativeModulating(object):
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

    def generate(self, t: np.ndarray) -> np.ndarray:
        delay = (
            t**4 / 24 * self.snap
            + t**3 / 6 * self.jerk
            + t**2 / 2 * self.acc
            + t * self.vel
            + self.shift
        )
        return t + delay / utils.c_val


@jitclass(spec=[("amplitude", types.f8), ("period", types.f8), ("phi", types.f8)])
class CircularModulating(object):
    def __init__(self, amplitude: float, period: float, phi: float) -> None:
        self.amplitude = amplitude
        self.period = period
        self.phi = phi

    def generate(self, t: np.ndarray) -> np.ndarray:
        delay = self.amplitude * np.sin(2 * np.pi / self.period * t + self.phi)
        return t + delay / utils.c_val


@jitclass(
    spec=[
        ("p_orb", types.f8),
        ("ecc", types.f8),
        ("phi", types.f8),
        ("a", types.f8),
        ("om", types.f8),
        ("inc", types.f8),
    ]
)
class KeplerianModulating(object):
    def __init__(self, p_orb, ecc, phi, a, om, inc):
        self.p_orb = p_orb
        self.ecc = ecc
        self.phi = phi
        self.a = a
        self.om = om
        self.inc = inc

    def generate(self, t: np.ndarray) -> np.ndarray:
        delay = kepler.KeplerianZ(
            t, self.p_orb, self.ecc, self.phi, self.a, self.om, self.inc
        )
        return t + delay / utils.c_val


type_to_mods = {
    "derivative": DerivativeModulating,
    "circular": CircularModulating,
    "keplerian": KeplerianModulating,
}


def von_mises_pulse_shape(proper_time, dt, period, width, pos=0.5) -> np.ndarray:
    kappa = np.log(2.0) / (2.0 * np.sin(np.pi * width / 2.0) ** 2)
    phase_radians = (proper_time / period - pos) * (2 * np.pi)
    return np.exp(kappa * (np.cos(phase_radians) - 1.0))


class PulseShape(object):
    """_summary_

    Parameters
    ----------
    proper_time : np.ndarray
        Proper time array.
    dt : float
        Time resolution.
    period : float
        Period of the pulsar.
    shape : str, optional
        Shape of the pulse, by default "gaussian"
    width : float, optional
        Duty cycle of the pulse (FWTM) in phase units, by default 0.1
    pos : float, optional
        Phase of the pulse peak, by default 0.5
    """

    def __init__(
        self,
        proper_time: np.ndarray,
        dt: float,
        period: float,
        shape: str = "gaussian",
        width: float = 0.1,
        pos: float = 0.5,
    ) -> None:
        self.proper_time = proper_time
        self.dt = dt
        self.period = period
        self.shape = shape
        self.width = width
        self.pos = pos

        self._rv = self._set_rv()

    @property
    def rv(self):
        return self._rv

    def generate(self) -> np.ndarray:
        x0 = (self.proper_time % self.period) / self.period
        x1 = ((self.proper_time + self.dt) % self.period) / self.period
        return (x0 < x1) * (self.rv.cdf(x1) - self.rv.cdf(x0))

    def _set_rv(self):
        if self.shape == "boxcar":
            rv = stats.uniform(loc=self.pos, scale=self.width)
        elif self.shape == "gaussian":
            width = self.width / (2 * np.sqrt(2 * np.log(10)))
            rv = stats.norm(loc=self.pos, scale=width)
        elif self.shape == "von_mises":
            kappa = np.log(2.0) / (2.0 * np.sin(np.pi * self.width / 2.0) ** 2)
            rv = stats.vonmises(loc=self.pos, kappa=kappa)
        else:
            raise ValueError(f"Unknown pulse shape: {self.shape}")
        return rv
