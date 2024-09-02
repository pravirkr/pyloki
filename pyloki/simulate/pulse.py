from __future__ import annotations

import attrs
import numpy as np
from scipy import stats

from pyloki.io.timeseries import TimeSeries
from pyloki.simulate import modulate


@attrs.define(auto_attribs=True, kw_only=True)
class SignalConfig:
    """A pulsar signal generator configuration.

    Parameters
    ----------
    period : float
        Period of the pulsar.
    dt : float
        Time resolution of the generated time series
    nsamps : int, optional
        Number of samples in the generated time series, by default 2**21
    snr : float, optional
        Signal to noise ratio of the folded pulse profile, by default 100
    ducy : float, optional
        Duty cycle of the pulse (FWTM) in fractional phase, by default 0.1
        For slow pulsars, ducy ~ 0.03, for millisecond pulsars, ducy ~ 0.1 - 0.3
    over_sampling : int, optional
        Over sampling factor for the folded phase bins, by default 1
    mod_type : str, optional
        Type of modulation, by default "derivative"
    mod_kwargs : dict, optional
        Modulation function parameters, by default None
    """

    period: float
    dt: float
    nsamps: int = 2**21
    snr: float = 100
    ducy: float = 0.1
    over_sampling: float = 1.0
    mod_type: str = "derivative"
    mod_kwargs: dict | None = None
    _mod_func: modulate.DerivativeModulating = attrs.field(init=False, repr=False)

    def __attrs_post_init__(self) -> None:
        self._set_mod_func(self.mod_type, self.mod_kwargs)
        self._check()

    @property
    def freq(self) -> float:
        """Pulsar frequency."""
        return 1 / self.period

    @property
    def tobs(self) -> float:
        """Total observation time."""
        return self.nsamps * self.dt

    @property
    def tol_bins(self) -> float:
        """Pulsar pulse width in bins."""
        return self.ducy * self.period / self.dt

    @property
    def fold_bins(self) -> int:
        """Number of phase bins in the folded profile."""
        return int(self.period / self.dt / self.over_sampling)

    @property
    def fold_bins_ideal(self) -> int:
        """Number of ideal phase bins in the folded profile."""
        return int(self.period / self.dt)

    @property
    def mod_func(self) -> modulate.DerivativeModulating:
        """Modulation function."""
        return self._mod_func

    @property
    def proper_time(self) -> np.ndarray:
        """Proper time array."""
        return self.mod_func.generate(np.arange(0, self.tobs, self.dt), self.tobs / 2)

    def get_updated(self, update_dict: dict) -> SignalConfig:
        new = attrs.asdict(self, filter=attrs.filters.exclude("_mod_func"))
        if update_dict is not None:
            new.update(update_dict)
        new_checked = {
            key: value for key, value in new.items() if key in attrs.asdict(self)
        }
        return SignalConfig(**new_checked)

    def generate(
        self,
        shape: str = "gaussian",
        phi0: float = 0.5,
    ) -> TimeSeries:
        pulse = PulseShape(
            self.proper_time,
            self.dt,
            self.period,
            shape=shape,
            width=self.ducy,
            pos=phi0,
        )
        signal = pulse.generate()
        stdnoise = np.sqrt(self.nsamps * self.ducy) / self.snr / self.tol_bins
        rng = np.random.default_rng()
        signal += rng.normal(0, stdnoise, self.nsamps)
        signal_v = np.ones(self.nsamps) * stdnoise**2
        return TimeSeries(signal, signal_v, self.dt)

    def _set_mod_func(
        self,
        mod_type: str = "derivative",
        mod_kwargs: dict | None = None,
    ) -> None:
        if mod_kwargs is None:
            mod_kwargs = {}
        self._mod_func = modulate.type_to_mods[mod_type](**mod_kwargs)

    def _check(self) -> None:
        if self.ducy <= 0 or self.ducy >= 1:
            msg = f"Duty cycle ({self.ducy}) should be in (0, 1)"
            raise ValueError(msg)


class PulseShape:
    """Generate a pulse shape.

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
    def rv(self) -> stats.rv_continuous:
        return self._rv

    def generate(self) -> np.ndarray:
        x0 = (self.proper_time % self.period) / self.period
        x1 = ((self.proper_time + self.dt) % self.period) / self.period
        return (x0 < x1) * (self.rv.cdf(x1) - self.rv.cdf(x0))

    def _set_rv(self) -> stats.rv_continuous:
        if self.shape == "boxcar":
            rv = stats.uniform(loc=self.pos, scale=self.width)
        elif self.shape == "gaussian":
            width = self.width / (2 * np.sqrt(2 * np.log(10)))
            rv = stats.norm(loc=self.pos, scale=width)
        elif self.shape == "von_mises":
            kappa = np.log(2.0) / (2.0 * np.sin(np.pi * self.width / 2.0) ** 2)
            rv = stats.vonmises(loc=self.pos, kappa=kappa)
        else:
            msg = f"Unknown pulse shape: {self.shape}"
            raise ValueError(msg)
        return rv


def von_mises_pulse_shape(
    proper_time: np.ndarray,
    dt: float,  # noqa: ARG001
    period: float,
    width: float,
    pos: float = 0.5,
) -> np.ndarray:
    kappa = np.log(2.0) / (2.0 * np.sin(np.pi * width / 2.0) ** 2)
    phase_radians = (proper_time / period - pos) * (2 * np.pi)
    return np.exp(kappa * (np.cos(phase_radians) - 1.0))
