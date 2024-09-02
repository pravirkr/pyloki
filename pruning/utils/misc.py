from __future__ import annotations

import logging

import numpy as np
from astropy import constants
from rich.logging import RichHandler
from spyden import TemplateBank, snratio

C_VAL = float(constants.c.value)

def get_indices(
    proper_time: np.ndarray,
    periods: float | list | np.ndarray,
    nbins: int,
) -> np.ndarray:
    """Calculate the indices of the folded time series.

    Parameters
    ----------
    proper_time : np.ndarray
        The proper time of the signal
    period : float | list | np.ndarray
        The period of the signal
    nbins : int
        The number of bins in the folded time series

    Returns
    -------
    np.ndarray
        The indices of the folded time series
    """
    if isinstance(periods, (float, int)):
        periods = [periods]
    periods = np.asarray(periods)
    factor = nbins / periods[:, np.newaxis]
    indices = np.round((proper_time % periods[:, np.newaxis]) * factor) % nbins
    return indices.astype(np.uint32).squeeze()


class Spyden:
    def __init__(
        self,
        profile: np.ndarray,
        tempwidth_max: int | None = None,
        template_kind: str = "boxcar",
    ) -> None:
        self._profile = profile
        self._template_kind = template_kind

        if tempwidth_max is None:
            tempwidth_max = min(len(profile), 32)

        # Generate a list of noise-free normalized pulse templates
        if template_kind == "boxcar":
            bank = TemplateBank.boxcars(list(range(1, tempwidth_max)))
        elif template_kind == "gaussian":
            max_temp_size = int(np.ceil(3.5 * 32 / 2.35)) * 2 + 1
            if len(profile) < max_temp_size:
                tempwidth_max = int(((len(profile) - 1) / 2) * 2.35 / 3.5) - 1
            bank = TemplateBank.gaussians(np.logspace(0, np.log10(tempwidth_max), 50))
        else:
            msg = f"template {template_kind} not implemented in spyden"
            raise ValueError(msg)

        snrmap, mu, sigma, models = snratio(profile, bank, mu=0.0, sigma=1.0)
        self.bank = bank
        self.snrmap = snrmap[0]
        self.mu = mu[0]
        self.sigma = sigma[0]
        self.best_model = models[0]

        self._best_temp = self.bank[
            np.unravel_index(self.snrmap.argmax(), self.snrmap.shape)[0]
        ]

    @property
    def template_kind(self) -> str:
        return self._template_kind

    @property
    def snr(self) -> float:
        return self.snrmap[np.unravel_index(self.snrmap.argmax(), self.snrmap.shape)]

    @property
    def best_width(self) -> int:
        # in bins
        return self._best_temp.shape_params["w"]

    @property
    def ref_bin(self) -> int:
        """Return peak bin (Gaussian) or start bin (boxcar)."""
        return np.unravel_index(self.snrmap.argmax(), self.snrmap.shape)[1]

    @property
    def on_pulse(self) -> list[int]:
        if self.template_kind == "boxcar":
            on_pulse_idx = [self.ref_bin, self.ref_bin + self.best_width]
        else:
            on_pulse_idx = [
                self.ref_bin - round(self.best_width),
                self.ref_bin + round(self.best_width),
            ]
        return on_pulse_idx


def get_logger(
    name: str,
    *,
    level: int | str = logging.INFO,
    quiet: bool = False,
    log_file: str | None = None,
) -> logging.Logger:
    """Get a fancy configured logger.

    Parameters
    ----------
    name : str
        logger name
    level : int or str, optional
        logging level, by default logging.INFO
    quiet : bool, optional
        if True set `level` as logging.WARNING, by default False
    log_file : str, optional
        path to log file, by default None

    Returns
    -------
    logging.Logger
        a logging object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING if quiet else level)
    logformat = "- %(name)s - %(message)s"
    formatter = logging.Formatter(fmt=logformat)
    if not logger.hasHandlers():
        handler = RichHandler(
            show_path=False,
            rich_tracebacks=True,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

