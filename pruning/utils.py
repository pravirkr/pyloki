from astropy import constants
import numpy as np
from numpy.polynomial import chebyshev as cheb

import logging
from datetime import datetime
from rich.text import Text
from rich.logging import RichHandler
from rich.console import Console

from spyden import TemplateBank, snratio

c_val = constants.c.value


def cartesian_prod(arr_list: list[np.ndarray]) -> np.ndarray:
    mesh = np.meshgrid(*arr_list, indexing="ij")
    flattened_mesh = [arr.ravel() for arr in mesh]
    return np.vstack(flattened_mesh).T


def cartesian_prod_st(arr_list: list[np.ndarray]) -> np.ndarray:
    """Twice as fast as cartesian_prod."""
    la = len(arr_list)
    dtype = np.result_type(*arr_list)
    cart = np.empty([la] + [len(arr) for arr in arr_list], dtype=dtype)
    for iarr, arr in enumerate(np.ix_(*arr_list)):
        cart[iarr, ...] = arr
    return cart.reshape(la, -1).T


def get_indices(
    proper_time: np.ndarray, periods: float | list | np.ndarray, nbins: int
) -> np.ndarray:
    """Calculates the indices of the folded time series.

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


def pad_with_inf(param_list: list[np.ndarray]) -> np.ndarray:
    """Pad a list of arrays with inf to make them all the same length.

    Parameters
    ----------
    param_list : list[np.ndarray]
        List of arrays to pad.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    maxlen = np.max(list(map(len, param_list)))
    output = np.zeros([len(param_list), maxlen])
    output += np.inf
    for iarr, arr in enumerate(param_list):
        output[iarr][: len(arr)] = arr
    return output


def snail_access_scheme(nchunks: int, ref_idx: int) -> np.ndarray:
    """Get an access pattern for the chunks to implement the snail scheme

    Parameters
    ----------
    nchunks : int
        number of chunks
    ind_ref : int
        index of the chunk to start with

    Returns
    -------
    np.ndarray
        access pattern for the chunks
    """
    return np.argsort(np.abs(np.arange(nchunks) - ref_idx))


class Spyden(object):
    def __init__(self, profile, tempwidth_max=None, template_kind="boxcar"):
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
            raise ValueError(f"template {template_kind} not implemented in spyden")

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
    def template_kind(self):
        return self._template_kind

    @property
    def snr(self):
        return self.snrmap[np.unravel_index(self.snrmap.argmax(), self.snrmap.shape)]

    @property
    def best_width(self):
        # in bins
        return self._best_temp.shape_params["w"]

    @property
    def ref_bin(self):
        """
        For Gaussian, return peak bin.
        For boxcar, return start bin.
        """
        return np.unravel_index(self.snrmap.argmax(), self.snrmap.shape)[1]

    @property
    def on_pulse(self):
        if self.template_kind == "boxcar":
            return [self.ref_bin, self.ref_bin + self.best_width]
        elif self.template_kind == "gaussian":
            return [
                self.ref_bin - round(self.best_width),
                self.ref_bin + round(self.best_width),
            ]

def generate_chebyshev_polys_table_numpy(order, n_derivs):
    tab = np.zeros((n_derivs + 1, order + 1, order + 1))
    for ideriv in range(n_derivs + 1):
        for iorder in range(order + 1):
            poly_coeffs = cheb.cheb2poly(cheb.Chebyshev.basis(iorder).deriv(ideriv).coef)
            tab[ideriv, iorder, : poly_coeffs.size] = poly_coeffs
    return tab


def get_logger(
    name: str, level: int | str = logging.INFO, quiet: bool = False
) -> logging.Logger:
    """Get a fancy logging utility using Rich library.

    Parameters
    ----------
    name : str
        logger name
    level : int or str, optional
        logging level, by default logging.INFO
    quiet : bool, optional
        if True set `level` as logging.WARNING, by default False

    Returns
    -------
    logging.Logger
        a logging object
    """
    logger = logging.getLogger(name)
    if quiet:
        logger.setLevel(logging.WARNING)
    else:
        logger.setLevel(level)

    logformat = "- %(name)s - %(message)s"
    formatter = logging.Formatter(fmt=logformat)

    if not logger.hasHandlers():
        handler = RichHandler(
            show_level=False,
            show_path=False,
            rich_tracebacks=True,
            log_time_format=_time_formatter,
            console=Console(width=170)
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _time_formatter(timestamp: datetime) -> Text:
    return Text(timestamp.isoformat(sep=" ", timespec="milliseconds"))
