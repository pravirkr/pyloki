from __future__ import annotations
from numba import njit, types, prange
from numba.experimental import jitclass
import numpy as np


@njit
def gaussian_template(width: int, size: int) -> np.ndarray:
    sigma = width / (2 * np.sqrt(2 * np.log(2)))
    xmax = int(np.ceil(3.5 * sigma))
    xx = np.arange(-xmax, xmax + 1)
    data = np.exp(-(xx**2) / (2 * sigma**2))
    padded_data = np.zeros(size)
    padded_data[size // 2 - xmax : size // 2 + xmax + 1] = data
    return normalise(padded_data)


@njit
def boxcar_template(width: int, size: int) -> np.ndarray:
    data = np.ones(width)
    padded_data = np.zeros(size)
    padded_data[: len(data)] = data
    return normalise(padded_data)


@njit
def normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise data to unit square sum.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Normalised array.
    """
    return arr / (np.dot(arr, arr) ** 0.5)


@njit
def cpadpow2(arr: np.ndarray) -> np.ndarray:
    """Circularly pad the last dimension of 'arr' to a length that
    is a power of 2.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Padded array.
    """
    nbins = arr.shape[-1]
    padded_length = 2 ** int(np.ceil(np.log2(nbins)))
    padding_needed = padded_length - nbins
    return np.concatenate((arr, arr[..., :padding_needed]), axis=-1)


@njit
def cpad2len(arr: np.ndarray, size: int) -> np.ndarray:
    """Circularly pad the last dimension of ndarray 'arr' to given length with zeros."""
    padding_needed = size - arr.shape[-1]
    zero_arr = np.zeros(arr.shape[:-1] + (padding_needed,))
    return np.concatenate((arr, zero_arr), axis=-1)


@jitclass(
    spec=[
        ("widths", types.i8[:]),
        ("size", types.i8),
        ("shape", types.string),
        ("_templates", types.f8[:, :]),
    ]
)
class MatchedFilter(object):
    def __init__(self, widths: np.ndarray, size: int, shape: str = "boxcar") -> None:
        self.widths = widths
        self.shape = shape
        self._templates = self._init_template_bank(size)

    @property
    def templates(self) -> np.ndarray:
        return self._templates

    @property
    def ntemp(self):
        return len(self.templates)

    def compute_ts(self, ts: np.ndarray) -> np.ndarray:
        ts_e, ts_v = ts
        fold = ts_e / np.sqrt(ts_v)
        return self.compute_snr(fold)

    def compute_snr(self, data: np.ndarray) -> np.ndarray:
        nbins = data.shape[-1]
        folds = data.reshape(-1, nbins).astype(np.float32)
        nprof, nbins = folds.shape

        xx = cpadpow2(folds)
        yy = cpad2len(self.templates, xx.shape[-1])
        fx = np.fft.rfft(xx).reshape(nprof, 1, -1)
        fy = np.fft.rfft(yy).reshape(1, self.ntemp, -1)
        snr = np.fft.irfft(fx * fy)

        result = np.empty((nprof, self.ntemp), dtype=np.float32)
        for iprof in range(nprof):
            for jtemp in range(self.ntemp):
                result[iprof, jtemp] = np.max(snr[iprof, jtemp, :nbins])
        return result.reshape((*data.shape[:-1], self.ntemp))

    def _init_template_bank(self, size: int) -> np.ndarray:
        templates = np.empty((len(self.widths), size), dtype=np.float64)
        if self.shape == "boxcar":
            for iw, width in enumerate(self.widths):
                templates[iw] = boxcar_template(width, size)
        elif self.shape == "gaussian":
            for iw, width in enumerate(self.widths):
                templates[iw] = gaussian_template(width, size)
        else:
            raise ValueError(f"Unknown template shape: {self.shape}")
        return templates


@njit
def snr_score_func(combined_res):
    ts_e, ts_v = combined_res
    fold = ts_e / np.sqrt(ts_v)
    widths = generate_width_trials(len(fold), ducy_max=0.3, wtsp=1.1)
    return np.max(boxcar_snr_1d(fold, widths))


@njit
def generate_width_trials(fold_bins, ducy_max=0.2, wtsp=1.5):
    widths = []
    ww = 1
    wmax = int(max(1, ducy_max * fold_bins))
    while ww <= wmax:
        widths.append(ww)
        ww = int(max(ww + 1, wtsp * ww))
    return np.asarray(widths)


@njit
def boxcar_snr(data: np.ndarray, widths: np.ndarray, stdnoise: float = 1.0) -> np.ndarray:
    widths = np.asarray(widths, dtype=np.uint64)
    nbins = data.shape[-1]
    folds = data.reshape(-1, nbins).astype(np.float32)
    snrs = boxcar_snr_2d(folds, widths, stdnoise)
    return snrs.reshape((*data.shape[:-1], widths.size))


@njit(parallel=True, fastmath=True)
def boxcar_snr_2d(
    folds: np.ndarray, widths: np.ndarray, stdnoise: float = 1.0
) -> np.ndarray:
    nfolds = len(folds)
    snrs = np.zeros(shape=(nfolds, widths.size), dtype=np.float32)
    for ifold in prange(nfolds):
        snrs[ifold] = boxcar_snr_1d(folds[ifold], widths, stdnoise)
    return snrs


@njit(fastmath=True)
def boxcar_snr_1d(
    norm_data: np.ndarray, widths: np.ndarray, stdnoise: float = 1.0
) -> np.ndarray:
    size = len(norm_data)
    data_cumsum = np.cumsum(norm_data)
    total_sum = data_cumsum[-1]
    prefix_sum = np.concatenate(
        (data_cumsum, total_sum + np.cumsum(norm_data[: max(widths)]))
    )
    snr = np.zeros(len(widths), dtype=np.float32)
    for iw, width in enumerate(widths):
        height = 1 / np.sqrt(width)  # boxcar height = +h
        dmax = np.max(prefix_sum[width : width + size] - prefix_sum[:size])
        snr[iw] = height * dmax / stdnoise
    return snr
