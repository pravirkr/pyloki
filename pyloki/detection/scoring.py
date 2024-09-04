from __future__ import annotations

import numpy as np
from numba import njit, prange, types
from numba.experimental import jitclass

from pyloki.utils import math, np_utils


@njit(cache=True, fastmath=True)
def normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise data to zero mean and unit square sum.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Normalised array.
    """
    arr_norm = arr - np.mean(arr)
    return arr_norm / (np.dot(arr_norm, arr_norm) ** 0.5)


@njit(cache=True, fastmath=True)
def gen_gaussian_templates(widths: np.ndarray, nbins: int) -> np.ndarray:
    templates = np.empty((len(widths), nbins), dtype=np.float32)
    for iw in range(len(widths)):
        width = widths[iw]
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        xmax = int(np.ceil(3.5 * sigma))
        xx = np.arange(-xmax, xmax + 1)
        data = np.exp(-(xx**2) / (2 * sigma**2))
        padded_data = np.zeros(nbins)
        if nbins >= 2 * xmax + 1:
            padded_data[nbins // 2 - xmax : nbins // 2 + xmax + 1] = data
        else:
            start = xmax - nbins // 2
            end = start + nbins
            padded_data[:nbins] = data[start:end]
        templates[iw] = normalise(padded_data)
    return templates


@njit(cache=True, fastmath=True)
def gen_boxcar_templates(widths: np.ndarray, nbins: int) -> np.ndarray:
    templates = np.empty((len(widths), nbins), dtype=np.float32)
    for iw in range(len(widths)):
        data = np.ones(widths[iw])
        padded_data = np.zeros(nbins)
        padded_data[: len(data)] = data
        templates[iw] = normalise(padded_data)
    return templates


@njit(cache=True, fastmath=True)
def compute_snr_fft(data: np.ndarray, templates: np.ndarray) -> np.ndarray:
    ntemp = len(templates)
    nbins = data.shape[-1]
    folds = data.reshape(-1, nbins).astype(np.float32)
    nprof, nbins = folds.shape

    xx = np_utils.cpadpow2(folds)
    yy = np_utils.cpad2len(templates, xx.shape[-1])
    fx = np.fft.rfft(xx).reshape(nprof, 1, -1)
    fy = np.fft.rfft(yy).reshape(1, ntemp, -1)
    snr = np.fft.irfft(fx * fy)

    result = np.empty((nprof, ntemp), dtype=np.float32)
    for iprof in range(nprof):
        for jtemp in range(ntemp):
            result[iprof, jtemp] = np.max(snr[iprof, jtemp, :nbins])
    return result.reshape((*data.shape[:-1], ntemp))


@njit(cache=True, fastmath=True)
def get_e_mat(templates: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    ntemp, nbins = templates.shape

    total_shifts = 0
    for itemp in range(ntemp):
        total_shifts += nbins // shifts[itemp]

    temp_bank = np.empty((total_shifts, nbins), dtype=templates.dtype)

    row_idx = 0
    for itemp in range(ntemp):
        for jbin in range(0, nbins, shifts[itemp]):
            temp = np.roll(templates[itemp], jbin)
            temp_bank[row_idx] = temp
            row_idx += 1

    return temp_bank


@jitclass(
    spec=[
        ("widths", types.i8[:]),
        ("nbins", types.i8),
        ("shift", types.i8),
        ("shape", types.string),
        ("templates", types.f4[:, :]),
        ("shifts", types.i8[:]),
        ("e_mat", types.f4[:, ::1]),
    ],
)
class MatchedFilter:
    """Matched filter class for computing SNR for a folded suggestion.

    Parameters
    ----------
    widths : np.ndarray
        _description_
    nbins : int
        Number of bins in the folded profile.
    shifts : int, optional
        Shifts to apply to the templates, by default 0.
    shape : str, optional
        Template shape, either 'boxcar' or 'gaussian', by default 'boxcar'.

    Notes
    -----
    - The templates are normalised to unit square sum.
    - The templates are circularly shifted by the given shifts.
    - Implements FFT for cases in which nbins >> 64.
    - Guassian filters are 5-10% more sensitive than boxcars for Gaussian pulses.
    """

    def __init__(
        self,
        widths: np.ndarray,
        nbins: int,
        shift: int = 1,
        shape: str = "boxcar",
    ) -> None:
        self.widths = widths
        self.nbins = nbins
        self.shape = shape
        self.templates = self._init_template_bank(nbins)
        self.shifts = self._init_shifts(shift)
        self.e_mat = get_e_mat(self.templates, self.shifts)

    @property
    def ntemp(self) -> int:
        return len(self.templates)

    def compute_ts(self, ts_comb: np.ndarray) -> np.ndarray:
        ts_e, ts_v = ts_comb
        fold = ts_e / np.sqrt(ts_v)
        return compute_snr_fft(fold, self.templates)

    def compute_match(self, ts_comb: np.ndarray) -> np.ndarray:
        ts_e, ts_v = ts_comb
        fold = ts_e / np.sqrt(ts_v)
        return np.dot(self.e_mat, fold)

    def compute_match_double(self, ts_comb: np.ndarray) -> np.ndarray:
        """Compute the double match filter.

        Parameters
        ----------
        ts_comb : np.ndarray
            _description_

        Returns
        -------
        np.ndarray
            _description_

        Notes
        -----
        - Assuming that the ratio between the peaks is unknown, the score is
        the sum of the squared S/N of the two features.
        - Currently making the assumption that by this stage the sensitivity of
        all cells in the fold is identical (Notice that this assumption is not
        the same as not knowing the mask. it is merely that the mask's fold at all
        places is similar (i.e., only O(n) changes matter, and cause only
        slightly reduced sensitivity, no false positives!)
        """
        n_filters = len(self.e_mat)
        inner_product_matrix = np.dot(self.e_mat, self.e_mat.T)
        look_elsewhere_effect_penalty_double = np.log2(
            (n_filters * n_filters - 1) / 2.0,
        )
        look_elsewhere_effect_penalty_single = np.log2(n_filters)
        ts_e, ts_v = ts_comb
        fold = ts_e / np.sqrt(ts_v)
        scores = np.dot(self.e_mat, fold)
        # not considering the negative maxima as relevant
        scores_max_single = np.max(scores) ** 2
        good_indices = np.argsort(scores)[-10:]
        scores_max_double = scores_max_single
        s = 0
        # loop is only on the 10 best indices, saving from 128**2/2 -> 10**2/2
        for i_ind in range(10):
            i = good_indices[i_ind]
            for j_ind in range(i_ind + 1, 10):
                j = good_indices[j_ind]
                gamma_ij = inner_product_matrix[i, j]
                if abs(gamma_ij) < 0.99:
                    s = (
                        scores[i] ** 2
                        + scores[j] ** 2
                        - 2 * gamma_ij * scores[i] * scores[j]
                    ) / (1 - gamma_ij**2)
                if s > scores_max_double:
                    scores_max_double = s
        # comparing scores should be on the basis of log-prob measure
        # score that uses 2 filters must have a look elsewhere effect
        # penalty of the ratio of amount of trials.
        # subtract - look_elsewhere_effect_penalty_single
        x_single = (
            math.chi_sq_minus_logsf_func(scores_max_single, 1)
            - look_elsewhere_effect_penalty_single
        )
        x_double = (
            math.chi_sq_minus_logsf_func(scores_max_double, 2)
            - look_elsewhere_effect_penalty_double
        )
        return math.norm_isf_func(max(x_single, x_double))

    def _init_template_bank(self, nbins: int) -> np.ndarray:
        if self.shape == "boxcar":
            templates = gen_boxcar_templates(self.widths, nbins)
        elif self.shape == "gaussian":
            templates = gen_gaussian_templates(self.widths, nbins)
        else:
            msg = f"Unknown template shape: {self.shape}"
            raise ValueError(msg)
        return templates

    def _init_shifts(self, shift: int) -> np.ndarray:
        if shift == 0:
            msg = "Shift must be greater than 0."
            raise ValueError(msg)
        return np.ones(self.ntemp, dtype=np.int64) * shift


@njit(cache=True, fastmath=True)
def snr_score_func(combined_res: np.ndarray) -> int:
    ts_e, ts_v = combined_res
    fold = ts_e / np.sqrt(ts_v)
    widths = generate_width_trials(len(fold), ducy_max=0.3, wtsp=1.1)
    return np.max(boxcar_snr_1d(fold, widths))


@njit(cache=True, fastmath=True)
def harmonic_summing_score_func(combined_res: np.ndarray, n_harmonics: int) -> float:
    ts_e, ts_v = combined_res
    fold = ts_e / np.sqrt(ts_v)
    fold_ft = np.fft.fft(fold) / np.sqrt(len(fold))

    best_score = -np.inf
    raw_score = 0
    for i in range(1, n_harmonics + 1):
        raw_score += np.abs(fold_ft[i]) ** 2
        score = math.chi_sq_minus_logsf_func(raw_score * 2, 2 * i)
        if score > best_score:
            best_score = score
    return math.norm_isf_func(best_score)


@njit(cache=True, fastmath=True)
def generate_width_trials(
    fold_bins: int,
    ducy_max: float = 0.2,
    wtsp: float = 1.5,
) -> np.ndarray:
    widths = []
    ww = 1
    wmax = int(max(1, ducy_max * fold_bins))
    while ww <= wmax:
        widths.append(ww)
        ww = int(max(ww + 1, wtsp * ww))
    return np.asarray(widths)


# Note: No cache=True when using parallel=True (numba issue)
@njit(fastmath=True)
def boxcar_snr(
    data: np.ndarray,
    widths: np.ndarray,
    stdnoise: float = 1.0,
) -> np.ndarray:
    widths = np.asarray(widths, dtype=np.uint64)
    nbins = data.shape[-1]
    folds = data.reshape(-1, nbins).astype(np.float32)
    snrs = boxcar_snr_2d(folds, widths, stdnoise)
    return snrs.reshape((*data.shape[:-1], widths.size))


@njit(parallel=True, fastmath=False)
def boxcar_snr_2d(
    folds: np.ndarray,
    widths: np.ndarray,
    stdnoise: float = 1.0,
) -> np.ndarray:
    nfolds = len(folds)
    snrs = np.zeros(shape=(nfolds, widths.size), dtype=np.float32)
    for ifold in prange(nfolds):
        snrs[ifold] = boxcar_snr_1d(folds[ifold], widths, stdnoise)
    return snrs


@njit(cache=True, fastmath=True)
def boxcar_snr_1d(
    norm_data: np.ndarray,
    widths: np.ndarray,
    stdnoise: float = 1.0,
) -> np.ndarray:
    size = len(norm_data)
    data_cumsum = np.cumsum(norm_data)
    total_sum = data_cumsum[-1]
    prefix_sum = np.concatenate(
        (data_cumsum, total_sum + np.cumsum(norm_data[: max(widths)])),
    )
    snr = np.zeros(len(widths), dtype=np.float32)
    for iw, width in enumerate(widths):
        height = np.sqrt((size - width) / (size * width))
        b = width * height / (size - width)
        dmax = np.max(prefix_sum[width : width + size] - prefix_sum[:size])
        snr[iw] = ((height + b) * dmax - b * total_sum) / stdnoise
    return snr


@njit(cache=True, fastmath=True)
def boxcar_snr_1do(
    norm_data: np.ndarray,
    widths: np.ndarray,
    stdnoise: float = 1.0,
) -> np.ndarray:
    size = len(norm_data)
    data_cumsum = np.cumsum(norm_data)
    total_sum = data_cumsum[-1]
    prefix_sum = np.concatenate(
        (data_cumsum, total_sum + np.cumsum(norm_data[: max(widths)])),
    )
    snr = np.zeros(len(widths), dtype=np.float32)
    for iw, width in enumerate(widths):
        height = 1 / np.sqrt(width)  # boxcar height = +h
        dmax = np.max(prefix_sum[width : width + size] - prefix_sum[:size])
        snr[iw] = height * dmax / stdnoise
    return snr
