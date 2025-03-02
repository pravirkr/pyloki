from __future__ import annotations

import numpy as np
from numba import njit, prange

from pyloki.utils import math, np_utils


@njit(cache=True, fastmath=True)
def generate_box_width_trials(
    fold_bins: int,
    ducy_max: float = 0.2,
    spacing_factor: float = 1.5,
) -> np.ndarray:
    """
    Generate boxcar width trials for matched filtering.

    Parameters
    ----------
    fold_bins : int
        Number of bins in the folded profile.
    ducy_max : float, optional
        Maximum ducy cycle, by default 0.2
    spacing_factor : float, optional
        Spacing factor for the widths, by default 1.5

    Returns
    -------
    np.ndarray
        Width trials for matched filtering.
    """
    wmax = int(max(1, ducy_max * fold_bins))
    widths = [1]
    while widths[-1] <= wmax:
        next_width = int(max(widths[-1] + 1, int(spacing_factor * widths[-1])))
        if next_width > wmax:
            break
        widths.append(next_width)
    return np.asarray(widths, dtype=np.int32)


@njit(cache=True, fastmath=True)
def boxcar_snr_1d(
    norm_data: np.ndarray,
    widths: np.ndarray,
    stdnoise: float = 1.0,
) -> np.ndarray:
    """Compute the SNR for a 1D profile using boxcar filters.

    Parameters
    ----------
    norm_data : np.ndarray
        Normalised data profile.
    widths : np.ndarray
        Widths of the boxcar filters.
    stdnoise : float, optional
        Standard deviation of the noise, by default 1.0

    Returns
    -------
    np.ndarray
        SNR scores for the given widths.
    """
    size = len(norm_data)
    data_cumsum = np.cumsum(norm_data)
    total_sum = data_cumsum[-1]
    max_width = np.max(widths)
    prefix_sum = np.concatenate(
        (data_cumsum, total_sum + np.cumsum(norm_data[:max_width])),
    )
    snr = np.empty(len(widths), dtype=np.float32)
    for iw, width in enumerate(widths):
        height = np.sqrt((size - width) / (size * width))
        b = width * height / (size - width)
        dmax = np.max(prefix_sum[width : width + size] - prefix_sum[:size])
        snr[iw] = ((height + b) * dmax - b * total_sum) / stdnoise
    return snr


@njit(cache=True, fastmath=True)
def snr_score_func(combined_res: np.ndarray) -> float:
    ts_e, ts_v = combined_res
    fold = ts_e / np.sqrt(ts_v)
    widths = generate_box_width_trials(len(fold), ducy_max=0.3, spacing_factor=1.1)
    return np.max(boxcar_snr_1d(fold, widths))


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
    snrs = _boxcar_snr_2d(folds, widths, stdnoise)
    return snrs.reshape((*data.shape[:-1], widths.size))


@njit(parallel=True, fastmath=False)
def _boxcar_snr_2d(
    folds: np.ndarray,
    widths: np.ndarray,
    stdnoise: float = 1.0,
) -> np.ndarray:
    nfolds, _ = folds.shape
    snrs = np.empty(shape=(nfolds, widths.size), dtype=np.float32)
    for ifold in prange(nfolds):
        snrs[ifold] = boxcar_snr_1d(folds[ifold], widths, stdnoise)
    return snrs


class MatchedFilter:
    """
    Matched filter class for computing SNR for a folded suggestion.

    Parameters
    ----------
    widths : np.ndarray
        Widths of the templates.
    nbins : int
        Number of bins in the folded profile.
    shifts : int, optional
        Shifts to apply to the templates, by default 0.
    kind : {"boxcar", "gaussian"}, optional
        Shape of the templates, by default "boxcar".

    Notes
    -----
    - The templates are mean-free and normalised to unit power.
    - The templates are circularly shifted by the given shifts.
    - Implements FFT for cases in which nbins >> 64.
    - Guassian filters are 5-10% more sensitive than boxcars for Gaussian pulses.
    """

    def __init__(
        self,
        widths: np.ndarray,
        nbins: int,
        shift: int = 1,
        kind: str = "boxcar",
    ) -> None:
        self.widths = widths
        self.nbins = nbins
        self.kind = kind
        self.templates = self._init_template_bank(nbins)
        self.shifts = self._init_shifts(shift)
        self.e_mat = _get_e_mat(self.templates, self.shifts)

    @property
    def ntemp(self) -> int:
        return len(self.templates)

    def compute_fft(self, folds: np.ndarray) -> np.ndarray:
        return _compute_snr_fft(folds, self.templates)

    def compute_dot(self, folds: np.ndarray) -> np.ndarray:
        return np.dot(self.e_mat, folds)

    def compute_dot_double(self, folds: np.ndarray) -> np.ndarray:
        return _compute_snr_double(folds, self.e_mat)

    def _init_template_bank(self, nbins: int) -> np.ndarray:
        if self.shape == "boxcar":
            templates = _gen_boxcar_templates(self.widths, nbins)
        elif self.shape == "gaussian":
            templates = _gen_gaussian_templates(self.widths, nbins)
        else:
            msg = f"Unknown template shape: {self.shape}"
            raise ValueError(msg)
        return templates

    def _init_shifts(self, shift: int) -> np.ndarray:
        if shift == 0:
            msg = "Shift must be greater than 0."
            raise ValueError(msg)
        return np.full(self.ntemp, shift, dtype=np.int64)


@njit(cache=True, fastmath=True)
def _normalise(arr: np.ndarray) -> np.ndarray:
    """
    Normalise to zero mean and unit power.

    Parameters
    ----------
    arr : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Normalised array with zero mean and unit power.
    """
    arr_norm = arr - np.mean(arr)
    return arr_norm / np.sqrt(np.dot(arr_norm, arr_norm))


@njit(cache=True, fastmath=True)
def _gen_boxcar_templates(widths: np.ndarray, nbins: int) -> np.ndarray:
    """
    Generate boxcar templates.

    Parameters
    ----------
    widths : np.ndarray
        Widths of the boxcar templates.
    nbins : int
        Number of bins in the templates.

    Returns
    -------
    np.ndarray
        Normalised boxcar templates.
    """
    templates = np.zeros((len(widths), nbins), dtype=np.float32)
    for iw, width in enumerate(widths):
        templates[iw, :width] = 1
        templates[iw] = _normalise(templates[iw])
    return templates


@njit(cache=True, fastmath=True)
def _gen_gaussian_templates(widths: np.ndarray, nbins: int) -> np.ndarray:
    """
    Generate Gaussian templates.

    Parameters
    ----------
    widths : np.ndarray
        Widths of the Gaussian templates.
    nbins : int
        Number of bins in the templates.

    Returns
    -------
    np.ndarray
        Normalised Gaussian templates.
    """
    templates = np.zeros((len(widths), nbins), dtype=np.float32)
    for iw, width in enumerate(widths):
        sigma = width / (2 * np.sqrt(2 * np.log(2)))
        xmax = int(np.ceil(3.5 * sigma))
        xx = np.arange(-xmax, xmax + 1)
        data = np.exp(-(xx**2) / (2 * sigma**2))
        if nbins >= 2 * xmax + 1:
            templates[iw, nbins // 2 - xmax : nbins // 2 + xmax + 1] = data
        else:
            start = xmax - nbins // 2
            end = start + nbins
            templates[iw, :nbins] = data[start:end]
        templates[iw] = _normalise(templates[iw])
    return templates


@njit(cache=True, fastmath=True)
def _get_e_mat(templates: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """
    Generate the matrix of templates.

    Parameters
    ----------
    templates : np.ndarray
        Normalised templates.
    shifts : np.ndarray
        Shifts to apply to the templates.

    Returns
    -------
    np.ndarray
        Matrix of templates.
    """
    ntemp, nbins = templates.shape
    total_shifts = np.sum(nbins // shifts)
    temp_bank = np.empty((total_shifts, nbins), dtype=templates.dtype)
    row_idx = 0
    for itemp in range(ntemp):
        for jbin in range(0, nbins, shifts[itemp]):
            temp_bank[row_idx] = np.roll(templates[itemp], jbin)
            row_idx += 1
    return temp_bank


@njit(cache=True, fastmath=True)
def _compute_snr_fft(data: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """
    Compute the SNR using FFT (convolution theorem).

    Parameters
    ----------
    data : np.ndarray
        Folded normalised data profiles (nprof, nbins).
    templates : np.ndarray
        Normalised templates (ntemp, nbins).

    Returns
    -------
    np.ndarray
        SNR scores (nprof, ntemp).
    """
    folds = data.reshape(-1, data.shape[-1]).astype(np.float32)
    nprof, nbins = folds.shape
    ntemp, _ = templates.shape

    xx = np_utils.cpadpow2(folds)
    yy = np_utils.cpad2len(templates, xx.shape[-1])
    fx = np.fft.rfft(xx)
    fy = np.fft.rfft(yy)

    result = np.empty((nprof, ntemp), dtype=np.float32)
    for iprof in range(nprof):
        for jtemp in range(ntemp):
            snr = np.fft.irfft(fx[iprof] * fy[jtemp])
            result[iprof, jtemp] = np.max(snr[:nbins])
    return result.reshape((*data.shape[:-1], ntemp))


@njit(cache=True, fastmath=True, parallel=True)
def _compute_snr_double(
    data: np.ndarray,
    e_mat: np.ndarray,
    n_corr: int = 10,
) -> np.ndarray:
    """
    Compute the SNR using convolution for two-peaked features.

    Parameters
    ----------
    data : np.ndarray
        Folded normalised data profile (nprof, nbins).
    e_mat : np.ndarray
        Matrix of normalized templates (nfilters, nbins).
    n_corr : int, optional
        Number of correlations to consider, by default 10.

    Returns
    -------
    np.ndarray
        SNR scores (nprof).

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
    folds = data.reshape(-1, data.shape[-1])
    nprof, _ = folds.shape
    n_filters, _ = e_mat.shape

    lee_penalty_double = np.log2((n_filters * n_filters - 1) / 2)
    lee_penalty_single = np.log2(n_filters)
    inner_product_matrix = np.dot(e_mat, e_mat.T)

    results = np.empty(nprof, dtype=np.float32)
    for iprof in prange(nprof):
        scores = np.dot(e_mat, folds[iprof])
        scores_max_single = np.max(scores) ** 2

        # good indices are the indices of the top scores
        # not considering the negative maxima as relevant
        good_indices = np.argsort(scores)[-n_corr:]
        scores_max_double = scores_max_single
        s = 0
        for i in range(n_corr - 1):
            for j in range(i + 1, n_corr):
                idx_i, idx_j = good_indices[i], good_indices[j]
                gamma_ij = inner_product_matrix[idx_i, idx_j]
                if abs(gamma_ij) < 0.99:
                    s = (
                        scores[idx_i] ** 2
                        + scores[idx_j] ** 2
                        - 2 * gamma_ij * scores[idx_i] * scores[idx_j]
                    ) / (1 - gamma_ij**2)
                scores_max_double = max(s, scores_max_double)
        # comparing scores should be on the basis of log-prob measure
        # score that uses 2 filters must have a look elsewhere effect
        # penalty of the ratio of amount of trials.
        # subtract - look_elsewhere_effect_penalty_single
        x_single = (
            math.chi_sq_minus_logsf_func(scores_max_single, 1) - lee_penalty_single
        )
        x_double = (
            math.chi_sq_minus_logsf_func(scores_max_double, 2) - lee_penalty_double
        )
        results[iprof] = math.norm_isf_func(max(x_single, x_double))
    return results.reshape(data.shape[:-1])


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
        best_score = max(score, best_score)
    return math.norm_isf_func(best_score)
