from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit, prange, types

from pyloki.utils import maths, np_utils


@njit(cache=True, fastmath=True)
def generate_box_width_trials(
    fold_bins: int,
    ducy_max: float = 0.2,
    wtsp: float = 1.5,
) -> npt.NDArray[np.int64]:
    """Generate boxcar width trials for matched filtering.

    Parameters
    ----------
    fold_bins : int
        Number of bins in the folded profile.
    ducy_max : float, optional
        Maximum ducy cycle, by default 0.2
    wtsp : float, optional
        Spacing factor for the widths, by default 1.5

    Returns
    -------
    np.ndarray
        Width trials for matched filtering.
    """
    wmax = int(max(1, ducy_max * fold_bins))
    widths = [1]
    while widths[-1] <= wmax:
        next_width = int(max(widths[-1] + 1, int(wtsp * widths[-1])))
        if next_width > wmax:
            break
        widths.append(next_width)
    return np.asarray(widths, dtype=np.int64)


@njit("void(f4[::1], f4[::1])", cache=True, fastmath=True)
def circular_prefix_sum(
    x: npt.NDArray[np.float32],
    out: npt.NDArray[np.float32],
) -> None:
    """Compute circular prefix sum efficiently."""
    nbins = len(x)
    nsum = len(out)
    if nbins == 0 or nsum == 0:
        return
    # Initial prefix sum (inclusive scan)
    out[0] = x[0]
    for i in range(1, min(nbins, nsum)):
        out[i] = out[i - 1] + x[i]
    if nsum <= nbins:
        return
    # Handle wrap around
    last_sum = out[nbins - 1]
    n_wraps = nsum // nbins
    extra = nsum % nbins

    for i in range(1, n_wraps):
        offset = i * nbins
        for j in range(nbins):
            out[offset + j] = out[j] + i * last_sum

    if extra > 0:
        offset = n_wraps * nbins
        for j in range(extra):
            out[offset + j] = out[j] + n_wraps * last_sum


@njit("f4(f4[::1], f4[::1])", cache=True, fastmath=True)
def diff_max(x: npt.NDArray[np.float32], y: npt.NDArray[np.float32]) -> np.float32:
    """Find the maximum difference between two arrays efficiently.

    Not vectorised on some architectures. Slower for large arrays.
    """
    max_diff = -np.finfo(np.float32).max
    for i in range(len(x)):
        diff = x[i] - y[i]
        max_diff = max(max_diff, diff)
    return max_diff


@njit("f4(f4[::1], f4[::1])", cache=True, fastmath=True)
def diff_max_buffer(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.float32],
) -> np.float32:
    """Find the maximum difference between two arrays efficiently."""
    buffer = np.empty(len(x), dtype=np.float32)
    for i in range(len(x)):
        buffer[i] = x[i] - y[i]
    return np.max(buffer)


@njit(
    "f4[::1](f4[::1], i8[::1], f8)",
    cache=True,
    fastmath=True,
    locals={"size_w": types.f4},
)
def boxcar_snr_1d(
    norm_data: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
    stdnoise: float = 1.0,
) -> npt.NDArray[np.float32]:
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
    n_widths = len(widths)
    max_width = np.max(widths)
    prefix_sum = np.empty(size + max_width, dtype=np.float32)
    circular_prefix_sum(norm_data, prefix_sum)
    total_sum = prefix_sum[size - 1]
    snr = np.empty(n_widths, dtype=np.float32)
    inv_stdnoise = 1.0 / stdnoise
    for iw, width in enumerate(widths):
        size_w = size - width
        height = np.sqrt(size_w / (size * width))
        b = width * height / size_w
        dmax = diff_max(prefix_sum[width : width + size], prefix_sum[:size])
        snr[iw] = ((height + b) * dmax - (b * total_sum)) * inv_stdnoise
    return snr


@njit("f4[:, ::1](f4[:, ::1], i8[::1], f8)", cache=True, fastmath=True, parallel=True)
def boxcar_snr_2d(
    folds: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
    stdnoise: float = 1.0,
) -> npt.NDArray[np.float32]:
    nfolds, _ = folds.shape
    snrs = np.empty(shape=(nfolds, widths.size), dtype=np.float32)
    for ifold in prange(nfolds):
        snrs[ifold] = boxcar_snr_1d(folds[ifold], widths, stdnoise)
    return snrs


@njit(cache=True, fastmath=True)
def boxcar_snr_nd(
    data: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
    stdnoise: float = 1.0,
) -> npt.NDArray[np.float32]:
    nbins = data.shape[-1]
    folds = data.reshape(-1, nbins).astype(np.float32)
    snrs = boxcar_snr_2d(folds, widths, stdnoise)
    return snrs.reshape((*data.shape[:-1], widths.size))


@njit("f4(f4[:, ::1], i8[::1])", cache=True, fastmath=True)
def snr_score_func(
    combined_res: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> float:
    """Compute the SNR score for a folded suggestion."""
    ts_e, ts_v = combined_res
    fold = ts_e / np.sqrt(ts_v)
    return np.max(boxcar_snr_1d(fold, widths, 1.0))


@njit("f4(c8[:, ::1], i8[::1])", cache=True, fastmath=True)
def snr_score_func_complex(
    combined_res: npt.NDArray[np.complex64],
    widths: npt.NDArray[np.int64],
) -> float:
    """Compute the SNR score for a folded suggestion."""
    combined_res_t = np.fft.irfft(combined_res)
    return snr_score_func(combined_res_t, widths)


@njit("f4[::1](f4[:, :, ::1], i8[::1])", cache=True, fastmath=True, error_model="numpy")
def snr_score_batch_func(
    combined_res_batch: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Compute the SNR score for a batched folded suggestion."""
    batch_size, _, nbins = combined_res_batch.shape
    n_widths = len(widths)
    max_width = np.max(widths)
    prefix_sum = np.empty(nbins + max_width, dtype=np.float32)
    scores = np.empty(batch_size, dtype=np.float32)
    fold_norm = np.empty(nbins, dtype=np.float32)

    for i in range(batch_size):
        ts_e = combined_res_batch[i, 0, :]
        ts_v = combined_res_batch[i, 1, :]
        for j in range(nbins):
            fold_norm[j] = ts_e[j] / np.sqrt(ts_v[j])
        # Compute prefix sum
        circular_prefix_sum(fold_norm, prefix_sum)
        total_sum = prefix_sum[nbins - 1]

        # Compute SNR for each width
        max_snr = -np.finfo(np.float32).max
        for iw in range(n_widths):
            width = widths[iw]
            size_w = nbins - width
            height = np.sqrt(size_w / (nbins * width))
            b = width * height / size_w
            dmax = diff_max(prefix_sum[width : width + nbins], prefix_sum[:nbins])
            snr = (height + b) * dmax - (b * total_sum)
            max_snr = max(max_snr, snr)
        scores[i] = max_snr
    return scores


@njit("f4[::1](c8[:, :, ::1], i8[::1])", cache=True, fastmath=True, error_model="numpy")
def snr_score_batch_func_complex(
    combined_res_batch: npt.NDArray[np.complex64],
    widths: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Compute the SNR score for a batched folded suggestion stored in FFT format."""
    combined_res_batch_t = np.fft.irfft(combined_res_batch)
    return snr_score_batch_func(combined_res_batch_t, widths)


class MatchedFilter:
    """Matched filter class for computing SNR for a folded suggestion.

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
        if self.kind == "boxcar":
            templates = _gen_boxcar_templates(self.widths, nbins)
        elif self.kind == "gaussian":
            templates = _gen_gaussian_templates(self.widths, nbins)
        else:
            msg = f"Unknown template shape: {self.kind}"
            raise ValueError(msg)
        return templates

    def _init_shifts(self, shift: int) -> np.ndarray:
        if shift == 0:
            msg = "Shift must be greater than 0."
            raise ValueError(msg)
        return np.full(self.ntemp, shift, dtype=np.int64)


@njit("f4[::1](f4[::1])", cache=True, fastmath=True)
def _normalise(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Normalise to zero mean and unit energy.

    Parameters
    ----------
    arr : NDArray[np.float32]
        Input array.

    Returns
    -------
    NDArray[np.float32]
        Normalised array with zero mean and unit energy.
    """
    arr_mean = arr - np.mean(arr)
    arr_mean = arr_mean.astype(np.float32)
    norm = np.sqrt(np.dot(arr_mean, arr_mean))
    if norm == 0:
        return arr_mean
    return arr_mean / norm


@njit(cache=True, fastmath=True)
def _gen_boxcar_templates(
    widths: npt.NDArray[np.int64],
    nbins: int,
) -> npt.NDArray[np.float32]:
    """Generate boxcar templates.

    Parameters
    ----------
    widths : NDArray[np.int64]
        Widths of the boxcar templates.
    nbins : int
        Number of bins in the templates.

    Returns
    -------
    NDArray[np.float32]
        Normalised boxcar templates.
    """
    templates = np.zeros((len(widths), nbins), dtype=np.float32)
    for iw, width in enumerate(widths):
        templates[iw, :width] = 1
        templates[iw] = _normalise(templates[iw])
    return templates


@njit(cache=True, fastmath=True)
def _gen_gaussian_templates(
    widths: npt.NDArray[np.int64],
    nbins: int,
) -> npt.NDArray[np.float32]:
    """Generate Gaussian templates.

    Parameters
    ----------
    widths : NDArray[np.int64]
        Widths of the Gaussian templates.
    nbins : int
        Number of bins in the templates.

    Returns
    -------
    NDArray[np.float32]
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


@njit("f4[:, ::1](f4[:, ::1], i8[::1])", cache=True, fastmath=True)
def _get_e_mat(
    templates: npt.NDArray[np.float32],
    shifts: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Generate the matrix of templates.

    Parameters
    ----------
    templates : NDArray[np.float32]
        Normalised templates.
    shifts : NDArray[np.int64]
        Shifts (in bins) to apply to the templates

    Returns
    -------
    NDArray[np.float32]
        2D array of templates (nfilters, nbins).
    """
    ntemp, nbins = templates.shape
    total_shifts = int(np.sum(nbins // shifts))
    temp_bank = np.empty((total_shifts, nbins), dtype=np.float32)
    row_idx = 0
    for itemp in range(ntemp):
        for jbin in range(0, nbins, shifts[itemp]):
            temp_bank[row_idx] = np.roll(templates[itemp], jbin)
            row_idx += 1
    return temp_bank


@njit(cache=True, fastmath=True)
def compute_snr_e_mat(
    norm_data: npt.NDArray[np.float32],
    e_mat: npt.NDArray[np.float32],
    shifts: npt.NDArray[np.int64],
    stdnoise: float = 1.0,
) -> np.ndarray:
    projections = np.dot(e_mat, norm_data)
    _, nbins = e_mat.shape
    shifts_per_template = nbins // shifts
    # Extract maximum projection per template
    snr = np.empty(len(shifts_per_template), dtype=np.float32)
    start = 0
    for i, n_shifts in enumerate(shifts_per_template):
        end = start + int(n_shifts)
        snr[i] = np.max(projections[start:end]) / stdnoise
        start = end
    return snr


@njit(cache=True, fastmath=True)
def compute_snr_e_mat_2d(
    norm_data: npt.NDArray[np.float32],
    e_mat: npt.NDArray[np.float32],
    shifts: npt.NDArray[np.int64],
    stdnoise: float = 1.0,
) -> npt.NDArray[np.float32]:
    _, nbins = e_mat.shape
    n_templates = len(shifts)
    n_counts = norm_data.shape[0]
    # Compute projections: (n_counts, n_templates)
    projections = np.dot(norm_data, e_mat.T)

    # Compute cumulative shift indices
    shifts_per_template = nbins // shifts
    shift_indices = np.zeros(n_templates + 1, dtype=np.int64)
    for i, n_shifts in enumerate(shifts_per_template):
        shift_indices[i + 1] = shift_indices[i] + n_shifts
    snr = np.empty((n_counts, n_templates), dtype=np.float32)

    # Compute max projection per template across shifts
    for i in range(n_counts):
        proj_row = projections[i, :]
        for j in range(n_templates):
            start = shift_indices[j]
            end = shift_indices[j + 1]
            max_proj = proj_row[start]
            for k in range(start + 1, end):
                max_proj = max(max_proj, proj_row[k])
            snr[i, j] = max_proj / stdnoise
    return snr


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
            maths.chi_sq_minus_logsf_func(scores_max_single, 1) - lee_penalty_single
        )
        x_double = (
            maths.chi_sq_minus_logsf_func(scores_max_double, 2) - lee_penalty_double
        )
        results[iprof] = maths.norm_isf_func(max(x_single, x_double))
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
        score = maths.chi_sq_minus_logsf_func(raw_score * 2, 2 * i)
        best_score = max(score, best_score)
    return maths.norm_isf_func(best_score)
