from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
import numpy.typing as npt
from numba import njit, prange, types

from pyloki.utils import maths

if TYPE_CHECKING:
    from collections.abc import Sequence


@njit(cache=True, fastmath=True)
def generate_box_width_trials(
    nbins: int,
    ducy_max: float = 0.2,
    wtsp: float = 1.5,
) -> npt.NDArray[np.int64]:
    """Generate boxcar width trials for matched filtering.

    Parameters
    ----------
    nbins : int
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
    if nbins < 2:
        msg = "nbins must be at least 2."
        raise ValueError(msg)
    wmax = min(nbins - 1, int(max(1, ducy_max * nbins)))
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
    if size == 0 or n_widths == 0:
        return np.zeros(n_widths, dtype=np.float32)
    max_width = max(0, int(np.max(widths)))
    prefix_sum = np.empty(size + max_width, dtype=np.float32)
    circular_prefix_sum(norm_data, prefix_sum)
    total_sum = prefix_sum[size - 1]
    snr = np.empty(n_widths, dtype=np.float32)
    inv_stdnoise = 1.0 / stdnoise
    for iw, width in enumerate(widths):
        if width <= 0 or width >= size:
            snr[iw] = 0.0
            continue
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


@njit("f4[:, ::1](f4[:, ::1], i8[::1], f8)", cache=True, fastmath=True)
def boxcar_snr_2d_serial(
    folds: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
    stdnoise: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Compute boxcar S/N serially (`boxcar_snr_2d` twin for parallel regions).

    Identical results; only the `prange` is a plain `range`. Calling the parallel
    version from inside another parallel region is a nested parallel launch, which
    numba's default `workqueue` threading layer cannot do -- it aborts the process
    with "Concurrent access has been detected". Nesting would not buy anything here
    anyway, since the enclosing loop already saturates the workers.
    """
    nfolds, _ = folds.shape
    snrs = np.empty(shape=(nfolds, widths.size), dtype=np.float32)
    for ifold in range(nfolds):
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


@njit(
    "f4[::1](f4[::1], f4[::1], i8[::1])",
    cache=True,
    fastmath=True,
    locals={"size_w": types.f4},
)
def boxcar_snr_beta_1d(
    ts_e: npt.NDArray[np.float32],
    ts_v: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Compute the optimal SNR_beta for a 1D profile using boxcar filters.

    Parameters
    ----------
    ts_e : np.ndarray
        Weighted data profile P_w(b).
    ts_v : np.ndarray
        Weight/Variance profile P_s(b).
    widths : np.ndarray
        Widths of the boxcar filters.

    Returns
    -------
    np.ndarray
        Max SNR scores for the given widths.
    """
    size = len(ts_e)
    n_widths = len(widths)
    if size == 0 or n_widths == 0:
        return np.zeros(n_widths, dtype=np.float32)
    max_width = max(0, int(np.max(widths)))

    # Allocate and compute prefix sums for BOTH arrays
    prefix_w = np.empty(size + max_width, dtype=np.float32)
    prefix_s = np.empty(size + max_width, dtype=np.float32)
    circular_prefix_sum(ts_e, prefix_w)
    circular_prefix_sum(ts_v, prefix_s)

    total_w = prefix_w[size - 1]
    total_s = prefix_s[size - 1]
    snr = np.empty(n_widths, dtype=np.float32)
    for iw, width in enumerate(widths):
        if width <= 0 or width >= size:
            snr[iw] = 0.0
            continue
        size_w = size - width
        # Zero-mean template coefficients
        height = np.sqrt(size_w / (size * width))
        b = width * height / size_w
        max_snr = -np.finfo(np.float32).max
        for i in range(size):
            dw = prefix_w[i + width] - prefix_w[i]
            ds = prefix_s[i + width] - prefix_s[i]
            num = (height + b) * dw - b * total_w
            den2 = (height * height) * ds + (b * b) * (total_s - ds)
            if den2 > 0:
                current_snr = num / np.float32(np.sqrt(den2))
                max_snr = max(max_snr, current_snr)

        snr[iw] = max_snr

    return snr


@njit("f4[::1](f4[::1], f4[::1], i8[::1])", cache=True, fastmath=True)
def boxcar_snr_glrt_1d(
    ts_e: npt.NDArray[np.float32],
    ts_v: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    r"""Compute exact heteroscedastic GLRT boxcar S/N for a 1D ``(E, V)`` profile.

    Under :math:`H_0: E_b \sim \mathcal{N}(\mu_0 V_b, V_b)` and
    :math:`H_1: E_b \sim \mathcal{N}((\mu_0 + A T_b) V_b, V_b)`, the profile
    likelihood (zero-mean background-marginalized) signed S/N for window
    :math:`W = [i, i + w)` is:

    .. math::
        \mathrm{SNR}_{\mathrm{GLRT}}(i, w)
        = \frac{E_W - (V_W / V_{\mathrm{tot}}) E_{\mathrm{tot}}}
                {\sqrt{V_W (1 - V_W / V_{\mathrm{tot}})}}

    which handles arbitrary non-uniform weights including zero-weight masked
    bins (``ts_v[b] == 0``) and reduces identically to ``boxcar_snr_1d`` when
    ``ts_v`` is uniform.
    """
    size = len(ts_e)
    n_widths = len(widths)
    if size == 0 or n_widths == 0:
        return np.zeros(n_widths, dtype=np.float32)
    max_width = max(0, int(np.max(widths)))
    prefix_e = np.empty(size + max_width, dtype=np.float32)
    prefix_v = np.empty(size + max_width, dtype=np.float32)
    circular_prefix_sum(ts_e, prefix_e)
    circular_prefix_sum(ts_v, prefix_v)

    total_e = float(prefix_e[size - 1])
    total_v = float(prefix_v[size - 1])
    snr = np.zeros(n_widths, dtype=np.float32)
    if total_v <= 0.0:
        return snr

    eps = 1e-12 * total_v
    for iw in range(n_widths):
        width = widths[iw]
        if width <= 0 or width >= size:
            snr[iw] = np.float32(0.0)
            continue
        max_snr = -np.finfo(np.float32).max
        has_valid = False
        for i in range(size):
            ew = float(prefix_e[i + width] - prefix_e[i])
            vw = float(prefix_v[i + width] - prefix_v[i])
            voff = total_v - vw
            if vw > eps and voff > eps:
                frac = vw / total_v
                num = ew - frac * total_e
                den = np.sqrt(vw * (1.0 - frac))
                if den > 0.0:
                    cur = np.float32(num / den)
                    if not has_valid or cur > max_snr:
                        max_snr = cur
                        has_valid = True
        snr[iw] = max_snr if has_valid else np.float32(0.0)
    return snr


@njit(
    "f4[:, ::1](f4[:, ::1], f4[:, ::1], i8[::1])",
    cache=True,
    fastmath=True,
    parallel=True,
)
def boxcar_snr_glrt_2d(
    ts_e_2d: npt.NDArray[np.float32],
    ts_v_2d: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Compute heteroscedastic GLRT boxcar S/N in parallel for 2D ``(E, V)`` arrays."""
    nfolds, _ = ts_e_2d.shape
    snrs = np.empty((nfolds, widths.size), dtype=np.float32)
    for ifold in prange(nfolds):
        snrs[ifold] = boxcar_snr_glrt_1d(ts_e_2d[ifold], ts_v_2d[ifold], widths)
    return snrs


@njit("f4[:, ::1](f4[:, ::1], f4[:, ::1], i8[::1])", cache=True, fastmath=True)
def boxcar_snr_glrt_2d_serial(
    ts_e_2d: npt.NDArray[np.float32],
    ts_v_2d: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Compute heteroscedastic GLRT boxcar S/N serially for parallel regions."""
    nfolds, _ = ts_e_2d.shape
    snrs = np.empty((nfolds, widths.size), dtype=np.float32)
    for ifold in range(nfolds):
        snrs[ifold] = boxcar_snr_glrt_1d(ts_e_2d[ifold], ts_v_2d[ifold], widths)
    return snrs


@njit("f4[:, ::1](f4[:, :, ::1], i8[::1])", cache=True, fastmath=True, parallel=True)
def boxcar_snr_glrt_3d(
    folds_ev: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Compute heteroscedastic GLRT boxcar S/N for ``(nfolds, 2, nbins)`` folds."""
    nfolds = folds_ev.shape[0]
    snrs = np.empty((nfolds, widths.size), dtype=np.float32)
    for ifold in prange(nfolds):
        snrs[ifold] = boxcar_snr_glrt_1d(
            folds_ev[ifold, 0, :],
            folds_ev[ifold, 1, :],
            widths,
        )
    return snrs


@njit("f4(f4[:, ::1], i8[::1])", cache=True, fastmath=True)
def snr_glrt_score_func(
    combined_res: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> float:
    """Compute max heteroscedastic GLRT boxcar S/N for a single ``(2, nbins)`` fold."""
    return float(np.max(boxcar_snr_glrt_1d(combined_res[0], combined_res[1], widths)))


@njit("f4[::1](f4[:, :, ::1], i8[::1])", cache=True, fastmath=True, parallel=True)
def snr_glrt_score_batch_func(
    combined_res_batch: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Compute max heteroscedastic GLRT boxcar S/N for a batch of folds."""
    batch_size = combined_res_batch.shape[0]
    scores = np.empty(batch_size, dtype=np.float32)
    for i in prange(batch_size):
        scores[i] = snr_glrt_score_func(combined_res_batch[i], widths)
    return scores


@njit("f4[::1](f4[:], f4[:])", cache=True, fastmath=True)
def _standardize_ev_1d(
    ts_e: npt.NDArray[np.float32],
    ts_v: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Standardize ``ts_e / sqrt(ts_v)`` safely, zeroing bins with ``ts_v <= 0``."""
    n = ts_e.size
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        v = ts_v[i]
        if v > np.float32(0.0):
            out[i] = ts_e[i] / np.sqrt(v)
        else:
            out[i] = np.float32(0.0)
    return out


@njit("f4(f4[:, ::1], i8[::1])", cache=True, fastmath=True)
def snr_score_func(
    combined_res: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> float:
    """Compute the SNR score for a folded suggestion."""
    ts_e, ts_v = combined_res
    fold = _standardize_ev_1d(ts_e, ts_v)
    return float(np.max(boxcar_snr_1d(fold, widths, 1.0)))


@njit("f4(c8[:, ::1], i8[::1])", cache=True, fastmath=True)
def snr_score_func_complex(
    combined_res: npt.NDArray[np.complex64],
    widths: npt.NDArray[np.int64],
) -> float:
    """Compute the SNR score for a folded suggestion."""
    combined_res_t = np.fft.irfft(combined_res).astype(np.float32)
    return snr_score_func(combined_res_t, widths)


@njit("f4[::1](f4[:, :, ::1], i8[::1])", cache=True, fastmath=True, error_model="numpy")
def snr_score_batch_func(
    combined_res_batch: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Compute the SNR score for a batched folded suggestion."""
    batch_size, _, nbins = combined_res_batch.shape
    n_widths = len(widths)
    if batch_size == 0 or nbins == 0 or n_widths == 0:
        return np.zeros(batch_size, dtype=np.float32)
    max_width = max(0, int(np.max(widths)))
    prefix_sum = np.empty(nbins + max_width, dtype=np.float32)
    scores = np.empty(batch_size, dtype=np.float32)
    fold_norm = np.empty(nbins, dtype=np.float32)

    for i in range(batch_size):
        ts_e = combined_res_batch[i, 0, :]
        ts_v = combined_res_batch[i, 1, :]
        for j in range(nbins):
            v = ts_v[j]
            if v > np.float32(0.0):
                fold_norm[j] = ts_e[j] / np.sqrt(v)
            else:
                fold_norm[j] = np.float32(0.0)
        # Compute prefix sum
        circular_prefix_sum(fold_norm, prefix_sum)
        total_sum = prefix_sum[nbins - 1]

        # Compute SNR for each width
        max_snr = -np.finfo(np.float32).max
        for iw in range(n_widths):
            width = widths[iw]
            if width <= 0 or width >= nbins:
                snr = np.float32(0.0)
            else:
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
    combined_res_batch_t = np.fft.irfft(combined_res_batch).astype(np.float32)
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
        folds_arr = np.asarray(folds, dtype=np.float32)
        if folds_arr.ndim == 0 or folds_arr.shape[-1] != self.nbins:
            msg = f"Expected trailing axis {self.nbins}, got shape {folds_arr.shape}"
            raise ValueError(msg)
        folds_2d = np.ascontiguousarray(folds_arr.reshape(-1, self.nbins))
        snr_2d = compute_snr_e_mat_2d(folds_2d, self.e_mat, self.shifts, 1.0)
        return snr_2d.reshape((*folds_arr.shape[:-1], self.ntemp))

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
        if shift <= 0:
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
    if nbins < 2:
        msg = "nbins must be at least 2."
        raise ValueError(msg)
    templates = np.zeros((len(widths), nbins), dtype=np.float32)
    for iw, width in enumerate(widths):
        if width <= 0 or width >= nbins:
            msg = "Boxcar widths must satisfy 0 < width < nbins."
            raise ValueError(msg)
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
    if nbins < 1:
        msg = "nbins must be positive."
        raise ValueError(msg)
    templates = np.zeros((len(widths), nbins), dtype=np.float32)
    for iw, width in enumerate(widths):
        if width <= 0:
            msg = "Template widths must be positive."
            raise ValueError(msg)
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
    if np.any(shifts <= 0):
        msg = "shifts must be positive."
        raise ValueError(msg)
    ntemp, nbins = templates.shape
    shifts_per_template = (nbins + shifts - 1) // shifts
    total_shifts = int(np.sum(shifts_per_template))
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
    shifts_per_template = (nbins + shifts - 1) // shifts
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
    shifts_per_template = (nbins + shifts - 1) // shifts
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
    Compute the SNR using FFT circular cross-correlation.

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
    if nprof == 0:
        return np.empty((*data.shape[:-1], ntemp), dtype=np.float32)
    templates_f32 = templates.astype(np.float32)

    fx = np.fft.rfft(folds, n=nbins)
    fy_conj = np.conj(np.fft.rfft(templates_f32, n=nbins))

    result = np.empty((nprof, ntemp), dtype=np.float32)
    for iprof in range(nprof):
        corr = np.fft.irfft(fx[iprof] * fy_conj, n=nbins)
        for jtemp in range(ntemp):
            result[iprof, jtemp] = np.max(corr[jtemp])
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
    if e_mat.shape[0] < 1:
        msg = "e_mat must contain at least one template."
        raise ValueError(msg)
    if n_corr < 1:
        msg = "n_corr must be at least 1."
        raise ValueError(msg)
    folds = data.reshape(-1, data.shape[-1]).astype(np.float32)
    nprof, _ = folds.shape
    if nprof == 0:
        return np.empty(data.shape[:-1], dtype=np.float32)
    e_mat_f32 = e_mat.astype(np.float32)
    n_filters, _ = e_mat_f32.shape
    k_corr = min(n_corr, n_filters)

    lee_penalty_single = np.log(n_filters)
    n_pairs = (n_filters * (n_filters - 1)) // 2
    lee_penalty_double = np.log(n_pairs) if n_pairs > 0 else 0.0
    inner_product_matrix = np.dot(e_mat_f32, e_mat_f32.T)

    results = np.empty(nprof, dtype=np.float32)
    for iprof in prange(nprof):
        scores = np.dot(e_mat_f32, folds[iprof])
        max_single = max(0.0, float(np.max(scores)))
        scores_max_single = max_single * max_single

        # good indices are the indices of the top scores
        # Require positive solved component amplitudes (a_i > 0, a_j > 0) so a
        # single bright pulse plus an adjacent negative trough is rejected while
        # disjoint positive pulses with negative baseline correlation are kept.
        good_indices = np.argsort(scores)[-k_corr:]
        scores_max_double = 0.0
        has_double = False
        for i in range(k_corr - 1):
            for j in range(i + 1, k_corr):
                idx_i, idx_j = good_indices[i], good_indices[j]
                si, sj = float(scores[idx_i]), float(scores[idx_j])
                gamma_ij = float(inner_product_matrix[idx_i, idx_j])
                if abs(gamma_ij) < 0.99:
                    num_i = si - gamma_ij * sj
                    num_j = sj - gamma_ij * si
                    if num_i > 0.0 and num_j > 0.0:
                        denom = 1.0 - gamma_ij * gamma_ij
                        s = (si * num_i + sj * num_j) / denom
                        scores_max_double = max(scores_max_double, s)
                        has_double = True
        # Compare hypotheses via minus log survival probability with natural-log
        # look-elsewhere penalties (exact 2-DOF survival: -ln exp(-s/2) = s / 2).
        x_single = (
            maths.chi_sq_minus_logsf_func(scores_max_single, 1) - lee_penalty_single
        )
        if has_double:
            x_double = 0.5 * scores_max_double - lee_penalty_double
            best_x = max(x_single, x_double)
        else:
            best_x = x_single
        results[iprof] = maths.norm_isf_func(best_x)
    return results.reshape(data.shape[:-1])


@njit(cache=True, fastmath=True)
def harmonic_summing_score_func(combined_res: np.ndarray, n_harmonics: int) -> float:
    """Compute trial-penalized Gaussian significance via harmonic summing.

    Harmonics are clamped at ``min(n_harmonics, (nbins - 1) // 2, 32)`` to
    exclude Nyquist/Hermitian-mirror bins and stay within the 64-DOF chi-square
    lookup table, and a conservative ``-ln(max_h)`` union-bound trials penalty
    is subtracted across the harmonic cutoffs.
    """
    ts_e, ts_v = combined_res
    fold = _standardize_ev_1d(ts_e, ts_v)
    nbins = len(fold)
    max_h = min(n_harmonics, (nbins - 1) // 2, 32)
    if max_h < 1:
        msg = (
            "n_harmonics and profile length must allow at least 1 independent harmonic."
        )
        raise ValueError(msg)
    fold_ft = np.fft.rfft(fold) / np.sqrt(nbins)
    lee_penalty = np.log(max_h)

    best_score = -np.inf
    raw_score = 0.0
    for i in range(1, max_h + 1):
        raw_score += np.abs(fold_ft[i]) ** 2
        score = maths.chi_sq_minus_logsf_func(raw_score * 2.0, 2 * i)
        best_score = max(score, best_score)
    return maths.norm_isf_func(best_score - lee_penalty)


@njit(
    "Tuple((f8, i8, i8))(f4[::1], f4[::1])",
    cache=True,
    fastmath=True,
)
def kadane_circular_normalized_1d(
    norm_data: npt.NDArray[np.float32],
    biases: npt.NDArray[np.float32],
) -> tuple[float, int, int]:
    """Zero-allocation circular Kadane. Returns (SNR score, width, start).

    Uses inverted sub-array subtraction for O(1) space complexity.
    The maximum circular subarray can only exist in two states:

    Unwrapped: It sits contiguous in the middle of the array - Standard Kadane

    Wrapped: It sits on the edges, crossing the boundary. If the maximum sum
    wraps around the edges, then the elements it excludes form a contiguous block
    in the middle. By definition, this excluded block must be the minimum
    contiguous subarray.

    Therefore: Wrapped Max Sum = Total Array Sum - Minimum Subarray Sum
    """
    size = len(norm_data)

    # 1. Compute mean for dynamic DC subtraction
    total_sum = np.float64(0.0)
    for i in range(size):
        total_sum += norm_data[i]
    mean_val = total_sum / size

    best_global_snr = -np.finfo(np.float64).max
    best_global_w = 1
    best_global_start = 0

    for k in range(len(biases)):
        bias = np.float64(biases[k])

        # State for Standard Max Kadane (Unwrapped)
        max_sum = -np.finfo(np.float64).max
        cur_max = np.float64(0.0)
        start_max, end_max, tmp_start_max = 0, 0, 0

        # State for Min Kadane (Excluded block for Wrapped)
        min_sum = np.finfo(np.float64).max
        cur_min = np.float64(0.0)
        start_min, end_min, tmp_start_min = 0, 0, 0

        for i in range(size):
            # Subtract mean AND the Kadane geometric bias penalty
            val = np.float64(norm_data[i]) - mean_val - bias

            # 1. Track the Maximum Contiguous Subarray
            cur_max += val
            if cur_max > max_sum:
                max_sum = cur_max
                start_max = tmp_start_max
                end_max = i
            if cur_max < 0.0:
                cur_max = np.float64(0.0)
                tmp_start_max = i + 1

            # 2. Track the Minimum Contiguous Subarray
            cur_min += val
            if cur_min < min_sum:
                min_sum = cur_min
                start_min = tmp_start_min
                end_min = i
            if cur_min > 0.0:
                cur_min = np.float64(0.0)
                tmp_start_min = i + 1

        # Candidate 1: Non-wrapped maximum
        best_biased_sum = max_sum
        best_start = start_max
        best_width = end_max - start_max + 1

        # Candidate 2: Wrapped maximum
        excluded_width = end_min - start_min + 1
        wrapped_width = size - excluded_width

        if wrapped_width > 0:
            # Wrapped maximum = Total Array Sum - Minimum Excluded Block
            wrapped_sum = (-size * bias) - min_sum
            if wrapped_sum > best_biased_sum:
                best_biased_sum = wrapped_sum
                best_start = (end_min + 1) % size
                best_width = wrapped_width

        # Recover the mean-subtracted sum (remove the bias penalty)
        unbiased_sum = best_biased_sum + best_width * bias

        # Apply the exact production Boxcar normalization scaling
        if 0 < best_width < size:
            snr = unbiased_sum * np.sqrt(size / (best_width * (size - best_width)))
        else:
            snr = 0.0

        if snr > best_global_snr:
            best_global_snr = snr
            best_global_w = best_width
            best_global_start = best_start

    return float(best_global_snr), best_global_w, best_global_start


@njit("f4(f4[::1], i8[::1])", cache=True, fastmath=True)
def double_boxcar_snr_1d(
    norm_data: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> float:
    r"""Compute trial-penalized single- or disjoint-double-peak boxcar S/N.

    Finds the strongest single boxcar window ``(s1, w1)`` and its best
    non-overlapping partner ``(s2, w2)`` on the phase circle, evaluates the
    exact correlated 2-DOF :math:`\chi^2_2` statistic with off-pulse correlation
    :math:`\gamma_{12} = -\sqrt{w_1 w_2 / ((B - w_1)(B - w_2))}`, and converts
    to equivalent Gaussian significance via natural-log look-elsewhere penalties.
    """
    size = len(norm_data)
    n_widths = len(widths)
    if size < 2 or n_widths == 0:
        return 0.0
    max_width = max(0, int(np.max(widths)))
    psum = np.empty(size + max_width, dtype=np.float32)
    circular_prefix_sum(norm_data, psum)
    total_sum = float(psum[size - 1])

    best_z1 = 0.0
    best_s1 = 0
    best_w1 = 0
    n_valid_widths = 0

    for iw in range(n_widths):
        w = int(widths[iw])
        if w <= 0 or w >= size:
            continue
        n_valid_widths += 1
        size_w = float(size - w)
        height = np.sqrt(size_w / (float(size) * float(w)))
        b = float(w) * height / size_w
        hb = height + b
        for i in range(size):
            dw = float(psum[i + w] - psum[i])
            z = hb * dw - b * total_sum
            if z > best_z1:
                best_z1 = z
                best_s1 = (i + 1) % size
                best_w1 = w

    if n_valid_widths == 0:
        return 0.0

    n_trials = n_valid_widths * size
    lee_single = np.log(float(n_trials))
    n_pairs = (n_trials * (n_trials - 1)) // 2
    lee_double = np.log(float(n_pairs)) if n_pairs > 0 else 0.0

    x_single = maths.chi_sq_minus_logsf_func(best_z1 * best_z1, 1) - lee_single

    if best_w1 <= 0 or best_z1 <= 0.0:
        return float(maths.norm_isf_func(x_single))

    best_s2_chi2 = 0.0
    has_double = False
    for iw in range(n_widths):
        w2 = int(widths[iw])
        if w2 <= 0 or w2 + best_w1 >= size:
            continue
        size_w2 = float(size - w2)
        height2 = np.sqrt(size_w2 / (float(size) * float(w2)))
        b2 = float(w2) * height2 / size_w2
        hb2 = height2 + b2
        gamma = -np.sqrt(
            (float(best_w1) * float(w2)) / (float(size - best_w1) * size_w2),
        )
        if abs(gamma) >= 0.99:
            continue
        inv_denom = 1.0 / (1.0 - gamma * gamma)
        for i in range(size):
            s2 = (i + 1) % size
            dist = (s2 - best_s1) % size
            if dist < best_w1 or dist + w2 > size:
                continue
            dw2 = float(psum[i + w2] - psum[i])
            z2 = hb2 * dw2 - b2 * total_sum
            if z2 <= 0.0:
                continue
            num1 = best_z1 - gamma * z2
            num2 = z2 - gamma * best_z1
            if num1 > 0.0 and num2 > 0.0:
                s_val = (best_z1 * num1 + z2 * num2) * inv_denom
                if s_val > best_s2_chi2:
                    best_s2_chi2 = s_val
                    has_double = True

    if has_double:
        x_double = 0.5 * best_s2_chi2 - lee_double
        best_x = max(x_single, x_double)
    else:
        best_x = x_single
    return float(maths.norm_isf_func(best_x))


@njit("f4[::1](f4[:, ::1], i8[::1])", cache=True, fastmath=True, parallel=True)
def double_boxcar_snr_2d(
    folds: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
) -> npt.NDArray[np.float32]:
    """Compute disjoint two-window boxcar S/N in parallel for 2D profiles."""
    nfolds = folds.shape[0]
    scores = np.empty(nfolds, dtype=np.float32)
    for i in prange(nfolds):
        scores[i] = double_boxcar_snr_1d(folds[i], widths)
    return scores


@njit("f4[::1](f4[:, :, ::1], i8)", cache=True, fastmath=True, parallel=True)
def harmonic_summing_snr_2d(
    folds_ev: npt.NDArray[np.float32],
    n_harmonics: int,
) -> npt.NDArray[np.float32]:
    """Compute trial-penalized Fourier :math:`Z^2_m` harmonic-summing S/N."""
    nfolds = folds_ev.shape[0]
    scores = np.empty(nfolds, dtype=np.float32)
    for i in prange(nfolds):
        scores[i] = harmonic_summing_score_func(folds_ev[i], n_harmonics)
    return scores


@njit("f4(f4[:, ::1], i8[::1], i8)", cache=True, fastmath=True)
def combined_score_func(
    combined_res: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
    n_harmonics: int,
) -> float:
    """Compute hybrid ``max(harmonic_z2, boxcar)`` S/N for a ``(2, nbins)`` fold."""
    s_box = snr_score_func(combined_res, widths)
    s_harm = harmonic_summing_score_func(combined_res, n_harmonics)
    return float(max(s_box, s_harm))


@njit("f4[::1](f4[:, :, ::1], i8[::1], i8)", cache=True, fastmath=True, parallel=True)
def combined_score_2d(
    folds_ev: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
    n_harmonics: int,
) -> npt.NDArray[np.float32]:
    """Compute hybrid ``max(harmonic_z2, boxcar)`` S/N in parallel."""
    nfolds = folds_ev.shape[0]
    scores = np.empty(nfolds, dtype=np.float32)
    for i in prange(nfolds):
        scores[i] = combined_score_func(folds_ev[i], widths, n_harmonics)
    return scores


def generate_kadane_biases(
    nbins: int,
    ducy_max: float = 0.2,
    n_biases: int = 6,
) -> npt.NDArray[np.float32]:
    """Generate geometric Kadane bias penalties spanning ``[1/nbins, ducy_max]``."""
    if nbins < 2:
        msg = "nbins must be at least 2."
        raise ValueError(msg)
    if n_biases < 1:
        msg = "n_biases must be at least 1."
        raise ValueError(msg)
    wmax = max(1, min(nbins - 1, int(ducy_max * nbins)))
    # Optimal bias scales inversely with sqrt(width)
    return np.geomspace(1.0 / np.sqrt(wmax), 1.0, n_biases, dtype=np.float32)


@njit("f4[::1](f4[:, ::1], f4[::1])", cache=True, fastmath=True, parallel=True)
def kadane_circular_snr_2d(
    folds: npt.NDArray[np.float32],
    biases: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute circular Kadane S/N in parallel for 2D normalized profiles."""
    nfolds = folds.shape[0]
    scores = np.empty(nfolds, dtype=np.float32)
    for i in prange(nfolds):
        snr, _, _ = kadane_circular_normalized_1d(folds[i], biases)
        scores[i] = np.float32(snr)
    return scores


ScoreMode = Literal[
    "boxcar",
    "boxcar_glrt",
    "double_boxcar",
    "double_matched",
    "harmonic_z2",
    "combined",
    "kadane",
]


def score_profiles(
    folds_ev: npt.NDArray[np.float32],
    widths: npt.NDArray[np.int64],
    *,
    mode: ScoreMode = "boxcar",
    n_harmonics: int = 16,
    e_mat: npt.NDArray[np.float32] | None = None,
    biases: npt.NDArray[np.float32] | None = None,
    n_corr: int = 10,
) -> npt.NDArray[np.float32]:
    """Score a batch of ``(nfolds, 2, nbins)`` ``(E, V)`` folds with ``mode``.

    Parameters
    ----------
    folds_ev : NDArray[np.float32]
        Folded profiles with shape ``(nfolds, 2, nbins)`` where axis 1 holds
        ``(ts_e, ts_v)``.
    widths : NDArray[np.int64]
        Boxcar width trials.
    mode : ScoreMode, optional
        Scoring algorithm, by default ``"boxcar"``.
    n_harmonics : int, optional
        Maximum Fourier harmonics for ``"harmonic_z2"`` / ``"combined"``.
    e_mat : NDArray[np.float32] | None, optional
        Normalized template matrix for ``"double_matched"``. If ``None``, built
        automatically from ``widths`` using ``MatchedFilter``.
    biases : NDArray[np.float32] | None, optional
        Kadane bias penalties for ``"kadane"``. If ``None``, generated via
        ``generate_kadane_biases``.
    n_corr : int, optional
        Top-K correlated templates for ``"double_matched"``, by default 10.

    Returns
    -------
    NDArray[np.float32]
        1D array of S/N scores with shape ``(nfolds,)``.
    """
    arr = np.ascontiguousarray(folds_ev, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != 2:
        msg = f"folds_ev must have shape (nfolds, 2, nbins), got {arr.shape}"
        raise ValueError(msg)
    widths_i64 = np.ascontiguousarray(widths, dtype=np.int64)

    if mode == "boxcar":
        result = snr_score_batch_func(arr, widths_i64)
    elif mode == "boxcar_glrt":
        result = snr_glrt_score_batch_func(arr, widths_i64)
    elif mode == "harmonic_z2":
        result = harmonic_summing_snr_2d(arr, n_harmonics)
    elif mode == "combined":
        result = combined_score_2d(arr, widths_i64, n_harmonics)
    elif mode in ("double_boxcar", "double_matched", "kadane"):
        ts_e = arr[:, 0, :]
        ts_v = arr[:, 1, :]
        norm = np.divide(
            ts_e,
            np.sqrt(np.maximum(ts_v, 0.0)),
            out=np.zeros_like(ts_e, dtype=np.float32),
            where=ts_v > 0.0,
        )
        norm = np.ascontiguousarray(norm, dtype=np.float32)
        if mode == "double_boxcar":
            result = double_boxcar_snr_2d(norm, widths_i64)
        elif mode == "double_matched":
            if e_mat is None:
                mf = MatchedFilter(widths_i64, arr.shape[2], kind="gaussian")
                e_mat = mf.e_mat
            result = _compute_snr_double(
                norm,
                np.ascontiguousarray(e_mat, dtype=np.float32),
                n_corr,
            )
        else:
            if biases is None:
                biases = generate_kadane_biases(arr.shape[2])
            result = kadane_circular_snr_2d(
                norm,
                np.ascontiguousarray(biases, dtype=np.float32),
            )
    else:
        msg = f"Unsupported score mode: {mode!r}"
        raise ValueError(msg)
    return result


# ---------------------------------------------------------------------------
# 2-State Markov / HMM Forward-DP Scorer for Nulling & Eclipsing Pulsars
# ---------------------------------------------------------------------------


@njit(cache=True)
def _logaddexp(a: float, b: float) -> float:
    """Overflow-safe symmetric log(exp(a) + exp(b))."""
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


@njit(cache=True)
def markov_hmm_log_llr(
    z_arr: np.ndarray,
    amp: float,
    p_on_off: float,
    p_off_on: float,
) -> float:
    """2-state Markov HMM forward-DP log Bayes factor ``ln(p(z | H1) / p(z | H0))``.

    Evaluates standardized per-segment matched-filter scores ``z_arr`` (where
    ``z_i ~ N(0, 1)`` under OFF state ``s_i = 0`` and ``z_i ~ N(amp, 1)`` under
    ON state ``s_i = 1``) in log-likelihood-ratio space using overflow-safe
    ``logaddexp``.
    """
    n = z_arr.size
    if n == 0:
        return 0.0
    p10 = min(max(p_on_off, 1e-12), 1.0 - 1e-12)
    p01 = min(max(p_off_on, 1e-12), 1.0 - 1e-12)
    pi0 = p10 / (p01 + p10)
    pi1 = p01 / (p01 + p10)

    log_t00 = math.log(1.0 - p01)
    log_t01 = math.log(p01)
    log_t10 = math.log(p10)
    log_t11 = math.log(1.0 - p10)

    half_a2 = 0.5 * amp * amp
    a0 = math.log(pi0)
    a1 = math.log(pi1) + (amp * float(z_arr[0]) - half_a2)

    for i in range(1, n):
        llr_i = amp * float(z_arr[i]) - half_a2
        next_a0 = _logaddexp(a0 + log_t00, a1 + log_t10)
        next_a1 = _logaddexp(a0 + log_t01, a1 + log_t11) + llr_i
        a0 = next_a0
        a1 = next_a1
    return _logaddexp(a0, a1)


@njit(cache=True)
def viterbi_null_mask(
    z_arr: np.ndarray,
    amp: float,
    p_on_off: float,
    p_off_on: float,
) -> np.ndarray:
    """Decode the MAP ON/OFF state sequence (``True`` = ON) via Viterbi DP."""
    n = z_arr.size
    mask = np.zeros(n, dtype=np.bool_)
    if n == 0:
        return mask

    p10 = min(max(p_on_off, 1e-12), 1.0 - 1e-12)
    p01 = min(max(p_off_on, 1e-12), 1.0 - 1e-12)
    pi0 = p10 / (p01 + p10)
    pi1 = p01 / (p01 + p10)

    log_t00 = math.log(1.0 - p01)
    log_t01 = math.log(p01)
    log_t10 = math.log(p10)
    log_t11 = math.log(1.0 - p10)
    half_a2 = 0.5 * amp * amp

    backptr = np.zeros((n, 2), dtype=np.int8)
    v0 = math.log(pi0)
    v1 = math.log(pi1) + (amp * float(z_arr[0]) - half_a2)

    for i in range(1, n):
        llr_i = amp * float(z_arr[i]) - half_a2
        c00 = v0 + log_t00
        c10 = v1 + log_t10
        if c00 >= c10:
            next_v0 = c00
            backptr[i, 0] = 0
        else:
            next_v0 = c10
            backptr[i, 0] = 1

        c01 = v0 + log_t01
        c11 = v1 + log_t11
        if c01 >= c11:
            next_v1 = c01 + llr_i
            backptr[i, 1] = 0
        else:
            next_v1 = c11 + llr_i
            backptr[i, 1] = 1

        v0 = next_v0
        v1 = next_v1

    state = 1 if v1 >= v0 else 0
    mask[n - 1] = state == 1
    for i in range(n - 1, 0, -1):
        state = int(backptr[i, state])
        mask[i - 1] = state == 1
    return mask


@njit(cache=True)
def markov_hmm_posterior_on(
    z_arr: np.ndarray,
    amp: float,
    p_on_off: float,
    p_off_on: float,
) -> np.ndarray:
    """Compute forward-backward posterior ON probability ``P(s_i = 1 | z)``."""
    n = z_arr.size
    post = np.zeros(n, dtype=np.float64)
    if n == 0:
        return post

    p10 = min(max(p_on_off, 1e-12), 1.0 - 1e-12)
    p01 = min(max(p_off_on, 1e-12), 1.0 - 1e-12)
    pi0 = p10 / (p01 + p10)
    pi1 = p01 / (p01 + p10)

    log_t00 = math.log(1.0 - p01)
    log_t01 = math.log(p01)
    log_t10 = math.log(p10)
    log_t11 = math.log(1.0 - p10)
    half_a2 = 0.5 * amp * amp

    alpha0 = np.empty(n, dtype=np.float64)
    alpha1 = np.empty(n, dtype=np.float64)
    alpha0[0] = math.log(pi0)
    alpha1[0] = math.log(pi1) + (amp * float(z_arr[0]) - half_a2)
    for i in range(1, n):
        llr_i = amp * float(z_arr[i]) - half_a2
        alpha0[i] = _logaddexp(alpha0[i - 1] + log_t00, alpha1[i - 1] + log_t10)
        alpha1[i] = _logaddexp(alpha0[i - 1] + log_t01, alpha1[i - 1] + log_t11) + llr_i

    log_z_last = _logaddexp(alpha0[n - 1], alpha1[n - 1])
    b0 = 0.0
    b1 = 0.0
    post[n - 1] = math.exp(alpha1[n - 1] - log_z_last)
    for i in range(n - 2, -1, -1):
        llr_next = amp * float(z_arr[i + 1]) - half_a2
        nb0 = _logaddexp(log_t00 + b0, log_t01 + llr_next + b1)
        nb1 = _logaddexp(log_t10 + b0, log_t11 + llr_next + b1)
        b0 = nb0
        b1 = nb1
        log_z_i = _logaddexp(alpha0[i] + b0, alpha1[i] + b1)
        post[i] = math.exp(alpha1[i] + b1 - log_z_i)
    return post


def markov_hmm_grid_score(
    z_arr: npt.ArrayLike,
    *,
    amps: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    p_on_offs: Sequence[float] = (0.02, 0.1, 0.25),
    p_off_ons: Sequence[float] = (0.01, 0.05, 0.2),
) -> tuple[float, float, float, float]:
    """Maximize ``markov_hmm_log_llr`` over a coarse ``(amp, p_on_off, p_off_on)`` grid.

    Returns ``(best_log_llr, best_amp, best_p_on_off, best_p_off_on)``.
    """
    z = np.ascontiguousarray(z_arr, dtype=np.float64)
    best_llr = -np.inf
    best_params = (float(amps[0]), float(p_on_offs[0]), float(p_off_ons[0]))
    for a in amps:
        for p10 in p_on_offs:
            for p01 in p_off_ons:
                val = float(markov_hmm_log_llr(z, float(a), float(p10), float(p01)))
                if val > best_llr:
                    best_llr = val
                    best_params = (float(a), float(p10), float(p01))
    return best_llr, best_params[0], best_params[1], best_params[2]
