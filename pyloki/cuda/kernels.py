import cupy as cp
import numpy as np
from numba import cuda


@cuda.jit(device=True, fastmath=True)
def get_phase_idx_device(
    proper_time: float,
    freq: float,
    nbins: int,
    delay: float,
) -> int:
    phase = ((proper_time + delay) * freq) % 1.0
    iphase = int(phase * nbins + 0.5)
    if iphase == nbins:
        return 0
    return iphase


@cuda.jit(fastmath=True)
def brutefold_kernel(
    ts_e: cp.ndarray,
    ts_v: cp.ndarray,
    freq_arr: cp.ndarray,
    fold: cp.ndarray,
    segment_len: int,
    tsamp: float,
    t_ref: float,
) -> None:
    iseg = cuda.blockIdx.x
    ifreq = cuda.blockIdx.y
    nsamples = ts_e.shape[0]
    nsegments, nfreqs, _, nbins = fold.shape
    if ifreq >= nfreqs:
        return

    # Threads in this block will process samples for the current segment (iseg)
    # Each thread starts at threadIdx.x and strides by blockDim.x
    start_isamp_in_segment = cuda.threadIdx.x
    stride_isamp_in_segment = cuda.blockDim.x

    for isamp_in_segment in range(
        start_isamp_in_segment,
        segment_len,
        stride_isamp_in_segment,
    ):
        isamp = iseg * segment_len + isamp_in_segment
        if isamp < nsamples:
            proper_time = isamp_in_segment * tsamp - t_ref
            iphase = get_phase_idx_device(
                proper_time,
                freq_arr[ifreq],
                nbins,
                0.0,
            )
            cuda.atomic.add(fold, (iseg, ifreq, 0, iphase), ts_e[isamp])
            cuda.atomic.add(fold, (iseg, ifreq, 1, iphase), ts_v[isamp])


def brutefold_start_cuda(
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    freq_arr: np.ndarray,
    segment_len: int,
    nbins: int,
    tsamp: float,
    t_ref: float = 0.0,
    *,
    return_on_host: bool = True,
) -> np.ndarray | cp.ndarray:
    """Fold a time series using CUDA for a given set of frequencies.

    Parameters
    ----------
    ts_e : np.ndarray
        Time series signal (intensity).
    ts_v : np.ndarray
        Time series variance.
    freq_arr : np.ndarray
        Array of frequencies to fold (Hz).
    segment_len : int
        Length of the segment (in samples) to fold.
    nbins : int
        Number of bins in the folded profile.
    tsamp : float
        Sampling time of the time series (s).
    t_ref : float, optional
        Reference time relative to segment start (s), (default: 0.0).
    return_on_host : bool, optional
        If True (default), returns result as NumPy array on host.
        If False, returns result as CuPy array on GPU.

    Returns
    -------
    np.ndarray | cp.ndarray
        Folded time series with shape (nsegments, nfreqs, 2, nbins).
        Type depends on `return_on_host`.
    """
    nfreqs = len(freq_arr)
    nsamples = len(ts_e)
    nsegments = int(np.ceil(nsamples / segment_len))

    ts_e_gpu = cp.asarray(ts_e)
    ts_v_gpu = cp.asarray(ts_v)
    freq_arr_gpu = cp.asarray(freq_arr)
    fold_gpu = cp.zeros((nsegments, nfreqs, 2, nbins), dtype=cp.float32)

    # --- Kernel Launch Configuration ---
    # Grid dimensions: (number of segments, number of frequencies)
    blocks_per_grid = (nsegments, nfreqs)
    threads_per_block_config = (256,)

    # Launch kernel
    brutefold_kernel[blocks_per_grid, threads_per_block_config](
        ts_e_gpu,
        ts_v_gpu,
        freq_arr_gpu,
        fold_gpu,
        segment_len,
        tsamp,
        t_ref,
    )
    if return_on_host:
        cuda.synchronize()
        return fold_gpu.get()
    return fold_gpu
