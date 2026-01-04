from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit, types

from pyloki.utils import np_utils

if TYPE_CHECKING:
    from collections.abc import Callable


@njit(cache=True, fastmath=True)
def get_leaves(param_arr: types.ListType, dparams: np.ndarray) -> np.ndarray:
    """Get the leaf parameter sets for pruning.

    Parameters
    ----------
    param_arr : types.ListType
        Parameter array containing the parameter values for each dimension.
    dparams : np.ndarray
        Parameter step sizes for each dimension in a 1D array.

    Returns
    -------
    np.ndarray
        Array of leaf parameter sets.
    """
    param_cart = np_utils.cartesian_prod(param_arr)
    param_mat = np.expand_dims(param_cart, axis=2)
    dparams_set = np.broadcast_to(np.expand_dims(dparams, 1), param_mat.shape)
    return np.concatenate((param_mat, dparams_set), axis=2)


@njit(cache=True, fastmath=True)
def get_leaves_opt(
    param_arr: types.ListType,
    dparams: np.ndarray,
) -> np.ndarray:
    nparams = len(param_arr) # ty: ignore
    shapes = np.empty(nparams, dtype=np.int64)
    for i in range(nparams):
        shapes[i] = len(param_arr[i])
    total_size = np.prod(shapes)
    leaves_taylor = np.empty((total_size, nparams, 2), dtype=np.float64)

    if total_size == 0:
        return leaves_taylor  # Return empty array if no leaves

    # Fill column 1 (dparams) - this is constant for each parameter across leaves
    for j in range(nparams):
        leaves_taylor[:, j, 1] = dparams[j]

    # Fill column 0 (parameter values) using Cartesian product logic
    # Similar logic to the optimized cartesian_prod, but fills directly
    elements_per_cycle = np.empty(nparams, dtype=np.int64)
    elements_in_block = total_size

    for i in range(nparams - 1, -1, -1):
        elements_in_block //= shapes[i]
        elements_per_cycle[i] = elements_in_block

    for i in range(total_size):
        for j in range(nparams):
            arr = param_arr[j]
            # Calculate index within the specific parameter's array
            idx = (i // elements_per_cycle[j]) % shapes[j]
            leaves_taylor[i, j, 0] = arr[idx]

    return leaves_taylor


@njit(cache=True, fastmath=True)
def load_folds_1d(fold: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """Load the fold from the input structure (1D-case).

    Parameters
    ----------
    fold : np.ndarray
        Input fold structure with shape (nsegments, nfreqs, 2, nbins).
    iseg : int
        Index of the segment.
    param_idx : np.ndarray
        Index of the parameter with shape [ifreq].

    Returns
    -------
    np.ndarray
        Fold with shape (2, nbins).
    """
    return fold[iseg, param_idx[-1]]


@njit(cache=True, fastmath=True)
def load_folds_2d(fold: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """Load the fold from the input structure (2D-case).

    Parameters
    ----------
    fold : np.ndarray
        Input fold structure with shape (nsegments, naccels, nfreqs, 2, nbins).
    iseg : int
        Index of the segment.
    param_idx : np.ndarray
        Index of the parameter with shape [iacc, ifreq].

    Returns
    -------
    np.ndarray
        Fold with shape (2, nbins).
    """
    return fold[iseg, param_idx[-2], param_idx[-1]]


@njit(cache=True, fastmath=True)
def load_folds_3d(fold: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (nsegments, njerks, naccels, nfreqs, 2, nbins)."""
    return fold[iseg, param_idx[-3], param_idx[-2], param_idx[-1]]


@njit(cache=True, fastmath=True)
def load_folds_4d(fold: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (nsegments, nsnap, njerks, naccels, nfreqs, 2, nbins)."""
    return fold[iseg, param_idx[-4], param_idx[-3], param_idx[-2], param_idx[-1]]


@njit(cache=True, fastmath=True)
def load_folds_5d(fold: np.ndarray, iseg: int, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (nsnap, ncrackles, nsnap, njerks, naccels, nfreqs, 2, nbins)."""
    return fold[
        iseg,
        param_idx[-5],
        param_idx[-4],
        param_idx[-3],
        param_idx[-2],
        param_idx[-1],
    ]


@njit(cache=True, fastmath=True)
def load_prune_folds_1d(fold: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (nfreqs, 2, nbins)."""
    # Single slice case: param_idx is 1D
    if param_idx.ndim == 1:
        return fold[param_idx[-1]]

    # Batched case: param_idx is 2D
    nbins = fold.shape[-1]
    batch_size = param_idx.shape[0]
    result = np.empty((batch_size, 2, nbins), dtype=fold.dtype)
    for i in range(batch_size):
        freq_idx = param_idx[i, -1]
        for j in range(2):
            for k in range(nbins):
                result[i, j, k] = fold[freq_idx, j, k]
    return result


@njit(cache=True, fastmath=True)
def load_prune_folds_2d(fold: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (naccels, nfreqs, 2, nbins)."""
    # Single slice case: param_idx is 1D
    if param_idx.ndim == 1:
        return fold[param_idx[-2], param_idx[-1]]

    # Batched case: param_idx is 2D
    nbins = fold.shape[-1]
    batch_size = param_idx.shape[0]
    result = np.empty((batch_size, 2, nbins), dtype=fold.dtype)
    for i in range(batch_size):
        accel_idx = param_idx[i, -2]
        freq_idx = param_idx[i, -1]
        for j in range(2):
            for k in range(nbins):
                result[i, j, k] = fold[accel_idx, freq_idx, j, k]

    return result


@njit(cache=True, fastmath=True)
def load_prune_folds_3d(fold: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (njerks, naccels, nfreqs, 2, nbins)."""
    # Single slice case: param_idx is 1D
    if param_idx.ndim == 1:
        return fold[0, param_idx[-2], param_idx[-1]]

    # Batched case: param_idx is 2D
    nbins = fold.shape[-1]
    batch_size = param_idx.shape[0]
    result = np.empty((batch_size, 2, nbins), dtype=fold.dtype)
    for i in range(batch_size):
        accel_idx = param_idx[i, -2]
        freq_idx = param_idx[i, -1]
        for j in range(2):
            for k in range(nbins):
                result[i, j, k] = fold[0, accel_idx, freq_idx, j, k]
    return result


@njit(cache=True, fastmath=True)
def load_prune_folds_4d(fold: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (nsnap, njerks, naccels, nfreqs, 2, nbins)."""
    # Single slice case: param_idx is 1D
    if param_idx.ndim == 1:
        return fold[0, 0, param_idx[-2], param_idx[-1]]

    # Batched case: param_idx is 2D
    nbins = fold.shape[-1]
    batch_size = param_idx.shape[0]
    result = np.empty((batch_size, 2, nbins), dtype=fold.dtype)
    for i in range(batch_size):
        accel_idx = param_idx[i, -2]
        freq_idx = param_idx[i, -1]
        for j in range(2):
            for k in range(nbins):
                result[i, j, k] = fold[0, 0, accel_idx, freq_idx, j, k]
    return result


@njit(cache=True, fastmath=True)
def load_prune_folds_5d(fold: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (nsnap, ncrackles, nsnap, njerks, naccels, nfreqs, 2, nbins)."""
    # Single slice case: param_idx is 1D
    if param_idx.ndim == 1:
        return fold[0, 0, 0, param_idx[-2], param_idx[-1]]

    # Batched case: param_idx is 2D
    nbins = fold.shape[-1]
    batch_size = param_idx.shape[0]
    result = np.empty((batch_size, 2, nbins), dtype=fold.dtype)
    for i in range(batch_size):
        accel_idx = param_idx[i, -2]
        freq_idx = param_idx[i, -1]
        for j in range(2):
            for k in range(nbins):
                result[i, j, k] = fold[0, 0, 0, accel_idx, freq_idx, j, k]
    return result


def set_ffa_load_func(
    nparams: int,
) -> Callable[[np.ndarray, int, np.ndarray], np.ndarray]:
    """Set the appropriate load function based on the number of parameters.

    Parameters
    ----------
    nparams : int
        Number of search parameters (dimensions).

    Returns
    -------
    Callable[[np.ndarray, int, np.ndarray], np.ndarray]
        The appropriate load function for the given number of parameters.
    """
    nparams_to_load_func = {
        1: load_folds_1d,
        2: load_folds_2d,
        3: load_folds_3d,
        4: load_folds_4d,
        5: load_folds_5d,
    }
    return nparams_to_load_func[nparams]


def set_prune_load_func(
    nparams: int,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Set the appropriate load function for the pruning based on the number of parameters.

    Parameters
    ----------
    nparams : int
        Number of search parameters (dimensions).

    Returns
    -------
    Callable[[np.ndarray, np.ndarray], np.ndarray]
        The appropriate load function for the given number of parameters.
    """
    nparams_to_load_func = {
        1: load_prune_folds_1d,
        2: load_prune_folds_2d,
        3: load_prune_folds_3d,
        4: load_prune_folds_4d,
        5: load_prune_folds_5d,
    }
    return nparams_to_load_func[nparams]


@njit(cache=True, fastmath=True)
def add(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
    return data0 + data1


@njit(cache=True, fastmath=True)
def pack(data: np.ndarray) -> np.ndarray:
    return data


@njit(cache=True, fastmath=True)
def shift(data: np.ndarray, phase_shift: int) -> np.ndarray:
    return np_utils.nb_roll(data, phase_shift, axis=-1)


@njit(["f4[:,::1](f4[:,::1],f4[:,::1],f8,f8)"], cache=True, fastmath=True)
def shift_add(
    data_tail: np.ndarray,
    data_head: np.ndarray,
    phase_shift_tail: float,
    phase_shift_head: float,
) -> np.ndarray:
    n_comps, nbins = data_tail.shape
    res = np.empty((n_comps, nbins), dtype=data_tail.dtype)
    phase_shift_tail_float = np.float32(phase_shift_tail)
    phase_shift_head_float = np.float32(phase_shift_head)
    shift_tail = round(phase_shift_tail_float) % nbins
    shift_head = round(phase_shift_head_float) % nbins
    for j in range(nbins):
        idx1 = (j - shift_tail) % nbins
        idx2 = (j - shift_head) % nbins
        res[0, j] = data_tail[0, idx1] + data_head[0, idx2]
        res[1, j] = data_tail[1, idx1] + data_head[1, idx2]
    return res


@njit(cache=True, fastmath=True)
def shift_add_complex_direct(
    data_tail: np.ndarray,
    data_head: np.ndarray,
    phase_shift_tail: float,
    phase_shift_head: float,
) -> np.ndarray:
    _, nbins_f = data_tail.shape
    nbins = (nbins_f - 1) * 2
    phase_shift_tail_float = np.float32(phase_shift_tail)
    phase_shift_head_float = np.float32(phase_shift_head)
    k = np.arange(nbins_f)
    phase1 = np.exp(-2j * np.pi * k * phase_shift_tail_float / nbins)
    phase2 = np.exp(-2j * np.pi * k * phase_shift_head_float / nbins)
    return (data_tail * phase1) + (data_head * phase2)


@njit(["c8[:,::1](c8[:,::1], c8[:,::1], f8, f8)"], cache=True, fastmath=True)
def shift_add_complex(
    data_tail: np.ndarray,
    data_head: np.ndarray,
    phase_shift_tail: float,
    phase_shift_head: float,
) -> np.ndarray:
    n_comps, nbins_f = data_tail.shape
    res = np.empty((n_comps, nbins_f), dtype=data_tail.dtype)
    nbins = (nbins_f - 1) * 2

    # Precompute the angular steps
    phase_shift_tail_float = np.float32(phase_shift_tail)
    phase_shift_head_float = np.float32(phase_shift_head)
    step_tail = np.float32(-2.0 * np.pi * phase_shift_tail_float / nbins)
    step_head = np.float32(-2.0 * np.pi * phase_shift_head_float / nbins)

    # Compute the complex delta (rotation factors)
    delta_tail = np.complex64(np.cos(step_tail) + 1j * np.sin(step_tail))
    delta_head = np.complex64(np.cos(step_head) + 1j * np.sin(step_head))
    phase_tail = np.complex64(1.0 + 0.0j)
    phase_head = np.complex64(1.0 + 0.0j)

    for k in range(nbins_f):
        res[0, k] = data_tail[0, k] * phase_tail + data_head[0, k] * phase_head
        res[1, k] = data_tail[1, k] * phase_tail + data_head[1, k] * phase_head
        phase_tail *= delta_tail
        phase_head *= delta_head
    return res


@njit(
    ["f4[:,:,::1](f4[:,:,::1],f8[::1],f4[:,:,::1], i8[::1])"],
    cache=True,
    fastmath=True,
)
def shift_add_batch(
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    n_batch, n_comps, nbins = segment_batch.shape
    res = np.empty((n_batch, n_comps, nbins), dtype=segment_batch.dtype)
    for irow in range(n_batch):
        shift_float = np.float32(shift_batch[irow])
        shift = round(shift_float) % nbins
        fold_row = folds[isuggest_batch[irow]]
        src_idx = (-shift) % nbins
        for j in range(nbins):
            res[irow, 0, j] = fold_row[0, j] + segment_batch[irow, 0, src_idx]
            res[irow, 1, j] = fold_row[1, j] + segment_batch[irow, 1, src_idx]
            src_idx += 1
            if src_idx == nbins:
                src_idx = 0
    return res


@njit(
    ["c8[:,:,::1](c8[:,:,::1],f8[::1],c8[:,:,::1], i8[::1])"],
    cache=True,
    fastmath=True,
)
def shift_add_complex_batch(
    segment_batch: np.ndarray,
    shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    n_batch, n_comps, nbins_f = segment_batch.shape
    res = np.empty((n_batch, n_comps, nbins_f), dtype=segment_batch.dtype)
    nbins = (nbins_f - 1) * 2
    for irow in range(n_batch):
        shift_float = np.float32(shift_batch[irow])
        # Precompute phase step and delta
        angle = np.float32(-2.0 * np.float32(np.pi) * shift_float / nbins)
        delta = np.complex64(np.cos(angle) + 1j * np.sin(angle))
        phase = np.complex64(1.0 + 0.0j)
        fold = folds[isuggest_batch[irow]]
        for k in range(nbins_f):
            res[irow, 0, k] = segment_batch[irow, 0, k] * phase + fold[0, k]
            res[irow, 1, k] = segment_batch[irow, 1, k] * phase + fold[1, k]
            phase *= delta

    return res


@njit(cache=True, fastmath=True)
def get_trans_matrix(
    coord_next: tuple[float, float],  # noqa: ARG001
    coord_prev: tuple[float, float],  # noqa: ARG001
) -> np.ndarray:
    return np.eye(2)


@njit(cache=True, fastmath=True)
def get_validation_params(
    coord_add: tuple[float, float],  # noqa: ARG001
) -> tuple[np.ndarray, np.ndarray, float]:
    return np.array([1, 2, 3]), np.array([4, 5, 6]), 0.1


@njit(["f4[:,:,::1](f4[:,:,::1],f8[::1])"], cache=True, fastmath=True)
def shift_3d_batch(leaves_batch: np.ndarray, shift_batch: np.ndarray) -> np.ndarray:
    n_batch, n_comps, nbins = leaves_batch.shape
    res = np.empty((n_batch, n_comps, nbins), dtype=leaves_batch.dtype)
    for irow in range(n_batch):
        shift_float = np.float32(shift_batch[irow])
        shift = round(shift_float) % nbins
        src_idx = (-shift) % nbins
        for j in range(nbins):
            res[irow, 0, j] = leaves_batch[irow, 0, src_idx]
            res[irow, 1, j] = leaves_batch[irow, 1, src_idx]
            src_idx += 1
            if src_idx == nbins:
                src_idx = 0
    return res


@njit(["c8[:,:,::1](c8[:,:,::1],f8[::1])"], cache=True, fastmath=True)
def shift_3d_complex_batch(
    leaves_batch: np.ndarray,
    shift_batch: np.ndarray,
) -> np.ndarray:
    n_batch, n_comps, nbins_f = leaves_batch.shape
    res = np.empty((n_batch, n_comps, nbins_f), dtype=leaves_batch.dtype)
    nbins = (nbins_f - 1) * 2
    for irow in range(n_batch):
        shift_float = np.float32(shift_batch[irow])
        # Precompute phase step and delta
        angle = np.float32(-2.0 * np.float32(np.pi) * shift_float / nbins)
        delta = np.complex64(np.cos(angle) + 1j * np.sin(angle))
        phase = np.complex64(1.0 + 0.0j)
        for k in range(nbins_f):
            res[irow, 0, k] = leaves_batch[irow, 0, k] * phase
            res[irow, 1, k] = leaves_batch[irow, 1, k] * phase
            phase *= delta

    return res
