from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from pyloki.utils import np_utils

if TYPE_CHECKING:
    from collections.abc import Callable


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


@njit(cache=True, fastmath=True)
def shift_add_batch(
    segment_batch: np.ndarray,
    phase_shift_batch: np.ndarray,
    folds: np.ndarray,
    isuggest_batch: np.ndarray,
) -> np.ndarray:
    n_batch, n_comps, n_cols = segment_batch.shape
    res = np.empty((n_batch, n_comps, n_cols), dtype=segment_batch.dtype)
    for irow in range(n_batch):
        shift = phase_shift_batch[irow] % n_cols
        fold_row = folds[isuggest_batch[irow]]
        src_idx = (-shift) % n_cols
        for j in range(n_cols):
            res[irow, 0, j] = fold_row[0, j] + segment_batch[irow, 0, src_idx]
            res[irow, 1, j] = fold_row[1, j] + segment_batch[irow, 1, src_idx]
            src_idx += 1
            if src_idx == n_cols:
                src_idx = 0
    return res


@njit(cache=True, fastmath=True)
def get_trans_matrix(
    coord_cur: tuple[float, float],  # noqa: ARG001
    coord_prev: tuple[float, float],  # noqa: ARG001
) -> np.ndarray:
    return np.eye(2)


@njit(cache=True, fastmath=True)
def get_validation_params() -> tuple[np.ndarray, np.ndarray, float]:
    return np.array([1, 2, 3]), np.array([4, 5, 6]), 0.1
