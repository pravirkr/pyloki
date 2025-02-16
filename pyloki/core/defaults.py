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
    return fold[param_idx[-1]]


@njit(cache=True, fastmath=True)
def load_prune_folds_2d(fold: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (naccels, nfreqs, 2, nbins)."""
    return fold[param_idx[-2], param_idx[-1]]


@njit(cache=True, fastmath=True)
def load_prune_folds_3d(fold: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (njerks, naccels, nfreqs, 2, nbins)."""
    return fold[0, param_idx[-2], param_idx[-1]]


@njit(cache=True, fastmath=True)
def load_prune_folds_4d(fold: np.ndarray, param_idx: np.ndarray) -> np.ndarray:
    """Fold shape: (nsnap, njerks, naccels, nfreqs, 2, nbins)."""
    return fold[0, 0, param_idx[-2], param_idx[-1]]


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
def get_trans_matrix(
    coord_cur: tuple[float, float],  # noqa: ARG001
    coord_prev: tuple[float, float],  # noqa: ARG001
) -> np.ndarray:
    return np.eye(2)


@njit(cache=True, fastmath=True)
def get_validation_params() -> tuple[np.ndarray, np.ndarray, float]:
    return np.array([1, 2, 3]), np.array([4, 5, 6]), 0.1
