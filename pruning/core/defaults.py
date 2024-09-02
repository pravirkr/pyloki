import numpy as np
from numba import njit

from pruning.utils import math


@njit(cache=True)
def add(data0: np.ndarray, data1: np.ndarray) -> np.ndarray:
    return data0 + data1


@njit(cache=True)
def pack(data: np.ndarray) -> np.ndarray:
    return data


@njit(cache=True)
def shift(data: np.ndarray, phase_shift: int) -> np.ndarray:
    return math.nb_roll2d(data, phase_shift)


@njit(cache=True)
def get_trans_matrix(
    coord_cur: tuple[float, float],  # noqa: ARG001
    coord_prev: tuple[float, float],  # noqa: ARG001
) -> np.ndarray:
    return np.eye(2)
