# ruff: noqa: ARG001

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np
from numba import njit, prange, types
from numba.experimental import structref
from numba.extending import overload_method

from pyloki.core import common, taylor
from pyloki.utils.timing import Timer

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyloki.config import PulsarSearchConfig


@structref.register
class FFASearchDPFunctsTemplate(types.StructRef):
    pass


class FFASearchDPFuncts(structref.StructRefProxy):
    """A container class for the functions used in the FFA search.

    Parameters
    ----------
    cfg : PulsarSearchConfig
        Configuration object for the search.
    """

    def __new__(cls, cfg: PulsarSearchConfig) -> Self:
        """Create a new instance of FFASearchDPFuncts."""
        return ffa_search_dp_functs_init(
            cfg.tsamp,
            cfg.nbins,
            cfg.bseg_brute,
            cfg.use_fft_shifts,
        )

    def init(
        self,
        ts_e: np.ndarray,
        ts_v: np.ndarray,
        param_arr: types.ListType[types.Array],
    ) -> np.ndarray:
        """Receives the data and parameter array and returns the initial fold."""
        return init_func(self, ts_e, ts_v, param_arr)

    def resolve(
        self,
        pset_cur: np.ndarray,
        parr_prev: np.ndarray,
        ffa_level: int,
        latter: int,
    ) -> tuple[np.ndarray, int]:
        """Resolve the current parameters among the previous level parameters."""
        return resolve_func(
            self,
            pset_cur,
            parr_prev,
            ffa_level,
            latter,
        )

    def resolve_fft(
        self,
        pset_cur: np.ndarray,
        parr_prev: np.ndarray,
        ffa_level: int,
        latter: int,
    ) -> tuple[np.ndarray, float]:
        return resolve_fft_func(
            self,
            pset_cur,
            parr_prev,
            ffa_level,
            latter,
        )

    def pack(self, data: np.ndarray, ffa_level: int) -> np.ndarray:
        """Bit packing rule for the FFA search."""
        return pack_func(self, data, ffa_level)

    def shift_add(
        self,
        data_tail: np.ndarray,
        data_head: np.ndarray,
        phase_shift_tail: int,
        phase_shift_head: int,
    ) -> np.ndarray:
        """Add two data segments after shifting them by a phase shift.

        Can only handle integer phase shifts.
        """
        return shift_add_func(
            self,
            data_tail,
            data_head,
            phase_shift_tail,
            phase_shift_head,
        )

    def shift_add_fft(
        self,
        data_tail: np.ndarray,
        data_head: np.ndarray,
        phase_shift_tail: float,
        phase_shift_head: float,
    ) -> np.ndarray:
        """Add two data segments after shifting them by a phase shift.

        Can handle non-integer phase shifts. Input data arrays are assumed to be FFTed.
        """
        return shift_add_fft_func(
            self,
            data_tail,
            data_head,
            phase_shift_tail,
            phase_shift_head,
        )


fields_ffa_search_dp_funcs = [
    ("tsamp", types.float64),
    ("nbins", types.int64),
    ("bseg_brute", types.int64),
    ("use_fft_shifts", types.bool_),
]

structref.define_boxing(FFASearchDPFunctsTemplate, FFASearchDPFuncts)
FFASearchDPFunctsType = FFASearchDPFunctsTemplate(fields_ffa_search_dp_funcs)


@njit(cache=True, fastmath=True)
def ffa_search_dp_functs_init(
    tsamp: float,
    nbins: int,
    bseg_brute: int,
    use_fft_shifts: bool,
) -> FFASearchDPFuncts:
    self = structref.new(FFASearchDPFunctsType)
    self.tsamp = tsamp
    self.nbins = nbins
    self.bseg_brute = bseg_brute
    self.use_fft_shifts = use_fft_shifts
    return self


@njit(cache=True, fastmath=True)
def init_func(
    self: FFASearchDPFuncts,
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    param_arr: types.ListType[types.Array],
) -> np.ndarray:
    return taylor.ffa_taylor_init(
        ts_e,
        ts_v,
        param_arr,
        self.bseg_brute,
        self.nbins,
        self.tsamp,
    )


@njit(cache=True, fastmath=True)
def resolve_func(
    self: FFASearchDPFuncts,
    pset_cur: np.ndarray,
    parr_prev: np.ndarray,
    ffa_level: int,
    latter: int,
) -> tuple[np.ndarray, int]:
    tseg_brute = self.bseg_brute * self.tsamp
    return taylor.ffa_taylor_resolve(
        pset_cur,
        parr_prev,
        ffa_level,
        latter,
        tseg_brute,
        self.nbins,
    )


@njit(cache=True, fastmath=True)
def resolve_fft_func(
    self: FFASearchDPFuncts,
    pset_cur: np.ndarray,
    parr_prev: np.ndarray,
    ffa_level: int,
    latter: int,
) -> tuple[np.ndarray, float]:
    tseg_brute = self.bseg_brute * self.tsamp
    return taylor.ffa_taylor_resolve_fft(
        pset_cur,
        parr_prev,
        ffa_level,
        latter,
        tseg_brute,
        self.nbins,
    )


@njit(cache=True, fastmath=True)
def pack_func(self: FFASearchDPFuncts, data: np.ndarray, ffa_level: int) -> np.ndarray:
    return common.pack(data)


@njit(cache=True, fastmath=True)
def shift_add_func(
    self: FFASearchDPFuncts,
    data_tail: np.ndarray,
    data_head: np.ndarray,
    phase_shift_tail: int,
    phase_shift_head: int,
) -> np.ndarray:
    return common.shift_add(
        data_tail,
        data_head,
        phase_shift_tail,
        phase_shift_head,
    )


@njit(cache=True, fastmath=True)
def shift_add_fft_func(
    self: FFASearchDPFuncts,
    data_tail: np.ndarray,
    data_head: np.ndarray,
    phase_shift_tail: float,
    phase_shift_head: float,
) -> np.ndarray:
    nbins_f = data_tail.shape[-1]
    k = np.arange(nbins_f)
    fold_bins = self.nbins
    phase1 = np.exp(-2j * np.pi * k * phase_shift_tail / fold_bins)
    phase2 = np.exp(-2j * np.pi * k * phase_shift_head / fold_bins)
    return (data_tail * phase1) + (data_head * phase2)


@overload_method(FFASearchDPFunctsTemplate, "init")
def ol_init_func(
    self: FFASearchDPFuncts,
    ts_e: np.ndarray,
    ts_v: np.ndarray,
    param_arr: types.ListType[types.Array],
) -> types.FunctionType:
    def impl(
        self: FFASearchDPFuncts,
        ts_e: np.ndarray,
        ts_v: np.ndarray,
        param_arr: types.ListType[types.Array],
    ) -> np.ndarray:
        return init_func(self, ts_e, ts_v, param_arr)

    return impl


@overload_method(FFASearchDPFunctsTemplate, "resolve")
def ol_resolve(
    self: FFASearchDPFuncts,
    pset_cur: np.ndarray,
    parr_prev: np.ndarray,
    ffa_level: int,
    latter: int,
) -> types.FunctionType:
    def impl(
        self: FFASearchDPFuncts,
        pset_cur: np.ndarray,
        parr_prev: np.ndarray,
        ffa_level: int,
        latter: int,
    ) -> tuple[np.ndarray, int]:
        return resolve_func(self, pset_cur, parr_prev, ffa_level, latter)

    return impl


@overload_method(FFASearchDPFunctsTemplate, "resolve_fft")
def ol_resolve_fft(
    self: FFASearchDPFuncts,
    pset_cur: np.ndarray,
    parr_prev: np.ndarray,
    ffa_level: int,
    latter: int,
) -> types.FunctionType:
    def impl(
        self: FFASearchDPFuncts,
        pset_cur: np.ndarray,
        parr_prev: np.ndarray,
        ffa_level: int,
        latter: int,
    ) -> tuple[np.ndarray, float]:
        return resolve_fft_func(self, pset_cur, parr_prev, ffa_level, latter)

    return impl


@overload_method(FFASearchDPFunctsTemplate, "pack")
def ol_pack_func(
    self: FFASearchDPFuncts,
    data: np.ndarray,
    ffa_level: int,
) -> types.FunctionType:
    def impl(
        self: FFASearchDPFuncts,
        data: np.ndarray,
        ffa_level: int,
    ) -> np.ndarray:
        return pack_func(self, data, ffa_level)

    return impl


@overload_method(FFASearchDPFunctsTemplate, "shift_add")
def ol_shift_add_func(
    self: FFASearchDPFuncts,
    data_tail: np.ndarray,
    data_head: np.ndarray,
    phase_shift_tail: int,
    phase_shift_head: int,
) -> types.FunctionType:
    def impl(
        self: FFASearchDPFuncts,
        data_tail: np.ndarray,
        data_head: np.ndarray,
        phase_shift_tail: int,
        phase_shift_head: int,
    ) -> np.ndarray:
        return shift_add_func(
            self,
            data_tail,
            data_head,
            phase_shift_tail,
            phase_shift_head,
        )

    return impl


@overload_method(FFASearchDPFunctsTemplate, "shift_add_fft")
def ol_shift_add_fft_func(
    self: FFASearchDPFuncts,
    data_tail: np.ndarray,
    data_head: np.ndarray,
    phase_shift_tail: float,
    phase_shift_head: float,
) -> types.FunctionType:
    def impl(
        self: FFASearchDPFuncts,
        data_tail: np.ndarray,
        data_head: np.ndarray,
        phase_shift_tail: float,
        phase_shift_head: float,
    ) -> np.ndarray:
        return shift_add_fft_func(
            self,
            data_tail,
            data_head,
            phase_shift_tail,
            phase_shift_head,
        )

    return impl


@Timer(name="unify_fold")
@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def unify_fold(
    fold_in: np.ndarray,
    param_arr_prev: types.ListType[types.Array],
    fold_out: np.ndarray,
    param_cart_cur: np.ndarray,
    ffa_level: int,
    dp_funcs: FFASearchDPFuncts,
    load_func: Callable[[np.ndarray, int, np.ndarray], np.ndarray],
) -> None:
    """Unify the fold by combining the two folds from the previous level.

    Parameters
    ----------
    fold_in : np.ndarray
        Input fold structure from the previous level.
    param_arr_prev : types.ListType[types.Array]
        Parameter array from the previous level.
    fold_out : np.ndarray
        Output fold structure for the current level.
    param_cart_cur : np.ndarray
        Cartesian product of the parameter array for the current level.
    ffa_level : int
        Current level of the FFA search.
    dp_funcs : FFASearchDPFuncts
        A container for the dynamic programming functions.
    load_func : Callable[[np.ndarray, int, np.ndarray], np.ndarray]
        A function to load the fold from the input structure.
    """
    for iparam_set in prange(len(param_cart_cur)):
        p_set = param_cart_cur[iparam_set]

        # Resolve parameters for tail and head
        p_idx_tail, phase_shift_tail = dp_funcs.resolve(
            p_set,
            param_arr_prev,
            ffa_level,
            0,
        )
        p_idx_head, phase_shift_head = dp_funcs.resolve(
            p_set,
            param_arr_prev,
            ffa_level,
            1,
        )
        for ipair in range(fold_out.shape[0]):
            fold_tail = load_func(fold_in, ipair * 2, p_idx_tail)
            fold_head = load_func(fold_in, ipair * 2 + 1, p_idx_head)
            fold_out[ipair, iparam_set] = dp_funcs.shift_add(
                fold_tail,
                fold_head,
                phase_shift_tail,
                phase_shift_head,
            )


@Timer(name="unify_fold")
@njit(cache=True, fastmath=True, parallel=True, nogil=True)
def unify_fold_fft(
    fold_in: np.ndarray,
    param_arr_prev: types.ListType[types.Array],
    fold_out: np.ndarray,
    param_cart_cur: np.ndarray,
    ffa_level: int,
    dp_funcs: FFASearchDPFuncts,
    load_func: Callable[[np.ndarray, int, np.ndarray], np.ndarray],
) -> None:
    for iparam_set in prange(len(param_cart_cur)):
        p_set = param_cart_cur[iparam_set]

        # Resolve parameters for tail and head
        p_idx_tail, phase_shift_tail = dp_funcs.resolve_fft(
            p_set,
            param_arr_prev,
            ffa_level,
            0,
        )
        p_idx_head, phase_shift_head = dp_funcs.resolve_fft(
            p_set,
            param_arr_prev,
            ffa_level,
            1,
        )
        for ipair in range(fold_out.shape[0]):
            fold_tail = load_func(fold_in, ipair * 2, p_idx_tail)
            fold_head = load_func(fold_in, ipair * 2 + 1, p_idx_head)
            fold_out[ipair, iparam_set] = dp_funcs.shift_add_fft(
                fold_tail,
                fold_head,
                phase_shift_tail,
                phase_shift_head,
            )
