from __future__ import annotations
import numpy as np
import ctypes
from numba import njit, vectorize
from numba.extending import get_cython_function_address

addr = get_cython_function_address("scipy.special.cython_special", "binom")
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
cbinom_func = functype(addr)


@vectorize("float64(float64, float64)")
def nbinom(xx, yy):
    return cbinom_func(xx, yy)


def fact_factory(n_tab_out=100):
    fact_tab = np.ones(n_tab_out)

    @njit(cache=True)
    def _fact(num, n_tab=n_tab_out):  # noqa: WPS430
        if num < n_tab:
            return fact_tab[num]
        ret = 1
        for nn in range(1, num + 1):
            ret *= nn
        return ret

    for ii in range(n_tab_out):
        fact_tab[ii] = _fact(ii, 0)

    @vectorize(cache=True)
    def fact_vec(num):  # noqa: WPS430
        return _fact(num)

    return fact_vec


fact = fact_factory(120)
