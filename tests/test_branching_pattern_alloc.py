"""The branching-pattern generators write every error column they propagate (#43).

They build their error vectors with ``np.empty``. Whether an unwritten column shows up
depends on what is in memory, so this fixes the content: the kernel's Python source runs
with the module's ``np.empty`` replaced by one that fills float arrays with NaN. Any
column the generator reads before writing then poisons the error propagation under the
``quadrature`` and ``conservative`` tilings, and the pattern differs from the control
or the ``ceil`` of a NaN raises. With every column written, the poisoned run returns the
control's pattern exactly, for every generator, grid and tiling.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from numba import typed

from pyloki.core import chebyshev, circular, taylor

NBINS = 64
ARGS = (10.0, 16, NBINS, 1.0, 8)  # tseg_ffa, nsegments, nbins, eta, ref_seg
PARAM_ARR_3 = typed.List([np.array([0.0]), np.array([0.0]), np.array([100.0])])
DPARAMS_3 = np.array([2e-4, 2.0, 0.05])
PARAM_ARR_5 = typed.List([np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([100.0])])
DPARAMS_5 = np.array([1e-9, 1e-6, 2e-4, 2.0, 0.05])

GENERATORS = {
    "taylor": (taylor, taylor.generate_bp_poly_taylor, PARAM_ARR_3, DPARAMS_3, {"use_cheby_coarsening": False}),
    "chebyshev": (chebyshev, chebyshev.generate_bp_poly_chebyshev, PARAM_ARR_3, DPARAMS_3, {}),
    "circular": (circular, circular.generate_bp_circ_taylor, PARAM_ARR_5, DPARAMS_5, {}),
}


def _poisoned_numpy() -> types.ModuleType:
    shim = types.ModuleType("numpy_poisoned")
    shim.__dict__.update(np.__dict__)

    def empty(shape, dtype=float, *args, **kwargs):  # noqa: ANN001, ANN202, ARG001
        dt = np.dtype(dtype)
        if dt.kind == "f":
            return np.full(shape, np.nan, dtype=dt)
        return np.empty(shape, dtype=dt)

    shim.empty = empty
    return shim


@pytest.mark.parametrize("strategy", ["aggressive", "quadrature", "conservative"])
@pytest.mark.parametrize("moving", [True, False])
@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_pattern_does_not_depend_on_unwritten_memory(monkeypatch, name, moving, strategy) -> None:
    mod, kernel, param_arr, dparams, kwargs = GENERATORS[name]
    control = kernel.py_func(param_arr, dparams, *ARGS, use_moving_grid=moving, tiling_strategy=strategy, **kwargs)
    monkeypatch.setattr(mod, "np", _poisoned_numpy())
    poisoned = kernel.py_func(param_arr, dparams, *ARGS, use_moving_grid=moving, tiling_strategy=strategy, **kwargs)
    assert np.isfinite(poisoned).all()
    np.testing.assert_array_equal(poisoned, control)
