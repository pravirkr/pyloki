"""Kernels that build their output with ``np.empty`` write every slot of it.

An unwritten slot in an ``np.empty`` output holds whatever was in memory, so a test that
compares two calls sees it only when the memory happens to differ (#41 was found that way:
on #40's CI all three Pythons failed, each with a different value). This module fixes the
content: each kernel's Python source runs with its module's ``np.empty`` replaced by one
that fills float arrays with NaN, signed integers with the dtype's minimum and unsigned
with its maximum, and the result must equal a control run with the real ``numpy``. Both
runs do the same arithmetic on the same inputs and differ only in unwritten memory, so a
failure means an unwritten slot, not a numerical difference. Only allocations in the
kernel's own source are poisoned: its callees run compiled, with the real ``numpy``, so a
case speaks for that kernel's slots and nothing else. The same check found #43.

Covered here: the kernels that assemble their output field by field (the shape of #41 and
#43), the circular resolvers (their rows are filled per leaf class), and the FFA plan.
``circ_taylor_branch_batch`` and the branching-pattern generators carry this check in
their own fix PRs. ``pruning_iteration_batched`` needs a live pruning state and is not
run; by reading, it writes every column of its width-(n+2) backtrack row (0, 1..n and n+1).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from numba import typed

from pyloki import ffa, kepler
from pyloki.config import ParamLimits, PulsarSearchConfig
from pyloki.core import circular, common, fold
from pyloki.simulation import modulate
from pyloki.utils import np_utils, psr_utils, transforms

# ----------------------------------------------------------------- the check


def poisoned_numpy() -> types.ModuleType:
    """A stand-in for ``numpy`` whose ``empty``/``empty_like`` fill with sentinels."""
    shim = types.ModuleType("numpy_poisoned")
    shim.__dict__.update(np.__dict__)

    def fill_for(dt: np.dtype):
        if dt.kind in "fc":
            return np.nan
        if dt.kind == "i":
            return np.iinfo(dt).min
        if dt.kind == "u":
            return np.iinfo(dt).max
        return None

    def empty(shape, dtype=float, order="C", **kwargs):  # noqa: ANN001, ANN202, ARG001
        dt = np.dtype(dtype)
        fill = fill_for(dt)
        return np.empty(shape, dtype=dt) if fill is None else np.full(shape, fill, dtype=dt)

    def empty_like(prototype, dtype=None, **kwargs):  # noqa: ANN001, ANN202, ARG001
        dt = np.dtype(dtype) if dtype is not None else np.asarray(prototype).dtype
        fill = fill_for(dt)
        return np.empty_like(prototype, dtype=dt) if fill is None else np.full_like(prototype, fill, dtype=dt)

    shim.empty = empty
    shim.empty_like = empty_like
    return shim


def _copies(args: tuple) -> tuple:
    return tuple(a.copy() if isinstance(a, np.ndarray) else a for a in args)


def _assert_same(control, poisoned) -> None:
    if isinstance(control, np.ndarray):
        np.testing.assert_array_equal(poisoned, control)
    elif isinstance(control, (tuple, list)):
        assert len(poisoned) == len(control)
        for c, p in zip(control, poisoned, strict=True):
            _assert_same(c, p)
    else:
        assert poisoned == control


def check(monkeypatch, module, func, *args, **kwargs) -> None:
    """Run ``func`` (a kernel's ``py_func`` or a plain function) both ways and compare."""
    control = func(*_copies(args), **kwargs)
    monkeypatch.setattr(module, "np", poisoned_numpy())
    poisoned = func(*_copies(args), **kwargs)
    monkeypatch.undo()
    _assert_same(control, poisoned)


# ----------------------------------------------------------------- fixtures

F0 = 100.0
T_INIT = 10.0
AMP, OMEGA, PHASE = 3e4, 2 * np.pi / 400.0, 0.7
STRATEGIES = ("aggressive", "quadrature", "conservative")


def _orbit_state(t: float) -> np.ndarray:
    """[d5..d0] at t of x(t) = AMP sin(OMEGA (t - T_INIT) + PHASE)."""
    a, w, arg = AMP, OMEGA, OMEGA * (t - T_INIT) + PHASE
    return np.array(
        [a * w**5 * np.cos(arg), a * w**4 * np.sin(arg), -a * w**3 * np.cos(arg),
         -a * w**2 * np.sin(arg), a * w * np.cos(arg), a * np.sin(arg)],
    )


def _leaf(state: np.ndarray, steps: np.ndarray) -> np.ndarray:
    leaf = np.zeros((1, 7, 2))
    leaf[0, :6, 0] = state
    leaf[0, :6, 1] = steps
    leaf[0, -1, 0] = F0
    return leaf


def _circular_leaves() -> np.ndarray:
    """One leaf of each class the circular resolvers distinguish."""
    orbit = _orbit_state(T_INIT)
    snap = _leaf(orbit, np.abs(orbit) * 1e-3 + 1e-12)
    hole_state = orbit.copy()
    hole_state[1] = hole_state[3] = 0.0
    hole = _leaf(hole_state, np.abs(orbit) * 1e-3 + np.array([0, 1, 0, 1, 0, 0]))
    taylor = _leaf(np.array([0.0, 0.0, 0.0, 5.0, 2e3, 0.0]), np.array([1.0, 1.0, 1.0, 0.01, 10.0, 0.0]))
    return np.concatenate([snap, hole, taylor])


def _taylor_full(n_params: int = 4, n: int = 3) -> np.ndarray:
    """(n, n_params, 2) Taylor leaves with values and positive errors."""
    rng = np.random.default_rng(7)
    out = np.zeros((n, n_params, 2))
    out[..., 0] = rng.normal(size=(n, n_params)) * 10.0 ** -np.arange(n_params)[::-1]
    out[..., 1] = np.abs(rng.normal(size=(n, n_params))) * 1e-3 + 1e-9
    return out


CIRC_LIMITS = np.array([[-20.0, 20.0], [99.0, 101.0]])
CIRC_COUNTS = np.array([16, 64])
NBINS = 64


# ----------------------------------------------------------------- utils/transforms


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_shift_taylor_full(monkeypatch, strategy: str) -> None:
    check(monkeypatch, transforms, transforms.shift_taylor_full.py_func, _taylor_full(), 12.5, strategy)


def test_taylor_to_circular_full(monkeypatch) -> None:
    orbit = _orbit_state(T_INIT)
    sets = np.zeros((2, 4, 2))
    sets[:, :, 0] = orbit[1:5]  # snap, jerk, accel, vel of a real orbit
    sets[:, :, 1] = np.abs(orbit[1:5]) * 1e-3 + 1e-12
    check(monkeypatch, transforms, transforms.taylor_to_circular_full.py_func, sets)


@pytest.mark.parametrize("in_hole", [False, True])
def test_shift_taylor_circular_params(monkeypatch, in_hole: bool) -> None:
    state = _orbit_state(T_INIT)
    if in_hole:
        state[1] = state[3] = 0.0
    check(monkeypatch, transforms, transforms.shift_taylor_circular_params.py_func,
          np.stack([state, state * 1.01]), 20.0, in_hole)


@pytest.mark.parametrize("strategy", ["aggressive", "quadrature"])  # conservative is NotImplemented there
def test_shift_taylor_circular_errors(monkeypatch, strategy: str) -> None:
    errors = np.abs(_orbit_state(T_INIT)) * 1e-3 + 1e-12
    check(monkeypatch, transforms, transforms.shift_taylor_circular_errors.py_func,
          np.stack([errors, errors * 2]), 20.0, 300.0, strategy)


def test_taylor_to_cheby_full(monkeypatch) -> None:
    check(monkeypatch, transforms, transforms.taylor_to_cheby_full.py_func, _taylor_full(), 5.0)


def test_cheby_to_taylor_full(monkeypatch) -> None:
    alpha = transforms.taylor_to_cheby_full(_taylor_full(), 5.0)
    check(monkeypatch, transforms, transforms.cheby_to_taylor_full.py_func, alpha, 5.0)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_shift_cheby_full(monkeypatch, strategy: str) -> None:
    alpha = transforms.taylor_to_cheby_full(_taylor_full(), 5.0)
    check(monkeypatch, transforms, transforms.shift_cheby_full.py_func, alpha, (30.0, 8.0), (20.0, 5.0), strategy)


# ----------------------------------------------------------------- utils/psr_utils


@pytest.mark.parametrize("use_cheby", [True, False])
@pytest.mark.parametrize("kernel", [psr_utils.poly_taylor_shift_d_vec, psr_utils.poly_taylor_shift_d_f_vec])
def test_poly_taylor_shift_factors(monkeypatch, kernel, use_cheby: bool) -> None:
    f_cur = np.array([99.5, 100.0, 100.5])
    dparam_old = np.tile(np.array([2e-4, 2.0, 0.05]), (3, 1))
    dparam_new = dparam_old / 3
    check(monkeypatch, psr_utils, kernel.py_func, dparam_old, dparam_new, 40.0, NBINS, f_cur, 0.0, use_cheby)


# ----------------------------------------------------------------- core/common, utils/np_utils, core/fold


def test_get_leaves_opt(monkeypatch) -> None:
    param_arr = typed.List([np.array([-1.0, 0.0, 1.0]), np.array([99.0, 100.0]), np.array([0.0])])
    check(monkeypatch, common, common.get_leaves_opt.py_func, param_arr, np.array([0.5, 0.25, 1.0]))


def test_cartesian_prod_padded(monkeypatch) -> None:
    padded = np.zeros((2, 3, 4))
    padded[0, 0, :2] = [1.0, 2.0]
    padded[0, 1, :3] = [10.0, 20.0, 30.0]
    padded[0, 2, :1] = [100.0]
    padded[1, 0, :1] = [-1.0]
    padded[1, 1, :2] = [-10.0, -20.0]
    padded[1, 2, :4] = [-100.0, -200.0, -300.0, -400.0]
    counts = np.array([[2, 3, 1], [1, 2, 4]], dtype=np.int64)
    check(monkeypatch, np_utils, np_utils.cartesian_prod_padded.py_func, padded, counts, 2, 3)


def test_brutefold_start_complex_opt(monkeypatch) -> None:
    rng = np.random.default_rng(3)
    nsegments, segment_len = 4, 256
    ts_e = rng.standard_normal(nsegments * segment_len).astype(np.float32)
    ts_v = np.ones_like(ts_e)
    check(monkeypatch, fold, fold.brutefold_start_complex_opt.py_func,
          ts_e, ts_v, np.array([99.0, 100.0, 101.0]), segment_len, 16, 1e-3, 0.0)


# ----------------------------------------------------------------- kepler, simulation


def test_poly_projection_matrix(monkeypatch) -> None:
    check(monkeypatch, kepler, kepler.poly_projection_matrix.py_func, np.linspace(-1.0, 1.0, 40), 4)


@pytest.mark.parametrize("n", [0, 3, 4, 9])
def test_to_derivatives_series(monkeypatch, n: int) -> None:
    orbit = modulate.CircularModulating(p_orb=5000.0, psi=0.7, x_orb=0.1)
    control = orbit.to_derivatives_series(n)
    monkeypatch.setattr(modulate, "np", poisoned_numpy())
    poisoned = orbit.to_derivatives_series(n)
    monkeypatch.undo()
    assert set(control) == {"coeffs"}  # its only output
    np.testing.assert_array_equal(poisoned["coeffs"], control["coeffs"])


# ----------------------------------------------------------------- core/circular resolvers (rows filled per leaf class)


def test_circular_fixture_has_one_leaf_per_class() -> None:
    """The resolver cases are not vacuous: the batch holds one leaf of each class."""
    snap, crackle, taylor = circular.get_circ_taylor_mask(_circular_leaves(), 2.0)
    assert (snap.tolist(), crackle.tolist(), taylor.tolist()) == ([0], [1], [2])


@pytest.mark.parametrize("kernel", [circular.circ_taylor_resolve_batch, circular.circ_taylor_fixed_resolve_batch])
def test_circular_resolvers(monkeypatch, kernel) -> None:
    check(monkeypatch, circular, kernel.py_func, _circular_leaves(), (30.0037, 1.0), (18.0, 1.0), (T_INIT, 1.0),
          CIRC_COUNTS, CIRC_LIMITS, NBINS, 2.0)


def test_circular_ascend_resolver(monkeypatch) -> None:
    coords = np.array([[12.0, 1.0], [20.0, 1.0], [30.0037, 1.0]])
    check(monkeypatch, circular, circular.circ_taylor_ascend_resolve_batch.py_func, _circular_leaves(), coords,
          (18.0, 1.0), CIRC_COUNTS, CIRC_LIMITS, NBINS, 2.0)


# ----------------------------------------------------------------- ffa.FFAPlan (attributes filled in configure_plan)


def test_ffa_plan_configure(monkeypatch) -> None:
    nsamps, dt = 2**18, 1e-4
    limits = ParamLimits.from_upper((99.0, 101.0), [0.0, 5.0], (-20.0, 20.0), nsamps * dt)
    cfg = PulsarSearchConfig(nsamps=nsamps, tsamp=dt, nbins=32, eta=1.0, param_limits=limits.limits,
                             bseg_brute=2**10, bseg_ffa=2**14, prune_poly_order=3)
    control = ffa.FFAPlan(cfg)
    monkeypatch.setattr(ffa, "np", poisoned_numpy())
    poisoned = ffa.FFAPlan(cfg)
    monkeypatch.undo()
    arrays = {k: v for k, v in vars(control).items() if isinstance(v, np.ndarray)}
    assert arrays, "the plan holds arrays"
    for name, value in arrays.items():
        np.testing.assert_array_equal(getattr(poisoned, name), value)
    # The per-level parameter grids live in a list of lists of arrays.
    assert len(poisoned.params) == len(control.params) > 0
    for level_c, level_p in zip(control.params, poisoned.params, strict=True):
        assert len(level_p) == len(level_c)
        for c, p in zip(level_c, level_p, strict=True):
            np.testing.assert_array_equal(p, c)
