"""``circ_taylor_branch_batch`` writes every slot of its output (#41).

The kernel builds its output with ``np.empty``. Whether an unwritten slot shows up
depends on what is in memory, so the check here fixes the content: it runs the kernel's
Python source with the module's ``np.empty`` replaced by one that fills float arrays
with NaN and integer arrays with the dtype's minimum. Any slot the kernel does not write
is then NaN, deterministically. Both paths of the kernel are exercised: the early
return when no leaf needs crackle expansion, and the expansion path.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from pyloki.core import circular
from pyloki.utils import psr_utils

NBINS = 64
SIG = 2.0
COORD = (20.0, 12.0)
T_INIT = 10.0
AMP, OMEGA, PHASE = 3e4, 2 * np.pi / 400.0, 0.7


def _orbit_state(t: float) -> np.ndarray:
    """Return [d5..d0] at t of x(t) = AMP sin(OMEGA (t - T_INIT) + PHASE)."""
    a, w, arg = AMP, OMEGA, OMEGA * (t - T_INIT) + PHASE
    return np.array(
        [
            a * w**5 * np.cos(arg),
            a * w**4 * np.sin(arg),
            -a * w**3 * np.cos(arg),
            -a * w**2 * np.sin(arg),
            a * w * np.cos(arg),
            a * np.sin(arg),
        ],
    )


def _leaf(state: np.ndarray, steps: np.ndarray, f0: float = 100.0) -> np.ndarray:
    leaf = np.zeros((1, 7, 2))
    leaf[0, :6, 0] = state
    leaf[0, :6, 1] = steps
    leaf[0, -1, 0] = f0
    return leaf


def _snap_leaf() -> np.ndarray:
    state = _orbit_state(T_INIT)
    return _leaf(state, np.abs(state) * 1e-3 + 1e-12)


def _hole_leaf() -> np.ndarray:
    """Snap and accel vanish; crackle and jerk are significant and physical."""
    state = _orbit_state(T_INIT)
    state[1] = state[3] = 0.0
    steps = np.abs(_orbit_state(T_INIT)) * 1e-3 + np.array([0, 1, 0, 1, 0, 0])
    return _leaf(state, steps)


def _taylor_leaf(vel: float = 2e3, accel: float = 5.0) -> np.ndarray:
    state = np.array([0.0, 0.0, 0.0, accel, vel, 0.0])
    return _leaf(state, np.array([1.0, 1.0, 1.0, 0.01, 10.0, 0.0]))


def _with_steps(leaves: np.ndarray, factors: np.ndarray) -> np.ndarray:
    """Set the leaves' steps to multiples of the branching step, so some axes branch."""
    dnew = psr_utils.poly_taylor_step_d_vec(5, COORD[1], NBINS, 1.0, leaves[:, -1, 0], t_ref=0)
    leaves[:, :5, 1] = dnew * factors
    return leaves


def _expansion_input() -> np.ndarray:
    """A hole leaf whose crackle needs branching: the crackle-expansion path."""
    leaves = np.concatenate([_snap_leaf(), _hole_leaf(), _taylor_leaf()])
    leaves = _with_steps(leaves, np.array([3.3, 2.1, 2.4, 0.5, 2.2]))
    leaves[1, 2, 1] = abs(leaves[1, 2, 0]) / 10
    return leaves


def _fast_input() -> np.ndarray:
    """No crackle branching: the early return."""
    leaves = np.concatenate([_snap_leaf(), _taylor_leaf()])
    return _with_steps(leaves, np.array([0.5, 0.5, 2.4, 0.5, 2.2]))


def _poisoned_numpy() -> types.ModuleType:
    shim = types.ModuleType("numpy_poisoned")
    shim.__dict__.update(np.__dict__)

    def empty(shape, dtype=float, *args, **kwargs):  # noqa: ANN001, ANN202, ARG001
        dt = np.dtype(dtype)
        fill = np.nan if dt.kind == "f" else np.iinfo(dt).min
        return np.full(shape, fill, dtype=dt)

    shim.empty = empty
    return shim


@pytest.mark.parametrize("make_input", [_fast_input, _expansion_input])
def test_branch_writes_every_output_slot(monkeypatch, make_input) -> None:
    leaves = make_input()
    monkeypatch.setattr(circular, "np", _poisoned_numpy())
    out, origins = circular.circ_taylor_branch_batch.py_func(
        leaves, COORD, NBINS, 1.0, 5, 64, SIG
    )
    assert len(out) > len(leaves)  # something branched, so the path did real work
    assert np.isfinite(out).all()
    assert (origins >= 0).all()
