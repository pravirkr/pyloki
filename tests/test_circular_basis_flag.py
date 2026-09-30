"""`circ_taylor_resolve_batch` flags each leaf with the class that moved it.

The flag, ``leaves[:, -1, 1]``, is 1 for a snap-identified circular leaf, 2 for a
crackle-identified ("hole") one and 0 for a Taylor one, and it reaches the results
through the report. A batch holding all three classes checks that each write lands on
its own class.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyloki.core import circular
from tests.jit_utils import jit_variants

AMP, OMEGA, PHASE = 3e4, 2 * np.pi / 400.0, 0.7


def _orbit() -> np.ndarray:
    """[d5..d0] of x(t) = AMP sin(OMEGA t + PHASE) at t = 0."""
    a, w, ph = AMP, OMEGA, PHASE
    return np.array(
        [
            a * w**5 * np.cos(ph),
            a * w**4 * np.sin(ph),
            -a * w**3 * np.cos(ph),
            -a * w**2 * np.sin(ph),
            a * w * np.cos(ph),
            a * np.sin(ph),
        ],
    )


def _leaves() -> np.ndarray:
    orbit = _orbit()
    leaves = np.zeros((3, 7, 2))
    leaves[:, -1, 0] = 100.0
    # Snap-identified: the whole orbit, every term significant.
    leaves[0, :6, 0] = orbit
    leaves[0, :6, 1] = np.abs(orbit) * 1e-3
    # Crackle-identified: snap and accel vanish, crackle and jerk are significant.
    hole = orbit.copy()
    hole[[1, 3]] = 0.0
    leaves[1, :6, 0] = hole
    leaves[1, :6, 1] = np.abs(orbit) * 1e-3 + [0, 1, 0, 1, 0, 0]
    # Taylor: only an acceleration.
    leaves[2, 3, 0] = 5.0
    leaves[2, :4, 1] = [1.0, 1.0, 1.0, 0.01]
    leaves[:, -1, 1] = 9.0  # none of the flag values
    return leaves


@pytest.mark.parametrize("impl", jit_variants(circular.circ_taylor_resolve_batch))
def test_moving_grid_flags_every_class(impl) -> None:
    leaves = _leaves()
    impl(
        leaves,
        (30.0, 1.0),
        (10.0, 1.0),
        (10.0, 1.0),
        np.array([16, 64]),
        np.array([[-20.0, 20.0], [99.0, 101.0]]),
        64,
        2.0,
    )
    np.testing.assert_array_equal(leaves[:, -1, 1], [1, 2, 0])
