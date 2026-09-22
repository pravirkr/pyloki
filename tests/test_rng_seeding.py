"""Every RNG in the library must be steerable from its caller.

The library used to call `np.random.default_rng()` with no argument in eight
places and expose no way to seed any of them, so a run could not be repeated.
`np.random.seed` does not help: `default_rng` ignores the legacy global seed,
which is what made the defect easy to miss -- the usual reflex appears to work
and changes nothing.

Each public entry point now takes `seed`, accepting an int, a
`np.random.Generator`, or `None`. **`None` is still the default and still draws
fresh entropy**, so these tests also pin that the fix did not quietly make the
library deterministic for callers who never asked for it.

`test_no_bare_default_rng_in_library` enumerates the call sites from the source
rather than from a list typed here, so a new unseeded generator fails this file
instead of going unnoticed.

Reproducibility holds under parallel execution too. The two kernels that draw
inside a `prange` take one generator per iteration rather than sharing one, so
their output does not depend on numba's thread count -- see
`TestThresholding.test_run_reproduces_under_parallelism`.
"""

from __future__ import annotations

import ast
import pathlib
from typing import TYPE_CHECKING

import numba
import numpy as np
import pytest

import pyloki
from pyloki.detection import thresholding

# aliased: pytest would otherwise try to collect the library's `Test*` class
from pyloki.sensitivity.sim_ffa import TestFFASensitivity as FFASensitivitySim
from pyloki.simulation.pulse import PulseSignalConfig

if TYPE_CHECKING:
    from pyloki.detection.schemes import StatesInfo

# Small enough that the whole file runs in a few seconds; nothing here depends
# on the configuration, only on whether two runs of it agree.
NSAMPS = 2**16
BRANCHING_PATTERN = np.full(4, 2.0)
SCHEME_KW = {
    "branching_pattern": BRANCHING_PATTERN,
    "ref_ducy": 0.1,
    "nbins": 64,
    "ntrials": 256,
    "snr_final": 8.0,
}
GENERATE_METHODS = ("generate", "generate_simple", "generate_noise", "generate_old")


def _config(seed: int | np.random.Generator | None) -> PulseSignalConfig:
    return PulseSignalConfig(
        period=0.01,
        dt=64e-6,
        nsamps=NSAMPS,
        snr=20,
        ducy=0.1,
        mod_kwargs={"acc": 100.0},
        seed=seed,
    )


def _survival(info: StatesInfo) -> np.ndarray:
    return info.get_info("success_h1_cumul")


class TestPulseSignalConfig:
    """`simulation/pulse.py` -- the four noise draws."""

    @pytest.mark.parametrize("method", GENERATE_METHODS)
    def test_same_seed_reproduces(self, method: str) -> None:
        first = getattr(_config(42), method)().ts_e
        second = getattr(_config(42), method)().ts_e
        np.testing.assert_array_equal(
            first, second, err_msg=f"{method}() is not reproducible under seed=42",
        )

    @pytest.mark.parametrize("method", GENERATE_METHODS)
    def test_different_seeds_differ(self, method: str) -> None:
        assert not np.array_equal(
            getattr(_config(42), method)().ts_e, getattr(_config(43), method)().ts_e,
        ), f"{method}() ignores the seed -- 42 and 43 gave the same noise"

    @pytest.mark.parametrize("method", GENERATE_METHODS)
    def test_unseeded_still_draws_fresh_entropy(self, method: str) -> None:
        """The default must not have become deterministic."""
        assert not np.array_equal(
            getattr(_config(None), method)().ts_e,
            getattr(_config(None), method)().ts_e,
        ), f"{method}() is deterministic without a seed; default behaviour changed"

    def test_successive_calls_advance_the_stream(self) -> None:
        """A seed fixes the *sequence*, not every call.

        Otherwise a loop over one seeded config would silently draw the same
        realisation every iteration, which is the trap in re-seeding per call.
        """
        cfg = _config(42)
        first, second = cfg.generate_noise().ts_e, cfg.generate_noise().ts_e
        assert not np.array_equal(first, second), (
            "two successive draws from one seeded config are identical; "
            "the generator is being re-created per call"
        )
        replay = _config(42)
        np.testing.assert_array_equal(replay.generate_noise().ts_e, first)
        np.testing.assert_array_equal(replay.generate_noise().ts_e, second)

    def test_accepts_a_generator(self) -> None:
        first = _config(np.random.default_rng(9)).generate_noise().ts_e
        second = _config(np.random.default_rng(9)).generate_noise().ts_e
        np.testing.assert_array_equal(first, second)

    def test_seed_survives_get_updated(self) -> None:
        """`get_updated` rebuilds the config through the constructor."""
        updated = _config(42).get_updated({"snr": 30})
        assert updated.seed == 42
        assert updated.snr == 30

    def test_stored_generator_does_not_break_equality(self) -> None:
        """Generators compare by identity; that must not leak into config `==`."""
        assert _config(42) == _config(42)
        assert _config(None) == _config(None)


class TestThresholding:
    """`detection/thresholding.py` -- the three ladder generators."""

    def test_determine_scheme_same_seed_reproduces(self) -> None:
        first = thresholding.determine_scheme(
            1.0 / BRANCHING_PATTERN, seed=7, **SCHEME_KW,
        )
        second = thresholding.determine_scheme(
            1.0 / BRANCHING_PATTERN, seed=7, **SCHEME_KW,
        )
        np.testing.assert_array_equal(first.thresholds, second.thresholds)

    def test_determine_scheme_different_seeds_differ(self) -> None:
        first = thresholding.determine_scheme(
            1.0 / BRANCHING_PATTERN, seed=7, **SCHEME_KW,
        )
        second = thresholding.determine_scheme(
            1.0 / BRANCHING_PATTERN, seed=8, **SCHEME_KW,
        )
        assert not np.array_equal(first.thresholds, second.thresholds)

    def test_determine_scheme_unseeded_still_varies(self) -> None:
        first = thresholding.determine_scheme(1.0 / BRANCHING_PATTERN, **SCHEME_KW)
        second = thresholding.determine_scheme(1.0 / BRANCHING_PATTERN, **SCHEME_KW)
        assert not np.array_equal(first.thresholds, second.thresholds), (
            "unseeded ladders are now identical; default behaviour changed"
        )

    def test_evaluate_scheme_same_seed_reproduces(self) -> None:
        ladder = np.asarray(
            thresholding.determine_scheme(
                1.0 / BRANCHING_PATTERN, seed=7, **SCHEME_KW,
            ).thresholds,
            dtype=np.float64,
        )
        first = thresholding.evaluate_scheme(ladder, seed=3, **SCHEME_KW)
        second = thresholding.evaluate_scheme(ladder, seed=3, **SCHEME_KW)
        np.testing.assert_array_equal(_survival(first), _survival(second))

    def test_dynamic_threshold_scheme_seeds_its_generator(self) -> None:
        """The seed reaches `self.rng`. That is NOT the same as `run()` reproducing."""

        def draws(seed: int | None) -> np.ndarray:
            scheme = thresholding.DynamicThresholdScheme(
                nthresholds=20, seed=seed, **SCHEME_KW,
            )
            return scheme.rng.standard_normal(8)

        np.testing.assert_array_equal(draws(5), draws(5))
        assert not np.array_equal(draws(None), draws(None))

    @pytest.mark.parametrize("mode", ["legacy", "improved"])
    def test_run_reproduces_under_parallelism(self, mode: str) -> None:
        """`run()` reproduces from a seed regardless of how many threads numba uses.

        Both kernels that consume randomness (`run_stage_legacy`,
        `pre_simulate_stage_folds`) draw inside a `prange`. A single shared generator
        there is not enough: which thread takes which draw depends on scheduling, so a
        seed would fix the stream and still leave the result varying. Each iteration
        gets its own generator instead, derived from `(entropy, istage)`, which makes
        the output independent of thread order.

        Compares `success_h0`, not `threshold`: thresholds come off a fixed
        `np.linspace` and are identical even unseeded, so comparing them would pass
        vacuously. The backtracked ladder is no good either -- it is a quantised
        summary, and two genuinely different states can produce the same one.
        """

        def success_h0(seed: int | None, nthreads: int) -> np.ndarray:
            original = numba.get_num_threads()
            numba.set_num_threads(nthreads)
            try:
                scheme = thresholding.DynamicThresholdScheme(
                    np.array([2.0] * 6),
                    ref_ducy=0.1,
                    nbins=32,
                    ntrials=256,
                    nprobs=8,
                    nthresholds=20,
                    snr_final=8.0,
                    mode=mode,
                    seed=seed,
                )
                scheme.run(thres_neigh=5)
                return np.asarray(scheme.states["success_h0"]).copy()
            finally:
                numba.set_num_threads(original)

        many = numba.get_num_threads()
        np.testing.assert_array_equal(
            success_h0(11, many),
            success_h0(11, many),
            err_msg="seeded run() is not reproducible on repeated runs",
        )
        if many > 1:
            np.testing.assert_array_equal(
                success_h0(11, many),
                success_h0(11, 1),
                err_msg=(
                    "seeded run() depends on the thread count -- a shared generator "
                    "is being consumed inside a prange"
                ),
            )
        # Negative controls, so the assertions above cannot pass vacuously.
        assert not np.array_equal(success_h0(11, many), success_h0(12, many)), (
            "run() ignores the seed entirely"
        )
        assert not np.array_equal(success_h0(None, many), success_h0(None, many)), (
            "run() is deterministic without a seed; default behaviour changed"
        )


class TestSimFFA:
    """`sensitivity/sim_ffa.py`."""

    def test_same_seed_reproduces(self) -> None:
        cfg = PulseSignalConfig(period=0.01, dt=64e-6, nsamps=NSAMPS, snr=20, ducy=0.1)
        limits = np.array([[100.0, 200.0], [-10.0, 10.0]])

        def draws(seed: int | None) -> np.ndarray:
            sim = FFASensitivitySim(cfg=cfg, param_limits=limits, seed=seed)
            return sim.rng.standard_normal(8)

        np.testing.assert_array_equal(draws(5), draws(5))
        assert not np.array_equal(draws(None), draws(None))


def test_no_bare_default_rng_in_library() -> None:
    """No `default_rng()` anywhere in `src/` may be called with no argument.

    Enumerated from the source so that a newly added unseeded generator fails
    here rather than reintroducing the defect unnoticed.
    """
    root = pathlib.Path(pyloki.__file__).parent
    offenders = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "default_rng"
                and not node.args
                and not node.keywords
            ):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not offenders, (
        "unseeded np.random.default_rng() call(s) -- the caller cannot reproduce "
        f"this run: {', '.join(offenders)}"
    )
