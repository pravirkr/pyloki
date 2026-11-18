"""Every RNG in the library must be steerable from its caller.

Each public entry point that draws randomness takes `seed`, accepting an int, a
`np.random.Generator`, or `None`:

- the same seed reproduces the output, and different seeds change it;
- **`None` is the default and draws fresh entropy**, so a caller who never asked for
  determinism does not get it;
- `np.random.seed` is no substitute: `default_rng` ignores the legacy global seed, so
  the library must never draw from an unseeded or global generator.
  `test_no_unseeded_rng_in_library` enumerates the call sites from the source rather
  than from a list typed here, so a new unseeded generator fails this file.

Reproducibility holds under parallel execution too. The two kernels that draw
inside a `prange` take one generator per iteration rather than sharing one, so
their output does not depend on numba's thread count -- see
`TestThresholding.test_run_is_independent_of_thread_count`.
"""

from __future__ import annotations

import ast
import pathlib
import textwrap
from typing import TYPE_CHECKING

import numba
import numpy as np
import pytest

import pyloki
from pyloki.detection import thresholding
from pyloki.sensitivity import sim_ffa
from pyloki.simulation.pulse import PulseSignalConfig
from tests._threads import num_threads

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
            first,
            second,
            err_msg=f"{method}() is not reproducible under seed=42",
        )

    @pytest.mark.parametrize("method", GENERATE_METHODS)
    def test_different_seeds_differ(self, method: str) -> None:
        assert not np.array_equal(
            getattr(_config(42), method)().ts_e,
            getattr(_config(43), method)().ts_e,
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
            1.0 / BRANCHING_PATTERN,
            seed=7,
            **SCHEME_KW,
        )
        second = thresholding.determine_scheme(
            1.0 / BRANCHING_PATTERN,
            seed=7,
            **SCHEME_KW,
        )
        np.testing.assert_array_equal(first.thresholds, second.thresholds)

    def test_determine_scheme_different_seeds_differ(self) -> None:
        first = thresholding.determine_scheme(
            1.0 / BRANCHING_PATTERN,
            seed=7,
            **SCHEME_KW,
        )
        second = thresholding.determine_scheme(
            1.0 / BRANCHING_PATTERN,
            seed=8,
            **SCHEME_KW,
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
                1.0 / BRANCHING_PATTERN,
                seed=7,
                **SCHEME_KW,
            ).thresholds,
            dtype=np.float64,
        )
        first = thresholding.evaluate_scheme(ladder, seed=3, **SCHEME_KW)
        second = thresholding.evaluate_scheme(ladder, seed=3, **SCHEME_KW)
        np.testing.assert_array_equal(_survival(first), _survival(second))

    @pytest.mark.parametrize("mode", ["legacy", "improved"])
    def test_dynamic_threshold_scheme_init_reproduces(self, mode: str) -> None:
        """The constructor simulates the first stage's folds from `seed`."""

        def first_stage(seed: int | None) -> np.ndarray:
            scheme = thresholding.DynamicThresholdScheme(
                nthresholds=20,
                mode=mode,
                seed=seed,
                **SCHEME_KW,
            )
            folds = [f.folds_h0 for f in scheme.folds_in if f.folds_h0.size]
            assert folds, "the first stage holds no simulated folds"
            return np.concatenate(folds)

        np.testing.assert_array_equal(first_stage(5), first_stage(5))
        assert not np.array_equal(first_stage(5), first_stage(6))
        assert not np.array_equal(first_stage(None), first_stage(None))

    @staticmethod
    def _success_h0(mode: str, seed: int | None) -> np.ndarray:
        """Return `run()`'s `success_h0`, which depends on every draw the run makes.

        Not `threshold`: thresholds come off a fixed `np.linspace` and are identical
        even unseeded, so comparing them would pass vacuously. Nor the backtracked
        ladder: it is a quantised summary, and two genuinely different states can
        produce the same one.
        """
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

    @pytest.mark.parametrize("mode", ["legacy", "improved"])
    def test_run_reproduces(self, mode: str) -> None:
        seeded = self._success_h0(mode, 11)
        np.testing.assert_array_equal(
            self._success_h0(mode, 11),
            seeded,
            err_msg="seeded run() is not reproducible on repeated runs",
        )
        # Negative controls, so the assertion above cannot pass vacuously.
        assert not np.array_equal(self._success_h0(mode, 12), seeded), (
            "run() ignores the seed entirely"
        )
        assert not np.array_equal(
            self._success_h0(mode, None), self._success_h0(mode, None)
        ), "run() is deterministic without a seed; default behaviour changed"

    @pytest.mark.parametrize("mode", ["legacy", "improved"])
    def test_run_is_independent_of_thread_count(self, mode: str) -> None:
        """Both kernels that consume randomness draw inside a `prange`.

        `run_stage_legacy` and `pre_simulate_stage_folds`. A single shared generator
        there is not enough: which thread takes which draw depends on scheduling, so a
        seed would fix the stream and still leave the result varying. Each iteration
        gets its own generator, derived from `(entropy, istage)`.
        """
        many = numba.config.NUMBA_NUM_THREADS
        if many < 2:
            pytest.skip("needs >= 2 numba threads to compare against one")
        with num_threads(many):
            parallel = self._success_h0(mode, 11)
        with num_threads(1):
            serial = self._success_h0(mode, 11)
        np.testing.assert_array_equal(
            parallel,
            serial,
            err_msg=(
                "seeded run() depends on the thread count -- a shared generator "
                "is being consumed inside a prange"
            ),
        )


class TestSimFFA:
    """`sensitivity/sim_ffa.py`: the simulator's own seed jitters the pulse phase."""

    def test_same_seed_reproduces(self, monkeypatch) -> None:
        # The FFA searches are deterministic given the simulated series; the seed
        # reaches only the series, through phi0. Stub the searches out, so the losses
        # left to compare are the ones read straight off the series' folds.
        monkeypatch.setattr(
            sim_ffa, "test_sensitivity_ffa", lambda *_, **__: (1.0, 1.0, 1.0)
        )

        def losses(seed: int | None) -> np.ndarray:
            # A fresh, fixed-seed config per run: its noise must not differ.
            cfg = PulseSignalConfig(
                period=0.01,
                dt=64e-6,
                nsamps=NSAMPS,
                snr=20,
                ducy=0.1,
                mod_kwargs={"acc": 5.0},
                seed=1,
            )
            sim = sim_ffa.TestFFASensitivity(
                cfg=cfg,
                param_limits=np.array([[-10.0, 10.0], [99.0, 101.0]]),
                ducy_arr=np.array([0.1]),
                eta_arr=np.array([1]),
                ds_arr=np.array([1.0, 1.5]),
                seed=seed,
                quiet=True,
            )
            # One ducy step of `run()`, without `_save`'s output file.
            parts = sim._execute(cfg)  # noqa: SLF001
            return np.concatenate([part.ravel() for part in parts])

        np.testing.assert_array_equal(losses(5), losses(5))
        assert not np.array_equal(losses(5), losses(6))
        assert not np.array_equal(losses(None), losses(None))


# Constructors of seedable generators; any other `*.random.<name>(...)` draws from
# numpy's legacy global state, which no `seed` argument reaches.
SEEDABLE = frozenset(
    {
        "default_rng",
        "Generator",
        "SeedSequence",
        "BitGenerator",
        "PCG64",
        "PCG64DXSM",
        "MT19937",
        "Philox",
        "SFC64",
    }
)


def _unseeded_rng_calls(source: str) -> list[int]:
    """Line numbers of RNG calls in `source` that the caller cannot seed."""
    lines = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name == "default_rng":
            seed = node.args[0] if node.args else None
            seed = next((k.value for k in node.keywords if k.arg == "seed"), seed)
            if seed is None or (isinstance(seed, ast.Constant) and seed.value is None):
                lines.append(node.lineno)
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "random"
            and func.attr not in SEEDABLE
        ):
            lines.append(node.lineno)
    return lines


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        ("np.random.default_rng()", True),
        ("np.random.default_rng(None)", True),
        ("np.random.default_rng(seed=None)", True),
        ("from numpy.random import default_rng\ndefault_rng()", True),
        ("np.random.normal(0.0, 1.0, 5)", True),
        ("numpy.random.seed(1)", True),
        ("np.random.default_rng(seed)", False),
        ("from numpy.random import default_rng\ndefault_rng(7)", False),
        ("np.random.SeedSequence(seed).entropy", False),
        ("np.random.Generator(np.random.PCG64(1))", False),
        ("rng.normal(0.0, 1.0)", False),
    ],
)
def test_unseeded_rng_detector(source: str, flagged: bool) -> None:
    assert bool(_unseeded_rng_calls(textwrap.dedent(source))) is flagged


def test_no_unseeded_rng_in_library() -> None:
    """No call in `src/` draws from a generator its caller cannot seed."""
    root = pathlib.Path(pyloki.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{lineno}"
        for path in sorted(root.rglob("*.py"))
        for lineno in _unseeded_rng_calls(path.read_text())
    ]
    assert not offenders, (
        "unseeded or global-state numpy RNG call(s) -- the caller cannot reproduce "
        f"this run: {', '.join(offenders)}"
    )
