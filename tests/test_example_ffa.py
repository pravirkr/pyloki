"""CI regression tests converted from the three FFA example notebooks.

`examples/pyloki_ffa_freq.ipynb`, `pyloki_ffa_accel.ipynb` and `pyloki_ffa_jerk.ipynb`
are one pipeline with one, two and three search parameters respectively, so they are
parametrised here rather than copied into three near-identical modules.

**Why these are worth their runtime.** They cover three things no other test touches:

1. `pyloki.search.ffa_search`, a public entry point. The EP tests drive
   `DynamicProgramming` directly and never go through it.
2. `use_fourier=False`. Every EP test uses `True`, so the direct (non-Fourier) FFA
   backend was entirely untested. Each notebook runs both, and so does this.
3. `Periodogram` (as opposed to `ScatteredPeriodogram`) and its `find_best_params` /
   `find_best_indices` / `get_indices_summary` accessors.

There is no pruning here, which is why these are cheap relative to what they cover.

**Assertions are on grid indices, not parameter values.** `get_indices_summary` compares
the peak's index with the index nearest the injection, which is the natural statement
for an FFA periodogram: it is tighter than a parameter tolerance and needs no
uncertainty-based non-vacuity guard. The guard instead checks that each searched axis
has more than one cell -- otherwise "the peak is in the right cell" is trivially true.
That guard is not hypothetical: at `2**21` the jerk case has a **1-cell** jerk axis, and
its jerk index check would have been meaningless.

**Sizing, measured per case** (notebook value -> here):

| case | nsamps | grid shape | why |
|---|---|---|---|
| freq | `2**24 -> 2**20` | `(88102, 17)` | 1 param; cheap at any size |
| accel | `2**23 -> 2**21` | `(138, 550, 6)` | 138 accel cells, so the axis binds |
| jerk | `2**23 -> 2**22` | `(6, 33, 180, 6)` | see below |

The jerk case is the only one that needed care: `2**21` collapses the jerk axis to a
single cell, `2**22` gives 6, and the notebook's own `2**23` gives 44 but costs ~30 s
for both backends. `2**22` is the cheapest size at which the jerk check means anything.

`snr` is raised to 20 (notebooks use 10) for the same reason as the EP tests: a CI test
must not sit near threshold. Everything else -- `eta`, `ducy`, `ducy_max`, `wtsp`,
`bseg_brute` as a fraction of `nsamps`, the parameter limits -- is the notebook's.

**Measured stability**, index offset from the true cell, over both backends:

- freq, 8 realisations (16 measurements): 14x `0`, one `-1`, one `+1`.
- accel, 8 realisations (16): offsets `0` or `+1` on both axes.
- jerk, 6 realisations (12): 11 within 1 bin, one outlier at `jerk:-2, freq:-2`.

The tolerance is 3 bins, i.e. 1.5x the worst observed. A 1-bin tolerance would have
failed the jerk case roughly 8% of the time -- worth measuring rather than assuming.

Cost: ~14.7 s locally for all 27 tests, and consistently so -- the numba disk cache
does not make repeat runs cheap here, because each pytest invocation is a fresh process
and the time goes on process setup plus simulating ~7.3M samples across the three cases.
(Within a single process the searches themselves are 0.04-0.3 s once compiled, which is
what the sizing probes measured; that speed is not what a CI run sees.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from pyloki.config import ParamLimits, PulsarSearchConfig
from pyloki.search import ffa_search
from pyloki.simulation.pulse import PulseSignalConfig
from pyloki.utils import np_utils

DT = 64e-6
NBINS = 64
SNR = 20.0  # notebooks use 10; see docstring
INDEX_TOL = 3  # bins; worst observed offset was 2
MIN_SNR = 12.0  # peak must be a real detection, well above noise
BACKENDS = (False, True)  # use_fourier


@dataclass(frozen=True)
class FfaCase:
    """One notebook's configuration."""

    name: str
    period: float
    nsamps: int
    brute_div: int
    eta: int
    ducy: float
    ducy_max: float
    wtsp: float
    # Injected kinematic parameters beyond frequency, notebook values.
    mod_kwargs: dict[str, float] = field(default_factory=dict)
    # `from_upper` arguments for the multi-parameter cases; None for the freq case,
    # which passes an explicit frequency range instead.
    upper_params: tuple[float, ...] | None = None
    upper_range: tuple[float, float] | None = None

    def build(self):
        """Simulate and configure exactly as the notebook does."""
        cfg = PulseSignalConfig(
            period=self.period,
            dt=DT,
            nsamps=self.nsamps,
            snr=SNR,
            ducy=self.ducy,
            mod_kwargs=dict(self.mod_kwargs),
        )
        tim_data = cfg.generate(shape="gaussian")
        if self.upper_params is None:
            # pyloki_ffa_freq.ipynb: an explicit frequency range.
            param_limits = np.array([(1 / 0.0075, 1 / 0.0065)])
        else:
            param_limits = ParamLimits.from_upper(
                cfg.freq, list(self.upper_params), self.upper_range, cfg.tobs
            ).limits
        truth = {"freq": cfg.freq}
        if "acc" in self.mod_kwargs:
            truth["accel"] = self.mod_kwargs["acc"]
        if "jerk" in self.mod_kwargs:
            truth["jerk"] = self.mod_kwargs["jerk"]
        return cfg, tim_data, param_limits, truth


CASES = (
    FfaCase(
        name="freq",
        period=0.007,
        nsamps=2**20,
        brute_div=16384,
        eta=1,
        ducy=0.05,
        ducy_max=0.5,
        wtsp=1.2,
    ),
    FfaCase(
        name="accel",
        period=0.007,
        nsamps=2**21,
        brute_div=256,
        eta=1,
        ducy=0.1,
        ducy_max=0.2,
        wtsp=1.5,
        mod_kwargs={"acc": 200.0},
        upper_params=(200.0,),
        upper_range=(-2000.0, 2000.0),
    ),
    FfaCase(
        name="jerk",
        period=0.010,
        nsamps=2**22,
        brute_div=128,
        eta=2,
        ducy=0.1,
        ducy_max=0.2,
        wtsp=1.5,
        mod_kwargs={"acc": 300.0, "jerk": 1.0},
        upper_params=(1.0, 300.0),
        upper_range=(-2.5, 2.5),
    ),
)


@pytest.fixture(scope="module", params=CASES, ids=lambda c: c.name)
def ffa_result(request) -> dict:
    """Run `ffa_search` once per case, on both backends, over the same data."""
    case: FfaCase = request.param
    cfg, tim_data, param_limits, truth = case.build()

    results = {}
    for use_fourier in BACKENDS:
        search_cfg = PulsarSearchConfig(
            nsamps=cfg.nsamps,
            tsamp=cfg.dt,
            nbins=NBINS,
            eta=case.eta,
            param_limits=param_limits,
            bseg_brute=cfg.nsamps // case.brute_div,
            ducy_max=case.ducy_max,
            wtsp=case.wtsp,
            use_fourier=use_fourier,
        )
        _dyp, pgram = ffa_search(
            tim_data, search_cfg, quiet=True, show_progress=False
        )
        results[use_fourier] = pgram
    return {"case": case, "truth": truth, "pgrams": results}


def _index_offsets(pgram, truth: dict[str, float]) -> dict[str, int]:
    """Best-peak index minus the index nearest the injected value, per axis."""
    best_indices = pgram.find_best_indices()
    offsets = {}
    for dim, idx in zip(pgram.data.dims, best_indices, strict=False):
        name = str(dim)
        if name in truth:
            true_idx = np_utils.find_nearest_sorted_idx(
                pgram.params[name], truth[name]
            )
            offsets[name] = int(idx) - int(true_idx)
    return offsets


@pytest.mark.parametrize("use_fourier", BACKENDS)
def test_search_returns_a_periodogram(ffa_result, use_fourier: bool) -> None:
    pgram = ffa_result["pgrams"][use_fourier]
    assert pgram.data.size > 0
    # Every injected parameter must appear as a periodogram axis.
    for name in ffa_result["truth"]:
        assert name in pgram.params, f"{name!r} missing from periodogram axes"


@pytest.mark.parametrize("use_fourier", BACKENDS)
def test_searched_axes_are_not_degenerate(ffa_result, use_fourier: bool) -> None:
    """Guard: an axis with one cell makes the index check below vacuous.

    This is exactly what ruled out `2**21` for the jerk case, where the jerk axis
    collapsed to a single cell.
    """
    pgram = ffa_result["pgrams"][use_fourier]
    for name in ffa_result["truth"]:
        ncells = len(pgram.params[name])
        assert ncells > 1, (
            f"{name!r} axis has {ncells} cell(s), so 'the peak is in the right cell' "
            f"would be trivially true -- this case needs a larger nsamps"
        )


@pytest.mark.parametrize("use_fourier", BACKENDS)
def test_peak_is_a_real_detection(ffa_result, use_fourier: bool) -> None:
    pgram = ffa_result["pgrams"][use_fourier]
    peak = float(pgram.find_best_params()["snr"])
    assert peak > MIN_SNR, f"peak S/N {peak:.2f} is not a detection"


@pytest.mark.parametrize("use_fourier", BACKENDS)
def test_peak_lands_on_the_injected_cell(ffa_result, use_fourier: bool) -> None:
    pgram = ffa_result["pgrams"][use_fourier]
    offsets = _index_offsets(pgram, ffa_result["truth"])
    assert offsets, "no searched axis matched an injected parameter"
    for name, offset in offsets.items():
        assert abs(offset) <= INDEX_TOL, (
            f"{name!r} peak is {offset:+d} cells from the injected value "
            f"(tolerance {INDEX_TOL}); case={ffa_result['case'].name}, "
            f"use_fourier={use_fourier}"
        )


def test_backends_agree(ffa_result) -> None:
    """The Fourier and direct FFA backends must find the same cell.

    A cross-check that needs no reference value: whatever the right answer is, the two
    implementations should agree on it. This is the only assertion here that would catch
    a regression in one backend but not the other.
    """
    direct = _index_offsets(ffa_result["pgrams"][False], ffa_result["truth"])
    fourier = _index_offsets(ffa_result["pgrams"][True], ffa_result["truth"])
    assert direct.keys() == fourier.keys()
    for name in direct:
        assert abs(direct[name] - fourier[name]) <= INDEX_TOL, (
            f"backends disagree on {name!r}: direct {direct[name]:+d} vs "
            f"fourier {fourier[name]:+d} cells from the injection"
        )
