"""The Extreme-Pruning pipeline shared by the example tests.

Each example module builds its own signal and search limits; this runs the rest the
way the notebooks do: FFA dynamic programming, the branching pattern, a "constant"
threshold scheme (`1 / branching` survivors per stage, as in
examples/optimal_thresholds.ipynb), pruning, and loading the candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from pyloki.config import PulsarSearchConfig
from pyloki.detection import thresholding
from pyloki.ffa import DynamicProgramming
from pyloki.periodogram import ScatteredPeriodogram
from pyloki.prune import prune_dyp_tree

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from pyloki.simulation.pulse import PulseSignalConfig

DUCY_MAX, WTSP, NTRIALS = 0.5, 1.2, 1024


@dataclass(frozen=True)
class EpSearch:
    """The search-side settings that differ between the examples."""

    nsegments: int
    prune_poly_order: int
    kind: str  # branching-pattern kind, e.g. "poly_taylor_moving"
    max_sugg: int
    ffa_levels: int = 16  # bseg_ffa // bseg_brute
    nbins: int = 64
    eta: int = 1
    branch_max: int = 32
    config_kwargs: dict[str, Any] = field(default_factory=dict)


def build_ep(
    signal: PulseSignalConfig,
    param_limits: np.ndarray,
    search: EpSearch,
    seed: int,
) -> tuple[DynamicProgramming, np.ndarray, np.ndarray]:
    """Simulate, run the FFA stage and derive the threshold ladder.

    Returns the executed `DynamicProgramming`, the thresholds and the branching
    pattern, ready for `prune_dyp_tree`.
    """
    tim_data = signal.generate(shape="gaussian")
    bseg_ffa = signal.nsamps // search.nsegments
    search_cfg = PulsarSearchConfig(
        nsamps=signal.nsamps,
        tsamp=signal.dt,
        nbins=search.nbins,
        eta=search.eta,
        param_limits=param_limits,
        bseg_brute=bseg_ffa // search.ffa_levels,
        bseg_ffa=bseg_ffa,
        prune_poly_order=search.prune_poly_order,
        ducy_max=DUCY_MAX,
        wtsp=WTSP,
        use_fourier=True,
        branch_max=search.branch_max,
        **search.config_kwargs,
    )
    dyp = DynamicProgramming(tim_data, search_cfg)
    dyp.initialize()
    dyp.execute()

    # `n_runs=1` prunes from the middle segment, so build the ladder for that one.
    branching_pattern = np.asarray(
        search_cfg.generate_branching_pattern(
            kind=search.kind, ref_seg=dyp.nsegments // 2
        ),
        dtype=np.float64,
    )
    scheme = thresholding.determine_scheme(
        1.0 / branching_pattern,
        branching_pattern,
        ref_ducy=signal.ducy,
        nbins=search.nbins,
        ntrials=NTRIALS,
        snr_final=signal.snr,
        ducy_max=DUCY_MAX,
        wtsp=WTSP,
        seed=seed,
    )
    return dyp, np.asarray(scheme.thresholds, dtype=np.float64), branching_pattern


def run_ep(
    signal: PulseSignalConfig,
    param_limits: np.ndarray,
    search: EpSearch,
    seed: int,
    outdir: Path,
) -> dict:
    """Run the whole pipeline once and return the candidates and its ladder."""
    dyp, thresholds, branching_pattern = build_ep(signal, param_limits, search, seed)
    result_file = prune_dyp_tree(
        dyp,
        thresholds,
        n_runs=1,
        max_sugg=search.max_sugg,
        outdir=str(outdir),
        file_prefix="ep",
        poly_basis="taylor",
        n_workers=1,
        use_moving_grid=True,
    )
    return {
        "data": ScatteredPeriodogram.load(result_file).data,
        "nsegments": dyp.nsegments,
        "thresholds": thresholds,
        "branching_pattern": branching_pattern,
        "snr": signal.snr,
    }


def best_candidate(result: dict) -> pd.Series:
    data = result["data"]
    return data.loc[data["score"].idxmax()]


# Checks every example makes on its pinned run.


def check_candidates(result: dict, columns: list[str]) -> None:
    data = result["data"]
    assert len(data) > 0, "no candidates survived; the search found nothing at all"
    for column in [*columns, "score"]:
        assert column in data.columns, f"missing column {column!r}"


def check_stage_count(result: dict, nsegments: int) -> None:
    """`nsegments` is driven by `bseg_ffa`, and the threshold ladder must match it."""
    assert result["nsegments"] == nsegments
    assert len(result["thresholds"]) == len(result["branching_pattern"])


# Recovery tolerances are in units of the best candidate's own reported uncertainty
# d<param>: k = 2 allows the peak to land one grid cell off (measured: it sometimes
# does), but not two.
RECOVERY_K = 2.0
# The best candidate's score against the injected S/N. Measured 0.92-1.00 over the
# pinned and sweep seeds of all three EP examples.
MIN_SCORE_FRACTION = 0.8


def check_recovered(
    result: dict,
    name: str,
    truth: float,
    max_unc_fraction: float,
    seed: object,
    *,
    k: float = RECOVERY_K,
    abs_tol: float | None = None,
) -> None:
    """Assert `|best[name] - truth| <= k * d<name>`, guarded against vacuity.

    The guard requires `d<name> < max_unc_fraction * |truth|`, so the recovery check
    cannot pass merely because the search left the parameter unconstrained.
    `abs_tol` replaces `k * d<name>` where the reported uncertainty is known to be
    optimistic; callers must say why.
    """
    best = best_candidate(result)
    unc = float(best[f"d{name}"])
    assert unc < max_unc_fraction * abs(truth), (
        f"{name} is not actually constrained (seed={seed}): d{name}={unc:.5g} "
        f"against |{name}|={abs(truth):.5g} -- the recovery check would be vacuous"
    )
    tol = k * unc if abs_tol is None else abs_tol
    got = float(best[name])
    error = abs(got - truth)
    assert error <= tol, (
        f"{name} not recovered (seed={seed}): got {got:.10g}, want {truth:.10g} "
        f"(error {error:.5g} = {error / unc:.2f} x d{name} > tolerance {tol:.5g})"
    )


def check_detection_strength(result: dict, seed: object) -> None:
    """Assert the best candidate carries most of the injected S/N, not just survives."""
    best_score = float(result["data"]["score"].max())
    floor = MIN_SCORE_FRACTION * result["snr"]
    assert best_score >= floor, (
        f"best score {best_score:.3f} is below {MIN_SCORE_FRACTION} x injected "
        f"snr {result['snr']} (seed={seed})"
    )
