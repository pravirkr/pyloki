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


def check_clears_final_threshold(result: dict) -> None:
    best_score = float(result["data"]["score"].max())
    final_threshold = float(result["thresholds"][-1])
    assert best_score > final_threshold, (
        f"best score {best_score:.3f} does not clear the final threshold "
        f"{final_threshold:.3f}"
    )
