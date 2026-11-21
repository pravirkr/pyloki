"""Synthetic eccentric Keplerian blind-search benchmark.

Compares empirical-SVD orbital pruning (``SVDPruningRunner``) against
Chebyshev polynomial EP across orbital eccentricities ``e in [0.0, 0.5]``
on synthetic binaries.
"""

from __future__ import annotations

import argparse

import numpy as np

from pyloki.config import PulsarSearchConfig
from pyloki.core import svd_kepler
from pyloki.dynamic.dyn_svd import SVDPruningRunner
from pyloki.ffa import DynamicProgramming
from pyloki.search import ep_search
from pyloki.simulation.pulse import PulseSignalConfig
from pyloki.utils.misc import C_VAL
from pyloki.utils.snail import MiddleOutScheme


def _build_span_axes(
    nsegments: int,
    t_seg: float,
    ref_seg: int,
    n_sub: int = 8,
) -> list[np.ndarray]:
    scheme = MiddleOutScheme(
        nsegments=nsegments,
        ref_idx=ref_seg,
        tsegment=t_seg,
        stride=1,
    )
    axes: list[np.ndarray] = []
    segs_so_far: list[int] = []
    for lvl in range(nsegments):
        segs_so_far.append(scheme.get_segment_idx(lvl))
        lo = min(segs_so_far) * t_seg
        hi = (max(segs_so_far) + 1) * t_seg
        n_pts = max(8, len(segs_so_far) * n_sub + 1)
        axes.append(np.linspace(lo, hi, n_pts, dtype=np.float64))
    return axes


def run_benchmark(
    ecc_values: tuple[float, ...] = (0.0, 0.2, 0.4),
    *,
    nsamps: int = 2**14,
    dt: float = 1e-3,
    period: float = 0.02,
    p_orb: float = 400.0,
    x_orb: float = 0.15,
    snr: float = 24.0,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Run an eccentric sweep comparing Chebyshev EP and empirical-SVD pruning."""
    nbins = 32
    bseg_brute = 2**10
    bseg_ffa = 2**11
    nsegments = nsamps // bseg_ffa
    t_seg = bseg_ffa * dt
    f0 = 1.0 / period
    ref_seg = nsegments // 2

    span_axes = _build_span_axes(nsegments, t_seg, ref_seg)
    basis = svd_kepler.SVDOrbitalBasis.build(
        span_axes,
        p_orb_range=(0.75 * p_orb, 1.5 * p_orb),
        x_orb_range=(0.0, 1.5 * x_orb),
        ecc_range=(0.0, max(ecc_values) + 0.05),
        tau_tol=period / nbins,
        n_examples=300,
        seed=seed,
    )
    svd_runner = SVDPruningRunner(basis)

    results: list[dict[str, float]] = []
    for idx, ecc in enumerate(ecc_values):
        sig = PulseSignalConfig(
            period=period,
            dt=dt,
            nsamps=nsamps,
            snr=snr,
            ducy=0.1,
            mod_type="keplerian",
            mod_kwargs={
                "p_orb": p_orb,
                "a": x_orb * C_VAL,
                "ecc": ecc,
                "om": 0.6,
                "phi": 0.4,
                "inc": 0.5 * np.pi,
            },
            mod_tref=0.0,
            seed=seed + idx + 1,
        )
        ts = sig.generate(shape="gaussian")
        cfg = PulsarSearchConfig(
            nsamps=nsamps,
            tsamp=dt,
            nbins=nbins,
            eta=1.0,
            param_limits=np.array([[-15.0, 15.0], [f0 - 0.15, f0 + 0.15]]),
            prune_poly_order=2,
            bseg_brute=bseg_brute,
            bseg_ffa=bseg_ffa,
            use_fourier=False,
        )
        thresholds = np.full(nsegments - 1, 1.0, dtype=np.float32)

        ep_res = ep_search(
            ts,
            cfg,
            threshold_scheme=thresholds,
            poly_basis="chebyshev",
            n_runs=1,
            ref_segs=[nsegments // 2],
            quiet=True,
        )

        dyp = DynamicProgramming(ts, cfg)
        dyp.initialize()
        dyp.execute(show_progress=False)
        svd_res = svd_runner.execute(
            dyp.fold,
            dyp.param_arr,
            dyp.param_grid_count,
            cfg.param_limits,
            cfg,
            t_seg=t_seg,
            threshold_scheme=thresholds,
            ref_seg=ref_seg,
        )
        svd_snr = float(svd_res.scores[0]) if svd_res.scores.size > 0 else 0.0
        results.append(
            {
                "ecc": float(ecc),
                "cheby_ep_snr": float(ep_res.best_score),
                "svd_ep_snr": svd_snr,
            },
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eccs",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.4],
        help="Orbital eccentricities to benchmark",
    )
    args = parser.parse_args()
    rows = run_benchmark(tuple(args.eccs))
    for r in rows:
        print(
            f"e={r['ecc']:.2f} | cheby_ep_snr={r['cheby_ep_snr']:.2f} | "
            f"svd_ep_snr={r['svd_ep_snr']:.2f}",
        )


if __name__ == "__main__":
    main()
