from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import attrs
import numpy as np

from pyloki.config import PulsarSearchConfig, compute_max_coherent_subarc
from pyloki.detection import scoring
from pyloki.detection.thresholding import determine_scheme, schedule_survive_probs
from pyloki.ffa import DynamicProgramming
from pyloki.periodogram import Periodogram
from pyloki.prune import Pruning, sliding_arc_windows
from pyloki.utils.misc import get_logger, quiet_logger
from pyloki.utils.np_utils import determine_ref_segs_pareto

if TYPE_CHECKING:
    from numpy import typing as npt

    from pyloki.detection.scoring import ScoreMode
    from pyloki.io.timeseries import TimeSeries

logger = get_logger(__name__)


def ffa_search(
    tseries: TimeSeries,
    search_cfg: PulsarSearchConfig,
    *,
    quiet: bool = False,
    show_progress: bool = False,
) -> tuple[DynamicProgramming, Periodogram]:
    """Perform a Fast Folding Algorithm search on a time series.

    Parameters
    ----------
    tseries : TimeSeries
        The time series to search.
    search_cfg : PulsarSearchConfig
        The configuration object for the search.
    quiet : bool, optional
        If True, suppresses logging output, by default False.
    show_progress : bool, optional
        Whether to show progress bar, by default False.

    Returns
    -------
    tuple[DynamicProgramming, Periodogram]
        The DynamicProgramming object and the Periodogram object.
    """
    with quiet_logger(quiet=quiet):
        dyp = DynamicProgramming(tseries, search_cfg)
        dyp.initialize()
        dyp.execute(show_progress=show_progress)
        if search_cfg.score_mode == "boxcar":
            folds = dyp.get_fold_norm()
            snrs = scoring.boxcar_snr_nd(folds, search_cfg.score_widths, 1.0)
            width_axis = search_cfg.score_widths
        elif search_cfg.score_mode == "boxcar_glrt":
            fold_t = (
                np.fft.irfft(dyp.fold, n=search_cfg.nbins).astype(np.float32)
                if search_cfg.use_fourier
                else dyp.fold.astype(np.float32)
            )
            param_shape = fold_t.shape[1:-2]
            flat_ev = np.ascontiguousarray(
                fold_t[0].reshape(-1, 2, search_cfg.nbins),
                dtype=np.float32,
            )
            snrs_flat = scoring.boxcar_snr_glrt_3d(flat_ev, search_cfg.score_widths)
            snrs = snrs_flat.reshape((*param_shape, search_cfg.score_widths.size))
            width_axis = search_cfg.score_widths
        else:
            fold_t = (
                np.fft.irfft(dyp.fold, n=search_cfg.nbins).astype(np.float32)
                if search_cfg.use_fourier
                else dyp.fold.astype(np.float32)
            )
            param_shape = fold_t.shape[1:-2]
            flat_ev = np.ascontiguousarray(
                fold_t[0].reshape(-1, 2, search_cfg.nbins),
                dtype=np.float32,
            )
            snrs_flat = scoring.score_profiles(
                flat_ev,
                search_cfg.score_widths,
                mode=search_cfg.score_mode,
                n_harmonics=search_cfg.score_n_harmonics,
                biases=search_cfg.kadane_biases,
            )
            snrs = snrs_flat.reshape((*param_shape, 1))
            width_axis = np.array([0], dtype=np.int64)
    pgram = Periodogram(
        params={"width": width_axis, **dyp.param_arr_dict},
        snrs=snrs,
        tobs=tseries.tobs,
    )
    return dyp, pgram


@attrs.frozen(auto_attribs=True, kw_only=True)
class EPSearchResult:
    """Result container returned by ``ep_search``."""

    dyp: DynamicProgramming
    pruning: Pruning
    threshold_scheme: npt.NDArray[np.float32]
    ref_seg: int
    param_names: tuple[str, ...]
    best_params: npt.NDArray[np.float64]
    best_score: float
    best_fold: npt.NDArray[np.float32]
    pre_ascend_score: float
    t_ref: float = 0.0
    arc_slice: tuple[int, int] | None = None


@attrs.frozen(auto_attribs=True, kw_only=True)
class SnrRetentionResult:
    """S/N retention metrics relative to folding with the exact noiseless ephemeris."""

    snr_exact: float
    snr_mid: float
    snr_ascend: float
    snr_refold: float
    rho_mid: float
    rho_ascend: float
    rho_ts: float
    ep_result: EPSearchResult
    delta_f_over_df: float = float("nan")
    delta_a_over_da: float = float("nan")


def _build_default_threshold_scheme(
    cfg: PulsarSearchConfig,
    dyp: DynamicProgramming,
    *,
    poly_basis: str,
    use_moving_grid: bool,
    ref_ducy: float,
    snr_final: float,
    seed: int | None,
) -> npt.NDArray[np.float32]:
    if cfg.prune_poly_order == 5:
        kind = "circ_taylor_moving" if use_moving_grid else "circ_taylor_fixed"
    elif poly_basis == "chebyshev":
        kind = "poly_chebyshev_moving" if use_moving_grid else "poly_chebyshev_fixed"
    else:
        kind = "poly_taylor_moving" if use_moving_grid else "poly_taylor_fixed"
    bp = cfg.generate_branching_pattern(kind=kind)
    nstages = dyp.nsegments - 1
    bp = bp[:nstages]
    sp, _, _ = schedule_survive_probs(bp)
    info = determine_scheme(
        sp,
        bp,
        ref_ducy=ref_ducy,
        nbins=cfg.nbins,
        ntrials=1024,
        snr_final=snr_final,
        ducy_max=cfg.ducy_max,
        wtsp=cfg.wtsp,
        seed=seed,
    )
    return info.thresholds


def _run_ep_on_dyp(
    dyp_full: DynamicProgramming,
    search_cfg: PulsarSearchConfig,
    *,
    threshold_scheme: npt.NDArray[np.float32] | None = None,
    n_runs: int = 1,
    ref_segs: list[int] | None = None,
    ascend_levels: list[int] | None = None,
    poly_basis: str = "taylor",
    use_moving_grid: bool = True,
    max_sugg: int = 2**16,
    batch_size: int = 1024,
    arc_slice: tuple[int, int] | None = None,
    stop_snr: float | None = None,
    ref_ducy: float = 0.1,
    snr_final: float = 10.0,
    seed: int | None = None,
) -> EPSearchResult:
    """Execute EP on an already-computed P-FFA object (optionally sliced)."""
    dyp_run = dyp_full.slice_segments(*arc_slice) if arc_slice is not None else dyp_full

    if threshold_scheme is None:
        scheme = _build_default_threshold_scheme(
            search_cfg,
            dyp_run,
            poly_basis=poly_basis,
            use_moving_grid=use_moving_grid,
            ref_ducy=ref_ducy,
            snr_final=snr_final,
            seed=seed,
        )
    else:
        scheme = np.asarray(threshold_scheme, dtype=np.float32)

    segs = (
        determine_ref_segs_pareto(dyp_run.nsegments, n_runs)
        if ref_segs is None
        else list(ref_segs)
    )
    if not segs:
        msg = "ref_segs must not be empty"
        raise ValueError(msg)

    best_score = -np.inf
    best_pre_ascend = -np.inf
    best_params = np.empty(0, dtype=np.float64)
    best_param_names: tuple[str, ...] = ()
    best_fold = np.empty((2, search_cfg.nbins), dtype=np.float32)
    best_seg = segs[0]
    best_prn: Pruning | None = None
    t_arc_offset = float(arc_slice[0]) * dyp_full.tseg if arc_slice is not None else 0.0
    best_t_ref = t_arc_offset + 0.5 * dyp_run.nsegments * dyp_run.tseg
    all_names = ("crackle", "snap", "jerk", "accel", "freq")

    with tempfile.TemporaryDirectory(prefix="pyloki_ep_") as tmpdir:
        tmp_path = Path(tmpdir)
        for idx_run, seg in enumerate(segs):
            prn = Pruning(
                dyp_run,
                scheme,
                max_sugg=max_sugg,
                batch_size=batch_size,
                poly_basis=poly_basis,
                use_moving_grid=use_moving_grid,
            )
            prn.execute(
                seg,
                ascend_levels=ascend_levels,
                outdir=tmp_path,
                task_id=idx_run,
            )
            if prn.world_tree.valid_size == 0:
                if best_prn is None:
                    best_prn = prn
                continue
            scores_post = prn.world_tree.scores
            scores_pre = prn.world_tree.scores_ep
            best_idx = int(np.argmax(scores_post))
            post_score = float(scores_post[best_idx])
            pre_score = (
                float(scores_pre[best_idx])
                if scores_pre.size > best_idx
                else post_score
            )
            if post_score > best_score:
                coord_end = prn.scheme.get_previous_coord(
                    prn.prune_level + 1,
                    use_moving_grid,
                )
                coord_report = prn.scheme.get_report_coord(use_moving_grid)
                leaves_report = prn.prune_funcs.report(
                    prn.world_tree.leaves,
                    coord_report,
                    coord_end,
                )
                raw_fold = prn.world_tree.folds[best_idx]
                fold_t = (
                    np.fft.irfft(raw_fold, n=search_cfg.nbins).astype(np.float32)
                    if search_cfg.use_fourier
                    else np.asarray(raw_fold, dtype=np.float32).copy()
                )
                best_score = post_score
                best_pre_ascend = pre_score
                best_params = np.asarray(
                    leaves_report[best_idx, :-2, 0],
                    dtype=np.float64,
                ).copy()
                best_param_names = all_names[-len(best_params) :]
                best_fold = fold_t
                best_seg = seg
                best_prn = prn
                best_t_ref = t_arc_offset + float(coord_report[0])
            if stop_snr is not None and best_score >= stop_snr:
                break

    if best_prn is None:
        msg = "Pruning did not execute any segment"
        raise RuntimeError(msg)

    return EPSearchResult(
        dyp=dyp_run,
        pruning=best_prn,
        threshold_scheme=scheme,
        ref_seg=best_seg,
        param_names=best_param_names,
        best_params=best_params,
        best_score=best_score,
        best_fold=best_fold,
        pre_ascend_score=best_pre_ascend,
        t_ref=best_t_ref,
        arc_slice=arc_slice,
    )


def ep_search(
    tseries: TimeSeries,
    search_cfg: PulsarSearchConfig,
    *,
    threshold_scheme: npt.NDArray[np.float32] | None = None,
    n_runs: int = 1,
    ref_segs: list[int] | None = None,
    ascend_levels: list[int] | None = None,
    poly_basis: str = "taylor",
    use_moving_grid: bool = True,
    max_sugg: int = 2**16,
    batch_size: int = 1024,
    arc_slice: tuple[int, int] | None = None,
    stop_snr: float | None = None,
    ref_ducy: float = 0.1,
    snr_final: float = 10.0,
    seed: int | None = None,
    quiet: bool = False,
    show_progress: bool = False,
) -> EPSearchResult:
    """Run an end-to-end P-FFA + Extreme Pruning search in one call.

    Computes the base P-FFA fold tensor once, optionally slices to ``arc_slice``,
    builds an empirical threshold scheme if none is provided, and executes
    ``Pruning`` over ``ref_segs`` (with optional ``stop_snr`` early stop).
    """
    with quiet_logger(quiet=quiet):
        dyp = DynamicProgramming(tseries, search_cfg)
        dyp.initialize()
        dyp.execute(show_progress=show_progress)
        return _run_ep_on_dyp(
            dyp,
            search_cfg,
            threshold_scheme=threshold_scheme,
            n_runs=n_runs,
            ref_segs=ref_segs,
            ascend_levels=ascend_levels,
            poly_basis=poly_basis,
            use_moving_grid=use_moving_grid,
            max_sugg=max_sugg,
            batch_size=batch_size,
            arc_slice=arc_slice,
            stop_snr=stop_snr,
            ref_ducy=ref_ducy,
            snr_final=snr_final,
            seed=seed,
        )


def ep_sliding_arc_search(
    tseries: TimeSeries,
    search_cfg: PulsarSearchConfig,
    *,
    arc_len: int | None = None,
    stride: int | None = None,
    threshold_scheme: npt.NDArray[np.float32] | None = None,
    n_runs: int = 1,
    poly_basis: str = "taylor",
    use_moving_grid: bool = True,
    max_sugg: int = 2**16,
    batch_size: int = 1024,
    stop_snr: float | None = None,
    ref_ducy: float = 0.1,
    snr_final: float = 10.0,
    seed: int | None = None,
    quiet: bool = False,
    show_progress: bool = False,
) -> list[EPSearchResult]:
    """Run single-P-FFA sliding-arc EP across contiguous sub-arc windows.

    Computes the base P-FFA fold tensor **once**, derives sub-arc windows via
    ``sliding_arc_windows`` (optionally sizing ``arc_len`` from
    ``compute_max_coherent_subarc`` when ``p_orb_min > 0`` and ``x_max > 0``),
    and executes EP on shallow segment-sliced views sharing the same fold memory.
    """
    with quiet_logger(quiet=quiet):
        dyp = DynamicProgramming(tseries, search_cfg)
        dyp.initialize()
        dyp.execute(show_progress=show_progress)

        if arc_len is None:
            if search_cfg.p_orb_min > 0.0 and search_cfg.x_max > 0.0:
                t_max = compute_max_coherent_subarc(
                    search_cfg.prune_poly_order,
                    search_cfg.p_orb_min,
                    search_cfg.x_max,
                    1.0 / search_cfg.f_max,
                    search_cfg.nbins,
                    eta=search_cfg.eta,
                )
                eff_arc = min(dyp.nsegments, max(2, int(t_max // dyp.tseg)))
            else:
                eff_arc = dyp.nsegments
        else:
            eff_arc = int(arc_len)

        windows = sliding_arc_windows(dyp.nsegments, eff_arc, stride=stride)
        results: list[EPSearchResult] = []
        for idx, win in enumerate(windows):
            win_seed = None if seed is None else seed + idx
            res = _run_ep_on_dyp(
                dyp,
                search_cfg,
                threshold_scheme=threshold_scheme,
                n_runs=n_runs,
                poly_basis=poly_basis,
                use_moving_grid=use_moving_grid,
                max_sugg=max_sugg,
                batch_size=batch_size,
                arc_slice=win,
                stop_snr=stop_snr,
                ref_ducy=ref_ducy,
                snr_final=snr_final,
                seed=win_seed,
            )
            results.append(res)
            if stop_snr is not None and res.best_score >= stop_snr:
                break
        return results


def refold_candidates(
    tseries: TimeSeries,
    param_sets: npt.NDArray[np.float64],
    nbins: int,
    *,
    param_names: list[str] | tuple[str, ...],
    mod_tref: float | None = None,
    widths: npt.NDArray[np.int64] | None = None,
    nsubints: int = 1,
    score_mode: ScoreMode = "boxcar",
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Refold raw ``(ts_e, ts_v)`` along each candidate's continuous phase model.

    Parameters
    ----------
    tseries : TimeSeries
        Input time series.
    param_sets : NDArray[np.float64]
        Candidate parameter vectors of shape ``(ncands, nparams)`` matching
        ``param_names`` (canonical descending order ending in ``"freq"``).
    nbins : int
        Number of phase bins.
    param_names : sequence of str
        Names corresponding to columns of ``param_sets`` (e.g.
        ``["accel", "freq"]``, ``["jerk", "accel", "freq"]``, or
        ``["crackle", "snap", "jerk", "accel", "freq"]``).
    mod_tref : float | None, optional
        Reference epoch in seconds; defaults to ``tseries.tobs / 2``.
    widths : NDArray[np.int64] | None, optional
        Boxcar width trials; defaults to ``generate_box_width_trials(nbins)``.
    nsubints : int, optional
        Number of sub-integrations to fold, by default 1.
    score_mode : ScoreMode, optional
        Profile scoring mode, by default ``"boxcar"``.

    Returns
    -------
    tuple[NDArray[np.float32], NDArray[np.float32]]
        ``(snrs_refold, refold_profiles_ev)`` where ``refold_profiles_ev`` has
        shape ``(ncands, 2, nbins)`` (for ``nsubints == 1``) or
        ``(ncands, nsubints, 2, nbins)``.
    """
    params = np.atleast_2d(np.asarray(param_sets, dtype=np.float64))
    ncands, nparams = params.shape
    if nparams != len(param_names) or param_names[-1] != "freq":
        msg = (
            f"param_names ({param_names}) must match param_sets columns "
            "and end in 'freq'"
        )
        raise ValueError(msg)

    tref = tseries.tobs / 2.0 if mod_tref is None else float(mod_tref)
    w = (
        scoring.generate_box_width_trials(nbins)
        if widths is None
        else np.asarray(widths, dtype=np.int64)
    )
    if ncands == 0:
        empty_shape = (0, 2, nbins) if nsubints == 1 else (0, nsubints, 2, nbins)
        return np.empty(0, dtype=np.float32), np.empty(empty_shape, dtype=np.float32)

    deriv_map = {"accel": 2, "jerk": 3, "snap": 4, "crackle": 5}
    max_order = max((deriv_map.get(name, 0) for name in param_names[:-1]), default=0)

    profiles_list: list[np.ndarray] = []
    integrated_ev = np.empty((ncands, 2, nbins), dtype=np.float32)
    for icand in range(ncands):
        row = params[icand]
        freq = float(row[-1])
        if max_order <= 4:
            mod_kwargs = {"acc": 0.0, "jerk": 0.0, "snap": 0.0}
            for col_idx, name in enumerate(param_names[:-1]):
                if name == "accel":
                    mod_kwargs["acc"] = float(row[col_idx])
                elif name == "jerk":
                    mod_kwargs["jerk"] = float(row[col_idx])
                elif name == "snap":
                    mod_kwargs["snap"] = float(row[col_idx])
            mod_type = "derivative"
            kwargs: dict = mod_kwargs
        else:
            coeffs = np.zeros(max_order + 1, dtype=np.float64)
            for col_idx, name in enumerate(param_names[:-1]):
                order = deriv_map.get(name)
                if order is not None:
                    coeffs[order] = float(row[col_idx])
            mod_type = "derivative_series"
            kwargs = {"coeffs": coeffs}

        fold = tseries.fold_ephem(
            freq,
            nbins,
            nsubints=nsubints,
            normalize=False,
            mod_type=mod_type,
            mod_kwargs=kwargs,
            mod_tref=tref,
        )
        profiles_list.append(fold)
        integrated_ev[icand] = fold if nsubints == 1 else np.sum(fold, axis=0)

    snrs = scoring.score_profiles(integrated_ev, w, mode=score_mode)
    return snrs.astype(np.float32), np.asarray(profiles_list, dtype=np.float32)


def measure_snr_retention(
    tseries: TimeSeries,
    search_cfg: PulsarSearchConfig,
    *,
    true_freq: float,
    mod_type: str = "derivative",
    mod_kwargs: dict | None = None,
    mod_tref: float | None = None,
    threshold_scheme: npt.NDArray[np.float32] | None = None,
    n_runs: int = 1,
    ref_segs: list[int] | None = None,
    poly_basis: str = "taylor",
    use_moving_grid: bool = True,
    max_sugg: int = 2**16,
    score_mode: ScoreMode = "boxcar",
    seed: int | None = None,
) -> SnrRetentionResult:
    """Measure directed EP S/N retention relative to exact-ephemeris raw folding.

    Returns ``(rho_mid, rho_ascend, rho_ts)`` comparing pre-ascend EP S/N,
    post-ascend EP S/N, and raw time-series refold S/N against folding raw
    ``(ts_e, ts_v)`` along the exact noiseless ephemeris.
    """
    tref = tseries.tobs / 2.0 if mod_tref is None else float(mod_tref)
    exact_ev = tseries.fold_ephem(
        true_freq,
        search_cfg.nbins,
        nsubints=1,
        normalize=False,
        mod_type=mod_type,
        mod_kwargs=mod_kwargs,
        mod_tref=tref,
    )
    snr_exact = float(
        scoring.score_profiles(
            exact_ev[np.newaxis, :, :].astype(np.float32),
            search_cfg.score_widths,
            mode=score_mode,
        )[0],
    )

    nsegments_ffa = int(np.ceil(search_cfg.nsamps / search_cfg.bseg_ffa))
    final_level = nsegments_ffa - 1
    ep_res = ep_search(
        tseries,
        search_cfg,
        threshold_scheme=threshold_scheme,
        n_runs=n_runs,
        ref_segs=ref_segs,
        ascend_levels=[final_level],
        poly_basis=poly_basis,
        use_moving_grid=use_moving_grid,
        max_sugg=max_sugg,
        seed=seed,
        quiet=True,
    )

    if ep_res.best_params.size > 0:
        snrs_refold, _ = refold_candidates(
            tseries,
            ep_res.best_params[np.newaxis, :],
            search_cfg.nbins,
            param_names=ep_res.param_names,
            mod_tref=ep_res.t_ref,
            widths=search_cfg.score_widths,
            score_mode=score_mode,
        )
        snr_refold = float(snrs_refold[0])
    else:
        snr_refold = 0.0

    denom = snr_exact if snr_exact > 0.0 else float("nan")
    delta_f_over_df = float("nan")
    delta_a_over_da = float("nan")
    if ep_res.best_params.size > 0 and "freq" in ep_res.param_names:
        f_idx = ep_res.param_names.index("freq")
        df_full = float(search_cfg.eta) / (search_cfg.nbins * tseries.tobs)
        delta_f_over_df = abs(
            float(ep_res.best_params[f_idx]) - float(true_freq)
        ) / max(df_full, 1e-18)
        if "accel" in ep_res.param_names and mod_kwargs and "acc" in mod_kwargs:
            a_idx = ep_res.param_names.index("accel")
            da_full = (
                2.0
                * float(search_cfg.eta)
                * 299792458.0
                / (search_cfg.nbins * float(true_freq) * tseries.tobs**2)
            )
            delta_a_over_da = abs(
                float(ep_res.best_params[a_idx]) - float(mod_kwargs["acc"])
            ) / max(da_full, 1e-18)
    return SnrRetentionResult(
        snr_exact=snr_exact,
        snr_mid=ep_res.pre_ascend_score,
        snr_ascend=ep_res.best_score,
        snr_refold=snr_refold,
        rho_mid=ep_res.pre_ascend_score / denom,
        rho_ascend=ep_res.best_score / denom,
        rho_ts=snr_refold / denom,
        ep_result=ep_res,
        delta_f_over_df=delta_f_over_df,
        delta_a_over_da=delta_a_over_da,
    )


def decompose_fold_quality(
    tseries: TimeSeries,
    search_cfg: PulsarSearchConfig,
    *,
    true_freq: float,
    mod_type: str = "derivative",
    mod_kwargs: dict | None = None,
    mod_tref: float | None = None,
    oversample: int = 4,
) -> dict[str, float]:
    """Decompose S/N loss into finite-nbins binning loss vs FFA shift-rounding loss.

    Compares an ``oversample * nbins`` high-resolution ideal ephemeris fold,
    an ``nbins`` ideal ephemeris fold, and the full-observation FFA peak S/N.
    """
    tref = tseries.tobs / 2.0 if mod_tref is None else float(mod_tref)
    hi_bins = search_cfg.nbins * max(2, int(oversample))
    hi_widths = scoring.generate_box_width_trials(
        hi_bins,
        ducy_max=search_cfg.ducy_max,
        wtsp=search_cfg.wtsp,
    )
    fold_ideal_hi = tseries.fold_ephem(
        true_freq,
        hi_bins,
        nsubints=1,
        normalize=False,
        mod_type=mod_type,
        mod_kwargs=mod_kwargs,
        mod_tref=tref,
    )
    snr_ideal_hi = float(
        scoring.score_profiles(
            fold_ideal_hi[np.newaxis, :, :].astype(np.float32),
            hi_widths,
            mode=search_cfg.score_mode,
        )[0],
    )
    fold_ideal_nb = tseries.fold_ephem(
        true_freq,
        search_cfg.nbins,
        nsubints=1,
        normalize=False,
        mod_type=mod_type,
        mod_kwargs=mod_kwargs,
        mod_tref=tref,
    )
    snr_ideal_nb = float(
        scoring.score_profiles(
            fold_ideal_nb[np.newaxis, :, :].astype(np.float32),
            search_cfg.score_widths,
            mode=search_cfg.score_mode,
        )[0],
    )
    _, pgram = ffa_search(tseries, search_cfg, quiet=True)
    snr_ffa = float(np.max(pgram.snrs))
    return {
        "snr_ideal_oversampled": snr_ideal_hi,
        "snr_ideal_nbins": snr_ideal_nb,
        "snr_ffa": snr_ffa,
        "binning_retention": snr_ideal_nb / max(snr_ideal_hi, 1e-12),
        "ffa_retention_vs_nbins": snr_ffa / max(snr_ideal_nb, 1e-12),
        "total_retention": snr_ffa / max(snr_ideal_hi, 1e-12),
    }
