import numpy as np
from numba import typed
from pruning.periodogram import Periodogram
from pruning import prune, thresholds, ffa, scores
from pruning.base import SearchParams
from pruning.timeseries import SignalParams, generate_signal


def search_pulsar(ts_data, params, true_params):
    dyp = ffa.DynamicProgramming(ts_data, params)
    dyp.initialize()
    dyp.do_iterations()
    folds = dyp.fold[0][..., 0, :] / np.sqrt(dyp.fold[0][..., 1, :])

    true_period_idx, _ = dyp.dp_funcns.resolve(
        true_params, dyp.param_arr, dyp.ffa_level, 0
    )

    widths = scores.generate_width_trials(params.nbins, ducy_max=0.1, wtsp=1)
    snrs = scores.boxcar_snr(folds, widths)
    periods = dyp.param_arr[-1]
    best_indices = np.unravel_index(np.nanargmax(snrs), snrs.shape)

    print("\n*** Search results ***")
    print(f"True period index: {true_period_idx}")
    print(f"Search best index: {best_indices}")
    if len(best_indices) > 2:
        accels = dyp.param_arr[-2]
        print(f"Best acceleration = {accels[best_indices[-3]]}")
    print(f"Best period = {periods[best_indices[-2]]}")
    print(f"Best width = {widths[best_indices[-1]]}")
    print(f"Best S/N = {np.nanmax(snrs)}")

    if len(best_indices) > 2:
        snrs = snrs[best_indices[-3]]
    pgram = Periodogram(widths, periods, snrs, ts_data.tobs)
    return dyp, pgram


def pruning_search(dyp: ffa.DynamicProgramming, target_snr: float, snr_margin: float):
    bound_scheme = thresholds.threshold_scheme_bound(dyp.nchunks, target_snr, snr_margin)
    trials_scheme = thresholds.threshold_scheme_trials(dyp.nchunks, dyp.nparams)
    threshold_scheme = np.minimum(bound_scheme, trials_scheme)
    prn = prune.Pruning(dyp, threshold_scheme, ref_ind=12, max_sugg=131072)
    prn.initialize()
    return prn


def search_pulsar_pruning(ts_data, params, true_params):
    dyp = ffa.DynamicProgramming(ts_data, params)
    dyp.initialize()
    dyp.do_iterations("prune")
    folds = dyp.fold[0][..., 0, :] / np.sqrt(dyp.fold[0][..., 1, :])

    true_period_idx, _ = dyp.dp_funcns.resolve(
        true_params, dyp.param_arr, dyp.ffa_level, 0
    )

    widths = scores.generate_width_trials(params.nbins, ducy_max=0.1, wtsp=1)
    snrs = scores.boxcar_snr(folds, widths)
    periods = dyp.param_arr[-1]
    best_indices = np.unravel_index(np.nanargmax(snrs), snrs.shape)

    print("\n*** Search results ***")
    print(f"True period index: {true_period_idx}")
    print(f"Search best index: {best_indices}")
    if len(best_indices) > 2:
        accels = dyp.param_arr[-2]
        print(f"Best acceleration = {accels[best_indices[-3]]}")
    print(f"Best period = {periods[best_indices[-2]]}")
    print(f"Best width = {widths[best_indices[-1]]}")
    print(f"Best S/N = {np.nanmax(snrs)}")

    if len(best_indices) > 2:
        snrs = snrs[best_indices[-3]]
    pgram = Periodogram(widths, periods, snrs, ts_data.tobs)
    return dyp, pgram


def test_pulsar_search_periodicity(psr_type="msp", tol=None, fold_bins=None):
    if psr_type == "msp":
        period = 1.012345678910111213 * 1e-2  # (s)
        dt = 8.192e-5
    elif psr_type == "regular":
        period = 1.212345678910111213  # (s)
        dt = 8.192e-3
    signal_params = SignalParams(period, dt)
    psr_signal = generate_signal(signal_params)

    param_limits = typed.List([(signal_params.period * 0.5, signal_params.period * 1.6)])
    if tol is None:
        tol = signal_params.tol
    if fold_bins is None:
        fold_bins = signal_params.fold_bins
    print(f"Using tolerance: {tol}, fold_bins: {fold_bins}")
    search_params = SearchParams(
        signal_params.nsamps, signal_params.dt, tol, fold_bins, param_limits
    )
    true_params = np.array([signal_params.period])
    dyp, pgram = search_pulsar(psr_signal.ts_data, search_params, true_params)

    fig1 = psr_signal.plot(figsize=(10, 6.5))
    fig2 = pgram.plot(figsize=(10, 5))
    fig1.show()
    fig2.show()
    return dyp


def test_pulsar_search_accn(psr_type="msp", accel=100, tol=None, fold_bins=None):
    if psr_type == "msp":
        period = 1.012345678910111213 * 1e-2  # (s)
        dt = 8.192e-5
    elif psr_type == "regular":
        period = 1.212345678910111213  # (s)
        dt = 8.192e-2
    signal_params = SignalParams(period, dt, mod_kwargs={"acc": accel})
    psr_signal = generate_signal(signal_params)

    param_limits = typed.List(
        [
            (-3.1 * abs(accel), 3.1 * abs(accel)),
            (signal_params.period * 0.95, signal_params.period * 1.1),
        ]
    )
    if tol is None:
        tol = signal_params.tol
    if fold_bins is None:
        fold_bins = signal_params.fold_bins
    print(f"Using tolerance: {tol}, fold_bins: {fold_bins}")
    search_params = SearchParams(
        signal_params.nsamps,
        signal_params.dt,
        tol,
        fold_bins,
        param_limits,
    )
    true_params = np.array([accel, signal_params.period])
    dyp, pgram = search_pulsar(psr_signal.ts_data, search_params, true_params)

    fig1 = psr_signal.plot(figsize=(10, 6.5))
    fig2 = pgram.plot(figsize=(10, 5))
    fig1.show()
    fig2.show()
    return dyp
