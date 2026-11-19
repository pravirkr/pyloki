"""Unit tests for `pyloki.io.timeseries`."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib import pyplot as plt

from pyloki.detection.scoring import boxcar_snr_1d
from pyloki.io import preprocess_kernels as pk
from pyloki.io.preprocess import (
    Birdie,
    ObservationMetadata,
    PreprocessOptions,
    apply_dm0_rfi_mask,
    generate_dm0_rfi_mask,
    preprocess_timeseries,
)
from pyloki.io.timeseries import TimeSeries
from pyloki.simulation.pulse import PulseSignalConfig
from tests.jit_utils import jit_variants

mpl.use("Agg")


class TestTimeSeriesDownsample:
    def test_scales_dt_and_divides_variance_preserving_snr(self) -> None:
        cfg = PulseSignalConfig(
            period=0.04,
            dt=1e-4,
            nsamps=2**15,
            snr=25.0,
            ducy=0.1,
            seed=42,
        )
        ts = cfg.generate(shape="gaussian")
        ts_ds = ts.downsample(4)

        assert ts_ds.dt == pytest.approx(4 * ts.dt)
        assert ts_ds.nsamps == ts.nsamps // 4
        assert ts_ds.tobs == pytest.approx(ts.tobs)
        np.testing.assert_allclose(ts_ds.ts_v, ts.ts_v[::4] / 4.0, rtol=1e-6)

        widths_orig = np.arange(1, 200)
        snr_orig = float(
            boxcar_snr_1d(
                ts.fold_ephem(cfg.freq, 400),
                widths_orig,
                1.0,
            ).max(),
        )
        widths_ds = np.arange(1, 50)
        snr_ds = float(
            boxcar_snr_1d(
                ts_ds.fold_ephem(cfg.freq, 100),
                widths_ds,
                1.0,
            ).max(),
        )
        assert abs(snr_ds - snr_orig) < 2.0

    def test_truncates_remainder_and_validates_factor(self) -> None:
        ts = TimeSeries(np.arange(11, dtype=np.float32), np.ones(11), 0.5)
        ds = ts.downsample(4)
        assert ds.nsamps == 2
        assert ds.dt == pytest.approx(2.0)
        np.testing.assert_allclose(ds.ts_e, [1.5, 5.5], rtol=1e-6)
        np.testing.assert_allclose(ds.ts_v, [0.25, 0.25], rtol=1e-6)

        with pytest.raises(ValueError, match="factor must be >= 1"):
            ts.downsample(0)
        with pytest.raises(ValueError, match="exceeds nsamps"):
            ts.downsample(12)


@pytest.mark.parametrize(("nsamps", "dt"), [(1006, 3e-3), (3018, 1e-3), (1000, 1e-3)])
def test_fold_ephem_time_grid_matches_nsamps(nsamps: int, dt: float) -> None:
    # np.arange(0, nsamps * dt, dt) can return nsamps + 1 points through float
    # rounding (e.g. nsamps=1006, dt=3e-3), which made brutefold reject the
    # series as a length mismatch.
    ts = TimeSeries(np.zeros(nsamps, np.float32), np.ones(nsamps, np.float32), dt)
    assert ts.fold_ephem(freq=10.0, nbins=16).shape == (16,)


def test_downsampled_series_folds() -> None:
    x = np.random.default_rng(0).standard_normal(3018).astype(np.float32)
    ds = TimeSeries(x, np.ones_like(x), 1e-3).downsample(3)
    assert ds.fold_ephem(freq=10.0, nbins=32).shape == (32,)


def test_fold_ephem_period_longer_than_tobs_message() -> None:
    ts = TimeSeries(np.zeros(100, np.float32), np.ones(100, np.float32), 0.1)
    with pytest.raises(ValueError, match="greater than the total data length"):
        ts.fold_ephem(freq=0.05, nbins=16)


def test_plot_fold_forwards_mod_type_and_kwargs_to_subints() -> None:
    cfg = PulseSignalConfig(
        period=0.01,
        dt=1e-4,
        nsamps=2**16,
        snr=40.0,
        ducy=0.08,
        mod_type="derivative",
        mod_kwargs={"acc": 1500.0},
        seed=7,
    )
    ts = cfg.generate(shape="gaussian", phi0=0.5)
    fig = ts.plot_fold(
        cfg.freq,
        nbins=64,
        nsubints=8,
        mod_type="derivative",
        mod_kwargs={"acc": 1500.0},
    )
    try:
        ax_subints = fig.axes[1]
        img = ax_subints.images[0].get_array()
        assert img is not None
        peak_bins = np.argmax(img, axis=1)
        # All 8 sub-integrations must stay phase-aligned around bin 32 (phi0=0.5).
        assert np.all(np.abs(peak_bins - 32) <= 2)
    finally:
        plt.close(fig)


class TestObservationMetadata:
    def test_valid_and_shifted(self) -> None:
        meta = ObservationMetadata(
            dm=0.0,
            start_mjd=60000.0,
            barycentric=True,
            source="J0737-3039A",
            input_path="test.dat",
            original_nsamps=10000,
        )
        shifted = meta.shifted(start_sample=8640, dt=0.1)
        assert shifted.start_mjd == pytest.approx(60000.0 + 864.0 / 86400.0)
        assert shifted.dm == 0.0
        assert shifted.source == "J0737-3039A"

        no_mjd = ObservationMetadata(dm=10.5)
        assert no_mjd.shifted(100, 1e-3).start_mjd is None

    def test_validation_rejects_bad_values(self) -> None:
        with pytest.raises(ValueError, match="DM"):
            ObservationMetadata(dm=-1.0)
        with pytest.raises(ValueError, match="DM"):
            ObservationMetadata(dm=float("nan"))
        with pytest.raises(ValueError, match="start_mjd"):
            ObservationMetadata(start_mjd=0.0)
        with pytest.raises(ValueError, match="original_nsamps"):
            ObservationMetadata(original_nsamps=-5)


class TestPreprocessTimeseries:
    def test_robust_white_noise_and_linear_drift(self) -> None:
        rng = np.random.default_rng(123)
        dt = 1e-3
        n = 8192
        t = np.arange(n) * dt
        raw = 50.0 + 2.5 * t + rng.standard_normal(n) * 3.0

        opts = PreprocessOptions(
            method="robust",
            filter_window=1.0,
            window_blocks=32,
            block_scales=(0.064, 0.256),
        )
        ts_e, ts_v, rep = preprocess_timeseries(raw, dt, opts)

        assert ts_e.dtype == np.float32
        assert ts_v.dtype == np.float32
        assert np.all(np.isfinite(ts_e))
        assert np.all(np.isfinite(ts_v))
        assert np.all(ts_v >= 0.0)
        assert np.all(ts_e[ts_v == 0.0] == 0.0)
        assert rep.method == "robust"
        assert rep.nsamps == n
        assert rep.global_scale == pytest.approx(3.0, rel=0.15)
        # Median nonzero ts_v is normalized to ~1.0
        assert float(np.median(ts_v[ts_v > 0.0])) == pytest.approx(1.0, rel=0.05)
        # Residual mean should be near 0 (linear trend removed)
        assert abs(float(np.mean(ts_e[ts_v > 0.0]))) < 0.1

    def test_robust_heteroscedastic_inverse_variance_weights(self) -> None:
        rng = np.random.default_rng(42)
        dt = 1e-3
        n = 16384
        raw = rng.standard_normal(n)
        raw[n // 2 :] *= 4.0  # 16x variance in second half

        opts = PreprocessOptions(
            method="robust",
            filter_window=1.0,
            window_blocks=16,
            block_scales=(),
            clip_sigma=0.0,
        )
        _ts_e, ts_v, _rep = preprocess_timeseries(raw, dt, opts)
        v_first = float(np.median(ts_v[: n // 4]))
        v_second = float(np.median(ts_v[3 * n // 4 :]))
        assert v_first / v_second == pytest.approx(16.0, rel=0.25)

    def test_robust_masks_rfi_burst_and_clips_spike(self) -> None:
        rng = np.random.default_rng(7)
        dt = 1e-3
        n = 8192
        raw = rng.standard_normal(n)
        # Inject a 128-sample broadband RFI burst and an isolated spike
        raw[2000:2128] += 15.0
        raw[5000] += 30.0

        opts = PreprocessOptions(
            method="robust",
            filter_window=2.0,
            window_blocks=32,
            block_scales=(0.064, 0.128),
            block_sigma=5.0,
            clip_sigma=6.0,
        )
        ts_e, ts_v, rep = preprocess_timeseries(raw, dt, opts)

        assert rep.n_masked > 0
        assert rep.n_clipped > 0
        assert rep.zero_weight_fraction > 0.0
        assert ts_e[5000] == 0.0
        assert ts_v[5000] == 0.0
        assert np.all(ts_v[2016:2112] == 0.0)
        assert np.all(ts_e[2016:2112] == 0.0)

    def test_robust_multiplicative_gain_and_negative_baseline_error(self) -> None:
        rng = np.random.default_rng(9)
        n = 4096
        dt = 1e-3
        raw_pos = 100.0 + rng.standard_normal(n)
        opts = PreprocessOptions(
            method="robust",
            gain_model="multiplicative",
            block_scales=(),
        )
        ts_e, ts_v, _ = preprocess_timeseries(raw_pos, dt, opts)
        assert np.all(np.isfinite(ts_e))
        assert np.all(ts_v >= 0.0)

        raw_zero_mean = rng.standard_normal(n)
        with pytest.raises(ValueError, match="positive baseline"):
            preprocess_timeseries(raw_zero_mean, dt, opts)

    @pytest.mark.parametrize("zap_mode", ["noise", "phase_kept", "zero"])
    def test_robust_periodic_and_birdie_zap(self, zap_mode: str) -> None:
        rng = np.random.default_rng(11)
        dt = 1e-3
        n = 4096
        t = np.arange(n) * dt
        # Strong 50 Hz mains birdie
        raw = rng.standard_normal(n) + 5.0 * np.sin(2.0 * np.pi * 50.0 * t)
        k50 = round(50.0 * n * dt)
        raw_p50 = float(np.abs(np.fft.rfft(raw)[k50]) ** 2)

        opts = PreprocessOptions(
            method="robust",
            filter_window=1.0,
            block_scales=(),
            clip_sigma=0.0,
            birdies=(Birdie(freq=50.0, width=1.0),),
            zap_mode=zap_mode,  # type: ignore[arg-type]
            zap_seed=42,
        )
        ts_e, _ts_v, rep = preprocess_timeseries(raw, dt, opts)
        assert rep.n_zapped >= 1
        clean_p50 = float(np.abs(np.fft.rfft(ts_e)[k50]) ** 2)
        assert clean_p50 < 0.05 * raw_p50

    def test_zscore_and_validation_errors(self) -> None:
        rng = np.random.default_rng(5)
        raw = rng.standard_normal(512) * 2.0 + 10.0
        for scale in ("iqr", "mad", "std"):
            opts = PreprocessOptions(
                method="zscore",
                filter_window=0.0,
                scale_method=scale,  # type: ignore[arg-type]
            )
            ts_e, ts_v, rep = preprocess_timeseries(raw, 1e-3, opts)
            assert rep.method == "zscore"
            np.testing.assert_array_equal(ts_v, np.ones(512, dtype=np.float32))
            assert float(np.std(ts_e)) == pytest.approx(1.0, rel=0.15)

        with pytest.raises(ValueError, match="constant"):
            preprocess_timeseries(np.ones(128), 1e-3)
        with pytest.raises(ValueError, match="non-finite"):
            preprocess_timeseries(np.array([1.0, np.nan] * 32), 1e-3)
        with pytest.raises(ValueError, match="at least 16"):
            preprocess_timeseries(np.arange(8, dtype=np.float64), 1e-3)
        with pytest.raises(ValueError, match="filter_window"):
            PreprocessOptions(filter_window=-1.0)


class TestDM0RfiMask:
    def test_generate_and_apply_preserves_zero_weight_invariant(self) -> None:
        rng = np.random.default_rng(101)
        dt = 1e-3
        n = 4096
        t = np.arange(n) * dt
        # DM=0 series with impulsive spike, pre-masked sample, and 60 Hz birdie
        dm0_e = rng.standard_normal(n) + 4.0 * np.sin(2.0 * np.pi * 60.0 * t)
        dm0_v = np.ones(n, dtype=np.float64)
        dm0_e[500] = 50.0
        dm0_e[600] = 0.0
        dm0_v[600] = 0.0

        rfi_mask = generate_dm0_rfi_mask(
            dm0_e,
            dm0_v,
            dt,
            t_sigma=6.0,
            f_cutoff_hz=10.0,
            f_sigma=8.0,
        )
        assert 500 in rfi_mask.mask_t
        assert 600 in rfi_mask.mask_t
        k60 = round(60.0 * n * dt)
        assert k60 in rfi_mask.mask_f

        target_e = rng.standard_normal(n) + 3.0 * np.sin(2.0 * np.pi * 60.0 * t)
        target_v = np.full(n, 2.0, dtype=np.float32)
        ts = TimeSeries(target_e, target_v, dt)
        cleaned = ts.apply_dm0_rfi_mask(rfi_mask, zap_mode="noise", seed=7)

        # Invariant: masked samples must have ts_e == 0 and ts_v == 0
        assert np.all(cleaned.ts_e[rfi_mask.mask_t] == 0.0)
        assert np.all(cleaned.ts_v[rfi_mask.mask_t] == 0.0)
        # 60 Hz birdie power is strongly suppressed
        raw_p60 = float(np.abs(np.fft.rfft(target_e)[k60]) ** 2)
        clean_p60 = float(np.abs(np.fft.rfft(cleaned.ts_e)[k60]) ** 2)
        assert clean_p60 < 0.05 * raw_p60

        # Length mismatch raises cleanly
        with pytest.raises(ValueError, match="does not match"):
            apply_dm0_rfi_mask(target_e[:100], target_v[:100], rfi_mask)


class TestTimeSeriesFromRawAndFromTim:
    def test_from_raw_attaches_metadata(self) -> None:
        rng = np.random.default_rng(3)
        raw = rng.standard_normal(1024)
        meta = ObservationMetadata(dm=12.5, start_mjd=60100.5, source="PSR_TEST")
        ts = TimeSeries.from_raw(raw, 1e-3, metadata=meta)
        assert ts.metadata == meta
        assert ts.downsample(2).metadata == meta

    def test_from_tim_subinterval_and_robust_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        rng = np.random.default_rng(19)
        data = rng.standard_normal(4096).astype(np.float32)

        class _FakeHeader:
            tsamp = 1e-3
            dm = 238.75
            tstart = 60000.0
            barycentric = True
            source = "Ter5A"
            nsamples = 4096

        class _FakeSigTS:
            def __init__(self, arr: np.ndarray, hdr: _FakeHeader) -> None:
                self.data = arr
                self.header = hdr

            def deredden(self, **_kwargs: object) -> _FakeSigTS:
                return self

            def normalise(self, **_kwargs: object) -> _FakeSigTS:
                return self

        monkeypatch.setattr(
            "pyloki.io.timeseries.sigTimeSeries",
            type(
                "_MockSigTS",
                (),
                {
                    "from_dat": staticmethod(
                        lambda _path: _FakeSigTS(data, _FakeHeader()),
                    ),
                    "__new__": lambda _cls, arr, hdr: _FakeSigTS(arr, hdr),
                },
            ),
        )

        ts_rob = TimeSeries.from_tim(
            "fake.dat",
            start_sample=512,
            nsamps=2048,
            preprocess_method="robust",
        )
        assert ts_rob.nsamps == 2048
        assert ts_rob.metadata is not None
        assert ts_rob.metadata.dm == pytest.approx(238.75)
        assert ts_rob.metadata.source == "Ter5A"
        assert ts_rob.metadata.start_mjd == pytest.approx(
            60000.0 + (512 * 1e-3) / 86400.0,
        )
        assert np.all(ts_rob.ts_v >= 0.0)

        ts_zs = TimeSeries.from_tim(
            "fake.dat",
            start_sample=100,
            nsamps=500,
            preprocess_method="zscore",
        )
        assert ts_zs.nsamps == 500
        np.testing.assert_array_equal(ts_zs.ts_v, np.ones(500, dtype=np.float32))


class TestPreprocessKernels:
    @pytest.mark.parametrize("fn", jit_variants(pk.fill_holes))
    def test_fill_holes(self, fn) -> None:
        arr = np.array([0.0, 2.0, 0.0, 0.0, 8.0, 0.0], dtype=np.float64)
        have = np.array([False, True, False, False, True, False], dtype=np.bool_)
        assert fn(arr, have) is True
        np.testing.assert_allclose(arr, [2.0, 2.0, 4.0, 6.0, 8.0, 8.0])
        assert fn(arr, np.zeros(6, dtype=np.bool_)) is False

    @pytest.mark.parametrize("fn", jit_variants(pk.reflected_running_mean))
    def test_reflected_running_mean_preserves_linear_trend(self, fn) -> None:
        x = np.linspace(1.0, 10.0, 19, dtype=np.float64)
        np.testing.assert_allclose(fn(x, 5), x, atol=1e-12)

    @pytest.mark.parametrize("fn", jit_variants(pk.masked_running_median))
    def test_masked_running_median(self, fn) -> None:
        x = np.array([1.0, 2.0, 100.0, 4.0, 5.0], dtype=np.float64)
        usable = np.array([True, True, False, True, True], dtype=np.bool_)
        out, ok = fn(x, usable, 3)
        assert ok is True
        np.testing.assert_allclose(out, [1.0, 1.5, 3.0, 4.5, 5.0])

    @pytest.mark.parametrize("fn", jit_variants(pk.masked_running_mean))
    def test_masked_running_mean(self, fn) -> None:
        x = np.array([2.0, 99.0, 6.0, 8.0], dtype=np.float64)
        usable = np.array([True, False, True, True], dtype=np.bool_)
        out, ok = fn(x, usable, 3)
        assert ok is True
        np.testing.assert_allclose(out, [2.0, 4.0, 7.0, 7.0])

    @pytest.mark.parametrize("fn", jit_variants(pk.interpolate_blocks))
    def test_interpolate_blocks(self, fn) -> None:
        vals = np.array([10.0, 20.0], dtype=np.float64)
        out = fn(8, 4, vals)
        assert out[0] == pytest.approx(10.0)
        assert out[-1] == pytest.approx(20.0)
        assert out[3] < out[4]

    @pytest.mark.parametrize("fn", jit_variants(pk.longest_zero_run))
    def test_longest_zero_run(self, fn) -> None:
        arr = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0], dtype=np.float32)
        assert fn(arr) == 3
