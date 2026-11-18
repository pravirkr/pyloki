"""Unit tests for `pyloki.io.timeseries`."""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib import pyplot as plt

from pyloki.detection.scoring import boxcar_snr_1d
from pyloki.io.timeseries import TimeSeries
from pyloki.simulation.pulse import PulseSignalConfig

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
