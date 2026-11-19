"""Unit tests for `pyloki.periodogram`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pytest
from matplotlib import pyplot as plt

from pyloki.periodogram import Periodogram, PruningStatsPlotter, ScatteredPeriodogram

if TYPE_CHECKING:
    from pathlib import Path

mpl.use("Agg")


class TestPeriodogram:
    def _make_pgram(self) -> tuple[Periodogram, np.ndarray]:
        accel = np.array([-2.0, 0.0, 2.0])
        freq = np.array([99.0, 100.0, 101.0, 102.0])
        width = np.array([1, 2])
        # Canonical dims order: ("accel", "freq", "width") -> shape (3, 4, 2)
        rng = np.random.default_rng(12)
        snrs = rng.uniform(1.0, 10.0, size=(3, 4, 2))
        pgram = Periodogram(
            params={"accel": accel, "freq": freq, "width": width},
            snrs=snrs,
            tobs=100.0,
        )
        return pgram, snrs

    def test_plot_2d_orientation_for_both_axis_orders(self) -> None:
        pgram, snrs = self._make_pgram()
        expected = snrs.max(axis=2)  # shape (3, 4) for (accel, freq)

        fig_af = pgram.plot_2d("accel", "freq")
        try:
            z_af = fig_af.axes[0].images[0].get_array()
            assert z_af is not None
            assert z_af.shape == (4, 3)  # rows=freq (y), cols=accel (x)
            np.testing.assert_allclose(z_af, expected.T)
        finally:
            plt.close(fig_af)

        fig_fa = pgram.plot_2d("freq", "accel")
        try:
            z_fa = fig_fa.axes[0].images[0].get_array()
            assert z_fa is not None
            assert z_fa.shape == (3, 4)  # rows=accel (y), cols=freq (x)
            np.testing.assert_allclose(z_fa, expected)
        finally:
            plt.close(fig_fa)

    def test_plot_2d_rejects_same_or_fixed_axes(self) -> None:
        pgram, _ = self._make_pgram()
        with pytest.raises(ValueError, match="Invalid 2D parameter selection"):
            pgram.plot_2d("accel", "accel")
        with pytest.raises(ValueError, match="Invalid 2D parameter selection"):
            pgram.plot_2d("accel", "freq", fixed_param_idx={"accel": 0})

    def test_save_and_load_round_trip_and_closes_handle(self, tmp_path: Path) -> None:
        pgram, snrs = self._make_pgram()
        path = tmp_path / "pgram.nc"
        pgram.save(str(path))
        loaded = Periodogram.load(str(path))
        path.unlink()
        assert not path.exists()
        assert loaded.tobs == pytest.approx(100.0)
        np.testing.assert_allclose(loaded.snrs, snrs)


class TestPruningStatsPlotter:
    def test_empty_and_missing_run_id_plot_without_legend_error(self) -> None:
        plotter = PruningStatsPlotter()
        fig_empty = plotter.plot_level_stats()
        try:
            assert len(fig_empty.axes) == 1
        finally:
            plt.close(fig_empty)

        ls = np.zeros(
            2,
            dtype=np.dtype([("level", np.int32), ("n_leaves_surv", np.int32)]),
        )
        ls["level"] = [1, 2]
        ls["n_leaves_surv"] = [10, 5]
        plotter.add_run(ls, "000_00")

        fig_missing = plotter.plot_level_stats(run_id="999_00")
        try:
            assert len(fig_missing.axes) == 1
        finally:
            plt.close(fig_missing)

        fig_valid = plotter.plot_level_stats(run_id="000_00")
        try:
            assert len(fig_valid.axes[0].get_lines()) >= 1
        finally:
            plt.close(fig_valid)


def test_scattered_periodogram_plot_scores_single_run_histogram() -> None:
    sp = ScatteredPeriodogram(param_names=["accel", "freq"])
    param_sets = np.zeros((4, 4, 2))
    sp.add_run(
        param_sets,
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([5.0, 6.0, 7.0, 8.0]),
        "000_00",
    )
    fig = sp.plot_scores(kind="hist")
    try:
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)
