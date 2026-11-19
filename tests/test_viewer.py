"""Headless unit tests for `pyloki.apps.pyloki_viewer.ThresholdAnalyzerApp`."""

from __future__ import annotations

from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
from click.testing import CliRunner
from matplotlib import pyplot as plt

from pyloki.apps import pyloki_viewer
from pyloki.apps.pyloki_viewer import (
    ThresholdAnalyzerApp,
    WaterfallLabParams,
    WaterfallLabState,
)
from pyloki.orbit_fit import fit_phase_curve_fourier

mpl.use("Agg")


class _VarStub:
    def __init__(self, value: str = "") -> None:
        self._value = value

    def get(self) -> str:
        return self._value

    def set(self, value: str) -> None:
        self._value = value


class _ComboStub:
    def __init__(self, var: _VarStub) -> None:
        self._var = var
        self._values: list[str] = []
        self._idx = -1
        self._state = "disabled"

    def __getitem__(self, key: str) -> object:
        if key == "values":
            return self._values
        return self._state

    def __setitem__(self, key: str, value: object) -> None:
        if key == "values":
            self._values = list(value)  # type: ignore[arg-type]
        elif key == "state":
            self._state = str(value)

    def current(self, index: int | None = None) -> int:
        if index is not None:
            self._idx = index
            if 0 <= index < len(self._values):
                self._var.set(self._values[index])
        return self._idx


class _CanvasStub:
    def __init__(self) -> None:
        self.draws = 0

    def draw(self) -> None:
        self.draws += 1


class _RootStub:
    def __init__(self) -> None:
        self.title_text = "Dynamic Threshold Scheme Analyzer"

    def title(self, text: str) -> None:
        self.title_text = text


def _make_headless_app() -> tuple[ThresholdAnalyzerApp, plt.Figure]:
    app = object.__new__(ThresholdAnalyzerApp)
    app.root = _RootStub()  # type: ignore[assignment]
    app.analyzer = None
    app.current_file = None
    app.best_path = None
    app.min_prob = None
    app.best_prob_var = _VarStub()  # type: ignore[assignment]
    app.file_label_var = _VarStub("No file loaded")  # type: ignore[assignment]
    fig, axes = plt.subplots(2, 2)
    app.fig = fig
    app.axes = axes
    app.canvas = _CanvasStub()  # type: ignore[assignment]
    app.unload_button = {"state": "disabled"}  # type: ignore[assignment]
    app.best_prob_combo = _ComboStub(app.best_prob_var)  # type: ignore[assignment]
    app.print_button = {"state": "disabled"}  # type: ignore[assignment]
    app.save_plot_button = {"state": "disabled"}  # type: ignore[assignment]
    app.save_path_button = {"state": "disabled"}  # type: ignore[assignment]
    return app, fig


class _AnalyserStub:
    def __init__(self, probs: np.ndarray, *, fail_plot: bool = False) -> None:
        self.probs = probs
        self.fail_plot = fail_plot
        self.seen_probs: list[float] = []

    def plot_paths(
        self, best_prob: float, min_prob: float, fig: plt.Figure
    ) -> tuple[plt.Figure, SimpleNamespace]:
        if self.fail_plot:
            msg = "plot failed"
            raise ValueError(msg)
        self.seen_probs.append(best_prob)
        return fig, SimpleNamespace(marker=min_prob)

    def update_best_path(
        self, best_prob: float, fig: plt.Figure
    ) -> tuple[plt.Figure, SimpleNamespace]:
        self.seen_probs.append(best_prob)
        # Simulate lookup failing if display rounding overshoots exact prob:
        if best_prob > float(np.max(self.probs)):
            msg = f"No best path found for probability {best_prob}"
            raise ValueError(msg)
        return fig, SimpleNamespace(prob=best_prob)


def test_load_file_error_and_empty_probs_preserve_state(monkeypatch) -> None:
    app, fig = _make_headless_app()
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        pyloki_viewer.messagebox,
        "showerror",
        lambda title, msg: errors.append((title, msg)),
    )

    good = _AnalyserStub(np.array([0.5, 0.89996], dtype=np.float32))
    monkeypatch.setattr(
        pyloki_viewer.DynamicThresholdSchemeAnalyser,
        "from_file",
        lambda _path: good,
    )
    app.load_file("good.h5")
    assert app.analyzer is good
    assert app.current_file == "good.h5"
    assert app.file_label_var.get() == "File: good.h5"
    prev_path = app.best_path

    # Failed reload must leave existing loaded state untouched.
    def _raise_load(_path: str) -> object:
        msg = "corrupt file"
        raise OSError(msg)

    monkeypatch.setattr(
        pyloki_viewer.DynamicThresholdSchemeAnalyser,
        "from_file",
        _raise_load,
    )
    app.load_file("bad.h5")
    assert len(errors) == 1
    assert app.analyzer is good
    assert app.current_file == "good.h5"
    assert app.file_label_var.get() == "File: good.h5"
    assert app.best_path is prev_path

    # Empty probs array must report an error and return cleanly.
    empty = _AnalyserStub(np.array([], dtype=np.float32))
    monkeypatch.setattr(
        pyloki_viewer.DynamicThresholdSchemeAnalyser,
        "from_file",
        lambda _path: empty,
    )
    app.load_file("empty.h5")
    assert len(errors) == 2
    assert app.analyzer is good

    # plot_paths failure must also preserve prior loaded state.
    bad_plot = _AnalyserStub(np.array([0.5], dtype=np.float32), fail_plot=True)
    monkeypatch.setattr(
        pyloki_viewer.DynamicThresholdSchemeAnalyser,
        "from_file",
        lambda _path: bad_plot,
    )
    app.load_file("bad_plot.h5")
    assert len(errors) == 3
    assert app.analyzer is good
    plt.close(fig)


def test_on_best_prob_changed_uses_exact_combo_index_value(monkeypatch) -> None:
    app, fig = _make_headless_app()
    errors: list[tuple[str, str]] = []
    monkeypatch.setattr(
        pyloki_viewer.messagebox,
        "showerror",
        lambda title, msg: errors.append((title, msg)),
    )

    # 0.89996 formats to "0.9000" (f"{p:.4f}") which would overshoot max(probs).
    probs = np.array([0.5, 0.89996], dtype=np.float32)
    analyser = _AnalyserStub(probs)
    monkeypatch.setattr(
        pyloki_viewer.DynamicThresholdSchemeAnalyser,
        "from_file",
        lambda _path: analyser,
    )
    app.load_file("exact.h5")
    assert app.best_prob_combo["values"] == ["0.5000", "0.9000"]

    app.best_prob_combo.current(1)
    app.on_best_prob_changed()
    assert not errors
    assert analyser.seen_probs[-1] == float(probs[1])
    plt.close(fig)


def test_cli_help_runs_without_tkagg() -> None:
    runner = CliRunner()
    res = runner.invoke(pyloki_viewer.main, ["--help"])
    assert res.exit_code == 0
    assert "scheme" in res.output


def test_waterfall_lab_state_shifts_and_renders() -> None:
    rng = np.random.default_rng(19)
    n_seg, n_bins = 16, 32
    t_seg = np.linspace(0.0, 1600.0, n_seg, dtype=np.float64)
    t_c = t_seg - float(np.mean(t_seg))
    df0_true = 2.0e-3  # drifts by 3.2 cycles across 1600 s

    folds = np.empty((n_seg, 2, n_bins), dtype=np.float32)
    folds[:, 0, :] = rng.normal(0.0, 1.0, size=(n_seg, n_bins))
    folds[:, 1, :] = 1.0
    for i in range(n_seg):
        drift_bin = round((-df0_true * t_c[i]) * n_bins) % n_bins
        folds[i, 0, (16 + drift_bin) % n_bins] += 5.0

    # Fit a dummy circular overlay for render verification
    om = 2.0 * np.pi / 3200.0
    delays = 0.5 * np.sin(om * t_seg + 0.3)
    overlay = fit_phase_curve_fourier(t_seg, delays, (2400.0, 4000.0))

    state = WaterfallLabState(folds, t_seg, f0_ref=100.0, orbit_overlay=overlay)
    _, _, snr_uncorrected = state.compute()

    state.params = WaterfallLabParams(
        delta_f0=df0_true,
        time_rebin=2,
        phase_rebin=2,
    )
    rebinned, profile, snr_corrected = state.compute()
    assert rebinned.shape == (n_seg // 2, n_bins // 2)
    assert profile.shape == (n_bins,)
    assert snr_corrected > snr_uncorrected + 5.0

    fig = plt.figure()
    rendered_snr, _ = state.render(fig)
    assert rendered_snr == snr_corrected
    assert len(fig.axes) == 2
    plt.close(fig)
