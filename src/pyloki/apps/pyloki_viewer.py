from __future__ import annotations

import json
import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING

import attrs
import numpy as np
import rich_click as click
from matplotlib import pyplot as plt
from sigpyproc.viz.styles import set_seaborn

from pyloki import kepler
from pyloki.detection import scoring
from pyloki.detection.schemes import DynamicThresholdSchemeAnalyser, StatesInfo
from pyloki.io.cands import bin2d, extract_segment_waterfall
from pyloki.orbit_fit import FourierOrbitFit, KeplerianFitResult
from pyloki.utils.misc import get_logger

if TYPE_CHECKING:
    from matplotlib import figure as mpl_figure
    from numpy import typing as npt

logger = get_logger(__name__)


class ThresholdAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Dynamic Threshold Scheme Analyzer")
        # Set minimum window size before calculating initial geometry
        self.root.minsize(900, 600)

        # Configure grid weight to make it responsive
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=0)  # Control panel

        # Variables
        self.analyzer: DynamicThresholdSchemeAnalyser | None = None
        self.current_file: str | None = None
        self.best_path: StatesInfo | None = None
        self.min_prob: float | None = None
        self.best_prob_var = tk.StringVar()
        self.file_label_var = tk.StringVar(value="No file loaded")
        self._base_font_size: int = 12  # Store base font size for scaling

        # Apply base styles once
        self.setup_style()
        self.create_layout()
        self.create_figure()
        self.create_control_panel()
        self.toggle_controls(enabled=False)

        logger.info("ThresholdAnalyzerApp initialized")

    def setup_style(self) -> None:
        """Set up the style for the application."""
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TButton", font=("Helvetica", self._base_font_size))
        style.configure("TLabel", font=("Helvetica", self._base_font_size))
        style.configure("TFrame", background="#f0f0f0")
        style.configure("Control.TFrame", background="#e0e0e0", relief="raised")

        set_seaborn(font_size=self._base_font_size)

    def create_layout(self) -> None:
        """Create the main tabbed layout of the application."""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=5, pady=5)

        self.scheme_tab = ttk.Frame(self.notebook)
        self.scheme_tab.columnconfigure(0, weight=1)
        self.scheme_tab.rowconfigure(0, weight=1)
        self.scheme_tab.rowconfigure(1, weight=0)
        self.notebook.add(self.scheme_tab, text="Threshold Scheme")

        self.main_frame = ttk.Frame(self.scheme_tab)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        self.control_frame = ttk.Frame(self.scheme_tab, style="Control.TFrame")
        self.control_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        self.waterfall_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.waterfall_tab, text="Waterfall Lab")
        self.waterfall_app: WaterfallLabApp | None = None
        wf_bar = ttk.Frame(self.waterfall_tab)
        wf_bar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        ttk.Button(
            wf_bar,
            text="Load Waterfall (.npz)",
            command=self._browse_waterfall,
        ).pack(side=tk.LEFT, padx=5)
        self.waterfall_container = ttk.Frame(self.waterfall_tab)
        self.waterfall_container.pack(fill=tk.BOTH, expand=True)

    def _browse_waterfall(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Waterfall NPZ File",
            filetypes=[("NumPy archives", "*.npz"), ("All files", "*.*")],
        )
        if path:
            self.load_waterfall(path)

    def load_waterfall(
        self,
        waterfall_npz: str | Path,
        *,
        f0: float = 100.0,
        orbit_overlay: FourierOrbitFit | KeplerianFitResult | None = None,
    ) -> None:
        """Load a waterfall ``.npz`` file into the embedded Waterfall Lab tab."""
        for child in self.waterfall_container.winfo_children():
            child.destroy()
        data = np.load(waterfall_npz)
        state = WaterfallLabState(
            data["folds"],
            data["segment_times"],
            f0_ref=f0,
            orbit_overlay=orbit_overlay,
        )
        self.waterfall_app = WaterfallLabApp(self.waterfall_container, state)

    def create_figure(self) -> None:
        """Create the matplotlib figure and canvas."""
        from matplotlib.backends.backend_tkagg import (  # noqa: PLC0415
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )

        self.fig = plt.Figure(figsize=(12, 8), dpi=100, constrained_layout=True)
        self.axes = self.fig.subplots(2, 2)

        # Create canvas and toolbar
        self.canvas = FigureCanvasTkAgg(self.fig, self.main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")

        # Toolbar
        self.toolbar_frame = ttk.Frame(self.main_frame)
        self.toolbar_frame.grid(row=1, column=0, sticky="ew")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

    def create_control_panel(self) -> None:
        """Create the control panel with buttons and dropdown."""
        # File information and operations
        file_frame = ttk.Frame(self.control_frame)
        file_frame.pack(side=tk.LEFT, padx=10, pady=5)

        # File label
        file_label = ttk.Label(file_frame, textvariable=self.file_label_var)
        file_label.pack(side=tk.LEFT, padx=5)

        # Load button
        self.load_button = ttk.Button(
            file_frame,
            text="Load File",
            command=self.load_file,
        )
        self.load_button.pack(side=tk.LEFT, padx=5)

        # Unload button
        self.unload_button = ttk.Button(
            file_frame,
            text="Unload File",
            command=self.unload_file,
        )
        self.unload_button.pack(side=tk.LEFT, padx=5)

        # Separator
        separator = ttk.Separator(self.control_frame, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)

        # Best probability selection
        prob_frame = ttk.Frame(self.control_frame)
        prob_frame.pack(side=tk.LEFT, padx=10, pady=5)

        ttk.Label(prob_frame, text="Best Probability:").pack(side=tk.LEFT, padx=5)
        self.best_prob_combo = ttk.Combobox(
            prob_frame,
            textvariable=self.best_prob_var,
            state="readonly",
            width=10,
        )
        self.best_prob_combo.pack(side=tk.LEFT, padx=5)

        # Bind the combobox to update function
        self.best_prob_combo.bind("<<ComboboxSelected>>", self.on_best_prob_changed)

        # Separator
        separator = ttk.Separator(self.control_frame, orient=tk.VERTICAL)
        separator.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)

        # Action buttons
        action_frame = ttk.Frame(self.control_frame)
        action_frame.pack(side=tk.LEFT, padx=10, pady=5)

        # Print summary button
        self.print_button = ttk.Button(
            action_frame,
            text="Print Summary",
            command=self.print_summary,
        )
        self.print_button.pack(side=tk.LEFT, padx=5)

        # Save plot button
        self.save_plot_button = ttk.Button(
            action_frame,
            text="Save Plot",
            command=self.save_plot,
        )
        self.save_plot_button.pack(side=tk.LEFT, padx=5)

        # Save best path button
        self.save_path_button = ttk.Button(
            action_frame,
            text="Save Best Path",
            command=self.save_best_path,
        )
        self.save_path_button.pack(side=tk.LEFT, padx=5)

    def toggle_controls(self, *, enabled: bool = True) -> None:
        """Enable or disable controls based on whether a file is loaded."""
        state = "normal" if enabled else "disabled"
        self.unload_button["state"] = state
        self.best_prob_combo["state"] = "readonly" if enabled else "disabled"
        self.print_button["state"] = state
        self.save_plot_button["state"] = state
        self.save_path_button["state"] = state

    def load_file(self, filename: str | None = None) -> None:
        """Load an HDF5 file.

        Parameters
        ----------
        filename : str | None, optional
            Path to the file to load. If None, a file dialog will be shown.
        """
        if filename is None:
            filename = filedialog.askopenfilename(
                title="Select HDF5 File",
                filetypes=[("HDF5 Files", "*.h5 *.hdf5"), ("All Files", "*.*")],
            )
        if not filename:
            return
        try:
            analyzer = DynamicThresholdSchemeAnalyser.from_file(filename)
            logger.info(f"File loaded: {filename}")
        except Exception as e:
            logger.exception("Error loading file")
            messagebox.showerror("Error", f"Failed to load file: {e!s}")
            return

        if len(analyzer.probs) == 0:
            logger.error("No probabilities found in analyzer")
            messagebox.showerror("Error", "No probabilities found in analyzer")
            return

        min_prob = float(analyzer.probs[0])
        best_prob = float(analyzer.probs[0])
        try:
            _, best_path = analyzer.plot_paths(
                best_prob,
                min_prob=min_prob,
                fig=self.fig,
            )
        except Exception as e:
            logger.exception("Error plotting threshold paths")
            messagebox.showerror("Error", f"Failed to plot threshold paths: {e!s}")
            return

        self.analyzer = analyzer
        self.min_prob = min_prob
        self.best_path = best_path
        self.current_file = filename
        self.file_label_var.set(f"File: {Path(filename).name}")
        # Update window title
        self.root.title(f"Dynamic Threshold Scheme Analyzer - {Path(filename).name}")

        # Populate combobox
        self.best_prob_combo["values"] = [f"{prob:.4f}" for prob in self.analyzer.probs]
        self.best_prob_var.set(f"{self.analyzer.probs[0]:.4f}")
        self.best_prob_combo.current(0)

        # Enable controls
        self.toggle_controls(enabled=True)
        self.canvas.draw()

    def unload_file(self) -> None:
        """Unload the current file."""
        if self.analyzer is None:
            return

        self.analyzer = None
        self.current_file = None
        self.best_path = None
        self.file_label_var.set("No file loaded")

        # Reset window title
        self.root.title("Dynamic Threshold Scheme Analyzer")

        # Clear the axes
        for ax in self.axes.ravel():
            ax.clear()
        self.canvas.draw()

        # Disable controls
        self.toggle_controls(enabled=False)

        # Clear combobox
        self.best_prob_combo["values"] = []
        self.best_prob_var.set("")

        logger.info("File unloaded")

    def on_best_prob_changed(self, event: tk.Event | None = None) -> None:  # noqa: ARG002
        """Handle changes to the best probability selection."""
        if self.analyzer is None:
            return

        try:
            idx = self.best_prob_combo.current()
            if 0 <= idx < len(self.analyzer.probs):
                best_prob = float(self.analyzer.probs[idx])
            else:
                best_prob = float(self.best_prob_var.get())
            _, self.best_path = self.analyzer.update_best_path(best_prob, self.fig)
            logger.info(f"Best path updated with best_prob={best_prob}")

        except Exception as e:
            logger.exception("Error updating best path")
            messagebox.showerror("Error", f"Failed to update best path: {e!s}")

    def print_summary(self) -> None:
        """Print the summary of the best path to the console using rich table."""
        if self.best_path is None:
            return

        logger.info("Printing summary to console")
        self.best_path.print_summary()

    def save_plot(self) -> None:
        """Save the current plot to a file."""
        if self.analyzer is None:
            return

        filename = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("PDF Document", "*.pdf"),
                ("SVG Image", "*.svg"),
                ("All Files", "*.*"),
            ],
        )

        if not filename:
            return

        try:
            self.fig.savefig(filename, bbox_inches="tight", dpi=300)
            logger.info(f"Plot saved to {filename}")
            messagebox.showinfo("Success", f"Plot saved to {filename}")
        except Exception as e:
            logger.exception("Error saving plot")
            messagebox.showerror("Error", f"Failed to save plot: {e!s}")

    def save_best_path(self) -> None:
        """Save the best path to a JSON file."""
        if self.best_path is None:
            return

        filename = filedialog.asksaveasfilename(
            title="Save Best Path",
            defaultextension=".json",
            filetypes=[("JSON File", "*.json"), ("All Files", "*.*")],
        )

        if not filename:
            return

        try:
            self.best_path.save(filename)
            logger.info(f"Best path saved to {filename}")
            messagebox.showinfo("Success", f"Best path saved to {filename}")
        except Exception as e:
            logger.exception("Error saving best path")
            messagebox.showerror("Error", f"Failed to save best path: {e!s}")

    def run(
        self,
        default_file: str | None = None,
        display: tuple[int, int] = (1350, 900),
    ) -> None:
        """Run the application.

        Parameters
        ----------
        default_file : str | None, optional
            Path to a default scheme file to load on startup, by default None.
        display : tuple[int, int], optional
            Display size for the GUI, by default (1350, 900).
        """
        self.root.update()
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = max(display[0], 600)
        window_height = max(display[1], 400)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")

        # Load default file if provided
        if default_file is not None:
            self.load_file(filename=default_file)

        self.root.mainloop()


@attrs.define(auto_attribs=True, kw_only=True)
class WaterfallLabParams:
    """Interactive slider parameters for the waterfall refolding lab."""

    delta_f0: float = 0.0
    delta_f1: float = 0.0
    f2: float = 0.0
    p_orb: float = 0.0
    x_orb: float = 0.0
    ecc: float = 0.0
    aop: float = 0.0
    phi_orb: float = 0.0
    time_rebin: int = 1
    phase_rebin: int = 1


class WaterfallLabState:
    """Headless state and rendering engine for the interactive waterfall slider lab.

    Shifts ``(n_seg, 2, n_bins)`` (or ``(n_seg, n_bins)``) segment folds in real
    time under polynomial + Keplerian slider adjustments, rebins via S/N-preserving
    ``bin2d``, recomputes the integrated profile and GLRT S/N, and renders an
    optional ``FourierOrbitFit`` or ``KeplerianFitResult`` phase overlay.
    """

    def __init__(
        self,
        folds: npt.ArrayLike,
        segment_times: npt.ArrayLike,
        *,
        f0_ref: float = 100.0,
        orbit_overlay: FourierOrbitFit | KeplerianFitResult | None = None,
        params: WaterfallLabParams | None = None,
    ) -> None:
        arr = np.asarray(folds, dtype=np.float32)
        if arr.ndim == 2:
            ev = np.empty((arr.shape[0], 2, arr.shape[1]), dtype=np.float32)
            ev[:, 0, :] = arr
            ev[:, 1, :] = 1.0
            self.segment_folds = ev
        elif arr.ndim == 3 and arr.shape[1] == 2:
            self.segment_folds = arr.copy()
        else:
            msg = (
                "folds must have shape (n_seg, n_bins) or (n_seg, 2, n_bins), "
                f"got {arr.shape}"
            )
            raise ValueError(msg)

        self.segment_times = np.asarray(segment_times, dtype=np.float64)
        if self.segment_times.shape != (self.segment_folds.shape[0],):
            msg = "segment_times length must match folds.shape[0]"
            raise ValueError(msg)
        self.f0_ref = float(f0_ref)
        self.orbit_overlay = orbit_overlay
        self.params = params if params is not None else WaterfallLabParams()
        self._widths = scoring.generate_box_width_trials(self.nbins)

    @property
    def nsegments(self) -> int:
        return int(self.segment_folds.shape[0])

    @property
    def nbins(self) -> int:
        return int(self.segment_folds.shape[2])

    def phase_shifts_bins(self) -> npt.NDArray[np.int64]:
        """Compute integer bin roll shifts per segment from current slider params."""
        p = self.params
        t_c = self.segment_times - float(np.mean(self.segment_times))
        dphi = p.delta_f0 * t_c + 0.5 * p.delta_f1 * (t_c**2) + (p.f2 / 6.0) * (t_c**3)
        if p.p_orb > 0.0 and p.x_orb > 0.0:
            z_sec = kepler.keplerian_z(
                self.segment_times,
                p.p_orb,
                float(np.clip(p.ecc, 0.0, 0.95)),
                p.phi_orb,
                p.x_orb,
                p.aop,
                0.5 * math.pi,
            )
            dphi = dphi - self.f0_ref * (z_sec - float(z_sec[0]))
        return np.rint(dphi * self.nbins).astype(np.int64) % self.nbins

    def compute(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
        """Apply current shifts and return ``(rebinned_waterfall, profile, snr)``."""
        shifts = self.phase_shifts_bins()
        wf = extract_segment_waterfall(
            self.segment_folds,
            phase_shifts=shifts,
            widths=self._widths,
        )
        rebinned = bin2d(
            wf.amp_map,
            bt=max(1, int(self.params.time_rebin)),
            bp=max(1, int(self.params.phase_rebin)),
        )
        return rebinned, wf.integrated_profile, wf.best_snr

    def render(self, fig: mpl_figure.Figure) -> tuple[float, mpl_figure.Figure]:
        """Render the live 2D waterfall, integrated profile, and orbit overlay."""
        rebinned, profile, snr = self.compute()
        fig.clear()
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.5], hspace=0.25)
        ax_prof = fig.add_subplot(gs[0])
        ax_wf = fig.add_subplot(gs[1])

        phase = np.linspace(0.0, 1.0, profile.size, endpoint=False)
        ax_prof.step(phase, profile, where="mid", color="tab:blue")
        ax_prof.axhline(0.0, color="0.6", ls="--", lw=0.8)
        ax_prof.set_ylabel("Normalized S/N")
        ax_prof.set_title(f"Integrated Profile (GLRT S/N = {snr:.2f})")

        ax_wf.imshow(
            rebinned,
            aspect="auto",
            origin="lower",
            extent=(
                0.0,
                1.0,
                float(self.segment_times[0]),
                float(self.segment_times[-1]),
            ),
            cmap="viridis",
        )
        if self.orbit_overlay is not None:
            t_arr = self.segment_times
            if isinstance(self.orbit_overlay, FourierOrbitFit):
                om = float(self.orbit_overlay.omega_orb)
                x = float(self.orbit_overlay.x_orb)
                phi = float(self.orbit_overlay.phi_orb)
                delay_s = x * np.sin(om * t_arr + phi)
            else:
                delay_s = kepler.keplerian_z(
                    t_arr,
                    float(self.orbit_overlay.p_orb),
                    float(self.orbit_overlay.ecc),
                    float(self.orbit_overlay.phi),
                    float(self.orbit_overlay.x_orb),
                    float(self.orbit_overlay.aop),
                    0.5 * math.pi,
                )
            shifts_phase = self.phase_shifts_bins().astype(np.float64) / float(
                self.nbins,
            )
            overlay_phase = np.mod(
                0.5 + self.f0_ref * (delay_s - float(delay_s[0])) + shifts_phase,
                1.0,
            )
            # Break line segments across phase wraps to avoid horizontal jump artifacts
            jumps = np.flatnonzero(np.abs(np.diff(overlay_phase)) > 0.5) + 1
            if jumps.size > 0:
                plot_phase = np.insert(overlay_phase, jumps, np.nan)
                plot_t = np.insert(t_arr, jumps, np.nan)
            else:
                plot_phase = overlay_phase
                plot_t = t_arr
            ax_wf.plot(plot_phase, plot_t, color="tab:red", lw=1.2, ls="--")

        ax_wf.set_xlabel("Pulse Phase")
        ax_wf.set_ylabel("Time (s)")
        return snr, fig


def load_orbit_overlay(
    path: str | Path,
) -> FourierOrbitFit | KeplerianFitResult:
    """Load an orbit overlay from a JSON file for ``WaterfallLabState``.

    Accepts either a circular/Fourier JSON object containing
    ``{"omega_orb" | "p_orb", "x_orb", "phi_orb"}`` or a Keplerian JSON object
    containing ``{"p_orb", "x_orb", "ecc", "aop", "phi"}``.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        msg = f"Orbit overlay JSON must be an object, got {type(payload).__name__}"
        raise TypeError(msg)
    if "ecc" in payload and "aop" in payload:
        return KeplerianFitResult(
            f0=float(payload.get("f0", 0.0)),
            p_orb=float(payload["p_orb"]),
            x_orb=float(payload["x_orb"]),
            ecc=float(payload["ecc"]),
            aop=float(payload["aop"]),
            phi=float(payload["phi"]),
            loss=float(payload.get("loss", 0.0)),
            rms_residual=float(payload.get("rms_residual", 0.0)),
            residuals=np.empty(0, dtype=np.float64),
            success=bool(payload.get("success", True)),
        )
    if "omega_orb" in payload:
        om = float(payload["omega_orb"])
        p_orb = 2.0 * math.pi / om
    elif "p_orb" in payload:
        p_orb = float(payload["p_orb"])
        om = 2.0 * math.pi / p_orb
    else:
        msg = "Orbit overlay JSON must contain 'p_orb' or 'omega_orb'"
        raise ValueError(msg)
    x_orb = float(payload["x_orb"])
    phi_orb = float(payload.get("phi_orb", payload.get("phi", 0.0)))
    return FourierOrbitFit(
        p_orb=p_orb,
        omega_orb=om,
        x_orb=x_orb,
        phi_orb=phi_orb,
        ecc_approx=float(payload.get("ecc_approx", 0.0)),
        df_over_f=float(payload.get("df_over_f", 0.0)),
        rms_residual=float(payload.get("rms_residual", 0.0)),
        amplitudes=np.array([x_orb], dtype=np.float64),
        phases=np.array([phi_orb], dtype=np.float64),
        coeffs=np.zeros(4, dtype=np.float64),
        coeff_uncertainties=np.zeros(4, dtype=np.float64),
        residuals=np.empty(0, dtype=np.float64),
        omega_grid=np.array([om], dtype=np.float64),
        rms_curve=np.array([0.0], dtype=np.float64),
        t_center=float(payload.get("t_center", 0.0)),
    )


class WaterfallLabApp:
    """Tkinter interactive waterfall slider lab backed by ``WaterfallLabState``."""

    def __init__(self, root: tk.Misc, state: WaterfallLabState) -> None:
        from matplotlib.backends.backend_tkagg import (  # noqa: PLC0415
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )

        self.root = root
        self.state = state
        if isinstance(self.root, tk.Tk | tk.Toplevel):
            self.root.title("PyLoki Interactive Waterfall Lab")
            self.root.minsize(950, 650)

        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(9, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()

        slider_frame = ttk.Frame(main_frame)
        slider_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=8)

        self.vars: dict[str, tk.DoubleVar] = {}
        slider_specs = [
            ("delta_f0", -0.01, 0.01, 1e-5),
            ("delta_f1", -1e-6, 1e-6, 1e-9),
            ("f2", -1e-9, 1e-9, 1e-12),
            ("p_orb", 0.0, 20000.0, 10.0),
            ("x_orb", 0.0, 10.0, 0.01),
            ("ecc", 0.0, 0.9, 0.01),
            ("aop", 0.0, 2.0 * math.pi, 0.05),
            ("phi_orb", 0.0, 2.0 * math.pi, 0.05),
            ("time_rebin", 1.0, max(1.0, float(state.nsegments)), 1.0),
            ("phase_rebin", 1.0, max(1.0, float(state.nbins // 2)), 1.0),
        ]
        for name, lo, hi, res in slider_specs:
            ttk.Label(slider_frame, text=name).pack(anchor="w")
            var = tk.DoubleVar(value=float(getattr(state.params, name)))
            self.vars[name] = var
            scale = tk.Scale(
                slider_frame,
                from_=lo,
                to=hi,
                resolution=res,
                orient=tk.HORIZONTAL,
                variable=var,
                length=220,
                command=lambda _v: self.refresh(),
            )
            scale.pack(fill=tk.X, pady=2)
        self.refresh()

    def refresh(self) -> float:
        for name, var in self.vars.items():
            val = var.get()
            if name in ("time_rebin", "phase_rebin"):
                setattr(self.state.params, name, max(1, round(val)))
            else:
                setattr(self.state.params, name, float(val))
        snr, _ = self.state.render(self.fig)
        self.canvas.draw()
        return snr


@click.group(
    context_settings={"help_option_names": ["-h", "--help"], "show_default": True},
)
def main() -> None:
    pass


@main.command()
@click.argument("scheme_file", type=click.Path(exists=True))
@click.option(
    "-d",
    "--display",
    help="Display size for the GUI",
    type=(int, int),
    default=(1350, 900),
)
@click.option(
    "--waterfall",
    "waterfall_npz",
    type=click.Path(exists=True),
    default=None,
    help="Optional waterfall .npz file to pre-load in the Waterfall Lab tab",
)
@click.option("--f0", type=float, default=100.0, help="Reference spin frequency (Hz)")
@click.option(
    "--orbit-overlay",
    type=click.Path(exists=True),
    default=None,
    help="Optional orbit overlay JSON file for the Waterfall Lab tab",
)
def scheme(
    scheme_file: str,
    display: tuple[int, int],
    waterfall_npz: str | None,
    f0: float,
    orbit_overlay: str | None,
) -> None:
    root = tk.Tk()
    app = ThresholdAnalyzerApp(root)
    if waterfall_npz is not None:
        overlay = load_orbit_overlay(orbit_overlay) if orbit_overlay else None
        app.load_waterfall(waterfall_npz, f0=f0, orbit_overlay=overlay)
    app.run(default_file=scheme_file, display=display)


@main.command()
@click.argument("waterfall_npz", type=click.Path(exists=True))
@click.option("--f0", type=float, default=100.0, help="Reference spin frequency (Hz)")
@click.option(
    "--orbit-overlay",
    type=click.Path(exists=True),
    default=None,
    help="Optional orbit overlay JSON file (Fourier or Keplerian parameters)",
)
def waterfall(
    waterfall_npz: str,
    f0: float,
    orbit_overlay: str | None,
) -> None:
    data = np.load(waterfall_npz)
    overlay = load_orbit_overlay(orbit_overlay) if orbit_overlay else None
    state = WaterfallLabState(
        data["folds"],
        data["segment_times"],
        f0_ref=f0,
        orbit_overlay=overlay,
    )
    root = tk.Tk()
    WaterfallLabApp(root, state)
    root.mainloop()


if __name__ == "__main__":
    main()
