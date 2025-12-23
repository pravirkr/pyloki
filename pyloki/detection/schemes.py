from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import TYPE_CHECKING, Self

import attrs
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.ticker import FormatStrFormatter
from rich.table import Table
from scipy import stats
from sigpyproc.viz.styles import set_seaborn

from pyloki.utils.misc import CONSOLE, get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from numpy import typing as npt


def bound_scheme(nstages: int, snr_bound: float) -> npt.NDArray[np.float32]:
    """Threshold scheme using the bound on the target S/N.

    This scheme is based on the assumption that the S/N thresholds (S/N^2) grow
    linearly with the stage index.

    Parameters
    ----------
    nstages : int
        Number of stages in the threshold scheme.
    snr_bound : float
        Upper bound on the target S/N.

    Returns
    -------
    NDArray[np.float32]
        Thresholds for each stage.
    """
    nsegments = nstages + 1
    thresh_sn2 = np.arange(1, nsegments + 1) * snr_bound**2 / nsegments
    thresh_sn2 = thresh_sn2.astype(np.float32)
    return np.sqrt(thresh_sn2[1:])


def trials_scheme(
    branching_pattern: npt.NDArray[np.float32],
    trials_start: int = 1,
) -> npt.NDArray[np.float32]:
    """Threshold scheme using the FAR of the tree.

    This scheme is based on the assumption that we are willing to accept a false alarm
    of 1 in :math:`N` trials at each stage.

    Parameters
    ----------
    branching_pattern : NDArray[np.float32]
        Branching pattern for each stage.
    trials_start : int
        Starting number of trials at stage 0, by default 1.

    Returns
    -------
    NDArray[np.float32]
        Thresholds for each stage.
    """
    trials = np.cumprod(branching_pattern) * trials_start
    return stats.norm.isf(1 / trials)


@attrs.frozen(auto_attribs=True, kw_only=True)
class StateInfo:
    """Class to save the state of the threshold scheme.

    Parameters
    ----------
    success_h0 : float, optional
        Success probability for H0 hypothesis, by default 1.
    success_h1 : float, optional
        Success probability for H1 hypothesis, by default 1.
    complexity : float, optional
        Number of options for H0 hypothesis, by default 1.
    complexity_cumul : float, optional
        Cumulative complexity/number of options for H0 hypothesis, by default 1.
    success_h1_cumul : float, optional
        Cumulative success/survival probability for H1 hypothesis, by default 1.
    nbranches : float, optional
        Number of branches for the current stage, by default 1.
    """

    success_h0: float
    success_h1: float
    complexity: float
    complexity_cumul: float
    success_h1_cumul: float
    nbranches: float
    threshold: float

    @property
    def cost(self) -> float:
        return self.complexity_cumul / self.success_h1_cumul

    @classmethod
    def from_record(cls, state: np.recarray) -> Self:
        field_names = set(attrs.fields_dict(cls).keys())
        dtype_names = state.dtype.names
        if dtype_names is None:
            msg = "State dtype has no field names"
            raise ValueError(msg)
        missing = field_names - set(dtype_names)
        if missing:
            msg = f"State dtype is missing required fields: {missing}"
            raise ValueError(msg)
        return cls(**{name: float(state[name].item()) for name in field_names})


@attrs.define(frozen=True)
class StatesInfo:
    """Class to handle the information of the states in the threshold scheme."""

    entries: list[StateInfo] = attrs.Factory(list)

    @property
    def thresholds(self) -> npt.NDArray[np.float32]:
        """Get list of thresholds for this scheme."""
        return np.array([entry.threshold for entry in self.entries], dtype=np.float32)

    def get_info(self, key: str) -> npt.NDArray[np.float32]:
        """Get list of values for a given key for all entries."""
        return np.array(
            [getattr(entry, key) for entry in self.entries],
            dtype=np.float32,
        )

    def print_summary(self) -> None:
        """Print a summary of the threshold scheme using rich table."""
        branching_pattern = self.get_info("nbranches")
        survive_prob = self.get_info("success_h1_cumul")[-1]
        pruning_complexity = np.log2(self.get_info("complexity_cumul")[-1])
        total_cost = np.log2(self.get_info("cost")[-1])
        mean_bp = 2 ** (np.mean(np.log2(branching_pattern)))
        n_options = np.sum(np.log2(branching_pattern))
        n_independent = len(self.thresholds) / self.thresholds[-1]
        total_survive_prob = 1 - (1 - survive_prob) ** n_independent

        # Format arrays for easy copy-pasting
        bp_str = np.array2string(
            branching_pattern,
            precision=1,
            floatmode="fixed",
            separator=",",
            threshold=np.iinfo(np.int32).max,
        )
        thresh_str = np.array2string(
            self.thresholds,
            precision=1,
            floatmode="fixed",
            separator=",",
            threshold=np.iinfo(np.int32).max,
        )
        # Print arrays first
        CONSOLE.print(f"[bold cyan]Branching Pattern:[/bold cyan] {bp_str}")
        CONSOLE.print(f"[bold cyan]Thresholds:[/bold cyan] {thresh_str}")
        CONSOLE.print()  # Add a blank line for separation

        # Create a rich table
        table = Table(
            title="Scheme Metrics Summary",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")

        # Add rows to the table
        table.add_row("Branching mean", f"{mean_bp:.2f}")
        table.add_row("Branching max", f"{np.max(branching_pattern):.2f}")
        table.add_row("Total enumerated options", f"{n_options:.2f}")
        table.add_row("Pruning complexity", f"{pruning_complexity:.2f}")
        table.add_row("Crude survival probability", f"{survive_prob:.2f}")
        table.add_row("Total cost", f"{total_cost:.2f}")
        table.add_row("Number of independent trials", f"{n_independent:.2f}")
        table.add_row("Total survival probability", f"{total_survive_prob:.2f}")
        CONSOLE.print(table)

    def save(self, filename: str) -> None:
        """Save the StatesInfo object to a file."""
        with Path(filename).open("w") as fp:
            json.dump(attrs.asdict(self), fp)

    @classmethod
    def load(cls, filename: str) -> StatesInfo:
        """Load a StatesInfo object from a file."""
        with Path(filename).open("r") as fp:
            data = json.load(fp)
            entries = [StateInfo(**entry) for entry in data["entries"]]
            return cls(entries=entries)


class DynamicThresholdSchemeAnalyser:
    def __init__(
        self,
        states: npt.NDArray,
        thresholds: npt.NDArray[np.float32],
        probs: npt.NDArray[np.float32],
        branching_pattern: npt.NDArray[np.float32],
        guess_path: npt.NDArray[np.float32],
        beam_width: float,
    ) -> None:
        self.states = states
        self.thresholds = thresholds
        self.probs = probs
        self.branching_pattern = branching_pattern
        self.guess_path = guess_path
        self.beam_width = beam_width
        self.nstages, self.nthresholds, self.nprobs = states.shape

    def backtrack_best(self, min_probs: list[float] | None) -> list[StatesInfo]:
        """Backtrack the best paths in the threshold scheme.

        Parameters
        ----------
        min_probs : list[float] | None
            Minimum success probability for the best path, by default None.

        Returns
        -------
        list[StatesInfo]
            List of the best paths as StatesInfo objects.
        """
        if min_probs is None:
            min_probs = [self.probs[0]]
        final_states = self.states[-1][self.states[-1]["is_empty"] == False]
        backtrack_states_info = []
        for min_prob in min_probs:
            filtered_states = final_states[final_states["success_h1_cumul"] >= min_prob]
            if len(filtered_states) == 0:
                logger.warning(f"No states found for min_prob {min_prob}")
                backtrack_states_info.append(StatesInfo([]))
                continue
            best_state = filtered_states[np.argmin(filtered_states["cost"])]
            backtrack_states = [StateInfo.from_record(best_state)]
            prev_threshold = best_state["threshold_prev"]
            prev_success_h1_cumul = best_state["success_h1_cumul_prev"]
            for istage in range(self.nstages - 2, -1, -1):
                ithres = np.argmin(np.abs(self.thresholds - prev_threshold))
                iprob = np.digitize(prev_success_h1_cumul, self.probs) - 1
                if iprob < 0:
                    msg = (
                        f"Backtracking failed at stage {istage} for threshold "
                        f"{prev_threshold} and success probability "
                        f"{prev_success_h1_cumul}"
                    )
                    raise ValueError(msg)
                prev_state = self.states[istage, ithres, iprob]
                if prev_state["is_empty"]:
                    msg = (
                        f"Backtracking failed at stage {istage} for threshold "
                        f"{prev_threshold} and success probability "
                        f"{prev_success_h1_cumul}"
                    )
                    raise ValueError(msg)
                backtrack_states.insert(0, StateInfo.from_record(prev_state))
                prev_threshold = prev_state["threshold_prev"]
                prev_success_h1_cumul = prev_state["success_h1_cumul_prev"]
            backtrack_states_info.append(StatesInfo(backtrack_states))
        return backtrack_states_info

    def backtrack_all(self, min_prob: float) -> list[StatesInfo]:
        """Backtrack all paths in the threshold scheme that meet the minimum success.

        Parameters
        ----------
        min_prob : float
            Minimum success probability for the best path.

        Returns
        -------
        list[StatesInfo]
            List of the best paths as StatesInfo objects.
        """
        final_states = self.states[-1][self.states[-1]["is_empty"] == False]
        filtered_states = final_states[final_states["success_h1_cumul"] >= min_prob]
        if len(filtered_states) == 0:
            return []
        backtrack_states_info = []
        for final_state in filtered_states:
            backtrack_states = [StateInfo.from_record(final_state)]
            prev_threshold = final_state["threshold_prev"]
            prev_success_h1_cumul = final_state["success_h1_cumul_prev"]
            for istage in range(self.nstages - 2, -1, -1):
                ithres = np.argmin(np.abs(self.thresholds - prev_threshold))
                iprob = np.digitize(prev_success_h1_cumul, self.probs) - 1
                if iprob < 0:
                    msg = (
                        f"Backtracking failed at stage {istage} for threshold "
                        f"{prev_threshold} and success probability "
                        f"{prev_success_h1_cumul}"
                    )
                    raise ValueError(msg)
                prev_state = self.states[istage, ithres, iprob]
                if prev_state["is_empty"]:
                    msg = (
                        f"Backtracking failed at stage {istage} for threshold "
                        f"{prev_threshold} and success probability "
                        f"{prev_success_h1_cumul}"
                    )
                    raise ValueError(msg)
                backtrack_states.insert(0, StateInfo.from_record(prev_state))
                prev_threshold = prev_state["threshold_prev"]
                prev_success_h1_cumul = prev_state["success_h1_cumul_prev"]
            backtrack_states_info.append(StatesInfo(backtrack_states))
        return backtrack_states_info

    def plot_paths(
        self,
        best_prob: float,
        min_prob: float,
        fig: plt.Figure | None = None,
    ) -> tuple[plt.Figure, StatesInfo]:
        paths = self.backtrack_all(min_prob)
        best_path = self.backtrack_best(min_probs=[best_prob])[0]
        if len(best_path.entries) == 0:
            msg = f"No best path found for probability {best_prob}"
            raise ValueError(msg)
        if fig is None:
            fig, axes = plt.subplots(
                2,
                2,
                figsize=(12, 8),
                dpi=120,
                constrained_layout=True,
            )
            ax1, ax2, ax3, ax4 = axes.flatten()
        else:
            axes = fig.get_axes()
            for ax in axes:
                ax.clear()
            ax1, ax2, ax3, ax4 = axes
        x = np.arange(1, self.nstages + 1)

        label = f"Best: P(H1) = {best_prob:.2f}"
        for path in paths:
            ax1.plot(x, path.thresholds**2, "b-", alpha=0.2)
        ax1.plot(x, best_path.thresholds**2, "r-", label=label)
        ax1.plot(x, self.guess_path**2, color="navy", ls="--", label="Guess path")
        upper_bound = np.minimum(self.guess_path + self.beam_width, self.thresholds[-1])
        lower_bound = np.maximum(self.guess_path - self.beam_width, 0)
        ax1.fill_between(
            x,
            lower_bound**2,
            upper_bound**2,
            color="cornflowerblue",
            alpha=0.2,
            label="Beam width",
        )
        ax1.set_xlabel("Pruning stage")
        ax1.set_ylabel("S/N squared")
        ax1.set_title("Threshold scheme")
        ax1.set_ylim(-0.5, self.thresholds[-1] ** 2 + 0.5)
        ax1.tick_params(axis="both", which="major")

        plot_configs = [
            (
                ax2,
                "complexity",
                "False Alarm",
                r"Number of $H_{0}$ options",
            ),
            (
                ax3,
                "complexity_cumul",
                r"Cumulative complexity $H_{0}$",
                r"Cumulative $H_{0}$ complexity",
            ),
            (
                ax4,
                "success_h1_cumul",
                r"Cumulative Success $H_{1}$",
                "Detection Probability",
            ),
        ]

        for ax, info, title, ylabel in plot_configs:
            for path in paths:
                ax.plot(x, path.get_info(info), "b-", alpha=0.2)
            ax.plot(x, best_path.get_info(info), "r-", label="Best path")
            ax.set_xlabel("Pruning stage")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.tick_params(axis="both", which="major")
            ax.set_yscale("log")

        ax2_current_ylim = ax2.get_ylim()
        ax2.plot(
            x,
            np.cumprod(self.branching_pattern),
            color="k",
            ls="--",
            label="Total options",
        )
        ax2.set_ylim(bottom=0.05, top=ax2_current_ylim[1])
        ax1.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax3.legend(loc="lower right")
        ax4.legend(loc="upper right")
        # Show grid in probability plots
        for pval in self.probs[1:]:
            ax4.axhline(y=pval, color="gray", alpha=0.15, linestyle="-", zorder=0)
        ax4.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        return fig, best_path

    def update_best_path(
        self,
        best_prob: float,
        fig: plt.Figure,
    ) -> tuple[plt.Figure, StatesInfo]:
        """Update only the best path in the plots without recomputing all paths."""
        best_path = self.backtrack_best(min_probs=[best_prob])[0]
        if len(best_path.entries) == 0:
            msg = f"No best path found for probability {best_prob}"
            raise ValueError(msg)

        x = np.arange(1, self.nstages + 1)
        axes = fig.get_axes()
        # For each axis, update only the best path line
        for idx, info in enumerate(
            [None, "complexity", "complexity_cumul", "success_h1_cumul"],
        ):
            ax = axes[idx]

            # Find and remove the previous best path line
            for line in ax.get_lines():
                label = line.get_label()
                if label == "Best path" or (
                    isinstance(label, str)
                    and label.startswith(
                        "Best: P(H1)",
                    )
                ):
                    line.remove()

            # Add the new best path line
            if info is None:  # Threshold scheme plot
                label = f"Best: P(H1) = {best_prob:.2f}"
                ax.plot(x, best_path.thresholds**2, "r-", label=label)
            else:
                ax.plot(x, best_path.get_info(info), "r-", label="Best path")

            # Update the legend
            ax.legend(loc="upper right")
        fig.canvas.draw_idle()
        return fig, best_path

    def plot_slice(
        self,
        stage: int,
        attribute: str = "success_h1_cumul",
        fmt: str = ".3f",
        cmap: str = "viridis",
        figsize: tuple[float, float] = (12, 8),
        annot_size: float = 8,
    ) -> plt.Figure:
        if not 0 <= stage < self.nstages:
            msg = f"Stage must be between 0 and {self.nstages - 1}, got {stage}"
            raise ValueError(msg)
        if self.states.dtype.names is None or attribute not in self.states.dtype.names:
            msg = f"Attribute must be one of {self.states.dtype.names}, got {attribute}"
            raise ValueError(msg)

        # Extract 2D slice directly from structured array
        cum_score = self.states[stage, :, :][attribute].astype(float)
        mask = self.states[stage, :, :]["is_empty"] == True
        cum_score[mask] = np.nan

        df = pd.DataFrame(
            cum_score,
            index=pd.Index(self.thresholds, name="Thresholds", dtype=np.float32),
            columns=pd.Index(range(self.nprobs), name="Success Prob bins"),
        )
        fig, ax = plt.subplots(figsize=figsize)
        heatmap = sns.heatmap(
            df,
            annot=True,
            annot_kws={"size": annot_size},
            fmt=fmt,
            cmap=cmap,
            linewidth=0.5,
            linecolor="gray",
            cbar_kws={
                "shrink": 0.8,
                "pad": 0.02,
            },
            ax=ax,
        )
        ax.set_title(f"Stage {stage}: {attribute.replace('_', ' ').title()}")
        ax.tick_params(axis="both", which="major")
        cbar = heatmap.collections[0].colorbar
        cbar.ax.set_ylabel(attribute.replace("_", " ").title())
        cbar.ax.tick_params()

        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    def save(self, filename: str) -> None:
        """Save the StatesInfo object to a file."""
        np.savez(
            filename,
            thresholds=self.thresholds,
            probs=self.probs,
            states=self.states,
            allow_pickle=True,
        )

    @classmethod
    def from_file(cls, filename: str) -> DynamicThresholdSchemeAnalyser:
        """Load a DynamicThresholdScheme object from an hdf5 file."""
        with h5py.File(filename, "r") as f:
            branching_pattern = f["branching_pattern"][:]
            thresholds = f["thresholds"][:]
            probs = f["probs"][:]
            guess_path = f["guess_path"][:]
            beam_width = f.attrs["beam_width"]
            states = f["states"][:]
        return cls(
            states,
            thresholds,
            probs,
            branching_pattern,
            guess_path,
            beam_width,
        )


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
        """Create the main layout of the application."""
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # Configure main frame grid
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(0, weight=1)

        # Control panel frame
        self.control_frame = ttk.Frame(self.root, style="Control.TFrame")
        self.control_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

    def create_figure(self) -> None:
        """Create the matplotlib figure and canvas."""
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
            self.analyzer = DynamicThresholdSchemeAnalyser.from_file(filename)
            logger.info(f"File loaded: {filename}")
        except Exception as e:
            logger.exception("Error loading file")
            messagebox.showerror("Error", f"Failed to load file: {e!s}")

        if self.analyzer is None:
            return
        self.current_file = filename
        self.file_label_var.set(f"File: {Path(filename).name}")
        # Update window title
        self.root.title(f"Dynamic Threshold Scheme Analyzer - {Path(filename).name}")
        # Set initial min_prob to the smallest available probability
        self.min_prob = self.analyzer.probs[0]
        if self.min_prob is None:
            logger.error("No probabilities found in analyzer")
            messagebox.showerror("Error", "No probabilities found in analyzer")
            return

        # Populate combobox
        self.best_prob_combo["values"] = [f"{prob:.4f}" for prob in self.analyzer.probs]
        self.best_prob_var.set(f"{self.analyzer.probs[0]:.4f}")

        # Enable controls
        self.toggle_controls(enabled=True)

        # Generate initial plots
        best_prob = self.analyzer.probs[0]
        _, self.best_path = self.analyzer.plot_paths(
            best_prob,
            min_prob=self.min_prob,
            fig=self.fig,
        )
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
            # Get the selected probability
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
