from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import rich_click as click
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sigpyproc.viz.styles import set_seaborn

from pyloki.detection.schemes import DynamicThresholdSchemeAnalyser, StatesInfo
from pyloki.utils.misc import get_logger

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
def scheme(scheme_file: str, display: tuple[int, int]) -> None:
    root = tk.Tk()
    app = ThresholdAnalyzerApp(root)
    app.run(default_file=scheme_file, display=display)


if __name__ == "__main__":
    main()
