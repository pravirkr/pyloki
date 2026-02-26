from __future__ import annotations

import tkinter as tk

import rich_click as click

from pyloki.detection.schemes import ThresholdAnalyzerApp
from pyloki.utils.misc import get_logger

logger = get_logger(__name__)


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
