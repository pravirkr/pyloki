from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from astropy import constants
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    ProgressType,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

C_VAL = float(constants.c.value)


def get_indices(
    proper_time: np.ndarray,
    periods: float | list | np.ndarray,
    nbins: int,
) -> np.ndarray:
    """Calculate the indices of the folded time series.

    Parameters
    ----------
    proper_time : np.ndarray
        The proper time of the signal
    period : float | list | np.ndarray
        The period of the signal
    nbins : int
        The number of bins in the folded time series

    Returns
    -------
    np.ndarray
        The indices of the folded time series
    """
    if isinstance(periods, float | int):
        periods = [periods]
    periods = np.asarray(periods)
    factor = nbins / periods[:, np.newaxis]
    indices = np.round((proper_time % periods[:, np.newaxis]) * factor) % nbins
    return indices.astype(np.uint32).squeeze()

def get_logger(
    name: str,
    *,
    level: int | str = logging.INFO,
    quiet: bool = False,
    log_file: str | None = None,
) -> logging.Logger:
    """Get a fancy configured logger.

    Parameters
    ----------
    name : str
        logger name
    level : int or str, optional
        logging level, by default logging.INFO
    quiet : bool, optional
        if True set `level` as logging.WARNING, by default False
    log_file : str, optional
        path to log file, by default None

    Returns
    -------
    logging.Logger
        a logging object
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING if quiet else level)
    logformat = "- %(name)s - %(message)s"
    formatter = logging.Formatter(fmt=logformat)
    if not logger.hasHandlers():
        handler = RichHandler(
            show_path=False,
            rich_tracebacks=True,
            log_time_format="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class ScoreColumn(ProgressColumn):
    def __init__(self) -> None:
        super().__init__()
        self.score = 0.0

    def update_score(self, score: float) -> None:
        self.score = score

    def render(self, task: ProgressType) -> Text:  # noqa: ARG002
        return Text(f"Score: {self.score:.2f}", style="cyan")


class LeavesColumn(ProgressColumn):
    def __init__(self) -> None:
        super().__init__()
        self.leaves = 0.0

    def update_leaves(self, leaves: float) -> None:
        self.leaves = leaves

    def render(self, task: ProgressType) -> Text:  # noqa: ARG002
        return Text(f"Leaves: {self.leaves:.2f}", style="cyan")

def prune_track(
    sequence: Sequence[ProgressType] | Iterable[ProgressType],
    description: str = "Working...",
    total: float | None = None,
    get_score: Callable[[], float] | None = None,
    get_leaves: Callable[[], float] | None = None,
) -> Iterable[ProgressType]:
    columns: list[ProgressColumn] = (
        [TextColumn("[progress.description]{task.description}")] if description else []
    )

    score_column = ScoreColumn()
    leaves_column = LeavesColumn()
    columns.extend(
        (
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            TextColumn("•"),
            TimeRemainingColumn(elapsed_when_finished=True),
            TextColumn("•"),
            score_column,
            TextColumn("•"),
            leaves_column,
        ),
    )
    progress = Progress(*columns)
    task_id = progress.add_task(description, total=total)

    def track_progress() -> Iterable[ProgressType]:
        for item in sequence:
            if get_score:
                score_column.update_score(get_score())
            if get_leaves:
                leaves_column.update_leaves(get_leaves())
            progress.update(task_id, advance=1)
            yield item

    with progress:
        yield from track_progress()
