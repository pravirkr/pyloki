from __future__ import annotations

import logging
import multiprocessing
import time
from contextlib import contextmanager
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import TYPE_CHECKING, Self, TypeVar

from astropy import constants
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.text import Text

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Sequence
    from multiprocessing.managers import DictProxy
    from queue import Queue
    from typing import Any

    from rich.progress import ProgressType, Task


C_VAL = float(constants.c.value)  # ty: ignore[unresolved-attribute]
T = TypeVar("T")
CONSOLE = Console()


def mkdir_p(dir_path: str | Path, must_not_exist: list[str] | None = None) -> Path:
    """Create a directory and all intermediate directories if they do not exist.

    Parameters
    ----------
    dir_path : str | Path
        Path to directory.
    must_not_exist : list[str], optional
        List of files that must not exist in the directory, by default None.
    """
    path = Path(dir_path)

    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        msg = f"Failed to create directory {path}: {e}"
        raise OSError(msg) from e

    if must_not_exist:
        existing = [f for f in must_not_exist if (path / f).exists()]
        if existing:
            msg = (
                f"Files already exist in directory {path}: {existing}\n"
                f"Provide a different directory or remove the files."
            )
            raise FileExistsError(msg)
    return path


def get_rich_handler(console: Console | None = None) -> logging.Handler:
    console = console or CONSOLE
    handler = RichHandler(
        console=console,
        show_path=False,
        rich_tracebacks=True,
        log_time_format="%Y-%m-%d %H:%M:%S",
        markup=True,
    )
    formatter = logging.Formatter(fmt="- %(name)s - %(message)s")
    handler.setFormatter(formatter)
    return handler


def setup_root_logging(
    package_name: str,
    *,
    console: Console | None = None,
    level: int | str = logging.INFO,
    quiet: bool = False,
    log_file: str | None = None,
) -> logging.Logger:
    """Set up logging for the entire package.

    This should be called once when the package is imported/initialized.

    Parameters
    ----------
    package_name : str
        Root package name.
    console : Console, optional
        Rich console instance
    level : int or str, optional
        Logging level, by default logging.INFO
    quiet : bool, optional
        If True set level to WARNING, by default False
    log_file : str, optional
        Path to log file, by default None

    Returns
    -------
    logging.Logger
        Root package logger
    """
    console = console or CONSOLE
    root_logger = logging.getLogger(package_name)
    root_logger.setLevel(logging.WARNING if quiet else level)

    # Clear any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    handler = get_rich_handler(console)
    root_logger.addHandler(handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(fmt="- %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    # Prevent propagation to root logger to avoid duplicate messages
    root_logger.propagate = False
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    This creates a child logger that inherits from the package root logger.
    Make sure setup_root_logging() has been called first.

    Parameters
    ----------
    name : str
        Module name (typically __name__)

    Returns
    -------
    logging.Logger
        Module logger that inherits from package root
    """
    return logging.getLogger(name)


def get_worker_logger(
    name: str,
    log_queue: Queue,
    level: int | str = logging.INFO,
) -> logging.Logger:
    """Get a logger for a worker process that sends logs through the queue.

    Parameters
    ----------
    name : str
        Logger name.
    level : int or str, optional
        logging level, by default logging.INFO

    Returns
    -------
    logging.Logger
        Logger configured to send logs to the main process
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    # Add queue handler to send logs to the main process
    handler = QueueHandler(log_queue)
    formatter = logging.Formatter(fmt="%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


@contextmanager
def quiet_logger(
    logger_name: str | None = None,
    *,
    quiet: bool = False,
    level: int = logging.ERROR,
) -> Generator[None, None, None]:
    """Context manager to temporarily change logging level.

    Parameters
    ----------
    logger_name : str, optional
        Logger name to quiet. If None, affects the root package logger.
        Use your package root name (e.g., 'pyloki') to quiet all loggers.
    quiet : bool, optional
        Whether to actually quiet the logger
    level : int, optional
        Level to set when quiet=True, by default logging.ERROR
    """
    if not quiet:
        yield
        return
    if logger_name is None:
        logger_name = "pyloki"

    target_logger = logging.getLogger(logger_name)
    original_level = target_logger.level

    target_logger.setLevel(level)
    try:
        yield
    finally:
        target_logger.setLevel(original_level)


class ScoreColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        score = task.fields.get("score", 0.0)
        return Text(f"Score: {score:.2f}", style="red")


class LeavesColumn(ProgressColumn):
    def render(self, task: Task) -> Text:
        leaves = task.fields.get("leaves", 0.0)
        return Text(f"Leaves: {leaves:.2f}", style="green")


def track_progress(
    sequence: Sequence[ProgressType] | Iterable[ProgressType],
    description: str = "Working...",
    total: float | None = None,
    get_score: Callable[[], float] | None = None,
    get_leaves: Callable[[], float] | None = None,
    console: Console | None = None,
    *,
    transient: bool = True,
    show_progress: bool = True,
) -> Iterable[ProgressType]:
    if not show_progress:
        yield from sequence
        return

    columns: list[ProgressColumn] = [
        TextColumn("[progress.description]{task.description}"),
        SpinnerColumn(),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        TextColumn("•"),
        TimeRemainingColumn(elapsed_when_finished=True),
    ]
    if get_score:
        columns.extend([TextColumn("•"), ScoreColumn()])
    if get_leaves:
        columns.extend([TextColumn("•"), LeavesColumn()])
    progress = Progress(*columns, console=console or CONSOLE, transient=transient)
    task_kwargs: dict[str, Any] = {"total": total}
    if get_score:
        task_kwargs["score"] = 0.0
    if get_leaves:
        task_kwargs["leaves"] = 0.0
    task_id = progress.add_task(f"[blue]{description}", **task_kwargs)

    def process_sequence() -> Iterable[ProgressType]:
        for item in sequence:
            updates: dict[str, Any] = {}
            if get_score:
                updates["score"] = get_score()
            if get_leaves:
                updates["leaves"] = get_leaves()
            if updates:
                progress.update(task_id, **updates)
            progress.update(task_id, advance=1)
            yield item

    with progress:
        yield from process_sequence()


def prune_track(
    sequence: Sequence[ProgressType],
    description: str = "Working...",
    shared_progress: DictProxy | None = None,
    task_id: int | None = None,
    total: float | None = None,
    get_score: Callable[[], float] | None = None,
    get_leaves: Callable[[], float] | None = None,
) -> Iterable[ProgressType]:
    """Track progress in both single-process and multi-process modes."""
    if shared_progress is None or task_id is None:
        yield from track_progress(
            sequence,
            description=description,
            total=total,
            get_score=get_score,
            get_leaves=get_leaves,
            console=CONSOLE,
        )
        return

    # In multi-process mode, use the shared progress manager
    def update_managed_dict(**kwargs) -> None:
        local_copy = dict(shared_progress[task_id])
        local_copy.update(kwargs)
        shared_progress[task_id] = local_copy  # reassign so the manager sees it

    total = total or len(sequence)
    update_managed_dict(total=total, visible=True)
    for idx, item in enumerate(sequence):
        updates: dict[str, int | float] = {"completed": idx + 1}
        if get_score:
            updates["score"] = get_score()
        if get_leaves:
            updates["leaves"] = get_leaves()
        update_managed_dict(**updates)
        yield item
    update_managed_dict(visible=False)


class MultiprocessProgressTracker:
    """A class to track progress across multiple processes using Rich.

    This class creates a shared state between processes and provides
    a single unified progress display for all worker processes.
    """

    def __init__(
        self,
        description: str = "Processing",
        console: Console | None = None,
    ) -> None:
        """Initialize the progress tracker.

        Parameters
        ----------
        description : str
            Description for the overall progress bar.
        """
        self.description = description
        self.console = console or CONSOLE

    def __enter__(self) -> Self:
        # Manager + shared dict + log queue
        self.manager = multiprocessing.Manager()
        self.shared_progress = self.manager.dict()
        self.log_queue = self.manager.Queue()
        # Create a handler that uses the same console as the tracker
        # Set up a listener for logs from worker processes
        self.log_listener = QueueListener(
            self.log_queue,
            get_rich_handler(self.console),
        )
        self.log_listener.start()

        # Single unified Progress instance
        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(),
            BarColumn(),
            TaskProgressColumn(show_speed=True),
            TextColumn("•"),
            TimeRemainingColumn(elapsed_when_finished=True),
            TextColumn("•"),
            ScoreColumn(),
            TextColumn("•"),
            LeavesColumn(),
            console=self.console,
            transient=True,
        )
        self.progress.start()
        self.overall_task_id = self.progress.add_task(
            f"[green]{self.description}",
            total=0,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        """Clean up manager resources."""
        self.log_listener.stop()
        self.progress.stop()
        self.manager.shutdown()

    def add_task(self, description: str, total: int) -> int:
        """Add a subtask to the overall progress.

        Parameters
        ----------
        description : str
            Description for the subtask.
        total : int
            Total number of steps for the subtask.

        Returns
        -------
        int
            Task ID for the subtask.
        """
        task_id = self.progress.add_task(
            f"[blue]{description}",
            total=total,
            visible=False,
            score=0.0,
            leaves=0.0,
        )
        self.shared_progress[task_id] = {
            "completed": 0,
            "total": total,
            "score": 0.0,
            "leaves": 0.0,
            "visible": False,
        }
        overall_total = self.progress.tasks[self.overall_task_id].total or 0
        self.progress.update(self.overall_task_id, total=overall_total + total)
        return task_id

    def update_progress(self) -> None:
        """Update the progress bar with the current state of the shared progress."""
        finished = 0
        for task_id, data in self.shared_progress.items():
            self.progress.update(
                task_id,
                completed=data["completed"],
                total=data["total"],
                visible=data["visible"],
                score=data["score"],
                leaves=data["leaves"],
            )
            finished += data["completed"]
        self.progress.update(self.overall_task_id, completed=finished)

    def collect_results(self, futures_to_seg: dict) -> tuple[list, list]:
        """Collect results from futures."""
        poll_interval = 0.25
        # Wait for all futures to complete
        while any(not f.done() for f in futures_to_seg):
            self.update_progress()
            time.sleep(poll_interval)
        # One final update
        self.update_progress()
        results = []
        errors = []
        for future, ref_seg in futures_to_seg.items():
            try:
                _ = future.result()
                results.append(ref_seg)
            except (RuntimeError, ValueError) as e:
                errors.append((ref_seg, str(e)))
        return results, errors


class PicklableStructRefWrapper[T]:
    """A generic wrapper to make Numba structref objects picklable.

    This class stores the constructor and arguments needed to create a structref object,
    recreating it on demand in a multiprocessing-safe way.

    Type Parameters:
        T: The type of the structref object being wrapped (e.g., FFASearchDPFuncts).
    """

    def __init__(self, constructor: Callable[..., T], *args, **kwargs) -> None:  # noqa: ANN002
        self._constructor = constructor
        self._args = args
        self._kwargs = kwargs
        self._instance: T | None = None

    def get_instance(self) -> T:
        """Get or create the wrapped structref instance."""
        if self._instance is None:
            self._instance = self._constructor(*self._args, **self._kwargs)
        return self._instance

    def __getattr__(self, name: str) -> T:
        """Delegate attribute access to the wrapped structref object."""
        if self._instance is None:
            self._instance = self._constructor(*self._args, **self._kwargs)
        return getattr(self._instance, name)

    def __reduce__(self) -> tuple[type, tuple, dict]:
        """Implement a custom reducer for pickling."""
        return (self.__class__, (self._constructor, *self._args), self._kwargs)
