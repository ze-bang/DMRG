"""Rich-based logging setup used by the CLI and examples."""

from __future__ import annotations

import logging

from rich.logging import RichHandler

__all__ = ["setup_logging"]


def setup_logging(level: str | int = "INFO", *, force: bool = False) -> None:
    """Install a single Rich-based root handler.

    Idempotent: calling it twice is safe.  Pass ``force=True`` to wipe any
    pre-existing handlers (useful in notebooks).
    """
    root = logging.getLogger()
    if force:
        for h in list(root.handlers):
            root.removeHandler(h)
    if not root.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    root.setLevel(level)
