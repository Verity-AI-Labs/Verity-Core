"""Logging setup for verity-core and the tools built on it.

Every module logs through the ``verity_core`` logger namespace using the standard
library only, so a tool can raise or lower verbosity for the whole library from its own
entry point without verity-core imposing any handler of its own.

Messages are written as ``event key=value`` so that a 200-environment run stays
greppable: ``rg 'exec ' audit.log`` or ``rg 'env_id=swe-gym/task-42'`` pulls out one
environment's whole story without a log-parsing library.

Level conventions:

``DEBUG``
    Routine, high-volume operations: every command executed, every cache lookup.
``INFO``
    Lifecycle events worth seeing on a normal run: environment loaded, container
    started or stopped, scorecard written.
``WARNING``
    Recoverable problems that change how a result should be read: an unreadable cache
    entry, a partial-credit grader falling back to binary, a command hitting its
    timeout, a submission that would not apply.
``ERROR``
    Failures: the daemon is unreachable, a container will not start, a manifest is
    invalid, the model endpoint rejected the request.
"""

from __future__ import annotations

import logging
import sys
from typing import IO

NAMESPACE = "verity_core"
"""Root logger name for the whole library."""

DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

__all__ = [
    "DEFAULT_FORMAT",
    "NAMESPACE",
    "configure_logging",
    "get_logger",
]


def get_logger(name: str) -> logging.Logger:
    """Return the logger for a module inside the ``verity_core`` namespace."""
    return logging.getLogger(name)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    stream: IO[str] | None = None,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
    force: bool = False,
) -> logging.Logger:
    """Attach a stream handler to the ``verity_core`` logger and set its level.

    Meant for entry points: the CLI calls this, and a tool may call it instead. It is
    idempotent, so repeated calls will not stack duplicate handlers; pass ``force`` to
    replace handlers already attached.

    Record propagation is left alone rather than switched off. Turning it off would stop
    duplicate output when the root logger is also configured, but it would equally stop
    a tool from routing verity-core records into its own aggregation, and that is the
    more valuable of the two.
    """
    logger = logging.getLogger(NAMESPACE)
    logger.setLevel(level)

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    # NullHandler is not a StreamHandler, so the placeholder installed at import time
    # does not count as "already configured" here.
    if not any(isinstance(handler, logging.StreamHandler) for handler in logger.handlers):
        handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        logger.addHandler(handler)

    return logger
