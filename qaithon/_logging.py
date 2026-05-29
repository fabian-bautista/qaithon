"""Logging setup for Qaithon.

Convention: every module gets its logger via ``get_logger(__name__)``.
Qaithon does NOT call ``logging.basicConfig`` — that's the application's
responsibility. By default no handler is attached, so Qaithon stays silent
unless the user opts in.

If users want to see Qaithon's logs they can do::

    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("qaithon").setLevel(logging.DEBUG)

or call :func:`enable_default_logging` for a one-liner during development.
"""

from __future__ import annotations

import logging

__all__ = ["enable_default_logging", "get_logger"]

# The root logger for the entire package. Every module-level logger is a child of this.
_ROOT_LOGGER_NAME = "qaithon"


def get_logger(name: str) -> logging.Logger:
    """Return a logger scoped under the ``qaithon`` namespace.

    Args:
        name: Usually the module's ``__name__``. The leading ``"qaithon."``
            is preserved so logging hierarchy keeps working.

    Returns:
        A standard ``logging.Logger`` instance.
    """
    return logging.getLogger(name)


def enable_default_logging(level: int = logging.INFO) -> None:
    """Attach a default stream handler to the ``qaithon`` logger.

    Convenience helper for development. **Do not call this from library code**
    — it's intended for notebook / script usage.

    Args:
        level: Logging level for the qaithon root logger. Defaults to ``INFO``.
    """
    root = logging.getLogger(_ROOT_LOGGER_NAME)
    if not root.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False
