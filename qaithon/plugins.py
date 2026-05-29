"""Discover backends published by third-party packages via entry points.

Qaithon supports an ecosystem of third-party backends. The discovery
mechanism uses Python's standard ``entry_points`` API: a third-party
package registers its backend under the ``qaithon.backends`` group, and
:func:`discover` (called once at startup) walks the entry points and
calls each backend's registration callable.

Third-party packages declare their plugin in ``pyproject.toml``::

    [project.entry-points."qaithon.backends"]
    lightmatter = "lightmatter_qaithon:register"

…where ``lightmatter_qaithon.register`` is a zero-argument function that
calls :func:`qaithon.backends.register_backend` internally.

Plugins are loaded eagerly when :func:`discover` is invoked; failures in
one plugin do not affect the others — Qaithon logs and moves on.
"""

from __future__ import annotations

from dataclasses import dataclass

from qaithon._logging import get_logger

__all__ = ["DiscoveredPlugin", "discover", "list_plugins"]

logger = get_logger(__name__)

ENTRY_POINT_GROUP = "qaithon.backends"


@dataclass(frozen=True, slots=True)
class DiscoveredPlugin:
    """One plugin found via entry point discovery."""

    name: str
    distribution: str
    loaded: bool
    error: str | None = None


_DISCOVERED: list[DiscoveredPlugin] = []


def _entry_points():
    """Return the entry_points for ``qaithon.backends`` across Python versions."""
    from importlib.metadata import entry_points

    eps = entry_points()
    # importlib.metadata signature changed in 3.10. The modern API returns
    # an `EntryPoints` object supporting `.select(group=...)`.
    if hasattr(eps, "select"):
        return eps.select(group=ENTRY_POINT_GROUP)
    # Pre-3.10 fallback: dict keyed by group.
    return eps.get(ENTRY_POINT_GROUP, [])  # type: ignore[union-attr]


def discover() -> list[DiscoveredPlugin]:
    """Load every plugin registered under ``qaithon.backends``.

    Each entry point is expected to point at a callable that registers
    backend(s) into Qaithon's default registry. The callable runs once;
    its side effects are persisted in the process.

    Returns:
        List of :class:`DiscoveredPlugin` describing every entry point
        seen — both the ones that loaded successfully and the ones that
        failed.
    """
    _DISCOVERED.clear()
    for ep in _entry_points():
        name = ep.name
        distribution = (
            getattr(ep, "dist", None).name  # type: ignore[union-attr]
            if hasattr(ep, "dist") and ep.dist is not None
            else "unknown"
        )
        try:
            callable_obj = ep.load()
            callable_obj()
            _DISCOVERED.append(
                DiscoveredPlugin(name=name, distribution=distribution, loaded=True)
            )
            logger.info("Loaded plugin %r from distribution %r", name, distribution)
        except Exception as exc:  # noqa: BLE001
            _DISCOVERED.append(
                DiscoveredPlugin(
                    name=name,
                    distribution=distribution,
                    loaded=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            logger.warning(
                "Plugin %r from %r failed to load: %s", name, distribution, exc
            )
    return list(_DISCOVERED)


def list_plugins() -> list[DiscoveredPlugin]:
    """Return the plugins discovered in the last :func:`discover` call."""
    return list(_DISCOVERED)
