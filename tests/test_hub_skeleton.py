"""Tests that the Hub namespace is reachable.

This file used to assert NotImplementedError for every Hub call. Now that
the loader is wired to ``huggingface_hub`` for real, the contract is:
calls that hit the network surface either real results (when valid) or
``HubError`` (when something goes wrong). The dedicated round-trip tests
live in ``test_hub_real.py``.
"""

from __future__ import annotations

import qaithon
from qaithon.hub.loader import HubError, HubNotImplementedError


class TestNamespace:
    def test_qaithon_hub_is_importable(self):
        assert hasattr(qaithon, "hub")
        assert hasattr(qaithon.hub, "load_block")
        assert hasattr(qaithon.hub, "push_block")
        assert hasattr(qaithon.hub, "list_blocks")

    def test_legacy_error_class_still_compatible(self):
        # Old callers may catch HubNotImplementedError; keep the type as a
        # subclass of NotImplementedError so legacy ``except`` clauses match.
        assert issubclass(HubNotImplementedError, NotImplementedError)
        assert issubclass(HubNotImplementedError, HubError)
