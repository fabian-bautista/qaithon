"""Tests for plugin entry-point discovery."""

from __future__ import annotations

from qaithon import plugins


class TestDiscovery:
    def test_discover_returns_list(self):
        result = plugins.discover()
        assert isinstance(result, list)

    def test_list_plugins_returns_after_discover(self):
        plugins.discover()
        assert isinstance(plugins.list_plugins(), list)

    def test_entry_points_group_constant(self):
        assert plugins.ENTRY_POINT_GROUP == "qaithon.backends"
