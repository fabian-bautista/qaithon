"""Tests for the persistent disk cache."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from qaithon.cache import DiskMatmulCache

_HAS_SAFETENSORS = importlib.util.find_spec("safetensors") is not None

pytestmark = pytest.mark.skipif(
    not _HAS_SAFETENSORS, reason="safetensors required for disk cache"
)


class TestDiskMatmulCache:
    def test_put_then_get(self, tmp_path):
        cache = DiskMatmulCache(str(tmp_path / "cache"), max_size_mb=10)
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        key = cache.make_key("mock", x, w, None)
        value = torch.randn(2, 3)
        cache.put(key, value)
        loaded = cache.get(key)
        assert loaded is not None
        assert torch.allclose(loaded, value)

    def test_miss_returns_none(self, tmp_path):
        cache = DiskMatmulCache(str(tmp_path / "cache"))
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        key = cache.make_key("mock", x, w, None)
        assert cache.get(key) is None

    def test_key_stability(self, tmp_path):
        cache = DiskMatmulCache(str(tmp_path / "cache"))
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        k1 = cache.make_key("mock", x, w, None)
        k2 = cache.make_key("mock", x, w, None)
        assert k1 == k2
