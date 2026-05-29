"""Tests for the LRU cache wrapper on backends."""

from __future__ import annotations

import pytest
import torch

from qaithon.backends import get_backend
from qaithon.cache import MatmulCache, cached


class TestMatmulCache:
    def test_rejects_zero_capacity(self):
        with pytest.raises(ValueError, match="positive"):
            MatmulCache(capacity=0)

    def test_put_then_get(self):
        c = MatmulCache(capacity=4)
        t = torch.randn(2, 3)
        key = c.make_key("mock", torch.randn(2, 4), torch.randn(3, 4), None)
        c.put(key, t)
        assert torch.allclose(c.get(key), t)
        assert c.stats.hits == 1

    def test_eviction_when_full(self):
        c = MatmulCache(capacity=2)
        for i in range(3):
            key = c.make_key(str(i), torch.tensor([float(i)]), torch.tensor([1.0]), None)
            c.put(key, torch.tensor([float(i)]))
        assert c.size == 2
        assert c.stats.evictions == 1

    def test_key_stable_across_calls(self):
        c = MatmulCache()
        x = torch.randn(2, 3)
        w = torch.randn(4, 3)
        k1 = c.make_key("mock", x, w, None)
        k2 = c.make_key("mock", x, w, None)
        assert k1 == k2


class TestCachedBackendWrapper:
    def test_passthrough_first_call(self):
        backend = cached(get_backend("mock"))
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        y = backend.matmul(x, w)
        assert y.shape == (2, 3)
        assert backend.stats.misses == 1
        assert backend.stats.hits == 0

    def test_hit_on_second_identical_call(self):
        backend = cached(get_backend("mock"))
        x = torch.randn(2, 4)
        w = torch.randn(3, 4)
        backend.matmul(x, w)
        backend.matmul(x, w)
        assert backend.stats.hits == 1

    def test_different_inputs_are_misses(self):
        backend = cached(get_backend("mock"))
        w = torch.randn(3, 4)
        backend.matmul(torch.randn(2, 4), w)
        backend.matmul(torch.randn(2, 4), w)
        # Different x → cache miss both times.
        assert backend.stats.misses == 2

    def test_profile_unchanged_by_wrapper(self):
        backend = cached(get_backend("mock"))
        assert backend.profile.name == "mock"
