"""Opt-in caching of backend computations.

Some backends — especially cloud QPUs — bill per shot and impose seconds-long
queues. If your workflow recomputes the same forward many times (sweep,
ablation, demo, validation), you want to cache.

:class:`MatmulCache` is a thread-safe LRU keyed by a hash of
``(backend_name, x, weight, bias)``. The default capacity is small and
in-memory; disk and Redis adapters are planned for v0.x.

Use the :func:`cached` decorator to wrap any :class:`Backend` instance so its
``matmul`` becomes cache-aware without touching the rest of the pipeline:

    from qaithon.backends import get_backend
    from qaithon.cache import cached

    backend = cached(get_backend("ibm.quantum"))
    # subsequent matmul calls with identical inputs return from cache

Caching is **opt-in**. The compiler does not wrap backends by default;
correctness must never depend on the cache being present.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from qaithon._logging import get_logger
from qaithon.backends.base import Backend

if TYPE_CHECKING:
    pass

__all__ = ["DiskMatmulCache", "MatmulCache", "cached"]

logger = get_logger(__name__)


def _hash_tensor(t: torch.Tensor) -> bytes:
    """Compute a stable hash of a tensor's contents + shape + dtype.

    Detaches and moves to CPU contiguous bytes; deterministic across runs.
    """
    arr = t.detach().to("cpu").contiguous()
    return hashlib.blake2b(
        arr.numpy().tobytes() + str(arr.shape).encode() + str(arr.dtype).encode(),
        digest_size=16,
    ).digest()


@dataclass
class _CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class MatmulCache:
    """Thread-safe LRU cache for ``Backend.matmul`` results.

    Args:
        capacity: Maximum number of distinct (input, weight, bias) tuples
            kept in memory. When exceeded, the least-recently-used entry is
            evicted.

    Notes:
        * The cache stores tensors directly on the same device they were
          computed on, so it does not move data between devices.
        * The cache is **never** shared across processes. Each process gets
          its own.
    """

    def __init__(self, capacity: int = 1024) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}.")
        self._capacity = capacity
        self._store: OrderedDict[bytes, torch.Tensor] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = _CacheStats()

    def make_key(
        self,
        backend_name: str,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> bytes:
        """Compute a stable cache key for the call."""
        parts = [backend_name.encode(), _hash_tensor(x), _hash_tensor(weight)]
        if bias is not None:
            parts.append(_hash_tensor(bias))
        return hashlib.blake2b(b"".join(parts), digest_size=16).digest()

    def get(self, key: bytes) -> torch.Tensor | None:
        """Return the cached tensor or ``None`` on miss."""
        with self._lock:
            if key in self._store:
                self.stats.hits += 1
                # Move to end (most recently used).
                self._store.move_to_end(key)
                return self._store[key].detach().clone()
            self.stats.misses += 1
            return None

    def put(self, key: bytes, value: torch.Tensor) -> None:
        """Insert a tensor under ``key``, evicting LRU if at capacity."""
        with self._lock:
            self._store[key] = value.detach()
            self._store.move_to_end(key)
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)
                self.stats.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


class _CachedBackend(Backend):
    """Internal wrapper exposing the same :class:`Backend` interface."""

    def __init__(self, inner: Backend, cache: MatmulCache | None = None) -> None:
        self._inner = inner
        self._cache = cache or MatmulCache()
        # Propagate profile so the selector sees the wrapped backend as the same.
        self.profile = inner.profile

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        key = self._cache.make_key(self.profile.name, x, weight, bias)
        cached_value = self._cache.get(key)
        if cached_value is not None:
            return cached_value
        result = self._inner.matmul(x, weight, bias)
        self._cache.put(key, result)
        return result

    def is_available(self) -> bool:
        return self._inner.is_available()

    @property
    def stats(self) -> _CacheStats:
        return self._cache.stats


class DiskMatmulCache:
    """Persistent on-disk variant of :class:`MatmulCache`.

    Uses safetensors for tensor storage. Each entry lives in a file named
    after the cache key; the filesystem itself is the index. This is the
    right choice when the same matmul (same input + same weight) is going
    to repeat across processes — typical for expensive cloud QPU calls
    during iterative experimentation.

    Args:
        cache_dir: Directory where cached tensors live. Created if absent.
        max_size_mb: Soft cap. When exceeded, oldest files are evicted by
            modification time. ``None`` disables eviction.

    Notes:
        * Eviction is lazy — checked on every ``put``. Not LRU; uses file
          mtime as a proxy.
        * Tensors are stored on the CPU. Callers receive copies on the
          requested device.
    """

    def __init__(self, cache_dir: str, max_size_mb: int | None = 512) -> None:
        import pathlib

        self._path = pathlib.Path(cache_dir)
        self._path.mkdir(parents=True, exist_ok=True)
        self._max_bytes = max_size_mb * 1024 * 1024 if max_size_mb else None
        self.stats = _CacheStats()

    def _key_path(self, key: bytes) -> "pathlib.Path":  # type: ignore[name-defined]
        return self._path / (key.hex() + ".safetensors")

    def make_key(
        self,
        backend_name: str,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> bytes:
        """Same key derivation as the in-memory cache for cross-compat."""
        parts = [backend_name.encode(), _hash_tensor(x), _hash_tensor(weight)]
        if bias is not None:
            parts.append(_hash_tensor(bias))
        return hashlib.blake2b(b"".join(parts), digest_size=16).digest()

    def get(self, key: bytes) -> torch.Tensor | None:
        try:
            from safetensors.torch import load_file
        except ImportError:
            logger.warning(
                "DiskMatmulCache requires safetensors. Install with "
                "`pip install qaithon[huggingface]`. Disk cache disabled."
            )
            return None
        path = self._key_path(key)
        if not path.exists():
            self.stats.misses += 1
            return None
        try:
            data = load_file(str(path))
            self.stats.hits += 1
            return data["result"]
        except Exception as exc:  # noqa: BLE001
            logger.debug("DiskMatmulCache read failed for %s: %s", path.name, exc)
            self.stats.misses += 1
            return None

    def put(self, key: bytes, value: torch.Tensor) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError:
            return
        path = self._key_path(key)
        save_file({"result": value.detach().cpu()}, str(path))
        if self._max_bytes is not None:
            self._maybe_evict()

    def _maybe_evict(self) -> None:
        files = sorted(self._path.glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
        total = sum(p.stat().st_size for p in files)
        while total > (self._max_bytes or 0) and files:
            victim = files.pop(0)
            try:
                size = victim.stat().st_size
                victim.unlink()
                total -= size
                self.stats.evictions += 1
            except OSError:
                break


def cached(backend: Backend, capacity: int = 1024) -> Backend:
    """Return a cache-aware version of ``backend``.

    Args:
        backend: Any :class:`Backend` instance.
        capacity: LRU cache capacity (number of distinct matmul calls).

    Returns:
        A wrapped :class:`Backend` that returns cached results when the
        input tensors match a prior call, and falls through to the
        original backend on miss.

    Example:
        >>> from qaithon.backends import get_backend
        >>> from qaithon.cache import cached
        >>> backend = cached(get_backend("quandela.sim"), capacity=512)
        >>> # Same call twice — second one is from cache:
        >>> import torch
        >>> x = torch.rand(2, 4)
        >>> w = torch.randn(3, 4)
        >>> _ = backend.matmul(x, w)
        >>> _ = backend.matmul(x, w)
        >>> backend.stats.hits
        1
    """
    cache = MatmulCache(capacity=capacity)
    return _CachedBackend(backend, cache)
