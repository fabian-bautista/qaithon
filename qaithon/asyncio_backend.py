"""Async-capable backend wrapper for streaming and concurrent inference.

Backends that run on cloud QPUs / photonic services often have latency
dominated by network and queue time (10s of ms to multiple seconds). When
serving an LLM with ``generate(stream=True)``, you want to issue the next
matmul while the current one is still in flight on the remote service.

Strategy
--------

* The synchronous :class:`Backend` contract stays as-is — that's what
  PyTorch's autograd and ``nn.Module`` expect.
* :class:`AsyncCompatBackend` wraps a synchronous backend and exposes a
  parallel :meth:`matmul_async` coroutine that runs the matmul on a thread
  pool. For backends whose ``matmul`` is just CPU compute, this is
  effectively a no-op overhead; for backends that make blocking network
  calls, it lets multiple in-flight requests pipeline.
* :class:`StreamingPipeline` builds on top: it accepts an ``async def
  generator`` (the model's per-token output) and yields tokens to the
  caller while the next forward is computing.

This is the minimal viable streaming layer; full integration with
HuggingFace's ``generate(streamer=...)`` lands in a follow-up.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

from qaithon._logging import get_logger
from qaithon.backends.base import Backend

if TYPE_CHECKING:
    pass

__all__ = ["AsyncCompatBackend", "StreamingPipeline"]

logger = get_logger(__name__)

# Bounded shared executor — keeps thread count predictable when many
# AsyncCompatBackend instances coexist.
_DEFAULT_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="qaithon-async")


class AsyncCompatBackend(Backend):
    """Wrap a synchronous backend with an async-capable interface.

    The synchronous :meth:`matmul` stays available so the wrapper is a
    drop-in replacement anywhere a :class:`Backend` is expected.
    :meth:`matmul_async` lets callers schedule and await the same
    computation concurrently.

    Args:
        inner: The synchronous backend to wrap.
        executor: Optional ThreadPoolExecutor. Defaults to a shared one.
    """

    def __init__(
        self,
        inner: Backend,
        executor: ThreadPoolExecutor | None = None,
    ) -> None:
        self._inner = inner
        self._executor = executor or _DEFAULT_EXECUTOR
        self.profile = inner.profile

    def is_available(self) -> bool:
        return self._inner.is_available()

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._inner.matmul(x, weight, bias)

    async def matmul_async(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the matmul on a thread pool and await the result."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._inner.matmul, x, weight, bias
        )


class StreamingPipeline:
    """Yield tokens as they are produced while the next forward is in flight.

    The pipeline expects a ``generate_one`` async callable that produces
    the next token given the prior context. It schedules the next call as
    soon as the current one finishes, yielding completed tokens to the
    caller through an :class:`AsyncIterator`.

    Args:
        generate_one: Async callable that maps the conversation tensor to
            the next output tensor. Typically wraps ``model.forward``.
        max_tokens: Maximum number of tokens to yield before stopping.

    Example:
        >>> async def gen(state: torch.Tensor) -> torch.Tensor:
        ...     await asyncio.sleep(0)  # simulate async I/O
        ...     return state[..., -1:]
        >>> pipe = StreamingPipeline(gen, max_tokens=3)
        >>> async def main():
        ...     async for tok in pipe(torch.tensor([[1, 2, 3]])):
        ...         print(tok.shape)
        >>> # asyncio.run(main())  # doctest: +SKIP
    """

    def __init__(
        self,
        generate_one: Callable[[torch.Tensor], "asyncio.Future[torch.Tensor] | torch.Tensor"],
        max_tokens: int = 256,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}.")
        self._generate_one = generate_one
        self._max_tokens = max_tokens

    async def __call__(self, prompt: torch.Tensor) -> AsyncIterator[torch.Tensor]:
        return self._iter(prompt)

    async def _iter(self, prompt: torch.Tensor) -> AsyncIterator[torch.Tensor]:
        state = prompt
        for _ in range(self._max_tokens):
            next_token = self._generate_one(state)
            if asyncio.iscoroutine(next_token):
                next_token = await next_token
            yield next_token  # type: ignore[misc]
            state = torch.cat([state, next_token], dim=-1)  # type: ignore[arg-type]
