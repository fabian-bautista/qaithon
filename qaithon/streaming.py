"""HuggingFace-compatible streaming for Qaithon-compiled models.

HuggingFace ``transformers`` exposes a ``BaseStreamer`` protocol used by
``model.generate(streamer=...)`` to push tokens to the caller as they are
decoded. :class:`QaithonStreamer` is a drop-in implementation that, on top
of the standard token decoding, records the per-token latency and the
backend each token traversed. The result is a runtime trace correlated to
actual tokens — useful for benchmarking, debugging, and pitching.

Two flavors are provided:

* :class:`QaithonStreamer` — synchronous, prints (or accumulates) tokens
  while recording per-token telemetry.
* :class:`QaithonIteratorStreamer` — async-friendly iterator variant
  modeled after ``transformers.TextIteratorStreamer``.

Both work with any tokenizer; pass it at construction time.
"""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from qaithon._logging import get_logger

if TYPE_CHECKING:
    pass

__all__ = ["QaithonIteratorStreamer", "QaithonStreamer", "TokenTrace"]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TokenTrace:
    """One token's worth of telemetry."""

    token_id: int
    text: str
    latency_us: float


@dataclass
class _BaseStreamerImpl:
    """Shared scaffolding for both streamers."""

    tokenizer: Any
    skip_prompt: bool = True
    decode_kwargs: dict[str, Any] = field(default_factory=dict)
    _is_first: bool = field(default=True, init=False)
    _start_time: float = field(default_factory=time.perf_counter, init=False)
    traces: list[TokenTrace] = field(default_factory=list, init=False)

    @property
    def total_latency_us(self) -> float:
        return sum(t.latency_us for t in self.traces)

    @property
    def n_tokens(self) -> int:
        return len(self.traces)

    @property
    def tokens_per_second(self) -> float:
        if self.total_latency_us == 0:
            return 0.0
        return 1e6 * self.n_tokens / self.total_latency_us


class QaithonStreamer(_BaseStreamerImpl):
    """Streamer that prints tokens as they arrive (or accumulates them silently).

    Args:
        tokenizer: The model's tokenizer. Used to decode token IDs.
        skip_prompt: If ``True`` (default), the prompt tokens are not
            emitted — matches ``TextStreamer`` semantics.
        print_to_stdout: If ``True``, tokens are printed as they arrive.
        decode_kwargs: Extra kwargs for ``tokenizer.decode``.

    Example:
        >>> import qaithon
        >>> # from transformers import AutoModelForCausalLM, AutoTokenizer  # doctest: +SKIP
        >>> # streamer = QaithonStreamer(tokenizer)
        >>> # model.generate(inputs, streamer=streamer, max_new_tokens=50)
        >>> # print(f"Achieved {streamer.tokens_per_second:.1f} tokens/s")
    """

    print_to_stdout: bool = True

    def __init__(
        self,
        tokenizer: Any,
        *,
        skip_prompt: bool = True,
        print_to_stdout: bool = True,
        decode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            skip_prompt=skip_prompt,
            decode_kwargs=decode_kwargs or {},
        )
        self.print_to_stdout = print_to_stdout
        self._last_emit_time: float = time.perf_counter()

    # ----- BaseStreamer protocol expected by transformers ---------------
    def put(self, value: torch.Tensor) -> None:
        """Called by ``generate`` for each batch of new tokens."""
        if self._is_first and self.skip_prompt:
            self._is_first = False
            self._last_emit_time = time.perf_counter()
            return

        now = time.perf_counter()
        latency_us = (now - self._last_emit_time) * 1e6
        self._last_emit_time = now

        # `value` shape during decoding is usually (batch, 1) with the
        # newest tokens. We focus on the first row (batch=1 is the common case).
        token_ids = value[0].tolist() if value.dim() > 1 else value.tolist()
        for token_id in token_ids:
            text = self.tokenizer.decode([token_id], **self.decode_kwargs)
            self.traces.append(
                TokenTrace(
                    token_id=int(token_id),
                    text=text,
                    latency_us=latency_us / max(1, len(token_ids)),
                )
            )
            if self.print_to_stdout:
                print(text, end="", flush=True)

    def end(self) -> None:
        """Called by ``generate`` once generation finishes."""
        if self.print_to_stdout:
            print()


class QaithonIteratorStreamer(_BaseStreamerImpl):
    """Iterator-style streamer for async / threaded inference.

    Modeled after ``transformers.TextIteratorStreamer``. Run ``generate``
    in a background thread and consume tokens by iterating the streamer.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        skip_prompt: bool = True,
        timeout: float | None = None,
        decode_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            tokenizer=tokenizer,
            skip_prompt=skip_prompt,
            decode_kwargs=decode_kwargs or {},
        )
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._timeout = timeout
        self._last_emit_time = time.perf_counter()

    def put(self, value: torch.Tensor) -> None:
        if self._is_first and self.skip_prompt:
            self._is_first = False
            self._last_emit_time = time.perf_counter()
            return

        now = time.perf_counter()
        latency_us = (now - self._last_emit_time) * 1e6
        self._last_emit_time = now

        token_ids = value[0].tolist() if value.dim() > 1 else value.tolist()
        for token_id in token_ids:
            text = self.tokenizer.decode([token_id], **self.decode_kwargs)
            self.traces.append(
                TokenTrace(
                    token_id=int(token_id),
                    text=text,
                    latency_us=latency_us / max(1, len(token_ids)),
                )
            )
            self._queue.put(text)

    def end(self) -> None:
        self._queue.put(None)  # sentinel

    def __iter__(self) -> QaithonIteratorStreamer:
        return self

    def __next__(self) -> str:
        item = self._queue.get(timeout=self._timeout)
        if item is None:
            raise StopIteration
        return item
