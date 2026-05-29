"""Tests for the HuggingFace-compatible streamers."""

from __future__ import annotations

import pytest
import torch

from qaithon.streaming import QaithonIteratorStreamer, QaithonStreamer, TokenTrace


class _FakeTokenizer:
    """Minimal tokenizer mock that just stringifies ids."""

    def decode(self, ids, **_kwargs):
        return "|".join(str(i) for i in ids) + " "


class TestQaithonStreamer:
    def test_records_tokens(self):
        streamer = QaithonStreamer(
            tokenizer=_FakeTokenizer(),
            print_to_stdout=False,
            skip_prompt=False,
        )
        streamer.put(torch.tensor([[1, 2, 3]]))
        streamer.end()
        assert streamer.n_tokens == 3
        assert streamer.tokens_per_second >= 0

    def test_skip_prompt_default(self):
        streamer = QaithonStreamer(
            tokenizer=_FakeTokenizer(), print_to_stdout=False, skip_prompt=True
        )
        # First call is the prompt and should be skipped.
        streamer.put(torch.tensor([[10, 20, 30]]))
        # Second call is the first generated token.
        streamer.put(torch.tensor([[99]]))
        streamer.end()
        assert streamer.n_tokens == 1

    def test_trace_records_text(self):
        streamer = QaithonStreamer(
            tokenizer=_FakeTokenizer(), print_to_stdout=False, skip_prompt=False
        )
        streamer.put(torch.tensor([[7]]))
        assert streamer.traces[0].token_id == 7
        assert isinstance(streamer.traces[0], TokenTrace)


class TestQaithonIteratorStreamer:
    def test_iterator_yields_tokens(self):
        streamer = QaithonIteratorStreamer(tokenizer=_FakeTokenizer(), skip_prompt=False)
        streamer.put(torch.tensor([[1, 2]]))
        streamer.end()
        tokens = list(streamer)
        assert len(tokens) == 2

    def test_iterator_stops_at_sentinel(self):
        streamer = QaithonIteratorStreamer(tokenizer=_FakeTokenizer(), skip_prompt=False)
        streamer.put(torch.tensor([[5]]))
        streamer.end()
        # Drain once successfully, then StopIteration.
        next(iter(streamer))
        with pytest.raises(StopIteration):
            next(iter(streamer))
