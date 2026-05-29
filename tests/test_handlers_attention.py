"""Tests for the attention-detection helpers."""

from __future__ import annotations

from torch import nn

from qaithon.handlers.attention import AttentionInfo, list_attention_modules


class _FakeLlamaSdpaAttention(nn.Module):
    pass


class _FakeFlashAttention2(nn.Module):
    pass


class _RegularLayer(nn.Module):
    pass


class TestDetection:
    def test_identifies_sdpa(self):
        model = nn.Sequential(_FakeLlamaSdpaAttention())
        infos = list_attention_modules(model)
        assert len(infos) == 1
        assert infos[0].kind == "sdpa"

    def test_identifies_flash(self):
        model = nn.Sequential(_FakeFlashAttention2())
        infos = list_attention_modules(model)
        assert len(infos) == 1
        assert infos[0].kind == "flash"

    def test_identifies_pytorch_multihead(self):
        model = nn.Sequential(nn.MultiheadAttention(8, 2))
        infos = list_attention_modules(model)
        assert any(i.kind == "multihead" for i in infos)

    def test_returns_empty_for_regular_layers(self):
        model = nn.Sequential(_RegularLayer(), nn.Linear(4, 4))
        infos = list_attention_modules(model)
        assert infos == []

    def test_attention_info_is_immutable(self):
        info = AttentionInfo(name="foo", kind="sdpa", class_name="X")
        try:
            info.kind = "flash"  # type: ignore[misc]
        except Exception:
            pass
        # Frozen dataclass: assignment is rejected.
        assert info.kind == "sdpa"
