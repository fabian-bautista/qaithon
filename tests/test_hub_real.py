"""Tests for the real (huggingface_hub-backed) Hub implementation.

These tests do not hit the network — they exercise the local code paths
(error messages, metadata parsing, class loading) using mocks and
``tmp_path``. Real end-to-end push/load tests will live in a separate
integration suite tagged ``@pytest.mark.integration``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from qaithon.hub.loader import (
    HubError,
    HubNotImplementedError,
    _instantiate_block,
)


class TestErrorTypes:
    def test_hub_not_implemented_is_compat_subclass(self):
        assert issubclass(HubNotImplementedError, NotImplementedError)
        assert issubclass(HubNotImplementedError, HubError)


class TestInstantiate:
    def test_rejects_missing_class_field(self, tmp_path):
        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"")
        with pytest.raises(HubError, match="qualified.*class"):
            _instantiate_block({"init_kwargs": {}}, str(weights))

    def test_rejects_unimportable_class(self, tmp_path):
        weights = tmp_path / "model.safetensors"
        weights.write_bytes(b"")
        with pytest.raises(HubError, match="not importable"):
            _instantiate_block(
                {"class": "no_such_pkg.NoSuchModule", "init_kwargs": {}},
                str(weights),
            )


@pytest.mark.skipif(
    importlib.util.find_spec("huggingface_hub") is None,
    reason="huggingface_hub not installed",
)
class TestRoundtripStub:
    def test_push_then_load_locally(self, tmp_path, monkeypatch):
        """Verify the safetensors + JSON round-trip path without hitting HF."""
        import torch
        import torch.nn as nn
        from safetensors.torch import save_file

        from qaithon.hub.loader import _METADATA_FILENAME, _WEIGHTS_FILENAME

        block_dir = tmp_path / "block"
        block_dir.mkdir()

        layer = nn.Linear(4, 4)
        save_file(layer.state_dict(), str(block_dir / _WEIGHTS_FILENAME))
        (block_dir / _METADATA_FILENAME).write_text(
            json.dumps(
                {
                    "class": "torch.nn.modules.linear.Linear",
                    "init_kwargs": {"in_features": 4, "out_features": 4},
                }
            )
        )

        loaded = _instantiate_block(
            json.loads(Path(block_dir / _METADATA_FILENAME).read_text()),
            str(block_dir / _WEIGHTS_FILENAME),
        )
        assert isinstance(loaded, nn.Linear)
        x = torch.randn(2, 4)
        # Outputs identical because we round-tripped the same weights.
        assert torch.allclose(loaded(x), layer(x))
