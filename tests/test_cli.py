"""Tests for the ``qaithon`` CLI entry point."""

from __future__ import annotations

import json

import pytest

from qaithon.cli import _build_parser, main


class TestParser:
    def test_parser_builds(self):
        parser = _build_parser()
        assert parser is not None

    def test_list_backends_subcommand(self):
        parser = _build_parser()
        args = parser.parse_args(["list-backends"])
        assert args.command == "list-backends"

    def test_inspect_requires_model_id(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["inspect"])

    def test_compile_accepts_options(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "compile",
                "gpt2",
                "--backend",
                "mock",
                "--optimize-for",
                "energy",
            ]
        )
        assert args.model_id == "gpt2"
        assert args.backend == ["mock"]
        assert args.optimize_for == "energy"


class TestListBackends:
    def test_runs_without_error(self, capsys):
        rc = main(["list-backends"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "mock" in out
        assert "quandela.sim" in out


class TestInspectJSON:
    def test_inspect_outputs_json_for_local_model(self, capsys, monkeypatch):
        """Stub out the HF loader so the test does not require network access."""
        import torch.nn as nn

        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def forward(self, x):
                return self.fc(x)

        # Monkeypatch the loader to return our local Tiny.
        from qaithon import cli

        monkeypatch.setattr(cli, "_load_hf_model", lambda _id: Tiny())

        rc = main(["inspect", "tiny", "--json"])
        out = capsys.readouterr().out
        assert rc == 0
        data = json.loads(out)
        assert data["model_class"] == "Tiny"
        assert any(m["name"] == "fc" for m in data["replaceable"])
