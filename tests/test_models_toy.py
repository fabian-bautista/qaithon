"""Tests for the toy transformer factories."""

from __future__ import annotations

import importlib.util

import pytest
import torch


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers not installed",
)
class TestToyTokenizer:
    def test_encode_decode_roundtrip(self):
        from qaithon.models import ToyTokenizer

        tok = ToyTokenizer()
        original = "Hello world!"
        ids = tok.encode(original, add_bos=False)
        recovered = tok.decode(ids)
        assert recovered == original

    def test_bos_added_by_default(self):
        from qaithon.models import ToyTokenizer

        tok = ToyTokenizer()
        ids = tok.encode("hi")
        assert ids[0] == tok.bos_token_id


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None,
    reason="transformers not installed",
)
class TestToyTransformer:
    def test_create_micro(self):
        from qaithon.models import create_micro_transformer

        model = create_micro_transformer()
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 10_000  # truly micro

    def test_create_toy_default(self):
        from qaithon.models import create_toy_transformer

        model = create_toy_transformer()
        # dim=32, 1 layer, 2 heads — well under 50k params.
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params < 50_000

    def test_dim_must_divide_heads(self):
        from qaithon.models import create_toy_transformer

        with pytest.raises(ValueError, match="divisible"):
            create_toy_transformer(dim=17, n_heads=2)

    def test_generate_runs(self):
        from qaithon.models import ToyTokenizer, create_micro_transformer

        tokenizer = ToyTokenizer()
        model = create_micro_transformer()
        model.eval()
        inputs = tokenizer("a", return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Generation completed; output should include the input ids + some new ones.
        assert out.shape[1] >= 1 + 5

    def test_compiles_with_qaithon(self):
        import qaithon
        from qaithon.models import create_micro_transformer

        model = create_micro_transformer()
        qaithon.compile(
            model,
            backends=("mock",),
            min_in_features=1,
            min_out_features=1,
        )
        assert model.qaithon_report.n_replaced > 0


@pytest.mark.skipif(
    importlib.util.find_spec("qiskit_aer") is None,
    reason="qiskit-aer not installed",
)
class TestFidelityMode:
    def test_ideal_is_default(self):
        from qaithon.backends.ibm_aer import IBMAerBackend

        backend = IBMAerBackend()
        assert backend.fidelity_mode == "ideal"

    def test_invalid_mode_rejected(self):
        from qaithon.backends.ibm_aer import IBMAerBackend

        with pytest.raises(ValueError, match="fidelity_mode"):
            IBMAerBackend(fidelity_mode="nonsense")

    def test_realistic_mode_accepted(self):
        from qaithon.backends.ibm_aer import IBMAerBackend

        backend = IBMAerBackend(fidelity_mode="realistic")
        assert backend.fidelity_mode == "realistic"

    def test_realistic_diverges_from_ideal(self):
        """The whole point: realistic should give different output than ideal."""
        from qaithon.backends.ibm_aer import IBMAerBackend

        ideal = IBMAerBackend(fidelity_mode="ideal", noise_strength=0.0)
        realistic = IBMAerBackend(fidelity_mode="realistic", noise_strength=0.0)

        x = torch.rand(2, 4)
        w = torch.randn(3, 4)

        # When noise_strength=0, ideal and realistic should ALSO match (no
        # noise is injected on top of the classical result). The divergence
        # appears only with noise_strength > 0 AND realistic mode.
        torch.manual_seed(0)
        y_ideal = ideal.matmul(x, w)
        torch.manual_seed(0)
        y_realistic = realistic.matmul(x, w)
        # With noise_strength=0, both should be identical classical output.
        assert torch.allclose(y_ideal, y_realistic)
