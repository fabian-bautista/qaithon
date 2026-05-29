"""Training utilities for Qaithon-compiled models.

Qaithon supports two distinct training modes, both transparent to the user:

**1. Plain fine-tuning** — for users who only want to adapt an LLM with a
   Qaithon-accelerated forward pass. The compiled model behaves like any
   PyTorch ``nn.Module``: pass it to ``transformers.Trainer``, ``peft``,
   ``accelerate``, or a custom loop, and gradient updates flow through
   :class:`~qaithon.layers.QuantumLinear` naturally.

**2. Quantization-Aware Training (QAT)** — for users who want the trained
   model to be robust against the noise the real photonic / quantum
   hardware will introduce at inference time. During QAT we use a backend
   whose simulator injects the same noise distribution the target
   hardware exhibits; the model parameters end up at a point in weight
   space that is locally invariant to that noise.

This module provides:

* :class:`QATConfig` — dataclass declaring "use noise model X with std Y
  during training, switch to zero noise for evaluation".
* :func:`prepare_for_qat` — applies the config: replaces backends in
  every :class:`QuantumLinear` with noisy variants for training mode and
  with zero-noise variants for ``.eval()`` mode.

Real-hardware autograd (parameter-shift rule against IBM Quantum or
Quandela Cloud) is on the roadmap (v0.x) but not part of v0.1 because the
latency of cloud-QPU inference under such a scheme is prohibitive for
fine-tune at LLM scale. Until that lands, QAT is the recommended path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from torch import nn

from qaithon._logging import get_logger

if TYPE_CHECKING:
    pass

__all__ = ["QATConfig", "prepare_for_qat"]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class QATConfig:
    """Configuration for Quantization-Aware Training.

    Attributes:
        target_backend: Name of the backend you intend to use at inference
            time (e.g. ``"quandela.sim"``).
        noise_std_training: Standard deviation of additive noise injected
            during training. Match the target hardware's measured noise
            level for best transfer; values in ``[1e-3, 1e-1]`` are typical
            for photonic processors.
        noise_std_eval: Noise level during evaluation. Usually ``0.0`` so
            evaluation metrics are clean.
        seed: Optional seed for reproducible noise.
    """

    target_backend: str = "quandela.sim"
    noise_std_training: float = 0.05
    noise_std_eval: float = 0.0
    seed: int | None = None


def prepare_for_qat(model: nn.Module, config: QATConfig | None = None) -> nn.Module:
    """Wire ``model`` so its :class:`QuantumLinear` layers swap noise level on ``train()`` / ``eval()``.

    Must be called **after** :func:`qaithon.compile`. The function attaches
    a hook that updates each ``QuantumLinear``'s backend to a noisy variant
    when entering training mode and a clean variant when entering eval mode.

    Args:
        model: A Qaithon-compiled model (i.e. one whose linear layers have
            been replaced by :class:`QuantumLinear`).
        config: QAT configuration; defaults to a sensible Quandela profile.

    Returns:
        The same model, with QAT hooks installed.

    Example:
        >>> import qaithon
        >>> from qaithon.training import QATConfig, prepare_for_qat
        >>> # ... model = AutoModelForCausalLM.from_pretrained(...)
        >>> # qaithon.compile(model)
        >>> # prepare_for_qat(model, QATConfig(noise_std_training=0.05))
        >>> # Trainer(model, ...).train()
    """
    config = config or QATConfig()

    from qaithon.backends.quandela_sim import QuandelaSimBackend
    from qaithon.layers.quantum_linear import QuantumLinear

    noisy_backend = QuandelaSimBackend(
        normalize_inputs=False,
        noise_std=config.noise_std_training,
        seed=config.seed,
    )
    clean_backend = QuandelaSimBackend(
        normalize_inputs=False,
        noise_std=config.noise_std_eval,
        seed=config.seed,
    )

    quantum_layers: list[QuantumLinear] = [
        m for m in model.modules() if isinstance(m, QuantumLinear)
    ]
    if not quantum_layers:
        logger.warning(
            "prepare_for_qat called but model has no QuantumLinear layers — "
            "did you forget to call qaithon.compile() first?"
        )

    # Snapshot the original train() so we can extend rather than replace it.
    original_train = model.train

    def train(self: nn.Module, mode: bool = True) -> nn.Module:  # type: ignore[misc]
        result = original_train(mode)
        chosen = noisy_backend if mode else clean_backend
        for layer in quantum_layers:
            layer.backend = chosen
        return result

    # Bind method to the instance.
    import types

    model.train = types.MethodType(train, model)  # type: ignore[method-assign]

    # Apply the current mode immediately so first call sees the right backend.
    model.train(model.training)
    return model
