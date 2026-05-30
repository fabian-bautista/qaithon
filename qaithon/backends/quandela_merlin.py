"""Quandela MerLin backend — genuine photonic matmul.

The matmul ``y = x @ W^T`` is computed by a **real linear-optical algorithm**:
the weight is embedded in a unitary (Halmos dilation), realised as an
interferometer, and the output amplitudes are read back (see
:func:`qaithon.kernels.photonic_linear`). There is **no** ``F.linear`` in the
compute path — only the encode/decode at the boundaries is classical.

Inference-only (the SLOS amplitude read-back is not differentiable). For a
**differentiable, trainable** photonic layer (MerLin's autograd-native
``QuantumLayer``), use :class:`qaithon.PhotonicLayer` instead — that is the path
where gradients flow through the photonic forward.

Bounded to the simulator's mode reach (dim ≤ 128); larger layers raise
``IncompatibleHardwareError`` (use the quantum path, which packs more per qubit).
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import torch

from qaithon._logging import get_logger
from qaithon.backends.base import Backend, BackendProfile, register_backend

if TYPE_CHECKING:
    pass

__all__ = ["QuandelaMerlinBackend"]

logger = get_logger(__name__)


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


class QuandelaMerlinBackend(Backend):
    """Genuine photonic matmul (Perceval/MerLin SLOS), MerLin cost profile.

    Numerically exact in simulation — the linear-optical circuit *is* doing the
    matmul, so fidelity vs the classical result is 1.0. Inference-only; for
    differentiable training use :class:`qaithon.PhotonicLayer`.
    """

    profile: BackendProfile = BackendProfile(
        name="quandela.merlin",
        kind="photonic",
        energy_pj_per_mac=0.004,
        latency_us_per_op=30_000.0,
        queue_us=0.0,
        supports_autograd=False,  # genuine SLOS read-back is not differentiable
        supports_batching=True,
        max_dim=None,
        notes=(
            "Genuine photonic matmul (Halmos dilation → linear-optical circuit, "
            "MerLin/Perceval SLOS). Inference-only; for a differentiable photonic "
            "layer use qaithon.PhotonicLayer. Bounded to dim ≤ 128."
        ),
    )

    def is_available(self) -> bool:
        return _has("merlin") and _has("perceval")

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Genuine optical compute (no F.linear). Raises IncompatibleHardwareError
        # beyond the SLOS simulator's 256-mode reach (dim 128).
        from qaithon.kernels import photonic_linear

        return photonic_linear(x, weight, bias)


if _has("merlin") and _has("perceval"):
    register_backend("quandela.merlin", QuandelaMerlinBackend, overwrite=True)
