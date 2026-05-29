"""Optional DeepQuantum-flavored backend.

DeepQuantum (by TuringQ) is uniquely positioned in our backend mix because
it natively combines qubit and photonic primitives behind a single PyTorch-
native ``nn.Module`` interface. It is the only library currently active that
exposes ``UnitaryDecomposer`` and bosonic states (Cat/GKP) directly — these
become valuable in v0.x when we move past linear approximation toward
unitary-decomposition-based execution (today this backend is, like the
other simulators, ``F.linear``-equivalent with its own cost profile).

Marked as the "power-tool" backend in the roadmap: opt-in via
``pip install qaithon[deepquantum]``. Auto-registers only when the
``deepquantum`` package is importable, so machines without it don't see
DeepQuantum at all in :func:`qaithon.backends.list_backends`.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qaithon._logging import get_logger
from qaithon.backends.base import Backend, BackendProfile, register_backend

if TYPE_CHECKING:
    pass

__all__ = ["DeepQuantumBackend"]

logger = get_logger(__name__)


# Cost profile: DeepQuantum simulators land between PennyLane and dedicated
# C++ backends in benchmarks. These are conservative placeholder numbers;
# the v0.x calibration pass will replace them with measured ones.
_DEEPQUANTUM_ENERGY_PJ_PER_MAC = 0.4
_DEEPQUANTUM_LATENCY_US_PER_OP = 75.0


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


class DeepQuantumBackend(Backend):
    """DeepQuantum-flavored backend (TuringQ).

    Same matmul contract as every other backend (``F.linear``-equivalent for
    v0.1). Distinct from PennyLane because it natively supports bosonic
    states and unitary decomposition primitives — useful when, in v0.x, the
    forward switches to a real ``QubitCircuit`` constructed from the weight
    matrix's SVD.
    """

    profile: BackendProfile = BackendProfile(
        name="deepquantum",
        kind="quantum",
        energy_pj_per_mac=_DEEPQUANTUM_ENERGY_PJ_PER_MAC,
        latency_us_per_op=_DEEPQUANTUM_LATENCY_US_PER_OP,
        queue_us=0.0,
        supports_autograd=True,
        supports_batching=True,
        max_dim=None,
        notes=(
            "DeepQuantum (TuringQ). Exposes qubit + photonic + bosonic primitives "
            "in a single PyTorch-native module. Today: F.linear-equivalent with "
            "its own cost profile. Future: route forward through QubitCircuit "
            "constructed via UnitaryDecomposer when weight is unitary or near-unitary."
        ),
    )

    def is_available(self) -> bool:
        return _has("deepquantum")

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # GENUINE compute via the real quantum kernel (qubit circuit), not F.linear.
        from qaithon.kernels import quantum_linear

        return quantum_linear(x, weight, bias)


# Conditional registration: only registers if deepquantum is importable.
# Users without the [deepquantum] extra don't see it in list_backends().
if _has("deepquantum"):
    register_backend("deepquantum", DeepQuantumBackend, overwrite=True)
