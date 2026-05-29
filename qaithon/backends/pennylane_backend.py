"""Backend that delegates to PennyLane simulators or hardware via plugins.

PennyLane is the broadest QML framework in the ecosystem. With one adapter,
Qaithon gains access to:

* ``default.qubit`` — pure Python simulator, fast for small circuits.
* ``lightning.qubit`` — C++ simulator, much faster.
* ``qiskit.remote`` — IBM Quantum hardware (via ``pennylane-qiskit``).
* ``braket.aws.qubit`` — AWS Braket QPUs (via ``pennylane-braket``).

We pick this approach (one backend, parametrized by device) over native
Qiskit / Braket backends because the PennyLane plugins are maintained by
Xanadu + the respective vendors. Reimplementing them ourselves would
duplicate maintenance and CI matrix without value (GAP-007).

Current scope (v0.1)
--------------------

This backend honors the :class:`Backend` contract by computing
``F.linear``-equivalent forward — same caveat as
:class:`QuandelaSimBackend`. The cost profile is derived from typical
simulator characteristics (or QPU queue times for the remote variants).
A future ``pennylane_circuit.py`` will add a backend that constructs a
real parametrized circuit and uses parameter-shift gradients, but that
breaks drop-in semantics and is not appropriate for v0.1's "compile any
HF model" promise.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qaithon._logging import get_logger
from qaithon.backends.base import Backend, BackendProfile, register_backend


def _safe_has_spec(name: str) -> bool:
    """Return True if a dotted module path is importable, without raising.

    ``importlib.util.find_spec`` raises ``ModuleNotFoundError`` if a parent
    package on the way doesn't exist (e.g. checking ``amazon.braket`` when
    ``amazon`` is not a package). We swallow that and return ``False``.
    """
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False

if TYPE_CHECKING:
    pass

__all__ = [
    "AWSBraketSimBackend",
    "IBMQuantumSimBackend",
    "PennyLaneSimBackend",
]

logger = get_logger(__name__)

# Cost numbers grounded in publicly available figures:
# - default.qubit: pure Python, ~10 microsec per gate per qubit on consumer CPU.
# - lightning.qubit: C++, ~10x faster.
# - IBM Quantum Heron: queue can vary from seconds to hours; we use a
#   conservative 30 s for "balanced" scheduling decisions.
# - AWS Braket SV1 simulator: 100 microsec/op typical.

_PENNYLANE_DEFAULT_ENERGY = 0.8       # pJ/MAC; simulator is classical, similar to CPU.
_PENNYLANE_DEFAULT_LATENCY = 200.0    # us per matmul, pessimistic for default.qubit.

_IBM_QUEUE_US = 30_000_000.0          # 30 s queue (median Open Plan).
_IBM_ENERGY = 0.01                    # pJ/MAC, optimistic if the QPU is used efficiently.
_IBM_LATENCY = 5_000.0                # us per op on hardware (sub-ms gates × depth).

_BRAKET_QUEUE_US = 5_000_000.0        # 5 s typical
_BRAKET_ENERGY = 0.5                  # pJ/MAC for SV1
_BRAKET_LATENCY = 1_000.0


class PennyLaneSimBackend(Backend):
    """PennyLane-flavored simulator backend.

    Numerically equivalent to ``F.linear`` while exposing a PennyLane-style
    cost profile to the compiler. When PennyLane is installed, a future
    upgrade can switch the internal forward to a real ``qml.qnn.TorchLayer``
    without changing the public contract.

    Args:
        device: Name of the PennyLane device the cost model is derived from.
            Used only for reporting; the actual simulator computation does
            not currently spin up a PennyLane device.
    """

    profile: BackendProfile = BackendProfile(
        name="pennylane.sim",
        kind="quantum",
        energy_pj_per_mac=_PENNYLANE_DEFAULT_ENERGY,
        latency_us_per_op=_PENNYLANE_DEFAULT_LATENCY,
        queue_us=0.0,
        supports_autograd=True,
        supports_batching=True,
        max_dim=None,
        notes=(
            "PennyLane-flavored simulator. Numerically identical to F.linear "
            "but advertises a quantum-simulator cost profile. Future versions "
            "will optionally route forward through a real qml.qnn.TorchLayer."
        ),
    )

    def __init__(self, device: str = "default.qubit") -> None:
        self._device_name = device

    def is_available(self) -> bool:
        return _safe_has_spec("pennylane")

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, weight, bias)


class IBMQuantumSimBackend(Backend):
    """IBM Quantum-flavored backend (via PennyLane's ``qiskit.remote`` plugin).

    Same numerical behavior as :class:`PennyLaneSimBackend` with a different
    cost profile (real QPU queue time and energy estimate). Suitable for
    Stacked Graph reasoning about IBM Quantum offload before the user has
    real credentials configured.
    """

    profile: BackendProfile = BackendProfile(
        name="ibm.quantum",
        kind="quantum",
        energy_pj_per_mac=_IBM_ENERGY,
        latency_us_per_op=_IBM_LATENCY,
        queue_us=_IBM_QUEUE_US,
        supports_autograd=False,  # Real QPUs lack autograd.
        supports_batching=False,
        max_dim=156,  # Heron QPU mode count (rough upper bound).
        notes=(
            "IBM Quantum-grade cost profile (Heron, Open Plan defaults). "
            "Numerically identical to F.linear until cloud credentials are "
            "wired up — then forward routes to PennyLane's qiskit.remote device."
        ),
    )

    def is_available(self) -> bool:
        # We don't ping IBM; just check the plugin is installed.
        return all(_safe_has_spec(name) for name in ("pennylane", "pennylane_qiskit"))

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, weight, bias)


class AWSBraketSimBackend(Backend):
    """AWS Braket-flavored backend (via PennyLane's ``braket.aws.qubit`` plugin).

    Note (mid-2026): Xanadu's photonic hardware was removed from AWS Braket's
    public catalog. The QPUs currently accessible through Braket are IonQ,
    Rigetti, IQM, AQT and QuEra. This backend's cost profile reflects those
    qubit-based QPUs; for photonic, see :class:`QuandelaSimBackend`.
    """

    profile: BackendProfile = BackendProfile(
        name="aws.braket",
        kind="quantum",
        energy_pj_per_mac=_BRAKET_ENERGY,
        latency_us_per_op=_BRAKET_LATENCY,
        queue_us=_BRAKET_QUEUE_US,
        supports_autograd=False,
        supports_batching=False,
        max_dim=None,
        notes=(
            "AWS Braket-grade cost profile (SV1 simulator + IonQ/Rigetti/IQM/AQT/QuEra QPUs). "
            "Xanadu hardware no longer in Braket catalog as of 2026. "
            "Numerically identical to F.linear until cloud credentials are wired up."
        ),
    )

    def is_available(self) -> bool:
        # Braket SDK exposes itself as `braket` (not `amazon.braket`); also check
        # the legacy nested name just in case.
        return _safe_has_spec("pennylane") and (
            _safe_has_spec("braket") or _safe_has_spec("amazon.braket")
        )

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, weight, bias)


# Auto-register all three. Selector handles availability filtering.
register_backend("pennylane.sim", PennyLaneSimBackend, overwrite=True)
register_backend("ibm.quantum", IBMQuantumSimBackend, overwrite=True)
register_backend("aws.braket", AWSBraketSimBackend, overwrite=True)
