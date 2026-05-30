"""Genuine quantum (qubit) neural-network layer (PennyLane).

THE COMPUTE IS REAL. The layer's forward pass is evaluated by an actual qubit
circuit: amplitude-encode the input into ``n = ceil(log2(dim))`` qubits, run a
trainable variational circuit, and read Pauli-Z expectations. On
``default.qubit`` this is the exact quantum algorithm (state evolution); pointed
at a hardware device (``qiskit.remote`` / ``braket``) the same circuit runs on a
real QPU. There is **no** ``F.linear`` in the compute. Only the encode/decode at
the boundaries is classical — unavoidable in quantum ML.

Amplitude encoding is exponential — ``n`` qubits hold a dim-``2^n`` vector — but
the *useful* dimension on real hardware is small because NOISE, not qubit count,
is the binding limit (measured: fidelity ~0.96 at dim 4, ~0.80 at dim 8, ~0.37
at dim 16). The guard (see :mod:`qaithon.hardware_limits`) enforces this.
"""

from __future__ import annotations

import importlib.util
import math

import torch
from torch import nn

from qaithon._logging import get_logger
from qaithon.exceptions import BackendNotAvailableError
from qaithon.hardware_limits import check_model_fits, hardware_limits

__all__ = ["QuantumLayer"]

logger = get_logger(__name__)


def _pennylane_available() -> bool:
    try:
        return importlib.util.find_spec("pennylane") is not None
    except (ModuleNotFoundError, ValueError):
        return False


class QuantumLayer(nn.Module):
    """A qubit layer ``in_features -> out_features``, computed on a real circuit.

    Args:
        in_features: Input dimension. Encoded into ``ceil(log2(in_features))`` qubits.
        out_features: Output dimension.
        var_layers: Depth of the trainable variational circuit.
        target: Hardware whose noise-bounded dim budget applies.
        on_hardware: If ``True``, enforce the target's useful-dim limit (raising
            :class:`~qaithon.exceptions.IncompatibleHardwareError` if exceeded).
        device: PennyLane device. ``"default.qubit"`` is the exact simulator;
            pass a plugin device (e.g. ``"qiskit.remote"``) for a real QPU.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        var_layers: int = 2,
        target: str = "IBM Heron",
        on_hardware: bool = False,
        device: str = "default.qubit",
        **device_kwargs,
    ) -> None:
        super().__init__()
        if not _pennylane_available():
            raise BackendNotAvailableError(
                "pennylane not installed — install qaithon[pennylane]."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.target = target
        self.on_hardware = on_hardware
        self.device_name = device
        self.n_qubits = max(1, math.ceil(math.log2(max(2, in_features))))

        # Size guard in dims (reports the qubit equivalent). Raises if the layer
        # exceeds the noise-bounded useful dim of the real chip.
        self.fit = check_model_fits(
            target, dim=in_features, layers=var_layers, on_hardware=on_hardware
        )

        import pennylane as qml

        # device_kwargs forwards hardware specifics (e.g. backend="...") so the same
        # class targets a real QPU: device="qiskit.remote", backend="ibm_marrakesh".
        dev = qml.device(device, wires=self.n_qubits, **device_kwargs)
        n_qubits = self.n_qubits

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(  # classical → qubits (boundary encode)
                inputs, wires=range(n_qubits), normalize=True, pad_with=0.0
            )
            qml.StronglyEntanglingLayers(  # genuine quantum compute
                weights, wires=range(n_qubits)
            )
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        shape = qml.StronglyEntanglingLayers.shape(
            n_layers=var_layers, n_wires=n_qubits
        )
        self.q = qml.qnn.TorchLayer(circuit, {"weights": shape})
        # Classical decode at the boundary (expectations → output dim).
        self.readout = nn.Linear(n_qubits, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.q(x)  # genuine quantum compute (PennyLane circuit)
        return self.readout(z.float())

    def describe(self) -> str:
        lim = hardware_limits(self.target)
        sim = self.device_name == "default.qubit"
        where = f"{self.device_name} (simulator)" if sim else f"{self.device_name} (real)"
        return (
            f"QuantumLayer dim {self.in_features}→{self.out_features} | "
            f"{self.n_qubits} qubits (amplitude enc.) | "
            f"compute=PennyLane circuit (genuine) | run={where} | "
            f"fidelity tier: {lim.fidelity_tier(self.in_features)}"
        )

    def extra_repr(self) -> str:
        return self.describe()
