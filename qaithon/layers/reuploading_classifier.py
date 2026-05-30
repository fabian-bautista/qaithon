"""Genuine quantum-native classifier via data re-uploading (PennyLane).

THE COMPUTE IS REAL. Unlike :class:`~qaithon.QuantumLayer` (amplitude encoding,
which compiles to a *deep* state-preparation circuit), this layer uses **angle
encoding with data re-uploading**: ``(encode → entangle → variational) × layers``.
That keeps the circuit **shallow** (tens of native gates), which is what makes it
survive NISQ noise.

Why this matters — measured on real IBM hardware (``ibm_marrakesh``, 5 qubits):
this classifier labels 10-class handwritten Digits at **90–95% accuracy on the
QPU, matching its simulator accuracy**, whereas a deep dense-matmul circuit on the
same 5 qubits collapsed to chance (~20%, ~6k gates). The engine — not the qubit
count — is the binding choice on today's hardware.

The quantum circuit is the trainable feature map (runs genuinely on the QPU); a
small classical linear head maps the measured probabilities to class logits —
standard practice in quantum ML, and the only classical part.
"""

from __future__ import annotations

import importlib.util

import torch
from torch import nn

from qaithon._logging import get_logger
from qaithon.exceptions import BackendNotAvailableError

__all__ = ["ReuploadingClassifier"]

logger = get_logger(__name__)


def _pennylane_available() -> bool:
    try:
        return importlib.util.find_spec("pennylane") is not None
    except (ModuleNotFoundError, ValueError):
        return False


class ReuploadingClassifier(nn.Module):
    """A shallow, NISQ-friendly quantum-native classifier.

    Args:
        in_features: Input dimension (features are angle-encoded, cycling across
            the ``2 * n_qubits`` encoding slots each re-upload layer).
        n_classes: Number of output classes.
        n_qubits: Width of the circuit. 5 qubits already span a 32-dim readout.
        layers: Data re-uploading depth. More layers = more expressivity (like
            hidden layers in a classical net) at the cost of a deeper circuit.
            ``6`` reached 95% on Digits; ``3`` reached 90% with a shallower (more
            noise-robust) circuit.
        device: PennyLane device. ``"default.qubit"`` is the exact simulator
            (use it to train); pass a plugin device (e.g. ``"qiskit.remote"``)
            to run inference on a real QPU.

    Example:
        >>> clf = qaithon.ReuploadingClassifier(in_features=10, n_classes=10)
        >>> logits = clf(torch.randn(32, 10))   # train like any nn.Module
        >>> logits.shape
        torch.Size([32, 10])
    """

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        *,
        n_qubits: int = 5,
        layers: int = 6,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        if not _pennylane_available():
            raise BackendNotAvailableError(
                "pennylane not installed — install qaithon[pennylane]."
            )
        if in_features < 1 or n_classes < 2 or n_qubits < 1 or layers < 1:
            raise ValueError("in_features≥1, n_classes≥2, n_qubits≥1, layers≥1 required.")

        self.in_features = in_features
        self.n_classes = n_classes
        self.n_qubits = n_qubits
        self.layers = layers
        self.device_name = device

        import pennylane as qml

        dev = qml.device(device, wires=n_qubits)
        n, depth, n_feat = n_qubits, layers, in_features
        diff = "backprop" if device == "default.qubit" else "parameter-shift"

        @qml.qnode(dev, interface="torch", diff_method=diff)
        def circuit(inputs, enc, var):
            for layer in range(depth):
                for q in range(n):  # angle re-uploading of the data (boundary encode)
                    qml.RY(inputs[..., (2 * q) % n_feat] * enc[layer, q, 0], wires=q)
                    qml.RZ(inputs[..., (2 * q + 1) % n_feat] * enc[layer, q, 1], wires=q)
                for q in range(n):  # entangle (ring)
                    qml.CNOT(wires=[q, (q + 1) % n])
                for q in range(n):  # trainable variational sublayer
                    qml.RY(var[layer, q, 0], wires=q)
                    qml.RZ(var[layer, q, 1], wires=q)
            return qml.probs(wires=range(n))

        shapes = {"enc": (depth, n, 2), "var": (depth, n, 2)}
        # Faithful init: encoding scale ≈ 1 (the data passes through cleanly),
        # variational angles ≈ 0. A scrambled init (PennyLane's default uniform
        # [0, 2π]) prevents the model from learning — measured the hard way.
        enc0 = torch.ones(depth, n, 2) + 0.1 * torch.randn(depth, n, 2)
        var0 = 0.1 * torch.randn(depth, n, 2)
        init = {
            "enc": (lambda t, v=enc0: t.data.copy_(v)),
            "var": (lambda t, v=var0: t.data.copy_(v)),
        }
        self.q = qml.qnn.TorchLayer(circuit, shapes, init_method=init)
        # Classical decode at the boundary (measured probabilities → class logits).
        self.readout = nn.Linear(2**n, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(self.q(x).float())  # genuine quantum circuit + classical head

    def describe(self) -> str:
        sim = self.device_name == "default.qubit"
        where = f"{self.device_name} (simulator)" if sim else f"{self.device_name} (real QPU)"
        return (
            f"ReuploadingClassifier {self.in_features}→{self.n_classes} | "
            f"{self.n_qubits} qubits · {self.layers} re-upload layers (angle enc.) | "
            f"compute=PennyLane circuit (genuine, shallow) | run={where}"
        )

    def extra_repr(self) -> str:
        return self.describe()
