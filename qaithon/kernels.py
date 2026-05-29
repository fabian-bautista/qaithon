"""Genuine quantum / photonic linear kernels.

These compute ``y = x @ W.T + b`` (a standard nn.Linear) by running the **real**
algorithm — never ``F.linear``:

* :func:`photonic_linear` — encodes each input vector as a single photon's
  amplitudes, sends it through a linear-optical interferometer that realises a
  unitary *dilation* of ``W``, and reads the output amplitudes (Perceval SLOS).
  Verified: reproduces ``W·x`` for arbitrary real ``W`` at machine precision.

* :func:`quantum_linear` — amplitude-encodes the input into qubits, applies the
  same unitary dilation as a gate, and reads the output amplitudes (Qiskit
  statevector). Exact in simulation.

Both work by *unitary dilation* (Halmos): any contraction ``W`` (‖W‖≤1) embeds
in the top-left block of a unitary ``U`` of twice the size. A real ``W`` yields a
real orthogonal ``U``, so the output is real.

THE COMPUTE IS THE CIRCUIT. The only classical steps are normalisation and the
unavoidable encode/decode at the boundaries. Size is bounded by the simulator /
hardware (photonic SLOS caps at 256 modes → dim 128).
"""

from __future__ import annotations

import importlib.util

import numpy as np
import torch

from qaithon._logging import get_logger

__all__ = ["photonic_linear", "quantum_linear", "PHOTONIC_MAX_DIM"]

logger = get_logger(__name__)

# Perceval SLOS caps Fock-state size at 256 modes; the dilation uses 2·dim modes.
PHOTONIC_MAX_DIM = 128


def _dilate(W: np.ndarray) -> tuple[np.ndarray, float, int]:
    """Return (U, scale, n) where U (2n×2n) is unitary and its top-left n×n block
    is ``W/scale`` (a contraction). ``W`` is padded to square first."""
    from scipy.linalg import sqrtm

    out_f, in_f = W.shape
    n = max(out_f, in_f)
    Wsq = np.zeros((n, n), dtype=float)
    Wsq[:out_f, :in_f] = W
    scale = np.linalg.norm(Wsq, 2) * 1.01 + 1e-12
    Wc = Wsq / scale
    I = np.eye(n)
    A = sqrtm(I - Wc @ Wc.T).real
    B = sqrtm(I - Wc.T @ Wc).real
    U = np.block([[Wc, A], [B, -Wc.T]]).astype(complex)
    return U, scale, n


def _prep(x: torch.Tensor, W: torch.Tensor) -> tuple[np.ndarray, np.ndarray, int, int]:
    Wn = W.detach().cpu().numpy().astype(float)
    out_f, in_f = Wn.shape
    if max(out_f, in_f) > PHOTONIC_MAX_DIM:
        logger.warning(
            "Genuine kernel on dim %d exceeds the photonic simulator's reach "
            "(%d); this is for the quantum path or will be slow/limited.",
            max(out_f, in_f),
            PHOTONIC_MAX_DIM,
        )
    return Wn, x.detach().cpu().numpy().astype(float), out_f, in_f


def photonic_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """y = x @ W.T (+ b), computed by a real linear-optical circuit (Perceval)."""
    if not (importlib.util.find_spec("perceval") and importlib.util.find_spec("merlin")):
        raise RuntimeError("perceval/merlin required for the photonic kernel.")
    import perceval as pcvl
    from perceval.simulators import Simulator

    Wn, xn, out_f, in_f = _prep(x, weight)
    U, scale, n = _dilate(Wn)
    circuit = pcvl.Unitary(pcvl.Matrix(U))
    sim = Simulator(pcvl.BackendFactory.get_backend("SLOS"))
    sim.set_circuit(circuit)

    flat = xn.reshape(-1, in_f)
    out = np.zeros((flat.shape[0], out_f), dtype=float)
    for r, xv in enumerate(flat):
        vec = np.zeros(2 * n)
        vec[:in_f] = xv
        nrm = np.linalg.norm(vec)
        if nrm < 1e-12:
            continue
        vec = vec / nrm
        sv = pcvl.StateVector()
        for i in range(2 * n):
            if abs(vec[i]) > 1e-12:
                e = [0] * (2 * n)
                e[i] = 1
                sv += complex(vec[i]) * pcvl.BasicState(e)
        ev = sim.evolve(sv)
        y = np.array(
            [
                complex(ev[pcvl.BasicState([1 if k == i else 0 for k in range(2 * n)])]).real
                for i in range(out_f)
            ]
        )
        out[r] = y * scale * nrm

    res = torch.from_numpy(out).to(x.dtype).reshape(*x.shape[:-1], out_f)
    if bias is not None:
        res = res + bias.detach().cpu().to(x.dtype)
    return res


def quantum_linear(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """y = x @ W.T (+ b), computed on a real qubit circuit (Qiskit statevector)."""
    if not importlib.util.find_spec("qiskit"):
        raise RuntimeError("qiskit required for the quantum kernel.")
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Statevector

    Wn, xn, out_f, in_f = _prep(x, weight)
    U, scale, n = _dilate(Wn)
    dim = 2 * n
    q = max(1, int(np.ceil(np.log2(dim))))
    pad = 2**q
    Upad = np.eye(pad, dtype=complex)
    Upad[:dim, :dim] = U
    gate = UnitaryGate(Upad)

    flat = xn.reshape(-1, in_f)
    out = np.zeros((flat.shape[0], out_f), dtype=float)
    for r, xv in enumerate(flat):
        amp = np.zeros(pad, dtype=complex)
        amp[:in_f] = xv
        nrm = np.linalg.norm(amp)
        if nrm < 1e-12:
            continue
        amp = amp / nrm
        sv = Statevector(amp).evolve(gate)
        y = sv.data[:out_f].real
        out[r] = y * scale * nrm

    res = torch.from_numpy(out).to(x.dtype).reshape(*x.shape[:-1], out_f)
    if bias is not None:
        res = res + bias.detach().cpu().to(x.dtype)
    return res
