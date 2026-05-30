"""Data re-uploading VQC on Digits 10-class @ 5 qubits, on REAL IBM hardware.

Reproduces the "changing the algorithm" result in the project README: the
same 10-class handwritten-Digits problem where a dense-matmul circuit collapsed
to chance (20%) at 5 qubits, solved instead by a SHALLOW quantum-native circuit
(:class:`qaithon.ReuploadingClassifier`'s circuit) that retains most of its
accuracy on the QPU.

Pipeline: train on the simulator (free) → VERIFY the Qiskit circuit matches
PennyLane on Aer (no quota) → run the test-set inference on real ``ibm_marrakesh``
(one batched job per depth). The quantum circuit is the trainable feature map; a
small classical linear head reads its measured probabilities (standard QML).

Requirements: an IBM Quantum token (free open plan), ``qaithon[pennylane]``,
``qiskit-ibm-runtime``, ``qiskit-aer``, ``scikit-learn``. Submits real jobs and
consumes (free) quota.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import SamplerV2
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

from qaithon.backends.ibm_heron import IBMHeronBackend

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)

N, NF = 5, 10  # 5 qubits, 10 PCA features (= 2 angle slots per qubit)


def _data(per_class: int = 8):
    d = load_digits()
    x = PCA(n_components=NF, random_state=0).fit_transform(d.data).astype("float32")
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    y = d.target.astype("int64")
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]
    n_tr = int(0.7 * len(x))
    xtr, ytr, xte, yte = x[:n_tr], y[:n_tr], x[n_tr:], y[n_tr:]
    keep = np.concatenate([np.where(yte == c)[0][:per_class] for c in range(10)])
    return torch.tensor(xtr), torch.tensor(ytr), torch.tensor(xte[keep]), yte[keep]


def _train(L: int, xtr, ytr):
    dev = qml.device("default.qubit", wires=N)

    @qml.qnode(dev, interface="torch")
    def circ(inputs, enc, var):
        for l in range(L):
            for q in range(N):
                qml.RY(inputs[..., 2 * q] * enc[l, q, 0], wires=q)
                qml.RZ(inputs[..., 2 * q + 1] * enc[l, q, 1], wires=q)
            for q in range(N):
                qml.CNOT(wires=[q, (q + 1) % N])
            for q in range(N):
                qml.RY(var[l, q, 0], wires=q)
                qml.RZ(var[l, q, 1], wires=q)
        return qml.probs(wires=range(N))

    init = {  # faithful init (enc≈1, var≈0) — scrambled init prevents learning
        "enc": (lambda t, v=torch.ones(L, N, 2) + 0.1 * torch.randn(L, N, 2): t.data.copy_(v)),
        "var": (lambda t, v=0.1 * torch.randn(L, N, 2): t.data.copy_(v)),
    }
    ql = qml.qnn.TorchLayer(circ, {"enc": (L, N, 2), "var": (L, N, 2)}, init_method=init)
    model = nn.Sequential(ql, nn.Linear(2**N, 10))
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(400):
        opt.zero_grad()
        F.cross_entropy(model(xtr), ytr).backward()
        opt.step()
    w = dict(model[0].named_parameters())
    return model, circ, w["enc"].detach().numpy(), w["var"].detach().numpy()


def _qiskit_circuit(x, enc, var, L):
    qc = QuantumCircuit(N, N)
    for l in range(L):
        for q in range(N):
            qc.ry(float(x[2 * q] * enc[l, q, 0]), q)
            qc.rz(float(x[2 * q + 1] * enc[l, q, 1]), q)
        for q in range(N):
            qc.cx(q, (q + 1) % N)
        for q in range(N):
            qc.ry(float(var[l, q, 0]), q)
            qc.rz(float(var[l, q, 1]), q)
    qc.measure(range(N), range(N))
    return qc


def _counts_to_probs(counts, reverse):
    p = np.zeros(2**N)
    tot = sum(counts.values())
    for b, c in counts.items():
        p[int((b[::-1] if reverse else b), 2)] = c / tot
    return p


def main(depths=(3, 6), per_class: int = 8) -> None:
    xtr, ytr, xte, yte = _data(per_class)
    aer = AerSimulator()
    print(f"=== Data re-uploading VQC | Digits 10-class | 5 qubits | test={len(yte)} ===")
    for L in depths:
        model, circ, enc, var = _train(L, xtr, ytr)
        with torch.no_grad():
            sim_acc = (model(xte).argmax(1).numpy() == yte).mean()
        # verify Qiskit == PennyLane on Aer (no quota)
        w = dict(model[0].named_parameters())
        p_pl = circ(xte[0], w["enc"].detach(), w["var"].detach()).detach().numpy()
        ref = aer.run(transpile(_qiskit_circuit(xte[0], enc, var, L), aer), shots=20000).result().get_counts()
        err_f = np.abs(_counts_to_probs(ref, False) - p_pl).sum()
        err_r = np.abs(_counts_to_probs(ref, True) - p_pl).sum()
        reverse = err_r < err_f
        if min(err_f, err_r) > 0.08:
            print(f"L={L}: ABORT — Qiskit/PennyLane mismatch ({min(err_f, err_r):.3f})")
            continue
        # run inference on real hardware (one batched job)
        backend = IBMHeronBackend(mode="execute")._pick_backend()
        circuits = transpile([_qiskit_circuit(x, enc, var, L) for x in xte], backend, optimization_level=3)
        res = SamplerV2(mode=backend).run(circuits, shots=2048).result()
        probs = []
        for i in range(len(xte)):
            try:
                cnt = res[i].data.c.get_counts()
            except AttributeError:
                cnt = res[i].data.meas.get_counts()
            probs.append(_counts_to_probs(cnt, reverse))
        with torch.no_grad():
            hw_acc = (model[1](torch.tensor(np.array(probs), dtype=torch.float32)).argmax(1).numpy() == yte).mean()
        print(f"L={L}: sim={sim_acc:.1%}  hardware={hw_acc:.1%}  "
              f"retention={hw_acc / max(sim_acc, 1e-9):.0%}  (dense matmul was 20%, classical ~95%)")


if __name__ == "__main__":
    main()
