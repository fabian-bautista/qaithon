"""Genuine quantum experiments on REAL IBM hardware (IBM Heron).

This script reproduces the real-hardware results reported in
the project README → "Results — measured on real hardware":

1. **AI inference on a QPU** — train a linear classifier on Iris and run the
   inference of its test set on a physical IBM Heron processor (3 qubits).
2. **Collapse curve** — run a genuine dense matmul at growing dimensions
   (2 → 32, i.e. 2 → 6 qubits) and measure how fidelity falls as the dilation
   unitary's gate count explodes.

Everything runs through the library itself —
``IBMHeronBackend(mode="execute").matmul(x, W)`` — not via raw Qiskit.

Requirements
------------
* An IBM Quantum account/token configured (``qaithon.set_ibm_token(...)`` or the
  ``QISKIT_IBM_TOKEN`` environment variable). The free "open" plan is enough.
* ``pip install -e ".[pennylane]"`` plus ``qiskit-ibm-runtime``.

⚠️  This submits **real jobs** to a physical QPU and consumes your (free) monthly
quota. Each job takes seconds to minutes depending on the queue. The collapse
curve at dim 32 transpiles a 64×64 dense unitary (~24k gates) and is slow.

Limitation: measurement yields ``|amplitude|^2`` only, so output *magnitudes* are
measured on hardware while *signs* are reconstructed from the ideal (full sign
recovery needs a Hadamard test — planned).
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from qaithon.backends.ibm_heron import IBMHeronBackend

warnings.filterwarnings("ignore")
torch.manual_seed(0)
np.random.seed(0)


def _load_dataset() -> tuple[np.ndarray, np.ndarray, str]:
    """Iris if scikit-learn is available, else synthetic 3-class 4D blobs."""
    try:
        from sklearn.datasets import load_iris

        data = load_iris()
        return data.data.astype("float32"), data.target.astype("int64"), "Iris"
    except Exception:  # noqa: BLE001
        centers = np.array([[2, 2, 0, 0], [-2, -2, 2, 0], [0, 0, -2, 2]], float)
        x = np.vstack([c + 0.6 * np.random.randn(50, 4) for c in centers])
        y = np.repeat([0, 1, 2], 50)
        return x.astype("float32"), y.astype("int64"), "synthetic 3-class 4D blobs"


def experiment_classifier() -> None:
    """Train a linear classifier and run its inference on a real QPU."""
    x, y, name = _load_dataset()
    x = (x - x.mean(0)) / (x.std(0) + 1e-8)
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]
    n_train = int(0.7 * len(x))
    x_tr, y_tr, x_te, y_te = x[:n_train], y[:n_train], x[n_train:], y[n_train:]
    keep = np.concatenate([np.where(y_te == c)[0][:7] for c in (0, 1, 2)])
    x_te, y_te = x_te[keep], y_te[keep]

    clf = nn.Linear(4, 3)
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    x_tr_t, y_tr_t = torch.tensor(x_tr), torch.tensor(y_tr)
    for _ in range(400):
        opt.zero_grad()
        F.cross_entropy(clf(x_tr_t), y_tr_t).backward()
        opt.step()

    x_te_t = torch.tensor(x_te)
    acc_classic = (clf(x_te_t).argmax(1).numpy() == y_te).mean()

    print(f"\n=== EXP 1: AI classifier inference on IBM Heron (3 qubits) ===")
    print(f"Dataset: {name} | test={len(y_te)} samples | classical acc={acc_classic:.1%}")
    backend = IBMHeronBackend(mode="execute", shots=4096)
    logits = backend.matmul(x_te_t, clf.weight.detach(), clf.bias.detach())
    acc_hw = (logits.argmax(1).numpy() == y_te).mean()
    info = backend.last_execute
    print(f"  device={info['device']} qubits={info['n_qubits']} circuits={info['n_rows']}"
          f" mean_fidelity={info['fidelity']:.3f}")
    print(f"  accuracy on PHYSICAL quantum hardware: {acc_hw:.1%} (classical {acc_classic:.1%})")


def experiment_collapse_curve(dims: tuple[int, ...] = (2, 4, 8, 16, 32)) -> None:
    """Measure fidelity vs dimension for a genuine dense matmul on real hardware."""
    print(f"\n=== EXP 2: collapse curve (genuine matmul vs size) ===")
    print(f"{'dim':>4} {'qubits':>7} {'gates':>7} {'fidelity':>9} {'rel_err':>8} {'lat(s)':>7}")
    for dim in dims:
        w = torch.randn(dim, dim)
        x = torch.randn(1, dim)
        backend = IBMHeronBackend(mode="execute", shots=2048)
        y = backend.matmul(x, w)
        ref = F.linear(x, w)
        rel = float((y - ref).norm() / ref.norm())
        info = backend.last_execute
        print(f"{dim:>4} {info['n_qubits']:>7} {info['n_gates']:>7} "
              f"{info['fidelity']:>9.3f} {rel:>8.2f} {info['latency_s']:>7.1f}")


def _expanded_classifier(in_feat: int, n_pca: int | None, n_classes: int, epochs: int):
    """Train a linear classifier on Iris/Digits with `in_feat` input features."""
    if n_pca is not None:  # Digits → PCA
        from sklearn.datasets import load_digits
        from sklearn.decomposition import PCA

        data = load_digits()
        x = PCA(n_components=n_pca, random_state=0).fit_transform(data.data)
        y = data.target.astype("int64")
    else:  # Iris expanded to 8 features (originals + squares)
        from sklearn.datasets import load_iris

        d = load_iris()
        x = np.concatenate([d.data, d.data**2], axis=1)
        y = d.target.astype("int64")
    x = ((x - x.mean(0)) / (x.std(0) + 1e-8)).astype("float32")
    idx = np.random.permutation(len(x))
    x, y = x[idx], y[idx]
    n_train = int(0.7 * len(x))
    clf = nn.Linear(x.shape[1], n_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=0.05)
    x_tr, y_tr = torch.tensor(x[:n_train]), torch.tensor(y[:n_train])
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(clf(x_tr), y_tr).backward()
        opt.step()
    per_class = 7 if n_classes == 3 else 2
    keep = np.concatenate([np.where(y[n_train:] == c)[0][:per_class] for c in range(n_classes)])
    x_te = torch.tensor(x[n_train:][keep])
    y_te = y[n_train:][keep]
    return clf, x_te, y_te


def experiment_mitigation() -> None:
    """A/B test software error mitigation on larger 4- and 5-qubit classifiers."""
    print(f"\n=== EXP 3: software error mitigation (A/B) ===")
    cases = [
        ("Iris-8feat 8->3 (4 qubits)", None, None, 3, 500),
        ("Digits PCA-16 16->10 (5 qubits)", 16, 16, 10, 600),
    ]
    for name, n_pca, in_feat, n_classes, epochs in cases:
        clf, x_te, y_te = _expanded_classifier(in_feat or 8, n_pca, n_classes, epochs)
        acc_classic = (clf(x_te).argmax(1).numpy() == y_te.numpy()).mean()
        print(f"\n{name} | classical acc={acc_classic:.1%}")
        for mit in (False, True):
            backend = IBMHeronBackend(mode="execute", shots=2048, mitigation=mit)
            logits = backend.matmul(x_te, clf.weight.detach(), clf.bias.detach())
            acc = (logits.argmax(1).numpy() == y_te.numpy()).mean()
            info = backend.last_execute
            tag = "mitigated  " if mit else "unmitigated"
            print(f"  {tag}: qubits={info['n_qubits']} gates~{info['n_gates']} "
                  f"fidelity={info['fidelity']:.3f} accuracy={acc:.1%}")


if __name__ == "__main__":
    experiment_classifier()
    experiment_collapse_curve()
    experiment_mitigation()
    print("\nDone. See the project README for the interpretation.")
