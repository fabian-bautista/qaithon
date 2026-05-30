"""Genuine photonic matmul on REAL hardware (Quandela Belenos).

The photonic counterpart of ``run_vqc_hardware.py`` — same kind of experiment as
on quantum (a genuine matmul, fidelity of the measured result vs the ideal), but
on a physical photonic QPU. Reproduces the photonic vs quantum comparison in the
project README.

This is a THIN wrapper over the library: it just calls
``QuandelaBelenosBackend(mode="execute").matmul(x, W)``. All the photonic logic
(single-photon encoding, the unitary-dilation interferometer, sampling and
reconstruction) lives in the backend.

Result on Belenos (qpu:belenos): dim2 0.998, dim4 0.977, dim6 0.953 — graceful,
but caps at 12 modes (dim 6). Quantum (IBM Heron): dim2 0.998, dim4 0.967,
dim8 0.705. Opposite limits.

Requirements: a Quandela token (``qaithon.set_quandela_token(...)`` / QUANDELA_TOKEN)
and ``perceval``/``merlin``. Submits real jobs. Pass ``platform_name="local:slos"``
to run on a local simulator instead (no cloud, no credits).
"""

from __future__ import annotations

import warnings

import torch
import torch.nn.functional as F

from qaithon.backends.quandela_belenos import QuandelaBelenosBackend

warnings.filterwarnings("ignore")
torch.manual_seed(0)


def main(dims=(2, 4, 6), platform_name: str = "qpu:belenos", shots: int = 2000) -> None:
    print(f"=== Genuine photonic matmul on {platform_name} ===")
    print(f"{'dim':>4} {'modes':>6} {'fidelity':>9} {'rel_err':>8}")
    for dim in dims:
        weight = torch.randn(dim, dim)
        x = torch.randn(1, dim)
        backend = QuandelaBelenosBackend(
            mode="execute", platform_name=platform_name, shots=shots
        )
        y = backend.matmul(x, weight)                       # genuine photonic matmul
        rel = float((y - F.linear(x, weight)).norm() / F.linear(x, weight).norm())
        info = backend.last_execute
        print(f"{dim:>4} {info['modes']:>6} {info['fidelity']:>9.3f} {rel:>8.3f}")
    print("\nQuantum (IBM Heron) for comparison: dim2 0.998, dim4 0.967, dim8 0.705.")


if __name__ == "__main__":
    main()
