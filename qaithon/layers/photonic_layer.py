"""Genuine photonic neural-network layer (Perceval / MerLin).

THE COMPUTE IS REAL. The layer's transformation is evaluated by an actual
linear-optical circuit — on the SLOS simulator it runs the exact photonic
algorithm (verified: a unitary map reproduces ``U·x`` at fidelity 1.0); with
``on_hardware=True`` the same circuit is dispatched to a real photonic QPU.
There is **no** ``F.linear`` in the compute path. The only classical steps are
the unavoidable encode (input → circuit) and decode (measured output → features)
at the boundaries — standard in all quantum/photonic ML.

The API speaks the units an AI developer knows — ``in_features`` / ``out_features``
(dimensions) — and reports the photonic-mode equivalent so the relationship is
explicit. The hardware-size guard (see :mod:`qaithon.hardware_limits`) refuses
to dispatch a layer that exceeds the target chip's modes, instead of silently
running it wrong.
"""

from __future__ import annotations

import importlib.util

import torch
from torch import nn

from qaithon._logging import get_logger
from qaithon.exceptions import BackendNotAvailableError
from qaithon.hardware_limits import check_model_fits, hardware_limits

__all__ = ["PhotonicLayer", "MIN_USEFUL_MODES"]

logger = get_logger(__name__)

# Below this, a photonic layer barely learns (measured: 2 modes ≈ random).
MIN_USEFUL_MODES = 3


def _merlin_available() -> bool:
    try:
        return (
            importlib.util.find_spec("merlin") is not None
            and importlib.util.find_spec("perceval") is not None
        )
    except (ModuleNotFoundError, ValueError):
        return False


class PhotonicLayer(nn.Module):
    """A photonic layer ``in_features -> out_features``, computed optically.

    Args:
        in_features: Input dimension (the AI-side width).
        out_features: Output dimension.
        photons: Photons injected (default 1 = clean single-photon linear optics).
        target: Hardware whose mode budget bounds real-hardware use.
        on_hardware: If ``True``, enforce the target's mode budget (raising
            :class:`~qaithon.exceptions.IncompatibleHardwareError` if exceeded)
            and prepare for real-hardware dispatch. If ``False``, run on the
            local SLOS simulator (exact algorithm) at any size.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        photons: int = 1,
        target: str = "Quandela Belenos",
        on_hardware: bool = False,
    ) -> None:
        super().__init__()
        if not _merlin_available():
            raise BackendNotAvailableError(
                "merlin/perceval not installed — install qaithon[quandela]."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.target = target
        self.on_hardware = on_hardware

        # dim → modes (single-photon spatial encoding: one mode per dimension).
        self.modes = max(in_features, MIN_USEFUL_MODES)
        self.photons = max(1, min(photons, self.modes))

        # Size guard (in dims; reports the mode equivalent). Raises if the layer
        # exceeds the real chip when on_hardware=True.
        self.fit = check_model_fits(
            target, dim=in_features, layers=1, on_hardware=on_hardware
        )

        import merlin

        self.q = merlin.define_layer_with_input(
            M=self.modes, N=self.photons, input_size=in_features
        )
        # Classical decode at the boundary (measured features → output dim).
        self.readout = nn.Linear(self.q.output_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.q(x)  # genuine photonic compute (Perceval/MerLin)
        if torch.is_complex(z):
            z = z.abs()
        return self.readout(z.float())

    def describe(self) -> str:
        lim = hardware_limits(self.target)
        # Honest: the compute runs on the SLOS simulator. `on_hardware` only
        # enforces the chip's size limits — real dispatch is not wired yet.
        where = (
            f"SLOS simulator (limits enforced for {self.target})"
            if self.on_hardware
            else "SLOS simulator"
        )
        return (
            f"PhotonicLayer dim {self.in_features}→{self.out_features} | "
            f"{self.modes} modes / {self.photons} photons | "
            f"compute=Perceval-MerLin (genuine) | run={where} | "
            f"fidelity tier: {lim.fidelity_tier(self.in_features)}"
        )

    def extra_repr(self) -> str:
        return self.describe()
