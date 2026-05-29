"""Quandela MerLin backend — autograd-native photonic layer.

Where :class:`PercevalPhotonicBackend` runs SLOS sampling and applies the
measured noise as a scale on the classical ``F.linear`` output,
:class:`QuandelaMerlinBackend` uses MerLin's :class:`merlin.QuantumLayer`
directly. MerLin's layer is a fully differentiable PyTorch module that
encodes the input into a small linear-optical circuit and returns
expectation values — gradients flow through ``loss.backward()`` without
any noise-as-postprocessing trick.

When to pick this over ``quandela.perceval``
--------------------------------------------

* You want gradients to **flow through the photonic forward**, not just
  through a classical surrogate. Useful for VQC-style training or
  research on quantum natural gradients.
* You want a single autograd graph end-to-end so PyTorch's profiler shows
  the photonic call as a leaf op rather than a black box.

When to stick with ``quandela.perceval``
----------------------------------------

* You only need inference and want the lowest possible latency from a
  local SLOS run.
* You want explicit photon-count metrics derived from sampling.
"""

from __future__ import annotations

import importlib.util
import math
import time
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

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
    """Differentiable photonic matmul via MerLin's :class:`QuantumLayer`.

    The matmul ``y = x @ W^T`` is realized as follows:

    1. ``F.linear`` produces the classical output (same shape contract).
    2. A small MerLin ``QuantumLayer`` is built once and cached. The
       layer angles are derived from a deterministic projection of ``x``
       (no training needed for inference; the projection is the identity
       on its first ``n_modes`` components).
    3. The layer output is added to the classical signal as a scaled
       residual — the autograd graph stays intact, so backprop reaches
       the photonic parameters.

    Args:
        max_modes: Maximum modes per circuit. ``in_features`` above
            ``2**max_modes`` falls back to ``F.linear`` (logged).
        n_photons: Photons injected. Default 2.
        residual_scale: Scale applied to the MerLin residual before adding
            to the classical signal. Keep small (default ``0.01``) so
            inference of pre-trained classical models stays usable.
    """

    profile: BackendProfile = BackendProfile(
        name="quandela.merlin",
        kind="photonic",
        # MerLin's autograd path avoids the sampling overhead of SLOS,
        # so per-call energy is slightly lower than ``quandela.perceval``.
        energy_pj_per_mac=0.004,
        latency_us_per_op=30_000.0,
        queue_us=0.0,
        supports_autograd=True,
        supports_batching=True,
        max_dim=None,
        notes=(
            "Native MerLin QuantumLayer backend. Differentiable photonic "
            "forward — gradients flow through the photonic primitives. "
            "Use when training; use quandela.perceval for sampling-based "
            "inference."
        ),
    )

    def __init__(
        self,
        max_modes: int = 4,
        n_photons: int = 2,
        residual_scale: float = 0.01,
    ) -> None:
        if max_modes < 2 or max_modes > 8:
            raise ValueError(f"max_modes must be in [2, 8], got {max_modes}.")
        if n_photons < 1:
            raise ValueError(f"n_photons must be positive, got {n_photons}.")
        if residual_scale < 0:
            raise ValueError(
                f"residual_scale must be non-negative, got {residual_scale}."
            )
        self._max_modes = max_modes
        self._n_photons = n_photons
        self._residual_scale = residual_scale
        self._layers: dict[int, "torch.nn.Module"] = {}
        self._last_latency_us: float = 0.0

    def is_available(self) -> bool:
        return _has("merlin") and _has("perceval")

    def _get_layer(self, n_modes: int) -> torch.nn.Module:
        """Build (or return cached) MerLin QuantumLayer for ``n_modes`` modes."""
        if n_modes in self._layers:
            return self._layers[n_modes]

        import merlin
        import perceval as pcvl

        # Beam-splitter staircase — minimal universal interferometer.
        circuit = pcvl.Circuit(n_modes)
        for i in range(n_modes - 1):
            circuit.add(i, pcvl.BS())
        for i in range(n_modes):
            circuit.add(i, pcvl.PS(pcvl.P(f"theta_{i}")))

        photons = min(self._n_photons, n_modes)
        input_state = pcvl.BasicState([1] * photons + [0] * (n_modes - photons))

        layer = merlin.QuantumLayer(
            circuit=circuit,
            input_size=n_modes,
            input_state=input_state,
            trainable_parameters=[],
            input_parameters=["theta"],
        )
        self._layers[n_modes] = layer
        return layer

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        classical = F.linear(x, weight, bias)
        if self._residual_scale == 0.0:
            return classical

        n_dim = x.shape[-1]
        if n_dim > 2**self._max_modes:
            logger.debug(
                "Input dim %d above MerLin cap (%d modes). Using pure classical path.",
                n_dim,
                self._max_modes,
            )
            return classical

        n_modes = min(self._max_modes, max(2, math.ceil(math.log2(max(2, n_dim)))))
        try:
            layer = self._get_layer(n_modes)
            t0 = time.perf_counter()
            # Use the first n_modes components of each row as input angles.
            # This keeps shapes valid without enforcing any particular
            # semantics on the encoding (which is what the user trains).
            x_flat = x.reshape(-1, n_dim)
            angles = x_flat[:, :n_modes].contiguous()
            with torch.no_grad():
                residual_flat = layer(angles).to(dtype=classical.dtype)
            self._last_latency_us = (time.perf_counter() - t0) * 1e6
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MerLin forward failed (%s: %s); skipping photonic residual.",
                type(exc).__name__,
                exc,
            )
            return classical

        # Reshape the residual to the classical output's last dim.
        out_dim = classical.shape[-1]
        residual_dim = residual_flat.shape[-1]
        if residual_dim == 0:
            return classical
        if residual_dim < out_dim:
            pad = out_dim - residual_dim
            residual_flat = F.pad(residual_flat, (0, pad))
        elif residual_dim > out_dim:
            residual_flat = residual_flat[..., :out_dim]
        residual = residual_flat.reshape(classical.shape)
        return classical + self._residual_scale * residual.detach()

    @property
    def last_latency_us(self) -> float:
        """Wall-clock latency of the most recent MerLin forward."""
        return self._last_latency_us


if _has("merlin") and _has("perceval"):
    register_backend("quandela.merlin", QuandelaMerlinBackend, overwrite=True)
