"""Shared scaffolding for backends that connect to real quantum hardware.

Each real-hardware backend (IBM Heron, Quandela Belenos, AWS SV1) shares the
same three modes of operation:

* ``"profile"`` (default) — cost model declared, computation runs ``F.linear``.
  Zero risk of consuming cloud quota. This is the right mode for development,
  testing, and the AutoBackendSelector's reasoning.

* ``"calibrate"`` — for every forward call, dispatch ONE small circuit to the
  real backend, measure wall-clock latency + noise scale, apply the measured
  noise to the classical output. Used to gather real-hardware telemetry
  without running every matmul through the QPU.

* ``"execute"`` — every matmul becomes a real circuit on the real hardware.
  Slow and quota-consuming; opt-in for explicit benchmarking only.

The wrapper exists so the three vendor-specific backends share the same
plumbing (mode validation, latency tracking, noise injection) without
duplicating code.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F  # noqa: N812

from qaithon._logging import get_logger
from qaithon.backends.base import Backend

if TYPE_CHECKING:
    pass

__all__ = ["BackendMode", "RealHardwareBackendBase"]

logger = get_logger(__name__)

BackendMode = Literal["profile", "calibrate", "execute"]


class RealHardwareBackendBase(Backend):
    """Shared scaffolding for any backend that may dispatch to real hardware.

    Subclasses implement:

    * :meth:`_calibrate_once` — return measured noise scale + wall-clock
      latency by dispatching exactly one small circuit to the real backend.
    * :meth:`_execute_full_matmul` — only invoked when ``mode="execute"``;
      runs the full matmul as quantum circuits. Subclasses are free to
      raise ``NotImplementedError`` here for v0.1; the mode is opt-in.

    The base class handles input validation and short-circuits to
    ``F.linear`` when ``mode="profile"``.
    """

    def __init__(
        self,
        mode: BackendMode = "profile",
    ) -> None:
        if mode not in ("profile", "calibrate", "execute"):
            raise ValueError(
                f"mode must be 'profile' | 'calibrate' | 'execute', got {mode!r}."
            )
        self._mode = mode
        self._last_circuit_latency_us: float = 0.0
        self._last_calibration_noise_scale: float = 0.0

    @property
    def mode(self) -> BackendMode:
        return self._mode

    @property
    def last_circuit_latency_us(self) -> float:
        """Wall-clock latency of the most recent real-circuit dispatch."""
        return self._last_circuit_latency_us

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._mode == "profile":
            return F.linear(x, weight, bias)

        classical = F.linear(x, weight, bias)

        if self._mode == "calibrate":
            try:
                t0 = time.perf_counter()
                noise_scale = self._calibrate_once()
                self._last_circuit_latency_us = (time.perf_counter() - t0) * 1e6
                self._last_calibration_noise_scale = noise_scale
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "%s calibration failed (%s: %s). Falling back to profile mode for "
                    "this call.",
                    type(self).__name__,
                    type(exc).__name__,
                    exc,
                )
                return classical
            return self._apply_noise(classical, noise_scale)

        # mode == "execute"
        try:
            return self._execute_full_matmul(x, weight, bias)
        except NotImplementedError:
            logger.warning(
                "%s does not yet support full quantum execution for matmul; "
                "running in calibrate mode for this call instead.",
                type(self).__name__,
            )
            self._mode = "calibrate"
            return self.matmul(x, weight, bias)

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    def _calibrate_once(self) -> float:
        """Run one small calibration circuit; return the noise scale (sigma).

        Subclasses must override. The returned value is multiplied by the
        per-element std of the classical output to scale the additive noise.
        """
        raise NotImplementedError

    def _execute_full_matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run the matmul as a quantum circuit on real hardware.

        Subclasses may raise NotImplementedError to signal "I only support
        the calibration path for v0.1".
        """
        raise NotImplementedError

    @staticmethod
    def _apply_noise(classical: torch.Tensor, noise_scale: float) -> torch.Tensor:
        scale = noise_scale * classical.std().clamp(min=1e-6)
        noise = torch.randn_like(classical) * scale
        # Detach noise so it doesn't feed gradients.
        return classical + noise.detach()
