"""Reference backend used for tests and as a runnable contract specification.

:class:`MockBackend` implements :class:`~qaithon.backends.base.Backend` by
delegating to ``torch.nn.functional.linear``. It is numerically *identical*
to a plain ``nn.Linear`` forward, which makes it the canonical fixture for:

* Unit-testing the compile pipeline without depending on photonic/quantum libs.
* Verifying that ``qaithon.compile(model, backend="mock")`` preserves the
  model's output bit-for-bit, gradients included.
* Demonstrating to backend implementors what a minimal but correct
  :class:`Backend` looks like.

It also accepts an optional ``noise_std`` knob so it can mimic noisy hardware
for tests that want to stress robustness without involving real backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812 — canonical PyTorch alias

from qaithon._logging import get_logger
from qaithon.backends.base import Backend, BackendProfile, register_backend

if TYPE_CHECKING:
    pass

__all__ = ["MockBackend"]

logger = get_logger(__name__)


class MockBackend(Backend):
    """Deterministic classical backend used as a reference implementation.

    Computes ``F.linear(x, weight, bias)`` exactly. Optionally adds Gaussian
    noise scaled by ``noise_std`` to the output, which is useful for tests
    that want to simulate the imperfection of real hardware without bringing
    in a heavyweight backend.

    Args:
        noise_std: Standard deviation of additive Gaussian noise on the output.
            Defaults to ``0.0`` (perfect identity to ``F.linear``).
        seed: Optional seed for the noise generator. When ``None`` and
            ``noise_std > 0``, noise uses the global PyTorch RNG.

    Example:
        >>> import torch
        >>> from qaithon.backends.mock import MockBackend
        >>> backend = MockBackend()
        >>> x = torch.randn(2, 3)
        >>> w = torch.randn(4, 3)
        >>> y = backend.matmul(x, w)
        >>> y.shape
        torch.Size([2, 4])
    """

    profile: BackendProfile = BackendProfile(
        name="mock",
        kind="mock",
        # Honest cost: mock runs F.linear on the CPU/GPU like any classical
        # path. Roughly aligned with the classical baseline the selector
        # uses internally. Photonic / quantum backends advertise lower
        # numbers and rightfully win when the user optimizes for energy.
        energy_pj_per_mac=1.0,
        latency_us_per_op=0.5,
        queue_us=0.0,
        supports_autograd=True,
        supports_batching=True,
        max_dim=None,
        notes="Reference implementation. Numerically identical to F.linear.",
    )

    def __init__(self, noise_std: float = 0.0, seed: int | None = None) -> None:
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}.")
        self._noise_std = noise_std
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``F.linear(x, weight, bias)`` with optional Gaussian noise.

        See :meth:`qaithon.backends.base.Backend.matmul` for the contract.
        """
        out = F.linear(x, weight, bias)
        if self._noise_std > 0:
            # Generate noise on the same device as the output so AMP / MPS work.
            noise = torch.randn(
                out.shape,
                generator=self._generator,
                dtype=out.dtype,
                device=out.device,
            )
            out = out + self._noise_std * noise
        return out


# Auto-register at import time so users can do `qaithon.compile(model, backend="mock")`
# without manually registering. Future backends follow the same pattern in their modules.
register_backend("mock", MockBackend, overwrite=True)
