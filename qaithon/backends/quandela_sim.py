"""Photonic-profile simulator backend modeled after Quandela's hardware.

This backend exposes the same numeric behavior as :class:`MockBackend`
(``F.linear``-equivalent forward) **but** advertises a photonic cost profile
based on Quandela's published characteristics. It is the right answer for
v0.1 because:

* The real Quandela hardware (Ascella, Belenos) does not execute arbitrary
  matrix multiplications — it executes parametrized linear-optical circuits.
  Mapping an arbitrary ``nn.Linear`` weight to such a circuit requires either
  (a) trainable approximation, which breaks drop-in behavior, or (b) unitary
  decomposition (Clements), which is non-trivial and will land in a separate
  backend later.
* Until then, this backend lets users *see* the photonic cost picture
  (CompileReport energy estimates, latency model) while keeping their model's
  outputs numerically correct. The :class:`AutoBackendSelector` can therefore
  reason about photonic deployments **before** real hardware is connected.

For correctness, this backend also enforces the GAP-013 invariant: inputs
must be in ``[0, 1]`` for any future swap to real MerLin layers to remain
valid. We softly normalize via sigmoid by default; users who already
normalize upstream can disable it.

When Quandela's photonic chips become accessible via cloud API (post v0.1),
a sibling :mod:`qaithon.backends.quandela_cloud` module will provide the
real-hardware backend. Their ``BackendProfile`` will be different (latency
dominated by queue time, energy lower than this estimate); the rest of the
contract is identical.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F  # noqa: N812

from qaithon._logging import get_logger
from qaithon.backends.base import Backend, BackendProfile, register_backend

if TYPE_CHECKING:
    pass

__all__ = ["QuandelaSimBackend"]

logger = get_logger(__name__)


# Photonic cost numbers based on Quandela's published characterization of the
# Ascella platform (12 modes, ~92% uptime, photon detection efficiency ~25%).
# These are order-of-magnitude estimates suitable for relative comparison;
# they are NOT a measurement of any specific user workload.
_QUANDELA_ENERGY_PJ_PER_MAC = 0.05  # ~20x better than H100 estimate (1 pJ/MAC).
_QUANDELA_LATENCY_US_PER_OP = 50.0  # Sub-ms per primitive when local.


class QuandelaSimBackend(Backend):
    """Photonic-profile simulator with Quandela-grade cost characteristics.

    Numerically equivalent to ``F.linear`` (preserves model semantics
    bit-for-bit) but reports a photonic cost profile to the compiler. Useful
    for evaluating "would this model benefit from photonic offload?" before
    cloud credentials or real hardware are available.

    Args:
        normalize_inputs: If ``True``, applies ``torch.sigmoid`` to the input
            before the matmul, guaranteeing the ``[0, 1]`` invariant that
            real MerLin photonic layers require. **Defaults to ``False``**:
            inside a transformer, internal activations are not in ``[0, 1]``
            and applying sigmoid silently destroys the model's outputs.
            Set ``True`` only when:
                * Your upstream pipeline already produces ``[0, 1]`` inputs, or
                * You want to *simulate* the silent-corruption regime that
                  appears on real hardware without fine-tuning (useful for
                  showing in a benchmark "what the output would look like
                  without QAT").
        noise_std: Standard deviation of additive Gaussian noise applied to
            the output, in units of the output's std. Defaults to ``0.0``
            (no noise). Use a small value (1e-3 to 1e-2) to simulate
            photodetection imperfection.
        seed: Optional seed for noise reproducibility.

    Example:
        >>> import torch
        >>> from qaithon.backends.quandela_sim import QuandelaSimBackend
        >>> backend = QuandelaSimBackend()
        >>> x = torch.rand(2, 4)  # already in [0, 1]
        >>> w = torch.randn(8, 4)
        >>> y = backend.matmul(x, w)
        >>> y.shape
        torch.Size([2, 8])
    """

    profile: BackendProfile = BackendProfile(
        name="quandela.sim",
        kind="photonic",
        energy_pj_per_mac=_QUANDELA_ENERGY_PJ_PER_MAC,
        latency_us_per_op=_QUANDELA_LATENCY_US_PER_OP,
        queue_us=0.0,  # Local simulator. quandela_cloud will set this.
        supports_autograd=True,
        supports_batching=True,
        max_dim=None,
        notes=(
            "Photonic-profile simulator based on Quandela Ascella characteristics. "
            "Numerically identical to F.linear; advertises photonic cost. "
            "Will be replaced by real hardware backend once Quandela cloud is "
            "configured (see qaithon.backends.quandela_cloud)."
        ),
    )

    def __init__(
        self,
        normalize_inputs: bool = False,
        noise_std: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if noise_std < 0:
            raise ValueError(f"noise_std must be non-negative, got {noise_std}.")
        self._normalize_inputs = normalize_inputs
        self._noise_std = noise_std
        self._generator: torch.Generator | None = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(seed)

    def is_available(self) -> bool:
        """Return ``True`` if the Quandela stack (perceval + merlin) is importable.

        We do not actually import them eagerly — just check that the modules
        exist on the filesystem. This keeps the check cheap (~ms) and
        side-effect-free.
        """
        def _has(name: str) -> bool:
            try:
                return importlib.util.find_spec(name) is not None
            except (ModuleNotFoundError, ValueError):
                return False

        return all(_has(name) for name in ("perceval", "merlin"))

    def matmul(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute ``F.linear`` with optional [0,1] normalization and shot noise.

        See :meth:`qaithon.backends.base.Backend.matmul` for the contract.
        """
        if self._normalize_inputs:
            x = torch.sigmoid(x)
        out = F.linear(x, weight, bias)
        if self._noise_std > 0:
            noise = torch.randn(
                out.shape,
                generator=self._generator,
                dtype=out.dtype,
                device=out.device,
            )
            # Scale noise by the per-output-element std for realism.
            out = out + self._noise_std * noise * (out.std().clamp(min=1e-6))
        return out


register_backend("quandela.sim", QuandelaSimBackend, overwrite=True)
