"""Drop-in replacement for :class:`torch.nn.Linear` backed by a Qaithon backend.

:class:`QuantumLinear` is the workhorse layer of Qaithon's MVP. It exposes the
**exact same public surface** as ``nn.Linear`` (``in_features``,
``out_features``, ``weight``, ``bias``) so any code that inspects the
attributes of a linear layer — HuggingFace's ``accelerate``, ``peft``,
``bitsandbytes``, or hand-written introspection — keeps working unchanged.
The only difference is that its ``forward`` delegates the matmul to a
configurable :class:`~qaithon.backends.base.Backend`.

Design choices
--------------

* The weight is stored as a real ``nn.Parameter`` so autograd works without
  any custom function. The backend is responsible for routing the gradient
  through its own machinery; the mock backend trivially does this via
  ``F.linear``.
* ``QuantumLinear.from_linear(linear, backend=...)`` is a class-method
  constructor that adopts an existing ``nn.Linear`` (its weights and bias),
  which is what the walker uses to swap layers in-place.
* Repr is intentionally compatible with ``nn.Linear``'s repr but adds the
  backend name, so ``print(model)`` shows what was replaced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from qaithon._logging import get_logger
from qaithon.backends.base import Backend, get_backend

if TYPE_CHECKING:
    pass

__all__ = ["QuantumLinear"]

logger = get_logger(__name__)


def _normalize_linear_inputs(linear: nn.Module) -> tuple[int, int, torch.Tensor]:
    """Return ``(in_features, out_features, weight_2d_in_linear_layout)``.

    Normalizes the source module's weight to the ``nn.Linear`` convention
    ``(out_features, in_features)`` regardless of whether the source is
    ``nn.Linear`` or ``transformers.pytorch_utils.Conv1D`` (which stores its
    weight as ``(in_features, out_features)``).
    """
    if isinstance(linear, nn.Linear):
        return linear.in_features, linear.out_features, linear.weight
    # Conv1D path: identified by class name to avoid an import dependency.
    qualified = f"{type(linear).__module__}.{type(linear).__name__}"
    if qualified == "transformers.pytorch_utils.Conv1D":
        nf = int(linear.nf)  # type: ignore[attr-defined]
        weight = linear.weight  # type: ignore[union-attr]
        in_features = int(weight.shape[0])
        # Conv1D weight is (in, out). nn.Linear is (out, in). Transpose.
        return in_features, nf, weight.transpose(0, 1).contiguous()
    raise TypeError(
        f"QuantumLinear.from_linear does not know how to adopt "
        f"{type(linear).__name__} (qualified: {qualified})."
    )


class QuantumLinear(nn.Module):
    """Linear layer whose forward pass is executed by a Qaithon backend.

    Mirrors the public interface of :class:`torch.nn.Linear`:

    Attributes:
        in_features: Size of the input's last dimension.
        out_features: Size of the output's last dimension.
        weight: Trainable weight tensor of shape ``(out_features, in_features)``.
        bias: Optional trainable bias of shape ``(out_features,)``, or ``None``.
        backend: The :class:`Backend` instance used to compute the matmul.

    Args:
        in_features: Same as ``nn.Linear``.
        out_features: Same as ``nn.Linear``.
        bias: Whether to include a bias term.
        backend: Either a string (looked up in the default registry) or an
            already-instantiated :class:`Backend`. Defaults to ``"mock"``.
        device: Optional device placement, matching ``nn.Linear``'s API.
        dtype: Optional dtype, matching ``nn.Linear``'s API.

    Example:
        >>> import torch
        >>> from qaithon.layers.quantum_linear import QuantumLinear
        >>> layer = QuantumLinear(8, 4, backend="mock")
        >>> x = torch.randn(2, 8)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([2, 4])
    """

    in_features: int
    out_features: int
    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        backend: str | Backend = "mock",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError(
                f"in_features and out_features must be positive, "
                f"got in={in_features}, out={out_features}."
            )
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self._backend = self._resolve_backend(backend)
        self.reset_parameters()

    @staticmethod
    def _resolve_backend(backend: str | Backend) -> Backend:
        """Normalize a string or :class:`Backend` to a :class:`Backend` instance."""
        if isinstance(backend, Backend):
            return backend
        if isinstance(backend, str):
            return get_backend(backend)
        raise TypeError(
            f"backend must be a string or Backend instance, got {type(backend).__name__}."
        )

    @property
    def backend(self) -> Backend:
        """The backend used by this layer's forward pass."""
        return self._backend

    @backend.setter
    def backend(self, backend: str | Backend) -> None:
        """Replace the backend without rebuilding the layer."""
        self._backend = self._resolve_backend(backend)

    def reset_parameters(self) -> None:
        """Initialize weights and bias using the same scheme as ``nn.Linear``."""
        # PyTorch's nn.Linear uses kaiming_uniform_ with a=sqrt(5).
        # Replicating it exactly so swapped layers behave identically when re-initialized.
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate the linear projection to the configured backend.

        Args:
            x: Input tensor with shape ``(..., in_features)``.

        Returns:
            Tensor of shape ``(..., out_features)``.
        """
        return self._backend.matmul(x, self.weight, self.bias)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Module,
        *,
        backend: str | Backend = "mock",
        copy_weights: bool = True,
    ) -> QuantumLinear:
        """Build a :class:`QuantumLinear` that adopts an existing linear layer.

        Accepts both ``nn.Linear`` and HuggingFace's
        ``transformers.pytorch_utils.Conv1D`` (used by GPT-2 / GPT-Neo /
        BLOOM). Conv1D stores its weight transposed vs ``nn.Linear``; we
        normalize to ``nn.Linear``'s ``(out, in)`` layout on the way in.

        Args:
            linear: Source module — either ``nn.Linear`` or HF Conv1D.
            backend: Backend name or instance for the new layer.
            copy_weights: If ``True`` (default), parameters are initialized
                from the source. If ``False``, the new layer is left at
                default (random) initialization — useful for ablations.

        Returns:
            A :class:`QuantumLinear` with the same shape, device and dtype.
        """
        in_features, out_features, weight_2d = _normalize_linear_inputs(linear)
        bias_param = getattr(linear, "bias", None)
        has_bias = bias_param is not None
        new_layer = cls(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            backend=backend,
            device=weight_2d.device,
            dtype=weight_2d.dtype,
        )
        if copy_weights:
            with torch.no_grad():
                new_layer.weight.copy_(weight_2d.detach())
                if has_bias and new_layer.bias is not None:
                    new_layer.bias.copy_(bias_param.detach())
        return new_layer

    def extra_repr(self) -> str:
        """Return a string describing this layer for ``print(model)``."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, backend={self._backend.profile.name!r}"
        )
