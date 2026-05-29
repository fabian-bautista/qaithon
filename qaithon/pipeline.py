"""Composability primitives — chain compiled models together.

For the v0.1 MVP the pipeline does one thing: take a list of models and run
them in sequence, threading the output of each into the input of the next.
The point is to keep the API stable so users can already think in
"pipeline" terms; richer routing (per-token, per-layer, dynamic) lands in
a later release.

LangChain's Runnable hierarchy inspires this API. We deliberately do **not**
reproduce LangChain's complexity (config dicts, lazy graphs, runnable
parallel/branch). Start small; expand only when concrete use cases demand it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import torch
from torch import nn

from qaithon._logging import get_logger

if TYPE_CHECKING:
    pass

__all__ = ["Pipeline"]

logger = get_logger(__name__)


class Pipeline(nn.Module):
    """Sequentially apply a list of (compiled) models or callables.

    Each element of the pipeline must accept a tensor and return a tensor.
    Tensors flow left-to-right; if any element raises, the exception
    propagates with the offending index included in the error message.

    This is the minimal viable composability primitive. It is intentionally
    small — adding "magic" (auto-bridging shapes, automatic routing,
    parallel branches) would land it in the over-engineered LangChain
    territory we want to avoid.

    Args:
        stages: Iterable of either ``nn.Module`` or any callable
            ``Tensor -> Tensor``.

    Example:
        >>> import torch
        >>> import qaithon
        >>> from qaithon.pipeline import Pipeline
        >>> from torch import nn
        >>> class A(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc = nn.Linear(4, 8)
        ...     def forward(self, x):
        ...         return self.fc(x)
        >>> class B(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.fc = nn.Linear(8, 2)
        ...     def forward(self, x):
        ...         return self.fc(x)
        >>> a, b = A(), B()
        >>> qaithon.compile(a)  # doctest: +ELLIPSIS
        A(...)
        >>> qaithon.compile(b)  # doctest: +ELLIPSIS
        B(...)
        >>> pipe = Pipeline([a, b])
        >>> y = pipe(torch.randn(3, 4))
        >>> y.shape
        torch.Size([3, 2])
    """

    def __init__(self, stages: Iterable[nn.Module | Callable[..., torch.Tensor]]) -> None:
        super().__init__()
        stage_list = list(stages)
        if not stage_list:
            raise ValueError("Pipeline requires at least one stage.")
        # Wrap nn.Modules in a ModuleList so they participate in `model.to`,
        # `model.parameters`, state_dict, etc. Non-Module callables stay
        # as-is — Pipeline still routes through them but cannot register them.
        self._stages: list[nn.Module | Callable[..., torch.Tensor]] = []
        self._modules_holder = nn.ModuleList()
        for stage in stage_list:
            if isinstance(stage, nn.Module):
                self._modules_holder.append(stage)
                self._stages.append(stage)
            elif callable(stage):
                self._stages.append(stage)
            else:
                raise TypeError(
                    f"Pipeline stages must be nn.Module or callable, "
                    f"got {type(stage).__name__}."
                )

    def __len__(self) -> int:
        return len(self._stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Thread ``x`` through every stage in order."""
        for i, stage in enumerate(self._stages):
            try:
                x = stage(x)
            except Exception as exc:
                raise RuntimeError(
                    f"Pipeline stage {i} ({type(stage).__name__}) raised: {exc}"
                ) from exc
        return x

    def extra_repr(self) -> str:
        return f"n_stages={len(self._stages)}"
