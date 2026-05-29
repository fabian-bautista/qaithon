"""Architecture-specific handlers that the walker / compiler consults.

Most HuggingFace transformers compile cleanly with the generic walker.
A few architectures need special handling because they break standard
assumptions (Mixtral's 3D expert weights, Phi-3's fused projections, …).

Each handler module exposes either:

* a ``register()`` function that hooks into the analyzer, OR
* a stateless ``transform(model)`` function the compiler calls before
  the generic walker.

For v0.1 we ship Mixtral's handler as a worked example; additional
handlers are added on a per-architecture basis.
"""

from __future__ import annotations

from qaithon.handlers.architecture import (
    ArchitectureProfile,
    detect_architecture,
    list_architectures,
    recommend_config,
    register_architecture,
)
from qaithon.handlers.attention import AttentionInfo, list_attention_modules
from qaithon.handlers.mixtral import (
    is_mixtral_model,
    list_mixtral_experts,
    transform_mixtral_experts,
)

__all__ = [
    "ArchitectureProfile",
    "AttentionInfo",
    "detect_architecture",
    "is_mixtral_model",
    "list_architectures",
    "list_attention_modules",
    "list_mixtral_experts",
    "recommend_config",
    "register_architecture",
    "transform_mixtral_experts",
]
