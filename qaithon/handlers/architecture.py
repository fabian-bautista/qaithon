"""Automatic architecture detection and per-family configuration.

The promise of ``qaithon.compile(model)`` without arguments is that
Qaithon *decides* everything. To honor that for any HuggingFace LLM, the
library needs to know — for each architecture family — which layers are
safe to offload, which to leave classical, and what backend characteristics
are appropriate.

This module is the registry where that knowledge lives. It exposes two
functions used by the compiler:

* :func:`detect_architecture` — inspect a model and return a stable family
  identifier (``"llama"``, ``"mistral"``, ``"gpt2"``, ``"mixtral"``,
  ``"phi"``, ``"qwen"``, ``"gemma"``, or ``"generic"``).
* :func:`recommend_config` — given the family and the user's optimization
  objective, return an :class:`ArchitectureProfile` describing recommended
  defaults: which layer-name patterns to skip, minimum-size threshold,
  whether MoE experts must be transformed, and any per-family hints.

The registry is open: third parties can register their own families via
:func:`register_architecture` without modifying Qaithon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from torch import nn

from qaithon._logging import get_logger

if TYPE_CHECKING:
    pass

__all__ = [
    "ArchitectureProfile",
    "detect_architecture",
    "list_architectures",
    "recommend_config",
    "register_architecture",
]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ArchitectureProfile:
    """Recommended defaults Qaithon applies for a given model family.

    Attributes:
        family: Stable identifier (``"llama"``, ``"gpt2"``, etc.).
        description: Human-readable description for logs and reports.
        skip_name_patterns: Substrings that, if present in a layer's
            fully-qualified name, cause it to be skipped. Conventional:
            ``"lm_head"``, ``"embed_tokens"``, ``"position_embedding"``.
        min_in_features: Minimum input dimension a layer must have to be
            considered worth replacing. Below this, the overhead exceeds
            the gain.
        min_out_features: Same idea for the output dimension.
        moe_aware: Whether to invoke the Mixtral expert transformation pass.
        notes: Free-form notes shown in the CompileReport.
    """

    family: str
    description: str
    skip_name_patterns: tuple[str, ...] = field(default_factory=tuple)
    min_in_features: int = 64
    min_out_features: int = 64
    moe_aware: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, ArchitectureProfile] = {}


def register_architecture(profile: ArchitectureProfile, *, overwrite: bool = False) -> None:
    """Register an architecture profile. Open/Closed extension point."""
    if not overwrite and profile.family in _REGISTRY:
        raise ValueError(
            f"Architecture family {profile.family!r} is already registered. "
            "Pass overwrite=True to replace it."
        )
    _REGISTRY[profile.family] = profile


def list_architectures() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY.keys()))


def recommend_config(family: str) -> ArchitectureProfile:
    """Return the profile registered under ``family``, or the generic fallback."""
    return _REGISTRY.get(family, _REGISTRY["generic"])


# ---------------------------------------------------------------------------
# Default profiles for the families HF distributes most commonly
# ---------------------------------------------------------------------------
_DEFAULTS: tuple[ArchitectureProfile, ...] = (
    ArchitectureProfile(
        family="generic",
        description="Fallback profile for any transformer-shaped model.",
        skip_name_patterns=("lm_head", "embed_tokens", "embeddings"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="Generic defaults. Replace any large nn.Linear / Conv1D outside the embedding/head paths.",
    ),
    ArchitectureProfile(
        family="gpt2",
        description="OpenAI GPT-2 family (and GPT-Neo, since both use Conv1D).",
        skip_name_patterns=("lm_head", "wte", "wpe"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="Conv1D layout. Tied weights between wte and lm_head are detected automatically.",
    ),
    ArchitectureProfile(
        family="llama",
        description="Meta Llama 1/2/3 family.",
        skip_name_patterns=("lm_head", "embed_tokens", "norm"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="Standard RoPE + RMSNorm + SwiGLU. Q/K/V/O proj and MLP gate/up/down are safe targets.",
    ),
    ArchitectureProfile(
        family="mistral",
        description="Mistral 7B family with sliding-window attention.",
        skip_name_patterns=("lm_head", "embed_tokens", "norm"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="Architecture matches Llama closely; same defaults apply.",
    ),
    ArchitectureProfile(
        family="mixtral",
        description="Mistral MoE family (8x7B, 8x22B).",
        skip_name_patterns=("lm_head", "embed_tokens", "norm", "gate.weight"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=True,
        notes="Experts are stored as 3D Parameter — handler invoked automatically.",
    ),
    ArchitectureProfile(
        family="phi",
        description="Microsoft Phi-2 / Phi-3 family.",
        skip_name_patterns=("lm_head", "embed_tokens", "norm"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="Phi-3 fuses Q/K/V into a single qkv_proj; the walker handles it as one Linear.",
    ),
    ArchitectureProfile(
        family="qwen",
        description="Alibaba Qwen 1.5/2 family.",
        skip_name_patterns=("lm_head", "embed_tokens", "norm"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="Aggressive Grouped Query Attention (GQA). Standard projections.",
    ),
    ArchitectureProfile(
        family="gemma",
        description="Google Gemma family with tied embeddings.",
        skip_name_patterns=("lm_head", "embed_tokens", "norm"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="Strong embedding tying — accelerate.find_tied_parameters catches it; no extra work needed.",
    ),
    ArchitectureProfile(
        family="bert",
        description="BERT-family encoders (BERT, RoBERTa, DistilBERT).",
        skip_name_patterns=("cls", "pooler", "embeddings"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="Encoder-only. Attention pre-norm vs post-norm doesn't affect Qaithon.",
    ),
    ArchitectureProfile(
        family="starcoder",
        description="BigCode StarCoder / StarCoder2.",
        skip_name_patterns=("lm_head", "wte", "wpe"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="Code LLM with extended context. Standard Conv1D-or-Linear projections.",
    ),
    ArchitectureProfile(
        family="deepseek",
        description="DeepSeek V1/V2/V3 (including MoE variants).",
        skip_name_patterns=("lm_head", "embed_tokens", "norm"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=True,  # DeepSeek-V2/V3 are MoE.
        notes="MoE-aware. DeepSeek-V2 uses MLA (multi-latent attention) — current handler treats it as standard.",
    ),
    ArchitectureProfile(
        family="falcon",
        description="TII Falcon family (7B, 40B, 180B).",
        skip_name_patterns=("lm_head", "word_embeddings", "ln"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="Multi-query attention. Bias=False is common; from_linear handles both.",
    ),
    ArchitectureProfile(
        family="mamba",
        description="State-space model family (Mamba, Mamba-2).",
        skip_name_patterns=("lm_head", "embeddings", "in_proj", "out_proj"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="NOT a transformer — SSM. Most layers are 1D conv + selective scan; Qaithon offloads only the Linear projections that remain.",
    ),
    ArchitectureProfile(
        family="pythia",
        description="EleutherAI Pythia / Pythia-Suite family.",
        skip_name_patterns=("embed_out", "embed_in", "norm"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="GPT-NeoX style. Standard projections.",
    ),
    ArchitectureProfile(
        family="opt",
        description="Meta OPT family (125M to 175B).",
        skip_name_patterns=("lm_head", "embed_tokens", "embed_positions"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="Predecessor to Llama. Pre-norm transformer. Standard projections.",
    ),
    ArchitectureProfile(
        family="t5",
        description="Google T5 / Flan-T5 / mT5 encoder-decoder.",
        skip_name_patterns=("lm_head", "shared", "embed_tokens", "relative_attention_bias"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="Encoder-decoder with relative position bias. Standard Linear projections in self/cross attention.",
    ),
    ArchitectureProfile(
        family="bloom",
        description="BigScience BLOOM (multilingual).",
        skip_name_patterns=("lm_head", "word_embeddings", "ln"),
        min_in_features=64,
        min_out_features=64,
        moe_aware=False,
        notes="ALiBi positional bias. Standard projections.",
    ),
)

for _p in _DEFAULTS:
    register_architecture(_p, overwrite=True)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------
# Hints applied in order. First match wins. The order matters: more specific
# names appear before more generic ones (mixtral before mistral, phi3 before phi).
_DETECTION_RULES: tuple[tuple[str, Callable[[str], bool]], ...] = (
    ("mixtral", lambda cls: cls.startswith("Mixtral")),
    ("deepseek", lambda cls: cls.startswith("DeepSeek") or cls.startswith("Deepseek")),
    ("llama", lambda cls: cls.startswith("Llama") or cls.startswith("CodeLlama")),
    ("mistral", lambda cls: cls.startswith("Mistral")),
    ("phi", lambda cls: cls.startswith("Phi") or cls.startswith("Phi3")),
    ("qwen", lambda cls: cls.startswith("Qwen") or cls.startswith("Qwen2")),
    ("gemma", lambda cls: cls.startswith("Gemma")),
    ("starcoder", lambda cls: cls.startswith("Starcoder") or cls.startswith("StarCoder") or cls.startswith("GPTBigCode")),
    ("falcon", lambda cls: cls.startswith("Falcon")),
    ("mamba", lambda cls: cls.startswith("Mamba")),
    ("pythia", lambda cls: cls.startswith("GPTNeoX") or cls.startswith("Pythia")),
    ("opt", lambda cls: cls.startswith("OPT")),
    ("t5", lambda cls: cls.startswith("T5") or cls.startswith("MT5") or cls.startswith("FlanT5")),
    ("bloom", lambda cls: cls.startswith("Bloom")),
    ("gpt2", lambda cls: cls.startswith("GPT2") or cls.startswith("GPTNeo")),
    ("bert", lambda cls: cls.startswith("Bert") or cls.startswith("Roberta") or cls.startswith("DistilBert")),
)


def detect_architecture(model: nn.Module) -> str:
    """Return the canonical family identifier for ``model``.

    Detection inspects the model's class name and its submodules; this is
    enough for the families HuggingFace distributes today. Unknown models
    return ``"generic"`` — Qaithon still works on them, with conservative
    defaults.
    """
    cls_name = type(model).__name__

    # Try the top-level class name.
    for family, predicate in _DETECTION_RULES:
        if predicate(cls_name):
            return family

    # Fall back to scanning child classes. Useful when the user passes a
    # PreTrainedModel wrapper whose top-level class is generic but whose
    # transformer attribute reveals the family.
    for child in model.modules():
        child_cls = type(child).__name__
        for family, predicate in _DETECTION_RULES:
            if predicate(child_cls):
                return family

    return "generic"
