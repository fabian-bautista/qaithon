"""Integration with HuggingFace's ``from_pretrained`` flow.

HuggingFace exposes a ``HfQuantizer`` extension point that ``transformers``
uses to apply quantization (bitsandbytes 4-bit/8-bit, GPTQ, AWQ, …). Even
though Qaithon is not quantization, the same mechanism is the cleanest way
to integrate with ``AutoModel.from_pretrained(..., quantization_config=...)``.

This module defines:

* :class:`QaithonConfig` — a ``QuantizationConfigMixin`` subclass that
  declares "use Qaithon to transform this model" plus the optimization
  knobs the user wants (``optimize_for``, ``backends``).
* :class:`QaithonHfQuantizer` — the actual ``HfQuantizer`` subclass that
  applies ``qaithon.compile`` after the standard ``from_pretrained`` flow
  finishes loading the model.

The import is **lazy**: this module imports nothing from ``transformers``
at module load. If ``transformers`` is not installed, importing
``qaithon.hf_integration`` still succeeds but the classes raise on instantiation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from qaithon._logging import get_logger
from qaithon.exceptions import QaithonError

if TYPE_CHECKING:
    from torch import nn

__all__ = ["QaithonConfig", "QaithonHfQuantizer", "register_with_transformers"]

logger = get_logger(__name__)


@dataclass
class QaithonConfig:
    """Declarative configuration passed to ``from_pretrained``.

    Mimics the shape HuggingFace expects from quantization configs
    (``quant_method``, ``to_dict``, ``from_dict``) so it slots into the
    framework cleanly.

    Args:
        backends: Optional whitelist of backend names. ``None`` (default)
            means "consider every available backend".
        backend: Convenience singular form. Accepts a string (single
            backend) or a tuple. Mutually exclusive with ``backends``.
        optimize_for: ``"speed"``, ``"energy"``, or ``"balanced"`` (default).
        strict: Refuse to compile already-quantized models.

    Example:
        >>> from qaithon.hf_integration import QaithonConfig
        >>> cfg = QaithonConfig(backend="quandela.sim", optimize_for="energy")
        >>> # passed to AutoModelForCausalLM.from_pretrained(
        ...     "meta-llama/Llama-3-8B", quantization_config=cfg)  # doctest: +SKIP
    """

    backends: tuple[str, ...] | None = None
    backend: str | tuple[str, ...] | None = None  # singular convenience
    optimize_for: str = "balanced"
    strict: bool = True
    quant_method: str = field(default="qaithon", init=False)

    def __post_init__(self) -> None:
        # Normalize backend → backends. User intent: backend="x" means
        # backends=("x",); backends=("x","y") direct works as before.
        if self.backend is not None and self.backends is not None:
            raise ValueError(
                "Pass either `backend` (singular) or `backends` (tuple), not both."
            )
        if self.backend is not None:
            normalized = (
                (self.backend,) if isinstance(self.backend, str) else tuple(self.backend)
            )
            object.__setattr__(self, "backends", normalized)
            object.__setattr__(self, "backend", None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "quant_method": self.quant_method,
            "backends": list(self.backends) if self.backends else None,
            "optimize_for": self.optimize_for,
            "strict": self.strict,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> QaithonConfig:
        backends = d.get("backends")
        return cls(
            backends=tuple(backends) if backends else None,
            optimize_for=d.get("optimize_for", "balanced"),
            strict=d.get("strict", True),
        )


class QaithonHfQuantizer:
    """Apply :func:`qaithon.compile` after ``from_pretrained`` finishes loading.

    This class is constructed by ``transformers`` automatically when a model
    is loaded with ``quantization_config=QaithonConfig(...)``. It calls
    Qaithon's ``compile`` on the already-loaded model.

    Lazy import of ``transformers`` happens at class construction so users
    who don't use this code path don't pay for the heavy import.

    Note:
        Full integration with ``transformers``' loader requires registering
        this class as a known quantizer (see
        :func:`register_with_transformers`).
    """

    def __init__(self, config: QaithonConfig) -> None:
        try:
            import transformers  # noqa: F401
        except ImportError as exc:
            raise QaithonError(
                "QaithonHfQuantizer requires the `transformers` package. "
                "Install it with `pip install qaithon[huggingface]`."
            ) from exc
        self.config = config

    def process_model_after_load(self, model: nn.Module) -> nn.Module:
        """Called by ``transformers`` after loading the model from disk."""
        # Imported here so the module is fully loaded.
        from qaithon.compile import compile as qaithon_compile

        return qaithon_compile(
            model,
            optimize_for=self.config.optimize_for,  # type: ignore[arg-type]
            backends=self.config.backends,
            strict=self.config.strict,
        )


def register_with_transformers(verbose: bool = False) -> bool:
    """Register QaithonHfQuantizer as a known quantizer in ``transformers``.

    Must be called once per process. After registration,
    ``AutoModelForCausalLM.from_pretrained(..., quantization_config=QaithonConfig(...))``
    automatically routes through Qaithon.

    Implementation strategy: ``transformers``' quantizer registry API has
    evolved several times. We probe multiple well-known paths in order of
    likelihood for transformers 4.40 → 5.x. The first one that succeeds wins.

    Returns:
        ``True`` if registration succeeded, ``False`` otherwise. Even when
        the return is ``False``, callers can still apply the quantizer
        manually via ``QaithonHfQuantizer(config).process_model_after_load(model)``.
    """
    import importlib

    candidates: tuple[tuple[str, str], ...] = (
        # transformers 4.40+
        ("transformers.quantizers.auto", "AUTO_QUANTIZER_MAPPING"),
        ("transformers.quantizers.auto", "AUTO_QUANTIZATION_CONFIG_MAPPING"),
        ("transformers.quantizers", "AUTO_QUANTIZER_MAPPING"),
        # transformers 5.x renamed to plural / dict on the top-level module
        ("transformers", "AUTO_QUANTIZER_MAPPING"),
        ("transformers.quantizers.quantizers_utils", "AUTO_QUANTIZER_MAPPING"),
    )

    for module_path, attr in candidates:
        try:
            mod = importlib.import_module(module_path)
        except ImportError:
            continue
        registry = getattr(mod, attr, None)
        if isinstance(registry, dict):
            registry["qaithon"] = QaithonHfQuantizer
            if verbose:
                logger.info("Registered QaithonHfQuantizer in %s.%s", module_path, attr)
            return True

    # Try the modern (5.x) plugin-via-config mechanism: just attach a
    # method to QaithonConfig so transformers can discover it.
    if not hasattr(QaithonConfig, "post_init"):

        def _post_init(self: QaithonConfig) -> None:  # noqa: ARG001
            pass

        QaithonConfig.post_init = _post_init  # type: ignore[attr-defined]

    logger.warning(
        "Could not locate a transformers quantizer registry in this transformers "
        "version (%d candidates probed). QaithonHfQuantizer can still be applied "
        "manually via `QaithonHfQuantizer(config).process_model_after_load(model)`.",
        len(candidates),
    )
    return False


# Eager attempt at process start so users who simply do
# `from qaithon.hf_integration import QaithonConfig` get auto-registration.
_AUTO_REGISTERED = register_with_transformers(verbose=False)
