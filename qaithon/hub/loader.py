"""HuggingFace Hub integration for Qaithon blocks.

Implementation strategy
-----------------------

We treat HF Hub as the artifact store: each Qaithon block is a HF repo
following the naming convention ``qaithon/<family>-<variant>-v<N>`` and
containing two files:

* ``model.safetensors`` — the block's weights.
* ``qaithon_block.json`` — metadata: family, variant, version, target
  architecture (e.g. "llama"), the Python class to instantiate
  (``"qaithon.layers.QuantumLinear"`` or a third-party plugin), and the
  ``BackendProfile`` recommended for the block.

The Python class is referenced as a fully-qualified import path so loading
is purely declarative — no eval, no dynamic Python. If the class is in a
third-party package the user installed separately, it will be importable
naturally; if not, loading fails with a clear "install package X" error.

Until the first block is published, these functions surface a clear
"published catalog is empty" error rather than pretending to work. The
API surface is stable so downstream code doesn't change once content lands.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from qaithon._logging import get_logger
from qaithon.exceptions import QaithonError

if TYPE_CHECKING:
    from torch import nn

__all__ = [
    "HubError",
    "HubNotImplementedError",
    "list_blocks",
    "load_block",
    "push_block",
]

logger = get_logger(__name__)

_DEFAULT_NAMESPACE = "qaithon"
_METADATA_FILENAME = "qaithon_block.json"
_WEIGHTS_FILENAME = "model.safetensors"


class HubError(QaithonError):
    """Base error raised by Qaithon Hub operations."""


class HubNotImplementedError(HubError, NotImplementedError):
    """Backwards-compatible alias kept while the catalog is empty.

    Raised by callers that hit functionality not yet shippable; subclasses
    :class:`NotImplementedError` so user ``except NotImplementedError``
    still catches it.
    """


def _require_hf_hub() -> Any:
    """Return the ``huggingface_hub`` module or raise with a clear hint."""
    if importlib.util.find_spec("huggingface_hub") is None:
        raise HubError(
            "huggingface_hub is not installed. Install it with "
            "`pip install qaithon[huggingface]` to use the Qaithon Hub."
        )
    return importlib.import_module("huggingface_hub")


def list_blocks(
    family: str | None = None,
    *,
    namespace: str = _DEFAULT_NAMESPACE,
) -> list[str]:
    """List blocks visible on the Hub.

    Args:
        family: Optional family filter (e.g. ``"attention"``, ``"ffn"``).
        namespace: HF organization or user to list from. Defaults to the
            project's reserved namespace.

    Returns:
        List of canonical block names (``"qaithon/<family>-<variant>-vN"``).

    Raises:
        HubError: If the catalog query fails.
    """
    hf = _require_hf_hub()
    try:
        repos = hf.list_models(author=namespace, full=False)
    except Exception as exc:  # noqa: BLE001
        raise HubError(
            f"Failed to list blocks for namespace {namespace!r}: {exc}"
        ) from exc

    names: list[str] = []
    for repo in repos:
        # huggingface_hub returns ModelInfo-like objects. Extract id.
        repo_id = getattr(repo, "id", None) or getattr(repo, "modelId", None) or str(repo)
        if family is not None and f"-{family}-" not in repo_id and not repo_id.endswith(f"-{family}"):
            continue
        names.append(repo_id)
    return names


def load_block(
    name: str,
    *,
    revision: str = "main",
    cache_dir: str | Path | None = None,
) -> nn.Module:
    """Download a Qaithon block from the Hub and instantiate it.

    Args:
        name: Block identifier (e.g. ``"qaithon/quantum-attention-v1"``).
        revision: Git-like revision (branch, tag, or commit) on HF.
        cache_dir: Optional local cache directory; defaults to HF cache.

    Returns:
        ``nn.Module`` honoring the Tensor-in / Tensor-out contract.

    Raises:
        HubError: If the block is missing required metadata or weights.
        ImportError: If the block's declared class lives in a package not
            installed locally.
    """
    hf = _require_hf_hub()

    # Download the two artifacts.
    try:
        metadata_path = hf.hf_hub_download(
            repo_id=name,
            filename=_METADATA_FILENAME,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        weights_path = hf.hf_hub_download(
            repo_id=name,
            filename=_WEIGHTS_FILENAME,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    except Exception as exc:  # noqa: BLE001
        raise HubError(
            f"Failed to download block {name!r} at revision {revision!r}: {exc}. "
            "Either the block does not exist yet or your HuggingFace credentials "
            "are missing for a private repo."
        ) from exc

    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    return _instantiate_block(metadata, weights_path)


def _instantiate_block(metadata: dict[str, Any], weights_path: str) -> nn.Module:
    """Materialize the declared class and load its weights."""
    qualified = metadata.get("class")
    if not qualified or "." not in qualified:
        raise HubError(
            "Block metadata is missing a fully-qualified `class` field "
            f"(got: {qualified!r})."
        )
    module_path, _, class_name = qualified.rpartition(".")
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
    except (ImportError, AttributeError) as exc:
        raise HubError(
            f"Block declares class {qualified!r} but it is not importable on "
            "this machine. Install the required package and retry."
        ) from exc

    init_kwargs = metadata.get("init_kwargs", {})
    instance = cls(**init_kwargs)

    # Load weights via safetensors.
    if importlib.util.find_spec("safetensors") is None:
        raise HubError(
            "safetensors is not installed; install it with "
            "`pip install qaithon[huggingface]`."
        )
    from safetensors.torch import load_file

    state_dict = load_file(weights_path)
    instance.load_state_dict(state_dict)
    return instance


def push_block(
    name: str,
    module: nn.Module,
    *,
    metadata: dict[str, Any] | None = None,
    commit_message: str = "Update block",
    private: bool = False,
    revision: str = "main",
) -> str:
    """Upload a Qaithon block to the Hub.

    Args:
        name: Target repo id (``"qaithon/<family>-<variant>-vN"``).
        module: The trained module to upload. Must have a standard
            ``state_dict`` (no custom storage) for safetensors export.
        metadata: Extra metadata to merge into ``qaithon_block.json``. The
            ``class`` field is auto-filled from ``type(module)`` if not set.
        commit_message: Message for the HF Hub commit.
        private: Whether the repo should be marked private.
        revision: Branch / revision to push to.

    Returns:
        URL of the resulting repo.
    """
    hf = _require_hf_hub()

    if importlib.util.find_spec("safetensors") is None:
        raise HubError(
            "safetensors is not installed; install it with "
            "`pip install qaithon[huggingface]`."
        )

    import tempfile

    from safetensors.torch import save_file

    qualified_class = f"{type(module).__module__}.{type(module).__name__}"
    full_metadata: dict[str, Any] = {
        "class": qualified_class,
        "init_kwargs": {},
    }
    if metadata:
        full_metadata.update(metadata)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        save_file(module.state_dict(), str(tmp_dir / _WEIGHTS_FILENAME))
        (tmp_dir / _METADATA_FILENAME).write_text(
            json.dumps(full_metadata, indent=2),
            encoding="utf-8",
        )

        try:
            hf.create_repo(name, private=private, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            raise HubError(f"Failed to create or access repo {name!r}: {exc}") from exc

        try:
            hf.upload_folder(
                repo_id=name,
                folder_path=str(tmp_dir),
                commit_message=commit_message,
                revision=revision,
            )
        except Exception as exc:  # noqa: BLE001
            raise HubError(f"Failed to upload block to {name!r}: {exc}") from exc

    return f"https://huggingface.co/{name}"
