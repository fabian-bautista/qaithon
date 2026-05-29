"""Tiny datasets bundled with Qaithon for experimentation.

We deliberately avoid ``huggingface_hub.datasets`` here — that pulls in a
heavy dependency tree and ties the lab to HF's caching semantics. The
datasets here are small enough (Shakespeare ~1 MB, TinyStories sample
~2 MB) to download on first use and store under ``~/.qaithon/datasets``.
"""

from __future__ import annotations

import hashlib
import pathlib
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from qaithon._logging import get_logger
from qaithon.models import ToyTokenizer

if TYPE_CHECKING:
    pass

__all__ = ["CharDataset", "load_dataset"]

logger = get_logger(__name__)

_CACHE_DIR = pathlib.Path.home() / ".qaithon" / "datasets"


# Public, free-to-use char-level corpora.
_DATASETS = {
    "shakespeare": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "filename": "shakespeare.txt",
        "description": "Karpathy's char-level Shakespeare (~1 MB).",
    },
    "wizard_of_oz": {
        "url": "https://www.gutenberg.org/cache/epub/55/pg55.txt",
        "filename": "wizard_of_oz.txt",
        "description": "The Wonderful Wizard of Oz, Project Gutenberg (~200 KB).",
    },
}


class CharDataset(Dataset):
    """Character-level dataset that yields fixed-length token sequences.

    Args:
        text: Raw string corpus.
        tokenizer: A :class:`qaithon.models.ToyTokenizer` instance.
        block_size: Length of each training sample (number of tokens).

    Each ``__getitem__`` returns ``(input_ids, target_ids)`` — the target
    is the input shifted by one position (standard language modeling).
    """

    def __init__(
        self,
        text: str,
        tokenizer: ToyTokenizer,
        block_size: int = 64,
    ) -> None:
        if block_size < 4:
            raise ValueError(f"block_size must be >= 4, got {block_size}.")
        self._tokenizer = tokenizer
        self._block_size = block_size

        token_ids = tokenizer.encode(text, add_bos=False)
        if len(token_ids) <= block_size:
            raise ValueError(
                f"Corpus too short ({len(token_ids)} tokens) for block_size={block_size}."
            )
        self._tokens = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return self._tokens.size(0) - self._block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self._tokens[idx : idx + self._block_size + 1]
        return chunk[:-1], chunk[1:]


@dataclass(frozen=True, slots=True)
class _DatasetInfo:
    """Lightweight handle to a downloaded corpus."""

    name: str
    text: str
    n_chars: int
    cached_path: pathlib.Path


def _ensure_downloaded(name: str) -> pathlib.Path:
    if name not in _DATASETS:
        raise KeyError(
            f"Unknown dataset {name!r}. Available: {sorted(_DATASETS.keys())}"
        )
    info = _DATASETS[name]
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / info["filename"]
    if path.is_file():
        return path
    logger.info("Downloading dataset %r from %s", name, info["url"])
    with urllib.request.urlopen(info["url"], timeout=30) as resp:
        path.write_bytes(resp.read())
    logger.info(
        "Cached %r at %s (%d bytes, sha=%s)",
        name,
        path,
        path.stat().st_size,
        hashlib.sha256(path.read_bytes()).hexdigest()[:12],
    )
    return path


def load_dataset(
    name: str,
    *,
    tokenizer: ToyTokenizer | None = None,
    block_size: int = 64,
) -> CharDataset:
    """Load and tokenize a built-in corpus.

    Args:
        name: One of ``"shakespeare"`` | ``"wizard_of_oz"``.
        tokenizer: Optional tokenizer; defaults to a fresh :class:`ToyTokenizer`.
        block_size: Length of each training sample.

    Returns:
        :class:`CharDataset` ready to feed into :func:`qaithon.lab.train`.

    Example:
        >>> from qaithon.lab import load_dataset
        >>> ds = load_dataset("shakespeare", block_size=64)
        >>> len(ds)  # doctest: +SKIP
        ~1000000
    """
    path = _ensure_downloaded(name)
    text = path.read_text(encoding="utf-8", errors="ignore")
    return CharDataset(
        text=text,
        tokenizer=tokenizer or ToyTokenizer(),
        block_size=block_size,
    )
