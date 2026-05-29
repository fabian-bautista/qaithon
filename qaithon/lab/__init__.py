"""All-in-one workspace for experimenting with Qaithon-compiled models.

The :mod:`qaithon.lab` namespace bundles the pieces a developer needs to
go from "I have an idea" to "I have a trained toy model running on a
quantum backend" in one or two lines.

It is intentionally separate from the core library: production users who
just want to compile a HuggingFace model never touch ``qaithon.lab``.
Researchers and developers experimenting locally can.

Modules:

* :mod:`qaithon.lab.datasets` — tiny character-level corpora bundled or
  downloaded on first use.
* :mod:`qaithon.lab.training` — minimal training loop with built-in QAT
  support, no HuggingFace ``Trainer`` dependency required.
* :mod:`qaithon.lab.inference` — convenience wrappers over
  ``model.generate`` with the toy tokenizer plugged in.
"""

from __future__ import annotations

from qaithon.lab.datasets import CharDataset, load_dataset
from qaithon.lab.inference import generate
from qaithon.lab.training import train

__all__ = ["CharDataset", "generate", "load_dataset", "train"]
