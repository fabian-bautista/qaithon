"""Minimal training loop with built-in Quantization-Aware Training support.

This is a deliberately small training utility — ~80 lines, no
HuggingFace ``Trainer`` dependency, no ``accelerate``, no DeepSpeed. The
goal is "one function call trains a toy model end to end". For real
production training, users should fall through to ``Trainer`` /
``peft`` / etc.; this is for the lab.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader

from qaithon._logging import get_logger

if TYPE_CHECKING:
    from torch.utils.data import Dataset

__all__ = ["TrainingResult", "train"]

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """What :func:`train` returns when it finishes.

    Attributes:
        steps: Number of optimizer steps executed.
        final_loss: Cross-entropy loss of the last logged batch.
        loss_history: One float per logged batch.
        total_time_s: Wall-clock duration.
        device: The torch device the training ran on.
    """

    steps: int
    final_loss: float
    loss_history: list[float] = field(default_factory=list)
    total_time_s: float = 0.0
    device: str = "cpu"


def train(
    model: nn.Module,
    dataset: "Dataset",
    *,
    steps: int = 1000,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    target_hardware: str | None = None,
    device: str | torch.device | None = None,
    log_every: int = 100,
    seed: int = 0,
) -> TrainingResult:
    """Train ``model`` on ``dataset`` for ``steps`` optimizer iterations.

    Args:
        model: Any ``nn.Module``. Typically a Qaithon-compiled toy
            transformer.
        dataset: A torch ``Dataset`` yielding ``(input_ids, target_ids)``.
            :class:`qaithon.lab.CharDataset` works directly.
        steps: Number of optimizer steps.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        target_hardware: Optional name of a hardware target (e.g.
            ``"IBM Heron"``, ``"IBM Starling"``). When set, the function
            **validates the model fits the target before training**, and
            raises :class:`~qaithon.exceptions.IncompatibleHardwareError`
            with concrete recommendations if not. This avoids burning
            training time on a model the destination QPU cannot run.
        device: Where to run. Defaults to MPS if available, then CUDA,
            then CPU.
        log_every: How often to record the running loss.
        seed: For reproducibility.

    Returns:
        :class:`TrainingResult` with loss curve and timing.

    Example:
        >>> import qaithon
        >>> from qaithon.lab import load_dataset, train
        >>> model = qaithon.models.create_toy_transformer(dim=64, n_layers=2)
        >>> qaithon.compile(model, backends=("quandela.sim",),
        ...                 min_in_features=1, min_out_features=1)
        >>> ds = load_dataset("shakespeare", block_size=64)
        >>> result = train(model, ds, steps=500)  # doctest: +SKIP
        >>> print(result.final_loss)  # doctest: +SKIP
    """
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}.")

    # Fail-fast hardware validation: if the user named a target, check the
    # model fits before we spend a single optimizer step.
    if target_hardware is not None:
        from qaithon.exceptions import IncompatibleHardwareError
        from qaithon.qubits import validate_for_hardware

        validation = validate_for_hardware(model, target=target_hardware)
        if not validation.fits:
            raise IncompatibleHardwareError(
                reason=(
                    f"Cannot train this model for {validation.target.name}: "
                    + "; ".join(validation.reasons)
                ),
                recommendations=list(validation.recommendations),
            )
        logger.info(
            "Hardware validation passed: model fits %s "
            "(%d qubits required, %d available; depth %d ≤ %d).",
            validation.target.name,
            validation.report.max_qubits_block,
            validation.target.logical_qubits or validation.target.physical_qubits,
            validation.report.max_circuit_depth,
            validation.target.max_coherent_depth,
        )

    torch.manual_seed(seed)
    device_ = torch.device(_pick_device(device))
    logger.info("Training on device: %s", device_)


    model.to(device_)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    loss_history: list[float] = []
    last_logged_loss = float("inf")
    step = 0
    t_start = time.perf_counter()

    while step < steps:
        for inputs, targets in loader:
            if step >= steps:
                break
            inputs, targets = inputs.to(device_), targets.to(device_)

            optimizer.zero_grad()
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % log_every == 0 or step == steps:
                last_logged_loss = float(loss.item())
                loss_history.append(last_logged_loss)
                logger.info(
                    "step=%6d/%d  loss=%.4f  elapsed=%5.1fs",
                    step,
                    steps,
                    last_logged_loss,
                    time.perf_counter() - t_start,
                )

    model.eval()
    return TrainingResult(
        steps=step,
        final_loss=last_logged_loss,
        loss_history=loss_history,
        total_time_s=time.perf_counter() - t_start,
        device=str(device_),
    )


def _pick_device(requested: str | torch.device | None) -> str:
    if requested is not None:
        return str(requested)
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
