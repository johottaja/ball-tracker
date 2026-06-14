from __future__ import annotations

import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import TRAINING_SETS_DIR
from .dataset import load_dataset
from .model import ThrowGRU, ThrowGRUConfig


class ProgressCallback(Protocol):
    def __call__(
        self,
        *,
        epoch: int,
        total_epochs: int,
        train_loss: float,
        val_loss: float | None,
        val_accuracy: float | None,
        val_precision: float | None,
        val_recall: float | None,
        message: str,
    ) -> None: ...


@dataclass(frozen=True)
class TrainingHyperparameters:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 30
    validation_split: float = 0.2
    seed: int = 42
    pos_weight: float = 1.0


@dataclass
class TrainingResult:
    model: ThrowGRU
    set_name: str
    buffer_size: int
    hyperparameters: TrainingHyperparameters
    history: list[dict[str, float]]
    final_metrics: dict[str, float]


def list_training_sets() -> list[str]:
    if not TRAINING_SETS_DIR.is_dir():
        return []
    return sorted(path.stem for path in TRAINING_SETS_DIR.glob("*.npz"))


def training_set_path(set_name: str) -> Path:
    return TRAINING_SETS_DIR / f"{set_name}.npz"


def _build_clip_frame_masks(
    data: dict[str, np.ndarray],
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = data["clip_offsets"]
    counts = data["clip_frame_counts"]
    total_frames = len(data["labels"])
    clip_ids = np.empty(total_frames, dtype=np.int32)

    for clip_index, (offset, count) in enumerate(zip(offsets, counts, strict=True)):
        end = int(offset) + int(count)
        clip_ids[int(offset) : end] = clip_index

    num_clips = len(offsets)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_clips)
    if num_clips >= 2 and val_ratio > 0:
        val_count = max(1, int(round(num_clips * val_ratio)))
        val_clip_ids = set(perm[:val_count].tolist())
    else:
        val_clip_ids = set()

    in_val = np.isin(clip_ids, list(val_clip_ids))
    train_mask = ~in_val
    val_mask = in_val
    return train_mask, val_mask


def _prepare_arrays(
    data: dict[str, np.ndarray],
    hyperparameters: TrainingHyperparameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    windows = np.nan_to_num(data["windows"].astype(np.float32), nan=0.0)
    labels = data["labels"].astype(np.float32)
    sides = data["sides"]
    buffer_size = int(data["buffer_size"])

    valid = sides >= 0
    train_mask, val_mask = _build_clip_frame_masks(
        data,
        hyperparameters.validation_split,
        hyperparameters.seed,
    )
    train_mask &= valid
    val_mask &= valid

    return (
        windows[train_mask],
        labels[train_mask],
        windows[val_mask],
        labels[val_mask],
        buffer_size,
    )


def _make_loader(
    windows: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
) -> DataLoader | None:
    if len(labels) == 0:
        return None
    dataset = TensorDataset(
        torch.from_numpy(windows),
        torch.from_numpy(labels),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _binary_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, float]:
    if labels.numel() == 0:
        return 0.0, 0.0, 0.0

    preds = (logits >= 0).float()
    accuracy = (preds == labels).float().mean().item()

    positive = labels == 1
    negative = labels == 0
    true_positive = ((preds == 1) & positive).sum().item()
    false_positive = ((preds == 1) & negative).sum().item()
    false_negative = ((preds == 0) & positive).sum().item()

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    return accuracy, precision, recall


def _run_epoch(
    model: ThrowGRU,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float, float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch_windows, batch_labels in loader:
        if is_train and optimizer is not None:
            optimizer.zero_grad()

        logits = model(batch_windows)
        loss = criterion(logits, batch_labels)
        if is_train and optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = batch_labels.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_logits.append(logits.detach())
        all_labels.append(batch_labels)

    merged_logits = torch.cat(all_logits)
    merged_labels = torch.cat(all_labels)
    accuracy, precision, recall = _binary_metrics(merged_logits, merged_labels)
    mean_loss = total_loss / max(total_samples, 1)
    return mean_loss, accuracy, precision, recall


def train_throw_model(
    set_name: str,
    hyperparameters: TrainingHyperparameters,
    *,
    progress: ProgressCallback | None = None,
    stop_event: threading.Event | None = None,
) -> TrainingResult:
    path = training_set_path(set_name)
    if not path.is_file():
        raise FileNotFoundError(f"Training set not found: {path}")

    torch.manual_seed(hyperparameters.seed)
    data = load_dataset(path)
    train_x, train_y, val_x, val_y, buffer_size = _prepare_arrays(
        data,
        hyperparameters,
    )

    if len(train_y) == 0:
        raise ValueError("No valid training frames after filtering missing pose data.")

    train_loader = _make_loader(
        train_x,
        train_y,
        hyperparameters.batch_size,
        shuffle=True,
    )
    assert train_loader is not None

    val_loader = _make_loader(
        val_x,
        val_y,
        hyperparameters.batch_size,
        shuffle=False,
    )

    model_config = ThrowGRUConfig(
        input_size=4,
        hidden_size=hyperparameters.hidden_size,
        num_layers=hyperparameters.num_layers,
        dropout=hyperparameters.dropout,
    )
    model = ThrowGRU(model_config)
    pos_weight = (
        torch.tensor([hyperparameters.pos_weight], dtype=torch.float32)
        if hyperparameters.pos_weight != 1.0
        else None
    )
    criterion: nn.Module = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters.learning_rate)

    history: list[dict[str, float]] = []
    final_metrics: dict[str, float] = {}

    for epoch in range(1, hyperparameters.epochs + 1):
        if stop_event is not None and stop_event.is_set():
            break

        train_loss, train_acc, train_prec, train_rec = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_loss: float | None = None
        val_accuracy: float | None = None
        val_precision: float | None = None
        val_recall: float | None = None

        if val_loader is not None:
            val_loss, val_accuracy, val_precision, val_recall = _run_epoch(
                model,
                val_loader,
                criterion,
                None,
            )
            final_metrics = {
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
            }
        else:
            final_metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_precision": train_prec,
                "train_recall": train_rec,
            }

        epoch_metrics: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_precision": train_prec,
            "train_recall": train_rec,
        }
        if val_loss is not None:
            epoch_metrics.update(
                {
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy or 0.0,
                    "val_precision": val_precision or 0.0,
                    "val_recall": val_recall or 0.0,
                },
            )
        history.append(epoch_metrics)

        if progress is not None:
            if val_loader is not None:
                message = (
                    f"Epoch {epoch}/{hyperparameters.epochs} — "
                    f"train loss {train_loss:.4f}, acc {train_acc:.3f} — "
                    f"val loss {val_loss:.4f}, acc {val_accuracy:.3f}, "
                    f"prec {val_precision:.3f}, rec {val_recall:.3f}"
                )
            else:
                message = (
                    f"Epoch {epoch}/{hyperparameters.epochs} — "
                    f"train loss {train_loss:.4f}, acc {train_acc:.3f}, "
                    f"prec {train_prec:.3f}, rec {train_rec:.3f}"
                )
            progress(
                epoch=epoch,
                total_epochs=hyperparameters.epochs,
                train_loss=train_loss,
                val_loss=val_loss,
                val_accuracy=val_accuracy,
                val_precision=val_precision,
                val_recall=val_recall,
                message=message,
            )

    return TrainingResult(
        model=model,
        set_name=set_name,
        buffer_size=buffer_size,
        hyperparameters=hyperparameters,
        history=history,
        final_metrics=final_metrics,
    )


def hyperparameters_to_dict(hyperparameters: TrainingHyperparameters) -> dict:
    return asdict(hyperparameters)
