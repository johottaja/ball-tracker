from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ThrowGRUConfig:
    input_size: int = 4
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1


class ThrowGRU(nn.Module):
    def __init__(self, config: ThrowGRUConfig) -> None:
        super().__init__()
        self.config = config
        gru_dropout = config.dropout if config.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            config.input_size,
            config.hidden_size,
            config.num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Linear(config.hidden_size, 1)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        """windows: (batch, seq_len, features) -> logits (batch,)"""
        output, _ = self.gru(windows)
        last = output[:, -1, :]
        return self.head(last).squeeze(-1)


def save_throw_model(
    path: Path,
    model: ThrowGRU,
    *,
    set_name: str,
    buffer_size: int,
    training_config: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "model_config": asdict(model.config),
        "set_name": set_name,
        "buffer_size": buffer_size,
        "training_config": training_config,
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_throw_model(path: Path, map_location: str | torch.device = "cpu") -> tuple[
    ThrowGRU,
    dict[str, Any],
]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = ThrowGRUConfig(**payload["model_config"])
    model = ThrowGRU(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metadata = {
        key: payload[key]
        for key in ("set_name", "buffer_size", "training_config", "metrics")
        if key in payload
    }
    return model, metadata
