from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


def make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU()
    raise ValueError(f"Unknown activation '{name}'")


@dataclass(frozen=True)
class MLPConfig:
    layer_dims: Tuple[int, ...] = (2, 50, 50, 50, 50, 3)
    activation: str = "tanh"
    init: str = "xavier_uniform"


class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig = MLPConfig()):
        super().__init__()
        self.cfg = cfg

        dims = tuple(int(d) for d in cfg.layer_dims)
        if len(dims) < 2 or any(d <= 0 for d in dims):
            raise ValueError("MLPConfig.layer_dims must contain at least two positive dimensions.")

        activation = make_activation(cfg.activation)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation)

        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def reset_parameters(self) -> None:
        init = self.cfg.init.lower()

        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if init == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif init == "xavier_normal":
                nn.init.xavier_normal_(module.weight)
            elif init == "kaiming_uniform":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            elif init == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            else:
                raise ValueError(f"Unknown init '{self.cfg.init}'")
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)            


__all__ = ["MLP", "MLPConfig", "make_activation"]
