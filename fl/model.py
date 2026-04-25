from typing import Iterable, List

import torch
import torch.nn as nn


class HARClassifier(nn.Module):
    """Feed-forward multiclass classifier for 50-dim PCA HAR features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 6,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev = hidden_dim

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_model(input_dim: int, num_classes: int = 6) -> HARClassifier:
    return HARClassifier(input_dim=input_dim, num_classes=num_classes)
