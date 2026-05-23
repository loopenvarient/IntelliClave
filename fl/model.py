"""
fl/model.py

Model architectures for IntelliClave FL.

Three architectures are available — select via get_model(model_type=...):

  "mlp"               : Feed-forward MLP (default). Fast, works well on most
                        tabular datasets. Hidden dims: 128 → 64.

  "resnet-tabular"    : Residual MLP with skip connections. Better gradient
                        flow for deeper networks; useful when MLP underfits.

  "transformer-tabular": Lightweight Transformer encoder over feature tokens.
                        Captures feature interactions; best for datasets with
                        many correlated features (100+).

All architectures accept any input_dim and num_classes — no hardcoded sizes.
"""
from typing import Iterable, List, Literal

import torch
import torch.nn as nn

ModelType = Literal["mlp", "resnet-tabular", "transformer-tabular"]


# ─────────────────────────────────────────────────────────────────────────────
# MLP (default)
# ─────────────────────────────────────────────────────────────────────────────

class FLClassifier(nn.Module):
    """
    Generic feed-forward MLP classifier.
    Works with any tabular dataset — input_dim and num_classes are inferred
    at runtime from the data.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hidden_dim

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Backward-compat alias
HARClassifier = FLClassifier


# ─────────────────────────────────────────────────────────────────────────────
# Residual MLP (ResNet-Tabular)
# ─────────────────────────────────────────────────────────────────────────────

class _ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ResNetTabular(nn.Module):
    """
    Residual MLP for tabular data.
    Projects input to a fixed hidden_dim, then applies N residual blocks,
    then classifies. Skip connections help with deeper networks.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(
            *[_ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# Transformer-Tabular
# ─────────────────────────────────────────────────────────────────────────────

class TransformerTabular(nn.Module):
    """
    Lightweight Transformer encoder for tabular data.

    Each feature is treated as a token (projected to embed_dim). A small
    Transformer encoder captures feature interactions, then the CLS token
    is used for classification.

    Best suited for datasets with many correlated features (100+).
    For small feature counts (<20), MLP or ResNet-Tabular will likely perform
    equally well with less compute.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Project each feature scalar to embed_dim
        self.feature_embed = nn.Linear(1, embed_dim)
        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Positional encoding (simple learned)
        self.pos_embed = nn.Parameter(torch.zeros(1, input_dim + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, F = x.shape
        # (B, F, 1) → (B, F, embed_dim)
        tokens = self.feature_embed(x.unsqueeze(-1))
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)          # (B, F+1, embed_dim)
        tokens = tokens + self.pos_embed[:, :F + 1, :]
        encoded = self.encoder(tokens)                     # (B, F+1, embed_dim)
        return self.head(encoded[:, 0])                    # CLS token → logits


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def get_model(
    input_dim: int,
    num_classes: int,
    model_type: ModelType = "mlp",
    dropout: float = 0.3,
) -> nn.Module:
    """
    Build and return a model for FL classification.

    Parameters
    ----------
    input_dim  : number of input features (inferred from CSV at runtime)
    num_classes: number of output classes  (inferred from CSV at runtime)
    model_type : "mlp" | "resnet-tabular" | "transformer-tabular"
    dropout    : dropout rate (default 0.3). Pass 0.0 when using DP training
                 to avoid compounding DP noise with dropout variance — the
                 privacy guarantee already acts as regularisation.
    """
    if model_type == "resnet-tabular":
        return ResNetTabular(input_dim=input_dim, num_classes=num_classes,
                             dropout=dropout)
    if model_type == "transformer-tabular":
        return TransformerTabular(input_dim=input_dim, num_classes=num_classes,
                                  dropout=dropout)
    # default: mlp
    return FLClassifier(input_dim=input_dim, num_classes=num_classes,
                        dropout=dropout)
