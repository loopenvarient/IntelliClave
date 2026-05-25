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

Model Inversion Defence
-----------------------
Use get_defended_model() instead of get_model() when serving predictions.
It wraps any architecture in PrivacyWrapper, which applies two defences at
inference time:

  1. Output perturbation — Laplace noise added to raw logits before softmax.
     Breaks the clean gradient signal the inversion optimizer needs.

  2. Temperature scaling — logits divided by T > 1 before softmax.
     Flattens confidence peaks so reconstructed inputs have lower cosine
     similarity to real class centroids.

Training is unaffected: PrivacyWrapper.forward() is only active during
model.eval() (i.e. torch.no_grad() inference calls). During training
(model.train()) it passes logits through unchanged so DP-SGD and loss
computation work exactly as before.
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
# Model Inversion Defence — PrivacyWrapper
# ─────────────────────────────────────────────────────────────────────────────

class PrivacyWrapper(nn.Module):
    """
    Wraps any classifier with two output-space model-inversion defences.

    Defences are applied ONLY during eval() (i.e. inference / torch.no_grad()).
    During train() they are bypassed so that DP-SGD, loss computation, and
    Opacus gradient hooks all see clean, unperturbed logits.

    Parameters
    ----------
    base_model   : any nn.Module that returns raw logits
    noise_scale  : scale of Laplace noise added to logits (default 0.5).
                   Increase until avg cosine similarity in the attack drops
                   below 0.6. Values above ~2.0 start degrading top-1 accuracy.
    temperature  : softmax temperature T > 1 (default 4.0).
                   Divides logits before softmax — flattens confidence peaks.
                   Does not change the argmax (predicted class), only probabilities.
    enabled      : set False to disable both defences (ablation / attack testing).

    Usage
    -----
    # At inference (dashboard /predict endpoint):
    defended = PrivacyWrapper(base_model, noise_scale=0.5, temperature=4.0)
    defended.eval()
    with torch.no_grad():
        probs = defended(x)   # returns softmax probabilities, not logits

    # During FL training — use base_model directly, not the wrapper:
    base_model.train()
    logits = base_model(x)    # raw logits, defences inactive
    loss = criterion(logits, y)
    """

    def __init__(
        self,
        base_model: nn.Module,
        noise_scale: float = 0.5,
        temperature: float = 4.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.base_model  = base_model
        self.noise_scale = noise_scale
        self.temperature = temperature
        self.enabled     = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x)

        # Defences only active during inference (eval mode + no_grad context).
        # self.training is False whenever model.eval() has been called.
        if self.enabled and not self.training:
            # 1. Output perturbation — Laplace noise on raw logits.
            #    Breaks the clean gradient signal the inversion optimizer follows.
            noise  = torch.distributions.Laplace(
                torch.zeros_like(logits),
                self.noise_scale * torch.ones_like(logits),
            ).sample()
            logits = logits + noise

            # 2. Temperature scaling — flatten confidence peaks.
            #    Does not change argmax; only squeezes the probability vector.
            logits = logits / self.temperature

        # Always return softmax probabilities from the defended model so the
        # /predict endpoint doesn't need to call softmax separately.
        # During training this is still raw logits / temperature=1 → same as before
        # because the temperature branch above is skipped.
        return torch.softmax(logits, dim=-1) if not self.training else logits


# ─────────────────────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────────────────────

def get_model(
    input_dim: int,
    num_classes: int,
    model_type: ModelType = "mlp",
) -> nn.Module:
    """
    Build and return a bare model for FL training.

    Parameters
    ----------
    input_dim  : number of input features (inferred from CSV at runtime)
    num_classes: number of output classes  (inferred from CSV at runtime)
    model_type : "mlp" | "resnet-tabular" | "transformer-tabular"

    Returns raw logits — use get_defended_model() for inference serving.
    """
    if model_type == "resnet-tabular":
        return ResNetTabular(input_dim=input_dim, num_classes=num_classes)
    if model_type == "transformer-tabular":
        return TransformerTabular(input_dim=input_dim, num_classes=num_classes)
    return FLClassifier(input_dim=input_dim, num_classes=num_classes)


def get_defended_model(
    input_dim: int,
    num_classes: int,
    model_type: ModelType = "mlp",
    noise_scale: float = 0.5,
    temperature: float = 4.0,
    enabled: bool = True,
) -> PrivacyWrapper:
    """
    Build a model wrapped in PrivacyWrapper for inference serving.

    Call this in the dashboard /predict endpoint instead of get_model().
    The returned wrapper shares no state with the training model — load
    state_dict into wrapper.base_model after calling this function:

        wrapper = get_defended_model(input_dim, num_classes)
        wrapper.base_model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        wrapper.eval()

    Parameters
    ----------
    noise_scale : Laplace noise scale on logits (default from MI_NOISE_SCALE).
    temperature : softmax temperature divisor  (default from MI_TEMPERATURE).
    enabled     : master switch — False disables both defences.
    """
    base = get_model(input_dim, num_classes, model_type)
    return PrivacyWrapper(
        base_model=base,
        noise_scale=noise_scale,
        temperature=temperature,
        enabled=enabled,
    )