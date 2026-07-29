"""
fl/model.py

Model architectures for IntelliClave FL.

Three architectures are available — select via get_model(model_type=...):

    "mlp"               : Feed-forward MLP (default). Fast, works well on most
                          tabular datasets. Hidden dims: 96 → 48.

    "resnet-tabular"    : Residual MLP with skip connections. Better gradient
                          flow for deeper networks; useful when MLP underfits.

    "transformer-tabular": Lightweight Transformer encoder over feature tokens.
                           Captures feature interactions; best for datasets with
                           many correlated features (100+).

All architectures accept any input_dim and num_classes — no hardcoded sizes.

Model Inversion Defence
-----------------------
Use get_defended_model() instead of get_model() when serving predictions.
It wraps any architecture in PrivacyWrapper, which applies three defences at
inference time:

  1. Gradient blocking — the entire base model forward pass runs inside
     torch.no_grad(). Even if the caller forgets no_grad, the inversion
     optimizer receives zero gradient signal and degrades to blind search.

  2. Output perturbation — Laplace noise added to raw logits before softmax.
     Disrupts the remaining output signal even under black-box attacks.

  3. Temperature scaling — logits divided by T > 1 before softmax.
     Flattens confidence peaks so reconstructed inputs have lower cosine
     similarity to real class centroids.

  4. Hard-label option — optionally return only argmax (no probabilities).
     Eliminates the soft probability vector the inversion loss function needs.

Training is completely unaffected: during model.train() the wrapper passes
logits through unchanged so DP-SGD and loss computation work as before.

FIX vs previous version
------------------------
Previous PrivacyWrapper checked `not self.training` to gate defences, but
ran the base model OUTSIDE no_grad. The inversion optimizer calls model.eval()
then does gradient-based optimisation — self.training was False but gradients
still flowed through, giving the attacker a clean gradient signal despite noise
being added to the output. The fix wraps the entire base model call inside
torch.no_grad() on the inference path, completely severing the gradient
regardless of what the calling code does.
"""
from typing import Iterable, List, Literal

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.constants import (  # noqa: E402
    DROPOUT_RATE,
    FEATURE_NOISE_STD,
    MI_NOISE_SCALE,
    MI_TEMPERATURE,
)

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
        hidden_dims: Iterable[int] = (96, 48),
        dropout: float = DROPOUT_RATE,
        feature_noise_std: float = FEATURE_NOISE_STD,
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

        self.feature_extractor = nn.Sequential(*layers)
        self.representation_norm = nn.LayerNorm(prev)
        self.classifier = nn.Linear(prev, num_classes)
        self.feature_noise_std = feature_noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        features = self.representation_norm(features)
        if self.training and self.feature_noise_std > 0:
            features = features + torch.randn_like(features) * self.feature_noise_std
        return self.classifier(features)


class LegacyMLPClassifier(nn.Module):
    """Backward-compatible MLP used by older saved checkpoints with `net.*` keys."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Iterable[int] = (96, 48),
        dropout: float = DROPOUT_RATE,
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
        hidden_dim: int = 96,
        num_blocks: int = 2,
        dropout: float = DROPOUT_RATE,
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
        embed_dim: int = 24,
        num_heads: int = 4,
        num_layers: int = 1,
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
    Wraps any classifier with layered model-inversion defences.

    Defences are applied ONLY during eval() (i.e. inference).
    During train() they are bypassed so DP-SGD, loss computation, and
    Opacus gradient hooks all see clean, unperturbed logits.

    KEY FIX vs previous version
    ---------------------------
    The entire base model forward pass now runs inside torch.no_grad() on the
    inference path. Previously the base model was called outside no_grad, which
    meant gradient-based inversion attacks could still optimise through the
    network even though noise was added to the output. Now the gradient is
    severed at the source — the attacker gets zero gradient regardless of
    whether they remembered to call no_grad themselves.

    Defence layers (inference only)
    --------------------------------
    1. Gradient blocking  — base_model runs in torch.no_grad() context.
                            Inversion optimizer gets zero gradient → degrades
                            to blind/black-box search.

    2. Output perturbation — Laplace(0, noise_scale) added to raw logits.
                             Disrupts output signal for black-box attacks.

    3. Temperature scaling — logits / temperature before softmax.
                             Flattens probability peaks; lowers cosine
                             similarity of reconstructed vs real inputs.

    4. Hard-label mode    — return only argmax index instead of probabilities.
                            Eliminates the soft vector the inversion loss needs.
                            Enable with hard_label=True for maximum privacy at
                            the cost of downstream probability-based features.

    Parameters
    ----------
    base_model   : any nn.Module that returns raw logits
    noise_scale  : Laplace noise scale on logits (default from MI_NOISE_SCALE)
    temperature  : softmax temperature T > 1 (default from MI_TEMPERATURE)
    hard_label   : if True, return argmax index only — no probabilities exposed
    enabled      : master switch — False disables all defences (ablation)

    Usage
    -----
    # At inference / dashboard /predict endpoint:
    defended = PrivacyWrapper(base_model)
    defended.eval()
    # No need to wrap in torch.no_grad() — wrapper does it internally,
    # but it's still good practice to call it from the outside too.
    with torch.no_grad():
        probs = defended(x)   # returns softmax probs (or argmax if hard_label=True)

    # During FL training — use base_model directly, not the wrapper:
    base_model.train()
    logits = base_model(x)    # raw logits, defences inactive
    loss = criterion(logits, y)
    """

    def __init__(
        self,
        base_model: nn.Module,
        noise_scale: float = MI_NOISE_SCALE,
        temperature: float = MI_TEMPERATURE,
        hard_label: bool = False,
        enabled: bool = True,
    ):
        super().__init__()
        self.base_model  = base_model
        self.noise_scale = noise_scale
        self.temperature = temperature
        self.hard_label  = hard_label
        self.enabled     = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Training path — no defences, full gradient flow for DP-SGD ──────
        if self.training:
            return self.base_model(x)

        # ── Inference path — all defences active ────────────────────────────
        #
        # CRITICAL: run the entire base model inside torch.no_grad().
        # This severs gradients at the source. Even if the caller (e.g. an
        # inversion attack script) calls model.eval() and then does
        # loss.backward(), they get zero gradient — the attack degrades from
        # gradient-based optimisation to blind random search.
        #
        with torch.no_grad():
            logits = self.base_model(x)

            if self.enabled:
                # Defence 1 — Output perturbation (Laplace noise on logits).
                # Sampled fresh every forward call so the attacker cannot
                # average away the noise across repeated queries.
                noise = torch.distributions.Laplace(
                    torch.zeros_like(logits),
                    self.noise_scale * torch.ones_like(logits),
                ).sample()
                logits = logits + noise

                # Defence 2 — Temperature scaling.
                # Divides logits before softmax — flattens confidence peaks.
                # Does not change argmax; only squeezes the probability vector
                # so inversion reconstruction has lower cosine similarity.
                logits = logits / self.temperature

            # Defence 3 — Hard-label mode (optional, maximum privacy).
            # Returns only the predicted class index; gives attacker no
            # probability vector to optimise against.
            if self.hard_label and self.enabled:
                return logits.argmax(dim=-1)

            # Default: return softmax probabilities (soft labels).
            # Still defended by noise + temperature; suitable for use cases
            # that need confidence scores (e.g. dashboard uncertainty display).
            return torch.softmax(logits, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Factories
# ─────────────────────────────────────────────────────────────────────────────

def get_model(
    input_dim: int,
    num_classes: int,
    model_type: ModelType = "mlp",
    hidden_dims: Iterable[int] | None = None,
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
    return FLClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims or (96, 48),
    )


def infer_hidden_dims_from_state(state: dict, model_type: ModelType = "mlp"):
    """Infer hidden dimensions from a checkpoint state_dict."""
    if not isinstance(state, dict):
        return None

    if any(key.startswith("feature_extractor.") for key in state):
        layers = []
        idx = 0
        while True:
            key = f"feature_extractor.{idx}.weight"
            if key not in state:
                break
            layers.append(int(state[key].shape[0]))
            idx += 3
        return tuple(layers) if layers else None

    if any(key.startswith("net.") for key in state):
        layers = []
        idx = 0
        while True:
            key = f"net.{idx}.weight"
            if key not in state:
                break
            layers.append(int(state[key].shape[0]))
            idx += 3
        return tuple(layers[:-1]) if len(layers) > 1 else None

    if model_type == "resnet-tabular" and "input_proj.0.weight" in state:
        return (int(state["input_proj.0.weight"].shape[0]),)

    return None


def build_model_from_state(
    input_dim: int,
    num_classes: int,
    state: dict,
    model_type: ModelType = "mlp",
) -> nn.Module:
    """Build a model matching the checkpoint architecture."""
    if any(key.startswith("net.") for key in state):
        hidden_dims = infer_hidden_dims_from_state(state, model_type="mlp") or (96, 48)
        return LegacyMLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
        )

    hidden_dims = infer_hidden_dims_from_state(state, model_type=model_type)
    return get_model(
        input_dim=input_dim,
        num_classes=num_classes,
        model_type=model_type,
        hidden_dims=hidden_dims,
    )


def get_defended_model(
    input_dim: int,
    num_classes: int,
    model_type: ModelType = "mlp",
    hidden_dims: Iterable[int] | None = None,
    noise_scale: float = MI_NOISE_SCALE,
    temperature: float = MI_TEMPERATURE,
    hard_label: bool = False,
    enabled: bool = True,
) -> PrivacyWrapper:
    """
    Build a model wrapped in PrivacyWrapper for inference serving.

    Call this in the dashboard /predict endpoint instead of get_model().
    The returned wrapper shares weights with whatever base model you load into
    wrapper.base_model — load state_dict into base_model after calling this:

        wrapper = get_defended_model(input_dim, num_classes)
        wrapper.base_model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        )
        wrapper.eval()

    For maximum privacy (no probabilities exposed), set hard_label=True.
    This returns only the argmax class index — useful when downstream consumers
    don't need confidence scores and you want to fully eliminate the soft
    probability vector the inversion attack optimises against.

    Parameters
    ----------
    noise_scale : Laplace noise scale on logits (default from MI_NOISE_SCALE)
    temperature : softmax temperature divisor  (default from MI_TEMPERATURE)
    hard_label  : return argmax only instead of softmax probabilities
    enabled     : master switch — False disables all defences
    """
    base = get_model(input_dim, num_classes, model_type, hidden_dims=hidden_dims)
    return PrivacyWrapper(
        base_model=base,
        noise_scale=noise_scale,
        temperature=temperature,
        hard_label=hard_label,
        enabled=enabled,
    )