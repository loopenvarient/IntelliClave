"""
tee/full_stack_test/pytorch_test.py

PyTorch forward pass inside Gramine.
Tests that torch runs correctly inside the enclave with a generic
feed-forward classifier — dimensions are not tied to any specific dataset.

Run:
    gramine-manifest pytorch_test.manifest.template pytorch_test.manifest
    gramine-direct python3 pytorch_test.py
"""

import sys
import torch
import torch.nn as nn

print("=" * 50)
print("IntelliClave — PyTorch in Gramine Test")
print("=" * 50)

# ── Generic feed-forward classifier ──────────────────
INPUT_DIM   = 32   # synthetic — not tied to any dataset
NUM_CLASSES = 4

class FLClassifier(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# ── Forward pass ──────────────────────────────────────
model = FLClassifier(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
model.eval()

batch = torch.randn(4, INPUT_DIM)
with torch.no_grad():
    logits = model(batch)
    preds  = logits.argmax(dim=1)

print(f"  PyTorch version : {torch.__version__}")
print(f"  Input shape     : {list(batch.shape)}")
print(f"  Output shape    : {list(logits.shape)}")
print(f"  Predictions     : {preds.tolist()}")
print()
print("TORCH IN GRAMINE OK ✓")
print("=" * 50)
