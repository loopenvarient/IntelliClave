"""
tee/full_stack_test/pytorch_test.py

PyTorch forward pass inside Gramine.
Input:  50 PCA features (matches IntelliClave HARClassifier)
Output: 6 class logits

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

# ── Model matching HARClassifier architecture ─────────
class HARClassifier(nn.Module):
    def __init__(self, input_dim=50, num_classes=6):
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
model = HARClassifier(input_dim=50, num_classes=6)
model.eval()

batch = torch.randn(4, 50)          # batch of 4 samples, 50 features each
with torch.no_grad():
    logits = model(batch)           # shape: (4, 6)
    preds  = logits.argmax(dim=1)   # predicted class per sample

print(f"  PyTorch version : {torch.__version__}")
print(f"  Input shape     : {list(batch.shape)}")
print(f"  Output shape    : {list(logits.shape)}")
print(f"  Predictions     : {preds.tolist()}")
print()
print("TORCH IN GRAMINE OK ✓")
print("=" * 50)
