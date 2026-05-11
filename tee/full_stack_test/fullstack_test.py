"""
tee/full_stack_test/fullstack_test.py

Full stack test: torch + flwr + opacus all working inside Gramine.
Also runs one training step with PrivacyEngine active.

This confirms the entire FL+DP pipeline can run inside the enclave.

Run:
    gramine-manifest fullstack_test.manifest.template fullstack_test.manifest
    gramine-direct python3 fullstack_test.py
"""

import sys

print("=" * 55)
print("IntelliClave — Full Stack in Gramine Test")
print("torch + flwr + opacus inside gramine-direct")
print("=" * 55)

# ── 1. PyTorch ────────────────────────────────────────
print("\n[1] Importing torch...", end=" ", flush=True)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
print(f"OK  (v{torch.__version__})")

# ── 2. Flower ─────────────────────────────────────────
print("[2] Importing flwr...", end=" ", flush=True)
import flwr as fl
print(f"OK  (v{fl.__version__})")

# ── 3. Opacus ─────────────────────────────────────────
print("[3] Importing opacus...", end=" ", flush=True)
from opacus import PrivacyEngine
print(f"OK  (v{opacus.__version__})")
import opacus

# ── 4. Model ──────────────────────────────────────────
print("\n[4] Building FLClassifier (synthetic dims)...", end=" ", flush=True)

class FLClassifier(nn.Module):
    def __init__(self, input_dim=32, num_classes=4):
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

# Use small synthetic dims — the enclave test only checks that the stack works,
# not that the model matches any specific dataset.
INPUT_DIM   = 32
NUM_CLASSES = 4
model = FLClassifier(input_dim=INPUT_DIM, num_classes=NUM_CLASSES)
print("OK")

# ── 5. Synthetic data ─────────────────────────────────
print(f"[5] Creating synthetic data (256 × {INPUT_DIM})...", end=" ", flush=True)
X = torch.randn(256, INPUT_DIM)
y = torch.randint(0, NUM_CLASSES, (256,))
dataset    = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
print("OK")

# ── 6. DP training step ───────────────────────────────
print("[6] Attaching Opacus PrivacyEngine (ε=10, δ=1e-3)...", end=" ", flush=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    epochs=1,
    target_epsilon=10.0,
    target_delta=1e-3,
    max_grad_norm=1.0,
)
print("OK")

print("[7] Running one DP training step...", end=" ", flush=True)
model.train()
X_b, y_b = next(iter(dataloader))
optimizer.zero_grad()
loss = criterion(model(X_b), y_b)
loss.backward()
optimizer.step()
epsilon = privacy_engine.get_epsilon(delta=1e-3)
print(f"OK  (loss={loss.item():.4f}, ε={epsilon:.4f})")

# ── Summary ───────────────────────────────────────────
print()
print("=" * 55)
print("FULL STACK IN GRAMINE OK ✓")
print(f"  torch  {torch.__version__}")
print(f"  flwr   {fl.__version__}")
print(f"  opacus {opacus.__version__}")
print(f"  DP training step: loss={loss.item():.4f}, ε={epsilon:.4f}")
print("=" * 55)
