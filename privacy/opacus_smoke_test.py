"""
Opacus smoke test — 6-class HAR model with DP.
Architecture: 50 -> 128 -> 64 -> 6
No BatchNorm anywhere. GroupNorm only.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine


class HARModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(50, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, x):
        return self.net(x)


torch.manual_seed(42)
N = 256
X = torch.randn(N, 50)
y = torch.randint(0, 6, (N,))
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

model     = HARModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

privacy_engine = PrivacyEngine()
model, optimizer, loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=loader,
    target_epsilon=2.0,
    target_delta=1e-5,
    epochs=5,
    max_grad_norm=1.0,
)

print("=== Opacus Smoke Test ===")
print(f"{'Step':<6} {'Loss':<10} {'epsilon'}")
print("-" * 32)

prev_eps = 0.0
for step, (xb, yb) in enumerate(loader):
    if step >= 5:
        break
    optimizer.zero_grad()
    loss = criterion(model(xb), yb)
    loss.backward()
    optimizer.step()

    eps = privacy_engine.get_epsilon(delta=1e-5)
    print(f"  {step+1:<4} {loss.item():<10.4f} e={eps:.4f}  {'up' if eps > prev_eps else '='}")
    assert eps >= prev_eps, f"epsilon decreased at step {step+1}!"
    prev_eps = eps

print()
print("Opacus confirmed for 6 classes")
print(f"Final epsilon={prev_eps:.4f}, delta=1e-5")
