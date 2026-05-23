# privacy/dp_trainer.py
import json
import os

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


class DPTrainer:
    def __init__(self, model, target_epsilon=10.0, target_delta=None,
                 max_grad_norm=2.0, epochs=3, num_classes=None, lr=1e-3,
                 dp_lr_boost: float = 2.0):
        """
        model          : any nn.Module — passed in, not defined here
        target_epsilon : privacy budget (default 10.0)
        target_delta   : set per-client based on their row count (1/n)
        max_grad_norm  : gradient clipping threshold for DP-SGD (default 2.0).
                         Sweep-optimised for eps=10 on UCI HAR — run
                         privacy/clipping_norm_sweep.py to find the optimal value
                         for your model and dataset at other epsilon values.
        epochs         : local epochs per FL round
        num_classes    : number of output classes — used to load class weights.
                         If None, class weights are not applied.
        lr             : base learning rate for the Adam optimizer (default 1e-3).
                         The effective LR is lr × dp_lr_boost.
        dp_lr_boost    : multiplier applied to lr when DP is active (default 2.0).
                         DP noise reduces the effective gradient signal; a higher
                         LR helps the model learn faster per step. Set to 1.0 to
                         disable the boost.
        """
        # Fix any BatchNorm → GroupNorm automatically
        self.model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(self.model, strict=False)
        if errors:
            raise ValueError(f"Model incompatible with Opacus: {errors}")

        self.target_epsilon = target_epsilon
        self.target_delta   = target_delta
        self.max_grad_norm  = max_grad_norm
        self.epochs         = epochs

        # Apply LR boost — DP noise reduces effective gradient signal so a
        # higher LR helps recover accuracy without changing the privacy guarantee.
        effective_lr = lr * dp_lr_boost
        if dp_lr_boost != 1.0:
            print(
                f"[DPTrainer] LR boost: {lr} × {dp_lr_boost} = {effective_lr:.5f} "
                f"(helps overcome DP noise)"
            )

        # Load class weights generically (integer-keyed JSON)
        weights = self._load_class_weights(num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=effective_lr)
        self.privacy_engine = PrivacyEngine()
        self._attached = False
        # Noise multiplier is set by Opacus during attach() — exposed via property
        self._noise_multiplier: float = None

    @staticmethod
    def _load_class_weights(num_classes):
        """
        Load data/class_weights.json if it exists and matches num_classes.
        Supports both integer-keyed {"0": w, "1": w, ...} and legacy
        name-keyed {"WALKING": w, ...} formats.
        Returns a FloatTensor or None.
        """
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        path  = os.path.join(_root, "data", "class_weights.json")
        if not os.path.exists(path) or num_classes is None:
            return None
        with open(path) as f:
            d = json.load(f)
        # Filter comment keys
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        if len(d) != num_classes:
            print(f"[DPTrainer] WARNING: class_weights.json has {len(d)} entries "
                  f"but num_classes={num_classes} — ignoring weights.")
            return None
        try:
            ordered = [d[k] for k in sorted(d.keys(), key=int)]
        except ValueError:
            ordered = [d[k] for k in sorted(d.keys())]
        return torch.FloatTensor(ordered)

    def attach(self, dataloader):
        self.model, self.optimizer, self.dataloader = \
            self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=dataloader,
                epochs=self.epochs,
                target_epsilon=self.target_epsilon,
                target_delta=self.target_delta,
                max_grad_norm=self.max_grad_norm,
            )
        # Capture the noise multiplier Opacus computed for this configuration.
        # This is the key diagnostic value: noise_multiplier = σ / max_grad_norm.
        # A very high value means DP is dominating the gradient signal.
        # A very low value means the privacy guarantee is weak.
        try:
            self._noise_multiplier = float(self.optimizer.noise_multiplier)
        except AttributeError:
            self._noise_multiplier = None

        # Pre-flight warning: low ε + high noise multiplier = accuracy collapse
        if (self._noise_multiplier is not None
                and self.target_epsilon < 5.0
                and self._noise_multiplier > 3.0):
            noise_std = self._noise_multiplier * self.max_grad_norm
            print(
                f"\n[DPTrainer] *** ACCURACY COLLAPSE WARNING ***\n"
                f"  target_epsilon={self.target_epsilon} is low and "
                f"noise_multiplier={self._noise_multiplier:.2f} is very high.\n"
                f"  noise_std = {self._noise_multiplier:.2f} x {self.max_grad_norm} "
                f"= {noise_std:.2f} -- gradient signal will be overwhelmed.\n"
                f"  Fix: run privacy/clipping_norm_sweep.py --epsilon {self.target_epsilon} "
                f"to find the optimal max_grad_norm before training.\n"
            )

        self._attached = True
        return self

    @property
    def noise_multiplier(self) -> float:
        """
        The noise multiplier σ computed by Opacus for this (ε, δ, C, epochs)
        configuration. Only available after attach() is called.

        noise_multiplier = σ / max_grad_norm
        The actual noise std added to each gradient is σ = noise_multiplier × max_grad_norm.

        Rule of thumb:
          σ < 0.5  → very little noise, weak privacy
          σ ≈ 1.0  → moderate noise, reasonable privacy
          σ > 2.0  → heavy noise, accuracy will degrade significantly
          σ > 5.0  → accuracy collapse likely (this is what ε=1 produces)
        """
        return self._noise_multiplier

    def calibration_report(self) -> dict:
        """
        Return a dict summarising the DP-SGD calibration for this trainer.
        Call after attach() to see the actual noise level Opacus computed.

        Use this to diagnose accuracy collapse:
          - If noise_multiplier is very high (> 3), the clipping norm is too
            low relative to the target epsilon, or epsilon is too tight.
          - If accuracy is near-random at ε=1, check noise_multiplier first.
        """
        if not self._attached:
            return {"error": "Call attach(dataloader) first"}
        return {
            "target_epsilon":   self.target_epsilon,
            "target_delta":     self.target_delta,
            "max_grad_norm":    self.max_grad_norm,
            "noise_multiplier": self._noise_multiplier,
            "noise_std":        (
                round(self._noise_multiplier * self.max_grad_norm, 6)
                if self._noise_multiplier is not None else None
            ),
            "epochs":           self.epochs,
            "interpretation": (
                "HEAVY NOISE — accuracy collapse likely"
                if self._noise_multiplier is not None and self._noise_multiplier > 3.0 else
                "MODERATE NOISE — expect accuracy degradation"
                if self._noise_multiplier is not None and self._noise_multiplier > 1.0 else
                "LIGHT NOISE — privacy guarantee may be weak"
                if self._noise_multiplier is not None else
                "unknown"
            ),
        }

    def train_one_round(self):
        assert self._attached, "Call attach(dataloader) before training"
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in self.dataloader:
            self.optimizer.zero_grad()
            output = self.model(X_batch)
            loss   = self.criterion(output, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)
        return {
            "loss":             round(total_loss / len(self.dataloader), 4),
            "epsilon":          round(epsilon, 4),
            "delta":            self.target_delta,
            "noise_multiplier": self._noise_multiplier,
        }

    def get_model(self):
        return self.model

    def get_epsilon(self):
        return self.privacy_engine.get_epsilon(delta=self.target_delta)
