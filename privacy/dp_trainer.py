# privacy/dp_trainer.py
import json
import os

import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


class DPTrainer:
    def __init__(self, model, target_epsilon=10.0, target_delta=None,
                 max_grad_norm=1.0, epochs=3, num_classes=None, lr=1e-3):
        """
        model          : any nn.Module — passed in, not defined here
        target_epsilon : privacy budget (default 10.0)
        target_delta   : set per-client based on their row count (1/n)
        max_grad_norm  : gradient clipping threshold
        epochs         : local epochs per FL round
        num_classes    : number of output classes — used to load class weights.
                         If None, class weights are not applied.
        lr             : learning rate for the Adam optimizer (default 1e-3)
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

        # Load class weights generically (integer-keyed JSON)
        weights = self._load_class_weights(num_classes)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.privacy_engine = PrivacyEngine()
        self._attached = False

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
        self._attached = True
        return self

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
            "loss":    round(total_loss / len(self.dataloader), 4),
            "epsilon": round(epsilon, 4),
            "delta":   self.target_delta,
        }

    def get_model(self):
        return self.model

    def get_epsilon(self):
        return self.privacy_engine.get_epsilon(delta=self.target_delta)
