# privacy/dp_trainer.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

class DPTrainer:
    def __init__(self, model, target_epsilon=10.0, target_delta=None,
                 max_grad_norm=1.0, epochs=3):
        """
        model          : M1's MLP — passed in, not defined here
        target_epsilon : privacy budget (we use 10.0 as default)
        target_delta   : set per-client based on their row count
        max_grad_norm  : gradient clipping threshold
        epochs         : local epochs per FL round
        """
        # Fix any BatchNorm → GroupNorm automatically
        self.model = ModuleValidator.fix(model)
        errors = ModuleValidator.validate(self.model, strict=False)
        if errors:
            raise ValueError(f"Model incompatible with Opacus: {errors}")

        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs

        # Load class weights from M1's file
        import json
        with open("data/class_weights.json") as f:
            weights_dict = json.load(f)
        # Order must match label order 0–5
        activity_order = ["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS",
                          "SITTING","STANDING","LAYING"]
        weights = torch.FloatTensor([weights_dict[a] for a in activity_order])
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.privacy_engine = PrivacyEngine()
        self._attached = False

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
            loss = self.criterion(output, y_batch)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        epsilon = self.privacy_engine.get_epsilon(delta=self.target_delta)
        return {
            "loss": round(total_loss / len(self.dataloader), 4),
            "epsilon": round(epsilon, 4),
            "delta": self.target_delta
        }

    def get_model(self):
        return self.model

    def get_epsilon(self):
        return self.privacy_engine.get_epsilon(delta=self.target_delta)