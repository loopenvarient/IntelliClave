# privacy/dp_flower_client.py
import flwr as fl
import torch
from collections import OrderedDict
from data_loader import load_client_data
from dp_trainer import DPTrainer

# Per-client deltas from confirmed row counts
CLIENT_DELTAS = {
    "client1": 1/3105,   # 3.22e-4
    "client2": 1/3426,   # 2.92e-4
    "client3": 1/3768,   # 2.65e-4
}

class DPFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id, model, csv_path):
        """
        client_id : "client1", "client2", or "client3"
        model     : M1's MLP — imported from fl/model.py (wherever M1 puts it)
        csv_path  : path to the frozen client CSV
        """
        self.client_id = client_id
        delta = CLIENT_DELTAS[client_id]
        self.loader, _ = load_client_data(csv_path, delta)

        self.trainer = DPTrainer(
            model=model,
            target_epsilon=10.0,
            target_delta=delta,
            max_grad_norm=1.0,
            epochs=3
        )
        self.trainer.attach(self.loader)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in
                self.trainer.get_model().state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in
             zip(self.trainer.get_model().state_dict().keys(), parameters)}
        )
        self.trainer.get_model().load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self.trainer.train_one_round()
        print(f"[{self.client_id}] ε={metrics['epsilon']:.4f} | "
              f"loss={metrics['loss']:.4f}")
        return (
            self.get_parameters(config),
            len(self.loader.dataset),
            {"epsilon": metrics["epsilon"], "loss": metrics["loss"],
             "client_id": self.client_id}
        )

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        model = self.trainer.get_model()
        model.eval()
        correct = total = 0
        total_loss = 0.0
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            for X_batch, y_batch in self.loader:
                out = model(X_batch)
                total_loss += criterion(out, y_batch).item()
                correct += (out.argmax(1) == y_batch).sum().item()
                total += len(y_batch)
        return (
            total_loss / len(self.loader),
            total,
            {"accuracy": correct / total, "client_id": self.client_id}
        )