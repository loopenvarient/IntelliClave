# privacy/budget_monitor.py
import json
from datetime import datetime

class BudgetMonitor:
    def __init__(self, max_epsilon=10.0):
        self.max_epsilon = max_epsilon
        self.log = []

    def record(self, round_num, client_id, epsilon, delta, loss, accuracy=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "round": round_num,
            "client_id": client_id,
            "epsilon": epsilon,
            "delta": delta,
            "loss": loss,
            "accuracy": accuracy,
            "budget_remaining": round(self.max_epsilon - epsilon, 4),
            "budget_exhausted": epsilon >= self.max_epsilon
        }
        self.log.append(entry)
        return entry

    def save(self, path="results/privacy_log.json"):
        with open(path, "w") as f:
            json.dump(self.log, f, indent=2)
        print(f"✅ Privacy log saved to {path}")

    def is_exhausted(self, current_epsilon):
        return current_epsilon >= self.max_epsilon