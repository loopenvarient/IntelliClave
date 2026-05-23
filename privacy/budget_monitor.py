# privacy/budget_monitor.py
"""
Privacy budget monitor for IntelliClave FL.

Tracks per-round epsilon from each client and exposes the cumulative
privacy loss across all rounds.

Issue 5 fix
-----------
The original implementation stored per-round epsilon but never computed
or surfaced cumulative epsilon. This is misleading: the privacy guarantee
is the cumulative value, not the per-round value.

Cumulative epsilon under Opacus
--------------------------------
Opacus uses the Renyi DP accountant (moments accountant) internally.
When you call privacy_engine.get_epsilon(delta) after N rounds of
training, it returns the CUMULATIVE epsilon for the full training run
up to that point. It is NOT a per-round increment.

What the server logs as "epsilon" in fit_res.metrics is therefore
already the cumulative epsilon from the client's perspective (since
the PrivacyEngine is attached once for the full training duration and
the accountant accumulates across all steps).

This monitor makes that explicit by:
  1. Storing the raw epsilon value from each client (= cumulative)
  2. Tracking the maximum cumulative epsilon seen per client
  3. Flagging when cumulative epsilon exceeds the budget
  4. Writing a clear cumulative_summary to the privacy log
"""
import json
import os
from datetime import datetime


class BudgetMonitor:
    def __init__(self, max_epsilon: float = 10.0):
        self.max_epsilon = max_epsilon
        self.log = []
        # {client_id: max_epsilon_seen} -- latest Opacus value IS cumulative
        self._cumulative: dict = {}

    def record(
        self,
        round_num: int,
        client_id: str,
        epsilon: float,
        delta: float,
        loss: float,
        accuracy: float = None,
    ) -> dict:
        """
        Record a privacy measurement for one client at one round.

        epsilon is the CUMULATIVE epsilon reported by Opacus
        (privacy_engine.get_epsilon(delta)), not a per-round increment.
        Opacus uses the Renyi DP accountant which gives tight cumulative
        bounds across all training steps seen so far.
        """
        cid = str(client_id)
        prev = self._cumulative.get(cid, 0.0)
        self._cumulative[cid] = max(prev, float(epsilon))

        entry = {
            "timestamp":          datetime.now().isoformat(),
            "round":              round_num,
            "client_id":          cid,
            "epsilon_cumulative": round(float(epsilon), 6),
            "delta":              delta,
            "loss":               loss,
            "accuracy":           accuracy,
            "budget_remaining":   round(self.max_epsilon - float(epsilon), 4),
            "budget_exhausted":   float(epsilon) >= self.max_epsilon,
            "accountant_note": (
                "Opacus Renyi DP accountant -- epsilon is cumulative across "
                "all rounds, not per-round. Compare to max_epsilon."
            ),
        }
        self.log.append(entry)
        return entry

    @property
    def worst_case_cumulative_epsilon(self) -> float:
        """
        Maximum cumulative epsilon across all clients.
        This is the binding privacy guarantee -- the weakest protection
        any participant has received.
        """
        return max(self._cumulative.values()) if self._cumulative else 0.0

    def cumulative_summary(self) -> dict:
        wc = self.worst_case_cumulative_epsilon
        return {
            "max_epsilon_budget":            self.max_epsilon,
            "worst_case_cumulative_epsilon": round(wc, 6),
            "budget_fraction_consumed":      round(wc / self.max_epsilon, 4)
                                             if self.max_epsilon > 0 else None,
            "budget_exhausted":              wc >= self.max_epsilon,
            "per_client_cumulative":         {
                cid: round(eps, 6)
                for cid, eps in self._cumulative.items()
            },
            "accountant":  "Renyi DP (Opacus moments accountant)",
            "composition": (
                "Tight -- Opacus tracks cumulative privacy loss internally. "
                "The reported epsilon is the cumulative bound, not a "
                "per-round value multiplied by rounds."
            ),
        }

    def save(self, path: str = "results/privacy_log.json") -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        output = {
            "log":                self.log,
            "cumulative_summary": self.cumulative_summary(),
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        wc = self.worst_case_cumulative_epsilon
        print(f"[BudgetMonitor] Privacy log saved to {path}")
        print(f"[BudgetMonitor] Worst-case cumulative epsilon: "
              f"{wc:.4f} / {self.max_epsilon} "
              f"({'EXHAUSTED' if wc >= self.max_epsilon else 'within budget'})")

    def is_exhausted(self, current_epsilon: float) -> bool:
        return float(current_epsilon) >= self.max_epsilon