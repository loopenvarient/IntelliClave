"""
privacy/run_budget_monitor.py

Load a saved FL privacy log and display the budget consumption table.

Usage:
    python privacy/run_budget_monitor.py
    python privacy/run_budget_monitor.py --max-epsilon 5.0
    python privacy/run_budget_monitor.py --privacy-json results/fl_rounds/fl_privacy.json
    python privacy/run_budget_monitor.py --out results/my_privacy_log.json
"""
import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.append(_HERE)

from budget_monitor import BudgetMonitor  # noqa: E402

sys.path.insert(0, os.path.join(_ROOT, "config"))
from constants import DEFAULT_EPSILON     # noqa: E402

DEFAULT_PRIVACY_JSON = os.path.join(_ROOT, "results", "fl_rounds", "fl_privacy.json")
DEFAULT_OUT          = os.path.join(_ROOT, "results", "privacy_log.json")


def main(max_epsilon: float = DEFAULT_EPSILON,
         privacy_json: str = DEFAULT_PRIVACY_JSON,
         out: str = DEFAULT_OUT):

    if not os.path.exists(privacy_json):
        raise FileNotFoundError(
            f"Privacy log not found: {privacy_json}\n"
            "Run the FL training first to generate fl_privacy.json."
        )

    monitor = BudgetMonitor(max_epsilon=max_epsilon)

    with open(privacy_json) as f:
        privacy_data = json.load(f)

    for round_entry in privacy_data:
        for client in round_entry["clients"]:
            # fl_privacy.json uses "epsilon" key; BudgetMonitor.record() accepts
            # it and stores it as "epsilon_cumulative" (the Opacus value IS cumulative)
            monitor.record(
                round_num=round_entry["round"],
                client_id=client["client_id"],
                epsilon=client["epsilon"],
                delta=client.get("delta") or 0.0,
                loss=0.0,
            )

    os.makedirs(os.path.dirname(out), exist_ok=True)
    monitor.save(out)

    # Load the saved log (new format: {"log": [...], "cumulative_summary": {...}})
    with open(out) as f:
        saved = json.load(f)

    log = saved.get("log", saved) if isinstance(saved, dict) else saved
    summary = saved.get("cumulative_summary", {}) if isinstance(saved, dict) else {}

    print()
    print(f"{'Round':<8} {'Client':<12} {'Epsilon (cumul)':>16} {'Remaining':>12} {'Exhausted':>10}")
    print("-" * 62)
    for entry in log:
        # Support both old key ("epsilon") and new key ("epsilon_cumulative")
        eps = entry.get("epsilon_cumulative") or entry.get("epsilon") or 0.0
        print(
            f"{entry['round']:<8} "
            f"{entry['client_id']:<12} "
            f"{eps:>16.4f} "
            f"{entry['budget_remaining']:>12.4f} "
            f"{str(entry['budget_exhausted']):>10}"
        )

    if summary:
        print()
        wc = summary.get("worst_case_cumulative_epsilon", "n/a")
        budget = summary.get("max_epsilon_budget", max_epsilon)
        exhausted = summary.get("budget_exhausted", False)
        print(f"Worst-case cumulative epsilon: {wc} / {budget} "
              f"({'EXHAUSTED' if exhausted else 'within budget'})")
        print(f"Accountant: {summary.get('accountant', 'unknown')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Display FL privacy budget consumption from a saved log.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-epsilon",   type=float, default=DEFAULT_EPSILON,
                        help="Maximum allowed privacy budget (ε).")
    parser.add_argument("--privacy-json",  default=DEFAULT_PRIVACY_JSON,
                        help="Path to the fl_privacy.json produced by FL training.")
    parser.add_argument("--out",           default=DEFAULT_OUT,
                        help="Output path for the processed privacy log JSON.")
    args = parser.parse_args()
    main(
        max_epsilon=args.max_epsilon,
        privacy_json=args.privacy_json,
        out=args.out,
    )
