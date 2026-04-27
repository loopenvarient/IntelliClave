import sys
import json
sys.path.append('privacy')
from budget_monitor import BudgetMonitor

monitor = BudgetMonitor(max_epsilon=10.0)

with open('results/fl_rounds/fl_privacy.json') as f:
    privacy_data = json.load(f)

for round_entry in privacy_data:
    for client in round_entry['clients']:
        monitor.record(
            round_num=round_entry['round'],
            client_id=client['client_id'],
            epsilon=client['epsilon'],
            delta=client['delta'],
            loss=0.0
        )

monitor.save('results/privacy_log.json')
print('Budget monitor saved to results/privacy_log.json')

with open('results/privacy_log.json') as f:
    log = json.load(f)

print()
print(f"{'Round':<8} {'Client':<12} {'Epsilon':>10} {'Remaining':>12} {'Exhausted':>10}")
print("-" * 56)
for entry in log:
    print(
        f"{entry['round']:<8} "
        f"{entry['client_id']:<12} "
        f"{entry['epsilon']:>10.4f} "
        f"{entry['budget_remaining']:>12.4f} "
        f"{str(entry['budget_exhausted']):>10}"
    )