# M2 — FL + DP Integration

## 1. Overview

M1 built a complete Federated Learning pipeline using Flower and FedAvg. M2's job was to integrate Differential Privacy into that pipeline without overwriting M1's work.

The integration follows a clean separation of concerns:

- M1 owns: data loading, model architecture, FL server, FL client base, FedAvg aggregation
- M2 owns: Opacus PrivacyEngine attachment, DP-SGD training, epsilon tracking, privacy budget monitoring, epsilon sweep experiments, all evaluation graphs

DP is applied **client-side only**, during local training, before any weights are sent to the server. The server performs standard FedAvg aggregation and is unaware of DP internals — it simply logs the epsilon values that clients report back.

---

## 2. How FL + DP Works Together

### Standard FL Round (M1 baseline)

```
Server sends global weights
    ↓
Each client loads weights → trains locally → sends updated weights back
    ↓
Server averages weights (FedAvg) → saves global model
    ↓
Repeat for N rounds
```

### FL + DP Round (M2 integration)

```
Server sends global weights
    ↓
Each client loads weights
    ↓
[M2] Opacus wraps model + optimizer + DataLoader
    ↓
Local training runs with DP-SGD:
    - Per-sample gradients computed individually
    - Each gradient clipped to max_grad_norm = 1.0
    - Gaussian noise added (calibrated to target epsilon)
    - Noisy gradient used to update model
    ↓
[M2] Epsilon consumed is measured and added to metrics
    ↓
Client sends updated weights + epsilon back to server
    ↓
Server performs FedAvg (unchanged) + logs epsilon to fl_privacy.json
    ↓
Repeat for N rounds — epsilon accumulates across rounds
```

The key guarantee: even if an attacker intercepts the model weights sent to the server, they cannot determine whether any individual person's sensor data was in the training set — with probability at least (1 − δ), the information leakage is bounded by ε.

---

## 3. Files — What Each One Does

### M2-owned files 

| File | Location | Purpose |
|------|----------|---------|
| `dp_trainer.py` | `privacy/` | Core DP wrapper class. Takes any PyTorch model and attaches Opacus PrivacyEngine. Used by epsilon sweep and standalone tests. |
| `data_loader.py` | `privacy/` | ⚠️ Retired from FL pipeline. M1's `data_utils.py` handles all data loading. Kept for reference only. |
| `dp_flower_client.py` | `privacy/` | ⚠️ Retired. M2's DP is now integrated directly into M1's `fl_client.py`. Kept for reference only. |
| `budget_monitor.py` | `privacy/` | Reads `fl_privacy.json` and produces `privacy_log.json`. Tracks epsilon consumed, budget remaining, and exhaustion status per client per round. |
| `epsilon_sweep.py` | `privacy/` | Runs two standalone experiments on real HAR data. Experiment 1: accuracy at each epsilon value. Experiment 2: epsilon accumulation across 5 FL rounds. Saves JSON results and 3 PNG plots. |
| `validate_model.py` | `privacy/` | Runs Opacus compatibility check on M1's model. Run this before any DP training if M1 changes the model architecture. |
| `run_budget_monitor.py` | project root | Script to run budget monitor and print formatted summary table. Run after FL completes. |

### M1-owned files modified by M2 (extensions only)

| File | Location | What M2 Added |
|------|----------|---------------|
| `fl_client.py` | `fl/` | `use_dp`, `target_epsilon`, `max_grad_norm`, `num_fl_rounds` parameters. `_attach_privacy_engine()` method. Epsilon logged in `fit()` metrics. Three CLI flags: `--dp`, `--epsilon`, `--rounds`. |
| `train_local.py` | `fl/` | `use_dp`, `target_epsilon`, `max_grad_norm` parameters in `train_local()`. Epsilon added to epoch history. Three CLI flags: `--dp`, `--epsilon`, `--max-grad-norm`. |
| `fl_server.py` | `fl/` | `self.privacy_log` list. Epsilon extraction from client fit metrics in `aggregate_fit()`. Writes `results/fl_rounds/fl_privacy.json` after each round. |


## 4. Key Design Decisions

### Why DP is attached in `__init__`, not in `fit()`

Opacus requires knowing the **total number of training steps upfront** to calibrate the noise multiplier correctly. If DP were attached inside `fit()` (once per round), Opacus would calibrate noise for `local_epochs` only, causing epsilon to overshoot the target across multiple rounds.

By attaching once in `__init__` with `epochs = local_epochs × num_fl_rounds`, Opacus computes the correct noise multiplier for the full training duration. The epsilon budget is then correctly distributed across all rounds.

**This was the root cause of the ε=21 overshoot in the first run.** After the fix, epsilon stays at or below the target across all 5 rounds.

### Why M1's `train_one_epoch()` needs no changes

Opacus replaces the optimizer and DataLoader transparently during `make_private_with_epsilon()`. The same training loop that M1 wrote continues to work — it calls `optimizer.step()` as normal, but the optimizer is now a `DPOptimizer` internally performing gradient clipping and noise addition before each update.

### Why each client has a different delta

Delta (δ) is set to `1 / n_train` where `n_train` is the number of training samples for that specific client. Each client has a different dataset size after the 80/20 train/test split:

| Client | Total rows | Train rows | δ |
|--------|-----------|------------|---|
| Client 1 (FitLife) | 3,105 | 2,484 | 4.03e-4 |
| Client 2 (MediTrack) | 3,426 | 2,741 | 3.65e-4 |
| Client 3 (CareWatch) | 3,768 | 3,014 | 3.32e-4 |

### Why the epsilon sweep uses only client1

The sweep isolates the DP mechanism from the FL aggregation effect. Running all 3 clients would mix both effects and make it impossible to cleanly attribute accuracy changes to epsilon. Client 1 is the active-profile dataset (FitLife) — the hardest to classify — giving a conservative lower bound on accuracy.

---

## 5. Output Files

After a complete run, the following files are produced:

```
results/
├── fl_rounds/
│   ├── fl_metrics.json          — FL accuracy and loss per round (M1)
│   ├── fl_privacy.json          — Epsilon per client per round (M2)
│   ├── global_model_latest.pth  — Final DP-trained global model
│   ├── global_model_round_N.pth — Checkpoint after each round
│   ├── global_model_eval.json   — Per-client evaluation of final model
│   └── round_N.npz              — Raw weight snapshots
├── epsilon_sweep.json           — Accuracy at each epsilon value
├── epsilon_rounds.json          — Epsilon consumed per FL round
├── privacy_log.json             — Budget monitor output
└── plots/
    ├── accuracy_vs_epsilon.png  — Privacy-utility tradeoff curve
    ├── epsilon_over_rounds.png  — Budget consumption across rounds
    └── dp_summary.png           — Combined summary chart
```

---

## 6. Running the Complete FL + DP Pipeline


### Step 1 — Verify M1's baseline (no DP)

Run standalone local training to confirm each client's data loads and trains correctly:

```bash
python fl/train_local.py --csv data/processed/client1.csv --epochs 10
python fl/train_local.py --csv data/processed/client2.csv --epochs 10
python fl/train_local.py --csv data/processed/client3.csv --epochs 10
```

Expected: accuracy above 90% by epoch 10 on all three clients.

---

### Step 2 — Verify DP standalone (single client, no FL)

Test DP training on one client before involving the FL system:

```bash
python fl/train_local.py --csv data/processed/client1.csv --epochs 5 --dp --epsilon 10.0
```

Expected output includes:
```
[M2-DP] Attaching PrivacyEngine: ε=10.0, δ=4.03e-04, total_epochs=5
[M2-DP] PrivacyEngine attached successfully
  Epoch   1 loss=x.xxxx accuracy=x.xxxx macro_f1=x.xxxx ε=x.xxxx
  Epoch   5 loss=x.xxxx accuracy=x.xxxx macro_f1=x.xxxx ε=9.xxxx
[M2-DP] Final privacy budget consumed: ε=9.xxxx
```

Accuracy will be lower than the no-DP baseline — this is expected and correct.

---

### Step 3 — Run M1's FL baseline (no DP, for comparison)

Open 4 terminals. Activate venv in each. Run from project root.

**Terminal 1:**
```bash
python fl/run_server.py --rounds 5 --min-clients 3
```

Wait for:
```
Flower ECE server running (5 rounds)...
```

**Terminal 2:**
```bash
python fl/run_client.py --id 1
```

**Terminal 3:**
```bash
python fl/run_client.py --id 2
```

**Terminal 4:**
```bash
python fl/run_client.py --id 3
```

Expected final result (M1 baseline): accuracy ≈ 96–97%, macro-F1 ≈ 97%.

---

### Step 4 — Run FL + DP pipeline

Open 4 terminals. Activate venv in each.

**Terminal 1 — Server (command unchanged from M1):**
```bash
python fl/run_server.py --rounds 5 --min-clients 3
```

**Terminal 2:**
```bash
python fl/fl_client.py --csv data/processed/client1.csv --id 1 --dp --epsilon 10.0 --rounds 5
```

**Terminal 3:**
```bash
python fl/fl_client.py --csv data/processed/client2.csv --id 2 --dp --epsilon 10.0 --rounds 5
```

**Terminal 4:**
```bash
python fl/fl_client.py --csv data/processed/client3.csv --id 3 --dp --epsilon 10.0 --rounds 5
```

> **Important:** `--rounds 5` must match `--rounds 5` on the server. This tells Opacus the total training duration so it calibrates noise correctly. If you change the number of rounds on the server, update the client command to match.

**Expected client output per round:**
```
[Client 1][M2-DP] PrivacyEngine attached: ε=10.0, δ=4.03e-04, total_epochs=15 (3 local × 5 rounds)
[Client 1][M2-DP] loss=x.xxxx acc=x.xxxx ε=x.xxxx
```

**Expected server output per round:**
```
[Server][M2-DP] Round 1 avg_ε=x.xxxx — saved to fl_privacy.json
[Server] Round 1 {'round': 1, 'loss': x.xxxxx, 'accuracy': x.xxxxx, 'macro_f1': x.xxxxx}
```

Epsilon should stay below 10.0 across all 5 rounds.

---

### Step 5 — Evaluate the DP global model

```bash
python fl/evaluate_global_model.py
```

Expected output:
```
client1.csv: loss=x.xxxxx accuracy=x.xxxxx macro_f1=x.xxxxx
client2.csv: loss=x.xxxxx accuracy=x.xxxxx macro_f1=x.xxxxx
client3.csv: loss=x.xxxxx accuracy=x.xxxxx macro_f1=x.xxxxx
Overall weighted: loss=x.xxxxx accuracy=x.xxxxx macro_f1=x.xxxxx
Saved -> results\fl_rounds\global_model_eval.json
```

Compare against M1 baseline (≈97%). The DP-trained model will be lower — this gap is the **privacy cost** and is your key evaluation result.

---

### Step 6 — Run budget monitor

```bash
python run_budget_monitor.py
```

Expected output:
```
Round    Client       Epsilon    Remaining  Exhausted
1        1            x.xxxx     x.xxxx     False
1        2            x.xxxx     x.xxxx     False
1        3            x.xxxx     x.xxxx     False
...
5        3            9.xxxx     0.xxxx     False
```

All rows should show `Exhausted=False` when the `--rounds` flag is set correctly.

---

### Step 7 — Run epsilon sweep and generate plots

```bash
python privacy/epsilon_sweep.py
```

This runs both experiments on real client1 data and saves all outputs. Takes approximately 5–8 minutes.

Expected terminal summary:
```
Target ε |   Actual ε |   Test Acc |  Train Acc | δ
--------------------------------------------------------------
     1.0 |     0.9932 |     50.89% |     52.71% | 4.03e-04
     2.0 |     1.9949 |     62.00% |     64.60% | 4.03e-04
     5.0 |     4.9941 |     61.51% |     63.65% | 4.03e-04
    10.0 |     9.9917 |     69.89% |     74.18% | 4.03e-04
    20.0 |    19.9929 |     77.46% |     80.55% | 4.03e-04

Round |   ε consumed |  Remaining |   Test Acc | Exhausted
--------------------------------------------------------------
    1 |       4.7837 |     5.2163 |     60.06% | False
    2 |       6.4060 |     3.5940 |     72.95% | False
    3 |       7.7362 |     2.2638 |     80.03% | False
    4 |       8.9152 |     1.0848 |     82.77% | False
    5 |       9.9974 |     0.0026 |     87.28% | False
```

Open the saved plots:
```bash
start results\plots\accuracy_vs_epsilon.png
start results\plots\epsilon_over_rounds.png
start results\plots\dp_summary.png
```

---


Manual checks:

| Check | What to look for | File |
|-------|-----------------|------|
| Epsilon stays ≤ 10.0 | All round entries show avg_epsilon < 10 | `fl_privacy.json` |
| Budget never exhausted | All `budget_exhausted` entries are False | `privacy_log.json` |
| Accuracy improves each round | Loss decreasing, accuracy increasing | `fl_metrics.json` |
| DP costs accuracy | Final accuracy < M1 baseline (96.99%) | `global_model_eval.json` |
| Sweep is monotonic | Higher epsilon → higher accuracy | `epsilon_sweep.json` |

---

## 8. Confirmed Results

### M1 Baseline (no DP)

| Metric | Value |
|--------|-------|
| Final accuracy | 96.99% |
| Final macro-F1 | 97.04% |
| Rounds | 10 |

### FL + DP Run (ε = 10.0, 5 rounds)

| Client | Accuracy | Macro-F1 |
|--------|----------|----------|
| Client 1 — FitLife | 90.18% | 89.92% |
| Client 2 — MediTrack | 88.19% | 88.22% |
| Client 3 — CareWatch | 93.90% | 94.34% |
| **Overall weighted** | **90.88%** | **90.97%** |

**Privacy cost: −6.1% accuracy in exchange for ε=10 differential privacy guarantee.**

### Epsilon Sweep (client1, standalone)

| Target ε | Test Accuracy |
|----------|--------------|
| 1.0 | 50.89% |
| 2.0 | 62.00% |
| 5.0 | 61.51% |
| 10.0 | 69.89% |
| 20.0 | 77.46% |

### Budget Consumption (5 FL rounds, ε = 10.0)

| Round | ε consumed | Remaining | Test Accuracy |
|-------|-----------|-----------|---------------|
| 1 | 4.7837 | 5.2163 | 60.06% |
| 2 | 6.4060 | 3.5940 | 72.95% |
| 3 | 7.7362 | 2.2638 | 80.03% |
| 4 | 8.9152 | 1.0848 | 82.77% |
| 5 | 9.9974 | 0.0026 | 87.28% |

---

## 9. DP Parameters Reference

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `target_epsilon` | 10.0 | Privacy budget — lower is more private |
| `target_delta` | 1/n_train | Failure probability — set to 1 over training set size |
| `max_grad_norm` | 1.0 | Gradient clipping threshold |
| `local_epochs` | 3 | Training epochs per FL round |
| `num_fl_rounds` | 5 | Total FL rounds — must match server |
| `total_epochs` | 15 | `local_epochs × num_fl_rounds` — passed to Opacus |
| `batch_size` | 32 | Set by M1's `data_utils.py` |

---
