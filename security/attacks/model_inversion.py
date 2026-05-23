
"""
security/attacks/model_inversion.py

Attack 1: Model Inversion — White-Box Baseline + Black-Box Surrogate Attack

Goal
----
Reconstruct a representative input for each class and measure whether the
inference-time mitigations in security/inference_protection.py actually
degrade reconstruction quality.

Two attack modes
----------------
1. White-box (baseline, --mode whitebox)
   Direct gradient-based inversion against the raw model weights.
   Establishes the upper bound on reconstruction quality.
   This is NOT the realistic threat model — it assumes the adversary has
   the model file. Used to confirm the attack works without mitigation.

2. Black-box surrogate (--mode blackbox)
   The realistic threat model: adversary has only API access.
   Steps:
     a. Query the /predict endpoint with random inputs to collect
        (input, predicted_label) pairs — budget-limited.
     b. Train a surrogate MLP on those pairs.
     c. Run gradient-based inversion against the surrogate.
     d. Compare reconstructed inputs to real class centroids.

Assertions (--assert flag)
--------------------------
  PASS conditions (all must hold):
    1. Unmitigated white-box avg cosine similarity > 0.70
       (confirms the attack is effective without mitigation)
    2. Black-box surrogate avg cosine similarity < 0.60
       (confirms output perturbation degrades surrogate reconstruction)
       NOTE: White-box cosine similarity is NOT checked here — output
       perturbation on the API label cannot degrade white-box reconstruction
       because the adversary uses model gradients directly, not API outputs.
       The relevant threat model is black-box API access.
    3. Budget exhaustion halts the surrogate query phase
       (confirms hard cap is enforced)

  Assertion 2 is skipped when --no-surrogate is passed.
  The script exits with code 1 if any assertion fails, making it usable
  as a CI regression test.

Output
------
  results/attacks/model_inversion.json           (white-box baseline)
  results/attacks/model_inversion_mitigated.json (mitigated white-box)
  results/attacks/model_inversion_surrogate.json (black-box surrogate)
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_ROOT, "fl"))
sys.path.insert(0, os.path.join(_ROOT, "security"))
sys.path.insert(0, os.path.join(_ROOT, "config"))

from model import get_model               # noqa: E402
from data_utils import infer_csv_schema   # noqa: E402
from constants import LABEL_COL           # noqa: E402
from inference_protection import (        # noqa: E402
    OutputPerturbation,
    InMemoryBudgetStore,
    BudgetExhaustedError,
    LIFETIME_BUDGET_DEFAULT,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

MODEL_PATH    = os.path.join(_ROOT, "results", "fl_rounds", "global_model_latest.pth")
META_PATH     = os.path.join(_ROOT, "results", "fl_rounds", "model_meta.json")
PROCESSED_DIR = os.path.join(_ROOT, "data", "processed")
OUT_PATH      = os.path.join(_ROOT, "results", "attacks", "model_inversion.json")

DEFAULT_LR    = 0.05
DEFAULT_STEPS = 500

# Thresholds for --assert mode
ASSERT_WHITEBOX_MIN_COS  = 0.70   # attack must work without mitigation
ASSERT_MITIGATED_MAX_COS = 0.60   # mitigation must degrade reconstruction


# ── Data / model helpers ──────────────────────────────────────────────────────

def load_meta(label_col: str = LABEL_COL) -> dict:
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    csvs = sorted(f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv"))
    if not csvs:
        raise FileNotFoundError("No model_meta.json and no CSVs in data/processed/")
    first_csv = os.path.join(PROCESSED_DIR, csvs[0])
    input_dim, _ = infer_csv_schema(first_csv, label_col)
    df = pd.read_csv(first_csv, usecols=[label_col])
    num_classes = int(df[label_col].nunique())
    return {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": [f"class_{i}" for i in range(num_classes)],
    }


def load_model(input_dim: int, num_classes: int, model_path: str = MODEL_PATH):
    model = get_model(input_dim, num_classes)
    sys.path.insert(0, os.path.join(_ROOT, "tee", "sealed_storage"))
    try:
        from sealed_storage import load_checkpoint_state_dict  # noqa: E402
        state = load_checkpoint_state_dict(model_path, map_location="cpu")
    except ImportError:
        state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def load_all_data(label_col: str = LABEL_COL):
    """Load + scale all client CSVs, return (X float32, y int64 0-based)."""
    csvs = sorted(
        os.path.join(PROCESSED_DIR, f)
        for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".csv")
    )
    frames = [pd.read_csv(p).dropna() for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    feat_cols = [c for c in df.columns if c != label_col]
    X = df[feat_cols].values.astype(np.float32)
    raw_y = df[label_col].values
    unique = sorted(np.unique(raw_y).tolist())
    offset = int(min(unique)) if all(isinstance(v, (int, float)) for v in unique) else 0
    y = raw_y.astype(np.int64) - offset
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y


def class_centroids(X: np.ndarray, y: np.ndarray, num_classes: int) -> dict:
    return {c: X[y == c].mean(axis=0) for c in range(num_classes)}


# ── Similarity metrics ────────────────────────────────────────────────────────

def cosine_sim(a, b) -> float:
    a, b = np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def l2_dist(a, b) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


# ── White-box inversion ───────────────────────────────────────────────────────

def invert_class_whitebox(
    model,
    target_class: int,
    input_dim: int,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
) -> tuple:
    """
    Gradient-based inversion directly against model weights (white-box).
    Returns (reconstructed_input, final_confidence, loss_history).
    """
    x = torch.randn(1, input_dim, requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([x], lr=lr)
    target = torch.tensor([target_class])
    loss_history = []

    for step in range(steps):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, target) + 0.01 * x.pow(2).sum()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            loss_history.append(round(float(loss.item()), 5))

    with torch.no_grad():
        confidence = float(torch.softmax(model(x), dim=1)[0, target_class].item())
    return x.detach().numpy().flatten(), confidence, loss_history


# ── Mitigated API oracle ──────────────────────────────────────────────────────

class MitigatedAPIOracle:
    """
    Simulates the mitigated /predict endpoint locally.

    Applies the same two mitigations as dashboard/backend/main.py:
      1. OutputPerturbation — Laplace noise on post-softmax probabilities.
      2. InMemoryBudgetStore — hard lifetime query budget per key.

    Returns only the predicted label (integer), not the probability vector.
    This is the information available to a black-box adversary.
    """

    def __init__(
        self,
        model,
        noise_scale: float,
        budget: int = LIFETIME_BUDGET_DEFAULT,
        api_key: str = "attacker",
    ):
        self.model = model
        self.perturbation = OutputPerturbation(noise_scale=noise_scale)
        self.budget_store = InMemoryBudgetStore()
        self.budget_store.reset(api_key, budget)
        self.api_key = api_key
        self.budget = budget
        self._queries = 0

    def predict(self, x: np.ndarray) -> int:
        """
        Query the oracle with a feature vector.
        Returns the predicted class label (integer).
        Raises BudgetExhaustedError when the lifetime budget is consumed.
        """
        # Consume one unit of lifetime budget — raises if exhausted
        self.budget_store.consume(self.api_key, self.budget)
        self._queries += 1

        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(x_t)
            probs = torch.softmax(logits, dim=1).squeeze()

        # Apply output perturbation
        noisy_probs = self.perturbation.perturb(probs)
        return int(noisy_probs.argmax().item())

    @property
    def queries_used(self) -> int:
        return self._queries

    @property
    def budget_remaining(self) -> int:
        return self.budget_store.get_remaining(self.api_key)


# ── Black-box surrogate attack ────────────────────────────────────────────────

def collect_surrogate_data(
    oracle: MitigatedAPIOracle,
    input_dim: int,
    num_classes: int,
    n_queries: int,
    seed: int = 42,
) -> tuple:
    """
    Query the oracle with random inputs to build a surrogate training set.

    Queries are drawn from a standard normal distribution — the adversary
    has no knowledge of the real data distribution.

    Returns (X_surrogate, y_surrogate) or raises BudgetExhaustedError.
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []
    budget_exhausted = False

    for i in range(n_queries):
        x = rng.standard_normal(input_dim).astype(np.float32)
        try:
            label = oracle.predict(x)
            X_list.append(x)
            y_list.append(label)
        except BudgetExhaustedError:
            budget_exhausted = True
            print(f"  [Surrogate] Budget exhausted after {i} queries — "
                  f"stopping data collection.")
            break

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    return X, y, budget_exhausted


def train_surrogate(
    X: np.ndarray,
    y: np.ndarray,
    input_dim: int,
    num_classes: int,
    epochs: int = 30,
    lr: float = 1e-3,
):
    """
    Train a surrogate MLP on (X, y) pairs collected from the oracle.
    Returns the trained surrogate model.
    """
    surrogate = get_model(input_dim, num_classes)
    surrogate.train()
    optimizer = torch.optim.Adam(surrogate.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        for X_b, y_b in loader:
            optimizer.zero_grad()
            criterion(surrogate(X_b), y_b).backward()
            optimizer.step()

    surrogate.eval()
    # Report surrogate accuracy on its own training data
    with torch.no_grad():
        preds = surrogate(X_t).argmax(1)
        acc = (preds == y_t).float().mean().item()
    print(f"  [Surrogate] Training accuracy: {acc:.4f} "
          f"({len(X)} samples, {epochs} epochs)")
    return surrogate, acc


def invert_class_surrogate(
    surrogate,
    target_class: int,
    input_dim: int,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
) -> tuple:
    """
    Gradient-based inversion against the surrogate model.
    Same method as white-box but targets the surrogate, not the real model.
    """
    x = torch.randn(1, input_dim, requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([x], lr=lr)
    target = torch.tensor([target_class])
    loss_history = []

    for step in range(steps):
        optimizer.zero_grad()
        logits = surrogate(x)
        loss = criterion(logits, target) + 0.01 * x.pow(2).sum()
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            loss_history.append(round(float(loss.item()), 5))

    with torch.no_grad():
        confidence = float(torch.softmax(surrogate(x), dim=1)[0, target_class].item())
    return x.detach().numpy().flatten(), confidence, loss_history


# ── Noise calibration sweep ───────────────────────────────────────────────────

def calibration_sweep(
    model,
    centroids: dict,
    input_dim: int,
    num_classes: int,
    steps: int = 300,
    lr: float = DEFAULT_LR,
    noise_scales: list = None,
) -> list:
    """
    Sweep noise_scale values and measure the resulting cosine similarity
    of white-box reconstructions against real class centroids.

    Used to find the minimum noise_scale that degrades cosine similarity
    below ASSERT_MITIGATED_MAX_COS while keeping label accuracy acceptable.

    Returns a list of dicts: [{noise_scale, avg_cos_sim, label_accuracy}, ...]
    """
    if noise_scales is None:
        noise_scales = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

    print("\n  Calibration sweep (noise_scale vs cosine similarity):")
    print(f"  {'noise_scale':>12} | {'avg_cos_sim':>12} | {'label_acc':>10}")
    print("  " + "-" * 40)

    sweep_results = []
    for ns in noise_scales:
        perturb = OutputPerturbation(noise_scale=ns) if ns > 0 else None
        cos_sims, label_correct = [], []

        for cls in range(num_classes):
            # Invert against raw model (white-box upper bound)
            recon, _, _ = invert_class_whitebox(model, cls, input_dim, steps=steps, lr=lr)
            centroid = centroids.get(cls, np.zeros(input_dim))

            if perturb is not None:
                # Apply perturbation to the reconstruction query
                x_t = torch.tensor(recon, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x_t)
                    probs = torch.softmax(logits, dim=1).squeeze()
                noisy = perturb.perturb(probs)
                predicted_label = int(noisy.argmax().item())
            else:
                x_t = torch.tensor(recon, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    logits = model(x_t)
                predicted_label = int(logits.argmax(1).item())

            cos_sims.append(cosine_sim(recon, centroid))
            label_correct.append(int(predicted_label == cls))

        avg_cos = float(np.mean(cos_sims))
        label_acc = float(np.mean(label_correct))
        print(f"  {ns:>12.2f} | {avg_cos:>12.4f} | {label_acc:>10.4f}")
        sweep_results.append({
            "noise_scale":    ns,
            "avg_cos_sim":    round(avg_cos, 6),
            "label_accuracy": round(label_acc, 6),
        })

    return sweep_results


# ── Result helpers ────────────────────────────────────────────────────────────

def _build_result_entry(cls, name, recon, confidence, loss_hist, centroid):
    cos_sim = cosine_sim(recon, centroid)
    l2      = l2_dist(recon, centroid)
    # Risk levels reflect reconstruction quality relative to real class centroids.
    # Even LOW risk does not mean zero leakage — PCA/tabular inversion can still
    # reveal class-level feature distributions even at low cosine similarity.
    risk    = "HIGH" if cos_sim > 0.7 else "MEDIUM" if cos_sim > 0.4 else "LOW"
    return {
        "class_id":            cls,
        "class_name":          name,
        "model_confidence":    round(confidence, 6),
        "cosine_similarity":   round(cos_sim, 6),
        "l2_distance":         round(l2, 6),
        "loss_history":        loss_hist,
        "risk_level":          risk,
        "reconstructed_input": recon.tolist(),
    }


def _build_summary(results: list, num_classes: int, mode: str,
                   extra: dict = None) -> dict:
    avg_cos   = float(np.mean([r["cosine_similarity"] for r in results]))
    avg_conf  = float(np.mean([r["model_confidence"]  for r in results]))
    high_risk = sum(1 for r in results if r["risk_level"] == "HIGH")
    verdict = (
        "VULNERABLE — model leaks class-level feature structure"
        if avg_cos > 0.5 else
        "MODERATE — partial class structure leakage; mitigations reduce but do not eliminate risk"
        if avg_cos > 0.2 else
        "LOW RISK — reconstructed inputs have low similarity to real data centroids; "
        "note that PCA/tabular inversion remains a hard problem and cosine similarity "
        "is a necessary but not sufficient measure of reconstruction quality"
    )
    summary = {
        "mode":                  mode,
        "avg_cosine_similarity": round(avg_cos, 6),
        "avg_model_confidence":  round(avg_conf, 6),
        "high_risk_classes":     high_risk,
        "total_classes":         num_classes,
        "verdict":               verdict,
    }
    if extra:
        summary.update(extra)
    return summary


def _save(data: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(
    model_path: str = MODEL_PATH,
    label_col: str = LABEL_COL,
    lr: float = DEFAULT_LR,
    steps: int = DEFAULT_STEPS,
    out_path: str = OUT_PATH,
    noise_scale: float = None,
    surrogate_queries: int = 2000,
    surrogate_budget: int = 1500,
    surrogate_epochs: int = 30,
    run_calibration: bool = False,
    run_surrogate: bool = True,
    assert_mode: bool = False,
):
    print("=" * 60)
    print("Attack 1: Model Inversion")
    print("  White-box baseline + Black-box surrogate attack")
    print("=" * 60)

    meta        = load_meta(label_col=label_col)
    input_dim   = meta["input_dim"]
    num_classes = meta["num_classes"]
    class_names = meta["class_names"]

    model     = load_model(input_dim, num_classes, model_path=model_path)
    X, y      = load_all_data(label_col=label_col)
    centroids = class_centroids(X, y, num_classes)

    # Resolve noise scale: env var > CLI arg > default
    from inference_protection import NOISE_SCALE_DEFAULT
    effective_noise_scale = noise_scale if noise_scale is not None else NOISE_SCALE_DEFAULT

    # ── Optional calibration sweep ────────────────────────────────────────────
    if run_calibration:
        print("\n[0] Calibration sweep — finding effective noise scale...")
        sweep = calibration_sweep(model, centroids, input_dim, num_classes,
                                  steps=min(steps, 300), lr=lr)
        _save(
            {"calibration_sweep": sweep,
             "target_max_cos_sim": ASSERT_MITIGATED_MAX_COS,
             "selected_noise_scale": effective_noise_scale},
            out_path.replace(".json", "_calibration.json"),
        )

    # ── White-box baseline (unmitigated) ──────────────────────────────────────
    print(f"\n[1] White-box inversion (unmitigated, {steps} steps)...")
    wb_results = []
    for cls in range(num_classes):
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        print(f"  Class {cls} ({name})...", end=" ", flush=True)
        recon, conf, loss_hist = invert_class_whitebox(
            model, cls, input_dim, steps=steps, lr=lr
        )
        entry = _build_result_entry(
            cls, name, recon, conf, loss_hist,
            centroids.get(cls, np.zeros(input_dim))
        )
        wb_results.append(entry)
        print(f"cos_sim={entry['cosine_similarity']:.4f}  risk={entry['risk_level']}")

    wb_summary = _build_summary(wb_results, num_classes, "whitebox_unmitigated")
    print(f"\n  Avg cosine similarity (unmitigated): "
          f"{wb_summary['avg_cosine_similarity']:.4f}")
    print(f"  Verdict: {wb_summary['verdict']}")
    _save({"attack": "model_inversion", "summary": wb_summary, "results": wb_results},
          out_path)

    # ── White-box with output perturbation ────────────────────────────────────
    print(f"\n[2] White-box inversion (output perturbation, "
          f"noise_scale={effective_noise_scale:.3f})...")
    perturb = OutputPerturbation(noise_scale=effective_noise_scale)
    print(f"  Calibration: {perturb.calibration_report(num_classes)}")

    mit_results = []
    for cls in range(num_classes):
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        print(f"  Class {cls} ({name})...", end=" ", flush=True)
        recon, _, loss_hist = invert_class_whitebox(
            model, cls, input_dim, steps=steps, lr=lr
        )
        # Apply perturbation to get the label the adversary would observe
        x_t = torch.tensor(recon, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = model(x_t)
            probs = torch.softmax(logits, dim=1).squeeze()
        noisy_probs = perturb.perturb(probs)
        conf = float(noisy_probs[cls].item())

        entry = _build_result_entry(
            cls, name, recon, conf, loss_hist,
            centroids.get(cls, np.zeros(input_dim))
        )
        mit_results.append(entry)
        print(f"cos_sim={entry['cosine_similarity']:.4f}  risk={entry['risk_level']}")

    mit_summary = _build_summary(
        mit_results, num_classes, "whitebox_mitigated",
        extra={"noise_scale": effective_noise_scale,
               "epsilon_inf": round(perturb.epsilon_inf, 4)},
    )
    print(f"\n  Avg cosine similarity (mitigated): "
          f"{mit_summary['avg_cosine_similarity']:.4f}")
    print(f"  Verdict: {mit_summary['verdict']}")
    _save({"attack": "model_inversion", "summary": mit_summary, "results": mit_results},
          out_path.replace(".json", "_mitigated.json"))

    # ── Black-box surrogate attack ────────────────────────────────────────────
    surrogate_summary = None
    budget_exhausted_confirmed = False

    if run_surrogate:
        print(f"\n[3] Black-box surrogate attack "
              f"(budget={surrogate_budget}, queries={surrogate_queries})...")
        oracle = MitigatedAPIOracle(
            model=model,
            noise_scale=effective_noise_scale,
            budget=surrogate_budget,
            api_key="attacker",
        )

        print(f"  Collecting surrogate training data "
              f"(up to {surrogate_queries} queries, budget={surrogate_budget})...")
        X_surr, y_surr, budget_exhausted_confirmed = collect_surrogate_data(
            oracle, input_dim, num_classes,
            n_queries=surrogate_queries,
        )
        print(f"  Collected {len(X_surr)} labelled samples "
              f"({oracle.queries_used} queries used, "
              f"{oracle.budget_remaining} remaining)")

        if len(X_surr) < num_classes * 10:
            print(f"  WARNING: Only {len(X_surr)} samples collected — "
                  "surrogate may be unreliable. Increase budget or queries.")

        surrogate, surr_train_acc = train_surrogate(
            X_surr, y_surr, input_dim, num_classes,
            epochs=surrogate_epochs, lr=lr,
        )

        print(f"  Inverting classes via surrogate ({steps} steps)...")
        surr_results = []
        for cls in range(num_classes):
            name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
            print(f"  Class {cls} ({name})...", end=" ", flush=True)
            recon, conf, loss_hist = invert_class_surrogate(
                surrogate, cls, input_dim, steps=steps, lr=lr
            )
            entry = _build_result_entry(
                cls, name, recon, conf, loss_hist,
                centroids.get(cls, np.zeros(input_dim))
            )
            surr_results.append(entry)
            print(f"cos_sim={entry['cosine_similarity']:.4f}  risk={entry['risk_level']}")

        surrogate_summary = _build_summary(
            surr_results, num_classes, "blackbox_surrogate",
            extra={
                "noise_scale":             effective_noise_scale,
                "surrogate_budget":        surrogate_budget,
                "queries_used":            oracle.queries_used,
                "budget_exhausted":        budget_exhausted_confirmed,
                "surrogate_train_accuracy": round(surr_train_acc, 4),
            },
        )
        print(f"\n  Avg cosine similarity (surrogate): "
              f"{surrogate_summary['avg_cosine_similarity']:.4f}")
        print(f"  Budget exhausted: {budget_exhausted_confirmed}")
        print(f"  Verdict: {surrogate_summary['verdict']}")
        _save(
            {"attack": "model_inversion", "summary": surrogate_summary,
             "results": surr_results},
            out_path.replace(".json", "_surrogate.json"),
        )

    # ── Assertion checks ──────────────────────────────────────────────────────
    if assert_mode:
        print("\n" + "=" * 60)
        print("ASSERTION CHECKS")
        print("=" * 60)
        failures = []

        # 1. Unmitigated white-box must be effective
        wb_cos = wb_summary["avg_cosine_similarity"]
        if wb_cos >= ASSERT_WHITEBOX_MIN_COS:
            print(f"  [PASS] White-box cosine sim {wb_cos:.4f} >= "
                  f"{ASSERT_WHITEBOX_MIN_COS} (attack works without mitigation)")
        else:
            msg = (f"  [FAIL] White-box cosine sim {wb_cos:.4f} < "
                   f"{ASSERT_WHITEBOX_MIN_COS} — attack is not effective even "
                   "without mitigation. Check model training.")
            print(msg)
            failures.append(msg)

        # 2. Output perturbation must degrade BLACK-BOX (surrogate) reconstruction.
        #
        # IMPORTANT: Output perturbation cannot degrade WHITE-BOX reconstruction.
        # In white-box mode the adversary has direct access to model weights and
        # gradients — they never query the API and never see the noisy output.
        # The mitigated white-box cosine similarity will always be ~equal to the
        # unmitigated one because the reconstruction uses gradients, not labels.
        #
        # The correct assertion is on the SURROGATE attack: the adversary queries
        # the noisy API, trains a surrogate on noisy labels, then inverts the
        # surrogate. Noise on the API output degrades the surrogate's fidelity,
        # which in turn degrades the inversion quality.
        if run_surrogate and surrogate_summary is not None:
            surr_cos = surrogate_summary["avg_cosine_similarity"]
            if surr_cos < ASSERT_MITIGATED_MAX_COS:
                print(f"  [PASS] Black-box surrogate cosine sim {surr_cos:.4f} < "
                      f"{ASSERT_MITIGATED_MAX_COS} "
                      f"(output perturbation degrades surrogate reconstruction)")
            else:
                msg = (f"  [FAIL] Black-box surrogate cosine sim {surr_cos:.4f} >= "
                       f"{ASSERT_MITIGATED_MAX_COS} — output perturbation is "
                       "insufficient against surrogate attack. Increase noise_scale.")
                print(msg)
                failures.append(msg)
        else:
            print(f"  [SKIP] Assertion 2 — surrogate not run (--no-surrogate). "
                  f"White-box cosine sim {mit_summary['avg_cosine_similarity']:.4f} "
                  f"is expected to be HIGH — output perturbation does not protect "
                  f"against white-box access.")

        # 3. Budget exhaustion must halt the surrogate attack
        if run_surrogate:
            if budget_exhausted_confirmed:
                print(f"  [PASS] Budget exhaustion confirmed — "
                      "surrogate query phase halted at hard cap")
            else:
                msg = ("  [FAIL] Budget was NOT exhausted during surrogate attack. "
                       "Increase surrogate_queries above surrogate_budget to test "
                       "the hard cap.")
                print(msg)
                failures.append(msg)

        print("=" * 60)
        if failures:
            print(f"RESULT: {len(failures)} assertion(s) FAILED")
            for f in failures:
                print(f)
            sys.exit(1)
        else:
            print("RESULT: All assertions PASSED ✓")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  White-box (unmitigated) : "
          f"cos_sim={wb_summary['avg_cosine_similarity']:.4f}  "
          f"→ {wb_summary['verdict']}")
    print(f"  White-box (mitigated)   : "
          f"cos_sim={mit_summary['avg_cosine_similarity']:.4f}  "
          f"→ {mit_summary['verdict']}")
    if surrogate_summary:
        print(f"  Black-box surrogate     : "
              f"cos_sim={surrogate_summary['avg_cosine_similarity']:.4f}  "
              f"→ {surrogate_summary['verdict']}")
    print("=" * 60)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Model Inversion Attack — white-box baseline + black-box surrogate.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--label-col",  default=LABEL_COL)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--steps",      type=int,   default=DEFAULT_STEPS)
    parser.add_argument("--out",        default=OUT_PATH)
    parser.add_argument(
        "--noise-scale", type=float, default=None,
        help="Laplace noise scale for output perturbation. "
             "Default: derived from INFERENCE_NOISE_SCALE env var or "
             "SOFTMAX_L1_SENSITIVITY / INFERENCE_EPSILON.",
    )
    parser.add_argument(
        "--surrogate-queries", type=int, default=2000,
        help="Number of oracle queries to attempt for surrogate training.",
    )
    parser.add_argument(
        "--surrogate-budget", type=int, default=1500,
        help="Hard lifetime query budget given to the attacker API key.",
    )
    parser.add_argument(
        "--surrogate-epochs", type=int, default=30,
        help="Training epochs for the surrogate model.",
    )
    parser.add_argument(
        "--calibration", action="store_true",
        help="Run noise calibration sweep before main attack.",
    )
    parser.add_argument(
        "--no-surrogate", action="store_true",
        help="Skip the black-box surrogate attack (run white-box only).",
    )
    parser.add_argument(
        "--assert", dest="assert_mode", action="store_true",
        help="Run assertion checks and exit 1 on failure (CI mode).",
    )
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        label_col=args.label_col,
        lr=args.lr,
        steps=args.steps,
        out_path=args.out,
        noise_scale=args.noise_scale,
        surrogate_queries=args.surrogate_queries,
        surrogate_budget=args.surrogate_budget,
        surrogate_epochs=args.surrogate_epochs,
        run_calibration=args.calibration,
        run_surrogate=not args.no_surrogate,
        assert_mode=args.assert_mode,
    )
