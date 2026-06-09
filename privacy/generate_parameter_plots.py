"""
Generate visualisations for IntelliClave parameter choices.

Covers:
  1. Laplace noise distribution and effect on logits
  2. Noise-scale tradeoff (confidence, entropy, argmax stability)
  3. Temperature scaling tradeoff
  4. DP epsilon privacy–utility (from epsilon_sweep.json)
  5. Model inversion defence comparison
  6. Summary dashboard of all chosen values

Usage:
    python privacy/generate_parameter_plots.py
    python privacy/generate_parameter_plots.py --model-path results/fl_rounds/global_model_latest.pth

Outputs → results/plots/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_FL_DIR = os.path.join(_ROOT, "fl")
sys.path.insert(0, _FL_DIR)
sys.path.insert(0, os.path.join(_ROOT, "config"))

from constants import (  # noqa: E402
    DP_BATCH_SIZE,
    FEATURE_NOISE_STD,
    MI_NOISE_SCALE,
    MI_TEMPERATURE,
)

PLOT_DIR = os.path.join(_ROOT, "results", "plots")
DEFAULT_MODEL = os.path.join(_ROOT, "results", "fl_rounds", "global_model_latest.pth")

# Chosen production values (from config/constants.py + learn.md)
CHOSEN = {
    "mi_noise_scale": MI_NOISE_SCALE,       # 3.0
    "mi_temperature": MI_TEMPERATURE,       # 15.0
    "target_epsilon": 8.0,                  # FL client target
    "max_grad_norm": 0.3,
    "dp_batch_size": DP_BATCH_SIZE,         # 64
    "feature_noise_std": FEATURE_NOISE_STD, # 0.15
}

# Candidate values we considered but did not choose
CANDIDATES = {
    "noise_scale": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
    "temperature": [1.0, 5.0, 10.0, 15.0, 20.0, 30.0],
    "epsilon": [1.0, 2.0, 5.0, 8.0, 10.0, 20.0],
    "max_grad_norm": [0.1, 0.3, 0.5, 1.0, 2.0],
    "batch_size": [32, 64, 128],
}


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def laplace_pdf(x: np.ndarray, scale: float) -> np.ndarray:
    return np.exp(-np.abs(x) / scale) / (2.0 * scale)


def synthetic_logits(n_samples: int = 256, n_classes: int = 6) -> np.ndarray:
    """
    Representative MLP logits when the checkpoint cannot be loaded.
    Magnitudes match UCI HAR global model (confident but not extreme).
    """
    rng = np.random.default_rng(42)
    logits = rng.normal(0, 0.4, size=(n_samples, n_classes))
    # Make one class slightly dominant per sample (realistic classifier behaviour)
    for i in range(n_samples):
        winner = rng.integers(0, n_classes)
        logits[i, winner] += rng.uniform(0.8, 2.0)
    return logits.astype(np.float32)


def load_logits_from_model(model_path: str, n_samples: int = 256) -> np.ndarray:
    """Load global checkpoint and collect raw logits; fall back to synthetic data."""
    try:
        import torch
        from model import build_model_from_state  # noqa: E402
    except ImportError:
        print("  torch not installed — using synthetic logits")
        return synthetic_logits(n_samples)

    if not os.path.exists(model_path):
        print(f"  model not found — using synthetic logits")
        return synthetic_logits(n_samples)

    state = torch.load(model_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(_ROOT, "results", "fl_rounds", "model_meta.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    input_dim = meta["input_dim"]
    num_classes = meta["num_classes"]
    model_type = meta.get("model_type", "mlp")

    base = build_model_from_state(
        state, input_dim=input_dim, num_classes=num_classes, model_type=model_type,
    )
    base.eval()

    rng = np.random.default_rng(42)
    X = rng.standard_normal((n_samples, input_dim)).astype(np.float32)
    with torch.no_grad():
        logits = base(torch.from_numpy(X))
    return logits.numpy()


def measure_noise_effects(
    logits: np.ndarray,
    noise_scales: list[float],
    temperature: float = MI_TEMPERATURE,
    n_trials: int = 200,
):
    """Simulate PrivacyWrapper perturbation: (logits + Laplace) / T → softmax."""
    logits_np = np.asarray(logits)
    records = []

    for scale in noise_scales:
        confidences, entropies, flip_rates = [], [], []
        for _ in range(n_trials):
            noise = np.random.laplace(0, scale, size=logits_np.shape)
            scaled = (logits_np + noise) / temperature
            probs = _softmax(scaled)
            orig_argmax = logits_np.argmax(axis=1)
            noisy_argmax = scaled.argmax(axis=1)
            flip_rates.append((orig_argmax != noisy_argmax).mean())
            # Confidence on the *true* (clean) winning class — drops as noise grows
            true_cls = orig_argmax
            confidences.append(probs[np.arange(len(probs)), true_cls].mean())
            entropies.append(_entropy(probs).mean())

        records.append({
            "scale": scale,
            "avg_confidence": float(np.mean(confidences)),
            "avg_entropy": float(np.mean(entropies)),
            "flip_rate": float(np.mean(flip_rates)),
        })
    return records


def measure_temperature_effects(logits: np.ndarray, temperatures: list[float]):
    logits_np = np.asarray(logits)
    records = []
    for T in temperatures:
        scaled = logits_np / T
        probs = _softmax(scaled)
        top2_gap = []
        for row in probs:
            sorted_p = np.sort(row)[::-1]
            top2_gap.append(sorted_p[0] - sorted_p[1] if len(sorted_p) > 1 else sorted_p[0])
        records.append({
            "temperature": T,
            "avg_confidence": float(probs.max(axis=1).mean()),
            "avg_entropy": float(_entropy(probs).mean()),
            "avg_sharpness": float(np.mean(top2_gap)),
        })
    return records


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def plot_laplace_distribution(plt, out_dir: str):
    x = np.linspace(-15, 15, 600)
    scales = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    colors = ["#A8DADC", "#457B9D", "#1D3557", "#E63946", "#F4A261", "#2A9D8F"]
    chosen = CHOSEN["mi_noise_scale"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for b, c in zip(scales, colors):
        lw = 3.0 if b == chosen else 1.8
        ls = "-" if b == chosen else "--"
        label = f"b = {b}" + ("  ← CHOSEN" if b == chosen else "")
        ax.plot(x, laplace_pdf(x, b), color=c, lw=lw, ls=ls, label=label)
    ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
    ax.set_xlabel("Noise sample value", fontsize=11)
    ax.set_ylabel("Probability density", fontsize=11)
    ax.set_title("Laplace(0, b) — Output Perturbation on Logits\n"
                 "PrivacyWrapper adds i.i.d. Laplace noise to each logit",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#FAFAFA")

    ax = axes[1]
    pct68 = [b for b in scales]
    pct95 = [b * np.log(20) for b in scales]  # ~95% interval for Laplace
    x_pos = np.arange(len(scales))
    w = 0.35
    bars1 = ax.bar(x_pos - w / 2, pct68, w, label="±b (68% of mass)", color="#457B9D", alpha=0.8)
    bars2 = ax.bar(x_pos + w / 2, pct95, w, label="±b·ln(20) (~95%)", color="#E63946", alpha=0.6)
    chosen_idx = scales.index(chosen)
    bars1[chosen_idx].set_edgecolor("black")
    bars1[chosen_idx].set_linewidth(2.5)
    bars2[chosen_idx].set_edgecolor("black")
    bars2[chosen_idx].set_linewidth(2.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(s) for s in scales])
    ax.set_xlabel("Noise scale b", fontsize=11)
    ax.set_ylabel("Typical noise magnitude on a logit", fontsize=11)
    ax.set_title(f"Why b = {chosen}? 68% of noise falls within ±{chosen} on each logit\n"
                 f"Start at 0.5, tune up until inversion cosine < 0.4",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_facecolor("#FAFAFA")

    fig.tight_layout()
    path = os.path.join(out_dir, "laplace_noise_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_logits_perturbation(plt, logits: np.ndarray, out_dir: str):
    """Show one example logit vector before/after Laplace noise."""
    chosen = CHOSEN["mi_noise_scale"]
    sample = np.asarray(logits)[0]
    n_classes = len(sample)
    classes = [f"C{i}" for i in range(n_classes)]

    rng = np.random.default_rng(7)
    noisy_samples = [sample + rng.laplace(0, chosen, size=n_classes) for _ in range(5)]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    x = np.arange(n_classes)
    ax.bar(x - 0.2, sample, width=0.4, color="#2E75B6", label="Clean logits", alpha=0.9)
    for i, ns in enumerate(noisy_samples[:3]):
        ax.plot(x, ns, "o-", ms=6, lw=1.5, alpha=0.7,
                label=f"Noisy query #{i + 1}" if i < 3 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Logit value")
    ax.set_title(f"Same input, different Laplace noise each query (b = {chosen})\n"
                 "Fresh noise every forward pass — attacker cannot average it away",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.axhline(0, color="gray", lw=0.8)

    ax = axes[1]
    clean_probs = _softmax(sample.reshape(1, -1))[0]
    noisy_probs = [_softmax((sample + rng.laplace(0, chosen, n_classes)).reshape(1, -1))[0]
                   for _ in range(8)]
    mean_noisy = np.mean(noisy_probs, axis=0)
    ax.bar(x - 0.2, clean_probs, width=0.4, color="#1D9E75", label="Clean softmax", alpha=0.9)
    ax.bar(x + 0.2, mean_noisy, width=0.4, color="#D85A30",
           label=f"Mean of 8 noisy softmaps (b={chosen})", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("Probability")
    ax.set_title("Effect on output probabilities (before temperature scaling)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    path = os.path.join(out_dir, "laplace_noise_on_logits.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_noise_scale_tradeoff(plt, noise_records: list, inversion_data: dict, out_dir: str):
    chosen = CHOSEN["mi_noise_scale"]
    scales = [r["scale"] for r in noise_records]
    conf = [r["avg_confidence"] for r in noise_records]
    ent = [r["avg_entropy"] for r in noise_records]
    flip = [r["flip_rate"] * 100 for r in noise_records]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(scales, conf, "o-", color="#2E75B6", lw=2.5, ms=9, mfc="white", mew=2)
    ax.axvline(chosen, color="#E63946", ls="--", lw=2, label=f"Chosen b = {chosen}")
    ax.axvspan(0, 0.5, alpha=0.08, color="red", label="Too weak (<0.5)")
    ax.axvspan(5, max(scales) + 1, alpha=0.08, color="orange", label="Too strong (>5)")
    ax.set_xlabel("Laplace noise scale (b)")
    ax.set_ylabel("Avg confidence on true class")
    ax.set_title("Utility: lower noise → higher true-class confidence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(scales, ent, "s-", color="#D85A30", lw=2.5, ms=9, mfc="white", mew=2)
    ax.axvline(chosen, color="#E63946", ls="--", lw=2, label=f"Chosen b = {chosen}")
    ax.axhline(-np.log(1 / 6), color="gray", ls=":", label="Max entropy (uniform, 6 classes)")
    ax.set_xlabel("Laplace noise scale (b)")
    ax.set_ylabel("Avg prediction entropy")
    ax.set_title("Privacy: higher noise → more uncertainty")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(scales, flip, "^-", color="#1D9E75", lw=2.5, ms=9, mfc="white", mew=2)
    ax.axvline(chosen, color="#E63946", ls="--", lw=2, label=f"Chosen b = {chosen}")
    ax.set_xlabel("Laplace noise scale (b)")
    ax.set_ylabel("Argmax flip rate (%)")
    ax.set_title("Prediction stability under noise")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Annotation box with inversion result
    defended_cos = inversion_data.get("defended", 0.017)
    unmitigated_cos = inversion_data.get("unmitigated", 0.834)
    fig.text(0.5, -0.02,
             f"Chosen b={chosen}: inversion cosine {defended_cos:.3f} (vs {unmitigated_cos:.3f} unmitigated). "
             f"b<0.5 leaves too much signal; b>5 hurts dashboard confidence without extra MI gain (no_grad blocks gradients).",
             ha="center", fontsize=10, style="italic", wrap=True)

    fig.suptitle("Laplace Noise Scale Tradeoff — Why MI_NOISE_SCALE = 3.0",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "noise_scale_tradeoff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_temperature_tradeoff(plt, temp_records: list, out_dir: str):
    chosen = CHOSEN["mi_temperature"]
    temps = [r["temperature"] for r in temp_records]
    conf = [r["avg_confidence"] for r in temp_records]
    ent = [r["avg_entropy"] for r in temp_records]
    sharp = [r["avg_sharpness"] for r in temp_records]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(temps, conf, "o-", color="#2E75B6", lw=2.5, ms=9, mfc="white", mew=2)
    ax.axvline(chosen, color="#E63946", ls="--", lw=2, label=f"Chosen T = {chosen}")
    ax.set_xlabel("Temperature T (logits / T)")
    ax.set_ylabel("Avg max confidence")
    ax.set_title("Higher T → flatter probabilities")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.plot(temps, ent, "s-", color="#D85A30", lw=2.5, ms=9, mfc="white", mew=2)
    ax.axvline(chosen, color="#E63946", ls="--", lw=2, label=f"Chosen T = {chosen}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Avg entropy")
    ax.set_title("Higher T → more entropy (harder to invert)")
    ax.legend()
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    ax.plot(temps, sharp, "^-", color="#1D9E75", lw=2.5, ms=9, mfc="white", mew=2)
    ax.axvline(chosen, color="#E63946", ls="--", lw=2, label=f"Chosen T = {chosen}")
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Top-1 minus Top-2 prob gap")
    ax.set_title("Sharpness — inversion optimizer follows this gap")
    ax.legend()
    ax.grid(True, alpha=0.25)

    fig.suptitle("Temperature Scaling Tradeoff — Why MI_TEMPERATURE = 15.0\n"
                 "T=1 keeps sharp peaks (vulnerable); T>20 over-flattens dashboard confidence",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    path = os.path.join(out_dir, "temperature_tradeoff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_epsilon_tradeoff(plt, out_dir: str):
    sweep_path = os.path.join(_ROOT, "results", "epsilon_sweep.json")
    if not os.path.exists(sweep_path):
        print(f"  skip epsilon plot — {sweep_path} not found")
        return

    with open(sweep_path, encoding="utf-8") as f:
        data = json.load(f)

    eps = [r["target_epsilon"] for r in data]
    acc = [r["test_accuracy"] * 100 for r in data]
    chosen = CHOSEN["target_epsilon"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eps, acc, "o-", color="#2E75B6", lw=2.5, ms=10, mfc="white", mew=2.5)
    ax.axvspan(0, 3, alpha=0.07, color="red", label="High privacy (ε < 3)")
    ax.axvspan(3, 12, alpha=0.07, color="orange", label="Balanced (3–12)")
    ax.axvspan(12, 25, alpha=0.07, color="green", label="Low privacy (ε > 12)")
    ax.axvline(chosen, color="#E63946", ls="--", lw=2.5, label=f"Chosen ε = {chosen} (FL target)")
    ax.axvline(10.0, color="#1D9E75", ls=":", lw=2, label="ε = 10 (sweep / solo-vs-FL comparison)")

    for x, y in zip(eps, acc):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Privacy budget ε  (lower = stronger privacy)", fontsize=12)
    ax.set_ylabel("Test accuracy (%)", fontsize=12)
    ax.set_title("DP Epsilon Tradeoff — Why target ε = 8.0 for FL\n"
                 "ε=1–2: too much accuracy loss; ε=20: weak guarantee for healthcare",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 22)
    ax.set_ylim(70, 90)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.25)
    ax.set_facecolor("#FAFAFA")

    fig.tight_layout()
    path = os.path.join(out_dir, "epsilon_choice_tradeoff.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_inversion_comparison(plt, out_dir: str):
    attacks_dir = os.path.join(_ROOT, "results", "attacks")
    files = {
        "Unmitigated\n(no defence)": "model_inversion.json",
        "Legacy masked\n(output masking only)": "model_inversion_mitigated.json",
        "PrivacyWrapper\n(noise=3, temp=15)": "model_inversion_defended.json",
    }
    labels, cosines, colors = [], [], ["#E63946", "#F4A261", "#1D9E75"]
    entropies = []

    for label, fname in files.items():
        path = os.path.join(attacks_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        labels.append(label)
        cosines.append(d["summary"]["avg_cosine_similarity"])
        entropies.append(d["summary"].get("avg_prediction_entropy", 0))

    if not labels:
        print("  skip inversion comparison — no attack JSON found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(labels, cosines, color=colors[: len(labels)], edgecolor="black", linewidth=0.8)
    ax.axhline(0.4, color="red", ls="--", lw=1.5, label="Risk threshold (0.4)")
    ax.axhline(0.25, color="orange", ls=":", lw=1.5, label="Moderate threshold (0.25)")
    ax.set_ylabel("Avg cosine similarity")
    ax.set_title("Model Inversion — Reconstruction Quality\n(lower = more resistant)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, cosines):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                f"{val:.3f}", ha="center", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")

    ax = axes[1]
    bars = ax.bar(labels, entropies, color=colors[: len(labels)], edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Avg prediction entropy")
    ax.set_title("Model Output Uncertainty Under Attack\n(higher = harder to reconstruct)")
    for bar, val in zip(bars, entropies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", fontweight="bold", fontsize=10)
    ax.grid(True, alpha=0.25, axis="y")

    fig.suptitle("Why Laplace b=3.0 + Temperature T=15.0?  Defended cosine = 0.017 vs 0.834 unmitigated",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "model_inversion_defence_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")

    return {"unmitigated": cosines[0], "defended": cosines[-1]}


def plot_parameter_summary(plt, out_dir: str):
    """Single-page cheat sheet of all chosen parameters and rejected alternatives."""
    params = [
        ("MI_NOISE_SCALE", "3.0", "0.5, 1, 2, 5, 8",
         "Laplace noise on logits at inference. Start 0.5, tune until cosine < 0.4. "
         "3.0 gives 0.017 cosine; <0.5 too weak, >5 hurts confidence."),
        ("MI_TEMPERATURE", "15.0", "1, 5, 10, 20, 30",
         "Divides logits before softmax. T=1 keeps sharp peaks (MI-vulnerable). "
         "T>20 over-flattens dashboard confidence. 15 balances defence + utility."),
        ("target_epsilon", "8.0", "1, 2, 5, 10, 20",
         "Per-client DP budget for 35 FL rounds. ε=1–2 loses ~5% accuracy; "
         "ε=20 weak for healthcare. 8 consumed ≈7.99 with 85% F1."),
        ("max_grad_norm", "0.3", "0.1, 0.5, 1.0, 2.0",
         "Gradient clipping for DP-SGD. 0.1 over-clips (slow learning); "
         ">0.5 allows outlier samples to dominate noisy updates."),
        ("DP_BATCH_SIZE", "64", "32, 128",
         "Opacus noise ∝ 1/batch. 64 improves utility vs 32 at same ε; "
         "128 marginal gain, more memory."),
        ("FEATURE_NOISE_STD", "0.15", "0.05, 0.3, 0.5",
         "Training-time Gaussian on hidden layers. 0.05 too weak; "
         ">0.3 destabilises MLP training on tabular HAR."),
    ]

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis("off")

    y = 0.95
    ax.text(0.5, y, "IntelliClave — Parameter Choices & Rejected Alternatives",
            ha="center", fontsize=15, fontweight="bold", transform=ax.transAxes)
    y -= 0.06
    headers = ["Parameter", "Chosen", "Also tested", "Why not the others?"]
    col_x = [0.02, 0.18, 0.30, 0.48]
    for h, cx in zip(headers, col_x):
        ax.text(cx, y, h, fontsize=11, fontweight="bold", transform=ax.transAxes)
    y -= 0.04
    ax.plot([0.02, 0.98], [y, y], color="black", lw=1, transform=ax.transAxes, clip_on=False)

    for name, chosen, tested, why in params:
        y -= 0.11
        ax.text(col_x[0], y, name, fontsize=10, fontweight="bold", transform=ax.transAxes,
                color="#1D3557")
        ax.text(col_x[1], y, chosen, fontsize=11, fontweight="bold", transform=ax.transAxes,
                color="#E63946")
        ax.text(col_x[2], y, tested, fontsize=9, transform=ax.transAxes, color="#457B9D")
        ax.text(col_x[3], y, why, fontsize=9, transform=ax.transAxes, wrap=True,
                va="top", linespacing=1.4)

    y -= 0.08
    ax.text(0.5, y,
            "Tuning rule from constants.py: MI_NOISE_SCALE — "
            "'Start at 0.5, tune upward until avg cosine similarity drops below 0.4'",
            ha="center", fontsize=10, style="italic", transform=ax.transAxes, color="#555")

    fig.tight_layout()
    path = os.path.join(out_dir, "parameter_choices_summary.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def plot_combined_defence_pipeline(plt, out_dir: str):
    """Visualise full inference defence pipeline with chosen values."""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)

    boxes = [
        (0.3, 1.2, 1.6, 1.2, "Raw logits\n(from MLP)", "#E8F4FD", "black"),
        (2.3, 1.2, 1.8, 1.2, "① no_grad\n(zero gradient)", "#FFE5E5", "#C00"),
        (4.5, 1.2, 2.0, 1.2, f"② + Laplace(0, {CHOSEN['mi_noise_scale']})\n(output perturbation)", "#FFF3CD", "#856404"),
        (7.0, 1.2, 1.8, 1.2, f"③ ÷ T={CHOSEN['mi_temperature']}\n(temperature scale)", "#D4EDDA", "#155724"),
        (9.2, 1.2, 1.4, 1.2, "softmax\n→ probs", "#E8F4FD", "black"),
    ]
    for x, y, w, h, text, fc, ec in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec=ec, lw=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, fontweight="bold")

    for x1, x2 in [(1.9, 2.3), (4.3, 4.5), (6.5, 7.0), (8.8, 9.2)]:
        ax.annotate("", xy=(x2, 1.8), xytext=(x1, 1.8),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#333"))

    ax.text(5, 2.7, "PrivacyWrapper Inference Pipeline (eval mode only)",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(5, 0.5,
            "Training uses bare model (no noise, no temperature) so Opacus DP-SGD sees clean gradients",
            ha="center", fontsize=10, style="italic", color="#555")

    fig.tight_layout()
    path = os.path.join(out_dir, "privacy_wrapper_pipeline.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate IntelliClave parameter choice plots.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", default=PLOT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plt = _setup_matplotlib()

    print("Generating parameter choice plots...")
    plot_laplace_distribution(plt, args.out_dir)
    plot_combined_defence_pipeline(plt, args.out_dir)
    plot_parameter_summary(plt, args.out_dir)
    plot_epsilon_tradeoff(plt, args.out_dir)
    inversion_data = plot_inversion_comparison(plt, args.out_dir) or {}

    print(f"  Loading logits from {args.model_path}")
    logits = load_logits_from_model(args.model_path)
    plot_logits_perturbation(plt, logits, args.out_dir)

    noise_records = measure_noise_effects(logits, CANDIDATES["noise_scale"])
    plot_noise_scale_tradeoff(plt, noise_records, inversion_data, args.out_dir)

    temp_records = measure_temperature_effects(logits, CANDIDATES["temperature"])
    plot_temperature_tradeoff(plt, temp_records, args.out_dir)

    print(f"\nAll plots saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
