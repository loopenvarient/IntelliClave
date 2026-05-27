"""
Shared project-wide constants.

Import these instead of repeating magic numbers across scripts:

    from config.constants import DEFAULT_EPSILON, DEFAULT_N_CLIENTS
"""

# ── Privacy ───────────────────────────────────────────────────────────────────
DEFAULT_EPSILON: float = 1.0   # balanced DP budget: resistant MI results with better utility
HIGH_PRIVACY_EPSILON: float = 0.5   # stronger privacy, lower utility
DEFAULT_DELTA_FALLBACK: float = 1e-5  # used only when row count is unavailable

# ── Federated Learning ────────────────────────────────────────────────────────
DEFAULT_N_CLIENTS: int   = 3
DEFAULT_FL_ROUNDS: int   = 10
DEFAULT_LOCAL_EPOCHS: int = 3

# ── Training ──────────────────────────────────────────────────────────────────
DEFAULT_LR: float         = 1e-3
DEFAULT_BATCH_SIZE: int   = 32   # canonical batch size used by FL training
DP_BATCH_SIZE: int        = 64   # larger batch used by DP/sweep experiments
                                  # (Opacus noise scales with batch size — keep separate)

# ── Data ──────────────────────────────────────────────────────────────────────
LABEL_COL: str    = "label"
RANDOM_SEED: int  = 42
TEST_SPLIT: float = 0.2

# ── Model Inversion Defence ───────────────────────────────────────────────────
# Output perturbation: Laplace noise added to logits before softmax at inference.
# Higher = stronger privacy, slightly lower accuracy. Start at 0.5, tune upward
# until avg cosine similarity in model_inversion.py drops below 0.4.
MI_NOISE_SCALE: float = 3.0

# Temperature scaling: divides logits by T before softmax.
# Flattens confidence peaks — reduces the gradient signal the inversion optimizer
# follows. Increase further if surrogate attacks remain strong.
MI_TEMPERATURE: float = 15.0

# Inference/output privacy for serving predictions.
# Round probabilities to the nearest 0.1, expose only top-k predictions, and
# optionally add randomized response to reduce surrogate learning signal.
OUTPUT_PROB_ROUNDING_STEP: float = 0.1
OUTPUT_TOP_K: int = 1
OUTPUT_RANDOM_RESPONSE_PROB: float = 0.1

# Master switch — set False to disable both defences (e.g. for ablation studies
# or when running the attack scripts themselves in unmitigated mode).
MI_DEFENCE_ENABLED: bool = True

FEATURE_NOISE_STD = 0.2
DROPOUT_RATE = 0.6
LABEL_SMOOTHING = 0.15
CONFIDENCE_PENALTY = 0.03
WEIGHT_DECAY = 1e-4
