# config/constants.py
"""
Shared project-wide constants.

Import these instead of repeating magic numbers across scripts:

    from config.constants import DEFAULT_EPSILON, DEFAULT_N_CLIENTS
"""

# ── Privacy ───────────────────────────────────────────────────────────────────
DEFAULT_EPSILON: float = 10.0   # DP privacy budget (ε)
DEFAULT_DELTA_FALLBACK: float = 1e-5  # used only when row count is unavailable
DEFAULT_MAX_GRAD_NORM: float = 2.0  # DP-SGD clipping norm C (sweep-optimised for ε=10)

# ── Federated Learning ────────────────────────────────────────────────────────
DEFAULT_N_CLIENTS: int   = 3
DEFAULT_FL_ROUNDS: int   = 10
DEFAULT_LOCAL_EPOCHS: int = 3

# ── Training ──────────────────────────────────────────────────────────────────
DEFAULT_LR: float         = 1e-3
DEFAULT_BATCH_SIZE: int   = 32   # canonical batch size used by non-DP FL training

# ── DP accuracy optimisation defaults ────────────────────────────────────────
# These three settings work together to reduce the ~5.7% accuracy cost of DP:
#
# 1. DP_BATCH_SIZE — larger batches reduce per-sample noise by averaging more
#    gradients before the noise is added. Opacus noise_std ∝ 1/√batch_size,
#    so doubling the batch size cuts noise_std by ~29%.
#    Used automatically by IntelliClaveClient when use_dp=True.
#
# 2. DP_LR_BOOST — DP noise reduces the effective gradient signal. A 2× LR
#    boost helps the model learn faster per step without changing the privacy
#    guarantee. Applied on top of DEFAULT_LR in DPTrainer and fl_client.py.
#
# 3. DP_DROPOUT — DP noise already acts as strong regularisation (it adds
#    calibrated noise to every gradient update). Stacking Dropout(0.3) on top
#    adds variance without adding privacy, compounding the accuracy cost.
#    Set to 0.0 when DP is active; the model is built with this value via
#    get_model(dropout=DP_DROPOUT) in IntelliClaveClient.
#
# Combined effect on UCI HAR (ε=10, 5 rounds, 3 clients):
#   Before: 91.27% (−5.72% vs no-DP baseline of 96.99%)
#   After:  ~94–95% (estimated −2–3% cost) — run epsilon_sweep.py to verify.
DP_BATCH_SIZE: int    = 64    # 2× DEFAULT_BATCH_SIZE
DP_LR_BOOST: float    = 2.0   # multiplier applied to DEFAULT_LR under DP
DP_DROPOUT: float     = 0.0   # dropout rate when DP is active (0 = disabled)

# ── Inference API (P0 — black-box model inversion) ───────────────────────────
INFERENCE_EPSILON_DEFAULT: float = 4.0
LIFETIME_QUERY_BUDGET_DEFAULT: int = 10_000
RATE_LIMIT_MAX_DEFAULT: int = 100
RATE_LIMIT_WINDOW_SECS_DEFAULT: int = 60

# ── Data ──────────────────────────────────────────────────────────────────────
LABEL_COL: str    = "label"
RANDOM_SEED: int  = 42
TEST_SPLIT: float = 0.2
