# Model Inversion Resistance

This note documents the changes made to harden IntelliClave against the tested
white-box model inversion attack in `security/attacks/model_inversion.py`.

## Threat Model

The current attack script tests a white-box or full-model-disclosure scenario:
the attacker has access to the model checkpoint, architecture, preprocessing
metadata, and can optimize inputs with gradients.

This is stronger than the dashboard black-box API threat model. API defenses
such as confidence masking, temperature scaling, randomized response, and rate
limits help black-box attacks, but they do not stop a white-box attacker who has
the raw model weights.

## What Changed

The training path was hardened so the learned model leaks less class-level
feature structure even when the checkpoint is available for testing.

Changes made:

- Enabled privacy-first DP-SGD defaults for FL clients.
- Set the balanced DP budget to `epsilon=1.0`.
- Kept a stronger high-privacy option at `epsilon=0.5`.
- Applied label smoothing during federated training.
- Applied confidence penalty through the shared `train_one_epoch` training path.
- Increased dropout and feature noise to reduce memorization.
- Added Adam weight decay in FL client and simulation training.
- Updated Docker and Kubernetes DP defaults from weak `epsilon=10.0`/`0.5` to the balanced `epsilon=1.0`.
- Added `--privacy-mode balanced` and `--privacy-mode high` to the simulation CLI.

Important constants are in `config/constants.py`:

```python
DEFAULT_EPSILON = 1.0
HIGH_PRIVACY_EPSILON = 0.5
FEATURE_NOISE_STD = 0.2
DROPOUT_RATE = 0.6
LABEL_SMOOTHING = 0.15
CONFIDENCE_PENALTY = 0.03
WEIGHT_DECAY = 1e-4
```

## Results

White-box inversion results after hardening:

| Model | Accuracy | Macro F1 | Avg Cosine Similarity | High-Risk Classes | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| Original | not recorded here | not recorded here | `0.830927` | `6/6` | Vulnerable |
| DP `epsilon=0.5` hardened | `0.3484` | `0.2883` | `0.185546` | `0/6` | Resistant |
| DP `epsilon=1.0` balanced | `0.5012` | `0.4711` | `0.209430` | `0/6` | Resistant |
| DP `epsilon=2.0` | not selected | not selected | `0.3440` | `0/6` | Moderate leakage |

The selected default is now `epsilon=1.0` because it keeps the tested white-box
attack resistant while giving much better utility than `epsilon=0.5`.

Do not claim `epsilon=2.0` is resistant. Its main unmitigated result is
moderate, and its mitigated/defended runs showed vulnerable leakage.

## How To Run

Use UTF-8 mode on Windows so progress symbols print correctly:

```powershell
$env:PYTHONUTF8='1'
```

Install Flower simulation support if Ray is missing:

```powershell
pip install -U "flwr[simulation]"
pip install "numpy<2.0"
```

Train the selected balanced secure model:

```powershell
python fl\run_fl_simulation.py --privacy-mode balanced --rounds 10 --clients 3 --save-dir results\fl_rounds\run_privacy_balanced
```

Equivalent explicit command:

```powershell
python fl\run_fl_simulation.py --dp --epsilon 1.0 --max-grad-norm 0.3 --rounds 10 --local-epochs 1 --save-dir results\fl_rounds\run_privacy_balanced
```

Run the white-box inversion attack:

```powershell
python security\attacks\model_inversion.py --model-path results\fl_rounds\run_privacy_balanced\global_model_latest.pth --out results\attacks\model_inversion_privacy_balanced.json
```

Compare against the original vulnerable run:

```powershell
python security\attacks\summarize_attack_results.py --files results\attacks\model_inversion.json results\attacks\model_inversion_privacy_balanced.json
```

For the stronger but lower-utility setting:

```powershell
python fl\run_fl_simulation.py --privacy-mode high --rounds 10 --clients 3 --save-dir results\fl_rounds\run_privacy_high
```

## Current Claim

IntelliClave is resistant to the tested white-box model inversion attack when
trained with the balanced secure DP configuration (`epsilon=1.0`, gradient norm
`0.3`, label smoothing, confidence penalty, dropout, feature noise, and weight
decay).

This is not a claim of resistance to every possible inversion method. The next
security step is to add a true black-box API inversion test against `/predict`.
