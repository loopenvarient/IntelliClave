# report/

Project documentation, threat analysis, and results writeups.

## Purpose

This folder holds **academic and technical documentation** for IntelliClave: FL+DP integration narrative, STRIDE threat model, security findings, deployment scenarios, and final results summaries. It is **not executed at runtime** — it documents and cites outputs from other folders.

## Key files

| Path | Content |
|------|---------|
| `README.md` | Index of all report sections (this file's siblings) |
| `project_inventory.md` | Most complete per-file map + run guide in the repo |
| `final_results_summary.md` | Consolidated experiment numbers and interpretations |
| `fl_process_flow.md` | End-to-end FL process description |
| `fl_dp_integration.md` | How Opacus DP integrates with Flower clients |
| `deployment_scenario.md` | Hospital / multi-site deployment narrative |
| `security_report.md` | Security evaluation overview |
| `security_section.md` | Security chapter content |
| `tee_section.md` | TEE / Gramine chapter content |
| `stride_complete.md` | Full STRIDE analysis |
| `stride_spoofing.md` | Spoofing threats |
| `stride_tampering.md` | Tampering threats (includes gradient poisoning) |
| `stride_info_disclosure.md` | Information disclosure threats |
| `figures/` | PNG assets for the report (regenerate via `evaluation/generate_graph6.py`) |

## How it connects

- Cites numbers from **`results/`**, root `status.json`, `attestation.json`
- Cross-references **`fl/`**, **`privacy/`**, **`security/`**, **`tee/`**, **`crypto/`**
- Complements root **`README.md`** and **`HANDOFF.md`**

## Usage

Open and edit markdown directly. Regenerate figures before final PDF export:

```bash
python evaluation/generate_graph6.py
python privacy/epsilon_sweep.py --epsilons 1 2 5 10 20
```
