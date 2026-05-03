# IntelliClave — Report Directory

All report sections, documentation, and figures are collected here.

---

## Report Sections

| File | Content |
|------|---------|
| `tee_section.md` | TEE architecture, Gramine, attestation, sealed storage, benchmarks |
| `security_section.md` | STRIDE analysis, attack simulations, crypto, DP results |
| `fl_dp_integration.md` | FL + DP integration design, running instructions, confirmed results |
| `fl_process_flow.md` | FL pipeline file-by-file walkthrough, round flow, data flow |
| `deployment_scenario.md` | FitLife / MediTrack / CareWatch deployment narrative, regulatory compliance |

---

## STRIDE Analysis

| File | Category |
|------|---------|
| `stride_complete.md` | All 6 categories — full threat table |
| `stride_spoofing.md` | S1–S3 detailed with attack paths and evidence |
| `stride_tampering.md` | T1–T4 detailed with test results |
| `stride_info_disclosure.md` | I1–I4 detailed with attack simulation numbers |

---

## Full Security Report

`security_report.md` — consolidated report covering:
- Crypto layer (4/4 tests)
- STRIDE summary (21 threats, all mitigated)
- Attack simulation results (3 attacks)
- DP results (ε=10, AUC=0.503)
- TEE overhead (35.2% avg)
- K8s security
- Residual risks

---

## Figures

| File | Description |
|------|-------------|
| `figures/accuracy_vs_epsilon.png` | Privacy-utility tradeoff curve (ε vs test accuracy) |
| `figures/epsilon_over_rounds.png` | Privacy budget consumption across FL rounds |
| `figures/dp_summary.png` | Combined DP summary chart |
| `figures/client_distributions.png` | Non-IID class distribution across 3 clients |
| `figures/graph6_final_results.png` | **Graph 6** — 4-panel final results summary |

---

## Key Numbers (Quick Reference)

| Metric | Value |
|--------|-------|
| FL baseline accuracy (no DP, 10 rounds) | 96.99% |
| FL+DP accuracy (ε=10, 5 rounds) | 91.27% |
| Privacy cost | −5.72% |
| Membership inference AUC | 0.503 (random) |
| Gradient poisoning drop (100%) | −3.78% |
| TEE overhead | 35.2% avg |
| TEE overhead as % of FL round | 0.14% |
| Crypto tests | 4/4 PASS |
| Attestation | VERIFIED |
| Sealed storage | All tests PASS |
