# IntelliClave - Final Results Summary

This document summarizes the final commands executed, what each command tested, and the observed results. It is designed as a source for the final report and presentation slides.

## Executive Summary

| Area | Command / Test | Main Result | Verdict |
|---|---|---:|---|
| Data preparation | Dirichlet non-IID split, alpha=0.5, 3 clients, no PCA | 10,299 total HAR samples split across 3 clients | Non-IID label distribution created successfully |
| Membership inference | No-DP baseline attack | Avg AUC = 0.5050, avg confidence gap = 0.0024 | Resistant / Low risk |
| Gradient poisoning (FedAvg baseline) | Label-flip attack, standard FedAvg | 100% poison: accuracy 91.85% → 15.04% (drop 0.7681) | Vulnerable / High risk |
| Gradient poisoning (trimmed mean) | Same attack with robust aggregation (`--robust`) | 100% poison: accuracy 81.95% → 80.93% (drop 0.0102) | Resistant / Low risk |
| Crypto layer | Encrypt/decrypt, integrity, tamper rejection, fresh nonce | 4/4 tests passed | Passed |
| TLS certificates | CA, server, and client certificate generation | Bundle generated and verified | Passed |
| TEE attestation | Server quote + client verification + rogue server rejection | All clients verified, rogue server blocked | Passed |
| Model inversion defense | Defended attack result file | Avg cosine similarity = 0.0170 | Resistant / Low risk |

## 1. Data Preparation: Non-IID Dirichlet Split

### Command

```powershell
python data/datascripts/pipeline.py --mode textfiles --partition dirichlet --dirichlet-alpha 0.5 --n-clients 3 --no-pca
```

### Purpose

This command converted the UCI HAR raw text files into three federated client CSV files using a Dirichlet partition. The Dirichlet parameter `alpha=0.5` intentionally creates a non-IID label distribution across clients. PCA was disabled, so the full 561-feature HAR vectors were retained.

### Input Dataset

| Item | Value |
|---|---:|
| Total samples | 10,299 |
| Feature count | 561 |
| Label count | 6 classes |
| Subject records | 10,299 |
| PCA | Disabled |
| Partition type | Dirichlet |
| Dirichlet alpha | 0.5 |
| Number of clients | 3 |

### Client Splits

| Client | Rows | Features | Label 0 | Label 1 | Label 2 | Label 3 | Label 4 | Label 5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| client1 | 5,529 | 561 | 1,494 | 1,184 | 401 | 785 | 782 | 883 |
| client2 | 2,288 | 561 | 27 | 331 | 651 | 238 | 8 | 1,033 |
| client3 | 2,482 | 561 | 201 | 29 | 354 | 754 | 1,116 | 28 |

### Interpretation

The split is clearly non-IID. Each client has all six classes, but the class proportions differ strongly:

- `client1` is dominated by labels 0 and 1.
- `client2` is dominated by labels 2 and 5, with very few label 0 and label 4 samples.
- `client3` is dominated by labels 3 and 4, with very few label 1 and label 5 samples.

This is a stronger and more realistic federated learning distribution than an IID/random split because each client sees a different local data profile.

### Verification Output

All generated CSVs passed validation:

| Check | Result |
|---|---|
| No missing values | Passed |
| No infinite values | Passed |
| Same feature schema across clients | Passed |
| All clients contain labels 0-5 | Passed |
| Distribution chart generated | `data/client_distributions.png` |

## 2. Membership Inference Attack

### Command

```powershell
python security\attacks\membership_inference.py
```

### Purpose

This attack checks whether an adversary can infer whether a sample was part of the model's training data. The attack compares model confidence on member samples versus non-member samples. AUC near 0.5 means the attack is close to random guessing.

### Result: No-DP Baseline

| Client | Members | Non-members | Member Confidence | Non-member Confidence | Confidence Gap | AUC | Accuracy | Risk |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| client1 | 3,870 | 1,659 | 0.5876 | 0.5867 | 0.0009 | 0.5023 | 0.6999 | Low |
| client2 | 1,601 | 687 | 0.5233 | 0.5169 | 0.0064 | 0.5090 | 0.7002 | Low |
| client3 | 1,737 | 745 | 0.4982 | 0.4983 | -0.0001 | 0.5036 | 0.6998 | Low |

### Summary

| Metric | Value |
|---|---:|
| Average AUC | 0.5050 |
| Average accuracy | 0.7000 |
| Average confidence gap | 0.0024 |
| High-risk clients | 0 |
| Verdict | Resistant |

### Interpretation

The attack is ineffective because the AUC is almost exactly random guessing. The confidence gap between training and non-training samples is also very small. This indicates the model is not strongly leaking membership information through prediction confidence.

## 3. Gradient Poisoning Attack

This attack simulates a malicious federated client that flips local training labels before sending updates. It is run in **two modes**: a FedAvg baseline (shows the threat) and a **trimmed-mean robust rerun** (shows the mitigation). The dashboard reports the robust result from `results/attacks/gradient_poisoning_robust.json`.

### Shared configuration

| Setting | Value |
|---|---:|
| FL rounds | 5 |
| Local epochs | 3 |
| Poisoned client | Client 1 |
| Flip target | Class 1 / class_2 |
| Attack type | Label flip |

The script also reported:

```text
model_meta.json input_dim=50 differs from CSV feature count=561. Using CSV dim for this simulation.
```

This is expected after regenerating the dataset with `--no-pca`, because the new CSV files contain 561 raw HAR features instead of the earlier 50 PCA features. The attack script correctly used the CSV feature dimension.

---

### 3a. FedAvg baseline (threat demonstration)

#### Command

```powershell
python security\attacks\gradient_poisoning.py
```

#### Output file

`results/attacks/gradient_poisoning.json`

#### Poison rate sweep

| Poison Rate | Accuracy | Accuracy Drop | Macro F1 | F1 Drop |
|---:|---:|---:|---:|---:|
| 0% clean baseline | 0.9185 | 0.0000 | 0.9189 | 0.0000 |
| 10% | 0.9107 | -0.0078 | 0.9123 | -0.0066 |
| 30% | 0.9010 | -0.0175 | 0.9021 | -0.0168 |
| 50% | 0.8991 | -0.0194 | 0.8959 | -0.0230 |
| 100% | 0.1504 | -0.7681 | 0.0443 | -0.8746 |

#### Summary

| Metric | Value |
|---|---:|
| Clean baseline accuracy | 0.9185 |
| Clean baseline macro F1 | 0.9189 |
| 100% poison accuracy | 0.1504 |
| 100% poison macro F1 | 0.0443 |
| Accuracy drop | 0.7681 |
| F1 drop | 0.8746 |
| Risk | High |
| Verdict | Vulnerable |

#### Interpretation

Under standard FedAvg, the system tolerates small and moderate poisoning (10–50% label flips) but **collapses at 100% poison**: global accuracy falls from 91.85% to 15.04%. This establishes that the Byzantine threat is real when aggregation blindly averages all client updates.

---

### 3b. Trimmed-mean defence (mitigation rerun)

#### Command

```powershell
python security\attacks\gradient_poisoning.py --robust
```

#### Output file

`results/attacks/gradient_poisoning_robust.json`

#### Defence mechanism

**Trimmed-mean aggregation** drops the client update whose full weight vector is farthest from the centroid of all updates, then averages the remaining honest clients. With 3 clients and 1 fully poisoned participant, the outlier update is excluded each round.

#### Poison rate sweep

| Poison Rate | Accuracy | Accuracy Drop vs Baseline | Macro F1 | F1 Drop vs Baseline |
|---:|---:|---:|---:|---:|
| 0% clean baseline | 0.8195 | 0.0000 | 0.8238 | 0.0000 |
| 10% | 0.8079 | -0.0116 | 0.8067 | -0.0172 |
| 30% | 0.8549 | +0.0354 | 0.8535 | +0.0296 |
| 50% | 0.8399 | +0.0204 | 0.8389 | +0.0151 |
| 100% | 0.8093 | -0.0102 | 0.8127 | -0.0111 |

#### Summary

| Metric | Value |
|---|---:|
| Clean baseline accuracy | 0.8195 |
| Clean baseline macro F1 | 0.8238 |
| 100% poison accuracy | 0.8093 |
| 100% poison macro F1 | 0.8127 |
| Accuracy drop at 100% poison | 0.0102 |
| F1 drop at 100% poison | 0.0111 |
| Risk | Low |
| Verdict | Resistant |
| Aggregation | trimmed_mean |

#### Interpretation

With trimmed-mean aggregation, the same worst-case attack (100% label-flip on Client 1) causes only a **1.02 percentage point** accuracy drop (81.95% → 80.93%). Macro F1 drops by 1.11 points. The poisoned client's outlier weights are rejected before they reach the global model.

**Before vs after mitigation (100% poison):**

| Aggregation | Baseline Acc | 100% Poison Acc | Drop |
|---|---:|---:|---:|
| FedAvg | 0.9185 | 0.1504 | 0.7681 |
| Trimmed mean | 0.8195 | 0.8093 | 0.0102 |

#### Deployment note

This robust result is validated in the **attack simulation** and shown on the **dashboard**. The live FL server (`fl/fl_server.py`) still uses FedAvg/FedProx weighted averaging — trimmed mean is the recommended next step for production aggregation.

#### Additional mitigations (complementary)

- Krum or coordinate-wise median aggregation (alternative robust rules).
- Per-client update anomaly detection before aggregation.
- DP-SGD gradient clipping (limits per-sample influence during training).
- Track per-client update drift across FL rounds.

## 4. Crypto Layer Tests

### Command

```powershell
python crypto/certs/test_crypto.py
```

### Purpose

This test validates secure encryption and integrity behavior for model/update payloads.

### Results

| Test | Description | Result |
|---|---|---|
| Test 1 | Encrypt then decrypt returns the same arrays | Pass |
| Test 2 | Integrity check accepts valid data | Pass |
| Test 3 | Tampered ciphertext is rejected | Pass |
| Test 4 | Same plaintext twice creates different ciphertext using fresh nonce | Pass |

### Summary

| Metric | Value |
|---|---:|
| Tests passed | 4/4 |
| Max absolute difference after decrypt | 0.00e+00 |
| Tampered ciphertext rejected | Yes |
| Fresh nonce confirmed | Yes |
| Verdict | All crypto tests passed |

### Interpretation

The crypto layer preserves model/update correctness after decryption, detects tampering, and avoids deterministic ciphertext reuse by generating fresh nonces.

## 5. TLS Certificate Generation

### Command

```powershell
python crypto/certs/generate_tls_certs.py
```

### Purpose

This command generated a TLS certificate bundle for secure communication between the FL server and FL clients.

### Generated Files

| Certificate / Key | Path |
|---|---|
| CA certificate | `crypto/certs/tls/ca.crt` |
| Server certificate | `crypto/certs/tls/server.crt` |
| Server key | `crypto/certs/tls/server.key` |
| Client certificate | `crypto/certs/tls/client.crt` |
| Client key | `crypto/certs/tls/client.key` |

### Verification

| Item | Value |
|---|---|
| CA subject | IntelliClave Root CA |
| Server subject | fl-server |
| Client subject | fl-client |
| Valid until | 2027-06-07 |
| Bundle verification | Passed |

### Example Usage

Server:

```powershell
python fl/run_server.py --tls `
  --tls-ca crypto/certs/tls/ca.crt `
  --tls-cert crypto/certs/tls/server.crt `
  --tls-key crypto/certs/tls/server.key
```

Client:

```powershell
python fl/run_client.py --id 1 --tls `
  --tls-ca crypto/certs/tls/ca.crt `
  --tls-cert crypto/certs/tls/client.crt `
  --tls-key crypto/certs/tls/client.key
```

### Interpretation

TLS support is ready for secure FL server-client communication. The generated bundle establishes a local CA and separate certificates for server and client authentication.

## 6. TEE Attestation Integration

### Command

```powershell
python tee/attestation/attestation_integration.py
```

### Purpose

This demo validates that clients verify the FL server's enclave identity before connecting, and that a rogue server with the wrong measurement is rejected.

### Attestation Record

| Field | Value |
|---|---|
| TEE verified | true |
| Enclave ID | `intelliclave-enclave-v1` |
| Platform | Intel SGX simulated |
| Mode | gramine-direct |
| Environment | WSL2 |
| Status | VERIFIED |
| Quote size | 218 bytes |
| Integrity hash prefix | `3f2bbc8c935d5a10` |

### Demo Results

| Step | Result |
|---|---|
| Server quote generated | Passed |
| Client 1 verified server MRENCLAVE | Passed |
| Client 2 verified server MRENCLAVE | Passed |
| Client 3 verified server MRENCLAVE | Passed |
| Rogue server simulation | Blocked |
| Final demo status | Passed |

### Interpretation

The attestation flow is correctly wired into the FL server-client trust model. Legitimate clients accept the verified server measurement, while a rogue server with an incorrect MRENCLAVE is rejected.

## 7. Model Inversion Defense Result

### Source

```text
results/attacks/model_inversion_defended.json
```

### Purpose

Model inversion attempts to reconstruct representative input data from model outputs. The defended mode used noise and temperature smoothing.

### Summary

| Metric | Value |
|---|---:|
| Mode | defended, noise=3.0, temp=15.0 |
| Average cosine similarity | 0.0170 |
| Average model confidence | 0.2458 |
| Average prediction entropy | 1.7547 |
| Average prediction sharpness | 0.0636 |
| High-risk classes | 0 |
| Verdict | Resistant |

### Interpretation

The defended inversion attack failed to reconstruct meaningful inputs. The average cosine similarity is close to zero, confidence is low, entropy is high, and no class was marked high risk.

## 8. Current Run Status Snapshot

### Source

```text
status.json
```

| Field | Value |
|---|---:|
| Round | 35 / 35 |
| Clients | 3 |
| Accuracy | 0.85 |
| Macro F1 | 0.78 |
| Loss | 0.45824 |
| Epsilon | 7.99471 |
| Noise scale | 0.5 |
| Temperature | 1.0 |
| Early stopped | false |
| Save directory | `results/fl_rounds/run_20260607_160153` |

## 9. Main Conclusions

1. The final dataset is non-IID.
   - The Dirichlet split with `alpha=0.5` created visibly skewed label distributions across clients.

2. Membership inference risk is low.
   - Average AUC is 0.5050, almost random guessing.
   - Confidence gap is only 0.0024.

3. Gradient poisoning: threat demonstrated, mitigation validated.
   - **FedAvg baseline:** a full label-flip by one client reduced accuracy from 91.85% to 15.04% (drop 0.7681) — vulnerable.
   - **Trimmed-mean defence:** the same 100% attack reduced accuracy from 81.95% to 80.93% (drop 0.0102) — resistant.
   - Dashboard reports the robust result; live `fl_server.py` should adopt trimmed mean (or Krum) for production parity.

4. The crypto layer works correctly.
   - Encryption/decryption is lossless.
   - Tampering is rejected.
   - Fresh nonces prevent repeated plaintext from producing identical ciphertext.

5. TLS setup is complete.
   - CA, server certificate, and client certificate were generated and verified.

6. Attestation flow works.
   - All clients accepted the legitimate server.
   - The rogue server simulation was correctly blocked.

7. Model inversion defense is effective in the defended configuration.
   - Reconstructed inputs do not match real data.
   - No high-risk classes were detected.

## 10. PPT-Ready Slide Outline

### Slide 1 - Project Goal

- IntelliClave secures federated learning using DP, cryptography, TLS, and TEE attestation.
- Evaluation covers utility, privacy leakage, poisoning resilience, secure communication, and trusted execution.

### Slide 2 - Final Data Setup

- UCI HAR dataset: 10,299 samples, 561 features, 6 activity classes.
- Three FL clients generated using Dirichlet non-IID partition.
- `alpha=0.5` creates heterogeneous client distributions.
- All CSV validation checks passed.

### Slide 3 - Non-IID Client Distribution

- Client 1: dominated by labels 0 and 1.
- Client 2: dominated by labels 2 and 5.
- Client 3: dominated by labels 3 and 4.
- Use figure: `data/client_distributions.png`.

### Slide 4 - Membership Inference Attack

- No-DP baseline was tested.
- Avg AUC = 0.5050, approximately random guessing.
- Avg confidence gap = 0.0024.
- Verdict: resistant, low risk.

### Slide 5 - Gradient Poisoning Attack

- Label-flip attack on one malicious client (same threat, two aggregation modes).
- **FedAvg baseline:** 91.85% → 15.04% at 100% poison (drop 76.81 pp) — vulnerable.
- **Trimmed-mean defence:** 81.95% → 80.93% at 100% poison (drop 1.02 pp) — resistant.
- Dashboard shows robust result; motivates wiring trimmed mean into `fl_server.py`.

### Slide 6 - Crypto and TLS

- Crypto tests: 4/4 passed.
- Tampered ciphertext rejected.
- Fresh nonce confirmed.
- TLS CA/server/client certificates generated and verified.

### Slide 7 - TEE Attestation

- Server enclave quote generated.
- All three clients verified the server MRENCLAVE.
- Rogue server simulation was rejected.
- Verdict: attestation integration passed.

### Slide 8 - Model Inversion Defense

- Defended mode used noise and temperature smoothing.
- Avg cosine similarity = 0.0170.
- High-risk classes = 0.
- Verdict: resistant.

### Slide 9 - Final Security Takeaway

- Strong results: membership inference, model inversion defense, crypto, TLS, attestation, gradient poisoning (with trimmed mean).
- FedAvg baseline confirms the Byzantine threat is real; trimmed-mean rerun shows the fix works.
- Recommended production step: adopt trimmed-mean (or Krum) in `fl_server.py` aggregation.

## 11. Report-Ready Short Paragraph

The final IntelliClave evaluation used a Dirichlet non-IID split of the UCI HAR dataset across three clients, preserving all 561 raw features. The generated client datasets passed integrity validation and showed intentionally skewed class distributions. Membership inference against the no-DP baseline produced an average AUC of 0.5050 and an average confidence gap of 0.0024, indicating near-random attack performance and low privacy leakage through confidence scores. The defended model inversion experiment was also resistant, with average cosine similarity of only 0.0170 and zero high-risk classes. For gradient poisoning, a FedAvg baseline run showed that a full label-flip by one poisoned client reduced global accuracy from 91.85% to 15.04%, confirming the Byzantine threat under standard aggregation. A follow-up run with trimmed-mean robust aggregation on the same attack reduced the 100% poison impact to only a 1.02 percentage point drop (81.95% to 80.93%), demonstrating that outlier client updates can be rejected effectively. The crypto layer passed all four tests, including lossless decryption, integrity verification, tamper rejection, and fresh nonce behavior. TLS certificates were generated and verified for CA, server, and client roles. TEE attestation was successfully demonstrated: all clients verified the legitimate server measurement, while a rogue server simulation was blocked. The recommended production follow-up is to wire trimmed-mean (or Krum) aggregation into the live FL server so training matches the validated attack-lab defence.

