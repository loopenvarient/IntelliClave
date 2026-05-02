# IntelliClave — Deployment Scenario

## Use Case: Federated Activity Recognition Across Health & Insurance Companies

### Problem Statement

Three health and insurance companies each hold smartphone sensor data from their users.
They want to collaboratively train a shared human activity recognition (HAR) model to
improve accuracy across diverse user populations — without sharing any raw user data.

---

### The Three Companies

**Company A — FitLife (Fitness App)**
- Users: active individuals, fitness enthusiasts
- Data profile: high proportion of WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS
- Mapped to: Client 1 (subjects 1–10, active profile)
- Concern: competitive sensitivity of user behaviour data

**Company B — MediTrack (Healthcare Monitoring)**
- Users: general patients, mixed age groups, post-operative monitoring
- Data profile: balanced mix of all 6 activities
- Mapped to: Client 2 (subjects 11–20, mixed profile)
- Concern: HIPAA compliance, patient confidentiality

**Company C — CareWatch (Elderly Care Monitoring)**
- Users: elderly residents, assisted living facilities
- Data profile: high proportion of SITTING, STANDING, LAYING
- Mapped to: Client 3 (subjects 21–30, sedentary profile)
- Concern: GDPR, PMDC patient data sovereignty

---

### Why Federated Learning

- Raw sensor data never leaves each company's infrastructure
- Only model weight updates (gradients) are shared with the central aggregator
- IntelliClave acts as the trusted aggregation server
- Each round: local training → encrypted gradient upload → FedAvg aggregation → global model update

---

### Regulatory Compliance

| Regulation     | Requirement                          | How IntelliClave Satisfies It                          |
|----------------|--------------------------------------|--------------------------------------------------------|
| GDPR Art. 25   | Privacy by design & by default       | Raw data never transmitted; PCA reduces identifiability |
| GDPR Art. 5(1) | Data minimisation                    | Only 50 PCA features used, not raw 561-dim signals     |
| HIPAA          | Minimum necessary standard           | Model gradients contain no patient-identifiable data   |
| HIPAA          | Administrative safeguards            | TLS-encrypted communication between clients and server |
| PMDC           | Patient data stays local             | Each client trains on-premise; no data egress          |

---

### Data Flow

```
Company A (Client 1)          Company B (Client 2)          Company C (Client 3)
  Local sensor data              Local sensor data              Local sensor data
       ↓                               ↓                               ↓
  Local PCA transform           Local PCA transform           Local PCA transform
       ↓                               ↓                               ↓
  Local model training          Local model training          Local model training
       ↓                               ↓                               ↓
  Encrypted gradients  ────→  IntelliClave Aggregator  ←────  Encrypted gradients
                                       ↓
                              FedAvg → Global Model
                                       ↓
                         Distributed back to all clients
```

---

### Non-IID Heterogeneity

The three companies have genuinely different data distributions (non-IID), which is the
realistic federated learning challenge IntelliClave is designed to handle:

| Client     | WALK%  | SIT%   | STAND% | LAY%   |
|------------|--------|--------|--------|--------|
| Company A  | 19.2%  | 15.9%  | 17.2%  | 17.3%  |
| Company B  | 16.1%  | 17.4%  | 19.1%  | 19.2%  |
| Company C  | 15.3%  | 18.3%  | 19.1%  | 19.8%  |
