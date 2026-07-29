# IntelliClave Bug Fixes — Commit Log

This document details all changes made during the code audit and fix session. You can use these as reference commits or copy the commit messages directly.

---

## **Commit 1: Fix DP Training Batch Dropping for Opacus Compatibility**

**Type:** Bug Fix  
**Severity:** High  
**Category:** Differential Privacy

### Summary
When Opacus DP training is enabled, the batch size argument was parsed but never passed through the call chain. Additionally, `drop_last=True` was not set on DataLoaders, which is **required by Opacus** for accurate privacy accounting. This caused training to fail with Opacus DP enabled.

### Changes

**File: `fl/fl_client.py`**
- Added `batch_size: int = 32` parameter to `IntelliClaveClient.__init__()` (line 44)
- Store batch_size as instance variable (line 73)
- Pass `drop_last_for_dp=use_dp` to `load_csv_data()` call (line 76)
- Updated `start_client()` signature to accept `batch_size` parameter (line 249)
- Pass batch_size through to client instantiation (line 287)

**File: `fl/train_local.py`**
- Added `drop_last_for_dp=use_dp` to `load_csv_data()` call in `train_local()` (line 98)
- This ensures local training respects Opacus batch dropping requirements

**File: `fl/run_client.py`**
- Added `batch_size=args.batch_size` to `start_client()` call (line 87)
- Now properly passes the `--batch-size` CLI argument through the entire chain

### Root Cause
- `--batch-size` CLI arg was parsed by argparse but never used
- `drop_last_for_dp` parameter wasn't passed when Opacus enabled
- Opacus requires `drop_last=True` during DataLoader initialization for privacy budget calculation

### Testing
- Local training with `--dp` flag should now work correctly
- Distributed training with `--dp` should respect batch size limits

### Related Issues
- Opacus PrivacyEngine.make_private_with_epsilon() requires drop_last=True on DataLoader

---

## **Commit 2: Implement Global Normalization Coordination & Metadata Persistence**

**Type:** Bug Fix  
**Severity:** High  
**Category:** Preprocessing & Data Normalization

### Summary
Global normalization statistics were computed but never saved or used, causing train/inference mismatch in federated learning. Preprocessing metadata was lost after training, breaking the `/predict` endpoint. Fixes ensure all clients use identical feature scaling and metadata persists with checkpoints.

### Changes

**File: `fl/fl_server.py`**
1. Modified `SaveModelStrategy.__init__()`:
   - Added `preprocessing_metadata: Optional[Dict] = None` parameter (line 131)
   - Store as `self._preprocessing_metadata = preprocessing_metadata or {}` (line 149)

2. Enhanced `_save_pth()` method:
   - Added call to `save_preprocessing_metadata()` after model checkpoint (lines 305-313)
   - Saves mean, std, feature names alongside model weights
   - Ensures inference uses identical normalization as training

3. Updated `build_strategy()` function:
   - Added `preprocessing_metadata: Optional[Dict] = None` parameter (line 504)
   - Pass to `common_kwargs` dict (line 523)
   - Updated docstring (line 514)

4. Enhanced `start_server()` function:
   - Compute global mean/std from all client CSVs (lines 599-601)
   - Package into metadata dict with feature names (lines 620-631)
   - Pass to `build_strategy()` initialization (line 648)

**File: `fl/train_local.py`**
- Added preprocessing metadata saving after model checkpoint (lines 181-188)
- Uses metadata extracted from `load_csv_data()` to ensure consistency
- Works for standalone local training without FL

### Root Cause
- `coordinate_global_normalization()` existed but was never called
- `SaveModelStrategy` didn't preserve normalization statistics
- Dashboard `/predict` endpoint expected preprocessing.json but it was never created

### Impact
- ✅ All clients now use coordinated global normalization
- ✅ Preprocessing metadata saved alongside checkpoints
- ✅ Dashboard inference uses same scaling as training
- ✅ Works for both FL (server) and local training modes

### Testing
- Train with FL simulation: `python fl/run_fl_simulation.py`
- Verify `preprocessing.json` created in results directory
- Run dashboard prediction: should use correct normalization

---

## **Commit 3: Fix Class Label Inference to Use Correct Column Names**

**Type:** Bug Fix  
**Severity:** Medium  
**Category:** Data Processing

### Summary
The code hardcoded `label_column="Activity"` (UCI HAR-specific) instead of using the standard `target_col="label"` used by all client CSVs. This caused feature name extraction to fail, breaking preprocessing metadata generation.

### Changes

**File: `fl/fl_server.py`**
- Line 629: Changed `label_column="Activity"` → `target_col="label"`
- Now uses correct parameter name matching the actual CSV schema
- Ensures feature names extracted from correct CSV columns

### Root Cause
- Dataset-specific column name was hardcoded
- All client CSVs use "label" column, not "Activity"
- Feature names were extracted from wrong section of CSV

### Impact
- ✅ Works with any dataset that uses standard "label" column
- ✅ Preprocessing metadata now includes correct feature names
- ✅ Class label inference works for both numeric and string labels

### Testing
- Verify feature names in preprocessing.json match actual CSV columns
- Test with different datasets using "label" column name

---

## **Commit 4: Fix Docker Crypto Key Mounting for Auto-Generation on Startup**

**Type:** Bug Fix  
**Severity:** High  
**Category:** Cryptography & Deployment

### Summary
The crypto key directory was mounted as read-only (`:ro`), preventing the server from generating RSA keys on first startup. This broke the auto-key-generation feature needed for development. Production deployment guidance was missing.

### Changes

**File: `docker/docker-compose.yml`**
- Line 36: Removed `:ro` suffix from server crypto mount
  - Before: `../crypto/certs/keys:/app/crypto/certs/keys:ro`
  - After: `../crypto/certs/keys:/app/crypto/certs/keys`
- Client mounts remain read-only (correct)

**File: `docker/generate_compose.py`**
- Line 44: Removed `:ro` from generated server crypto mounts
- Ensures all multi-client deployments support key generation

**File: `crypto/certs/crypto_context.py`**
- Enhanced `load_or_create()` docstring (lines 60-62)
- Documents Docker mount behavior and production setup
- References new CRYPTO_SETUP.md guide

**File: `README.md`**
- Added reference to `docker/CRYPTO_SETUP.md` in Docker section (line 274)
- Documents auto-key-generation process

**File: `docker/CRYPTO_SETUP.md` (NEW)**
- Development setup: Auto-generation in RW mounts
- Production setup: Docker secrets + Kubernetes secrets
- Security best practices & key rotation
- Troubleshooting & error resolution

### Root Cause
- Read-only mount (`-ro`) prevented write operations
- `CryptoContext.load_or_create()` writes keys if they don't exist (lines 72-75)
- Permission denied error on first run with `--crypto` flag

### Impact
- ✅ RSA keys auto-generated on first Docker startup
- ✅ Development workflow no longer requires pre-generated keys
- ✅ Production deployment guide provided for Docker secrets + Kubernetes
- ✅ Security documentation for key management

### Testing
- Run `docker compose up --build` without pre-generating keys
- Keys should be created in `crypto/certs/keys/` directory
- Verify both `server_private.pem` and `server_public.pem` exist
- Encryption should work in FL training with `--crypto` flag

### Security Notes
- Development: RW mount allows auto-generation (convenient for testing)
- Production: Use Docker secrets or Kubernetes secrets (secure)
- Client mounts remain read-only (cannot leak private key)

---

## **Commit Format for Git**

If using `git commit` directly, use this format:

```bash
# Commit 1
git add fl/fl_client.py fl/train_local.py fl/run_client.py
git commit -m "Fix DP Training batch dropping for Opacus compatibility

- Added batch_size parameter to IntelliClaveClient and start_client()
- Pass batch_size=args.batch_size through call chain in run_client.py
- Set drop_last_for_dp=use_dp for Opacus DataLoader compatibility

Opacus PrivacyEngine requires drop_last=True on DataLoaders for accurate
privacy budget accounting. The --batch-size CLI argument was parsed but
never used in the downstream calls." \
--trailer="Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

# Commit 2
git add fl/fl_server.py fl/train_local.py
git commit -m "Implement global normalization coordination & metadata persistence

- Add preprocessing_metadata to SaveModelStrategy.__init__()
- Save normalization stats in _save_pth() alongside model checkpoint
- Compute global mean/std from all client CSVs in start_server()
- Package metadata with feature names for inference
- Save preprocessing.json for standalone local training

This ensures all FL clients use identical feature scaling (global normalization)
and metadata persists with checkpoints so the dashboard /predict endpoint
can load the same normalization used during training." \
--trailer="Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

# Commit 3
git add fl/fl_server.py
git commit -m "Fix class label inference to use correct column names

- Change hardcoded label_column='Activity' to target_col='label'
- Now works with any dataset using standard 'label' column
- Ensures feature names extracted from correct CSV columns

The Activity column was UCI HAR-specific. All client CSVs use 'label'
column by default, so this fixes feature name extraction in preprocessing." \
--trailer="Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"

# Commit 4
git add docker/docker-compose.yml docker/generate_compose.py crypto/certs/crypto_context.py docker/CRYPTO_SETUP.md README.md
git commit -m "Fix Docker crypto key mounting & add production deployment guide

- Remove :ro from server crypto mount in docker-compose.yml
- Remove :ro from generated compose files in generate_compose.py
- Update crypto_context.py docstring with Docker setup guidance
- Add comprehensive docker/CRYPTO_SETUP.md guide

Development: Keys auto-generate on first startup (RW mount)
Production: Use Docker secrets or Kubernetes secrets (secure)

CryptoContext.load_or_create() writes RSA keys if they don't exist,
which requires a read-write mount. Read-only mount prevented key
generation on first run with --crypto flag." \
--trailer="Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
```

---

## **How to Apply These Commits**

### Option A: Copy/Paste Approach
```bash
cd IntelliClave

# Stage files for Commit 1
git add fl/fl_client.py fl/train_local.py fl/run_client.py
git commit -m "Fix DP Training batch dropping for Opacus compatibility

- Added batch_size parameter to IntelliClaveClient and start_client()
- Pass batch_size=args.batch_size through call chain in run_client.py
- Set drop_last_for_dp=use_dp for Opacus DataLoader compatibility"

# Repeat for Commits 2, 3, and 4...
```

### Option B: Single Aggregated Commit
```bash
git add .
git commit -m "Fix 4 critical bugs: DP training, preprocessing, class labels, crypto mounts

Includes:
1. DP Training: Fix batch dropping for Opacus compatibility
2. Preprocessing: Implement global normalization & metadata persistence
3. Class Labels: Fix hardcoded 'Activity' column reference
4. Docker Crypto: Fix read-only mount, add production guide"
```

---

## **File Summary**

| File | Commits | Changes | Lines Modified |
|------|---------|---------|-----------------|
| `fl/fl_client.py` | Commit 1 | 3 edits | +10 |
| `fl/train_local.py` | Commit 1, 2 | 2 edits | +18 |
| `fl/run_client.py` | Commit 1 | 1 edit | +1 |
| `fl/fl_server.py` | Commit 2, 3 | 7 edits | +45 |
| `docker/docker-compose.yml` | Commit 4 | 1 edit | -1 |
| `docker/generate_compose.py` | Commit 4 | 1 edit | -1 |
| `crypto/certs/crypto_context.py` | Commit 4 | 1 edit | +3 |
| `docker/CRYPTO_SETUP.md` | Commit 4 | NEW | +267 |
| `README.md` | Commit 4 | 1 edit | +2 |
| **TOTAL** | **4 commits** | **18 edits** | **~344** |

---

## **Verification Checklist**

After applying commits, verify:

- [ ] All 4 bugs fixed and committed
- [ ] No merge conflicts
- [ ] `git log --oneline` shows 4 new commits
- [ ] Each commit message is descriptive and actionable
- [ ] Files compile without syntax errors
- [ ] Tests pass (if applicable)
- [ ] Documentation is updated (README, CRYPTO_SETUP.md)

---

## **Next Steps**

After these commits are merged:

1. **Test the fixes:**
   ```bash
   # Test DP training
   python fl/run_fl_simulation.py --rounds 5 --dp --epsilon 10.0
   
   # Test Docker deployment
   docker compose -f docker/docker-compose.yml up --build
   
   # Test dashboard with preprocessing
   python fl/run_fl_simulation.py && cd dashboard/backend && uvicorn main:app
   ```

2. **Remaining bugs to fix:**
   - Bug #5: API authentication (optional, previously undone)
   - Bug #6: Rate limiting (Redis-backed solution)
   - Bug #7: TLS certificate validation

3. **Code review:**
   - Verify no regressions in existing functionality
   - Check that all changes align with existing code style
   - Ensure documentation is complete

---

**Generated:** 2026-05-24 09:23 UTC  
**Session:** IntelliClave Bug Audit & Fix  
**Total Fixes:** 4 critical bugs  
**Files Changed:** 9 files modified/created  
**Status:** Ready for commit
