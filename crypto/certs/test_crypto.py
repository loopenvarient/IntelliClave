"""
crypto/certs/test_crypto.py

4 crypto tests — all must pass.

Tests:
  1. Encrypt then decrypt → same array back
  2. HMAC on valid data   → accepted
  3. HMAC on tampered data → rejected
  4. Same plaintext twice  → different ciphertext (IND-CPA)

Run:
    python3 crypto/certs/test_crypto.py
"""

import base64
import hashlib
import hmac
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from crypto_layer import (
    generate_rsa_keypair,
    encrypt_weights,
    decrypt_weights,
    aes_encrypt,
    aes_decrypt,
    weights_to_bytes,
    fingerprint_weights,
    verify_weights_fingerprint,
)

PASS = "  ✓ PASS"
FAIL = "  ✗ FAIL"

results = []

print("=" * 55)
print("IntelliClave — Crypto Layer Tests")
print("=" * 55)

# ── Shared fixtures ───────────────────────────────────────────────────────────
private_key, public_key = generate_rsa_keypair()

# Simulate HARClassifier weight arrays
dummy_weights = [
    np.random.randn(128, 50).astype(np.float32),   # fc1 weight
    np.random.randn(128).astype(np.float32),        # fc1 bias
    np.random.randn(64, 128).astype(np.float32),    # fc2 weight
    np.random.randn(64).astype(np.float32),         # fc2 bias
    np.random.randn(6, 64).astype(np.float32),      # output weight
    np.random.randn(6).astype(np.float32),          # output bias
]

# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Encrypt → Decrypt → same arrays back
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Test 1] Encrypt then decrypt → same array back")
try:
    payload   = encrypt_weights(dummy_weights, public_key)
    recovered = decrypt_weights(payload, private_key)

    assert len(recovered) == len(dummy_weights), "Array count mismatch"
    for i, (orig, recv) in enumerate(zip(dummy_weights, recovered)):
        assert np.allclose(orig, recv, atol=1e-6), f"Array {i} mismatch"

    print(f"  Encrypted payload size : {len(str(payload))} chars")
    print(f"  Arrays recovered       : {len(recovered)}")
    print(f"  Max abs diff           : {max(np.abs(o - r).max() for o, r in zip(dummy_weights, recovered)):.2e}")
    print(PASS)
    results.append(True)
except Exception as e:
    print(f"  Error: {e}")
    print(FAIL)
    results.append(False)

# ─────────────────────────────────────────────────────────────────────────────
# Test 2: HMAC / integrity check on valid data → accepted
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Test 2] Integrity check on valid data → accepted")
try:
    fp = fingerprint_weights(dummy_weights)
    ok = verify_weights_fingerprint(dummy_weights, fp)
    assert ok, "Fingerprint should match"

    # Also verify the weight_hash field in the encrypted payload
    payload2  = encrypt_weights(dummy_weights, public_key)
    recovered2 = decrypt_weights(payload2, private_key)  # raises if hash fails
    assert len(recovered2) == len(dummy_weights)

    print(f"  SHA-256 fingerprint    : {fp[:24]}...")
    print(f"  Fingerprint match      : {ok}")
    print(PASS)
    results.append(True)
except Exception as e:
    print(f"  Error: {e}")
    print(FAIL)
    results.append(False)

# ─────────────────────────────────────────────────────────────────────────────
# Test 3: HMAC on tampered data → rejected
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Test 3] Tampered ciphertext → rejected")
try:
    payload3 = encrypt_weights(dummy_weights, public_key)

    # Flip bits in the ciphertext
    raw = base64.b64decode(payload3["ciphertext"])
    raw_list = bytearray(raw)
    raw_list[10] ^= 0xFF
    payload3["ciphertext"] = base64.b64encode(bytes(raw_list)).decode()

    rejected = False
    try:
        decrypt_weights(payload3, private_key)
    except Exception:
        rejected = True

    assert rejected, "Tampered payload should have been rejected"
    print("  Tampered ciphertext    : bits flipped at byte 10")
    print("  Decryption rejected    : True")
    print(PASS)
    results.append(True)
except Exception as e:
    print(f"  Error: {e}")
    print(FAIL)
    results.append(False)

# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Same plaintext twice → different ciphertext (IND-CPA / fresh nonce)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Test 4] Same plaintext twice → different ciphertext (fresh nonce)")
try:
    p_a = encrypt_weights(dummy_weights, public_key)
    p_b = encrypt_weights(dummy_weights, public_key)

    ct_a = p_a["ciphertext"]
    ct_b = p_b["ciphertext"]
    nonce_a = p_a["nonce"]
    nonce_b = p_b["nonce"]

    assert ct_a != ct_b,    "Ciphertexts must differ (fresh AES key each time)"
    assert nonce_a != nonce_b, "Nonces must differ"

    print(f"  Ciphertext A (first 16): {ct_a[:16]}...")
    print(f"  Ciphertext B (first 16): {ct_b[:16]}...")
    print(f"  Ciphertexts differ     : {ct_a != ct_b}")
    print(f"  Nonces differ          : {nonce_a != nonce_b}")
    print(PASS)
    results.append(True)
except Exception as e:
    print(f"  Error: {e}")
    print(FAIL)
    results.append(False)

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 55)
passed = sum(results)
total  = len(results)
print(f"Results: {passed}/{total} tests passed")
for i, ok in enumerate(results, 1):
    print(f"  Test {i}: {'PASS' if ok else 'FAIL'}")

if passed == total:
    print()
    print("ALL CRYPTO TESTS PASSED ✓")
else:
    print()
    print("SOME TESTS FAILED ✗")
    sys.exit(1)

print("=" * 55)
