"""
crypto/certs/crypto_layer.py

Crypto layer for IntelliClave FL communication.
Encrypts model weights/gradients before they leave a client,
decrypts them on the server side after aggregation.

Scheme: AES-256-GCM (symmetric, per-session key) with RSA-2048 key encapsulation.
  - Each FL round generates a fresh AES session key.
  - The session key is RSA-encrypted with the server's public key.
  - Model weights (numpy arrays) are serialised → encrypted with AES-GCM.
  - Server decrypts the session key with its RSA private key, then decrypts weights.

Why AES-GCM?
  - Authenticated encryption: integrity + confidentiality in one pass.
  - Fast enough for the weight tensors we're sending (~63 KB per round).
  - GCM tag catches any tampering in transit (STRIDE: Tampering mitigation).

Why RSA for key exchange?
  - Asymmetric: clients only need the server's public key (no shared secret).
  - 2048-bit gives adequate security for a research prototype.
  - In production this would be replaced by a proper PKI / TLS mutual auth.

Usage (standalone demo):
    python crypto/certs/crypto_layer.py
"""

import os
import io
import json
import base64
import hashlib

import numpy as np

from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


# ── Key generation ─────────────────────────────────────────────────────────────

def generate_rsa_keypair(key_size: int = 2048):
    """
    Generate an RSA keypair.
    Returns (private_key, public_key) as cryptography objects.
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=key_size,
    )
    return private_key, private_key.public_key()


def serialize_public_key(public_key) -> bytes:
    """PEM-encode a public key for storage or transmission."""
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def load_public_key(pem_bytes: bytes):
    """Load a PEM-encoded public key."""
    return serialization.load_pem_public_key(pem_bytes)


def serialize_private_key(private_key, password: bytes = None) -> bytes:
    """PEM-encode a private key (optionally password-protected)."""
    encryption = (
        serialization.BestAvailableEncryption(password)
        if password
        else serialization.NoEncryption()
    )
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=encryption,
    )


def load_private_key(pem_bytes: bytes, password: bytes = None):
    """Load a PEM-encoded private key."""
    return serialization.load_pem_private_key(pem_bytes, password=password)


# ── Weight serialisation ───────────────────────────────────────────────────────

def weights_to_bytes(weights: list) -> bytes:
    """
    Serialise a list of numpy arrays (FL model weights) to bytes.
    Uses numpy's npz format — same as what Flower uses internally.
    """
    buf = io.BytesIO()
    np.savez(buf, *weights)
    return buf.getvalue()


def bytes_to_weights(data: bytes) -> list:
    """Deserialise bytes back to a list of numpy arrays."""
    buf = io.BytesIO(data)
    npz = np.load(buf, allow_pickle=False)
    return [npz[key] for key in sorted(npz.files)]


# ── AES-GCM encryption ─────────────────────────────────────────────────────────

def aes_encrypt(plaintext: bytes) -> tuple:
    """
    Encrypt plaintext with a fresh AES-256-GCM session key.

    Returns:
        session_key : bytes  (32 bytes, random)
        nonce       : bytes  (12 bytes, random — GCM standard)
        ciphertext  : bytes  (plaintext + 16-byte GCM auth tag appended)
    """
    session_key = os.urandom(32)   # AES-256
    nonce = os.urandom(12)         # GCM nonce
    aesgcm = AESGCM(session_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)
    return session_key, nonce, ciphertext


def aes_decrypt(session_key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    """
    Decrypt AES-256-GCM ciphertext.
    Raises cryptography.exceptions.InvalidTag if the ciphertext was tampered with.
    """
    aesgcm = AESGCM(session_key)
    return aesgcm.decrypt(nonce, ciphertext, associated_data=None)


# ── RSA key encapsulation ──────────────────────────────────────────────────────

def rsa_encrypt_key(session_key: bytes, server_public_key) -> bytes:
    """
    Encrypt the AES session key with the server's RSA public key.
    Uses OAEP + SHA-256 — the recommended padding for RSA encryption.
    """
    return server_public_key.encrypt(
        session_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


def rsa_decrypt_key(encrypted_key: bytes, server_private_key) -> bytes:
    """Decrypt the AES session key with the server's RSA private key."""
    return server_private_key.decrypt(
        encrypted_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )


# ── High-level FL weight encryption / decryption ──────────────────────────────

def encrypt_weights(weights: list, server_public_key) -> dict:
    """
    Encrypt a list of numpy weight arrays for transmission to the FL server.

    Flow:
        weights → npz bytes → AES-GCM encrypt → RSA-wrap session key

    Returns a dict (JSON-serialisable via base64) with:
        encrypted_key  : RSA-encrypted AES session key (b64)
        nonce          : AES-GCM nonce (b64)
        ciphertext     : encrypted weight bytes (b64)
        weight_hash    : SHA-256 of plaintext bytes (hex) — integrity check
    """
    plaintext = weights_to_bytes(weights)
    weight_hash = hashlib.sha256(plaintext).hexdigest()

    session_key, nonce, ciphertext = aes_encrypt(plaintext)
    encrypted_key = rsa_encrypt_key(session_key, server_public_key)

    return {
        "encrypted_key": base64.b64encode(encrypted_key).decode(),
        "nonce":         base64.b64encode(nonce).decode(),
        "ciphertext":    base64.b64encode(ciphertext).decode(),
        "weight_hash":   weight_hash,
    }


def decrypt_weights(payload: dict, server_private_key) -> list:
    """
    Decrypt a weight payload produced by encrypt_weights().

    Verifies the SHA-256 hash after decryption to confirm integrity.
    Raises ValueError if the hash does not match.
    Raises cryptography.exceptions.InvalidTag if GCM auth tag fails (tampering).

    Returns a list of numpy arrays.
    """
    encrypted_key = base64.b64decode(payload["encrypted_key"])
    nonce         = base64.b64decode(payload["nonce"])
    ciphertext    = base64.b64decode(payload["ciphertext"])
    expected_hash = payload["weight_hash"]

    session_key = rsa_decrypt_key(encrypted_key, server_private_key)
    plaintext   = aes_decrypt(session_key, nonce, ciphertext)

    actual_hash = hashlib.sha256(plaintext).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"Weight integrity check failed: "
            f"expected {expected_hash}, got {actual_hash}"
        )

    return bytes_to_weights(plaintext)


# ── Payload serialisation (for wire transport) ─────────────────────────────────

def payload_to_json(payload: dict) -> str:
    """Serialise an encrypted payload to a JSON string."""
    return json.dumps(payload)


def payload_from_json(json_str: str) -> dict:
    """Deserialise an encrypted payload from a JSON string."""
    return json.loads(json_str)


# ── Integrity / fingerprint helpers ───────────────────────────────────────────

def fingerprint_weights(weights: list) -> str:
    """
    Compute a SHA-256 fingerprint of model weights.
    Useful for verifying that aggregated weights haven't been tampered with.
    """
    return hashlib.sha256(weights_to_bytes(weights)).hexdigest()


def verify_weights_fingerprint(weights: list, expected_hash: str) -> bool:
    """Return True if the weights match the expected SHA-256 hash."""
    return fingerprint_weights(weights) == expected_hash


# ── Standalone demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("IntelliClave Crypto Layer — Demo")
    print("=" * 55)

    # 1. Generate server keypair (done once at server startup)
    print("\n[1] Generating RSA-2048 server keypair...")
    private_key, public_key = generate_rsa_keypair()
    print("    ✅ Keypair generated")

    # 2. Simulate model weights (same shape as HARClassifier)
    print("\n[2] Simulating HARClassifier weight arrays...")
    dummy_weights = [
        np.random.randn(128, 50).astype(np.float32),   # fc1 weight
        np.random.randn(128).astype(np.float32),        # fc1 bias
        np.random.randn(64, 128).astype(np.float32),    # fc2 weight
        np.random.randn(64).astype(np.float32),         # fc2 bias
        np.random.randn(6, 64).astype(np.float32),      # output weight
        np.random.randn(6).astype(np.float32),          # output bias
    ]
    total_params = sum(w.size for w in dummy_weights)
    print(f"    Arrays: {len(dummy_weights)}, Total params: {total_params:,}")

    # 3. Client encrypts weights before sending
    print("\n[3] Client encrypting weights...")
    payload = encrypt_weights(dummy_weights, public_key)
    payload_json = payload_to_json(payload)
    print(f"    ✅ Encrypted payload size: {len(payload_json):,} bytes")
    print(f"    Weight hash (plaintext): {payload['weight_hash'][:16]}...")

    # 4. Server decrypts weights after receiving
    print("\n[4] Server decrypting weights...")
    received_payload = payload_from_json(payload_json)
    recovered_weights = decrypt_weights(received_payload, private_key)
    print(f"    ✅ Decrypted {len(recovered_weights)} weight arrays")

    # 5. Verify round-trip integrity
    print("\n[5] Verifying round-trip integrity...")
    for i, (orig, recv) in enumerate(zip(dummy_weights, recovered_weights)):
        assert np.allclose(orig, recv), f"Mismatch in array {i}"
    print("    ✅ All weight arrays match exactly")

    # 6. Verify fingerprint helper
    fp = fingerprint_weights(dummy_weights)
    assert verify_weights_fingerprint(dummy_weights, fp)
    print(f"    ✅ Fingerprint verified: {fp[:16]}...")

    # 7. Tamper detection demo
    print("\n[6] Tamper detection demo...")
    tampered = dict(received_payload)
    raw = base64.b64decode(tampered["ciphertext"])
    raw_list = bytearray(raw)
    raw_list[10] ^= 0xFF   # flip bits in ciphertext
    tampered["ciphertext"] = base64.b64encode(bytes(raw_list)).decode()
    try:
        decrypt_weights(tampered, private_key)
        print("    ❌ Tamper NOT detected (unexpected)")
    except Exception as e:
        print(f"    ✅ Tamper detected: {type(e).__name__}")

    print("\n" + "=" * 55)
    print("Crypto layer demo complete ✅")
    print("=" * 55)
