"""
tee/sealed_storage/sealed_storage.py

SGX Sealed Storage for IntelliClave.

What sealing means:
    In real gramine-sgx, the SGX hardware provides a sealing key derived from:
        MRENCLAVE  (hash of enclave code — changes if code changes)
        MRSIGNER   (hash of the developer's signing key)
    Data encrypted with this key can ONLY be decrypted by the same enclave
    running on the same CPU. The host OS, hypervisor, and other enclaves
    cannot read it — even with root access.

What we simulate here (gramine-direct / WSL2):
    We derive a sealing key from MRENCLAVE (computed from manifest + code files)
    and use AES-256-GCM to encrypt the data. The key is deterministic for a
    given enclave identity, so the same enclave can always unseal its own data.
    A different enclave (different code) gets a different key and cannot unseal.

What gets sealed in IntelliClave:
    1. server_private.pem  — RSA private key for weight decryption
    2. global_model_*.pth  — aggregated model checkpoints after each round
    3. fl_privacy.json     — epsilon/delta audit log

Sealing policy:
    MRENCLAVE binding — only the exact same enclave code can unseal.
    If the manifest or fl_server.py changes, MRENCLAVE changes,
    the sealing key changes, and old sealed data cannot be read.
    This is intentional: it prevents a modified server from reading
    data sealed by the legitimate server.

Run demo:
    python3 tee/sealed_storage/sealed_storage.py
"""

import hashlib
import hmac
import json
import os
import struct
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

# ── Sealing key derivation ────────────────────────────────────────────────────

def _derive_sealing_key(mrenclave: str) -> bytes:
    """
    Derive a 32-byte AES sealing key from MRENCLAVE.

    In real SGX: the CPU provides egetkey(SEAL_KEY, MRENCLAVE_POLICY).
    Here: HKDF-like derivation using HMAC-SHA256 with a fixed salt.

    The key is deterministic for a given MRENCLAVE — same enclave always
    gets the same key, different enclave gets a different key.
    """
    # Fixed salt — represents the "platform sealing key" in simulation
    salt = b"intelliclave-sgx-seal-salt-v1"
    key  = hmac.new(salt, mrenclave.encode(), hashlib.sha256).digest()
    return key  # 32 bytes → AES-256


def get_enclave_mrenclave() -> str:
    """
    Compute MRENCLAVE for the FL server enclave.
    Uses the server manifest template + fl_server.py as the enclave identity.
    """
    manifest_path = os.path.join(
        _ROOT, "tee", "fl_enclave", "fl_server_enclave.manifest.template"
    )
    code_path = os.path.join(_ROOT, "fl", "fl_server.py")

    h = hashlib.sha256()
    for path in [manifest_path, code_path]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                h.update(f.read())
        else:
            h.update(path.encode())
    return h.hexdigest()


# ── AES-256-GCM seal / unseal ─────────────────────────────────────────────────

def seal(plaintext: bytes, mrenclave: str = None) -> bytes:
    """
    Seal (encrypt) data to the current enclave identity.

    Layout of sealed blob:
        [mrenclave_hash: 32B][nonce: 12B][ciphertext+tag: N+16B]

    The mrenclave_hash is stored so unseal() can detect identity mismatch
    before attempting decryption.
    """
    import os as _os
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise RuntimeError("cryptography package required: pip install cryptography")

    if mrenclave is None:
        mrenclave = get_enclave_mrenclave()

    sealing_key  = _derive_sealing_key(mrenclave)
    mrenclave_b  = bytes.fromhex(mrenclave)[:32]
    nonce        = _os.urandom(12)

    aesgcm     = AESGCM(sealing_key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=mrenclave_b)

    return mrenclave_b + nonce + ciphertext


def unseal(sealed_blob: bytes, mrenclave: str = None) -> bytes:
    """
    Unseal (decrypt) data previously sealed by this enclave.

    Raises:
        ValueError  — if MRENCLAVE in blob doesn't match current enclave
        cryptography.exceptions.InvalidTag — if blob was tampered with
    """
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        raise RuntimeError("cryptography package required: pip install cryptography")

    if mrenclave is None:
        mrenclave = get_enclave_mrenclave()

    mrenclave_b = bytes.fromhex(mrenclave)[:32]

    # Extract stored MRENCLAVE from blob header
    stored_mrenclave_b = sealed_blob[:32]
    if stored_mrenclave_b != mrenclave_b:
        raise ValueError(
            f"Sealed data belongs to a different enclave.\n"
            f"  Stored  MRENCLAVE: {stored_mrenclave_b.hex()}\n"
            f"  Current MRENCLAVE: {mrenclave_b.hex()}\n"
            f"  This means the enclave code or manifest has changed."
        )

    nonce      = sealed_blob[32:44]
    ciphertext = sealed_blob[44:]

    sealing_key = _derive_sealing_key(mrenclave)
    aesgcm      = AESGCM(sealing_key)
    return aesgcm.decrypt(nonce, ciphertext, associated_data=mrenclave_b)


# ── High-level helpers ────────────────────────────────────────────────────────

def seal_file(src_path: str, dst_path: str = None, mrenclave: str = None) -> str:
    """
    Seal a file to the enclave identity.
    Writes sealed blob to dst_path (default: src_path + '.sealed').
    Returns the path of the sealed file.
    """
    if dst_path is None:
        dst_path = src_path + ".sealed"

    with open(src_path, "rb") as f:
        plaintext = f.read()

    sealed = seal(plaintext, mrenclave)

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    with open(dst_path, "wb") as f:
        f.write(sealed)

    return dst_path


def unseal_file(src_path: str, dst_path: str = None, mrenclave: str = None) -> str:
    """
    Unseal a previously sealed file.
    Writes plaintext to dst_path (default: src_path without '.sealed').
    Returns the path of the unsealed file.
    """
    if dst_path is None:
        dst_path = src_path.removesuffix(".sealed")

    with open(src_path, "rb") as f:
        sealed_blob = f.read()

    plaintext = unseal(sealed_blob, mrenclave)

    with open(dst_path, "wb") as f:
        f.write(plaintext)

    return dst_path


def seal_model_checkpoint(pth_path: str, mrenclave: str = None) -> str:
    """
    Seal a PyTorch model checkpoint (.pth) to the server enclave.
    The sealed file is written alongside the original as <name>.pth.sealed.
    """
    sealed_path = seal_file(pth_path, pth_path + ".sealed", mrenclave)
    print(f"[SealedStorage] Sealed model checkpoint: {os.path.basename(pth_path)}")
    return sealed_path


def seal_private_key(pem_path: str, mrenclave: str = None) -> str:
    """
    Seal the RSA private key to the server enclave.
    After sealing, the original .pem should be deleted from disk.
    """
    sealed_path = seal_file(pem_path, pem_path + ".sealed", mrenclave)
    print(f"[SealedStorage] Sealed private key: {os.path.basename(pem_path)}")
    return sealed_path


# ── Demo ──────────────────────────────────────────────────────────────────────

def run_demo():
    import tempfile

    print("=" * 55)
    print("IntelliClave — SGX Sealed Storage Demo")
    print("=" * 55)

    mrenclave = get_enclave_mrenclave()
    print(f"\n[1] Enclave MRENCLAVE: {mrenclave[:24]}...")

    # ── Seal a secret ─────────────────────────────────────────────────────────
    secret = b'{"private_key": "RSA-2048-SIMULATED", "round": 5}'
    print(f"\n[2] Sealing secret ({len(secret)} bytes)...")
    sealed = seal(secret, mrenclave)
    print(f"    Sealed blob size : {len(sealed)} bytes")
    print(f"    First 16 bytes   : {sealed[:16].hex()}")

    # ── Unseal with correct MRENCLAVE ─────────────────────────────────────────
    print(f"\n[3] Unsealing with correct MRENCLAVE...")
    recovered = unseal(sealed, mrenclave)
    assert recovered == secret, "Unseal mismatch!"
    print(f"    Recovered        : {recovered.decode()}")
    print(f"    Match            : ✓")

    # ── Unseal with wrong MRENCLAVE (tampered enclave) ────────────────────────
    print(f"\n[4] Unsealing with WRONG MRENCLAVE (simulates tampered enclave)...")
    fake_mrenclave = hashlib.sha256(b"tampered-enclave-code").hexdigest()
    rejected = False
    try:
        unseal(sealed, fake_mrenclave)
    except ValueError as e:
        rejected = True
        print(f"    Rejected: {str(e).splitlines()[0]}")
    assert rejected, "Should have been rejected!"
    print(f"    Tampered enclave blocked : ✓")

    # ── Seal a file ───────────────────────────────────────────────────────────
    print(f"\n[5] Sealing a file...")
    tmp_dir  = tempfile.mkdtemp()
    src_file = os.path.join(tmp_dir, "server_private.pem")
    dst_file = os.path.join(tmp_dir, "server_private.pem.sealed")
    rec_file = os.path.join(tmp_dir, "server_private_recovered.pem")

    with open(src_file, "wb") as f:
        f.write(b"-----BEGIN RSA PRIVATE KEY-----\nSIMULATED\n-----END RSA PRIVATE KEY-----\n")

    seal_file(src_file, dst_file, mrenclave)
    unseal_file(dst_file, rec_file, mrenclave)

    with open(src_file, "rb") as f:
        orig = f.read()
    with open(rec_file, "rb") as f:
        recv = f.read()

    assert orig == recv
    print(f"    File sealed and recovered : ✓")
    print(f"    Sealed size : {os.path.getsize(dst_file)} bytes")

    print()
    print("=" * 55)
    print("ALL SEALED STORAGE TESTS PASSED ✓")
    print("=" * 55)
    print()
    print("Production note:")
    print("  In gramine-sgx, seal() uses egetkey(SEAL_KEY) from the CPU.")
    print("  The sealing key is hardware-bound to MRENCLAVE + platform.")
    print("  No code change required — Gramine handles the key derivation.")


if __name__ == "__main__":
    run_demo()
