"""
crypto/certs/crypto_context.py

CryptoContext — manages the server RSA keypair lifecycle for IntelliClave FL.

The server creates one CryptoContext at startup and shares its public key
with all clients. Clients use the public key to encrypt weights before
returning them; the server uses the private key to decrypt before aggregation.

Key files (written to crypto/certs/keys/ by default):
    server_private.pem  — server private key  (never leaves the server)
    server_public.pem   — server public key   (distributed to clients)
"""

import os
import sys

# resolve project root so this works from any cwd
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

from crypto_layer import (  # noqa: E402
    generate_rsa_keypair,
    serialize_private_key,
    serialize_public_key,
    load_private_key,
    load_public_key,
)

_DEFAULT_KEY_DIR = os.path.join(_HERE, "keys")


class CryptoContext:
    """
    Holds the server RSA keypair and exposes helpers used by fl_server / fl_client.

    Usage — server side:
        ctx = CryptoContext.load_or_create()
        public_pem = ctx.public_pem   # send/share with clients

    Usage — client side:
        ctx = CryptoContext.from_public_pem(public_pem)
        # ctx.encrypt_weights(weights) → payload dict
        # ctx.decrypt_weights(payload) → raises — clients don't hold private key
    """

    def __init__(self, private_key=None, public_key=None):
        self._private_key = private_key
        self._public_key  = public_key

    # ── constructors ──────────────────────────────────────────────────────────

    @classmethod
    def load_or_create(cls, key_dir: str = _DEFAULT_KEY_DIR) -> "CryptoContext":
        """
        Load existing keypair from disk, or generate and save a new one.
        Called once at server startup.
        """
        os.makedirs(key_dir, exist_ok=True)
        priv_path = os.path.join(key_dir, "server_private.pem")
        pub_path  = os.path.join(key_dir, "server_public.pem")

        if os.path.exists(priv_path) and os.path.exists(pub_path):
            with open(priv_path, "rb") as f:
                private_key = load_private_key(f.read())
            with open(pub_path, "rb") as f:
                public_key = load_public_key(f.read())
            print("[CryptoContext] Loaded existing keypair from", key_dir)
        else:
            private_key, public_key = generate_rsa_keypair()
            with open(priv_path, "wb") as f:
                f.write(serialize_private_key(private_key))
            with open(pub_path, "wb") as f:
                f.write(serialize_public_key(public_key))
            print("[CryptoContext] Generated new RSA-2048 keypair →", key_dir)

        return cls(private_key=private_key, public_key=public_key)

    @classmethod
    def from_public_pem(cls, pem_bytes: bytes) -> "CryptoContext":
        """
        Create a client-side context that only holds the server public key.
        Clients can encrypt but not decrypt.
        """
        return cls(public_key=load_public_key(pem_bytes))

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def public_pem(self) -> bytes:
        return serialize_public_key(self._public_key)

    # ── encrypt / decrypt (thin wrappers around crypto_layer) ─────────────────

    def encrypt_weights(self, weights: list) -> dict:
        """Encrypt weight arrays with the server public key."""
        from crypto_layer import encrypt_weights  # noqa: E402
        return encrypt_weights(weights, self._public_key)

    def decrypt_weights(self, payload: dict) -> list:
        """Decrypt a weight payload with the server private key."""
        if self._private_key is None:
            raise RuntimeError("No private key — this context is client-side only.")
        from crypto_layer import decrypt_weights  # noqa: E402
        return decrypt_weights(payload, self._private_key)
