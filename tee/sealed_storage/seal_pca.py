"""
tee/sealed_storage/seal_pca.py

Seal pca_model.pkl inside the TEE using the existing sealed storage
infrastructure.

Why this matters
────────────────
The PCA model is the white-box attack path that output perturbation does
not close. An adversary with access to pca_model.pkl can:
  1. Invert the PCA transform: x_raw = pca.inverse_transform(x_pca)
  2. Map any reconstructed PCA-space vector back to the original 561-dim
     sensor feature space.
  3. Bypass output perturbation entirely by working in raw feature space.

Sealing pca_model.pkl to the server enclave (MRENCLAVE-bound AES-256-GCM)
means the file on disk is ciphertext. Only the exact server enclave can
unseal it. The host OS, other processes, and modified server code all get
a different MRENCLAVE and therefore a different sealing key — they cannot
read the plaintext PCA model.

Usage
─────
  # Seal (run once after PCA is fitted, or after any re-fit):
  python tee/sealed_storage/seal_pca.py --seal

  # Verify the sealed file can be unsealed by the current enclave:
  python tee/sealed_storage/seal_pca.py --verify

  # Unseal to a temp path and load into memory (used by inference code):
  from tee.sealed_storage.seal_pca import load_sealed_pca
  pca = load_sealed_pca()

  # Seal + delete plaintext (production hardening):
  python tee/sealed_storage/seal_pca.py --seal --delete-plaintext

Integration with inference
──────────────────────────
Replace any direct pickle.load(open("data/samples/pca_model.pkl")) call
with load_sealed_pca(). The function unseals to an in-memory buffer —
the plaintext never touches disk.
"""

import argparse
import io
import os
import pickle
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

from sealed_storage import seal, unseal, get_enclave_mrenclave  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────────────

PCA_PLAINTEXT_PATH = os.path.join(_ROOT, "data", "samples", "pca_model.pkl")
PCA_SEALED_PATH    = os.path.join(_ROOT, "data", "samples", "pca_model.pkl.sealed")


# ── Core helpers ──────────────────────────────────────────────────────────────

def seal_pca(
    src_path: str = PCA_PLAINTEXT_PATH,
    dst_path: str = PCA_SEALED_PATH,
    mrenclave: str = None,
    delete_plaintext: bool = False,
) -> str:
    """
    Seal pca_model.pkl to the current enclave identity.

    Parameters
    ----------
    src_path         : path to the plaintext .pkl file
    dst_path         : path to write the sealed blob
    mrenclave        : override MRENCLAVE (default: computed from manifest + code)
    delete_plaintext : if True, delete src_path after sealing (production mode)

    Returns
    -------
    str : path of the sealed file
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"PCA model not found at {src_path}. "
            "Run data/datascripts/pipeline.py first to generate it."
        )

    with open(src_path, "rb") as f:
        plaintext = f.read()

    if mrenclave is None:
        mrenclave = get_enclave_mrenclave()

    sealed_blob = seal(plaintext, mrenclave)

    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    with open(dst_path, "wb") as f:
        f.write(sealed_blob)

    print(f"[SealPCA] Sealed {os.path.basename(src_path)} "
          f"({len(plaintext):,} bytes plaintext → {len(sealed_blob):,} bytes sealed)")
    print(f"[SealPCA] MRENCLAVE: {mrenclave[:24]}...")
    print(f"[SealPCA] Sealed file: {dst_path}")

    if delete_plaintext:
        os.remove(src_path)
        print(f"[SealPCA] Plaintext deleted: {src_path}")
        print("[SealPCA] WARNING: plaintext is gone — only the sealed file remains. "
              "Re-seal from source if you need to regenerate.")

    return dst_path


def unseal_pca_to_bytes(
    src_path: str = PCA_SEALED_PATH,
    mrenclave: str = None,
) -> bytes:
    """
    Unseal pca_model.pkl.sealed and return the plaintext bytes.

    Raises ValueError if the sealed file was created by a different enclave.
    Raises FileNotFoundError if the sealed file does not exist.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"Sealed PCA model not found at {src_path}. "
            "Run: python tee/sealed_storage/seal_pca.py --seal"
        )

    with open(src_path, "rb") as f:
        sealed_blob = f.read()

    if mrenclave is None:
        mrenclave = get_enclave_mrenclave()

    return unseal(sealed_blob, mrenclave)


def load_sealed_pca(
    sealed_path: str = PCA_SEALED_PATH,
    plaintext_fallback: str = PCA_PLAINTEXT_PATH,
    mrenclave: str = None,
):
    """
    Load the PCA model from the sealed file into memory.

    The plaintext never touches disk — unsealing happens in-memory.

    Falls back to the plaintext .pkl if the sealed file does not exist
    (development mode). Logs a warning when falling back so it is visible
    in CI and production logs.

    Parameters
    ----------
    sealed_path       : path to the .pkl.sealed file
    plaintext_fallback: path to the plaintext .pkl (fallback only)
    mrenclave         : override MRENCLAVE

    Returns
    -------
    sklearn PCA object (or whatever was pickled)
    """
    if os.path.exists(sealed_path):
        plaintext_bytes = unseal_pca_to_bytes(sealed_path, mrenclave)
        pca = pickle.loads(plaintext_bytes)  # noqa: S301 — trusted enclave-sealed data
        print(f"[SealPCA] PCA model loaded from sealed storage ✓")
        return pca

    if os.path.exists(plaintext_fallback):
        import warnings
        warnings.warn(
            f"[SealPCA] WARNING: Loading PCA model from plaintext fallback "
            f"({plaintext_fallback}). "
            "This exposes the PCA model to white-box inversion attacks. "
            "Run 'python tee/sealed_storage/seal_pca.py --seal' to seal it.",
            UserWarning,
            stacklevel=2,
        )
        with open(plaintext_fallback, "rb") as f:
            return pickle.load(f)  # noqa: S301

    raise FileNotFoundError(
        f"PCA model not found at sealed path ({sealed_path}) "
        f"or plaintext fallback ({plaintext_fallback}). "
        "Run data/datascripts/pipeline.py to generate the PCA model, "
        "then seal it with: python tee/sealed_storage/seal_pca.py --seal"
    )


def verify_sealed_pca(
    sealed_path: str = PCA_SEALED_PATH,
    mrenclave: str = None,
) -> bool:
    """
    Verify that the sealed PCA file can be unsealed by the current enclave
    and that the result deserialises to a valid sklearn PCA object.

    Returns True on success, raises on failure.
    """
    plaintext_bytes = unseal_pca_to_bytes(sealed_path, mrenclave)
    pca = pickle.loads(plaintext_bytes)  # noqa: S301

    # Basic sanity checks
    if not hasattr(pca, "transform"):
        raise ValueError("Unsealed object does not have a .transform() method — "
                         "not a valid sklearn PCA model.")
    if not hasattr(pca, "components_"):
        raise ValueError("Unsealed object has no .components_ — "
                         "PCA was not fitted before sealing.")

    n_components = pca.n_components_
    n_features   = pca.components_.shape[1]
    print(f"[SealPCA] Verification passed ✓")
    print(f"[SealPCA]   n_components : {n_components}")
    print(f"[SealPCA]   n_features   : {n_features}")
    print(f"[SealPCA]   explained_var: "
          f"{pca.explained_variance_ratio_.sum():.4f}")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Seal / verify the PCA model inside the TEE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seal", action="store_true",
        help="Seal pca_model.pkl to the current enclave identity.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify the sealed file can be unsealed and deserialised.",
    )
    parser.add_argument(
        "--delete-plaintext", action="store_true",
        help="Delete the plaintext .pkl after sealing (production hardening).",
    )
    parser.add_argument(
        "--src", default=PCA_PLAINTEXT_PATH,
        help="Path to the plaintext pca_model.pkl.",
    )
    parser.add_argument(
        "--dst", default=PCA_SEALED_PATH,
        help="Path to write the sealed blob.",
    )
    args = parser.parse_args()

    if not args.seal and not args.verify:
        parser.print_help()
        return

    if args.seal:
        seal_pca(
            src_path=args.src,
            dst_path=args.dst,
            delete_plaintext=args.delete_plaintext,
        )

    if args.verify:
        verify_sealed_pca(sealed_path=args.dst)


if __name__ == "__main__":
    main()
