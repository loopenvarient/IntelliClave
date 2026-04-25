import hashlib
import json

def compute_mrenclave(manifest: str, code: str):
    data = (manifest + code).encode()
    return hashlib.sha256(data).hexdigest()


def generate_quote(mrenclave: str):
    quote = {
        "mrenclave": mrenclave,
        "status": "SIMULATED_SGX_QUOTE"
    }
    return json.dumps(quote).encode()

def parse_quote(quote_bytes):
    return json.loads(quote_bytes.decode())


def verify_quote(quote_bytes, expected_mrenclave):
    quote = parse_quote(quote_bytes)
    return quote["mrenclave"] == expected_mrenclave


def run_attestation_demo():
    manifest = "dummy_manifest"
    code = "dummy_code"

    mrenclave = compute_mrenclave(manifest, code)
    quote = generate_quote(mrenclave)

    result = verify_quote(quote, mrenclave)

    print("ATTESTATION:", "VERIFIED" if result else "FAILED")   
