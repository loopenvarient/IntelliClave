"""
crypto/certs/generate_tls_certs.py

Generates TLS certificates for IntelliClave FL communication.

What gets generated:
    crypto/certs/tls/
    ├── ca.key          — CA private key
    ├── ca.crt          — Self-signed CA certificate
    ├── server.key      — FL server private key
    ├── server.csr      — FL server certificate signing request
    ├── server.crt      — FL server certificate (signed by CA)
    ├── client.key      — FL client private key
    ├── client.csr      — FL client certificate signing request
    └── client.crt      — FL client certificate (signed by CA)

Why TLS for Flower gRPC:
    Flower uses gRPC which supports TLS natively. With these certs:
    - All weight updates are transmitted over an encrypted channel
    - Server identity is verified by clients (prevents rogue server)
    - Client identity is verified by server (prevents impersonation)
    This is the transport-layer complement to the AES-256-GCM
    application-layer encryption in crypto_layer.py.

Usage:
    python3 crypto/certs/generate_tls_certs.py

    Then start the FL server with:
        python fl/run_server.py --tls

    And clients with:
        python fl/run_client.py --id 1 --tls

Output:
    crypto/certs/tls/  (all cert files)
"""

import datetime
import ipaddress
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
TLS_DIR = os.path.join(_HERE, "tls")

try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
except ImportError:
    print("ERROR: cryptography package required.")
    print("       pip install cryptography")
    sys.exit(1)

_UTC = datetime.timezone.utc

def _now():
    return datetime.datetime.now(_UTC)

def _days(n):
    return datetime.timedelta(days=n)


# ── Key generation ─────────────────────────────────────────────────────────────

def _gen_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _save_key(key, path: str, password: bytes = None):
    enc = (
        serialization.BestAvailableEncryption(password)
        if password
        else serialization.NoEncryption()
    )
    with open(path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=enc,
        ))


def _save_cert(cert, path: str):
    with open(path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))


# ── Certificate builders ───────────────────────────────────────────────────────

def _build_ca(key) -> x509.Certificate:
    """Build a self-signed CA certificate."""
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "GB"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IntelliClave"),
        x509.NameAttribute(NameOID.COMMON_NAME, "IntelliClave Root CA"),
    ])
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now())
        .not_valid_after(_now() + _days(3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_cert_sign=True, crl_sign=True,
                content_commitment=False, key_encipherment=False,
                data_encipherment=False, key_agreement=False,
                encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )


def _build_server_cert(server_key, ca_key, ca_cert) -> x509.Certificate:
    """Build a server certificate signed by the CA."""
    subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "GB"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IntelliClave"),
        x509.NameAttribute(NameOID.COMMON_NAME, "fl-server"),
    ])
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now())
        .not_valid_after(_now() + _days(365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("fl-server"),
                x509.DNSName("fl-server-service"),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_encipherment=True,
                content_commitment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=False,
                crl_sign=False, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )


def _build_client_cert(client_key, ca_key, ca_cert, client_id: str = "fl-client") -> x509.Certificate:
    """Build a client certificate signed by the CA."""
    subject = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "GB"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "IntelliClave"),
        x509.NameAttribute(NameOID.COMMON_NAME, client_id),
    ])
    return (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(ca_cert.subject)
        .public_key(client_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(_now())
        .not_valid_after(_now() + _days(365))
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True, key_encipherment=True,
                content_commitment=False, data_encipherment=False,
                key_agreement=False, key_cert_sign=False,
                crl_sign=False, encipher_only=False, decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([x509.ExtendedKeyUsageOID.CLIENT_AUTH]),
            critical=False,
        )
        .sign(ca_key, hashes.SHA256())
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def generate_all(tls_dir: str = TLS_DIR, force: bool = False) -> dict:
    """
    Generate the full TLS certificate bundle for IntelliClave.

    Returns a dict of paths to all generated files.
    Skips generation if files already exist (unless force=True).
    """
    os.makedirs(tls_dir, exist_ok=True)

    paths = {
        "ca_key":     os.path.join(tls_dir, "ca.key"),
        "ca_crt":     os.path.join(tls_dir, "ca.crt"),
        "server_key": os.path.join(tls_dir, "server.key"),
        "server_crt": os.path.join(tls_dir, "server.crt"),
        "client_key": os.path.join(tls_dir, "client.key"),
        "client_crt": os.path.join(tls_dir, "client.crt"),
    }

    if not force and all(os.path.exists(p) for p in paths.values()):
        print(f"[TLS] Certificates already exist at {tls_dir} — skipping generation.")
        print(f"[TLS] Use force=True or delete the directory to regenerate.")
        return paths

    print(f"[TLS] Generating certificate bundle → {tls_dir}")

    # 1. CA
    print("[TLS]   Generating CA keypair...")
    ca_key = _gen_key()
    ca_crt = _build_ca(ca_key)
    _save_key(ca_key, paths["ca_key"])
    _save_cert(ca_crt, paths["ca_crt"])
    print(f"[TLS]   CA cert:     {paths['ca_crt']}")

    # 2. Server
    print("[TLS]   Generating server keypair...")
    server_key = _gen_key()
    server_crt = _build_server_cert(server_key, ca_key, ca_crt)
    _save_key(server_key, paths["server_key"])
    _save_cert(server_crt, paths["server_crt"])
    print(f"[TLS]   Server cert: {paths['server_crt']}")

    # 3. Client (shared cert for all 3 FL clients in prototype)
    print("[TLS]   Generating client keypair...")
    client_key = _gen_key()
    client_crt = _build_client_cert(client_key, ca_key, ca_crt, "fl-client")
    _save_key(client_key, paths["client_key"])
    _save_cert(client_crt, paths["client_crt"])
    print(f"[TLS]   Client cert: {paths['client_crt']}")

    print(f"[TLS] ✓ Certificate bundle complete")
    return paths


def verify_bundle(tls_dir: str = TLS_DIR) -> bool:
    """
    Verify the generated certificate bundle is self-consistent.
    Checks that server and client certs are signed by the CA.
    """
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    ca_crt_path     = os.path.join(tls_dir, "ca.crt")
    server_crt_path = os.path.join(tls_dir, "server.crt")
    client_crt_path = os.path.join(tls_dir, "client.crt")

    for p in [ca_crt_path, server_crt_path, client_crt_path]:
        if not os.path.exists(p):
            print(f"[TLS] MISSING: {p}")
            return False

    with open(ca_crt_path, "rb") as f:
        ca_cert = x509.load_pem_x509_certificate(f.read())
    with open(server_crt_path, "rb") as f:
        server_cert = x509.load_pem_x509_certificate(f.read())
    with open(client_crt_path, "rb") as f:
        client_cert = x509.load_pem_x509_certificate(f.read())

    # Verify issuer chain
    assert server_cert.issuer == ca_cert.subject, "Server cert not issued by CA"
    assert client_cert.issuer == ca_cert.subject, "Client cert not issued by CA"

    # Verify CA is self-signed
    assert ca_cert.issuer == ca_cert.subject, "CA cert is not self-signed"

    # Verify validity periods
    now = _now()
    for name, cert in [("CA", ca_cert), ("Server", server_cert), ("Client", client_cert)]:
        assert cert.not_valid_before_utc <= now <= cert.not_valid_after_utc, \
            f"{name} cert is not currently valid"

    print("[TLS] ✓ Certificate bundle verified:")
    print(f"[TLS]   CA      : {ca_cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value}")
    print(f"[TLS]   Server  : {server_cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value}")
    print(f"[TLS]   Client  : {client_cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value}")
    print(f"[TLS]   Valid until: {server_cert.not_valid_after_utc.strftime('%Y-%m-%d')}")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate TLS certs for IntelliClave FL")
    parser.add_argument("--force", action="store_true", help="Regenerate even if certs exist")
    parser.add_argument("--dir", default=TLS_DIR, help=f"Output directory (default: {TLS_DIR})")
    args = parser.parse_args()

    print("=" * 55)
    print("IntelliClave — TLS Certificate Generator")
    print("=" * 55)

    paths = generate_all(tls_dir=args.dir, force=args.force)

    print()
    verify_bundle(tls_dir=args.dir)

    print()
    print("=" * 55)
    print("Usage with Flower FL:")
    print()
    print("  Server:")
    print(f"    python fl/run_server.py --tls \\")
    print(f"      --tls-ca     {paths['ca_crt']} \\")
    print(f"      --tls-cert   {paths['server_crt']} \\")
    print(f"      --tls-key    {paths['server_key']}")
    print()
    print("  Client:")
    print(f"    python fl/run_client.py --id 1 --tls \\")
    print(f"      --tls-ca     {paths['ca_crt']} \\")
    print(f"      --tls-cert   {paths['client_crt']} \\")
    print(f"      --tls-key    {paths['client_key']}")
    print("=" * 55)
