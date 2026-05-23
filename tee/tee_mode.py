"""
tee/tee_mode.py — Detect whether real Intel SGX hardware is available.

gramine-direct (WSL2 / dev laptops): userspace only, simulated quotes.
gramine-sgx (production): hardware enclave, real attestation.
"""

from __future__ import annotations

import os
import platform
import shutil


def sgx_device_available() -> bool:
    """True when the host exposes an SGX character device."""
    return any(
        os.path.exists(p)
        for p in ("/dev/sgx_enclave", "/dev/isgx", "/dev/sgx")
    )


def gramine_sgx_installed() -> bool:
    return shutil.which("gramine-sgx") is not None


def detect_tee_mode() -> dict:
    """
    Return metadata for attestation records and the dashboard.

    Keys: mode, simulation_mode, platform, environment, sgx_available
    """
    sgx_hw = sgx_device_available()
    if sgx_hw:
        return {
            "mode":             "gramine-sgx",
            "simulation_mode":  False,
            "platform":         "Intel SGX (hardware)",
            "environment":      platform.system(),
            "sgx_available":    True,
            "gramine_sgx_path": shutil.which("gramine-sgx"),
        }

    env = "WSL2" if "microsoft" in platform.uname().release.lower() else platform.system()
    return {
        "mode":             "gramine-direct",
        "simulation_mode":  True,
        "platform":         "Intel SGX (simulated — no hardware enclave)",
        "environment":      env,
        "sgx_available":    False,
        "gramine_sgx_path": shutil.which("gramine-sgx"),
        "warning": (
            "gramine-direct runs entirely in userspace with no hardware isolation. "
            "Attestation quotes and sealed storage are simulated. "
            "Use gramine-sgx on SGX-capable hardware for production."
        ),
    }


def enrich_attestation_record(record: dict) -> dict:
    """Merge detect_tee_mode() into an existing attestation.json dict."""
    info = detect_tee_mode()
    out = dict(record)
    # Prefer explicit mode in the file when set; otherwise use detection.
    mode = out.get("mode") or info["mode"]
    if mode == "gramine-direct":
        out["simulation_mode"] = True
    elif mode == "gramine-sgx" and info["sgx_available"]:
        out["simulation_mode"] = False
    else:
        out.setdefault("simulation_mode", info["simulation_mode"])
    out.setdefault("mode", mode)
    out.setdefault("platform", info["platform"])
    out.setdefault("environment", info.get("environment", info["environment"]))
    out["sgx_available"] = info["sgx_available"]
    if out.get("simulation_mode") and "warning" not in out:
        out["warning"] = info.get("warning")
    return out
