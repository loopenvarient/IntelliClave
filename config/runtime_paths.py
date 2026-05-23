"""
config/runtime_paths.py

Paths for dashboard-visible runtime files (status, attestation).

Local dev: reads/writes repo root and results/ (dual-write for compatibility).
Docker/K8s: set DASHBOARD_STATE_DIR=/app/results so all services share one volume.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def dashboard_state_dir() -> str:
    env = os.environ.get("DASHBOARD_STATE_DIR", "").strip()
    if env:
        return os.path.abspath(env)
    return os.path.join(project_root(), "results")


def _candidate_paths(filename: str) -> List[str]:
    """Search order: shared state dir, then repo root (legacy)."""
    root = project_root()
    state = dashboard_state_dir()
    return [
        os.path.join(state, filename),
        os.path.join(root, filename),
    ]


def resolve_runtime_file(filename: str) -> Optional[str]:
    """Return first existing path for filename, or None."""
    for path in _candidate_paths(filename):
        if os.path.isfile(path):
            return path
    return None


def primary_runtime_path(filename: str) -> str:
    """Canonical write path (under DASHBOARD_STATE_DIR / results/)."""
    path = os.path.join(dashboard_state_dir(), filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def legacy_runtime_path(filename: str) -> str:
    return os.path.join(project_root(), filename)


def write_json_runtime(filename: str, data: dict) -> str:
    """
    Write JSON to the canonical path and mirror to repo root for local tooling.
    Returns the canonical path written.
    """
    primary = primary_runtime_path(filename)
    with open(primary, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    legacy = legacy_runtime_path(filename)
    if os.path.normpath(legacy) != os.path.normpath(primary):
        with open(legacy, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return primary


def read_json_runtime(filename: str) -> dict:
    """Load JSON from first existing candidate path."""
    path = resolve_runtime_file(filename)
    if path is None:
        raise FileNotFoundError(
            f"{filename} not found under {dashboard_state_dir()} or repo root"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)
