"""
dashboard/backend/training_freshness.py

Training-run freshness for the dashboard (replaces brittle 24h mtime heuristic).

Status fields written by the FL server (status.json):
  last_training_completed_at   — ISO-8601 UTC when the last round finished
  expected_training_cadence_hours — operator-defined "fresh" window (default 24)
  training_active              — True while a run is in progress

Derived fields added by the dashboard API:
  training_freshness           — live | stale | training | pre-recorded | unknown
  stale_hours                  — hours since last_training_completed_at
  freshness_note               — human-readable explanation
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


def _parse_iso_utc(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def _hours_since(dt: datetime) -> float:
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)


def compute_training_freshness(
    data: dict,
    rel_path: str,
    root: str,
    *,
    default_cadence_hours: float = None,
) -> Dict[str, Any]:
    """
    Enrich status/results payloads with explicit freshness metadata.
    """
    cadence = default_cadence_hours
    if cadence is None:
        cadence = float(
            data.get("expected_training_cadence_hours")
            or os.environ.get("EXPECTED_TRAINING_CADENCE_HOURS", "24")
        )

    last_at_str = data.get("last_training_completed_at") or data.get("last_training_at")
    training_active = bool(data.get("training_active", False))

    # Fallback: file mtime when server did not embed a timestamp (legacy runs)
    mtime_hours: Optional[float] = None
    path = os.path.join(root, rel_path)
    if os.path.exists(path):
        mtime_hours = (time.time() - os.path.getmtime(path)) / 3600.0

    last_dt = _parse_iso_utc(last_at_str) if last_at_str else None
    if last_dt is None and mtime_hours is not None:
        last_dt = datetime.fromtimestamp(
            os.path.getmtime(path), tz=timezone.utc
        )

    stale_hours: Optional[float] = _hours_since(last_dt) if last_dt else mtime_hours

    rounds = data.get("rounds") or data.get("total_rounds") or 0
    if isinstance(rounds, list):
        rounds = len(rounds)
    has_real_rounds = int(rounds) > 0

    if training_active:
        freshness = "training"
        note = "FL training is in progress."
    elif last_dt is None and not has_real_rounds:
        freshness = "pre-recorded"
        note = (
            "No training timestamp in status — using bundled demo metrics. "
            "Run: python fl/run_fl_simulation.py --rounds 10 --clients 3"
        )
    elif last_dt is None:
        freshness = "unknown"
        note = "Training timestamp missing; file age alone is unreliable."
    elif stale_hours is not None and stale_hours <= cadence:
        freshness = "live"
        note = (
            f"Last training completed {stale_hours:.1f}h ago "
            f"(within {cadence:.0f}h cadence)."
        )
    else:
        freshness = "stale"
        note = (
            f"Stale as of {stale_hours:.1f}h ago — last completed "
            f"{last_at_str or 'run'} (cadence {cadence:.0f}h). "
            "Start a new FL run to refresh metrics."
        )

    return {
        **data,
        "expected_training_cadence_hours": cadence,
        "last_training_completed_at": last_at_str or (
            last_dt.strftime("%Y-%m-%dT%H:%M:%SZ") if last_dt else None
        ),
        "training_freshness": freshness,
        "stale_hours": round(stale_hours, 2) if stale_hours is not None else None,
        "freshness_note": note,
        # Legacy field for older frontend builds
        "data_source": "live" if freshness in ("live", "training") else (
            "pre-recorded" if freshness == "pre-recorded" else "stale"
        ),
        "data_source_note": note,
    }
