"""I/O helpers for Exp3 SeQUeNCe outputs."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def repo_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        commit = None
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        status = None
    return {"git_commit": commit, "git_status_short": status}
