"""Tiny reproducible sweep runner over toy Eve experiments.

A sweep is a base experiment config plus axes: a mapping from dotted config
paths (e.g. ``"topology.seed"``, ``"reward.abort_reward"``) to value lists.
Runs are the cartesian product. Each run directory gets the resolved config,
run metadata (with scope labels), metrics, and — on failure — an error status
file. Existing completed runs (summary.json present) are skipped for resume.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import traceback
from pathlib import Path
from typing import Any

from .runner import ExperimentConfigError, run_eve_experiment


def set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ExperimentConfigError(f"unknown config section {part!r} in {dotted_key!r}")
        node = node[part]
    if parts[-1] not in node:
        raise ExperimentConfigError(f"unknown config key {dotted_key!r}")
    node[parts[-1]] = value


def expand_axes(axes: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Deterministic cartesian product of axis assignments."""
    if not axes:
        return [{}]
    keys = sorted(axes)
    combos = itertools.product(*(axes[k] for k in keys))
    return [dict(zip(keys, combo)) for combo in combos]


def run_id_for(assignment: dict[str, Any]) -> str:
    if not assignment:
        return "run-base"
    text = json.dumps(assignment, sort_keys=True)
    return "run-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def run_sweep(base_config: dict[str, Any], axes: dict[str, list[Any]],
              output_dir: Path, *, skip_existing: bool = True) -> list[dict[str, Any]]:
    """Execute the sweep; returns a manifest of {run_id, assignment, status}."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for assignment in expand_axes(axes):
        run_id = run_id_for(assignment)
        run_dir = output_dir / run_id
        entry: dict[str, Any] = {"run_id": run_id, "assignment": assignment}
        if skip_existing and (run_dir / "summary.json").exists():
            entry["status"] = "skipped_existing"
            manifest.append(entry)
            continue
        config = copy.deepcopy(base_config)
        for key, value in assignment.items():
            set_dotted(config, key, value)
        try:
            run_eve_experiment(config, run_dir)
            entry["status"] = "ok"
        except Exception as exc:  # noqa: BLE001 - sweep must record and continue
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "error.json").write_text(json.dumps({
                "status": "failed",
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "assignment": assignment,
            }, indent=2), encoding="utf-8")
            entry["status"] = "failed"
            entry["error"] = repr(exc)
        manifest.append(entry)
    (output_dir / "sweep_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
