"""Markdown report generation for one experiment run directory."""

from __future__ import annotations

import json
from pathlib import Path

SCOPE_WARNING = (
    "> **Scope warning:** this run uses the *toy* game-mechanics stack. "
    "Nothing in this report is a physical simulation result or a security "
    "claim. Literature-scoped parameters are still TODO (see "
    "`configs/physical/*.json`).")


def _load_if_exists(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def generate_run_report(run_dir: Path) -> str:
    """Summarize a run directory (as written by run_eve_experiment) as Markdown."""
    run_dir = Path(run_dir)
    metadata = _load_if_exists(run_dir / "run_metadata.json")
    config = _load_if_exists(run_dir / "experiment_config.json")
    summary = _load_if_exists(run_dir / "summary.json")
    training_metrics = _load_if_exists(run_dir / "training_metrics.json")

    lines = [f"# Run report: {run_dir.name}", "", SCOPE_WARNING, ""]

    lines += ["## Provenance", ""]
    for key in ("timestamp_utc", "seed", "git_commit", "scope"):
        lines.append(f"- {key}: `{metadata.get(key, 'missing')}`")
    lines.append("")

    if config:
        lines += ["## Config", "", "```json",
                  json.dumps(config, indent=2, sort_keys=True), "```", ""]

    lines += ["## Key metrics", ""]
    merged = {**summary, **training_metrics}
    if merged:
        for key in sorted(merged):
            lines.append(f"- {key}: {merged[key]}")
    else:
        lines.append("- no metrics files found")
    lines.append("")

    lines += ["## Caveats", "",
              "- Rewards are game-design quantities, not information measures.",
              "- The trial generator is a toy correlation rule "
              "(`sequence_game/protocol/toy_trial.py`).",
              "- Unresolved scientific TODOs are listed in "
              "`docs/HISTOGRAM_PROJECT_HANDOFF.md` and the physical configs.",
              ""]
    return "\n".join(lines)


def write_run_report(run_dir: Path) -> Path:
    out = Path(run_dir) / "report.md"
    out.write_text(generate_run_report(run_dir), encoding="utf-8")
    return out
