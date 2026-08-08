#!/usr/bin/env python3
"""Render the detector-aware ordered Prime-and-Probe trajectory figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.prime_probe_artifacts import (  # noqa: E402
    _plot_ordered_phase_dynamics,
)
from sequence_game.experiments.histogram_colearning.prime_probe_detection_corpus import (  # noqa: E402
    DEFAULT_PRIME_PROBE_DETECTION_RUN_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_PRIME_PROBE_DETECTION_RUN_DIR,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "paper"
            / "pqqs2026"
            / "figures"
            / "prime_probe_ordered_dynamics.png"
        ),
    )
    args = parser.parse_args()

    summary = _read_json(args.run_dir / "summary.json")
    if not bool(summary.get("complete")):
        raise ValueError("the detector rerun is incomplete")
    manifest = _read_json(args.run_dir / "run_manifest.json")
    runs = _read_jsonl(args.run_dir / "results.jsonl")
    comparisons = _read_jsonl(
        args.run_dir / "paired_comparisons.jsonl"
    )
    checkpoints = {
        path.stem: _read_json(path)
        for path in sorted(
            (args.run_dir / "graph_checkpoints").glob("*.json")
        )
    }
    if len(runs) != int(manifest["expected_condition_runs"]):
        raise ValueError("the detector rerun has incomplete condition rows")
    if len(checkpoints) != len(manifest["cases"]):
        raise ValueError("the detector rerun has incomplete graph checkpoints")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot_ordered_phase_dynamics(
        manifest,
        runs,
        comparisons,
        checkpoints,
        args.output,
    )
    print(
        json.dumps(
            {
                "source": str(args.run_dir),
                "png": str(args.output),
                "pdf": str(args.output.with_suffix(".pdf")),
                "svg": str(args.output.with_suffix(".svg")),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


if __name__ == "__main__":
    main()
