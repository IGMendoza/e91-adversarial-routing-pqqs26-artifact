#!/usr/bin/env python3
"""Build the corpus-wide ordered intervention and alarm figure."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=REPO_ROOT / "runs/histogram_prime_probe_followup_105_500k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "paper/pqqs2026/figures/prime_probe_ordered_dynamics.png",
    )
    args = parser.parse_args()
    manifest = json.loads((args.run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    runs = [
        json.loads(line)
        for line in (args.run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checkpoints = {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in (args.run_dir / "graph_checkpoints").glob("*.json")
    }
    _plot_ordered_phase_dynamics(manifest, runs, [], checkpoints, args.output)


if __name__ == "__main__":
    main()
