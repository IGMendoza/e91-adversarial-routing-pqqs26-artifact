#!/usr/bin/env python
"""Evaluate topology-held-out DT checks for an Exp3 sequence run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sequence_game.experiments.exp3_sequence.dt_holdout import (
    evaluate_dt_holdout_payload,
    markdown_table,
    strong_generalization_drops,
)


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir
    payload_path = args.payload or run_dir / "dt" / "dt_payload.json"
    max_depth = args.max_depth
    if max_depth is None:
        max_depth = _config_dt_max_depth(run_dir) or 3

    payload = json.loads(payload_path.read_text())
    results = evaluate_dt_holdout_payload(payload, max_depth=max_depth)

    out_dir = args.out_dir or payload_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "dt_topology_holdout.csv"
    md_path = out_dir / "dt_topology_holdout.md"
    _write_csv(csv_path, results)
    md_text = markdown_table(results)
    md_path.write_text(md_text + "\n")

    print(md_text)
    print()
    print(f"max_depth: {max_depth}")
    print(f"wrote: {csv_path}")
    print(f"wrote: {md_path}")

    drops = strong_generalization_drops(results)
    if drops:
        print()
        print("Strong train-vs-topology-held-out drops:")
        for result in drops:
            drop = result.train_r2 - result.topology_heldout_r2
            print(
                f"- {result.target}: train R2={result.train_r2:.4f}, "
                f"held-out R2={result.topology_heldout_r2:.4f}, "
                f"drop={drop:.4f}"
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/exp3_dynamic_500k"),
        help="Exp3 sequence run directory.",
    )
    parser.add_argument(
        "--payload",
        type=Path,
        default=None,
        help="Optional explicit dt_payload.json path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for CSV/Markdown. Defaults to the payload directory.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="DecisionTreeRegressor max_depth. Defaults to run config dt_max_depth, then 3.",
    )
    return parser.parse_args()


def _config_dt_max_depth(run_dir: Path) -> int | None:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None
    payload = json.loads(config_path.read_text())
    value = payload.get("config", {}).get("dt_max_depth")
    return int(value) if value is not None else None


def _write_csv(path: Path, results) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "rows",
                "topologies",
                "train_r2",
                "topology_heldout_r2",
                "topology_heldout_mae",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())


if __name__ == "__main__":
    main()
