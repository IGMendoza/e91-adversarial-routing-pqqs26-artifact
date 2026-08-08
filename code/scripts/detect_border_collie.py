#!/usr/bin/env python3
"""Detect the "border collie" bait-and-switch pattern in real Exp3 self-play.

Reads the real per-turn trajectory data an ``exp3_sequence_results.py``
production run already logs (``online_runs/exp3_vs_exp3/<graph_id>.json``,
one snapshot of both players' full strategy vectors every
``online_step_record_stride`` turns), restricted to graphs with
``node_cut_size == 3`` (three node-disjoint Alice-Bob paths -- the corpus's
closest analogue to a minimal 3-parallel-path "triple diamond" topology),
and computes a lagged-correlation score for each: does Eve's attack weight
on Alice's eventual favorite route move opposite to Alice's rising
commitment early on, then track it closely late (``score = corr_late -
corr_early``)?

Example (the 234-graph small_synthetic production run)::

    .venv/bin/python scripts/detect_border_collie.py \\
      --corpus-db-path runs/exp3_topology_bench/corpus_small_synthetic_hop7.sqlite \\
      --production-dir runs/exp3_small_synth_234_production \\
      --out-json runs/rule_clusters/border_collie_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sequence_game.experiments.exp3_sequence.corpus import load_graph_cases_from_sqlite
from sequence_game.graph_corpus.shape_features import compute_shape_features
from sequence_game.topology import TopologyIR
from sequence_game.xai.border_collie import compute_border_collie_result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus-db-path", type=Path, required=True)
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cases_by_id = {c.graph_id: c for c in load_graph_cases_from_sqlite(args.corpus_db_path)}

    candidates = []
    for gid, case in cases_by_id.items():
        ir = TopologyIR.from_dict(case.ir_dict)
        shape = compute_shape_features(ir, case.alice, case.bob)
        if shape.node_cut_size == 3:
            candidates.append(gid)
    print(f"candidate graphs (node_cut_size == 3): {len(candidates)}")

    results = []
    skipped = 0
    for gid in candidates:
        payoff_path = args.production_dir / "payoff_matrices" / f"{gid}.json"
        online_path = args.production_dir / "online_runs" / "exp3_vs_exp3" / f"{gid}.json"
        if not payoff_path.exists() or not online_path.exists():
            skipped += 1
            continue
        payoff_json = json.loads(payoff_path.read_text())
        online_json = json.loads(online_path.read_text())
        result = compute_border_collie_result(
            gid, cases_by_id[gid].routes, payoff_json["action_ids"], online_json.get("steps") or [],
        )
        if result is None:
            skipped += 1
            continue
        results.append(result)

    print(f"analyzed: {len(results)}  skipped (missing data / degenerate): {skipped}")
    results.sort(key=lambda r: r.score, reverse=True)

    print("\n=== top 10 by border-collie score (corr_late - corr_early) ===")
    for r in results[:10]:
        print(
            f"{r.graph_id}: score={r.score:+.3f}  corr_early={r.corr_early:+.3f}  "
            f"corr_late={r.corr_late:+.3f}  alice_final={r.favorite_final_weight:.3f}"
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps([asdict(r) for r in results], indent=2, sort_keys=True))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
