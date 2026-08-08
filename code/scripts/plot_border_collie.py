#!/usr/bin/env python3
"""Plot detect_border_collie.py's output: score distribution + clearest example.

Left panel: histogram of border-collie scores (corr_late - corr_early)
across all candidate graphs, with the "reversal" threshold marked. Right
panel: Alice's weight on her favorite route vs. Eve's aggregate attack
weight on the actions covering it, over turns, for the single clearest
example -- visually showing the early anti-correlation / late
correlation reversal.

Usage:
    .venv/bin/python scripts/plot_border_collie.py \\
      runs/rule_clusters/border_collie_results.json \\
      --production-dir runs/exp3_small_synth_234_production \\
      --corpus-db-path runs/exp3_topology_bench/corpus_small_synthetic_hop7.sqlite \\
      --out-dir runs/rule_clusters/figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_ALICE_COLOR = "#2a78d6"  # slot 1: blue
_EVE_COLOR = "#eb6834"    # slot 6: orange
_MUTED = "#898781"
_GRID = "#e1e0d9"
_THRESHOLD = 0.3


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    results = json.loads(args.results_json.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "sequence_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.4))

    scores = [r["score"] for r in results]
    ax1.hist(scores, bins=20, color=_EVE_COLOR, alpha=0.85, zorder=3)
    ax1.axvline(_THRESHOLD, color=_MUTED, linewidth=1.2, linestyle="--", zorder=2,
                label=f"reversal threshold ({_THRESHOLD})")
    n_above = sum(1 for s in scores if s > _THRESHOLD)
    ax1.set_title(f"Score distribution (n={len(scores)}, {n_above} above threshold)")
    ax1.set_xlabel("border-collie score (corr_late - corr_early)")
    ax1.set_ylabel("graph count")
    ax1.legend(frameon=False, fontsize=9)

    top = max(results, key=lambda r: r["score"])
    graph_id = top["graph_id"]
    online_path = args.production_dir / "online_runs" / "exp3_vs_exp3" / f"{graph_id}.json"

    from sequence_game.experiments.exp3_sequence.corpus import load_graph_cases_from_sqlite
    from sequence_game.xai.border_collie import action_covers_route
    import numpy as np

    cases_by_id = {c.graph_id: c for c in load_graph_cases_from_sqlite(args.corpus_db_path)}
    case = cases_by_id[graph_id]
    payoff_json = json.loads((args.production_dir / "payoff_matrices" / f"{graph_id}.json").read_text())
    online_json = json.loads(online_path.read_text())
    steps = online_json["steps"]
    action_ids = payoff_json["action_ids"]

    alice_mat = np.array([s["alice_strategy"] for s in steps])
    eve_mat = np.array([s["eve_strategy"] for s in steps])
    turns = np.array([s["turn"] for s in steps])
    favorite_idx = int(np.argmax(alice_mat[-1]))
    favorite_route = case.routes[favorite_idx]
    covering = [i for i, aid in enumerate(action_ids) if action_covers_route(aid, favorite_route)]

    alice_series = alice_mat[:, favorite_idx]
    eve_series = eve_mat[:, covering].sum(axis=1)

    ax2.plot(turns, alice_series, color=_ALICE_COLOR, linewidth=2, label="Alice weight on favorite route", zorder=3)
    ax2.plot(turns, eve_series, color=_EVE_COLOR, linewidth=2, label="Eve weight on covering action(s)", zorder=3)
    half_turn = turns[len(turns) // 2]
    ax2.axvline(half_turn, color=_MUTED, linewidth=1, linestyle="--", zorder=2, label="early/late split")
    ax2.set_title(f"Clearest example ({graph_id.split('_')[0]}): score={top['score']:.2f}")
    ax2.set_xlabel("turn")
    ax2.set_ylabel("strategy weight")
    ax2.legend(frameon=False, fontsize=8, loc="upper left")

    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(_MUTED)
        ax.tick_params(colors=_MUTED)
        ax.grid(axis="y", color=_GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle('"Border collie" bait-and-switch: real but not shape-predictable')
    fig.tight_layout()
    out_path = args.out_dir / "border_collie.png"
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_path}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_json", type=Path, help="Output of detect_border_collie.py.")
    parser.add_argument("--production-dir", type=Path, required=True)
    parser.add_argument("--corpus-db-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
