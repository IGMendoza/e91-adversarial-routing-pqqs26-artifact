#!/usr/bin/env python3
"""Plot Level 2 root-split cluster frequencies produced by fit_rule_clusters.py.

Writes one horizontal bar chart per target (Eve's per-graph action-strategy
trees, Alice's per-graph route-strategy trees): each bar is one
(root_feature, direction) cluster, length = fraction of graphs whose
per-graph tree's root split falls in that cluster.

Usage:
    .venv/bin/python scripts/plot_rule_clusters.py \\
      runs/rule_clusters/level2_clusters.json \\
      --out-dir runs/rule_clusters/figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_MUTED = "#898781"
_GRID = "#e1e0d9"
_POSITIVE = "#2a78d6"   # slot 1: blue -- higher feature value -> higher target
_NEGATIVE = "#eb6834"   # slot 6: orange -- higher feature value -> lower target
_NO_SPLIT = "#c3c2b7"   # neutral -- no data-driven split found


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report: dict[str, Any] = json.loads(args.report_json.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "sequence_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for label, target_report in report.items():
        _plot_one_target(label, target_report, args.out_dir, plt)


def _plot_one_target(label: str, target_report: dict[str, Any], out_dir: Path, plt: Any) -> None:
    clusters = target_report["clusters"]
    num_fit = target_report["num_graphs_fit"]

    labels, fractions, colors = [], [], []
    for cluster in clusters:
        feature = cluster["root_feature"]
        direction = cluster["root_direction"]
        if feature is None:
            labels.append("(no split)")
            colors.append(_NO_SPLIT)
        else:
            arrow = "↑" if direction == "positive" else "↓"
            labels.append(f"{feature} {arrow}")
            colors.append(_POSITIVE if direction == "positive" else _NEGATIVE)
        fractions.append(100.0 * cluster["count"] / max(1, num_fit))

    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(labels) + 1.2))
    y = range(len(labels))
    ax.barh(list(y), fractions, color=colors, zorder=3)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f"% of {num_fit} graphs (root-split cluster)")
    ax.set_title(f"Level 2 root-split clusters — {label}")
    ax.set_xlim(0, 100)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_MUTED)
    ax.tick_params(colors=_MUTED)
    ax.grid(axis="x", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for yi, frac in zip(y, fractions):
        ax.text(frac + 1.5, yi, f"{frac:.1f}%", va="center", fontsize=9, color=_MUTED)

    fig.tight_layout()
    out_path = out_dir / f"{label}_root_split_clusters.png"
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"-> {out_path}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("report_json", type=Path, help="Output of fit_rule_clusters.py.")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
