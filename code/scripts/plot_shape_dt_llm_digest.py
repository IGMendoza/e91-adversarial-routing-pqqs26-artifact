#!/usr/bin/env python3
"""Plot the shape-DT LLM digest depth sweep produced by run_shape_dt_llm_digest.py.

Reads ``digest_results.json`` and writes:

- ``loto_r2_vs_depth.png`` -- real leave-one-topology-out R^2 for both
  targets (matrix_retention, exploitability) as tree depth increases. Not
  model-dependent (same DT is shared across models at a given depth), so
  one line per target.
- ``digest_quality_vs_depth.png`` -- small multiples, one panel per
  mechanical check (numeric faithfulness, dominant-feature-named rate,
  section completeness, scope-clean rate), one line per model, so
  digestion-quality trends across depth are visible per model without a
  dual-axis chart.

Standalone and idempotent -- safe to re-run any time more depths/models
have been added to the input JSON.

Usage:
    .venv/bin/python scripts/plot_shape_dt_llm_digest.py \\
      runs/shape_dt_llm_digest/digest_results.json \\
      --out-dir runs/shape_dt_llm_digest/figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

# Fixed categorical slots (validated palette, dataviz skill) -- assigned by
# entity (model / target), never cycled or repainted when the series set changes.
_MODEL_COLORS = {
    "phi4:14b": "#2a78d6",       # slot 1: blue
    "llama3.1:8b": "#008300",    # slot 2: green
}
_TARGET_COLORS = {
    "retention": "#2a78d6",      # slot 1: blue
    "exploitability": "#eb6834",  # slot 6: orange
}
_MUTED = "#898781"
_GRID = "#e1e0d9"


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    records: list[dict[str, Any]] = json.loads(args.digest_json.read_text(encoding="utf-8"))
    if not records:
        raise SystemExit(f"no records found in {args.digest_json}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "sequence_matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _plot_loto_r2_vs_depth(records, args.out_dir, plt)
    _plot_digest_quality_vs_depth(records, args.out_dir, plt)


def _style_axes(ax, muted=_MUTED, grid=_GRID) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(muted)
    ax.tick_params(colors=muted)
    ax.grid(axis="y", color=grid, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _plot_loto_r2_vs_depth(records: list[dict[str, Any]], out_dir: Path, plt: Any) -> None:
    by_depth: dict[int, dict[str, float]] = {}
    for r in records:
        by_depth.setdefault(r["depth"], {
            "retention": r["retention"]["r2_loo"],
            "exploitability": r["exploitability"]["r2_loo"],
        })
    depths = sorted(by_depth)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for target, label in (("retention", "matrix_retention"), ("exploitability", "exploitability")):
        values = [by_depth[d][target] for d in depths]
        ax.plot(depths, values, marker="o", markersize=6, linewidth=2,
                 color=_TARGET_COLORS[target], label=label, zorder=3)
    ax.axhline(0.0, color=_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.set_xlabel("tree depth")
    ax.set_ylabel("LOTO R² (real, leave-one-topology-out)")
    ax.set_title("Pooled shape-DT generalization vs. tree depth")
    ax.set_xticks(depths)
    _style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path = out_dir / "loto_r2_vs_depth.png"
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)
    print(f"-> {out_path}")


def _plot_digest_quality_vs_depth(records: list[dict[str, Any]], out_dir: Path, plt: Any) -> None:
    by_model_depth: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for r in records:
        score = r["score"]
        dominant_named_rate = (
            int(score["dominant_feature_retention"]["named"])
            + int(score["dominant_feature_exploitability"]["named"])
        ) / 2.0
        sections = score["sections"]
        sections_total = len(sections["present"]) + len(sections["missing"])
        sections_pct = 100.0 * len(sections["present"]) / max(1, sections_total)
        by_model_depth[r["model"]][r["depth"]] = {
            "faithfulness_pct": 100.0 * score["faithfulness"]["score"],
            "dominant_feature_named_pct": 100.0 * dominant_named_rate,
            "sections_complete_pct": sections_pct,
            "scope_clean_pct": 100.0 if score["scope_overclaim"]["clean"] else 0.0,
        }
    models = sorted(by_model_depth)
    depths = sorted({d for per_model in by_model_depth.values() for d in per_model})

    metrics = [
        ("faithfulness_pct", "numeric faithfulness"),
        ("dominant_feature_named_pct", "dominant feature named"),
        ("sections_complete_pct", "sections complete"),
        ("scope_clean_pct", "scope-overclaim clean"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4.0), sharey=True)
    for ax, (metric_key, metric_label) in zip(axes, metrics):
        for model in models:
            per_depth = by_model_depth[model]
            values = [per_depth[d].get(metric_key) for d in depths if d in per_depth]
            xs = [d for d in depths if d in per_depth]
            ax.plot(xs, values, marker="o", markersize=6, linewidth=2,
                     color=_MODEL_COLORS.get(model, _MUTED), label=model, zorder=3)
        ax.set_title(metric_label, fontsize=10)
        ax.set_xlabel("tree depth")
        ax.set_xticks(depths)
        ax.set_ylim(-5, 105)
        _style_axes(ax)
    axes[0].set_ylabel("%")
    axes[-1].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle("DT-grounded digest quality vs. tree depth, by model", y=1.02)
    fig.tight_layout()
    out_path = out_dir / "digest_quality_vs_depth.png"
    fig.savefig(out_path, dpi=150, facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)
    print(f"-> {out_path}")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("digest_json", type=Path, help="Output of run_shape_dt_llm_digest.py.")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
