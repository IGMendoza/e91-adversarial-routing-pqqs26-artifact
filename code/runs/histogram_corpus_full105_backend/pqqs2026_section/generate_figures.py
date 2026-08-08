"""Render the four real-corpus DT-digestion figures for the new PQQS'26
section ``dt_digestion_real_corpus_section.tex`` directly from the real JSON
output under ``runs/histogram_corpus_full105_backend/``.

Every number plotted here is read from the real JSON files at render time --
nothing is hand-transcribed into this script. Run with:

    .venv/bin/python runs/histogram_corpus_full105_backend/pqqs2026_section/generate_figures.py

Scientific scope: empirical_fit -- these are descriptive plots of real
cross-validated fits and real mechanical LLM-digest scoring, not a physical
or security claim.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "runs" / "histogram_corpus_full105_backend"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TURNS = ["100k", "500k"]
PAIRINGS = ["tsallis_rv_eta1_vs_tsallis_rv_eta1", "linear_goptimal_eta2_vs_tsallis_rv_eta1"]
PAIRING_LABEL = {
    "tsallis_rv_eta1_vs_tsallis_rv_eta1": "Tsallis-INF vs Tsallis-INF (symmetric)",
    "linear_goptimal_eta2_vs_tsallis_rv_eta1": "LinearExp2 vs Tsallis-INF (asymmetric)",
}
# Matches this paper's own Fig. tree-generalization convention: blue circle
# for the LinearExp2/Tsallis-INF pairing, orange square for Tsallis-INF/
# Tsallis-INF, held fixed across every figure in this new section.
PAIRING_COLOR = {
    "linear_goptimal_eta2_vs_tsallis_rv_eta1": "#2a78d6",  # blue
    "tsallis_rv_eta1_vs_tsallis_rv_eta1": "#eb6834",       # orange
}
PAIRING_MARKER = {
    "linear_goptimal_eta2_vs_tsallis_rv_eta1": "o",
    "tsallis_rv_eta1_vs_tsallis_rv_eta1": "s",
}

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e5e5e2",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────
# Figure 1: LOTO R^2 vs. tree depth, both targets, both pairings, both turns
# ─────────────────────────────────────────────────────────────────────────

def fig_generalization():
    # Pull the full depth sweep {1,2,3,4,6,8} from the Ollama digest_results
    # files -- shape_dt_results_*.json only stores the single depth-3
    # "headline" tree, but the per-depth digest_results.json (from which the
    # LLM digest prompts were built) contains the complete swept LOTO curve
    # for both targets at every depth actually fit.
    curves = {}  # (turns, pairing) -> {"depth": [...], "retention": [...], "exploitability": [...]}
    for turns in TURNS:
        for pairing in PAIRINGS:
            p = BASE / turns / f"llm_digest_{pairing}" / "digest_results.json"
            d = load_json(p)
            by_depth = {}
            for e in d:
                by_depth.setdefault(e["depth"], e)
            depths = sorted(by_depth)
            curves[(turns, pairing)] = {
                "depth": depths,
                "retention": [by_depth[dep]["retention"]["r2_loo"] for dep in depths],
                "exploitability": [by_depth[dep]["exploitability"]["r2_loo"] for dep in depths],
            }

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.2), sharex=True, sharey="row")
    target_rows = [("retention", "LOTO $R^2$ (histogram_final_retention)"),
                   ("exploitability", "LOTO $R^2$ (exploitability)")]
    for row, (target, ylabel) in enumerate(target_rows):
        for col, turns in enumerate(TURNS):
            ax = axes[row][col]
            for pairing in PAIRINGS:
                c = curves[(turns, pairing)]
                ax.plot(
                    c["depth"], c[target],
                    color=PAIRING_COLOR[pairing], marker=PAIRING_MARKER[pairing],
                    markersize=7, linewidth=2.2,
                    label=PAIRING_LABEL[pairing] if (row == 0 and col == 0) else None,
                )
            ax.axhline(0.0, color="#9a9a95", linewidth=1.0, linestyle=(0, (3, 2)))
            if row == 0:
                ax.set_title(f"{turns.replace('k', ',000')} turns", fontsize=12)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=10.5)
            if row == 1:
                ax.set_xlabel("Tree depth")
            ax.set_xticks([1, 2, 3, 4, 6, 8])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False, fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "real_digest_generalization_by_depth.pdf", bbox_inches="tight")
    fig.savefig(OUT / "real_digest_generalization_by_depth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return curves


# ─────────────────────────────────────────────────────────────────────────
# Figure 2: digestion faithfulness by depth, Ollama vs Claude-family
# ─────────────────────────────────────────────────────────────────────────

def fig_checks():
    ollama_by_depth = {}
    for turns in TURNS:
        for pairing in PAIRINGS:
            p = BASE / turns / f"llm_digest_{pairing}" / "digest_results.json"
            for e in load_json(p):
                ollama_by_depth.setdefault(e["depth"], []).append(e["score"]["faithfulness"]["score"])

    claude_by_depth = {}
    for turns in TURNS:
        p = BASE / turns / "llm_digest_tsallis_rv_eta1_vs_tsallis_rv_eta1" / "claude_family_responses" / "digest_results.json"
        for e in load_json(p):
            claude_by_depth.setdefault(e["depth"], []).append(e["score"]["faithfulness"]["score"])

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for label, series, color, marker in [
        ("Local (phi4:14b, llama3.1:8b; 4 runs x 6 depths = 48 responses)", ollama_by_depth, "#2a78d6", "o"),
        ("Frontier (Haiku/Sonnet/Fable/Opus; symmetric pairing, 2 turn counts x 4 depths = 32 responses)", claude_by_depth, "#4a3aa7", "^"),
    ]:
        depths = sorted(series)
        means = [float(np.mean(series[dep])) for dep in depths]
        lo = [float(np.min(series[dep])) for dep in depths]
        hi = [float(np.max(series[dep])) for dep in depths]
        ax.plot(depths, means, color=color, marker=marker, markersize=8, linewidth=2.2, label=label)
        ax.fill_between(depths, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.set_xlabel("Tree depth")
    ax.set_ylabel("Mechanical faithfulness score\n(mean, shaded band = observed range)")
    ax.set_xticks([1, 2, 3, 4, 6, 8])
    ax.set_ylim(0.55, 1.03)
    ax.legend(loc="lower center", frameon=False, fontsize=9.0, bbox_to_anchor=(0.5, -0.42))
    fig.tight_layout()
    fig.savefig(OUT / "real_digest_checks_by_depth.pdf", bbox_inches="tight")
    fig.savefig(OUT / "real_digest_checks_by_depth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Figure 3: root-split cluster frequencies, Alice and Eve
# ─────────────────────────────────────────────────────────────────────────

DISPLAY_NAME = {
    ("mean_overlap_with_other_routes", "negative"): "route overlap $\\downarrow$",
    ("internal_node_count", "negative"): "internal nodes $\\downarrow$",
    ("length_m", "negative"): "route length $\\downarrow$",
    ("hop_count", "positive"): "hop count $\\uparrow$",
    ("hop_count", "negative"): "hop count $\\downarrow$",
    ("target_route_coverage", "positive"): "target coverage $\\uparrow$",
    ("target_route_coverage", "negative"): "target coverage $\\downarrow$",
    ("is_edge_dos", "positive"): "edge DoS",
    ("is_edge_information_probe", "positive"): "edge info. probe",
    ("is_swap_denial", "positive"): "swap denial",
    ("is_memory_degradation", "positive"): "memory degradation",
    ("is_repeater_memory_measure_z", "positive"): "memory measure $Z$",
    ("is_repeater_memory_measure_x", "positive"): "memory measure $X$",
    ("target_is_bottleneck", "positive"): "target is bottleneck",
}


def _top_categories(all_run_clusters, n=7):
    # union of (feature, direction) keys, ranked by mean fraction across runs
    totals = {}
    counts = {}
    for clusters in all_run_clusters:
        for c in clusters:
            key = (c["root_feature"], c["root_direction"])
            totals[key] = totals.get(key, 0.0) + c["fraction"]
            counts[key] = counts.get(key, 0) + 1
    ranked = sorted(totals, key=lambda k: -totals[k])
    return ranked[:n]


def fig_clusters():
    data = {}  # (turns, pairing) -> {"alice": clusters, "eve": clusters}
    for turns in TURNS:
        for pairing in PAIRINGS:
            p = BASE / turns / f"rule_clusters_{pairing}.json"
            d = load_json(p)
            data[(turns, pairing)] = {
                "alice": d["alice_route_strategy"]["clusters"],
                "eve": d["eve_action_strategy"]["clusters"],
            }

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.6))
    run_order = [(t, p) for p in PAIRINGS for t in TURNS]
    n_runs = len(run_order)
    bar_h = 0.8 / n_runs

    for ax, who, title in [
        (axes[0], "alice", "Alice's per-route trees (route-choice root split)"),
        (axes[1], "eve", "Eve's per-action trees (action-choice root split)"),
    ]:
        cats = _top_categories([data[k][who] for k in run_order], n=7)
        ylabels = [DISPLAY_NAME.get(k, f"{k[0]} {k[1]}") for k in cats]
        y_pos = np.arange(len(cats))
        for i, (turns, pairing) in enumerate(run_order):
            clusters = {(c["root_feature"], c["root_direction"]): c["fraction"] for c in data[(turns, pairing)][who]}
            vals = [clusters.get(k, 0.0) for k in cats]
            offset = (i - (n_runs - 1) / 2) * bar_h
            hatch = "///" if turns == "500k" else None
            ax.barh(
                y_pos + offset, vals, height=bar_h * 0.92,
                color=PAIRING_COLOR[pairing], hatch=hatch, edgecolor="white", linewidth=0.6,
                label=f"{PAIRING_LABEL[pairing].split(' (')[0]}, {turns.replace('k', ',000')} turns" if who == "alice" else None,
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ylabels, fontsize=10.5)
        ax.invert_yaxis()
        ax.set_xlabel("Fraction of fitted per-graph trees")
        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=11.5, loc="left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=2, frameon=False, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "real_digest_root_clusters.pdf", bbox_inches="tight")
    fig.savefig(OUT / "real_digest_root_clusters.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Figure 4: border-collie score distribution, node_cut_size == 3 subset
# ─────────────────────────────────────────────────────────────────────────

def fig_border_collie():
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharex=True, sharey=True)
    bins = np.linspace(-2.0, 2.0, 21)
    for row, pairing in enumerate(PAIRINGS):
        for col, turns in enumerate(TURNS):
            ax = axes[row][col]
            p = BASE / turns / f"border_collie_{pairing}.json"
            scores = [x["score"] for x in load_json(p)]
            ax.hist(scores, bins=bins, color=PAIRING_COLOR[pairing], edgecolor="white", linewidth=0.6)
            ax.axvline(0.3, color="#9a9a95", linewidth=1.2, linestyle=(0, (3, 2)))
            ax.set_title(
                f"{PAIRING_LABEL[pairing].split(' (')[0]}, {turns.replace('k', ',000')} turns"
                f"\n(n={len(scores)}, max={max(scores):.2f}, min={min(scores):.2f})",
                fontsize=10,
            )
            if row == 1:
                ax.set_xlabel("Lagged-correlation score $q_g$")
            if col == 0:
                ax.set_ylabel("Graphs")
    fig.tight_layout()
    fig.savefig(OUT / "real_digest_border_collie.pdf", bbox_inches="tight")
    fig.savefig(OUT / "real_digest_border_collie.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    curves = fig_generalization()
    fig_checks()
    fig_clusters()
    fig_border_collie()
    print("Wrote figures to", OUT)
    print()
    print("Generalization curves (for section-text cross-check):")
    for (turns, pairing), c in curves.items():
        print(f"  {turns} {pairing}")
        print(f"    depth: {c['depth']}")
        print(f"    retention r2_loo: {c['retention']}")
        print(f"    exploitability r2_loo: {c['exploitability']}")
