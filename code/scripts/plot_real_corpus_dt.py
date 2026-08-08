"""Chart: LOTO R2 vs tree depth for the real 105-graph corpus, one line per
pairing, matching the style/scope of the existing pooled-shape-dt-
generalization-tree-depth.jpg but using this paper's own corpus/backend
data instead of the 234-graph small-synthetic Exp3 pilot.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "runs" / "real_corpus_dt_results.json"
OUT_PATH = REPO_ROOT / "runs" / "real_corpus_dt_generalization_by_depth.png"

LABELS = {
    "linear_goptimal_eta2_vs_tsallis_rv_eta1": "LinearExp2 (Alice) vs Tsallis-INF (Eve)",
    "tsallis_rv_eta1_vs_tsallis_rv_eta1": "Tsallis-INF vs Tsallis-INF",
}
COLORS = {
    "linear_goptimal_eta2_vs_tsallis_rv_eta1": "#2a78d6",  # blue, palette slot 1
    "tsallis_rv_eta1_vs_tsallis_rv_eta1": "#eb6834",       # orange, palette slot 2
}
MARKERS = {
    "linear_goptimal_eta2_vs_tsallis_rv_eta1": "o",
    "tsallis_rv_eta1_vs_tsallis_rv_eta1": "s",
}

results = json.loads(RESULTS_PATH.read_text())

fig, ax = plt.subplots(figsize=(5.2, 3.6), dpi=200)

depths = [1, 2, 3, 4, 6, 8]
for pairing_label, data in results.items():
    r2s = [data["depth_results"][str(d)]["loto_r2"] for d in depths]
    ax.plot(
        depths,
        r2s,
        marker=MARKERS[pairing_label],
        color=COLORS[pairing_label],
        linewidth=2,
        markersize=6,
        label=LABELS[pairing_label],
    )

ax.axhline(0.0, color="#767e8a", linewidth=1, linestyle=":", zorder=0)
ax.set_xlabel("Tree depth")
ax.set_ylabel("Leave-one-topology-out $R^2$ (exploitability)")
ax.set_xticks(depths)
ax.set_title("Real 105-graph corpus, $n=105$ per pairing", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.grid(axis="y", color="#dde1e6", linewidth=0.6, zorder=-1)

fig.tight_layout()
fig.savefig(OUT_PATH)
print(f"wrote {OUT_PATH}")
