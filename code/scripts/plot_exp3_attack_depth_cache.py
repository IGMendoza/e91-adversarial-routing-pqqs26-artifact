"""Plot cached 7-hop Eve attack-location diagnostics.

This script is intentionally cache-only. It reads the Exp3 SeQUeNCe attack
SQLite database and never calls the SeQUeNCe backend.
"""

from __future__ import annotations

import argparse
import csv
import os
import sqlite3
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_DB = Path(
    "sequence_game/experiments/exp3_sequence/attack_baselines.sqlite"
)
DEFAULT_OUT_DIR = Path("runs/exp3_sequence_attack_cache_check/figures")


@dataclass(frozen=True)
class AttackDepthRow:
    attack_kind: str
    target_kind: str
    target_index: int
    target_id: str
    sample_index: int
    signed_chsh_s: float
    abs_chsh_s: float
    qber: float

    @property
    def depth(self) -> int:
        return self.target_index + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot CHSH/QBER by Eve attack depth from cached attack samples."
        )
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--hop-count", type=int, default=7)
    return parser.parse_args()


def load_rows(db_path: Path, hop_count: int) -> list[AttackDepthRow]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT
                attack_kind,
                target_kind,
                target_index,
                target_id,
                sample_index,
                chsh_s,
                qber
            FROM attack_samples
            WHERE hop_count = ?
              AND chsh_s IS NOT NULL
              AND qber IS NOT NULL
            ORDER BY attack_kind, target_index, sample_index
            """,
            (hop_count,),
        ).fetchall()
    finally:
        con.close()

    return [
        AttackDepthRow(
            attack_kind=str(row[0]),
            target_kind=str(row[1]),
            target_index=int(row[2]),
            target_id=str(row[3]),
            sample_index=int(row[4]),
            signed_chsh_s=float(row[5]),
            abs_chsh_s=abs(float(row[5])),
            qber=float(row[6]),
        )
        for row in rows
    ]


def summarize(rows: list[AttackDepthRow]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str, str], list[AttackDepthRow]] = defaultdict(list)
    for row in rows:
        grouped[
            (row.attack_kind, row.depth, row.target_kind, row.target_id)
        ].append(row)

    summary = []
    for (attack_kind, depth, target_kind, target_id), bucket in sorted(grouped.items()):
        signed_s = np.asarray([row.signed_chsh_s for row in bucket], dtype=float)
        abs_s = np.asarray([row.abs_chsh_s for row in bucket], dtype=float)
        qber = np.asarray([row.qber for row in bucket], dtype=float)
        summary.append({
            "attack_kind": attack_kind,
            "depth": depth,
            "target_kind": target_kind,
            "target_id": target_id,
            "samples": len(bucket),
            "mean_signed_chsh_s": float(np.mean(signed_s)),
            "sd_signed_chsh_s": float(np.std(signed_s, ddof=1)) if len(bucket) > 1 else 0.0,
            "mean_abs_chsh_s": float(np.mean(abs_s)),
            "sd_abs_chsh_s": float(np.std(abs_s, ddof=1)) if len(bucket) > 1 else 0.0,
            "mean_qber": float(np.mean(qber)),
            "sd_qber": float(np.std(qber, ddof=1)) if len(bucket) > 1 else 0.0,
        })
    return summary


def write_summary_csv(path: Path, summary: list[dict[str, object]]) -> None:
    if not summary:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def plot(rows: list[AttackDepthRow], summary: list[dict[str, object]],
         out_path: Path, hop_count: int) -> None:
    os.environ["MPLCONFIGDIR"] = str(
        Path(tempfile.gettempdir()) / "sequence_matplotlib"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "edge_intercept_resend": "#d45087",
        "memory_degradation": "#2f7ed8",
    }
    labels = {
        "edge_intercept_resend": "intercept-resend edge",
        "memory_degradation": "memory degradation node",
    }
    metrics = [
        ("signed_chsh_s", "mean_signed_chsh_s", "sd_signed_chsh_s", "CHSH S"),
        ("abs_chsh_s", "mean_abs_chsh_s", "sd_abs_chsh_s", "|CHSH S|"),
        ("qber", "mean_qber", "sd_qber", "QBER"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(9.4, 8.4), sharex=True)
    for ax, (row_attr, mean_key, sd_key, ylabel) in zip(axes, metrics):
        for attack_kind in sorted({row.attack_kind for row in rows}):
            color = colors.get(attack_kind, "#333333")
            attack_rows = [row for row in rows if row.attack_kind == attack_kind]
            attack_summary = [
                row for row in summary if row["attack_kind"] == attack_kind
            ]
            depths = np.asarray(
                [int(row["depth"]) for row in attack_summary],
                dtype=float,
            )
            means = np.asarray(
                [float(row[mean_key]) for row in attack_summary],
                dtype=float,
            )
            sds = np.asarray(
                [float(row[sd_key]) for row in attack_summary],
                dtype=float,
            )
            if len(depths) == 0:
                continue
            x_raw = np.asarray([row.depth for row in attack_rows], dtype=float)
            y_raw = np.asarray([getattr(row, row_attr) for row in attack_rows],
                               dtype=float)
            jitter = np.asarray([
                ((row.sample_index % 7) - 3) * 0.018
                for row in attack_rows
            ])
            ax.scatter(
                x_raw + jitter,
                y_raw,
                s=16,
                color=color,
                alpha=0.25,
                linewidths=0,
            )
            ax.fill_between(
                depths,
                means - sds,
                means + sds,
                color=color,
                alpha=0.13,
                linewidth=0,
            )
            ax.plot(
                depths,
                means,
                marker="o",
                markersize=5,
                linewidth=2.0,
                color=color,
                label=f"{labels.get(attack_kind, attack_kind)} (band=1 sd)",
            )
        ax.set_ylabel(ylabel)
        ax.grid(color="#d8d8d8", linewidth=0.7, alpha=0.7)
        ax.legend(fontsize=8, frameon=False, loc="best")

    axes[0].axhline(0.0, color="#202124", linestyle=":", linewidth=1.0)
    axes[1].axhline(2.0, color="#202124", linestyle=":", linewidth=1.2)
    axes[2].axhline(0.15, color="#202124", linestyle=":", linewidth=1.2)
    axes[0].set_title(
        f"Cached {hop_count}-hop attack-location diagnostics "
        "(SQLite attack samples only)"
    )
    axes[2].set_xlabel("attack depth along route (1-based edge/node index)")
    axes[2].set_xticks(range(1, hop_count + 1))
    axes[1].set_ylim(bottom=0.0)
    axes[2].set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.db, args.hop_count)
    if not rows:
        raise SystemExit(
            f"no cached rows found in {args.db} for hop_count={args.hop_count}"
        )

    summary = summarize(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / f"attack_depth_hop{args.hop_count}_chsh_qber.png"
    csv_path = args.out_dir / f"attack_depth_hop{args.hop_count}_summary.csv"
    plot(rows, summary, png_path, args.hop_count)
    write_summary_csv(csv_path, summary)

    counts = defaultdict(int)
    for row in rows:
        counts[row.attack_kind] += 1
    print(f"read cached rows from {args.db}")
    for attack_kind in sorted(counts):
        print(f"{attack_kind}: {counts[attack_kind]} rows")
    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
