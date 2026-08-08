"""Merge partial benchmark runs into a single report per query state.

Usage:
    python scripts/merge_xai_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

XAI_DIR = Path(__file__).parent.parent / "results" / "xai"


def merge_model_results(primary: dict, *patches: dict) -> dict:
    """Take primary result, overwrite errored models with results from patches."""
    by_model: dict[str, dict] = {m["model"]: m for m in primary["models"]}
    for patch in patches:
        for m in patch["models"]:
            existing = by_model.get(m["model"])
            if existing is None or existing.get("error"):
                by_model[m["model"]] = m
    merged = dict(primary)
    merged["models"] = list(by_model.values())
    return merged


def write_summary_table(results_by_state: dict[str, dict]) -> None:
    print("\n=== Faithfulness Summary Table ===")
    print(f"{'Model':<30} ", end="")
    for state in results_by_state:
        print(f"{state[:20]:<22}", end="")
    print()
    print("-" * (30 + 22 * len(results_by_state)))

    all_models = []
    for r in results_by_state.values():
        for m in r["models"]:
            if m["model"] not in all_models:
                all_models.append(m["model"])

    for model in all_models:
        print(f"{model:<30} ", end="")
        for r in results_by_state.values():
            entry = next((m for m in r["models"] if m["model"] == model), None)
            if entry is None:
                print(f"{'N/A':<22}", end="")
            elif entry.get("error"):
                print(f"{'ERROR':<22}", end="")
            else:
                score = f"{entry['faithfulness_score']:.2%} ({entry['total_numeric_claims']}n)"
                print(f"{score:<22}", end="")
        print()


def main() -> None:
    state1_primary = json.loads(
        (XAI_DIR / "benchmark_state1_no_attack_accepted.json").read_text()
    )
    state2_primary = json.loads(
        (XAI_DIR / "benchmark_state2_swap_denial_r0_failure.json").read_text()
    )

    retry_path = XAI_DIR / "benchmark_state1_retry.json"
    if retry_path.exists():
        retry = json.loads(retry_path.read_text())
        state1_merged = merge_model_results(state1_primary, retry)
    else:
        state1_merged = state1_primary

    (XAI_DIR / "state1_merged.json").write_text(
        json.dumps(state1_merged, indent=2, sort_keys=True)
    )

    results_by_state = {
        "no_attack/accepted": state1_merged,
        "swap_denial_r0/failure": state2_primary,
    }

    write_summary_table(results_by_state)

    full_report = {
        "states": {
            "no_attack_accepted": state1_merged,
            "swap_denial_r0_delivery_failure": state2_primary,
        }
    }
    out = XAI_DIR / "full_benchmark_report.json"
    out.write_text(json.dumps(full_report, indent=2, sort_keys=True))
    print(f"\nFull report saved → {out}")


if __name__ == "__main__":
    main()
