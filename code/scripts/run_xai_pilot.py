"""End-to-end XAI pilot: Q-table → DT surrogate → LLM benchmark.

Usage:
    python scripts/run_xai_pilot.py
    python scripts/run_xai_pilot.py --models llama3.2:3b qwen3:8b
    python scripts/run_xai_pilot.py --dry-run   # DT only, skip LLM queries
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results" / "repeater_runtime"
XAI_OUTPUT_DIR = ROOT / "results" / "xai"

from sequence_game.xai.surrogate import (
    RouteFeatures,
    build_dataset,
    encode_state,
    extract_decision_path,
    fit_surrogate_dt,
    format_path_text,
)
from sequence_game.xai.policy_explainer import PolicyExplanation
from sequence_game.xai.prompt_builder import CounterfactualRow, PromptContext
from sequence_game.xai.benchmark import BENCHMARK_MODELS, run_benchmark


def load_q_table_entries(path: Path) -> tuple[list[dict], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["q_table_entries"]["entries"], data["action_space"]


def load_outcome_table(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["rows"]


def build_counterfactual_rows(
    outcome_rows: list[dict],
    active_route: str,
    chosen_action: str,
) -> tuple[CounterfactualRow, list[CounterfactualRow]]:
    on_route = [r for r in outcome_rows if r["selected_route"] == active_route]
    chosen: CounterfactualRow | None = None
    alternatives: list[CounterfactualRow] = []
    for row in on_route:
        cf = CounterfactualRow(
            action=row["attack_action"],
            route=row["selected_route"],
            outcome=row["public_outcome"],
            fidelity=row.get("fidelity"),
            qber=row.get("qber"),
            quality_metric=row.get("quality_metric"),
            is_chosen=(row["attack_action"] == chosen_action),
        )
        if cf.is_chosen:
            chosen = cf
        else:
            alternatives.append(cf)
    if chosen is None:
        raise ValueError(f"chosen action {chosen_action!r} not found in outcome table for {active_route}")
    return chosen, alternatives


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="XAI pilot benchmark")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Ollama model tags to benchmark (default: all)")
    parser.add_argument("--out", default=None,
                        help="output path for benchmark JSON (default: results/xai/pilot_benchmark.json)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fit DT and print prompt only; skip LLM queries")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--query-state", nargs=2, default=["no_attack", "accepted"],
                        metavar=("PREV_ACTION", "PREV_OUTCOME"),
                        help="Q-table state to explain (default: no_attack accepted)")
    args = parser.parse_args(argv)

    print("=== XAI Pilot: Q-table → DT surrogate → LLM benchmark ===\n")

    entries, action_names = load_q_table_entries(
        RESULTS_DIR / "q_eve_two_route_training_summary.json"
    )
    outcome_rows = load_outcome_table(
        RESULTS_DIR / "two_route_runtime_outcome_table.json"
    )

    # Filter start state — uninformative (all-zero Q-values)
    entries = [e for e in entries if e["state"][0] != "start"]

    # Reference two-route topology: symmetric r0/r1
    r0 = RouteFeatures(bottleneck_score=0.5, hops=1, length_m=10_000.0, overlap_with_other=0)
    r1 = RouteFeatures(bottleneck_score=0.5, hops=1, length_m=10_000.0, overlap_with_other=0)

    print("Fitting surrogate decision tree...")
    dataset = build_dataset(entries, action_names, r0, r1)
    dt = fit_surrogate_dt(dataset, max_depth=4)
    train_fidelity = float(np.mean(dt.predict(dataset.X) == dataset.y))
    print(f"  Training fidelity: {train_fidelity:.2%}  ({len(entries)} samples)")

    prev_action, prev_outcome = args.query_state
    state_feats = encode_state(prev_action, prev_outcome)
    route_feats = [r0.bottleneck_score, r1.bottleneck_score, float(r0.hops), float(r1.hops)]
    query_X = np.array([state_feats + route_feats])

    path = extract_decision_path(dt, dataset, query_X)
    print(f"\nDecision path for state ({prev_action!r}, {prev_outcome!r}):")
    print(format_path_text(path))

    chosen, alternatives = build_counterfactual_rows(
        outcome_rows,
        active_route="route_r0",
        chosen_action=path.predicted_action,
    )

    # Script uses surrogate DT fitted on real Q-table entries
    prev_outcome = chosen.outcome if chosen.outcome not in ("counterfactual — not simulated",) else "accepted"
    pe = PolicyExplanation(
        text=format_path_text(path),
        method="surrogate_dt",
        predicted_action=path.predicted_action,
        prev_outcome_is_failure=prev_outcome in {"delivery_failure", "miss", "blocked", "qber_abort"},
        dt_fidelity=path.fidelity,
    )
    ctx = PromptContext(
        policy_explanation=pe,
        chosen_row=chosen,
        alternative_rows=alternatives,
        r0_bottleneck=r0.bottleneck_score,
        r1_bottleneck=r1.bottleneck_score,
        r0_hops=r0.hops,
        r1_hops=r1.hops,
    )

    from sequence_game.xai.prompt_builder import build_prompt
    prompt = build_prompt(ctx)

    if args.dry_run:
        print("\n--- Generated prompt (dry run) ---")
        print(prompt)
        return

    models = args.models or BENCHMARK_MODELS
    out_path = Path(args.out) if args.out else XAI_OUTPUT_DIR / "pilot_benchmark.json"
    print(f"\nRunning LLM benchmark on {len(models)} model(s)...")
    result = run_benchmark(
        ctx,
        models=models,
        temperature=args.temperature,
        output_path=out_path,
    )

    print("\n=== Summary ===")
    sorted_results = sorted(result.model_results, key=lambda r: -r.faithfulness.score)
    for mr in sorted_results:
        if mr.error:
            print(f"  {mr.model:<30} ERROR: {mr.error}")
        else:
            print(
                f"  {mr.model:<30} faithfulness={mr.faithfulness.score:.2%}"
                f"  numeric_claims={mr.faithfulness.total_numeric_claims}"
                f"  hallucinated={len(mr.faithfulness.hallucinated)}"
            )


if __name__ == "__main__":
    main()
