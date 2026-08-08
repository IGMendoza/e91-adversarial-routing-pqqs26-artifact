#!/usr/bin/env python3
"""Single-corpus-assignment DQN benchmark with persisted DT artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.paper_results import _corpus_records_from_manifest
from sequence_game.corpus.runner import (
    ROUTE_CHOICE_FEATURE_NAMES,
    STRUCTURAL_FEATURE_NAMES,
    corpus_candidate_routes,
    ir_to_demo_topology,
    per_graph_route_choice_rows,
    per_graph_structural_rows,
)
from sequence_game.gui_demo.demo_runner import _route_corridor_attack_scope, run_binary_collision
from sequence_game.xai.cross_graph_dt import fit_cross_graph_dt
from sequence_game.xai.dt_refit_observer import (
    DynamicDTRefitConfig,
    DynamicDTRefitObserver,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _dt_payload(dt_result: Any) -> dict[str, Any]:
    importances = [
        {"feature": feature, "importance": round(float(importance), 6)}
        for feature, importance in zip(
            dt_result.feature_names,
            dt_result.dt.feature_importances_,
        )
        if float(importance) > 0.0
    ]
    importances.sort(key=lambda row: row["importance"], reverse=True)
    return {
        "target_key": dt_result.target_key,
        "num_rows": dt_result.num_rows,
        "num_topologies": dt_result.num_topologies,
        "r2_score": dt_result.r2_score,
        "feature_names": dt_result.feature_names,
        "feature_importances": importances,
        "rules_text": dt_result.rules_text,
    }


def _write_timing_artifacts(
    out_dir: Path,
    tick_rows: list[dict[str, float]],
    *,
    refit_steps: list[int] | None = None,
) -> None:
    csv_path = out_dir / "time_per_50_ticks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["step_tick", "last_5_steps_s", "last_5_avg_step_s"],
        )
        writer.writeheader()
        writer.writerows(tick_rows)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    ax.plot(
        [row["step_tick"] for row in tick_rows],
        [row["last_5_steps_s"] for row in tick_rows],
        color="#2563eb",
        linewidth=1.4,
    )
    for refit_step in refit_steps or []:
        ax.axvline(refit_step, color="#dc2626", alpha=0.2, linewidth=0.8)
    ax.set_title("DQN vs DQN single-graph benchmark: last-5-step wall time")
    ax.set_xlabel("Training step tick")
    ax.set_ylabel("Wall time for last 5 steps (s)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "time_per_50_ticks.png")
    plt.close(fig)


def _dt_config_from_args(args: argparse.Namespace) -> DynamicDTRefitConfig:
    return DynamicDTRefitConfig(
        mode=args.dt_refit_mode,
        observer_interval=args.dt_observer_interval,
        fixed_refit_interval=args.dt_fixed_refit_interval,
        epsilon=args.dt_epsilon,
        delta=args.dt_delta,
        policy_drift_threshold=args.dt_policy_drift_threshold,
        min_refit_gap=args.dt_min_refit_gap,
        max_refit_gap=args.dt_max_refit_gap,
        log_snapshot_count=args.dt_log_snapshot_count,
        probe_state_count=args.dt_probe_state_count,
        final_step_refit=args.dt_final_step_refit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="runs/graph_corpus_snap_as_733_top2/corpus_manifest.json",
        help="Corpus manifest path.",
    )
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt-depth", type=int, default=4)
    parser.add_argument(
        "--dt-refit-mode",
        choices=["fixed", "logarithmic", "dynamic", "off"],
        default="dynamic",
        help="DT explanation refit schedule. fixed preserves legacy/debug every-k behavior.",
    )
    parser.add_argument("--dt-observer-interval", type=int, default=50)
    parser.add_argument("--dt-fixed-refit-interval", type=int, default=50)
    parser.add_argument("--dt-epsilon", type=float, default=0.10)
    parser.add_argument("--dt-delta", type=float, default=0.05)
    parser.add_argument("--dt-policy-drift-threshold", type=float, default=0.05)
    parser.add_argument("--dt-min-refit-gap", type=int, default=None)
    parser.add_argument("--dt-max-refit-gap", type=int, default=2000)
    parser.add_argument("--dt-log-snapshot-count", type=int, default=8)
    parser.add_argument("--dt-probe-state-count", type=int, default=2048)
    parser.add_argument(
        "--dt-final-step-refit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out-dir",
        default="results/benchmarks/snap_as_733_top2_first_5hop_dqn_dqn_10k_with_dt",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_start = time.perf_counter()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = _ROOT / manifest_path
    records = _corpus_records_from_manifest(json.loads(manifest_path.read_text()))
    record = records[args.record_index]
    topology_path = Path(record["topology_path"])
    assignment_path = Path(record["assignment_path"])
    if not topology_path.is_absolute():
        topology_path = _ROOT / topology_path
    if not assignment_path.is_absolute():
        assignment_path = _ROOT / assignment_path

    ir_dict = json.loads(topology_path.read_text())
    assignment = json.loads(assignment_path.read_text())
    demo_topology = ir_to_demo_topology(ir_dict)

    enum_start = time.perf_counter()
    routes, route_meta = corpus_candidate_routes(
        ir_dict,
        record["alice"],
        record["bob"],
        assignment,
    )
    enum_s = time.perf_counter() - enum_start
    _write_json(out_dir / "routes.json", routes)
    eve_corridor_nodes, eve_corridor_edges = _route_corridor_attack_scope(
        routes,
        record["alice"],
        record["bob"],
    )
    eve_action_count = len(eve_corridor_nodes) + len(eve_corridor_edges)

    step_durations: list[float] = []
    tick_rows: list[dict[str, float]] = []
    dt_config = _dt_config_from_args(args)
    dt_observer = DynamicDTRefitObserver(
        dt_config,
        total_steps=args.steps,
        alice_action_count=len(routes),
        eve_action_count=eve_action_count,
    )
    snapshot_dir = out_dir / "dt_snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    latest_dt_payloads: dict[str, Any] = {"eve": None, "alice": None}

    def _step_timing(step_idx: int, elapsed_s: float) -> None:
        step_durations.append(elapsed_s)
        step_num = step_idx + 1
        if step_num % 50 == 0:
            last_5_s = sum(step_durations[-5:])
            tick_rows.append({
                "step_tick": step_num,
                "last_5_steps_s": last_5_s,
                "last_5_avg_step_s": last_5_s / 5.0,
            })
            if step_num % 1000 == 0:
                print(json.dumps({
                    "progress_step": step_num,
                    "last_5_steps_s": round(last_5_s, 6),
                    "ticks_recorded": len(tick_rows),
                }), flush=True)

    def _refit_dt_snapshot(event: Any, context: dict[str, Any]) -> None:
        fit_start = time.perf_counter()
        step = int(context["step"])
        eve_q = context["eve_q"].to_dict()
        alice_q = context["alice_q"].to_dict()
        action_names = [action["id"] for action in context["actions"]]
        current_steps = list(context["steps"])
        combo = {
            "game_mode": "attack_surface_dos",
            "eve": "deep_q",
            "alice": "deep_q",
            "key": "attack_surface_dos__deep_q__deep_q",
            "group": "single_graph_benchmark",
        }
        family = str(record.get("family") or "unknown")
        eve_rows_for_dt = per_graph_structural_rows(
            action_names,
            eve_q,
            routes,
            assignment,
            str(record["graph_id"]),
            family,
            combo=combo,
        )
        alice_rows_for_dt = per_graph_route_choice_rows(
            routes,
            current_steps,
            assignment,
            str(record["graph_id"]),
            family,
            combo=combo,
        )
        output_paths: list[str] = []
        leaf_count = 0
        if eve_rows_for_dt:
            eve_dt = fit_cross_graph_dt(
                eve_rows_for_dt,
                max_depth=args.dt_depth,
                target_key="q_rank",
                feature_names=STRUCTURAL_FEATURE_NAMES,
            )
            eve_payload = _dt_payload(eve_dt)
            latest_dt_payloads["eve"] = eve_payload
            eve_path = snapshot_dir / f"refit_{step:05d}_eve_attack_choice_dt.json"
            eve_rules_path = snapshot_dir / f"refit_{step:05d}_eve_attack_choice_dt_rules.txt"
            _write_json(eve_path, eve_payload)
            eve_rules_path.write_text(eve_dt.rules_text, encoding="utf-8")
            output_paths.extend([str(eve_path), str(eve_rules_path)])
            leaf_count += int(eve_dt.dt.get_n_leaves())
        if alice_rows_for_dt:
            alice_dt = fit_cross_graph_dt(
                alice_rows_for_dt,
                max_depth=args.dt_depth,
                target_key="choice_rate",
                feature_names=ROUTE_CHOICE_FEATURE_NAMES,
            )
            alice_payload = _dt_payload(alice_dt)
            latest_dt_payloads["alice"] = alice_payload
            alice_path = snapshot_dir / f"refit_{step:05d}_alice_route_choice_dt.json"
            alice_rules_path = snapshot_dir / f"refit_{step:05d}_alice_route_choice_dt_rules.txt"
            _write_json(alice_path, alice_payload)
            alice_rules_path.write_text(alice_dt.rules_text, encoding="utf-8")
            output_paths.extend([str(alice_path), str(alice_rules_path)])
            leaf_count += int(alice_dt.dt.get_n_leaves())
        event.record_fit(
            wall_clock_seconds=time.perf_counter() - fit_start,
            training_sample_count=len(eve_rows_for_dt) + len(alice_rows_for_dt),
            dt_depth=args.dt_depth,
            dt_leaf_count=leaf_count,
            output_paths=output_paths,
        )

    train_start = time.perf_counter()
    result = run_binary_collision(
        demo_topology,
        record["alice"],
        record["bob"],
        eve_algo="deep_q",
        alice_algo="deep_q",
        control_model="attack_surface_dos",
        num_steps=args.steps,
        seed=args.seed,
        route_candidates=routes,
        step_timing_callback=_step_timing,
        include_alice_q_table=True,
        dt_refit_observer=dt_observer,
        dt_refit_callback=_refit_dt_snapshot,
    )
    train_s = time.perf_counter() - train_start

    summary = result["summary"]
    q_tables = summary["q_tables"]
    metrics = summary["metrics"]
    eve_q = q_tables["eve"]["raw"]
    alice_q = q_tables["alice"]["raw"]

    _write_json(out_dir / "eve_q_table.json", eve_q)
    _write_json(out_dir / "eve_action_names.json", q_tables["eve"]["action_names"])
    _write_json(out_dir / "alice_q_table.json", alice_q)
    _write_json(out_dir / "alice_action_names.json", q_tables["alice"]["action_names"])
    _write_jsonl(out_dir / "steps.jsonl", result["steps"])

    combo = {
        "game_mode": "attack_surface_dos",
        "eve": "deep_q",
        "alice": "deep_q",
        "key": "attack_surface_dos__deep_q__deep_q",
        "group": "single_graph_benchmark",
    }
    family = str(record.get("family") or "unknown")
    eve_rows = per_graph_structural_rows(
        q_tables["eve"]["action_names"],
        eve_q,
        routes,
        assignment,
        str(record["graph_id"]),
        family,
        combo=combo,
    )
    alice_rows = per_graph_route_choice_rows(
        routes,
        result["steps"],
        assignment,
        str(record["graph_id"]),
        family,
        combo=combo,
    )
    _write_jsonl(out_dir / "eve_structural_rows.jsonl", eve_rows)
    _write_jsonl(out_dir / "alice_route_choice_rows.jsonl", alice_rows)

    if latest_dt_payloads["eve"] is None and eve_rows and args.dt_final_step_refit:
        eve_dt = fit_cross_graph_dt(
            eve_rows,
            max_depth=args.dt_depth,
            target_key="q_rank",
            feature_names=STRUCTURAL_FEATURE_NAMES,
        )
        latest_dt_payloads["eve"] = _dt_payload(eve_dt)
    if latest_dt_payloads["alice"] is None and alice_rows and args.dt_final_step_refit:
        alice_dt = fit_cross_graph_dt(
            alice_rows,
            max_depth=args.dt_depth,
            target_key="choice_rate",
            feature_names=ROUTE_CHOICE_FEATURE_NAMES,
        )
        latest_dt_payloads["alice"] = _dt_payload(alice_dt)
    if latest_dt_payloads["eve"] is not None:
        _write_json(out_dir / "eve_attack_choice_dt.json", latest_dt_payloads["eve"])
        (out_dir / "eve_attack_choice_dt_rules.txt").write_text(
            latest_dt_payloads["eve"]["rules_text"],
            encoding="utf-8",
        )
    if latest_dt_payloads["alice"] is not None:
        _write_json(out_dir / "alice_route_choice_dt.json", latest_dt_payloads["alice"])
        (out_dir / "alice_route_choice_dt_rules.txt").write_text(
            latest_dt_payloads["alice"]["rules_text"],
            encoding="utf-8",
        )

    dt_observer.write_outputs(out_dir)
    observer_summary = dt_observer.summary()
    _write_timing_artifacts(out_dir, tick_rows, refit_steps=observer_summary["refit_steps"])
    last_5_values = [row["last_5_steps_s"] for row in tick_rows]
    last_5_mean = sum(last_5_values) / len(last_5_values) if last_5_values else None
    artifact_summary = {
        "benchmark": "single_graph_5hop_deep_q_vs_deep_q_attack_surface_dos_with_dt",
        "graph_id": record["graph_id"],
        "assignment_id": record.get("assignment_id"),
        "alice": record["alice"],
        "bob": record["bob"],
        "shortest_hops": record.get("shortest_hops"),
        **route_meta,
        "route_hops_seen": sorted({route["hop_count"] for route in routes}),
        "alice_route_actions": len(routes),
        "eve_actions": len(q_tables["eve"]["action_names"]),
        "eve_corridor_internal_nodes": metrics.get("attack_surface_node_count"),
        "eve_corridor_edges": metrics.get("attack_surface_edge_count"),
        "num_steps": args.steps,
        "alice_dqn_update_calls": args.steps,
        "eve_dqn_update_calls": args.steps,
        "dqn_refit_checkpoints_per_agent": args.steps // 50,
        "dt_refit_mode": args.dt_refit_mode,
        "dt_refit_count": observer_summary["refit_count"],
        "fixed_50_dt_refit_count_equivalent": observer_summary[
            "fixed_50_refit_count_equivalent"
        ],
        "estimated_dt_refits_avoided": observer_summary["estimated_refits_avoided"],
        "route_enumeration_wall_s": enum_s,
        "training_wall_s": train_s,
        "pipeline_wall_s_excluding_artifact_write": time.perf_counter() - pipeline_start,
        "tick_count": len(tick_rows),
        "last_5_steps_s_mean": last_5_mean,
        "last_5_steps_s_min": min(last_5_values) if last_5_values else None,
        "last_5_steps_s_max": max(last_5_values) if last_5_values else None,
        "collision_rate": metrics.get("collision_rate"),
        "route_entropy": metrics.get("route_entropy"),
        "route_switches": metrics.get("route_switches"),
        "eve_dt": (
            {
                "r2_score": latest_dt_payloads["eve"]["r2_score"],
                "num_rows": latest_dt_payloads["eve"]["num_rows"],
                "top_features": latest_dt_payloads["eve"]["feature_importances"][:8],
            }
            if latest_dt_payloads["eve"] is not None else None
        ),
        "alice_route_dt": (
            {
                "r2_score": latest_dt_payloads["alice"]["r2_score"],
                "num_rows": latest_dt_payloads["alice"]["num_rows"],
                "top_features": latest_dt_payloads["alice"]["feature_importances"][:8],
            }
            if latest_dt_payloads["alice"] is not None else None
        ),
        "dt_refit_summary": observer_summary,
        "no_convergence_claim": True,
    }
    _write_json(out_dir / "summary.json", artifact_summary)

    slim_summary = dict(summary)
    slim_summary["q_tables"] = {
        "eve": {k: v for k, v in q_tables["eve"].items() if k != "raw"},
        "alice": {k: v for k, v in q_tables["alice"].items() if k != "raw"},
    }
    _write_json(out_dir / "run_summary_without_raw_q.json", slim_summary)

    print(json.dumps({
        "summary": artifact_summary,
        "out_dir": str(out_dir),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
