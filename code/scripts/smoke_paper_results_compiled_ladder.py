#!/usr/bin/env python3
"""Smoke the paper_results compiled-histogram backend on small ladder routes.

This harness verifies the paper-results dispatch path, not just the isolated
compiled backend.  It builds one artificial corpus graph per requested minimum
hop depth, runs a tiny paper-results corpus training job for each graph, and
reports:

* whether the compiled histogram backend was actually used;
* whether hop-based rolling-reservation calibration was applied;
* cache misses for first-seen route/action histograms and hits on revisits;
* whether Alice selected the deliberately included longer candidate route;
* wall-clock time by depth.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.paper_results as paper_results
from sequence_game.corpus.compiled_histogram_backend import (
    DEFAULT_CONVERSION_RESPONSE_PATH,
    DEFAULT_DETECTOR_PROFILE_PATH,
)
from sequence_game.corpus.runner import corpus_candidate_routes


def _edge_id(u: str, v: str) -> str:
    return "-".join(sorted((str(u), str(v))))


def _write_ladder_like_manifest(
    root: Path,
    *,
    min_hops: int,
    longer_extra_hops: int,
    edge_length_m: float,
) -> Path:
    """Write a one-assignment corpus with a genuine two-rail ladder.

    The previous smoke fixture used two disjoint rails plus one longer rail.
    That was useful for cache smoke tests but did not stress Alice's route
    action set.  This fixture builds a real ladder with rung edges, so all
    every simple Alice--Bob path is enumerated by the paper-results corpus
    runner.
    """

    graph_id = f"compiled_ladder_min_h{min_hops}"
    assignment_id = f"a_min_h{min_hops}"
    graph_dir = root / "accepted" / graph_id
    assignment_dir = graph_dir / "assignments"
    assignment_dir.mkdir(parents=True, exist_ok=True)
    alice = "alice"
    bob = "bob"
    nodes: dict[str, dict[str, Any]] = {
        alice: {"node_id": alice, "roles": ["alice"], "coordinates": [0.0, 0.0]},
        bob: {"node_id": bob, "roles": ["bob"], "coordinates": [float(min_hops), 0.0]},
    }
    edges_by_id: dict[str, dict[str, Any]] = {}

    def add_node(node_id: str, x: float, y: float) -> None:
        nodes.setdefault(
            node_id,
            {
                "node_id": node_id,
                "roles": ["swap_candidate"],
                "coordinates": [float(x), float(y)],
            },
        )

    def add_edge(u: str, v: str) -> None:
        edge_id = _edge_id(u, v)
        edges_by_id.setdefault(
            edge_id,
            {
                "edge_id": edge_id,
                "u": str(u),
                "v": str(v),
                "length_m": float(edge_length_m),
                "eve_eligible": True,
            },
        )

    if min_hops == 1:
        add_edge(alice, bob)
        add_node("alt2_1", 0.5, 1.0)
        add_edge(alice, "alt2_1")
        add_edge("alt2_1", bob)
        add_node("alt3_1", 0.33, -1.0)
        add_node("alt3_2", 0.66, -1.0)
        add_edge(alice, "alt3_1")
        add_edge("alt3_1", "alt3_2")
        add_edge("alt3_2", bob)
    else:
        columns = int(min_hops) - 1
        for idx in range(1, columns + 1):
            add_node(f"top_{idx}", idx, 1.0)
            add_node(f"bot_{idx}", idx, -1.0)
            add_edge(f"top_{idx}", f"bot_{idx}")
            if idx < columns:
                add_edge(f"top_{idx}", f"top_{idx + 1}")
                add_edge(f"bot_{idx}", f"bot_{idx + 1}")
        add_edge(alice, "top_1")
        add_edge(alice, "bot_1")
        add_edge(f"top_{columns}", bob)
        add_edge(f"bot_{columns}", bob)

    topology = {
        "nodes": nodes,
        "edges": list(edges_by_id.values()),
        "metadata": {
            "generator": "smoke_paper_results_compiled_ladder",
            "min_hops": int(min_hops),
            "topology_kind": "two_rail_ladder_with_rungs",
            "legacy_longer_extra_hops_argument": int(longer_extra_hops),
            "edge_length_m": float(edge_length_m),
        },
    }
    routes, route_selection = corpus_candidate_routes(
        topology,
        alice,
        bob,
        {"shortest_hops": int(min_hops)},
    )
    route_count_by_hops: dict[str, int] = {}
    for route in routes:
        key = str(int(route["hop_count"]))
        route_count_by_hops[key] = route_count_by_hops.get(key, 0) + 1
    shortest_hops = min((int(route["hop_count"]) for route in routes), default=int(min_hops))
    longest_hops = max((int(route["hop_count"]) for route in routes), default=int(min_hops))

    topology_path = graph_dir / "topology.json"
    topology_path.write_text(json.dumps(topology, indent=2), encoding="utf-8")

    assignment = {
        "assignment_id": assignment_id,
        "graph_id": graph_id,
        "alice": alice,
        "bob": bob,
        "candidate_routes": {
            "count": len(routes),
            "truncated": bool(route_selection.get("route_enumeration_truncated", False)),
            "count_by_hops": route_count_by_hops,
        },
        "shortest_route": {"hop_count": int(shortest_hops)},
        "longest_route": {"hop_count": int(longest_hops)},
        "smallest_set_of_nodes_on_each_candidate_route": {
            "size": 0,
            "sets": [],
            "exact": True,
        },
        "smallest_set_of_edges_on_each_candidate_route": {
            "size": 0,
            "sets": [],
            "exact": True,
        },
    }
    assignment_path = assignment_dir / f"{assignment_id}.json"
    assignment_path.write_text(json.dumps(assignment, indent=2), encoding="utf-8")

    manifest = {
        "graphs": [{
            "graph_id": graph_id,
            "family": "compiled_ladder_smoke",
            "source": "synthetic",
            "scope_label": "REPEATER_RUNTIME",
            "topology_path": str(topology_path.relative_to(root)),
            "assignment_ids": [assignment_id],
            "assignment_count": 1,
            "node_count": len(nodes),
            "edge_count": len(edges_by_id),
        }],
        "assignments": [{
            "graph_id": graph_id,
            "assignment_id": assignment_id,
            "assignment_path": str(assignment_path.relative_to(root)),
            "assignment_policy": "compiled_ladder_smoke",
            "assignment_search_mode": "fixed_min_hop",
            "alice": alice,
            "bob": bob,
            "route_count": len(routes),
            "shortest_hops": int(shortest_hops),
            "longest_hops": int(longest_hops),
            "route_count_by_hops": route_count_by_hops,
            "strategy_validity": "valid_multi_route" if len(routes) > 1 else "valid_single_route",
        }],
        "counts": {"graphs": 1, "assignments": 1},
        "config": {
            "generator": "smoke_paper_results_compiled_ladder",
            "topology_kind": "two_rail_ladder_with_rungs",
            "min_hops": int(min_hops),
            "shortest_hops": int(shortest_hops),
            "longest_hops": int(longest_hops),
            "route_count": len(routes),
            "route_count_by_hops": route_count_by_hops,
            "legacy_longer_extra_hops_argument": int(longer_extra_hops),
            "edge_length_m": float(edge_length_m),
        },
        "pruned_or_skipped": [],
        "scope_label": "REPEATER_RUNTIME",
        "script": "smoke_paper_results_compiled_ladder.py",
    }
    manifest_path = root / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _compiled_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "evaluated_cache_size": int(args.evaluated_cache_size),
        "compile_key_length": int(args.compile_key_length),
        "fock_cutoff": int(args.fock_cutoff),
        "detector_kind": str(args.detector_kind),
        "detector_profile_path": args.detector_profile,
        "detector_response_path": args.detector_response_file,
        "fiber_response_path": args.fiber_response_file,
        "conversion_response_path": args.conversion_response_file,
        "allow_wavelength_mismatch": bool(args.allow_wavelength_mismatch),
        "allow_ideal_detector_reference": bool(args.allow_ideal_detector_reference),
        "require_explicit_device_stack": True,
        "rolling_hop_calibration": {
            "kind": "hop_based_10km",
            "edge_length_km": 10.0,
            "accepted_alive_time_s": float(args.accepted_alive_time_s),
            "route_availability_target": float(args.route_availability_target),
            "chsh_target": float(args.chsh_target),
            "age_points_per_edge": int(args.age_points_per_edge),
            "selection_policy": "youngest_live",
            "detector_kind": str(args.detector_kind),
            "detector_profile_path": args.detector_profile,
            "detector_response_path": args.detector_response_file,
            "fiber_response_path": args.fiber_response_file,
            "conversion_response_path": args.conversion_response_file,
            "allow_wavelength_mismatch": bool(args.allow_wavelength_mismatch),
            "allow_ideal_detector_reference": bool(args.allow_ideal_detector_reference),
            "fock_cutoff": int(args.fock_cutoff),
            "c_max": int(args.c_max),
        },
    }


def _combo(args: argparse.Namespace) -> dict[str, str]:
    return {
        "game_mode": "attack_surface_all",
        "eve": str(args.eve_algo),
        "alice": str(args.alice_algo),
        "group": "smoke",
        "key": f"attack_surface_all__{args.eve_algo}__{args.alice_algo}",
    }


def _route_choice_summary(rows: Sequence[dict[str, Any]], min_hops: int) -> dict[str, Any]:
    by_hop: dict[str, int] = {}
    total = 0
    longer = 0
    for row in rows:
        features = row.get("features") or []
        route_hops = int(round(float(features[0]))) if features else None
        chosen = int(row.get("chosen_count") or 0)
        total += chosen
        if route_hops is not None:
            by_hop[str(route_hops)] = by_hop.get(str(route_hops), 0) + chosen
            if route_hops > int(min_hops):
                longer += chosen
    return {
        "chosen_count_total": total,
        "chosen_count_by_route_hops": by_hop,
        "longer_route_chosen_count": longer,
        "longer_route_seen": longer > 0,
    }


def _summarize_depth(depth_dir: Path, combo_key: str, *, min_hops: int, wall_seconds: float) -> dict[str, Any]:
    per_graph = json.loads((depth_dir / "data" / "corpus_per_graph.json").read_text())
    checkpoint = json.loads((depth_dir / "data" / "corpus_checkpoint.json").read_text())
    manifest = json.loads((depth_dir.parent / "corpus" / "corpus_manifest.json").read_text())
    row = per_graph[0]
    checkpoint_row = checkpoint[0]
    runtime_metrics = row["combo_results"][combo_key]["runtime_metrics"]
    backend = runtime_metrics["compiled_histogram_backend"]
    stats = backend.get("stats", {})
    delta = backend.get("stats_delta", {})
    route_choice = _route_choice_summary(
        checkpoint_row.get("route_choice_rows", []),
        min_hops,
    )
    hop_keys = sorted(int(key) for key in backend.get("hop_calibrations", {}))
    return {
        "min_hops": int(min_hops),
        "graph_id": row["graph_id"],
        "num_routes": int(row["num_routes"]),
        "graph_shortest_hops": int(row["graph_S"]),
        "graph_longest_hops": int(row["graph_L"]),
        "num_actions": int(row["num_actions"]),
        "route_count_by_hops": manifest.get("config", {}).get("route_count_by_hops", {}),
        "steps": int(row["num_steps"]),
        "wall_seconds": float(wall_seconds),
        "mean_step_ms_wall": 1000.0 * float(wall_seconds) / max(1, int(row["num_steps"])),
        "backend": runtime_metrics["backend"],
        "compiled_histogram_runtime_executed": bool(
            runtime_metrics["compiled_histogram_runtime_executed"]
        ),
        "sequence_runtime_executed": bool(runtime_metrics["sequence_runtime_executed"]),
        "backend_pool_enabled": bool(backend.get("backend_pool_enabled")),
        "backend_count": int(backend.get("backend_count", 0)),
        "calibrated_hops_observed": hop_keys,
        "stats": stats,
        "stats_delta": delta,
        "cache_reused": (
            float(delta.get("kernel_cache_hits", 0.0)) > 0.0
            and float(delta.get("evaluated_cache_hits", 0.0)) > 0.0
        ),
        "route_choice": route_choice,
        "public_outcome_counts": runtime_metrics.get("public_outcome_counts", {}),
        "eve_attack_selection_counts": runtime_metrics.get("eve_attack_selection_counts", {}),
        "qber": runtime_metrics.get("qber"),
        "route_entropy": runtime_metrics.get("route_entropy"),
        "route_switches": runtime_metrics.get("route_switches"),
        "calibrations": backend.get("hop_calibrations", {}),
    }


def _markdown(summary: dict[str, Any]) -> str:
    rows = summary["depth_rows"]
    headers = [
        "min h",
        "routes",
        "route h",
        "actions",
        "steps",
        "wall s",
        "ms/step",
        "compiles",
        "k hits",
        "eval misses",
        "eval hits",
        "longer chosen",
        "cal h",
        "route counts",
    ]
    lines = [
        "# Compiled Histogram Paper-Results Ladder Smoke",
        "",
        "This smoke uses `paper_results.run_corpus_pipeline` with the compiled histogram runtime backend and hop-based 10 km rolling-reservation calibration.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        delta = row["stats_delta"]
        route_choice = row["route_choice"]
        lines.append(
            "| "
            + " | ".join([
                str(row["min_hops"]),
                str(row["num_routes"]),
                f"{row['graph_shortest_hops']}..{row['graph_longest_hops']}",
                str(row["num_actions"]),
                str(row["steps"]),
                f"{row['wall_seconds']:.3f}",
                f"{row['mean_step_ms_wall']:.3f}",
                str(int(delta.get("kernel_compiles", 0))),
                str(int(delta.get("kernel_cache_hits", 0))),
                str(int(delta.get("evaluated_cache_misses", 0))),
                str(int(delta.get("evaluated_cache_hits", 0))),
                str(route_choice["longer_route_chosen_count"]),
                ",".join(str(item) for item in row["calibrated_hops_observed"]),
                ",".join(
                    f"{hop}:{count}"
                    for hop, count in sorted(
                        row.get("route_count_by_hops", {}).items(),
                        key=lambda item: int(item[0]),
                    )
                ),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Contract Checks",
        "",
        f"- compiled backend used for all rows: `{summary['checks']['compiled_backend_used_all']}`",
        f"- SeQUeNCe runtime bypassed for all rows: `{summary['checks']['sequence_runtime_bypassed_all']}`",
        f"- hop calibration observed for all selected route depths: `{summary['checks']['hop_calibration_observed_all']}`",
        f"- cache hits observed for all depths: `{summary['checks']['cache_reused_all']}`",
        f"- at least one longer route selected: `{summary['checks']['longer_route_selected_any']}`",
        "",
    ])
    return "\n".join(lines)


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    combo = _combo(args)
    config = _compiled_config(args)
    depth_rows: list[dict[str, Any]] = []
    for min_hops in range(int(args.min_hops), int(args.max_min_hops) + 1):
        depth_root = run_root / f"h{min_hops:02d}"
        corpus_root = depth_root / "corpus"
        manifest_path = _write_ladder_like_manifest(
            corpus_root,
            min_hops=min_hops,
            longer_extra_hops=int(args.longer_extra_hops),
            edge_length_m=float(args.edge_length_m),
        )
        out_dir = depth_root / "paper_results"
        started = time.perf_counter()
        paper_results.run_corpus_pipeline(
            str(manifest_path),
            models=[],
            out_dir=out_dir,
            ts=f"compiled_ladder_h{min_hops}",
            max_episodes=int(args.steps),
            seed=int(args.seed) + min_hops,
            no_llm=True,
            workers=1,
            checkpoint_every=1,
            combos=[combo],
            runtime_backend=paper_results.RUNTIME_BACKEND_COMPILED_HISTOGRAM,
            compiled_histogram_backend_config=config,
            assignment_selection=paper_results.ASSIGNMENT_SELECTION_ALL,
            experiment_suite_id="compiled_ladder_smoke",
            write_json_checkpoint=True,
        )
        wall = time.perf_counter() - started
        depth_rows.append(
            _summarize_depth(
                out_dir,
                combo["key"],
                min_hops=min_hops,
                wall_seconds=wall,
            )
        )

    checks = {
        "compiled_backend_used_all": all(
            row["compiled_histogram_runtime_executed"] and row["backend"] == paper_results.COMPILED_HISTOGRAM_RUNTIME_BACKEND
            for row in depth_rows
        ),
        "sequence_runtime_bypassed_all": all(
            not row["sequence_runtime_executed"] for row in depth_rows
        ),
        "hop_calibration_observed_all": all(
            row["calibrated_hops_observed"] for row in depth_rows
        ),
        "cache_reused_all": all(row["cache_reused"] for row in depth_rows),
        "longer_route_selected_any": any(
            row["route_choice"]["longer_route_seen"] for row in depth_rows
        ),
    }
    summary = {
        "smoke_version": 1,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": vars(args),
        "compiled_histogram_backend_config": config,
        "combo": combo,
        "depth_rows": depth_rows,
        "checks": checks,
    }
    json_path = run_root / "compiled_ladder_smoke_summary.json"
    md_path = run_root / "compiled_ladder_smoke_summary.md"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    md_path.write_text(_markdown(summary), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="runs/compiled_histogram_ladder_smoke")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-hops", type=int, default=1)
    parser.add_argument("--max-min-hops", type=int, default=10)
    parser.add_argument("--longer-extra-hops", type=int, default=2)
    parser.add_argument("--edge-length-m", type=float, default=10_000.0)
    parser.add_argument("--alice-algo", default="q_learning")
    parser.add_argument("--eve-algo", default="q_learning")
    parser.add_argument("--evaluated-cache-size", type=int, default=4096)
    parser.add_argument("--compile-key-length", type=int, default=128)
    parser.add_argument("--detector-kind", default="pnr_snspd")
    parser.add_argument("--detector-profile", default=str(DEFAULT_DETECTOR_PROFILE_PATH))
    parser.add_argument("--detector-response-file", default=None)
    parser.add_argument("--fiber-response-file", default=None)
    parser.add_argument("--conversion-response-file", default=str(DEFAULT_CONVERSION_RESPONSE_PATH))
    parser.add_argument("--allow-wavelength-mismatch", action="store_true")
    parser.add_argument("--allow-ideal-detector-reference", action="store_true")
    parser.add_argument("--fock-cutoff", type=int, default=2)
    parser.add_argument("--accepted-alive-time-s", type=float, default=0.18)
    parser.add_argument("--route-availability-target", type=float, default=0.95)
    parser.add_argument("--chsh-target", type=float, default=2.10)
    parser.add_argument("--age-points-per-edge", type=int, default=8)
    parser.add_argument("--c-max", type=int, default=64)
    args = parser.parse_args()
    run_smoke(args)


if __name__ == "__main__":
    main()
