#!/usr/bin/env python3
"""Slim stage benchmark for one synthetic paper corpus case.

This script profiles an existing case execution by wrapping public module
boundaries at runtime. It does not change game dynamics, Q update math,
SeQUeNCe runtime behavior, attack definitions, graph generation, DT fitting,
or LLM prompts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "sequence_matplotlib"))

import sequence_game.corpus.e91_runtime_game as e91_runtime
import sequence_game.corpus.runner as corpus_runner
import sequence_game.protocol.repeater_trial as repeater_trial
import sequence_game.sequence_build.repeater_e91_builder as repeater_builder
from sequence.app.e91_endpoint import E91EndpointApp
from sequence.kernel.timeline import Timeline
from sequence.topology.router_net_topo import RouterNetTopo
from sequence_game.rl.deep_q_learning import DeepQNetwork
from sequence_game.rl.q_learning import QTable
from sequence_game.xai.cross_graph_dt import fit_cross_graph_dt
from sequence_game.xai.dt_refit_observer import DynamicDTRefitConfig, DynamicDTRefitObserver


PUBLICATION_COMBOS: dict[str, dict[str, str]] = {}
for _combo in [
    {
        "game_mode": "attack_surface_all",
        "eve": "random",
        "alice": "random",
        "group": "reference",
    },
    {
        "game_mode": "attack_surface_all",
        "eve": "deep_q",
        "alice": "random",
        "group": "reference",
    },
    {
        "game_mode": "attack_surface_all",
        "eve": "route_aware_oracle",
        "alice": "cautious_greedy",
        "group": "reference",
    },
    {
        "game_mode": "attack_surface_all",
        "eve": "random",
        "alice": "cautious_greedy",
        "group": "reference",
    },
    {
        "game_mode": "attack_surface_all",
        "eve": "deep_q",
        "alice": "cautious_greedy",
        "group": "main",
    },
    {
        "game_mode": "attack_surface_all",
        "eve": "deep_q",
        "alice": "deep_q",
        "group": "main",
    },
]:
    _combo["key"] = f"{_combo['game_mode']}__{_combo['eve']}__{_combo['alice']}"
    PUBLICATION_COMBOS[_combo["key"]] = _combo

LEARNED_EVE_ALGOS = frozenset({"q_learning", "deep_q"})
LEARNED_ALICE_ALGOS = frozenset({"q_learning", "deep_q"})


class StageTimer:
    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)
        self.timeline_runs: list[dict[str, Any]] = []

    @contextmanager
    def time(self, stage: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self._samples[stage].append(time.perf_counter() - started)

    def summary(self, wall_seconds: float) -> dict[str, dict[str, float | int]]:
        return {
            stage: _stage_stats(samples, wall_seconds)
            for stage, samples in sorted(
                self._samples.items(),
                key=lambda item: (-sum(item[1]), item[0]),
            )
        }

    def record_timeline_run(self, row: dict[str, Any]) -> None:
        self.timeline_runs.append(row)


def _stage_stats(samples: list[float], wall_seconds: float) -> dict[str, float | int]:
    total = sum(samples)
    count = len(samples)
    return {
        "count": count,
        "total_s": round(total, 6),
        "pct_wall_inclusive": round((100.0 * total / wall_seconds), 3)
        if wall_seconds > 0.0
        else 0.0,
        "mean_ms": round((1000.0 * total / count), 3) if count else 0.0,
        "p50_ms": round(1000.0 * _percentile(samples, 0.50), 3),
        "p90_ms": round(1000.0 * _percentile(samples, 0.90), 3),
        "p95_ms": round(1000.0 * _percentile(samples, 0.95), 3),
        "max_ms": round(1000.0 * max(samples), 3) if samples else 0.0,
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[max(0, min(len(ordered) - 1, idx))]


def install_stage_wrappers(timer: StageTimer) -> None:
    _wrap_attr(corpus_runner, "corpus_candidate_routes", "route_enumeration", timer)
    _wrap_attr(corpus_runner, "sequence_route_diagnostics", "route_diagnostics", timer)
    _wrap_attr(e91_runtime, "e91_runtime_action_count", "runtime_action_count_build", timer)
    _wrap_attr(e91_runtime, "_run_selected_action", "sequence_selected_action_total", timer)
    _wrap_attr(e91_runtime, "build_attacked_hops", "sequence_build_attacked_hops", timer)
    _wrap_attr(e91_runtime, "make_station_factory", "sequence_make_station_factory", timer)
    _wrap_attr(e91_runtime, "run_e91_trial", "sequence_edge_run_e91_trial", timer)
    _wrap_attr(
        e91_runtime,
        "run_fixed_repeater_e91_trial",
        "sequence_repeater_run_fixed_repeater_e91_trial",
        timer,
    )
    _wrap_attr(
        repeater_trial,
        "build_fixed_repeater_e91_line",
        "repeater_build_fixed_line_total",
        timer,
    )
    _wrap_attr(
        repeater_builder,
        "build_fixed_repeater_e91_line",
        "repeater_builder_build_fixed_line_total",
        timer,
    )
    _wrap_attr(repeater_builder, "_topology_config", "repeater_topology_config_build", timer)
    _wrap_attr(repeater_builder, "_write_topology_config", "repeater_topology_config_write", timer)
    _wrap_attr(RouterNetTopo, "__init__", "sequence_router_net_topo_init", timer)
    _wrap_attr(E91EndpointApp, "start", "sequence_e91_endpoint_app_start", timer)
    _wrap_attr(Timeline, "init", "sequence_timeline_init", timer)
    _wrap_timeline_run(timer)
    _wrap_attr(e91_runtime, "compute_information_gain", "info_gain_compute", timer)
    _wrap_attr(e91_runtime, "_step_record", "step_record_public_payload", timer)

    _wrap_attr(DeepQNetwork, "update", "dqn_update_total", timer)
    _wrap_attr(DeepQNetwork, "_refit", "dqn_refit_mlp_loop", timer)
    _wrap_attr(DeepQNetwork, "to_dict", "dqn_to_dict", timer)
    _wrap_attr(QTable, "update", "qtable_update", timer)

    _wrap_attr(DynamicDTRefitObserver, "record_transition", "dt_observer_record_transition", timer)
    _wrap_attr(DynamicDTRefitObserver, "observe", "dt_observer_observe", timer)
    _wrap_attr(
        DynamicDTRefitObserver,
        "refit_event_from_tick",
        "dt_observer_event_from_tick",
        timer,
    )
    _wrap_attr(DynamicDTRefitObserver, "mark_refit", "dt_observer_mark_refit", timer)
    _wrap_attr(DynamicDTRefitObserver, "write_outputs", "dt_observer_write_outputs", timer)

    try:
        from sklearn.neural_network import MLPRegressor

        _wrap_attr(MLPRegressor, "fit", "sklearn_mlpregressor_fit_lbfgs", timer)
    except Exception:
        pass


def _wrap_attr(obj: Any, attr: str, stage: str, timer: StageTimer) -> None:
    original = getattr(obj, attr, None)
    if original is None or getattr(original, "_stage_benchmark_wrapped", False):
        return

    @wraps(original)
    def _wrapped(*args, **kwargs):
        with timer.time(stage):
            return original(*args, **kwargs)

    _wrapped._stage_benchmark_wrapped = True  # type: ignore[attr-defined]
    setattr(obj, attr, _wrapped)


def _wrap_timeline_run(timer: StageTimer) -> None:
    original = getattr(Timeline, "run", None)
    if original is None or getattr(original, "_stage_benchmark_wrapped", False):
        return

    @wraps(original)
    def _wrapped(self, *args, **kwargs):
        before_schedule = int(getattr(self, "schedule_counter", 0))
        before_run = int(getattr(self, "run_counter", 0))
        before_time = int(getattr(self, "time", 0))
        before_queue = len(getattr(self, "events", ()))
        started = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            timer._samples["sequence_timeline_run"].append(elapsed)
            after_schedule = int(getattr(self, "schedule_counter", 0))
            after_run = int(getattr(self, "run_counter", 0))
            after_time = int(getattr(self, "time", 0))
            after_queue = len(getattr(self, "events", ()))
            run_delta = after_run - before_run
            schedule_delta = after_schedule - before_schedule
            timer.record_timeline_run({
                "kind": _timeline_kind(self),
                "elapsed_s": round(elapsed, 6),
                "run_events": run_delta,
                "scheduled_events": schedule_delta,
                "events_per_s": round(run_delta / elapsed, 3) if elapsed > 0.0 else None,
                "initial_time_ps": before_time,
                "final_time_ps": after_time,
                "stop_time_ps": int(getattr(self, "stop_time", 0)),
                "initial_queue_len": before_queue,
                "final_queue_len": after_queue,
            })

    _wrapped._stage_benchmark_wrapped = True  # type: ignore[attr-defined]
    setattr(Timeline, "run", _wrapped)


def _timeline_kind(timeline: Timeline) -> str:
    class_names = {
        type(entity).__name__
        for entity in getattr(timeline, "entities", {}).values()
    }
    if "QuantumRouter" in class_names or "BSMNode" in class_names:
        return "repeater_router_net"
    return "edge_e91_line"


def _corpus_records_from_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("accepted"):
        return list(manifest["accepted"])

    graphs_by_id = {graph["graph_id"]: graph for graph in manifest.get("graphs", [])}
    records = []
    for assignment in manifest.get("assignments", []):
        graph = graphs_by_id.get(assignment.get("graph_id"), {})
        if not graph:
            continue
        records.append({
            **graph,
            **assignment,
            "candidate_route_count": assignment.get("route_count"),
        })
    return records


def _record_family(record: dict[str, Any]) -> str:
    family = str(record.get("family") or "unknown")
    if family != "unknown":
        return family
    graph_id = str(record.get("graph_id") or "")
    for candidate in (
        "bounded_er",
        "layered_parallel",
        "braess",
        "bottleneck",
        "diamond",
        "ladder",
    ):
        if candidate in graph_id:
            return candidate
    return family


def _route_count_hint(record: dict[str, Any]) -> int:
    for key in ("candidate_route_count", "route_count", "graph_C"):
        value = record.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    route_structure = record.get("route_structure") or {}
    routes = route_structure.get("candidate_routes") or []
    return len(routes)


def select_record(
    records: list[dict[str, Any]],
    *,
    family: str | None,
    graph_id: str | None,
    assignment_id: str | None,
    min_route_count: int,
) -> dict[str, Any]:
    candidates = records
    if graph_id:
        candidates = [record for record in candidates if record.get("graph_id") == graph_id]
    if assignment_id:
        candidates = [
            record for record in candidates
            if str(record.get("assignment_id") or "") == assignment_id
        ]
    if family:
        candidates = [
            record for record in candidates
            if _record_family(record) == family
        ]
    if not candidates:
        raise SystemExit("no corpus record matched the requested graph/family/assignment")

    preferred = [
        record for record in candidates
        if _route_count_hint(record) >= min_route_count
    ]
    candidates = preferred or candidates
    return sorted(
        candidates,
        key=lambda record: (
            -_route_count_hint(record),
            str(record.get("graph_id") or ""),
            str(record.get("assignment_id") or ""),
        ),
    )[0]


def _resolve_corpus_path(corpus_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute() or path.exists():
        return path
    return corpus_root / path


def _safe_path_part(value: Any) -> str:
    text = str(value)
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


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
        "dt_depth": int(dt_result.dt.get_depth()),
        "dt_leaf_count": int(dt_result.dt.get_n_leaves()),
    }


def make_dt_refit_callback(
    *,
    combo: dict[str, Any],
    combo_dir: Path,
    timer: StageTimer,
    routes: list[dict[str, Any]],
    assignment: dict[str, Any],
    graph_id: str,
    family: str,
    dt_depth: int,
) -> Callable[[Any, dict[str, Any]], None]:
    def _callback(event: Any, context: dict[str, Any]) -> None:
        with timer.time("dt_refit_callback_total"):
            fit_start = time.perf_counter()
            event_dir = combo_dir / f"refit_{int(event.refit_step):05d}"
            event_dir.mkdir(parents=True, exist_ok=True)
            output_paths: list[str] = []
            sample_count = 0
            max_depth_seen = 0
            leaf_count = 0

            if combo["eve"] in LEARNED_EVE_ALGOS and context.get("eve_q") is not None:
                eve_q = context["eve_q"].to_dict()
                action_names = [str(action["id"]) for action in context["actions"]]
                with timer.time("structural_rows_eve"):
                    rows = corpus_runner.per_graph_structural_rows(
                        action_names,
                        eve_q,
                        routes,
                        assignment,
                        graph_id,
                        family,
                        combo=combo,
                    )
                if rows:
                    with timer.time("dt_fit_cross_graph"):
                        eve_dt = fit_cross_graph_dt(
                            rows,
                            max_depth=dt_depth,
                            target_key="q_rank",
                            feature_names=corpus_runner.STRUCTURAL_FEATURE_NAMES,
                        )
                    payload = _dt_payload(eve_dt)
                    json_path = event_dir / "eve_attack_choice_dt.json"
                    rules_path = event_dir / "eve_attack_choice_dt_rules.txt"
                    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
                    rules_path.write_text(payload["rules_text"], encoding="utf-8")
                    output_paths.extend([str(json_path), str(rules_path)])
                    sample_count += int(eve_dt.num_rows)
                    max_depth_seen = max(max_depth_seen, int(eve_dt.dt.get_depth()))
                    leaf_count += int(eve_dt.dt.get_n_leaves())

            if combo["alice"] in LEARNED_ALICE_ALGOS:
                with timer.time("route_choice_rows_alice"):
                    rows = corpus_runner.per_graph_route_choice_rows(
                        routes,
                        context["steps"],
                        assignment,
                        graph_id,
                        family,
                        combo=combo,
                    )
                if rows:
                    with timer.time("dt_fit_cross_graph"):
                        alice_dt = fit_cross_graph_dt(
                            rows,
                            max_depth=dt_depth,
                            target_key="choice_rate",
                            feature_names=corpus_runner.ROUTE_CHOICE_FEATURE_NAMES,
                        )
                    payload = _dt_payload(alice_dt)
                    json_path = event_dir / "alice_route_choice_dt.json"
                    rules_path = event_dir / "alice_route_choice_dt_rules.txt"
                    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
                    rules_path.write_text(payload["rules_text"], encoding="utf-8")
                    output_paths.extend([str(json_path), str(rules_path)])
                    sample_count += int(alice_dt.num_rows)
                    max_depth_seen = max(max_depth_seen, int(alice_dt.dt.get_depth()))
                    leaf_count += int(alice_dt.dt.get_n_leaves())

            event.record_fit(
                wall_clock_seconds=time.perf_counter() - fit_start,
                training_sample_count=sample_count,
                dt_depth=max_depth_seen,
                dt_leaf_count=leaf_count,
                output_paths=output_paths,
            )

    return _callback


def benchmark_case(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    corpus_root = manifest_path.parent
    records = _corpus_records_from_manifest(manifest)
    record = select_record(
        records,
        family=args.family,
        graph_id=args.graph_id,
        assignment_id=args.assignment_id,
        min_route_count=args.min_route_count,
    )
    combo = dict(PUBLICATION_COMBOS[args.combo_key])
    graph_id = str(record["graph_id"])
    family = _record_family(record)
    topology_path = _resolve_corpus_path(corpus_root, record["topology_path"])
    assignment_path = _resolve_corpus_path(corpus_root, record["assignment_path"])

    timer = StageTimer()
    install_stage_wrappers(timer)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir) / run_id
    dt_output_dir = out_dir / "dt_refit_observer"
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    ir_dict = json.loads(topology_path.read_text(encoding="utf-8"))
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    alice = str(record["alice"])
    bob = str(record["bob"])

    routes, route_selection = corpus_runner.corpus_candidate_routes(
        ir_dict,
        alice,
        bob,
        assignment,
    )
    route_diagnostics = corpus_runner.sequence_route_diagnostics(
        ir_dict,
        routes,
        assignment,
    )
    eve_action_count = e91_runtime.e91_runtime_action_count(
        ir_dict,
        routes,
        combo["game_mode"],
    )

    observer = None
    refit_callback = None
    combo_dt_dir = None
    if args.dt_refit_mode != "off":
        dt_config = DynamicDTRefitConfig(
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
            final_step_refit=True,
        )
        observer = DynamicDTRefitObserver(
            dt_config,
            total_steps=args.steps,
            alice_action_count=len(routes) if combo["alice"] in LEARNED_ALICE_ALGOS else 0,
            eve_action_count=eve_action_count if combo["eve"] in LEARNED_EVE_ALGOS else 0,
        )
        combo_dt_dir = (
            dt_output_dir
            / _safe_path_part(graph_id)
            / _safe_path_part(record.get("assignment_id") or "assignment")
            / _safe_path_part(combo["key"])
        )
        combo_dt_dir.mkdir(parents=True, exist_ok=True)
        refit_callback = make_dt_refit_callback(
            combo=combo,
            combo_dir=combo_dt_dir,
            timer=timer,
            routes=routes,
            assignment=assignment,
            graph_id=graph_id,
            family=family,
            dt_depth=args.dt_depth,
        )

    with timer.time("runtime_game_total"):
        result = e91_runtime.run_e91_graph_runtime_game(
            ir_dict,
            alice,
            bob,
            routes=routes,
            game_mode=combo["game_mode"],
            eve_algo=combo["eve"],
            alice_algo=combo["alice"],
            num_steps=args.steps,
            seed=args.seed,
            final_window_size=min(200, args.steps),
            dt_refit_observer=observer,
            dt_refit_callback=refit_callback,
        )

    if observer is not None and combo_dt_dir is not None:
        observer.write_outputs(combo_dt_dir)

    steps = result["steps"]
    summary = result["summary"]
    q_tables = summary.get("q_tables", {})
    post_rows = {"eve_structural_rows": 0, "alice_route_choice_rows": 0}
    if combo["eve"] in LEARNED_EVE_ALGOS:
        with timer.time("post_run_structural_rows_eve"):
            rows = corpus_runner.per_graph_structural_rows(
                q_tables["eve"]["action_names"],
                q_tables["eve"]["raw"],
                routes,
                assignment,
                graph_id,
                family,
                combo=combo,
            )
        post_rows["eve_structural_rows"] = len(rows)
    if combo["alice"] in LEARNED_ALICE_ALGOS:
        with timer.time("post_run_route_choice_rows_alice"):
            rows = corpus_runner.per_graph_route_choice_rows(
                routes,
                steps,
                assignment,
                graph_id,
                family,
                combo=combo,
            )
        post_rows["alice_route_choice_rows"] = len(rows)

    wall_seconds = time.perf_counter() - started
    dt_refit_summary = observer.summary() if observer is not None else None
    if dt_refit_summary is not None:
        dt_refit_summary = {
            **dt_refit_summary,
            "tick_count": len(observer.ticks),
        }

    payload = {
        "benchmark_version": 1,
        "notes": [
            "Stage timings are inclusive when one wrapped stage calls another wrapped stage.",
            "sklearn_mlpregressor_fit_lbfgs is nested inside dqn_refit_mlp_loop, which is nested inside dqn_update_total.",
            "No route-hop cap is used; corpus_candidate_routes runs unbounded simple-path enumeration.",
        ],
        "case": {
            "graph_id": graph_id,
            "assignment_id": record.get("assignment_id"),
            "family": family,
            "alice": alice,
            "bob": bob,
            "manifest_route_count_hint": _route_count_hint(record),
            "enumerated_route_count": len(routes),
            "route_selection_policy": route_selection.get("route_selection_policy"),
            "route_diagnostics": route_diagnostics,
        },
        "combo": combo,
        "seed": args.seed,
        "steps": args.steps,
        "eve_action_count": eve_action_count,
        "post_run_rows": post_rows,
        "wall_seconds": round(wall_seconds, 6),
        "dt_refit_summary": dt_refit_summary,
        "runtime_metrics": summary.get("metrics", {}),
        "timings": timer.summary(wall_seconds),
        "timeline_runs": timer.timeline_runs,
        "artifacts": {
            "out_dir": str(out_dir),
            "stage_profile_json": str(out_dir / "stage_profile.json"),
            "dt_refit_dir": str(combo_dt_dir) if combo_dt_dir is not None else None,
        },
    }
    (out_dir / "stage_profile.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    return payload


def print_summary(payload: dict[str, Any], *, top: int) -> None:
    case = payload["case"]
    print(
        "Case: "
        f"{case['graph_id']} / {case.get('assignment_id')} "
        f"family={case['family']} routes={case['enumerated_route_count']} "
        f"actions={payload['eve_action_count']} "
        f"combo={payload['combo']['key']} steps={payload['steps']}"
    )
    print(f"Wall: {payload['wall_seconds']:.3f}s")
    dt_summary = payload.get("dt_refit_summary")
    if dt_summary:
        print(
            "DT observer: "
            f"mode={dt_summary['refit_mode']} "
            f"ticks={dt_summary.get('tick_count', 0)} "
            f"refits={dt_summary['refit_count']} "
            f"refit_steps={dt_summary['refit_steps']}"
        )
    print()
    timeline_runs = payload.get("timeline_runs") or []
    if timeline_runs:
        print("Timeline event execution:")
        print(f"{'kind':24} {'runs':>6} {'events':>12} {'sched':>12} {'elapsed_s':>10} {'events/s':>10} {'max_run_s':>10}")
        by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in timeline_runs:
            by_kind[str(row.get("kind") or "unknown")].append(row)
        for kind, rows in sorted(by_kind.items()):
            elapsed = sum(float(row.get("elapsed_s") or 0.0) for row in rows)
            events = sum(int(row.get("run_events") or 0) for row in rows)
            scheduled = sum(int(row.get("scheduled_events") or 0) for row in rows)
            max_run = max(float(row.get("elapsed_s") or 0.0) for row in rows)
            print(
                f"{kind[:24]:24} "
                f"{len(rows):6d} "
                f"{events:12d} "
                f"{scheduled:12d} "
                f"{elapsed:10.3f} "
                f"{(events / elapsed if elapsed > 0.0 else 0.0):10.1f} "
                f"{max_run:10.3f}"
            )
        print()
    print("Top inclusive stage timings:")
    print(f"{'stage':48} {'count':>7} {'total_s':>10} {'pct':>8} {'mean_ms':>10} {'p95_ms':>10} {'max_ms':>10}")
    for stage, stats in list(payload["timings"].items())[:top]:
        print(
            f"{stage[:48]:48} "
            f"{stats['count']:7d} "
            f"{stats['total_s']:10.3f} "
            f"{stats['pct_wall_inclusive']:7.2f}% "
            f"{stats['mean_ms']:10.3f} "
            f"{stats['p95_ms']:10.3f} "
            f"{stats['max_ms']:10.3f}"
        )
    print()
    print(f"Wrote {payload['artifacts']['stage_profile_json']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a slim per-stage benchmark for one paper synthetic corpus case.",
    )
    parser.add_argument(
        "--manifest",
        default="runs/graph_corpus_paper_v2/corpus_manifest.json",
        help="Synthetic corpus manifest to sample from.",
    )
    parser.add_argument(
        "--family",
        default="layered_parallel",
        help="Synthetic family to sample. Use an empty string to allow any family.",
    )
    parser.add_argument("--graph-id", default=None)
    parser.add_argument("--assignment-id", default=None)
    parser.add_argument("--min-route-count", type=int, default=128)
    parser.add_argument(
        "--combo-key",
        choices=sorted(PUBLICATION_COMBOS),
        default="attack_surface_all__deep_q__deep_q",
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="results/benchmarks/paper_case_stage_profile")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--dt-refit-mode",
        choices=["fixed", "logarithmic", "dynamic", "off"],
        default="dynamic",
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
    parser.add_argument("--dt-depth", type=int, default=4)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args()
    if args.family == "":
        args.family = None
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.min_route_count < 0:
        raise SystemExit("--min-route-count must be >= 0")
    return args


def main() -> None:
    args = parse_args()
    payload = benchmark_case(args)
    print_summary(payload, top=args.top)


if __name__ == "__main__":
    main()
