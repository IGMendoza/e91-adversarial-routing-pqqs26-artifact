"""Top-level orchestration for `scripts/exp3_sequence_results.py`."""

from __future__ import annotations

import time
import multiprocessing as mp
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .backend import build_actions, _action_hits_route
from .attack_cache import (
    attack_cache_summary,
    attack_route_cache_key,
    cached_attack_sample_count,
)
from .baseline_cache import (
    baseline_cache_summary,
    cached_baseline_sample_count,
    route_physics_cache_key,
)
from .config import Exp3SequenceConfig
from .corpus import (
    GraphCase,
    corpus_summary,
    load_graph_cases_from_sqlite,
    read_corpus_sqlite_metadata,
)
from .dt_rows import build_dt_payload
from .health import (
    GraphHealthResult,
    run_graph_health_check,
    select_health_route_indices,
)
from .io import repo_metadata, write_json
from .online import OnlineRunSummary, run_online_condition
from .oracle import OracleSummary, solve_oracle
from .payoff import (
    CellTask,
    PayoffEstimate,
    assemble_payoff,
    plan_cell_tasks,
    run_cell_task,
)
from .plots import write_figures

REUSED_COMPONENTS = (
    "sequence_game.experiments.exp3_sequence.corpus.load_graph_cases_from_sqlite",
    "sequence_game.corpus.runner.corpus_candidate_routes (fixed corpus artifact build)",
    "sequence_game.protocol.repeater_trial.run_fixed_repeater_chsh_trial",
    "sequence_game.protocol.repeater_trial.run_fixed_repeater_e91_trial",
    "sequence_game.corpus.e91_runtime_game._default_models",
    "sequence_game.corpus.e91_runtime_game._multiplexed_memory_efficiency",
    "sequence_game.game_theory.Exp3",
    "sequence_game.game_theory.solve_zero_sum_lp",
    "sequence_game.game_theory.exploitability",
    "sequence_game.xai.cross_graph_dt.fit_cross_graph_dt",
)


@dataclass(frozen=True)
class GraphRunResult:
    graph_id: str
    actions: list[Any]
    payoff: PayoffEstimate
    oracle: OracleSummary
    online: dict[str, OnlineRunSummary]


def run_pipeline(config: Exp3SequenceConfig, *, progress: bool = True) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = repo_metadata()
    write_json(out_dir / "config.json", {
        "config": config.to_dict(),
        "repo": repo,
        "backend": "sequence_repeater_e91",
        "reused_components": list(REUSED_COMPONENTS),
    })

    cases = load_graph_cases_from_sqlite(config.corpus_db_path, limit=config.max_graphs)
    corpus_metadata = read_corpus_sqlite_metadata(config.corpus_db_path)
    corpus_stats = corpus_summary(cases)
    write_json(out_dir / "corpus.json", {
        "source": {
            "kind": "sqlite",
            "path": str(config.corpus_db_path),
        },
        "metadata": corpus_metadata,
        "summary": corpus_stats,
        "graphs": [case.to_dict() for case in cases],
        "graph_count": len(cases),
    })

    health = _run_health_checks(cases, config, progress=progress)
    health_path = out_dir / "baseline_health.json"
    health_payload = _health_payload(health, config)
    write_json(health_path, health_payload)
    failed_health = [
        graph_id for graph_id, result in health.items()
        if not result.healthy
    ]
    if failed_health:
        examples = ", ".join(failed_health[:8])
        raise RuntimeError(
            "baseline no-attack health failed for "
            f"{len(failed_health)}/{len(cases)} graphs; see {health_path}; "
            f"examples: {examples}"
        )

    payoffs: dict[str, PayoffEstimate] = {}
    oracles: dict[str, OracleSummary] = {}
    online: dict[str, dict[str, OnlineRunSummary]] = {}
    actions_by_graph = {}
    if progress:
        _print_run_header(cases, config, corpus_metadata)
    for result in _run_cases(cases, config, progress=progress):
        actions_by_graph[result.graph_id] = result.actions
        payoffs[result.graph_id] = result.payoff
        oracles[result.graph_id] = result.oracle
        online[result.graph_id] = result.online
        write_json(out_dir / "payoff_matrices" / f"{result.graph_id}.json", result.payoff.to_dict())
        write_json(out_dir / "oracle" / f"{result.graph_id}.json", result.oracle.to_dict())
        for condition, run in result.online.items():
            write_json(
                out_dir / "online_runs" / condition / f"{result.graph_id}.json",
                run.to_dict(),
            )

    oracle_summary = {
        graph_id: oracle.to_dict()
        for graph_id, oracle in oracles.items()
    }
    online_summary = {
        graph_id: {
            condition: run.to_dict()
            for condition, run in by_condition.items()
        }
        for graph_id, by_condition in online.items()
    }
    write_json(out_dir / "oracle_summary.json", oracle_summary)
    write_json(out_dir / "exp3_summary.json", online_summary)

    dt_payload = build_dt_payload(
        cases,
        payoffs,
        oracles,
        online,
        actions_by_graph,
        max_depth=config.dt_max_depth,
    )
    write_json(out_dir / "dt" / "dt_payload.json", dt_payload)
    _write_dt_artifacts(out_dir / "dt", dt_payload)
    figure_paths = write_figures(
        out_dir / "figures",
        cases,
        oracles,
        online,
        payoffs,
        attack_cache_db_path=config.attack_cache_db_path,
        baseline_cache_db_path=config.baseline_cache_db_path,
    )
    summary = {
        "out_dir": str(out_dir),
        "graph_count": len(cases),
        "corpus": {
            "source": "sqlite",
            "path": str(config.corpus_db_path),
            "metadata": corpus_metadata,
            "summary": corpus_stats,
        },
        "backend": "sequence_repeater_e91",
        "security_monitor": config.security_monitor,
        "alice_acceptance_rule": config.alice_acceptance_rule,
        "trials_per_cell": config.trials_per_cell,
        "online_turns": config.online_turns,
        "exp3_schedule": config.to_dict()["exp3_schedule"],
        "workers": config.workers,
        "repo": repo,
        "sequence_trial_config": {
            "start_time_ps": config.start_time_ps,
            "end_time_ps": config.end_time_ps,
            "stop_time_ps": config.stop_time_ps,
            "memory_pairs_per_trial": config.memory_pairs_per_trial,
            "qber_threshold": config.qber_threshold,
            "min_key_pairs": config.min_key_pairs,
            "request_fidelity": config.request_fidelity,
            "sequence_memory_fidelity_override": config.sequence_memory_fidelity_override,
            "baseline_cache_db_path": str(config.baseline_cache_db_path),
            "baseline_cache": baseline_cache_summary(config.baseline_cache_db_path),
            "attack_cache_db_path": str(config.attack_cache_db_path),
            "attack_cache": attack_cache_summary(config.attack_cache_db_path),
        },
        "reused_components": list(REUSED_COMPONENTS),
        "oracle_summary_path": str(out_dir / "oracle_summary.json"),
        "exp3_summary_path": str(out_dir / "exp3_summary.json"),
        "dt_payload_path": str(out_dir / "dt" / "dt_payload.json"),
        "baseline_health_path": str(health_path),
        "dt_artifact_paths": {
            "graph_value": str(out_dir / "dt" / "graph_value_dt.json"),
            "action_strategy": str(out_dir / "dt" / "action_strategy_dt.json"),
            "route_strategy": str(out_dir / "dt" / "route_strategy_dt.json"),
        },
        "figures": figure_paths,
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(out_dir / "run_summary.json", summary)
    return summary


def _health_payload(
        health: dict[str, GraphHealthResult],
        config: Exp3SequenceConfig,
) -> dict[str, Any]:
    total_trials = sum(result.trial_count for result in health.values())
    accepted = sum(result.accepted_count for result in health.values())
    qualified = sum(result.qualified_count for result in health.values())
    failed = [
        graph_id for graph_id, result in health.items()
        if not result.healthy
    ]
    return {
        "status": "healthy" if not failed else "failed",
        "graph_count": len(health),
        "failed_graph_count": len(failed),
        "failed_graphs": failed,
        "trial_count": total_trials,
        "accepted_count": accepted,
        "qualified_count": qualified,
        "accepted_rate": accepted / total_trials if total_trials else 0.0,
        "qualified_rate": qualified / total_trials if total_trials else 0.0,
        "config": {
            "routes_per_graph": config.baseline_health_routes_per_graph,
            "trials_per_route": config.baseline_health_trials_per_route,
            "min_accepted_trials": config.baseline_health_min_accepted_trials,
            "accept_rate_warn": config.baseline_health_accept_rate_warn,
            "min_chsh_s": config.baseline_health_min_chsh_s,
            "min_delivered_pairs": config.baseline_health_min_delivered_pairs,
            "security_monitor": config.security_monitor,
            "memory_pairs_per_trial": config.memory_pairs_per_trial,
        },
        "graphs": {
            graph_id: result.to_dict()
            for graph_id, result in health.items()
        },
    }


def _run_health_checks(
        cases: list[GraphCase],
        config: Exp3SequenceConfig,
        *,
        progress: bool,
) -> dict[str, GraphHealthResult]:
    if progress:
        print(
            "baseline health: "
            f"routes_per_graph={config.baseline_health_routes_per_graph} "
            f"trials_per_route={config.baseline_health_trials_per_route} "
            f"min_accepted={config.baseline_health_min_accepted_trials} "
            f"min_chsh_s={config.baseline_health_min_chsh_s:.2f} "
            f"min_delivered_pairs={config.baseline_health_min_delivered_pairs}",
            flush=True,
        )
    if config.workers == 1 or len(cases) <= 1:
        results = {}
        for index, case in enumerate(cases):
            result = _run_one_health_check(index, len(cases), case, config, progress)
            results[result.graph_id] = result
        return results

    results_by_index: dict[int, GraphHealthResult] = {}
    mp_context = _process_context()
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=mp_context) as pool:
        futures = {
            pool.submit(_run_one_health_check, index, len(cases), case, config, progress): index
            for index, case in enumerate(cases)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            result = future.result()
            results_by_index[index] = result
            if progress:
                print(
                    f"[{completed}/{len(cases)}] health {result.graph_id} "
                    f"accepted={result.accepted_count}/{result.trial_count} "
                    f"qualified={result.qualified_count}/{result.trial_count} "
                    f"status={result.status}",
                    flush=True,
                )
    return {
        cases[index].graph_id: results_by_index[index]
        for index in range(len(cases))
    }


def _run_one_health_check(
        index: int,
        total: int,
        case: GraphCase,
        config: Exp3SequenceConfig,
        progress: bool,
) -> GraphHealthResult:
    started = time.perf_counter()

    def log(message: str) -> None:
        if progress:
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"health {index + 1:02d}/{total:02d} "
                f"{case.graph_id}: {message} "
                f"({time.perf_counter() - started:.1f}s)",
                flush=True,
            )

    return run_graph_health_check(
        case,
        config,
        seed_base=config.seed + index * 1_000_000 + 50_000,
        progress_callback=log if progress else None,
    )


def _write_dt_artifacts(dt_dir: Path, payload: dict[str, Any]) -> None:
    trees = payload.get("trees", {})
    write_json(dt_dir / "graph_value_dt.json", {
        "rows": payload.get("graph_rows", []),
        "trees": {
            key: trees.get(key)
            for key in ("graph_oracle_retention",)
        },
    })
    write_json(dt_dir / "action_strategy_dt.json", {
        "rows": payload.get("action_rows", []),
        "trees": {
            key: trees.get(key)
            for key in (
                "action_oracle_eve_strategy_prob",
                "action_expected_denial_under_oracle_alice",
            )
        },
    })
    write_json(dt_dir / "route_strategy_dt.json", {
        "rows": payload.get("route_rows", []),
        "trees": {
            key: trees.get(key)
            for key in ("route_oracle_alice_strategy_prob",)
        },
    })


def _run_cases(cases: list[GraphCase], config: Exp3SequenceConfig, *,
               progress: bool) -> list[GraphRunResult]:
    """Run payoff, oracle, and online phases over all graphs.

    Work is parallelized below the graph level: payoff estimation is one
    global pool of deduplicated cell tasks and each online (graph, condition)
    run is its own task, so a single large graph no longer serializes an
    entire worker for days.
    """

    actions_list = [
        build_actions(case.routes, config.action_kinds) for case in cases
    ]
    plans = [
        plan_cell_tasks(
            case.routes,
            actions_list[index],
            trials_per_cell=config.trials_per_cell,
            seed_base=config.seed,
        )
        for index, case in enumerate(cases)
    ]
    payoffs = _run_payoff_phase(
        cases, actions_list, plans, config, progress=progress)
    oracles = []
    for case, payoff in zip(cases, payoffs):
        oracle = solve_oracle(payoff)
        oracles.append(oracle)
        if progress:
            retention = (
                f"{oracle.retention:.3f}" if oracle.retention is not None else "none"
            )
            print(
                f"oracle {case.graph_id}: value={oracle.value:.3f} "
                f"baseline={oracle.baseline_rate:.3f} "
                f"retention={retention} status={oracle.status}",
                flush=True,
            )
    online_list = _run_online_phase(
        cases, actions_list, payoffs, oracles, config, progress=progress)
    return [
        GraphRunResult(
            graph_id=case.graph_id,
            actions=actions_list[index],
            payoff=payoffs[index],
            oracle=oracles[index],
            online=online_list[index],
        )
        for index, case in enumerate(cases)
    ]


def _run_payoff_phase(
        cases: list[GraphCase],
        actions_list: list[list[Any]],
        plans: list[tuple[list[CellTask], dict[tuple[int, int], int]]],
        config: Exp3SequenceConfig,
        *,
        progress: bool,
) -> list[PayoffEstimate]:
    unique_tasks: dict[tuple[Any, ...], tuple[int, int, CellTask]] = {}
    for index, (tasks, _) in enumerate(plans):
        for task_index, task in enumerate(tasks):
            unique_tasks.setdefault(task.cache_key, (index, task_index, task))
    total_tasks = len(unique_tasks)
    started = time.perf_counter()
    results: dict[int, dict[int, Any]] = {index: {} for index in range(len(cases))}
    global_results: dict[tuple[Any, ...], Any] = {}
    if config.workers == 1 or total_tasks <= 1:
        completed = 0
        for cache_key, (index, _task_index, task) in unique_tasks.items():
            global_results[cache_key] = run_cell_task(
                cases[index], actions_list[index], config, task)
            completed += 1
            _log_phase_progress(
                progress, "route baselines", completed, total_tasks,
                started, cases[index].graph_id)
    else:
        mp_context = _process_context()
        with ProcessPoolExecutor(max_workers=config.workers, mp_context=mp_context) as pool:
            futures = {}
            for cache_key, (index, _task_index, task) in unique_tasks.items():
                future = pool.submit(
                    run_cell_task, cases[index], actions_list[index], config, task)
                futures[future] = (cache_key, index)
            for completed, future in enumerate(as_completed(futures), 1):
                cache_key, index = futures[future]
                global_results[cache_key] = future.result()
                _log_phase_progress(
                    progress, "route baselines", completed, total_tasks,
                    started, cases[index].graph_id)
    for index, (tasks, _) in enumerate(plans):
        for task_index, task in enumerate(tasks):
            results[index][task_index] = global_results[task.cache_key]
    return [
        assemble_payoff(
            case,
            actions_list[index],
            config,
            seed_base=config.seed,
            tasks=plans[index][0],
            cell_to_task=plans[index][1],
            results_by_task=results[index],
        )
        for index, case in enumerate(cases)
    ]


def _run_online_phase(
        cases: list[GraphCase],
        actions_list: list[list[Any]],
        payoffs: list[PayoffEstimate],
        oracles: list[OracleSummary],
        config: Exp3SequenceConfig,
        *,
        progress: bool,
) -> list[dict[str, OnlineRunSummary]]:
    online: list[dict[str, OnlineRunSummary]] = [{} for _ in cases]
    if config.online_turns <= 0:
        return online
    pairs = [
        (index, condition_index)
        for index in range(len(cases))
        for condition_index in range(len(config.conditions))
    ]
    started = time.perf_counter()
    if config.workers == 1 or len(pairs) <= 1:
        for completed, (index, condition_index) in enumerate(pairs, 1):
            case = cases[index]
            condition = config.conditions[condition_index]

            def log(message: str) -> None:
                if progress:
                    print(
                        f"[{time.strftime('%H:%M:%S')}] online "
                        f"{case.graph_id} {condition.key}: {message}",
                        flush=True,
                    )

            run = run_online_condition(
                case,
                actions_list[index],
                payoffs[index],
                oracles[index],
                condition,
                config,
                seed=config.seed + index * 1_000_000 + condition_index * 100_000,
                progress_callback=log if progress else None,
            )
            online[index][condition.key] = run
            _log_phase_progress(
                progress, "online runs", completed, len(pairs),
                started, f"{case.graph_id} {condition.key}")
    else:
        mp_context = _process_context()
        with ProcessPoolExecutor(max_workers=config.workers, mp_context=mp_context) as pool:
            futures = {
                pool.submit(
                    run_online_condition,
                    cases[index],
                    actions_list[index],
                    payoffs[index],
                    oracles[index],
                    config.conditions[condition_index],
                    config,
                    seed=config.seed + index * 1_000_000 + condition_index * 100_000,
                    progress_callback=None,
                ): (index, condition_index)
                for index, condition_index in pairs
            }
            for completed, future in enumerate(as_completed(futures), 1):
                index, condition_index = futures[future]
                condition = config.conditions[condition_index]
                run = future.result()
                online[index][condition.key] = run
                if progress:
                    print(
                        f"[{completed}/{len(pairs)}] online done "
                        f"{cases[index].graph_id} {condition.key} "
                        f"final_key={run.final_key_rate:.3f} "
                        f"hit={run.hit_rate:.3f} "
                        f"exploitability={run.exploitability_vs_payoff:.3f}",
                        flush=True,
                    )
    return online


def _log_phase_progress(progress: bool, label: str, completed: int, total: int,
                        started: float, last_item: str) -> None:
    if not progress:
        return
    every = max(1, total // 200)
    if completed not in (1, total) and completed % every != 0:
        return
    elapsed = time.perf_counter() - started
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta_minutes = (total - completed) / rate / 60.0 if rate > 0 else float("inf")
    print(
        f"[{time.strftime('%H:%M:%S')}] {label} {completed}/{total} "
        f"({completed / max(1, total):.1%}) last={last_item} "
        f"elapsed={elapsed / 60.0:.1f}m eta={eta_minutes:.1f}m",
        flush=True,
    )


def _process_context() -> Any:
    try:
        return mp.get_context("fork")
    except ValueError:
        return None


def _print_run_header(
        cases: list[GraphCase],
        config: Exp3SequenceConfig,
        corpus_metadata: dict[str, Any],
) -> None:
    actions = [build_actions(case.routes, config.action_kinds) for case in cases]
    cells = [len(case.routes) * len(case_actions) for case, case_actions in zip(cases, actions)]
    unique_cells = sum(
        len(plan_cell_tasks(
            case.routes,
            case_actions,
            trials_per_cell=config.trials_per_cell,
            seed_base=config.seed,
        )[0])
        for case, case_actions in zip(cases, actions)
    )
    unique_route_baselines = len({
        task.cache_key
        for case, case_actions in zip(cases, actions)
        for task in plan_cell_tasks(
            case.routes,
            case_actions,
            trials_per_cell=config.trials_per_cell,
            seed_base=config.seed,
        )[0]
    })
    payoff_cache_keys = {
        task.cache_key
        for case, case_actions in zip(cases, actions)
        for task in plan_cell_tasks(
            case.routes,
            case_actions,
            trials_per_cell=config.trials_per_cell,
            seed_base=config.seed,
        )[0]
    }
    attack_hit_cells = 0
    attack_cache_keys = set()
    for case, case_actions in zip(cases, actions):
        for route in case.routes:
            for action in case_actions:
                if _action_hits_route(action, route):
                    attack_hit_cells += 1
                    attack_cache_keys.add(attack_route_cache_key(route, action.attack_type))
    health_cache_keys = [
        route_physics_cache_key(case.routes[route_index])
        for case in cases
        for route_index in select_health_route_indices(
            case.routes, config.baseline_health_routes_per_graph)
    ]
    route_lengths_km = [
        float(route["total_length_m"]) / 1000.0
        for case in cases
        for route in case.routes
    ]
    repeaters = [
        float(len(route.get("internal_nodes") or []))
        for case in cases
        for route in case.routes
    ]
    node_disjoint = [case.features["node_disjoint_paths"] for case in cases]
    overlap_edges = []
    for case in cases:
        for left, right in combinations(case.routes, 2):
            overlap_edges.append(float(len(_route_edge_set(left) & _route_edge_set(right))))
    cache = baseline_cache_summary(config.baseline_cache_db_path)
    payoff_cached_samples, sequence_payoff_trials = _cache_coverage_counts(
        payoff_cache_keys,
        sample_count=config.trials_per_cell,
        db_path=config.baseline_cache_db_path,
    )
    health_cached_samples, sequence_health_trials = _cache_coverage_counts(
        health_cache_keys,
        sample_count=config.baseline_health_trials_per_route,
        db_path=config.baseline_cache_db_path,
    )
    attack_cached_samples, attack_missing_samples = _attack_cache_coverage_counts(
        attack_cache_keys,
        sample_count=config.attack_payoff_samples_per_route,
        db_path=config.attack_cache_db_path,
    )
    print(
        "corpus source: "
        f"sqlite={config.corpus_db_path} "
        f"artifact_graphs={corpus_metadata.get('graph_count', 'unknown')} "
        f"artifact_max_hops={corpus_metadata.get('max_route_hops', 'unknown')} "
        f"loaded_graphs={len(cases)}",
        flush=True,
    )
    print(f"running {len(cases)} graph cases with workers={config.workers}", flush=True)
    print(
        "work: "
        f"routes={sum(len(case.routes) for case in cases)} "
        f"actions={sum(len(case_actions) for case_actions in actions)} "
        f"health_cached_samples={health_cached_samples} "
        f"health_sequence_trials={sequence_health_trials} "
        f"payoff_cells={sum(cells)} "
        f"payoff_route_tasks={unique_cells} "
        f"payoff_unique_route_baselines={unique_route_baselines} "
        f"attack_hit_cells={attack_hit_cells} "
        f"attack_cached_samples={attack_cached_samples} "
        f"attack_missing_samples={attack_missing_samples} "
        "(hit cells use attack cache; misses reuse clean baselines) "
        f"payoff_cached_samples={payoff_cached_samples} "
        f"payoff_sequence_trials={sequence_payoff_trials} "
        f"online_turns={len(cases) * len(config.conditions) * config.online_turns}",
        flush=True,
    )
    print(
        "corpus: "
        f"route_km={_mean_sd(route_lengths_km)} "
        f"repeaters_per_route={_mean_sd(repeaters)} "
        f"node_disjoint={_mean_sd(node_disjoint)} "
        f"edge_overlap_pair={_mean_sd(overlap_edges)}",
        flush=True,
    )
    print(
        "sequence: "
        f"monitor={config.security_monitor} "
        f"memory_pairs={config.memory_pairs_per_trial} "
        f"baseline_samples_per_route={config.trials_per_cell} "
        f"attack_samples_per_route={config.attack_payoff_samples_per_route} "
        f"timing=start=max({config.start_time_ps}ps, "
        f"{config.sequence_setup_traversals:g}*route_one_way_ps), "
        f"window={config.repeater_window_ps}ps, stop_margin={config.stop_margin_ps}ps",
        flush=True,
    )
    print(
        "baseline cache: "
        f"path={config.baseline_cache_db_path} "
        f"exists={cache.get('exists', False)} "
        f"samples={cache.get('sample_count', 0)}",
        flush=True,
    )
    attack_cache = attack_cache_summary(config.attack_cache_db_path)
    print(
        "attack cache: "
        f"path={config.attack_cache_db_path} "
        f"exists={attack_cache.get('exists', False)} "
        f"samples={attack_cache.get('sample_count', 0)}",
        flush=True,
    )


def _route_edge_set(route: dict[str, Any]) -> set[Any]:
    if route.get("edge_ids"):
        return set(str(edge_id) for edge_id in route["edge_ids"])
    path = [str(node) for node in route.get("path", [])]
    return {tuple(sorted((u, v))) for u, v in zip(path, path[1:])}


def _cache_coverage_counts(
        cache_keys: Any,
        *,
        sample_count: int,
        db_path: Path,
) -> tuple[int, int]:
    cached = 0
    sequence_trials = 0
    for cache_key in cache_keys:
        available = cached_baseline_sample_count(db_path, cache_key=cache_key)
        if available >= sample_count:
            cached += sample_count
        else:
            sequence_trials += sample_count
    return cached, sequence_trials


def _attack_cache_coverage_counts(
        cache_keys: Any,
        *,
        sample_count: int,
        db_path: Path,
) -> tuple[int, int]:
    cached = 0
    missing = 0
    for cache_key in cache_keys:
        available = cached_attack_sample_count(db_path, cache_key=cache_key)
        if available >= sample_count:
            cached += sample_count
        else:
            missing += sample_count
    return cached, missing


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "n=0"
    sd = stdev(values) if len(values) > 1 else 0.0
    return f"mean={mean(values):.3f},sd={sd:.3f},n={len(values)}"
