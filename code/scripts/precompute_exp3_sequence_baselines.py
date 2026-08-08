#!/usr/bin/env python3
"""Precompute clean SeQUeNCe baselines for normalized Exp3 route profiles."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "sequence_matplotlib")

from sequence_game.experiments.exp3_sequence.backend import (  # noqa: E402
    ActionSpec,
    SequenceRouteEvaluator,
)
from sequence_game.experiments.exp3_sequence.baseline_cache import (  # noqa: E402
    BaselineCacheTask,
    baseline_cache_summary,
    build_baseline_tasks,
    existing_baseline_sample_keys,
    initialize_baseline_cache,
    insert_baseline_sample,
    standard_route_length_profiles,
)
from sequence_game.experiments.exp3_sequence.config import (  # noqa: E402
    BASELINE_CACHE_SQLITE_PATH,
    BASELINE_PRECOMPUTE_SAMPLES_PER_HOP,
    BASELINE_PRECOMPUTE_SEED,
    MAX_ROUTE_HOPS,
    Exp3SequenceConfig,
    default_worker_count,
)
from sequence_game.experiments.exp3_sequence.io import repo_metadata  # noqa: E402
from sequence_game.experiments.exp3_sequence.runner import _process_context  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("workers must be >= 1")

    db_path = BASELINE_CACHE_SQLITE_PATH
    if args.force and db_path.exists():
        db_path.unlink()

    config = Exp3SequenceConfig(
        workers=args.workers,
        baseline_cache_db_path=db_path,
    )
    tasks = build_baseline_tasks(
        max_hops=MAX_ROUTE_HOPS,
        samples_per_hop=BASELINE_PRECOMPUTE_SAMPLES_PER_HOP,
        seed=BASELINE_PRECOMPUTE_SEED,
    )
    initialize_baseline_cache(
        db_path,
        metadata={
            "generated_by": "scripts/precompute_exp3_sequence_baselines.py",
            "backend": "sequence_repeater_e91_chsh",
            "samples_per_hop": BASELINE_PRECOMPUTE_SAMPLES_PER_HOP,
            "max_hops": MAX_ROUTE_HOPS,
            "seed": BASELINE_PRECOMPUTE_SEED,
            "length_profiles_m": {
                str(hops): list(vector)
                for hops, vector in standard_route_length_profiles(MAX_ROUTE_HOPS).items()
            },
            "config": config.to_dict(),
            "repo": repo_metadata(),
        },
    )
    existing = existing_baseline_sample_keys(db_path)
    pending = [
        task for task in tasks
        if (task.hop_count, task.sample_index) not in existing
    ]
    print(
        f"baseline precompute: db={db_path} total={len(tasks)} "
        f"existing={len(existing)} pending={len(pending)} workers={args.workers}",
        flush=True,
    )
    if not pending:
        print(f"summary={baseline_cache_summary(db_path)}", flush=True)
        return

    started = time.perf_counter()
    completed = 0
    if args.workers == 1 or len(pending) == 1:
        for task in pending:
            task, result, wall_seconds = _run_task(task, config)
            insert_baseline_sample(
                db_path, task=task, result=result, wall_seconds=wall_seconds)
            completed += 1
            _print_progress(completed, len(pending), task, result, started, wall_seconds)
    else:
        mp_context = _process_context()
        with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=mp_context,
        ) as pool:
            futures = {
                pool.submit(_run_task, task, config): task
                for task in pending
            }
            for future in as_completed(futures):
                task, result, wall_seconds = future.result()
                insert_baseline_sample(
                    db_path, task=task, result=result, wall_seconds=wall_seconds)
                completed += 1
                _print_progress(
                    completed, len(pending), task, result, started, wall_seconds)

    print(f"summary={baseline_cache_summary(db_path)}", flush=True)


def _run_task(
        task: BaselineCacheTask,
        config: Exp3SequenceConfig,
) -> tuple[BaselineCacheTask, Any, float]:
    route = _route_for_task(task)
    evaluator = SequenceRouteEvaluator(_ir_for_route(route), [route], config)
    started = time.perf_counter()
    result = evaluator.evaluate(
        0,
        ActionSpec("no_attack", "none", "", "no_attack"),
        seed=task.seed,
        trial_id=f"baseline_h{task.hop_count}_s{task.sample_index}",
    )
    return task, result, time.perf_counter() - started


def _route_for_task(task: BaselineCacheTask) -> dict[str, Any]:
    nodes = _nodes_for_hops(task.hop_count)
    edge_ids = [
        f"{nodes[index]}-{nodes[index + 1]}"
        for index in range(len(nodes) - 1)
    ]
    total = float(sum(task.length_vector_m))
    return {
        "route_id": task.route_id,
        "path": nodes,
        "edge_ids": edge_ids,
        "internal_nodes": nodes[1:-1],
        "hop_count": task.hop_count,
        "edge_lengths_m": list(task.length_vector_m),
        "total_length": total,
        "total_length_m": total,
        "length_profile_id": f"standard_h{task.hop_count}",
    }


def _ir_for_route(route: dict[str, Any]) -> dict[str, Any]:
    nodes = {
        node: {
            "node_id": node,
            "roles": _roles_for_node(index, len(route["path"])),
            "hardware_profile_id": None,
            "coordinates": [float(index), 0.0],
        }
        for index, node in enumerate(route["path"])
    }
    edges = []
    for index, (u, v) in enumerate(zip(route["path"], route["path"][1:])):
        edges.append({
            "edge_id": f"e{index}",
            "u": u,
            "v": v,
            "length_m": float(route["edge_lengths_m"][index]),
            "channel_profile_id": None,
            "eve_eligible": True,
            "notes": "",
        })
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "generator": "exp3_sequence_baseline_precompute",
            "params": {
                "length_profile_id": route["length_profile_id"],
                "route_edge_lengths_are_canonical_for_sequence_runtime": True,
            },
            "seed": None,
        },
    }


def _nodes_for_hops(hops: int) -> list[str]:
    nodes = ["alice"]
    nodes.extend(f"r{index}" for index in range(1, hops))
    nodes.append("bob")
    return nodes


def _roles_for_node(index: int, node_count: int) -> list[str]:
    if index == 0:
        return ["alice"]
    if index == node_count - 1:
        return ["bob"]
    return ["swap_candidate"]


def _print_progress(
        completed: int,
        total: int,
        task: BaselineCacheTask,
        result: Any,
        started: float,
        wall_seconds: float,
) -> None:
    elapsed = time.perf_counter() - started
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta_minutes = (total - completed) / rate / 60.0 if rate > 0 else 0.0
    s_value = "none" if result.chsh_s is None else f"{float(result.chsh_s):.3f}"
    print(
        f"[{completed}/{total}] h={task.hop_count} sample={task.sample_index} "
        f"outcome={result.public_outcome} accepted={int(result.accepted)} "
        f"delivered={result.delivered_count} S={s_value} "
        f"wall={wall_seconds:.1f}s elapsed={elapsed / 60.0:.1f}m "
        f"eta={eta_minutes:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
