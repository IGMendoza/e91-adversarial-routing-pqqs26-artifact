#!/usr/bin/env python3
"""Precompute attacked SeQUeNCe samples for normalized Exp3 route profiles."""

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

from sequence_game.experiments.exp3_sequence.attack_cache import (  # noqa: E402
    AttackCacheTask,
    attack_cache_summary,
    build_attack_tasks,
    existing_attack_sample_keys,
    initialize_attack_cache,
    insert_attack_sample,
)
from sequence_game.experiments.exp3_sequence.backend import (  # noqa: E402
    SequenceRouteEvaluator,
)
from sequence_game.experiments.exp3_sequence.baseline_cache import (  # noqa: E402
    standard_route_length_profiles,
)
from sequence_game.experiments.exp3_sequence.config import (  # noqa: E402
    ATTACK_CACHE_SQLITE_PATH,
    ATTACK_PRECOMPUTE_SAMPLES_PER_HOP,
    ATTACK_PRECOMPUTE_SEED,
    MAX_ROUTE_HOPS,
    Exp3SequenceConfig,
)
from sequence_game.experiments.exp3_sequence.io import repo_metadata  # noqa: E402
from sequence_game.experiments.exp3_sequence.runner import _process_context  # noqa: E402

DEFAULT_WORKERS = 8
EVE_INFORMATION_STATUS = (
    "sequence_repeater_backend_records_hit_and_attack_location_only; "
    "private_information_metric_not_computed"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("workers must be >= 1")

    db_path = ATTACK_CACHE_SQLITE_PATH
    if args.force and db_path.exists():
        db_path.unlink()

    config = Exp3SequenceConfig(
        workers=args.workers,
        attack_cache_db_path=db_path,
    )
    tasks = build_attack_tasks(
        max_hops=MAX_ROUTE_HOPS,
        samples_per_hop=ATTACK_PRECOMPUTE_SAMPLES_PER_HOP,
        seed=ATTACK_PRECOMPUTE_SEED,
    )
    initialize_attack_cache(
        db_path,
        metadata={
            "generated_by": "scripts/precompute_exp3_sequence_attacks.py",
            "backend": "sequence_repeater_e91_chsh",
            "samples_per_hop": ATTACK_PRECOMPUTE_SAMPLES_PER_HOP,
            "max_hops": MAX_ROUTE_HOPS,
            "seed": ATTACK_PRECOMPUTE_SEED,
            "attack_kinds": [
                "edge_intercept_resend",
                "memory_degradation",
            ],
            "target_sampling": (
                "balanced randomized target index per attack kind and hop depth"
            ),
            "length_profiles_m": {
                str(hops): list(vector)
                for hops, vector in standard_route_length_profiles(MAX_ROUTE_HOPS).items()
            },
            "config": config.to_dict(),
            "repo": repo_metadata(),
            "eve_information_status": EVE_INFORMATION_STATUS,
        },
    )
    existing = existing_attack_sample_keys(db_path)
    pending = [
        task for task in tasks
        if (task.attack_kind, task.hop_count, task.sample_index) not in existing
    ]
    print(
        f"attack precompute: db={db_path} total={len(tasks)} "
        f"existing={len(existing)} pending={len(pending)} "
        f"workers={args.workers}",
        flush=True,
    )
    print(
        "task plan: edge_intercept_resend=64*(h=1..7)=448; "
        "memory_degradation=64*(h=2..7)=384; total=832",
        flush=True,
    )
    if not pending:
        print(f"summary={attack_cache_summary(db_path)}", flush=True)
        return

    started = time.perf_counter()
    completed = 0
    if args.workers == 1 or len(pending) == 1:
        for task in pending:
            task, result, wall_seconds = _run_task(task, config)
            insert_attack_sample(
                db_path,
                task=task,
                result=result,
                wall_seconds=wall_seconds,
                eve_information_status=EVE_INFORMATION_STATUS,
            )
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
                insert_attack_sample(
                    db_path,
                    task=task,
                    result=result,
                    wall_seconds=wall_seconds,
                    eve_information_status=EVE_INFORMATION_STATUS,
                )
                completed += 1
                _print_progress(
                    completed, len(pending), task, result, started, wall_seconds)

    print(f"summary={attack_cache_summary(db_path)}", flush=True)


def _run_task(
        task: AttackCacheTask,
        config: Exp3SequenceConfig,
) -> tuple[AttackCacheTask, Any, float]:
    route = _route_for_task(task)
    evaluator = SequenceRouteEvaluator(_ir_for_route(route), [route], config)
    started = time.perf_counter()
    result = evaluator.evaluate(
        0,
        task.action,
        seed=task.seed,
        trial_id=(
            f"attack_{task.attack_kind}_h{task.hop_count}"
            f"_s{task.sample_index}_target{task.target_index}"
        ),
    )
    return task, result, time.perf_counter() - started


def _route_for_task(task: AttackCacheTask) -> dict[str, Any]:
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
            "generator": "exp3_sequence_attack_precompute",
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
        task: AttackCacheTask,
        result: Any,
        started: float,
        wall_seconds: float,
) -> None:
    elapsed = time.perf_counter() - started
    rate = completed / elapsed if elapsed > 0 else 0.0
    eta_minutes = (total - completed) / rate / 60.0 if rate > 0 else 0.0
    s_value = "none" if result.chsh_s is None else f"{float(result.chsh_s):.3f}"
    qber = "none" if result.qber is None else f"{float(result.qber):.4f}"
    print(
        f"[{completed}/{total}] kind={task.attack_kind} h={task.hop_count} "
        f"sample={task.sample_index} target={task.target_index + 1}/"
        f"{task.target_count}:{task.target_id} "
        f"outcome={result.public_outcome} accepted={int(result.accepted)} "
        f"hit={int(result.active_route_attacked)} delivered={result.delivered_count} "
        f"S={s_value} qber={qber} wall={wall_seconds:.1f}s "
        f"elapsed={elapsed / 60.0:.1f}m eta={eta_minutes:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
