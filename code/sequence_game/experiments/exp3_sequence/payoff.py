"""Payoff matrix estimation from SeQUeNCe trials."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .acceptance import alice_accepts_turn, chsh_passes
from .backend import ActionSpec, SequenceRouteEvaluator, TurnResult, _action_hits_route
from .attack_cache import (
    attack_route_cache_key,
    cached_attack_sample_count,
    load_cached_attack_samples,
)
from .baseline_cache import (
    load_cached_baseline_results,
    route_physics_cache_key,
)
from .config import Exp3SequenceConfig
from .corpus import GraphCase


@dataclass(frozen=True)
class CellTask:
    """One clean route-baseline task for payoff assembly."""

    route_index: int
    action_index: int
    seed_start: int
    trials: int
    cache_key: tuple[Any, ...]

    @property
    def seed_end(self) -> int:
        return self.seed_start + self.trials - 1


@dataclass(frozen=True)
class CellStats:
    route_id: str
    action_id: str
    seed_start: int
    seed_end: int
    trial_count: int
    accepted_count: int
    chsh_abort_count: int
    qber_abort_count: int
    delivery_failure_count: int
    accepted_rate: float
    accepted_ci_low: float
    accepted_ci_high: float
    accepted_ci_half_width: float
    mean_chsh_s: float | None
    chsh_s_count: int
    chsh_s_min: float | None
    chsh_s_max: float | None
    chsh_adequately_sampled_count: int
    mean_qber: float | None
    qber_count: int
    qber_min: float | None
    qber_max: float | None
    mean_delivered_count: float
    mean_sifted_count: float
    sequence_timing: dict[str, Any]
    warnings: tuple[str, ...]
    # Action whose trials produced these stats; differs from action_id when a
    # route-missing action reuses the route's shared no-attack trials.
    simulated_action_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "action_id": self.action_id,
            "seed_start": self.seed_start,
            "seed_end": self.seed_end,
            "trial_count": self.trial_count,
            "accepted_count": self.accepted_count,
            "chsh_abort_count": self.chsh_abort_count,
            "qber_abort_count": self.qber_abort_count,
            "delivery_failure_count": self.delivery_failure_count,
            "accepted_rate": self.accepted_rate,
            "accepted_ci_low": self.accepted_ci_low,
            "accepted_ci_high": self.accepted_ci_high,
            "accepted_ci_half_width": self.accepted_ci_half_width,
            "mean_chsh_s": self.mean_chsh_s,
            "chsh_s_count": self.chsh_s_count,
            "chsh_s_min": self.chsh_s_min,
            "chsh_s_max": self.chsh_s_max,
            "chsh_adequately_sampled_count": self.chsh_adequately_sampled_count,
            "mean_qber": self.mean_qber,
            "qber_count": self.qber_count,
            "qber_min": self.qber_min,
            "qber_max": self.qber_max,
            "mean_delivered_count": self.mean_delivered_count,
            "mean_sifted_count": self.mean_sifted_count,
            "sequence_timing": dict(self.sequence_timing),
            "warnings": list(self.warnings),
            "simulated_action_id": self.simulated_action_id,
        }


@dataclass(frozen=True)
class PayoffEstimate:
    graph_id: str
    route_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    payoff: np.ndarray
    cells: tuple[CellStats, ...]
    seed_start: int
    seed_end: int
    backend: str = "sequence_repeater_e91"
    security_monitor: str = "chsh"
    trials_per_cell: int = 0
    memory_pairs_per_trial: int = 0
    ci_half_width_warn: float = 0.0
    sequence_trial_config: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "backend": self.backend,
            "security_monitor": self.security_monitor,
            "trials_per_cell": self.trials_per_cell,
            "memory_pairs_per_trial": self.memory_pairs_per_trial,
            "ci_half_width_warn": self.ci_half_width_warn,
            "sequence_trial_config": dict(self.sequence_trial_config or {}),
            "route_ids": list(self.route_ids),
            "action_ids": list(self.action_ids),
            "payoff": self.payoff.tolist(),
            "cells": [cell.to_dict() for cell in self.cells],
            "seed_start": self.seed_start,
            "seed_end": self.seed_end,
        }


def plan_cell_tasks(
        routes: list[dict[str, Any]],
        actions: list[ActionSpec],
        *,
        trials_per_cell: int,
        seed_base: int,
) -> tuple[list[CellTask], dict[tuple[int, int], int]]:
    """Plan clean route-baseline tasks and map payoff cells to them."""

    tasks: list[CellTask] = []
    cell_to_task: dict[tuple[int, int], int] = {}
    task_by_route_key: dict[tuple[Any, ...], int] = {}
    no_attack_index = _no_attack_index(actions)
    for route_index, route in enumerate(routes):
        route_key = route_physics_cache_key(route)
        task_index = task_by_route_key.get(route_key)
        if task_index is None:
            task_index = len(tasks)
            task_by_route_key[route_key] = task_index
            tasks.append(CellTask(
                route_index=route_index,
                action_index=no_attack_index,
                seed_start=_seed_start_for_cache_key(
                    seed_base, route_key, trials_per_cell),
                trials=trials_per_cell,
                cache_key=route_key,
            ))
        for action_index, action in enumerate(actions):
            cell_to_task[(route_index, action_index)] = task_index
    return tasks, cell_to_task


def _no_attack_index(actions: list[ActionSpec]) -> int:
    for index, action in enumerate(actions):
        if action.attack_type == "no_attack":
            return index
    raise ValueError("payoff assembly requires a no_attack action")


def _seed_start_for_cache_key(
        seed_base: int,
        cache_key: tuple[Any, ...],
        trials_per_cell: int,
) -> int:
    digest = hashlib.sha256(repr(cache_key).encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big") % 1_000_000_000
    return int(seed_base) + offset * max(1, int(trials_per_cell))


def run_cell_task(case: GraphCase, actions: list[ActionSpec],
                  config: Exp3SequenceConfig, task: CellTask) -> list[TurnResult]:
    """Run the SeQUeNCe trials of one simulated payoff cell."""

    cached = load_cached_baseline_results(
        config.baseline_cache_db_path,
        cache_key=task.cache_key,
        sample_count=task.trials,
        seed=task.seed_start,
    )
    if cached is not None:
        return cached

    evaluator = SequenceRouteEvaluator(case.ir_dict, case.routes, config)
    action = actions[task.action_index]
    results: list[TurnResult] = []
    for trial in range(task.trials):
        results.append(evaluator.evaluate(
            task.route_index,
            action,
            seed=task.seed_start + trial,
            trial_id=(
                f"{case.graph_id}_r{task.route_index}"
                f"_a{task.action_index}_t{trial}"
            ),
        ))
    return results


def assemble_payoff(case: GraphCase, actions: list[ActionSpec],
                    config: Exp3SequenceConfig, *, seed_base: int,
                    tasks: list[CellTask],
                    cell_to_task: dict[tuple[int, int], int],
                    results_by_task: dict[int, list[TurnResult]]) -> PayoffEstimate:
    payoff = np.zeros((len(case.routes), len(actions)), dtype=float)
    cells: list[CellStats] = []
    baseline_by_task: dict[int, CellStats] = {}
    for route_index, route in enumerate(case.routes):
        for action_index, action in enumerate(actions):
            task_index = cell_to_task[(route_index, action_index)]
            baseline = baseline_by_task.get(task_index)
            if baseline is None:
                task = tasks[task_index]
                baseline = summarize_cell(
                    route_id=str(route["route_id"]),
                    action_id=actions[task.action_index].action_id,
                    results=results_by_task[task_index],
                    seed_start=task.seed_start,
                    seed_end=task.seed_end,
                    ci_half_width_warn=config.ci_half_width_warn,
                    simulated_action_id=actions[task.action_index].action_id,
                    sequence_timing_extra={
                        "route_hop_count": int(route["hop_count"]),
                    },
                    config=config,
                )
                baseline_by_task[task_index] = baseline
            if _action_hits_route(action, route):
                cell = attack_cache_cell(
                    route_id=str(route["route_id"]),
                    route=route,
                    action=action,
                    config=config,
                    seed_base=seed_base,
                )
            else:
                cell = clone_cell_for_action(
                    route_id=str(route["route_id"]),
                    action_id=action.action_id,
                    source=baseline,
                )
            payoff[route_index, action_index] = cell.accepted_rate
            cells.append(cell)
    seed_end = seed_base - 1
    if tasks:
        seed_end = max(task.seed_end for task in tasks)
    return PayoffEstimate(
        graph_id=case.graph_id,
        route_ids=tuple(str(route["route_id"]) for route in case.routes),
        action_ids=tuple(action.action_id for action in actions),
        payoff=payoff,
        cells=tuple(cells),
        seed_start=seed_base,
        seed_end=seed_end,
        backend="sequence_repeater_e91",
        security_monitor=config.security_monitor,
        trials_per_cell=config.trials_per_cell,
        memory_pairs_per_trial=config.memory_pairs_per_trial,
        ci_half_width_warn=config.ci_half_width_warn,
        sequence_trial_config={
            "payoff_model": (
                "clean_route_baseline_cache_plus_attack_route_profile_cache"
            ),
            "attack_cache_db_path": str(config.attack_cache_db_path),
            "attack_payoff_samples_per_route": config.attack_payoff_samples_per_route,
            "attack_cache_role": "active hit cells are sampled by route profile and attack kind",
            "start_time_ps": config.start_time_ps,
            "end_time_ps": config.end_time_ps,
            "stop_time_ps": config.stop_time_ps,
            "request_fidelity": config.request_fidelity,
            "qber_threshold": config.qber_threshold,
            "min_key_pairs": config.min_key_pairs,
            "alice_acceptance_rule": config.alice_acceptance_rule,
            "alice_key_rate_shaping_weight": config.alice_key_rate_shaping_weight,
            "swapping_success_prob": config.swapping_success_prob,
            "swapping_degradation": config.swapping_degradation,
            "route_timing_policy": (
                "start=max(config.start_time_ps, "
                "sequence_setup_traversals * one_way_classical_ps)"
            ),
            "sequence_setup_traversals": config.sequence_setup_traversals,
        },
    )


def clone_cell_for_action(route_id: str, action_id: str,
                          source: CellStats) -> CellStats:
    return CellStats(
        route_id=route_id,
        action_id=action_id,
        seed_start=source.seed_start,
        seed_end=source.seed_end,
        trial_count=source.trial_count,
        accepted_count=source.accepted_count,
        chsh_abort_count=source.chsh_abort_count,
        qber_abort_count=source.qber_abort_count,
        delivery_failure_count=source.delivery_failure_count,
        accepted_rate=source.accepted_rate,
        accepted_ci_low=source.accepted_ci_low,
        accepted_ci_high=source.accepted_ci_high,
        accepted_ci_half_width=source.accepted_ci_half_width,
        mean_chsh_s=source.mean_chsh_s,
        chsh_s_count=source.chsh_s_count,
        chsh_s_min=source.chsh_s_min,
        chsh_s_max=source.chsh_s_max,
        chsh_adequately_sampled_count=source.chsh_adequately_sampled_count,
        mean_qber=source.mean_qber,
        qber_count=source.qber_count,
        qber_min=source.qber_min,
        qber_max=source.qber_max,
        mean_delivered_count=source.mean_delivered_count,
        mean_sifted_count=source.mean_sifted_count,
        sequence_timing={
            **source.sequence_timing,
            "payoff_model": "clean_route_baseline_cache",
            "source_simulated_action_id": source.simulated_action_id,
        },
        warnings=source.warnings,
        simulated_action_id=source.simulated_action_id,
    )


def attack_cache_cell(
        *,
        route_id: str,
        route: dict[str, Any],
        action: ActionSpec,
        config: Exp3SequenceConfig,
        seed_base: int,
) -> CellStats:
    cache_key = attack_route_cache_key(route, action.attack_type)
    sample_count = config.attack_payoff_samples_per_route
    selection_seed = _seed_start_for_cache_key(
        seed_base,
        cache_key + ("route", route_id, "action", action.action_id),
        sample_count,
    )
    samples = load_cached_attack_samples(
        config.attack_cache_db_path,
        cache_key=cache_key,
        sample_count=sample_count,
        seed=selection_seed,
    )
    if samples is None:
        available = cached_attack_sample_count(
            config.attack_cache_db_path,
            cache_key=cache_key,
        )
        raise RuntimeError(
            "attack cache missing or undersampled for active hit cell: "
            f"route_id={route_id} action_id={action.action_id} "
            f"cache_key={cache_key!r} required={sample_count} available={available} "
            f"db={config.attack_cache_db_path}"
        )
    results = [sample.result for sample in samples]
    target_counts: dict[str, int] = {}
    for sample in samples:
        target_counts[sample.target_id] = target_counts.get(sample.target_id, 0) + 1
    seed_values = [sample.seed for sample in samples]
    return summarize_cell(
        route_id=route_id,
        action_id=action.action_id,
        results=results,
        seed_start=min(seed_values),
        seed_end=max(seed_values),
        ci_half_width_warn=config.ci_half_width_warn,
        simulated_action_id=f"attack_cache:{action.attack_type}",
        sequence_timing_extra={
            "payoff_model": "attack_route_profile_cache",
            "attack_cache_db_path": str(config.attack_cache_db_path),
            "attack_cache_key": list(cache_key),
            "attack_cache_selection_seed": selection_seed,
            "attack_cache_sample_count": len(samples),
            "attack_cache_selected_sample_indices": [
                sample.sample_index for sample in samples
            ],
            "attack_cache_target_counts": dict(sorted(target_counts.items())),
            "attack_cache_target_role": (
                "metadata only; payoff samples are grouped by route profile "
                "and attack kind"
            ),
            "attack_cache_hop_count": int(route["hop_count"]),
            "route_hop_count": int(route["hop_count"]),
            "requested_action_id": action.action_id,
            "eve_information_status": sorted({
                sample.eve_information_status for sample in samples
            }),
        },
        config=config,
    )


def estimate_payoff(case: GraphCase, actions: list[ActionSpec],
                    config: Exp3SequenceConfig, *, seed_base: int,
                    progress_callback: Any = None) -> PayoffEstimate:
    tasks, cell_to_task = plan_cell_tasks(
        case.routes,
        actions,
        trials_per_cell=config.trials_per_cell,
        seed_base=seed_base,
    )
    results_by_task: dict[int, list[TurnResult]] = {}
    progress_every = max(1, min(25, len(tasks) // 20 or 1))
    for task_index, task in enumerate(tasks):
        results_by_task[task_index] = run_cell_task(case, actions, config, task)
        completed = task_index + 1
        if progress_callback and (
            completed == 1
            or completed == len(tasks)
            or completed % progress_every == 0
        ):
            accepted = sum(
                1 for result in results_by_task[task_index]
                if alice_accepts_turn(result, config)
            )
            progress_callback(
                "payoff "
                f"{completed}/{len(tasks)} route baselines "
                f"({completed / max(1, len(tasks)):.1%}); "
                f"last accepted={accepted / max(1, task.trials):.3f}"
            )
    return assemble_payoff(
        case,
        actions,
        config,
        seed_base=seed_base,
        tasks=tasks,
        cell_to_task=cell_to_task,
        results_by_task=results_by_task,
    )


def summarize_cell(route_id: str, action_id: str, results: list[TurnResult],
                   seed_start: int, seed_end: int,
                   ci_half_width_warn: float,
                   simulated_action_id: str | None = None,
                   sequence_timing_extra: dict[str, Any] | None = None,
                   config: Exp3SequenceConfig | None = None) -> CellStats:
    n = len(results)
    accepted = sum(1 for result in results if _accepted_for_cell(result, config))
    cached_accepted = sum(1 for result in results if result.accepted)
    low, high = wilson_interval(accepted, n)
    half_width = (high - low) / 2.0
    warnings = []
    if half_width > ci_half_width_warn:
        warnings.append("accepted_rate_ci_wide")
    if any(result.chsh_adequately_sampled is False for result in results):
        warnings.append("chsh_under_sampled_trial")
    chsh_values = [result.chsh_s for result in results if result.chsh_s is not None]
    qbers = [result.qber for result in results if result.qber is not None]
    sequence_timing = dict(results[0].sequence_timing) if results else {}
    if sequence_timing_extra:
        sequence_timing.update(sequence_timing_extra)
    if config is not None:
        sequence_timing.update({
            "alice_acceptance_rule": config.alice_acceptance_rule,
            "cached_protocol_accepted_count": cached_accepted,
            "effective_accepted_count": accepted,
            "effective_accepted_rate": accepted / n if n else 0.0,
            "qber_is_diagnostic_not_hard_veto": (
                config.alice_acceptance_rule == "chsh_only"
            ),
        })
        if action_id == "no_attack" or simulated_action_id == "no_attack":
            chsh_pass_count = sum(1 for result in results if chsh_passes(result))
            raw_false_signal_count = n - cached_accepted
            effective_false_stop_count = n - accepted
            sequence_timing.update({
                "no_attack_chsh_pass_count": chsh_pass_count,
                "no_attack_chsh_pass_rate": chsh_pass_count / n if n else 0.0,
                "no_attack_false_signal_count": raw_false_signal_count,
                "no_attack_false_signal_rate": (
                    raw_false_signal_count / n if n else 0.0
                ),
                "no_attack_effective_false_stop_abort_count": (
                    effective_false_stop_count
                ),
                "no_attack_effective_false_stop_abort_rate": (
                    effective_false_stop_count / n if n else 0.0
                ),
            })
    return CellStats(
        route_id=route_id,
        action_id=action_id,
        seed_start=seed_start,
        seed_end=seed_end,
        trial_count=n,
        accepted_count=accepted,
        chsh_abort_count=sum(1 for result in results if result.public_outcome == "chsh_abort"),
        qber_abort_count=sum(1 for result in results if result.public_outcome == "qber_abort"),
        delivery_failure_count=sum(1 for result in results if result.public_outcome == "delivery_failure"),
        accepted_rate=accepted / n if n else 0.0,
        accepted_ci_low=low,
        accepted_ci_high=high,
        accepted_ci_half_width=half_width,
        mean_chsh_s=_mean(chsh_values),
        chsh_s_count=len(chsh_values),
        chsh_s_min=_min(chsh_values),
        chsh_s_max=_max(chsh_values),
        chsh_adequately_sampled_count=sum(1 for result in results if result.chsh_adequately_sampled is True),
        mean_qber=_mean(qbers),
        qber_count=len(qbers),
        qber_min=_min(qbers),
        qber_max=_max(qbers),
        mean_delivered_count=sum(result.delivered_count for result in results) / n if n else 0.0,
        mean_sifted_count=sum(result.sifted_count for result in results) / n if n else 0.0,
        sequence_timing=sequence_timing,
        warnings=tuple(warnings),
        simulated_action_id=(
            action_id if simulated_action_id is None else simulated_action_id
        ),
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def payoff_from_dict(payload: dict[str, Any]) -> PayoffEstimate:
    return PayoffEstimate(
        graph_id=str(payload["graph_id"]),
        route_ids=tuple(payload["route_ids"]),
        action_ids=tuple(payload["action_ids"]),
        payoff=np.asarray(payload["payoff"], dtype=float),
        cells=tuple(_cell_from_dict(cell) for cell in payload.get("cells", [])),
        seed_start=int(payload.get("seed_start", 0)),
        seed_end=int(payload.get("seed_end", 0)),
        backend=str(payload.get("backend", "sequence_repeater_e91")),
        security_monitor=str(payload.get("security_monitor", "chsh")),
        trials_per_cell=int(payload.get("trials_per_cell", 0)),
        memory_pairs_per_trial=int(payload.get("memory_pairs_per_trial", 0)),
        ci_half_width_warn=float(payload.get("ci_half_width_warn", 0.0)),
        sequence_trial_config=dict(payload.get("sequence_trial_config") or {}),
    )


def _cell_from_dict(payload: dict[str, Any]) -> CellStats:
    return CellStats(
        route_id=str(payload["route_id"]),
        action_id=str(payload["action_id"]),
        seed_start=int(payload.get("seed_start", 0)),
        seed_end=int(payload.get("seed_end", 0)),
        trial_count=int(payload["trial_count"]),
        accepted_count=int(payload["accepted_count"]),
        chsh_abort_count=int(payload["chsh_abort_count"]),
        qber_abort_count=int(payload["qber_abort_count"]),
        delivery_failure_count=int(payload["delivery_failure_count"]),
        accepted_rate=float(payload["accepted_rate"]),
        accepted_ci_low=float(payload["accepted_ci_low"]),
        accepted_ci_high=float(payload["accepted_ci_high"]),
        accepted_ci_half_width=float(payload["accepted_ci_half_width"]),
        mean_chsh_s=payload.get("mean_chsh_s"),
        chsh_s_count=int(payload.get("chsh_s_count", 0)),
        chsh_s_min=payload.get("chsh_s_min"),
        chsh_s_max=payload.get("chsh_s_max"),
        chsh_adequately_sampled_count=int(payload.get("chsh_adequately_sampled_count", 0)),
        mean_qber=payload.get("mean_qber"),
        qber_count=int(payload.get("qber_count", 0)),
        qber_min=payload.get("qber_min"),
        qber_max=payload.get("qber_max"),
        mean_delivered_count=float(payload["mean_delivered_count"]),
        mean_sifted_count=float(payload.get("mean_sifted_count", 0.0)),
        sequence_timing=dict(payload.get("sequence_timing") or {}),
        warnings=tuple(payload.get("warnings", [])),
        simulated_action_id=str(
            payload.get("simulated_action_id", payload.get("action_id", ""))
        ),
    )


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _min(values: list[float]) -> float | None:
    return None if not values else float(min(values))


def _max(values: list[float]) -> float | None:
    return None if not values else float(max(values))


def _accepted_for_cell(
        result: TurnResult,
        config: Exp3SequenceConfig | None,
) -> bool:
    if config is None:
        return bool(result.accepted)
    return alice_accepts_turn(result, config)
