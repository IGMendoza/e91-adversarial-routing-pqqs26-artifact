"""No-attack SeQUeNCe baseline health checks for the Exp3 corpus."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .acceptance import alice_accepts_turn
from .backend import ActionSpec, SequenceRouteEvaluator
from .baseline_cache import load_cached_baseline_results, route_physics_cache_key
from .config import Exp3SequenceConfig
from .corpus import GraphCase


NO_ATTACK = ActionSpec("no_attack", "none", "", "no_attack")


@dataclass(frozen=True)
class RouteHealthResult:
    route_index: int
    route_id: str
    route_length_km: float
    trial_count: int
    accepted_count: int
    qualified_count: int
    outcome_counts: dict[str, int]
    mean_chsh_s: float | None
    mean_qber: float | None
    mean_delivered_count: float
    mean_sifted_count: float
    timing: dict[str, Any]
    result_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_index": self.route_index,
            "route_id": self.route_id,
            "route_length_km": self.route_length_km,
            "trial_count": self.trial_count,
            "accepted_count": self.accepted_count,
            "qualified_count": self.qualified_count,
            "outcome_counts": dict(self.outcome_counts),
            "mean_chsh_s": self.mean_chsh_s,
            "mean_qber": self.mean_qber,
            "mean_delivered_count": self.mean_delivered_count,
            "mean_sifted_count": self.mean_sifted_count,
            "timing": dict(self.timing),
            "result_source": self.result_source,
        }


@dataclass(frozen=True)
class GraphHealthResult:
    graph_id: str
    family: str
    status: str
    healthy: bool
    trial_count: int
    accepted_count: int
    qualified_count: int
    accepted_rate: float
    qualified_rate: float
    outcome_counts: dict[str, int]
    selected_route_indices: tuple[int, ...]
    route_results: tuple[RouteHealthResult, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "family": self.family,
            "status": self.status,
            "healthy": self.healthy,
            "trial_count": self.trial_count,
            "accepted_count": self.accepted_count,
            "qualified_count": self.qualified_count,
            "accepted_rate": self.accepted_rate,
            "qualified_rate": self.qualified_rate,
            "outcome_counts": dict(self.outcome_counts),
            "selected_route_indices": list(self.selected_route_indices),
            "route_results": [row.to_dict() for row in self.route_results],
            "warnings": list(self.warnings),
        }


def run_graph_health_check(
        case: GraphCase,
        config: Exp3SequenceConfig,
        *,
        seed_base: int,
        progress_callback: Any = None,
) -> GraphHealthResult:
    """Run representative no-attack trials and require at least one acceptance."""

    route_indices = select_health_route_indices(case.routes, config.baseline_health_routes_per_graph)
    if progress_callback:
        progress_callback(
            "baseline health start "
            f"routes={len(case.routes)} selected={list(route_indices)} "
            f"trials={len(route_indices) * config.baseline_health_trials_per_route}"
        )
    route_results = []
    cached_route_count = 0
    live_route_count = 0
    evaluator: SequenceRouteEvaluator | None = None
    seed = seed_base
    for route_index in route_indices:
        route = case.routes[route_index]
        route_seed = seed
        results = load_cached_baseline_results(
            config.baseline_cache_db_path,
            cache_key=route_physics_cache_key(route),
            sample_count=config.baseline_health_trials_per_route,
            seed=route_seed,
        )
        if results is not None:
            result_source = "baseline_cache"
            cached_route_count += 1
        else:
            result_source = "sequence_runtime"
            live_route_count += 1
            if evaluator is None:
                evaluator = SequenceRouteEvaluator(case.ir_dict, case.routes, config)
            results = []
            for trial in range(config.baseline_health_trials_per_route):
                result = evaluator.evaluate(
                    route_index,
                    NO_ATTACK,
                    seed=seed,
                    trial_id=f"{case.graph_id}_baseline_health_r{route_index}_t{trial}",
                )
                results.append(result)
                seed += 1
        route_results.append(_summarize_route(
            route_index,
            route,
            results,
            config,
            result_source=result_source,
        ))
        if result_source == "baseline_cache":
            seed += config.baseline_health_trials_per_route

    if progress_callback:
        progress_callback(
            "baseline health source "
            f"cached_routes={cached_route_count} "
            f"sequence_routes={live_route_count}"
        )

    total_trials = sum(row.trial_count for row in route_results)
    accepted = sum(row.accepted_count for row in route_results)
    qualified = sum(row.qualified_count for row in route_results)
    outcomes = Counter()
    for row in route_results:
        outcomes.update(row.outcome_counts)
    accepted_rate = accepted / total_trials if total_trials else 0.0
    qualified_rate = qualified / total_trials if total_trials else 0.0
    warnings = []
    if accepted < config.baseline_health_min_accepted_trials:
        warnings.append("no_no_attack_acceptance")
    if qualified < config.baseline_health_min_accepted_trials:
        warnings.append("no_qualified_no_attack_acceptance")
    if accepted_rate < config.baseline_health_accept_rate_warn:
        warnings.append("low_no_attack_acceptance_rate")
    if qualified < accepted:
        warnings.append("accepted_but_bell_margin_or_delivery_low")
    for outcome in ("chsh_abort", "qber_abort", "delivery_failure"):
        if total_trials > 0 and outcomes.get(outcome, 0) == total_trials:
            warnings.append(f"all_{outcome}")
    healthy = qualified >= config.baseline_health_min_accepted_trials
    status = "healthy" if healthy else "no_attack_baseline_failed"
    if progress_callback:
        progress_callback(
            "baseline health done "
            f"accepted={accepted}/{total_trials} "
            f"qualified={qualified}/{total_trials} "
            f"outcomes={dict(outcomes)} status={status}"
        )
    return GraphHealthResult(
        graph_id=case.graph_id,
        family=case.family,
        status=status,
        healthy=healthy,
        trial_count=total_trials,
        accepted_count=accepted,
        qualified_count=qualified,
        accepted_rate=accepted_rate,
        qualified_rate=qualified_rate,
        outcome_counts=dict(outcomes),
        selected_route_indices=tuple(route_indices),
        route_results=tuple(route_results),
        warnings=tuple(warnings),
    )


def select_health_route_indices(routes: list[dict[str, Any]], count: int) -> tuple[int, ...]:
    if not routes:
        return ()
    ordered = sorted(
        range(len(routes)),
        key=lambda idx: (
            float(routes[idx].get("total_length_m", routes[idx].get("total_length", 0.0)) or 0.0),
            idx,
        ),
    )
    candidates = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    selected = []
    for idx in candidates:
        if idx not in selected:
            selected.append(idx)
        if len(selected) >= count:
            break
    for idx in ordered:
        if len(selected) >= count:
            break
        if idx not in selected:
            selected.append(idx)
    return tuple(selected)


def _summarize_route(
        route_index: int,
        route: dict[str, Any],
        results: list[Any],
        config: Exp3SequenceConfig,
        *,
        result_source: str,
) -> RouteHealthResult:
    outcomes = Counter(str(result.public_outcome) for result in results)
    accepted = sum(1 for result in results if alice_accepts_turn(result, config))
    qualified = sum(
        1 for result in results
        if (
            alice_accepts_turn(result, config)
            and result.chsh_s is not None
            and float(result.chsh_s) >= config.baseline_health_min_chsh_s
            and int(result.delivered_count) >= config.baseline_health_min_delivered_pairs
        )
    )
    chsh_values = [float(result.chsh_s) for result in results if result.chsh_s is not None]
    qbers = [float(result.qber) for result in results if result.qber is not None]
    return RouteHealthResult(
        route_index=route_index,
        route_id=str(route["route_id"]),
        route_length_km=float(route.get("total_length_m", 0.0)) / 1000.0,
        trial_count=len(results),
        accepted_count=accepted,
        qualified_count=qualified,
        outcome_counts=dict(outcomes),
        mean_chsh_s=_mean(chsh_values),
        mean_qber=_mean(qbers),
        mean_delivered_count=_mean([float(result.delivered_count) for result in results]) or 0.0,
        mean_sifted_count=_mean([float(result.sifted_count) for result in results]) or 0.0,
        timing={
            **(dict(results[0].sequence_timing) if results else {}),
            "alice_acceptance_rule": config.alice_acceptance_rule,
            "qber_is_diagnostic_not_hard_veto": (
                config.alice_acceptance_rule == "chsh_only"
            ),
        },
        result_source=result_source,
    )


def _mean(values: list[float]) -> float | None:
    return None if not values else float(sum(values) / len(values))
