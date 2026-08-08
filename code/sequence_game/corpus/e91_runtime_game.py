"""Graph-route E91 runtime game for paper and corpus attack-surface runs.

This backend uses SeQUeNCe E91/BBM92 route runners, not the binary-collision
control model and not the GUI fixture. Alice chooses among the graph's
candidate routes. Eve chooses attacks over the route corridor:

* ``edge_dos`` maps to the SeQUeNCe ``added_loss`` Eve station.
* ``edge_information_probe`` maps to the SeQUeNCe ``intercept_resend`` Eve
  station and records empirical ``I(K;E|P)`` via ``compute_information_gain``.
* ``swap_denial`` and ``memory_degradation`` map to selected-route repeater
  runtime attacks on interior route nodes.
* ``repeater_memory_measure_Z`` and ``repeater_memory_measure_X`` map to
  pre-swap repeater-memory measurement hooks on interior route nodes.

Only aggregate information metrics are emitted. Raw keys, bases, outcomes,
Eve observations, and sifted indices remain private to the trial internals.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.periods import detect_route_period
from ..claims import (
    IMPLEMENTED_AND_TESTABLE,
    REPEATER_RUNTIME,
    assert_no_private_public_fields,
    validate_information_gain_reward,
)
from ..eve.information import InfoGainResult, compute_information_gain
from ..eve.repeater_attacks import RepeaterAttackSpec
from ..eve.sequence_attacks import (
    EdgeAttackSpec,
    build_attacked_hops,
    make_station_factory,
    spec_active_on_path,
)
from ..physical import load_models_from_dir
from ..protocol.postprocessing import SiftingConfig
from ..protocol.repeater_trial import (
    RepeaterE91RunConfig,
    run_fixed_repeater_chsh_trial,
    run_fixed_repeater_e91_trial,
)
from ..protocol.sequence_trial import E91RunConfig, run_e91_trial
from ..rl.q_learning import QLearningConfig, QTable, epsilon_greedy
from ..routing.route import route_id_for_path
from ..sequence_build.repeater_e91_builder import FixedRepeaterPath
from ..topology.ir import EdgeRecord, NodeRecord, TopologyIR, TopologyMetadata

DEFAULT_E91_PERIODS_PER_STEP = 1200
DEFAULT_E91_QBER_THRESHOLD = 0.15
DEFAULT_E91_MIN_SIFTED_SAMPLES = 20
DEFAULT_REPEATER_MEMORY_PAIRS_PER_STEP = 8
DEFAULT_REPEATER_MIN_SIFTED_SAMPLES = 1
# A valid per-step CHSH Bell test needs >=30 samples per setting across 9
# settings (chsh_core.DEFAULT_MIN_CHSH_CELL_COUNT); ~350 delivered pairs keeps
# every CHSH cell comfortably above the guard.
DEFAULT_REPEATER_CHSH_PAIRS_PER_STEP = 350
E91_RUNTIME_BACKEND = "graph_route_sequence_e91_runtime"
_ATTACK_SURFACE_MODES = frozenset({
    "attack_surface_dos",
    "attack_surface_mixed",
    "attack_surface_all",
})
_INFORMATION_GAIN_MODES = frozenset({
    "attack_surface_mixed",
    "attack_surface_all",
})
# E91: Eve wins only by availability denial (the key is rejected). There is no
# confidentiality/"information_exposure" win, because an accepted key is provably
# information-free under device-independent E91 security. Denial reasons:
#   delivery_failure - too few pairs delivered (DoS / undersampled block)
#   qber_abort       - Bell-secure but key error rate above threshold
#   chsh_abort       - Bell violation lost (S <= 2): no security certificate
_EVE_WIN_OUTCOMES = frozenset({
    "delivery_failure",
    "qber_abort",
    "chsh_abort",
})
_DENIAL_OUTCOMES = _EVE_WIN_OUTCOMES


def _information_gain_enabled(game_mode: str) -> bool:
    return game_mode in _INFORMATION_GAIN_MODES


def demo_topology_to_ir_dict(topology: dict[str, Any], *, graph_id: str = "paper_topology") -> dict[str, Any]:
    """Convert a demo/paper topology dict into a TopologyIR dict."""

    node_ids = [str(node["id"]) for node in topology.get("nodes", [])]
    nodes = {node_id: {"node_id": node_id, "roles": [], "coordinates": [0.0, 0.0]}
             for node_id in node_ids}
    for node in topology.get("nodes", []):
        node_id = str(node["id"])
        nodes[node_id]["coordinates"] = [
            float(node.get("x", 0.0)),
            float(node.get("y", 0.0)),
        ]
    edges = []
    for edge in topology.get("edges", []):
        u = str(edge["u"])
        v = str(edge["v"])
        edges.append({
            "edge_id": _edge_label(u, v),
            "u": u,
            "v": v,
            "length_m": float(edge.get("length_m", edge.get("length", 1.0))),
            "eve_eligible": True,
        })
    return {
        "nodes": nodes,
        "edges": edges,
        "metadata": {"generator": "demo_topology", "params": {"graph_id": graph_id}, "seed": None},
    }


def e91_runtime_action_count(
    ir_dict: dict[str, Any],
    routes: list[dict[str, Any]],
    game_mode: str,
) -> int:
    ir = TopologyIR.from_dict(ir_dict)
    return len(_runtime_actions(ir, routes, game_mode))


def run_e91_graph_runtime_game(
    ir_dict: dict[str, Any],
    alice: str,
    bob: str,
    *,
    routes: list[dict[str, Any]] | None,
    game_mode: str,
    eve_algo: str,
    alice_algo: str,
    num_steps: int,
    seed: int,
    final_window_size: int = 20,
    e91_periods_per_step: int = DEFAULT_E91_PERIODS_PER_STEP,
    repeater_memory_pairs_per_step: int = DEFAULT_REPEATER_MEMORY_PAIRS_PER_STEP,
    attack_cost: float = 0.05,
    dt_refit_observer: Any = None,
    dt_refit_callback: Any = None,
    progress_every: int = 0,
    progress_label: str | None = None,
    security_monitor: str = "qber",
) -> dict[str, Any]:
    """Run one graph attack-surface combo through SeQUeNCe E91 route trials.

    ``security_monitor`` selects the repeater-route key acceptance test:
    ``"qber"`` (default) uses the Z/X BBM92 raw-key + QBER runner; ``"chsh"``
    uses the physical Ekert-91 CHSH-S Bell monitor
    (:func:`run_fixed_repeater_chsh_trial`), which emits ``chsh_abort`` when the
    Bell violation is lost. A valid CHSH block needs many delivered pairs, so in
    ``"chsh"`` mode ``repeater_memory_pairs_per_step`` is raised to at least
    :data:`DEFAULT_REPEATER_CHSH_PAIRS_PER_STEP`.
    """

    if game_mode not in _ATTACK_SURFACE_MODES:
        raise ValueError(f"E91 runtime backend does not support {game_mode!r}")
    if security_monitor not in ("qber", "chsh"):
        raise ValueError(f"unsupported security_monitor {security_monitor!r}")
    if num_steps < 1:
        raise ValueError("num_steps must be >= 1")
    if e91_periods_per_step < 1:
        raise ValueError("e91_periods_per_step must be >= 1")
    if repeater_memory_pairs_per_step < 1:
        raise ValueError("repeater_memory_pairs_per_step must be >= 1")
    if security_monitor == "chsh":
        # A statistically valid per-step Bell test needs many delivered pairs.
        repeater_memory_pairs_per_step = max(
            repeater_memory_pairs_per_step, DEFAULT_REPEATER_CHSH_PAIRS_PER_STEP)
    if _information_gain_enabled(game_mode):
        # Information gain is measured as a diagnostic in mixed/all modes but is
        # never rewarded (weight 0): under device-independent E91 security Eve
        # holds no information about an accepted key, and a rejected key is
        # discarded, so there is no confidentiality reward.
        validate_information_gain_reward(
            0.0,
            metric_source="sequence_game.eve.information.compute_information_gain",
            public_transcript_mapping=(
                "aggregate empirical I(K;E|P) diagnostic; not rewarded; "
                "K/E samples and sifted indices remain private"
            ),
        )

    started = time.perf_counter()
    ir = TopologyIR.from_dict(ir_dict)
    runtime_routes = _normalize_routes(ir, alice, bob, routes)
    if not runtime_routes:
        raise ValueError(f"no Alice-Bob routes available for {alice!r}->{bob!r}")
    actions = _runtime_actions(ir, runtime_routes, game_mode)
    source_model, detector_model, fiber_model, memory_model = _default_models()
    edge_run_config = E91RunConfig(
        num_periods=e91_periods_per_step,
        sifting=SiftingConfig(
            qber_threshold=DEFAULT_E91_QBER_THRESHOLD,
            min_sifted_samples=DEFAULT_E91_MIN_SIFTED_SAMPLES,
        ),
    )
    repeater_run_config = RepeaterE91RunConfig(
        memory_pairs=repeater_memory_pairs_per_step,
        sifting=SiftingConfig(
            qber_threshold=DEFAULT_E91_QBER_THRESHOLD,
            min_sifted_samples=DEFAULT_REPEATER_MIN_SIFTED_SAMPLES,
        ),
        request_fidelity=0.01,
        start_time_ps=1_000_000,
        end_time_ps=80_000_000,
        stop_time_ps=100_000_000,
        swapping_success_prob=1.0,
        swapping_degradation=1.0,
        memory_efficiency_override=_multiplexed_memory_efficiency(memory_model),
    )

    rng = np.random.default_rng(seed)
    eve_q = _make_agent(eve_algo, len(actions))
    alice_q = _make_agent(alice_algo, len(runtime_routes))
    q_cfg = QLearningConfig(alpha=0.2, gamma=0.9)

    avoided_until: dict[str, int] = {}
    previous_eve_action = "start"
    previous_public_outcome = "start"
    previous_route_id = "start"
    steps: list[dict[str, Any]] = []
    step_seconds: list[float] = []
    progress_interval = max(0, int(progress_every or 0))
    progress_name = progress_label or f"{game_mode}__{eve_algo}__{alice_algo}"
    eve_win_count = 0

    for step_idx in range(num_steps):
        step_started = time.perf_counter()
        route, route_idx, alice_state, fallback_used = _select_route(
            runtime_routes,
            alice_algo,
            alice_q,
            previous_route_id,
            previous_public_outcome,
            avoided_until,
            step_idx,
            num_steps,
            rng,
        )
        action, eve_action_idx, eve_state = _select_action(
            actions,
            route,
            eve_algo,
            eve_q,
            previous_eve_action,
            previous_public_outcome,
            step_idx,
            num_steps,
            rng,
        )
        trial_id = (
            f"e91_graph_{game_mode}_{eve_algo}_{alice_algo}_"
            f"seed{seed}_step{step_idx}_{route['route_id']}_{action['id']}"
        )
        transcript, info, active_route_attacked, runtime_metrics = _run_selected_action(
            ir=ir,
            route=route,
            action=action,
            source_model=source_model,
            detector_model=detector_model,
            fiber_model=fiber_model,
            memory_model=memory_model,
            edge_run_config=edge_run_config,
            repeater_run_config=repeater_run_config,
            trial_id=trial_id,
            seed=seed + 100_000 + step_idx,
            eve_name=f"eve_{step_idx}",
            attack_cost=attack_cost,
            security_monitor=security_monitor,
        )
        public_outcome = _public_outcome(transcript, info, action, game_mode)
        if public_outcome in _EVE_WIN_OUTCOMES:
            eve_win_count += 1
        failure_or_detection = public_outcome in _DENIAL_OUTCOMES
        if failure_or_detection and alice_algo == "cautious_greedy":
            avoided_until[route["route_id"]] = step_idx + 1

        eve_reward, reward_components = _eve_reward_components(
            public_outcome,
            action,
            info,
            game_mode,
            attack_cost,
        )
        alice_reward = _alice_reward(public_outcome)
        if fallback_used:
            alice_reward -= 0.2

        next_eve_state = (action["id"], public_outcome)
        if eve_algo in {"q_learning", "deep_q"}:
            eve_q.update(eve_state, eve_action_idx, eve_reward, next_eve_state, q_cfg)
        if alice_algo in {"q_learning", "deep_q"}:
            next_avoid = _avoidance_signature(runtime_routes, avoided_until, step_idx)
            next_alice_state = (route["route_id"], public_outcome, next_avoid)
            alice_q.update(alice_state, route_idx, alice_reward, next_alice_state, q_cfg)

        record = _step_record(
            step_idx,
            game_mode,
            alice_algo,
            eve_algo,
            route,
            action,
            public_outcome,
            alice_reward,
            eve_reward,
            fallback_used,
            active_route_attacked=active_route_attacked,
            transcript=transcript,
            info=info,
            reward_components=reward_components,
            e91_periods_per_step=e91_periods_per_step,
            repeater_memory_pairs_per_step=repeater_memory_pairs_per_step,
            runtime_metrics=runtime_metrics,
        )
        steps.append(record)
        step_elapsed = time.perf_counter() - step_started
        step_seconds.append(step_elapsed)
        step_num = step_idx + 1
        if progress_interval and (
            step_num % progress_interval == 0 or step_num == num_steps
        ):
            elapsed = time.perf_counter() - started
            print(
                "  [case-progress] "
                f"{progress_name} step={step_num}/{num_steps} "
                f"routes={len(runtime_routes)} actions={len(actions)} "
                f"eve_win={eve_win_count / step_num:.2%} "
                f"mean_step={elapsed / step_num:.3f}s "
                f"elapsed={elapsed:.1f}s "
                f"last={public_outcome}",
                flush=True,
            )

        if dt_refit_observer is not None:
            dt_refit_observer.record_transition(
                alice_action=route_idx,
                eve_action=eve_action_idx,
                alice_route=route.get("route_id"),
                eve_target=action.get("target"),
                reward=eve_reward,
            )
            tick = dt_refit_observer.observe(
                step=step_idx + 1,
                alice_q=alice_q if alice_algo in {"q_learning", "deep_q"} else None,
                eve_q=eve_q if eve_algo in {"q_learning", "deep_q"} else None,
            )
            if tick is not None and tick.refit_recommended:
                event = dt_refit_observer.refit_event_from_tick(tick)
                if dt_refit_callback is not None:
                    dt_refit_callback(event, {
                        "step": step_idx + 1,
                        "routes": runtime_routes,
                        "actions": actions,
                        "steps": steps,
                        "alice_q": alice_q,
                        "eve_q": eve_q,
                        "alice_algo": alice_algo,
                        "eve_algo": eve_algo,
                        "control_model": game_mode,
                    })
                dt_refit_observer.mark_refit(
                    step=step_idx + 1,
                    alice_q=alice_q if alice_algo in {"q_learning", "deep_q"} else None,
                    eve_q=eve_q if eve_algo in {"q_learning", "deep_q"} else None,
                    reason=tick.refit_reason,
                )

        previous_eve_action = action["id"]
        previous_public_outcome = public_outcome
        previous_route_id = route["route_id"]

    final_window = steps[-min(final_window_size, len(steps)):]
    summary = _summary(
        mode=game_mode,
        seed=seed,
        alice=alice,
        bob=bob,
        alice_algo=alice_algo,
        eve_algo=eve_algo,
        routes=runtime_routes,
        actions=actions,
        steps=steps,
        final_window=final_window,
        final_window_size=final_window_size,
        eve_q=eve_q,
        alice_q=alice_q,
        include_alice_q=alice_algo in {"q_learning", "deep_q"},
        started=started,
        step_seconds=step_seconds,
        e91_periods_per_step=e91_periods_per_step,
        repeater_memory_pairs_per_step=repeater_memory_pairs_per_step,
    )
    return {
        "summary": summary,
        "steps": steps,
        "final_window": final_window,
        "scope_label": REPEATER_RUNTIME,
    }


def _default_config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "physical"


@lru_cache(maxsize=1)
def _default_models() -> tuple[Any, Any, Any, Any]:
    registry = load_models_from_dir(_default_config_dir(), require_resolved=True)

    def by_kind(kind: str) -> Any:
        return next(model for model in registry.models.values() if model.device_kind == kind)

    return (
        by_kind("source"),
        by_kind("detector"),
        by_kind("fiber_channel"),
        by_kind("memory"),
    )


def _multiplexed_memory_efficiency(memory_model: Any) -> float | None:
    """Effective multiplexed memory efficiency for the SeQUeNCe repeater path.

    SeQUeNCe's Memory has a single per-attempt efficiency scalar and no native
    multiplexing, so a multiplexed node (e.g. Cui2025's multiple-excitation
    scheme, effective branching ratio BR_max) is approximated by substituting
    the effective branching ratio for the single-mode per-excitation efficiency.
    Returns None when the model carries no multiplexing_model, leaving the
    single-mode value in place. The model's ``efficiency`` field is never
    mutated, so other backends keep the single-mode value.
    """

    model = getattr(memory_model, "parameters", {}).get("multiplexing_model")
    if not isinstance(model, dict):
        return None
    effective = model.get("effective_branching_ratio")
    return None if effective is None else float(effective)


def _zero_information_gain(transcript: Any) -> InfoGainResult:
    return InfoGainResult(
        num_sifted=len(getattr(transcript, "sifted_indices", ()) or ()),
        num_eve_records=0,
        mutual_information_bits=0.0,
        fraction_correct=0.5,
    )


def _run_selected_action(
    *,
    ir: TopologyIR,
    route: dict[str, Any],
    action: dict[str, Any],
    source_model: Any,
    detector_model: Any,
    fiber_model: Any,
    memory_model: Any,
    edge_run_config: E91RunConfig,
    repeater_run_config: RepeaterE91RunConfig,
    trial_id: str,
    seed: int,
    eve_name: str,
    attack_cost: float,
    security_monitor: str = "qber",
) -> tuple[Any, InfoGainResult, bool, dict[str, Any]]:
    if action.get("kind") == "node":
        return _run_repeater_node_action(
            route=route,
            action=action,
            memory_model=memory_model,
            fiber_model=fiber_model,
            run_config=repeater_run_config,
            trial_id=trial_id,
            seed=seed,
            attack_cost=attack_cost,
            security_monitor=security_monitor,
        )
    spec = _action_to_edge_attack(action, attack_cost=attack_cost)
    active_route_attacked = spec_active_on_path(route["path"], spec)
    hops, eve_nodes = build_attacked_hops(
        ir,
        route["path"],
        fiber_model,
        spec,
        eve_name=eve_name,
    )
    factory = make_station_factory(spec, eve_name=eve_name)
    transcript, built = run_e91_trial(
        hops,
        alice=route["path"][0],
        bob=route["path"][-1],
        source_model=source_model,
        detector_model=detector_model,
        run_config=edge_run_config,
        trial_id=trial_id,
        seed=seed,
        station_factory=factory,
        eve_nodes=eve_nodes,
    )
    info = compute_information_gain(transcript, built)
    runtime_metrics = {
        "runtime_engine": "edge_e91_trial",
        "runtime_attack_applied": spec.to_dict(),
        "configured_repeater_attack": None,
        "edge_attack_kind": spec.kind,
        "node_attack_kind": None,
        "fidelity": None,
        "amer_style_d_eff_mean": None,
        "memory_intervention_public": None,
        "router_net_summary": None,
    }
    return transcript, info, active_route_attacked, runtime_metrics


def _run_repeater_node_action(
    *,
    route: dict[str, Any],
    action: dict[str, Any],
    memory_model: Any,
    fiber_model: Any,
    run_config: RepeaterE91RunConfig,
    trial_id: str,
    seed: int,
    attack_cost: float,
    security_monitor: str = "qber",
) -> tuple[Any, InfoGainResult, bool, dict[str, Any]]:
    configured_spec = _action_to_repeater_attack(action, attack_cost=attack_cost)
    path = tuple(str(node) for node in route["path"])
    active_route_attacked = (
        configured_spec.is_active and configured_spec.target_node in set(path[1:-1])
    )
    spec = configured_spec if active_route_attacked else RepeaterAttackSpec(kind="none")
    edge_lengths = tuple(
        float(length)
        for length in route.get("edge_lengths_m", ())
    )
    fixed_path = FixedRepeaterPath(
        nodes=path,
        edge_lengths_m=edge_lengths or tuple(
            float(route.get("total_length_m", route.get("total_length", 0.0)))
            / max(1, len(path) - 1)
            for _ in range(max(0, len(path) - 1))
        ),
    )
    runner = (run_fixed_repeater_chsh_trial if security_monitor == "chsh"
              else run_fixed_repeater_e91_trial)
    transcript, built = runner(
        fixed_path,
        memory_model=memory_model,
        fiber_model=fiber_model,
        run_config=run_config,
        trial_id=trial_id,
        seed=seed,
        attack=spec,
    )
    summary = {
        key: value for key, value in built.summary().items()
        if key != "config_path"
    }
    memory_public = summary.get("pre_swap_memory_intervention_public")
    runtime_metrics = {
        "runtime_engine": (
            "repeater_chsh_trial" if security_monitor == "chsh"
            else "repeater_e91_trial"),
        "runtime_attack_applied": spec.to_dict(),
        "configured_repeater_attack": configured_spec.to_dict(),
        "edge_attack_kind": None,
        "node_attack_kind": spec.kind,
        "fidelity": transcript.extra.get("mean_actual_fidelity"),
        "amer_style_d_eff_mean": transcript.extra.get("amer_style_d_eff_mean"),
        "memory_intervention_public": memory_public,
        "router_net_summary": summary,
        # CHSH-mode Bell diagnostics (None under the Z/X monitor). Only surface S
        # when the block is adequately sampled; an undersampled S is noise and can
        # spuriously exceed the Tsirelson bound, so it must not be presented as a
        # physical value (the outcome is delivery_failure in that case anyway).
        "chsh_s": (
            transcript.extra.get("chsh_s")
            if transcript.extra.get("chsh_adequately_sampled") else None),
        "chsh_violates_bell": (
            transcript.extra.get("chsh_violates_bell")
            if transcript.extra.get("chsh_adequately_sampled") else None),
        "chsh_adequately_sampled": transcript.extra.get("chsh_adequately_sampled"),
    }
    return transcript, _zero_information_gain(transcript), active_route_attacked, runtime_metrics


def _make_agent(algo: str, num_actions: int):
    if algo == "deep_q":
        from ..rl.deep_q_learning import DeepQNetwork

        return DeepQNetwork(num_actions)
    return QTable(num_actions)


def _normalize_routes(
    ir: TopologyIR,
    alice: str,
    bob: str,
    routes: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if routes is None:
        from ..routing.route import enumerate_simple_paths, make_route

        route_objs = [
            make_route(ir, path)
            for path in enumerate_simple_paths(ir, alice, bob)
        ]
        routes = [route.to_dict() for route in route_objs]
    normalized = []
    for route in routes:
        path = [str(node) for node in route.get("path", [])]
        if len(path) < 2:
            continue
        edge_ids = [_edge_label(u, v) for u, v in zip(path, path[1:])]
        edge_lengths = [
            float(ir.edge_between(u, v).length_m)  # type: ignore[union-attr]
            for u, v in zip(path, path[1:])
        ]
        total_length = sum(
            edge_lengths
        )
        normalized.append({
            "route_id": str(route.get("route_id") or route_id_for_path(tuple(path))),
            "path": path,
            "edge_ids": edge_ids,
            "edge_lengths_m": edge_lengths,
            "total_length": total_length,
            "total_length_m": total_length,
            "hop_count": len(edge_ids),
            "internal_nodes": path[1:-1],
        })
    normalized.sort(key=lambda row: (
        int(row["hop_count"]),
        float(row["total_length"]),
        tuple(row["path"]),
    ))
    return normalized


def _runtime_actions(
    ir: TopologyIR,
    routes: list[dict[str, Any]],
    game_mode: str,
) -> list[dict[str, Any]]:
    if game_mode not in _ATTACK_SURFACE_MODES:
        raise ValueError(f"unsupported E91 runtime game mode {game_mode!r}")
    edge_targets = sorted({
        _edge_label(u, v)
        for route in routes
        for u, v in zip(route["path"], route["path"][1:])
        if ir.edge_between(u, v) is not None
    })
    node_targets = sorted({
        str(node)
        for route in routes
        for node in route.get("internal_nodes", route["path"][1:-1])
    })
    actions: list[dict[str, Any]] = [{
        "id": "no_attack",
        "kind": "none",
        "target": "",
        "attack_type": "no_attack",
        "cost": 0.0,
    }]
    actions.extend({
        "id": f"edge_dos:{target}",
        "kind": "edge",
        "target": target,
        "attack_type": "edge_dos",
        "cost": 0.05,
    } for target in edge_targets)
    for target in node_targets:
        actions.append({
            "id": f"swap_denial:{target}",
            "kind": "node",
            "target": target,
            "attack_type": "swap_denial",
            "cost": 0.05,
        })
        actions.append({
            "id": f"memory_degradation:{target}",
            "kind": "node",
            "target": target,
            "attack_type": "memory_degradation",
            "cost": 0.05,
        })
    if _information_gain_enabled(game_mode):
        actions.extend({
            "id": f"edge_information_probe:{target}",
            "kind": "edge",
            "target": target,
            "attack_type": "edge_information_probe",
            "cost": 0.05,
        } for target in edge_targets)
        for target in node_targets:
            actions.append({
                "id": f"repeater_memory_measure_Z:{target}",
                "kind": "node",
                "target": target,
                "attack_type": "repeater_memory_measure_Z",
                "cost": 0.05,
            })
            actions.append({
                "id": f"repeater_memory_measure_X:{target}",
                "kind": "node",
                "target": target,
                "attack_type": "repeater_memory_measure_X",
                "cost": 0.05,
            })
    return actions


def _edge_label(u: str, v: str) -> str:
    a, b = sorted((str(u), str(v)))
    return f"{a}-{b}"


def _edge_endpoints(ir: TopologyIR, target: str) -> tuple[str, str]:
    for edge in ir.edges:
        labels = {
            str(edge.edge_id),
            _edge_label(edge.u, edge.v),
            f"{min(edge.u, edge.v)}--{max(edge.u, edge.v)}",
        }
        if target in labels:
            return edge.u, edge.v
    if "-" in target:
        u, v = target.split("-", 1)
        return u, v
    raise ValueError(f"cannot resolve edge target {target!r}")


def _action_to_edge_attack(action: dict[str, Any], *, attack_cost: float) -> EdgeAttackSpec:
    attack_type = action.get("attack_type")
    if attack_type == "no_attack":
        return EdgeAttackSpec(kind="none", cost=0.0)
    u, v = action["target"].split("-", 1)
    if attack_type == "edge_dos":
        return EdgeAttackSpec(
            kind="added_loss",
            target_u=u,
            target_v=v,
            drop_probability=1.0,
            cost=attack_cost,
        )
    if attack_type == "edge_information_probe":
        return EdgeAttackSpec(
            kind="intercept_resend",
            target_u=u,
            target_v=v,
            basis_choice="random",
            cost=attack_cost,
        )
    raise ValueError(f"unsupported runtime action {action!r}")


def _action_to_repeater_attack(
    action: dict[str, Any],
    *,
    attack_cost: float,
) -> RepeaterAttackSpec:
    attack_type = action.get("attack_type")
    target = str(action.get("target") or "")
    if attack_type == "no_attack":
        return RepeaterAttackSpec(kind="none", cost=0.0)
    if attack_type == "swap_denial":
        return RepeaterAttackSpec(
            kind="swap_denial",
            target_node=target,
            attack_id=str(action["id"]),
            cost=attack_cost,
        )
    if attack_type == "memory_degradation":
        return RepeaterAttackSpec(
            kind="memory_degradation",
            target_node=target,
            memory_fidelity_multiplier=0.5,
            attack_id=str(action["id"]),
            cost=attack_cost,
        )
    if attack_type in {"repeater_memory_measure_Z", "repeater_memory_measure_X"}:
        basis = "Z" if attack_type.endswith("_Z") else "X"
        return RepeaterAttackSpec(
            kind="repeater_memory_measurement",
            target_node=target,
            basis=basis,
            attack_id=str(action["id"]),
            cost=attack_cost,
        )
    raise ValueError(f"unsupported repeater runtime action {action!r}")


_MAX_LITERAL_AVOIDANCE_BITS = 64


def _avoidance_signature(
    routes: list[dict[str, Any]],
    avoided_until: dict[str, int],
    step_idx: int,
) -> str:
    bits = "".join(
        "1" if step_idx <= avoided_until.get(route["route_id"], -1) else "0"
        for route in routes
    )
    if len(bits) <= _MAX_LITERAL_AVOIDANCE_BITS:
        return bits
    # Route-dense graphs would otherwise embed a num_routes-length bitstring
    # in every Alice state tuple. The digest is an injective-in-practice
    # relabeling: distinct bitsets still map to distinct state tokens, so
    # learning dynamics are unchanged.
    digest = hashlib.sha1(bits.encode("ascii")).hexdigest()[:16]
    return f"b{len(bits)}:{digest}"


def _select_route(
    routes: list[dict[str, Any]],
    alice_algo: str,
    alice_q: Any,
    previous_route_id: str,
    previous_public_outcome: str,
    avoided_until: dict[str, int],
    step_idx: int,
    num_steps: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], int, tuple[str, ...] | None, bool]:
    avoided_signature = _avoidance_signature(routes, avoided_until, step_idx)
    if alice_algo in {"q_learning", "deep_q"}:
        state = (previous_route_id, previous_public_outcome, avoided_signature)
        epsilon = max(0.05, 0.3 * (1 - step_idx / max(1, num_steps - 1)))
        route_idx = epsilon_greedy(alice_q, state, epsilon, rng)
        return routes[route_idx], route_idx, state, False
    if alice_algo == "random":
        route_idx = int(rng.integers(len(routes)))
        return routes[route_idx], route_idx, None, False
    if alice_algo == "cautious_greedy":
        available = [
            route for route in routes
            if step_idx > avoided_until.get(route["route_id"], -1)
        ]
        fallback_used = not available
        route = (available or routes)[0]
        return route, routes.index(route), None, fallback_used
    raise ValueError(f"unsupported Alice algorithm {alice_algo!r}")


def _select_action(
    actions: list[dict[str, Any]],
    route: dict[str, Any],
    eve_algo: str,
    eve_q: Any,
    previous_eve_action: str,
    previous_public_outcome: str,
    step_idx: int,
    num_steps: int,
    rng: np.random.Generator,
) -> tuple[dict[str, Any], int, tuple[str, ...]]:
    if eve_algo == "route_aware_oracle":
        route_edges = set(route.get("edge_ids") or [])
        action = next(
            (
                candidate for candidate in actions
                if candidate["attack_type"] == "edge_dos"
                and candidate["target"] in route_edges
            ),
            actions[0],
        )
        return action, actions.index(action), ("privileged_route", route["route_id"])
    state = (previous_eve_action, previous_public_outcome)
    if eve_algo == "random":
        action_idx = int(rng.integers(len(actions)))
        return actions[action_idx], action_idx, state
    if eve_algo in {"q_learning", "deep_q"}:
        epsilon = max(0.05, 0.35 * (1 - step_idx / max(1, num_steps - 1)))
        action_idx = epsilon_greedy(eve_q, state, epsilon, rng)
        return actions[action_idx], action_idx, state
    raise ValueError(f"unsupported Eve algorithm {eve_algo!r}")


def _public_outcome(
    transcript: Any,
    info: InfoGainResult,
    action: dict[str, Any],
    game_mode: str,
) -> str:
    # E91 device-independent security: an accepted key is provably free of Eve
    # information, so an accepted key is always an Alice win regardless of any
    # measured information-gain diagnostic. Eve has no confidentiality win; her
    # only win is availability denial (the key is rejected). Information-bearing
    # moves that actually disturb the state are caught here as qber_abort.
    if transcript.accepted:
        return "accepted"
    # CHSH backend emits its decision directly as the abort reason; the Z/X
    # backend uses "qber_above_threshold". Bell-violation loss is chsh_abort.
    if transcript.abort_reason == "chsh_abort":
        return "chsh_abort"
    if transcript.abort_reason in ("qber_above_threshold", "qber_abort"):
        return "qber_abort"
    return "delivery_failure"


def _eve_reward_components(
    public_outcome: str,
    action: dict[str, Any],
    info: InfoGainResult,
    game_mode: str,
    attack_cost: float,
) -> tuple[float, dict[str, Any]]:
    # E91 has no confidentiality win for Eve: an accepted key is provably
    # information-free, and a rejected key is discarded, so Eve keeps nothing in
    # either case. Her reward is availability denial only. mutual_information_bits
    # remains a measured diagnostic (mixed/all modes) but is never rewarded.
    denial_reward = 1.0 if public_outcome in _DENIAL_OUTCOMES else 0.0
    info_reward = 0.0
    cost = 0.0 if action.get("attack_type") == "no_attack" else float(attack_cost)
    reward = denial_reward + info_reward - cost
    components = {
        "scope_label": REPEATER_RUNTIME,
        "public_outcome_reward": denial_reward,
        "information_gain_reward": info_reward,
        "information_gain_rewarded": False,
        "attack_cost": cost,
        "mixed_reward_enabled": False,
        "information_gain_measured": _information_gain_enabled(game_mode),
        "metric_source": (
            "sequence_game.eve.information.compute_information_gain"
            if _information_gain_enabled(game_mode)
            else None
        ),
    }
    return float(reward), components


def _alice_reward(public_outcome: str) -> float:
    if public_outcome == "accepted":
        return 1.0
    return -1.0


def _step_record(
    step_idx: int,
    mode: str,
    alice_policy: str,
    eve_policy: str,
    route: dict[str, Any],
    action: dict[str, Any],
    public_outcome: str,
    alice_reward: float,
    eve_reward: float,
    fallback_used: bool,
    *,
    active_route_attacked: bool,
    transcript: Any,
    info: InfoGainResult,
    reward_components: dict[str, Any],
    e91_periods_per_step: int,
    repeater_memory_pairs_per_step: int,
    runtime_metrics: dict[str, Any],
) -> dict[str, Any]:
    fidelity = runtime_metrics.get("fidelity")
    quality_metrics = {
        "qber": transcript.qber_estimate,
        "fidelity": fidelity,
        "amer_style_d_eff_mean": runtime_metrics.get("amer_style_d_eff_mean"),
        "delivery_success": bool(transcript.generation_successes > 0),
    }
    info_payload = info.to_dict()
    record = {
        "step": step_idx,
        "mode": mode,
        "scope_label": REPEATER_RUNTIME,
        "backend": E91_RUNTIME_BACKEND,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "alice_policy": alice_policy,
        "eve_policy": eve_policy,
        "alice_route_id": route["route_id"],
        "alice_path": list(route["path"]),
        "eve_action": {
            "id": action["id"],
            "kind": action["kind"],
            "target": action["target"],
            "attack_type": action["attack_type"],
        },
        "runtime_engine": runtime_metrics.get("runtime_engine"),
        "runtime_attack_applied": runtime_metrics.get("runtime_attack_applied"),
        "configured_repeater_attack": runtime_metrics.get("configured_repeater_attack"),
        "edge_attack_kind": runtime_metrics.get("edge_attack_kind"),
        "node_attack_kind": runtime_metrics.get("node_attack_kind"),
        "public_outcome": public_outcome,
        "collision": public_outcome in _DENIAL_OUTCOMES,
        "collision_or_delivery_failure": public_outcome in _DENIAL_OUTCOMES,
        # Infosec-aware denial reason for the decision trees: why did Eve deny
        # the key? (Bell-detected eavesdropping vs key-error vs availability.)
        "denial_reason": public_outcome if public_outcome in _DENIAL_OUTCOMES else None,
        "security_certificate_lost": public_outcome == "chsh_abort",
        "chsh_s": runtime_metrics.get("chsh_s"),
        "chsh_violates_bell": runtime_metrics.get("chsh_violates_bell"),
        "alice_reward": float(alice_reward),
        "eve_reward": float(eve_reward),
        "fallback_used": bool(fallback_used),
        "route_features": {
            "hops": route["hop_count"],
            "length": route["total_length"],
            "internal_nodes": list(route.get("internal_nodes") or []),
            "edges": list(route.get("edge_ids") or []),
            "scope_label": REPEATER_RUNTIME,
        },
        "quality_metrics": quality_metrics,
        "delivery_success": quality_metrics["delivery_success"],
        "delivered_count": transcript.generation_successes,
        "sifted_count": len(transcript.sifted_indices),
        "qber": transcript.qber_estimate,
        "fidelity": fidelity,
        "amer_style_d_eff_mean": runtime_metrics.get("amer_style_d_eff_mean"),
        "active_route_attacked": bool(active_route_attacked),
        "memory_intervention_public": runtime_metrics.get("memory_intervention_public"),
        "router_net_summary": runtime_metrics.get("router_net_summary"),
        "information_gain_reward_enabled": False,
        "information_gain_metric": info_payload,
        "eve_information_metric": info_payload,
        "e91_periods_per_step": int(e91_periods_per_step),
        "repeater_memory_pairs_per_step": int(repeater_memory_pairs_per_step),
        "metrics": {
            "scope_label": REPEATER_RUNTIME,
            "backend": E91_RUNTIME_BACKEND,
            "runtime_engine": runtime_metrics.get("runtime_engine"),
            **quality_metrics,
            "runtime_attack_applied": runtime_metrics.get("runtime_attack_applied"),
            "configured_repeater_attack": runtime_metrics.get("configured_repeater_attack"),
            "edge_attack_kind": runtime_metrics.get("edge_attack_kind"),
            "node_attack_kind": runtime_metrics.get("node_attack_kind"),
            "memory_intervention_public": runtime_metrics.get("memory_intervention_public"),
            "reward_components": reward_components,
            "active_route_attacked": bool(active_route_attacked),
            "information_gain_reward_enabled": False,
            "information_gain_metric": info_payload,
            "e91_periods_per_step": int(e91_periods_per_step),
            "repeater_memory_pairs_per_step": int(repeater_memory_pairs_per_step),
        },
    }
    assert_no_private_public_fields(record)
    return record


def _summary(
    *,
    mode: str,
    seed: int,
    alice: str,
    bob: str,
    alice_algo: str,
    eve_algo: str,
    routes: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    final_window: list[dict[str, Any]],
    final_window_size: int,
    eve_q: Any,
    alice_q: Any,
    include_alice_q: bool,
    started: float,
    step_seconds: list[float],
    e91_periods_per_step: int,
    repeater_memory_pairs_per_step: int,
) -> dict[str, Any]:
    route_ids = [step["alice_route_id"] for step in steps if step.get("alice_route_id")]
    outcomes = Counter(step.get("public_outcome") for step in steps)
    qbers = [
        float(step["qber"]) for step in steps
        if step.get("qber") is not None
    ]
    infos = [
        float((step.get("information_gain_metric") or {}).get("mutual_information_bits") or 0.0)
        for step in steps
    ]
    edge_targets = sorted({
        action["target"] for action in actions if action.get("kind") == "edge"
    })
    node_targets = sorted({
        action["target"] for action in actions if action.get("kind") == "node"
    })
    fidelities = [
        float(step["fidelity"]) for step in steps
        if step.get("fidelity") is not None
    ]
    runtime_engines = sorted({
        str(step.get("runtime_engine")) for step in steps
        if step.get("runtime_engine")
    })
    eve_q_export = _bounded_q_export(eve_q, _EVE_Q_EXPORT_MAX_ENTRIES)
    alice_q_export = (
        _bounded_q_export(alice_q, _ALICE_Q_EXPORT_MAX_ENTRIES)
        if include_alice_q else None
    )
    summary = {
        "run_id": f"{mode}-{seed}-{len(steps)}",
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "backend": E91_RUNTIME_BACKEND,
        "seed": seed,
        "mode": mode,
        "num_steps": len(steps),
        "alice": alice,
        "bob": bob,
        "routes": routes,
        "final_window": final_window,
        "q_tables": {
            "eve": {
                "top_actions": _q_top_actions(eve_q, actions, export=eve_q_export),
                "raw": eve_q_export,
                "action_names": [action["id"] for action in actions],
                "observation_state": ["previous_action", "previous_public_outcome"],
                "route_aware_oracle_label": (
                    "privileged/control" if eve_algo == "route_aware_oracle" else None
                ),
                "enabled_attack_kinds": sorted({
                    action["attack_type"] for action in actions
                    if action["attack_type"] != "no_attack"
                }),
                "attack_surface_scope": "sequence_edge_and_repeater_route_corridor",
                "control_model": mode,
                "reward_model": (
                    "dos_plus_empirical_mutual_information"
                    if _information_gain_enabled(mode)
                    else "dos_only"
                ),
                "information_gain_reward_enabled": False,
            },
            "alice": {
                "top_actions": _q_top_actions(
                    alice_q,
                    [{"id": route["route_id"], "kind": "route", "target": route["path"]}
                     for route in routes],
                    export=alice_q_export,
                ) if include_alice_q else [],
                "raw": alice_q_export,
                "action_names": [route["route_id"] for route in routes],
                "observation_state": [
                    "previous_route",
                    "previous_public_outcome",
                    "avoided_route_bitset",
                ],
            },
        },
        "metrics": {
            "scope_label": REPEATER_RUNTIME,
            "backend": E91_RUNTIME_BACKEND,
            "training_runtime_ms": round((time.perf_counter() - started) * 1000, 3),
            "training_iterations": len(steps),
            "training_execution": "online route/action updates; each step runs SeQUeNCe E91",
            "control_model": mode,
            "eve_algo": eve_algo,
            "alice_algo": alice_algo,
            "sequence_runtime_executed": True,
            "full_graph_runtime": True,
            "selected_path_runtime": True,
            "runtime_engines": runtime_engines,
            "runtime_case_count": len(steps),
            "e91_periods_per_step": int(e91_periods_per_step),
            "repeater_memory_pairs_per_step": int(repeater_memory_pairs_per_step),
            "reward_model": (
                "dos_plus_empirical_mutual_information"
                if _information_gain_enabled(mode)
                else "dos_only"
            ),
            "route_entropy": _route_entropy(route_ids),
            "route_switches": sum(1 for a, b in zip(route_ids, route_ids[1:]) if a != b),
            "detected_period": detect_route_period(route_ids),
            "enabled_attack_kinds": sorted({
                action["attack_type"] for action in actions
                if action["attack_type"] != "no_attack"
            }),
            "attack_surface_scope": "sequence_edge_and_repeater_route_corridor",
            "attack_surface_node_count": len(node_targets),
            "attack_surface_edge_count": len(edge_targets),
            "attack_surface_node_targets": node_targets,
            "attack_surface_edge_targets": edge_targets,
            "information_gain_reward_enabled": False,
            "information_gain_metric_source": (
                "sequence_game.eve.information.compute_information_gain"
                if _information_gain_enabled(mode)
                else None
            ),
            "information_gain_mean_bits": float(np.mean(infos)) if infos else 0.0,
            "information_gain_max_bits": max(infos) if infos else 0.0,
            "information_exposures": outcomes.get("information_exposure", 0),
            "delivery_failures": outcomes.get("delivery_failure", 0),
            "aborts": outcomes.get("qber_abort", 0),
            # Infosec-aware denial breakdown (why did E91 fail to make a key?):
            # chsh_abort = Bell violation lost (eavesdropping detected),
            # qber_abort = key too noisy, delivery_failure = availability denial.
            "chsh_aborts": outcomes.get("chsh_abort", 0),
            "denial_reason_counts": {
                reason: outcomes.get(reason, 0)
                for reason in ("delivery_failure", "qber_abort", "chsh_abort")
                if outcomes.get(reason, 0)
            },
            "public_outcome_counts": dict(sorted(outcomes.items())),
            "qber": float(np.mean(qbers)) if qbers else None,
            "qber_summary": _numeric_summary(qbers),
            "qber_by_eve_win": _numeric_by_eve_win(steps, "qber"),
            "fidelity_by_eve_win": _numeric_by_eve_win(steps, "fidelity"),
            "information_gain_by_eve_win": _information_gain_by_eve_win(steps),
            "quality_by_public_outcome": _quality_groups(
                steps,
                lambda step: str(step.get("public_outcome") or "unknown"),
            ),
            "quality_by_attack_type": _quality_groups(
                steps,
                lambda step: str((step.get("eve_action") or {}).get("attack_type") or "unknown"),
            ),
            "eve_attack_selection_counts": dict(sorted(Counter(
                str((step.get("eve_action") or {}).get("attack_type") or "unknown")
                for step in steps
            ).items())),
            "fidelity": float(np.mean(fidelities)) if fidelities else None,
            "fidelity_summary": _numeric_summary(fidelities),
            "delivered_pair_count": sum(
                int(step.get("delivered_count") or 0) for step in steps
            ),
            "final_replay_window": final_window_size,
            "mean_step_wall_clock_seconds": (
                float(np.mean(step_seconds)) if step_seconds else None
            ),
            "no_convergence_claim": True,
        },
    }
    assert_no_private_public_fields(summary)
    return summary


def _is_eve_win_step(step: dict[str, Any]) -> bool:
    # Reporting-only helper (never used for game dynamics). Direct/compiled
    # transcript step records carry the authoritative ``eve_win`` flag and can
    # emit "chsh_abort"; sequence-backend records fall back to the outcome set.
    flag = step.get("eve_win")
    if isinstance(flag, bool):
        return flag
    outcome = str(step.get("public_outcome"))
    return outcome in _EVE_WIN_OUTCOMES or outcome == "chsh_abort"


def _numeric_by_eve_win(steps: list[dict[str, Any]], field: str) -> dict[str, Any]:
    return {
        "eve_win": _numeric_summary([
            float(step[field]) for step in steps
            if _is_eve_win_step(step) and step.get(field) is not None
        ]),
        "no_eve_win": _numeric_summary([
            float(step[field]) for step in steps
            if not _is_eve_win_step(step) and step.get(field) is not None
        ]),
    }


def _information_gain_bits(step: dict[str, Any]) -> float:
    return float(
        (step.get("information_gain_metric") or {}).get("mutual_information_bits")
        or 0.0
    )


def _information_gain_by_eve_win(steps: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "eve_win": _numeric_summary([
            _information_gain_bits(step) for step in steps
            if _is_eve_win_step(step)
        ]),
        "no_eve_win": _numeric_summary([
            _information_gain_bits(step) for step in steps
            if not _is_eve_win_step(step)
        ]),
    }


def _quality_groups(
    steps: list[dict[str, Any]],
    key_fn: Any,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for step in steps:
        grouped[str(key_fn(step))].append(step)
    return {
        key: _quality_group_summary(group_steps)
        for key, group_steps in sorted(grouped.items())
    }


def _quality_group_summary(group_steps: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(group_steps)
    if count <= 0:
        return {
            "count": 0,
            "qber_summary": _numeric_summary([]),
            "fidelity_summary": _numeric_summary([]),
            "information_gain_summary": _numeric_summary([]),
            "delivery_success_rate": None,
            "active_route_attacked_rate": None,
            "eve_win_rate": None,
            "public_outcome_counts": {},
        }
    return {
        "count": count,
        "qber_summary": _numeric_summary([
            float(step["qber"]) for step in group_steps
            if step.get("qber") is not None
        ]),
        "fidelity_summary": _numeric_summary([
            float(step["fidelity"]) for step in group_steps
            if step.get("fidelity") is not None
        ]),
        "information_gain_summary": _numeric_summary([
            _information_gain_bits(step) for step in group_steps
        ]),
        "delivery_success_rate": sum(
            1 for step in group_steps if step.get("delivery_success")
        ) / count,
        "active_route_attacked_rate": sum(
            1 for step in group_steps if step.get("active_route_attacked")
        ) / count,
        "eve_win_rate": sum(
            1 for step in group_steps if _is_eve_win_step(step)
        ) / count,
        "public_outcome_counts": dict(sorted(Counter(
            str(step.get("public_outcome") or "unknown")
            for step in group_steps
        ).items())),
    }


def _route_entropy(route_ids: list[str]) -> float:
    if not route_ids:
        return 0.0
    total = len(route_ids)
    counts = Counter(route_ids)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


# Eve budget preserves the historical dense export for every action space seen
# so far; the Alice budget is tighter because route-dense graphs give Alice
# ~30k actions and her raw export has no downstream consumers.
_EVE_Q_EXPORT_MAX_ENTRIES = 2_000_000
_ALICE_Q_EXPORT_MAX_ENTRIES = 200_000


def _bounded_q_export(q_table: Any, max_entries: int) -> dict[str, Any]:
    """Q-table export capped at max_entries states x actions Q values.

    Keeps the tail of the underlying export's entry order when truncating
    (most recently seen states for DeepQNetwork, sorted-last states for
    QTable) and records q_export metadata so downstream consumers can tell
    the export is partial.
    """
    exported = q_table.to_dict()
    entries = exported.get("entries") or []
    num_actions = int(exported.get("num_actions") or 0)
    if num_actions > 0 and len(entries) * num_actions > max_entries:
        keep = max(1, max_entries // num_actions)
        exported = dict(exported)
        exported["entries"] = entries[-keep:]
        meta = dict(exported.get("q_export") or {})
        meta.update({
            "mode": meta.get("mode", "bounded"),
            "truncated": True,
            "exported_state_count": min(keep, len(entries)),
            "total_state_count": len(entries),
        })
        exported["q_export"] = meta
    return exported


def _q_top_actions(
    table: Any,
    actions: list[dict[str, Any]],
    limit: int = 8,
    export: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    scores = [0.0] * len(actions)
    entries_source = export if export is not None else table.to_dict()
    for entry in entries_source.get("entries", []):
        for index, value in enumerate(entry.get("q", [])):
            if index < len(scores):
                scores[index] = max(scores[index], float(value))
    ranked = sorted(range(len(actions)), key=lambda idx: (-scores[idx], actions[idx]["id"]))[:limit]
    return [
        {
            "action": actions[index]["id"],
            "kind": actions[index].get("kind", ""),
            "target": actions[index].get("target", ""),
            "attack_type": actions[index].get("attack_type", actions[index].get("kind", "")),
            "q": round(scores[index], 4),
        }
        for index in ranked
    ]
