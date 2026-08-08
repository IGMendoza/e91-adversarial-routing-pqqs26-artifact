"""SeQUeNCe route/action evaluator for the Exp3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sequence_game.corpus.e91_runtime_game import (
    _default_models,
    _multiplexed_memory_efficiency,
)
from sequence_game.eve.repeater_attacks import RepeaterAttackSpec
from sequence_game.eve.feedback import (
    EVE_FEEDBACK_MODEL,
    eve_feedback_from_public_utility,
)
from sequence_game.protocol.postprocessing import SiftingConfig
from sequence_game.protocol.repeater_trial import (
    RepeaterE91RunConfig,
    run_fixed_repeater_chsh_trial,
    run_fixed_repeater_e91_trial,
)
from sequence_game.sequence_build.repeater_e91_builder import FixedRepeaterPath
from sequence_game.topology import TopologyIR

from .config import FIBER_CLASSICAL_SPEED_M_PER_PS, Exp3SequenceConfig
from .models import apply_sequence_memory_fidelity_override


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    kind: str
    target: str
    attack_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.action_id,
            "kind": self.kind,
            "target": self.target,
            "attack_type": self.attack_type,
        }


@dataclass(frozen=True)
class TurnResult:
    public_outcome: str
    alice_reward: float
    eve_hit_reward: float
    active_route_attacked: bool
    accepted: bool
    qber: float | None
    chsh_s: float | None
    chsh_adequately_sampled: bool | None
    delivered_count: int
    sifted_count: int
    fidelity: float | None
    runtime_engine: str
    runtime_attack_applied: dict[str, Any]
    sequence_timing: dict[str, Any]
    eve_feedback_reward: float | None = None
    eve_feedback_model: str = "legacy_target_overlap_diagnostic_only"
    eve_feedback_observation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_outcome": self.public_outcome,
            "alice_reward": self.alice_reward,
            # Compatibility name below is retained for existing SQLite cache
            # schemas.  It is never supplied to the learner.
            "eve_hit_reward": self.eve_hit_reward,
            "eve_target_overlap_diagnostic": self.eve_hit_reward,
            "active_route_attacked": self.active_route_attacked,
            "accepted": self.accepted,
            "qber": self.qber,
            "chsh_s": self.chsh_s,
            "chsh_adequately_sampled": self.chsh_adequately_sampled,
            "delivered_count": self.delivered_count,
            "sifted_count": self.sifted_count,
            "fidelity": self.fidelity,
            "runtime_engine": self.runtime_engine,
            "runtime_attack_applied": self.runtime_attack_applied,
            "sequence_timing": dict(self.sequence_timing),
            "eve_feedback_reward": self.eve_feedback_reward,
            "eve_feedback_model": self.eve_feedback_model,
            "eve_feedback_observation": dict(self.eve_feedback_observation),
        }


class SequenceRouteEvaluator:
    """Evaluate one route/action pair with SeQUeNCe repeater E91 trials."""

    def __init__(self, ir_dict: dict[str, Any], routes: list[dict[str, Any]],
                 config: Exp3SequenceConfig):
        self.ir = TopologyIR.from_dict(ir_dict)
        self.routes = routes
        self.config = config
        _source, _detector, self.fiber_model, self.memory_model = _default_models()
        self.memory_model = apply_sequence_memory_fidelity_override(
            self.memory_model, config)

    def evaluate(self, route_index: int, action: ActionSpec, *, seed: int,
                 trial_id: str) -> TurnResult:
        route = self.routes[route_index]
        target_active = _action_hits_route(action, route)
        attack = (
            _action_to_repeater_attack(action, route)
            if target_active else RepeaterAttackSpec()
        )
        path = FixedRepeaterPath(
            nodes=tuple(str(node) for node in route["path"]),
            edge_lengths_m=tuple(float(length) for length in route.get("edge_lengths_m", ()))
            or _fallback_lengths(route),
        )
        run_config, sequence_timing = self._run_config_for_route(route)
        runner = (
            run_fixed_repeater_chsh_trial
            if self.config.security_monitor == "chsh"
            else run_fixed_repeater_e91_trial
        )
        transcript, built = runner(
            path,
            memory_model=self.memory_model,
            fiber_model=self.fiber_model,
            run_config=run_config,
            trial_id=trial_id,
            seed=seed,
            attack=attack,
        )
        public_outcome = _public_outcome(transcript)
        alice_reward = 1.0 if transcript.accepted else 0.0
        local_kind, local_count = _eve_local_observation(built, attack)
        eve_feedback = eve_feedback_from_public_utility(
            public_outcome=public_outcome,
            alice_utility=alice_reward,
            local_observation_kind=local_kind,
            local_observation_count=local_count,
        )
        # Retained for offline diagnostics and old cache readers only.  The
        # online learner consumes eve_feedback_reward, never this overlap bit.
        eve_hit_reward = 1.0 if target_active and action.attack_type != "no_attack" else 0.0
        extra = getattr(transcript, "extra", {}) or {}
        return TurnResult(
            public_outcome=public_outcome,
            alice_reward=alice_reward,
            eve_hit_reward=eve_hit_reward,
            active_route_attacked=bool(target_active),
            accepted=bool(transcript.accepted),
            qber=transcript.qber_estimate,
            chsh_s=extra.get("chsh_s"),
            chsh_adequately_sampled=extra.get("chsh_adequately_sampled"),
            delivered_count=int(transcript.generation_successes),
            sifted_count=int(len(getattr(transcript, "sifted_indices", ()) or ())),
            fidelity=extra.get("mean_actual_fidelity"),
            runtime_engine=(
                "repeater_chsh_trial"
                if self.config.security_monitor == "chsh"
                else "repeater_e91_trial"
            ),
            runtime_attack_applied=attack.to_dict(),
            sequence_timing=sequence_timing,
            eve_feedback_reward=eve_feedback.reward,
            eve_feedback_model=EVE_FEEDBACK_MODEL,
            eve_feedback_observation=eve_feedback.to_dict(),
        )

    def _run_config_for_route(
            self,
            route: dict[str, Any],
    ) -> tuple[RepeaterE91RunConfig, dict[str, Any]]:
        total_length_m = float(route.get("total_length_m", route.get("total_length", 0.0)) or 0.0)
        light_speed = float(
            getattr(self.fiber_model, "parameters", {}).get(
                "light_speed",
                FIBER_CLASSICAL_SPEED_M_PER_PS,
            )
        )
        one_way_ps = total_length_m / light_speed if light_speed > 0 else 0.0
        start_time_ps = max(
            int(self.config.start_time_ps),
            int(self.config.sequence_setup_traversals * one_way_ps),
        )
        end_time_ps = start_time_ps + self.config.repeater_window_ps
        stop_time_ps = end_time_ps + self.config.stop_margin_ps
        timing = {
            "start_time_ps": start_time_ps,
            "end_time_ps": end_time_ps,
            "stop_time_ps": stop_time_ps,
            "route_total_length_m": total_length_m,
            "one_way_classical_ps": one_way_ps,
            "policy": (
                "start=max(config.start_time_ps, "
                "sequence_setup_traversals * one_way_classical_ps); "
                "end=start+repeater_window_ps; stop=end+stop_margin_ps"
            ),
            "sequence_setup_traversals": self.config.sequence_setup_traversals,
            "sequence_memory_fidelity_override": (
                self.config.sequence_memory_fidelity_override
            ),
            "stop_on_pair_target": True,
        }
        return RepeaterE91RunConfig(
            memory_pairs=self.config.memory_pairs_per_trial,
            sifting=SiftingConfig(
                qber_threshold=self.config.qber_threshold,
                min_sifted_samples=self.config.min_key_pairs,
            ),
            request_fidelity=self.config.request_fidelity,
            start_time_ps=start_time_ps,
            end_time_ps=end_time_ps,
            stop_time_ps=stop_time_ps,
            swapping_success_prob=self.config.swapping_success_prob,
            swapping_degradation=self.config.swapping_degradation,
            memory_efficiency_override=_multiplexed_memory_efficiency(self.memory_model),
            stop_on_pair_target=True,
        ), timing


def build_actions(routes: list[dict[str, Any]], action_kinds: tuple[str, ...]) -> list[ActionSpec]:
    nodes = sorted({
        str(node)
        for route in routes
        for node in (route.get("internal_nodes") or [])
    })
    edges = sorted({
        _edge_label(u, v)
        for route in routes
        for u, v in zip(route.get("path", []), route.get("path", [])[1:])
    })
    actions = [ActionSpec("no_attack", "none", "", "no_attack")]
    if "edge_intercept_resend" in action_kinds:
        for target in edges:
            actions.append(ActionSpec(
                f"edge_intercept_resend:{target}",
                "edge",
                target,
                "edge_intercept_resend",
            ))
    if "memory_degradation" in action_kinds:
        for target in nodes:
            actions.append(ActionSpec(
                f"memory_degradation:{target}",
                "node",
                target,
                "memory_degradation",
            ))
    return actions


def _action_hits_route(action: ActionSpec, route: dict[str, Any]) -> bool:
    if action.attack_type == "no_attack":
        return False
    if action.kind == "edge":
        return action.target in _route_edge_labels(route)
    return action.target in set(str(node) for node in (route.get("internal_nodes") or []))


def _action_to_repeater_attack(action: ActionSpec, route: dict[str, Any]) -> RepeaterAttackSpec:
    if action.attack_type == "edge_intercept_resend":
        u, v = _oriented_route_edge(route, action.target)
        return RepeaterAttackSpec(
            kind="edge_intercept_resend",
            target_edge=action.target,
            target_u=u,
            target_v=v,
            basis="Z",
            attack_id=action.action_id,
        )
    if action.attack_type == "memory_degradation":
        return RepeaterAttackSpec(
            kind="memory_degradation",
            target_node=action.target,
            memory_fidelity_multiplier=0.5,
            attack_id=action.action_id,
        )
    raise ValueError(f"unsupported action {action.action_id!r}")


def _edge_label(u: str, v: str) -> str:
    left, right = sorted((str(u), str(v)))
    return f"{left}-{right}"


def _route_edge_labels(route: dict[str, Any]) -> set[str]:
    path = [str(node) for node in route.get("path", [])]
    labels = {_edge_label(u, v) for u, v in zip(path, path[1:])}
    labels.update(str(edge_id) for edge_id in route.get("edge_ids", ()))
    return labels


def _oriented_route_edge(route: dict[str, Any], target: str) -> tuple[str, str]:
    path = [str(node) for node in route.get("path", [])]
    for u, v in zip(path, path[1:]):
        if target in {_edge_label(u, v), f"{u}-{v}", f"{v}-{u}"}:
            return u, v
    edge_ids = [str(edge_id) for edge_id in route.get("edge_ids", ())]
    for edge_id, u, v in zip(edge_ids, path, path[1:]):
        if target == edge_id:
            return u, v
    raise ValueError(f"edge target {target!r} is not on route {route.get('route_id')!r}")


def _public_outcome(transcript: Any) -> str:
    if transcript.accepted:
        return "accepted"
    if transcript.abort_reason in {"chsh_abort", "qber_abort", "qber_above_threshold"}:
        return "qber_abort" if transcript.abort_reason == "qber_above_threshold" else transcript.abort_reason
    return "delivery_failure"


def _eve_local_observation(built: Any, attack: RepeaterAttackSpec) -> tuple[str, int]:
    """Return records produced by Eve's physical hook, not route membership."""

    if attack.kind == "edge_intercept_resend":
        count = len(getattr(built, "edge_intercept_records", ()) or ())
        return "intercept_resend_measurement_record", count
    if attack.kind in {
        "repeater_memory_measurement",
        "repeater_memory_dephasing_probe",
    }:
        count = len(getattr(built, "memory_intervention_records", ()) or ())
        return "pre_swap_memory_intervention_record", count
    # memory_degradation is currently a configured device perturbation.  Its
    # successful installation is not evidence that Alice used that memory.
    return "none", 0


def _fallback_lengths(route: dict[str, Any]) -> tuple[float, ...]:
    hops = max(1, len(route.get("path") or []) - 1)
    total = float(route.get("total_length_m", route.get("total_length", 0.0)) or 0.0)
    return tuple(total / hops for _ in range(hops))
