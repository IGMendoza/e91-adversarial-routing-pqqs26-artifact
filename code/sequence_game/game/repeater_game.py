"""Fixed-path repeater E91 strategic-game wrapper.

This is the first repeater-infrastructure game surface. Alice's route is fixed
and explicit; Eve chooses an interior repeater plus one supported corruption
vector. Rewards are availability/abort/quality only. Information gain is not
computed or rewarded in this pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..claims import CONTROL_GAME, REPEATER_RUNTIME
from ..eve.repeater_attacks import (
    SUPPORTED_REPEATER_GAME_KINDS,
    RepeaterAttackSpec,
)
from ..physical.registry import PhysicalModel
from ..protocol.repeater_trial import (
    RepeaterE91RunConfig,
    run_fixed_repeater_e91_trial,
)
from ..sequence_build.repeater_e91_builder import FixedRepeaterPath


class RepeaterGameError(ValueError):
    """Invalid fixed-path repeater-game configuration."""


@dataclass(frozen=True)
class RepeaterGameConfig:
    memory_model: PhysicalModel
    fiber_model: PhysicalModel
    run_config: RepeaterE91RunConfig
    path: FixedRepeaterPath = field(default_factory=FixedRepeaterPath)
    num_trials: int = 3
    base_seed: int = 0
    attack_kinds: tuple[str, ...] = SUPPORTED_REPEATER_GAME_KINDS
    memory_degradation_multiplier: float = 0.5
    attack_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.num_trials < 1:
            raise RepeaterGameError("num_trials must be >= 1")
        bad = [kind for kind in self.attack_kinds
               if kind not in SUPPORTED_REPEATER_GAME_KINDS]
        if bad:
            raise RepeaterGameError(
                f"unsupported repeater attack kinds {bad}; "
                f"allowed {SUPPORTED_REPEATER_GAME_KINDS}")
        if not 0 <= self.memory_degradation_multiplier <= 1:
            raise RepeaterGameError("memory_degradation_multiplier must be in [0, 1]")
        if self.attack_cost < 0:
            raise RepeaterGameError("attack_cost must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_model": self.memory_model.model_name,
            "fiber_model": self.fiber_model.model_name,
            "run_config": self.run_config.to_dict(),
            "path": list(self.path.nodes),
            "edge_lengths_m": list(self.path.edge_lengths_m),
            "num_trials": self.num_trials,
            "base_seed": self.base_seed,
            "attack_kinds": list(self.attack_kinds),
            "memory_degradation_multiplier": self.memory_degradation_multiplier,
            "attack_cost": self.attack_cost,
            "scope_label": REPEATER_RUNTIME,
            "reward_scope": "availability_abort_quality_only",
            "information_gain_reward_enabled": False,
        }


@dataclass(frozen=True)
class RepeaterStrategyOutcome:
    strategy_id: str
    kind: str
    target_node: str
    trials: int
    accept_rate: float
    prevents_key: bool
    mean_qber: Optional[float]
    mean_sifted: float
    mean_fidelity: Optional[float]
    mean_amer_d_eff: Optional[float]
    mean_reward: float
    mean_information_bits: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "kind": self.kind,
            "target_node": self.target_node,
            "trials": self.trials,
            "accept_rate": self.accept_rate,
            "prevents_key": self.prevents_key,
            "mean_qber": self.mean_qber,
            "mean_sifted": self.mean_sifted,
            "mean_fidelity": self.mean_fidelity,
            "mean_amer_d_eff": self.mean_amer_d_eff,
            "mean_reward": self.mean_reward,
            "mean_information_bits": self.mean_information_bits,
            "scope_label": REPEATER_RUNTIME,
            "reward_scope": "availability_abort_quality_only",
        }


@dataclass
class RepeaterGameResult:
    path: tuple[str, ...]
    outcomes: list[RepeaterStrategyOutcome]

    def baseline(self) -> RepeaterStrategyOutcome:
        return next(o for o in self.outcomes if o.kind == "none")

    def denial_strategies(self) -> list[RepeaterStrategyOutcome]:
        return [o for o in self.outcomes if o.kind != "none" and o.prevents_key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_label": REPEATER_RUNTIME,
            "path": list(self.path),
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
        }


@dataclass(frozen=True)
class TwoRouteRepeaterControlConfig:
    memory_model: PhysicalModel
    fiber_model: PhysicalModel
    run_config: RepeaterE91RunConfig
    routes: tuple[FixedRepeaterPath, FixedRepeaterPath]

    def __post_init__(self) -> None:
        if len(self.routes) != 2:
            raise RepeaterGameError("two-route control requires exactly two fixed paths")


@dataclass(frozen=True)
class TwoRouteRepeaterStep:
    active_route_index: int
    attacked_route_index: Optional[int]
    attack_kind: str
    public_outcome: str
    delivery_success: bool
    active_route_attacked: bool
    qber: Optional[float]
    fidelity: Optional[float]
    scope_label: str = REPEATER_RUNTIME

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_route_index": self.active_route_index,
            "attacked_route_index": self.attacked_route_index,
            "attack_kind": self.attack_kind,
            "public_outcome": self.public_outcome,
            "delivery_success": self.delivery_success,
            "active_route_attacked": self.active_route_attacked,
            "qber": self.qber,
            "fidelity": self.fidelity,
            "scope_label": self.scope_label,
            "reward_scope": "availability_abort_quality_only",
        }


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _reward(transcript, attack_cost: float) -> float:
    attempts = max(1, transcript.generation_attempts)
    availability_loss = 1.0 - (transcript.generation_successes / attempts)
    abort_gain = 1.0 if transcript.accepted is False else 0.0
    fidelity = transcript.extra.get("mean_actual_fidelity")
    quality_loss = 0.0 if fidelity is None else 1.0 - float(fidelity)
    qber_loss = 0.0 if transcript.qber_estimate is None else float(transcript.qber_estimate)
    return availability_loss + abort_gain + quality_loss + qber_loss - attack_cost


def _specs(config: RepeaterGameConfig) -> list[RepeaterAttackSpec]:
    specs = [RepeaterAttackSpec(kind="none", cost=0.0)]
    for node in config.path.nodes[1:-1]:
        for kind in config.attack_kinds:
            if kind == "memory_degradation":
                specs.append(RepeaterAttackSpec(
                    kind=kind,
                    target_node=node,
                    memory_fidelity_multiplier=config.memory_degradation_multiplier,
                    cost=config.attack_cost,
                ))
            else:
                specs.append(RepeaterAttackSpec(
                    kind=kind,
                    target_node=node,
                    cost=config.attack_cost,
                ))
    return specs


def _evaluate_spec(config: RepeaterGameConfig,
                   spec: RepeaterAttackSpec) -> RepeaterStrategyOutcome:
    accepts = 0
    qbers: list[float] = []
    sifteds: list[float] = []
    fidelities: list[float] = []
    d_effs: list[float] = []
    rewards: list[float] = []

    for k in range(config.num_trials):
        transcript, _built = run_fixed_repeater_e91_trial(
            config.path,
            memory_model=config.memory_model,
            fiber_model=config.fiber_model,
            run_config=config.run_config,
            trial_id=f"{spec.kind}-{k}",
            seed=config.base_seed + k,
            attack=spec,
        )
        if transcript.accepted:
            accepts += 1
        if transcript.qber_estimate is not None:
            qbers.append(transcript.qber_estimate)
        sifteds.append(float(len(transcript.sifted_indices)))
        fidelity = transcript.extra.get("mean_actual_fidelity")
        if fidelity is not None:
            fidelities.append(float(fidelity))
        d_eff = transcript.extra.get("amer_style_d_eff_mean")
        if d_eff is not None:
            d_effs.append(float(d_eff))
        rewards.append(_reward(transcript, spec.cost if spec.is_active else 0.0))

    strategy_id = "no_attack" if spec.kind == "none" else f"{spec.kind}:{spec.target_node}"
    return RepeaterStrategyOutcome(
        strategy_id=strategy_id,
        kind=spec.kind,
        target_node=spec.target_node,
        trials=config.num_trials,
        accept_rate=accepts / config.num_trials,
        prevents_key=(accepts == 0),
        mean_qber=_mean(qbers),
        mean_sifted=float(_mean(sifteds) or 0.0),
        mean_fidelity=_mean(fidelities),
        mean_amer_d_eff=_mean(d_effs),
        mean_reward=float(_mean(rewards) or 0.0),
    )


def evaluate_fixed_repeater_game(config: RepeaterGameConfig) -> RepeaterGameResult:
    """Evaluate Eve's pure strategies on the explicit fixed repeater path."""

    outcomes = [_evaluate_spec(config, spec) for spec in _specs(config)]
    return RepeaterGameResult(path=config.path.nodes, outcomes=outcomes)


def run_two_route_repeater_step(config: TwoRouteRepeaterControlConfig,
                                *,
                                active_route_index: int,
                                attacked_route_index: Optional[int],
                                attack_kind: str,
                                seed: int) -> TwoRouteRepeaterStep:
    """Run one selected fixed route while Eve may attack active or inactive path."""

    if active_route_index not in (0, 1):
        raise RepeaterGameError("active_route_index must be 0 or 1")
    if attacked_route_index is not None and attacked_route_index not in (0, 1):
        raise RepeaterGameError("attacked_route_index must be 0 or 1 or None")
    if attack_kind not in SUPPORTED_REPEATER_GAME_KINDS and attack_kind != "no_attack":
        raise RepeaterGameError(f"unsupported repeater attack kind {attack_kind!r}")

    active_route_attacked = (
        attacked_route_index == active_route_index and attack_kind != "no_attack"
    )
    attack = RepeaterAttackSpec(kind="none")
    if active_route_attacked:
        path = config.routes[active_route_index]
        target_node = path.nodes[1] if len(path.nodes) > 2 else ""
        attack = RepeaterAttackSpec(kind=attack_kind, target_node=target_node)

    transcript, _built = run_fixed_repeater_e91_trial(
        config.routes[active_route_index],
        memory_model=config.memory_model,
        fiber_model=config.fiber_model,
        run_config=config.run_config,
        trial_id=f"two-route-{active_route_index}-{seed}",
        seed=seed,
        attack=attack,
    )
    if transcript.accepted:
        outcome = "accepted"
    elif transcript.abort_reason == "qber_above_threshold":
        outcome = "qber_abort"
    else:
        outcome = "delivery_failure"
    return TwoRouteRepeaterStep(
        active_route_index=active_route_index,
        attacked_route_index=attacked_route_index,
        attack_kind=attack_kind,
        public_outcome=outcome,
        delivery_success=bool(transcript.generation_successes > 0),
        active_route_attacked=active_route_attacked,
        qber=transcript.qber_estimate,
        fidelity=transcript.extra.get("mean_actual_fidelity"),
    )


def deterministic_two_route_reroute(active_route_index: int, public_outcome: str) -> int:
    """Cautious two-route control policy: switch after public failure."""

    if active_route_index not in (0, 1):
        raise RepeaterGameError("active_route_index must be 0 or 1")
    if public_outcome in {"delivery_failure", "qber_abort", "blocked"}:
        return 1 - active_route_index
    return active_route_index


def route_aware_repeater_oracle_action(active_route_index: int,
                                       attack_kind: str) -> dict[str, Any]:
    """Privileged control action that knows the active route."""

    if active_route_index not in (0, 1):
        raise RepeaterGameError("active_route_index must be 0 or 1")
    return {
        "scope_label": CONTROL_GAME,
        "privileged_control": True,
        "active_route_index": active_route_index,
        "attacked_route_index": active_route_index,
        "attack_kind": attack_kind,
    }


def repeater_q_eve_public_state(previous_action: str,
                                previous_public_outcome: str) -> tuple[str, str]:
    """Q Eve state for repeater control: public history only."""

    return (str(previous_action), str(previous_public_outcome))
