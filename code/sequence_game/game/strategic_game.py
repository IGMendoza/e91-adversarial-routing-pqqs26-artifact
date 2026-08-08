"""Legacy strategic routing-vs-attack game on the bare-fiber E91/BBM92 control.

This module is retained for existing BBM92/bare-fiber experiments. New
repeater-network E91 payoff work should use ``game.repeater_game``.

The game (project step 5): Alice runs a *fixed deterministic* route to Bob; Eve
picks one fiber edge to attack (Alice/Bob nodes are off-limits, edges are fair
game). This module evaluates Eve's pure strategies against the fixed route and
returns the payoff table that answers the two research questions:

- **Can Eve deny the key?** -> ``prevents_key`` (the key never succeeds under the
  attack across all trials).
- **Can Eve learn the key?** -> ``mean_information_bits`` (I(K;E|P)) and
  ``mean_fraction_correct``, together with ``mean_qber``.

The instructive outcome it surfaces: intercept-resend *does* leak information
(~0.5 bits/sifted bit) but drives QBER to ~0.25, so the protocol aborts -- Eve
cannot learn the key without being detected. DoS (added loss) denies the key but
leaks nothing. This is QKD's security property shown empirically; it is a
simulator result, not a formal proof.

Scope: orchestration only (no ``sequence`` import); reuses the trial runner, the
Eve attacks, and the I(K;E|P) metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..eve.information import compute_information_gain
from ..eve.sequence_attacks import (
    SUPPORTED_GAME_KINDS,
    EdgeAttackSpec,
    build_attacked_hops,
    make_station_factory,
)
from ..physical.registry import PhysicalModel
from ..protocol.sequence_trial import E91RunConfig, run_e91_trial
from ..routing.policy import RoutingPolicy
from ..topology.ir import TopologyIR


class GameError(ValueError):
    """Invalid strategic-game configuration."""


@dataclass(frozen=True)
class GameConfig:
    alice: str
    bob: str
    source_model: PhysicalModel
    detector_model: PhysicalModel
    fiber_model: PhysicalModel
    run_config: E91RunConfig
    num_trials: int = 3
    base_seed: int = 0
    attack_kinds: tuple[str, ...] = ("intercept_resend", "added_loss")
    drop_probability: float = 1.0  # for added_loss strategies

    def __post_init__(self) -> None:
        if self.num_trials < 1:
            raise GameError("num_trials must be >= 1")
        bad = [k for k in self.attack_kinds if k not in SUPPORTED_GAME_KINDS]
        if bad:
            raise GameError(f"unsupported attack kinds {bad}; allowed {SUPPORTED_GAME_KINDS}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "alice": self.alice, "bob": self.bob,
            "source_model": self.source_model.model_name,
            "detector_model": self.detector_model.model_name,
            "fiber_model": self.fiber_model.model_name,
            "run_config": self.run_config.to_dict(),
            "num_trials": self.num_trials, "base_seed": self.base_seed,
            "attack_kinds": list(self.attack_kinds),
            "drop_probability": self.drop_probability,
        }


@dataclass(frozen=True)
class StrategyOutcome:
    strategy_id: str
    kind: str
    target_u: str
    target_v: str
    trials: int
    accept_rate: float
    prevents_key: bool            # key never accepted under this strategy
    mean_qber: Optional[float]
    mean_sifted: float
    mean_information_bits: float   # mean empirical I(K;E|P)
    mean_fraction_correct: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id, "kind": self.kind,
            "target_u": self.target_u, "target_v": self.target_v,
            "trials": self.trials, "accept_rate": self.accept_rate,
            "prevents_key": self.prevents_key, "mean_qber": self.mean_qber,
            "mean_sifted": self.mean_sifted,
            "mean_information_bits": self.mean_information_bits,
            "mean_fraction_correct": self.mean_fraction_correct,
        }


@dataclass
class GameResult:
    alice: str
    bob: str
    route: tuple[str, ...]
    outcomes: list[StrategyOutcome]

    def baseline(self) -> StrategyOutcome:
        return next(o for o in self.outcomes if o.kind == "none")

    def best_information_strategy(self) -> Optional[StrategyOutcome]:
        active = [o for o in self.outcomes if o.kind != "none"]
        return max(active, key=lambda o: o.mean_information_bits, default=None)

    def denial_strategies(self) -> list[StrategyOutcome]:
        return [o for o in self.outcomes if o.kind != "none" and o.prevents_key]

    def to_dict(self) -> dict[str, Any]:
        return {
            "alice": self.alice, "bob": self.bob, "route": list(self.route),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _evaluate_strategy(ir: TopologyIR, node_path: list[str], config: GameConfig,
                       spec: EdgeAttackSpec) -> StrategyOutcome:
    hops, eve_nodes = build_attacked_hops(ir, node_path, config.fiber_model, spec)
    factory = make_station_factory(spec)
    accepts = 0
    qbers: list[float] = []
    sifteds: list[float] = []
    infos: list[float] = []
    fracs: list[float] = []
    for k in range(config.num_trials):
        transcript, built = run_e91_trial(
            hops, alice=config.alice, bob=config.bob,
            source_model=config.source_model, detector_model=config.detector_model,
            run_config=config.run_config, trial_id=f"{spec.kind}-{k}",
            seed=config.base_seed + k, station_factory=factory, eve_nodes=eve_nodes)
        if transcript.accepted:
            accepts += 1
        if transcript.qber_estimate is not None:
            qbers.append(transcript.qber_estimate)
        sifteds.append(len(transcript.sifted_indices))
        ig = compute_information_gain(transcript, built)
        infos.append(ig.mutual_information_bits)
        fracs.append(ig.fraction_correct)

    if spec.kind == "none":
        sid = "no_attack"
    else:
        sid = f"{spec.kind}:{spec.target_u}-{spec.target_v}"
    return StrategyOutcome(
        strategy_id=sid, kind=spec.kind, target_u=spec.target_u, target_v=spec.target_v,
        trials=config.num_trials, accept_rate=accepts / config.num_trials,
        prevents_key=(accepts == 0), mean_qber=(_mean(qbers) if qbers else None),
        mean_sifted=_mean(sifteds), mean_information_bits=_mean(infos),
        mean_fraction_correct=_mean(fracs))


def evaluate_strategic_game(ir: TopologyIR, routing_policy: RoutingPolicy,
                            config: GameConfig) -> GameResult:
    """Run the fixed-Alice / Eve-picks-an-edge game and return the payoff table.

    Eve's strategy set: ``no_attack`` plus, for every edge on Alice's route, each
    configured attack kind. (Off-route edges are wasted attacks and are omitted.)
    """
    ir.validate()
    for nid in (config.alice, config.bob):
        if nid not in ir.nodes:
            raise GameError(f"unknown endpoint {nid!r}")
    route = routing_policy.select_route(ir, config.alice, config.bob)
    node_path = list(route.path)

    specs = [EdgeAttackSpec(kind="none")]
    for u, v in zip(node_path, node_path[1:]):
        for kind in config.attack_kinds:
            specs.append(EdgeAttackSpec(kind=kind, target_u=u, target_v=v,
                                        drop_probability=config.drop_probability))
    outcomes = [_evaluate_strategy(ir, node_path, config, s) for s in specs]
    return GameResult(alice=config.alice, bob=config.bob, route=tuple(node_path),
                      outcomes=outcomes)
