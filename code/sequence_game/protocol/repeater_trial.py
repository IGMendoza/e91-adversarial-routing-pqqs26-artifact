"""Repeater-backed E91 raw-key/QBER trial runner.

This module is the new governing protocol path for the repeater work. It drives
SeQUeNCe's RouterNetTopo repeater stack to deliver Alice--Bob entangled memories
over an explicit fixed route, then measures endpoint memories in Z/X through
``E91EndpointApp``.

The legacy bare-fiber BBM92 runner remains in ``sequence_trial.py`` and is not
deleted in this pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..sequence_build.e91_endpoint_app import (
    E91CHSHEndpointApp,
    E91EndpointApp,
    E91MemoryMeasurement,
)

from ..claims import REPEATER_RUNTIME
from ..eve.repeater_attacks import RepeaterAttackSpec
from ..physical.registry import PhysicalModel
from ..routing.route import route_id_for_path
from ..sequence_build.pre_swap_swapping import use_pre_swap_hook_swapping
from ..sequence_build.repeater_e91_builder import (
    BuiltRepeaterE91Line,
    FixedRepeaterPath,
    build_fixed_repeater_e91_line,
)
from .chsh_core import (
    ALICE_ANGLES,
    BOB_ANGLES,
    DEFAULT_MIN_CHSH_CELL_COUNT,
    compute_chsh_statistics,
    e91_outcome,
)
from .postprocessing import PostprocessingError, SiftingConfig, apply_postprocessing
from .state_witness import (
    EndpointBellStateWitnessCollector,
    EndpointPairMeasurementCoordinator,
)
from .transcript import TrialTranscript


@dataclass(frozen=True)
class RepeaterE91RunConfig:
    """Configuration for one fixed-path repeater E91 trial."""

    memory_pairs: int
    sifting: SiftingConfig
    request_fidelity: float = 0.01
    start_time_ps: int = 1_000_000_000
    end_time_ps: int = 5_000_000_000
    stop_time_ps: int = 6_000_000_000
    swapping_success_prob: float = 0.5
    swapping_degradation: float = 1.0
    scope: str = REPEATER_RUNTIME
    declare_werner_diagnostic: bool = True
    # SeQUeNCe repeater path only: substitute this effective per-attempt memory
    # efficiency (e.g. the Cui2025 multiplexed effective branching ratio) for
    # the memory model's single-mode value. None keeps the single-mode value.
    memory_efficiency_override: float | None = None
    # Stop the timeline as soon as both endpoints have measured memory_pairs
    # deliveries. The apps cap measurements at the reservation size, so the
    # transcript is identical to a full-window run; only post-completion event
    # churn (and its diagnostic attack-record counts) is skipped.
    stop_on_pair_target: bool = False

    def __post_init__(self) -> None:
        if self.memory_pairs < 1:
            raise PostprocessingError("memory_pairs must be >= 1")
        if len(self.sifting.basis_labels) != 2 or set(self.sifting.basis_labels) != {"Z", "X"}:
            raise PostprocessingError("RepeaterE91RunConfig currently supports Z/X bases")
        if not 0 < self.request_fidelity <= 1:
            raise PostprocessingError("request_fidelity must satisfy 0 < f <= 1")
        if not 0 <= self.start_time_ps < self.end_time_ps <= self.stop_time_ps:
            raise PostprocessingError(
                "need 0 <= start_time_ps < end_time_ps <= stop_time_ps")
        if not 0 <= self.swapping_success_prob <= 1:
            raise PostprocessingError("swapping_success_prob must be in [0, 1]")
        if not 0 <= self.swapping_degradation <= 1:
            raise PostprocessingError("swapping_degradation must be in [0, 1]")
        if (self.memory_efficiency_override is not None
                and not 0 <= self.memory_efficiency_override <= 1):
            raise PostprocessingError("memory_efficiency_override must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_pairs": self.memory_pairs,
            "sifting": self.sifting.to_dict(),
            "request_fidelity": self.request_fidelity,
            "start_time_ps": self.start_time_ps,
            "end_time_ps": self.end_time_ps,
            "stop_time_ps": self.stop_time_ps,
            "swapping_success_prob": self.swapping_success_prob,
            "swapping_degradation": self.swapping_degradation,
            "scope": self.scope,
            "declare_werner_diagnostic": self.declare_werner_diagnostic,
            "memory_efficiency_override": self.memory_efficiency_override,
            "stop_on_pair_target": self.stop_on_pair_target,
        }


def _install_pair_target_stop(built: BuiltRepeaterE91Line, alice_app, bob_app,
                              run_config: "RepeaterE91RunConfig") -> None:
    if not run_config.stop_on_pair_target:
        return
    target = run_config.memory_pairs
    timeline = built.timeline

    def stop_when_both_reach_target(_app) -> None:
        if (len(alice_app.measurements) >= target
                and len(bob_app.measurements) >= target):
            timeline.stop()

    alice_app.on_measurement = stop_when_both_reach_target
    bob_app.on_measurement = stop_when_both_reach_target


def _reverse_key(record: E91MemoryMeasurement) -> tuple[str, str, str, str]:
    return (record.remote_node, record.remote_memo, record.node, record.local_memo)


def _matched_endpoint_pairs(
        alice_measurements: list[E91MemoryMeasurement],
        bob_measurements: list[E91MemoryMeasurement],
) -> list[tuple[E91MemoryMeasurement, E91MemoryMeasurement]]:
    bob_by_key = {record.pair_key(): record for record in bob_measurements}
    pairs: list[tuple[E91MemoryMeasurement, E91MemoryMeasurement]] = []
    for alice_record in alice_measurements:
        bob_record = bob_by_key.get(_reverse_key(alice_record))
        if bob_record is not None:
            pairs.append((alice_record, bob_record))
    return sorted(pairs, key=lambda p: max(p[0].time_ps, p[1].time_ps))


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _amer_d_eff(fidelity: Optional[float]) -> Optional[float]:
    return None if fidelity is None else 4 * (1 - fidelity) / 3


def _install_endpoint_state_witness(
        built: BuiltRepeaterE91Line,
        alice_app: E91EndpointApp | E91CHSHEndpointApp,
        bob_app: E91EndpointApp | E91CHSHEndpointApp,
        *,
        alice: str,
        bob: str,
) -> EndpointBellStateWitnessCollector:
    key_by_memory_name: dict[str, int] = {}
    for node_name in (alice, bob):
        router = built.routers[node_name]
        memory_array = router.components[router.memo_arr_name]
        key_by_memory_name.update({
            memory.name: int(memory.qstate_key) for memory in memory_array
        })
    collector = EndpointBellStateWitnessCollector(
        quantum_manager=built.timeline.quantum_manager,
        alice_node=alice,
        bob_node=bob,
        qstate_key_by_memory_name=key_by_memory_name,
    )
    alice_app.on_pre_measurement = collector.observe
    bob_app.on_pre_measurement = collector.observe
    coordinator = EndpointPairMeasurementCoordinator()
    alice_app.on_qualified_memory = coordinator.coordinate
    bob_app.on_qualified_memory = coordinator.coordinate
    return collector


def run_fixed_repeater_e91_trial(
        path: FixedRepeaterPath,
        *,
        memory_model: PhysicalModel,
        fiber_model: PhysicalModel,
        run_config: RepeaterE91RunConfig,
        trial_id: str,
        seed: int,
        attack: RepeaterAttackSpec = RepeaterAttackSpec(),
) -> tuple[TrialTranscript, BuiltRepeaterE91Line]:
    """Run one fixed-path repeater E91 trial and return transcript + network."""

    built = build_fixed_repeater_e91_line(
        path,
        memory_model=memory_model,
        fiber_model=fiber_model,
        stop_time_ps=run_config.stop_time_ps,
        memory_size=run_config.memory_pairs * 2,
        seed=seed,
        swapping_success_prob=run_config.swapping_success_prob,
        swapping_degradation=run_config.swapping_degradation,
        attack=attack,
        memory_efficiency_override=run_config.memory_efficiency_override,
    )
    routers = built.routers
    alice, bob = path.nodes[0], path.nodes[-1]
    alice_app = E91EndpointApp(
        routers[alice],
        seed=seed + 10_001,
        basis_labels=run_config.sifting.basis_labels,
        release_after_measure=False,
    )
    bob_app = E91EndpointApp(
        routers[bob],
        seed=seed + 20_003,
        basis_labels=run_config.sifting.basis_labels,
        release_after_measure=False,
    )

    state_witness = _install_endpoint_state_witness(
        built, alice_app, bob_app, alice=alice, bob=bob)

    _install_pair_target_stop(built, alice_app, bob_app, run_config)
    alice_app.start(
        bob,
        run_config.start_time_ps,
        run_config.end_time_ps,
        run_config.memory_pairs,
        run_config.request_fidelity,
    )
    built.timeline.init()
    if built.pre_swap_hook_installed:
        # Pre-swap memory attacks need the first-party swapping subclass that
        # fires the installed hook; scope it to this run so other trials keep
        # the stock circuit swap.
        with use_pre_swap_hook_swapping():
            built.timeline.run()
    else:
        built.timeline.run()

    pairs = _matched_endpoint_pairs(alice_app.measurements, bob_app.measurements)
    fidelities = [min(a.fidelity, b.fidelity) for a, b in pairs]
    mean_fidelity = _mean(fidelities)

    transcript = TrialTranscript(
        trial_id=trial_id,
        route_id=route_id_for_path(path.nodes),
        route_path=path.nodes,
        scope=run_config.scope,
    )
    transcript.generation_attempts = run_config.memory_pairs
    transcript.generation_successes = len(pairs)
    if len(path.nodes) > 2:
        transcript.swap_attempts = run_config.memory_pairs
        transcript.swap_successes = len(pairs)
    transcript.alice_bases = tuple(a.basis for a, _ in pairs)
    transcript.bob_bases = tuple(b.basis for _, b in pairs)
    transcript.alice_outcomes = tuple(a.outcome for a, _ in pairs)
    transcript.bob_outcomes = tuple(b.outcome for _, b in pairs)
    if pairs:
        transcript.latency_ps = max(a.time_ps for a, _ in pairs) - run_config.start_time_ps
    transcript.extra.update({
        "fixed_path": list(path.nodes),
        "router_net_summary": built.summary(),
        "request_fidelity_delivery_threshold": run_config.request_fidelity,
        "request_fidelity_is_protocol_acceptance": False,
        "actual_fidelities": tuple(fidelities),
        "mean_actual_fidelity": mean_fidelity,
        "amer_style_d_eff_mean": (
            _amer_d_eff(mean_fidelity) if run_config.declare_werner_diagnostic else None
        ),
        "amer_style_d_eff_assumption": (
            "Bell/Werner diagnostic only; SeQUeNCe fidelity is not forced from D"
            if run_config.declare_werner_diagnostic else "not_declared"
        ),
        "endpoint_stage": (
            "Z/X raw-key plus finite-shot QBER decision; exact pre-measurement "
            "Ekert state witness stored as a private validation diagnostic"
        ),
        "endpoint_state_witness": state_witness.summary(
            qber_threshold=run_config.sifting.qber_threshold),
        "amer_final_key_formula": "not_applied; Amer formula is asymptotic",
        "information_gain_reward_enabled": False,
        "alice_endpoint_measurements": tuple(m.to_dict() for m in alice_app.measurements),
        "bob_endpoint_measurements": tuple(m.to_dict() for m in bob_app.measurements),
    })
    apply_postprocessing(transcript, run_config.sifting)
    return transcript, built


def _matched_chsh_pairs(alice_measurements, bob_measurements):
    """Pair Alice/Bob CHSH measurements on the same delivered memory pair."""
    bob_by_key = {m.pair_key(): m for m in bob_measurements}
    pairs = []
    for a in alice_measurements:
        b = bob_by_key.get(_reverse_key(a))
        if b is not None:
            pairs.append((a, b))
    return sorted(pairs, key=lambda p: max(p[0].time_ps, p[1].time_ps))


def run_fixed_repeater_chsh_trial(
        path: FixedRepeaterPath,
        *,
        memory_model: PhysicalModel,
        fiber_model: PhysicalModel,
        run_config: RepeaterE91RunConfig,
        trial_id: str,
        seed: int,
        attack: RepeaterAttackSpec = RepeaterAttackSpec(),
) -> tuple[TrialTranscript, BuiltRepeaterE91Line]:
    """Repeater E91 trial with a physical CHSH-S Bell monitor (Ekert-91).

    Alice/Bob measure delivered memories at the Ekert angle sets; the CHSH-S
    statistic and the device-independent decision (|S|<=2 -> chsh_abort, else
    qber>threshold -> qber_abort, else accepted) come from
    :mod:`sequence_game.protocol.chsh_core`. Scope: toy/simulator Bell-violation
    monitor, not a security proof.
    """

    built = build_fixed_repeater_e91_line(
        path,
        memory_model=memory_model,
        fiber_model=fiber_model,
        stop_time_ps=run_config.stop_time_ps,
        memory_size=run_config.memory_pairs * 2,
        seed=seed,
        swapping_success_prob=run_config.swapping_success_prob,
        swapping_degradation=run_config.swapping_degradation,
        attack=attack,
        memory_efficiency_override=run_config.memory_efficiency_override,
    )
    routers = built.routers
    alice, bob = path.nodes[0], path.nodes[-1]
    alice_app = E91CHSHEndpointApp(
        routers[alice], seed=seed + 10_001,
        polarization_angles=ALICE_ANGLES, release_after_measure=False)
    bob_app = E91CHSHEndpointApp(
        routers[bob], seed=seed + 20_003,
        polarization_angles=BOB_ANGLES, release_after_measure=False)

    state_witness = _install_endpoint_state_witness(
        built, alice_app, bob_app, alice=alice, bob=bob)

    _install_pair_target_stop(built, alice_app, bob_app, run_config)
    alice_app.start(
        bob, run_config.start_time_ps, run_config.end_time_ps,
        run_config.memory_pairs, run_config.request_fidelity)
    built.timeline.init()
    if built.pre_swap_hook_installed:
        with use_pre_swap_hook_swapping():
            built.timeline.run()
    else:
        built.timeline.run()

    pairs = _matched_chsh_pairs(alice_app.measurements, bob_app.measurements)
    stats = compute_chsh_statistics(
        a_idx=[a.setting_index for a, _ in pairs],
        b_idx=[b.setting_index for _, b in pairs],
        a_out=[a.outcome for a, _ in pairs],
        b_out=[b.outcome for _, b in pairs],
    )
    outcome = e91_outcome(
        stats,
        qber_threshold=run_config.sifting.qber_threshold,
        min_key_pairs=run_config.sifting.min_sifted_samples,
    )
    accepted = outcome == "accepted"
    fidelities = [min(a.fidelity, b.fidelity) for a, b in pairs]

    transcript = TrialTranscript(
        trial_id=trial_id,
        route_id=route_id_for_path(path.nodes),
        route_path=path.nodes,
        scope=run_config.scope,
    )
    transcript.generation_attempts = run_config.memory_pairs
    transcript.generation_successes = len(pairs)
    if len(path.nodes) > 2:
        transcript.swap_attempts = run_config.memory_pairs
        transcript.swap_successes = len(pairs)
    transcript.qber_estimate = stats.key_qber
    transcript.accepted = accepted
    transcript.abort_reason = None if accepted else outcome
    if pairs:
        transcript.latency_ps = max(a.time_ps for a, _ in pairs) - run_config.start_time_ps
    transcript.extra.update({
        "fixed_path": list(path.nodes),
        "router_net_summary": built.summary(),
        "endpoint_stage": "Ekert-91 3-angle CHSH-S Bell monitor (hard-Bell |S|<=2 abort)",
        "chsh_s": stats.chsh_s,
        "chsh_violates_bell": stats.violates_bell,
        "chsh_correlations": stats.correlations,
        "chsh_key_length": stats.key_length,
        "chsh_coincidences": stats.coincidences,
        "chsh_min_cell_count": stats.min_chsh_cell_count,
        # A trustworthy S needs enough samples per setting; below the minimum the
        # block is inconclusive (see chsh_core.DEFAULT_MIN_CHSH_CELL_COUNT) and
        # the decision falls through to delivery_failure rather than trusting S.
        "chsh_adequately_sampled": (
            stats.min_chsh_cell_count >= DEFAULT_MIN_CHSH_CELL_COUNT),
        "endpoint_state_witness": state_witness.summary(
            qber_threshold=run_config.sifting.qber_threshold),
        "public_outcome_decision": outcome,
        "mean_actual_fidelity": (sum(fidelities) / len(fidelities) if fidelities else None),
        "information_gain_reward_enabled": False,
        "alice_endpoint_measurements": tuple(m.to_dict() for m in alice_app.measurements),
        "bob_endpoint_measurements": tuple(m.to_dict() for m in bob_app.measurements),
    })
    return transcript, built
