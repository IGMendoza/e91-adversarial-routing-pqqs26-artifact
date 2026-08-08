"""Finite-horizon physical Prime-and-Probe interventions against Alice.

The leader acts only through declared histogram-backend Eve instruments.
Planning uses the normalized topology and designated feedback draws.  Standard
runs reserve the audit split for scoring.  Detector-enabled evaluation may
instead freeze a previously selected plan, commission Alice's detector from
feedback draws, and execute both players on the disjoint audit draws; in that
case those draws are the live held-out execution stream and are not reused for
post-hoc scoring.
"""

from __future__ import annotations

from collections import Counter, deque
import copy
from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from .calibration_bank import HistogramPayoffBank, validate_bank_against_case
from .calibration_game import CommonCellOutcomeStream
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .endpoint_evidence_bank import (
    HistogramEndpointEvidenceBank,
    validate_endpoint_evidence_bank_against_case,
)
from .learner_protocol import PlayerLearner, policy_distribution_statistics
from .learning import build_histogram_alice_learner
from .matched_payoff_endpoint_bank import (
    MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA,
)
from .quantum_stackelberg_detection import (
    BinomialEvidenceDetectorConfig,
    MatchedClassicalQuantumEndpointDetectors,
)


PRIME_PROBE_RESULT_SCHEMA = "histogram_prime_probe_run.v1"
PRIME_PROBE_PLAN_SCHEMA = "histogram_prime_probe_plan.v1"
PRIME_PROBE_WITHDRAWAL_STATE_SCHEMA = (
    "histogram_prime_probe_withdrawal_state_study.v2"
)

WITHDRAWAL_NATIVE_CONTINUATION = "native_continuation"
WITHDRAWAL_FULL_RESTART = "full_restart"
WITHDRAWAL_WARMUP_RESTORE = "warmup_checkpoint_restore"
WITHDRAWAL_STATE_STRATEGIES = (
    WITHDRAWAL_NATIVE_CONTINUATION,
    WITHDRAWAL_FULL_RESTART,
    WITHDRAWAL_WARMUP_RESTORE,
)


@dataclass(frozen=True)
class PrimeProbeSchedule:
    """Predeclared phase lengths and observation decision rule."""

    warmup_turns: int = 150_000
    prime_turns: int = 150_000
    probe_turns: int = 5_000
    exploit_turns: int = 50_000
    recovery_turns: int = 145_000
    record_every: int = 1_000
    rolling_window_turns: int = 5_000
    baseline_tail_turns: int = 50_000
    sustained_recovery_checkpoints: int = 3
    probe_confidence_z: float = 1.959963984540054

    def __post_init__(self) -> None:
        phase_lengths = (
            self.warmup_turns,
            self.prime_turns,
            self.probe_turns,
            self.exploit_turns,
            self.recovery_turns,
        )
        if min(phase_lengths) < 1:
            raise ValueError("all Prime-and-Probe phase lengths must be positive")
        if self.record_every < 1 or self.rolling_window_turns < 1:
            raise ValueError("record and rolling-window lengths must be positive")
        if not 1 <= self.baseline_tail_turns <= self.warmup_turns:
            raise ValueError("baseline_tail_turns must lie inside warmup")
        if self.sustained_recovery_checkpoints < 1:
            raise ValueError("sustained recovery checkpoints must be positive")
        if not math.isfinite(self.probe_confidence_z) or self.probe_confidence_z <= 0:
            raise ValueError("probe_confidence_z must be finite and positive")

    @property
    def total_turns(self) -> int:
        return (
            self.warmup_turns
            + self.prime_turns
            + self.probe_turns
            + self.exploit_turns
            + self.recovery_turns
        )

    @property
    def prime_start(self) -> int:
        return self.warmup_turns

    @property
    def probe_start(self) -> int:
        return self.prime_start + self.prime_turns

    @property
    def exploit_start(self) -> int:
        return self.probe_start + self.probe_turns

    @property
    def recovery_start(self) -> int:
        return self.exploit_start + self.exploit_turns

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "total_turns": self.total_turns}


@dataclass(frozen=True)
class PrimeProbePlan:
    """One topology-safe virtual-bottleneck intervention plan."""

    exploit_action_index: int
    exploit_action_id: str
    probe_action_index: int
    probe_action_id: str
    target_route_indices: tuple[int, ...]
    target_route_ids: tuple[str, ...]
    alternative_route_indices: tuple[int, ...]
    prime_action_indices: tuple[int, ...]
    prime_action_ids: tuple[str, ...]
    prime_probabilities: tuple[float, ...]
    topological_coverage_floor: float
    physical_prime_loss_floor: float
    physical_exploit_loss_mean: float
    probe_target_count_minimum: float
    probe_target_count_maximum: float
    probe_nontarget_count_minimum: float
    probe_nontarget_count_maximum: float
    selection_score: float

    def __post_init__(self) -> None:
        if not self.target_route_indices or not self.alternative_route_indices:
            raise ValueError("plan requires target and alternative routes")
        if not self.prime_action_indices:
            raise ValueError("plan requires at least one prime action")
        probabilities = np.asarray(self.prime_probabilities, dtype=float)
        if probabilities.shape != (len(self.prime_action_indices),):
            raise ValueError("prime probabilities do not match prime actions")
        if np.any(probabilities < 0.0) or not math.isclose(
            float(probabilities.sum()), 1.0, rel_tol=1e-10, abs_tol=1e-10
        ):
            raise ValueError("prime probabilities must form a simplex point")
        if self.probe_target_count_minimum <= self.probe_nontarget_count_maximum:
            raise ValueError("probe count ranges do not identify target-family mass")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": PRIME_PROBE_PLAN_SCHEMA,
            **asdict(self),
            "target_route_fraction_under_uniform_policy": (
                len(self.target_route_indices)
                / (len(self.target_route_indices) + len(self.alternative_route_indices))
            ),
            "planner_information": {
                "topology": "normalized route/resource incidence",
                "physics": "feedback split cell means only",
                "audit_split_loaded": False,
                "alice_policy_or_weights_used": False,
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrimeProbePlan":
        """Restore a frozen plan while ignoring derived metadata fields."""

        values = {
            field: payload[field]
            for field in cls.__dataclass_fields__
        }
        for field in (
            "target_route_indices",
            "target_route_ids",
            "alternative_route_indices",
            "prime_action_indices",
            "prime_action_ids",
            "prime_probabilities",
        ):
            values[field] = tuple(values[field])
        return cls(**values)


@dataclass
class _PrimeProbePrefixState:
    """In-memory branch point for withdrawal-time sensitivity analysis."""

    condition: str
    alice: PlayerLearner
    warmup_checkpoint: PlayerLearner
    outcome_stream: CommonCellOutcomeStream
    baseline_policy: np.ndarray
    baseline_service_lower: float
    baseline_service_upper: float
    probe_trigger: dict[str, Any] | None
    mean_alice_reward: float


def build_prime_probe_plan(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    *,
    selection_digest_blake2b_128: str | None = None,
) -> PrimeProbePlan:
    """Select the highest-scoring declared Prime-and-Probe plan."""

    return build_prime_probe_plans(
        case,
        bank,
        config,
        selection_digest_blake2b_128=selection_digest_blake2b_128,
    )[0]


def build_prime_probe_plans(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    *,
    selection_digest_blake2b_128: str | None = None,
) -> tuple[PrimeProbePlan, ...]:
    """Enumerate every feasible declared Prime-and-Probe plan.

    Each candidate exploit resource defines a target route family.  Prime
    actions are restricted to denial instruments that do not touch any target
    route.  A max--min linear program selects a distribution whose incidence
    coverage is as even as possible across every alternative route.  Feedback
    bank means then quantify prime loss, exploit loss, and probe visibility.
    """

    validate_bank_against_case(
        bank,
        case,
        config,
        selection_digest_blake2b_128=(
            str(selection_digest_blake2b_128)
            if selection_digest_blake2b_128 is not None
            else str(bank.metadata["selection_digest_blake2b_128"])
        ),
    )
    if bank.metadata.get("eve_observation_count_semantics") == (
        "binary_indicator_migrated_from_v1"
    ):
        raise ValueError(
            "Prime-and-Probe requires count-valued local instrument records; "
            "a migrated v1 binary observation bank is insufficient"
        )
    actions = case.actions(config.game_mode)
    active = route_action_incidence(case, actions)
    feedback_reward = bank.feedback.alice_payoff_matrix()
    feedback_counts = np.mean(bank.feedback.eve_observation_count, axis=2)
    no_attack_index = _unique_action_index(actions, "no_attack")
    denial_indices = tuple(
        index
        for index, action in enumerate(actions)
        if action["attack_type"] in {"edge_dos", "swap_denial"}
    )
    candidates: list[tuple[tuple[float, ...], PrimeProbePlan]] = []

    for exploit_index in denial_indices:
        probe_index = _matching_probe_index(actions, actions[exploit_index])
        if probe_index is None:
            continue
        target = tuple(map(int, np.flatnonzero(active[:, exploit_index])))
        alternatives = tuple(map(int, np.flatnonzero(~active[:, exploit_index])))
        if not target or not alternatives:
            continue
        safe_prime = tuple(
            index
            for index in denial_indices
            if index != exploit_index and not np.any(active[list(target), index])
        )
        if not safe_prime:
            continue
        coverage = active[np.ix_(alternatives, safe_prime)].astype(float)
        if np.any(np.sum(coverage, axis=1) == 0.0):
            continue
        probabilities, coverage_floor = _max_min_coverage_distribution(coverage)
        retained = probabilities > 1e-12
        prime_indices = tuple(
            index for index, keep in zip(safe_prime, retained) if keep
        )
        prime_probabilities = probabilities[retained]
        prime_probabilities /= prime_probabilities.sum()

        baseline = feedback_reward[list(alternatives), no_attack_index]
        prime_reward = (
            feedback_reward[np.ix_(alternatives, prime_indices)]
            @ prime_probabilities
        )
        prime_loss = baseline - prime_reward
        exploit_loss = (
            feedback_reward[list(target), no_attack_index]
            - feedback_reward[list(target), exploit_index]
        )
        target_counts = feedback_counts[list(target), probe_index]
        alternative_counts = feedback_counts[list(alternatives), probe_index]
        target_min = float(np.min(target_counts))
        target_max = float(np.max(target_counts))
        alternative_min = float(np.min(alternative_counts))
        alternative_max = float(np.max(alternative_counts))
        if target_min <= alternative_max:
            continue
        prime_floor = float(np.min(prime_loss))
        exploit_mean = float(np.mean(exploit_loss))
        visibility_contrast = (
            (target_min - alternative_max) / max(target_max, 1e-12)
        )
        score = prime_floor * exploit_mean * visibility_contrast
        if prime_floor <= 0.0 or exploit_mean <= 0.0 or score <= 0.0:
            continue
        plan = PrimeProbePlan(
            exploit_action_index=exploit_index,
            exploit_action_id=str(actions[exploit_index]["id"]),
            probe_action_index=probe_index,
            probe_action_id=str(actions[probe_index]["id"]),
            target_route_indices=target,
            target_route_ids=tuple(bank.route_ids[index] for index in target),
            alternative_route_indices=alternatives,
            prime_action_indices=prime_indices,
            prime_action_ids=tuple(
                str(actions[index]["id"]) for index in prime_indices
            ),
            prime_probabilities=tuple(map(float, prime_probabilities)),
            topological_coverage_floor=float(coverage_floor),
            physical_prime_loss_floor=prime_floor,
            physical_exploit_loss_mean=exploit_mean,
            probe_target_count_minimum=target_min,
            probe_target_count_maximum=target_max,
            probe_nontarget_count_minimum=alternative_min,
            probe_nontarget_count_maximum=alternative_max,
            selection_score=float(score),
        )
        candidates.append((
            (
                plan.selection_score,
                plan.physical_prime_loss_floor,
                plan.physical_exploit_loss_mean,
                plan.topological_coverage_floor,
                -float(len(plan.target_route_indices)),
            ),
            plan,
        ))

    if not candidates:
        raise ValueError(
            "no declared denial/probe resource admits a topology-safe "
            "Prime-and-Probe construction"
        )
    candidates.sort(key=lambda row: (row[0], row[1].exploit_action_id))
    return tuple(row[1] for row in reversed(candidates))


def prime_probe_target_family_representatives(
    plans: Sequence[PrimeProbePlan],
    *,
    maximum_families: int | None = None,
) -> tuple[PrimeProbePlan, ...]:
    """Retain the highest-ranked plan for each distinct target route family.

    This view is intended for broad plan screening. It does not delete the
    underlying physical action variants returned by
    :func:`build_prime_probe_plans`; variants within a selected target family
    can subsequently be compared with matched seeds.
    """

    plan_values = tuple(plans)
    if not plan_values:
        raise ValueError("at least one Prime-and-Probe plan is required")
    if maximum_families is not None and int(maximum_families) < 1:
        raise ValueError("maximum_families must be positive")
    retained: list[PrimeProbePlan] = []
    seen: set[tuple[int, ...]] = set()
    for plan in plan_values:
        family = tuple(plan.target_route_indices)
        if family in seen:
            continue
        retained.append(plan)
        seen.add(family)
        if (
            maximum_families is not None
            and len(retained) >= int(maximum_families)
        ):
            break
    return tuple(retained)


def route_action_incidence(
    case: HistogramGameCase,
    actions: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    """Return pre-play route/resource incidence for declared Eve actions."""

    result = np.zeros((len(case.routes), len(actions)), dtype=bool)
    for route_index, route in enumerate(case.routes):
        path = tuple(map(str, route["path"]))
        route_edges = {
            _edge_label(left, right) for left, right in zip(path, path[1:])
        }
        internal_nodes = set(path[1:-1])
        for action_index, action in enumerate(actions):
            family = str(action.get("attack_type") or "no_attack")
            target = str(action.get("target") or "")
            if family in {"edge_dos", "edge_information_probe"}:
                result[route_index, action_index] = target in route_edges
            elif family in {
                "swap_denial",
                "memory_degradation",
                "repeater_memory_measure_Z",
                "repeater_memory_measure_X",
            }:
                result[route_index, action_index] = target in internal_nodes
    return result


def run_ordered_prime_probe(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    plan: PrimeProbePlan,
    *,
    schedule: PrimeProbeSchedule = PrimeProbeSchedule(),
    seed: int,
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    include_audit_scoring: bool = True,
    endpoint_evidence_bank: HistogramEndpointEvidenceBank | None = None,
    execution_split: str = "feedback",
    detector_commissioning_draws_per_route: int = 32,
    detector_config: BinomialEvidenceDetectorConfig | None = None,
) -> dict[str, Any]:
    """Run the ordered intervention and optional Alice endpoint detector."""

    return _run_prime_probe(
        case,
        bank,
        config,
        plan,
        schedule=schedule,
        seed=seed,
        condition="ordered_prime_probe",
        fixed_actions=None,
        selection_digest_blake2b_128=selection_digest_blake2b_128,
        progress_callback=progress_callback,
        include_audit_scoring=include_audit_scoring,
        endpoint_evidence_bank=endpoint_evidence_bank,
        execution_split=execution_split,
        detector_commissioning_draws_per_route=(
            detector_commissioning_draws_per_route
        ),
        detector_config=detector_config,
    )


def run_shuffled_prime_probe_control(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    plan: PrimeProbePlan,
    *,
    schedule: PrimeProbeSchedule,
    seed: int,
    ordered_action_indices: Sequence[int],
    selection_digest_blake2b_128: str | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    include_audit_scoring: bool = True,
    endpoint_evidence_bank: HistogramEndpointEvidenceBank | None = None,
    execution_split: str = "feedback",
) -> dict[str, Any]:
    """Run an identical-action-multiset temporal-order control."""

    actions = np.asarray(ordered_action_indices, dtype=np.int64)
    if actions.shape != (schedule.total_turns,):
        raise ValueError("ordered action schedule has the wrong length")
    rng = np.random.default_rng([seed, 97])
    shuffled = actions[rng.permutation(actions.size)]
    if Counter(map(int, shuffled)) != Counter(map(int, actions)):
        raise AssertionError("shuffled control changed the action multiset")
    return _run_prime_probe(
        case,
        bank,
        config,
        plan,
        schedule=schedule,
        seed=seed,
        condition="shuffled_identical_action_multiset",
        fixed_actions=shuffled,
        selection_digest_blake2b_128=selection_digest_blake2b_128,
        progress_callback=progress_callback,
        include_audit_scoring=include_audit_scoring,
        endpoint_evidence_bank=endpoint_evidence_bank,
        execution_split=execution_split,
        detector_commissioning_draws_per_route=0,
        detector_config=None,
    )


def _run_prime_probe(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    plan: PrimeProbePlan,
    *,
    schedule: PrimeProbeSchedule,
    seed: int,
    condition: str,
    fixed_actions: np.ndarray | None,
    selection_digest_blake2b_128: str | None,
    progress_callback: Callable[[int, str], None] | None,
    include_audit_scoring: bool,
    endpoint_evidence_bank: HistogramEndpointEvidenceBank | None,
    execution_split: str,
    detector_commissioning_draws_per_route: int,
    detector_config: BinomialEvidenceDetectorConfig | None,
) -> dict[str, Any]:
    if execution_split not in {"feedback", "audit"}:
        raise ValueError("execution_split must be 'feedback' or 'audit'")
    if endpoint_evidence_bank is None and execution_split != "feedback":
        raise ValueError(
            "audit execution requires a matched endpoint evidence bank"
        )
    if endpoint_evidence_bank is None and detector_config is not None:
        raise ValueError(
            "an Alice detector requires a matched endpoint evidence bank"
        )
    if detector_config is not None and execution_split != "audit":
        raise ValueError(
            "Alice detector execution must use the audit split disjoint from "
            "feedback-split commissioning"
        )
    if endpoint_evidence_bank is not None and fixed_actions is not None:
        if detector_config is not None:
            raise ValueError(
                "the shuffled control does not run the Alice detector"
            )
    validate_bank_against_case(
        bank,
        case,
        config,
        selection_digest_blake2b_128=(
            str(selection_digest_blake2b_128)
            if selection_digest_blake2b_128 is not None
            else str(bank.metadata["selection_digest_blake2b_128"])
        ),
    )
    selection_digest = (
        str(selection_digest_blake2b_128)
        if selection_digest_blake2b_128 is not None
        else str(bank.metadata["selection_digest_blake2b_128"])
    )
    payoff_block = getattr(bank, execution_split)
    endpoint_block = None
    if endpoint_evidence_bank is not None:
        validate_endpoint_evidence_bank_against_case(
            endpoint_evidence_bank,
            case,
            config,
            selection_digest_blake2b_128=selection_digest,
        )
        if (
            bank.route_ids != endpoint_evidence_bank.route_ids
            or bank.action_ids != endpoint_evidence_bank.action_ids
            or payoff_block.draws_per_cell
            != getattr(endpoint_evidence_bank, execution_split).draws_per_cell
        ):
            raise ValueError(
                "matched payoff and endpoint evidence bank axes differ"
            )
        endpoint_block = getattr(endpoint_evidence_bank, execution_split)
        if not np.array_equal(
            payoff_block.alice_reward,
            endpoint_block.alice_reward,
        ):
            raise ValueError(
                "matched payoff and endpoint rewards are not draw-aligned"
            )
        if not np.array_equal(
            payoff_block.public_outcome_code,
            endpoint_block.public_outcome_code,
        ):
            raise ValueError(
                "matched payoff and endpoint outcomes are not draw-aligned"
            )
        _validate_matched_bank_provenance(
            bank,
            endpoint_evidence_bank,
        )
    if include_audit_scoring and execution_split == "audit":
        raise ValueError(
            "audit execution cannot reuse the same split for post-hoc scoring"
        )
    active_config = replace(config, turns=schedule.total_turns, seed=int(seed))
    actions = case.actions(active_config.game_mode)
    _validate_frozen_prime_probe_plan(
        plan,
        bank=bank,
        route_count=len(case.routes),
    )
    prior = np.full(len(case.routes), 1.0 / len(case.routes), dtype=float)
    implementation, metadata = build_histogram_alice_learner(
        case=case,
        config=active_config,
        prior=prior,
    )
    alice = PlayerLearner(
        player="alice",
        action_ids=bank.route_ids,
        learner=implementation,
        metadata=metadata,
    )
    stream = CommonCellOutcomeStream(
        len(bank.route_ids),
        len(bank.action_ids),
        payoff_block.draws_per_cell,
        seed=int(seed),
    )
    alice_detectors: MatchedClassicalQuantumEndpointDetectors | None = None
    detector_alarm_snapshots: dict[str, dict[str, Any]] = {}
    detector_trace: list[dict[str, Any]] = []
    detector_monitoring_complete = False
    if detector_config is not None:
        if endpoint_evidence_bank is None:
            raise AssertionError("validated endpoint evidence bank disappeared")
        if detector_config.horizon != schedule.total_turns:
            raise ValueError(
                "detector horizon must equal the Prime-and-Probe schedule"
            )
        commissioning_draws = int(
            detector_commissioning_draws_per_route
        )
        if not 2 <= commissioning_draws <= (
            endpoint_evidence_bank.feedback.draws_per_cell
        ):
            raise ValueError(
                "detector commissioning draws are outside the feedback bank"
            )
        alice_detectors = (
            MatchedClassicalQuantumEndpointDetectors.from_commissioning(
                _endpoint_commissioning_evidence_by_route(
                    endpoint_evidence_bank,
                    stop=commissioning_draws,
                ),
                detector_config,
            )
        )
    target_set = set(plan.target_route_indices)
    no_attack_index = bank.action_ids.index("no_attack")
    prime_rng = np.random.default_rng([seed, 53])
    prime_indices = np.asarray(plan.prime_action_indices, dtype=np.int64)
    prime_probabilities = np.asarray(plan.prime_probabilities, dtype=float)
    ordered_actions = np.full(schedule.total_turns, no_attack_index, dtype=np.int64)
    if fixed_actions is None:
        ordered_actions[schedule.prime_start:schedule.probe_start] = (
            prime_rng.choice(
                prime_indices,
                size=schedule.prime_turns,
                p=prime_probabilities,
            )
        )
        ordered_actions[schedule.probe_start:schedule.exploit_start] = (
            plan.probe_action_index
        )
    else:
        ordered_actions[:] = np.asarray(fixed_actions, dtype=np.int64)

    joint_counts = np.zeros(
        (len(bank.route_ids), len(bank.action_ids)), dtype=np.int64
    )
    phase_sums: dict[str, dict[str, float]] = {}
    phase_counts: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    recent_rewards: deque[float] = deque(maxlen=schedule.rolling_window_turns)
    recent_target: deque[int] = deque(maxlen=schedule.rolling_window_turns)
    probe_counts: list[int] = []
    probe_route_target_count = 0
    trajectory: list[dict[str, Any]] = []
    baseline_policy_rows: list[np.ndarray] = []
    baseline_policy: np.ndarray | None = None
    trigger: dict[str, Any] | None = None
    alice_reward_sum = 0.0
    eve_public_sum = 0.0
    started = time.perf_counter()

    for turn_index in range(schedule.total_turns):
        if fixed_actions is None and turn_index == schedule.exploit_start:
            trigger = _probe_trigger(
                probe_counts,
                plan,
                uniform_target_mass=len(target_set) / len(case.routes),
                z=schedule.probe_confidence_z,
            )
            if trigger["triggered"]:
                ordered_actions[
                    schedule.exploit_start:schedule.recovery_start
                ] = plan.exploit_action_index
        route_index = alice.sample()
        action_index = int(ordered_actions[turn_index])
        draw_index = stream.next_draw(route_index, action_index)
        payoff_observation = payoff_block.draw(
            route_index,
            action_index,
            draw_index,
        )
        endpoint_observation = (
            endpoint_block.draw(
                route_index,
                action_index,
                draw_index,
                requested_pair_count=(
                    endpoint_evidence_bank.requested_pairs_per_draw
                ),
            )
            if (
                endpoint_block is not None
                and endpoint_evidence_bank is not None
            )
            else None
        )
        reward = float(
            endpoint_observation["alice_reward"]
            if endpoint_observation is not None
            else payoff_observation["alice_reward"]
        )
        alice.update(reward)
        detector_rows: dict[str, Any] | None = None
        if (
            alice_detectors is not None
            and endpoint_observation is not None
            and not detector_monitoring_complete
        ):
            detector_rows = alice_detectors.update(
                route_index,
                endpoint_observation["endpoint_evidence"],
            )
            for detector_name, detector_row in detector_rows.items():
                if detector_row["newly_alarmed"]:
                    detector_alarm_snapshots[detector_name] = {
                        **dict(detector_row),
                        "absolute_turn": turn_index + 1,
                        "phase": _ordered_phase(turn_index, schedule),
                        "delay_from_prime_start_turns": (
                            turn_index + 1 - schedule.prime_start
                            if turn_index >= schedule.prime_start
                            else None
                        ),
                        "pre_prime_false_alarm": (
                            turn_index < schedule.prime_start
                        ),
                    }
            detector_monitoring_complete = bool(
                alice_detectors.classical.alarmed
                and alice_detectors.quantum.alarmed
                and alice_detectors.delivery_only.alarmed
                and alice_detectors.measurement_counts.alarmed
            )
        target_selected = int(route_index in target_set)
        action_cost = float(bank.action_costs[action_index])
        eve_public = 1.0 - reward - action_cost
        phase = (
            _ordered_phase(turn_index, schedule)
            if fixed_actions is None
            else "shuffled_control"
        )
        phase_counts[phase] += 1
        phase_row = phase_sums.setdefault(
            phase,
            {
                "alice_reward": 0.0,
                "eve_public_utility": 0.0,
                "target_selection": 0.0,
                "observation_count": 0.0,
            },
        )
        phase_row["alice_reward"] += reward
        phase_row["eve_public_utility"] += eve_public
        phase_row["target_selection"] += target_selected
        phase_row["observation_count"] += int(
            payoff_observation["eve_observation_count"]
        )
        action_counts[action_index] += 1
        joint_counts[route_index, action_index] += 1
        alice_reward_sum += reward
        eve_public_sum += eve_public
        recent_rewards.append(reward)
        recent_target.append(target_selected)
        if fixed_actions is None and phase == "probe":
            probe_counts.append(
                int(payoff_observation["eve_observation_count"])
            )
            probe_route_target_count += target_selected

        step = turn_index + 1
        if progress_callback is not None and (
            step == 1
            or step == schedule.total_turns
            or step % max(50_000, 10 * schedule.record_every) == 0
        ):
            progress_callback(step, phase)
        if step == 1 or step == schedule.total_turns or step % schedule.record_every == 0:
            policy = alice.policy(turn=step + 1)
            target_mass = float(np.sum(policy[list(target_set)]))
            baseline_tail_start = schedule.warmup_turns - schedule.baseline_tail_turns
            if baseline_tail_start < step <= schedule.warmup_turns:
                baseline_policy_rows.append(policy.copy())
            if step == schedule.warmup_turns:
                baseline_policy = np.mean(baseline_policy_rows, axis=0)
                baseline_policy /= baseline_policy.sum()
            row = {
                "turn": step,
                "condition": condition,
                "phase": phase,
                "alice_reward_mean": alice_reward_sum / step,
                "eve_public_utility_mean": eve_public_sum / step,
                "rolling_alice_reward": float(np.mean(recent_rewards)),
                "rolling_target_selection_rate": float(np.mean(recent_target)),
                "alice_target_family_policy_mass": target_mass,
                "alice_policy_effective_action_fraction": (
                    policy_distribution_statistics(policy)["effective_action_fraction"]
                ),
                "alice_policy_tv_from_warmup": (
                    _total_variation(policy, baseline_policy)
                    if baseline_policy is not None else None
                ),
                "cumulative_local_observation_count": int(sum(probe_counts)),
                "_alice_policy": policy,
            }
            if probe_counts:
                row["local_target_mass_interval"] = _local_mass_interval(
                    probe_counts,
                    plan,
                    z=schedule.probe_confidence_z,
                )
            if alice_detectors is not None:
                row["alice_detector"] = _detector_trajectory_row(
                    alice_detectors,
                    detector_alarm_snapshots,
                )
            trajectory.append(row)
        if detector_rows is not None and any(
            bool(row["newly_alarmed"]) for row in detector_rows.values()
        ):
            detector_trace.append({
                "turn": step,
                "phase": phase,
                **_detector_trajectory_row(
                    alice_detectors,
                    detector_alarm_snapshots,
                ),
            })

    if fixed_actions is None and trigger is None:
        trigger = _probe_trigger(
            probe_counts,
            plan,
            uniform_target_mass=len(target_set) / len(case.routes),
            z=schedule.probe_confidence_z,
        )
    if fixed_actions is not None:
        trigger = {
            "triggered": None,
            "decision_rule": "not_reapplied_in_temporal_order_control",
        }

    baseline = _baseline_envelope(trajectory, schedule)
    post_prime = next(
        row for row in trajectory if int(row["turn"]) == schedule.probe_start
    )
    virtual_bottleneck = {
        "offline_scoring_only": True,
        "post_prime_target_family_policy_mass": float(
            post_prime["alice_target_family_policy_mass"]
        ),
        "late_warmup_target_mass_mean": float(baseline["target_mass_mean"]),
        "post_prime_minus_late_warmup_mean": float(
            post_prime["alice_target_family_policy_mass"]
            - baseline["target_mass_mean"]
        ),
        "post_prime_exceeds_late_warmup_upper_quantile": bool(
            post_prime["alice_target_family_policy_mass"]
            > baseline["target_mass_upper_97p5"]
        ),
        "used_by_eve_trigger": False,
    }
    recovery = _recovery_telemetry(trajectory, schedule, baseline)
    for row in trajectory:
        row.pop("_alice_policy", None)
    audit_expected_reward = None
    if include_audit_scoring:
        audit_matrix = bank.audit.alice_payoff_matrix()
        audit_expected_reward = float(
            np.sum(joint_counts * audit_matrix) / schedule.total_turns
        )
    alice_detector = None
    if alice_detectors is not None:
        diagnostics = alice_detectors.diagnostics()
        alice_detector = {
            "enabled": True,
            "response": "monitor_only_native_learning",
            "execution_split": execution_split,
            "commissioning_split": "feedback",
            "commissioning_draws_per_route": int(
                detector_commissioning_draws_per_route
            ),
            "prime_first_turn": int(schedule.prime_start + 1),
            "outcomes": _detector_outcomes(
                alice_detectors,
                detector_alarm_snapshots,
                schedule,
                requested_pairs_per_turn=(
                    endpoint_evidence_bank.requested_pairs_per_draw
                ),
            ),
            "detectors": diagnostics,
            "alarm_trace": detector_trace,
            "information_boundary": diagnostics["information_boundary"],
        }
    action_indices = ordered_actions.astype(int).tolist()
    action_multiset = {
        bank.action_ids[index]: int(count)
        for index, count in sorted(action_counts.items())
    }
    phase_metrics = {
        phase: {
            "turns": int(phase_counts[phase]),
            "mean_alice_reward": values["alice_reward"] / phase_counts[phase],
            "mean_eve_public_utility": (
                values["eve_public_utility"] / phase_counts[phase]
            ),
            "target_selection_rate_offline": (
                values["target_selection"] / phase_counts[phase]
            ),
            "mean_local_observation_count": (
                values["observation_count"] / phase_counts[phase]
            ),
        }
        for phase, values in phase_sums.items()
    }
    return {
        "schema": PRIME_PROBE_RESULT_SCHEMA,
        "condition": condition,
        "case_id": bank.case_id,
        "graph_id": case.graph_id,
        "family": case.family,
        "seed": int(seed),
        "turns": schedule.total_turns,
        "elapsed_seconds": time.perf_counter() - started,
        "schedule": schedule.to_dict(),
        "plan": plan.to_dict(),
        "alice_learner": dict(alice.metadata),
        "alice_current_policy": alice.policy(turn=schedule.total_turns + 1).tolist(),
        "alice_empirical_strategy": alice.empirical_strategy().tolist(),
        "mean_alice_reward": alice_reward_sum / schedule.total_turns,
        "mean_eve_public_utility": eve_public_sum / schedule.total_turns,
        "audit_expected_alice_reward": audit_expected_reward,
        "execution_split": execution_split,
        "alice_endpoint_detector": alice_detector,
        "joint_action_counts": joint_counts.tolist(),
        "action_indices": action_indices,
        "action_multiset": action_multiset,
        "phase_metrics": phase_metrics,
        "probe_trigger": trigger,
        "probe_information_regimes": {
            "live_policy_upper_bound_offline": (
                next(
                    (
                        row["alice_target_family_policy_mass"]
                        for row in trajectory
                        if row["turn"] == schedule.exploit_start
                    ),
                    None,
                )
            ),
            "sampled_route_history_upper_bound_offline": (
                probe_route_target_count / len(probe_counts)
                if probe_counts else None
            ),
            "local_instrument_count_interval": (
                _local_mass_interval(
                    probe_counts,
                    plan,
                    z=schedule.probe_confidence_z,
                )
                if probe_counts else None
            ),
            "eta_and_update_rule_without_action_identity": (
                len(target_set) / len(case.routes)
            ),
        },
        "baseline_envelope": baseline,
        "virtual_bottleneck": virtual_bottleneck,
        "recovery": recovery,
        "trajectory": trajectory,
        "information_boundary": {
            "eve_online_observation": (
                "count-valued local record from the declared probe instrument"
            ),
            "probe_trigger_uses_route_identity": False,
            "probe_trigger_uses_alice_policy_or_weights": False,
            "probe_trigger_uses_target_active": False,
            "probe_trigger_uses_audit_split": (
                execution_split == "audit"
            ),
            "audit_split_read_for_offline_scoring": bool(
                include_audit_scoring
            ),
            "online_execution_split": execution_split,
            "audit_split_role": (
                "held-out live execution; not post-hoc scoring"
                if execution_split == "audit"
                else "post-hoc scoring only"
            ),
            "alice_endpoint_detector_enabled": (
                alice_detectors is not None
            ),
            "alice_detector_changes_learner_state": False,
            "alice_detector_observes_eve_local_record": False,
            "alice_detector_observes_eve_action_identity": False,
            "offline_diagnostics": [
                "Alice policy target-family mass",
                "sampled target-route indicator",
                "joint route/action counts",
            ],
            "physical_interventions_only": True,
            "reward_poisoning": False,
        },
    }


def run_prime_probe_withdrawal_state_study(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    plan: PrimeProbePlan,
    *,
    schedule: PrimeProbeSchedule,
    seed: int,
    selection_digest_blake2b_128: str | None = None,
) -> dict[str, Any]:
    """Branch learner state at withdrawal under attacked and clean prefixes.

    This is an oracle-timed sensitivity analysis, not an online response
    policy.  The experiment supplies the known withdrawal boundary and, for
    one branch, an in-memory learner checkpoint retained at the end of
    Warmup.  A clean sham prefix receives the same state-management operations
    at the same turns so their ordinary-operation cost is measurable.
    """

    selection_digest = (
        str(selection_digest_blake2b_128)
        if selection_digest_blake2b_128 is not None
        else str(bank.metadata["selection_digest_blake2b_128"])
    )
    validate_bank_against_case(
        bank,
        case,
        config,
        selection_digest_blake2b_128=selection_digest,
    )
    active_config = replace(
        config,
        turns=schedule.total_turns,
        seed=int(seed),
    )
    attacked = _run_prime_probe_prefix(
        case,
        bank,
        active_config,
        plan,
        schedule=schedule,
        seed=int(seed),
        condition="ordered_intervention_prefix",
        attack_enabled=True,
    )
    clean = _run_prime_probe_prefix(
        case,
        bank,
        active_config,
        plan,
        schedule=schedule,
        seed=int(seed),
        condition="clean_sham_prefix",
        attack_enabled=False,
    )
    if not bool((attacked.probe_trigger or {}).get("triggered")):
        raise RuntimeError(
            "withdrawal-state study requires the predeclared Probe to trigger"
        )

    attacked_bit_generator = type(
        attacked.alice.learner.rng.bit_generator
    )
    clean_bit_generator = type(clean.alice.learner.rng.bit_generator)
    if attacked_bit_generator is not clean_bit_generator:
        raise AssertionError(
            "attacked and clean prefixes use different RNG implementations"
        )
    prefix_rng_states_matched = (
        attacked.alice.learner.rng.bit_generator.state
        == clean.alice.learner.rng.bit_generator.state
    )
    recovery_stream_seed = int(
        np.random.SeedSequence([int(seed), 0x5253]).generate_state(
            1,
            dtype=np.uint64,
        )[0]
    )
    recovery_outcome_stream = CommonCellOutcomeStream(
        len(bank.route_ids),
        len(bank.action_ids),
        bank.draws_per_cell,
        seed=recovery_stream_seed,
    )
    recovery_rng = np.random.Generator(
        attacked_bit_generator(
            np.random.SeedSequence([int(seed), 0x524E47])
        )
    )
    recovery_rng_state = copy.deepcopy(
        recovery_rng.bit_generator.state
    )

    branch_rows: list[dict[str, Any]] = []
    for prefix in (attacked, clean):
        for strategy in WITHDRAWAL_STATE_STRATEGIES:
            branch_rows.append(
                _run_withdrawal_state_branch(
                    case,
                    bank,
                    active_config,
                    plan,
                    schedule=schedule,
                    prefix=prefix,
                    strategy=strategy,
                    recovery_outcome_stream=recovery_outcome_stream,
                    recovery_rng_state=recovery_rng_state,
                )
            )
    _assert_matched_state_operation_branches(branch_rows)
    paired_rows = _pair_withdrawal_state_branches(branch_rows)
    within_condition = _compare_withdrawal_strategies(branch_rows)
    difference_in_differences = _withdrawal_difference_in_differences(
        within_condition
    )
    return {
        "schema": PRIME_PROBE_WITHDRAWAL_STATE_SCHEMA,
        "case_id": bank.case_id,
        "graph_id": case.graph_id,
        "family": case.family,
        "seed": int(seed),
        "schedule": schedule.to_dict(),
        "plan": plan.to_dict(),
        "alice_learner": dict(attacked.alice.metadata),
        "probe_trigger": dict(attacked.probe_trigger or {}),
        "prefixes": {
            attacked.condition: {
                "turns": schedule.recovery_start,
                "mean_alice_reward": attacked.mean_alice_reward,
                "ending_alice_update_calls": attacked.alice.update_calls,
            },
            clean.condition: {
                "turns": schedule.recovery_start,
                "mean_alice_reward": clean.mean_alice_reward,
                "ending_alice_update_calls": clean.alice.update_calls,
            },
        },
        "branches": branch_rows,
        "attacked_clean_pairs": paired_rows,
        "within_condition_comparisons": within_condition,
        "difference_in_differences": difference_in_differences,
        "information_boundary": {
            "withdrawal_boundary_supplied_to_state_manager": True,
            "alice_endpoint_alarm_used": False,
            "change_point_estimated_online": False,
            "warmup_checkpoint_available_to_deployed_learner": False,
            "warmup_checkpoint_semantics": (
                "in-memory copy of the complete learner immediately after "
                "Warmup; restored statistics retain their Warmup turn count"
            ),
            "full_restart_semantics": (
                "production uniform initialization with turn count reset to "
                "zero"
            ),
            "sampling_rng_coupling": (
                "all six branches start from the same independently seeded "
                "post-withdrawal Alice sampling-RNG state"
            ),
            "prefix_sampling_rng_states_matched_at_withdrawal": (
                prefix_rng_states_matched
            ),
            "outcome_stream_coupling": (
                "all six branches start from one fresh post-withdrawal "
                "per-cell outcome-stream state; the nth visit to the same "
                "route/no-attack cell therefore uses the same feedback draw"
            ),
            "recovery_outcome_stream_seed": recovery_stream_seed,
            "audit_split_use": "offline branch scoring only",
            "physical_interventions_only": True,
            "reward_poisoning": False,
        },
        "claim_boundary": (
            "The study measures sensitivity to three state-management "
            "operations at a known withdrawal turn. It does not implement "
            "alarm-conditioned recovery, estimate intervention onset or "
            "withdrawal, or attribute learner differences solely to topology "
            "sharing."
        ),
    }


def _run_prime_probe_prefix(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    plan: PrimeProbePlan,
    *,
    schedule: PrimeProbeSchedule,
    seed: int,
    condition: str,
    attack_enabled: bool,
) -> _PrimeProbePrefixState:
    alice = _new_prime_probe_alice(case, bank, config)
    stream = CommonCellOutcomeStream(
        len(bank.route_ids),
        len(bank.action_ids),
        bank.draws_per_cell,
        seed=int(seed),
    )
    no_attack_index = bank.action_ids.index("no_attack")
    action_indices = np.full(
        schedule.recovery_start,
        no_attack_index,
        dtype=np.int64,
    )
    if attack_enabled:
        prime_rng = np.random.default_rng([int(seed), 53])
        action_indices[schedule.prime_start:schedule.probe_start] = (
            prime_rng.choice(
                np.asarray(plan.prime_action_indices, dtype=np.int64),
                size=schedule.prime_turns,
                p=np.asarray(plan.prime_probabilities, dtype=float),
            )
        )
        action_indices[schedule.probe_start:schedule.exploit_start] = (
            plan.probe_action_index
        )

    recent_rewards: deque[float] = deque(
        maxlen=schedule.rolling_window_turns
    )
    baseline_policies: list[np.ndarray] = []
    baseline_services: list[float] = []
    probe_counts: list[int] = []
    warmup_checkpoint: PlayerLearner | None = None
    trigger: dict[str, Any] | None = None
    reward_sum = 0.0

    for turn_index in range(schedule.recovery_start):
        if (
            attack_enabled
            and turn_index == schedule.exploit_start
        ):
            trigger = _probe_trigger(
                probe_counts,
                plan,
                uniform_target_mass=(
                    len(plan.target_route_indices) / len(case.routes)
                ),
                z=schedule.probe_confidence_z,
            )
            if trigger["triggered"]:
                action_indices[
                    schedule.exploit_start:schedule.recovery_start
                ] = plan.exploit_action_index

        route_index = alice.sample()
        action_index = int(action_indices[turn_index])
        draw_index = stream.next_draw(route_index, action_index)
        observation = bank.feedback.draw(
            route_index,
            action_index,
            draw_index,
        )
        reward = float(observation["alice_reward"])
        alice.update(reward)
        reward_sum += reward
        recent_rewards.append(reward)
        if (
            attack_enabled
            and schedule.probe_start
            <= turn_index
            < schedule.exploit_start
        ):
            probe_counts.append(int(observation["eve_observation_count"]))

        step = turn_index + 1
        if step == schedule.warmup_turns:
            warmup_checkpoint = copy.deepcopy(alice)
        if (
            step % schedule.record_every == 0
            or step == schedule.warmup_turns
        ):
            baseline_start = (
                schedule.warmup_turns - schedule.baseline_tail_turns
            )
            if baseline_start < step <= schedule.warmup_turns:
                baseline_policies.append(alice.policy())
                baseline_services.append(float(np.mean(recent_rewards)))

    if warmup_checkpoint is None:
        raise AssertionError("Warmup checkpoint was not retained")
    if not baseline_policies or not baseline_services:
        raise AssertionError("late-Warmup baseline was not recorded")
    baseline_policy = np.mean(baseline_policies, axis=0)
    baseline_policy /= np.sum(baseline_policy)
    services = np.asarray(baseline_services, dtype=float)
    if attack_enabled and trigger is None:
        trigger = _probe_trigger(
            probe_counts,
            plan,
            uniform_target_mass=(
                len(plan.target_route_indices) / len(case.routes)
            ),
            z=schedule.probe_confidence_z,
        )
    return _PrimeProbePrefixState(
        condition=condition,
        alice=alice,
        warmup_checkpoint=warmup_checkpoint,
        outcome_stream=stream,
        baseline_policy=baseline_policy,
        baseline_service_lower=float(np.quantile(services, 0.025)),
        baseline_service_upper=float(np.quantile(services, 0.975)),
        probe_trigger=trigger,
        mean_alice_reward=reward_sum / schedule.recovery_start,
    )


def _new_prime_probe_alice(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
) -> PlayerLearner:
    implementation, metadata = build_histogram_alice_learner(
        case=case,
        config=config,
        prior=np.full(len(case.routes), 1.0 / len(case.routes)),
    )
    return PlayerLearner(
        player="alice",
        action_ids=bank.route_ids,
        learner=implementation,
        metadata=metadata,
    )


def _run_withdrawal_state_branch(
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    plan: PrimeProbePlan,
    *,
    schedule: PrimeProbeSchedule,
    prefix: _PrimeProbePrefixState,
    strategy: str,
    recovery_outcome_stream: CommonCellOutcomeStream,
    recovery_rng_state: Mapping[str, Any],
) -> dict[str, Any]:
    if strategy not in WITHDRAWAL_STATE_STRATEGIES:
        raise ValueError(f"unsupported withdrawal strategy {strategy!r}")
    if strategy == WITHDRAWAL_NATIVE_CONTINUATION:
        alice = copy.deepcopy(prefix.alice)
    elif strategy == WITHDRAWAL_FULL_RESTART:
        alice = _new_prime_probe_alice(case, bank, config)
    else:
        alice = copy.deepcopy(prefix.warmup_checkpoint)
    alice.learner.rng.bit_generator.state = copy.deepcopy(recovery_rng_state)
    stream = copy.deepcopy(recovery_outcome_stream)

    starting_updates = int(alice.update_calls)
    no_attack_index = bank.action_ids.index("no_attack")
    audit_reward = (
        bank.audit.alice_payoff_matrix()[:, no_attack_index].astype(float)
    )
    best_audit_reward = float(np.max(audit_reward))
    target = np.asarray(plan.target_route_indices, dtype=np.int64)
    target_set = set(plan.target_route_indices)
    recent_rewards: deque[float] = deque(
        maxlen=schedule.rolling_window_turns
    )
    reward_sum = 0.0
    audit_reward_sum = 0.0
    audit_deficit_sum = 0.0
    target_count = 0
    trajectory: list[dict[str, Any]] = []

    for turn_index in range(schedule.recovery_turns):
        route_index = alice.sample()
        draw_index = stream.next_draw(route_index, no_attack_index)
        observation = bank.feedback.draw(
            route_index,
            no_attack_index,
            draw_index,
        )
        reward = float(observation["alice_reward"])
        alice.update(reward)
        route_audit_reward = float(audit_reward[route_index])
        reward_sum += reward
        audit_reward_sum += route_audit_reward
        audit_deficit_sum += best_audit_reward - route_audit_reward
        target_count += int(route_index in target_set)
        recent_rewards.append(reward)

        step = turn_index + 1
        if (
            step == 1
            or step == schedule.recovery_turns
            or step % schedule.record_every == 0
        ):
            policy = alice.policy()
            trajectory.append({
                "recovery_turn": step,
                "rolling_turn_count": len(recent_rewards),
                "rolling_alice_reward": float(np.mean(recent_rewards)),
                "target_family_policy_mass": float(np.sum(policy[target])),
                "policy_tv_from_late_warmup": _total_variation(
                    policy,
                    prefix.baseline_policy,
                ),
                "audit_policy_service": float(policy @ audit_reward),
                "_policy": policy,
            })

    final_policy = alice.policy()
    recovery = _withdrawal_service_recovery(
        trajectory,
        lower=prefix.baseline_service_lower,
        upper=prefix.baseline_service_upper,
        rolling_window_turns=schedule.rolling_window_turns,
        required=schedule.sustained_recovery_checkpoints,
        censor=schedule.recovery_turns,
    )
    checkpoint_mean_tv = float(np.mean([
        float(row["policy_tv_from_late_warmup"])
        for row in trajectory
    ]))
    public_trajectory = []
    for row in trajectory:
        public_trajectory.append({
            key: value for key, value in row.items() if key != "_policy"
        })
    return {
        "prefix_condition": prefix.condition,
        "strategy": strategy,
        "recovery_turns": schedule.recovery_turns,
        "starting_alice_update_calls": starting_updates,
        "ending_alice_update_calls": int(alice.update_calls),
        "mean_feedback_alice_reward": reward_sum / schedule.recovery_turns,
        "mean_audit_selected_route_service": (
            audit_reward_sum / schedule.recovery_turns
        ),
        "cumulative_audit_service_deficit": audit_deficit_sum,
        "mean_audit_service_deficit": (
            audit_deficit_sum / schedule.recovery_turns
        ),
        "best_audit_no_attack_route_service": best_audit_reward,
        "final_audit_policy_service": float(final_policy @ audit_reward),
        "final_target_family_policy_mass": float(
            np.sum(final_policy[target])
        ),
        "sampled_target_family_rate": (
            target_count / schedule.recovery_turns
        ),
        "final_policy_tv_from_late_warmup": _total_variation(
            final_policy,
            prefix.baseline_policy,
        ),
        "checkpoint_mean_policy_tv_from_late_warmup": checkpoint_mean_tv,
        "service_recovery": recovery,
        "current_policy": final_policy.tolist(),
        "trajectory": public_trajectory,
        "_trajectory_policies": [
            row["_policy"] for row in trajectory
        ],
    }


def _withdrawal_service_recovery(
    trajectory: Sequence[Mapping[str, Any]],
    *,
    lower: float,
    upper: float,
    rolling_window_turns: int,
    required: int,
    censor: int,
) -> dict[str, Any]:
    qualifying = []
    for row in trajectory:
        complete_window = (
            int(row["rolling_turn_count"]) >= int(rolling_window_turns)
        )
        reward = float(row["rolling_alice_reward"])
        qualifying.append(complete_window and reward >= lower)
    for index in range(max(0, len(qualifying) - required + 1)):
        if all(qualifying[index:index + required]):
            turn = int(trajectory[index]["recovery_turn"])
            return {
                "recovered": True,
                "latency_turns": turn,
                "right_censored": False,
                "censor_turns": int(censor),
                "baseline_lower_2p5": float(lower),
                "baseline_upper_97p5": float(upper),
                "criterion": (
                    "rolling_alice_reward >= baseline_lower_2p5"
                ),
            }
    return {
        "recovered": False,
        "latency_turns": None,
        "right_censored": True,
        "censor_turns": int(censor),
        "baseline_lower_2p5": float(lower),
        "baseline_upper_97p5": float(upper),
        "criterion": "rolling_alice_reward >= baseline_lower_2p5",
    }


def _pair_withdrawal_state_branches(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (str(row["prefix_condition"]), str(row["strategy"])): row
        for row in rows
    }
    paired: list[dict[str, Any]] = []
    for strategy in WITHDRAWAL_STATE_STRATEGIES:
        attacked = indexed[("ordered_intervention_prefix", strategy)]
        clean = indexed[("clean_sham_prefix", strategy)]
        attacked_policies = attacked.pop("_trajectory_policies")
        clean_policies = clean.pop("_trajectory_policies")
        if len(attacked_policies) != len(clean_policies):
            raise AssertionError("attacked and clean checkpoints differ")
        paired.append({
            "strategy": strategy,
            "attacked_minus_clean_mean_feedback_reward": (
                float(attacked["mean_feedback_alice_reward"])
                - float(clean["mean_feedback_alice_reward"])
            ),
            "attacked_minus_clean_cumulative_audit_service_deficit": (
                float(attacked["cumulative_audit_service_deficit"])
                - float(clean["cumulative_audit_service_deficit"])
            ),
            "attacked_minus_clean_final_audit_policy_service": (
                float(attacked["final_audit_policy_service"])
                - float(clean["final_audit_policy_service"])
            ),
            "attacked_clean_final_policy_tv": _total_variation(
                np.asarray(attacked["current_policy"], dtype=float),
                np.asarray(clean["current_policy"], dtype=float),
            ),
            "attacked_clean_checkpoint_mean_policy_tv": float(np.mean([
                _total_variation(attacked_policy, clean_policy)
                for attacked_policy, clean_policy in zip(
                    attacked_policies,
                    clean_policies,
                )
            ])),
        })
    return paired


def _assert_matched_state_operation_branches(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject prefix leakage after operations that erase the prefix."""

    indexed = {
        (str(row["prefix_condition"]), str(row["strategy"])): row
        for row in rows
    }
    exact_keys = (
        "mean_feedback_alice_reward",
        "mean_audit_selected_route_service",
        "cumulative_audit_service_deficit",
        "final_audit_policy_service",
        "current_policy",
        "trajectory",
        "service_recovery",
    )
    for strategy in (
        WITHDRAWAL_FULL_RESTART,
        WITHDRAWAL_WARMUP_RESTORE,
    ):
        attacked = indexed[("ordered_intervention_prefix", strategy)]
        clean = indexed[("clean_sham_prefix", strategy)]
        for key in exact_keys:
            if attacked[key] != clean[key]:
                raise AssertionError(
                    f"{strategy} retains prefix-dependent {key}"
                )


def _compare_withdrawal_strategies(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (str(row["prefix_condition"]), str(row["strategy"])): row
        for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    for condition in ("ordered_intervention_prefix", "clean_sham_prefix"):
        native = indexed[(condition, WITHDRAWAL_NATIVE_CONTINUATION)]
        for strategy in (
            WITHDRAWAL_FULL_RESTART,
            WITHDRAWAL_WARMUP_RESTORE,
        ):
            response = indexed[(condition, strategy)]
            comparisons.append({
                "prefix_condition": condition,
                "strategy": strategy,
                "strategy_minus_native_mean_feedback_reward": (
                    float(response["mean_feedback_alice_reward"])
                    - float(native["mean_feedback_alice_reward"])
                ),
                "strategy_minus_native_cumulative_audit_service_deficit": (
                    float(response["cumulative_audit_service_deficit"])
                    - float(native["cumulative_audit_service_deficit"])
                ),
                "strategy_minus_native_final_audit_policy_service": (
                    float(response["final_audit_policy_service"])
                    - float(native["final_audit_policy_service"])
                ),
            })
    return comparisons


def _withdrawal_difference_in_differences(
    comparisons: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Subtract each state operation's clean cost from its attacked effect."""

    indexed = {
        (str(row["prefix_condition"]), str(row["strategy"])): row
        for row in comparisons
    }
    rows: list[dict[str, Any]] = []
    for strategy in (
        WITHDRAWAL_FULL_RESTART,
        WITHDRAWAL_WARMUP_RESTORE,
    ):
        attacked = indexed[("ordered_intervention_prefix", strategy)]
        clean = indexed[("clean_sham_prefix", strategy)]
        rows.append({
            "strategy": strategy,
            "estimand": (
                "(strategy - native)_ordered_intervention_prefix - "
                "(strategy - native)_clean_sham_prefix"
            ),
            "audit_deficit_sign": (
                "negative means the state operation reduces attack-specific "
                "post-withdrawal audit service deficit after subtracting its "
                "clean-sham cost"
            ),
            "reward_and_service_sign": (
                "positive means the state operation improves the "
                "attack-specific outcome after subtracting its clean-sham "
                "effect"
            ),
            "did_cumulative_audit_service_deficit": (
                float(attacked[
                    "strategy_minus_native_cumulative_audit_service_deficit"
                ])
                - float(clean[
                    "strategy_minus_native_cumulative_audit_service_deficit"
                ])
            ),
            "did_mean_feedback_alice_reward": (
                float(attacked["strategy_minus_native_mean_feedback_reward"])
                - float(clean[
                    "strategy_minus_native_mean_feedback_reward"
                ])
            ),
            "did_final_audit_policy_service": (
                float(attacked[
                    "strategy_minus_native_final_audit_policy_service"
                ])
                - float(clean[
                    "strategy_minus_native_final_audit_policy_service"
                ])
            ),
        })
    return rows


def paired_prime_probe_comparison(
    ordered: Mapping[str, Any],
    shuffled: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare timing while certifying an identical action economy."""

    if ordered["action_multiset"] != shuffled["action_multiset"]:
        raise ValueError("ordered and shuffled runs do not share an action multiset")
    ordered_audit = ordered["audit_expected_alice_reward"]
    shuffled_audit = shuffled["audit_expected_alice_reward"]
    return {
        "graph_id": ordered["graph_id"],
        "seed": ordered["seed"],
        "alice_learner_mode": ordered["alice_learner"]["mode"],
        "identical_action_multiset": True,
        "ordered_minus_shuffled_mean_alice_reward": (
            ordered["mean_alice_reward"] - shuffled["mean_alice_reward"]
        ),
        "ordered_minus_shuffled_audit_expected_alice_reward": (
            ordered_audit - shuffled_audit
            if ordered_audit is not None and shuffled_audit is not None
            else None
        ),
        "ordered_minus_shuffled_mean_eve_public_utility": (
            ordered["mean_eve_public_utility"]
            - shuffled["mean_eve_public_utility"]
        ),
        "ordered_triggered": ordered["probe_trigger"]["triggered"],
        "ordered_recovery": dict(ordered["recovery"]),
    }


def _max_min_coverage_distribution(
    coverage: np.ndarray,
) -> tuple[np.ndarray, float]:
    matrix = np.asarray(coverage, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 1:
        raise ValueError("coverage matrix must be nonempty")
    action_count = matrix.shape[1]
    objective = np.zeros(action_count + 1, dtype=float)
    objective[-1] = -1.0
    inequality = np.column_stack((-matrix, np.ones(matrix.shape[0])))
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.zeros(matrix.shape[0]),
        A_eq=np.asarray([[*np.ones(action_count), 0.0]]),
        b_eq=np.ones(1),
        bounds=[(0.0, 1.0)] * action_count + [(0.0, 1.0)],
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"prime coverage LP failed: {result.message}")
    probabilities = np.maximum(np.asarray(result.x[:-1], dtype=float), 0.0)
    probabilities /= probabilities.sum()
    floor = float(np.min(matrix @ probabilities))
    return probabilities, floor


def _endpoint_commissioning_evidence_by_route(
    bank: HistogramEndpointEvidenceBank,
    *,
    stop: int,
) -> tuple[tuple[Any, ...], ...]:
    no_attack_index = bank.action_ids.index("no_attack")
    return tuple(
        tuple(
            bank.feedback.evidence(
                route_index,
                no_attack_index,
                draw_index,
                requested_pair_count=bank.requested_pairs_per_draw,
            )
            for draw_index in range(stop)
        )
        for route_index in range(len(bank.route_ids))
    )


def _validate_matched_bank_provenance(
    payoff_bank: HistogramPayoffBank,
    endpoint_bank: HistogramEndpointEvidenceBank,
) -> None:
    """Require evidence that both banks project the same sampled transcripts."""

    payoff_metadata = payoff_bank.metadata
    endpoint_metadata = endpoint_bank.metadata
    payoff_schema = str(payoff_metadata.get("matched_sampling_schema", ""))
    endpoint_schema = str(
        endpoint_metadata.get("matched_sampling_schema", "")
    )
    if (
        payoff_schema != MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA
        or endpoint_schema != MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA
    ):
        raise ValueError(
            "payoff and endpoint banks must come from the matched transcript "
            "builder"
        )
    payoff_id = str(
        payoff_metadata.get("matched_sampling_id_blake2b_128", "")
    )
    endpoint_id = str(
        endpoint_metadata.get("matched_sampling_id_blake2b_128", "")
    )
    if not payoff_id or payoff_id != endpoint_id:
        raise ValueError(
            "payoff and endpoint banks have different matched sampling IDs"
        )
    payoff_seeds = payoff_metadata.get("split_seeds")
    endpoint_seeds = endpoint_metadata.get("split_seeds")
    if (
        not isinstance(payoff_seeds, Mapping)
        or not isinstance(endpoint_seeds, Mapping)
        or dict(payoff_seeds) != dict(endpoint_seeds)
        or set(payoff_seeds) != {"feedback", "audit"}
        or int(payoff_seeds["feedback"]) == int(payoff_seeds["audit"])
    ):
        raise ValueError(
            "matched bank feedback and audit split seeds are absent, "
            "different, or not disjoint"
        )
    for metadata in (payoff_metadata, endpoint_metadata):
        if int(
            metadata.get(
                "matched_sampling_step_calls_per_coordinate",
                0,
            )
        ) != 1:
            raise ValueError(
                "matched banks must record one physics sample per coordinate"
            )


def _validate_frozen_prime_probe_plan(
    plan: PrimeProbePlan,
    *,
    bank: HistogramPayoffBank,
    route_count: int,
) -> None:
    """Reject a plan whose stored identities no longer match its indices."""

    action_pairs = (
        (plan.exploit_action_index, plan.exploit_action_id),
        (plan.probe_action_index, plan.probe_action_id),
        *tuple(zip(plan.prime_action_indices, plan.prime_action_ids)),
    )
    for index, expected_id in action_pairs:
        if (
            not 0 <= int(index) < len(bank.action_ids)
            or bank.action_ids[int(index)] != str(expected_id)
        ):
            raise ValueError(
                "frozen Prime-and-Probe action identity no longer matches "
                "its stored index"
            )
    if len(plan.target_route_indices) != len(plan.target_route_ids):
        raise ValueError("frozen target route indices and IDs differ")
    for index, expected_id in zip(
        plan.target_route_indices,
        plan.target_route_ids,
    ):
        if (
            not 0 <= int(index) < len(bank.route_ids)
            or bank.route_ids[int(index)] != str(expected_id)
        ):
            raise ValueError(
                "frozen Prime-and-Probe route identity no longer matches "
                "its stored index"
            )
    target = set(map(int, plan.target_route_indices))
    alternative = set(map(int, plan.alternative_route_indices))
    if (
        target & alternative
        or target | alternative != set(range(int(route_count)))
    ):
        raise ValueError(
            "frozen target and alternative routes do not partition the "
            "current route universe"
        )


def _detector_trajectory_row(
    detectors: MatchedClassicalQuantumEndpointDetectors,
    alarm_snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "delivery_only_log_e_value": float(
            detectors.delivery_only.log_e_value
        ),
        "classical_service_log_e_value": float(
            detectors.classical.log_e_value
        ),
        "quantum_endpoint_log_e_value": float(
            detectors.quantum.log_e_value
        ),
        "e91_measurement_counts_log_e_value": float(
            detectors.measurement_counts.log_e_value
        ),
        "threshold_log_e_value": float(
            detectors.quantum.threshold_log_e_value
        ),
        "delivery_only_alarm_turn": detectors.delivery_only.alarm_turn,
        "classical_service_alarm_turn": detectors.classical.alarm_turn,
        "quantum_endpoint_alarm_turn": detectors.quantum.alarm_turn,
        "e91_measurement_counts_alarm_turn": (
            detectors.measurement_counts.alarm_turn
        ),
        "alarm_snapshots": {
            str(name): dict(value)
            for name, value in alarm_snapshots.items()
        },
    }


def _detector_outcomes(
    detectors: MatchedClassicalQuantumEndpointDetectors,
    alarm_snapshots: Mapping[str, Mapping[str, Any]],
    schedule: PrimeProbeSchedule,
    *,
    requested_pairs_per_turn: int,
) -> dict[str, Any]:
    detector_rows = (
        (
            "delivery_only",
            "classical_delivery_only",
            detectors.delivery_only,
        ),
        (
            "classical_service",
            "classical_service_baseline",
            detectors.classical,
        ),
        (
            "quantum_endpoint",
            "quantum_endpoint_integrity",
            detectors.quantum,
        ),
        (
            "e91_measurement_counts_only",
            "e91_measurement_counts_only",
            detectors.measurement_counts,
        ),
    )
    outcomes: dict[str, Any] = {}
    for label, snapshot_key, detector in detector_rows:
        alarm_turn = detector.alarm_turn
        pre_prime = bool(
            alarm_turn is not None
            and alarm_turn <= schedule.prime_start
        )
        outcomes[label] = {
            "alarm_turn": alarm_turn,
            "alarm_phase": (
                _ordered_phase(alarm_turn - 1, schedule)
                if alarm_turn is not None
                else None
            ),
            "pre_prime_false_alarm": pre_prime,
            "detection_delay_from_prime_start_turns": (
                int(alarm_turn - schedule.prime_start)
                if alarm_turn is not None and not pre_prime
                else None
            ),
            "requested_pairs_from_prime_start_to_alarm": (
                int(alarm_turn - schedule.prime_start)
                * int(requested_pairs_per_turn)
                if alarm_turn is not None and not pre_prime
                else None
            ),
            "observed_through_turn": int(detector.turn),
            "right_censored": bool(
                alarm_turn is None
                and detector.turn >= schedule.total_turns
            ),
            "leading_alternative_at_alarm": (
                dict(alarm_snapshots[snapshot_key]).get(
                    "leading_alternative"
                )
                if snapshot_key in alarm_snapshots
                else None
            ),
        }
        leading = outcomes[label]["leading_alternative_at_alarm"]
        outcomes[label]["leading_channel_at_alarm"] = (
            str(leading["channel"])
            if isinstance(leading, Mapping) and "channel" in leading
            else None
        )
    return outcomes


def _matching_probe_index(
    actions: Sequence[Mapping[str, Any]],
    exploit: Mapping[str, Any],
) -> int | None:
    family = str(exploit["attack_type"])
    probe_family = (
        "edge_information_probe" if family == "edge_dos"
        else "repeater_memory_measure_Z" if family == "swap_denial"
        else None
    )
    if probe_family is None:
        return None
    target = str(exploit["target"])
    matches = [
        index
        for index, action in enumerate(actions)
        if action["attack_type"] == probe_family and str(action["target"]) == target
    ]
    return matches[0] if len(matches) == 1 else None


def _unique_action_index(
    actions: Sequence[Mapping[str, Any]], family: str
) -> int:
    matches = [
        index for index, action in enumerate(actions)
        if action["attack_type"] == family
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one {family!r} action, found {len(matches)}")
    return matches[0]


def _ordered_phase(turn_index: int, schedule: PrimeProbeSchedule) -> str:
    if turn_index < schedule.prime_start:
        return "warmup"
    if turn_index < schedule.probe_start:
        return "prime"
    if turn_index < schedule.exploit_start:
        return "probe"
    if turn_index < schedule.recovery_start:
        return "exploit"
    return "recovery"


def _local_mass_interval(
    counts: Sequence[int],
    plan: PrimeProbePlan,
    *,
    z: float,
) -> dict[str, float | int]:
    values = np.asarray(counts, dtype=float)
    if values.ndim != 1 or values.size < 1 or np.any(values < 0.0):
        raise ValueError("probe counts must be a nonnegative vector")
    mean = float(np.mean(values))
    standard_error = (
        float(np.std(values, ddof=1) / math.sqrt(values.size))
        if values.size > 1 else 0.0
    )
    mean_lower = max(0.0, mean - z * standard_error)
    mean_upper = mean + z * standard_error
    a_min = plan.probe_nontarget_count_minimum
    a_max = plan.probe_nontarget_count_maximum
    f_min = plan.probe_target_count_minimum
    f_max = plan.probe_target_count_maximum
    lower = np.clip((mean_lower - a_max) / (f_max - a_max), 0.0, 1.0)
    upper = np.clip((mean_upper - a_min) / (f_min - a_min), 0.0, 1.0)
    return {
        "samples": int(values.size),
        "mean_observation_count": mean,
        "mean_standard_error": standard_error,
        "target_mass_lower": float(min(lower, upper)),
        "target_mass_upper": float(max(lower, upper)),
        "confidence_z": float(z),
    }


def _probe_trigger(
    counts: Sequence[int],
    plan: PrimeProbePlan,
    *,
    uniform_target_mass: float,
    z: float,
) -> dict[str, Any]:
    interval = _local_mass_interval(counts, plan, z=z)
    triggered = interval["target_mass_lower"] > uniform_target_mass
    return {
        "triggered": bool(triggered),
        "decision_rule": (
            "exploit iff the lower endpoint of the count-calibrated target-"
            "mass interval exceeds the topology-only uniform target mass"
        ),
        "uniform_target_mass": float(uniform_target_mass),
        "local_interval": interval,
    }


def _baseline_envelope(
    trajectory: Sequence[Mapping[str, Any]],
    schedule: PrimeProbeSchedule,
) -> dict[str, Any]:
    start = schedule.warmup_turns - schedule.baseline_tail_turns
    rows = [
        row for row in trajectory
        if start < int(row["turn"]) <= schedule.warmup_turns
    ]
    if not rows:
        raise ValueError("trajectory has no late-warmup baseline rows")
    policies = [np.asarray(row["_alice_policy"], dtype=float) for row in rows]
    mean_policy = np.mean(policies, axis=0)
    mean_policy /= mean_policy.sum()
    policy_tv = np.asarray([
        _total_variation(policy, mean_policy) for policy in policies
    ])
    service = np.asarray([float(row["rolling_alice_reward"]) for row in rows])
    target = np.asarray([
        float(row["alice_target_family_policy_mass"]) for row in rows
    ])
    return {
        "source": "late-warmup recorded rolling windows",
        "record_count": len(rows),
        "service_lower_2p5": float(np.quantile(service, 0.025)),
        "target_mass_lower_2p5": float(np.quantile(target, 0.025)),
        "target_mass_upper_97p5": float(np.quantile(target, 0.975)),
        "target_mass_mean": float(np.mean(target)),
        "policy_tv_upper_97p5": float(np.quantile(policy_tv, 0.975)),
        "baseline_mean_policy": mean_policy.tolist(),
    }


def _recovery_telemetry(
    trajectory: Sequence[Mapping[str, Any]],
    schedule: PrimeProbeSchedule,
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [
        row for row in trajectory if int(row["turn"]) > schedule.recovery_start
    ]
    criteria = {
        "service": lambda row: float(row["rolling_alice_reward"])
        >= float(baseline["service_lower_2p5"]),
        "target_mass": lambda row: (
            float(baseline["target_mass_lower_2p5"])
            <= float(row["alice_target_family_policy_mass"])
            <= float(baseline["target_mass_upper_97p5"])
        ),
        "policy_tv": lambda row: (
            row["alice_policy_tv_from_warmup"] is not None
            and float(row["alice_policy_tv_from_warmup"])
            <= float(baseline["policy_tv_upper_97p5"])
        ),
    }
    result: dict[str, Any] = {
        "recovery_window_starts_after_turn": schedule.recovery_start,
        "sustained_checkpoints_required": schedule.sustained_recovery_checkpoints,
    }
    for name, criterion in criteria.items():
        result[name] = _first_sustained_recovery(
            rows,
            criterion,
            start=schedule.recovery_start,
            required=schedule.sustained_recovery_checkpoints,
            censor=schedule.recovery_turns,
        )
    result["joint"] = _first_sustained_recovery(
        rows,
        lambda row: all(criterion(row) for criterion in criteria.values()),
        start=schedule.recovery_start,
        required=schedule.sustained_recovery_checkpoints,
        censor=schedule.recovery_turns,
    )
    return result


def _first_sustained_recovery(
    rows: Sequence[Mapping[str, Any]],
    criterion: Any,
    *,
    start: int,
    required: int,
    censor: int,
) -> dict[str, Any]:
    streak = 0
    first_turn: int | None = None
    for row in rows:
        if criterion(row):
            streak += 1
            if streak == 1:
                first_turn = int(row["turn"])
            if streak >= required:
                assert first_turn is not None
                return {
                    "recovered": True,
                    "latency_turns": first_turn - start,
                    "right_censored": False,
                }
        else:
            streak = 0
            first_turn = None
    return {
        "recovered": False,
        "latency_turns": int(censor),
        "right_censored": True,
    }


def _total_variation(left: np.ndarray, right: np.ndarray | None) -> float:
    if right is None:
        raise ValueError("right policy is unavailable")
    return float(0.5 * np.sum(np.abs(np.asarray(left) - np.asarray(right))))


def _edge_label(left: str, right: str) -> str:
    first, second = sorted((str(left), str(right)))
    return f"{first}-{second}"
