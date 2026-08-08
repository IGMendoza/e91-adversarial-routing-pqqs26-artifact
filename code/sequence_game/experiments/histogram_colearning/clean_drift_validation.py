"""Clean commissioning and source-uncertainty drift validation.

The attack study freezes a route-conditioned endpoint e-process at a
1,200-request observation block.  This module validates that detector under
clean operation without changing its alarm threshold, horizon, channel set,
or commissioning pseudocount.

Two source-reported uncertainties in the active Cui rolling composite define
the physical stress axes: protected-memory Gaussian coherence and local
Bell-transform fidelity.  Stationary profiles are commissioned and monitored
at the same operating point.  Stale-commissioning scenarios intentionally
keep the nominal commissioning model while the clean physical profile differs
or changes during monitoring.  The latter are nuisance-drift alarms, not
stationary false alarms.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import beta as beta_distribution

from .config import ALICE_LEARNER_TSALLIS_INF, HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .learner_protocol import PlayerLearner, policy_distribution_statistics
from .learning import (
    bell_qualified_rate_utility,
    build_histogram_alice_learner,
)
from .physics import HistogramPhysicsPool
from .quantum_stackelberg_detection import (
    BinomialEvidenceDetectorConfig,
    MatchedClassicalQuantumEndpointDetectors,
    QuantumEndpointEvidence,
)


CLEAN_DRIFT_STUDY_SCHEMA = "histogram_clean_commissioning_drift_study.v1"
CLEAN_DRIFT_ROW_SCHEMA = "histogram_clean_drift_trajectory.v1"
CLEAN_DRIFT_SUMMARY_SCHEMA = "histogram_clean_drift_summary.v1"

NOMINAL_PROFILE = "nominal"
MEMORY_MINUS_PROFILE = "memory_coherence_minus_1sigma"
MEMORY_PLUS_PROFILE = "memory_coherence_plus_1sigma"
BSM_MINUS_PROFILE = "local_bsm_fidelity_minus_1sigma"
BSM_PLUS_PROFILE = "local_bsm_fidelity_plus_1sigma"

STATIONARY_RECOMMISSIONED = "stationary_recommissioned"
STALE_STATIC_MISMATCH = "stale_commissioning_static_mismatch"
STALE_CHANGE_POINT = "stale_commissioning_change_point"

DETECTOR_FIELDS = {
    "classical_delivery_only": "delivery_only",
    "classical_service_baseline": "classical_service",
    "quantum_endpoint_integrity": "quantum_endpoint",
}


@dataclass(frozen=True)
class PhysicalDriftProfile:
    """One source-grounded clean operating-point profile."""

    profile_id: str
    parameter_path: tuple[str, ...] | None
    nominal_value: float | None
    uncertainty: float | None
    sigma_offset: float
    resolved_value: float | None
    units: str | None
    source_semantics: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["parameter_path"] = (
            list(self.parameter_path)
            if self.parameter_path is not None
            else None
        )
        return payload


@dataclass(frozen=True)
class CleanDriftScenario:
    """Commissioning/monitoring relationship for a clean trajectory."""

    scenario_id: str
    classification: str
    commissioning_profile_id: str
    initial_monitoring_profile_id: str
    changed_monitoring_profile_id: str | None = None
    change_turn: int | None = None

    def __post_init__(self) -> None:
        if self.classification not in {
            STATIONARY_RECOMMISSIONED,
            STALE_STATIC_MISMATCH,
            STALE_CHANGE_POINT,
        }:
            raise ValueError("unsupported clean drift scenario classification")
        if self.classification == STATIONARY_RECOMMISSIONED:
            if (
                self.commissioning_profile_id
                != self.initial_monitoring_profile_id
                or self.changed_monitoring_profile_id is not None
                or self.change_turn is not None
            ):
                raise ValueError(
                    "stationary scenarios must commission and monitor one profile"
                )
        elif self.classification == STALE_STATIC_MISMATCH:
            if (
                self.commissioning_profile_id
                == self.initial_monitoring_profile_id
                or self.changed_monitoring_profile_id is not None
                or self.change_turn is not None
            ):
                raise ValueError(
                    "static mismatch requires distinct commissioning/monitor profiles"
                )
        else:
            if (
                self.changed_monitoring_profile_id is None
                or self.change_turn is None
                or self.change_turn < 1
                or self.commissioning_profile_id
                != self.initial_monitoring_profile_id
                or self.changed_monitoring_profile_id
                == self.initial_monitoring_profile_id
            ):
                raise ValueError("invalid clean change-point scenario")

    @property
    def alarm_semantics(self) -> str:
        if self.classification == STATIONARY_RECOMMISSIONED:
            return "clean_stationary_false_alarm"
        return "clean_nuisance_drift_alarm"

    @property
    def required_profile_ids(self) -> tuple[str, ...]:
        values = {
            self.commissioning_profile_id,
            self.initial_monitoring_profile_id,
        }
        if self.changed_monitoring_profile_id is not None:
            values.add(self.changed_monitoring_profile_id)
        return tuple(sorted(values))

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "alarm_semantics": self.alarm_semantics,
            "required_profile_ids": list(self.required_profile_ids),
        }


@dataclass(frozen=True)
class CleanDriftValidationConfig:
    """Frozen detector and replication contract for the clean study."""

    requested_pairs_per_turn: int = 1_200
    horizon_turns: int = 4_000
    commissioning_draws_per_route: int = 32
    replicates_per_case_scenario: int = 50
    detector_false_alarm_level: float = 0.01
    detector_commissioning_pseudocount: float = 0.5
    detector_change_fractions: tuple[float, ...] = (0.10, 0.25, 0.50)
    change_point_turn: int = 2_001
    root_seed: int = 2_026_072_301
    workers: int = 3

    def __post_init__(self) -> None:
        if self.requested_pairs_per_turn < 1:
            raise ValueError("requested_pairs_per_turn must be positive")
        if self.horizon_turns < 2:
            raise ValueError("horizon_turns must be at least two")
        if self.commissioning_draws_per_route < 2:
            raise ValueError(
                "commissioning_draws_per_route must be at least two"
            )
        if self.replicates_per_case_scenario < 1:
            raise ValueError("replicates must be positive")
        if not 1 < self.change_point_turn <= self.horizon_turns:
            raise ValueError("change_point_turn must lie inside the horizon")
        if self.workers < 1:
            raise ValueError("workers must be positive")
        BinomialEvidenceDetectorConfig(
            horizon=self.horizon_turns,
            model_false_alarm_level=self.detector_false_alarm_level,
            commissioning_pseudocount=(
                self.detector_commissioning_pseudocount
            ),
            change_fractions=self.detector_change_fractions,
        )

    @property
    def detector_config(self) -> BinomialEvidenceDetectorConfig:
        return BinomialEvidenceDetectorConfig(
            horizon=self.horizon_turns,
            model_false_alarm_level=self.detector_false_alarm_level,
            commissioning_pseudocount=(
                self.detector_commissioning_pseudocount
            ),
            change_fractions=self.detector_change_fractions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "detector_change_fractions": list(
                map(float, self.detector_change_fractions)
            ),
            "detector_config": self.detector_config.to_dict(),
            "route_schedule": (
                "fresh uniformly initialized native reduced-variance "
                "Tsallis-INF updated from selected-route clean public utility"
            ),
        }


def default_physical_drift_profiles(
    base_profile: Mapping[str, Any],
) -> tuple[PhysicalDriftProfile, ...]:
    """Resolve the two registered one-sigma source uncertainties."""

    memory = base_profile["memory_model"]["storage_decoherence_model"]
    local_bsm = base_profile["state_model"]["local_swap_model"][
        "bell_basis_transform_model"
    ]
    memory_nominal = float(memory["coherence_time_s"])
    memory_uncertainty = float(memory["coherence_time_uncertainty_s"])
    bsm_nominal = float(local_bsm["fidelity"])
    bsm_uncertainty = float(local_bsm["uncertainty"])
    return (
        PhysicalDriftProfile(
            profile_id=NOMINAL_PROFILE,
            parameter_path=None,
            nominal_value=None,
            uncertainty=None,
            sigma_offset=0.0,
            resolved_value=None,
            units=None,
            source_semantics=(
                "unchanged active Cui rolling purified near-term composite"
            ),
        ),
        PhysicalDriftProfile(
            profile_id=MEMORY_MINUS_PROFILE,
            parameter_path=(
                "memory_model",
                "storage_decoherence_model",
                "coherence_time_s",
            ),
            nominal_value=memory_nominal,
            uncertainty=memory_uncertainty,
            sigma_offset=-1.0,
            resolved_value=memory_nominal - memory_uncertainty,
            units="s",
            source_semantics=(
                "Cui protected-memory Ramsey-with-echo Gaussian fit, "
                "366 +/- 11 ms"
            ),
        ),
        PhysicalDriftProfile(
            profile_id=MEMORY_PLUS_PROFILE,
            parameter_path=(
                "memory_model",
                "storage_decoherence_model",
                "coherence_time_s",
            ),
            nominal_value=memory_nominal,
            uncertainty=memory_uncertainty,
            sigma_offset=1.0,
            resolved_value=memory_nominal + memory_uncertainty,
            units="s",
            source_semantics=(
                "Cui protected-memory Ramsey-with-echo Gaussian fit, "
                "366 +/- 11 ms"
            ),
        ),
        PhysicalDriftProfile(
            profile_id=BSM_MINUS_PROFILE,
            parameter_path=(
                "state_model",
                "local_swap_model",
                "bell_basis_transform_model",
                "fidelity",
            ),
            nominal_value=bsm_nominal,
            uncertainty=bsm_uncertainty,
            sigma_offset=-1.0,
            resolved_value=bsm_nominal - bsm_uncertainty,
            units="probability",
            source_semantics=(
                "Clark uncorrected Bell-preparation reciprocal transform "
                "envelope, 0.9994 +/- 0.0003"
            ),
        ),
        PhysicalDriftProfile(
            profile_id=BSM_PLUS_PROFILE,
            parameter_path=(
                "state_model",
                "local_swap_model",
                "bell_basis_transform_model",
                "fidelity",
            ),
            nominal_value=bsm_nominal,
            uncertainty=bsm_uncertainty,
            sigma_offset=1.0,
            resolved_value=bsm_nominal + bsm_uncertainty,
            units="probability",
            source_semantics=(
                "Clark uncorrected Bell-preparation reciprocal transform "
                "envelope, 0.9994 +/- 0.0003"
            ),
        ),
    )


def materialize_drift_profile(
    base_profile: Mapping[str, Any],
    drift: PhysicalDriftProfile,
) -> dict[str, Any]:
    """Create one validated-profile payload without mutating the source."""

    payload = deepcopy(dict(base_profile))
    base_id = str(payload["profile_id"])
    if drift.parameter_path is not None:
        cursor: dict[str, Any] = payload
        for key in drift.parameter_path[:-1]:
            cursor = cursor[key]
        leaf = drift.parameter_path[-1]
        observed = float(cursor[leaf])
        if not math.isclose(
            observed,
            float(drift.nominal_value),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError(
                f"base profile value changed at {drift.parameter_path}: "
                f"{observed} != {drift.nominal_value}"
            )
        cursor[leaf] = float(drift.resolved_value)
    payload["profile_id"] = f"{base_id}__clean_drift__{drift.profile_id}"
    payload["validation_stress"] = {
        "scope": "source_uncertainty_clean_drift_validation",
        **drift.to_dict(),
        "attack_enabled": False,
        "composition_boundary": (
            "one profile field is shifted while all other near-term "
            "composite components remain fixed"
        ),
    }
    payload["notes"] = (
        str(payload.get("notes", ""))
        + " Clean validation derivative; see validation_stress."
    ).strip()
    return payload


def default_clean_drift_scenarios(
    *,
    change_turn: int,
) -> tuple[CleanDriftScenario, ...]:
    """Return the frozen stationary, mismatch, and change-point matrix."""

    drift_ids = (
        MEMORY_MINUS_PROFILE,
        MEMORY_PLUS_PROFILE,
        BSM_MINUS_PROFILE,
        BSM_PLUS_PROFILE,
    )
    rows = [
        CleanDriftScenario(
            scenario_id="nominal_stationary",
            classification=STATIONARY_RECOMMISSIONED,
            commissioning_profile_id=NOMINAL_PROFILE,
            initial_monitoring_profile_id=NOMINAL_PROFILE,
        )
    ]
    for profile_id in drift_ids:
        rows.append(CleanDriftScenario(
            scenario_id=f"{profile_id}__recommissioned",
            classification=STATIONARY_RECOMMISSIONED,
            commissioning_profile_id=profile_id,
            initial_monitoring_profile_id=profile_id,
        ))
    for profile_id in drift_ids:
        rows.append(CleanDriftScenario(
            scenario_id=f"nominal_to_{profile_id}__static",
            classification=STALE_STATIC_MISMATCH,
            commissioning_profile_id=NOMINAL_PROFILE,
            initial_monitoring_profile_id=profile_id,
        ))
    for profile_id in (MEMORY_MINUS_PROFILE, BSM_MINUS_PROFILE):
        rows.append(CleanDriftScenario(
            scenario_id=f"nominal_to_{profile_id}__change_point",
            classification=STALE_CHANGE_POINT,
            commissioning_profile_id=NOMINAL_PROFILE,
            initial_monitoring_profile_id=NOMINAL_PROFILE,
            changed_monitoring_profile_id=profile_id,
            change_turn=int(change_turn),
        ))
    return tuple(rows)


def run_clean_drift_trajectory(
    case: HistogramGameCase,
    pools: Mapping[str, HistogramPhysicsPool],
    profile_paths: Mapping[str, Path],
    scenario: CleanDriftScenario,
    study_config: CleanDriftValidationConfig,
    *,
    partition: str,
    case_id: str,
    replicate: int,
    trajectory_seed: int,
) -> dict[str, Any]:
    """Run one fresh-commissioning, clean native-Tsallis trajectory."""

    missing = set(scenario.required_profile_ids) - set(pools)
    if missing:
        raise ValueError(f"missing physics pools for {sorted(missing)}")
    actions = case.actions("attack_surface_all")
    no_attack = next(
        action for action in actions if str(action["id"]) == "no_attack"
    )
    commissioning_pool = pools[scenario.commissioning_profile_id]
    commissioning = _sample_commissioning(
        commissioning_pool,
        no_attack,
        route_count=len(case.routes),
        draws_per_route=study_config.commissioning_draws_per_route,
        seed=_stable_seed(trajectory_seed, "commissioning"),
    )
    detectors = MatchedClassicalQuantumEndpointDetectors.from_commissioning(
        commissioning,
        study_config.detector_config,
    )

    initial_pool = pools[scenario.initial_monitoring_profile_id]
    learner_config = replace(
        HistogramCoLearningConfig(),
        turns=study_config.horizon_turns,
        final_window=min(1_000, study_config.horizon_turns),
        record_every=max(1, study_config.horizon_turns),
        seed=int(trajectory_seed),
        requested_pairs_per_turn=study_config.requested_pairs_per_turn,
        alice_learner_mode=ALICE_LEARNER_TSALLIS_INF,
        repeater_service_profile_path=profile_paths[
            scenario.initial_monitoring_profile_id
        ],
    )
    implementation, metadata = build_histogram_alice_learner(
        case=case,
        config=learner_config,
        prior=np.full(len(case.routes), 1.0 / len(case.routes), dtype=float),
    )
    alice = PlayerLearner(
        player="alice",
        action_ids=tuple(str(route["route_id"]) for route in case.routes),
        learner=implementation,
        metadata=metadata,
    )

    route_counts = np.zeros(len(case.routes), dtype=np.int64)
    public_outcomes: Counter[str] = Counter()
    delivered_total = 0
    requested_total = 0
    key_trials = 0
    key_errors = 0
    chsh_trials = np.zeros(4, dtype=np.int64)
    chsh_same = np.zeros(4, dtype=np.int64)
    reward_sum = 0.0
    monitor_seed = _stable_seed(trajectory_seed, "monitoring")
    for turn_index in range(study_config.horizon_turns):
        step = turn_index + 1
        profile_id = scenario.initial_monitoring_profile_id
        if (
            scenario.change_turn is not None
            and step >= scenario.change_turn
        ):
            profile_id = str(scenario.changed_monitoring_profile_id)
        active_pool = pools[profile_id]
        route_index = alice.sample()
        result = active_pool.step(
            route_index,
            no_attack,
            seed=monitor_seed,
            step_index=turn_index,
        )
        utility = bell_qualified_rate_utility(
            result=result,
            calibration=active_pool.calibrations[route_index],
            reference_service_rate_hz=active_pool.reference_service_rate_hz,
        )
        alice.update(utility.reward)
        evidence = QuantumEndpointEvidence.from_step_result(result)
        detectors.update(route_index, evidence)

        route_counts[route_index] += 1
        public_outcomes[str(result.public_outcome)] += 1
        requested_total += int(result.requested_pair_count)
        delivered_total += int(result.delivered_pair_count)
        key_trials += int(result.key_pair_count)
        key_errors += int(result.key_error_count)
        chsh_trials += np.asarray(
            evidence.chsh_setting_total_counts,
            dtype=np.int64,
        )
        chsh_same += np.asarray(
            evidence.chsh_setting_same_counts,
            dtype=np.int64,
        )
        reward_sum += float(utility.reward)

    diagnostics = detectors.diagnostics()
    detector_rows: dict[str, Any] = {}
    for diagnostics_key, output_key in DETECTOR_FIELDS.items():
        row = diagnostics[diagnostics_key]
        detector_rows[output_key] = {
            "alarmed": bool(row["alarmed"]),
            "alarm_turn": row["alarm_turn"],
            "maximum_log_e_value": float(row["maximum_log_e_value"]),
            "threshold_log_e_value": float(
                -math.log(study_config.detector_false_alarm_level)
            ),
            "leading_alternative": row["leading_alternative"],
        }
    correlations = np.divide(
        2.0 * chsh_same - chsh_trials,
        chsh_trials,
        out=np.full(4, np.nan, dtype=float),
        where=chsh_trials > 0,
    )
    chsh_s = (
        -float(correlations[0] + correlations[1] + correlations[2] - correlations[3])
        if np.all(np.isfinite(correlations))
        else None
    )
    return {
        "schema": CLEAN_DRIFT_ROW_SCHEMA,
        "partition": str(partition),
        "case_id": str(case_id),
        "family": str(case.family),
        "scenario_id": scenario.scenario_id,
        "scenario_classification": scenario.classification,
        "alarm_semantics": scenario.alarm_semantics,
        "change_turn": scenario.change_turn,
        "replicate": int(replicate),
        "trajectory_seed": int(trajectory_seed),
        "requested_pairs_per_turn": study_config.requested_pairs_per_turn,
        "horizon_turns": study_config.horizon_turns,
        "commissioning_draws_per_route": (
            study_config.commissioning_draws_per_route
        ),
        "detectors": detector_rows,
        "mean_alice_reward": reward_sum / study_config.horizon_turns,
        "public_outcome_counts": dict(public_outcomes),
        "physical_delivery_fraction": (
            delivered_total / requested_total if requested_total else None
        ),
        "aggregate_qber": (
            key_errors / key_trials if key_trials else None
        ),
        "aggregate_chsh_s": chsh_s,
        "key_trials": int(key_trials),
        "key_errors": int(key_errors),
        "chsh_setting_trials": chsh_trials.tolist(),
        "chsh_setting_same": chsh_same.tolist(),
        "route_counts": route_counts.tolist(),
        "final_policy_statistics": policy_distribution_statistics(
            alice.policy(turn=study_config.horizon_turns + 1)
        ),
        "attack_enabled": False,
        "observation_contract": (
            "selected route plus endpoint delivery/public-service counts; "
            "quantum arm additionally receives key-error and per-setting "
            "CHSH same/different counts"
        ),
    }


def aggregate_clean_drift_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate alarm incidence with pooled exact and topology spread."""

    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        for detector in DETECTOR_FIELDS.values():
            key = (
                str(row["scenario_id"]),
                str(row["alarm_semantics"]),
                detector,
            )
            grouped.setdefault(key, []).append(row)
    aggregates: list[dict[str, Any]] = []
    for (scenario_id, alarm_semantics, detector), cell in sorted(
        grouped.items()
    ):
        alarmed = np.asarray([
            bool(row["detectors"][detector]["alarmed"])
            for row in cell
        ], dtype=bool)
        alarm_turns = np.asarray([
            int(row["detectors"][detector]["alarm_turn"])
            for row in cell
            if row["detectors"][detector]["alarm_turn"] is not None
        ], dtype=float)
        maximum_log_e_values = np.asarray([
            float(row["detectors"][detector]["maximum_log_e_value"])
            for row in cell
        ], dtype=float)
        leading_counts: Counter[str] = Counter()
        for row in cell:
            leading = row["detectors"][detector]["leading_alternative"]
            if leading is None:
                leading_counts["none"] += 1
            else:
                leading_counts[
                    f"{leading['channel']}|{leading['direction']}|"
                    f"{float(leading['change_fraction']):g}"
                ] += 1
        change_turns = {
            row.get("change_turn") for row in cell
        }
        if len(change_turns) != 1:
            raise ValueError("aggregate cell mixes change-point contracts")
        change_turn = next(iter(change_turns))
        pre_change_alarm_count = None
        post_change_alarm_count = None
        post_change_delays: list[int] = []
        if change_turn is not None:
            pre_change_alarm_count = 0
            post_change_alarm_count = 0
            for row in cell:
                turn = row["detectors"][detector]["alarm_turn"]
                if turn is None:
                    continue
                if int(turn) < int(change_turn):
                    pre_change_alarm_count += 1
                else:
                    post_change_alarm_count += 1
                    post_change_delays.append(
                        int(turn) - int(change_turn) + 1
                    )
        topology_rates: dict[str, float] = {}
        for case_id in sorted({str(row["case_id"]) for row in cell}):
            selected = [
                row for row in cell if str(row["case_id"]) == case_id
            ]
            topology_rates[case_id] = float(np.mean([
                bool(row["detectors"][detector]["alarmed"])
                for row in selected
            ]))
        topology_values = np.asarray(
            list(topology_rates.values()),
            dtype=float,
        )
        lower, upper = clopper_pearson_interval(
            int(np.sum(alarmed)),
            int(alarmed.size),
        )
        aggregates.append({
            "scenario_id": scenario_id,
            "alarm_semantics": alarm_semantics,
            "detector": detector,
            "trajectory_count": int(alarmed.size),
            "alarm_count": int(np.sum(alarmed)),
            "pooled_alarm_rate": float(np.mean(alarmed)),
            "pooled_clopper_pearson_95": [lower, upper],
            "mean_alarm_turn_given_alarm": (
                float(np.mean(alarm_turns))
                if alarm_turns.size
                else None
            ),
            "median_alarm_turn_given_alarm": (
                float(np.median(alarm_turns))
                if alarm_turns.size
                else None
            ),
            "maximum_log_e_value_quantiles": {
                "q50": float(np.quantile(maximum_log_e_values, 0.50)),
                "q90": float(np.quantile(maximum_log_e_values, 0.90)),
                "q95": float(np.quantile(maximum_log_e_values, 0.95)),
                "q99": float(np.quantile(maximum_log_e_values, 0.99)),
                "maximum": float(np.max(maximum_log_e_values)),
            },
            "final_leading_alternative_counts": dict(leading_counts),
            "change_turn": change_turn,
            "pre_change_alarm_count": pre_change_alarm_count,
            "post_change_alarm_count": post_change_alarm_count,
            "mean_changed_profile_blocks_to_alarm": (
                float(np.mean(post_change_delays))
                if post_change_delays
                else None
            ),
            "topology_alarm_rates": topology_rates,
            "topology_macro_alarm_rate": float(np.mean(topology_values)),
            "topology_standard_error": (
                float(np.std(topology_values, ddof=1) / np.sqrt(topology_values.size))
                if topology_values.size > 1
                else 0.0
            ),
        })
    return aggregates


def clopper_pearson_interval(
    successes: int,
    trials: int,
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Two-sided exact binomial interval, including zero/all boundaries."""

    k = int(successes)
    n = int(trials)
    if n < 1 or not 0 <= k <= n:
        raise ValueError("successes/trials are inconsistent")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if k == 0 else float(
        beta_distribution.ppf(tail, k, n - k + 1)
    )
    upper = 1.0 if k == n else float(
        beta_distribution.ppf(1.0 - tail, k + 1, n - k)
    )
    return lower, upper


def freeze_digest(payload: Mapping[str, Any]) -> str:
    """Stable 128-bit identity for a clean-study preregistration."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _sample_commissioning(
    pool: HistogramPhysicsPool,
    no_attack: Mapping[str, Any],
    *,
    route_count: int,
    draws_per_route: int,
    seed: int,
) -> tuple[tuple[QuantumEndpointEvidence, ...], ...]:
    rows = []
    for route_index in range(route_count):
        evidence = []
        for draw in range(draws_per_route):
            result = pool.step(
                route_index,
                dict(no_attack),
                seed=int(seed),
                step_index=route_index * draws_per_route + draw,
            )
            evidence.append(QuantumEndpointEvidence.from_step_result(result))
        rows.append(tuple(evidence))
    return tuple(rows)


def _stable_seed(root_seed: int, *parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(root_seed).to_bytes(8, "little", signed=True))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") % (2**31 - 1)


def trajectory_seed(
    root_seed: int,
    case_id: str,
    replicate: int,
) -> int:
    """Pair scenarios within a case while separating cases and replicates."""

    return _stable_seed(root_seed, case_id, int(replicate), "trajectory")
