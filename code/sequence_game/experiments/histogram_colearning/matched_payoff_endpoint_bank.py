"""Matched payoff and Alice endpoint-evidence banks.

Each feedback/audit route--action--draw coordinate is sampled from compiled
histogram physics exactly once.  The resulting transcript is then projected
into two separate objects:

* ``HistogramPayoffBank`` retains the scalar game quantities and Eve's local
  observation count used by the existing game pipeline.
* ``HistogramEndpointEvidenceBank`` retains only Alice/Bob endpoint counts and
  public service quantities used by endpoint detectors.

The two projections share coordinates and provenance, but not array storage or
observation fields.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import copy
import hashlib
import json
import time
import threading
from typing import Any, Callable

import numpy as np

from sequence_game.corpus.compiled_histogram_backend import (
    COMPILED_HISTOGRAM_RUNTIME_VERSION,
)

from .calibration_bank import (
    PAYOFF_BANK_SPLITS,
    PUBLIC_OUTCOME_CODES,
    SIGNAL_CLASS_CODES,
    HistogramPayoffBank,
    PayoffSampleBlock,
)
from .calibration_cases import STEP6_SELECTION_DIGEST
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .endpoint_evidence_bank import (
    EndpointEvidenceSampleBlock,
    HistogramEndpointEvidenceBank,
)
from .learning import (
    assess_alice_public_signal,
    bell_qualified_rate_utility,
    normalize_eve_bandit_reward,
)
from .physics import HistogramPhysicsPool
from .quantum_stackelberg_detection import (
    CHSH_SETTING_LABELS,
    QuantumEndpointEvidence,
)


MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA = (
    "histogram_matched_payoff_endpoint_banks.v1"
)


def build_matched_histogram_payoff_and_endpoint_banks(
    case: HistogramGameCase,
    config: HistogramCoLearningConfig,
    *,
    samples_per_cell: int = 64,
    root_seed: int = 20_260_717,
    physics: HistogramPhysicsPool | None = None,
    progress_callback: Callable[[str], None] | None = None,
    selection_digest_blake2b_128: str = STEP6_SELECTION_DIGEST,
    evaluation_partition_loaded: bool = False,
    sampling_purpose: str = "matched_learning_and_endpoint_detection",
    route_workers: int = 1,
) -> tuple[HistogramPayoffBank, HistogramEndpointEvidenceBank]:
    """Build aligned payoff and endpoint banks from one sample per coordinate.

    ``feedback`` and ``audit`` use different deterministic seeds.  Within one
    split, every route--action--draw coordinate produces one
    :meth:`HistogramPhysicsPool.step` result; both returned banks are populated
    from that same result.
    """

    if samples_per_cell < 2:
        raise ValueError("samples_per_cell must be at least two")
    if route_workers < 1:
        raise ValueError("route_workers must be positive")
    started = time.perf_counter()
    active_physics = physics or HistogramPhysicsPool(case, config)
    calibrations = active_physics.calibrations
    actions = case.actions(config.game_mode)
    if len(calibrations) != len(case.routes) or not actions:
        raise ValueError("physics routes or Eve actions are inconsistent")

    route_ids = tuple(str(route["route_id"]) for route in case.routes)
    action_ids = tuple(str(action["id"]) for action in actions)
    action_costs = np.asarray(
        [float(action.get("cost", 0.0)) for action in actions],
        dtype=float,
    )
    maximum_cost = float(action_costs.max(initial=0.0))
    reference_rate = active_physics.reference_service_rate_hz
    shape = (len(route_ids), len(action_ids), int(samples_per_cell))
    split_seeds = {
        split: _stable_seed(
            root_seed,
            case.graph_id,
            case.assignment_id,
            MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA,
            split,
        )
        for split in PAYOFF_BANK_SPLITS
    }
    if len(set(split_seeds.values())) != len(PAYOFF_BANK_SPLITS):
        raise RuntimeError("matched feedback and audit seeds collided")

    payoff_blocks: dict[str, PayoffSampleBlock] = {}
    endpoint_blocks: dict[str, EndpointEvidenceSampleBlock] = {}
    for split in PAYOFF_BANK_SPLITS:
        seed = split_seeds[split]
        payoff_arrays = _empty_payoff_arrays(shape)
        endpoint_arrays = _empty_endpoint_arrays(shape)
        total = int(np.prod(shape))
        completed = 0
        progress_stride = max(samples_per_cell, total // 20)
        next_progress = progress_stride
        progress_lock = threading.Lock()

        def sample_route(route_row: tuple[int, Any]) -> None:
            nonlocal completed, next_progress
            route_index, calibration = route_row
            route_completed = 0
            for action_index, action in enumerate(actions):
                for draw_index in range(samples_per_cell):
                    step_index = (
                        (
                            route_index * len(actions)
                            + action_index
                        )
                        * samples_per_cell
                        + draw_index
                    )
                    result = active_physics.step(
                        route_index,
                        action,
                        seed=seed,
                        step_index=step_index,
                    )
                    utility = bell_qualified_rate_utility(
                        result=result,
                        calibration=calibration,
                        reference_service_rate_hz=reference_rate,
                    )
                    signal = assess_alice_public_signal(result)
                    corrected = bell_qualified_rate_utility(
                        result=result,
                        calibration=calibration,
                        reference_service_rate_hz=reference_rate,
                        bell_qualified_override=signal.latent_bell_qualified,
                    )
                    raw_eve_reward = (
                        1.0
                        - utility.reward
                        - action_costs[action_index]
                    )
                    eve_reward = normalize_eve_bandit_reward(
                        raw_eve_reward,
                        maximum_cost=maximum_cost,
                    )
                    endpoint_evidence = (
                        QuantumEndpointEvidence.from_step_result(result)
                    )
                    if endpoint_evidence.requested_pair_count != int(
                        config.requested_pairs_per_turn
                    ):
                        raise ValueError(
                            "compiled transcript requested-pair count differs "
                            "from the experiment configuration"
                        )

                    index = (route_index, action_index, draw_index)
                    _store_payoff_sample(
                        payoff_arrays,
                        index=index,
                        result=result,
                        utility=utility,
                        corrected_reward=corrected.reward,
                        eve_reward=eve_reward,
                        signal=signal,
                    )
                    _store_endpoint_sample(
                        endpoint_arrays,
                        index=index,
                        evidence=endpoint_evidence,
                        utility=utility,
                        public_outcome=str(result.public_outcome),
                    )
                    route_completed += 1
            with progress_lock:
                completed += route_completed
                if progress_callback is not None and (
                    completed >= next_progress or completed == total
                ):
                    progress_callback(
                        f"{case.family} matched {split}: "
                        f"{completed}/{total} draws"
                    )
                    while next_progress <= completed:
                        next_progress += progress_stride

        route_rows = tuple(enumerate(calibrations))
        if route_workers == 1:
            for route_row in route_rows:
                sample_route(route_row)
        else:
            with ThreadPoolExecutor(
                max_workers=min(int(route_workers), len(route_rows))
            ) as executor:
                tuple(executor.map(sample_route, route_rows))
        payoff_blocks[split] = PayoffSampleBlock(**payoff_arrays)
        endpoint_blocks[split] = EndpointEvidenceSampleBlock(
            **endpoint_arrays
        )

    sampling_id = _matched_sampling_id(
        case=case,
        config=config,
        route_ids=route_ids,
        action_ids=action_ids,
        samples_per_cell=samples_per_cell,
        root_seed=root_seed,
        split_seeds=split_seeds,
        selection_digest_blake2b_128=selection_digest_blake2b_128,
    )
    elapsed = time.perf_counter() - started
    physics_diagnostics = active_physics.diagnostics()
    common_metadata: dict[str, Any] = {
        "selection_digest_blake2b_128": str(
            selection_digest_blake2b_128
        ),
        "sampling_design": (
            "one compiled transcript per split/route/action/draw coordinate; "
            "the result is projected into separate payoff and Alice endpoint "
            "evidence banks; feedback and audit use disjoint seeds"
        ),
        "sampling_purpose": str(sampling_purpose),
        "matched_sampling_schema": MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA,
        "matched_sampling_id_blake2b_128": sampling_id,
        "matched_sampling_step_calls_per_coordinate": 1,
        "matched_sampling_shared_fields": [
            "alice_reward",
            "bell_qualified_rate_hz",
            "public_outcome_code",
        ],
        "samples_per_cell_per_split": int(samples_per_cell),
        "root_seed": int(root_seed),
        "split_seeds": dict(split_seeds),
        "game_mode": config.game_mode,
        "requested_pairs_per_draw": int(
            config.requested_pairs_per_turn
        ),
        "compiled_histogram_runtime_version": (
            COMPILED_HISTOGRAM_RUNTIME_VERSION
        ),
        "feedback_audit_seed_disjoint": True,
        "evaluation_partition_loaded": bool(evaluation_partition_loaded),
        "physics_config": _physics_config(config),
        "physics": physics_diagnostics,
        "elapsed_seconds": elapsed,
    }
    payoff_metadata = copy.deepcopy(common_metadata)
    payoff_metadata.update(
        {
            "matched_sampling_role": "game_payoff",
            "eve_observation_count_semantics": (
                "physical_count_per_transcript"
            ),
        }
    )
    endpoint_metadata = copy.deepcopy(common_metadata)
    endpoint_metadata.update(
        {
            "matched_sampling_role": "alice_endpoint_evidence",
            "information_boundary": (
                "Alice scalar public service and endpoint E91 measurement "
                "counts only; local intervention records and target-active "
                "diagnostics are not retained"
            ),
        }
    )

    payoff_bank = HistogramPayoffBank(
        case_id=f"{case.graph_id}::{case.assignment_id}",
        graph_id=case.graph_id,
        assignment_id=case.assignment_id,
        route_ids=route_ids,
        action_ids=action_ids,
        action_costs=action_costs,
        feedback=payoff_blocks["feedback"],
        audit=payoff_blocks["audit"],
        metadata=payoff_metadata,
    )
    endpoint_bank = HistogramEndpointEvidenceBank(
        case_id=payoff_bank.case_id,
        graph_id=case.graph_id,
        assignment_id=case.assignment_id,
        route_ids=route_ids,
        action_ids=action_ids,
        requested_pairs_per_draw=int(config.requested_pairs_per_turn),
        feedback=endpoint_blocks["feedback"],
        audit=endpoint_blocks["audit"],
        metadata=endpoint_metadata,
    )
    _validate_shared_projection(payoff_bank, endpoint_bank)
    return payoff_bank, endpoint_bank


def _store_payoff_sample(
    arrays: dict[str, np.ndarray],
    *,
    index: tuple[int, int, int],
    result: Any,
    utility: Any,
    corrected_reward: float,
    eve_reward: float,
    signal: Any,
) -> None:
    arrays["alice_reward"][index] = utility.reward
    arrays["corrected_alice_reward"][index] = corrected_reward
    arrays["eve_bandit_reward"][index] = eve_reward
    arrays["bell_qualified_rate_hz"][index] = (
        utility.estimated_bell_qualified_rate_hz
    )
    arrays["false_signal"][index] = signal.false_signal
    arrays["classifiable_signal"][index] = (
        signal.classifiable_quality_signal
    )
    arrays["target_active"][index] = bool(result.target_active)
    arrays["eve_observation_count"][index] = int(
        result.eve_observation_count
    )
    arrays["information_observed"][index] = bool(
        result.eve_observation_count > 0
    )
    try:
        arrays["public_outcome_code"][index] = PUBLIC_OUTCOME_CODES[
            str(result.public_outcome)
        ]
        arrays["signal_class_code"][index] = SIGNAL_CLASS_CODES[
            signal.classification
        ]
    except KeyError as exc:
        raise ValueError(
            f"unrecognized histogram diagnostic label {exc.args[0]!r}"
        ) from exc


def _store_endpoint_sample(
    arrays: dict[str, np.ndarray],
    *,
    index: tuple[int, int, int],
    evidence: QuantumEndpointEvidence,
    utility: Any,
    public_outcome: str,
) -> None:
    arrays["alice_reward"][index] = utility.reward
    arrays["bell_qualified_rate_hz"][index] = (
        utility.estimated_bell_qualified_rate_hz
    )
    arrays["delivered_pair_count"][index] = (
        evidence.delivered_pair_count
    )
    arrays["key_pair_count"][index] = evidence.key_pair_count
    arrays["key_error_count"][index] = evidence.key_error_count
    arrays["chsh_setting_total_counts"][index] = (
        evidence.chsh_setting_total_counts
    )
    arrays["chsh_setting_same_counts"][index] = (
        evidence.chsh_setting_same_counts
    )
    try:
        arrays["public_outcome_code"][index] = PUBLIC_OUTCOME_CODES[
            public_outcome
        ]
    except KeyError as exc:
        raise ValueError(
            f"unrecognized public outcome {exc.args[0]!r}"
        ) from exc


def _validate_shared_projection(
    payoff_bank: HistogramPayoffBank,
    endpoint_bank: HistogramEndpointEvidenceBank,
) -> None:
    if (
        payoff_bank.case_id != endpoint_bank.case_id
        or payoff_bank.route_ids != endpoint_bank.route_ids
        or payoff_bank.action_ids != endpoint_bank.action_ids
    ):
        raise RuntimeError("matched payoff and endpoint bank axes differ")
    for split in PAYOFF_BANK_SPLITS:
        payoff = getattr(payoff_bank, split)
        endpoint = getattr(endpoint_bank, split)
        for field in (
            "alice_reward",
            "bell_qualified_rate_hz",
            "public_outcome_code",
        ):
            if not np.array_equal(
                np.asarray(getattr(payoff, field)),
                np.asarray(getattr(endpoint, field)),
            ):
                raise RuntimeError(
                    f"matched bank shared field {field!r} differs in {split}"
                )


def _empty_payoff_arrays(
    shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    return {
        "alice_reward": np.empty(shape, dtype=np.float64),
        "corrected_alice_reward": np.empty(shape, dtype=np.float64),
        "eve_bandit_reward": np.empty(shape, dtype=np.float64),
        "bell_qualified_rate_hz": np.empty(shape, dtype=np.float64),
        "false_signal": np.empty(shape, dtype=np.bool_),
        "classifiable_signal": np.empty(shape, dtype=np.bool_),
        "target_active": np.empty(shape, dtype=np.bool_),
        "eve_observation_count": np.empty(shape, dtype=np.int32),
        "information_observed": np.empty(shape, dtype=np.bool_),
        "public_outcome_code": np.empty(shape, dtype=np.int8),
        "signal_class_code": np.empty(shape, dtype=np.int8),
    }


def _empty_endpoint_arrays(
    shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    chsh_shape = shape + (len(CHSH_SETTING_LABELS),)
    return {
        "alice_reward": np.empty(shape, dtype=np.float64),
        "bell_qualified_rate_hz": np.empty(shape, dtype=np.float64),
        "delivered_pair_count": np.empty(shape, dtype=np.int64),
        "key_pair_count": np.empty(shape, dtype=np.int64),
        "key_error_count": np.empty(shape, dtype=np.int64),
        "chsh_setting_total_counts": np.empty(
            chsh_shape, dtype=np.int64
        ),
        "chsh_setting_same_counts": np.empty(
            chsh_shape, dtype=np.int64
        ),
        "public_outcome_code": np.empty(shape, dtype=np.int8),
    }


def _matched_sampling_id(
    *,
    case: HistogramGameCase,
    config: HistogramCoLearningConfig,
    route_ids: tuple[str, ...],
    action_ids: tuple[str, ...],
    samples_per_cell: int,
    root_seed: int,
    split_seeds: dict[str, int],
    selection_digest_blake2b_128: str,
) -> str:
    payload = {
        "schema": MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA,
        "case_id": f"{case.graph_id}::{case.assignment_id}",
        "route_ids": list(route_ids),
        "action_ids": list(action_ids),
        "game_mode": config.game_mode,
        "requested_pairs_per_draw": int(
            config.requested_pairs_per_turn
        ),
        "samples_per_cell_per_split": int(samples_per_cell),
        "root_seed": int(root_seed),
        "split_seeds": dict(split_seeds),
        "selection_digest_blake2b_128": str(
            selection_digest_blake2b_128
        ),
        "physics_config": _physics_config(config),
        "compiled_histogram_runtime_version": (
            COMPILED_HISTOGRAM_RUNTIME_VERSION
        ),
    }
    digest = hashlib.blake2b(digest_size=16)
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


def _physics_config(
    config: HistogramCoLearningConfig,
) -> dict[str, Any]:
    return {
        "repeater_service_profile_path": str(
            config.repeater_service_profile_path
        ),
        "age_points_per_edge": config.age_points_per_edge,
        "accepted_alive_time_s": config.accepted_alive_time_s,
        "fock_cutoff": config.fock_cutoff,
        "detector_wavelength_approximation_allowed": (
            config.allow_known_1550_to_1558_detector_approximation
        ),
    }


def _stable_seed(root_seed: int, *parts: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(root_seed).to_bytes(8, "little", signed=True))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") % (2**31 - 1)
