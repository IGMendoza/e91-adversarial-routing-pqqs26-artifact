"""Equal-coverage histogram outcome banks for learner calibration."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
import threading
from typing import Any, Callable, Mapping

import numpy as np

from sequence_game.corpus.compiled_histogram_backend import (
    COMPILED_HISTOGRAM_RUNTIME_VERSION,
)

from .calibration_cases import STEP6_SELECTION_DIGEST
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .learning import (
    assess_alice_public_signal,
    bell_qualified_rate_utility,
    normalize_eve_bandit_reward,
)
from .physics import HistogramPhysicsPool


PAYOFF_BANK_SCHEMA = "histogram_calibration_payoff_bank.v2"
LEGACY_PAYOFF_BANK_SCHEMA = "histogram_calibration_payoff_bank.v1"
PAYOFF_BANK_SPLITS = ("feedback", "audit")
PUBLIC_OUTCOME_CODES = {
    "delivery_failure": 0,
    "qber_abort": 1,
    "chsh_abort": 2,
    "accepted": 3,
}
SIGNAL_CLASS_CODES = {
    "delivery_failure_not_quality_signal": 0,
    "missing_latent_quality_reference": 1,
    "true_accept": 2,
    "false_accept": 3,
    "false_abort": 4,
    "true_abort": 5,
}


@dataclass(frozen=True)
class PayoffSampleBlock:
    """Independent transcript-derived observations in every game cell."""

    alice_reward: np.ndarray
    corrected_alice_reward: np.ndarray
    eve_bandit_reward: np.ndarray
    bell_qualified_rate_hz: np.ndarray
    false_signal: np.ndarray
    classifiable_signal: np.ndarray
    target_active: np.ndarray
    eve_observation_count: np.ndarray
    information_observed: np.ndarray
    public_outcome_code: np.ndarray
    signal_class_code: np.ndarray

    def __post_init__(self) -> None:
        shape = np.asarray(self.alice_reward).shape
        if len(shape) != 3 or min(shape) < 1 or shape[2] < 2:
            raise ValueError(
                "payoff samples must have route x action x draw dimensions "
                "with at least two draws"
            )
        for name in (
            "corrected_alice_reward",
            "eve_bandit_reward",
            "bell_qualified_rate_hz",
            "false_signal",
            "classifiable_signal",
            "target_active",
            "eve_observation_count",
            "information_observed",
            "public_outcome_code",
            "signal_class_code",
        ):
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} has the wrong shape")
        for name in (
            "alice_reward",
            "corrected_alice_reward",
            "eve_bandit_reward",
        ):
            values = np.asarray(getattr(self, name), dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(
                values > 1.0
            ):
                raise ValueError(f"{name} must lie in [0, 1]")
        rates = np.asarray(self.bell_qualified_rate_hz, dtype=float)
        if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
            raise ValueError("bell_qualified_rate_hz must be finite and nonnegative")
        counts = np.asarray(self.eve_observation_count)
        if not np.issubdtype(counts.dtype, np.integer) or np.any(counts < 0):
            raise ValueError("eve_observation_count must be a nonnegative integer array")
        observed = np.asarray(self.information_observed, dtype=bool)
        if not np.array_equal(observed, counts > 0):
            raise ValueError(
                "information_observed must equal eve_observation_count > 0"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.alice_reward.shape)

    @property
    def draws_per_cell(self) -> int:
        return self.shape[2]

    def alice_payoff_matrix(self) -> np.ndarray:
        return np.mean(self.alice_reward, axis=2)

    def cell_standard_errors(self) -> np.ndarray:
        return np.std(self.alice_reward, axis=2, ddof=1) / math.sqrt(
            self.draws_per_cell
        )

    def false_signal_rate_matrix(self) -> np.ndarray:
        return np.mean(self.false_signal, axis=2)

    def draw(
        self,
        route_index: int,
        action_index: int,
        draw_index: int,
    ) -> dict[str, Any]:
        index = (int(route_index), int(action_index), int(draw_index))
        return {
            "alice_reward": float(self.alice_reward[index]),
            "corrected_alice_reward": float(
                self.corrected_alice_reward[index]
            ),
            "eve_bandit_reward": float(self.eve_bandit_reward[index]),
            "bell_qualified_rate_hz": float(
                self.bell_qualified_rate_hz[index]
            ),
            "false_signal": bool(self.false_signal[index]),
            "classifiable_signal": bool(self.classifiable_signal[index]),
            "target_active": bool(self.target_active[index]),
            "eve_observation_count": int(self.eve_observation_count[index]),
            "information_observed": bool(self.information_observed[index]),
            "public_outcome_code": int(self.public_outcome_code[index]),
            "signal_class_code": int(self.signal_class_code[index]),
        }


@dataclass(frozen=True)
class HistogramPayoffBank:
    """Disjoint feedback and scoring observations for one frozen case."""

    case_id: str
    graph_id: str
    assignment_id: str
    route_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    action_costs: np.ndarray
    feedback: PayoffSampleBlock
    audit: PayoffSampleBlock
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        shape = self.feedback.shape
        if self.audit.shape != shape:
            raise ValueError("feedback and audit blocks have different shapes")
        if shape[:2] != (len(self.route_ids), len(self.action_ids)):
            raise ValueError("sample shape does not match route/action identities")
        costs = np.asarray(self.action_costs, dtype=float)
        if costs.shape != (len(self.action_ids),):
            raise ValueError("action_costs has the wrong shape")
        if not np.all(np.isfinite(costs)) or np.any(costs < 0.0):
            raise ValueError("action costs must be finite and nonnegative")
        if len(set(self.route_ids)) != len(self.route_ids):
            raise ValueError("route ids are not unique")
        if len(set(self.action_ids)) != len(self.action_ids):
            raise ValueError("action ids are not unique")

    @property
    def draws_per_cell(self) -> int:
        return self.feedback.draws_per_cell

    def audit_payoff_matrix(self) -> np.ndarray:
        return self.audit.alice_payoff_matrix()

    def equivalent_zero_sum_matrix(self) -> np.ndarray:
        return self.audit_payoff_matrix() + self.action_costs[np.newaxis, :]

    def content_digest(self) -> str:
        cached = getattr(self, "_content_digest_cache", None)
        if cached is not None:
            return str(cached)
        value = _bank_content_digest(self, _BLOCK_ARRAY_NAMES)
        object.__setattr__(self, "_content_digest_cache", value)
        return value

    def to_summary(self) -> dict[str, Any]:
        audit = self.audit
        return {
            "schema": PAYOFF_BANK_SCHEMA,
            "case_id": self.case_id,
            "graph_id": self.graph_id,
            "assignment_id": self.assignment_id,
            "route_count": len(self.route_ids),
            "action_count": len(self.action_ids),
            "draws_per_cell_per_split": self.draws_per_cell,
            "total_histogram_draws": (
                2 * len(self.route_ids) * len(self.action_ids) * self.draws_per_cell
            ),
            "content_digest_blake2b_128": self.content_digest(),
            "audit": {
                "alice_payoff_minimum": float(np.min(audit.alice_reward)),
                "alice_payoff_maximum": float(np.max(audit.alice_reward)),
                "alice_payoff_mean": float(np.mean(audit.alice_reward)),
                "maximum_cell_standard_error": float(
                    np.max(audit.cell_standard_errors())
                ),
                "false_signal_rate": float(np.mean(audit.false_signal)),
                "classifiable_signal_rate": float(
                    np.mean(audit.classifiable_signal)
                ),
                "target_active_rate": float(np.mean(audit.target_active)),
                "information_observation_rate": float(
                    np.mean(audit.information_observed)
                ),
                "mean_eve_observation_count": float(
                    np.mean(audit.eve_observation_count)
                ),
            },
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.metadata,
            "schema": PAYOFF_BANK_SCHEMA,
            "case_id": self.case_id,
            "graph_id": self.graph_id,
            "assignment_id": self.assignment_id,
            "route_ids": list(self.route_ids),
            "action_ids": list(self.action_ids),
            "content_digest_blake2b_128": self.content_digest(),
        }
        arrays: dict[str, Any] = {
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
            "action_costs": self.action_costs,
        }
        for split in PAYOFF_BANK_SPLITS:
            block = getattr(self, split)
            for name in _BLOCK_ARRAY_NAMES:
                arrays[f"{split}_{name}"] = getattr(block, name)
        np.savez_compressed(destination, **arrays)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "HistogramPayoffBank":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            schema = str(metadata.get("schema", ""))
            if schema not in {PAYOFF_BANK_SCHEMA, LEGACY_PAYOFF_BANK_SCHEMA}:
                raise ValueError(f"unsupported payoff-bank schema {schema!r}")
            array_names = (
                _BLOCK_ARRAY_NAMES
                if schema == PAYOFF_BANK_SCHEMA
                else _LEGACY_BLOCK_ARRAY_NAMES
            )
            blocks = {}
            for split in PAYOFF_BANK_SPLITS:
                arrays = {
                    name: np.asarray(archive[f"{split}_{name}"]).copy()
                    for name in array_names
                }
                if schema == LEGACY_PAYOFF_BANK_SCHEMA:
                    arrays["eve_observation_count"] = np.asarray(
                        arrays["information_observed"], dtype=np.int32
                    )
                blocks[split] = PayoffSampleBlock(**arrays)
            bank = cls(
                case_id=str(metadata["case_id"]),
                graph_id=str(metadata["graph_id"]),
                assignment_id=str(metadata["assignment_id"]),
                route_ids=tuple(map(str, metadata["route_ids"])),
                action_ids=tuple(map(str, metadata["action_ids"])),
                action_costs=np.asarray(archive["action_costs"], dtype=float).copy(),
                feedback=blocks["feedback"],
                audit=blocks["audit"],
                metadata={
                    key: value for key, value in metadata.items()
                    if key not in {
                        "schema",
                        "case_id",
                        "graph_id",
                        "assignment_id",
                        "route_ids",
                        "action_ids",
                        "content_digest_blake2b_128",
                    }
                } | ({
                    "eve_observation_count_semantics": (
                        "binary_indicator_migrated_from_v1"
                    ),
                    "loaded_source_schema": LEGACY_PAYOFF_BANK_SCHEMA,
                } if schema == LEGACY_PAYOFF_BANK_SCHEMA else {}),
            )
        expected = metadata.get("content_digest_blake2b_128")
        observed = _bank_content_digest(bank, array_names)
        if observed != expected:
            raise ValueError(f"payoff-bank content digest mismatch in {path}")
        return bank


_BLOCK_ARRAY_NAMES = (
    "alice_reward",
    "corrected_alice_reward",
    "eve_bandit_reward",
    "bell_qualified_rate_hz",
    "false_signal",
    "classifiable_signal",
    "target_active",
    "eve_observation_count",
    "information_observed",
    "public_outcome_code",
    "signal_class_code",
)
_LEGACY_BLOCK_ARRAY_NAMES = tuple(
    name for name in _BLOCK_ARRAY_NAMES if name != "eve_observation_count"
)


def _bank_content_digest(
    bank: HistogramPayoffBank,
    array_names: tuple[str, ...],
) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(bank.case_id.encode("utf-8"))
    digest.update(json.dumps(bank.route_ids).encode("utf-8"))
    digest.update(json.dumps(bank.action_ids).encode("utf-8"))
    digest.update(np.ascontiguousarray(bank.action_costs).view(np.uint8))
    for split in PAYOFF_BANK_SPLITS:
        block = getattr(bank, split)
        for name in array_names:
            values = np.ascontiguousarray(getattr(block, name))
            digest.update(split.encode("ascii"))
            digest.update(name.encode("ascii"))
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(np.asarray(values.shape, dtype=np.int64).view(np.uint8))
            digest.update(values.view(np.uint8))
    return digest.hexdigest()


def build_histogram_payoff_bank(
    case: HistogramGameCase,
    config: HistogramCoLearningConfig,
    *,
    samples_per_cell: int = 64,
    root_seed: int = 20_260_715,
    physics: HistogramPhysicsPool | None = None,
    progress_callback: Callable[[str], None] | None = None,
    selection_digest_blake2b_128: str = STEP6_SELECTION_DIGEST,
    evaluation_partition_loaded: bool = False,
    sampling_purpose: str = "learner_calibration",
    sampling_design: str | None = None,
    route_workers: int = 1,
) -> HistogramPayoffBank:
    """Sample disjoint, equally covered feedback and scoring banks."""

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
        [float(action.get("cost", 0.0)) for action in actions], dtype=float
    )
    maximum_cost = float(action_costs.max(initial=0.0))
    reference_rate = active_physics.reference_service_rate_hz
    shape = (len(route_ids), len(action_ids), int(samples_per_cell))
    split_blocks: dict[str, PayoffSampleBlock] = {}
    split_seeds: dict[str, int] = {}
    for split_index, split in enumerate(PAYOFF_BANK_SPLITS):
        seed = _stable_seed(root_seed, case.graph_id, case.assignment_id, split)
        split_seeds[split] = seed
        arrays = _empty_block_arrays(shape)
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
                        (route_index * len(actions) + action_index)
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
                        1.0 - utility.reward - action_costs[action_index]
                    )
                    eve_reward = normalize_eve_bandit_reward(
                        raw_eve_reward,
                        maximum_cost=maximum_cost,
                    )
                    index = (route_index, action_index, draw_index)
                    arrays["alice_reward"][index] = utility.reward
                    arrays["corrected_alice_reward"][index] = corrected.reward
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
                        arrays["public_outcome_code"][index] = (
                            PUBLIC_OUTCOME_CODES[str(result.public_outcome)]
                        )
                        arrays["signal_class_code"][index] = (
                            SIGNAL_CLASS_CODES[signal.classification]
                        )
                    except KeyError as exc:
                        raise ValueError(
                            f"unrecognized histogram diagnostic label {exc.args[0]!r}"
                        ) from exc
                    route_completed += 1
            with progress_lock:
                completed += route_completed
                if progress_callback is not None and (
                    completed >= next_progress or completed == total
                ):
                    progress_callback(
                        f"{case.family} {split}: {completed}/{total} draws"
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
        split_blocks[split] = PayoffSampleBlock(**arrays)
    return HistogramPayoffBank(
        case_id=f"{case.graph_id}::{case.assignment_id}",
        graph_id=case.graph_id,
        assignment_id=case.assignment_id,
        route_ids=route_ids,
        action_ids=action_ids,
        action_costs=action_costs,
        feedback=split_blocks["feedback"],
        audit=split_blocks["audit"],
        metadata={
            "selection_digest_blake2b_128": str(
                selection_digest_blake2b_128
            ),
            "sampling_design": sampling_design or (
                "two disjoint equal-coverage histogram transcript banks; "
                "feedback outcomes are resampled during candidate screening "
                "and audit outcomes are reserved for scoring; confirmation "
                "uses fresh compiled-histogram transcript draws"
            ),
            "sampling_purpose": str(sampling_purpose),
            "compiled_histogram_runtime_version": (
                COMPILED_HISTOGRAM_RUNTIME_VERSION
            ),
            "eve_observation_count_semantics": "physical_count_per_transcript",
            "samples_per_cell_per_split": int(samples_per_cell),
            "root_seed": int(root_seed),
            "split_seeds": split_seeds,
            "game_mode": config.game_mode,
            "requested_pairs_per_draw": config.requested_pairs_per_turn,
            "physics_config": {
                "repeater_service_profile_path": str(
                    config.repeater_service_profile_path
                ),
                "age_points_per_edge": config.age_points_per_edge,
                "accepted_alive_time_s": config.accepted_alive_time_s,
                "fock_cutoff": config.fock_cutoff,
                "detector_wavelength_approximation_allowed": (
                    config.allow_known_1550_to_1558_detector_approximation
                ),
            },
            "feedback_audit_seed_disjoint": True,
            "evaluation_partition_loaded": bool(
                evaluation_partition_loaded
            ),
            "physics": active_physics.diagnostics(),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def validate_bank_against_case(
    bank: HistogramPayoffBank,
    case: HistogramGameCase,
    config: HistogramCoLearningConfig,
    *,
    selection_digest_blake2b_128: str = STEP6_SELECTION_DIGEST,
) -> None:
    """Reject stale banks before any learner calibration begins."""

    expected_case_id = f"{case.graph_id}::{case.assignment_id}"
    if bank.case_id != expected_case_id:
        raise ValueError("payoff bank belongs to a different case")
    routes = tuple(str(route["route_id"]) for route in case.routes)
    actions = case.actions(config.game_mode)
    action_ids = tuple(str(action["id"]) for action in actions)
    costs = np.asarray([float(action.get("cost", 0.0)) for action in actions])
    if bank.route_ids != routes or bank.action_ids != action_ids:
        raise ValueError("payoff bank route/action order changed")
    if not np.array_equal(bank.action_costs, costs):
        raise ValueError("payoff bank action costs changed")
    metadata = bank.metadata
    if metadata.get("compiled_histogram_runtime_version") != (
        COMPILED_HISTOGRAM_RUNTIME_VERSION
    ):
        raise ValueError(
            "payoff bank compiled histogram runtime semantics changed"
        )
    if metadata.get("selection_digest_blake2b_128") != str(
        selection_digest_blake2b_128
    ):
        raise ValueError("payoff bank does not match the frozen case selection")
    if metadata.get("game_mode") != config.game_mode:
        raise ValueError("payoff bank game mode changed")
    if int(metadata.get("requested_pairs_per_draw", -1)) != int(
        config.requested_pairs_per_turn
    ):
        raise ValueError("payoff bank transcript size changed")
    expected_physics = {
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
    if metadata.get("physics_config") != expected_physics:
        raise ValueError("payoff bank physics configuration changed")


def _empty_block_arrays(shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
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


def _stable_seed(root_seed: int, *parts: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(root_seed).to_bytes(8, "little", signed=True))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") % (2**31 - 1)
