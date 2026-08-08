"""Disjoint compiled-physics banks for Alice endpoint detection evidence.

This bank is intentionally separate from the frozen learner payoff-bank
schema.  Each draw preserves Alice's scalar public reward together with the
finite delivery, key-error, and per-setting CHSH counts from that same
compiled transcript.  It does not retain Eve action diagnostics beyond the
declared action-axis identity needed to replay an experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from sequence_game.corpus.compiled_histogram_backend import (
    COMPILED_HISTOGRAM_RUNTIME_VERSION,
)

from .calibration_bank import PUBLIC_OUTCOME_CODES
from .calibration_cases import STEP6_SELECTION_DIGEST
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .learning import bell_qualified_rate_utility
from .physics import HistogramPhysicsPool
from .quantum_stackelberg_detection import (
    CHSH_SETTING_LABELS,
    QuantumEndpointEvidence,
)


ENDPOINT_EVIDENCE_BANK_SCHEMA = "histogram_endpoint_evidence_bank.v1"
ENDPOINT_EVIDENCE_SPLITS = ("feedback", "audit")
_BLOCK_ARRAY_NAMES = (
    "alice_reward",
    "bell_qualified_rate_hz",
    "delivered_pair_count",
    "key_pair_count",
    "key_error_count",
    "chsh_setting_total_counts",
    "chsh_setting_same_counts",
    "public_outcome_code",
)


@dataclass(frozen=True)
class EndpointEvidenceSampleBlock:
    """Route x action x draw samples with one trailing CHSH-setting axis."""

    alice_reward: np.ndarray
    bell_qualified_rate_hz: np.ndarray
    delivered_pair_count: np.ndarray
    key_pair_count: np.ndarray
    key_error_count: np.ndarray
    chsh_setting_total_counts: np.ndarray
    chsh_setting_same_counts: np.ndarray
    public_outcome_code: np.ndarray

    def __post_init__(self) -> None:
        shape = np.asarray(self.alice_reward).shape
        if len(shape) != 3 or min(shape) < 1 or shape[2] < 2:
            raise ValueError(
                "endpoint evidence must have nonempty route x action x draw "
                "dimensions with at least two draws"
            )
        for name in (
            "bell_qualified_rate_hz",
            "delivered_pair_count",
            "key_pair_count",
            "key_error_count",
            "public_outcome_code",
        ):
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} has the wrong shape")
        chsh_shape = shape + (len(CHSH_SETTING_LABELS),)
        for name in (
            "chsh_setting_total_counts",
            "chsh_setting_same_counts",
        ):
            if np.asarray(getattr(self, name)).shape != chsh_shape:
                raise ValueError(f"{name} has the wrong shape")

        rewards = np.asarray(self.alice_reward, dtype=float)
        if (
            not np.all(np.isfinite(rewards))
            or np.any(rewards < 0.0)
            or np.any(rewards > 1.0)
        ):
            raise ValueError("alice_reward must lie in [0, 1]")
        rates = np.asarray(self.bell_qualified_rate_hz, dtype=float)
        if not np.all(np.isfinite(rates)) or np.any(rates < 0.0):
            raise ValueError(
                "bell_qualified_rate_hz must be finite and nonnegative"
            )
        for name in (
            "delivered_pair_count",
            "key_pair_count",
            "key_error_count",
            "chsh_setting_total_counts",
            "chsh_setting_same_counts",
            "public_outcome_code",
        ):
            values = np.asarray(getattr(self, name))
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"{name} must contain integers")
        delivered = np.asarray(self.delivered_pair_count, dtype=np.int64)
        key_pairs = np.asarray(self.key_pair_count, dtype=np.int64)
        key_errors = np.asarray(self.key_error_count, dtype=np.int64)
        chsh_totals = np.asarray(
            self.chsh_setting_total_counts, dtype=np.int64
        )
        chsh_same = np.asarray(
            self.chsh_setting_same_counts, dtype=np.int64
        )
        if (
            np.any(delivered < 0)
            or np.any(key_pairs < 0)
            or np.any(key_errors < 0)
            or np.any(chsh_totals < 0)
            or np.any(chsh_same < 0)
            or np.any(key_pairs > delivered)
            or np.any(key_errors > key_pairs)
            or np.any(chsh_same > chsh_totals)
            or np.any(key_pairs + np.sum(chsh_totals, axis=3) > delivered)
        ):
            raise ValueError("endpoint evidence contains inconsistent counts")
        outcomes = np.asarray(self.public_outcome_code, dtype=np.int64)
        if np.any(outcomes < 0) or np.any(
            outcomes >= len(PUBLIC_OUTCOME_CODES)
        ):
            raise ValueError("public_outcome_code is outside the declared set")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.alice_reward.shape)

    @property
    def draws_per_cell(self) -> int:
        return self.shape[2]

    def draw(
        self,
        route_index: int,
        action_index: int,
        draw_index: int,
        *,
        requested_pair_count: int,
    ) -> dict[str, Any]:
        index = (int(route_index), int(action_index), int(draw_index))
        evidence = self.evidence(
            route_index,
            action_index,
            draw_index,
            requested_pair_count=requested_pair_count,
        )
        return {
            "alice_reward": float(self.alice_reward[index]),
            "bell_qualified_rate_hz": float(
                self.bell_qualified_rate_hz[index]
            ),
            "public_outcome_code": int(self.public_outcome_code[index]),
            "endpoint_evidence": evidence,
        }

    def evidence(
        self,
        route_index: int,
        action_index: int,
        draw_index: int,
        *,
        requested_pair_count: int,
    ) -> QuantumEndpointEvidence:
        index = (int(route_index), int(action_index), int(draw_index))
        return QuantumEndpointEvidence(
            requested_pair_count=int(requested_pair_count),
            delivered_pair_count=int(self.delivered_pair_count[index]),
            key_pair_count=int(self.key_pair_count[index]),
            key_error_count=int(self.key_error_count[index]),
            public_service_accepted=(
                int(self.public_outcome_code[index])
                == PUBLIC_OUTCOME_CODES["accepted"]
            ),
            chsh_setting_total_counts=tuple(
                map(int, self.chsh_setting_total_counts[index])
            ),
            chsh_setting_same_counts=tuple(
                map(int, self.chsh_setting_same_counts[index])
            ),
        )


@dataclass(frozen=True)
class HistogramEndpointEvidenceBank:
    """Disjoint feedback and audit endpoint traces for one frozen case."""

    case_id: str
    graph_id: str
    assignment_id: str
    route_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    requested_pairs_per_draw: int
    feedback: EndpointEvidenceSampleBlock
    audit: EndpointEvidenceSampleBlock
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.feedback.shape != self.audit.shape:
            raise ValueError("feedback and audit evidence shapes differ")
        if self.feedback.shape[:2] != (
            len(self.route_ids),
            len(self.action_ids),
        ):
            raise ValueError(
                "evidence shape does not match route/action identities"
            )
        if self.requested_pairs_per_draw < 1:
            raise ValueError("requested_pairs_per_draw must be positive")
        for block in (self.feedback, self.audit):
            if np.any(
                np.asarray(block.delivered_pair_count)
                > self.requested_pairs_per_draw
            ):
                raise ValueError(
                    "delivered counts exceed requested_pairs_per_draw"
                )
        if len(set(self.route_ids)) != len(self.route_ids):
            raise ValueError("route ids are not unique")
        if len(set(self.action_ids)) != len(self.action_ids):
            raise ValueError("action ids are not unique")

    @property
    def draws_per_cell(self) -> int:
        return self.feedback.draws_per_cell

    def content_digest(self) -> str:
        cached = getattr(self, "_content_digest_cache", None)
        if cached is not None:
            return str(cached)
        value = _bank_content_digest(self)
        object.__setattr__(self, "_content_digest_cache", value)
        return value

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema": ENDPOINT_EVIDENCE_BANK_SCHEMA,
            "case_id": self.case_id,
            "graph_id": self.graph_id,
            "assignment_id": self.assignment_id,
            "route_count": len(self.route_ids),
            "action_count": len(self.action_ids),
            "draws_per_cell_per_split": self.draws_per_cell,
            "requested_pairs_per_draw": self.requested_pairs_per_draw,
            "total_compiled_transcript_draws": (
                2
                * len(self.route_ids)
                * len(self.action_ids)
                * self.draws_per_cell
            ),
            "content_digest_blake2b_128": self.content_digest(),
            "audit": {
                "mean_alice_reward": float(
                    np.mean(self.audit.alice_reward)
                ),
                "mean_delivery_fraction": float(
                    np.mean(self.audit.delivered_pair_count)
                    / self.requested_pairs_per_draw
                ),
                "total_key_trials": int(
                    np.sum(self.audit.key_pair_count)
                ),
                "total_chsh_trials": int(
                    np.sum(self.audit.chsh_setting_total_counts)
                ),
            },
            "information_boundary": {
                "retained": [
                    "declared route/action/draw coordinates",
                    "Alice scalar public reward and Bell-qualified rate",
                    "public terminal outcome code",
                    "requested/delivered pair counts",
                    "key-pair and key-error counts",
                    "per-setting CHSH trial and same-outcome counts",
                ],
                "excluded": [
                    "target-active flag",
                    "Eve observation transcript",
                    "expected simulator probabilities",
                    "private no-delivery-stage diagnostics",
                    "memory, herald, or BSM fields without an explicit "
                    "endpoint-visible observation contract",
                ],
            },
            "metadata": dict(self.metadata),
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            **self.metadata,
            "schema": ENDPOINT_EVIDENCE_BANK_SCHEMA,
            "case_id": self.case_id,
            "graph_id": self.graph_id,
            "assignment_id": self.assignment_id,
            "route_ids": list(self.route_ids),
            "action_ids": list(self.action_ids),
            "requested_pairs_per_draw": self.requested_pairs_per_draw,
            "content_digest_blake2b_128": self.content_digest(),
        }
        arrays: dict[str, Any] = {
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
        }
        for split in ENDPOINT_EVIDENCE_SPLITS:
            block = getattr(self, split)
            for name in _BLOCK_ARRAY_NAMES:
                arrays[f"{split}_{name}"] = getattr(block, name)
        np.savez_compressed(destination, **arrays)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "HistogramEndpointEvidenceBank":
        with np.load(Path(path), allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            schema = str(metadata.get("schema", ""))
            if schema != ENDPOINT_EVIDENCE_BANK_SCHEMA:
                raise ValueError(
                    f"unsupported endpoint evidence schema {schema!r}"
                )
            blocks = {
                split: EndpointEvidenceSampleBlock(**{
                    name: np.asarray(
                        archive[f"{split}_{name}"]
                    ).copy()
                    for name in _BLOCK_ARRAY_NAMES
                })
                for split in ENDPOINT_EVIDENCE_SPLITS
            }
            bank = cls(
                case_id=str(metadata["case_id"]),
                graph_id=str(metadata["graph_id"]),
                assignment_id=str(metadata["assignment_id"]),
                route_ids=tuple(map(str, metadata["route_ids"])),
                action_ids=tuple(map(str, metadata["action_ids"])),
                requested_pairs_per_draw=int(
                    metadata["requested_pairs_per_draw"]
                ),
                feedback=blocks["feedback"],
                audit=blocks["audit"],
                metadata={
                    key: value
                    for key, value in metadata.items()
                    if key
                    not in {
                        "schema",
                        "case_id",
                        "graph_id",
                        "assignment_id",
                        "route_ids",
                        "action_ids",
                        "requested_pairs_per_draw",
                        "content_digest_blake2b_128",
                    }
                },
            )
        expected = str(metadata["content_digest_blake2b_128"])
        if bank.content_digest() != expected:
            raise ValueError(
                f"endpoint evidence content digest mismatch in {path}"
            )
        return bank


def build_histogram_endpoint_evidence_bank(
    case: HistogramGameCase,
    config: HistogramCoLearningConfig,
    *,
    samples_per_cell: int = 64,
    root_seed: int = 20_260_716,
    physics: HistogramPhysicsPool | None = None,
    progress_callback: Callable[[str], None] | None = None,
    selection_digest_blake2b_128: str = STEP6_SELECTION_DIGEST,
    evaluation_partition_loaded: bool = False,
) -> HistogramEndpointEvidenceBank:
    """Sample disjoint equal-coverage endpoint evidence from compiled physics."""

    if samples_per_cell < 2:
        raise ValueError("samples_per_cell must be at least two")
    started = time.perf_counter()
    active_physics = physics or HistogramPhysicsPool(case, config)
    calibrations = active_physics.calibrations
    actions = case.actions(config.game_mode)
    if len(calibrations) != len(case.routes) or not actions:
        raise ValueError("physics routes or Eve actions are inconsistent")
    route_ids = tuple(str(route["route_id"]) for route in case.routes)
    action_ids = tuple(str(action["id"]) for action in actions)
    reference_rate = active_physics.reference_service_rate_hz
    shape = (len(route_ids), len(action_ids), int(samples_per_cell))
    split_blocks: dict[str, EndpointEvidenceSampleBlock] = {}
    split_seeds: dict[str, int] = {}
    for split in ENDPOINT_EVIDENCE_SPLITS:
        seed = _stable_seed(
            root_seed,
            case.graph_id,
            case.assignment_id,
            "endpoint_evidence",
            split,
        )
        split_seeds[split] = seed
        arrays = _empty_block_arrays(shape)
        total = int(np.prod(shape))
        completed = 0
        progress_stride = max(samples_per_cell, total // 20)
        next_progress = progress_stride
        for route_index, calibration in enumerate(calibrations):
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
                    evidence = QuantumEndpointEvidence.from_step_result(result)
                    index = (route_index, action_index, draw_index)
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
                        arrays["public_outcome_code"][index] = (
                            PUBLIC_OUTCOME_CODES[str(result.public_outcome)]
                        )
                    except KeyError as exc:
                        raise ValueError(
                            "unrecognized public outcome "
                            f"{exc.args[0]!r}"
                        ) from exc
                    completed += 1
                if progress_callback is not None and (
                    completed == samples_per_cell
                    or completed >= next_progress
                    or completed == total
                ):
                    progress_callback(
                        f"{case.family} endpoint {split}: "
                        f"{completed}/{total} draws"
                    )
                    while next_progress <= completed:
                        next_progress += progress_stride
        split_blocks[split] = EndpointEvidenceSampleBlock(**arrays)

    return HistogramEndpointEvidenceBank(
        case_id=f"{case.graph_id}::{case.assignment_id}",
        graph_id=case.graph_id,
        assignment_id=case.assignment_id,
        route_ids=route_ids,
        action_ids=action_ids,
        requested_pairs_per_draw=int(config.requested_pairs_per_turn),
        feedback=split_blocks["feedback"],
        audit=split_blocks["audit"],
        metadata={
            "selection_digest_blake2b_128": str(
                selection_digest_blake2b_128
            ),
            "sampling_design": (
                "two disjoint equal-coverage compiled transcript banks; "
                "feedback is reserved for commissioning/model construction "
                "and audit for held-out matched detector execution"
            ),
            "samples_per_cell_per_split": int(samples_per_cell),
            "root_seed": int(root_seed),
            "split_seeds": split_seeds,
            "game_mode": config.game_mode,
            "compiled_histogram_runtime_version": (
                COMPILED_HISTOGRAM_RUNTIME_VERSION
            ),
            "feedback_audit_seed_disjoint": True,
            "evaluation_partition_loaded": bool(
                evaluation_partition_loaded
            ),
            "physics_config": _physics_config(config),
            "physics": active_physics.diagnostics(),
            "elapsed_seconds": time.perf_counter() - started,
        },
    )


def validate_endpoint_evidence_bank_against_case(
    bank: HistogramEndpointEvidenceBank,
    case: HistogramGameCase,
    config: HistogramCoLearningConfig,
    *,
    selection_digest_blake2b_128: str = STEP6_SELECTION_DIGEST,
) -> None:
    """Reject stale or mismatched endpoint evidence before a study."""

    if bank.case_id != f"{case.graph_id}::{case.assignment_id}":
        raise ValueError("endpoint evidence bank belongs to a different case")
    route_ids = tuple(str(route["route_id"]) for route in case.routes)
    action_ids = tuple(
        str(action["id"]) for action in case.actions(config.game_mode)
    )
    if bank.route_ids != route_ids or bank.action_ids != action_ids:
        raise ValueError("endpoint evidence route/action order changed")
    if bank.requested_pairs_per_draw != config.requested_pairs_per_turn:
        raise ValueError("endpoint evidence transcript size changed")
    if bank.metadata.get("selection_digest_blake2b_128") != str(
        selection_digest_blake2b_128
    ):
        raise ValueError(
            "endpoint evidence does not match the frozen case selection"
        )
    if bank.metadata.get("game_mode") != config.game_mode:
        raise ValueError("endpoint evidence game mode changed")
    if bank.metadata.get("compiled_histogram_runtime_version") != (
        COMPILED_HISTOGRAM_RUNTIME_VERSION
    ):
        raise ValueError(
            "endpoint evidence compiled histogram runtime semantics changed"
        )
    if bank.metadata.get("physics_config") != _physics_config(config):
        raise ValueError("endpoint evidence physics configuration changed")


def _empty_block_arrays(
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


def _physics_config(config: HistogramCoLearningConfig) -> dict[str, Any]:
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


def _bank_content_digest(bank: HistogramEndpointEvidenceBank) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(bank.case_id.encode("utf-8"))
    digest.update(json.dumps(bank.route_ids).encode("utf-8"))
    digest.update(json.dumps(bank.action_ids).encode("utf-8"))
    digest.update(
        int(bank.requested_pairs_per_draw).to_bytes(
            8, "little", signed=False
        )
    )
    for split in ENDPOINT_EVIDENCE_SPLITS:
        block = getattr(bank, split)
        for name in _BLOCK_ARRAY_NAMES:
            values = np.ascontiguousarray(getattr(block, name))
            digest.update(split.encode("ascii"))
            digest.update(name.encode("ascii"))
            digest.update(str(values.dtype).encode("ascii"))
            digest.update(
                np.asarray(values.shape, dtype=np.int64).view(np.uint8)
            )
            digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _stable_seed(root_seed: int, *parts: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(root_seed).to_bytes(8, "little", signed=True))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") % (2**31 - 1)
