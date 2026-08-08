"""Paired endpoint/payoff banks across finite observation-block sizes.

One maximum-size compiled transcript is sampled per route/action/draw cell.
Smaller blocks are nested multivariate-hypergeometric subsets of that same
categorical transcript. Marginally, every derived block is an exact
multinomial draw at its declared size; jointly, block-size comparisons are
paired rather than confounded by unrelated transcript noise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np

from .calibration_bank import (
    PUBLIC_OUTCOME_CODES,
    SIGNAL_CLASS_CODES,
    HistogramPayoffBank,
    PayoffSampleBlock,
)
from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase
from .endpoint_evidence_bank import (
    EndpointEvidenceSampleBlock,
    HistogramEndpointEvidenceBank,
    validate_endpoint_evidence_bank_against_case,
)
from .learning import normalize_eve_bandit_reward
from .physics import HistogramPhysicsPool


PAIRED_BLOCK_EVIDENCE_SCHEMA = "histogram_paired_block_evidence.v1"
PAIRED_BLOCK_DIAGNOSTIC_SIZES = (64, 128)
PAIRED_BLOCK_PRIMARY_SIZES = (256, 512, 1_200)


@dataclass(frozen=True)
class PairedBlockSizeBanks:
    """Standard banks plus pairing diagnostics for each requested size."""

    maximum_block_size: int
    block_sizes: tuple[int, ...]
    endpoint_banks: Mapping[int, HistogramEndpointEvidenceBank]
    payoff_banks: Mapping[int, HistogramPayoffBank]
    summary: dict[str, Any]

    def __post_init__(self) -> None:
        if self.maximum_block_size != max(self.block_sizes):
            raise ValueError("maximum block size does not match block_sizes")
        expected = set(self.block_sizes)
        if set(self.endpoint_banks) != expected:
            raise ValueError("endpoint banks do not cover every block size")
        if set(self.payoff_banks) != expected:
            raise ValueError("payoff banks do not cover every block size")


def derive_paired_block_size_banks(
    case: HistogramGameCase,
    maximum_config: HistogramCoLearningConfig,
    source: HistogramEndpointEvidenceBank,
    *,
    physics: HistogramPhysicsPool,
    block_sizes: Sequence[int] = (
        *PAIRED_BLOCK_DIAGNOSTIC_SIZES,
        *PAIRED_BLOCK_PRIMARY_SIZES,
    ),
    thinning_root_seed: int = 20_260_720,
    selection_digest_blake2b_128: str,
    evaluation_partition_loaded: bool = False,
) -> PairedBlockSizeBanks:
    """Derive exact nested finite-block banks from one maximum transcript."""

    sizes = tuple(sorted(set(map(int, block_sizes))))
    if not sizes or sizes[0] < 1:
        raise ValueError("block_sizes must contain positive integers")
    maximum = sizes[-1]
    if source.requested_pairs_per_draw != maximum:
        raise ValueError(
            "source endpoint bank must use the maximum block size"
        )
    if maximum_config.requested_pairs_per_turn != maximum:
        raise ValueError(
            "maximum config and source endpoint bank sizes differ"
        )
    validate_endpoint_evidence_bank_against_case(
        source,
        case,
        maximum_config,
        selection_digest_blake2b_128=(
            selection_digest_blake2b_128
        ),
    )
    if physics.config.requested_pairs_per_turn != maximum:
        raise ValueError("physics pool was not built for maximum block size")

    actions = case.actions(maximum_config.game_mode)
    route_count = len(case.routes)
    action_count = len(actions)
    shape = source.feedback.shape
    if shape[:2] != (route_count, action_count):
        raise ValueError("source bank axes do not match case")

    cell_model = _cell_model(case, actions, physics)
    action_costs = np.asarray(
        [float(action.get("cost", 0.0)) for action in actions],
        dtype=float,
    )
    maximum_cost = float(action_costs.max(initial=0.0))
    endpoint_arrays = {
        size: {
            split: _empty_endpoint_arrays(shape)
            for split in ("feedback", "audit")
        }
        for size in sizes
    }
    payoff_arrays = {
        size: {
            split: _empty_payoff_arrays(shape)
            for split in ("feedback", "audit")
        }
        for size in sizes
    }

    for split_index, split in enumerate(("feedback", "audit")):
        source_block = getattr(source, split)
        for route_index in range(route_count):
            for action_index in range(action_count):
                for draw_index in range(shape[2]):
                    index = (
                        route_index,
                        action_index,
                        draw_index,
                    )
                    maximum_categories = _endpoint_categories(
                        source_block,
                        index,
                        maximum_block_size=maximum,
                    )
                    thinned = _nested_thinnings(
                        maximum_categories,
                        sizes=sizes,
                        seed=_stable_seed(
                            thinning_root_seed,
                            split,
                            str(route_index),
                            str(action_index),
                            str(draw_index),
                        ),
                    )
                    for size, categories in thinned.items():
                        _record_derived_draw(
                            categories,
                            size=size,
                            index=index,
                            route_index=route_index,
                            action_index=action_index,
                            cell_model=cell_model,
                            action_costs=action_costs,
                            maximum_cost=maximum_cost,
                            endpoint_arrays=endpoint_arrays[size][split],
                            payoff_arrays=payoff_arrays[size][split],
                        )

    endpoint_banks: dict[int, HistogramEndpointEvidenceBank] = {}
    payoff_banks: dict[int, HistogramPayoffBank] = {}
    source_digest = source.content_digest()
    pairing_contract = {
        "schema": PAIRED_BLOCK_EVIDENCE_SCHEMA,
        "maximum_block_size": maximum,
        "block_sizes": list(sizes),
        "thinning_root_seed": int(thinning_root_seed),
        "source_endpoint_digest_blake2b_128": source_digest,
        "coupling": (
            "nested multivariate-hypergeometric thinning from one "
            "maximum-size categorical transcript per split/cell/draw"
        ),
        "marginal_law": (
            "exact multinomial transcript at each declared block size"
        ),
        "protocol_decision_thresholds": {
            "min_key_pairs_for_qber_abort": _unique_integer(
                cell_model["min_key_pairs_for_qber_abort"]
            ),
            "min_chsh_pairs_for_chsh_abort": _unique_integer(
                cell_model["min_chsh_pairs_for_chsh_abort"]
            ),
        },
    }
    for size in sizes:
        config = replace(
            maximum_config,
            requested_pairs_per_turn=size,
        )
        metadata = {
            "selection_digest_blake2b_128": str(
                selection_digest_blake2b_128
            ),
            "sampling_design": pairing_contract["coupling"],
            "samples_per_cell_per_split": shape[2],
            "root_seed": int(thinning_root_seed),
            "game_mode": config.game_mode,
            "compiled_histogram_runtime_version": source.metadata[
                "compiled_histogram_runtime_version"
            ],
            "feedback_audit_seed_disjoint": True,
            "evaluation_partition_loaded": bool(
                evaluation_partition_loaded
            ),
            "physics_config": dict(source.metadata["physics_config"]),
            "paired_block_contract": pairing_contract,
        }
        endpoint_banks[size] = HistogramEndpointEvidenceBank(
            case_id=source.case_id,
            graph_id=source.graph_id,
            assignment_id=source.assignment_id,
            route_ids=source.route_ids,
            action_ids=source.action_ids,
            requested_pairs_per_draw=size,
            feedback=EndpointEvidenceSampleBlock(
                **endpoint_arrays[size]["feedback"]
            ),
            audit=EndpointEvidenceSampleBlock(
                **endpoint_arrays[size]["audit"]
            ),
            metadata=metadata,
        )
        payoff_banks[size] = HistogramPayoffBank(
            case_id=source.case_id,
            graph_id=source.graph_id,
            assignment_id=source.assignment_id,
            route_ids=source.route_ids,
            action_ids=source.action_ids,
            action_costs=action_costs.copy(),
            feedback=PayoffSampleBlock(
                **payoff_arrays[size]["feedback"]
            ),
            audit=PayoffSampleBlock(
                **payoff_arrays[size]["audit"]
            ),
            metadata={
                **metadata,
                "sampling_purpose": (
                    "paired_block_response_calibration"
                ),
                "requested_pairs_per_draw": size,
                "eve_observation_count_semantics": (
                    "not retained by endpoint thinning; zero placeholder "
                    "excluded from this response study"
                ),
            },
        )

    summary = {
        **pairing_contract,
        "case_id": source.case_id,
        "route_count": route_count,
        "action_count": action_count,
        "draws_per_cell_per_split": shape[2],
        "per_block_size": {
            str(size): _block_size_summary(
                endpoint_banks[size],
                min_key_pairs=pairing_contract[
                    "protocol_decision_thresholds"
                ]["min_key_pairs_for_qber_abort"],
                min_chsh_pairs=pairing_contract[
                    "protocol_decision_thresholds"
                ]["min_chsh_pairs_for_chsh_abort"],
            )
            for size in sizes
        },
    }
    return PairedBlockSizeBanks(
        maximum_block_size=maximum,
        block_sizes=sizes,
        endpoint_banks=endpoint_banks,
        payoff_banks=payoff_banks,
        summary=summary,
    )


def _cell_model(
    case: HistogramGameCase,
    actions: Sequence[dict[str, Any]],
    physics: HistogramPhysicsPool,
) -> dict[str, np.ndarray]:
    route_count = len(case.routes)
    action_count = len(actions)
    latent = np.zeros((route_count, action_count), dtype=bool)
    target_active = np.zeros(
        (route_count, action_count),
        dtype=bool,
    )
    qber_threshold = np.zeros(
        (route_count, action_count),
        dtype=float,
    )
    chsh_threshold = np.zeros(
        (route_count, action_count),
        dtype=float,
    )
    min_key_pairs = np.zeros(
        (route_count, action_count),
        dtype=np.int64,
    )
    min_chsh_pairs = np.zeros(
        (route_count, action_count),
        dtype=np.int64,
    )
    for route_index in range(route_count):
        for action_index, action in enumerate(actions):
            result = physics.step(
                route_index,
                action,
                seed=20_260_721,
                step_index=route_index * action_count + action_index,
            )
            expected_qber = result.expected_qber_from_histogram
            expected_chsh = result.chsh_S_expected
            latent[route_index, action_index] = bool(
                expected_qber is not None
                and expected_chsh is not None
                and float(expected_qber)
                <= float(result.qber_abort_threshold)
                and float(expected_chsh)
                >= float(result.chsh_abort_threshold)
            )
            target_active[route_index, action_index] = bool(
                result.target_active
            )
            qber_threshold[route_index, action_index] = float(
                result.qber_abort_threshold
            )
            chsh_threshold[route_index, action_index] = float(
                result.chsh_abort_threshold
            )
            min_key_pairs[route_index, action_index] = int(
                result.min_key_pairs_for_qber_abort
            )
            min_chsh_pairs[route_index, action_index] = int(
                result.min_chsh_pairs_for_chsh_abort
            )
    calibrations = physics.calibrations
    reference_rate = physics.reference_service_rate_hz
    return {
        "latent_bell_qualified": latent,
        "target_active": target_active,
        "qber_threshold": qber_threshold,
        "chsh_threshold": chsh_threshold,
        "min_key_pairs_for_qber_abort": min_key_pairs,
        "min_chsh_pairs_for_chsh_abort": min_chsh_pairs,
        "clean_delivery_probability": np.asarray([
            row.clean_delivery_probability for row in calibrations
        ], dtype=float),
        "route_rate_ratio": np.asarray([
            row.service_rate_hz / reference_rate for row in calibrations
        ], dtype=float),
        "reference_service_rate_hz": np.asarray(
            [reference_rate],
            dtype=float,
        ),
    }


def _endpoint_categories(
    block: EndpointEvidenceSampleBlock,
    index: tuple[int, int, int],
    *,
    maximum_block_size: int,
) -> np.ndarray:
    delivered = int(block.delivered_pair_count[index])
    key_pairs = int(block.key_pair_count[index])
    key_errors = int(block.key_error_count[index])
    totals = np.asarray(
        block.chsh_setting_total_counts[index],
        dtype=np.int64,
    )
    same = np.asarray(
        block.chsh_setting_same_counts[index],
        dtype=np.int64,
    )
    categories = [
        maximum_block_size - delivered,
        delivered - key_pairs - int(np.sum(totals)),
        key_pairs - key_errors,
        key_errors,
    ]
    for total, equal in zip(totals, same):
        categories.extend([int(equal), int(total - equal)])
    values = np.asarray(categories, dtype=np.int64)
    if np.any(values < 0) or int(np.sum(values)) != maximum_block_size:
        raise ValueError("source endpoint counts do not form a transcript")
    return values


def _nested_thinnings(
    maximum_categories: np.ndarray,
    *,
    sizes: Sequence[int],
    seed: int,
) -> dict[int, np.ndarray]:
    ordered = tuple(sorted(set(map(int, sizes)), reverse=True))
    if ordered[0] != int(np.sum(maximum_categories)):
        raise ValueError(
            "largest block size must equal the source category total"
        )
    rng = np.random.default_rng(int(seed))
    current = np.asarray(maximum_categories, dtype=np.int64).copy()
    current_size = ordered[0]
    output = {current_size: current.copy()}
    for next_size in ordered[1:]:
        if not 0 < next_size < current_size:
            raise ValueError("block sizes must be strictly nested")
        current = rng.multivariate_hypergeometric(
            current,
            next_size,
        )
        current_size = next_size
        output[current_size] = current.copy()
    return output


def _record_derived_draw(
    categories: np.ndarray,
    *,
    size: int,
    index: tuple[int, int, int],
    route_index: int,
    action_index: int,
    cell_model: Mapping[str, np.ndarray],
    action_costs: np.ndarray,
    maximum_cost: float,
    endpoint_arrays: dict[str, np.ndarray],
    payoff_arrays: dict[str, np.ndarray],
) -> None:
    delivered = int(size - categories[0])
    key_correct = int(categories[2])
    key_error = int(categories[3])
    key_pairs = key_correct + key_error
    chsh_same = np.asarray(
        [categories[4 + 2 * offset] for offset in range(4)],
        dtype=np.int64,
    )
    chsh_diff = np.asarray(
        [categories[5 + 2 * offset] for offset in range(4)],
        dtype=np.int64,
    )
    chsh_totals = chsh_same + chsh_diff
    qber_threshold = float(
        cell_model["qber_threshold"][route_index, action_index]
    )
    chsh_threshold = float(
        cell_model["chsh_threshold"][route_index, action_index]
    )
    min_key_pairs = int(
        cell_model["min_key_pairs_for_qber_abort"][
            route_index,
            action_index,
        ]
    )
    min_chsh_pairs = int(
        cell_model["min_chsh_pairs_for_chsh_abort"][
            route_index,
            action_index,
        ]
    )
    qber_abort = bool(
        key_pairs >= min_key_pairs
        and key_error / key_pairs > qber_threshold
    )
    chsh_s = _chsh_s(chsh_same, chsh_totals)
    chsh_abort = bool(
        not qber_abort
        and chsh_s is not None
        and int(np.sum(chsh_totals)) >= min_chsh_pairs
        and chsh_s < chsh_threshold
    )
    if delivered <= 0:
        outcome = "delivery_failure"
    elif qber_abort:
        outcome = "qber_abort"
    elif chsh_abort:
        outcome = "chsh_abort"
    else:
        outcome = "accepted"
    accepted = outcome == "accepted"
    clean_delivery = float(
        cell_model["clean_delivery_probability"][route_index]
    )
    rate_ratio = float(
        cell_model["route_rate_ratio"][route_index]
    )
    relative_yield = min(
        1.0,
        max(0.0, (delivered / size) / clean_delivery),
    )
    reward_if_qualified = rate_ratio * relative_yield
    alice_reward = reward_if_qualified if accepted else 0.0
    latent = bool(
        cell_model["latent_bell_qualified"][
            route_index,
            action_index,
        ]
    )
    corrected_reward = (
        reward_if_qualified if latent and delivered > 0 else 0.0
    )
    if delivered <= 0:
        signal_class = "delivery_failure_not_quality_signal"
        classifiable = False
        false_signal = False
    else:
        classifiable = True
        false_signal = accepted != latent
        if accepted and latent:
            signal_class = "true_accept"
        elif accepted:
            signal_class = "false_accept"
        elif latent:
            signal_class = "false_abort"
        else:
            signal_class = "true_abort"
    raw_eve_reward = (
        1.0 - alice_reward - float(action_costs[action_index])
    )
    eve_reward = normalize_eve_bandit_reward(
        raw_eve_reward,
        maximum_cost=maximum_cost,
    )
    reference_rate = float(
        cell_model["reference_service_rate_hz"][0]
    )

    endpoint_arrays["alice_reward"][index] = alice_reward
    endpoint_arrays["bell_qualified_rate_hz"][index] = (
        alice_reward * reference_rate
    )
    endpoint_arrays["delivered_pair_count"][index] = delivered
    endpoint_arrays["key_pair_count"][index] = key_pairs
    endpoint_arrays["key_error_count"][index] = key_error
    endpoint_arrays["chsh_setting_total_counts"][index] = (
        chsh_totals
    )
    endpoint_arrays["chsh_setting_same_counts"][index] = chsh_same
    endpoint_arrays["public_outcome_code"][index] = (
        PUBLIC_OUTCOME_CODES[outcome]
    )

    payoff_arrays["alice_reward"][index] = alice_reward
    payoff_arrays["corrected_alice_reward"][index] = corrected_reward
    payoff_arrays["eve_bandit_reward"][index] = eve_reward
    payoff_arrays["bell_qualified_rate_hz"][index] = (
        alice_reward * reference_rate
    )
    payoff_arrays["false_signal"][index] = false_signal
    payoff_arrays["classifiable_signal"][index] = classifiable
    payoff_arrays["target_active"][index] = bool(
        cell_model["target_active"][route_index, action_index]
    )
    payoff_arrays["eve_observation_count"][index] = 0
    payoff_arrays["information_observed"][index] = False
    payoff_arrays["public_outcome_code"][index] = (
        PUBLIC_OUTCOME_CODES[outcome]
    )
    payoff_arrays["signal_class_code"][index] = (
        SIGNAL_CLASS_CODES[signal_class]
    )


def _chsh_s(
    same: np.ndarray,
    totals: np.ndarray,
) -> float | None:
    if np.any(totals <= 0):
        return None
    correlations = (2.0 * same - totals) / totals
    return -float(
        correlations[0]
        + correlations[1]
        + correlations[2]
        - correlations[3]
    )


def _empty_endpoint_arrays(
    shape: tuple[int, int, int],
) -> dict[str, np.ndarray]:
    chsh_shape = shape + (4,)
    return {
        "alice_reward": np.empty(shape, dtype=np.float64),
        "bell_qualified_rate_hz": np.empty(shape, dtype=np.float64),
        "delivered_pair_count": np.empty(shape, dtype=np.int64),
        "key_pair_count": np.empty(shape, dtype=np.int64),
        "key_error_count": np.empty(shape, dtype=np.int64),
        "chsh_setting_total_counts": np.empty(
            chsh_shape,
            dtype=np.int64,
        ),
        "chsh_setting_same_counts": np.empty(
            chsh_shape,
            dtype=np.int64,
        ),
        "public_outcome_code": np.empty(shape, dtype=np.int8),
    }


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
        "eve_observation_count": np.zeros(shape, dtype=np.int32),
        "information_observed": np.zeros(shape, dtype=np.bool_),
        "public_outcome_code": np.empty(shape, dtype=np.int8),
        "signal_class_code": np.empty(shape, dtype=np.int8),
    }


def _block_size_summary(
    bank: HistogramEndpointEvidenceBank,
    *,
    min_key_pairs: int,
    min_chsh_pairs: int,
) -> dict[str, Any]:
    no_attack = bank.action_ids.index("no_attack")
    block = bank.audit
    key = np.asarray(block.key_pair_count[:, no_attack, :])
    chsh = np.sum(
        block.chsh_setting_total_counts[:, no_attack, :, :],
        axis=2,
    )
    setting_totals = np.asarray(
        block.chsh_setting_total_counts[:, no_attack, :, :]
    )
    accepted = (
        block.public_outcome_code[:, no_attack, :]
        == PUBLIC_OUTCOME_CODES["accepted"]
    )
    qber_abort = (
        block.public_outcome_code[:, no_attack, :]
        == PUBLIC_OUTCOME_CODES["qber_abort"]
    )
    chsh_abort = (
        block.public_outcome_code[:, no_attack, :]
        == PUBLIC_OUTCOME_CODES["chsh_abort"]
    )
    return {
        "requested_pairs_per_draw": bank.requested_pairs_per_draw,
        "mean_key_trials": float(np.mean(key)),
        "mean_chsh_trials": float(np.mean(chsh)),
        "mean_chsh_trials_per_setting": float(
            np.mean(chsh) / 4.0
        ),
        "min_key_pairs_for_qber_abort": int(min_key_pairs),
        "min_chsh_pairs_for_chsh_abort": int(min_chsh_pairs),
        "key_trial_count_below_abort_minimum_fraction": float(
            np.mean(key < min_key_pairs)
        ),
        "chsh_trial_count_below_abort_minimum_fraction": float(
            np.mean(chsh < min_chsh_pairs)
        ),
        "missing_chsh_setting_fraction": float(
            np.mean(np.any(setting_totals == 0, axis=2))
        ),
        "clean_acceptance_fraction": float(np.mean(accepted)),
        "clean_qber_abort_fraction": float(np.mean(qber_abort)),
        "clean_chsh_abort_fraction": float(np.mean(chsh_abort)),
        "endpoint_digest_blake2b_128": bank.content_digest(),
    }


def _unique_integer(values: np.ndarray) -> int:
    unique = np.unique(np.asarray(values, dtype=np.int64))
    if unique.shape != (1,):
        raise ValueError(
            "paired blocks require one protocol trial threshold per case"
        )
    return int(unique[0])


def _stable_seed(root_seed: int, *parts: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(root_seed).to_bytes(8, "little", signed=True))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") % (2**31 - 1)
