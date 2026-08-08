"""Fail-closed analysis of the matched 500k Prime-and-Probe detector bundle."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .prime_probe import PRIME_PROBE_RESULT_SCHEMA
from .prime_probe_detection_corpus import (
    PRIME_PROBE_DETECTION_CORPUS_SCHEMA,
    _validate_ordered_detection_run,
    _validate_shuffled_run,
)


PRIME_PROBE_DETECTION_ANALYSIS_SCHEMA = (
    "histogram_prime_probe_detection_analysis.v1"
)
E91_COUNT_DETECTOR_ARM = "e91_measurement_counts_only"
ORDERED_CONDITION = "ordered_prime_probe"
SHUFFLED_CONDITION = "shuffled_identical_action_multiset"
RECOVERY_CRITERIA = ("service", "target_mass", "policy_tv", "joint")

PairKey = tuple[str, str, int]


class PrimeProbeDetectionAnalysisError(ValueError):
    """The bundle is incomplete or violates its declared pairing contract."""


def analyze_prime_probe_detection_bundle(
    run_dir: str | Path,
) -> dict[str, Any]:
    """Validate and summarize one completed detector-aware study bundle.

    The function is read-only.  It rejects incomplete JSONL files, missing or
    duplicate condition pairs, stale summary counts, unmatched action
    multisets, and comparison rows that do not reproduce their source runs.
    """

    root = Path(run_dir)
    manifest = _read_json(root / "run_manifest.json")
    summary = _read_json(root / "summary.json")
    runs = _read_jsonl(root / "results.jsonl")
    comparisons = _read_jsonl(root / "paired_comparisons.jsonl")

    expected_pairs, defender_modes = _validate_manifest_and_summary(
        manifest,
        summary,
        runs,
        comparisons,
    )
    run_pairs = _validate_runs(
        manifest,
        runs,
        expected_pairs=expected_pairs,
        defender_modes=defender_modes,
    )
    comparison_rows = _validate_comparisons(
        manifest,
        comparisons,
        run_pairs=run_pairs,
        expected_pairs=expected_pairs,
    )

    learners: dict[str, Any] = {}
    for defender_label in sorted(defender_modes):
        keys = sorted(
            (
                key
                for key in expected_pairs
                if key[1] == defender_label
            ),
            key=lambda key: (key[0], key[2]),
        )
        ordered = [
            run_pairs[key][ORDERED_CONDITION]
            for key in keys
        ]
        shuffled = [
            run_pairs[key][SHUFFLED_CONDITION]
            for key in keys
        ]
        paired = [comparison_rows[key] for key in keys]
        learners[defender_label] = _learner_summary(
            defender_label,
            defender_modes[defender_label],
            ordered,
            shuffled,
            paired,
        )

    return {
        "schema": PRIME_PROBE_DETECTION_ANALYSIS_SCHEMA,
        "source_schema": PRIME_PROBE_DETECTION_CORPUS_SCHEMA,
        "complete": True,
        "detector_arm": E91_COUNT_DETECTOR_ARM,
        "quartile_method": "numpy.quantile(method='linear')",
        "condition_run_count": len(runs),
        "paired_comparison_count": len(comparisons),
        "learner_count": len(learners),
        "resume_contract_digest_blake2b_128": manifest[
            "resume_contract_digest_blake2b_128"
        ],
        "learners": learners,
    }


def _validate_manifest_and_summary(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> tuple[set[PairKey], dict[str, str]]:
    if manifest.get("schema") != PRIME_PROBE_DETECTION_CORPUS_SCHEMA:
        raise PrimeProbeDetectionAnalysisError(
            "Prime-and-Probe detector manifest schema changed"
        )
    if summary.get("schema") != PRIME_PROBE_DETECTION_CORPUS_SCHEMA:
        raise PrimeProbeDetectionAnalysisError(
            "Prime-and-Probe detector summary schema changed"
        )
    if summary.get("complete") is not True:
        raise PrimeProbeDetectionAnalysisError(
            "Prime-and-Probe detector bundle is not complete"
        )

    cases = list(manifest.get("cases") or ())
    defenders = list(manifest.get("defenders") or ())
    seeds = list(manifest.get("seeds") or ())
    graph_ids = [str(row["graph_id"]) for row in cases]
    defender_labels = [str(row["label"]) for row in defenders]
    seed_values = [_integer(seed, "manifest seed") for seed in seeds]
    if not graph_ids or not defender_labels or not seed_values:
        raise PrimeProbeDetectionAnalysisError(
            "manifest case, defender, and seed axes must be nonempty"
        )
    if (
        len(set(graph_ids)) != len(graph_ids)
        or len(set(defender_labels)) != len(defender_labels)
        or len(set(seed_values)) != len(seed_values)
    ):
        raise PrimeProbeDetectionAnalysisError(
            "manifest case, defender, or seed identity is duplicated"
        )
    defender_modes = {
        str(row["label"]): str(row["alice_learner_mode"])
        for row in defenders
    }
    expected_pairs = {
        (graph_id, defender_label, seed)
        for graph_id in graph_ids
        for defender_label in defender_labels
        for seed in seed_values
    }
    expected_pair_count = len(expected_pairs)
    declared = {
        "expected_condition_runs": 2 * expected_pair_count,
        "expected_paired_comparisons": expected_pair_count,
        "expected_detector_runs": expected_pair_count,
    }
    for field, expected in declared.items():
        if _integer(manifest.get(field), f"manifest {field}") != expected:
            raise PrimeProbeDetectionAnalysisError(
                f"manifest {field} does not match its declared axes"
            )

    summary_expected = {
        "graph_count": len(graph_ids),
        "condition_run_count": len(runs),
        "expected_condition_run_count": 2 * expected_pair_count,
        "paired_comparison_count": len(comparisons),
        "expected_paired_comparison_count": expected_pair_count,
        "detector_run_count": expected_pair_count,
        "expected_detector_run_count": expected_pair_count,
    }
    for field, expected in summary_expected.items():
        if _integer(summary.get(field), f"summary {field}") != expected:
            raise PrimeProbeDetectionAnalysisError(
                f"summary {field} does not match the completed files"
            )
    if len(runs) != 2 * expected_pair_count:
        raise PrimeProbeDetectionAnalysisError(
            "results.jsonl line count is incomplete"
        )
    if len(comparisons) != expected_pair_count:
        raise PrimeProbeDetectionAnalysisError(
            "paired_comparisons.jsonl line count is incomplete"
        )
    return expected_pairs, defender_modes


def _validate_runs(
    manifest: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    *,
    expected_pairs: set[PairKey],
    defender_modes: Mapping[str, str],
) -> dict[PairKey, dict[str, Mapping[str, Any]]]:
    total_turns = _integer(
        manifest["schedule"]["total_turns"],
        "manifest total turns",
    )
    resume_digest = str(
        manifest["resume_contract_digest_blake2b_128"]
    )
    pairs: dict[PairKey, dict[str, Mapping[str, Any]]] = {}
    for row in runs:
        if row.get("schema") != PRIME_PROBE_RESULT_SCHEMA:
            raise PrimeProbeDetectionAnalysisError(
                "result row schema changed"
            )
        key = _pair_key(row)
        if key not in expected_pairs:
            raise PrimeProbeDetectionAnalysisError(
                f"unknown result identity {key!r}"
            )
        condition = str(row.get("condition"))
        if condition not in {ORDERED_CONDITION, SHUFFLED_CONDITION}:
            raise PrimeProbeDetectionAnalysisError(
                f"unknown result condition {condition!r}"
            )
        pair = pairs.setdefault(key, {})
        if condition in pair:
            raise PrimeProbeDetectionAnalysisError(
                f"duplicate {condition} result for {key!r}"
            )
        if _integer(row.get("turns"), "result turns") != total_turns:
            raise PrimeProbeDetectionAnalysisError(
                f"result horizon changed for {key!r}"
            )
        if row.get("resume_contract_digest_blake2b_128") != resume_digest:
            raise PrimeProbeDetectionAnalysisError(
                f"result resume contract changed for {key!r}"
            )
        learner = row.get("alice_learner") or {}
        if learner.get("mode") != defender_modes[key[1]]:
            raise PrimeProbeDetectionAnalysisError(
                f"result learner mode changed for {key!r}"
            )
        if condition == ORDERED_CONDITION:
            _validate_ordered_detection_run(row, manifest)
            _validate_count_outcome(row, manifest)
            trigger = (row.get("probe_trigger") or {}).get("triggered")
            if not isinstance(trigger, bool):
                raise PrimeProbeDetectionAnalysisError(
                    f"Eve Probe decision is not Boolean for {key!r}"
                )
        else:
            _validate_shuffled_run(row, manifest)
        _validate_recovery(row, manifest, key)
        pair[condition] = row

    expected_conditions = {ORDERED_CONDITION, SHUFFLED_CONDITION}
    if set(pairs) != expected_pairs or any(
        set(pair) != expected_conditions for pair in pairs.values()
    ):
        raise PrimeProbeDetectionAnalysisError(
            "results do not contain one complete condition pair per identity"
        )
    return pairs


def _validate_count_outcome(
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    detector = run["alice_endpoint_detector"]
    outcomes = detector["outcomes"]
    if E91_COUNT_DETECTOR_ARM not in outcomes:
        raise PrimeProbeDetectionAnalysisError(
            "ordered result omits the E91-count-only detector"
        )
    outcome = outcomes[E91_COUNT_DETECTOR_ARM]
    total_turns = _integer(
        manifest["schedule"]["total_turns"],
        "manifest total turns",
    )
    prime_start = _integer(
        manifest["schedule"]["warmup_turns"],
        "manifest warmup turns",
    )
    requested_pairs = _integer(
        manifest["source_corpus_plan"]["requested_pairs_per_turn"],
        "requested pairs per turn",
    )
    alarm_turn_raw = outcome.get("alarm_turn")
    pre_prime = outcome.get("pre_prime_false_alarm")
    right_censored = outcome.get("right_censored")
    if not isinstance(pre_prime, bool) or not isinstance(
        right_censored, bool
    ):
        raise PrimeProbeDetectionAnalysisError(
            "detector alarm flags must be Boolean"
        )
    observed_through = _integer(
        outcome.get("observed_through_turn"),
        "detector observed-through turn",
    )
    if alarm_turn_raw is None:
        if (
            pre_prime
            or not right_censored
            or outcome.get("detection_delay_from_prime_start_turns")
            is not None
            or outcome.get("requested_pairs_from_prime_start_to_alarm")
            is not None
            or outcome.get("leading_channel_at_alarm") is not None
            or observed_through != total_turns
        ):
            raise PrimeProbeDetectionAnalysisError(
                "uncensored or partially populated no-alarm detector row"
            )
        return

    alarm_turn = _integer(alarm_turn_raw, "detector alarm turn")
    if not 1 <= alarm_turn <= total_turns:
        raise PrimeProbeDetectionAnalysisError(
            "detector alarm turn is outside the study horizon"
        )
    expected_pre_prime = alarm_turn <= prime_start
    if pre_prime != expected_pre_prime or right_censored:
        raise PrimeProbeDetectionAnalysisError(
            "detector alarm classification is inconsistent"
        )
    if observed_through < alarm_turn:
        raise PrimeProbeDetectionAnalysisError(
            "detector stopped before its recorded alarm"
        )
    leading_channel = outcome.get("leading_channel_at_alarm")
    leading = outcome.get("leading_alternative_at_alarm")
    if (
        not isinstance(leading_channel, str)
        or not leading_channel
        or not isinstance(leading, Mapping)
        or leading.get("channel") != leading_channel
    ):
        raise PrimeProbeDetectionAnalysisError(
            "alarmed detector row lacks a consistent leading channel"
        )
    delay = outcome.get("detection_delay_from_prime_start_turns")
    requested_to_alarm = outcome.get(
        "requested_pairs_from_prime_start_to_alarm"
    )
    if pre_prime:
        if delay is not None or requested_to_alarm is not None:
            raise PrimeProbeDetectionAnalysisError(
                "pre-Prime alarm has a post-Prime delay"
            )
    else:
        expected_delay = alarm_turn - prime_start
        if (
            _integer(delay, "detector delay") != expected_delay
            or _integer(
                requested_to_alarm,
                "requested pairs to alarm",
            )
            != expected_delay * requested_pairs
        ):
            raise PrimeProbeDetectionAnalysisError(
                "detector delay or requested-pair count is inconsistent"
            )


def _validate_recovery(
    run: Mapping[str, Any],
    manifest: Mapping[str, Any],
    key: PairKey,
) -> None:
    recovery = run.get("recovery")
    if not isinstance(recovery, Mapping):
        raise PrimeProbeDetectionAnalysisError(
            f"result omits recovery telemetry for {key!r}"
        )
    censor = _integer(
        manifest["schedule"]["recovery_turns"],
        "manifest recovery turns",
    )
    for criterion in RECOVERY_CRITERIA:
        value = recovery.get(criterion)
        if not isinstance(value, Mapping):
            raise PrimeProbeDetectionAnalysisError(
                f"recovery criterion {criterion!r} is missing for {key!r}"
            )
        recovered = value.get("recovered")
        right_censored = value.get("right_censored")
        if (
            not isinstance(recovered, bool)
            or not isinstance(right_censored, bool)
            or recovered == right_censored
        ):
            raise PrimeProbeDetectionAnalysisError(
                f"recovery censoring is inconsistent for {key!r}"
            )
        latency = _integer(
            value.get("latency_turns"),
            f"{criterion} recovery latency",
        )
        if not 0 <= latency <= censor:
            raise PrimeProbeDetectionAnalysisError(
                f"recovery latency is outside the horizon for {key!r}"
            )
        if right_censored and latency != censor:
            raise PrimeProbeDetectionAnalysisError(
                f"right-censored recovery does not use the censor for {key!r}"
            )


def _validate_comparisons(
    manifest: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    *,
    run_pairs: Mapping[
        PairKey, Mapping[str, Mapping[str, Any]]
    ],
    expected_pairs: set[PairKey],
) -> dict[PairKey, Mapping[str, Any]]:
    rows: dict[PairKey, Mapping[str, Any]] = {}
    for comparison in comparisons:
        key = _pair_key(comparison)
        if key not in expected_pairs or key in rows:
            raise PrimeProbeDetectionAnalysisError(
                f"missing, duplicate, or unknown comparison identity {key!r}"
            )
        ordered = run_pairs[key][ORDERED_CONDITION]
        shuffled = run_pairs[key][SHUFFLED_CONDITION]
        if comparison.get("identical_action_multiset") is not True:
            raise PrimeProbeDetectionAnalysisError(
                f"comparison is not action-multiset matched for {key!r}"
            )
        if ordered.get("action_multiset") != shuffled.get("action_multiset"):
            raise PrimeProbeDetectionAnalysisError(
                f"source run action multisets differ for {key!r}"
            )
        if (
            comparison.get("alice_learner_mode")
            != ordered["alice_learner"]["mode"]
        ):
            raise PrimeProbeDetectionAnalysisError(
                f"comparison learner mode differs for {key!r}"
            )
        ordered_triggered = ordered["probe_trigger"]["triggered"]
        if comparison.get("ordered_triggered") is not ordered_triggered:
            raise PrimeProbeDetectionAnalysisError(
                f"comparison Probe decision differs for {key!r}"
            )
        _require_matching_difference(
            comparison,
            "ordered_minus_shuffled_mean_alice_reward",
            ordered,
            shuffled,
            "mean_alice_reward",
            key,
        )
        _require_matching_difference(
            comparison,
            "ordered_minus_shuffled_mean_eve_public_utility",
            ordered,
            shuffled,
            "mean_eve_public_utility",
            key,
        )
        if (
            ordered.get("audit_expected_alice_reward") is not None
            or shuffled.get("audit_expected_alice_reward") is not None
            or comparison.get(
                "ordered_minus_shuffled_audit_expected_alice_reward"
            )
            is not None
        ):
            raise PrimeProbeDetectionAnalysisError(
                "detector rerun unexpectedly reuses live transcripts for "
                f"post-hoc audit scoring in {key!r}"
            )
        if comparison.get("ordered_recovery") != ordered.get("recovery"):
            raise PrimeProbeDetectionAnalysisError(
                f"comparison recovery differs from its ordered run for {key!r}"
            )
        expected_plan_digest = manifest["frozen_plans"][key[0]][
            "plan_digest_blake2b_128"
        ]
        if (
            comparison.get("source_plan_digest_blake2b_128")
            != expected_plan_digest
            or ordered.get("source_plan_digest_blake2b_128")
            != expected_plan_digest
            or shuffled.get("source_plan_digest_blake2b_128")
            != expected_plan_digest
        ):
            raise PrimeProbeDetectionAnalysisError(
                f"source plan digest differs for {key!r}"
            )
        rows[key] = comparison
    if set(rows) != expected_pairs:
        raise PrimeProbeDetectionAnalysisError(
            "comparison identities do not match result pairs"
        )
    return rows


def _require_matching_difference(
    comparison: Mapping[str, Any],
    comparison_field: str,
    ordered: Mapping[str, Any],
    shuffled: Mapping[str, Any],
    run_field: str,
    key: PairKey,
) -> None:
    reported = _finite_float(
        comparison.get(comparison_field),
        comparison_field,
    )
    expected = (
        _finite_float(ordered.get(run_field), run_field)
        - _finite_float(shuffled.get(run_field), run_field)
    )
    if not math.isclose(
        reported,
        expected,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise PrimeProbeDetectionAnalysisError(
            f"comparison field {comparison_field} does not reproduce {key!r}"
        )


def _learner_summary(
    defender_label: str,
    learner_mode: str,
    ordered: Sequence[Mapping[str, Any]],
    shuffled: Sequence[Mapping[str, Any]],
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    outcomes = [
        row["alice_endpoint_detector"]["outcomes"][
            E91_COUNT_DETECTOR_ARM
        ]
        for row in ordered
    ]
    pre_prime = [
        outcome
        for outcome in outcomes
        if outcome["pre_prime_false_alarm"]
    ]
    post_prime = [
        outcome
        for outcome in outcomes
        if (
            outcome["alarm_turn"] is not None
            and not outcome["pre_prime_false_alarm"]
        )
    ]
    censored = [
        outcome
        for outcome in outcomes
        if outcome["right_censored"]
    ]
    alarmed = [
        outcome
        for outcome in outcomes
        if outcome["alarm_turn"] is not None
    ]
    reward_differences = [
        _finite_float(
            row["ordered_minus_shuffled_mean_alice_reward"],
            "ordered-minus-shuffled Alice reward",
        )
        for row in comparisons
    ]
    triggers = [
        bool(row["probe_trigger"]["triggered"])
        for row in ordered
    ]
    return {
        "defender_label": defender_label,
        "alice_learner_mode": learner_mode,
        "run_count": len(ordered),
        "requested_pairs_per_turn": _integer(
            ordered[0]["config"]["requested_pairs_per_turn"],
            "requested pairs per turn",
        ),
        "alarm_counts": {
            "total": len(alarmed),
            "pre_prime_false_alarm": len(pre_prime),
            "post_prime": len(post_prime),
            "right_censored_no_alarm": len(censored),
        },
        "post_prime_delay_turns": _five_number_summary(
            [
                outcome["detection_delay_from_prime_start_turns"]
                for outcome in post_prime
            ]
        ),
        "post_prime_requested_pairs_to_alarm": _five_number_summary(
            [
                outcome["requested_pairs_from_prime_start_to_alarm"]
                for outcome in post_prime
            ]
        ),
        "leading_channel_counts": {
            "all_alarms": _counter(
                outcome["leading_channel_at_alarm"]
                for outcome in alarmed
            ),
            "pre_prime_false_alarms": _counter(
                outcome["leading_channel_at_alarm"]
                for outcome in pre_prime
            ),
            "post_prime_alarms": _counter(
                outcome["leading_channel_at_alarm"]
                for outcome in post_prime
            ),
        },
        "eve_private_probe_decisions": {
            "triggered": sum(triggers),
            "not_triggered": len(triggers) - sum(triggers),
        },
        "realized_ordered_minus_shuffled_alice_reward": {
            "five_number_summary": _five_number_summary(
                reward_differences
            ),
            "sign_counts": {
                "negative": sum(value < 0.0 for value in reward_differences),
                "zero": sum(value == 0.0 for value in reward_differences),
                "positive": sum(value > 0.0 for value in reward_differences),
            },
        },
        "recovery": _recovery_summary(ordered, shuffled),
    }


def _recovery_summary(
    ordered: Sequence[Mapping[str, Any]],
    shuffled: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    shuffled_by_key = {_pair_key(row): row for row in shuffled}
    result: dict[str, Any] = {}
    for criterion in RECOVERY_CRITERIA:
        states: Counter[tuple[bool, bool]] = Counter()
        ordered_recovered = 0
        shuffled_recovered = 0
        for ordered_row in ordered:
            shuffled_row = shuffled_by_key[_pair_key(ordered_row)]
            ordered_status = bool(
                ordered_row["recovery"][criterion]["recovered"]
            )
            shuffled_status = bool(
                shuffled_row["recovery"][criterion]["recovered"]
            )
            ordered_recovered += ordered_status
            shuffled_recovered += shuffled_status
            states[(ordered_status, shuffled_status)] += 1
        pair_count = len(ordered)
        result[criterion] = {
            "pair_count": pair_count,
            "ordered": {
                "recovered": ordered_recovered,
                "right_censored": pair_count - ordered_recovered,
            },
            "shuffled": {
                "recovered": shuffled_recovered,
                "right_censored": pair_count - shuffled_recovered,
            },
            "paired_status": {
                "both_recovered": states[(True, True)],
                "ordered_only_recovered": states[(True, False)],
                "shuffled_only_recovered": states[(False, True)],
                "neither_recovered": states[(False, False)],
            },
        }
    return result


def _five_number_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(
        [_finite_float(value, "summary value") for value in values],
        dtype=float,
    )
    if array.size == 0:
        return {
            "count": 0,
            "minimum": None,
            "q25": None,
            "median": None,
            "q75": None,
            "maximum": None,
        }
    return {
        "count": int(array.size),
        "minimum": float(np.min(array)),
        "q25": float(np.quantile(array, 0.25, method="linear")),
        "median": float(np.median(array)),
        "q75": float(np.quantile(array, 0.75, method="linear")),
        "maximum": float(np.max(array)),
    }


def _counter(values: Sequence[str] | Any) -> dict[str, int]:
    return dict(sorted(Counter(map(str, values)).items()))


def _pair_key(row: Mapping[str, Any]) -> PairKey:
    try:
        graph_id = str(row["graph_id"])
        defender_label = str(row["defender_label"])
        seed = _integer(row["seed"], "row seed")
    except KeyError as exc:
        raise PrimeProbeDetectionAnalysisError(
            f"row omits pairing field {exc.args[0]!r}"
        ) from exc
    return graph_id, defender_label, seed


def _integer(value: Any, label: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
    ):
        raise PrimeProbeDetectionAnalysisError(
            f"{label} must be an integer"
        )
    return int(value)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise PrimeProbeDetectionAnalysisError(
            f"{label} must be numeric"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PrimeProbeDetectionAnalysisError(
            f"{label} must be numeric"
        ) from exc
    if not math.isfinite(number):
        raise PrimeProbeDetectionAnalysisError(
            f"{label} must be finite"
        )
    return number


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrimeProbeDetectionAnalysisError(
            f"cannot read valid JSON from {path}"
        ) from exc
    if not isinstance(value, dict):
        raise PrimeProbeDetectionAnalysisError(
            f"{path} must contain one JSON object"
        )
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PrimeProbeDetectionAnalysisError(
            f"cannot read {path}"
        ) from exc
    if not lines or any(not line.strip() for line in lines):
        raise PrimeProbeDetectionAnalysisError(
            f"{path} is empty or contains blank lines"
        )
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PrimeProbeDetectionAnalysisError(
                f"{path}:{line_number} is not valid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise PrimeProbeDetectionAnalysisError(
                f"{path}:{line_number} is not a JSON object"
            )
        rows.append(row)
    return rows
