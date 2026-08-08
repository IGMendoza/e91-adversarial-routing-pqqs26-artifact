#!/usr/bin/env python3
"""Calibrate sequential endpoint responses on feedback, then audit once."""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
from typing import Any

from sequence_game.experiments.histogram_colearning.calibration_bank import (
    HistogramPayoffBank,
)
from sequence_game.experiments.histogram_colearning.config import (
    HistogramCoLearningConfig,
    REPO_ROOT,
)
from sequence_game.experiments.histogram_colearning.corpus_scale import (
    load_corpus_scale_case,
)
from sequence_game.experiments.histogram_colearning.endpoint_detection_study import (
    EndpointDetectionStudyConfig,
)
from sequence_game.experiments.histogram_colearning.endpoint_evidence_bank import (
    HistogramEndpointEvidenceBank,
)
from sequence_game.experiments.histogram_colearning.endpoint_response_calibration import (
    SequentialResponseCalibrationConfig,
    run_sequential_response_calibration_from_detection,
)
from sequence_game.experiments.histogram_colearning.stackelberg_bandit import (
    minimum_route_observation_design,
)
from sequence_game.experiments.histogram_colearning.topology_census import (
    available_ram_snapshot,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "histogram_sequential_endpoint_response_cernet_v1"
)
DEFAULT_PAYOFF_BANK = (
    REPO_ROOT
    / "runs"
    / "histogram_stackelberg_detection_cernet_100k"
    / "cernet_payoff_bank.npz"
)
DEFAULT_EVIDENCE_BANK = (
    REPO_ROOT
    / "runs"
    / "histogram_stackelberg_endpoint_detection_cernet_v2"
    / "cernet_endpoint_evidence_bank.npz"
)
DEFAULT_DETECTION_STUDY = (
    REPO_ROOT
    / "runs"
    / "histogram_stackelberg_endpoint_detection_cernet_v2"
    / "study.json"
)
DEFAULT_DETECTION_MANIFEST = (
    REPO_ROOT
    / "runs"
    / "histogram_stackelberg_detection_cernet_100k"
    / "run_manifest.json"
)
DEFAULT_CORPUS_MANIFEST = (
    REPO_ROOT / "runs" / "histogram_corpus_500k" / "run_manifest.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select sequential response parameters on disjoint feedback "
            "draws and execute the selected profiles once on audit draws."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--payoff-bank", type=Path, default=DEFAULT_PAYOFF_BANK)
    parser.add_argument(
        "--evidence-bank",
        type=Path,
        default=DEFAULT_EVIDENCE_BANK,
    )
    parser.add_argument(
        "--detection-study",
        type=Path,
        default=DEFAULT_DETECTION_STUDY,
    )
    parser.add_argument(
        "--detection-manifest",
        type=Path,
        default=DEFAULT_DETECTION_MANIFEST,
    )
    parser.add_argument(
        "--corpus-manifest",
        type=Path,
        default=DEFAULT_CORPUS_MANIFEST,
    )
    parser.add_argument(
        "--verification",
        action="store_true",
        help="Use a two-candidate feedback/audit plumbing check.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    detection_study = _read_json(args.detection_study)
    detection_manifest = _read_json(args.detection_manifest)
    corpus_manifest = _read_json(args.corpus_manifest)
    case = load_corpus_scale_case(
        detection_manifest["case"],
        corpus_manifest,
    )
    payoff_bank = HistogramPayoffBank.load(args.payoff_bank)
    evidence_bank = HistogramEndpointEvidenceBank.load(args.evidence_bank)
    source_study_config = _endpoint_detection_config(
        detection_study["study_config"]
    )
    learner_configs = _learner_configs(
        detection_manifest,
        study_config=source_study_config,
    )
    observation_design = minimum_route_observation_design(
        case,
        allowed_sensor_kinds=("node",),
    )
    calibration_config = (
        SequentialResponseCalibrationConfig(
            maximum_total_variations=(0.05, 0.20),
            arming_turns=(16,),
            containment_ceilings=(0.20,),
            posterior_window_turns=(64,),
        )
        if args.verification
        else SequentialResponseCalibrationConfig()
    )
    source_rows = [
        row
        for row in detection_study["rows"]
        if str(row["condition"]) == calibration_config.condition
    ]
    candidate_count = len(calibration_config.candidates())
    calibration_runs = (
        len(source_rows)
        * len(calibration_config.active_detectors)
        * candidate_count
    )
    audit_runs = (
        len(source_rows) * len(calibration_config.active_detectors)
    )
    preflight = {
        "case_id": payoff_bank.case_id,
        "route_count": len(payoff_bank.route_ids),
        "eve_action_count": len(payoff_bank.action_ids),
        "nominal_route_action_cells": (
            len(payoff_bank.route_ids) * len(payoff_bank.action_ids)
        ),
        "feedback_draws_per_cell": evidence_bank.draws_per_cell,
        "candidate_count": candidate_count,
        "source_row_count": len(source_rows),
        "calibration_run_count": calibration_runs,
        "audit_run_count": audit_runs,
        "calibration_config": calibration_config.to_dict(),
        "payoff_bank_bytes": args.payoff_bank.stat().st_size,
        "endpoint_evidence_bank_bytes": args.evidence_bank.stat().st_size,
        "payoff_bank_digest_blake2b_128": payoff_bank.content_digest(),
        "endpoint_evidence_bank_digest_blake2b_128": (
            evidence_bank.content_digest()
        ),
        "available_ram_before_run": available_ram_snapshot(),
        "physics_compilation_required": False,
        "cache_entries_required": 0,
        "execution_semantics": (
            "intervention model uses feedback draws 0:32; calibration uses "
            "feedback draws 32:96; selected profiles execute once on the "
            "seed-disjoint audit split"
        ),
    }
    _write_json(args.output / "preflight.json", preflight)
    print(
        "preflight: "
        f"{preflight['route_count']} routes, "
        f"{preflight['eve_action_count']} actions, "
        f"{candidate_count} candidates, "
        f"{calibration_runs} feedback runs, "
        f"{audit_runs} audit runs"
    )

    result = run_sequential_response_calibration_from_detection(
        case,
        payoff_bank,
        evidence_bank,
        learner_configs=learner_configs,
        observation_design=observation_design,
        detection_study=detection_study,
        calibration_config=calibration_config,
        selection_digest_blake2b_128=str(
            payoff_bank.metadata["selection_digest_blake2b_128"]
        ),
        progress_callback=print,
    )
    _write_json(args.output / "study.json", result)
    _write_json(
        args.output / "summary.json",
        {
            key: result[key]
            for key in (
                "schema",
                "case_id",
                "graph_id",
                "family",
                "calibration_config",
                "source_detection_schema",
                "candidate_count",
                "calibration_run_count",
                "audit_run_count",
                "elapsed_seconds",
                "selections",
                "audit_aggregates",
                "alarm_records",
                "information_boundary",
                "claim_boundary",
            )
        },
    )
    print(f"wrote {args.output / 'summary.json'}")


def _endpoint_detection_config(
    payload: dict[str, Any],
) -> EndpointDetectionStudyConfig:
    return EndpointDetectionStudyConfig(**{
        field.name: payload[field.name]
        for field in fields(EndpointDetectionStudyConfig)
    })


def _learner_configs(
    manifest: dict[str, Any],
    *,
    study_config: EndpointDetectionStudyConfig,
) -> dict[str, HistogramCoLearningConfig]:
    source = manifest["source_corpus_plan"]
    base = HistogramCoLearningConfig(
        turns=(
            study_config.maximum_follow_turns
            + study_config.prime_turns
            + study_config.exploit_turns
        ),
        requested_pairs_per_turn=int(source["requested_pairs_per_turn"]),
        final_window=min(
            int(source["final_window"]),
            study_config.prime_turns + study_config.exploit_turns,
        ),
        record_every=study_config.record_every,
        posthoc_samples_per_cell=0,
        seed=int(study_config.execution_seeds[0]),
        game_mode="attack_surface_all",
        age_points_per_edge=int(source["age_points_per_edge"]),
        evaluated_cache_size=max(
            256,
            int(manifest["case"]["eve_action_count"]) + 1,
        ),
    )
    return {
        str(row["label"]): replace(base, **dict(row["config_overrides"]))
        for row in manifest["defenders"]
    }


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
