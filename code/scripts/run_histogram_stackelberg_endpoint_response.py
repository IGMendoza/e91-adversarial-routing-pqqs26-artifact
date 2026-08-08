#!/usr/bin/env python3
"""Run matched alarm-conditioned Alice responses on frozen Cernet alarms."""

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
from sequence_game.experiments.histogram_colearning.endpoint_response_study import (
    EndpointResponseStudyConfig,
    run_endpoint_response_study_from_detection,
)
from sequence_game.experiments.histogram_colearning.stackelberg_bandit import (
    minimum_route_observation_design,
)
from sequence_game.experiments.histogram_colearning.topology_census import (
    available_ram_snapshot,
)


DEFAULT_OUTPUT = (
    REPO_ROOT / "runs" / "histogram_stackelberg_endpoint_response_cernet_v1"
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
            "Freeze classical/quantum endpoint alarms once and replay "
            "matched Alice response policies from the exact live state."
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
    parser.add_argument("--switching-penalty", type=float, default=0.05)
    parser.add_argument(
        "--compatibility-effect-floor",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--topology-sharing-trust-region-tv",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--independent-arm-trust-region-tv",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--topology-sharing-arming-turns",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--independent-arm-arming-turns",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--topology-sharing-containment-ceiling",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--independent-arm-containment-ceiling",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--posterior-pseudocount",
        type=float,
        default=0.5,
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
    response_config = EndpointResponseStudyConfig(
        switching_penalty=args.switching_penalty,
        compatibility_effect_floor=args.compatibility_effect_floor,
        topology_sharing_trust_region_tv=(
            args.topology_sharing_trust_region_tv
        ),
        independent_arm_trust_region_tv=(
            args.independent_arm_trust_region_tv
        ),
        topology_sharing_arming_turns=(
            args.topology_sharing_arming_turns
        ),
        independent_arm_arming_turns=(
            args.independent_arm_arming_turns
        ),
        topology_sharing_containment_ceiling=(
            args.topology_sharing_containment_ceiling
        ),
        independent_arm_containment_ceiling=(
            args.independent_arm_containment_ceiling
        ),
        posterior_pseudocount=args.posterior_pseudocount,
        record_every=source_study_config.record_every,
    )
    selected_source_rows = [
        row
        for row in detection_study["rows"]
        if row["condition"] in response_config.conditions
    ]
    preflight = {
        "case_id": payoff_bank.case_id,
        "route_count": len(payoff_bank.route_ids),
        "eve_action_count": len(payoff_bank.action_ids),
        "source_detection_study": str(args.detection_study),
        "payoff_bank": str(args.payoff_bank),
        "payoff_bank_bytes": args.payoff_bank.stat().st_size,
        "payoff_bank_digest_blake2b_128": payoff_bank.content_digest(),
        "endpoint_evidence_bank": str(args.evidence_bank),
        "endpoint_evidence_bank_bytes": args.evidence_bank.stat().st_size,
        "endpoint_evidence_bank_digest_blake2b_128": (
            evidence_bank.content_digest()
        ),
        "source_row_count": len(selected_source_rows),
        "alarm_record_count": (
            len(selected_source_rows)
            * len(response_config.active_detectors)
        ),
        "response_branch_count": (
            len(selected_source_rows)
            * len(response_config.active_detectors)
            * len(response_config.strategies)
        ),
        "response_study_config": response_config.to_dict(),
        "available_ram_before_run": available_ram_snapshot(),
        "physics_compilation_required": False,
        "execution_semantics": (
            "discover each held-out detector alarm once, then replay response "
            "branches without recomputing post-alarm detector evidence"
        ),
    }
    _write_json(args.output / "preflight.json", preflight)
    print(
        "preflight: "
        f"{preflight['route_count']} routes, "
        f"{preflight['eve_action_count']} actions, "
        f"{preflight['alarm_record_count']} alarm records, "
        f"{preflight['response_branch_count']} response branches"
    )

    study = run_endpoint_response_study_from_detection(
        case,
        payoff_bank,
        evidence_bank,
        learner_configs=learner_configs,
        observation_design=observation_design,
        detection_study=detection_study,
        response_study_config=response_config,
        selection_digest_blake2b_128=str(
            payoff_bank.metadata["selection_digest_blake2b_128"]
        ),
        progress_callback=print,
    )
    _write_json(args.output / "study.json", study)
    _write_json(
        args.output / "summary.json",
        {
            key: study[key]
            for key in (
                "schema",
                "case_id",
                "graph_id",
                "family",
                "response_study_config",
                "source_detection_schema",
                "source_study_config",
                "learner_configs",
                "run_count",
                "alarm_record_count",
                "elapsed_seconds",
                "aggregates",
                "replay_validation",
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
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
