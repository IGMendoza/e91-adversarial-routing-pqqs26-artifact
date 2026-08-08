#!/usr/bin/env python3
"""Run matched delivery-only and quantum-integrity detection on Cernet."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from sequence_game.experiments.histogram_colearning.calibration_bank import (
    HistogramPayoffBank,
    validate_bank_against_case,
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
    run_matched_endpoint_detection_study,
)
from sequence_game.experiments.histogram_colearning.endpoint_evidence_bank import (
    HistogramEndpointEvidenceBank,
    build_histogram_endpoint_evidence_bank,
    validate_endpoint_evidence_bank_against_case,
)
from sequence_game.experiments.histogram_colearning.physics import (
    HistogramPhysicsPool,
    histogram_cache_plan,
)
from sequence_game.experiments.histogram_colearning.stackelberg_bandit import (
    minimum_route_observation_design,
)
from sequence_game.experiments.histogram_colearning.topology_census import (
    available_ram_snapshot,
)


DEFAULT_OUTPUT = (
    REPO_ROOT / "runs" / "histogram_stackelberg_endpoint_detection_cernet"
)
DEFAULT_PAYOFF_BANK = (
    REPO_ROOT
    / "runs"
    / "histogram_stackelberg_detection_cernet_100k"
    / "cernet_payoff_bank.npz"
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
            "Build/load finite endpoint evidence and compare a physical-"
            "delivery detector with a matched delivery-plus-QBER/CHSH "
            "detector during passive-controller execution."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--payoff-bank",
        type=Path,
        default=DEFAULT_PAYOFF_BANK,
    )
    parser.add_argument("--evidence-bank", type=Path)
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
    parser.add_argument("--force-rebuild-evidence-bank", action="store_true")
    parser.add_argument(
        "--verification",
        action="store_true",
        help=(
            "Run a two-seed, short-horizon software-contract verification "
            "with the same compiled Cernet physics and action surface."
        ),
    )
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    corpus_manifest = _read_json(args.corpus_manifest)
    detection_manifest = _read_json(args.detection_manifest)
    case = load_corpus_scale_case(
        detection_manifest["case"],
        corpus_manifest,
    )
    payoff_bank = HistogramPayoffBank.load(args.payoff_bank)
    study_config = (
        EndpointDetectionStudyConfig(
            target_family_count=3,
            planning_seeds=(2_026_072_901, 2_026_072_902),
            execution_seeds=(2_026_071_901, 2_026_071_902),
            prime_turns=100,
            exploit_turns=100,
            record_every=25,
            commissioning_draws_per_route=8,
        )
        if args.verification
        else EndpointDetectionStudyConfig()
    )
    learner_configs = _learner_configs(
        detection_manifest,
        study_config=study_config,
    )
    base_config = next(iter(learner_configs.values()))
    selection_digest = str(
        payoff_bank.metadata["selection_digest_blake2b_128"]
    )
    validate_bank_against_case(
        payoff_bank,
        case,
        base_config,
        selection_digest_blake2b_128=selection_digest,
    )
    evidence_samples = (
        8
        if args.verification
        else int(detection_manifest["samples_per_cell_per_split"])
    )
    evidence_bank_path = (
        args.evidence_bank
        if args.evidence_bank is not None
        else output
        / (
            "cernet_endpoint_evidence_bank_verification.npz"
            if args.verification
            else "cernet_endpoint_evidence_bank.npz"
        )
    )

    cache_plan = histogram_cache_plan(case, base_config)
    nominal_cells = len(payoff_bank.route_ids) * len(payoff_bank.action_ids)
    estimated_evidence_bytes = (
        2 * nominal_cells * evidence_samples * _bytes_per_evidence_draw()
    )
    preflight = {
        "case_id": payoff_bank.case_id,
        "route_count": len(payoff_bank.route_ids),
        "eve_action_count": len(payoff_bank.action_ids),
        "nominal_route_action_cells": nominal_cells,
        "canonical_cache_entries": int(
            cache_plan["canonical_unique_move_count_total"]
        ),
        "estimated_physics_cache_bytes": int(
            cache_plan["estimated_full_action_surface_cache_bytes"]
        ),
        "estimated_physics_cache_mib": float(
            cache_plan["estimated_full_action_surface_cache_mib"]
        ),
        "configured_lru_entries_per_route": base_config.evaluated_cache_size,
        "minimum_lru_entries_per_route": int(
            cache_plan["minimum_entries_per_route_without_eviction"]
        ),
        "lru_sufficient_without_eviction": bool(
            cache_plan[
                "sufficient_for_declared_action_surface_without_eviction"
            ]
        ),
        "payoff_bank_path": str(args.payoff_bank),
        "payoff_bank_bytes": args.payoff_bank.stat().st_size,
        "payoff_bank_digest_blake2b_128": payoff_bank.content_digest(),
        "endpoint_evidence_bank_path": str(evidence_bank_path),
        "endpoint_evidence_samples_per_cell_per_split": evidence_samples,
        "estimated_endpoint_evidence_resident_bytes": (
            estimated_evidence_bytes
        ),
        "estimated_endpoint_evidence_resident_mib": (
            estimated_evidence_bytes / (1024.0 ** 2)
        ),
        "available_ram_before_bank": available_ram_snapshot(),
        "study_config": study_config.to_dict(),
        "physics_compilation_required": (
            not evidence_bank_path.exists()
            or args.force_rebuild_evidence_bank
        ),
        "information_boundary": (
            "the evidence bank retains public reward and finite endpoint "
            "delivery/QBER/CHSH counts; it excludes target-active and Eve "
            "private simulator diagnostics"
        ),
    }
    _write_json(output / "preflight.json", preflight)
    print(
        "preflight: "
        f"{preflight['route_count']} routes, "
        f"{preflight['eve_action_count']} actions, "
        f"{preflight['nominal_route_action_cells']} cells, "
        f"{preflight['canonical_cache_entries']} canonical entries, "
        f"{preflight['estimated_physics_cache_mib']:.1f} MiB physics cache, "
        f"{preflight['estimated_endpoint_evidence_resident_mib']:.1f} MiB "
        "endpoint evidence"
    )

    if (
        evidence_bank_path.exists()
        and not args.force_rebuild_evidence_bank
    ):
        print(f"loading endpoint evidence bank {evidence_bank_path}")
        evidence_bank = HistogramEndpointEvidenceBank.load(
            evidence_bank_path
        )
        validate_endpoint_evidence_bank_against_case(
            evidence_bank,
            case,
            base_config,
            selection_digest_blake2b_128=selection_digest,
        )
        if evidence_bank.draws_per_cell != evidence_samples:
            raise ValueError("existing endpoint evidence draw count changed")
    else:
        print("compiling endpoint evidence bank on demand")
        physics = HistogramPhysicsPool(case, base_config)
        evidence_bank = build_histogram_endpoint_evidence_bank(
            case,
            base_config,
            samples_per_cell=evidence_samples,
            root_seed=2_026_071_641,
            physics=physics,
            progress_callback=print,
            selection_digest_blake2b_128=selection_digest,
            evaluation_partition_loaded=True,
        )
        evidence_bank.save(evidence_bank_path)
        print(f"saved endpoint evidence bank {evidence_bank_path}")

    observation_design = minimum_route_observation_design(
        case,
        allowed_sensor_kinds=("node",),
    )
    study = run_matched_endpoint_detection_study(
        case,
        payoff_bank,
        evidence_bank,
        learner_configs=learner_configs,
        observation_design=observation_design,
        study_config=study_config,
        selection_digest_blake2b_128=selection_digest,
        progress_callback=print,
    )
    _write_json(output / "study.json", study)
    _write_json(
        output / "summary.json",
        {
            key: study[key]
            for key in (
                "schema",
                "case_id",
                "graph_id",
                "family",
                "study_config",
                "observation_design",
                "payoff_bank",
                "endpoint_evidence_bank",
                "plan_surface",
                "run_count",
                "planning_row_count",
                "elapsed_seconds",
                "aggregates",
                "information_boundary",
                "claim_boundary",
            )
        },
    )
    print(
        f"wrote {output / 'summary.json'} and {output / 'study.json'}"
    )


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


def _bytes_per_evidence_draw() -> int:
    return (
        8  # Alice reward
        + 8  # Bell-qualified rate
        + 8  # delivered count
        + 8  # key trial count
        + 8  # key error count
        + 4 * 8  # CHSH setting trial counts
        + 4 * 8  # CHSH same-outcome counts
        + 1  # public outcome code
    )


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
