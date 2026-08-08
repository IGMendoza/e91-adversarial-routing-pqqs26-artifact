#!/usr/bin/env python3
"""Run the corrected passive Stackelberg controller study on frozen Cernet."""

from __future__ import annotations

import argparse
from dataclasses import replace
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
from sequence_game.experiments.histogram_colearning.stackelberg_bandit import (
    minimum_route_observation_design,
)
from sequence_game.experiments.histogram_colearning.stackelberg_controller_study import (
    ControllerMatchedStudyConfig,
    run_matched_controller_study,
)
from sequence_game.experiments.histogram_colearning.topology_census import (
    available_ram_snapshot,
)


DEFAULT_OUTPUT = (
    REPO_ROOT / "runs" / "histogram_stackelberg_controller_cernet"
)
DEFAULT_BANK = (
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
            "Run passive follow, feedback-only Prime/Exploit planning, and "
            "matched held-out Cernet controls."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bank", type=Path, default=DEFAULT_BANK)
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
        help=(
            "Run a two-seed, short-horizon software-contract verification "
            "using the same compiled production physics bank."
        ),
    )
    args = parser.parse_args()

    corpus_manifest = _read_json(args.corpus_manifest)
    detection_manifest = _read_json(args.detection_manifest)
    case = load_corpus_scale_case(
        detection_manifest["case"],
        corpus_manifest,
    )
    bank = HistogramPayoffBank.load(args.bank)
    study_config = (
        ControllerMatchedStudyConfig(
            target_family_count=3,
            planning_seeds=(2_026_072_901, 2_026_072_902),
            execution_seeds=(2_026_071_901, 2_026_071_902),
            prime_turns=100,
            exploit_turns=100,
            record_every=50,
        )
        if args.verification
        else ControllerMatchedStudyConfig()
    )
    learner_configs = _learner_configs(
        detection_manifest,
        study_config=study_config,
    )
    observation_design = minimum_route_observation_design(
        case,
        allowed_sensor_kinds=("node",),
    )
    selection_digest = str(
        bank.metadata["selection_digest_blake2b_128"]
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    preflight = {
        "case_id": bank.case_id,
        "route_count": len(bank.route_ids),
        "eve_action_count": len(bank.action_ids),
        "nominal_route_action_cells": (
            len(bank.route_ids) * len(bank.action_ids)
        ),
        "canonical_cache_entries": int(
            detection_manifest["case"]["canonical_cache_entries"]
        ),
        "estimated_physics_cache_mib": float(
            detection_manifest["case"]["estimated_cache_mib"]
        ),
        "configured_lru_entries_per_route": int(
            next(iter(learner_configs.values())).evaluated_cache_size
        ),
        "payoff_bank_path": str(args.bank),
        "payoff_bank_bytes": args.bank.stat().st_size,
        "payoff_bank_draws_per_cell_per_split": bank.draws_per_cell,
        "payoff_bank_content_digest_blake2b_128": bank.content_digest(),
        "available_ram": available_ram_snapshot(),
        "observation_sensor_count": len(
            observation_design.selected_sensor_indices
        ),
        "exact_route_observation": (
            observation_design.exact_route_observation
        ),
        "study_config": study_config.to_dict(),
        "physics_compilation_required": False,
        "bank_reuse_semantics": (
            "load the frozen disjoint feedback/audit Cernet bank; planning "
            "uses feedback and execution uses audit"
        ),
    }
    _write_json(output / "preflight.json", preflight)
    print(
        "preflight: "
        f"{preflight['route_count']} routes, "
        f"{preflight['eve_action_count']} actions, "
        f"{preflight['nominal_route_action_cells']} cells, "
        f"{preflight['canonical_cache_entries']} canonical entries, "
        f"{preflight['estimated_physics_cache_mib']:.1f} MiB cache"
    )
    study = run_matched_controller_study(
        case,
        bank,
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
                "bank",
                "plan_surface",
                "run_count",
                "planning_row_count",
                "elapsed_seconds",
                "aggregates",
                "paired_contrasts",
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
    study_config: ControllerMatchedStudyConfig,
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
