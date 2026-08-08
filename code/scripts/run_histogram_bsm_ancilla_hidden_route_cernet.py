#!/usr/bin/env python3
"""Run the Cernet hidden-route XX/ZZ ancilla follower study."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from sequence_game.experiments.histogram_colearning.bsm_ancilla_hidden_route import (
    BSMAncillaHiddenRouteConfig,
    BSMAncillaObservationScenario,
    bsm_probe_actions_for_case,
    bsm_probe_cache_plan,
    run_bsm_ancilla_hidden_route_study,
)
from sequence_game.experiments.histogram_colearning.config import (
    HistogramCoLearningConfig,
    REPO_ROOT,
)
from sequence_game.experiments.histogram_colearning.corpus_scale import (
    load_corpus_scale_case,
)
from sequence_game.experiments.histogram_colearning.topology_census import (
    available_ram_snapshot,
)
from sequence_game.experiments.histogram_colearning.stackelberg_bandit import (
    minimum_route_observation_design,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "runs"
    / "histogram_bsm_ancilla_hidden_route_cernet_v2"
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
            "Filter Cernet learner state through latent routes using only "
            "compiled XX/ZZ ancilla records and public Alice reward."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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
        help="Use one seed and a reduced particle/commissioning contract.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    detection_manifest = _read_json(args.detection_manifest)
    corpus_manifest = _read_json(args.corpus_manifest)
    case = load_corpus_scale_case(
        detection_manifest["case"],
        corpus_manifest,
    )
    study_config = (
        BSMAncillaHiddenRouteConfig(
            maximum_follow_turns=20,
            particles_per_hypothesis=32,
            commissioning_draws_per_route_probe=2,
            scenarios=(
                BSMAncillaObservationScenario(
                    "separating_panel_every_turn",
                    1.0,
                    1.0,
                    None,
                ),
                BSMAncillaObservationScenario(
                    "two_probe_panel_quarter_turns_missed",
                    0.25,
                    0.75,
                    2,
                ),
            ),
            execution_seeds=(2_026_072_201,),
            record_every=5,
        )
        if args.verification
        else BSMAncillaHiddenRouteConfig()
    )
    learner_configs = _learner_configs(
        detection_manifest,
        study_config=study_config,
    )
    base_config = next(iter(learner_configs.values()))
    actions = bsm_probe_actions_for_case(
        case,
        stabilizers=study_config.stabilizers,
        dephasing_theta=study_config.dephasing_theta,
    )
    cache_plan = bsm_probe_cache_plan(
        case,
        base_config,
        actions,
    )
    observation_design = minimum_route_observation_design(
        case,
        allowed_sensor_kinds=("node",),
    )
    run_count = (
        len(study_config.scenarios)
        * len(learner_configs)
        * len(study_config.execution_seeds)
    )
    preflight = {
        "case_id": str(detection_manifest["case"]["case_id"]),
        "route_count": len(case.routes),
        "sensor_node_count": len({
            str(action["target"]) for action in actions
        }),
        "probe_action_count": len(actions),
        "minimum_separating_panel_size": len(
            observation_design.selected_sensor_labels
        ),
        "minimum_separating_panel_labels": list(
            observation_design.selected_sensor_labels
        ),
        "nominal_route_probe_cells": len(case.routes) * len(actions),
        "canonical_cache_entries": cache_plan[
            "canonical_unique_move_count_total"
        ],
        "estimated_resident_bytes": cache_plan[
            "estimated_resident_bytes"
        ],
        "estimated_resident_mib": cache_plan[
            "estimated_resident_mib"
        ],
        "configured_lru_entries_per_route": (
            base_config.evaluated_cache_size
        ),
        "maximum_canonical_moves_on_one_route": cache_plan[
            "maximum_canonical_moves_on_one_route"
        ],
        "cache_capacity_sufficient_without_eviction": cache_plan[
            "sufficient_without_eviction"
        ],
        "particles_per_hypothesis": (
            study_config.particles_per_hypothesis
        ),
        "learner_hypothesis_count": len(learner_configs),
        "total_particle_count": (
            study_config.particles_per_hypothesis
            * len(learner_configs)
        ),
        "scenario_count": len(study_config.scenarios),
        "execution_run_count": run_count,
        "commissioning_draws_per_route_probe": (
            study_config.commissioning_draws_per_route_probe
        ),
        "commissioning_physics_draws": (
            len(case.routes)
            * len(actions)
            * study_config.commissioning_draws_per_route_probe
        ),
        "available_ram_before_run": available_ram_snapshot(),
        "study_config": study_config.to_dict(),
        "execution_semantics": (
            "one shared compiled physics pool and commissioning table; "
            "each learner/seed/scenario run uses a fresh particle filter "
            "and matched clean Alice counterfactual"
        ),
    }
    _write_json(args.output / "preflight.json", preflight)
    print(
        "preflight: "
        f"{preflight['route_count']} routes, "
        f"{preflight['probe_action_count']} probes, "
        f"{preflight['nominal_route_probe_cells']} nominal cells, "
        f"{preflight['canonical_cache_entries']} canonical entries, "
        f"{preflight['estimated_resident_mib']:.2f} MiB estimated, "
        f"{run_count} executions"
    )

    result = run_bsm_ancilla_hidden_route_study(
        case,
        learner_configs,
        study_config=study_config,
        progress_callback=print,
    )
    _write_json(args.output / "study.json", result)
    _write_json(
        args.output / "summary.json",
        {
            key: result[key]
            for key in (
                "schema",
                "study_config",
                "probe_cache_plan",
                "probe_contract",
                "expected_service_invariance",
                "aggregates",
                "elapsed_seconds",
                "information_boundary",
                "scientific_scope",
                "claim_boundary",
            )
        },
    )
    print(f"wrote {args.output / 'summary.json'}")


def _learner_configs(
    manifest: dict[str, Any],
    *,
    study_config: BSMAncillaHiddenRouteConfig,
) -> dict[str, HistogramCoLearningConfig]:
    source = manifest["source_corpus_plan"]
    base = HistogramCoLearningConfig(
        turns=study_config.maximum_follow_turns,
        requested_pairs_per_turn=int(source["requested_pairs_per_turn"]),
        final_window=min(
            int(source["final_window"]),
            study_config.maximum_follow_turns,
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
        str(row["label"]): replace(
            base,
            **dict(row["config_overrides"]),
        )
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
