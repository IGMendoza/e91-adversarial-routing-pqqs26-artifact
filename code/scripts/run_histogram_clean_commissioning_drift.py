#!/usr/bin/env python3
"""Run the frozen 1,200-pair clean commissioning/drift validation."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping

import numpy as np

from sequence_game.experiments.histogram_colearning.calibration_cases import (
    STEP6_SELECTION_DIGEST,
    FrozenCalibrationCase,
    load_frozen_partition_cases,
    load_frozen_partition_specs,
)
from sequence_game.experiments.histogram_colearning.clean_drift_validation import (
    CLEAN_DRIFT_STUDY_SCHEMA,
    CLEAN_DRIFT_SUMMARY_SCHEMA,
    CleanDriftScenario,
    CleanDriftValidationConfig,
    aggregate_clean_drift_rows,
    default_clean_drift_scenarios,
    default_physical_drift_profiles,
    freeze_digest,
    materialize_drift_profile,
    run_clean_drift_trajectory,
    trajectory_seed,
)
from sequence_game.experiments.histogram_colearning.config import (
    HistogramCoLearningConfig,
    REPO_ROOT,
)
from sequence_game.experiments.histogram_colearning.physics import (
    CACHE_ENTRY_RAM_ESTIMATE_BYTES,
    HistogramPhysicsPool,
    histogram_cache_plan,
)
from sequence_game.experiments.histogram_colearning.topology_census import (
    available_ram_snapshot,
)
from sequence_game.physical.repeater_service import (
    load_repeater_service_profile,
)


DEFAULT_OUTPUT = (
    REPO_ROOT / "runs" / "histogram_clean_commissioning_drift_v1"
)
PHASES = ("preflight", "run", "summarize", "all")
_WORKER_CONTEXT: dict[str, Any] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the frozen 1,200-pair endpoint detector under clean "
            "stationary commissioning and source-uncertainty device drift."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=4_000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--verification",
        action="store_true",
        help=(
            "Use one calibration case, three scenarios, two replicates, "
            "100 turns, and one worker."
        ),
    )
    args = parser.parse_args()
    output = args.output
    if args.verification and output == DEFAULT_OUTPUT:
        output = output.with_name(output.name + "_verification")
    output.mkdir(parents=True, exist_ok=True)

    config = _study_config(args)
    base_path = HistogramCoLearningConfig().repeater_service_profile_path
    base_profile = _read_json(base_path)
    drift_profiles = default_physical_drift_profiles(base_profile)
    scenarios = default_clean_drift_scenarios(
        change_turn=config.change_point_turn,
    )
    partitions = ("calibration", "evaluation")
    specs_by_partition = {
        partition: load_frozen_partition_specs(partition)
        for partition in partitions
    }
    if args.verification:
        specs_by_partition = {
            "calibration": specs_by_partition["calibration"][:1],
        }
        scenarios = (
            scenarios[0],
            next(
                row for row in scenarios
                if row.scenario_id
                == "memory_coherence_minus_1sigma__recommissioned"
            ),
            next(
                row for row in scenarios
                if row.scenario_id
                == (
                    "nominal_to_memory_coherence_minus_1sigma"
                    "__change_point"
                )
            ),
        )

    profile_paths = _materialize_profiles(
        output,
        base_profile,
        drift_profiles,
    )
    freeze = _freeze_manifest(
        config=config,
        scenarios=scenarios,
        drift_profiles=drift_profiles,
        specs_by_partition=specs_by_partition,
        base_profile_path=base_path,
        base_profile=base_profile,
        verification=bool(args.verification),
    )
    _write_or_validate_freeze(output / "study_freeze.json", freeze)
    preflight = _preflight(
        config=config,
        scenarios=scenarios,
        drift_profiles=drift_profiles,
        specs_by_partition=specs_by_partition,
    )
    _write_json(output / "preflight.json", preflight)
    if args.phase == "preflight":
        print(output / "preflight.json", flush=True)
        return

    if args.phase in {"run", "all"}:
        for partition, specs in specs_by_partition.items():
            cases = load_frozen_partition_cases(
                partition,
                case_ids=[row.case_id for row in specs],
            )
            for spec, case in cases:
                _run_case(
                    partition,
                    spec,
                    case,
                    config=config,
                    scenarios=scenarios,
                    profile_paths=profile_paths,
                    output=output,
                    force=bool(args.force),
                )

    if args.phase in {"summarize", "all"}:
        _write_summary(
            output,
            freeze=freeze,
            config=config,
            scenarios=scenarios,
        )
        print(output / "summary.json", flush=True)


def _study_config(args: argparse.Namespace) -> CleanDriftValidationConfig:
    if args.verification:
        return CleanDriftValidationConfig(
            horizon_turns=100,
            commissioning_draws_per_route=4,
            replicates_per_case_scenario=2,
            change_point_turn=51,
            workers=1,
        )
    return CleanDriftValidationConfig(
        horizon_turns=int(args.horizon),
        replicates_per_case_scenario=int(args.replicates),
        change_point_turn=int(args.horizon) // 2 + 1,
        workers=int(args.workers),
    )


def _materialize_profiles(
    output: Path,
    base_profile: Mapping[str, Any],
    drift_profiles: tuple[Any, ...],
) -> dict[str, Path]:
    directory = output / "profiles"
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for drift in drift_profiles:
        payload = materialize_drift_profile(base_profile, drift)
        path = directory / f"{drift.profile_id}.json"
        _write_json(path, payload)
        load_repeater_service_profile(path)
        paths[drift.profile_id] = path.resolve()
    return paths


def _freeze_manifest(
    *,
    config: CleanDriftValidationConfig,
    scenarios: tuple[CleanDriftScenario, ...],
    drift_profiles: tuple[Any, ...],
    specs_by_partition: Mapping[str, tuple[FrozenCalibrationCase, ...]],
    base_profile_path: Path,
    base_profile: Mapping[str, Any],
    verification: bool,
) -> dict[str, Any]:
    core = {
        "schema": CLEAN_DRIFT_STUDY_SCHEMA,
        "verification": verification,
        "config": config.to_dict(),
        "scenarios": [row.to_dict() for row in scenarios],
        "physical_drift_profiles": [row.to_dict() for row in drift_profiles],
        "base_profile_path": str(base_profile_path),
        "base_profile_digest_blake2b_128": freeze_digest(base_profile),
        "selection_digest_blake2b_128": STEP6_SELECTION_DIGEST,
        "cases": {
            partition: [row.to_dict() for row in specs]
            for partition, specs in specs_by_partition.items()
        },
        "detector_contract": {
            "block_size": config.requested_pairs_per_turn,
            "horizon": config.horizon_turns,
            "commissioning_draws_per_route": (
                config.commissioning_draws_per_route
            ),
            "model_false_alarm_level": config.detector_false_alarm_level,
            "commissioning_pseudocount": (
                config.detector_commissioning_pseudocount
            ),
            "change_fractions": list(config.detector_change_fractions),
            "frozen_from_response_milestone": True,
        },
        "study_boundary": {
            "attack_enabled": False,
            "full_corpus_exposition_started": False,
            "stationary_alarm_label": "clean_stationary_false_alarm",
            "stale_commissioning_alarm_label": (
                "clean_nuisance_drift_alarm"
            ),
            "uncertainty_semantics": (
                "source-reported one-sigma parameter stress; not a measured "
                "temporal drift distribution"
            ),
        },
    }
    return {
        **core,
        "freeze_digest_blake2b_128": freeze_digest(core),
        "execution_commit": _git_head(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _write_or_validate_freeze(path: Path, freeze: Mapping[str, Any]) -> None:
    if not path.exists():
        _write_json(path, freeze)
        return
    existing = _read_json(path)
    if existing.get("freeze_digest_blake2b_128") != freeze.get(
        "freeze_digest_blake2b_128"
    ):
        raise RuntimeError(
            "existing clean-drift freeze differs; use a new output directory"
        )


def _preflight(
    *,
    config: CleanDriftValidationConfig,
    scenarios: tuple[CleanDriftScenario, ...],
    drift_profiles: tuple[Any, ...],
    specs_by_partition: Mapping[str, tuple[FrozenCalibrationCase, ...]],
) -> dict[str, Any]:
    specs = [row for rows in specs_by_partition.values() for row in rows]
    route_count = sum(row.route_count for row in specs)
    full_actions = sum(row.eve_action_count for row in specs)
    full_cells = sum(
        row.route_count * row.eve_action_count for row in specs
    )
    profile_count = len(drift_profiles)
    maximum_case_routes = max((row.route_count for row in specs), default=0)
    trajectory_count = (
        len(specs)
        * len(scenarios)
        * config.replicates_per_case_scenario
    )
    monitoring_observations = trajectory_count * config.horizon_turns
    commissioning_observations = sum(
        row.route_count
        * len(scenarios)
        * config.replicates_per_case_scenario
        * config.commissioning_draws_per_route
        for row in specs
    )
    return {
        "schema": "histogram_clean_drift_preflight.v1",
        "partition_case_counts": {
            key: len(value) for key, value in specs_by_partition.items()
        },
        "case_count": len(specs),
        "route_count_sum": route_count,
        "full_declared_eve_action_count_sum": full_actions,
        "full_declared_nominal_route_action_cells": full_cells,
        "clean_study_action_count_per_case": 1,
        "clean_study_nominal_cells_per_profile": route_count,
        "physical_profile_count": profile_count,
        "canonical_cache_entries_per_profile_across_serial_cases": route_count,
        "canonical_cache_entries_across_all_profiles_and_cases": (
            route_count * profile_count
        ),
        "maximum_concurrent_canonical_entries": (
            maximum_case_routes * profile_count
        ),
        "estimated_concurrent_resident_bytes": (
            maximum_case_routes
            * profile_count
            * CACHE_ENTRY_RAM_ESTIMATE_BYTES
        ),
        "estimated_concurrent_resident_mib": (
            maximum_case_routes
            * profile_count
            * CACHE_ENTRY_RAM_ESTIMATE_BYTES
            / (1024.0 ** 2)
        ),
        "configured_lru_capacity_per_route": 256,
        "required_lru_entries_per_route": 1,
        "available_ram": available_ram_snapshot(),
        "scenario_count": len(scenarios),
        "stationary_scenario_count": sum(
            row.alarm_semantics == "clean_stationary_false_alarm"
            for row in scenarios
        ),
        "stale_commissioning_scenario_count": sum(
            row.alarm_semantics == "clean_nuisance_drift_alarm"
            for row in scenarios
        ),
        "replicates_per_case_scenario": (
            config.replicates_per_case_scenario
        ),
        "trajectory_count": trajectory_count,
        "monitoring_observation_count": monitoring_observations,
        "commissioning_observation_count": commissioning_observations,
        "total_compiled_transcript_samples": (
            monitoring_observations + commissioning_observations
        ),
        "requested_pairs_represented_in_monitoring": (
            monitoring_observations * config.requested_pairs_per_turn
        ),
        "workers": config.workers,
        "execution_order": (
            "cases serial; physical profiles resident together within one "
            "case; trajectories forked over deterministic seeds"
        ),
        "full_corpus_exposition_started": False,
    }


def _run_case(
    partition: str,
    spec: FrozenCalibrationCase,
    case: Any,
    *,
    config: CleanDriftValidationConfig,
    scenarios: tuple[CleanDriftScenario, ...],
    profile_paths: Mapping[str, Path],
    output: Path,
    force: bool,
) -> None:
    case_dir = output / "checkpoints" / partition / _slug(spec)
    case_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        row for row in scenarios
        if force or not (case_dir / f"{row.scenario_id}.json").exists()
    ]
    if not pending:
        print(f"clean drift skip {partition}/{spec.family}: complete", flush=True)
        return
    required_profiles = sorted({
        profile_id
        for scenario in pending
        for profile_id in scenario.required_profile_ids
    })
    base_config = replace(
        HistogramCoLearningConfig(),
        turns=config.horizon_turns,
        requested_pairs_per_turn=config.requested_pairs_per_turn,
        evaluated_cache_size=256,
    )
    no_attack = next(
        action for action in case.actions("attack_surface_all")
        if str(action["id"]) == "no_attack"
    )
    pools: dict[str, HistogramPhysicsPool] = {}
    physics_started = time.perf_counter()
    for profile_id in required_profiles:
        pool_config = replace(
            base_config,
            repeater_service_profile_path=profile_paths[profile_id],
        )
        pool = HistogramPhysicsPool(case, pool_config)
        for route_index in range(len(case.routes)):
            pool.step(
                route_index,
                no_attack,
                seed=2_026_072_399,
                step_index=route_index,
            )
        pools[profile_id] = pool
    physics_rows = {
        profile_id: pool.diagnostics()
        for profile_id, pool in pools.items()
    }
    _write_json(case_dir / "physics.json", {
        "partition": partition,
        "case": spec.to_dict(),
        "profiles": physics_rows,
        "compile_and_prewarm_seconds": time.perf_counter() - physics_started,
        "available_ram_after_prewarm": available_ram_snapshot(),
        "full_action_cache_plan": histogram_cache_plan(case, base_config),
        "clean_study_cache_contract": {
            "action_count": 1,
            "canonical_entries_per_profile": len(case.routes),
            "required_entries_per_route": 1,
            "configured_entries_per_route": base_config.evaluated_cache_size,
        },
    })

    global _WORKER_CONTEXT
    _WORKER_CONTEXT = {
        "partition": partition,
        "case_id": spec.case_id,
        "case": case,
        "pools": pools,
        "profile_paths": dict(profile_paths),
        "config": config,
    }
    worker_pool = None
    if config.workers > 1:
        context = mp.get_context("fork")
        worker_pool = context.Pool(processes=config.workers)
    try:
        for scenario in pending:
            started = time.perf_counter()
            tasks = [
                (scenario, replicate)
                for replicate in range(config.replicates_per_case_scenario)
            ]
            if worker_pool is None:
                rows = [_trajectory_worker(task) for task in tasks]
            else:
                rows = []
                iterator = worker_pool.imap_unordered(
                    _trajectory_worker,
                    tasks,
                    chunksize=1,
                )
                progress_every = max(1, len(tasks) // 5)
                for completed, row in enumerate(iterator, start=1):
                    rows.append(row)
                    if (
                        completed % progress_every == 0
                        or completed == len(tasks)
                    ):
                        print(
                            f"clean drift {partition}/{spec.family}/"
                            f"{scenario.scenario_id}: "
                            f"{completed}/{len(tasks)}",
                            flush=True,
                        )
            rows.sort(key=lambda row: int(row["replicate"]))
            checkpoint = {
                "schema": CLEAN_DRIFT_STUDY_SCHEMA,
                "partition": partition,
                "case": spec.to_dict(),
                "scenario": scenario.to_dict(),
                "config": config.to_dict(),
                "trajectory_count": len(rows),
                "elapsed_seconds": time.perf_counter() - started,
                "rows": rows,
            }
            _write_json(case_dir / f"{scenario.scenario_id}.json", checkpoint)
            print(
                f"clean drift completed {partition}/{spec.family}/"
                f"{scenario.scenario_id}",
                flush=True,
            )
    finally:
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()
        _WORKER_CONTEXT = None
        pools.clear()
        gc.collect()


def _trajectory_worker(
    task: tuple[CleanDriftScenario, int],
) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("clean drift worker context is not initialized")
    scenario, replicate = task
    case = _WORKER_CONTEXT["case"]
    config = _WORKER_CONTEXT["config"]
    seed = trajectory_seed(
        config.root_seed,
        str(_WORKER_CONTEXT["case_id"]),
        replicate,
    )
    return run_clean_drift_trajectory(
        case,
        _WORKER_CONTEXT["pools"],
        _WORKER_CONTEXT["profile_paths"],
        scenario,
        config,
        partition=_WORKER_CONTEXT["partition"],
        case_id=_WORKER_CONTEXT["case_id"],
        replicate=replicate,
        trajectory_seed=seed,
    )


def _write_summary(
    output: Path,
    *,
    freeze: Mapping[str, Any],
    config: CleanDriftValidationConfig,
    scenarios: tuple[CleanDriftScenario, ...],
) -> None:
    checkpoints = sorted((output / "checkpoints").glob("*/*/*.json"))
    checkpoints = [path for path in checkpoints if path.name != "physics.json"]
    rows: list[dict[str, Any]] = []
    elapsed = 0.0
    for path in checkpoints:
        payload = _read_json(path)
        rows.extend(payload["rows"])
        elapsed += float(payload.get("elapsed_seconds", 0.0))
    expected = sum(
        len(partition_rows)
        for partition_rows in freeze["cases"].values()
    ) * len(scenarios) * config.replicates_per_case_scenario
    if len(rows) != expected:
        raise RuntimeError(
            f"clean drift summary is incomplete: {len(rows)} != {expected}"
        )
    rows.sort(key=lambda row: (
        str(row["partition"]),
        str(row["case_id"]),
        str(row["scenario_id"]),
        int(row["replicate"]),
    ))
    _write_json(output / "trajectory_rows.json", {
        "schema": CLEAN_DRIFT_STUDY_SCHEMA,
        "freeze_digest_blake2b_128": freeze[
            "freeze_digest_blake2b_128"
        ],
        "row_count": len(rows),
        "rows": rows,
    })
    service_rows = []
    for scenario in scenarios:
        cell = [row for row in rows if row["scenario_id"] == scenario.scenario_id]
        service_rows.append({
            "scenario_id": scenario.scenario_id,
            "scenario_classification": scenario.classification,
            "trajectory_count": len(cell),
            "mean_public_acceptance": float(np.mean([
                int(row["public_outcome_counts"].get("accepted", 0))
                / int(row["horizon_turns"])
                for row in cell
            ])),
            "mean_physical_delivery_fraction": float(np.mean([
                float(row["physical_delivery_fraction"]) for row in cell
            ])),
            "mean_aggregate_qber": float(np.mean([
                float(row["aggregate_qber"]) for row in cell
                if row["aggregate_qber"] is not None
            ])),
            "mean_aggregate_chsh_s": float(np.mean([
                float(row["aggregate_chsh_s"]) for row in cell
                if row["aggregate_chsh_s"] is not None
            ])),
            "mean_alice_reward": float(np.mean([
                float(row["mean_alice_reward"]) for row in cell
            ])),
        })
    summary = {
        "schema": CLEAN_DRIFT_SUMMARY_SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_commit": _git_head(),
        "freeze_digest_blake2b_128": freeze[
            "freeze_digest_blake2b_128"
        ],
        "trajectory_count": len(rows),
        "case_count": sum(len(value) for value in freeze["cases"].values()),
        "scenario_count": len(scenarios),
        "detector_aggregates": aggregate_clean_drift_rows(rows),
        "partition_detector_aggregates": {
            partition: aggregate_clean_drift_rows([
                row for row in rows if row["partition"] == partition
            ])
            for partition in sorted({str(row["partition"]) for row in rows})
        },
        "service_aggregates": service_rows,
        "summed_checkpoint_elapsed_seconds": elapsed,
        "statistical_boundary": {
            "pooled_interval": (
                "two-sided 95% Clopper-Pearson interval over independent "
                "trajectory seeds within the fixed ten-topology panel"
            ),
            "topology_uncertainty": (
                "equal-weight topology macro rate and standard error over "
                "the ten fixed cases"
            ),
            "stationary_alarm_semantics": (
                "empirical clean false alarm under matched commissioning "
                "and monitoring profile"
            ),
            "stale_commissioning_alarm_semantics": (
                "clean nuisance-drift alarm relative to a stale nominal "
                "commissioning model; not a stationary false positive"
            ),
        },
        "attack_enabled": False,
        "full_corpus_exposition_started": False,
        "ready_for_scientific_review": True,
    }
    _write_json(output / "summary.json", summary)


def _slug(spec: FrozenCalibrationCase) -> str:
    return f"{spec.selection_order:02d}_{spec.family.lower()}"


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
