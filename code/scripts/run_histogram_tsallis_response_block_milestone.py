#!/usr/bin/env python3
"""Run the staged Tsallis response/observation-block milestone."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
import multiprocessing as mp
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from sequence_game.experiments.histogram_colearning.calibration_bank import (
    HistogramPayoffBank,
    validate_bank_against_case,
)
from sequence_game.experiments.histogram_colearning.calibration_cases import (
    STEP6_SELECTION_DIGEST,
    FrozenCalibrationCase,
    load_frozen_calibration_cases,
    load_frozen_calibration_specs,
    load_frozen_evaluation_cases,
    load_frozen_evaluation_specs,
)
from sequence_game.experiments.histogram_colearning.config import (
    HistogramCoLearningConfig,
    REPO_ROOT,
)
from sequence_game.experiments.histogram_colearning.endpoint_evidence_bank import (
    HistogramEndpointEvidenceBank,
    build_histogram_endpoint_evidence_bank,
    validate_endpoint_evidence_bank_against_case,
)
from sequence_game.experiments.histogram_colearning.endpoint_response import (
    DETECTOR_CLASSICAL_SERVICE,
    DETECTOR_QUANTUM_ENDPOINT,
    discover_alarm_response_record,
    run_alarm_conditioned_response_continuation,
)
from sequence_game.experiments.histogram_colearning.endpoint_response_calibration import (
    SequentialResponseCalibrationConfig,
    SequentialResponseCandidate,
    sequential_response_config,
)
from sequence_game.experiments.histogram_colearning.paired_block_evidence import (
    derive_paired_block_size_banks,
)
from sequence_game.experiments.histogram_colearning.physics import (
    HistogramPhysicsPool,
    histogram_cache_plan,
)
from sequence_game.experiments.histogram_colearning.quantum_stackelberg_detection import (
    MatchedClassicalQuantumEndpointDetectors,
)
from sequence_game.experiments.histogram_colearning.response_block_milestone import (
    RESPONSE_BLOCK_FREEZE_SCHEMA,
    RESPONSE_BLOCK_MILESTONE_SCHEMA,
    TsallisResponseBlockMilestoneConfig,
    freeze_digest,
    response_result_row,
    select_feedback_response_profiles,
    validate_freeze_manifest,
)
from sequence_game.experiments.histogram_colearning.stackelberg_bandit import (
    minimum_route_observation_design,
    run_stackelberg_bandit_follow_runtime,
)
from sequence_game.experiments.histogram_colearning.stackelberg_controller_plans import (
    ControllerPrimeExploitPlan,
    build_controller_prime_exploit_plans,
)
from sequence_game.experiments.histogram_colearning.topology_census import (
    available_ram_snapshot,
)


DEFAULT_OUTPUT = (
    REPO_ROOT / "runs" / "histogram_tsallis_response_block_milestone_v1"
)
PHASES = ("preflight", "calibration", "audit", "evaluation", "all")
_FEEDBACK_WORKER_CONTEXT: dict[str, Any] | None = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build paired endpoint blocks, calibrate Tsallis response on the "
            "five frozen calibration graphs, freeze choices, and evaluate "
            "once on the five untouched graphs."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=PHASES, default="all")
    parser.add_argument(
        "--force-rebuild-banks",
        action="store_true",
    )
    parser.add_argument(
        "--verification",
        action="store_true",
        help=(
            "Use one case, two seeds, short schedules, eight endpoint draws, "
            "and two response candidates to verify plumbing."
        ),
    )
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = _milestone_config(verification=args.verification)
    _write_json(output / "milestone_config.json", {
        "schema": RESPONSE_BLOCK_MILESTONE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "verification": bool(args.verification),
        "config": config.to_dict(),
        "selection_digest_blake2b_128": STEP6_SELECTION_DIGEST,
        "full_corpus_exposition_in_scope": False,
    })

    calibration_specs = load_frozen_calibration_specs()
    if args.verification:
        calibration_specs = calibration_specs[:1]
    _write_json(
        output / "preflight_calibration.json",
        _preflight_from_specs(
            "calibration",
            calibration_specs,
            config,
        ),
    )
    if args.phase == "preflight":
        print(output / "preflight_calibration.json")
        return

    if args.phase in {"calibration", "all"}:
        calibration_cases = load_frozen_calibration_cases(
            case_ids=[row.case_id for row in calibration_specs],
        )
        _run_feedback_calibration(
            calibration_cases,
            config=config,
            output=output,
            force_rebuild_banks=args.force_rebuild_banks,
        )
    freeze_path = output / "response_block_freeze.json"
    if not freeze_path.exists():
        raise FileNotFoundError(
            "feedback calibration has not produced a freeze manifest"
        )
    freeze = _read_json(freeze_path)
    validate_freeze_manifest(freeze)

    if args.phase in {"audit", "all"}:
        calibration_cases = load_frozen_calibration_cases(
            case_ids=[row.case_id for row in calibration_specs],
        )
        _run_locked_partition(
            "calibration",
            calibration_cases,
            config=config,
            freeze=freeze,
            output=output,
            force_rebuild_banks=False,
        )

    if args.phase in {"evaluation", "all"}:
        # This is the first point at which evaluation metadata or cases load.
        evaluation_specs = load_frozen_evaluation_specs()
        if args.verification:
            evaluation_specs = evaluation_specs[:1]
        _write_json(
            output / "preflight_evaluation.json",
            _preflight_from_specs(
                "evaluation",
                evaluation_specs,
                config,
                freeze_digest_value=str(
                    freeze["freeze_digest_blake2b_128"]
                ),
            ),
        )
        evaluation_cases = load_frozen_evaluation_cases(
            case_ids=[row.case_id for row in evaluation_specs],
        )
        _run_locked_partition(
            "evaluation",
            evaluation_cases,
            config=config,
            freeze=freeze,
            output=output,
            force_rebuild_banks=args.force_rebuild_banks,
        )

    _write_final_summary(output, config, freeze)
    print(output / "summary.json")


def _milestone_config(
    *,
    verification: bool,
) -> TsallisResponseBlockMilestoneConfig:
    if not verification:
        return TsallisResponseBlockMilestoneConfig()
    response = SequentialResponseCalibrationConfig(
        active_detectors=(DETECTOR_QUANTUM_ENDPOINT,),
        maximum_total_variations=(0.05,),
        arming_turns=(4,),
        containment_ceilings=(0.10,),
        posterior_window_turns=(4,),
        posterior_refresh_turns=2,
        feedback_model_draw_start=0,
        feedback_model_draw_stop=4,
        feedback_calibration_draw_start=4,
        feedback_calibration_draw_stop=8,
        audit_draw_start=0,
        audit_draw_stop=8,
        response_workers=1,
        record_every=5,
    )
    return TsallisResponseBlockMilestoneConfig(
        diagnostic_block_sizes=(64,),
        primary_block_sizes=(256, 1_200),
        endpoint_samples_per_cell_per_split=8,
        maximum_follow_turns=10,
        follow_posterior_threshold=0.90,
        execution_seeds=(2_026_071_901, 2_026_071_902),
        prime_turns=20,
        exploit_turns=10,
        record_every=5,
        commissioning_draws_per_route=4,
        feedback_calibration_draw_start=4,
        feedback_calibration_draw_stop=8,
        audit_draw_start=0,
        audit_draw_stop=8,
        active_detectors=(DETECTOR_QUANTUM_ENDPOINT,),
        response_calibration=response,
    )


def _preflight_from_specs(
    partition: str,
    specs: Sequence[FrozenCalibrationCase],
    config: TsallisResponseBlockMilestoneConfig,
    *,
    freeze_digest_value: str | None = None,
) -> dict[str, Any]:
    cells = sum(row.route_count * row.eve_action_count for row in specs)
    endpoint_draws = (
        2
        * cells
        * config.endpoint_samples_per_cell_per_split
    )
    paired_bank_multiplier = len(config.block_sizes)
    candidate_count = len(config.response_calibration.candidates())
    feedback_runs = (
        len(specs)
        * len(config.block_sizes)
        * len(config.execution_seeds)
        * len(config.active_detectors)
        * candidate_count
    )
    locked_runs = (
        len(specs)
        * len(config.block_sizes)
        * len(config.execution_seeds)
        * len(config.active_detectors)
        * 6
    )
    return {
        "schema": "histogram_tsallis_response_block_preflight.v1",
        "partition": partition,
        "case_count": len(specs),
        "cases": [row.to_dict() for row in specs],
        "route_count_total": sum(row.route_count for row in specs),
        "eve_action_count_total": sum(
            row.eve_action_count for row in specs
        ),
        "nominal_route_action_cells": cells,
        "canonical_cache_entries": (
            "computed exactly per case immediately before bank compilation"
        ),
        "largest_declared_cache_mib": max(
            row.estimated_cache_mib for row in specs
        ),
        "all_declared_cache_mib": sum(
            row.estimated_cache_mib for row in specs
        ),
        "configured_lru_capacity_per_route": 256,
        "endpoint_draws_at_maximum_block": endpoint_draws,
        "paired_block_sizes": list(config.block_sizes),
        "paired_standard_bank_multiplier": paired_bank_multiplier,
        "candidate_count": candidate_count,
        "response_worker_count": config.response_workers,
        "response_worker_start_method": (
            "single_process"
            if config.response_workers == 1
            else "fork_copy_on_write"
        ),
        "feedback_calibration_run_count": (
            feedback_runs if partition == "calibration" else 0
        ),
        "locked_selected_native_attack_clean_run_count": locked_runs,
        "turns_per_response_run": config.schedule.total_turns,
        "requested_pairs_per_complete_run_by_block": {
            str(size): size * config.schedule.total_turns
            for size in config.block_sizes
        },
        "available_ram_before_partition": available_ram_snapshot(),
        "freeze_digest_blake2b_128": freeze_digest_value,
        "evaluation_partition_loaded": partition == "evaluation",
        "full_corpus_exposition_in_scope": False,
    }


def _run_feedback_calibration(
    loaded: Sequence[tuple[FrozenCalibrationCase, Any]],
    *,
    config: TsallisResponseBlockMilestoneConfig,
    output: Path,
    force_rebuild_banks: bool,
) -> None:
    _write_json(
        output / "runtime_preflight_calibration.json",
        _runtime_preflight("calibration", loaded, config),
    )
    all_rows: list[dict[str, Any]] = []
    plan_rows: list[dict[str, Any]] = []
    bank_rows: list[dict[str, Any]] = []
    for spec, case in loaded:
        print(f"calibration banks: {spec.family}")
        case_banks = _ensure_case_banks(
            "calibration",
            spec,
            case,
            config=config,
            output=output,
            force_rebuild=force_rebuild_banks,
        )
        bank_rows.append(case_banks["summary"])
        maximum_payoff, _maximum_evidence = _load_standard_banks(
            case_banks,
            config.maximum_block_size,
        )
        maximum_learners = _learner_configs(
            case,
            block_size=config.maximum_block_size,
            config=config,
        )
        plan, plan_contract = _fixed_memory_exploit_plan(
            case,
            maximum_payoff,
            maximum_learners["tsallis_inf_rv_eta1"],
        )
        del maximum_payoff, _maximum_evidence
        for block_size in config.block_sizes:
            checkpoint = (
                output
                / "feedback"
                / _slug(spec)
                / f"block_{block_size}.json"
            )
            if checkpoint.exists():
                payload = _read_json(checkpoint)
                all_rows.extend(payload["rows"])
                plan_rows.append({
                    **payload["plan"],
                    "case_id": spec.case_id,
                    "block_size": block_size,
                })
                print(f"feedback resume: {spec.family} block={block_size}")
                continue
            payoff_bank, evidence_bank = _load_standard_banks(
                case_banks,
                block_size,
            )
            learner_configs = _learner_configs(
                case,
                block_size=block_size,
                config=config,
            )
            observation_design = _observation_design(case)
            rows, clean_alarm_rows, follow_rows = _feedback_block_rows(
                spec,
                case,
                payoff_bank,
                evidence_bank,
                learner_configs=learner_configs,
                observation_design=observation_design,
                plan=plan,
                block_size=block_size,
                config=config,
            )
            payload = {
                "partition": "calibration",
                "case": spec.to_dict(),
                "block_size": block_size,
                "plan": {
                    **plan.to_dict(),
                    **plan_contract,
                },
                "observation_design": observation_design.to_dict(),
                "payoff_bank_digest_blake2b_128": (
                    payoff_bank.content_digest()
                ),
                "endpoint_bank_digest_blake2b_128": (
                    evidence_bank.content_digest()
                ),
                "rows": rows,
                "clean_alarm_rows": clean_alarm_rows,
                "follow_rows": follow_rows,
            }
            _write_json(checkpoint, payload)
            all_rows.extend(rows)
            plan_rows.append({
                **plan.to_dict(),
                **plan_contract,
                "case_id": spec.case_id,
                "block_size": block_size,
            })
            del payoff_bank, evidence_bank
            gc.collect()

    selection = select_feedback_response_profiles(
        all_rows,
        config.response_calibration.candidates(),
        primary_block_sizes=config.primary_block_sizes,
    )
    _write_json(output / "feedback_rows.json", {
        "schema": "histogram_tsallis_response_feedback_rows.v1",
        "rows": all_rows,
    })
    freeze = {
        "schema": RESPONSE_BLOCK_FREEZE_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_digest_blake2b_128": STEP6_SELECTION_DIGEST,
        "milestone_config": config.to_dict(),
        "selection": selection,
        "calibration_cases": [
            spec.to_dict() for spec, _case in loaded
        ],
        "calibration_bank_summaries": bank_rows,
        "feedback_plan_rows": plan_rows,
        "source_commit": _git_head(),
        "evaluation_partition_loaded_before_freeze": False,
        "audit_data_loaded_before_freeze": False,
        "full_corpus_exposition_authorized": False,
    }
    freeze["freeze_digest_blake2b_128"] = freeze_digest(freeze)
    _write_json(output / "response_block_freeze.json", freeze)
    validate_freeze_manifest(freeze)
    print(
        "froze feedback choices: "
        f"{freeze['freeze_digest_blake2b_128']}"
    )


def _feedback_block_rows(
    spec: FrozenCalibrationCase,
    case: Any,
    payoff_bank: HistogramPayoffBank,
    evidence_bank: HistogramEndpointEvidenceBank,
    *,
    learner_configs: Mapping[str, HistogramCoLearningConfig],
    observation_design: Any,
    plan: ControllerPrimeExploitPlan,
    block_size: int,
    config: TsallisResponseBlockMilestoneConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = config.response_calibration.candidates()
    commissioning = _commissioning(
        evidence_bank,
        stop=config.commissioning_draws_per_route,
    )
    clean_alarm_rows: list[dict[str, Any]] = []
    follow_rows: list[dict[str, Any]] = []
    follows: dict[int, Any] = {}
    alarms: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in config.execution_seeds:
        follow = _follow(
            case,
            payoff_bank,
            learner_configs,
            observation_design=observation_design,
            seed=seed,
            config=config,
        )
        follows[int(seed)] = follow
        follow_rows.append({
            "seed": int(seed),
            "turns": int(follow.report["turns"]),
            "decision": (
                dict(follow.report["decision"])
                if isinstance(follow.report.get("decision"), Mapping)
                else None
            ),
            "posterior": follow.tracker.posterior_by_label(),
        })
        for detector in config.active_detectors:
            alarm = discover_alarm_response_record(
                case,
                payoff_bank,
                evidence_bank,
                learner_configs["tsallis_inf_rv_eta1"],
                plan,
                schedule=config.schedule,
                alice=follow.clone_true_alice(
                    continuation_seed=seed
                ),
                observation_design=observation_design,
                detectors=_detectors(commissioning, config),
                active_detector=detector,
                seed=seed,
                attack_enabled=True,
                execution_split="feedback",
                execution_draw_start=(
                    config.feedback_calibration_draw_start
                ),
                execution_draw_stop=(
                    config.feedback_calibration_draw_stop
                ),
                selection_digest_blake2b_128=STEP6_SELECTION_DIGEST,
            )
            alarms[(int(seed), str(detector))] = alarm
            clean_alarm = discover_alarm_response_record(
                case,
                payoff_bank,
                evidence_bank,
                learner_configs["tsallis_inf_rv_eta1"],
                plan,
                schedule=config.schedule,
                alice=follow.clone_true_alice(
                    continuation_seed=seed
                ),
                observation_design=observation_design,
                detectors=_detectors(commissioning, config),
                active_detector=detector,
                seed=seed,
                attack_enabled=False,
                execution_split="feedback",
                execution_draw_start=(
                    config.feedback_calibration_draw_start
                ),
                execution_draw_stop=(
                    config.feedback_calibration_draw_stop
                ),
                selection_digest_blake2b_128=STEP6_SELECTION_DIGEST,
            )
            clean_alarm_rows.append(
                _alarm_row(
                    clean_alarm,
                    spec=spec,
                    block_size=block_size,
                    detector=detector,
                    seed=seed,
                    scenario="clean_no_attack",
                )
            )
    tasks = [
        (int(seed), str(detector), candidate)
        for seed in config.execution_seeds
        for detector in config.active_detectors
        for candidate in candidates
    ]
    global _FEEDBACK_WORKER_CONTEXT
    _FEEDBACK_WORKER_CONTEXT = {
        "spec": spec,
        "case": case,
        "payoff_bank": payoff_bank,
        "evidence_bank": evidence_bank,
        "learner_config": learner_configs["tsallis_inf_rv_eta1"],
        "observation_design": observation_design,
        "plan": plan,
        "block_size": int(block_size),
        "config": config,
        "commissioning": commissioning,
        "follows": follows,
        "alarms": alarms,
    }
    if config.response_workers == 1:
        rows = [_feedback_candidate_worker(task) for task in tasks]
    else:
        context = mp.get_context("fork")
        with context.Pool(processes=config.response_workers) as pool:
            rows = list(pool.imap_unordered(
                _feedback_candidate_worker,
                tasks,
                chunksize=1,
            ))
    _FEEDBACK_WORKER_CONTEXT = None
    rows.sort(key=lambda row: (
        int(row["seed"]),
        str(row["active_detector"]),
        str(row["candidate_id"]),
    ))
    return rows, clean_alarm_rows, follow_rows


def _feedback_candidate_worker(
    task: tuple[int, str, SequentialResponseCandidate],
) -> dict[str, Any]:
    context = _FEEDBACK_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("feedback worker context was not initialized")
    seed, detector, candidate = task
    config = context["config"]
    follow = context["follows"][int(seed)]
    response = run_alarm_conditioned_response_continuation(
        context["case"],
        context["payoff_bank"],
        context["evidence_bank"],
        context["learner_config"],
        context["plan"],
        schedule=config.schedule,
        alice=follow.clone_true_alice(continuation_seed=seed),
        observation_design=context["observation_design"],
        detectors=_detectors(context["commissioning"], config),
        response_config=sequential_response_config(
            config.response_calibration,
            candidate,
            detector,
        ),
                    seed=seed,
                    attack_enabled=True,
                    condition=_attack_scenario(context["plan"]),
        frozen_alarm_record=context["alarms"][(seed, detector)],
        execution_split="feedback",
        execution_draw_start=config.feedback_calibration_draw_start,
        execution_draw_stop=config.feedback_calibration_draw_stop,
        selection_digest_blake2b_128=STEP6_SELECTION_DIGEST,
    )
    spec = context["spec"]
    row = response_result_row(
        response,
        partition="calibration",
        case_id=spec.case_id,
        family=spec.family,
        block_size=context["block_size"],
        active_detector=detector,
        candidate=candidate,
        execution_role="feedback_candidate",
    )
    row["scenario"] = _attack_scenario(context["plan"])
    return row


def _run_locked_partition(
    partition: str,
    loaded: Sequence[tuple[FrozenCalibrationCase, Any]],
    *,
    config: TsallisResponseBlockMilestoneConfig,
    freeze: Mapping[str, Any],
    output: Path,
    force_rebuild_banks: bool,
) -> None:
    validate_freeze_manifest(freeze)
    _write_json(
        output / f"runtime_preflight_{partition}.json",
        _runtime_preflight(partition, loaded, config),
    )
    selected = _selected_candidates_by_block(freeze)
    raw_best = _raw_best_candidates_by_block(freeze)
    partition_rows: list[dict[str, Any]] = []
    bank_rows: list[dict[str, Any]] = []
    for spec, case in loaded:
        print(f"{partition} banks: {spec.family}")
        case_banks = _ensure_case_banks(
            partition,
            spec,
            case,
            config=config,
            output=output,
            force_rebuild=force_rebuild_banks,
        )
        bank_rows.append(case_banks["summary"])
        maximum_payoff, _maximum_evidence = _load_standard_banks(
            case_banks,
            config.maximum_block_size,
        )
        maximum_learners = _learner_configs(
            case,
            block_size=config.maximum_block_size,
            config=config,
        )
        plan, plan_contract = _fixed_memory_exploit_plan(
            case,
            maximum_payoff,
            maximum_learners["tsallis_inf_rv_eta1"],
        )
        del maximum_payoff, _maximum_evidence
        for block_size in config.block_sizes:
            checkpoint = (
                output
                / partition
                / _slug(spec)
                / f"block_{block_size}.json"
            )
            if checkpoint.exists():
                payload = _read_json(checkpoint)
                partition_rows.extend(payload["rows"])
                print(f"{partition} resume: {spec.family} {block_size}")
                continue
            payoff_bank, evidence_bank = _load_standard_banks(
                case_banks,
                block_size,
            )
            learner_configs = _learner_configs(
                case,
                block_size=block_size,
                config=config,
            )
            observation_design = _observation_design(case)
            rows = _locked_block_rows(
                partition,
                spec,
                case,
                payoff_bank,
                evidence_bank,
                learner_configs=learner_configs,
                observation_design=observation_design,
                plan=plan,
                block_size=block_size,
                selected=selected,
                raw_best=raw_best,
                config=config,
            )
            _write_json(checkpoint, {
                "partition": partition,
                "case": spec.to_dict(),
                "block_size": block_size,
                "freeze_digest_blake2b_128": freeze[
                    "freeze_digest_blake2b_128"
                ],
                "plan": {
                    **plan.to_dict(),
                    **plan_contract,
                },
                "observation_design": observation_design.to_dict(),
                "rows": rows,
            })
            partition_rows.extend(rows)
            del payoff_bank, evidence_bank
            gc.collect()
    summary = {
        "schema": (
            "histogram_tsallis_response_block_locked_partition.v1"
        ),
        "partition": partition,
        "freeze_digest_blake2b_128": freeze[
            "freeze_digest_blake2b_128"
        ],
        "execution_commit": _git_head(),
        "row_count": len(partition_rows),
        "rows": partition_rows,
        "aggregates": _aggregate_locked_rows(partition_rows),
        "bank_summaries": bank_rows,
        "full_corpus_exposition_in_scope": False,
    }
    _write_json(output / f"{partition}_results.json", summary)


def _locked_block_rows(
    partition: str,
    spec: FrozenCalibrationCase,
    case: Any,
    payoff_bank: HistogramPayoffBank,
    evidence_bank: HistogramEndpointEvidenceBank,
    *,
    learner_configs: Mapping[str, HistogramCoLearningConfig],
    observation_design: Any,
    plan: ControllerPrimeExploitPlan,
    block_size: int,
    selected: Mapping[tuple[str, int], SequentialResponseCandidate],
    raw_best: Mapping[tuple[str, int], SequentialResponseCandidate],
    config: TsallisResponseBlockMilestoneConfig,
) -> list[dict[str, Any]]:
    commissioning = _commissioning(
        evidence_bank,
        stop=config.commissioning_draws_per_route,
    )
    monitor = _monitor_candidate(config)
    rows: list[dict[str, Any]] = []
    for seed in config.execution_seeds:
        follow = _follow(
            case,
            payoff_bank,
            learner_configs,
            observation_design=observation_design,
            seed=seed,
            config=config,
        )
        for detector in config.active_detectors:
            candidate = selected[(detector, int(block_size))]
            for scenario, attack_enabled in (
                (_attack_scenario(plan), True),
                ("clean_no_attack_control", False),
            ):
                alarm = discover_alarm_response_record(
                    case,
                    payoff_bank,
                    evidence_bank,
                    learner_configs["tsallis_inf_rv_eta1"],
                    plan,
                    schedule=config.schedule,
                    alice=follow.clone_true_alice(
                        continuation_seed=seed
                    ),
                    observation_design=observation_design,
                    detectors=_detectors(commissioning, config),
                    active_detector=detector,
                    seed=seed,
                    attack_enabled=attack_enabled,
                    execution_split="audit",
                    execution_draw_start=config.audit_draw_start,
                    execution_draw_stop=config.audit_draw_stop,
                    selection_digest_blake2b_128=(
                        STEP6_SELECTION_DIGEST
                    ),
                )
                result_by_candidate: dict[str, dict[str, Any]] = {}
                for execution_role, active_candidate in (
                    ("selected_response", candidate),
                    (
                        "feedback_raw_best_response",
                        raw_best[(detector, int(block_size))],
                    ),
                    ("native_monitor_control", monitor),
                ):
                    cached = result_by_candidate.get(
                        active_candidate.candidate_id
                    )
                    if cached is not None:
                        row = dict(cached)
                        row["execution_role"] = execution_role
                        rows.append(row)
                        continue
                    response = run_alarm_conditioned_response_continuation(
                        case,
                        payoff_bank,
                        evidence_bank,
                        learner_configs["tsallis_inf_rv_eta1"],
                        plan,
                        schedule=config.schedule,
                        alice=follow.clone_true_alice(
                            continuation_seed=seed
                        ),
                        observation_design=observation_design,
                        detectors=_detectors(commissioning, config),
                        response_config=sequential_response_config(
                            config.response_calibration,
                            active_candidate,
                            detector,
                        ),
                        seed=seed,
                        attack_enabled=attack_enabled,
                        condition=scenario,
                        frozen_alarm_record=alarm,
                        execution_split="audit",
                        execution_draw_start=config.audit_draw_start,
                        execution_draw_stop=config.audit_draw_stop,
                        selection_digest_blake2b_128=(
                            STEP6_SELECTION_DIGEST
                        ),
                    )
                    row = response_result_row(
                        response,
                        partition=partition,
                        case_id=spec.case_id,
                        family=spec.family,
                        block_size=block_size,
                        active_detector=detector,
                        candidate=active_candidate,
                        execution_role=execution_role,
                    )
                    row["scenario"] = scenario
                    rows.append(row)
                    result_by_candidate[
                        active_candidate.candidate_id
                    ] = row
    return rows


def _ensure_case_banks(
    partition: str,
    spec: FrozenCalibrationCase,
    case: Any,
    *,
    config: TsallisResponseBlockMilestoneConfig,
    output: Path,
    force_rebuild: bool,
) -> dict[str, Any]:
    root = output / "banks" / partition / _slug(spec)
    summary_path = root / "summary.json"
    paths = {
        size: {
            "endpoint": root / f"endpoint_{size}.npz",
            "payoff": root / f"payoff_{size}.npz",
        }
        for size in config.block_sizes
    }
    if (
        summary_path.exists()
        and not force_rebuild
        and all(
            row["endpoint"].exists() and row["payoff"].exists()
            for row in paths.values()
        )
    ):
        return {
            "paths": paths,
            "summary": _read_json(summary_path),
        }

    maximum_config = _learner_configs(
        case,
        block_size=config.maximum_block_size,
        config=config,
    )["tsallis_inf_rv_eta1"]
    cache_plan = histogram_cache_plan(case, maximum_config)
    root.mkdir(parents=True, exist_ok=True)
    preflight = {
        "partition": partition,
        "case": spec.to_dict(),
        "route_count": len(case.routes),
        "eve_action_count": len(
            case.actions(maximum_config.game_mode)
        ),
        "nominal_route_action_cells": (
            len(case.routes)
            * len(case.actions(maximum_config.game_mode))
        ),
        "canonical_cache_entries": cache_plan[
            "canonical_unique_move_count_total"
        ],
        "estimated_resident_cache_bytes": cache_plan[
            "estimated_full_action_surface_cache_bytes"
        ],
        "estimated_resident_cache_mib": cache_plan[
            "estimated_full_action_surface_cache_mib"
        ],
        "configured_lru_capacity_per_route": (
            maximum_config.evaluated_cache_size
        ),
        "minimum_lru_capacity_per_route": cache_plan[
            "minimum_entries_per_route_without_eviction"
        ],
        "lru_sufficient_without_eviction": cache_plan[
            "sufficient_for_declared_action_surface_without_eviction"
        ],
        "available_ram_before_bank": available_ram_snapshot(),
        "paired_block_sizes": list(config.block_sizes),
        "samples_per_cell_per_split": (
            config.endpoint_samples_per_cell_per_split
        ),
    }
    _write_json(root / "preflight.json", preflight)
    print(
        f"bank preflight {spec.family}: {preflight['route_count']} routes, "
        f"{preflight['eve_action_count']} actions, "
        f"{preflight['canonical_cache_entries']} canonical entries, "
        f"{preflight['estimated_resident_cache_mib']:.1f} MiB"
    )

    physics = HistogramPhysicsPool(case, maximum_config)
    source_path = root / "endpoint_source_1200.npz"
    if source_path.exists() and not force_rebuild:
        source = HistogramEndpointEvidenceBank.load(source_path)
        validate_endpoint_evidence_bank_against_case(
            source,
            case,
            maximum_config,
            selection_digest_blake2b_128=STEP6_SELECTION_DIGEST,
        )
    else:
        source = build_histogram_endpoint_evidence_bank(
            case,
            maximum_config,
            samples_per_cell=(
                config.endpoint_samples_per_cell_per_split
            ),
            root_seed=config.endpoint_root_seed,
            physics=physics,
            progress_callback=print,
            selection_digest_blake2b_128=STEP6_SELECTION_DIGEST,
            evaluation_partition_loaded=partition == "evaluation",
        )
        source.save(source_path)
    paired = derive_paired_block_size_banks(
        case,
        maximum_config,
        source,
        physics=physics,
        block_sizes=config.block_sizes,
        thinning_root_seed=config.thinning_root_seed,
        selection_digest_blake2b_128=STEP6_SELECTION_DIGEST,
        evaluation_partition_loaded=partition == "evaluation",
    )
    block_rows = {}
    for size in config.block_sizes:
        paired.endpoint_banks[size].save(paths[size]["endpoint"])
        paired.payoff_banks[size].save(paths[size]["payoff"])
        block_rows[str(size)] = {
            "endpoint_path": str(paths[size]["endpoint"]),
            "payoff_path": str(paths[size]["payoff"]),
            "endpoint_digest_blake2b_128": (
                paired.endpoint_banks[size].content_digest()
            ),
            "payoff_digest_blake2b_128": (
                paired.payoff_banks[size].content_digest()
            ),
            "block_summary": paired.summary[
                "per_block_size"
            ][str(size)],
        }
    summary = {
        "partition": partition,
        "case": spec.to_dict(),
        "preflight": preflight,
        "paired_summary": paired.summary,
        "blocks": block_rows,
        "physics_diagnostics": physics.diagnostics(),
    }
    _write_json(summary_path, summary)
    del paired, source, physics
    gc.collect()
    return {"paths": paths, "summary": summary}


def _runtime_preflight(
    partition: str,
    loaded: Sequence[tuple[FrozenCalibrationCase, Any]],
    config: TsallisResponseBlockMilestoneConfig,
) -> dict[str, Any]:
    rows = []
    for spec, case in loaded:
        active = _learner_configs(
            case,
            block_size=config.maximum_block_size,
            config=config,
        )["tsallis_inf_rv_eta1"]
        plan = histogram_cache_plan(case, active)
        rows.append({
            "case_id": spec.case_id,
            "family": spec.family,
            "route_count": len(case.routes),
            "eve_action_count": len(
                case.actions(active.game_mode)
            ),
            "nominal_route_action_cells": (
                len(case.routes)
                * len(case.actions(active.game_mode))
            ),
            "canonical_cache_entries": plan[
                "canonical_unique_move_count_total"
            ],
            "estimated_resident_cache_bytes": plan[
                "estimated_full_action_surface_cache_bytes"
            ],
            "estimated_resident_cache_mib": plan[
                "estimated_full_action_surface_cache_mib"
            ],
            "configured_lru_capacity_per_route": (
                active.evaluated_cache_size
            ),
            "minimum_lru_capacity_per_route": plan[
                "minimum_entries_per_route_without_eviction"
            ],
            "lru_sufficient_without_eviction": plan[
                "sufficient_for_declared_action_surface_without_eviction"
            ],
        })
    return {
        "schema": "histogram_tsallis_response_runtime_preflight.v1",
        "partition": partition,
        "cases": rows,
        "canonical_cache_entries_total": sum(
            row["canonical_cache_entries"] for row in rows
        ),
        "estimated_resident_cache_bytes_total_if_concurrent": sum(
            row["estimated_resident_cache_bytes"] for row in rows
        ),
        "largest_single_case_cache_mib": max(
            row["estimated_resident_cache_mib"] for row in rows
        ),
        "execution": "one case physics pool resident at a time",
        "available_ram_before_run": available_ram_snapshot(),
    }


def _load_standard_banks(
    case_banks: Mapping[str, Any],
    block_size: int,
) -> tuple[HistogramPayoffBank, HistogramEndpointEvidenceBank]:
    row = case_banks["paths"][int(block_size)]
    return (
        HistogramPayoffBank.load(row["payoff"]),
        HistogramEndpointEvidenceBank.load(row["endpoint"]),
    )


def _learner_configs(
    case: Any,
    *,
    block_size: int,
    config: TsallisResponseBlockMilestoneConfig,
) -> dict[str, HistogramCoLearningConfig]:
    base = HistogramCoLearningConfig(
        turns=config.maximum_follow_turns + config.schedule.total_turns,
        requested_pairs_per_turn=int(block_size),
        final_window=min(1_000, config.schedule.total_turns),
        record_every=config.record_every,
        posthoc_samples_per_cell=0,
        seed=int(config.execution_seeds[0]),
        game_mode="attack_surface_all",
        age_points_per_edge=8,
        evaluated_cache_size=max(
            256,
            len(case.actions("attack_surface_all")) + 1,
        ),
    )
    return {
        "tsallis_inf_rv_eta1": replace(
            base,
            alice_learner_mode="tsallis_inf",
            alice_tsallis_inf_estimator="rv",
            alice_tsallis_inf_eta_scale=1.0,
        ),
        "linear_exp2_goptimal_eta2": replace(
            base,
            alice_learner_mode="linear_route_exp2",
            structured_exploration_mode="g_optimal",
            alice_exp3_eta_c=2.0,
        ),
    }


def _observation_design(case: Any) -> Any:
    design = minimum_route_observation_design(
        case,
        allowed_sensor_kinds=("node", "edge"),
    )
    if not design.exact_route_observation:
        raise ValueError(
            f"{case.family} lacks exact passive node/edge route separation"
        )
    return design


def _fixed_memory_exploit_plan(
    case: Any,
    bank: HistogramPayoffBank,
    learner_config: HistogramCoLearningConfig,
) -> tuple[ControllerPrimeExploitPlan, dict[str, Any]]:
    """Freeze one 1,200-pair memory-exploit plan across every block size."""

    try:
        plans = build_controller_prime_exploit_plans(
            case,
            bank,
            learner_config,
            prime_attack_types=("memory_degradation",),
            exploit_attack_types=("memory_degradation",),
            selection_digest_blake2b_128=STEP6_SELECTION_DIGEST,
        )
        contract = "memory_prime_and_memory_exploit"
    except ValueError as exc:
        if "no physical Prime/Exploit plan" not in str(exc):
            raise
        try:
            plans = build_controller_prime_exploit_plans(
                case,
                bank,
                learner_config,
                prime_attack_types=(
                    "edge_dos",
                    "swap_denial",
                    "memory_degradation",
                ),
                exploit_attack_types=("memory_degradation",),
                selection_digest_blake2b_128=STEP6_SELECTION_DIGEST,
            )
            contract = "availability_prime_memory_exploit"
        except ValueError as fallback_exc:
            if "no physical Prime/Exploit plan" not in str(fallback_exc):
                raise
            plans = build_controller_prime_exploit_plans(
                case,
                bank,
                learner_config,
                selection_digest_blake2b_128=STEP6_SELECTION_DIGEST,
            )
            contract = "feedback_best_feasible_controller"
    memory_exploit = plans[0].exploit_attack_type == "memory_degradation"
    return plans[0], {
        "cross_block_plan_source": "1200_pair_feedback_bank",
        "cross_block_plan_frozen": True,
        "plan_action_contract": contract,
        "exploit_is_memory_degradation": memory_exploit,
        "quantum_integrity_exploit_stratum": memory_exploit,
        "quantum_increment_interpretation": (
            "when availability interventions occur during Prime, detector "
            "differences in that phase are not an integrity-only comparison; "
            "a non-memory Exploit is additionally an availability-controller "
            "stratum and cannot support a memory-integrity increment claim"
        ),
    }


def _attack_scenario(plan: ControllerPrimeExploitPlan) -> str:
    return (
        "memory_exploit_controller_attack"
        if plan.exploit_attack_type == "memory_degradation"
        else "availability_controller_attack"
    )


def _follow(
    case: Any,
    bank: HistogramPayoffBank,
    learner_configs: Mapping[str, HistogramCoLearningConfig],
    *,
    observation_design: Any,
    seed: int,
    config: TsallisResponseBlockMilestoneConfig,
) -> Any:
    return run_stackelberg_bandit_follow_runtime(
        case,
        bank,
        true_alice_config=learner_configs["tsallis_inf_rv_eta1"],
        hypothesis_configs=learner_configs,
        observation_design=observation_design,
        turns=config.maximum_follow_turns,
        seed=int(seed),
        true_hypothesis_label="tsallis_inf_rv_eta1",
        record_every=max(1, min(10, config.record_every)),
        decision_posterior_threshold=(
            config.follow_posterior_threshold
        ),
        selection_digest_blake2b_128=STEP6_SELECTION_DIGEST,
    )


def _commissioning(
    bank: HistogramEndpointEvidenceBank,
    *,
    stop: int,
) -> tuple[tuple[Any, ...], ...]:
    no_attack = bank.action_ids.index("no_attack")
    return tuple(
        tuple(
            bank.feedback.evidence(
                route_index,
                no_attack,
                draw_index,
                requested_pair_count=bank.requested_pairs_per_draw,
            )
            for draw_index in range(stop)
        )
        for route_index in range(len(bank.route_ids))
    )


def _detectors(
    commissioning: Sequence[Sequence[Any]],
    config: TsallisResponseBlockMilestoneConfig,
) -> MatchedClassicalQuantumEndpointDetectors:
    return MatchedClassicalQuantumEndpointDetectors.from_commissioning(
        commissioning,
        config.detector_config,
    )


def _selected_candidates_by_block(
    freeze: Mapping[str, Any],
) -> dict[tuple[str, int], SequentialResponseCandidate]:
    selected = {}
    for row in freeze["selection"]["per_block"]:
        payload = row["selected_candidate"]
        selected[(str(row["active_detector"]), int(row["block_size"]))] = (
            _candidate_from_dict(payload)
        )
    return selected


def _raw_best_candidates_by_block(
    freeze: Mapping[str, Any],
) -> dict[tuple[str, int], SequentialResponseCandidate]:
    selected = {}
    for row in freeze["selection"]["per_block"]:
        payload = row["raw_best"]["candidate"]
        selected[(str(row["active_detector"]), int(row["block_size"]))] = (
            _candidate_from_dict(payload)
        )
    return selected


def _monitor_candidate(
    config: TsallisResponseBlockMilestoneConfig,
) -> SequentialResponseCandidate:
    monitors = [
        candidate
        for candidate in config.response_calibration.candidates()
        if candidate.containment_ceiling == 0.0
        and candidate.maximum_total_variation == 0.0
    ]
    if len(monitors) != 1:
        raise ValueError("response grid must contain one monitor candidate")
    return monitors[0]


def _candidate_from_dict(
    payload: Mapping[str, Any],
) -> SequentialResponseCandidate:
    return SequentialResponseCandidate(**{
        key: payload[key]
        for key in (
            "maximum_total_variation",
            "arming_turns",
            "containment_ceiling",
            "posterior_window_turns",
            "posterior_refresh_turns",
        )
    })


def _alarm_row(
    alarm: Mapping[str, Any],
    *,
    spec: FrozenCalibrationCase,
    block_size: int,
    detector: str,
    seed: int,
    scenario: str,
) -> dict[str, Any]:
    turn = alarm.get("alarm_turn")
    return {
        "case_id": spec.case_id,
        "family": spec.family,
        "block_size": int(block_size),
        "active_detector": str(detector),
        "seed": int(seed),
        "scenario": str(scenario),
        "alarm_turn": turn,
        "alarm_requested_pairs": (
            int(turn) * int(block_size)
            if turn is not None
            else None
        ),
        "execution_split": str(alarm["execution_split"]),
    }


def _aggregate_locked_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    values = tuple(dict(row) for row in rows)
    aggregates = []
    keys = sorted({
        (
            str(row["scenario"]),
            str(row["active_detector"]),
            int(row["block_size"]),
        )
        for row in values
    })
    for scenario, detector, block_size in keys:
        native = [
            row for row in values
            if row["scenario"] == scenario
            and row["active_detector"] == detector
            and int(row["block_size"]) == block_size
            and row["execution_role"] == "native_monitor_control"
        ]
        native_by_key = {
            (str(row["case_id"]), int(row["seed"])): row
            for row in native
        }
        for response_role in (
            "selected_response",
            "feedback_raw_best_response",
        ):
            response_rows = [
                row for row in values
                if row["scenario"] == scenario
                and row["active_detector"] == detector
                and int(row["block_size"]) == block_size
                and row["execution_role"] == response_role
            ]
            deltas = np.asarray([
                float(row["mean_alice_reward"])
                - float(native_by_key[
                    (str(row["case_id"]), int(row["seed"]))
                ]["mean_alice_reward"])
                for row in response_rows
            ])
            alarm_turns = [
                int(row["alarm_turn"])
                for row in response_rows
                if row["alarm_turn"] is not None
            ]
            alarm_pairs = [
                int(row["alarm_requested_pairs"])
                for row in response_rows
                if row["alarm_requested_pairs"] is not None
            ]
            aggregates.append({
                "scenario": scenario,
                "active_detector": detector,
                "block_size": block_size,
                "response_role": response_role,
                "run_pairs": len(response_rows),
                "alarm_rate": (
                    len(alarm_turns) / len(response_rows)
                    if response_rows else 0.0
                ),
                "mean_alarm_turn_given_alarm": (
                    float(np.mean(alarm_turns))
                    if alarm_turns else None
                ),
                "mean_alarm_requested_pairs_given_alarm": (
                    float(np.mean(alarm_pairs))
                    if alarm_pairs else None
                ),
                "mean_response_minus_native_reward": float(
                    np.mean(deltas)
                ),
                "standard_error_response_minus_native_reward": (
                    float(
                        np.std(deltas, ddof=1)
                        / np.sqrt(deltas.size)
                    )
                    if deltas.size > 1
                    else 0.0
                ),
                "mean_response_reward": float(np.mean([
                    float(row["mean_alice_reward"])
                    for row in response_rows
                ])),
                "mean_native_reward": float(np.mean([
                    float(row["mean_alice_reward"]) for row in native
                ])),
            })
    return aggregates


def _write_final_summary(
    output: Path,
    config: TsallisResponseBlockMilestoneConfig,
    freeze: Mapping[str, Any],
) -> None:
    calibration_path = output / "calibration_results.json"
    evaluation_path = output / "evaluation_results.json"
    summary = {
        "schema": RESPONSE_BLOCK_MILESTONE_SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_digest_blake2b_128": freeze[
            "freeze_digest_blake2b_128"
        ],
        "execution_commit": _git_head(),
        "selection": freeze["selection"],
        "calibration_validation": (
            _read_json(calibration_path)
            if calibration_path.exists()
            else None
        ),
        "untouched_evaluation": (
            _read_json(evaluation_path)
            if evaluation_path.exists()
            else None
        ),
        "block_size_clean_evidence": _collect_block_summaries(output),
        "latency_units": {
            "turns": "number of endpoint observation blocks",
            "requested_pairs": "alarm turn multiplied by block size",
        },
        "scientific_boundary": {
            "physics": (
                "compiled histogram E91/BBM92-style endpoint transcripts "
                "with finite delivery, key-error, and CHSH counts"
            ),
            "response_calibration_split": "feedback only",
            "calibration_audit_split": "fresh audit after freeze",
            "evaluation_partition_loaded_after_freeze": True,
            "tsallis_control": (
                "behavior-policy importance correction is unbiased for the "
                "declared mixture; native regret theorem is not asserted"
            ),
            "diagnostic_blocks": list(config.diagnostic_block_sizes),
            "primary_blocks": list(config.primary_block_sizes),
        },
        "full_corpus_exposition_started": False,
        "ready_for_user_validation_before_full_corpus": bool(
            calibration_path.exists() and evaluation_path.exists()
        ),
    }
    _write_json(output / "summary.json", summary)


def _collect_block_summaries(output: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((output / "banks").glob("*/*/summary.json")):
        payload = _read_json(path)
        for size, row in payload["blocks"].items():
            rows.append({
                "partition": payload["partition"],
                "case_id": payload["case"]["case_id"],
                "family": payload["case"]["family"],
                "block_size": int(size),
                **dict(row["block_summary"]),
            })
    return rows


def _slug(spec: FrozenCalibrationCase) -> str:
    return (
        f"{spec.selection_order:02d}_"
        f"{spec.family.lower().replace(' ', '_')}"
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
