"""Single-topology study of public detection and steering mitigation."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .calibration_bank import (
    HistogramPayoffBank,
    build_histogram_payoff_bank,
    validate_bank_against_case,
)
from .calibration_game import (
    CalibrationGameOracle,
    build_calibration_game_oracle,
    run_bank_two_player_learning,
)
from .config import HistogramCoLearningConfig, REPO_ROOT
from .corpus import HistogramGameCase
from .corpus_scale import load_corpus_scale_case
from .physics import HistogramPhysicsPool, histogram_cache_plan
from .stackelberg_detection import (
    DEFAULT_DETECTION_MITIGATION_CONDITIONS,
    DetectionMitigationCondition,
    StackelbergDetectionMitigationConfig,
    run_stackelberg_detection_mitigation,
    slice_payoff_sample_block,
)
from .stackelberg_eve import StackelbergCommitment, solve_stackelberg_oracle
from .stackelberg_eve_corpus import (
    DEFAULT_CORPUS_RUN_DIR,
    STACKELBERG_DEFENDERS,
    StackelbergDefender,
)
from .topology_census import available_ram_snapshot


STACKELBERG_DETECTION_STUDY_SCHEMA = (
    "histogram_stackelberg_detection_mitigation_study.v5"
)
DEFAULT_STACKELBERG_DETECTION_RUN_DIR = (
    REPO_ROOT / "runs" / "histogram_stackelberg_detection_cernet_100k"
)
DEFAULT_DETECTION_SEEDS = (
    20_260_719_01,
    20_260_719_02,
    20_260_719_03,
)
_BANK_BYTES_PER_CELL_DRAW_BOTH_SPLITS = 84
_WORKER_CONTEXT: tuple[
    HistogramGameCase,
    HistogramPayoffBank,
    HistogramCoLearningConfig,
    StackelbergDetectionMitigationConfig,
    StackelbergCommitment,
    str,
    str,
] | None = None
_COLEARNING_CONTROL_CONTEXT: tuple[
    HistogramGameCase,
    HistogramPayoffBank,
    HistogramCoLearningConfig,
    CalibrationGameOracle,
    str,
    str,
    int,
    tuple[int, ...],
    int,
] | None = None


ProgressCallback = Callable[[str], None]


def build_cernet_detection_manifest(
    corpus_manifest: Mapping[str, Any],
    *,
    turns: int = 100_000,
    samples_per_cell_per_split: int = 96,
    seeds: Sequence[int] = DEFAULT_DETECTION_SEEDS,
    experiment: StackelbergDetectionMitigationConfig | None = None,
    conditions: Sequence[DetectionMitigationCondition] = (
        DEFAULT_DETECTION_MITIGATION_CONDITIONS
    ),
) -> dict[str, Any]:
    """Freeze the Cernet case, information boundary, and matched conditions."""

    if turns < 2:
        raise ValueError("turns must be at least two")
    if samples_per_cell_per_split < 4:
        raise ValueError("samples_per_cell_per_split must be at least four")
    seed_values = tuple(map(int, seeds))
    if not seed_values or len(set(seed_values)) != len(seed_values):
        raise ValueError("seeds must be nonempty and unique")
    condition_values = tuple(conditions)
    if (
        not condition_values
        or len({row.name for row in condition_values}) != len(condition_values)
    ):
        raise ValueError("conditions must be nonempty and uniquely named")
    case_rows = [
        dict(row)
        for row in corpus_manifest["cases"]
        if str(row["family"]).lower() == "cernet"
    ]
    if len(case_rows) != 1:
        raise ValueError("corpus manifest must contain exactly one Cernet case")
    case_entry = case_rows[0]
    active_experiment = experiment or StackelbergDetectionMitigationConfig()
    active_experiment.validate(
        turns=turns,
        draws_per_cell=samples_per_cell_per_split,
    )
    estimated_bank_mib = (
        int(case_entry["nominal_route_action_cells"])
        * samples_per_cell_per_split
        * _BANK_BYTES_PER_CELL_DRAW_BOTH_SPLITS
        / 1024**2
    )
    stable = {
        "source_corpus_selection_digest_blake2b_128": str(
            corpus_manifest["selection_digest_blake2b_128"]
        ),
        "case": case_entry,
        "turns": int(turns),
        "samples_per_cell_per_split": int(samples_per_cell_per_split),
        "seeds": list(seed_values),
        "defenders": [row.to_dict() for row in STACKELBERG_DEFENDERS],
        "conditions": [row.to_dict() for row in condition_values],
        "experiment_config": active_experiment.to_dict(),
        "detector_implementation": (
            "route_conditioned_dirichlet_posterior_predictive_"
            "finite_horizon_mixture_e_process.v3"
        ),
        "detector_transition_telemetry": (
            "every turn from 50 turns before through 250 turns after the "
            "declared attack onset"
        ),
    }
    digest = hashlib.blake2b(
        json.dumps(stable, sort_keys=True, separators=(",", ":")).encode(),
        digest_size=16,
    ).hexdigest()
    return {
        "schema": STACKELBERG_DETECTION_STUDY_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_digest_blake2b_128": digest,
        **stable,
        "source_corpus_plan": dict(corpus_manifest["plan"]),
        "expected_run_count": (
            len(STACKELBERG_DEFENDERS)
            * len(condition_values)
            * len(seed_values)
        ),
        "preflight": {
            "graph_id": str(case_entry["graph_id"]),
            "family": str(case_entry["family"]),
            "route_count": int(case_entry["route_count"]),
            "eve_action_count": int(case_entry["eve_action_count"]),
            "nominal_route_action_cells": int(
                case_entry["nominal_route_action_cells"]
            ),
            "canonical_cache_entries": int(
                case_entry["canonical_cache_entries"]
            ),
            "estimated_physics_cache_mib": float(
                case_entry["estimated_cache_mib"]
            ),
            "estimated_feedback_audit_bank_mib": float(
                estimated_bank_mib
            ),
            "available_ram_at_manifest": available_ram_snapshot(),
            "run_parallelism_boundary": (
                "one compiled bank is created on demand; forked learner "
                "workers share its read-only pages"
            ),
        },
        "information_boundary": {
            "detector": (
                "selected route identity, public terminal outcome, and fixed "
                "clean commissioning counts only"
            ),
            "service_learner": (
                "selected service-route scalar public reward only"
            ),
            "sentinel_selection": (
                "topology-only G-optimal route-resource design"
            ),
            "audit_split": "held-out scoring only",
            "simulator_private_state_disclosed_online": False,
        },
        "scope": (
            "opt-in Cernet method study; existing steering and co-learning "
            "runners remain unchanged"
        ),
    }


def run_cernet_detection_study(
    corpus_manifest: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_STACKELBERG_DETECTION_RUN_DIR,
    workers: int = 2,
    force_rebuild_bank: bool = False,
    bank_only: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Compile one Cernet bank and run every matched learner/controller pair."""

    if manifest.get("schema") != STACKELBERG_DETECTION_STUDY_SCHEMA:
        raise ValueError("incompatible detection-study manifest")
    if workers < 1:
        raise ValueError("workers must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / "run_manifest.json", manifest)
    entry = dict(manifest["case"])
    case = load_corpus_scale_case(entry, corpus_manifest)
    source_plan = manifest["source_corpus_plan"]
    base_config = HistogramCoLearningConfig(
        turns=int(manifest["turns"]),
        requested_pairs_per_turn=int(source_plan["requested_pairs_per_turn"]),
        final_window=int(source_plan["final_window"]),
        record_every=int(manifest["experiment_config"]["record_every"]),
        posthoc_samples_per_cell=0,
        seed=int(manifest["seeds"][0]),
        game_mode="attack_surface_all",
        age_points_per_edge=int(source_plan["age_points_per_edge"]),
        evaluated_cache_size=max(
            256, int(entry["eve_action_count"]) + 1
        ),
    )
    cache_plan = histogram_cache_plan(case, base_config)
    if not cache_plan[
        "sufficient_for_declared_action_surface_without_eviction"
    ]:
        base_config = replace(
            base_config,
            evaluated_cache_size=int(
                cache_plan["minimum_entries_per_route_without_eviction"]
            ),
        )
        cache_plan = histogram_cache_plan(case, base_config)
    _write_json_atomic(
        output / "preflight.json",
        {
            "manifest": dict(manifest["preflight"]),
            "active_cache_plan": cache_plan,
            "available_ram_before_bank": available_ram_snapshot(),
            "workers": int(workers),
        },
    )

    bank_path = output / "cernet_payoff_bank.npz"
    selection_digest = str(
        manifest["source_corpus_selection_digest_blake2b_128"]
    )
    if bank_path.exists() and not force_rebuild_bank:
        progress = progress_callback or (lambda _: None)
        progress(f"loading verified payoff bank {bank_path}")
        bank = HistogramPayoffBank.load(bank_path)
        validate_bank_against_case(
            bank,
            case,
            base_config,
            selection_digest_blake2b_128=selection_digest,
        )
        if bank.draws_per_cell != int(
            manifest["samples_per_cell_per_split"]
        ):
            raise ValueError("existing payoff bank draw count changed")
        physics_diagnostics = bank.metadata.get("physics")
    else:
        progress = progress_callback or (lambda _: None)
        progress("compiling Cernet histogram physics bank on demand")
        physics = HistogramPhysicsPool(case, base_config)
        bank = build_histogram_payoff_bank(
            case,
            base_config,
            samples_per_cell=int(
                manifest["samples_per_cell_per_split"]
            ),
            root_seed=20_260_719_41,
            physics=physics,
            progress_callback=progress,
            selection_digest_blake2b_128=selection_digest,
            evaluation_partition_loaded=True,
            sampling_purpose=(
                "stackelberg_public_detection_and_mitigation"
            ),
            sampling_design=(
                "feedback draws are partitioned into clean commissioning and "
                "online feedback; the audit split remains held out"
            ),
        )
        bank.save(bank_path)
        physics_diagnostics = physics.diagnostics()
        progress(f"saved verified payoff bank {bank_path}")

    if bank_only:
        summary = {
            "schema": "histogram_stackelberg_detection_bank_build.v1",
            "bank_only": True,
            "bank_path": str(bank_path),
            "bank": bank.to_summary(),
            "physics": physics_diagnostics,
        }
        _write_json_atomic(output / "bank_summary.json", summary)
        return summary

    experiment = StackelbergDetectionMitigationConfig(
        **dict(manifest["experiment_config"])
    )
    online_feedback = slice_payoff_sample_block(
        bank.feedback,
        experiment.commissioning_draws_per_route,
        bank.draws_per_cell,
    )
    commitment = solve_stackelberg_oracle(
        online_feedback.alice_payoff_matrix(),
        bank.action_costs,
        minimum_response_margin=experiment.response_margin,
    ).selected
    _mark_bank_read_only(bank)

    defenders = [
        StackelbergDefender(
            label=str(row["label"]),
            alice_learner_mode=str(row["alice_learner_mode"]),
            config_overrides=dict(row["config_overrides"]),
        )
        for row in manifest["defenders"]
    ]
    conditions = [
        DetectionMitigationCondition(**dict(row))
        for row in manifest["conditions"]
    ]
    tasks = [
        {
            "defender": defender.to_dict(),
            "condition": condition.to_dict(),
            "seed": int(seed),
        }
        for defender in defenders
        for condition in conditions
        for seed in manifest["seeds"]
    ]
    existing = _load_existing_results(
        output / "results.jsonl",
        experiment_digest=str(
            manifest["experiment_digest_blake2b_128"]
        ),
    )
    existing_by_key = {
        _run_key(row): row for row in existing
    }
    pending = [
        task for task in tasks if _task_key(task) not in existing_by_key
    ]
    results = list(existing)

    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (
        case,
        bank,
        base_config,
        experiment,
        commitment,
        selection_digest,
        str(manifest["experiment_digest_blake2b_128"]),
    )
    progress(
        f"running {len(pending)} pending learner/controller cases "
        f"with {workers} worker(s)"
    )
    if workers == 1:
        for task in pending:
            row = _run_worker_task(task)
            results.append(row)
            _write_json_lines(output / "results.jsonl", results)
            progress(_completion_message(row, len(results), len(tasks)))
    else:
        context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)) if pending else 1,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_run_worker_task, task): task
                for task in pending
            }
            for future in as_completed(futures):
                row = future.result()
                results.append(row)
                _write_json_lines(output / "results.jsonl", results)
                progress(_completion_message(row, len(results), len(tasks)))
    _WORKER_CONTEXT = None

    results.sort(key=_result_sort_key)
    _write_json_lines(output / "results.jsonl", results)
    colearning_controls = _run_colearning_controls(
        case=case,
        bank=bank,
        online_feedback=online_feedback,
        base_config=base_config,
        selection_digest=selection_digest,
        experiment_digest=str(
            manifest["experiment_digest_blake2b_128"]
        ),
        evaluation_start_turn=experiment.attack_start_turn + 1,
        damage_window_turns=experiment.damage_window_turns,
        record_every=experiment.record_every,
        defenders=defenders,
        seeds=tuple(map(int, manifest["seeds"])),
        output=output,
        workers=workers,
        progress=progress,
    )
    summary = {
        "schema": STACKELBERG_DETECTION_STUDY_SCHEMA,
        "experiment_digest_blake2b_128": str(
            manifest["experiment_digest_blake2b_128"]
        ),
        "graph_id": case.graph_id,
        "family": case.family,
        "run_count": len(results),
        "expected_run_count": len(tasks),
        "complete": len(results) == len(tasks),
        "bank": bank.to_summary(),
        "bank_path": str(bank_path),
        "cache_plan": cache_plan,
        "physics": physics_diagnostics,
        "commitment": commitment.to_dict(
            action_ids=bank.action_ids,
            route_ids=bank.route_ids,
        ),
        "aggregates": _aggregate_results(results),
        "colearning_controls": _aggregate_colearning_controls(
            colearning_controls
        ),
        "available_ram_after_run": available_ram_snapshot(),
    }
    _write_json_atomic(output / "summary.json", summary)
    return summary


def _run_worker_task(task: Mapping[str, Any]) -> dict[str, Any]:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("detection-study worker context is unavailable")
    (
        case,
        bank,
        base_config,
        experiment,
        commitment,
        selection_digest,
        experiment_digest,
    ) = (
        _WORKER_CONTEXT
    )
    defender = StackelbergDefender(
        label=str(task["defender"]["label"]),
        alice_learner_mode=str(task["defender"]["alice_learner_mode"]),
        config_overrides=dict(task["defender"]["config_overrides"]),
    )
    condition = DetectionMitigationCondition(**dict(task["condition"]))
    seed = int(task["seed"])
    active_config = replace(
        base_config,
        **defender.config_overrides,
        seed=seed,
    )
    result = run_stackelberg_detection_mitigation(
        case,
        bank,
        active_config,
        experiment,
        condition,
        seed=seed,
        selection_digest_blake2b_128=selection_digest,
        precomputed_commitment=commitment,
    )
    result.update({
        "defender_label": defender.label,
        "experiment_digest_blake2b_128": experiment_digest,
        "bank_content_digest_blake2b_128": bank.content_digest(),
    })
    return result


def _run_colearning_controls(
    *,
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    online_feedback: Any,
    base_config: HistogramCoLearningConfig,
    selection_digest: str,
    experiment_digest: str,
    evaluation_start_turn: int,
    damage_window_turns: Sequence[int],
    record_every: int,
    defenders: Sequence[StackelbergDefender],
    seeds: Sequence[int],
    output: Path,
    workers: int,
    progress: ProgressCallback,
) -> list[dict[str, Any]]:
    comparison_audit = slice_payoff_sample_block(
        bank.audit,
        0,
        online_feedback.draws_per_cell,
    )
    comparison_bank = replace(
        bank,
        feedback=online_feedback,
        audit=comparison_audit,
        metadata={
            **bank.metadata,
            "sampling_purpose": (
                "matched_ordinary_tsallis_eve_vulnerability_control"
            ),
        },
    )
    oracle = build_calibration_game_oracle(comparison_bank)
    tasks = [
        {
            "defender": defender.to_dict(),
            "seed": int(seed),
        }
        for defender in defenders
        for seed in seeds
    ]
    path = output / "colearning_controls.jsonl"
    existing = _load_existing_control_results(
        path,
        experiment_digest=experiment_digest,
    )
    existing_by_key = {
        _control_run_key(row): row for row in existing
    }
    pending = [
        task for task in tasks
        if _control_task_key(task) not in existing_by_key
    ]
    rows = list(existing)
    global _COLEARNING_CONTROL_CONTEXT
    _COLEARNING_CONTROL_CONTEXT = (
        case,
        comparison_bank,
        base_config,
        oracle,
        selection_digest,
        experiment_digest,
        int(evaluation_start_turn),
        tuple(map(int, damage_window_turns)),
        int(record_every),
    )
    progress(
        f"running {len(pending)} pending ordinary Tsallis-Eve controls "
        f"with {workers} worker(s)"
    )
    if workers == 1:
        for task in pending:
            row = _run_colearning_control_task(task)
            rows.append(row)
            _write_control_json_lines(path, rows)
    else:
        context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=min(workers, len(pending)) if pending else 1,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_run_colearning_control_task, task): task
                for task in pending
            }
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                _write_control_json_lines(path, rows)
                progress(
                    f"co-learning control: {row['defender_label']} / "
                    f"seed {row['seed']}"
                )
    _COLEARNING_CONTROL_CONTEXT = None
    rows.sort(key=_control_result_sort_key)
    _write_control_json_lines(path, rows)
    return rows


def _run_colearning_control_task(
    task: Mapping[str, Any],
) -> dict[str, Any]:
    if _COLEARNING_CONTROL_CONTEXT is None:
        raise RuntimeError("co-learning control worker context is unavailable")
    (
        case,
        bank,
        base_config,
        oracle,
        selection_digest,
        experiment_digest,
        evaluation_start_turn,
        damage_window_turns,
        record_every,
    ) = _COLEARNING_CONTROL_CONTEXT
    defender = StackelbergDefender(
        label=str(task["defender"]["label"]),
        alice_learner_mode=str(task["defender"]["alice_learner_mode"]),
        config_overrides=dict(task["defender"]["config_overrides"]),
    )
    seed = int(task["seed"])
    config = replace(
        base_config,
        **defender.config_overrides,
        seed=seed,
        eve_learner_mode="tsallis_inf",
        eve_tsallis_inf_estimator="rv",
        eve_tsallis_inf_eta_scale=1.0,
    )
    run = run_bank_two_player_learning(
        case,
        bank,
        config,
        seed=seed,
        record_every=record_every,
        oracle=oracle,
        selection_digest_blake2b_128=selection_digest,
        reward_window_turns=damage_window_turns,
        reward_window_start_turn=evaluation_start_turn,
    )
    audit = bank.audit.alice_payoff_matrix()
    alice_empirical = np.asarray(run["alice_strategy"], dtype=float)
    eve_empirical = np.asarray(run["eve_strategy"], dtype=float)
    alice_current = np.asarray(run["alice_current_policy"], dtype=float)
    eve_current = np.asarray(run["eve_current_policy"], dtype=float)
    empirical_reward = float(alice_empirical @ audit @ eve_empirical)
    empirical_best = float(np.max(audit @ eve_empirical))
    current_reward = float(alice_current @ audit @ eve_current)
    current_best = float(np.max(audit @ eve_current))
    oracle_reward = float(
        oracle.alice_strategy @ audit @ oracle.eve_strategy
    )
    return {
        "schema": "histogram_cernet_colearning_damage_control.v2",
        "experiment_digest_blake2b_128": experiment_digest,
        "defender_label": defender.label,
        "seed": seed,
        "turns": int(config.turns),
        "alice_learner": run["alice_learner"],
        "eve_learner": run["eve_learner"],
        "mean_alice_reward": float(run["mean_alice_reward"]),
        "evaluation_start_turn": int(evaluation_start_turn),
        "mean_pre_evaluation_alice_reward": float(
            run["mean_alice_reward_before_evaluation_interval"]
        ),
        "mean_evaluation_interval_alice_reward": float(
            run["mean_alice_reward_evaluation_interval"]
        ),
        "alice_reward_window_minima": run[
            "alice_reward_window_minima"
        ],
        "trajectory": run["trajectory"],
        "schedule_semantics": (
            "ordinary simultaneous Alice/Eve co-learning begins at turn 1; "
            "damage-window evaluation begins at the strict study's declared "
            "attack onset for a common analysis interval"
        ),
        "alice_empirical_strategy": run["alice_strategy"],
        "eve_empirical_strategy": run["eve_strategy"],
        "alice_current_policy": run["alice_current_policy"],
        "eve_current_policy": run["eve_current_policy"],
        "audit_empirical_profile_alice_reward": empirical_reward,
        "audit_empirical_best_response_reward": empirical_best,
        "audit_empirical_reward_retention": (
            empirical_reward / empirical_best
            if empirical_best > 0.0 else None
        ),
        "audit_current_profile_alice_reward": current_reward,
        "audit_current_best_response_reward": current_best,
        "audit_current_reward_retention": (
            current_reward / current_best
            if current_best > 0.0 else None
        ),
        "audit_equilibrium_alice_reward": oracle_reward,
        "strategic": run["strategic"],
        "information_boundary": run["information_boundary"],
    }


def _aggregate_colearning_controls(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["defender_label"]), []).append(row)
    return [
        {
            "defender_label": defender,
            "seed_count": len(group),
            "mean_alice_reward": _mean([
                row["mean_alice_reward"] for row in group
            ]),
            "mean_evaluation_interval_alice_reward": _mean([
                row["mean_evaluation_interval_alice_reward"]
                for row in group
            ]),
            "mean_audit_current_profile_alice_reward": _mean([
                row["audit_current_profile_alice_reward"]
                for row in group
            ]),
            "mean_audit_current_best_response_reward": _mean([
                row["audit_current_best_response_reward"]
                for row in group
            ]),
            "mean_audit_current_reward_retention": _mean([
                row["audit_current_reward_retention"]
                for row in group
            ]),
            "alice_reward_window_minima": (
                _aggregate_reward_window_minima(
                    group,
                    field="alice_reward_window_minima",
                )
            ),
            "mean_exploitability": _mean([
                row["strategic"]["exploitability"] for row in group
            ]),
        }
        for defender, group in sorted(groups.items())
    ]


def _aggregate_results(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in results:
        groups.setdefault(
            (str(row["defender_label"]), str(row["condition"]["name"])),
            [],
        ).append(row)
    aggregates = []
    for (defender, condition), rows in sorted(groups.items()):
        delays = [
            float(row["detection_delay_turns"])
            for row in rows
            if row.get("detection_delay_turns") is not None
        ]
        aggregates.append({
            "defender_label": defender,
            "condition": condition,
            "seed_count": len(rows),
            "pre_attack_alarm_rate": _mean(
                [bool(row["pre_attack_alarm"]) for row in rows]
            ),
            "post_attack_detection_rate": _mean(
                [bool(row["detected_after_attack"]) for row in rows]
            ),
            "mean_detection_delay_turns_detected": (
                _mean(delays) if delays else None
            ),
            "mean_quantum_four_outcome_detection_delay_turns": _optional_mean([
                row["observer_comparison"][
                    "quantum_four_outcome_detection_delay_turns"
                ]
                for row in rows
                if row["observer_comparison"][
                    "quantum_four_outcome_detection_delay_turns"
                ] is not None
            ]),
            "mean_classical_binary_service_detection_delay_turns": (
                _optional_mean([
                    row["observer_comparison"][
                        "classical_binary_service_detection_delay_turns"
                    ]
                    for row in rows
                    if row["observer_comparison"][
                        "classical_binary_service_detection_delay_turns"
                    ] is not None
                ])
            ),
            "mean_monitoring_request_fraction": _mean(
                [row["monitoring_request_fraction"] for row in rows]
            ),
            "mean_service_reward": _mean(
                [row["mean_service_reward"] for row in rows]
            ),
            "mean_clean_window_service_reward": _optional_mean([
                row["mean_clean_window_service_reward"]
                for row in rows
                if row["mean_clean_window_service_reward"] is not None
            ]),
            "mean_attack_window_service_reward": _optional_mean([
                row["mean_attack_window_service_reward"]
                for row in rows
                if row["mean_attack_window_service_reward"] is not None
            ]),
            "application_reward_window_minima": (
                _aggregate_reward_window_minima(
                    rows,
                    field="application_reward_window_minima",
                )
            ),
            "mean_application_reward_per_total_request": _mean([
                row["mean_service_reward"]
                * row["service_request_count"]
                / row["turns"]
                for row in rows
            ]),
            "mean_attack_window_eve_public_utility": _optional_mean([
                row["mean_attack_window_eve_public_utility"]
                for row in rows
                if row["mean_attack_window_eve_public_utility"] is not None
            ]),
            "mean_final_intended_route_service_mass": _mean([
                row["final_target_mass"][
                    "intended_route_service_request_mass"
                ]
                for row in rows
            ]),
            "mean_final_intended_route_policy_mass": _mean([
                row["final_target_mass"][
                    "intended_route_current_service_policy_mass"
                ]
                for row in rows
            ]),
            "mean_learner_restart_count": _mean(
                [len(row["learner_restart_turns"]) for row in rows]
            ),
            "service_policy_freeze_rate": _mean([
                row["service_policy_frozen_turn"] is not None
                for row in rows
            ]),
        })
    return aggregates


def _aggregate_reward_window_minima(
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> list[dict[str, Any]]:
    by_window: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        for window in row[field]:
            by_window.setdefault(
                int(window["window_turns"]),
                [],
            ).append(window)
    if any(len(group) != len(rows) for group in by_window.values()):
        raise ValueError(f"{field} has inconsistent window coverage")
    return [
        {
            "window_turns": window_turns,
            "mean_minimum_reward": _mean([
                row["mean_reward"] for row in group
            ]),
            "median_window_start_turn": float(np.median([
                row["window_start_turn"] for row in group
            ])),
            "minimum_window_start_turn": int(min(
                row["window_start_turn"] for row in group
            )),
            "maximum_window_end_turn": int(max(
                row["window_end_turn"] for row in group
            )),
        }
        for window_turns, group in sorted(by_window.items())
    ]


def _mark_bank_read_only(bank: HistogramPayoffBank) -> None:
    bank.action_costs.flags.writeable = False
    for split in ("feedback", "audit"):
        block = getattr(bank, split)
        for values in vars(block).values():
            if isinstance(values, np.ndarray):
                values.flags.writeable = False


def _load_existing_results(
    path: Path,
    *,
    experiment_digest: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(
        row.get("experiment_digest_blake2b_128") != experiment_digest
        for row in rows
    ):
        raise ValueError("existing results belong to a different experiment")
    if len({_run_key(row) for row in rows}) != len(rows):
        raise ValueError("existing results contain duplicate run identities")
    return rows


def _load_existing_control_results(
    path: Path,
    *,
    experiment_digest: str,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any(
        row.get("experiment_digest_blake2b_128") != experiment_digest
        for row in rows
    ):
        raise ValueError(
            "existing co-learning controls belong to another experiment"
        )
    if len({_control_run_key(row) for row in rows}) != len(rows):
        raise ValueError("co-learning controls contain duplicate identities")
    return rows


def _task_key(task: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(task["defender"]["label"]),
        str(task["condition"]["name"]),
        int(task["seed"]),
    )


def _run_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["defender_label"]),
        str(row["condition"]["name"]),
        int(row["seed"]),
    )


def _result_sort_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return _run_key(row)


def _control_task_key(task: Mapping[str, Any]) -> tuple[str, int]:
    return (
        str(task["defender"]["label"]),
        int(task["seed"]),
    )


def _control_run_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return (
        str(row["defender_label"]),
        int(row["seed"]),
    )


def _control_result_sort_key(
    row: Mapping[str, Any],
) -> tuple[str, int]:
    return _control_run_key(row)


def _completion_message(
    row: Mapping[str, Any],
    completed: int,
    expected: int,
) -> str:
    return (
        f"{completed}/{expected}: {row['defender_label']} / "
        f"{row['condition']['name']} / seed {row['seed']}"
    )


def _mean(values: Sequence[Any]) -> float:
    if not values:
        return math.nan
    return float(np.mean(np.asarray(values, dtype=float)))


def _optional_mean(values: Sequence[Any]) -> float | None:
    return _mean(values) if values else None


def _write_json_lines(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True) + "\n"
            for row in sorted(rows, key=_result_sort_key)
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_control_json_lines(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True) + "\n"
            for row in sorted(rows, key=_control_result_sort_key)
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
