"""Manifest-locked two-sided Tsallis-INF calibration refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .calibration_bank import HistogramPayoffBank, validate_bank_against_case
from .calibration_candidates import (
    LearnerCalibrationCandidate,
    apply_candidate_pair,
    tsallis_refinement_candidates,
)
from .calibration_cases import (
    DEFAULT_STEP6_CASE_MANIFEST_PATH,
    STEP6_SELECTION_DIGEST,
    FrozenCalibrationCase,
    load_frozen_calibration_cases,
    load_frozen_calibration_specs,
)
from .calibration_game import (
    CalibrationGameOracle,
    build_calibration_game_oracle,
    run_bank_two_player_learning,
)
from .calibration_pipeline import (
    DEFAULT_STEP7_OUTPUT_DIR,
    Step7CalibrationPlan,
    _append_jsonl,
    _compact_run_row,
    _compile_confirmation_action_surface,
    _load_jsonl,
    _run_confirmation_task_in_process,
    _run_confirmation_tasks_parallel,
    _run_confirmation_timing_row,
    _validate_resumable_rows,
    _write_json,
)
from .config import HistogramCoLearningConfig, REPO_ROOT
from .corpus import HistogramGameCase
from .physics import HistogramPhysicsPool


TSALLIS_REFINEMENT_PLAN_SCHEMA = "histogram_tsallis_refinement_plan.v1"
TSALLIS_REFINEMENT_SCREEN_SCHEMA = "histogram_tsallis_refinement_screen.v1"
TSALLIS_REFINEMENT_CONFIRMATION_SCHEMA = (
    "histogram_tsallis_refinement_confirmation.v1"
)
DEFAULT_TSALLIS_REFINEMENT_OUTPUT_DIR = (
    REPO_ROOT / "runs" / "histogram_tsallis_refinement_5topology"
)
DEFAULT_STEP7_BANK_DIR = DEFAULT_STEP7_OUTPUT_DIR / "banks"


_SCREEN_WORKER_CONTEXT: dict[str, Any] | None = None


@dataclass(frozen=True)
class TsallisRefinementPlan:
    """Compute and information contract for the finer Tsallis-INF grid."""

    screening_turns: int = 10_000
    screening_seeds: tuple[int, ...] = (401, 402, 403)
    screening_workers: int = 6
    confirmation_turns: int = 100_000
    confirmation_seeds: tuple[int, ...] = (201, 202, 203)
    confirmation_record_every: int = 10_000
    confirmation_workers: int = 3
    confirmation_timing_turns: int = 10_000
    confirmation_timing_seed: int = 601

    def __post_init__(self) -> None:
        if self.screening_turns < 1 or self.confirmation_turns < 1:
            raise ValueError("calibration horizons must be positive")
        if not self.screening_seeds or not self.confirmation_seeds:
            raise ValueError("calibration seed sets must be nonempty")
        if len(set(self.screening_seeds)) != len(self.screening_seeds):
            raise ValueError("screening seeds are not unique")
        if len(set(self.confirmation_seeds)) != len(self.confirmation_seeds):
            raise ValueError("confirmation seeds are not unique")
        if set(self.screening_seeds) & set(self.confirmation_seeds):
            raise ValueError("screening and confirmation seeds must be disjoint")
        if self.screening_workers < 1 or self.confirmation_workers < 1:
            raise ValueError("worker counts must be positive")
        if self.confirmation_record_every < 1:
            raise ValueError("confirmation_record_every must be positive")
        if self.confirmation_timing_turns < 1:
            raise ValueError("confirmation_timing_turns must be positive")
        if self.confirmation_timing_seed in {
            *self.screening_seeds,
            *self.confirmation_seeds,
        }:
            raise ValueError("confirmation timing seed must be disjoint")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["screening_seeds"] = list(self.screening_seeds)
        payload["confirmation_seeds"] = list(self.confirmation_seeds)
        return payload


def tsallis_refinement_preflight(
    plan: TsallisRefinementPlan,
    *,
    manifest_path: str | Path = DEFAULT_STEP6_CASE_MANIFEST_PATH,
    bank_dir: str | Path = DEFAULT_STEP7_BANK_DIR,
    case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return exact grid, turn, bank, cache, and RAM boundaries."""

    specs = _selected_specs(manifest_path, case_ids)
    alice = tsallis_refinement_candidates("alice")
    eve = tsallis_refinement_candidates("eve")
    bank_root = Path(bank_dir)
    case_rows = []
    serialized_bank_bytes = 0
    for spec in specs:
        path = _bank_path(bank_root, spec)
        if not path.exists():
            raise FileNotFoundError(f"missing frozen Step 7 bank {path}")
        summary_path = path.with_suffix(".summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cache = summary["metadata"]["physics"]["cache_plan"]
        serialized_bank_bytes += path.stat().st_size
        case_rows.append({
            **spec.to_dict(),
            "bank_path": str(path),
            "bank_content_digest_blake2b_128": summary[
                "content_digest_blake2b_128"
            ],
            "nominal_route_action_cells": (
                spec.route_count * spec.eve_action_count
            ),
            "canonical_physics_entries": cache[
                "canonical_unique_move_count_total"
            ],
            "configured_cache_entry_capacity": cache[
                "configured_entry_capacity_all_routes"
            ],
            "measured_compiled_cache_mib": summary["metadata"]["physics"][
                "cache_memory"
            ]["combined_cache_mib"],
        })
    screening_runs = (
        len(specs)
        * len(plan.screening_seeds)
        * len(alice)
        * len(eve)
    )
    maximum_confirmation_pairs = 3
    maximum_confirmation_runs = (
        len(specs)
        * len(plan.confirmation_seeds)
        * maximum_confirmation_pairs
    )
    largest_cache_mib = max(row.estimated_cache_mib for row in specs)
    return {
        "schema": TSALLIS_REFINEMENT_PLAN_SCHEMA,
        "selection_digest_blake2b_128": STEP6_SELECTION_DIGEST,
        "case_count": len(specs),
        "cases": case_rows,
        "eta_grid": [
            row.config_overrides["alice_tsallis_inf_eta_scale"]
            for row in alice if row.config_overrides[
                "alice_tsallis_inf_estimator"
            ] == "rv"
        ],
        "estimators": ["rv", "iw"],
        "candidate_count_per_player": len(alice),
        "screening": {
            "crossplay_shape_per_case_seed": [len(alice), len(eve)],
            "run_count": screening_runs,
            "turns_per_run": plan.screening_turns,
            "total_turns": screening_runs * plan.screening_turns,
            "seeds": list(plan.screening_seeds),
            "worker_count": plan.screening_workers,
            "outcome_source": "frozen_step7_feedback_bank",
            "scoring_source": "frozen_step7_disjoint_audit_bank",
        },
        "confirmation": {
            "declared_pair_set": [
                "one_standard_error_selected_pair",
                "raw_minimum_pair",
                "previous_rv_eta1_pair",
            ],
            "maximum_distinct_pair_count": maximum_confirmation_pairs,
            "maximum_run_count": maximum_confirmation_runs,
            "turns_per_run": plan.confirmation_turns,
            "maximum_total_turns": (
                maximum_confirmation_runs * plan.confirmation_turns
            ),
            "seeds": list(plan.confirmation_seeds),
            "worker_count": plan.confirmation_workers,
            "worker_start_method": (
                "single_process"
                if plan.confirmation_workers == 1 else "fork_copy_on_write"
            ),
            "largest_declared_cache_mib": largest_cache_mib,
            "worst_case_cache_copy_upper_bound_mib": (
                largest_cache_mib * plan.confirmation_workers
            ),
            "timing_turns_per_pair_case": plan.confirmation_timing_turns,
            "timing_seed": plan.confirmation_timing_seed,
        },
        "resource_boundary": {
            "nominal_route_action_cells": sum(
                row.route_count * row.eve_action_count for row in specs
            ),
            "canonical_physics_entries": sum(
                int(row["canonical_physics_entries"]) for row in case_rows
            ),
            "serialized_bank_mib": serialized_bank_bytes / 2**20,
            "available_ram_mib_at_preflight": _available_ram_mib(),
            "concurrent_physics_cases": 1,
        },
        "information_boundary": {
            "calibration_partition_loaded": True,
            "evaluation_partition_loaded": False,
            "full_five_case_claim": case_ids is None,
            "case_membership_can_change": False,
            "online_feedback": "selected_action_scalar_reward_only",
        },
        "plan": plan.to_dict(),
    }


def run_tsallis_refinement(
    *,
    output_dir: str | Path = DEFAULT_TSALLIS_REFINEMENT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_STEP6_CASE_MANIFEST_PATH,
    bank_dir: str | Path = DEFAULT_STEP7_BANK_DIR,
    base_config: HistogramCoLearningConfig | None = None,
    plan: TsallisRefinementPlan | None = None,
    case_ids: Sequence[str] | None = None,
    phase: str = "all",
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Screen the full Tsallis grid and confirm its selected pair live."""

    if phase not in {"screening", "confirmation", "all"}:
        raise ValueError(f"unsupported Tsallis refinement phase {phase!r}")
    active_plan = plan or TsallisRefinementPlan()
    config = base_config or HistogramCoLearningConfig(
        turns=active_plan.screening_turns,
        posthoc_samples_per_cell=0,
        seed=active_plan.screening_seeds[0],
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    preflight = tsallis_refinement_preflight(
        active_plan,
        manifest_path=manifest_path,
        bank_dir=bank_dir,
        case_ids=case_ids,
    )
    preflight["base_config"] = config.to_dict()
    _write_json(destination / "tsallis_refinement_plan.json", preflight)
    cases = load_frozen_calibration_cases(manifest_path, case_ids=case_ids)
    banks = _load_frozen_banks(cases, bank_dir=bank_dir, config=config)
    screen = run_tsallis_refinement_screen(
        cases,
        banks=banks,
        config=config,
        plan=active_plan,
        output_dir=destination,
        progress_callback=progress_callback,
    )
    if phase == "screening":
        return {"phase": phase, "preflight": preflight, "screening": screen}
    confirmation = run_tsallis_refinement_confirmation(
        cases,
        banks=banks,
        config=config,
        plan=active_plan,
        screening=screen,
        output_dir=destination,
        progress_callback=progress_callback,
    )
    result = {
        "schema": TSALLIS_REFINEMENT_CONFIRMATION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight,
        "screening": screen,
        "confirmation": confirmation,
        "evaluation_partition_loaded": False,
    }
    _write_json(destination / "tsallis_refinement_results.json", result)
    return result


def run_tsallis_refinement_screen(
    cases: Sequence[tuple[FrozenCalibrationCase, HistogramGameCase]],
    *,
    banks: dict[str, HistogramPayoffBank],
    config: HistogramCoLearningConfig,
    plan: TsallisRefinementPlan,
    output_dir: str | Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the complete Alice-setting by Eve-setting bank cross-play."""

    alice_candidates = tsallis_refinement_candidates("alice")
    eve_candidates = tsallis_refinement_candidates("eve")
    path = Path(output_dir) / "screening_rows.jsonl"
    existing = _load_jsonl(path)
    expected_contracts = {}
    for spec, _case in cases:
        for alice in alice_candidates:
            for eve in eve_candidates:
                for seed in plan.screening_seeds:
                    run_key = f"{spec.case_id}|{alice.label}|{eve.label}|{seed}"
                    expected_contracts[run_key] = {
                        "phase": "tsallis_refinement_screening",
                        "tuned_player": "both",
                        "case_id": spec.case_id,
                        "seed": int(seed),
                        "turns": plan.screening_turns,
                        "candidate": alice.to_dict(),
                        "opponent": eve.to_dict(),
                        "bank_content_digest_blake2b_128": banks[
                            spec.case_id
                        ].content_digest(),
                        "learner_outcome_source": (
                            "empirical_resampling_from_feedback_bank"
                        ),
                        "scoring_source": "reserved_equal_coverage_audit_bank",
                        "physics_cache_compiled_before_learning": None,
                    }
    _validate_resumable_rows(
        existing,
        expected_contracts=expected_contracts,
        path=path,
    )
    rows_by_key = {str(row["run_key"]): row for row in existing}
    expected = len(expected_contracts)
    completed = len(rows_by_key)
    for spec, case in cases:
        bank = banks[spec.case_id]
        oracle = build_calibration_game_oracle(bank)
        tasks = [
            (f"{spec.case_id}|{alice.label}|{eve.label}|{seed}", alice, eve, seed)
            for alice in alice_candidates
            for eve in eve_candidates
            for seed in plan.screening_seeds
            if f"{spec.case_id}|{alice.label}|{eve.label}|{seed}" not in rows_by_key
        ]
        if not tasks:
            continue
        rows = _run_screen_tasks(
            tasks,
            case=case,
            bank=bank,
            config=config,
            plan=plan,
            oracle=oracle,
        )
        for row in rows:
            row["selection_digest_blake2b_128"] = STEP6_SELECTION_DIGEST
            _append_jsonl(path, row)
            rows_by_key[str(row["run_key"])] = row
            completed += 1
            if progress_callback is not None and (
                completed == 1 or completed == expected or completed % 50 == 0
            ):
                progress_callback(
                    f"Tsallis screening {completed}/{expected}: {spec.family} "
                    f"{row['alice_label']} vs {row['eve_label']}"
                )
    rows = list(rows_by_key.values())
    selection = {
        "alice": _select_refined_candidate(
            alice_candidates, rows, player="alice"
        ),
        "eve": _select_refined_candidate(eve_candidates, rows, player="eve"),
    }
    result = {
        "schema": TSALLIS_REFINEMENT_SCREEN_SCHEMA,
        "selection_digest_blake2b_128": STEP6_SELECTION_DIGEST,
        "screening_turns": plan.screening_turns,
        "screening_seeds": list(plan.screening_seeds),
        "candidate_count_per_player": len(alice_candidates),
        "crossplay_shape_per_case_seed": [
            len(alice_candidates), len(eve_candidates)
        ],
        "run_count": len(rows),
        "expected_run_count": expected,
        "complete": len(rows) == expected,
        "selection": selection,
        "pair_summaries": _screen_pair_summaries(rows),
        "rows_path": str(path),
        "evaluation_partition_loaded": False,
    }
    _write_json(Path(output_dir) / "screening_summary.json", result)
    return result


def run_tsallis_refinement_confirmation(
    cases: Sequence[tuple[FrozenCalibrationCase, HistogramGameCase]],
    *,
    banks: dict[str, HistogramPayoffBank],
    config: HistogramCoLearningConfig,
    plan: TsallisRefinementPlan,
    screening: dict[str, Any],
    output_dir: str | Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run fresh 100k confirmation for the predeclared selected/control pairs."""

    if not screening.get("complete"):
        raise ValueError("Tsallis screening must be complete before confirmation")
    pairs = _confirmation_pairs(screening)
    path = Path(output_dir) / "confirmation_rows.jsonl"
    timing_path = Path(output_dir) / "confirmation_timing_rows.jsonl"
    existing = _load_jsonl(path)
    existing_timing = _load_jsonl(timing_path)
    expected_contracts = {}
    expected_timing_contracts = {}
    pair_by_labels = {}
    for pair in pairs:
        alice = pair["alice"]
        eve = pair["eve"]
        pair_by_labels[(alice.label, eve.label)] = pair["pair_id"]
        for spec, _case in cases:
            for seed in plan.confirmation_seeds:
                run_key = f"{spec.case_id}|{alice.label}|{eve.label}|{seed}"
                expected_contracts[run_key] = {
                    "phase": "confirmation",
                    "tuned_player": "both",
                    "case_id": spec.case_id,
                    "seed": int(seed),
                    "turns": plan.confirmation_turns,
                    "candidate": alice.to_dict(),
                    "opponent": eve.to_dict(),
                    "bank_content_digest_blake2b_128": banks[
                        spec.case_id
                    ].content_digest(),
                    "learner_outcome_source": (
                        "fresh_compiled_histogram_transcript_per_turn"
                    ),
                    "scoring_source": "reserved_equal_coverage_audit_bank",
                    "physics_cache_compiled_before_learning": True,
                }
            timing_key = (
                f"{spec.case_id}|{alice.label}|{eve.label}|"
                f"timing-{plan.confirmation_timing_seed}"
            )
            expected_timing_contracts[timing_key] = {
                "phase": "confirmation_timing",
                "case_id": spec.case_id,
                "seed": int(plan.confirmation_timing_seed),
                "turns": int(plan.confirmation_timing_turns),
                "alice_label": alice.label,
                "eve_label": eve.label,
                "physics_cache_compiled_before_learning": True,
                "execution_worker_count": 1,
                "timing_scope": "single_worker_uncontended",
            }
    _validate_resumable_rows(existing, expected_contracts=expected_contracts, path=path)
    _validate_resumable_rows(
        existing_timing,
        expected_contracts=expected_timing_contracts,
        path=timing_path,
    )
    rows_by_key = {str(row["run_key"]): row for row in existing}
    timing_by_key = {str(row["run_key"]): row for row in existing_timing}
    completed = len(rows_by_key)
    expected = len(expected_contracts)
    compilation_rows = []
    confirmation_plan = Step7CalibrationPlan(
        screening_turns=plan.screening_turns,
        screening_seeds=plan.screening_seeds,
        confirmation_turns=plan.confirmation_turns,
        confirmation_seeds=plan.confirmation_seeds,
        confirmation_record_every=plan.confirmation_record_every,
        confirmation_workers=plan.confirmation_workers,
        confirmation_timing_turns=plan.confirmation_timing_turns,
        confirmation_timing_seed=plan.confirmation_timing_seed,
    )
    for spec, case in cases:
        bank = banks[spec.case_id]
        oracle = build_calibration_game_oracle(bank)
        missing_tasks = []
        for pair in pairs:
            alice = pair["alice"]
            eve = pair["eve"]
            for seed in plan.confirmation_seeds:
                run_key = f"{spec.case_id}|{alice.label}|{eve.label}|{seed}"
                if run_key not in rows_by_key:
                    missing_tasks.append((run_key, alice, eve, seed))
        missing_timing = []
        for pair in pairs:
            alice = pair["alice"]
            eve = pair["eve"]
            timing_key = (
                f"{spec.case_id}|{alice.label}|{eve.label}|"
                f"timing-{plan.confirmation_timing_seed}"
            )
            if timing_key not in timing_by_key:
                missing_timing.append((timing_key, alice, eve))
        if not missing_tasks and not missing_timing:
            compilation_rows.append({
                "case_id": spec.case_id,
                "status": "not_repeated_all_rows_already_present",
            })
            continue
        physics = HistogramPhysicsPool(case, config)
        _compile_confirmation_action_surface(
            case,
            physics=physics,
            config=config,
            progress_callback=progress_callback,
        )
        compilation_rows.append({
            "case_id": spec.case_id,
            "physics": physics.diagnostics(),
        })
        if plan.confirmation_workers == 1:
            generated = (
                _run_confirmation_task_in_process(
                    task,
                    case=case,
                    physics=physics,
                    config=config,
                    plan=confirmation_plan,
                    bank=bank,
                    oracle=oracle,
                    execution_worker_count=1,
                )
                for task in missing_tasks
            )
        else:
            generated = _run_confirmation_tasks_parallel(
                missing_tasks,
                case=case,
                physics=physics,
                config=config,
                plan=confirmation_plan,
                bank=bank,
                oracle=oracle,
            )
        for row in generated:
            row["pair_id"] = pair_by_labels[(row["alice_label"], row["eve_label"])]
            row["selection_digest_blake2b_128"] = STEP6_SELECTION_DIGEST
            _append_jsonl(path, row)
            rows_by_key[str(row["run_key"])] = row
            completed += 1
            if progress_callback is not None:
                progress_callback(
                    f"Tsallis confirmation {completed}/{expected}: {spec.family} "
                    f"{row['pair_id']} seed {row['seed']}"
                )
        for timing_key, alice, eve in missing_timing:
            row = _run_confirmation_timing_row(
                timing_key,
                case_id=spec.case_id,
                case=case,
                physics=physics,
                config=config,
                plan=confirmation_plan,
                alice=alice,
                eve=eve,
            )
            row["pair_id"] = pair_by_labels[(alice.label, eve.label)]
            _append_jsonl(timing_path, row)
            timing_by_key[timing_key] = row
        del physics
        gc.collect()
    rows = list(rows_by_key.values())
    timing_rows = list(timing_by_key.values())
    result = {
        "schema": TSALLIS_REFINEMENT_CONFIRMATION_SCHEMA,
        "confirmation_turns": plan.confirmation_turns,
        "confirmation_seeds": list(plan.confirmation_seeds),
        "pairs": [
            {
                "pair_id": pair["pair_id"],
                "alice": pair["alice"].to_dict(),
                "eve": pair["eve"].to_dict(),
            }
            for pair in pairs
        ],
        "run_count": len(rows),
        "expected_run_count": expected,
        "timing_run_count": len(timing_rows),
        "timing_expected_run_count": len(expected_timing_contracts),
        "complete": (
            len(rows) == expected
            and len(timing_rows) == len(expected_timing_contracts)
        ),
        "aggregate": _confirmation_aggregates(rows, timing_rows),
        "rows_path": str(path),
        "timing_rows_path": str(timing_path),
        "physics_action_surface_compilation": compilation_rows,
        "evaluation_partition_loaded": False,
    }
    _write_json(Path(output_dir) / "confirmation_summary.json", result)
    return result


def _run_screen_tasks(
    tasks: Sequence[tuple[str, LearnerCalibrationCandidate, LearnerCalibrationCandidate, int]],
    *,
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    plan: TsallisRefinementPlan,
    oracle: CalibrationGameOracle,
) -> Any:
    if plan.screening_workers == 1:
        for task in tasks:
            yield _run_screen_task(
                task,
                case=case,
                bank=bank,
                config=config,
                plan=plan,
                oracle=oracle,
            )
        return
    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError("parallel screening requires fork copy-on-write sharing")
    global _SCREEN_WORKER_CONTEXT
    if _SCREEN_WORKER_CONTEXT is not None:
        raise RuntimeError("screen worker context is already active")
    _SCREEN_WORKER_CONTEXT = {
        "case": case,
        "bank": bank,
        "config": config,
        "plan": plan,
        "oracle": oracle,
    }
    context = mp.get_context("fork")
    try:
        with context.Pool(processes=plan.screening_workers) as pool:
            yield from pool.imap_unordered(_run_screen_task_in_worker, tasks, chunksize=2)
    finally:
        _SCREEN_WORKER_CONTEXT = None


def _run_screen_task_in_worker(
    task: tuple[str, LearnerCalibrationCandidate, LearnerCalibrationCandidate, int],
) -> dict[str, Any]:
    context = _SCREEN_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("screen worker context is not initialized")
    return _run_screen_task(
        task,
        case=context["case"],
        bank=context["bank"],
        config=context["config"],
        plan=context["plan"],
        oracle=context["oracle"],
    )


def _run_screen_task(
    task: tuple[str, LearnerCalibrationCandidate, LearnerCalibrationCandidate, int],
    *,
    case: HistogramGameCase,
    bank: HistogramPayoffBank,
    config: HistogramCoLearningConfig,
    plan: TsallisRefinementPlan,
    oracle: CalibrationGameOracle,
) -> dict[str, Any]:
    run_key, alice, eve, seed = task
    active = apply_candidate_pair(
        config,
        alice=alice,
        eve=eve,
        turns=plan.screening_turns,
        seed=seed,
    )
    payload = run_bank_two_player_learning(
        case,
        bank,
        active,
        oracle=oracle,
        include_equilibrium_distance=True,
    )
    return _compact_run_row(
        payload,
        run_key=run_key,
        phase="tsallis_refinement_screening",
        tuned_player="both",
        candidate=alice,
        opponent=eve,
        retain_strategies=False,
    )


def _select_refined_candidate(
    candidates: Sequence[LearnerCalibrationCandidate],
    rows: Sequence[dict[str, Any]],
    *,
    player: str,
) -> dict[str, Any]:
    label_field = f"{player}_label"
    metric = f"{player}_external_regret_per_turn"
    summaries = []
    for candidate in candidates:
        observations = [
            row for row in rows if row[label_field] == candidate.label
        ]
        if not observations:
            raise ValueError(f"candidate {candidate.label!r} has no observations")
        seeds = sorted({int(row["seed"]) for row in observations})
        seed_means = [
            float(np.mean([
                float(row[metric]) for row in observations
                if int(row["seed"]) == seed
            ]))
            for seed in seeds
        ]
        case_means = {
            case_id: float(np.mean([
                float(row[metric]) for row in observations
                if str(row["case_id"]) == case_id
            ]))
            for case_id in sorted({str(row["case_id"]) for row in observations})
        }
        estimator_field = f"{player}_tsallis_inf_estimator"
        eta_field = f"{player}_tsallis_inf_eta_scale"
        summaries.append({
            "candidate": candidate.to_dict(),
            "estimator": candidate.config_overrides[estimator_field],
            "eta_scale": candidate.config_overrides[eta_field],
            "observation_count": len(observations),
            "seed_block_means": seed_means,
            "mean_external_regret_per_turn": float(np.mean(seed_means)),
            "standard_error": _standard_error(seed_means),
            "case_mean_external_regret_per_turn": case_means,
            "worst_case_mean_external_regret_per_turn": max(case_means.values()),
        })
    return {
        "player": player,
        "selection_metric": metric,
        "seed_block_definition": (
            "for each seed, macro-average equally over all five calibration "
            "cases and all opposing Tsallis estimator/eta settings"
        ),
        "selection_rule": (
            "find the minimum mean own-player external regret; admit settings "
            "within one seed-block standard error; select the admitted setting "
            "nearest the literature RV/eta-1 configuration"
        ),
        "candidate_summaries": summaries,
        "selected": _one_standard_error_selection(summaries),
        "selected_by_estimator": {
            estimator: _one_standard_error_selection([
                row for row in summaries if row["estimator"] == estimator
            ])
            for estimator in ("rv", "iw")
        },
    }


def _one_standard_error_selection(
    summaries: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    raw = min(
        summaries,
        key=lambda row: (
            row["mean_external_regret_per_turn"],
            row["worst_case_mean_external_regret_per_turn"],
            row["candidate"]["label"],
        ),
    )
    threshold = raw["mean_external_regret_per_turn"] + raw["standard_error"]
    eligible = [
        row for row in summaries
        if row["mean_external_regret_per_turn"] <= threshold
    ]
    selected = min(
        eligible,
        key=lambda row: (
            row["candidate"]["default_distance"],
            row["mean_external_regret_per_turn"],
            row["worst_case_mean_external_regret_per_turn"],
            row["candidate"]["label"],
        ),
    )
    return {
        "selected": selected,
        "raw_minimum": raw,
        "one_standard_error_threshold": threshold,
        "eligible_labels": [row["candidate"]["label"] for row in eligible],
    }


def _screen_pair_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    keys = sorted({(row["alice_label"], row["eve_label"]) for row in rows})
    for alice_label, eve_label in keys:
        selected = [
            row for row in rows
            if row["alice_label"] == alice_label and row["eve_label"] == eve_label
        ]
        result.append({
            "alice_label": alice_label,
            "eve_label": eve_label,
            "observation_count": len(selected),
            "mean_alice_external_regret_per_turn": float(np.mean([
                row["alice_external_regret_per_turn"] for row in selected
            ])),
            "mean_eve_external_regret_per_turn": float(np.mean([
                row["eve_external_regret_per_turn"] for row in selected
            ])),
            "mean_nash_gap": float(np.mean([
                row["exploitability"] for row in selected
            ])),
        })
    return result


def _confirmation_pairs(screening: dict[str, Any]) -> list[dict[str, Any]]:
    alice_selection = screening["selection"]["alice"]["selected"]
    eve_selection = screening["selection"]["eve"]["selected"]
    declarations = {
        "alice": {
            row.label: row for row in tsallis_refinement_candidates("alice")
        },
        "eve": {
            row.label: row for row in tsallis_refinement_candidates("eve")
        },
    }
    declared = [
        (
            "one_standard_error_selected_pair",
            alice_selection["selected"]["candidate"]["label"],
            eve_selection["selected"]["candidate"]["label"],
        ),
        (
            "raw_minimum_pair",
            alice_selection["raw_minimum"]["candidate"]["label"],
            eve_selection["raw_minimum"]["candidate"]["label"],
        ),
        ("previous_rv_eta1_pair", "tsallis_rv_eta1", "tsallis_rv_eta1"),
    ]
    result = []
    seen = set()
    for pair_id, alice_label, eve_label in declared:
        labels = (alice_label, eve_label)
        if labels in seen:
            continue
        seen.add(labels)
        result.append({
            "pair_id": pair_id,
            "alice": declarations["alice"][alice_label],
            "eve": declarations["eve"][eve_label],
        })
    return result


def _confirmation_aggregates(
    rows: Sequence[dict[str, Any]],
    timing_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    metrics = (
        "alice_external_regret_per_turn",
        "eve_external_regret_per_turn",
        "exploitability",
        "mean_alice_reward",
        "mean_bell_qualified_rate_hz",
        "false_signal_rate",
        "false_signal_policy_total_variation",
    )
    for pair_id in sorted({str(row["pair_id"]) for row in rows}):
        selected = [row for row in rows if row["pair_id"] == pair_id]
        seeds = sorted({int(row["seed"]) for row in selected})
        summary: dict[str, Any] = {
            "pair_id": pair_id,
            "alice_label": selected[0]["alice_label"],
            "eve_label": selected[0]["eve_label"],
            "run_count": len(selected),
            "case_metrics": {},
        }
        for metric in metrics:
            seed_means = [
                float(np.mean([
                    float(row[metric]) for row in selected
                    if int(row["seed"]) == seed
                ]))
                for seed in seeds
            ]
            summary[f"mean_{metric}"] = float(np.mean(seed_means))
            summary[f"standard_error_{metric}"] = _standard_error(seed_means)
        for case_id in sorted({str(row["case_id"]) for row in selected}):
            case_rows = [row for row in selected if row["case_id"] == case_id]
            summary["case_metrics"][case_id] = {
                metric: float(np.mean([row[metric] for row in case_rows]))
                for metric in metrics
            }
        selected_timing = [
            row for row in timing_rows if row["pair_id"] == pair_id
        ]
        summary["mean_uncontended_turns_per_second"] = float(np.mean([
            row["turns_per_second"] for row in selected_timing
        ])) if selected_timing else None
        result.append(summary)
    return result


def _load_frozen_banks(
    cases: Sequence[tuple[FrozenCalibrationCase, HistogramGameCase]],
    *,
    bank_dir: str | Path,
    config: HistogramCoLearningConfig,
) -> dict[str, HistogramPayoffBank]:
    root = Path(bank_dir)
    result = {}
    for spec, case in cases:
        bank = HistogramPayoffBank.load(_bank_path(root, spec))
        validate_bank_against_case(bank, case, config)
        result[spec.case_id] = bank
    return result


def _selected_specs(
    manifest_path: str | Path,
    case_ids: Sequence[str] | None,
) -> tuple[FrozenCalibrationCase, ...]:
    specs = load_frozen_calibration_specs(manifest_path)
    if case_ids is None:
        return specs
    requested = set(case_ids)
    unknown = requested - {row.case_id for row in specs}
    if unknown:
        raise ValueError(f"unknown calibration case ids: {sorted(unknown)}")
    selected = tuple(row for row in specs if row.case_id in requested)
    if not selected:
        raise ValueError("calibration case filter selected no cases")
    return selected


def _bank_path(root: Path, spec: FrozenCalibrationCase) -> Path:
    return root / f"{spec.selection_order:02d}_{spec.family}.npz"


def _standard_error(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    return (
        float(np.std(array, ddof=1) / math.sqrt(array.size))
        if array.size > 1 else 0.0
    )


def _available_ram_mib() -> float | None:
    path = Path("/proc/meminfo")
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024.0
    return None
