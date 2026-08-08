"""Step 7 calibration pipeline over the frozen Topology Bench partition."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import gc
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import numpy as np

from .calibration_bank import (
    HistogramPayoffBank,
    build_histogram_payoff_bank,
    validate_bank_against_case,
)
from .calibration_candidates import (
    LEARNER_FAMILIES,
    LearnerCalibrationCandidate,
    apply_candidate_pair,
    declared_calibration_candidates,
    default_anchor_candidates,
    select_candidate_by_external_regret,
    selected_candidates,
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
    score_calibration_behavior,
)
from .config import HistogramCoLearningConfig, REPO_ROOT
from .corpus import HistogramGameCase
from .payoff_structure import analyze_payoff_structure
from .physics import HistogramPhysicsPool
from .learning import run_histogram_colearning


STEP7_PLAN_SCHEMA = "histogram_step7_calibration_plan.v1"
STEP7_SCREENING_SCHEMA = "histogram_step7_screening.v1"
STEP7_CONFIRMATION_SCHEMA = "histogram_step7_confirmation.v1"
DEFAULT_STEP7_OUTPUT_DIR = REPO_ROOT / "runs" / "histogram_step7_calibration"


_CONFIRMATION_WORKER_CONTEXT: dict[str, Any] | None = None


@dataclass(frozen=True)
class Step7CalibrationPlan:
    """Frozen compute and information contract for Step 7."""

    samples_per_cell_per_split: int = 64
    payoff_bank_root_seed: int = 20_260_715
    screening_turns: int = 10_000
    screening_seeds: tuple[int, ...] = (101, 102, 103)
    confirmation_turns: int = 100_000
    confirmation_seeds: tuple[int, ...] = (201, 202, 203)
    confirmation_record_every: int = 10_000
    confirmation_workers: int = 1
    confirmation_timing_turns: int = 10_000
    confirmation_timing_seed: int = 301

    def __post_init__(self) -> None:
        if self.samples_per_cell_per_split < 2:
            raise ValueError("samples_per_cell_per_split must be at least two")
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
        if self.confirmation_record_every < 1:
            raise ValueError("confirmation_record_every must be positive")
        if self.confirmation_workers < 1:
            raise ValueError("confirmation_workers must be positive")
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


def step7_preflight(
    plan: Step7CalibrationPlan,
    *,
    manifest_path: str | Path = DEFAULT_STEP6_CASE_MANIFEST_PATH,
    case_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report exact run counts and declared resource boundaries before execution."""

    specs = load_frozen_calibration_specs(manifest_path)
    if case_ids is not None:
        requested = set(case_ids)
        unknown = requested - {row.case_id for row in specs}
        if unknown:
            raise ValueError(f"unknown calibration case ids: {sorted(unknown)}")
        specs = tuple(row for row in specs if row.case_id in requested)
        if not specs:
            raise ValueError("calibration case filter selected no cases")
    alice_candidates = declared_calibration_candidates("alice")
    eve_candidates = declared_calibration_candidates("eve")
    alice_anchors = default_anchor_candidates("alice")
    eve_anchors = default_anchor_candidates("eve")
    cells = sum(row.route_count * row.eve_action_count for row in specs)
    screening_runs = (
        len(specs)
        * len(plan.screening_seeds)
        * (
            len(alice_candidates) * len(eve_anchors)
            + len(eve_candidates) * len(alice_anchors)
        )
    )
    confirmation_runs = (
        len(specs)
        * len(plan.confirmation_seeds)
        * len(LEARNER_FAMILIES) ** 2
    )
    largest_cache_mib = max(row.estimated_cache_mib for row in specs)
    return {
        "schema": STEP7_PLAN_SCHEMA,
        "selection_digest_blake2b_128": STEP6_SELECTION_DIGEST,
        "case_count": len(specs),
        "cases": [row.to_dict() for row in specs],
        "candidate_count_per_player": len(alice_candidates),
        "opponent_anchor_count_per_player": len(alice_anchors),
        "payoff_bank": {
            "route_action_cells": cells,
            "draws_per_cell_per_split": plan.samples_per_cell_per_split,
            "split_count": 2,
            "total_histogram_draws": (
                2 * cells * plan.samples_per_cell_per_split
            ),
            "largest_declared_cache_mib": largest_cache_mib,
            "default_concurrent_physics_cases": 1,
        },
        "screening": {
            "run_count": screening_runs,
            "turns_per_run": plan.screening_turns,
            "total_turns": screening_runs * plan.screening_turns,
            "seeds": list(plan.screening_seeds),
        },
        "confirmation": {
            "run_count": confirmation_runs,
            "turns_per_run": plan.confirmation_turns,
            "total_turns": confirmation_runs * plan.confirmation_turns,
            "seeds": list(plan.confirmation_seeds),
            "crossplay_shape_per_case_seed": [
                len(LEARNER_FAMILIES),
                len(LEARNER_FAMILIES),
            ],
            "learner_outcome_source": (
                "fresh compiled-histogram transcript draw on every turn"
            ),
            "scoring_source": "reserved equal-coverage audit bank",
            "worker_count": plan.confirmation_workers,
            "worker_start_method": (
                "single_process"
                if plan.confirmation_workers == 1
                else "fork_copy_on_write"
            ),
            "worst_case_cache_copy_upper_bound_mib": (
                largest_cache_mib * plan.confirmation_workers
            ),
            "timing": {
                "run_count": len(specs) * len(LEARNER_FAMILIES) ** 2,
                "turns_per_run": plan.confirmation_timing_turns,
                "seed": plan.confirmation_timing_seed,
                "execution": "single_worker_uncontended",
            },
        },
        "information_boundary": {
            "calibration_partition_loaded": True,
            "development_partition_loaded": False,
            "evaluation_partition_loaded": False,
            "case_membership_can_change_after_payoff_audit": False,
            "learner_feedback_split": "feedback",
            "scoring_split": "audit",
            "confirmation_feedback_split": None,
            "confirmation_uses_live_compiled_histograms": True,
            "full_five_case_claim": case_ids is None,
        },
    }


def run_step7_calibration(
    *,
    output_dir: str | Path = DEFAULT_STEP7_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_STEP6_CASE_MANIFEST_PATH,
    base_config: HistogramCoLearningConfig | None = None,
    plan: Step7CalibrationPlan | None = None,
    case_ids: Sequence[str] | None = None,
    phase: str = "all",
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Build banks, audit payoff structure, calibrate, and confirm cross-play."""

    if phase not in {"banks", "screening", "confirmation", "all"}:
        raise ValueError(f"unsupported Step 7 phase {phase!r}")
    active_plan = plan or Step7CalibrationPlan()
    config = base_config or HistogramCoLearningConfig(
        turns=active_plan.screening_turns,
        posthoc_samples_per_cell=0,
        seed=active_plan.screening_seeds[0],
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    preflight = step7_preflight(
        active_plan,
        manifest_path=manifest_path,
        case_ids=case_ids,
    )
    if case_ids is not None:
        preflight["execution_case_filter"] = list(case_ids)
        preflight["full_five_case_claim"] = False
    else:
        preflight["execution_case_filter"] = None
        preflight["full_five_case_claim"] = True
    preflight["plan"] = active_plan.to_dict()
    preflight["base_config"] = config.to_dict()
    preflight["alice_candidates"] = [
        row.to_dict() for row in declared_calibration_candidates("alice")
    ]
    preflight["eve_candidates"] = [
        row.to_dict() for row in declared_calibration_candidates("eve")
    ]
    _write_json(destination / "step7_plan.json", preflight)
    loaded = load_frozen_calibration_cases(
        manifest_path,
        case_ids=case_ids,
    )
    banks = prepare_calibration_banks(
        loaded,
        config=config,
        plan=active_plan,
        output_dir=destination,
        progress_callback=progress_callback,
    )
    structure = audit_calibration_payoff_structure(
        loaded,
        banks=banks,
        config=config,
        output_dir=destination,
    )
    if phase == "banks":
        return {
            "phase": phase,
            "preflight": preflight,
            "payoff_structure": structure,
        }
    screening = run_symmetric_candidate_screening(
        loaded,
        banks=banks,
        config=config,
        plan=active_plan,
        output_dir=destination,
        progress_callback=progress_callback,
    )
    if phase == "screening":
        return {
            "phase": phase,
            "preflight": preflight,
            "payoff_structure": structure,
            "screening": screening,
        }
    confirmation = run_selected_crossplay_confirmation(
        loaded,
        banks=banks,
        config=config,
        plan=active_plan,
        screening=screening,
        output_dir=destination,
        progress_callback=progress_callback,
    )
    result = {
        "schema": STEP7_CONFIRMATION_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preflight": preflight,
        "payoff_structure": structure,
        "screening": screening,
        "confirmation": confirmation,
        "evaluation_partition_loaded": False,
    }
    _write_json(destination / "step7_results.json", result)
    return result


def prepare_calibration_banks(
    cases: Sequence[tuple[FrozenCalibrationCase, HistogramGameCase]],
    *,
    config: HistogramCoLearningConfig,
    plan: Step7CalibrationPlan,
    output_dir: str | Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, HistogramPayoffBank]:
    """Build or validate one serialized bank at a time to bound resident RAM."""

    bank_dir = Path(output_dir) / "banks"
    bank_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, HistogramPayoffBank] = {}
    summaries = []
    for index, (spec, case) in enumerate(cases, start=1):
        path = bank_dir / f"{spec.selection_order:02d}_{spec.family}.npz"
        if path.exists():
            bank = HistogramPayoffBank.load(path)
            validate_bank_against_case(bank, case, config)
            if bank.draws_per_cell != plan.samples_per_cell_per_split:
                raise ValueError(f"existing bank draw count changed in {path}")
            if int(bank.metadata.get("root_seed", -1)) != int(
                plan.payoff_bank_root_seed
            ):
                raise ValueError(f"existing bank root seed changed in {path}")
            if progress_callback is not None:
                progress_callback(
                    f"bank {index}/{len(cases)} {spec.family}: validated existing file"
                )
        else:
            if progress_callback is not None:
                progress_callback(
                    f"bank {index}/{len(cases)} {spec.family}: constructing physics pool"
                )
            physics = HistogramPhysicsPool(case, config)
            bank = build_histogram_payoff_bank(
                case,
                config,
                samples_per_cell=plan.samples_per_cell_per_split,
                root_seed=plan.payoff_bank_root_seed,
                physics=physics,
                progress_callback=progress_callback,
            )
            bank.save(path)
            del physics
            gc.collect()
        result[spec.case_id] = bank
        summary = bank.to_summary()
        summary["path"] = str(path)
        summaries.append(summary)
        _write_json(path.with_suffix(".summary.json"), summary)
    _write_json(bank_dir / "bank_manifest.json", {
        "schema": "histogram_calibration_payoff_bank_manifest.v1",
        "selection_digest_blake2b_128": STEP6_SELECTION_DIGEST,
        "bank_count": len(summaries),
        "banks": summaries,
        "evaluation_partition_loaded": False,
    })
    return result


def audit_calibration_payoff_structure(
    cases: Sequence[tuple[FrozenCalibrationCase, HistogramGameCase]],
    *,
    banks: dict[str, HistogramPayoffBank],
    config: HistogramCoLearningConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Measure two-sided bilinear span only after case membership is frozen."""

    audit_dir = Path(output_dir) / "payoff_structure"
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec, case in cases:
        bank = banks[spec.case_id]
        analysis = analyze_payoff_structure(
            payoff=bank.audit_payoff_matrix(),
            cell_standard_errors=bank.audit.cell_standard_errors(),
            case=case,
            actions=case.actions(config.game_mode),
        )
        analysis["case_id"] = spec.case_id
        analysis["selection_digest_blake2b_128"] = STEP6_SELECTION_DIGEST
        analysis["case_membership_changed"] = False
        analysis["bank_content_digest_blake2b_128"] = bank.content_digest()
        path = audit_dir / f"{spec.selection_order:02d}_{spec.family}.json"
        _write_json(path, analysis)
        rows.append({
            "case_id": spec.case_id,
            "family": spec.family,
            "route_count": spec.route_count,
            "action_count": spec.eve_action_count,
            "alice_feature_rank": analysis["route_resource_embedding"]["rank"],
            "selected_eve_feature_rank": analysis["eve_action_embeddings"][
                "family_target_topology"
            ]["rank"],
            "bilinear_in_sample_rmse": analysis["models"][
                "bilinear_family_target_topology"
            ]["rmse"],
            "bilinear_leave_route_out_rmse": analysis["models"][
                "bilinear_family_target_topology"
            ]["bilinear"]["leave_one_route_out"]["rmse"],
            "bilinear_leave_target_out_rmse": analysis["models"][
                "bilinear_family_target_topology"
            ]["bilinear"]["leave_one_target_out"]["rmse"],
            "bilinear_game_value_absolute_error": analysis["models"][
                "bilinear_family_target_topology"
            ]["absolute_equilibrium_value_error"],
            "path": str(path),
        })
    result = {
        "schema": "histogram_step7_payoff_structure.v1",
        "selection_digest_blake2b_128": STEP6_SELECTION_DIGEST,
        "case_count": len(rows),
        "cases": rows,
        "selection_revisited_after_fit": False,
        "evaluation_partition_loaded": False,
    }
    _write_json(audit_dir / "payoff_structure_summary.json", result)
    return result


def run_symmetric_candidate_screening(
    cases: Sequence[tuple[FrozenCalibrationCase, HistogramGameCase]],
    *,
    banks: dict[str, HistogramPayoffBank],
    config: HistogramCoLearningConfig,
    plan: Step7CalibrationPlan,
    output_dir: str | Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Tune Alice and Eve separately against all four default opponent families."""

    path = Path(output_dir) / "screening_rows.jsonl"
    existing = _load_jsonl(path)
    declarations = {
        "alice": declared_calibration_candidates("alice"),
        "eve": declared_calibration_candidates("eve"),
    }
    anchors = {
        "alice": default_anchor_candidates("alice"),
        "eve": default_anchor_candidates("eve"),
    }
    expected_contracts: dict[str, dict[str, Any]] = {}
    for player in ("alice", "eve"):
        opponent = "eve" if player == "alice" else "alice"
        for spec, _case in cases:
            for candidate in declarations[player]:
                for anchor in anchors[opponent]:
                    for seed in plan.screening_seeds:
                        run_key = (
                            f"{player}|{spec.case_id}|{candidate.label}|"
                            f"{anchor.label}|{seed}"
                        )
                        expected_contracts[run_key] = {
                            "phase": "screening",
                            "tuned_player": player,
                            "case_id": spec.case_id,
                            "seed": int(seed),
                            "turns": plan.screening_turns,
                            "candidate": candidate.to_dict(),
                            "opponent": anchor.to_dict(),
                            "bank_content_digest_blake2b_128": banks[
                                spec.case_id
                            ].content_digest(),
                            "learner_outcome_source": (
                                "empirical_resampling_from_feedback_bank"
                            ),
                            "scoring_source": (
                                "reserved_equal_coverage_audit_bank"
                            ),
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
    for player in ("alice", "eve"):
        opponent = "eve" if player == "alice" else "alice"
        for spec, case in cases:
            bank = banks[spec.case_id]
            oracle = build_calibration_game_oracle(bank)
            for candidate in declarations[player]:
                for anchor in anchors[opponent]:
                    for seed in plan.screening_seeds:
                        run_key = (
                            f"{player}|{spec.case_id}|{candidate.label}|"
                            f"{anchor.label}|{seed}"
                        )
                        if run_key in rows_by_key:
                            continue
                        alice_candidate = candidate if player == "alice" else anchor
                        eve_candidate = candidate if player == "eve" else anchor
                        active = apply_candidate_pair(
                            config,
                            alice=alice_candidate,
                            eve=eve_candidate,
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
                        row = _compact_run_row(
                            payload,
                            run_key=run_key,
                            phase="screening",
                            tuned_player=player,
                            candidate=candidate,
                            opponent=anchor,
                            retain_strategies=False,
                        )
                        _append_jsonl(path, row)
                        rows_by_key[run_key] = row
                        completed += 1
                        if progress_callback is not None and (
                            completed == 1
                            or completed == expected
                            or completed % 25 == 0
                        ):
                            progress_callback(
                                f"screening {completed}/{expected}: {player} "
                                f"{spec.family} {candidate.label} vs {anchor.label}"
                            )
    rows = list(rows_by_key.values())
    alice_rows = [row for row in rows if row["tuned_player"] == "alice"]
    eve_rows = [row for row in rows if row["tuned_player"] == "eve"]
    alice_selection = select_candidate_by_external_regret(
        declarations["alice"], alice_rows, player="alice"
    )
    eve_selection = select_candidate_by_external_regret(
        declarations["eve"], eve_rows, player="eve"
    )
    result = {
        "schema": STEP7_SCREENING_SCHEMA,
        "selection_digest_blake2b_128": STEP6_SELECTION_DIGEST,
        "screening_turns": plan.screening_turns,
        "screening_seeds": list(plan.screening_seeds),
        "run_count": len(rows),
        "expected_run_count": expected,
        "complete": len(rows) == expected,
        "alice": alice_selection,
        "eve": eve_selection,
        "rows_path": str(path),
        "evaluation_partition_loaded": False,
    }
    _write_json(Path(output_dir) / "screening_selection.json", result)
    return result


def run_selected_crossplay_confirmation(
    cases: Sequence[tuple[FrozenCalibrationCase, HistogramGameCase]],
    *,
    banks: dict[str, HistogramPayoffBank],
    config: HistogramCoLearningConfig,
    plan: Step7CalibrationPlan,
    screening: dict[str, Any],
    output_dir: str | Path,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run selected cross-play with fresh compiled-histogram transcript draws."""

    if not screening.get("complete"):
        raise ValueError("screening must be complete before confirmation")
    alice_candidates = selected_candidates(screening["alice"])
    eve_candidates = selected_candidates(screening["eve"])
    path = Path(output_dir) / "confirmation_rows.jsonl"
    timing_path = Path(output_dir) / "confirmation_timing_rows.jsonl"
    existing = _load_jsonl(path)
    existing_timing = _load_jsonl(timing_path)
    expected_contracts = {}
    expected_timing_contracts = {}
    for spec, _case in cases:
        for alice in alice_candidates:
            for eve in eve_candidates:
                for seed in plan.confirmation_seeds:
                    run_key = (
                        f"{spec.case_id}|{alice.label}|{eve.label}|{seed}"
                    )
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
                        "scoring_source": (
                            "reserved_equal_coverage_audit_bank"
                        ),
                        "physics_cache_compiled_before_learning": True,
                    }
                timing_run_key = (
                    f"{spec.case_id}|{alice.label}|{eve.label}|"
                    f"timing-{plan.confirmation_timing_seed}"
                )
                expected_timing_contracts[timing_run_key] = {
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
    _validate_resumable_rows(
        existing,
        expected_contracts=expected_contracts,
        path=path,
    )
    _validate_resumable_rows(
        existing_timing,
        expected_contracts=expected_timing_contracts,
        path=timing_path,
    )
    rows_by_key = {str(row["run_key"]): row for row in existing}
    timing_by_key = {
        str(row["run_key"]): row for row in existing_timing
    }
    expected = len(expected_contracts)
    timing_expected = len(expected_timing_contracts)
    previous_summary_path = Path(output_dir) / "confirmation_summary.json"
    if (
        len(rows_by_key) == expected
        and len(timing_by_key) == timing_expected
        and previous_summary_path.exists()
    ):
        previous = json.loads(previous_summary_path.read_text(encoding="utf-8"))
        if previous.get("complete") is True and int(
            previous.get("run_count", -1)
        ) == expected and int(
            previous.get("timing_run_count", -1)
        ) == timing_expected:
            return previous
    completed = len(rows_by_key)
    timing_completed = len(timing_by_key)
    compilation_rows = []
    for spec, case in cases:
        bank = banks[spec.case_id]
        oracle = build_calibration_game_oracle(bank)
        case_run_keys = {
            f"{spec.case_id}|{alice.label}|{eve.label}|{seed}"
            for alice in alice_candidates
            for eve in eve_candidates
            for seed in plan.confirmation_seeds
        }
        case_timing_keys = {
            (
                f"{spec.case_id}|{alice.label}|{eve.label}|"
                f"timing-{plan.confirmation_timing_seed}"
            )
            for alice in alice_candidates
            for eve in eve_candidates
        }
        if (
            case_run_keys <= set(rows_by_key)
            and case_timing_keys <= set(timing_by_key)
        ):
            compilation_rows.append({
                "case_id": spec.case_id,
                "status": (
                    "not_repeated_all_confirmation_and_timing_rows_already_present"
                ),
            })
            continue
        physics = HistogramPhysicsPool(case, config)
        compilation_started = time.perf_counter()
        _compile_confirmation_action_surface(
            case,
            physics=physics,
            config=config,
            progress_callback=progress_callback,
        )
        compilation_rows.append({
            "case_id": spec.case_id,
            "elapsed_seconds": time.perf_counter() - compilation_started,
            "physics": physics.diagnostics(),
        })
        missing_tasks = [
            (f"{spec.case_id}|{alice.label}|{eve.label}|{seed}", alice, eve, seed)
            for alice in alice_candidates
            for eve in eve_candidates
            for seed in plan.confirmation_seeds
            if f"{spec.case_id}|{alice.label}|{eve.label}|{seed}" not in rows_by_key
        ]
        if plan.confirmation_workers == 1:
            strategic_rows = (
                _run_confirmation_task_in_process(
                    task,
                    case=case,
                    physics=physics,
                    config=config,
                    plan=plan,
                    bank=bank,
                    oracle=oracle,
                    execution_worker_count=1,
                )
                for task in missing_tasks
            )
            completed = _consume_confirmation_rows(
                strategic_rows,
                path=path,
                rows_by_key=rows_by_key,
                completed=completed,
                expected=expected,
                case_family=spec.family,
                progress_callback=progress_callback,
            )
        elif missing_tasks:
            strategic_rows = _run_confirmation_tasks_parallel(
                missing_tasks,
                case=case,
                physics=physics,
                config=config,
                plan=plan,
                bank=bank,
                oracle=oracle,
            )
            completed = _consume_confirmation_rows(
                strategic_rows,
                path=path,
                rows_by_key=rows_by_key,
                completed=completed,
                expected=expected,
                case_family=spec.family,
                progress_callback=progress_callback,
            )
        for alice in alice_candidates:
            for eve in eve_candidates:
                timing_run_key = (
                    f"{spec.case_id}|{alice.label}|{eve.label}|"
                    f"timing-{plan.confirmation_timing_seed}"
                )
                if timing_run_key in timing_by_key:
                    continue
                timing_row = _run_confirmation_timing_row(
                    timing_run_key,
                    case_id=spec.case_id,
                    case=case,
                    physics=physics,
                    config=config,
                    plan=plan,
                    alice=alice,
                    eve=eve,
                )
                _append_jsonl(timing_path, timing_row)
                timing_by_key[timing_run_key] = timing_row
                timing_completed += 1
                if progress_callback is not None and (
                    timing_completed == 1
                    or timing_completed == timing_expected
                    or timing_completed % 10 == 0
                ):
                    progress_callback(
                        f"confirmation timing {timing_completed}/"
                        f"{timing_expected}: {spec.family} {alice.family} "
                        f"vs {eve.family}"
                    )
        del physics
        gc.collect()
    rows = list(rows_by_key.values())
    timing_rows = list(timing_by_key.values())
    result = {
        "schema": STEP7_CONFIRMATION_SCHEMA,
        "confirmation_turns": plan.confirmation_turns,
        "confirmation_seeds": list(plan.confirmation_seeds),
        "run_count": len(rows),
        "expected_run_count": expected,
        "timing_run_count": len(timing_rows),
        "timing_expected_run_count": timing_expected,
        "complete": (
            len(rows) == expected and len(timing_rows) == timing_expected
        ),
        "alice_selected": [row.to_dict() for row in alice_candidates],
        "eve_selected": [row.to_dict() for row in eve_candidates],
        "aggregate": _aggregate_confirmation_rows(
            rows,
            timing_rows=timing_rows,
        ),
        "rows_path": str(path),
        "timing_rows_path": str(timing_path),
        "physics_action_surface_compilation": compilation_rows,
        "parallel_execution": {
            "worker_count": plan.confirmation_workers,
            "start_method": (
                "single_process"
                if plan.confirmation_workers == 1
                else "fork_copy_on_write"
            ),
            "strategic_row_elapsed_time_boundary": (
                "worker-contended wall time when worker_count > 1; excluded "
                "from learner-family timing comparison"
            ),
            "timing_execution": "single_worker_uncontended",
            "timing_turns_per_run": plan.confirmation_timing_turns,
            "timing_seed": plan.confirmation_timing_seed,
        },
        "learner_outcome_source": (
            "fresh compiled-histogram transcript draw on every turn"
        ),
        "scoring_source": "reserved equal-coverage audit bank",
        "evaluation_partition_loaded": False,
    }
    _write_json(Path(output_dir) / "confirmation_summary.json", result)
    return result


def _run_confirmation_tasks_parallel(
    tasks: Sequence[
        tuple[
            str,
            LearnerCalibrationCandidate,
            LearnerCalibrationCandidate,
            int,
        ]
    ],
    *,
    case: HistogramGameCase,
    physics: HistogramPhysicsPool,
    config: HistogramCoLearningConfig,
    plan: Step7CalibrationPlan,
    bank: HistogramPayoffBank,
    oracle: CalibrationGameOracle,
) -> Any:
    """Yield independent strategic rows from forked cache-sharing workers."""

    if "fork" not in mp.get_all_start_methods():
        raise RuntimeError(
            "parallel confirmation requires fork copy-on-write cache sharing"
        )
    global _CONFIRMATION_WORKER_CONTEXT
    if _CONFIRMATION_WORKER_CONTEXT is not None:
        raise RuntimeError("confirmation worker context is already active")
    _CONFIRMATION_WORKER_CONTEXT = {
        "case": case,
        "physics": physics,
        "config": config,
        "plan": plan,
        "bank": bank,
        "oracle": oracle,
    }
    context = mp.get_context("fork")
    try:
        with context.Pool(processes=plan.confirmation_workers) as pool:
            yield from pool.imap_unordered(
                _run_confirmation_task_in_worker,
                tasks,
                chunksize=1,
            )
    finally:
        _CONFIRMATION_WORKER_CONTEXT = None


def _run_confirmation_task_in_worker(
    task: tuple[
        str,
        LearnerCalibrationCandidate,
        LearnerCalibrationCandidate,
        int,
    ],
) -> dict[str, Any]:
    context = _CONFIRMATION_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("confirmation worker context is not initialized")
    return _run_confirmation_task_in_process(
        task,
        case=context["case"],
        physics=context["physics"],
        config=context["config"],
        plan=context["plan"],
        bank=context["bank"],
        oracle=context["oracle"],
        execution_worker_count=context["plan"].confirmation_workers,
    )


def _run_confirmation_task_in_process(
    task: tuple[
        str,
        LearnerCalibrationCandidate,
        LearnerCalibrationCandidate,
        int,
    ],
    *,
    case: HistogramGameCase,
    physics: HistogramPhysicsPool,
    config: HistogramCoLearningConfig,
    plan: Step7CalibrationPlan,
    bank: HistogramPayoffBank,
    oracle: CalibrationGameOracle,
    execution_worker_count: int,
) -> dict[str, Any]:
    run_key, alice, eve, seed = task
    active = apply_candidate_pair(
        config,
        alice=alice,
        eve=eve,
        turns=plan.confirmation_turns,
        seed=seed,
    )
    active = replace(
        active,
        record_every=plan.confirmation_record_every,
        posthoc_samples_per_cell=0,
    )
    live = run_histogram_colearning(
        case,
        active,
        physics=physics,
    ).to_dict()
    payload = _adapt_live_confirmation_payload(
        live,
        bank=bank,
        oracle=oracle,
    )
    row = _compact_run_row(
        payload,
        run_key=run_key,
        phase="confirmation",
        tuned_player="both",
        candidate=alice,
        opponent=eve,
        retain_strategies=True,
    )
    row["execution_worker_count"] = int(execution_worker_count)
    row["elapsed_time_scope"] = (
        "single_worker_uncontended"
        if execution_worker_count == 1
        else "parallel_worker_contended"
    )
    return row


def _consume_confirmation_rows(
    rows: Any,
    *,
    path: Path,
    rows_by_key: dict[str, dict[str, Any]],
    completed: int,
    expected: int,
    case_family: str,
    progress_callback: Callable[[str], None] | None,
) -> int:
    for row in rows:
        run_key = str(row["run_key"])
        _append_jsonl(path, row)
        rows_by_key[run_key] = row
        completed += 1
        if progress_callback is not None and (
            completed == 1 or completed == expected or completed % 10 == 0
        ):
            progress_callback(
                f"confirmation {completed}/{expected}: {case_family} "
                f"{row['alice_family']} vs {row['eve_family']}"
            )
    return completed


def _run_confirmation_timing_row(
    run_key: str,
    *,
    case_id: str,
    case: HistogramGameCase,
    physics: HistogramPhysicsPool,
    config: HistogramCoLearningConfig,
    plan: Step7CalibrationPlan,
    alice: LearnerCalibrationCandidate,
    eve: LearnerCalibrationCandidate,
) -> dict[str, Any]:
    active = apply_candidate_pair(
        config,
        alice=alice,
        eve=eve,
        turns=plan.confirmation_timing_turns,
        seed=plan.confirmation_timing_seed,
    )
    active = replace(
        active,
        record_every=plan.confirmation_timing_turns,
        posthoc_samples_per_cell=0,
    )
    live = run_histogram_colearning(
        case,
        active,
        physics=physics,
    ).to_dict()
    telemetry = live["online_learning_telemetry"]
    return {
        "run_key": run_key,
        "phase": "confirmation_timing",
        "case_id": case_id,
        "seed": int(plan.confirmation_timing_seed),
        "turns": int(plan.confirmation_timing_turns),
        "alice_family": alice.family,
        "eve_family": eve.family,
        "alice_label": alice.label,
        "eve_label": eve.label,
        "elapsed_seconds": float(live["elapsed_seconds"]),
        "turns_per_second": float(live["turns_per_second"]),
        "alice_runtime": telemetry["alice"],
        "eve_runtime": telemetry["eve"],
        "physics_cache_compiled_before_learning": True,
        "execution_worker_count": 1,
        "timing_scope": "single_worker_uncontended",
        "measurement_boundary": (
            "runtime measurement only; excluded from strategic estimates"
        ),
    }


def _compile_confirmation_action_surface(
    case: HistogramGameCase,
    *,
    physics: HistogramPhysicsPool,
    config: HistogramCoLearningConfig,
    progress_callback: Callable[[str], None] | None,
) -> None:
    """Compile every canonical route/action histogram before timed comparisons."""

    actions = case.actions(config.game_mode)
    total = len(case.routes) * len(actions)
    completed = 0
    for route_index in range(len(case.routes)):
        for action_index, action in enumerate(actions):
            physics.step(
                route_index,
                action,
                seed=1_907_201,
                step_index=route_index * len(actions) + action_index,
            )
            completed += 1
            if progress_callback is not None and (
                completed == 1
                or completed == total
                or completed % max(1, total // 10) == 0
            ):
                progress_callback(
                    f"confirmation physics compilation {case.family}: "
                    f"{completed}/{total} route/action cells"
                )
    diagnostics = physics.diagnostics()
    cache_plan = diagnostics["cache_plan"]
    if not cache_plan["sufficient_for_declared_action_surface_without_eviction"]:
        raise RuntimeError(
            "configured histogram cache cannot retain the declared action surface"
        )
    evictions = float(
        (diagnostics.get("stats") or {}).get("evaluated_cache_evictions", 0.0)
    )
    if evictions != 0.0:
        raise RuntimeError(
            "histogram cache evicted entries during confirmation compilation"
        )


def _compact_run_row(
    payload: dict[str, Any],
    *,
    run_key: str,
    phase: str,
    tuned_player: str,
    candidate: LearnerCalibrationCandidate,
    opponent: LearnerCalibrationCandidate,
    retain_strategies: bool,
) -> dict[str, Any]:
    strategic = payload["strategic"]
    false_signal = payload["alice_false_signal_telemetry"]
    row = {
        "run_key": run_key,
        "phase": phase,
        "tuned_player": tuned_player,
        "case_id": payload["case_id"],
        "seed": payload["seed"],
        "turns": payload["turns"],
        "candidate_label": candidate.label,
        "candidate_family": candidate.family,
        "candidate": candidate.to_dict(),
        "opponent_label": opponent.label,
        "opponent_family": opponent.family,
        "opponent": opponent.to_dict(),
        "alice_family": (
            candidate.family if candidate.player == "alice" else opponent.family
        ),
        "eve_family": (
            candidate.family if candidate.player == "eve" else opponent.family
        ),
        "alice_label": (
            candidate.label if candidate.player == "alice" else opponent.label
        ),
        "eve_label": (
            candidate.label if candidate.player == "eve" else opponent.label
        ),
        "bank_content_digest_blake2b_128": payload[
            "bank_content_digest_blake2b_128"
        ],
        "physics_cache_compiled_before_learning": payload[
            "physics_cache_compiled_before_learning"
        ],
        "learner_outcome_source": payload["learner_outcome_source"],
        "scoring_source": payload["scoring_source"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "turns_per_second": payload["turns_per_second"],
        "mean_alice_reward": payload["mean_alice_reward"],
        "mean_eve_bandit_reward": payload["mean_eve_bandit_reward"],
        "mean_bell_qualified_rate_hz": payload[
            "mean_bell_qualified_rate_hz"
        ],
        "false_signal_rate": false_signal.get(
            "false_signal_rate",
            false_signal.get("false_signal_rate_all_turns"),
        ),
        "false_signal_policy_total_variation": false_signal[
            "policy_attribution"
        ]["policy_total_variation"],
        "target_active_rate_diagnostic": payload[
            "target_active_rate_diagnostic"
        ],
        "information_observation_rate": payload[
            "information_observation_rate"
        ],
        **strategic,
    }
    if retain_strategies:
        row.update({
            "alice_strategy": payload["alice_strategy"],
            "eve_strategy": payload["eve_strategy"],
            "alice_current_policy": payload["alice_current_policy"],
            "eve_current_policy": payload["eve_current_policy"],
            "joint_action_counts": payload["joint_action_counts"],
            "trajectory": payload["trajectory"],
            "outcome_counts": payload["outcome_counts"],
            "alice_runtime": payload["online_learning_telemetry"]["alice"],
            "eve_runtime": payload["online_learning_telemetry"]["eve"],
        })
    return row


def _adapt_live_confirmation_payload(
    live: dict[str, Any],
    *,
    bank: HistogramPayoffBank,
    oracle: CalibrationGameOracle,
) -> dict[str, Any]:
    telemetry = live["online_learning_telemetry"]
    joint_counts = np.asarray(telemetry["joint_action_counts"], dtype=np.int64)
    alice_strategy = np.asarray(live["alice_strategy"], dtype=float)
    eve_strategy = np.asarray(live["eve_strategy"], dtype=float)
    strategic = score_calibration_behavior(
        oracle,
        joint_counts=joint_counts,
        alice_strategy=alice_strategy,
        eve_strategy=eve_strategy,
        include_equilibrium_distance=True,
    )
    return {
        "case_id": bank.case_id,
        "bank_content_digest_blake2b_128": bank.content_digest(),
        "turns": int(live["turns"]),
        "seed": int(live["config"]["seed"]),
        "elapsed_seconds": float(live["elapsed_seconds"]),
        "turns_per_second": live["turns_per_second"],
        "alice_strategy": live["alice_strategy"],
        "eve_strategy": live["eve_strategy"],
        "alice_current_policy": live["alice_current_strategy"],
        "eve_current_policy": live["eve_current_strategy"],
        "joint_action_counts": telemetry["joint_action_counts"],
        "mean_alice_reward": live["mean_alice_utility"],
        "mean_eve_bandit_reward": live["mean_eve_bandit_reward"],
        "mean_bell_qualified_rate_hz": live["mean_bell_qualified_rate_hz"],
        "target_active_rate_diagnostic": live[
            "target_overlap_rate_diagnostic"
        ],
        "information_observation_rate": live["information_observation_rate"],
        "outcome_counts": live["outcome_counts"],
        "alice_false_signal_telemetry": live["alice_false_signal_telemetry"],
        "strategic": strategic,
        "trajectory": live["learning_curve"],
        "online_learning_telemetry": telemetry,
        "learner_outcome_source": (
            "fresh_compiled_histogram_transcript_per_turn"
        ),
        "scoring_source": "reserved_equal_coverage_audit_bank",
        "physics_cache_compiled_before_learning": True,
    }


def _aggregate_confirmation_rows(
    rows: Sequence[dict[str, Any]],
    *,
    timing_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = (
        "alice_external_regret_per_turn",
        "eve_external_regret_per_turn",
        "exploitability",
        "alice_equilibrium_set_total_variation",
        "eve_equilibrium_set_total_variation",
        "mean_alice_reward",
        "mean_bell_qualified_rate_hz",
        "false_signal_rate",
        "false_signal_policy_total_variation",
    )
    result = []
    for alice_family in LEARNER_FAMILIES:
        for eve_family in LEARNER_FAMILIES:
            selected = [
                row for row in rows
                if row["alice_family"] == alice_family
                and row["eve_family"] == eve_family
            ]
            if not selected:
                continue
            summary: dict[str, Any] = {
                "alice_family": alice_family,
                "eve_family": eve_family,
                "run_count": len(selected),
                "aggregation_contract": (
                    "equal-weight case/seed mean; seed-block uncertainty "
                    "macro-averages across the five fixed calibration cases"
                ),
            }
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in selected])
                summary[f"mean_{metric}"] = float(np.mean(values))
                summary[f"case_seed_run_standard_error_{metric}"] = (
                    float(np.std(values, ddof=1) / np.sqrt(values.size))
                    if values.size > 1 else 0.0
                )
                seed_block_means = [
                    float(np.mean([
                        float(row[metric]) for row in selected
                        if int(row["seed"]) == seed
                    ]))
                    for seed in sorted({int(row["seed"]) for row in selected})
                ]
                summary[f"seed_block_means_{metric}"] = seed_block_means
                summary[f"seed_block_standard_error_{metric}"] = (
                    float(
                        np.std(seed_block_means, ddof=1)
                        / np.sqrt(len(seed_block_means))
                    )
                    if len(seed_block_means) > 1 else 0.0
                )
                summary[f"case_means_{metric}"] = {
                    case_id: float(np.mean([
                        float(row[metric]) for row in selected
                        if str(row["case_id"]) == case_id
                    ]))
                    for case_id in sorted({
                        str(row["case_id"]) for row in selected
                    })
                }
            selected_timing = [
                row for row in timing_rows
                if row["alice_family"] == alice_family
                and row["eve_family"] == eve_family
            ]
            if len(selected_timing) != len({
                str(row["case_id"]) for row in selected
            }):
                raise ValueError(
                    "confirmation timing must contain one row per case and "
                    "learner-family pair"
                )
            timing_values = np.asarray([
                float(row["turns_per_second"]) for row in selected_timing
            ])
            summary["mean_turns_per_second"] = float(np.mean(timing_values))
            summary["case_standard_error_turns_per_second"] = (
                float(
                    np.std(timing_values, ddof=1)
                    / np.sqrt(timing_values.size)
                )
                if timing_values.size > 1 else 0.0
            )
            summary["case_means_turns_per_second"] = {
                str(row["case_id"]): float(row["turns_per_second"])
                for row in selected_timing
            }
            summary["timing_turns_per_run"] = int(selected_timing[0]["turns"])
            summary["timing_scope"] = "single_worker_uncontended"
            result.append(summary)
    return result


def _validate_resumable_rows(
    rows: Sequence[dict[str, Any]],
    *,
    expected_contracts: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    """Require every persisted run row to match the active frozen contract."""

    for row in rows:
        run_key = str(row["run_key"])
        expected = expected_contracts.get(run_key)
        if expected is None:
            raise ValueError(
                f"run key {run_key!r} in {path} is outside the active plan"
            )
        mismatched = [
            field for field, value in expected.items()
            if row.get(field) != value
        ]
        if mismatched:
            raise ValueError(
                f"run row {run_key!r} in {path} changed contract fields: "
                f"{mismatched}"
            )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict) or not row.get("run_key"):
                raise ValueError(f"invalid run row at {path}:{line_number}")
            rows.append(row)
    if len({row["run_key"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate run keys in {path}")
    return rows


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
