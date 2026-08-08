"""Five-topology withdrawal-time learner-state sensitivity study."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import inspect
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from sequence_game.corpus.compiled_histogram_backend import (
    COMPILED_HISTOGRAM_RUNTIME_VERSION,
)

from .calibration_bank import (
    PAYOFF_BANK_SCHEMA,
    build_histogram_payoff_bank,
)
from .config import HistogramCoLearningConfig, REPO_ROOT
from .corpus_scale import load_corpus_scale_case
from .physics import HistogramPhysicsPool, histogram_cache_plan
from .prime_probe import (
    PRIME_PROBE_PLAN_SCHEMA,
    PRIME_PROBE_WITHDRAWAL_STATE_SCHEMA,
    PrimeProbeSchedule,
    WITHDRAWAL_FULL_RESTART,
    WITHDRAWAL_NATIVE_CONTINUATION,
    WITHDRAWAL_STATE_STRATEGIES,
    WITHDRAWAL_WARMUP_RESTORE,
    _assert_matched_state_operation_branches,
    _compare_withdrawal_strategies,
    _pair_withdrawal_state_branches,
    _run_prime_probe_prefix,
    _run_withdrawal_state_branch,
    _withdrawal_difference_in_differences,
    _withdrawal_service_recovery,
    build_prime_probe_plan,
    run_prime_probe_withdrawal_state_study,
)
from .prime_probe_corpus import (
    PRIME_PROBE_STUDY_SCHEMA,
    PrimeProbeDefender,
)
from .topology_census import available_ram_snapshot


WITHDRAWAL_CORPUS_SCHEMA = (
    "histogram_prime_probe_withdrawal_state_corpus.v2"
)
WITHDRAWAL_GRAPH_CHECKPOINT_SCHEMA = (
    "histogram_prime_probe_withdrawal_state_graph_checkpoint.v2"
)
WITHDRAWAL_RESUME_CONTRACT_SCHEMA = (
    "histogram_prime_probe_withdrawal_state_resume_contract.v1"
)
WITHDRAWAL_IMPLEMENTATION_REVISION = (
    "oracle_withdrawal_state_branching.one_sided_service_recovery.v2"
)
WITHDRAWAL_RECOVERY_CRITERION_ID = (
    "complete_window_rolling_reward_at_least_late_warmup_2p5.v1"
)
DEFAULT_WITHDRAWAL_STATE_RUN_DIR = (
    REPO_ROOT / "runs" / "histogram_prime_probe_withdrawal_state_500k"
)
DEFAULT_SOURCE_PRIME_PROBE_RUN_DIR = (
    REPO_ROOT / "runs" / "histogram_prime_probe_500k"
)
WITHDRAWAL_BANK_ROOT_SEED = 20_260_716_41
WITHDRAWAL_BANK_SAMPLING_PURPOSE = "withdrawal_state_sensitivity"
WITHDRAWAL_BANK_SAMPLING_DESIGN = (
    "feedback split drives learner updates; audit split scores every "
    "withdrawal branch independently"
)
WITHDRAWAL_PREFIX_CONDITIONS = (
    "ordered_intervention_prefix",
    "clean_sham_prefix",
)

ProgressCallback = Callable[[str], None]


def build_withdrawal_state_manifest(
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze a state-management extension of one Prime-and-Probe manifest."""

    cases = [dict(row) for row in source_manifest["cases"]]
    defenders = [dict(row) for row in source_manifest["defenders"]]
    seeds = list(map(int, source_manifest["seeds"]))
    manifest = {
        **dict(source_manifest),
        "schema": WITHDRAWAL_CORPUS_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_prime_probe_schema": str(source_manifest["schema"]),
        "state_management_strategies": list(WITHDRAWAL_STATE_STRATEGIES),
        "prefix_conditions": list(WITHDRAWAL_PREFIX_CONDITIONS),
        "expected_study_runs": len(cases) * len(defenders) * len(seeds),
        "expected_branch_rows": (
            len(cases)
            * len(defenders)
            * len(seeds)
            * 2
            * len(WITHDRAWAL_STATE_STRATEGIES)
        ),
        "scientific_contract": {
            **dict(source_manifest["scientific_contract"]),
            "withdrawal_time": "supplied oracle boundary",
            "endpoint_alarm": False,
            "change_point_estimation": False,
            "clean_sham_state_operations": True,
            "audit_split": "branch scoring only",
        },
    }
    resume_contract = _resume_contract_payload(manifest)
    manifest["resume_contract"] = resume_contract
    manifest["resume_contract_digest_blake2b_128"] = _stable_digest(
        resume_contract
    )
    _validate_study_manifest(manifest)
    return manifest


def withdrawal_state_preflight(
    study_manifest: Mapping[str, Any],
    *,
    source_run_dir: str | Path = DEFAULT_SOURCE_PRIME_PROBE_RUN_DIR,
) -> dict[str, Any]:
    """Report action geometry, cache planning, branch count, and available RAM."""

    source = Path(source_run_dir)
    source_plans: dict[str, Mapping[str, Any]] = {}
    for path in sorted((source / "graph_checkpoints").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_plans[str(payload["graph_id"])] = payload["cache_plan"]
    case_rows = []
    for entry in study_manifest["cases"]:
        graph_id = str(entry["graph_id"])
        cache = source_plans.get(graph_id)
        case_rows.append({
            "graph_id": graph_id,
            "family": str(entry["family"]),
            "route_count": int(entry["route_count"]),
            "eve_action_count": int(entry["eve_action_count"]),
            "nominal_route_action_cells": (
                int(entry["route_count"])
                * int(entry["eve_action_count"])
            ),
            "canonical_cache_entries": (
                int(cache["canonical_unique_move_count_total"])
                if cache is not None else None
            ),
            "estimated_resident_cache_bytes": (
                int(cache["estimated_full_action_surface_cache_bytes"])
                if cache is not None else None
            ),
            "configured_lru_capacity_per_route": (
                int(cache["configured_entries_per_route"])
                if cache is not None else None
            ),
            "minimum_lru_capacity_per_route": (
                int(cache["minimum_entries_per_route_without_eviction"])
                if cache is not None else None
            ),
        })
    return {
        "schema": "histogram_prime_probe_withdrawal_preflight.v1",
        "case_count": len(case_rows),
        "cases": case_rows,
        "route_count_total": sum(row["route_count"] for row in case_rows),
        "eve_action_count_total": sum(
            row["eve_action_count"] for row in case_rows
        ),
        "nominal_route_action_cells_total": sum(
            row["nominal_route_action_cells"] for row in case_rows
        ),
        "canonical_cache_entries_total": _optional_sum(
            row["canonical_cache_entries"] for row in case_rows
        ),
        "estimated_resident_cache_bytes_if_concurrent": _optional_sum(
            row["estimated_resident_cache_bytes"] for row in case_rows
        ),
        "expected_study_runs": int(
            study_manifest["expected_study_runs"]
        ),
        "expected_branch_rows": int(
            study_manifest["expected_branch_rows"]
        ),
        "turns_per_attacked_or_clean_prefix": int(
            study_manifest["schedule"]["warmup_turns"]
            + study_manifest["schedule"]["prime_turns"]
            + study_manifest["schedule"]["probe_turns"]
            + study_manifest["schedule"]["exploit_turns"]
        ),
        "turns_per_recovery_branch": int(
            study_manifest["schedule"]["recovery_turns"]
        ),
        "available_ram_before_run": available_ram_snapshot(),
        "source_cache_plans_complete": len(source_plans) == len(case_rows),
    }


def run_withdrawal_state_corpus_study(
    corpus_manifest: Mapping[str, Any],
    study_manifest: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_WITHDRAWAL_STATE_RUN_DIR,
    graph_workers: int = 5,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute resumable graph-owned withdrawal-state comparisons."""

    _validate_study_manifest(study_manifest)
    output = Path(output_dir)
    checkpoints = output / "graph_checkpoints"
    progress_dir = output / "progress"
    checkpoints.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths = sorted(checkpoints.glob("*.json"))
    existing_manifest_path = output / "run_manifest.json"
    if checkpoint_paths:
        if not existing_manifest_path.is_file():
            raise ValueError(
                "withdrawal-state checkpoints exist without their run manifest"
            )
        existing_manifest = json.loads(
            existing_manifest_path.read_text(encoding="utf-8")
        )
        _validate_study_manifest(existing_manifest)
        if existing_manifest[
            "resume_contract_digest_blake2b_128"
        ] != study_manifest["resume_contract_digest_blake2b_128"]:
            raise ValueError(
                "withdrawal-state output belongs to a different resume contract"
            )
    _write_json_atomic(output / "run_manifest.json", study_manifest)

    case_by_graph = {
        str(row["graph_id"]): dict(row)
        for row in study_manifest["cases"]
    }
    complete: dict[str, dict[str, Any]] = {}
    for path in checkpoint_paths:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        graph_id = str(checkpoint.get("graph_id") or path.stem)
        if graph_id not in case_by_graph:
            raise ValueError(f"checkpoint {path} is outside the frozen study")
        _validate_checkpoint(
            checkpoint,
            case_by_graph[graph_id],
            study_manifest,
        )
        complete[graph_id] = checkpoint
    pending = [
        row for row in study_manifest["cases"]
        if str(row["graph_id"]) not in complete
    ]
    if pending:
        workers = min(max(1, int(graph_workers)), len(pending))
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("withdrawal-state execution requires fork")
        context = mp.get_context("fork")
        payloads = [
            {
                "entry": dict(entry),
                "corpus_manifest": dict(corpus_manifest),
                "study_manifest": dict(study_manifest),
                "output_dir": str(output),
            }
            for entry in pending
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            future_map = {
                executor.submit(_run_withdrawal_graph, payload): payload[
                    "entry"
                ]
                for payload in payloads
            }
            for future in as_completed(future_map):
                checkpoint = future.result()
                entry = future_map[future]
                _validate_checkpoint(
                    checkpoint,
                    case_by_graph[str(entry["graph_id"])],
                    study_manifest,
                )
                complete[str(checkpoint["graph_id"])] = checkpoint
                if progress_callback is not None:
                    progress_callback(
                        "withdrawal-state graph complete "
                        f"{len(complete)}/{len(case_by_graph)}: "
                        f"{checkpoint['family']}"
                    )

    ordered = [
        complete[str(row["graph_id"])]
        for row in study_manifest["cases"]
    ]
    runs = [
        row for checkpoint in ordered for row in checkpoint["runs"]
    ]
    if len(runs) != int(study_manifest["expected_study_runs"]):
        raise AssertionError("withdrawal-state study is incomplete")
    _write_jsonl_atomic(output / "results.jsonl", runs)
    summary = {
        "schema": WITHDRAWAL_CORPUS_SCHEMA,
        "complete": True,
        "graph_count": len(ordered),
        "study_run_count": len(runs),
        "branch_row_count": sum(len(row["branches"]) for row in runs),
        "elapsed_seconds_summed_across_graph_workers": float(
            sum(float(row["elapsed_seconds"]) for row in ordered)
        ),
        "descriptive_aggregates": _descriptive_aggregates(runs),
    }
    _write_json_atomic(output / "summary.json", summary)
    return summary


def _run_withdrawal_graph(payload: Mapping[str, Any]) -> dict[str, Any]:
    entry = payload["entry"]
    corpus_manifest = payload["corpus_manifest"]
    study = payload["study_manifest"]
    output = Path(payload["output_dir"])
    graph_id = str(entry["graph_id"])
    progress_path = output / "progress" / f"{graph_id}.json"
    started = time.perf_counter()

    def progress(phase: str, message: str) -> None:
        _write_json_atomic(progress_path, {
            "graph_id": graph_id,
            "family": entry["family"],
            "phase": phase,
            "message": message,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    progress("case", "loading frozen cycle-free route universe")
    case = load_corpus_scale_case(entry, corpus_manifest)
    source_plan = study["source_corpus_plan"]
    cache_size = max(256, int(entry["eve_action_count"]) + 1)
    base_config = _base_config_for_entry(
        study,
        entry,
        evaluated_cache_size=cache_size,
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

    progress("physics", "compiling equal-coverage feedback/audit bank")
    physics = HistogramPhysicsPool(case, base_config)
    selection_digest = str(
        study["source_corpus_selection_digest_blake2b_128"]
    )
    bank = build_histogram_payoff_bank(
        case,
        base_config,
        samples_per_cell=int(study["samples_per_cell_per_split"]),
        root_seed=WITHDRAWAL_BANK_ROOT_SEED,
        physics=physics,
        progress_callback=lambda message: progress("physics", message),
        selection_digest_blake2b_128=selection_digest,
        evaluation_partition_loaded=True,
        sampling_purpose=WITHDRAWAL_BANK_SAMPLING_PURPOSE,
        sampling_design=WITHDRAWAL_BANK_SAMPLING_DESIGN,
    )
    plan = build_prime_probe_plan(
        case,
        bank,
        base_config,
        selection_digest_blake2b_128=selection_digest,
    )
    schedule = PrimeProbeSchedule(**{
        key: value for key, value in study["schedule"].items()
        if key != "total_turns"
    })
    runs: list[dict[str, Any]] = []
    for defender_row in study["defenders"]:
        defender = PrimeProbeDefender(
            label=str(defender_row["label"]),
            alice_learner_mode=str(defender_row["alice_learner_mode"]),
            config_overrides=dict(defender_row["config_overrides"]),
        )
        for seed in map(int, study["seeds"]):
            progress(
                "learning",
                f"{defender.label}, seed {seed}: paired state branches",
            )
            config = replace(
                base_config,
                seed=seed,
                **defender.config_overrides,
            )
            run = run_prime_probe_withdrawal_state_study(
                case,
                bank,
                config,
                plan,
                schedule=schedule,
                seed=seed,
                selection_digest_blake2b_128=selection_digest,
            )
            run.update({
                "defender_label": defender.label,
                "defender": dict(defender_row),
                "config": config.to_dict(),
                "bank_content_digest_blake2b_128": bank.content_digest(),
                "source_corpus_selection_digest_blake2b_128": (
                    selection_digest
                ),
                "resume_contract_digest_blake2b_128": study[
                    "resume_contract_digest_blake2b_128"
                ],
            })
            runs.append(run)

    bank_summary = bank.to_summary()
    plan_payload = plan.to_dict()
    checkpoint = {
        "schema": WITHDRAWAL_GRAPH_CHECKPOINT_SCHEMA,
        "study_schema": WITHDRAWAL_CORPUS_SCHEMA,
        "run_schema": PRIME_PROBE_WITHDRAWAL_STATE_SCHEMA,
        "resume_contract_digest_blake2b_128": study[
            "resume_contract_digest_blake2b_128"
        ],
        "scientific_implementation_digest_blake2b_128": (
            study["resume_contract"][
                "scientific_implementation_digest_blake2b_128"
            ]
        ),
        "source_corpus_selection_digest_blake2b_128": selection_digest,
        "case_entry_digest_blake2b_128": _stable_digest(entry),
        "graph_id": graph_id,
        "family": case.family,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "schedule": schedule.to_dict(),
        "base_config": base_config.to_dict(),
        "case": case.to_dict(),
        "cache_plan": cache_plan,
        "bank_content_digest_blake2b_128": bank.content_digest(),
        "bank_summary_digest_blake2b_128": _stable_digest(bank_summary),
        "bank": bank_summary,
        "plan_digest_blake2b_128": _stable_digest(plan_payload),
        "plan": plan_payload,
        "physics": physics.diagnostics(),
        "runs": runs,
    }
    _write_json_atomic(
        output / "graph_checkpoints" / f"{graph_id}.json",
        checkpoint,
    )
    progress("complete", f"completed {len(runs)} paired studies")
    return checkpoint


def _descriptive_aggregates(
    runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for run in runs:
        label = str(run["defender_label"])
        for row in run["within_condition_comparisons"]:
            key = (
                label,
                str(row["prefix_condition"]),
                str(row["strategy"]),
            )
            grouped.setdefault(key, []).append(float(
                row[
                    "strategy_minus_native_cumulative_audit_service_deficit"
                ]
            ))
    rows = []
    for key, values in sorted(grouped.items()):
        label, condition, strategy = key
        array = np.asarray(values, dtype=float)
        rows.append({
            "defender_label": label,
            "prefix_condition": condition,
            "strategy": strategy,
            "run_count": int(array.size),
            "median_strategy_minus_native_cumulative_audit_service_deficit": (
                float(np.median(array))
            ),
            "mean_strategy_minus_native_cumulative_audit_service_deficit": (
                float(np.mean(array))
            ),
            "strategy_lower_deficit_count": int(np.sum(array < 0.0)),
            "strategy_equal_deficit_count": int(np.sum(array == 0.0)),
        })
    return rows


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    entry: Mapping[str, Any],
    study: Mapping[str, Any],
) -> None:
    _validate_study_manifest(study)
    if checkpoint.get("schema") != WITHDRAWAL_GRAPH_CHECKPOINT_SCHEMA:
        raise ValueError("withdrawal-state checkpoint schema changed")
    if checkpoint.get("study_schema") != WITHDRAWAL_CORPUS_SCHEMA:
        raise ValueError("withdrawal-state checkpoint study schema changed")
    if checkpoint.get("run_schema") != PRIME_PROBE_WITHDRAWAL_STATE_SCHEMA:
        raise ValueError("withdrawal-state checkpoint run schema changed")
    expected_resume_digest = str(
        study["resume_contract_digest_blake2b_128"]
    )
    if (
        checkpoint.get("resume_contract_digest_blake2b_128")
        != expected_resume_digest
    ):
        raise ValueError("withdrawal-state checkpoint resume contract changed")
    expected_implementation_digest = str(
        study["resume_contract"][
            "scientific_implementation_digest_blake2b_128"
        ]
    )
    if (
        checkpoint.get("scientific_implementation_digest_blake2b_128")
        != expected_implementation_digest
    ):
        raise ValueError(
            "withdrawal-state checkpoint scientific implementation changed"
        )
    if str(checkpoint.get("graph_id")) != str(entry["graph_id"]):
        raise ValueError("withdrawal-state checkpoint graph changed")
    if str(checkpoint.get("family", "")).lower() != str(
        entry["family"]
    ).lower():
        raise ValueError("withdrawal-state checkpoint family changed")
    if checkpoint.get("case_entry_digest_blake2b_128") != _stable_digest(
        entry
    ):
        raise ValueError("withdrawal-state checkpoint frozen case changed")
    expected_selection_digest = str(
        study["source_corpus_selection_digest_blake2b_128"]
    )
    if (
        checkpoint.get("source_corpus_selection_digest_blake2b_128")
        != expected_selection_digest
    ):
        raise ValueError("withdrawal-state checkpoint selection digest changed")
    expected_schedule = dict(study["schedule"])
    if dict(checkpoint.get("schedule") or {}) != expected_schedule:
        raise ValueError("withdrawal-state checkpoint schedule changed")

    case = dict(checkpoint.get("case") or {})
    _validate_checkpoint_case(case, entry)
    cache_plan = dict(checkpoint.get("cache_plan") or {})
    initial_cache_size = max(256, int(entry["eve_action_count"]) + 1)
    minimum_cache_size = int(
        cache_plan.get("minimum_entries_per_route_without_eviction", -1)
    )
    expected_cache_size = max(initial_cache_size, minimum_cache_size)
    if int(cache_plan.get("configured_entries_per_route", -1)) != (
        expected_cache_size
    ):
        raise ValueError("withdrawal-state checkpoint cache capacity changed")
    if not bool(
        cache_plan.get(
            "sufficient_for_declared_action_surface_without_eviction"
        )
    ):
        raise ValueError("withdrawal-state checkpoint cache plan is insufficient")
    if int(cache_plan.get("route_count", -1)) != int(entry["route_count"]):
        raise ValueError("withdrawal-state checkpoint cache route count changed")
    if int(cache_plan.get("declared_action_count", -1)) != int(
        entry["eve_action_count"]
    ):
        raise ValueError("withdrawal-state checkpoint cache action count changed")
    if int(cache_plan.get("global_route_action_pair_count", -1)) != (
        int(entry["route_count"]) * int(entry["eve_action_count"])
    ):
        raise ValueError("withdrawal-state checkpoint cache cell count changed")
    base_config = _base_config_for_entry(
        study,
        entry,
        evaluated_cache_size=expected_cache_size,
    )
    if dict(checkpoint.get("base_config") or {}) != base_config.to_dict():
        raise ValueError("withdrawal-state checkpoint base configuration changed")

    bank = dict(checkpoint.get("bank") or {})
    if checkpoint.get("bank_summary_digest_blake2b_128") != _stable_digest(
        bank
    ):
        raise ValueError("withdrawal-state checkpoint bank summary changed")
    bank_content_digest = str(
        checkpoint.get("bank_content_digest_blake2b_128") or ""
    )
    _validate_checkpoint_bank(
        bank,
        entry,
        study,
        base_config,
        bank_content_digest=bank_content_digest,
    )

    plan = dict(checkpoint.get("plan") or {})
    if checkpoint.get("plan_digest_blake2b_128") != _stable_digest(plan):
        raise ValueError("withdrawal-state checkpoint plan changed")
    _validate_checkpoint_plan(plan, entry)

    expected = len(study["defenders"]) * len(study["seeds"])
    runs = list(checkpoint.get("runs") or [])
    if len(runs) != expected:
        raise ValueError("withdrawal-state checkpoint is incomplete")
    observed = {
        (str(row["defender_label"]), int(row["seed"]))
        for row in runs
    }
    required = {
        (str(defender["label"]), int(seed))
        for defender in study["defenders"]
        for seed in study["seeds"]
    }
    if observed != required:
        raise ValueError("withdrawal-state checkpoint identities changed")
    defenders = {
        str(row["label"]): dict(row) for row in study["defenders"]
    }
    for row in runs:
        label = str(row["defender_label"])
        defender = defenders[label]
        seed = int(row["seed"])
        expected_config = replace(
            base_config,
            seed=seed,
            **dict(defender["config_overrides"]),
        )
        _validate_checkpoint_run(
            row,
            entry,
            study,
            defender,
            expected_config,
            plan=plan,
            bank_content_digest=bank_content_digest,
            resume_contract_digest=expected_resume_digest,
            selection_digest=expected_selection_digest,
        )


def _validate_study_manifest(study: Mapping[str, Any]) -> None:
    if study.get("schema") != WITHDRAWAL_CORPUS_SCHEMA:
        raise ValueError("withdrawal-state study schema changed")
    if study.get("source_prime_probe_schema") != PRIME_PROBE_STUDY_SCHEMA:
        raise ValueError("source Prime-and-Probe study schema changed")
    if tuple(study.get("state_management_strategies") or ()) != (
        WITHDRAWAL_STATE_STRATEGIES
    ):
        raise ValueError("withdrawal-state strategy set changed")
    if tuple(study.get("prefix_conditions") or ()) != (
        WITHDRAWAL_PREFIX_CONDITIONS
    ):
        raise ValueError("withdrawal-state prefix set changed")

    schedule_payload = dict(study.get("schedule") or {})
    total_turns = schedule_payload.pop("total_turns", None)
    try:
        resolved_schedule = PrimeProbeSchedule(**schedule_payload)
    except (TypeError, ValueError) as exc:
        raise ValueError("withdrawal-state schedule is invalid") from exc
    if total_turns != resolved_schedule.total_turns or dict(
        study["schedule"]
    ) != resolved_schedule.to_dict():
        raise ValueError("withdrawal-state schedule is inconsistent")
    source_plan = dict(study.get("source_corpus_plan") or {})
    if int(source_plan.get("turns", -1)) != resolved_schedule.total_turns:
        raise ValueError("withdrawal-state schedule differs from source horizon")

    cases = [dict(row) for row in study.get("cases") or ()]
    defenders = [dict(row) for row in study.get("defenders") or ()]
    seeds = [int(seed) for seed in study.get("seeds") or ()]
    graph_ids = [str(row.get("graph_id")) for row in cases]
    labels = [str(row.get("label")) for row in defenders]
    if not cases or len(set(graph_ids)) != len(graph_ids):
        raise ValueError("withdrawal-state cases are empty or duplicated")
    if not defenders or len(set(labels)) != len(labels):
        raise ValueError("withdrawal-state defenders are empty or duplicated")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("withdrawal-state seeds are empty or duplicated")
    for entry in cases:
        if len(entry.get("route_ids") or ()) != int(entry["route_count"]):
            raise ValueError("withdrawal-state case route identities changed")
        if len(entry.get("eve_action_ids") or ()) != int(
            entry["eve_action_count"]
        ):
            raise ValueError("withdrawal-state case action identities changed")
    for defender in defenders:
        overrides = dict(defender.get("config_overrides") or {})
        if overrides.get("alice_learner_mode") != defender.get(
            "alice_learner_mode"
        ):
            raise ValueError("withdrawal-state defender mode is inconsistent")
        if "seed" in overrides or "turns" in overrides:
            raise ValueError(
                "withdrawal-state defender overrides fixed run identity"
            )
    expected_runs = len(cases) * len(defenders) * len(seeds)
    expected_branches = (
        expected_runs
        * len(WITHDRAWAL_PREFIX_CONDITIONS)
        * len(WITHDRAWAL_STATE_STRATEGIES)
    )
    if int(study.get("expected_study_runs", -1)) != expected_runs:
        raise ValueError("withdrawal-state expected run count changed")
    if int(study.get("expected_branch_rows", -1)) != expected_branches:
        raise ValueError("withdrawal-state expected branch count changed")
    if int(study.get("samples_per_cell_per_split", -1)) < 2:
        raise ValueError("withdrawal-state bank coverage is invalid")

    expected_contract = _resume_contract_payload(study)
    if dict(study.get("resume_contract") or {}) != expected_contract:
        raise ValueError("withdrawal-state resume contract changed")
    if study.get("resume_contract_digest_blake2b_128") != _stable_digest(
        expected_contract
    ):
        raise ValueError("withdrawal-state resume contract digest changed")


def _resume_contract_payload(study: Mapping[str, Any]) -> dict[str, Any]:
    cases = [dict(row) for row in study["cases"]]
    return {
        "schema": WITHDRAWAL_RESUME_CONTRACT_SCHEMA,
        "implementation_revision": WITHDRAWAL_IMPLEMENTATION_REVISION,
        "scientific_implementation_digest_blake2b_128": (
            _scientific_implementation_digest()
        ),
        "compiled_histogram_runtime_version": (
            COMPILED_HISTOGRAM_RUNTIME_VERSION
        ),
        "payoff_bank_schema": PAYOFF_BANK_SCHEMA,
        "study_schema": WITHDRAWAL_CORPUS_SCHEMA,
        "graph_checkpoint_schema": WITHDRAWAL_GRAPH_CHECKPOINT_SCHEMA,
        "run_schema": PRIME_PROBE_WITHDRAWAL_STATE_SCHEMA,
        "plan_schema": PRIME_PROBE_PLAN_SCHEMA,
        "recovery_criterion_id": WITHDRAWAL_RECOVERY_CRITERION_ID,
        "recovery_criterion": (
            "rolling_alice_reward >= baseline_lower_2p5"
        ),
        "requires_complete_rolling_window": True,
        "bank_root_seed": WITHDRAWAL_BANK_ROOT_SEED,
        "bank_sampling_purpose": WITHDRAWAL_BANK_SAMPLING_PURPOSE,
        "bank_sampling_design": WITHDRAWAL_BANK_SAMPLING_DESIGN,
        "source_prime_probe_schema": str(
            study["source_prime_probe_schema"]
        ),
        "source_corpus_selection_digest_blake2b_128": str(
            study["source_corpus_selection_digest_blake2b_128"]
        ),
        "source_corpus_plan": dict(study["source_corpus_plan"]),
        "cases": cases,
        "defenders": [dict(row) for row in study["defenders"]],
        "seeds": list(map(int, study["seeds"])),
        "schedule": dict(study["schedule"]),
        "samples_per_cell_per_split": int(
            study["samples_per_cell_per_split"]
        ),
        "state_management_strategies": list(
            study["state_management_strategies"]
        ),
        "prefix_conditions": list(study["prefix_conditions"]),
        "expected_study_runs": int(study["expected_study_runs"]),
        "expected_branch_rows": int(study["expected_branch_rows"]),
        "scientific_contract": dict(study["scientific_contract"]),
        "base_config_templates": [
            {
                "graph_id": str(entry["graph_id"]),
                "config": _base_config_for_entry(study, entry).to_dict(),
            }
            for entry in cases
        ],
    }


def _scientific_implementation_digest() -> str:
    functions = (
        build_histogram_payoff_bank,
        build_prime_probe_plan,
        run_prime_probe_withdrawal_state_study,
        _run_prime_probe_prefix,
        _run_withdrawal_state_branch,
        _withdrawal_service_recovery,
        _assert_matched_state_operation_branches,
        _pair_withdrawal_state_branches,
        _compare_withdrawal_strategies,
        _withdrawal_difference_in_differences,
        _run_withdrawal_graph,
    )
    digest = hashlib.blake2b(digest_size=16)
    for function in functions:
        digest.update(
            f"{function.__module__}.{function.__qualname__}\0".encode("utf-8")
        )
        digest.update(inspect.getsource(function).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _base_config_for_entry(
    study: Mapping[str, Any],
    entry: Mapping[str, Any],
    *,
    evaluated_cache_size: int | None = None,
) -> HistogramCoLearningConfig:
    source_plan = study["source_corpus_plan"]
    cache_size = (
        max(256, int(entry["eve_action_count"]) + 1)
        if evaluated_cache_size is None
        else int(evaluated_cache_size)
    )
    return HistogramCoLearningConfig(
        turns=int(study["schedule"]["total_turns"]),
        requested_pairs_per_turn=int(
            source_plan["requested_pairs_per_turn"]
        ),
        final_window=int(source_plan["final_window"]),
        record_every=int(study["schedule"]["record_every"]),
        posthoc_samples_per_cell=0,
        seed=int(study["seeds"][0]),
        game_mode="attack_surface_all",
        age_points_per_edge=int(source_plan["age_points_per_edge"]),
        evaluated_cache_size=cache_size,
    )


def _validate_checkpoint_case(
    case: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> None:
    expected_case_id = str(
        entry.get("case_id")
        or f"{entry['graph_id']}::{entry['assignment_id']}"
    )
    checks = (
        ("graph_id", str(entry["graph_id"])),
        ("assignment_id", str(entry["assignment_id"])),
        ("family", str(entry["family"])),
        ("alice", str(entry["alice"])),
        ("bob", str(entry["bob"])),
    )
    for key, expected in checks:
        if str(case.get(key)) != expected:
            raise ValueError(
                f"withdrawal-state checkpoint case {key} changed"
            )
    if f"{case['graph_id']}::{case['assignment_id']}" != expected_case_id:
        raise ValueError("withdrawal-state checkpoint case id changed")
    if int(case.get("route_count", -1)) != int(entry["route_count"]):
        raise ValueError("withdrawal-state checkpoint case route count changed")
    if list(case.get("route_ids") or ()) != list(entry["route_ids"]):
        raise ValueError(
            "withdrawal-state checkpoint case route identities changed"
        )
    if dict(case.get("route_universe") or {}) != dict(
        entry.get("route_universe") or {}
    ):
        raise ValueError(
            "withdrawal-state checkpoint route-universe contract changed"
        )


def _validate_checkpoint_bank(
    bank: Mapping[str, Any],
    entry: Mapping[str, Any],
    study: Mapping[str, Any],
    base_config: HistogramCoLearningConfig,
    *,
    bank_content_digest: str,
) -> None:
    expected_case_id = str(
        entry.get("case_id")
        or f"{entry['graph_id']}::{entry['assignment_id']}"
    )
    if bank.get("schema") != PAYOFF_BANK_SCHEMA:
        raise ValueError("withdrawal-state checkpoint bank schema changed")
    if str(bank.get("case_id")) != expected_case_id:
        raise ValueError("withdrawal-state checkpoint bank case changed")
    if str(bank.get("graph_id")) != str(entry["graph_id"]):
        raise ValueError("withdrawal-state checkpoint bank graph changed")
    if str(bank.get("assignment_id")) != str(entry["assignment_id"]):
        raise ValueError("withdrawal-state checkpoint bank assignment changed")
    if int(bank.get("route_count", -1)) != int(entry["route_count"]):
        raise ValueError("withdrawal-state checkpoint bank route count changed")
    if int(bank.get("action_count", -1)) != int(
        entry["eve_action_count"]
    ):
        raise ValueError("withdrawal-state checkpoint bank action count changed")
    draws = int(study["samples_per_cell_per_split"])
    if int(bank.get("draws_per_cell_per_split", -1)) != draws:
        raise ValueError("withdrawal-state checkpoint bank coverage changed")
    expected_total = (
        2 * int(entry["route_count"]) * int(entry["eve_action_count"]) * draws
    )
    if int(bank.get("total_histogram_draws", -1)) != expected_total:
        raise ValueError("withdrawal-state checkpoint bank draw count changed")
    if not bank_content_digest or bank.get(
        "content_digest_blake2b_128"
    ) != bank_content_digest:
        raise ValueError("withdrawal-state checkpoint bank content changed")
    metadata = dict(bank.get("metadata") or {})
    expected_metadata = {
        "selection_digest_blake2b_128": str(
            study["source_corpus_selection_digest_blake2b_128"]
        ),
        "sampling_design": WITHDRAWAL_BANK_SAMPLING_DESIGN,
        "sampling_purpose": WITHDRAWAL_BANK_SAMPLING_PURPOSE,
        "compiled_histogram_runtime_version": (
            COMPILED_HISTOGRAM_RUNTIME_VERSION
        ),
        "samples_per_cell_per_split": draws,
        "root_seed": WITHDRAWAL_BANK_ROOT_SEED,
        "game_mode": base_config.game_mode,
        "requested_pairs_per_draw": base_config.requested_pairs_per_turn,
        "feedback_audit_seed_disjoint": True,
        "evaluation_partition_loaded": True,
        "physics_config": {
            "repeater_service_profile_path": str(
                base_config.repeater_service_profile_path
            ),
            "age_points_per_edge": base_config.age_points_per_edge,
            "accepted_alive_time_s": base_config.accepted_alive_time_s,
            "fock_cutoff": base_config.fock_cutoff,
            "detector_wavelength_approximation_allowed": (
                base_config.allow_known_1550_to_1558_detector_approximation
            ),
        },
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"withdrawal-state checkpoint bank metadata {key} changed"
            )
    split_seeds = dict(metadata.get("split_seeds") or {})
    if set(split_seeds) != {"feedback", "audit"}:
        raise ValueError(
            "withdrawal-state checkpoint bank split seeds changed"
        )
    if int(split_seeds["feedback"]) == int(split_seeds["audit"]):
        raise ValueError(
            "withdrawal-state checkpoint bank splits are not seed-disjoint"
        )


def _validate_checkpoint_plan(
    plan: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> None:
    if plan.get("schema") != PRIME_PROBE_PLAN_SCHEMA:
        raise ValueError("withdrawal-state checkpoint plan schema changed")
    route_ids = list(map(str, entry["route_ids"]))
    action_ids = list(map(str, entry["eve_action_ids"]))
    target_indices = list(map(int, plan.get("target_route_indices") or ()))
    target_ids = list(map(str, plan.get("target_route_ids") or ()))
    alternative_indices = list(
        map(int, plan.get("alternative_route_indices") or ())
    )
    if (
        not target_indices
        or not alternative_indices
        or target_ids != [route_ids[index] for index in target_indices]
    ):
        raise ValueError(
            "withdrawal-state checkpoint plan route identities changed"
        )
    if sorted(target_indices + alternative_indices) != list(
        range(len(route_ids))
    ):
        raise ValueError(
            "withdrawal-state checkpoint plan route partition changed"
        )
    for prefix in ("exploit", "probe"):
        index = int(plan.get(f"{prefix}_action_index", -1))
        if (
            index < 0
            or index >= len(action_ids)
            or str(plan.get(f"{prefix}_action_id")) != action_ids[index]
        ):
            raise ValueError(
                f"withdrawal-state checkpoint {prefix} action changed"
            )
    prime_indices = list(map(int, plan.get("prime_action_indices") or ()))
    prime_ids = list(map(str, plan.get("prime_action_ids") or ()))
    probabilities = np.asarray(
        plan.get("prime_probabilities") or (), dtype=float
    )
    if (
        not prime_indices
        or prime_ids != [action_ids[index] for index in prime_indices]
        or probabilities.shape != (len(prime_indices),)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not np.isclose(float(np.sum(probabilities)), 1.0)
    ):
        raise ValueError(
            "withdrawal-state checkpoint prime action mixture changed"
        )


def _validate_checkpoint_run(
    run: Mapping[str, Any],
    entry: Mapping[str, Any],
    study: Mapping[str, Any],
    defender: Mapping[str, Any],
    expected_config: HistogramCoLearningConfig,
    *,
    plan: Mapping[str, Any],
    bank_content_digest: str,
    resume_contract_digest: str,
    selection_digest: str,
) -> None:
    if run.get("schema") != PRIME_PROBE_WITHDRAWAL_STATE_SCHEMA:
        raise ValueError("withdrawal-state run schema changed")
    if str(run.get("graph_id")) != str(entry["graph_id"]):
        raise ValueError("withdrawal-state run graph changed")
    if str(run.get("family", "")).lower() != str(entry["family"]).lower():
        raise ValueError("withdrawal-state run family changed")
    expected_case_id = str(
        entry.get("case_id")
        or f"{entry['graph_id']}::{entry['assignment_id']}"
    )
    if str(run.get("case_id")) != expected_case_id:
        raise ValueError("withdrawal-state run case changed")
    if dict(run.get("schedule") or {}) != dict(study["schedule"]):
        raise ValueError("withdrawal-state run schedule changed")
    if dict(run.get("plan") or {}) != dict(plan):
        raise ValueError("withdrawal-state run plan changed")
    if dict(run.get("defender") or {}) != dict(defender):
        raise ValueError("withdrawal-state run defender changed")
    if dict(run.get("config") or {}) != expected_config.to_dict():
        raise ValueError("withdrawal-state run configuration changed")
    if str(run.get("defender_label")) != str(defender["label"]):
        raise ValueError("withdrawal-state run defender label changed")
    if int(run.get("seed", -1)) != int(expected_config.seed):
        raise ValueError("withdrawal-state run seed changed")
    if run.get("bank_content_digest_blake2b_128") != bank_content_digest:
        raise ValueError("withdrawal-state run bank content changed")
    if (
        run.get("source_corpus_selection_digest_blake2b_128")
        != selection_digest
    ):
        raise ValueError("withdrawal-state run selection digest changed")
    if (
        run.get("resume_contract_digest_blake2b_128")
        != resume_contract_digest
    ):
        raise ValueError("withdrawal-state run resume contract changed")
    learner = dict(run.get("alice_learner") or {})
    if learner.get("mode") != defender["alice_learner_mode"]:
        raise ValueError("withdrawal-state run learner mode changed")
    if int(learner.get("action_count", -1)) != int(entry["route_count"]):
        raise ValueError("withdrawal-state run learner action count changed")

    recovery_start = sum(
        int(study["schedule"][key])
        for key in (
            "warmup_turns",
            "prime_turns",
            "probe_turns",
            "exploit_turns",
        )
    )
    prefixes = dict(run.get("prefixes") or {})
    if set(prefixes) != set(WITHDRAWAL_PREFIX_CONDITIONS):
        raise ValueError("withdrawal-state run prefix identities changed")
    for condition in WITHDRAWAL_PREFIX_CONDITIONS:
        prefix = dict(prefixes[condition])
        if int(prefix.get("turns", -1)) != recovery_start:
            raise ValueError("withdrawal-state run prefix turn count changed")
        if int(prefix.get("ending_alice_update_calls", -1)) != recovery_start:
            raise ValueError("withdrawal-state run prefix updates changed")

    branches = list(run.get("branches") or ())
    expected_branch_keys = {
        (condition, strategy)
        for condition in WITHDRAWAL_PREFIX_CONDITIONS
        for strategy in WITHDRAWAL_STATE_STRATEGIES
    }
    observed_branch_keys = {
        (str(row.get("prefix_condition")), str(row.get("strategy")))
        for row in branches
    }
    if (
        len(branches) != len(expected_branch_keys)
        or observed_branch_keys != expected_branch_keys
    ):
        raise ValueError("withdrawal-state branch identities changed")
    for branch in branches:
        _validate_checkpoint_branch(
            branch,
            study,
            recovery_start,
            route_count=int(entry["route_count"]),
        )

    attacked_clean_pairs = list(run.get("attacked_clean_pairs") or ())
    if (
        len(attacked_clean_pairs) != len(WITHDRAWAL_STATE_STRATEGIES)
        or {
            str(row.get("strategy")) for row in attacked_clean_pairs
        } != set(WITHDRAWAL_STATE_STRATEGIES)
    ):
        raise ValueError("withdrawal-state attacked/clean pairs changed")
    expected_comparisons = {
        (condition, strategy)
        for condition in WITHDRAWAL_PREFIX_CONDITIONS
        for strategy in (
            WITHDRAWAL_FULL_RESTART,
            WITHDRAWAL_WARMUP_RESTORE,
        )
    }
    comparisons = list(run.get("within_condition_comparisons") or ())
    if (
        len(comparisons) != len(expected_comparisons)
        or {
            (
                str(row.get("prefix_condition")),
                str(row.get("strategy")),
            )
            for row in comparisons
        } != expected_comparisons
    ):
        raise ValueError(
            "withdrawal-state within-condition comparisons changed"
        )
    comparison_by_key = {
        (
            str(row["prefix_condition"]),
            str(row["strategy"]),
        ): row
        for row in comparisons
    }
    difference_rows = list(run.get("difference_in_differences") or ())
    expected_did_strategies = {
        WITHDRAWAL_FULL_RESTART,
        WITHDRAWAL_WARMUP_RESTORE,
    }
    if (
        len(difference_rows) != len(expected_did_strategies)
        or {
            str(row.get("strategy")) for row in difference_rows
        } != expected_did_strategies
    ):
        raise ValueError(
            "withdrawal-state difference-in-differences identities changed"
        )
    for row in difference_rows:
        strategy = str(row["strategy"])
        attacked = comparison_by_key[
            ("ordered_intervention_prefix", strategy)
        ]
        clean = comparison_by_key[("clean_sham_prefix", strategy)]
        expected_values = {
            "did_cumulative_audit_service_deficit": (
                float(attacked[
                    "strategy_minus_native_cumulative_audit_service_deficit"
                ])
                - float(clean[
                    "strategy_minus_native_cumulative_audit_service_deficit"
                ])
            ),
            "did_mean_feedback_alice_reward": (
                float(attacked[
                    "strategy_minus_native_mean_feedback_reward"
                ])
                - float(clean[
                    "strategy_minus_native_mean_feedback_reward"
                ])
            ),
            "did_final_audit_policy_service": (
                float(attacked[
                    "strategy_minus_native_final_audit_policy_service"
                ])
                - float(clean[
                    "strategy_minus_native_final_audit_policy_service"
                ])
            ),
        }
        for key, expected_value in expected_values.items():
            if float(row.get(key, np.nan)) != expected_value:
                raise ValueError(
                    "withdrawal-state difference-in-differences value changed"
                )


def _validate_checkpoint_branch(
    branch: Mapping[str, Any],
    study: Mapping[str, Any],
    recovery_start: int,
    *,
    route_count: int,
) -> None:
    strategy = str(branch["strategy"])
    recovery_turns = int(study["schedule"]["recovery_turns"])
    warmup_turns = int(study["schedule"]["warmup_turns"])
    expected_start = {
        WITHDRAWAL_NATIVE_CONTINUATION: recovery_start,
        WITHDRAWAL_FULL_RESTART: 0,
        WITHDRAWAL_WARMUP_RESTORE: warmup_turns,
    }[strategy]
    if int(branch.get("recovery_turns", -1)) != recovery_turns:
        raise ValueError("withdrawal-state branch recovery turn count changed")
    if int(branch.get("starting_alice_update_calls", -1)) != expected_start:
        raise ValueError("withdrawal-state branch starting state changed")
    if int(branch.get("ending_alice_update_calls", -1)) != (
        expected_start + recovery_turns
    ):
        raise ValueError("withdrawal-state branch update count changed")

    record_every = int(study["schedule"]["record_every"])
    expected_turns = sorted({
        1,
        recovery_turns,
        *range(record_every, recovery_turns + 1, record_every),
    })
    trajectory = list(branch.get("trajectory") or ())
    observed_turns = [
        int(row.get("recovery_turn", -1)) for row in trajectory
    ]
    if observed_turns != expected_turns:
        raise ValueError(
            "withdrawal-state branch trajectory checkpoints changed"
        )
    policy = np.asarray(branch.get("current_policy") or (), dtype=float)
    if (
        policy.ndim != 1
        or policy.size != int(route_count)
        or not np.all(np.isfinite(policy))
        or np.any(policy < 0.0)
        or not np.isclose(float(np.sum(policy)), 1.0)
    ):
        raise ValueError("withdrawal-state branch final policy is invalid")
    recovery = dict(branch.get("service_recovery") or {})
    if int(recovery.get("censor_turns", -1)) != recovery_turns:
        raise ValueError("withdrawal-state recovery censor changed")
    if recovery.get("criterion") != (
        "rolling_alice_reward >= baseline_lower_2p5"
    ):
        raise ValueError("withdrawal-state recovery criterion changed")
    recovered = bool(recovery.get("recovered"))
    right_censored = bool(recovery.get("right_censored"))
    latency = recovery.get("latency_turns")
    if recovered:
        if right_censored or latency is None or int(latency) not in expected_turns:
            raise ValueError("withdrawal-state recovered branch is inconsistent")
    elif not right_censored or latency is not None:
        raise ValueError("withdrawal-state censored branch is inconsistent")


def _stable_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _optional_sum(values: Sequence[int | None]) -> int | None:
    materialized = list(values)
    if any(value is None for value in materialized):
        return None
    return int(sum(int(value) for value in materialized if value is not None))


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(dict(row), sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
