"""Corpus-wide evaluation of calibrated Stackelberg route commitment.

The five calibration topologies remain the sole source of the selected
route-response margin.  This module evaluates that frozen margin and the
zero-margin boundary on every structurally eligible topology in the 105-graph
corpus.  One graph-derived seed is used per topology, matching the indexing
unit of the completed corpus co-learning experiment.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

from .calibration_bank import build_histogram_payoff_bank
from .config import HistogramCoLearningConfig, REPO_ROOT
from .corpus_scale import load_corpus_scale_case
from .physics import HistogramPhysicsPool, histogram_cache_plan
from .stackelberg_eve import solve_stackelberg_oracle
from .stackelberg_eve_corpus import STACKELBERG_DEFENDERS, StackelbergDefender
from .stackelberg_steering import (
    StackelbergSteeringConfig,
    feasible_stackelberg_response_margins,
    run_pessimistic_stackelberg_steering,
)
from .topology_census import available_ram_snapshot


STACKELBERG_FOLLOWUP_SCHEMA = "histogram_stackelberg_followup_corpus.v1"
STACKELBERG_FOLLOWUP_CHECKPOINT_SCHEMA = (
    "histogram_stackelberg_followup_checkpoint.v1"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runs" / "histogram_stackelberg_followup_105_500k"
)
DEFAULT_RESPONSE_MARGINS = (0.0, 0.01)
DEFAULT_BANK_ROOT_SEED = 20_260_729_01
CALIBRATION_GRAPH_IDS = (
    "0065_topology_bench_memorex_seed_na",
    "0043_topology_bench_hiberniaireland_seed_na",
    "0062_topology_bench_layer42_seed_na",
    "0059_topology_bench_karen_seed_na",
    "0009_topology_bench_cernet_seed_na",
)


ProgressCallback = Callable[[str], None]


def build_stackelberg_followup_manifest(
    corpus_manifest: Mapping[str, Any],
    *,
    turns: int = 500_000,
    response_margins: Sequence[float] = DEFAULT_RESPONSE_MARGINS,
    samples_per_cell_per_split: int = 64,
    record_every: int = 1_000,
) -> dict[str, Any]:
    """Freeze the corpus-wide evaluation without recalibrating the margin."""

    if corpus_manifest.get("endpoint_selection", {}).get("route_universe") != (
        "every simple cycle-free Alice--Bob path"
    ):
        raise ValueError("source corpus route universe changed")
    if turns < 1:
        raise ValueError("turns must be positive")
    if samples_per_cell_per_split < 2:
        raise ValueError("samples per cell must be at least two")
    margins = tuple(sorted(set(map(float, response_margins))))
    if margins != DEFAULT_RESPONSE_MARGINS:
        raise ValueError(
            "the follow-up requires the zero boundary and calibrated 0.01 "
            "route-response margin"
        )
    cases = [dict(row) for row in corpus_manifest["cases"]]
    root_seed = int(corpus_manifest["plan"]["root_seed"])
    seeds = {
        str(row["graph_id"]): _graph_seed(root_seed, str(row["graph_id"]))
        for row in cases
    }
    preflight_rows = [_preflight_row(row) for row in cases]
    structurally_eligible = [
        row for row in preflight_rows if row["route_count"] >= 2
    ]
    return {
        "schema": STACKELBERG_FOLLOWUP_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_corpus_selection_digest_blake2b_128": str(
            corpus_manifest["selection_digest_blake2b_128"]
        ),
        "source_corpus_plan": dict(corpus_manifest["plan"]),
        "cases": cases,
        "defenders": [row.to_dict() for row in STACKELBERG_DEFENDERS],
        "turns": int(turns),
        "response_margins": list(margins),
        "samples_per_cell_per_split": int(samples_per_cell_per_split),
        "record_every": int(record_every),
        "bank_root_seed": DEFAULT_BANK_ROOT_SEED,
        "graph_seeds": seeds,
        "maximum_condition_run_count": (
            len(structurally_eligible)
            * len(STACKELBERG_DEFENDERS)
            * len(margins)
        ),
        "margin_source": {
            "source": "five-topology calibration",
            "run_directory": "runs/histogram_stackelberg_steering_500k",
            "graph_ids": list(CALIBRATION_GRAPH_IDS),
            "selected_margin": 0.01,
            "selection_rule": "one-standard-error rule",
            "recalibrated_on_full_corpus": False,
        },
        "scientific_contract": {
            "indexing_unit": "topology",
            "route_universe": "every simple cycle-free Alice--Bob path",
            "seed_design": (
                "one deterministic graph-derived seed shared by both "
                "learners and both margins"
            ),
            "structural_eligibility": "at least two Alice routes",
            "feedback_bank": (
                "constructs and executes the stationary commitment"
            ),
            "audit_bank": "disjoint equal-coverage evaluation draws",
            "learner_observations": "selected-route scalar reward only",
            "leader_observations": (
                "declared feedback payoff matrix and action costs; no live "
                "Alice weights or policy"
            ),
        },
        "preflight": {
            "case_count": len(preflight_rows),
            "structurally_eligible_case_count": len(structurally_eligible),
            "structurally_ineligible_case_count": (
                len(preflight_rows) - len(structurally_eligible)
            ),
            "cases": preflight_rows,
            "route_count_total": sum(
                row["route_count"] for row in preflight_rows
            ),
            "eve_action_count_total": sum(
                row["eve_action_count"] for row in preflight_rows
            ),
            "nominal_route_action_cells_total": sum(
                row["nominal_route_action_cells"]
                for row in preflight_rows
            ),
            "canonical_cache_entries_total": sum(
                row["canonical_cache_entries"] for row in preflight_rows
            ),
            "estimated_resident_cache_bytes_if_concurrent": sum(
                row["estimated_resident_cache_bytes"]
                for row in preflight_rows
            ),
            "configured_lru_capacity_per_route": max(
                row["configured_lru_capacity_per_route"]
                for row in preflight_rows
            ),
            "available_ram_before_run": available_ram_snapshot(),
        },
    }


def run_stackelberg_followup_study(
    corpus_manifest: Mapping[str, Any],
    study_manifest: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    graph_workers: int = 8,
    route_workers: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run or resume graph-owned boundary and calibrated-margin evaluations."""

    _validate_manifest(study_manifest)
    output = Path(output_dir)
    checkpoint_dir = output / "graph_checkpoints"
    progress_dir = output / "progress"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    _check_existing_manifest(output, study_manifest)
    _write_json_atomic(output / "run_manifest.json", study_manifest)
    _write_json_atomic(output / "preflight.json", study_manifest["preflight"])

    case_by_graph = {
        str(row["graph_id"]): dict(row) for row in study_manifest["cases"]
    }
    complete: dict[str, dict[str, Any]] = {}
    for path in sorted(checkpoint_dir.glob("*.json")):
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        graph_id = str(checkpoint.get("graph_id") or path.stem)
        if graph_id not in case_by_graph:
            raise ValueError(f"checkpoint {path} lies outside the manifest")
        _validate_checkpoint(checkpoint, case_by_graph[graph_id], study_manifest)
        complete[graph_id] = checkpoint

    pending = [
        dict(row)
        for row in study_manifest["cases"]
        if str(row["graph_id"]) not in complete
    ]
    if pending:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("corpus Stackelberg execution requires fork")
        workers = min(max(1, int(graph_workers)), len(pending))
        context = mp.get_context("fork")
        payloads = [
            {
                "entry": entry,
                "corpus_manifest": dict(corpus_manifest),
                "study_manifest": dict(study_manifest),
                "output_dir": str(output),
                "route_workers": int(route_workers),
            }
            for entry in pending
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_run_graph, payload): payload["entry"]
                for payload in payloads
            }
            for future in as_completed(futures):
                checkpoint = future.result()
                complete[str(checkpoint["graph_id"])] = checkpoint
                if progress_callback is not None:
                    progress_callback(
                        "Stackelberg follow-up graph complete "
                        f"{len(complete)}/{len(case_by_graph)}: "
                        f"{checkpoint['family']} ({checkpoint['status']})"
                    )

    ordered = [
        complete[str(row["graph_id"])] for row in study_manifest["cases"]
    ]
    runs = [row for checkpoint in ordered for row in checkpoint["runs"]]
    _write_jsonl_atomic(output / "results.jsonl", runs)
    summary = _summarize(study_manifest, ordered, runs)
    _write_json_atomic(output / "summary.json", summary)
    return summary


def _run_graph(payload: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(payload["entry"])
    study = payload["study_manifest"]
    output = Path(payload["output_dir"])
    graph_id = str(entry["graph_id"])
    started = time.perf_counter()
    progress_path = output / "progress" / f"{graph_id}.json"

    def progress(phase: str, message: str) -> None:
        _write_json_atomic(progress_path, {
            "graph_id": graph_id,
            "family": entry["family"],
            "phase": phase,
            "message": message,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    if int(entry["route_count"]) < 2:
        checkpoint = _ineligible_checkpoint(
            entry,
            status="structurally_ineligible",
            reason="route commitment requires at least two Alice routes",
            elapsed_seconds=time.perf_counter() - started,
        )
        _write_json_atomic(
            output / "graph_checkpoints" / f"{graph_id}.json",
            checkpoint,
        )
        return checkpoint

    progress("case", "loading the frozen cycle-free route universe")
    case = load_corpus_scale_case(entry, payload["corpus_manifest"])
    source = study["source_corpus_plan"]
    seed = int(study["graph_seeds"][graph_id])
    base_config = HistogramCoLearningConfig(
        turns=int(study["turns"]),
        requested_pairs_per_turn=int(source["requested_pairs_per_turn"]),
        final_window=min(int(source["final_window"]), int(study["turns"])),
        record_every=int(study["record_every"]),
        posthoc_samples_per_cell=0,
        seed=seed,
        game_mode="attack_surface_all",
        age_points_per_edge=int(source["age_points_per_edge"]),
        evaluated_cache_size=max(256, int(entry["eve_action_count"]) + 1),
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

    progress("physics", "compiling the shared feedback and audit bank")
    physics = HistogramPhysicsPool(case, base_config)
    selection_digest = str(
        study["source_corpus_selection_digest_blake2b_128"]
    )
    bank = build_histogram_payoff_bank(
        case,
        base_config,
        samples_per_cell=int(study["samples_per_cell_per_split"]),
        root_seed=_graph_seed(int(study["bank_root_seed"]), graph_id),
        physics=physics,
        progress_callback=lambda message: progress("physics", message),
        selection_digest_blake2b_128=selection_digest,
        evaluation_partition_loaded=True,
        sampling_purpose="stackelberg_route_commitment_105_topology_followup",
        sampling_design=(
            "feedback split constructs and executes both commitments; audit "
            "split independently evaluates them"
        ),
        route_workers=int(payload.get("route_workers", 1)),
    )
    feedback_matrix = bank.feedback.alice_payoff_matrix()
    audit_matrix = bank.audit.alice_payoff_matrix()
    feedback_frontier = feasible_stackelberg_response_margins(
        feedback_matrix,
        bank.action_costs,
        study["response_margins"],
    )
    audit_frontier = feasible_stackelberg_response_margins(
        audit_matrix,
        bank.action_costs,
        study["response_margins"],
    )
    strict_margin = max(map(float, study["response_margins"]))
    strict_row = next(
        row
        for row in feedback_frontier
        if float(row["response_margin"]) == strict_margin
    )
    if not bool(strict_row["feasible"]):
        checkpoint = _ineligible_checkpoint(
            entry,
            status="calibrated_margin_infeasible",
            reason=(
                "the feedback bank admits no commitment with route-response "
                f"margin {strict_margin:g}"
            ),
            elapsed_seconds=time.perf_counter() - started,
            cache_plan=cache_plan,
            bank=bank.to_summary(),
            physics=physics.diagnostics(),
            feedback_margin_frontier=feedback_frontier,
            audit_margin_frontier=audit_frontier,
        )
        _write_json_atomic(
            output / "graph_checkpoints" / f"{graph_id}.json",
            checkpoint,
        )
        return checkpoint

    feedback_unconstrained = solve_stackelberg_oracle(
        feedback_matrix,
        bank.action_costs,
    )
    audit_unconstrained = solve_stackelberg_oracle(
        audit_matrix,
        bank.action_costs,
    )
    runs: list[dict[str, Any]] = []
    for defender_row in study["defenders"]:
        defender = StackelbergDefender(
            label=str(defender_row["label"]),
            alice_learner_mode=str(defender_row["alice_learner_mode"]),
            config_overrides=dict(defender_row["config_overrides"]),
        )
        for margin in map(float, study["response_margins"]):
            progress(
                "evaluation",
                f"{defender.label}, margin {margin:g}, seed {seed}",
            )
            config = replace(
                base_config,
                seed=seed,
                **defender.config_overrides,
            )
            run = run_pessimistic_stackelberg_steering(
                case,
                bank,
                config,
                StackelbergSteeringConfig(
                    response_margin=margin,
                    record_every=int(study["record_every"]),
                ),
                seed=seed,
                selection_digest_blake2b_128=selection_digest,
                progress_callback=(
                    lambda step, label=defender.label, active_margin=margin:
                    progress(
                        "evaluation",
                        f"{label}, margin {active_margin:g}: "
                        f"{step}/{study['turns']}",
                    )
                ),
                feedback_unconstrained_oracle=feedback_unconstrained,
                audit_unconstrained_oracle=audit_unconstrained,
            )
            run.update({
                "phase": "corpus_followup",
                "condition": (
                    "calibrated_margin" if margin > 0.0
                    else "zero_margin_boundary"
                ),
                "defender_label": defender.label,
                "bank_content_digest_blake2b_128": bank.content_digest(),
                "source_corpus_selection_digest_blake2b_128": (
                    selection_digest
                ),
                "graph_seed": seed,
                "margin_source": dict(study["margin_source"]),
            })
            runs.append(run)

    checkpoint = {
        "schema": STACKELBERG_FOLLOWUP_CHECKPOINT_SCHEMA,
        "graph_id": graph_id,
        "family": case.family,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "reason": None,
        "elapsed_seconds": time.perf_counter() - started,
        "case": case.to_dict(),
        "cache_plan": cache_plan,
        "bank": bank.to_summary(),
        "physics": physics.diagnostics(),
        "feedback_margin_frontier": feedback_frontier,
        "audit_margin_frontier": audit_frontier,
        "runs": runs,
    }
    _validate_checkpoint(checkpoint, entry, study)
    _write_json_atomic(
        output / "graph_checkpoints" / f"{graph_id}.json",
        checkpoint,
    )
    progress("complete", f"completed {len(runs)} condition runs")
    return checkpoint


def _preflight_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "graph_id": str(entry["graph_id"]),
        "family": str(entry["family"]),
        "route_count": int(entry["route_count"]),
        "eve_action_count": int(entry["eve_action_count"]),
        "nominal_route_action_cells": int(
            entry["nominal_route_action_cells"]
        ),
        "canonical_cache_entries": int(entry["canonical_cache_entries"]),
        "estimated_resident_cache_bytes": int(
            round(float(entry["estimated_cache_mib"]) * 1024**2)
        ),
        "configured_lru_capacity_per_route": max(
            256,
            int(entry["eve_action_count"]) + 1,
        ),
        "structurally_eligible": int(entry["route_count"]) >= 2,
    }


def _ineligible_checkpoint(
    entry: Mapping[str, Any],
    *,
    status: str,
    reason: str,
    elapsed_seconds: float,
    **diagnostics: Any,
) -> dict[str, Any]:
    return {
        "schema": STACKELBERG_FOLLOWUP_CHECKPOINT_SCHEMA,
        "graph_id": str(entry["graph_id"]),
        "family": str(entry["family"]),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "reason": reason,
        "elapsed_seconds": float(elapsed_seconds),
        "case": dict(entry),
        "runs": [],
        **diagnostics,
    }


def _summarize(
    study: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for checkpoint in checkpoints:
        status = str(checkpoint["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    strict = [row for row in runs if row["condition"] == "calibrated_margin"]
    boundary = [
        row for row in runs if row["condition"] == "zero_margin_boundary"
    ]
    strict_keys = {_run_key(row) for row in strict}
    boundary_keys = {_run_key(row) for row in boundary}
    if strict_keys != boundary_keys:
        raise AssertionError("strict and boundary condition identities differ")
    return {
        "schema": STACKELBERG_FOLLOWUP_SCHEMA,
        "complete": len(checkpoints) == len(study["cases"]),
        "graph_count": len(checkpoints),
        "status_counts": status_counts,
        "eligible_graph_count": status_counts.get("complete", 0),
        "condition_run_count": len(runs),
        "paired_topology_learner_count": len(strict_keys),
        "strict_audit_best_response_count": sum(
            bool(row["intended_route_is_audit_best_response"])
            for row in strict
        ),
        "strict_audit_best_response_total": len(strict),
        "elapsed_seconds_summed_across_graph_workers": float(sum(
            float(row["elapsed_seconds"]) for row in checkpoints
        )),
    }


def _run_key(row: Mapping[str, Any]) -> tuple[str, str, int]:
    return (
        str(row["graph_id"]),
        str(row["defender_label"]),
        int(row["seed"]),
    )


def _graph_seed(root_seed: int, graph_id: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(root_seed).to_bytes(8, "little", signed=True))
    digest.update(b"\0")
    digest.update(graph_id.encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") % (2**31 - 1)


def _validate_manifest(study: Mapping[str, Any]) -> None:
    if study.get("schema") != STACKELBERG_FOLLOWUP_SCHEMA:
        raise ValueError("Stackelberg follow-up manifest schema changed")
    if tuple(map(float, study["response_margins"])) != DEFAULT_RESPONSE_MARGINS:
        raise ValueError("Stackelberg follow-up response margins changed")
    if len(study["graph_seeds"]) != len(study["cases"]):
        raise ValueError("Stackelberg follow-up graph seeds are incomplete")
    if study.get("scientific_contract", {}).get("route_universe") != (
        "every simple cycle-free Alice--Bob path"
    ):
        raise ValueError("Stackelberg follow-up route universe changed")


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    entry: Mapping[str, Any],
    study: Mapping[str, Any],
) -> None:
    if checkpoint.get("schema") != STACKELBERG_FOLLOWUP_CHECKPOINT_SCHEMA:
        raise ValueError("Stackelberg follow-up checkpoint schema changed")
    if str(checkpoint.get("graph_id")) != str(entry["graph_id"]):
        raise ValueError("Stackelberg follow-up graph identity changed")
    status = str(checkpoint.get("status"))
    runs = list(checkpoint.get("runs") or [])
    if status != "complete":
        if runs:
            raise ValueError("ineligible checkpoint contains trajectory rows")
        return
    expected = {
        (str(defender["label"]), float(margin), int(
            study["graph_seeds"][str(entry["graph_id"])]
        ))
        for defender in study["defenders"]
        for margin in study["response_margins"]
    }
    observed = {
        (
            str(row["defender_label"]),
            float(row["steering_config"]["response_margin"]),
            int(row["seed"]),
        )
        for row in runs
    }
    if observed != expected or len(runs) != len(expected):
        raise ValueError("Stackelberg follow-up condition identities changed")
    digest = str(study["source_corpus_selection_digest_blake2b_128"])
    if any(
        row.get("source_corpus_selection_digest_blake2b_128") != digest
        for row in runs
    ):
        raise ValueError("Stackelberg follow-up selection digest changed")


def _check_existing_manifest(output: Path, study: Mapping[str, Any]) -> None:
    path = output / "run_manifest.json"
    if not path.is_file():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    stable_keys = (
        "schema",
        "source_corpus_selection_digest_blake2b_128",
        "turns",
        "response_margins",
        "samples_per_cell_per_split",
        "record_every",
        "bank_root_seed",
        "graph_seeds",
    )
    if any(existing.get(key) != study.get(key) for key in stable_keys):
        raise ValueError("existing Stackelberg follow-up manifest changed")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
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
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
