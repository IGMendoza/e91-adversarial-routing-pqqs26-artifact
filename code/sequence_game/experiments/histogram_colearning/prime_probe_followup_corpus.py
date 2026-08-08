"""Corpus-wide ordered Prime--Probe--Exploit evaluation with Alice alarms.

The controller and detector hyperparameters selected on five calibration
topologies are frozen.  This module applies them to every structurally
eligible graph in the 105-topology corpus.  Planning uses one payoff bank;
online reward, Eve's local Probe record, detector commissioning records, and
Alice's live endpoint records come from a second independently seeded matched
payoff/endpoint bank.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

from .calibration_bank import HistogramPayoffBank, build_histogram_payoff_bank
from .config import HistogramCoLearningConfig, REPO_ROOT
from .corpus_scale import load_corpus_scale_case
from .endpoint_evidence_bank import HistogramEndpointEvidenceBank
from .matched_payoff_endpoint_bank import (
    build_matched_histogram_payoff_and_endpoint_banks,
)
from .physics import HistogramPhysicsPool, histogram_cache_plan
from .prime_probe import (
    PrimeProbePlan,
    PrimeProbeSchedule,
    build_prime_probe_plan,
    run_ordered_prime_probe,
)
from .prime_probe_corpus import PRIME_PROBE_DEFENDERS, PrimeProbeDefender
from .prime_probe_detection_corpus import _validate_ordered_detection_run
from .quantum_stackelberg_detection import BinomialEvidenceDetectorConfig
from .topology_census import available_ram_snapshot


PRIME_PROBE_FOLLOWUP_SCHEMA = "histogram_prime_probe_followup_corpus.v1"
PRIME_PROBE_FOLLOWUP_CHECKPOINT_SCHEMA = (
    "histogram_prime_probe_followup_checkpoint.v1"
)
PRIME_PROBE_FOLLOWUP_PLAN_SCHEMA = "histogram_prime_probe_followup_plan.v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "histogram_prime_probe_followup_105_500k"
DEFAULT_PLANNING_BANK_ROOT_SEED = 20_260_807_01
DEFAULT_EXECUTION_BANK_ROOT_SEED = 20_260_807_02
DEFAULT_SAMPLES_PER_CELL_PER_SPLIT = 64
DEFAULT_COMMISSIONING_DRAWS_PER_ROUTE = 32
CALIBRATION_GRAPH_IDS = (
    "0065_topology_bench_memorex_seed_na",
    "0043_topology_bench_hiberniaireland_seed_na",
    "0062_topology_bench_layer42_seed_na",
    "0059_topology_bench_karen_seed_na",
    "0009_topology_bench_cernet_seed_na",
)


ProgressCallback = Callable[[str], None]


def build_prime_probe_followup_manifest(
    corpus_manifest: Mapping[str, Any],
    *,
    schedule: PrimeProbeSchedule = PrimeProbeSchedule(),
    samples_per_cell_per_split: int = DEFAULT_SAMPLES_PER_CELL_PER_SPLIT,
) -> dict[str, Any]:
    """Freeze the 105-topology extension of the calibrated ordered study."""

    if corpus_manifest.get("endpoint_selection", {}).get("route_universe") != (
        "every simple cycle-free Alice--Bob path"
    ):
        raise ValueError("source corpus route universe changed")
    if samples_per_cell_per_split < DEFAULT_COMMISSIONING_DRAWS_PER_ROUTE:
        raise ValueError("bank draws must cover detector commissioning")
    cases = [dict(row) for row in corpus_manifest["cases"]]
    root_seed = int(corpus_manifest["plan"]["root_seed"])
    graph_seeds = {
        str(row["graph_id"]): _graph_seed(root_seed, str(row["graph_id"]))
        for row in cases
    }
    rows = [_preflight_row(row) for row in cases]
    structurally_eligible = [row for row in rows if row["route_count"] >= 2]
    detector = {
        "commissioning_split": "feedback",
        "commissioning_action_id": "no_attack",
        "commissioning_draws_per_route": DEFAULT_COMMISSIONING_DRAWS_PER_ROUTE,
        "execution_split": "audit",
        "model_false_alarm_level": 0.01,
        "commissioning_pseudocount": 0.5,
        "change_fractions": [0.10, 0.25, 0.50],
        "horizon": schedule.total_turns,
        "response": "monitor_only_native_learning",
    }
    manifest: dict[str, Any] = {
        "schema": PRIME_PROBE_FOLLOWUP_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_corpus_selection_digest_blake2b_128": str(
            corpus_manifest["selection_digest_blake2b_128"]
        ),
        "source_corpus_plan": dict(corpus_manifest["plan"]),
        "cases": cases,
        "defenders": [row.to_dict() for row in PRIME_PROBE_DEFENDERS],
        "graph_seeds": graph_seeds,
        "schedule": schedule.to_dict(),
        "samples_per_cell_per_split": int(samples_per_cell_per_split),
        "planning_bank_root_seed": DEFAULT_PLANNING_BANK_ROOT_SEED,
        "execution_bank_root_seed": DEFAULT_EXECUTION_BANK_ROOT_SEED,
        "detector": detector,
        "maximum_ordered_run_count": (
            len(structurally_eligible) * len(PRIME_PROBE_DEFENDERS)
        ),
        "calibration_source": {
            "topology_count": 5,
            "graph_ids": list(CALIBRATION_GRAPH_IDS),
            "source_run_directory": "runs/histogram_prime_probe_detection_500k",
            "schedule_retuned": False,
            "plan_rule_retuned": False,
            "detector_retuned": False,
            "learner_retuned": False,
        },
        "scientific_contract": {
            "indexing_unit": "topology",
            "route_universe": "every simple cycle-free Alice--Bob path",
            "seed_design": "one deterministic graph-derived seed per topology",
            "structural_eligibility": "at least two Alice routes",
            "planning": "planning-bank feedback cell means only",
            "detector_commissioning": (
                "execution-bank feedback no-intervention records"
            ),
            "live_execution": (
                "independent execution-bank audit transcripts supply Alice "
                "reward, endpoint counts, and Eve local Probe counts"
            ),
            "detector_response": "monitor only; learner state is unchanged",
            "posthoc_audit_scoring": False,
        },
        "preflight": {
            "case_count": len(rows),
            "structurally_eligible_case_count": len(structurally_eligible),
            "structurally_ineligible_case_count": len(rows) - len(structurally_eligible),
            "cases": rows,
            "route_count_total": sum(row["route_count"] for row in rows),
            "eve_action_count_total": sum(row["eve_action_count"] for row in rows),
            "nominal_route_action_cells_total": sum(
                row["nominal_route_action_cells"] for row in rows
            ),
            "canonical_cache_entries_total": sum(
                row["canonical_cache_entries"] for row in rows
            ),
            "estimated_resident_cache_bytes_if_concurrent": sum(
                row["estimated_resident_cache_bytes"] for row in rows
            ),
            "configured_lru_capacity_per_route": max(
                row["configured_lru_capacity_per_route"] for row in rows
            ),
            "planning_histogram_draws_maximum": (
                2
                * int(samples_per_cell_per_split)
                * sum(row["nominal_route_action_cells"] for row in structurally_eligible)
            ),
            "matched_execution_draws_maximum": (
                2
                * int(samples_per_cell_per_split)
                * sum(row["nominal_route_action_cells"] for row in structurally_eligible)
            ),
            "available_ram_before_run": available_ram_snapshot(),
        },
    }
    manifest["resume_contract_digest_blake2b_128"] = _stable_digest(
        _resume_contract(manifest)
    )
    _validate_manifest(manifest)
    return manifest


def run_prime_probe_followup_study(
    corpus_manifest: Mapping[str, Any],
    study_manifest: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    graph_workers: int = 8,
    route_workers: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run or resume graph-owned ordered experiments."""

    _validate_manifest(study_manifest)
    output = Path(output_dir)
    for name in ("graph_checkpoints", "partial_graph_checkpoints", "plan_checkpoints", "progress", "banks"):
        (output / name).mkdir(parents=True, exist_ok=True)
    _check_existing_manifest(output, study_manifest)
    _write_json_atomic(output / "run_manifest.json", study_manifest)
    _write_json_atomic(output / "preflight.json", study_manifest["preflight"])

    case_by_graph = {
        str(row["graph_id"]): dict(row) for row in study_manifest["cases"]
    }
    complete: dict[str, dict[str, Any]] = {}
    for path in sorted((output / "graph_checkpoints").glob("*.json")):
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
            raise RuntimeError("ordered follow-up execution requires fork")
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
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {
                executor.submit(_run_graph, payload): payload["entry"]
                for payload in payloads
            }
            for future in as_completed(futures):
                checkpoint = future.result()
                complete[str(checkpoint["graph_id"])] = checkpoint
                if progress_callback is not None:
                    progress_callback(
                        "ordered follow-up graph complete "
                        f"{len(complete)}/{len(case_by_graph)}: "
                        f"{checkpoint['family']} ({checkpoint['status']})"
                    )

    checkpoints = [
        complete[str(row["graph_id"])] for row in study_manifest["cases"]
    ]
    runs = [row for checkpoint in checkpoints for row in checkpoint["runs"]]
    _write_jsonl_atomic(output / "results.jsonl", runs)
    summary = _summarize(study_manifest, checkpoints, runs)
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
        return _write_ineligible(
            output,
            entry,
            status="structurally_ineligible",
            reason="ordered route steering requires at least two Alice routes",
            elapsed_seconds=time.perf_counter() - started,
        )

    progress("case", "loading the frozen cycle-free route universe")
    case = load_corpus_scale_case(entry, payload["corpus_manifest"])
    base_config = _base_config(study, entry)
    cache_plan = histogram_cache_plan(case, base_config)
    if not cache_plan["sufficient_for_declared_action_surface_without_eviction"]:
        base_config = replace(
            base_config,
            evaluated_cache_size=int(
                cache_plan["minimum_entries_per_route_without_eviction"]
            ),
        )
        cache_plan = histogram_cache_plan(case, base_config)
    selection_digest = str(
        study["source_corpus_selection_digest_blake2b_128"]
    )

    physics_pool: HistogramPhysicsPool | None = None
    plan_path = output / "plan_checkpoints" / f"{graph_id}.json"
    if plan_path.is_file():
        plan_checkpoint = json.loads(plan_path.read_text(encoding="utf-8"))
        _validate_plan_checkpoint(plan_checkpoint, entry, study)
        if plan_checkpoint["status"] != "complete":
            return _write_ineligible(
                output,
                entry,
                status="plan_ineligible",
                reason=str(plan_checkpoint["reason"]),
                elapsed_seconds=time.perf_counter() - started,
                cache_plan=cache_plan,
                plan_checkpoint=plan_checkpoint,
            )
        plan = PrimeProbePlan.from_dict(plan_checkpoint["plan"])
    else:
        progress("planning", "compiling the independent planning bank")
        physics_pool = HistogramPhysicsPool(case, base_config)
        planning_bank = build_histogram_payoff_bank(
            case,
            base_config,
            samples_per_cell=int(study["samples_per_cell_per_split"]),
            root_seed=_graph_seed(int(study["planning_bank_root_seed"]), graph_id),
            physics=physics_pool,
            progress_callback=lambda message: progress("planning", message),
            selection_digest_blake2b_128=selection_digest,
            evaluation_partition_loaded=True,
            sampling_purpose="prime_probe_105_topology_plan_selection",
            sampling_design=(
                "feedback split selects the declared controller; this bank is "
                "independent of detector commissioning and live execution"
            ),
            route_workers=int(payload.get("route_workers", 1)),
        )
        try:
            plan = build_prime_probe_plan(
                case,
                planning_bank,
                base_config,
                selection_digest_blake2b_128=selection_digest,
            )
        except ValueError as exc:
            if "no declared denial/probe resource" not in str(exc):
                raise
            plan_checkpoint = {
                "schema": PRIME_PROBE_FOLLOWUP_PLAN_SCHEMA,
                "graph_id": graph_id,
                "family": case.family,
                "status": "plan_ineligible",
                "reason": str(exc),
                "resume_contract_digest_blake2b_128": study[
                    "resume_contract_digest_blake2b_128"
                ],
                "planning_bank": planning_bank.to_summary(),
                "physics": physics_pool.diagnostics(),
            }
            _write_json_atomic(plan_path, plan_checkpoint)
            return _write_ineligible(
                output,
                entry,
                status="plan_ineligible",
                reason=str(exc),
                elapsed_seconds=time.perf_counter() - started,
                cache_plan=cache_plan,
                plan_checkpoint=plan_checkpoint,
            )
        plan_checkpoint = {
            "schema": PRIME_PROBE_FOLLOWUP_PLAN_SCHEMA,
            "graph_id": graph_id,
            "family": case.family,
            "status": "complete",
            "reason": None,
            "resume_contract_digest_blake2b_128": study[
                "resume_contract_digest_blake2b_128"
            ],
            "plan": plan.to_dict(),
            "plan_digest_blake2b_128": _stable_digest(plan.to_dict()),
            "planning_bank": planning_bank.to_summary(),
            "physics": physics_pool.diagnostics(),
        }
        _write_json_atomic(plan_path, plan_checkpoint)
        del planning_bank

    progress("physics", "loading or compiling the matched execution bank")
    payoff_path = output / "banks" / f"{graph_id}.payoff.npz"
    endpoint_path = output / "banks" / f"{graph_id}.endpoint.npz"
    if payoff_path.is_file() and endpoint_path.is_file():
        payoff_bank = HistogramPayoffBank.load(payoff_path)
        endpoint_bank = HistogramEndpointEvidenceBank.load(endpoint_path)
        execution_physics = dict(payoff_bank.metadata.get("physics") or {})
    elif payoff_path.exists() or endpoint_path.exists():
        raise ValueError("only one matched execution bank exists")
    else:
        if physics_pool is None:
            physics_pool = HistogramPhysicsPool(case, base_config)
        payoff_bank, endpoint_bank = build_matched_histogram_payoff_and_endpoint_banks(
            case,
            base_config,
            samples_per_cell=int(study["samples_per_cell_per_split"]),
            root_seed=_graph_seed(int(study["execution_bank_root_seed"]), graph_id),
            physics=physics_pool,
            progress_callback=lambda message: progress("physics", message),
            selection_digest_blake2b_128=selection_digest,
            evaluation_partition_loaded=True,
            sampling_purpose="prime_probe_105_topology_live_detection",
            route_workers=int(payload.get("route_workers", 1)),
        )
        _save_bank_atomic(payoff_bank, payoff_path)
        _save_bank_atomic(endpoint_bank, endpoint_path)
        execution_physics = physics_pool.diagnostics()

    schedule = PrimeProbeSchedule(**{
        key: value for key, value in study["schedule"].items()
        if key != "total_turns"
    })
    detector_row = study["detector"]
    detector_config = BinomialEvidenceDetectorConfig(
        horizon=schedule.total_turns,
        model_false_alarm_level=float(detector_row["model_false_alarm_level"]),
        commissioning_pseudocount=float(detector_row["commissioning_pseudocount"]),
        change_fractions=tuple(map(float, detector_row["change_fractions"])),
    )
    graph_seed = int(study["graph_seeds"][graph_id])
    partial_path = output / "partial_graph_checkpoints" / f"{graph_id}.json"
    runs: list[dict[str, Any]] = []
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("resume_contract_digest_blake2b_128") != study[
            "resume_contract_digest_blake2b_128"
        ]:
            raise ValueError("partial ordered checkpoint contract changed")
        runs = [dict(row) for row in partial["runs"]]
    completed = {str(row["defender_label"]) for row in runs}
    for defender_row in study["defenders"]:
        defender = PrimeProbeDefender(
            label=str(defender_row["label"]),
            alice_learner_mode=str(defender_row["alice_learner_mode"]),
            config_overrides=dict(defender_row["config_overrides"]),
        )
        if defender.label in completed:
            continue
        config = replace(base_config, seed=graph_seed, **defender.config_overrides)
        progress("learning", f"{defender.label}, seed {graph_seed}: ordered run")
        run = run_ordered_prime_probe(
            case,
            payoff_bank,
            config,
            plan,
            schedule=schedule,
            seed=graph_seed,
            selection_digest_blake2b_128=selection_digest,
            progress_callback=lambda step, phase, label=defender.label: progress(
                "learning", f"{label}: {step}/{schedule.total_turns} ({phase})"
            ),
            include_audit_scoring=False,
            endpoint_evidence_bank=endpoint_bank,
            execution_split="audit",
            detector_commissioning_draws_per_route=int(
                detector_row["commissioning_draws_per_route"]
            ),
            detector_config=detector_config,
        )
        run.pop("action_indices")
        run.update({
            "defender_label": defender.label,
            "defender": defender.to_dict(),
            "config": config.to_dict(),
            "payoff_bank_content_digest_blake2b_128": payoff_bank.content_digest(),
            "endpoint_bank_content_digest_blake2b_128": endpoint_bank.content_digest(),
            "matched_sampling_id_blake2b_128": payoff_bank.metadata[
                "matched_sampling_id_blake2b_128"
            ],
            "source_plan_digest_blake2b_128": _stable_digest(plan.to_dict()),
            "source_corpus_selection_digest_blake2b_128": selection_digest,
            "resume_contract_digest_blake2b_128": study[
                "resume_contract_digest_blake2b_128"
            ],
            "graph_seed": graph_seed,
        })
        _validate_ordered_detection_run(run, study)
        runs.append(run)
        _write_json_atomic(partial_path, {
            "schema": "histogram_prime_probe_followup_partial.v1",
            "graph_id": graph_id,
            "resume_contract_digest_blake2b_128": study[
                "resume_contract_digest_blake2b_128"
            ],
            "runs": runs,
        })

    checkpoint = {
        "schema": PRIME_PROBE_FOLLOWUP_CHECKPOINT_SCHEMA,
        "graph_id": graph_id,
        "family": case.family,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "reason": None,
        "elapsed_seconds": time.perf_counter() - started,
        "resume_contract_digest_blake2b_128": study[
            "resume_contract_digest_blake2b_128"
        ],
        "case": case.to_dict(),
        "base_config": base_config.to_dict(),
        "cache_plan": cache_plan,
        "plan": plan.to_dict(),
        "plan_checkpoint": plan_checkpoint,
        "payoff_bank": payoff_bank.to_summary(),
        "endpoint_bank": endpoint_bank.to_summary(),
        "execution_physics": execution_physics,
        "runs": runs,
    }
    _validate_checkpoint(checkpoint, entry, study)
    _write_json_atomic(output / "graph_checkpoints" / f"{graph_id}.json", checkpoint)
    progress("complete", f"completed {len(runs)} ordered runs")
    return checkpoint


def _base_config(
    study: Mapping[str, Any], entry: Mapping[str, Any]
) -> HistogramCoLearningConfig:
    source = study["source_corpus_plan"]
    return HistogramCoLearningConfig(
        turns=int(study["schedule"]["total_turns"]),
        requested_pairs_per_turn=int(source["requested_pairs_per_turn"]),
        final_window=min(
            int(source["final_window"]), int(study["schedule"]["total_turns"])
        ),
        record_every=int(study["schedule"]["record_every"]),
        posthoc_samples_per_cell=0,
        seed=int(study["graph_seeds"][str(entry["graph_id"])]),
        game_mode="attack_surface_all",
        age_points_per_edge=int(source["age_points_per_edge"]),
        evaluated_cache_size=max(256, int(entry["eve_action_count"]) + 1),
    )


def _preflight_row(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "graph_id": str(entry["graph_id"]),
        "family": str(entry["family"]),
        "route_count": int(entry["route_count"]),
        "eve_action_count": int(entry["eve_action_count"]),
        "nominal_route_action_cells": int(entry["nominal_route_action_cells"]),
        "canonical_cache_entries": int(entry["canonical_cache_entries"]),
        "estimated_resident_cache_bytes": int(
            round(float(entry["estimated_cache_mib"]) * 1024**2)
        ),
        "configured_lru_capacity_per_route": max(
            256, int(entry["eve_action_count"]) + 1
        ),
        "structurally_eligible": int(entry["route_count"]) >= 2,
    }


def _write_ineligible(
    output: Path,
    entry: Mapping[str, Any],
    *,
    status: str,
    reason: str,
    elapsed_seconds: float,
    **diagnostics: Any,
) -> dict[str, Any]:
    checkpoint = {
        "schema": PRIME_PROBE_FOLLOWUP_CHECKPOINT_SCHEMA,
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
    _write_json_atomic(
        output / "graph_checkpoints" / f"{entry['graph_id']}.json", checkpoint
    )
    return checkpoint


def _summarize(
    study: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for checkpoint in checkpoints:
        status = str(checkpoint["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    trigger_count = sum(bool(row["probe_trigger"]["triggered"]) for row in runs)
    detector_outcomes = _detector_outcome_summaries(runs)
    measurement_counts = detector_outcomes["e91_measurement_counts_only"]
    calibration_ids = set(map(str, CALIBRATION_GRAPH_IDS))
    noncalibration_runs = [
        row for row in runs if str(row["graph_id"]) not in calibration_ids
    ]
    noncalibration_detector_outcomes = _detector_outcome_summaries(
        noncalibration_runs
    )
    noncalibration_measurement_counts = noncalibration_detector_outcomes[
        "e91_measurement_counts_only"
    ]
    return {
        "schema": PRIME_PROBE_FOLLOWUP_SCHEMA,
        "complete": len(checkpoints) == len(study["cases"]),
        "graph_count": len(checkpoints),
        "status_counts": status_counts,
        "eligible_graph_count": status_counts.get("complete", 0),
        "ordered_run_count": len(runs),
        "eve_probe_trigger_count": trigger_count,
        "eve_probe_trigger_total": len(runs),
        "alice_post_prime_alarm_count": int(
            measurement_counts["post_prime_alarm_count"]
        ),
        "alice_pre_prime_false_alarm_count": int(
            measurement_counts["pre_prime_false_alarm_count"]
        ),
        "alice_detector_run_total": int(measurement_counts["run_count"]),
        "alice_detector_outcomes": detector_outcomes,
        "noncalibration_evaluation": {
            "eligible_graph_count": len({
                str(row["graph_id"]) for row in noncalibration_runs
            }),
            "ordered_run_count": len(noncalibration_runs),
            "eve_probe_trigger_count": sum(
                bool(row["probe_trigger"]["triggered"])
                for row in noncalibration_runs
            ),
            "alice_post_prime_alarm_count": int(
                noncalibration_measurement_counts[
                    "post_prime_alarm_count"
                ]
            ),
            "alice_pre_prime_false_alarm_count": int(
                noncalibration_measurement_counts[
                    "pre_prime_false_alarm_count"
                ]
            ),
            "alice_detector_run_total": int(
                noncalibration_measurement_counts["run_count"]
            ),
            "alice_detector_outcomes": noncalibration_detector_outcomes,
        },
        "elapsed_seconds_summed_across_graph_workers": float(
            sum(float(row["elapsed_seconds"]) for row in checkpoints)
        ),
    }


def _detector_outcome_summaries(
    runs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    names = (
        "delivery_only",
        "classical_service",
        "quantum_endpoint",
        "e91_measurement_counts_only",
    )
    return {
        name: _detector_outcome_summary(runs, name)
        for name in names
    }


def _detector_outcome_summary(
    runs: Sequence[Mapping[str, Any]],
    outcome_name: str,
) -> dict[str, Any]:
    outcomes: list[Mapping[str, Any]] = []
    by_defender: dict[str, list[Mapping[str, Any]]] = {}
    for row in runs:
        detector = row.get("alice_endpoint_detector")
        if not isinstance(detector, Mapping):
            continue
        detector_outcomes = detector.get("outcomes")
        if not isinstance(detector_outcomes, Mapping):
            continue
        outcome = detector_outcomes.get(outcome_name)
        if not isinstance(outcome, Mapping):
            continue
        outcomes.append(outcome)
        defender = str(row["defender_label"])
        by_defender.setdefault(defender, []).append(outcome)

    summary = _detector_outcome_rows_summary(outcomes)
    summary["by_defender"] = {
        defender: _detector_outcome_rows_summary(rows)
        for defender, rows in sorted(by_defender.items())
    }
    return summary


def _detector_outcome_rows_summary(
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pre_prime = [
        row for row in outcomes if bool(row.get("pre_prime_false_alarm"))
    ]
    post_prime = [
        row
        for row in outcomes
        if not bool(row.get("pre_prime_false_alarm"))
        and row.get("alarm_turn") is not None
        and not bool(row.get("right_censored"))
    ]
    right_censored = [
        row
        for row in outcomes
        if not bool(row.get("pre_prime_false_alarm"))
        and (
            row.get("alarm_turn") is None
            or bool(row.get("right_censored"))
        )
    ]
    delays = [
        float(row["detection_delay_from_prime_start_turns"])
        for row in post_prime
    ]
    requested_pairs = [
        float(row["requested_pairs_from_prime_start_to_alarm"])
        for row in post_prime
        if row.get("requested_pairs_from_prime_start_to_alarm") is not None
    ]
    return {
        "run_count": len(outcomes),
        "post_prime_alarm_count": len(post_prime),
        "pre_prime_false_alarm_count": len(pre_prime),
        "right_censored_count": len(right_censored),
        "post_prime_delay_turns": _numeric_summary(delays),
        "post_prime_requested_pairs": _numeric_summary(requested_pairs),
    }


def _numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "minimum": float(min(values)),
        "maximum": float(max(values)),
    }


def _validate_manifest(study: Mapping[str, Any]) -> None:
    if study.get("schema") != PRIME_PROBE_FOLLOWUP_SCHEMA:
        raise ValueError("ordered follow-up manifest schema changed")
    if int(study["schedule"]["total_turns"]) != 500_000:
        raise ValueError("ordered follow-up horizon changed")
    if len(study["graph_seeds"]) != len(study["cases"]):
        raise ValueError("ordered follow-up graph seeds are incomplete")
    if study.get("scientific_contract", {}).get("route_universe") != (
        "every simple cycle-free Alice--Bob path"
    ):
        raise ValueError("ordered follow-up route universe changed")
    expected = _stable_digest(_resume_contract(study))
    if study.get("resume_contract_digest_blake2b_128") != expected:
        raise ValueError("ordered follow-up resume contract changed")


def _validate_plan_checkpoint(
    checkpoint: Mapping[str, Any],
    entry: Mapping[str, Any],
    study: Mapping[str, Any],
) -> None:
    if checkpoint.get("schema") != PRIME_PROBE_FOLLOWUP_PLAN_SCHEMA:
        raise ValueError("ordered follow-up plan schema changed")
    if str(checkpoint.get("graph_id")) != str(entry["graph_id"]):
        raise ValueError("ordered follow-up plan graph changed")
    if checkpoint.get("resume_contract_digest_blake2b_128") != study[
        "resume_contract_digest_blake2b_128"
    ]:
        raise ValueError("ordered follow-up plan contract changed")
    if checkpoint["status"] == "complete":
        plan = PrimeProbePlan.from_dict(checkpoint["plan"])
        if _stable_digest(plan.to_dict()) != checkpoint["plan_digest_blake2b_128"]:
            raise ValueError("ordered follow-up plan content changed")


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    entry: Mapping[str, Any],
    study: Mapping[str, Any],
) -> None:
    if checkpoint.get("schema") != PRIME_PROBE_FOLLOWUP_CHECKPOINT_SCHEMA:
        raise ValueError("ordered follow-up checkpoint schema changed")
    if str(checkpoint.get("graph_id")) != str(entry["graph_id"]):
        raise ValueError("ordered follow-up checkpoint graph changed")
    runs = list(checkpoint.get("runs") or [])
    if checkpoint["status"] != "complete":
        if runs:
            raise ValueError("ineligible ordered checkpoint contains runs")
        return
    expected_labels = {str(row["label"]) for row in study["defenders"]}
    if {str(row["defender_label"]) for row in runs} != expected_labels:
        raise ValueError("ordered follow-up defender identities changed")
    if len(runs) != len(expected_labels):
        raise ValueError("ordered follow-up checkpoint is incomplete")
    for run in runs:
        if int(run["seed"]) != int(study["graph_seeds"][str(entry["graph_id"])]):
            raise ValueError("ordered follow-up graph seed changed")
        _validate_ordered_detection_run(run, study)


def _resume_contract(study: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: study[key]
        for key in (
            "schema",
            "source_corpus_selection_digest_blake2b_128",
            "source_corpus_plan",
            "cases",
            "defenders",
            "graph_seeds",
            "schedule",
            "samples_per_cell_per_split",
            "planning_bank_root_seed",
            "execution_bank_root_seed",
            "detector",
        )
    }


def _check_existing_manifest(output: Path, study: Mapping[str, Any]) -> None:
    path = output / "run_manifest.json"
    if not path.is_file():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(existing)
    if existing["resume_contract_digest_blake2b_128"] != study[
        "resume_contract_digest_blake2b_128"
    ]:
        raise ValueError("output directory belongs to another ordered study")


def _graph_seed(root_seed: int, graph_id: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(root_seed).to_bytes(8, "little", signed=True))
    digest.update(b"\0")
    digest.update(graph_id.encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") % (2**31 - 1)


def _stable_digest(payload: Any) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _save_bank_atomic(bank: Any, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    bank.save(temporary)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(
    path: Path, rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
