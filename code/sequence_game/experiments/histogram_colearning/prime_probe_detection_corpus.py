"""Five-topology Prime-and-Probe execution with Alice endpoint detection.

The study reuses the exact cases, learners, seeds, schedules, and physical
intervention plans frozen by ``histogram_prime_probe_500k``.  It does not
re-plan Eve on the newly sampled physics bank.  Each live turn instead reads
Alice's reward, Alice's endpoint E91 counts, and Eve's local Probe count from
one matched audit-split transcript.  The feedback split is reserved for
commissioning Alice's route-conditioned detector on no-attack observations.
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
from typing import Any, Callable, Mapping

import numpy as np

from .calibration_bank import (
    HistogramPayoffBank,
    validate_bank_against_case,
)
from .config import HistogramCoLearningConfig, REPO_ROOT
from .corpus_scale import load_corpus_scale_case
from .endpoint_evidence_bank import (
    HistogramEndpointEvidenceBank,
    validate_endpoint_evidence_bank_against_case,
)
from .matched_payoff_endpoint_bank import (
    MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA,
    build_matched_histogram_payoff_and_endpoint_banks,
)
from .physics import HistogramPhysicsPool, histogram_cache_plan
from .prime_probe import (
    PRIME_PROBE_PLAN_SCHEMA,
    PRIME_PROBE_RESULT_SCHEMA,
    PrimeProbePlan,
    PrimeProbeSchedule,
    paired_prime_probe_comparison,
    run_ordered_prime_probe,
    run_shuffled_prime_probe_control,
)
from .prime_probe_corpus import (
    PRIME_PROBE_GRAPH_CHECKPOINT_SCHEMA,
    PRIME_PROBE_STUDY_SCHEMA,
    PrimeProbeDefender,
)
from .quantum_stackelberg_detection import BinomialEvidenceDetectorConfig
from .topology_census import available_ram_snapshot


PRIME_PROBE_DETECTION_CORPUS_SCHEMA = (
    "histogram_prime_probe_detection_corpus.v1"
)
PRIME_PROBE_DETECTION_GRAPH_CHECKPOINT_SCHEMA = (
    "histogram_prime_probe_detection_graph_checkpoint.v1"
)
PRIME_PROBE_DETECTION_PARTIAL_CHECKPOINT_SCHEMA = (
    "histogram_prime_probe_detection_partial_checkpoint.v1"
)
PRIME_PROBE_DETECTION_PREFLIGHT_SCHEMA = (
    "histogram_prime_probe_detection_preflight.v1"
)
PRIME_PROBE_DETECTION_IMPLEMENTATION_REVISION = (
    "matched_audit_execution.feedback_commissioned_monitor_only.v3"
)
DEFAULT_PRIME_PROBE_DETECTION_RUN_DIR = (
    REPO_ROOT / "runs" / "histogram_prime_probe_detection_500k"
)
DEFAULT_PRIME_PROBE_DETECTION_SMOKE_RUN_DIR = (
    REPO_ROOT / "runs" / "histogram_prime_probe_detection_smoke"
)
DEFAULT_SOURCE_PRIME_PROBE_RUN_DIR = (
    REPO_ROOT / "runs" / "histogram_prime_probe_500k"
)
DEFAULT_CORPUS_RUN_DIR = REPO_ROOT / "runs" / "histogram_corpus_500k"
DEFAULT_MATCHED_SAMPLES_PER_CELL_PER_SPLIT = 64
DEFAULT_DETECTOR_COMMISSIONING_DRAWS_PER_ROUTE = 32
DEFAULT_MATCHED_BANK_ROOT_SEED = 20_260_728_01
DEFAULT_DETECTOR_FALSE_ALARM_LEVEL = 0.01
DEFAULT_DETECTOR_PSEUDOCOUNT = 0.5
DEFAULT_DETECTOR_CHANGE_FRACTIONS = (0.10, 0.25, 0.50)


ProgressCallback = Callable[[str], None]


def load_source_prime_probe_bundle(
    source_run_dir: str | Path = DEFAULT_SOURCE_PRIME_PROBE_RUN_DIR,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load and verify the completed source manifest and graph checkpoints."""

    source = Path(source_run_dir)
    manifest_path = source / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"source Prime-and-Probe manifest is missing: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != PRIME_PROBE_STUDY_SCHEMA:
        raise ValueError("source Prime-and-Probe manifest schema changed")
    checkpoints: dict[str, dict[str, Any]] = {}
    for path in sorted((source / "graph_checkpoints").glob("*.json")):
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        if checkpoint.get("schema") != PRIME_PROBE_GRAPH_CHECKPOINT_SCHEMA:
            raise ValueError(f"source checkpoint schema changed: {path}")
        graph_id = str(checkpoint["graph_id"])
        if graph_id in checkpoints:
            raise ValueError(f"duplicate source checkpoint for {graph_id}")
        checkpoints[graph_id] = checkpoint
    expected_graph_ids = {
        str(entry["graph_id"]) for entry in manifest["cases"]
    }
    if set(checkpoints) != expected_graph_ids:
        raise ValueError(
            "source Prime-and-Probe checkpoints do not match the manifest"
        )
    for entry in manifest["cases"]:
        _validate_source_checkpoint(
            checkpoints[str(entry["graph_id"])],
            entry,
        )
    return manifest, checkpoints


def build_prime_probe_detection_manifest(
    source_manifest: Mapping[str, Any],
    source_checkpoints: Mapping[str, Mapping[str, Any]],
    *,
    samples_per_cell_per_split: int = (
        DEFAULT_MATCHED_SAMPLES_PER_CELL_PER_SPLIT
    ),
    smoke: bool = False,
) -> dict[str, Any]:
    """Freeze the matched-transcript detection rerun.

    Production mode copies the source cases, learners, seeds, and 500k schedule
    exactly.  Smoke mode uses the smallest source case, the first learner and
    seed, and a short schedule while retaining that case's exact route/action
    universe and frozen physical intervention plan.
    """

    if source_manifest.get("schema") != PRIME_PROBE_STUDY_SCHEMA:
        raise ValueError("source Prime-and-Probe manifest schema changed")
    if samples_per_cell_per_split < (
        DEFAULT_DETECTOR_COMMISSIONING_DRAWS_PER_ROUTE
    ):
        raise ValueError(
            "matched samples per cell must cover all detector commissioning "
            "draws"
        )
    cases = [dict(row) for row in source_manifest["cases"]]
    defenders = [dict(row) for row in source_manifest["defenders"]]
    seeds = list(map(int, source_manifest["seeds"]))
    schedule = dict(source_manifest["schedule"])
    if smoke:
        cases = [
            min(
                cases,
                key=lambda row: (
                    int(row["route_count"])
                    * int(row["eve_action_count"]),
                    str(row["graph_id"]),
                ),
            )
        ]
        defenders = defenders[:1]
        seeds = seeds[:1]
        schedule = PrimeProbeSchedule(
            warmup_turns=32,
            prime_turns=32,
            probe_turns=8,
            exploit_turns=16,
            recovery_turns=32,
            record_every=8,
            rolling_window_turns=8,
            baseline_tail_turns=16,
            sustained_recovery_checkpoints=2,
        ).to_dict()

    frozen_plans: dict[str, dict[str, Any]] = {}
    for entry in cases:
        graph_id = str(entry["graph_id"])
        try:
            source_checkpoint = source_checkpoints[graph_id]
        except KeyError as exc:
            raise ValueError(
                f"source checkpoint missing for {graph_id}"
            ) from exc
        _validate_source_checkpoint(source_checkpoint, entry)
        plan = dict(source_checkpoint["plan"])
        frozen_plans[graph_id] = {
            "plan": plan,
            "plan_digest_blake2b_128": _stable_digest(plan),
            "source_checkpoint_schema": str(source_checkpoint["schema"]),
            "source_checkpoint_created_at_utc": source_checkpoint.get(
                "created_at_utc"
            ),
            "source_bank_content_digest_blake2b_128": str(
                source_checkpoint["bank"][
                    "content_digest_blake2b_128"
                ]
            ),
            "source_cache_plan": _cache_plan_summary(
                source_checkpoint["cache_plan"]
            ),
        }

    detector = {
        "commissioning_split": "feedback",
        "commissioning_action_id": "no_attack",
        "commissioning_draws_per_route": (
            DEFAULT_DETECTOR_COMMISSIONING_DRAWS_PER_ROUTE
        ),
        "execution_split": "audit",
        "model_false_alarm_level": (
            DEFAULT_DETECTOR_FALSE_ALARM_LEVEL
        ),
        "commissioning_pseudocount": DEFAULT_DETECTOR_PSEUDOCOUNT,
        "change_fractions": list(DEFAULT_DETECTOR_CHANGE_FRACTIONS),
        "horizon": int(schedule["total_turns"]),
        "response": "monitor_only_native_learning",
    }
    manifest: dict[str, Any] = {
        "schema": PRIME_PROBE_DETECTION_CORPUS_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke": bool(smoke),
        "source_prime_probe_schema": str(source_manifest["schema"]),
        "source_prime_probe_manifest_digest_blake2b_128": _stable_digest(
            source_manifest
        ),
        "source_corpus_selection_digest_blake2b_128": str(
            source_manifest[
                "source_corpus_selection_digest_blake2b_128"
            ]
        ),
        "source_corpus_plan": dict(
            source_manifest["source_corpus_plan"]
        ),
        "cases": cases,
        "families": [str(row["family"]) for row in cases],
        "defenders": defenders,
        "seeds": seeds,
        "schedule": schedule,
        "frozen_plans": frozen_plans,
        "samples_per_cell_per_split": int(
            samples_per_cell_per_split
        ),
        "matched_bank_root_seed": DEFAULT_MATCHED_BANK_ROOT_SEED,
        "matched_bank_schema": MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA,
        "detector": detector,
        "expected_condition_runs": (
            len(cases) * len(defenders) * len(seeds) * 2
        ),
        "expected_paired_comparisons": (
            len(cases) * len(defenders) * len(seeds)
        ),
        "expected_detector_runs": (
            len(cases) * len(defenders) * len(seeds)
        ),
        "scientific_contract": {
            "source_plan": (
                "exact physical intervention plan copied from the completed "
                "source checkpoint; not reoptimized on the new banks"
            ),
            "live_turn_sampling": (
                "one audit-split compiled transcript supplies Alice reward, "
                "Alice endpoint E91 counts, and Eve local Probe count"
            ),
            "detector_commissioning": (
                "first 32 feedback-split no-attack draws on every route"
            ),
            "detector_active_from_turn": 1,
            "detector_response": "monitor only; learner state is unchanged",
            "posthoc_audit_scoring": False,
            "reward_poisoning": False,
            "alice_detector_excludes": [
                "Eve action identity",
                "Eve target identity",
                "Eve local Probe count",
                "target-active simulator diagnostic",
                "expected payoff matrix",
            ],
        },
    }
    resume_contract = _resume_contract_payload(manifest)
    manifest["resume_contract"] = resume_contract
    manifest["resume_contract_digest_blake2b_128"] = _stable_digest(
        resume_contract
    )
    _validate_study_manifest(manifest)
    return manifest


def prime_probe_detection_preflight(
    study_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Report exact action geometry, cache plan, run count, and available RAM."""

    _validate_study_manifest(study_manifest)
    rows = []
    for entry in study_manifest["cases"]:
        graph_id = str(entry["graph_id"])
        cache = study_manifest["frozen_plans"][graph_id][
            "source_cache_plan"
        ]
        rows.append({
            "graph_id": graph_id,
            "family": str(entry["family"]),
            "route_count": int(entry["route_count"]),
            "eve_action_count": int(entry["eve_action_count"]),
            "nominal_route_action_cells": (
                int(entry["route_count"])
                * int(entry["eve_action_count"])
            ),
            "canonical_cache_entries": int(
                cache["canonical_cache_entries"]
            ),
            "estimated_resident_cache_bytes": int(
                cache["estimated_resident_cache_bytes"]
            ),
            "configured_lru_capacity_per_route": int(
                cache["configured_lru_capacity_per_route"]
            ),
            "minimum_lru_capacity_per_route": int(
                cache["minimum_lru_capacity_per_route"]
            ),
        })
    return {
        "schema": PRIME_PROBE_DETECTION_PREFLIGHT_SCHEMA,
        "smoke": bool(study_manifest["smoke"]),
        "case_count": len(rows),
        "cases": rows,
        "route_count_total": sum(row["route_count"] for row in rows),
        "eve_action_count_total": sum(
            row["eve_action_count"] for row in rows
        ),
        "nominal_route_action_cells_total": sum(
            row["nominal_route_action_cells"] for row in rows
        ),
        "canonical_cache_entries_total": sum(
            row["canonical_cache_entries"] for row in rows
        ),
        "estimated_resident_cache_bytes_if_concurrent": sum(
            row["estimated_resident_cache_bytes"] for row in rows
        ),
        "samples_per_cell_per_split": int(
            study_manifest["samples_per_cell_per_split"]
        ),
        "matched_compiled_transcript_draws_total": (
            2
            * int(study_manifest["samples_per_cell_per_split"])
            * sum(row["nominal_route_action_cells"] for row in rows)
        ),
        "expected_condition_runs": int(
            study_manifest["expected_condition_runs"]
        ),
        "expected_detector_runs": int(
            study_manifest["expected_detector_runs"]
        ),
        "turns_per_condition": int(
            study_manifest["schedule"]["total_turns"]
        ),
        "available_ram_before_run": available_ram_snapshot(),
    }


def run_prime_probe_detection_study(
    corpus_manifest: Mapping[str, Any],
    study_manifest: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_PRIME_PROBE_DETECTION_RUN_DIR,
    graph_workers: int = 5,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Execute resumable graph-owned matched-transcript detector runs."""

    _validate_study_manifest(study_manifest)
    if graph_workers < 1:
        raise ValueError("graph_workers must be positive")
    output = Path(output_dir)
    checkpoints_dir = output / "graph_checkpoints"
    partial_checkpoints_dir = output / "partial_graph_checkpoints"
    progress_dir = output / "progress"
    banks_dir = output / "banks"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    partial_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    banks_dir.mkdir(parents=True, exist_ok=True)
    _check_existing_manifest(output, study_manifest)
    _write_json_atomic(output / "run_manifest.json", study_manifest)
    _write_json_atomic(
        output / "preflight.json",
        prime_probe_detection_preflight(study_manifest),
    )

    case_by_graph = {
        str(row["graph_id"]): dict(row)
        for row in study_manifest["cases"]
    }
    complete: dict[str, dict[str, Any]] = {}
    for path in sorted(checkpoints_dir.glob("*.json")):
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        graph_id = str(checkpoint.get("graph_id") or path.stem)
        if graph_id not in case_by_graph:
            raise ValueError(f"checkpoint {path} is outside the study")
        _validate_detection_checkpoint(
            checkpoint,
            case_by_graph[graph_id],
            study_manifest,
        )
        complete[graph_id] = checkpoint
    pending = [
        dict(row)
        for row in study_manifest["cases"]
        if str(row["graph_id"]) not in complete
    ]
    payloads = [
        {
            "entry": entry,
            "corpus_manifest": dict(corpus_manifest),
            "study_manifest": dict(study_manifest),
            "output_dir": str(output),
        }
        for entry in pending
    ]
    if graph_workers == 1:
        for payload in payloads:
            checkpoint = _run_detection_graph(payload)
            complete[str(checkpoint["graph_id"])] = checkpoint
            if progress_callback is not None:
                progress_callback(
                    "Prime-and-Probe detection graph complete "
                    f"{len(complete)}/{len(case_by_graph)}: "
                    f"{checkpoint['family']}"
                )
    elif payloads:
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError(
                "Prime-and-Probe detection execution requires fork"
            )
        context = mp.get_context("fork")
        workers = min(int(graph_workers), len(payloads))
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            future_map = {
                executor.submit(_run_detection_graph, payload): payload[
                    "entry"
                ]
                for payload in payloads
            }
            for future in as_completed(future_map):
                checkpoint = future.result()
                complete[str(checkpoint["graph_id"])] = checkpoint
                if progress_callback is not None:
                    progress_callback(
                        "Prime-and-Probe detection graph complete "
                        f"{len(complete)}/{len(case_by_graph)}: "
                        f"{checkpoint['family']}"
                    )

    ordered_checkpoints = [
        complete[str(row["graph_id"])]
        for row in study_manifest["cases"]
    ]
    runs = [
        row
        for checkpoint in ordered_checkpoints
        for row in checkpoint["runs"]
    ]
    comparisons = [
        row
        for checkpoint in ordered_checkpoints
        for row in checkpoint["paired_comparisons"]
    ]
    if len(runs) != int(study_manifest["expected_condition_runs"]):
        raise AssertionError("detection study condition runs are incomplete")
    if len(comparisons) != int(
        study_manifest["expected_paired_comparisons"]
    ):
        raise AssertionError("detection study comparisons are incomplete")
    _write_jsonl_atomic(output / "results.jsonl", runs)
    _write_jsonl_atomic(
        output / "paired_comparisons.jsonl",
        comparisons,
    )
    summary = {
        "schema": PRIME_PROBE_DETECTION_CORPUS_SCHEMA,
        "complete": True,
        "smoke": bool(study_manifest["smoke"]),
        "graph_count": len(ordered_checkpoints),
        "condition_run_count": len(runs),
        "expected_condition_run_count": int(
            study_manifest["expected_condition_runs"]
        ),
        "paired_comparison_count": len(comparisons),
        "expected_paired_comparison_count": int(
            study_manifest["expected_paired_comparisons"]
        ),
        "detector_run_count": sum(
            row["condition"] == "ordered_prime_probe"
            for row in runs
        ),
        "expected_detector_run_count": int(
            study_manifest["expected_detector_runs"]
        ),
        "elapsed_seconds_summed_across_graph_workers": float(
            sum(
                float(checkpoint["elapsed_seconds"])
                for checkpoint in ordered_checkpoints
            )
        ),
        "detector_descriptive_summary": (
            _detector_descriptive_summary(runs)
        ),
    }
    _write_json_atomic(output / "summary.json", summary)
    return summary


def _run_detection_graph(payload: Mapping[str, Any]) -> dict[str, Any]:
    entry = dict(payload["entry"])
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

    progress("case", "loading the frozen cycle-free route universe")
    case = load_corpus_scale_case(entry, corpus_manifest)
    _validate_loaded_case(case, entry)
    base_config = _base_config_for_entry(study, entry)
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

    payoff_path = output / "banks" / f"{graph_id}.payoff.npz"
    endpoint_path = output / "banks" / f"{graph_id}.endpoint.npz"
    payoff_bank, endpoint_bank, physics_diagnostics = (
        _load_or_build_matched_banks(
            case,
            base_config,
            study,
            payoff_path=payoff_path,
            endpoint_path=endpoint_path,
            progress=progress,
        )
    )
    frozen = study["frozen_plans"][graph_id]
    plan = PrimeProbePlan.from_dict(frozen["plan"])
    _validate_frozen_plan(case, payoff_bank, plan, frozen)
    schedule = PrimeProbeSchedule(**{
        key: value
        for key, value in study["schedule"].items()
        if key != "total_turns"
    })
    detector_row = study["detector"]
    detector_config = BinomialEvidenceDetectorConfig(
        horizon=schedule.total_turns,
        model_false_alarm_level=float(
            detector_row["model_false_alarm_level"]
        ),
        commissioning_pseudocount=float(
            detector_row["commissioning_pseudocount"]
        ),
        change_fractions=tuple(
            map(float, detector_row["change_fractions"])
        ),
    )
    runs: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    partial_path = (
        output / "partial_graph_checkpoints" / f"{graph_id}.json"
    )
    payoff_digest = payoff_bank.content_digest()
    endpoint_digest = endpoint_bank.content_digest()
    matched_sampling_id = str(
        payoff_bank.metadata["matched_sampling_id_blake2b_128"]
    )
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        _validate_detection_partial_checkpoint(
            partial,
            entry,
            study,
            payoff_bank_content_digest=payoff_digest,
            endpoint_bank_content_digest=endpoint_digest,
            matched_sampling_id=matched_sampling_id,
        )
        runs = [dict(row) for row in partial["runs"]]
        comparisons = [
            dict(row) for row in partial["paired_comparisons"]
        ]
    completed_pairs = _completed_pair_keys(runs, comparisons)
    for defender_row in study["defenders"]:
        defender = PrimeProbeDefender(
            label=str(defender_row["label"]),
            alice_learner_mode=str(
                defender_row["alice_learner_mode"]
            ),
            config_overrides=dict(defender_row["config_overrides"]),
        )
        for seed in map(int, study["seeds"]):
            pair_key = (defender.label, seed)
            if pair_key in completed_pairs:
                progress(
                    "learning",
                    f"{defender.label}, seed {seed}: restored completed pair",
                )
                continue
            config = replace(
                base_config,
                seed=seed,
                **defender.config_overrides,
            )
            progress(
                "learning",
                f"{defender.label}, seed {seed}: ordered detector run",
            )
            ordered = run_ordered_prime_probe(
                case,
                payoff_bank,
                config,
                plan,
                schedule=schedule,
                seed=seed,
                selection_digest_blake2b_128=study[
                    "source_corpus_selection_digest_blake2b_128"
                ],
                progress_callback=(
                    lambda step, phase, label=defender.label,
                    active_seed=seed: progress(
                        "learning",
                        f"{label}, seed {active_seed}: ordered "
                        f"{step}/{schedule.total_turns} ({phase})",
                    )
                ),
                include_audit_scoring=False,
                endpoint_evidence_bank=endpoint_bank,
                execution_split="audit",
                detector_commissioning_draws_per_route=int(
                    detector_row["commissioning_draws_per_route"]
                ),
                detector_config=detector_config,
            )
            action_indices = ordered.pop("action_indices")
            _annotate_run(
                ordered,
                defender=defender,
                config=config,
                study=study,
                payoff_bank=payoff_bank,
                endpoint_bank=endpoint_bank,
                frozen=frozen,
            )
            _validate_ordered_detection_run(ordered, study)

            progress(
                "learning",
                f"{defender.label}, seed {seed}: shuffled control",
            )
            shuffled = run_shuffled_prime_probe_control(
                case,
                payoff_bank,
                config,
                plan,
                schedule=schedule,
                seed=seed,
                ordered_action_indices=action_indices,
                selection_digest_blake2b_128=study[
                    "source_corpus_selection_digest_blake2b_128"
                ],
                progress_callback=(
                    lambda step, phase, label=defender.label,
                    active_seed=seed: progress(
                        "learning",
                        f"{label}, seed {active_seed}: shuffled "
                        f"{step}/{schedule.total_turns} ({phase})",
                    )
                ),
                include_audit_scoring=False,
                endpoint_evidence_bank=endpoint_bank,
                execution_split="audit",
            )
            shuffled.pop("action_indices")
            _annotate_run(
                shuffled,
                defender=defender,
                config=config,
                study=study,
                payoff_bank=payoff_bank,
                endpoint_bank=endpoint_bank,
                frozen=frozen,
            )
            _validate_shuffled_run(shuffled, study)
            comparison = paired_prime_probe_comparison(
                ordered,
                shuffled,
            )
            comparison["defender_label"] = defender.label
            comparison["source_plan_digest_blake2b_128"] = frozen[
                "plan_digest_blake2b_128"
            ]
            runs.extend((ordered, shuffled))
            comparisons.append(comparison)
            partial = _detection_partial_checkpoint_payload(
                graph_id=graph_id,
                family=case.family,
                study=study,
                frozen=frozen,
                payoff_bank_content_digest=payoff_digest,
                endpoint_bank_content_digest=endpoint_digest,
                matched_sampling_id=matched_sampling_id,
                runs=runs,
                comparisons=comparisons,
            )
            _validate_detection_partial_checkpoint(
                partial,
                entry,
                study,
                payoff_bank_content_digest=payoff_digest,
                endpoint_bank_content_digest=endpoint_digest,
                matched_sampling_id=matched_sampling_id,
            )
            _write_json_atomic(partial_path, partial)
            completed_pairs.add(pair_key)
            progress(
                "learning",
                f"{defender.label}, seed {seed}: pair checkpointed",
            )

    checkpoint = {
        "schema": PRIME_PROBE_DETECTION_GRAPH_CHECKPOINT_SCHEMA,
        "study_schema": PRIME_PROBE_DETECTION_CORPUS_SCHEMA,
        "graph_id": graph_id,
        "family": case.family,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "resume_contract_digest_blake2b_128": study[
            "resume_contract_digest_blake2b_128"
        ],
        "source_plan_digest_blake2b_128": frozen[
            "plan_digest_blake2b_128"
        ],
        "case": case.to_dict(),
        "base_config": base_config.to_dict(),
        "cache_plan": cache_plan,
        "payoff_bank": payoff_bank.to_summary(),
        "endpoint_bank": endpoint_bank.to_summary(),
        "bank_paths": {
            "payoff": str(payoff_path.relative_to(output)),
            "endpoint": str(endpoint_path.relative_to(output)),
        },
        "matched_sampling_id_blake2b_128": payoff_bank.metadata[
            "matched_sampling_id_blake2b_128"
        ],
        "plan": plan.to_dict(),
        "source_plan": dict(frozen),
        "physics": physics_diagnostics,
        "runs": runs,
        "paired_comparisons": comparisons,
    }
    _validate_detection_checkpoint(checkpoint, entry, study)
    _write_json_atomic(
        output / "graph_checkpoints" / f"{graph_id}.json",
        checkpoint,
    )
    progress("complete", f"completed {len(runs)} condition runs")
    return checkpoint


def _load_or_build_matched_banks(
    case: Any,
    config: HistogramCoLearningConfig,
    study: Mapping[str, Any],
    *,
    payoff_path: Path,
    endpoint_path: Path,
    progress: Callable[[str, str], None],
) -> tuple[
    HistogramPayoffBank,
    HistogramEndpointEvidenceBank,
    dict[str, Any],
]:
    if payoff_path.is_file() and endpoint_path.is_file():
        progress("physics", "loading verified matched payoff/endpoint banks")
        payoff = HistogramPayoffBank.load(payoff_path)
        endpoint = HistogramEndpointEvidenceBank.load(endpoint_path)
        diagnostics = dict(payoff.metadata.get("physics") or {})
    elif payoff_path.exists() or endpoint_path.exists():
        raise ValueError(
            "only one matched bank exists; refusing an ambiguous partial resume"
        )
    else:
        progress(
            "physics",
            "compiling one transcript per matched payoff/endpoint coordinate",
        )
        physics = HistogramPhysicsPool(case, config)
        payoff, endpoint = (
            build_matched_histogram_payoff_and_endpoint_banks(
                case,
                config,
                samples_per_cell=int(
                    study["samples_per_cell_per_split"]
                ),
                root_seed=int(study["matched_bank_root_seed"]),
                physics=physics,
                progress_callback=lambda message: progress(
                    "physics", message
                ),
                selection_digest_blake2b_128=str(
                    study[
                        "source_corpus_selection_digest_blake2b_128"
                    ]
                ),
                evaluation_partition_loaded=True,
                sampling_purpose=(
                    "prime_probe_alice_endpoint_detection_rerun"
                ),
            )
        )
        _save_bank_atomic(payoff, payoff_path)
        _save_bank_atomic(endpoint, endpoint_path)
        diagnostics = physics.diagnostics()
    _validate_matched_banks(case, config, study, payoff, endpoint)
    return payoff, endpoint, diagnostics


def _validate_matched_banks(
    case: Any,
    config: HistogramCoLearningConfig,
    study: Mapping[str, Any],
    payoff: HistogramPayoffBank,
    endpoint: HistogramEndpointEvidenceBank,
) -> None:
    selection_digest = str(
        study["source_corpus_selection_digest_blake2b_128"]
    )
    validate_bank_against_case(
        payoff,
        case,
        config,
        selection_digest_blake2b_128=selection_digest,
    )
    validate_endpoint_evidence_bank_against_case(
        endpoint,
        case,
        config,
        selection_digest_blake2b_128=selection_digest,
    )
    if (
        payoff.metadata.get("matched_sampling_schema")
        != MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA
        or endpoint.metadata.get("matched_sampling_schema")
        != MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA
    ):
        raise ValueError("banks were not produced by the matched builder")
    payoff_sampling_id = str(
        payoff.metadata.get("matched_sampling_id_blake2b_128")
    )
    if payoff_sampling_id != str(
        endpoint.metadata.get("matched_sampling_id_blake2b_128")
    ):
        raise ValueError("payoff and endpoint banks have different samples")
    if payoff.draws_per_cell != int(
        study["samples_per_cell_per_split"]
    ):
        raise ValueError("matched bank draw count changed")
    if (
        payoff.route_ids != endpoint.route_ids
        or payoff.action_ids != endpoint.action_ids
    ):
        raise ValueError("matched bank route/action axes changed")
    for split in ("feedback", "audit"):
        payoff_block = getattr(payoff, split)
        endpoint_block = getattr(endpoint, split)
        for field in (
            "alice_reward",
            "bell_qualified_rate_hz",
            "public_outcome_code",
        ):
            if not np.array_equal(
                np.asarray(getattr(payoff_block, field)),
                np.asarray(getattr(endpoint_block, field)),
            ):
                raise ValueError(
                    f"matched bank field {field!r} differs in {split}"
                )


def _base_config_for_entry(
    study: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> HistogramCoLearningConfig:
    source = study["source_corpus_plan"]
    return HistogramCoLearningConfig(
        turns=int(study["schedule"]["total_turns"]),
        requested_pairs_per_turn=int(
            source["requested_pairs_per_turn"]
        ),
        final_window=min(
            int(source["final_window"]),
            int(study["schedule"]["total_turns"]),
        ),
        record_every=int(study["schedule"]["record_every"]),
        posthoc_samples_per_cell=0,
        seed=int(study["seeds"][0]),
        game_mode="attack_surface_all",
        age_points_per_edge=int(source["age_points_per_edge"]),
        evaluated_cache_size=max(
            256,
            int(entry["eve_action_count"]) + 1,
        ),
    )


def _annotate_run(
    run: dict[str, Any],
    *,
    defender: PrimeProbeDefender,
    config: HistogramCoLearningConfig,
    study: Mapping[str, Any],
    payoff_bank: HistogramPayoffBank,
    endpoint_bank: HistogramEndpointEvidenceBank,
    frozen: Mapping[str, Any],
) -> None:
    run.update({
        "defender_label": defender.label,
        "defender": defender.to_dict(),
        "config": config.to_dict(),
        "payoff_bank_content_digest_blake2b_128": (
            payoff_bank.content_digest()
        ),
        "endpoint_bank_content_digest_blake2b_128": (
            endpoint_bank.content_digest()
        ),
        "matched_sampling_id_blake2b_128": payoff_bank.metadata[
            "matched_sampling_id_blake2b_128"
        ],
        "source_plan_digest_blake2b_128": frozen[
            "plan_digest_blake2b_128"
        ],
        "source_corpus_selection_digest_blake2b_128": study[
            "source_corpus_selection_digest_blake2b_128"
        ],
        "resume_contract_digest_blake2b_128": study[
            "resume_contract_digest_blake2b_128"
        ],
    })


def _validate_ordered_detection_run(
    run: Mapping[str, Any],
    study: Mapping[str, Any],
) -> None:
    if run.get("schema") != PRIME_PROBE_RESULT_SCHEMA:
        raise ValueError("ordered result schema changed")
    if run.get("condition") != "ordered_prime_probe":
        raise ValueError("ordered result condition changed")
    if run.get("execution_split") != "audit":
        raise ValueError("ordered result did not execute on the audit split")
    if run.get("audit_expected_alice_reward") is not None:
        raise ValueError("live audit execution was reused for scoring")
    detector = run.get("alice_endpoint_detector")
    if not isinstance(detector, Mapping) or not detector.get("enabled"):
        raise ValueError("ordered result is missing Alice's detector")
    expected = study["detector"]
    if (
        detector.get("response") != "monitor_only_native_learning"
        or detector.get("execution_split") != "audit"
        or detector.get("commissioning_split") != "feedback"
        or int(detector.get("commissioning_draws_per_route", -1))
        != int(expected["commissioning_draws_per_route"])
    ):
        raise ValueError("ordered detector boundary changed")
    boundary = run.get("information_boundary") or {}
    if (
        boundary.get("alice_detector_changes_learner_state") is not False
        or boundary.get("alice_detector_observes_eve_local_record")
        is not False
        or boundary.get("alice_detector_observes_eve_action_identity")
        is not False
    ):
        raise ValueError("Alice detector information boundary changed")
    outcomes = detector.get("outcomes") or {}
    if set(outcomes) != {
        "delivery_only",
        "classical_service",
        "quantum_endpoint",
        "e91_measurement_counts_only",
    }:
        raise ValueError("ordered result has incomplete detector outcomes")
    for outcome in outcomes.values():
        if "pre_prime_false_alarm" not in outcome:
            raise ValueError("detector outcome omits pre-Prime false alarms")


def _validate_shuffled_run(
    run: Mapping[str, Any],
    study: Mapping[str, Any],
) -> None:
    if run.get("schema") != PRIME_PROBE_RESULT_SCHEMA:
        raise ValueError("shuffled result schema changed")
    if run.get("condition") != "shuffled_identical_action_multiset":
        raise ValueError("shuffled result condition changed")
    if run.get("execution_split") != "audit":
        raise ValueError("shuffled result did not execute on the audit split")
    if run.get("audit_expected_alice_reward") is not None:
        raise ValueError("live audit execution was reused for scoring")
    if run.get("alice_endpoint_detector") is not None:
        raise ValueError("shuffled control unexpectedly ran a detector")
    if int(run["turns"]) != int(study["schedule"]["total_turns"]):
        raise ValueError("shuffled result horizon changed")


def _detection_partial_checkpoint_payload(
    *,
    graph_id: str,
    family: str,
    study: Mapping[str, Any],
    frozen: Mapping[str, Any],
    payoff_bank_content_digest: str,
    endpoint_bank_content_digest: str,
    matched_sampling_id: str,
    runs: list[Mapping[str, Any]],
    comparisons: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": PRIME_PROBE_DETECTION_PARTIAL_CHECKPOINT_SCHEMA,
        "study_schema": PRIME_PROBE_DETECTION_CORPUS_SCHEMA,
        "graph_id": str(graph_id),
        "family": str(family),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "resume_contract_digest_blake2b_128": study[
            "resume_contract_digest_blake2b_128"
        ],
        "source_plan_digest_blake2b_128": frozen[
            "plan_digest_blake2b_128"
        ],
        "payoff_bank_content_digest_blake2b_128": str(
            payoff_bank_content_digest
        ),
        "endpoint_bank_content_digest_blake2b_128": str(
            endpoint_bank_content_digest
        ),
        "matched_sampling_id_blake2b_128": str(matched_sampling_id),
        "runs": [dict(row) for row in runs],
        "paired_comparisons": [
            dict(row) for row in comparisons
        ],
    }


def _validate_detection_partial_checkpoint(
    checkpoint: Mapping[str, Any],
    entry: Mapping[str, Any],
    study: Mapping[str, Any],
    *,
    payoff_bank_content_digest: str,
    endpoint_bank_content_digest: str,
    matched_sampling_id: str,
) -> None:
    """Accept only complete ordered/shuffled learner--seed pairs."""

    if (
        checkpoint.get("schema")
        != PRIME_PROBE_DETECTION_PARTIAL_CHECKPOINT_SCHEMA
    ):
        raise ValueError("detection partial checkpoint schema changed")
    graph_id = str(entry["graph_id"])
    if str(checkpoint.get("graph_id")) != graph_id:
        raise ValueError("detection partial checkpoint graph changed")
    if str(checkpoint.get("family", "")).lower() != str(
        entry["family"]
    ).lower():
        raise ValueError("detection partial checkpoint family changed")
    if checkpoint.get(
        "resume_contract_digest_blake2b_128"
    ) != study["resume_contract_digest_blake2b_128"]:
        raise ValueError(
            "detection partial checkpoint resume contract changed"
        )
    frozen = study["frozen_plans"][graph_id]
    if checkpoint.get(
        "source_plan_digest_blake2b_128"
    ) != frozen["plan_digest_blake2b_128"]:
        raise ValueError("detection partial checkpoint source plan changed")
    if checkpoint.get(
        "payoff_bank_content_digest_blake2b_128"
    ) != payoff_bank_content_digest:
        raise ValueError("detection partial checkpoint payoff bank changed")
    if checkpoint.get(
        "endpoint_bank_content_digest_blake2b_128"
    ) != endpoint_bank_content_digest:
        raise ValueError("detection partial checkpoint endpoint bank changed")
    if checkpoint.get(
        "matched_sampling_id_blake2b_128"
    ) != matched_sampling_id:
        raise ValueError(
            "detection partial checkpoint matched sampling changed"
        )

    runs = list(checkpoint.get("runs") or [])
    comparisons = list(checkpoint.get("paired_comparisons") or [])
    if not runs or len(runs) % 2 or len(comparisons) * 2 != len(runs):
        raise ValueError(
            "detection partial checkpoint contains an incomplete pair"
        )
    allowed_pairs = {
        (str(defender["label"]), int(seed))
        for defender in study["defenders"]
        for seed in study["seeds"]
    }
    runs_by_pair: dict[
        tuple[str, int],
        dict[str, Mapping[str, Any]],
    ] = {}
    for run in runs:
        pair_key = (str(run["defender_label"]), int(run["seed"]))
        if pair_key not in allowed_pairs:
            raise ValueError(
                "detection partial checkpoint contains an unknown pair"
            )
        condition = str(run["condition"])
        pair_runs = runs_by_pair.setdefault(pair_key, {})
        if condition in pair_runs:
            raise ValueError(
                "detection partial checkpoint contains a duplicate run"
            )
        pair_runs[condition] = run
        if run.get(
            "payoff_bank_content_digest_blake2b_128"
        ) != payoff_bank_content_digest:
            raise ValueError("partial run payoff bank changed")
        if run.get(
            "endpoint_bank_content_digest_blake2b_128"
        ) != endpoint_bank_content_digest:
            raise ValueError("partial run endpoint bank changed")
        if run.get(
            "matched_sampling_id_blake2b_128"
        ) != matched_sampling_id:
            raise ValueError("partial run matched sampling changed")
        if run.get(
            "resume_contract_digest_blake2b_128"
        ) != study["resume_contract_digest_blake2b_128"]:
            raise ValueError("partial run resume contract changed")
        if condition == "ordered_prime_probe":
            _validate_ordered_detection_run(run, study)
        elif condition == "shuffled_identical_action_multiset":
            _validate_shuffled_run(run, study)
        else:
            raise ValueError(
                "detection partial checkpoint condition changed"
            )
    expected_conditions = {
        "ordered_prime_probe",
        "shuffled_identical_action_multiset",
    }
    if any(
        set(pair_runs) != expected_conditions
        for pair_runs in runs_by_pair.values()
    ):
        raise ValueError(
            "detection partial checkpoint contains a half-completed pair"
        )

    comparison_pairs: set[tuple[str, int]] = set()
    for comparison in comparisons:
        pair_key = (
            str(comparison["defender_label"]),
            int(comparison["seed"]),
        )
        if pair_key not in runs_by_pair or pair_key in comparison_pairs:
            raise ValueError(
                "detection partial checkpoint comparison identity changed"
            )
        if not bool(comparison.get("identical_action_multiset")):
            raise ValueError(
                "detection partial checkpoint has an unmatched control"
            )
        if comparison.get(
            "source_plan_digest_blake2b_128"
        ) != frozen["plan_digest_blake2b_128"]:
            raise ValueError("partial comparison source plan changed")
        comparison_pairs.add(pair_key)
    if comparison_pairs != set(runs_by_pair):
        raise ValueError(
            "detection partial checkpoint comparison is incomplete"
        )


def _completed_pair_keys(
    runs: list[Mapping[str, Any]],
    comparisons: list[Mapping[str, Any]],
) -> set[tuple[str, int]]:
    """Return pair identities after the partial validator established parity."""

    run_keys = {
        (str(run["defender_label"]), int(run["seed"]))
        for run in runs
    }
    comparison_keys = {
        (
            str(comparison["defender_label"]),
            int(comparison["seed"]),
        )
        for comparison in comparisons
    }
    if run_keys != comparison_keys:
        raise ValueError("completed detection pair identities differ")
    return run_keys


def _validate_detection_checkpoint(
    checkpoint: Mapping[str, Any],
    entry: Mapping[str, Any],
    study: Mapping[str, Any],
) -> None:
    if (
        checkpoint.get("schema")
        != PRIME_PROBE_DETECTION_GRAPH_CHECKPOINT_SCHEMA
    ):
        raise ValueError("detection checkpoint schema changed")
    graph_id = str(entry["graph_id"])
    if str(checkpoint.get("graph_id")) != graph_id:
        raise ValueError("detection checkpoint graph identity changed")
    if str(checkpoint.get("family", "")).lower() != str(
        entry["family"]
    ).lower():
        raise ValueError("detection checkpoint family changed")
    if checkpoint.get(
        "resume_contract_digest_blake2b_128"
    ) != study["resume_contract_digest_blake2b_128"]:
        raise ValueError("detection checkpoint resume contract changed")
    frozen = study["frozen_plans"][graph_id]
    if checkpoint.get(
        "source_plan_digest_blake2b_128"
    ) != frozen["plan_digest_blake2b_128"]:
        raise ValueError("detection checkpoint source plan changed")
    expected_pairs = len(study["defenders"]) * len(study["seeds"])
    runs = list(checkpoint.get("runs") or [])
    comparisons = list(checkpoint.get("paired_comparisons") or [])
    if len(runs) != 2 * expected_pairs:
        raise ValueError("detection checkpoint condition runs are incomplete")
    if len(comparisons) != expected_pairs:
        raise ValueError("detection checkpoint comparisons are incomplete")
    expected_keys = {
        (str(defender["label"]), int(seed), condition)
        for defender in study["defenders"]
        for seed in study["seeds"]
        for condition in (
            "ordered_prime_probe",
            "shuffled_identical_action_multiset",
        )
    }
    observed_keys = {
        (
            str(run["defender_label"]),
            int(run["seed"]),
            str(run["condition"]),
        )
        for run in runs
    }
    if observed_keys != expected_keys:
        raise ValueError("detection checkpoint run identities changed")
    for run in runs:
        if run.get(
            "resume_contract_digest_blake2b_128"
        ) != study["resume_contract_digest_blake2b_128"]:
            raise ValueError("detection run resume contract changed")
        if int(run["turns"]) != int(study["schedule"]["total_turns"]):
            raise ValueError("detection run horizon changed")
        if run["condition"] == "ordered_prime_probe":
            _validate_ordered_detection_run(run, study)
        else:
            _validate_shuffled_run(run, study)
    if not all(
        bool(row.get("identical_action_multiset"))
        for row in comparisons
    ):
        raise ValueError("detection checkpoint has an unmatched control")


def _validate_source_checkpoint(
    checkpoint: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> None:
    if checkpoint.get("schema") != PRIME_PROBE_GRAPH_CHECKPOINT_SCHEMA:
        raise ValueError("source checkpoint schema changed")
    if str(checkpoint.get("graph_id")) != str(entry["graph_id"]):
        raise ValueError("source checkpoint graph identity changed")
    if str(checkpoint.get("family", "")).lower() != str(
        entry["family"]
    ).lower():
        raise ValueError("source checkpoint family changed")
    source_case = checkpoint.get("case") or {}
    if tuple(source_case.get("route_ids") or ()) != tuple(
        entry["route_ids"]
    ):
        raise ValueError("source checkpoint route universe changed")
    plan = checkpoint.get("plan") or {}
    if plan.get("schema") != PRIME_PROBE_PLAN_SCHEMA:
        raise ValueError("source Prime-and-Probe plan schema changed")
    restored = PrimeProbePlan.from_dict(plan)
    _validate_plan_indices_and_ids(
        restored,
        tuple(map(str, entry["route_ids"])),
        tuple(map(str, entry["eve_action_ids"])),
    )


def _validate_loaded_case(case: Any, entry: Mapping[str, Any]) -> None:
    route_ids = tuple(str(route["route_id"]) for route in case.routes)
    action_ids = tuple(
        str(action["id"]) for action in case.actions("attack_surface_all")
    )
    if (
        str(case.graph_id) != str(entry["graph_id"])
        or str(case.assignment_id) != str(entry["assignment_id"])
        or route_ids != tuple(map(str, entry["route_ids"]))
        or action_ids != tuple(map(str, entry["eve_action_ids"]))
    ):
        raise ValueError("loaded case differs from the frozen action universe")


def _validate_frozen_plan(
    case: Any,
    bank: HistogramPayoffBank,
    plan: PrimeProbePlan,
    frozen: Mapping[str, Any],
) -> None:
    if _stable_digest(frozen["plan"]) != frozen[
        "plan_digest_blake2b_128"
    ]:
        raise ValueError("embedded source plan digest changed")
    _validate_plan_indices_and_ids(
        plan,
        bank.route_ids,
        bank.action_ids,
    )
    if str(case.graph_id) != str(bank.graph_id):
        raise ValueError("frozen plan case and matched bank differ")


def _validate_plan_indices_and_ids(
    plan: PrimeProbePlan,
    route_ids: tuple[str, ...],
    action_ids: tuple[str, ...],
) -> None:
    if tuple(route_ids[index] for index in plan.target_route_indices) != (
        plan.target_route_ids
    ):
        raise ValueError("source plan target route identities changed")
    if tuple(
        action_ids[index] for index in plan.prime_action_indices
    ) != plan.prime_action_ids:
        raise ValueError("source plan Prime action identities changed")
    if action_ids[plan.exploit_action_index] != plan.exploit_action_id:
        raise ValueError("source plan Exploit action identity changed")
    if action_ids[plan.probe_action_index] != plan.probe_action_id:
        raise ValueError("source plan Probe action identity changed")
    all_routes = set(range(len(route_ids)))
    if (
        set(plan.target_route_indices)
        | set(plan.alternative_route_indices)
    ) != all_routes:
        raise ValueError("source plan route partition changed")


def _validate_study_manifest(study: Mapping[str, Any]) -> None:
    if study.get("schema") != PRIME_PROBE_DETECTION_CORPUS_SCHEMA:
        raise ValueError("Prime-and-Probe detection schema changed")
    cases = list(study.get("cases") or [])
    defenders = list(study.get("defenders") or [])
    seeds = list(study.get("seeds") or [])
    if not cases or not defenders or not seeds:
        raise ValueError("detection study must include cases, learners, and seeds")
    if len({str(row["graph_id"]) for row in cases}) != len(cases):
        raise ValueError("detection study graph identities are not unique")
    schedule = study.get("schedule") or {}
    restored_schedule = PrimeProbeSchedule(**{
        key: value
        for key, value in schedule.items()
        if key != "total_turns"
    })
    if int(schedule.get("total_turns", -1)) != restored_schedule.total_turns:
        raise ValueError("detection schedule total is inconsistent")
    detector = study.get("detector") or {}
    if (
        detector.get("commissioning_split") != "feedback"
        or detector.get("commissioning_action_id") != "no_attack"
        or detector.get("execution_split") != "audit"
        or detector.get("response") != "monitor_only_native_learning"
        or int(detector.get("horizon", -1))
        != restored_schedule.total_turns
        or int(detector.get("commissioning_draws_per_route", -1))
        != DEFAULT_DETECTOR_COMMISSIONING_DRAWS_PER_ROUTE
        or float(detector.get("model_false_alarm_level", -1.0))
        != DEFAULT_DETECTOR_FALSE_ALARM_LEVEL
        or float(detector.get("commissioning_pseudocount", -1.0))
        != DEFAULT_DETECTOR_PSEUDOCOUNT
        or tuple(map(float, detector.get("change_fractions") or ()))
        != DEFAULT_DETECTOR_CHANGE_FRACTIONS
    ):
        raise ValueError("detection configuration changed")
    if int(study["samples_per_cell_per_split"]) < int(
        detector["commissioning_draws_per_route"]
    ):
        raise ValueError("matched bank cannot supply detector commissioning")
    frozen_plans = study.get("frozen_plans") or {}
    if set(frozen_plans) != {
        str(entry["graph_id"]) for entry in cases
    }:
        raise ValueError("embedded frozen plans do not match study cases")
    for entry in cases:
        frozen = frozen_plans[str(entry["graph_id"])]
        if _stable_digest(frozen["plan"]) != frozen[
            "plan_digest_blake2b_128"
        ]:
            raise ValueError("embedded source plan digest changed")
        _validate_plan_indices_and_ids(
            PrimeProbePlan.from_dict(frozen["plan"]),
            tuple(map(str, entry["route_ids"])),
            tuple(map(str, entry["eve_action_ids"])),
        )
    expected_pairs = len(cases) * len(defenders) * len(seeds)
    if (
        int(study["expected_condition_runs"]) != 2 * expected_pairs
        or int(study["expected_paired_comparisons"]) != expected_pairs
        or int(study["expected_detector_runs"]) != expected_pairs
    ):
        raise ValueError("detection study expected row counts changed")
    if study.get("resume_contract_digest_blake2b_128") != _stable_digest(
        study.get("resume_contract")
    ):
        raise ValueError("detection study resume contract digest changed")
    if study.get("resume_contract") != _resume_contract_payload(study):
        raise ValueError("detection study resume contract payload changed")


def _resume_contract_payload(
    study: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "histogram_prime_probe_detection_resume_contract.v1",
        "implementation_revision": (
            PRIME_PROBE_DETECTION_IMPLEMENTATION_REVISION
        ),
        "smoke": bool(study["smoke"]),
        "source_prime_probe_manifest_digest_blake2b_128": study[
            "source_prime_probe_manifest_digest_blake2b_128"
        ],
        "source_corpus_selection_digest_blake2b_128": study[
            "source_corpus_selection_digest_blake2b_128"
        ],
        "cases": [
            {
                "graph_id": row["graph_id"],
                "assignment_id": row["assignment_id"],
                "route_ids": list(row["route_ids"]),
                "eve_action_ids": list(row["eve_action_ids"]),
            }
            for row in study["cases"]
        ],
        "defenders": list(study["defenders"]),
        "seeds": list(study["seeds"]),
        "schedule": dict(study["schedule"]),
        "source_plan_digests": {
            graph_id: row["plan_digest_blake2b_128"]
            for graph_id, row in study["frozen_plans"].items()
        },
        "samples_per_cell_per_split": int(
            study["samples_per_cell_per_split"]
        ),
        "matched_bank_root_seed": int(study["matched_bank_root_seed"]),
        "detector": dict(study["detector"]),
    }


def _detector_descriptive_summary(
    runs: list[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = [
        row
        for row in runs
        if row["condition"] == "ordered_prime_probe"
    ]
    result: dict[str, Any] = {}
    for detector_name in (
        "delivery_only",
        "classical_service",
        "quantum_endpoint",
        "e91_measurement_counts_only",
    ):
        outcomes = [
            row["alice_endpoint_detector"]["outcomes"][detector_name]
            for row in ordered
        ]
        pre_prime = sum(
            bool(outcome["pre_prime_false_alarm"])
            for outcome in outcomes
        )
        post_prime_delays = [
            int(outcome["detection_delay_from_prime_start_turns"])
            for outcome in outcomes
            if (
                outcome["alarm_turn"] is not None
                and not outcome["pre_prime_false_alarm"]
            )
        ]
        result[detector_name] = {
            "run_count": len(outcomes),
            "alarm_count": sum(
                outcome["alarm_turn"] is not None
                for outcome in outcomes
            ),
            "pre_prime_false_alarm_count": pre_prime,
            "post_prime_detection_count": len(post_prime_delays),
            "post_prime_detection_delay_turns": {
                "minimum": min(post_prime_delays)
                if post_prime_delays else None,
                "median": statistics.median(post_prime_delays)
                if post_prime_delays else None,
                "maximum": max(post_prime_delays)
                if post_prime_delays else None,
            },
        }
    return result


def _cache_plan_summary(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "canonical_cache_entries": int(
            cache["canonical_unique_move_count_total"]
        ),
        "estimated_resident_cache_bytes": int(
            cache["estimated_full_action_surface_cache_bytes"]
        ),
        "configured_lru_capacity_per_route": int(
            cache["configured_entries_per_route"]
        ),
        "minimum_lru_capacity_per_route": int(
            cache["minimum_entries_per_route_without_eviction"]
        ),
    }


def _check_existing_manifest(
    output: Path,
    study: Mapping[str, Any],
) -> None:
    path = output / "run_manifest.json"
    if not path.is_file():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    _validate_study_manifest(existing)
    if existing[
        "resume_contract_digest_blake2b_128"
    ] != study["resume_contract_digest_blake2b_128"]:
        raise ValueError(
            "output directory belongs to another detection resume contract"
        )


def _save_bank_atomic(bank: Any, path: Path) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    bank.save(temporary)
    temporary.replace(path)


def _stable_digest(payload: Any) -> str:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return digest.hexdigest()


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
    rows: list[Mapping[str, Any]],
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
