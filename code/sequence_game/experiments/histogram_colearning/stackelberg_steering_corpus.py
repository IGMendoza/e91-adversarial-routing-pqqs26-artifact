"""Five-topology calibration and evaluation of strict Stackelberg steering."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .calibration_bank import build_histogram_payoff_bank
from .config import HistogramCoLearningConfig, REPO_ROOT
from .corpus_scale import load_corpus_scale_case
from .physics import HistogramPhysicsPool, histogram_cache_plan
from .stackelberg_eve import solve_stackelberg_oracle
from .stackelberg_eve_corpus import (
    DEFAULT_CORPUS_RUN_DIR,
    DEFAULT_STACKELBERG_FAMILIES,
    STACKELBERG_DEFENDERS,
    StackelbergDefender,
)
from .stackelberg_steering import (
    StackelbergSteeringConfig,
    feasible_stackelberg_response_margins,
    run_pessimistic_stackelberg_steering,
)
from .topology_census import available_ram_snapshot


STACKELBERG_STEERING_STUDY_SCHEMA = (
    "histogram_stackelberg_steering_study.v1"
)
STACKELBERG_STEERING_CHECKPOINT_SCHEMA = (
    "histogram_stackelberg_steering_checkpoint.v1"
)
DEFAULT_STACKELBERG_STEERING_RUN_DIR = (
    REPO_ROOT / "runs" / "histogram_stackelberg_steering_500k"
)
DEFAULT_MARGIN_CANDIDATES = (
    0.0,
    0.01,
    0.02,
    500_000.0 ** -0.25,
    0.05,
    0.08,
)
DEFAULT_CALIBRATION_SEEDS = (
    20_260_717_01,
    20_260_717_02,
    20_260_717_03,
)
DEFAULT_EVALUATION_SEEDS = (
    20_260_718_01,
    20_260_718_02,
    20_260_718_03,
)


ProgressCallback = Callable[[str], None]


def build_stackelberg_steering_manifest(
    corpus_manifest: Mapping[str, Any],
    *,
    calibration_turns: int = 100_000,
    evaluation_turns: int = 500_000,
    margin_candidates: Sequence[float] = DEFAULT_MARGIN_CANDIDATES,
    calibration_seeds: Sequence[int] = DEFAULT_CALIBRATION_SEEDS,
    evaluation_seeds: Sequence[int] = DEFAULT_EVALUATION_SEEDS,
    samples_per_cell_per_split: int = 64,
    record_every: int = 1_000,
    evaluation_margin_override: float | None = None,
) -> dict[str, Any]:
    """Freeze the steering margins, cases, defenders, and disjoint seeds."""

    if calibration_turns < 1 or evaluation_turns < 1:
        raise ValueError("calibration and evaluation turns must be positive")
    if samples_per_cell_per_split < 2:
        raise ValueError("samples per cell must be at least two")
    if record_every < 1:
        raise ValueError("record_every must be positive")
    margins = tuple(sorted(set(map(float, margin_candidates))))
    if not margins or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in margins
    ):
        raise ValueError("margin candidates must be finite values in [0, 1]")
    calibration_seed_tuple = tuple(map(int, calibration_seeds))
    evaluation_seed_tuple = tuple(map(int, evaluation_seeds))
    if not calibration_seed_tuple or not evaluation_seed_tuple:
        raise ValueError("both seed sets must be nonempty")
    if set(calibration_seed_tuple) & set(evaluation_seed_tuple):
        raise ValueError("calibration and evaluation seeds must be disjoint")
    if evaluation_margin_override is not None:
        override = float(evaluation_margin_override)
        if not math.isfinite(override) or override < 0.0 or override > 1.0:
            raise ValueError(
                "evaluation margin override must be a finite value in [0, 1]"
            )
        if not any(
            math.isclose(
                override,
                margin,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for margin in margins
        ):
            raise ValueError(
                "evaluation margin override must be a calibration candidate"
            )
    else:
        override = None

    by_family = {
        str(row["family"]).lower(): dict(row)
        for row in corpus_manifest["cases"]
    }
    missing = [
        family
        for family in DEFAULT_STACKELBERG_FAMILIES
        if family not in by_family
    ]
    if missing:
        raise ValueError(f"corpus manifest is missing steering cases: {missing}")
    cases = [by_family[family] for family in DEFAULT_STACKELBERG_FAMILIES]
    bytes_per_bank_draw = 84
    preflight_cases = [
        {
            "graph_id": str(row["graph_id"]),
            "family": str(row["family"]),
            "route_count": int(row["route_count"]),
            "eve_action_count": int(row["eve_action_count"]),
            "nominal_route_action_cells": int(
                row["nominal_route_action_cells"]
            ),
            "estimated_cache_mib": float(row["estimated_cache_mib"]),
            "estimated_feedback_audit_bank_mib": (
                int(row["nominal_route_action_cells"])
                * int(samples_per_cell_per_split)
                * bytes_per_bank_draw
                / 1024**2
            ),
        }
        for row in cases
    ]
    calibration_run_count = (
        len(cases)
        * len(STACKELBERG_DEFENDERS)
        * len(margins)
        * len(calibration_seed_tuple)
    )
    evaluation_run_count = (
        len(cases)
        * len(STACKELBERG_DEFENDERS)
        * len(evaluation_seed_tuple)
    )
    return {
        "schema": STACKELBERG_STEERING_STUDY_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_corpus_selection_digest_blake2b_128": corpus_manifest[
            "selection_digest_blake2b_128"
        ],
        "source_corpus_plan": dict(corpus_manifest["plan"]),
        "families": list(DEFAULT_STACKELBERG_FAMILIES),
        "cases": cases,
        "defenders": [row.to_dict() for row in STACKELBERG_DEFENDERS],
        "calibration_turns": int(calibration_turns),
        "evaluation_turns": int(evaluation_turns),
        "margin_candidates": list(margins),
        "calibration_seeds": list(calibration_seed_tuple),
        "evaluation_seeds": list(evaluation_seed_tuple),
        "samples_per_cell_per_split": int(samples_per_cell_per_split),
        "record_every": int(record_every),
        "evaluation_margin_override": override,
        "maximum_calibration_run_count": calibration_run_count,
        "expected_evaluation_run_count": evaluation_run_count,
        "selection_rule": {
            "metric": "per_turn_unconstrained_stackelberg_regret",
            "aggregation": "mean over calibration seeds",
            "uncertainty": "sample standard error over calibration seeds",
            "eligible_candidates": (
                "strictly positive response margins; zero is retained only "
                "as the unconstrained boundary control"
            ),
            "eligible_set": (
                "mean no larger than raw-minimum mean plus the raw-minimum "
                "standard error"
            ),
            "tie_break": (
                "smallest response margin in the positive eligible set"
            ),
        },
        "scientific_contract": {
            "source_method": [
                (
                    "Deng, Schneider, and Sivan (NeurIPS 2019): strict "
                    "best-response induction steers no-regret learners to "
                    "Stackelberg value"
                ),
                (
                    "Zhang, Ma, and Mazumdar (arXiv 2502.20770): pessimistic "
                    "facets trade leader value for finite-time unique-response "
                    "steering"
                ),
            ],
            "leader_information": (
                "declared histogram feedback payoff surface and action costs"
            ),
            "leader_exclusions": [
                "Alice live policy or weights",
                "audit split",
                "future feedback draws",
            ],
            "leader_action": (
                "stationary mixture over declared physical Eve interventions"
            ),
            "follower": "predeclared Alice route learner",
            "route_policy_manipulation": (
                "the commitment makes one route exceed every competing route "
                "by the calibrated public-reward margin"
            ),
            "audit": "disjoint equal-coverage scoring bank",
            "reward_poisoning": False,
        },
        "preflight": {
            "cases": preflight_cases,
            "total_nominal_route_action_cells": sum(
                row["nominal_route_action_cells"]
                for row in preflight_cases
            ),
            "estimated_parallel_cache_mib": sum(
                row["estimated_cache_mib"] for row in preflight_cases
            ),
            "estimated_parallel_feedback_audit_bank_mib": sum(
                row["estimated_feedback_audit_bank_mib"]
                for row in preflight_cases
            ),
            "host_ram_snapshot": available_ram_snapshot(),
        },
    }


def select_stackelberg_response_margin(
    calibration_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select a positive strict-response margin by one-standard-error rule."""

    grouped: dict[float, list[float]] = {}
    for row in calibration_rows:
        margin = float(row["steering_config"]["response_margin"])
        grouped.setdefault(margin, []).append(
            float(row["per_turn_unconstrained_stackelberg_regret"])
        )
    if not grouped:
        raise ValueError("margin selection requires calibration rows")
    summary = []
    for margin in sorted(grouped):
        values = np.asarray(grouped[margin], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("calibration regret contains nonfinite values")
        standard_error = (
            float(np.std(values, ddof=1) / math.sqrt(values.size))
            if values.size > 1 else 0.0
        )
        summary.append({
            "response_margin": margin,
            "run_count": int(values.size),
            "mean_regret": float(np.mean(values)),
            "standard_error": standard_error,
            "mean_eve_public_utility": float(
                np.mean([
                    float(row["mean_eve_public_utility"])
                    for row in calibration_rows
                    if math.isclose(
                        float(row["steering_config"]["response_margin"]),
                        margin,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ])
            ),
            "mean_intended_route_mass": float(
                np.mean([
                    float(row["intended_route_empirical_mass"])
                    for row in calibration_rows
                    if math.isclose(
                        float(row["steering_config"]["response_margin"]),
                        margin,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                ])
            ),
        })
    positive_summary = [
        row for row in summary if row["response_margin"] > 0.0
    ]
    if not positive_summary:
        raise ValueError(
            "strict Stackelberg steering requires a positive feasible margin"
        )
    raw = min(
        positive_summary,
        key=lambda row: (
            row["mean_regret"],
            row["response_margin"],
        ),
    )
    threshold = float(raw["mean_regret"] + raw["standard_error"])
    eligible = [
        row
        for row in positive_summary
        if row["mean_regret"] <= threshold + 1e-15
    ]
    selected = min(
        eligible,
        key=lambda row: row["response_margin"],
    )
    boundary = next(
        (
            row for row in summary
            if math.isclose(
                float(row["response_margin"]),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ),
        None,
    )
    return {
        "raw_minimum_margin": float(raw["response_margin"]),
        "raw_minimum_mean_regret": float(raw["mean_regret"]),
        "one_standard_error_threshold": threshold,
        "eligible_margins": [
            float(row["response_margin"]) for row in eligible
        ],
        "selected_margin": float(selected["response_margin"]),
        "selected_mean_regret": float(selected["mean_regret"]),
        "zero_margin_boundary_control": boundary,
        "candidate_summary": summary,
    }


def run_stackelberg_steering_study(
    corpus_manifest: Mapping[str, Any],
    study_manifest: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_STACKELBERG_STEERING_RUN_DIR,
    graph_workers: int = 5,
    calibration_source_dir: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run or resume graph-owned calibration and fresh 500k evaluation."""

    output = Path(output_dir)
    checkpoints = output / "graph_checkpoints"
    progress_dir = output / "progress"
    checkpoints.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / "run_manifest.json", dict(study_manifest))
    case_by_graph = {
        str(row["graph_id"]): dict(row) for row in study_manifest["cases"]
    }
    complete: dict[str, dict[str, Any]] = {}
    for path in checkpoints.glob("*.json"):
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
        row
        for row in study_manifest["cases"]
        if str(row["graph_id"]) not in complete
    ]
    if pending:
        calibration_source = (
            Path(calibration_source_dir)
            if calibration_source_dir is not None else None
        )
        if calibration_source is not None:
            missing_sources = [
                calibration_source / "graph_checkpoints"
                / f"{row['graph_id']}.json"
                for row in pending
                if not (
                    calibration_source / "graph_checkpoints"
                    / f"{row['graph_id']}.json"
                ).is_file()
            ]
            if missing_sources:
                raise FileNotFoundError(
                    "missing reusable calibration checkpoints: "
                    + ", ".join(map(str, missing_sources))
                )
        workers = min(max(1, int(graph_workers)), len(pending))
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("Stackelberg steering execution requires fork")
        context = mp.get_context("fork")
        payloads = [
            {
                "entry": dict(entry),
                "corpus_manifest": dict(corpus_manifest),
                "study_manifest": dict(study_manifest),
                "output_dir": str(output),
                "calibration_source_checkpoint": (
                    str(
                        calibration_source / "graph_checkpoints"
                        / f"{entry['graph_id']}.json"
                    )
                    if calibration_source is not None else None
                ),
            }
            for entry in pending
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            future_map = {
                executor.submit(_run_steering_graph, payload): payload["entry"]
                for payload in payloads
            }
            for future in as_completed(future_map):
                checkpoint = future.result()
                complete[str(checkpoint["graph_id"])] = checkpoint
                if progress_callback is not None:
                    progress_callback(
                        f"Steering graph complete {len(complete)}/"
                        f"{len(study_manifest['cases'])}: "
                        f"{checkpoint['family']}"
                    )
    ordered = [
        complete[str(row["graph_id"])] for row in study_manifest["cases"]
    ]
    calibration_rows = [
        row for checkpoint in ordered for row in checkpoint["calibration_runs"]
    ]
    evaluation_rows = [
        row for checkpoint in ordered for row in checkpoint["evaluation_runs"]
    ]
    _write_jsonl_atomic(
        output / "calibration_results.jsonl",
        calibration_rows,
    )
    _write_jsonl_atomic(output / "results.jsonl", evaluation_rows)
    summary = {
        "schema": STACKELBERG_STEERING_STUDY_SCHEMA,
        "complete": len(ordered) == len(study_manifest["cases"]),
        "graph_count": len(ordered),
        "calibration_run_count": len(calibration_rows),
        "evaluation_run_count": len(evaluation_rows),
        "expected_evaluation_run_count": int(
            study_manifest["expected_evaluation_run_count"]
        ),
        "selected_margins": {
            str(checkpoint["family"]): {
                str(row["defender_label"]): float(row["selected_margin"])
                for row in checkpoint["margin_selections"]
            }
            for checkpoint in ordered
        },
    }
    _write_json_atomic(output / "summary.json", summary)
    return summary


def _run_steering_graph(payload: Mapping[str, Any]) -> dict[str, Any]:
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
    base_config = HistogramCoLearningConfig(
        turns=int(study["evaluation_turns"]),
        requested_pairs_per_turn=int(source_plan["requested_pairs_per_turn"]),
        final_window=int(source_plan["final_window"]),
        record_every=int(study["record_every"]),
        posthoc_samples_per_cell=0,
        seed=int(study["calibration_seeds"][0]),
        game_mode="attack_surface_all",
        age_points_per_edge=int(source_plan["age_points_per_edge"]),
        evaluated_cache_size=cache_size,
    )
    cache_plan = histogram_cache_plan(case, base_config)
    if not cache_plan["sufficient_for_declared_action_surface_without_eviction"]:
        base_config = replace(
            base_config,
            evaluated_cache_size=int(
                cache_plan["minimum_entries_per_route_without_eviction"]
            ),
        )
        cache_plan = histogram_cache_plan(case, base_config)

    progress("physics", "compiling shared equal-coverage physical bank")
    physics = HistogramPhysicsPool(case, base_config)
    selection_digest = str(
        study["source_corpus_selection_digest_blake2b_128"]
    )
    bank = build_histogram_payoff_bank(
        case,
        base_config,
        samples_per_cell=int(study["samples_per_cell_per_split"]),
        root_seed=20_260_716_41,
        physics=physics,
        progress_callback=lambda message: progress("physics", message),
        selection_digest_blake2b_128=selection_digest,
        evaluation_partition_loaded=True,
        sampling_purpose="stackelberg_pessimistic_route_steering",
        sampling_design=(
            "feedback split computes and executes the strict-response "
            "commitment; audit split is held out for evaluation"
        ),
    )
    feedback_matrix = bank.feedback.alice_payoff_matrix()
    audit_matrix = bank.audit.alice_payoff_matrix()
    feedback_unconstrained = solve_stackelberg_oracle(
        feedback_matrix,
        bank.action_costs,
    )
    audit_unconstrained = solve_stackelberg_oracle(
        audit_matrix,
        bank.action_costs,
    )
    feedback_frontier = feasible_stackelberg_response_margins(
        feedback_matrix,
        bank.action_costs,
        study["margin_candidates"],
    )
    audit_frontier = feasible_stackelberg_response_margins(
        audit_matrix,
        bank.action_costs,
        study["margin_candidates"],
    )
    feasible_margins = [
        float(row["response_margin"])
        for row in feedback_frontier
        if row["feasible"]
    ]
    if not feasible_margins:
        raise RuntimeError(f"{case.family} has no feasible steering margin")

    calibration_runs = []
    margin_selections = []
    evaluation_runs = []
    calibration_source_checkpoint = payload.get(
        "calibration_source_checkpoint"
    )
    if calibration_source_checkpoint is not None:
        calibration_runs = _load_reusable_calibration_runs(
            Path(str(calibration_source_checkpoint)),
            entry=entry,
            study=study,
            feasible_margins=feasible_margins,
            bank_content_digest=bank.content_digest(),
            selection_digest=selection_digest,
        )
        progress(
            "calibration",
            f"reused {len(calibration_runs)} verified calibration runs",
        )
    for defender_row in study["defenders"]:
        defender = StackelbergDefender(
            label=str(defender_row["label"]),
            alice_learner_mode=str(defender_row["alice_learner_mode"]),
            config_overrides=dict(defender_row["config_overrides"]),
        )
        defender_calibration = [
            row
            for row in calibration_runs
            if str(row["defender_label"]) == defender.label
        ]
        if calibration_source_checkpoint is None:
            for margin in feasible_margins:
                for seed in map(int, study["calibration_seeds"]):
                    progress(
                        "calibration",
                        f"{defender.label}, margin {margin:g}, seed {seed}",
                    )
                    config = replace(
                        base_config,
                        **defender.config_overrides,
                        turns=int(study["calibration_turns"]),
                        seed=seed,
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
                        feedback_unconstrained_oracle=feedback_unconstrained,
                        audit_unconstrained_oracle=audit_unconstrained,
                    )
                    run.update({
                        "phase": "calibration",
                        "defender_label": defender.label,
                        "bank_content_digest_blake2b_128": (
                            bank.content_digest()
                        ),
                        "source_corpus_selection_digest_blake2b_128": (
                            selection_digest
                        ),
                    })
                    calibration_runs.append(run)
                    defender_calibration.append(run)
        selection = select_stackelberg_response_margin(
            defender_calibration
        )
        selection["defender_label"] = defender.label
        selected_margin = float(selection["selected_margin"])
        if study.get("evaluation_margin_override") is not None:
            selected_margin = float(study["evaluation_margin_override"])
            if not any(
                math.isclose(
                    selected_margin,
                    margin,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
                for margin in feasible_margins
            ):
                raise ValueError(
                    f"{case.family} cannot evaluate response margin "
                    f"{selected_margin:g}"
                )
            selection["calibrated_selected_margin"] = float(
                selection["selected_margin"]
            )
            selection["selected_margin"] = selected_margin
            selection["evaluation_margin_override"] = selected_margin
        margin_selections.append(selection)
        for seed in map(int, study["evaluation_seeds"]):
            progress(
                "evaluation",
                f"{defender.label}, margin {selected_margin:g}, seed {seed}",
            )
            config = replace(
                base_config,
                **defender.config_overrides,
                turns=int(study["evaluation_turns"]),
                seed=seed,
            )
            run = run_pessimistic_stackelberg_steering(
                case,
                bank,
                config,
                StackelbergSteeringConfig(
                    response_margin=selected_margin,
                    record_every=int(study["record_every"]),
                ),
                seed=seed,
                selection_digest_blake2b_128=selection_digest,
                progress_callback=lambda step, label=defender.label, active_seed=seed: progress(
                    "evaluation",
                    f"{label}, seed {active_seed}: "
                    f"{step}/{study['evaluation_turns']}",
                ),
                feedback_unconstrained_oracle=feedback_unconstrained,
                audit_unconstrained_oracle=audit_unconstrained,
            )
            run.update({
                "phase": "evaluation",
                "defender_label": defender.label,
                "margin_selection": selection,
                "bank_content_digest_blake2b_128": bank.content_digest(),
                "source_corpus_selection_digest_blake2b_128": selection_digest,
            })
            evaluation_runs.append(run)

    checkpoint = {
        "schema": STACKELBERG_STEERING_CHECKPOINT_SCHEMA,
        "graph_id": graph_id,
        "family": case.family,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "case": case.to_dict(),
        "cache_plan": cache_plan,
        "bank": bank.to_summary(),
        "physics": physics.diagnostics(),
        "calibration_source_checkpoint": (
            str(calibration_source_checkpoint)
            if calibration_source_checkpoint is not None else None
        ),
        "feedback_margin_frontier": feedback_frontier,
        "audit_margin_frontier": audit_frontier,
        "margin_selections": margin_selections,
        "calibration_runs": calibration_runs,
        "evaluation_runs": evaluation_runs,
    }
    _write_json_atomic(
        output / "graph_checkpoints" / f"{graph_id}.json",
        checkpoint,
    )
    progress(
        "complete",
        f"completed {len(calibration_runs)} calibration and "
        f"{len(evaluation_runs)} evaluation runs",
    )
    return checkpoint


def _load_reusable_calibration_runs(
    path: Path,
    *,
    entry: Mapping[str, Any],
    study: Mapping[str, Any],
    feasible_margins: Sequence[float],
    bank_content_digest: str,
    selection_digest: str,
) -> list[dict[str, Any]]:
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("schema") != STACKELBERG_STEERING_CHECKPOINT_SCHEMA:
        raise ValueError(f"{path} has an incompatible checkpoint schema")
    if str(checkpoint.get("graph_id")) != str(entry["graph_id"]):
        raise ValueError(f"{path} graph identity does not match")
    if str(checkpoint.get("family", "")).lower() != str(
        entry["family"]
    ).lower():
        raise ValueError(f"{path} graph family does not match")
    rows = [dict(row) for row in checkpoint.get("calibration_runs") or []]
    expected = {
        (str(defender["label"]), float(margin), int(seed))
        for defender in study["defenders"]
        for margin in feasible_margins
        for seed in study["calibration_seeds"]
    }
    observed = {
        (
            str(row.get("defender_label")),
            float(row["steering_config"]["response_margin"]),
            int(row["seed"]),
        )
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError(f"{path} calibration identities do not match")
    for row in rows:
        if int(row.get("turns", -1)) != int(study["calibration_turns"]):
            raise ValueError(f"{path} calibration horizon does not match")
        if row.get("phase") != "calibration":
            raise ValueError(f"{path} contains a non-calibration source row")
        if row.get("bank_content_digest_blake2b_128") != bank_content_digest:
            raise ValueError(f"{path} payoff bank digest does not match")
        if (
            row.get("source_corpus_selection_digest_blake2b_128")
            != selection_digest
        ):
            raise ValueError(f"{path} corpus selection digest does not match")
    return rows


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    entry: Mapping[str, Any],
    study: Mapping[str, Any],
) -> None:
    if checkpoint.get("schema") != STACKELBERG_STEERING_CHECKPOINT_SCHEMA:
        raise ValueError("steering checkpoint schema changed")
    if str(checkpoint.get("graph_id")) != str(entry["graph_id"]):
        raise ValueError("steering checkpoint graph identity changed")
    if str(checkpoint.get("family", "")).lower() != str(entry["family"]).lower():
        raise ValueError("steering checkpoint family changed")
    expected_digest = str(
        study["source_corpus_selection_digest_blake2b_128"]
    )
    evaluation_keys = {
        (str(defender["label"]), int(seed))
        for defender in study["defenders"]
        for seed in study["evaluation_seeds"]
    }
    observed_evaluation = {
        (str(row["defender_label"]), int(row["seed"]))
        for row in checkpoint.get("evaluation_runs") or []
    }
    if observed_evaluation != evaluation_keys:
        raise ValueError("steering evaluation identities changed")
    for row in (
        list(checkpoint.get("calibration_runs") or [])
        + list(checkpoint.get("evaluation_runs") or [])
    ):
        if row.get("source_corpus_selection_digest_blake2b_128") != expected_digest:
            raise ValueError("steering checkpoint selection digest changed")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
