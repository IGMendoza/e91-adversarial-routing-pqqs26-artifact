"""Five-topology 500k Stackelberg Eve study with graph-owned checkpoints."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from .calibration_bank import build_histogram_payoff_bank
from .config import (
    ALICE_LEARNER_LINEAR_ROUTE_EXP2,
    ALICE_LEARNER_TSALLIS_INF,
    HistogramCoLearningConfig,
    REPO_ROOT,
    STRUCTURED_EXPLORATION_G_OPTIMAL,
)
from .corpus_scale import load_corpus_scale_case
from .physics import HistogramPhysicsPool, histogram_cache_plan
from .stackelberg_eve_runtime import (
    StackelbergExploreCommitConfig,
    run_stackelberg_explore_then_commit,
)
from .stackelberg_eve import solve_stackelberg_oracle
from .topology_census import available_ram_snapshot


STACKELBERG_EVE_STUDY_SCHEMA = "histogram_stackelberg_eve_study.v1"
STACKELBERG_EVE_CHECKPOINT_SCHEMA = "histogram_stackelberg_eve_checkpoint.v1"
DEFAULT_STACKELBERG_EVE_RUN_DIR = (
    REPO_ROOT / "runs" / "histogram_stackelberg_eve_500k"
)
DEFAULT_CORPUS_RUN_DIR = REPO_ROOT / "runs" / "histogram_corpus_500k"
DEFAULT_STACKELBERG_FAMILIES = (
    "memorex",
    "hiberniaireland",
    "layer42",
    "karen",
    "cernet",
)
DEFAULT_STACKELBERG_SEEDS = (20_260_716_01, 20_260_716_02, 20_260_716_03)


@dataclass(frozen=True)
class StackelbergDefender:
    label: str
    alice_learner_mode: str
    config_overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STACKELBERG_DEFENDERS = (
    StackelbergDefender(
        label="tsallis_inf_rv_eta1",
        alice_learner_mode=ALICE_LEARNER_TSALLIS_INF,
        config_overrides={
            "alice_learner_mode": ALICE_LEARNER_TSALLIS_INF,
            "alice_tsallis_inf_estimator": "rv",
            "alice_tsallis_inf_eta_scale": 1.0,
        },
    ),
    StackelbergDefender(
        label="linear_exp2_goptimal_eta2",
        alice_learner_mode=ALICE_LEARNER_LINEAR_ROUTE_EXP2,
        config_overrides={
            "alice_learner_mode": ALICE_LEARNER_LINEAR_ROUTE_EXP2,
            "alice_exp3_eta_c": 2.0,
            "structured_exploration_mode": STRUCTURED_EXPLORATION_G_OPTIMAL,
        },
    ),
)


ProgressCallback = Callable[[str], None]


def default_stackelberg_learner_config() -> StackelbergExploreCommitConfig:
    """Return the predeclared 500k exploration and commitment schedule."""

    return StackelbergExploreCommitConfig(
        minimum_exploration_turns=50_000,
        maximum_exploration_turns=150_000,
        decision_interval=5_000,
        minimum_unique_draws_per_supported_cell=16,
        confidence_delta=0.05,
        record_every=1_000,
    )


def build_stackelberg_eve_manifest(
    corpus_manifest: Mapping[str, Any],
    *,
    turns: int = 500_000,
    learner_config: StackelbergExploreCommitConfig | None = None,
    samples_per_cell_per_split: int = 64,
    seeds: tuple[int, ...] = DEFAULT_STACKELBERG_SEEDS,
) -> dict[str, Any]:
    """Freeze the cases, defenders, information regime, and learning horizon."""

    active_learner_config = (
        learner_config or default_stackelberg_learner_config()
    )
    active_learner_config.validate_horizon(turns)
    if samples_per_cell_per_split < 2:
        raise ValueError("samples per cell must be at least two")
    by_family = {
        str(row["family"]).lower(): dict(row) for row in corpus_manifest["cases"]
    }
    missing = [
        family
        for family in DEFAULT_STACKELBERG_FAMILIES
        if family not in by_family
    ]
    if missing:
        raise ValueError(f"corpus manifest is missing Stackelberg cases: {missing}")
    cases = [by_family[family] for family in DEFAULT_STACKELBERG_FAMILIES]
    bytes_per_bank_draw = 84
    case_preflight = [
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
    return {
        "schema": STACKELBERG_EVE_STUDY_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_corpus_selection_digest_blake2b_128": corpus_manifest[
            "selection_digest_blake2b_128"
        ],
        "source_corpus_plan": dict(corpus_manifest["plan"]),
        "families": list(DEFAULT_STACKELBERG_FAMILIES),
        "cases": cases,
        "defenders": [row.to_dict() for row in STACKELBERG_DEFENDERS],
        "seeds": list(map(int, seeds)),
        "turns": int(turns),
        "learner_config": active_learner_config.to_dict(),
        "samples_per_cell_per_split": int(samples_per_cell_per_split),
        "expected_runs": len(cases) * len(STACKELBERG_DEFENDERS) * len(seeds),
        "preflight": {
            "cases": case_preflight,
            "total_nominal_route_action_cells": sum(
                row["nominal_route_action_cells"] for row in case_preflight
            ),
            "estimated_parallel_cache_mib": sum(
                row["estimated_cache_mib"] for row in case_preflight
            ),
            "estimated_parallel_feedback_audit_bank_mib": sum(
                row["estimated_feedback_audit_bank_mib"]
                for row in case_preflight
            ),
            "bank_byte_estimate_semantics": (
                "two splits times the declared route/action/draw cells and "
                "the exact dense array dtypes; Python/container overhead excluded"
            ),
            "host_ram_snapshot": available_ram_snapshot(),
        },
        "scientific_contract": {
            "leader_commitment": (
                "stationary mixed distribution over declared physical Eve actions"
            ),
            "follower": "predeclared Alice online route learner",
            "online_observations": [
                "selected Eve action and declared cost",
                "Alice selected route identity",
                "Alice public scalar service reward",
            ],
            "online_exclusions": [
                "Alice weights or policy",
                "counterfactual payoff cells",
                "feedback-bank matrix",
                "audit split",
            ],
            "oracle": "offline planning/scoring comparison only",
            "reward_poisoning": False,
            "prime_probe_control_modified": False,
        },
    }


def run_stackelberg_eve_study(
    corpus_manifest: Mapping[str, Any],
    study_manifest: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_STACKELBERG_EVE_RUN_DIR,
    graph_workers: int = 5,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run or resume the graph-owned Stackelberg comparisons."""

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
        workers = min(max(1, int(graph_workers)), len(pending))
        if "fork" not in mp.get_all_start_methods():
            raise RuntimeError("Stackelberg graph execution requires fork")
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
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            future_map = {
                executor.submit(_run_stackelberg_graph, payload): payload["entry"]
                for payload in payloads
            }
            for future in as_completed(future_map):
                checkpoint = future.result()
                complete[str(checkpoint["graph_id"])] = checkpoint
                if progress_callback is not None:
                    progress_callback(
                        f"Stackelberg graph complete {len(complete)}/"
                        f"{len(study_manifest['cases'])}: {checkpoint['family']}"
                    )
    ordered = [
        complete[str(row["graph_id"])] for row in study_manifest["cases"]
    ]
    runs = [run for checkpoint in ordered for run in checkpoint["runs"]]
    if len(runs) != int(study_manifest["expected_runs"]):
        raise AssertionError("completed checkpoints do not contain every run")
    _write_jsonl_atomic(output / "results.jsonl", runs)
    summary = {
        "schema": STACKELBERG_EVE_STUDY_SCHEMA,
        "complete": len(ordered) == len(study_manifest["cases"]),
        "graph_count": len(ordered),
        "run_count": len(runs),
        "expected_run_count": int(study_manifest["expected_runs"]),
        "response_certified_run_count": sum(
            bool(row["learned_commitment"]["response_certified"])
            for row in runs
        ),
    }
    _write_json_atomic(output / "summary.json", summary)
    return summary


def _run_stackelberg_graph(payload: Mapping[str, Any]) -> dict[str, Any]:
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
        turns=int(study["turns"]),
        requested_pairs_per_turn=int(source_plan["requested_pairs_per_turn"]),
        final_window=int(source_plan["final_window"]),
        record_every=int(study["learner_config"]["record_every"]),
        posthoc_samples_per_cell=0,
        seed=int(study["seeds"][0]),
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
        sampling_purpose="stackelberg_eve_explore_then_commit",
        sampling_design=(
            "feedback split supplies online public outcomes; audit split is "
            "reserved for oracle comparison and Stackelberg regret"
        ),
    )
    learner_config = StackelbergExploreCommitConfig(
        **study["learner_config"]
    )
    feedback_oracle = solve_stackelberg_oracle(
        bank.feedback.alice_payoff_matrix(),
        bank.action_costs,
    )
    audit_oracle = solve_stackelberg_oracle(
        bank.audit.alice_payoff_matrix(),
        bank.action_costs,
    )
    runs = []
    for defender_row in study["defenders"]:
        defender = StackelbergDefender(
            label=str(defender_row["label"]),
            alice_learner_mode=str(defender_row["alice_learner_mode"]),
            config_overrides=dict(defender_row["config_overrides"]),
        )
        for seed in map(int, study["seeds"]):
            progress(
                "learning",
                f"{defender.label}, seed {seed}: explore then commit",
            )
            config = replace(base_config, seed=seed, **defender.config_overrides)
            run = run_stackelberg_explore_then_commit(
                case,
                bank,
                config,
                learner_config,
                seed=seed,
                selection_digest_blake2b_128=selection_digest,
                progress_callback=lambda step, phase, label=defender.label, active_seed=seed: progress(
                    "learning",
                    f"{label}, seed {active_seed}: {step}/{study['turns']} ({phase})",
                ),
                feedback_oracle=feedback_oracle,
                audit_oracle=audit_oracle,
            )
            run.update({
                "defender_label": defender.label,
                "bank_content_digest_blake2b_128": bank.content_digest(),
                "source_corpus_selection_digest_blake2b_128": selection_digest,
            })
            runs.append(run)

    checkpoint = {
        "schema": STACKELBERG_EVE_CHECKPOINT_SCHEMA,
        "graph_id": graph_id,
        "family": case.family,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "case": case.to_dict(),
        "cache_plan": cache_plan,
        "bank": bank.to_summary(),
        "physics": physics.diagnostics(),
        "runs": runs,
    }
    _write_json_atomic(
        output / "graph_checkpoints" / f"{graph_id}.json",
        checkpoint,
    )
    progress("complete", f"completed {len(runs)} Stackelberg runs")
    return checkpoint


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    entry: Mapping[str, Any],
    study: Mapping[str, Any],
) -> None:
    if checkpoint.get("schema") != STACKELBERG_EVE_CHECKPOINT_SCHEMA:
        raise ValueError("Stackelberg checkpoint schema changed")
    if str(checkpoint.get("graph_id")) != str(entry["graph_id"]):
        raise ValueError("Stackelberg checkpoint graph identity changed")
    if str(checkpoint.get("family", "")).lower() != str(entry["family"]).lower():
        raise ValueError("Stackelberg checkpoint family changed")
    expected_digest = str(study["source_corpus_selection_digest_blake2b_128"])
    expected_keys = {
        (str(defender["label"]), int(seed))
        for defender in study["defenders"]
        for seed in study["seeds"]
    }
    runs = list(checkpoint.get("runs") or [])
    observed_keys = {
        (str(row["defender_label"]), int(row["seed"])) for row in runs
    }
    if observed_keys != expected_keys:
        raise ValueError("Stackelberg checkpoint run identities changed")
    for row in runs:
        if row.get("source_corpus_selection_digest_blake2b_128") != expected_digest:
            raise ValueError("Stackelberg checkpoint selection digest changed")
        if int(row["turns"]) != int(study["turns"]):
            raise ValueError("Stackelberg checkpoint turn count changed")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)
