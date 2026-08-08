"""Five-topology Prime-and-Probe defender study with graph checkpoints."""

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
from .prime_probe import (
    PrimeProbeSchedule,
    build_prime_probe_plan,
    paired_prime_probe_comparison,
    run_ordered_prime_probe,
    run_shuffled_prime_probe_control,
)


PRIME_PROBE_STUDY_SCHEMA = "histogram_prime_probe_study.v1"
PRIME_PROBE_GRAPH_CHECKPOINT_SCHEMA = "histogram_prime_probe_graph_checkpoint.v1"
DEFAULT_PRIME_PROBE_RUN_DIR = REPO_ROOT / "runs" / "histogram_prime_probe_500k"
DEFAULT_CORPUS_RUN_DIR = REPO_ROOT / "runs" / "histogram_corpus_500k"
DEFAULT_PRIME_PROBE_FAMILIES = (
    "memorex",
    "hiberniaireland",
    "layer42",
    "karen",
    "cernet",
)
DEFAULT_PRIME_PROBE_SEEDS = (20_260_716_01, 20_260_716_02, 20_260_716_03)


@dataclass(frozen=True)
class PrimeProbeDefender:
    label: str
    alice_learner_mode: str
    config_overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PRIME_PROBE_DEFENDERS = (
    PrimeProbeDefender(
        label="tsallis_inf_rv_eta1",
        alice_learner_mode=ALICE_LEARNER_TSALLIS_INF,
        config_overrides={
            "alice_learner_mode": ALICE_LEARNER_TSALLIS_INF,
            "alice_tsallis_inf_estimator": "rv",
            "alice_tsallis_inf_eta_scale": 1.0,
        },
    ),
    PrimeProbeDefender(
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


def build_prime_probe_manifest(
    corpus_manifest: Mapping[str, Any],
    *,
    schedule: PrimeProbeSchedule = PrimeProbeSchedule(),
    samples_per_cell_per_split: int = 64,
    seeds: tuple[int, ...] = DEFAULT_PRIME_PROBE_SEEDS,
) -> dict[str, Any]:
    """Freeze the five continuity cases and defender comparison."""

    if samples_per_cell_per_split < 2:
        raise ValueError("samples per cell must be at least two")
    by_family = {
        str(row["family"]).lower(): dict(row) for row in corpus_manifest["cases"]
    }
    missing = [family for family in DEFAULT_PRIME_PROBE_FAMILIES if family not in by_family]
    if missing:
        raise ValueError(f"corpus manifest is missing Prime-and-Probe cases: {missing}")
    cases = [by_family[family] for family in DEFAULT_PRIME_PROBE_FAMILIES]
    return {
        "schema": PRIME_PROBE_STUDY_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_corpus_selection_digest_blake2b_128": corpus_manifest[
            "selection_digest_blake2b_128"
        ],
        "source_corpus_plan": dict(corpus_manifest["plan"]),
        "families": list(DEFAULT_PRIME_PROBE_FAMILIES),
        "cases": cases,
        "defenders": [row.to_dict() for row in PRIME_PROBE_DEFENDERS],
        "seeds": list(map(int, seeds)),
        "schedule": schedule.to_dict(),
        "samples_per_cell_per_split": int(samples_per_cell_per_split),
        "expected_condition_runs": (
            len(cases) * len(PRIME_PROBE_DEFENDERS) * len(seeds) * 2
        ),
        "expected_paired_comparisons": (
            len(cases) * len(PRIME_PROBE_DEFENDERS) * len(seeds)
        ),
        "scientific_contract": {
            "eve_interventions": "declared physical histogram actions only",
            "reward_poisoning": False,
            "ordered_control": "identical action multiset under temporal permutation",
            "online_probe_signal": "count-valued local instrument record",
            "audit_split": "scoring only",
        },
    }


def run_prime_probe_study(
    corpus_manifest: Mapping[str, Any],
    study_manifest: Mapping[str, Any],
    *,
    output_dir: str | Path = DEFAULT_PRIME_PROBE_RUN_DIR,
    graph_workers: int = 5,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run resumable graph-owned Prime-and-Probe comparisons."""

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
        _validate_prime_probe_checkpoint(
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
            raise RuntimeError("Prime-and-Probe graph execution requires fork")
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
                executor.submit(_run_prime_probe_graph, payload): payload["entry"]
                for payload in payloads
            }
            for future in as_completed(future_map):
                checkpoint = future.result()
                complete[str(checkpoint["graph_id"])] = checkpoint
                if progress_callback is not None:
                    progress_callback(
                        f"Prime-and-Probe graph complete {len(complete)}/"
                        f"{len(study_manifest['cases'])}: {checkpoint['family']}"
                    )
    ordered_checkpoints = [
        complete[str(row["graph_id"])] for row in study_manifest["cases"]
    ]
    runs = [run for checkpoint in ordered_checkpoints for run in checkpoint["runs"]]
    comparisons = [
        row for checkpoint in ordered_checkpoints
        for row in checkpoint["paired_comparisons"]
    ]
    if len(runs) != int(study_manifest["expected_condition_runs"]):
        raise AssertionError("completed checkpoints do not contain every condition run")
    if len(comparisons) != int(study_manifest["expected_paired_comparisons"]):
        raise AssertionError("completed checkpoints do not contain every paired comparison")
    _write_jsonl_atomic(output / "results.jsonl", runs)
    _write_jsonl_atomic(output / "paired_comparisons.jsonl", comparisons)
    summary = {
        "schema": PRIME_PROBE_STUDY_SCHEMA,
        "complete": len(ordered_checkpoints) == len(study_manifest["cases"]),
        "graph_count": len(ordered_checkpoints),
        "condition_run_count": len(runs),
        "expected_condition_run_count": study_manifest["expected_condition_runs"],
        "paired_comparison_count": len(comparisons),
        "expected_paired_comparison_count": study_manifest[
            "expected_paired_comparisons"
        ],
    }
    _write_json_atomic(output / "summary.json", summary)
    return summary


def _run_prime_probe_graph(payload: Mapping[str, Any]) -> dict[str, Any]:
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
        turns=int(study["schedule"]["total_turns"]),
        requested_pairs_per_turn=int(source_plan["requested_pairs_per_turn"]),
        final_window=int(source_plan["final_window"]),
        record_every=int(study["schedule"]["record_every"]),
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

    progress("physics", "compiling equal-coverage physical planning/scoring bank")
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
        sampling_purpose="finite_horizon_stackelberg_prime_probe",
        sampling_design=(
            "feedback split supplies physical leader planning and learner draws; "
            "audit split remains reserved for scoring"
        ),
    )
    progress("planning", "solving topology-safe max-min prime distribution")
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
    comparisons: list[dict[str, Any]] = []
    for defender_row in study["defenders"]:
        defender = PrimeProbeDefender(
            label=str(defender_row["label"]),
            alice_learner_mode=str(defender_row["alice_learner_mode"]),
            config_overrides=dict(defender_row["config_overrides"]),
        )
        for seed in map(int, study["seeds"]):
            progress(
                "learning",
                f"{defender.label}, seed {seed}: ordered intervention",
            )
            config = replace(base_config, seed=seed, **defender.config_overrides)
            ordered = run_ordered_prime_probe(
                case,
                bank,
                config,
                plan,
                schedule=schedule,
                seed=seed,
                selection_digest_blake2b_128=selection_digest,
                progress_callback=lambda step, phase, label=defender.label, active_seed=seed: progress(
                    "learning",
                    f"{label}, seed {active_seed}: ordered {step}/{schedule.total_turns} ({phase})",
                ),
            )
            action_indices = ordered.pop("action_indices")
            ordered.update({
                "defender_label": defender.label,
                "bank_content_digest_blake2b_128": bank.content_digest(),
                "source_corpus_selection_digest_blake2b_128": selection_digest,
            })
            progress(
                "learning",
                f"{defender.label}, seed {seed}: shuffled action-order control",
            )
            shuffled = run_shuffled_prime_probe_control(
                case,
                bank,
                config,
                plan,
                schedule=schedule,
                seed=seed,
                ordered_action_indices=action_indices,
                selection_digest_blake2b_128=selection_digest,
                progress_callback=lambda step, phase, label=defender.label, active_seed=seed: progress(
                    "learning",
                    f"{label}, seed {active_seed}: shuffled {step}/{schedule.total_turns}",
                ),
            )
            shuffled.pop("action_indices")
            shuffled.update({
                "defender_label": defender.label,
                "bank_content_digest_blake2b_128": bank.content_digest(),
                "source_corpus_selection_digest_blake2b_128": selection_digest,
            })
            comparison = paired_prime_probe_comparison(ordered, shuffled)
            comparison["defender_label"] = defender.label
            runs.extend((ordered, shuffled))
            comparisons.append(comparison)

    diagnostics = physics.diagnostics()
    checkpoint = {
        "schema": PRIME_PROBE_GRAPH_CHECKPOINT_SCHEMA,
        "graph_id": graph_id,
        "family": case.family,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "case": case.to_dict(),
        "cache_plan": cache_plan,
        "bank": bank.to_summary(),
        "plan": plan.to_dict(),
        "physics": diagnostics,
        "runs": runs,
        "paired_comparisons": comparisons,
    }
    _write_json_atomic(
        output / "graph_checkpoints" / f"{graph_id}.json",
        checkpoint,
    )
    progress("complete", f"completed {len(runs)} condition runs")
    return checkpoint


def _validate_prime_probe_checkpoint(
    checkpoint: Mapping[str, Any],
    entry: Mapping[str, Any],
    study: Mapping[str, Any],
) -> None:
    """Reject stale or incomplete graph checkpoints before resumption."""

    if checkpoint.get("schema") != PRIME_PROBE_GRAPH_CHECKPOINT_SCHEMA:
        raise ValueError("Prime-and-Probe checkpoint schema changed")
    if str(checkpoint.get("graph_id")) != str(entry["graph_id"]):
        raise ValueError("Prime-and-Probe checkpoint graph identity changed")
    if str(checkpoint.get("family", "")).lower() != str(entry["family"]).lower():
        raise ValueError("Prime-and-Probe checkpoint family changed")
    expected_pairs = len(study["defenders"]) * len(study["seeds"])
    runs = list(checkpoint.get("runs") or [])
    comparisons = list(checkpoint.get("paired_comparisons") or [])
    if len(runs) != 2 * expected_pairs or len(comparisons) != expected_pairs:
        raise ValueError("Prime-and-Probe checkpoint is incomplete")
    expected_digest = str(study["source_corpus_selection_digest_blake2b_128"])
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
        (str(row["defender_label"]), int(row["seed"]), str(row["condition"]))
        for row in runs
    }
    if observed_keys != expected_keys:
        raise ValueError("Prime-and-Probe checkpoint condition identities changed")
    for row in runs:
        if row.get("source_corpus_selection_digest_blake2b_128") != expected_digest:
            raise ValueError("Prime-and-Probe checkpoint selection digest changed")
        if int(row["turns"]) != int(study["schedule"]["total_turns"]):
            raise ValueError("Prime-and-Probe checkpoint turn count changed")
    if not all(bool(row.get("identical_action_multiset")) for row in comparisons):
        raise ValueError("Prime-and-Probe checkpoint contains an unmatched control")


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
