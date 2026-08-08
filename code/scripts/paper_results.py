#!/usr/bin/env python3
r"""Paper results pipeline for the scoped QKD adversarial experiments.

The current publication suite runs six ``attack_surface_all`` conditions through
the graph-route SeQUeNCe/E91 runtime.  The all-surface mode exposes edge
``added_loss``, repeater ``swap_denial``, repeater ``memory_degradation``, edge
``intercept_resend``, and pre-swap repeater-memory Z/X measurement.  The
optional corpus pipeline uses the same runtime dispatch for attack-surface
combos, plus Path B-lite cross-graph structural DT outputs.

Output layout (timestamped so re-runs never overwrite):
  results/paper/<YYYYMMDDTHHMMSSZ>/
    figures/convergence_<group>.png            (suite runs; group = reference|main)
    figures/strategy_distribution_<group>.png  (suite runs)
    figures/q_table_heatmap_<group>.png        (suite runs)
    figures/corpus_win_rates_by_condition.png  (corpus runs)
    figures/corpus_learning_delta.png          (corpus runs)
    figures/corpus_attack_mix_by_condition.png (corpus runs)
    figures/corpus_win_rate_vs_route_depth.png (corpus runs)
    figures/corpus_dt_summary.png              (corpus runs)
    data/run_summaries.json
    data/metrics.json
    data/mitigation_evaluations.json   (mitigation quality metrics per model/algo)
    data/corpus_per_graph.json         (per-graph metrics if --corpus-manifest used)
    data/corpus_condition_summary.json (per-condition corpus aggregates)
    data/corpus_sequence_route_diagnostics.json (post-transpilation route sidecar)
    data/corpus_agg_stats.json         (primary Eve attack-choice aggregate + DT stats)
    data/corpus_agg_stats_by_combo.json (one Eve attack-choice DT per learned Eve condition)
    data/corpus_route_choice_agg_stats_by_combo.json (one Alice route-choice DT per learned Alice condition)
    data/corpus_dt_stats_by_emission.json (Eve attack-choice and Alice route-choice DT groups)
    summary.md                         (primary model summary)
    summary_prompt.txt                 (saved for audit)
    summaries/<model_slug>.md          (per-model summaries if --compare-llms)
    summaries/comparison.md            (side-by-side comparison if --compare-llms)
    summaries/mitigation_comparison.md (mitigation advice comparison if --compare-llms)
    summaries/cross_graph_explainer_<slug>.md   (structural capability explainer)
    summaries/graph_recommender_<slug>.md       (pro-Eve / pro-Alice recommender)

Usage:
  python scripts/paper_results.py
  python scripts/paper_results.py --steps 1000 --seeds 2 --no-llm
  python scripts/paper_results.py --llm phi4:14b --steps 3000 --seeds 5
  python scripts/paper_results.py --compare-llms phi4:14b,llama3.2:3b,qwen3:8b,qwen3.6:27b,qwq:32b,deepseek-r1:70b
  python scripts/paper_results.py --corpus-manifest runs/graph_corpus_paper_v1/corpus_manifest.json
  python scripts/paper_results.py --corpus-manifest runs/graph_corpus_paper_v1/corpus_manifest.json \\
      --corpus-max-episodes 300 --corpus-seed 42
  # Compiled-histogram backend (kernel + evaluated-histogram LRU caches,
  # hop-calibrated rolling reservation) on the SQLite Topology Bench corpus:
  python scripts/paper_results.py --no-suite --no-llm --no-mitigation \\
      --corpus-db runs/graph_corpus_sqlite_v1/graph_corpus.sqlite \\
      --corpus-id topology_bench_local \\
      --experiment-suite workshop_primary_six --assignment-selection top2_per_graph \\
      --runtime-backend compiled_histogram --compiled-histogram-hop-calibration-10km \\
      --workers 6
```powershell
$env:OMP_NUM_THREADS='1'; $env:OPENBLAS_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:NUMEXPR_NUM_THREADS='1'; $env:MPLCONFIGDIR="$env:TEMP\sequence_matplotlib"; .\.venv\Scripts\python.exe scripts\paper_results.py --out-dir runs\workshop_top2_primary_six_direct_e91 --run-id workshop_top2_primary_six_direct_e91_20260629 --no-suite --no-llm --no-mitigation --corpus-manifest runs\graph_corpus_paper_v2\corpus_manifest.json --assignment-selection top2_per_graph --experiment-suite workshop_primary_six --corpus-max-episodes 3000 --corpus-checkpoint-every 1 --workers 6 --runtime-backend direct_e91_transcript --corpus-dt-refit-mode off
```
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLCONFIGDIR", str(_ROOT / ".cache" / "matplotlib"))

def _configure_stdio_for_windows() -> None:
    """Avoid cp1252 crashes when PowerShell uses a legacy code page."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


_configure_stdio_for_windows()

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(_ROOT))

from sequence_game.claims import REPEATER_RUNTIME
from sequence_game.eve.feedback import eve_feedback_from_heralded_service
from sequence_game.gui_demo.demo_runner import run_binary_collision
import sequence_game.corpus.runner as corpus_runner_module
from sequence_game.corpus.runner import (
    LEARNED_ALICE_ALGOS,
    LEARNED_EVE_ALGOS,
    ROUTE_CHOICE_FEATURE_NAMES,
    run_one_corpus_graph,
)
from sequence_game.corpus.sqlite_store import (
    SQLiteGraphCorpusSource,
    load_completed_corpus_case_records,
    write_corpus_case_record,
    write_worker_status_payload,
)
from sequence_game.corpus.e91_runtime_game import (
    E91_RUNTIME_BACKEND,
    demo_topology_to_ir_dict,
    run_e91_graph_runtime_game,
)
from sequence_game.xai.cross_graph_dt import fit_cross_graph_dt, aggregate_corpus_stats
from sequence_game.xai.dt_refit_observer import DynamicDTRefitConfig
from sequence_game.xai.prompt_builder import (
    build_cross_graph_explainer_prompt,
    build_graph_recommender_prompt,
    build_route_choice_explainer_prompt,
    build_route_choice_recommender_prompt,
)

# ── Topology ──────────────────────────────────────────────────────────────────
# 2-route diamond: alice ─r0─ bob  (shorter)  and  alice ─r1─ bob  (longer).
# cautious-greedy Alice prefers r0 → oracle Eve always attacks r0.
# Both repeaters have bottleneck_score=0.5 (each on exactly one route).
PAPER_TOPOLOGY = {
    "nodes": [
        {"id": "alice", "x": 0,   "y": 0},
        {"id": "r0",    "x": 100, "y": -40},
        {"id": "r1",    "x": 100, "y": 40},
        {"id": "bob",   "x": 200, "y": 0},
    ],
    "edges": [
        {"u": "alice", "v": "r0",  "length": 50.0},
        {"u": "r0",    "v": "bob", "length": 50.0},
        {"u": "alice", "v": "r1",  "length": 80.0},
        {"u": "r1",    "v": "bob", "length": 80.0},
    ],
    "warnings": [],
}

# ── Experiment suite (contracted publication spec) ────────────────────────────
# 6 combinations across 2 groups; key = f"{game_mode}__{eve}__{alice}"
#
# group "reference" — attack_surface_all, control/sanity baselines
# group "main"      — attack_surface_all, deep-Q Eve primary results

EXPERIMENT_SUITE: list[dict] = [
    # ── Full-simulator control references (attack_surface_all) ─────────────────
    {"game_mode": "attack_surface_all",  "eve": "random",            "alice": "random",          "group": "reference"},
    {"game_mode": "attack_surface_all",  "eve": "deep_q",            "alice": "random",          "group": "reference"},
    {"game_mode": "attack_surface_all",  "eve": "route_aware_oracle","alice": "cautious_greedy", "group": "reference"},
    {"game_mode": "attack_surface_all",  "eve": "random",            "alice": "cautious_greedy", "group": "reference"},
    # ── Main objects ───────────────────────────────────────────────────────────
    {"game_mode": "attack_surface_all",  "eve": "deep_q",            "alice": "cautious_greedy", "group": "main"},
    {"game_mode": "attack_surface_all",  "eve": "deep_q",            "alice": "deep_q",          "group": "main"},
]

def _combo_key(combo: dict) -> str:
    return f"{combo['game_mode']}__{combo['eve']}__{combo['alice']}"

# Inject key into each combo (avoids recomputing everywhere)
for _c in EXPERIMENT_SUITE:
    _c["key"] = _combo_key(_c)

EXPERIMENT_SUITE_WORKSHOP_PRIMARY_SIX = "workshop_primary_six"
EXPERIMENT_SUITE_PRIMARY_SIX = "primary_six"
EXPERIMENT_SUITE_PUBLICATION_SIX = "publication_six"
EXPERIMENT_SUITE_WORKSHOP_MAIN_TWO = "workshop_main_two"
EXPERIMENT_SUITE_MAIN_TWO = "main_two"
EXPERIMENT_SUITE_PROJECT_MAIN_TWO = "project_main_two"
EXPERIMENT_SUITE_MAIN_TWO_COMBOS = [
    combo for combo in EXPERIMENT_SUITE
    if combo.get("group") == "main"
]
EXPERIMENT_SUITE_ALIASES = {
    EXPERIMENT_SUITE_WORKSHOP_PRIMARY_SIX: EXPERIMENT_SUITE,
    EXPERIMENT_SUITE_PRIMARY_SIX: EXPERIMENT_SUITE,
    EXPERIMENT_SUITE_PUBLICATION_SIX: EXPERIMENT_SUITE,
    EXPERIMENT_SUITE_WORKSHOP_MAIN_TWO: EXPERIMENT_SUITE_MAIN_TWO_COMBOS,
    EXPERIMENT_SUITE_MAIN_TWO: EXPERIMENT_SUITE_MAIN_TWO_COMBOS,
    EXPERIMENT_SUITE_PROJECT_MAIN_TWO: EXPERIMENT_SUITE_MAIN_TWO_COMBOS,
}

ASSIGNMENT_SELECTION_ALL = "all"
ASSIGNMENT_SELECTION_TOP2_PER_GRAPH = "top2_per_graph"
ASSIGNMENT_SELECTION_TOP2_LIMIT = 2
ASSIGNMENT_SELECTION_TOP2_MANIFEST_ORDER_RULE = "top2_per_graph_manifest_order"
_ASSIGNMENT_RANKING_FIELDS: tuple[tuple[str, str], ...] = (
    ("assignment_rank", "ascending"),
    ("rank", "ascending"),
    ("selection_rank", "ascending"),
    ("quality_rank", "ascending"),
    ("route_diversity_rank", "ascending"),
    ("assignment_score", "descending"),
    ("score", "descending"),
    ("quality_score", "descending"),
    ("route_diversity_score", "descending"),
    ("diversity_score", "descending"),
)

_EVE_COLORS = {
    "random":             "#8e9aaf",
    "q_learning":         "#41d19a",
    "deep_q":             "#5b9bd5",
    "route_aware_oracle": "#d13b3b",
}
# Corpus condition colors keyed by the (eve, alice) pair — the condition is
# the entity, so its hue must stay fixed no matter which subset is plotted.
# Palette validated in suite order for colorblind separation (worst adjacent
# deltaE 21.2); aqua/yellow sit below 3:1 surface contrast, so corpus figures
# always pair color with direct labels and Alice marker shapes.
_CONDITION_COLORS = {
    ("random",             "random"):          "#4a3aa7",
    ("deep_q",             "random"):          "#1baf7a",
    ("route_aware_oracle", "cautious_greedy"): "#e34948",
    ("random",             "cautious_greedy"): "#eda100",
    ("deep_q",             "cautious_greedy"): "#eb6834",
    ("deep_q",             "deep_q"):          "#2a78d6",
}
_ALICE_LINESTYLE = {
    "cautious_greedy": "-",
    "q_learning":      "--",
    "deep_q":          ":",
    "random":          "-.",
}
_MODE_MARKER = {
    "binary_collision":    "o",
    "attack_surface_dos":  "s",
    "attack_surface_mixed":"^",
    "attack_surface_all":  "D",
}
_EVE_LABEL = {
    "random":             "Random Eve",
    "q_learning":         "Q-learning Eve",
    "deep_q":             "DQN Eve",
    "route_aware_oracle": "Oracle Eve",
}
_ALICE_LABEL = {
    "cautious_greedy": "cautious-greedy Alice",
    "q_learning":      "Q-learning Alice",
    "deep_q":          "DQN Alice",
    "random":          "random Alice",
}
_MODE_LABEL = {
    "binary_collision":    "binary-collision",
    "attack_surface_dos":  "DoS surface",
    "attack_surface_mixed":"mixed surface",
    "attack_surface_all":  "all-six attack surface",
}

# Fixed attack-type stacking order + colorblind-safe (Okabe-Ito) colors.
# Legacy binary-collision kinds map onto the matching semantic family.
_ATTACK_TYPE_ORDER = [
    "no_attack", "none",
    "edge_dos", "edge_collision", "edge",
    "swap_denial", "node_dos", "node_collision", "node",
    "memory_degradation",
    "edge_information_probe", "information_probe",
    "repeater_memory_measure_Z",
    "repeater_memory_measure_X",
]
_ATTACK_TYPE_COLORS = {
    "no_attack": "#cccccc", "none": "#cccccc",
    "edge_dos": "#e69f00", "edge_collision": "#e69f00", "edge": "#e69f00",
    "swap_denial": "#56b4e9", "node_dos": "#56b4e9",
    "node_collision": "#56b4e9", "node": "#56b4e9",
    "memory_degradation": "#009e73",
    "edge_information_probe": "#d55e00", "information_probe": "#d55e00",
    "repeater_memory_measure_Z": "#0072b2",
    "repeater_memory_measure_X": "#cc79a7",
}


def _attack_types_in_fixed_order(present: set[str]) -> list[str]:
    ordered = [kind for kind in _ATTACK_TYPE_ORDER if kind in present]
    ordered.extend(sorted(present - set(_ATTACK_TYPE_ORDER)))
    return ordered


def _suite_label(combo: dict) -> str:
    return f"{_EVE_LABEL[combo['eve']]} vs {_ALICE_LABEL[combo['alice']]} [{_MODE_LABEL[combo['game_mode']]}]"

def _suite_color(combo: dict) -> str:
    return _EVE_COLORS[combo["eve"]]

def _suite_linestyle(combo: dict) -> str:
    return _ALICE_LINESTYLE[combo["alice"]]

# Legacy single-Alice / all-EVE constants kept for LLM prompt / mitigation helpers
ALICE_ALGO = "cautious_greedy"
EVE_ALGOS  = ["random", "q_learning", "deep_q", "route_aware_oracle"]
COLORS = _EVE_COLORS
LABELS = _EVE_LABEL


# ── Metrics helpers ───────────────────────────────────────────────────────────
# binary_collision: "hit" / "miss"
# attack_surface_* runtime rows use selected-path E91 outcomes:
# sequence backend: "accepted" / "delivery_failure" / "qber_abort" /
# "information_exposure"; direct/compiled backends additionally emit
# "chsh_abort" and report key exposure via the per-step ``eve_win`` /
# ``information_exposed`` flags instead of a dedicated public outcome.
# Eve-win accounting must match the reward definition the agents trained on
# (direct_e91_transcript: eve_win = outcome != "accepted" or exposed).
_EVE_WIN_OUTCOMES = {
    "hit",
    "delivery_failure",
    "qber_abort",
    "chsh_abort",
    "information_exposure",
}
_CORPUS_ROW_PAYLOAD_KEYS = frozenset({"structural_rows", "route_choice_rows"})
_CASE_HASH_DIGEST_SIZE = 16
RUNTIME_BACKEND_SEQUENCE = "sequence"
RUNTIME_BACKEND_DIRECT_E91 = "direct_e91_transcript"
DIRECT_E91_RUNTIME_BACKEND = "direct_e91_transcript"
RUNTIME_BACKEND_COMPILED_HISTOGRAM = "compiled_histogram"
COMPILED_HISTOGRAM_RUNTIME_BACKEND = "compiled_histogram_transcript"
DIRECT_E91_PREPARED_CACHE_KEY_VERSION = "paper_results_direct_e91_prepared_v1"
CASE_PROVENANCE_FIELDS = (
    "case_id",
    "graph_id",
    "assignment_id",
    "combo_key",
    "backend",
    "expected_backend",
    "sequence_backend_called",
    "sequence_trial_count",
    "topology_hash",
    "action_space_hash",
    "config_hash",
    "seed",
    "num_steps",
    "route_policy",
    "status",
)


class CaseValidationError(RuntimeError):
    """Raised when a paper-run case violates provenance/backend requirements."""


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _stable_hash(payload: Any, *, digest_size: int = _CASE_HASH_DIGEST_SIZE) -> str:
    return hashlib.blake2b(
        _canonical_json_bytes(payload),
        digest_size=digest_size,
    ).hexdigest()


def _stable_file_hash(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=_CASE_HASH_DIGEST_SIZE)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_backend_for_combo(
    combo: dict[str, Any],
    runtime_backend: str = RUNTIME_BACKEND_SEQUENCE,
) -> str:
    mode = str(combo.get("game_mode", ""))
    if mode.startswith("attack_surface_"):
        if runtime_backend == RUNTIME_BACKEND_DIRECT_E91:
            return DIRECT_E91_RUNTIME_BACKEND
        if runtime_backend == RUNTIME_BACKEND_COMPILED_HISTOGRAM:
            return COMPILED_HISTOGRAM_RUNTIME_BACKEND
        return "sequence"
    return "control"


def _planned_backend_for_combo(
    combo: dict[str, Any],
    runtime_backend: str = RUNTIME_BACKEND_SEQUENCE,
) -> str:
    expected = _expected_backend_for_combo(combo, runtime_backend)
    if expected == "sequence":
        return E91_RUNTIME_BACKEND
    if expected == DIRECT_E91_RUNTIME_BACKEND:
        return DIRECT_E91_RUNTIME_BACKEND
    if expected == COMPILED_HISTOGRAM_RUNTIME_BACKEND:
        return COMPILED_HISTOGRAM_RUNTIME_BACKEND
    return "binary_collision_control"


def _coerce_nonnegative_int(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _runtime_metrics_from_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {}
    summary = result.get("summary") or {}
    if not isinstance(summary, dict):
        return {}
    metrics = summary.get("metrics") or {}
    return metrics if isinstance(metrics, dict) else {}


def _combo_key_from_combo(combo: dict[str, Any]) -> str:
    return str(combo.get("key") or _combo_key(combo))


def _make_case_id(
    *,
    graph_id: str,
    assignment_id: str,
    combo_key: str,
    seed: int,
    num_steps: int,
    route_policy: str,
    backend: str,
    action_space_hash: str,
    config_hash: str,
) -> str:
    return _stable_hash({
        "graph_id": graph_id,
        "assignment_id": assignment_id,
        "combo_key": combo_key,
        "seed": int(seed),
        "steps": int(num_steps),
        "route_policy": route_policy,
        "backend": backend,
        "action_space_hash": action_space_hash,
        "config_hash": config_hash,
    })


def _make_case_plan(
    *,
    graph_id: str,
    assignment_id: str | None,
    combo: dict[str, Any],
    seed: int,
    num_steps: int,
    route_policy: str,
    topology_hash: str,
    action_space_hash: str,
    config_hash: str,
    runtime_backend: str = RUNTIME_BACKEND_SEQUENCE,
) -> dict[str, Any]:
    combo_key = _combo_key_from_combo(combo)
    backend = _planned_backend_for_combo(combo, runtime_backend)
    plan = {
        "graph_id": str(graph_id),
        "assignment_id": str(assignment_id or ""),
        "combo_key": combo_key,
        "backend": backend,
        "expected_backend": _expected_backend_for_combo(combo, runtime_backend),
        "sequence_backend_called": False,
        "sequence_trial_count": 0,
        "topology_hash": str(topology_hash),
        "action_space_hash": str(action_space_hash),
        "config_hash": str(config_hash),
        "seed": int(seed),
        "num_steps": int(num_steps),
        "route_policy": str(route_policy),
        "status": "planned",
    }
    plan["case_id"] = _make_case_id(
        graph_id=plan["graph_id"],
        assignment_id=plan["assignment_id"],
        combo_key=combo_key,
        seed=int(seed),
        num_steps=int(num_steps),
        route_policy=str(route_policy),
        backend=backend,
        action_space_hash=str(action_space_hash),
        config_hash=str(config_hash),
    )
    return plan


def _case_provenance_from_metrics(
    case_plan: dict[str, Any],
    metrics: dict[str, Any],
    *,
    status: str,
) -> dict[str, Any]:
    sequence_backend_called = bool(metrics.get("sequence_runtime_executed"))
    runtime_case_count = _coerce_nonnegative_int(metrics.get("runtime_case_count"))
    provenance = {
        field: case_plan.get(field)
        for field in CASE_PROVENANCE_FIELDS
        if field in case_plan
    }
    provenance.update({
        "sequence_backend_called": sequence_backend_called,
        "sequence_trial_count": runtime_case_count if sequence_backend_called else 0,
        "runtime_backend_called": bool(
            sequence_backend_called
            or metrics.get("direct_e91_runtime_executed")
            or metrics.get("compiled_histogram_runtime_executed")
        ),
        "runtime_case_count": runtime_case_count,
        "status": status,
    })
    observed_backend = metrics.get("backend")
    if observed_backend is not None:
        provenance["observed_backend"] = str(observed_backend)
    return provenance


def _require_case_provenance(record: dict[str, Any]) -> None:
    missing = [field for field in CASE_PROVENANCE_FIELDS if field not in record]
    if missing:
        raise CaseValidationError(
            "case provenance missing required field(s): " + ", ".join(missing)
        )


def _validate_case_record(record: dict[str, Any]) -> None:
    _require_case_provenance(record)
    if record.get("status") != "completed":
        raise CaseValidationError(
            f"case {record.get('case_id')} has non-completed status {record.get('status')!r}"
        )
    if record.get("expected_backend") == "sequence":
        if not record.get("sequence_backend_called"):
            raise CaseValidationError(
                f"case {record.get('case_id')} did not call the SeQUeNCe backend"
            )
        if _coerce_nonnegative_int(record.get("sequence_trial_count")) <= 0:
            raise CaseValidationError(
                f"case {record.get('case_id')} recorded zero SeQUeNCe trials"
            )


def _assert_sequence_backend_result(result: dict[str, Any]) -> None:
    metrics = _runtime_metrics_from_result(result)
    if not metrics.get("sequence_runtime_executed"):
        raise CaseValidationError("expected SeQUeNCe backend but it was not called")
    if _coerce_nonnegative_int(metrics.get("runtime_case_count")) <= 0:
        raise CaseValidationError("expected SeQUeNCe backend to execute at least one trial")


def _outcome_counts_from_steps(steps: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(step.get("public_outcome") or "unknown") for step in steps))


def _metric_split_from_counts(
    counts: dict[str, Any],
    *,
    information_exposures: Any = None,
) -> dict[str, float]:
    numeric_counts = {
        str(key): _coerce_nonnegative_int(value)
        for key, value in (counts or {}).items()
    }
    total = sum(numeric_counts.values())
    if total <= 0:
        return {
            "delivery_rate": 0.0,
            "delivery_failure_rate": 0.0,
            "qber_abort_rate": 0.0,
            "chsh_abort_rate": 0.0,
            "info_exposure_rate": 0.0,
            "secure_delivery_rate": 0.0,
            "eve_win_rate": 0.0,
        }
    accepted = numeric_counts.get("accepted", 0)
    outcome_exposures = numeric_counts.get("information_exposure", 0)
    # Direct/compiled backends report key exposure as a flag beside an
    # "accepted" public outcome rather than as its own outcome; fold the flag
    # count back out of the "accepted" bucket when the outcome key is absent.
    flag_exposures = 0
    if outcome_exposures == 0 and information_exposures is not None:
        flag_exposures = min(accepted, _coerce_nonnegative_int(information_exposures))
    exposures = outcome_exposures + flag_exposures
    secure_deliveries = accepted - flag_exposures
    delivery_failures = numeric_counts.get("delivery_failure", 0)
    qber_aborts = numeric_counts.get("qber_abort", 0)
    chsh_aborts = numeric_counts.get("chsh_abort", 0)
    eve_wins = (
        sum(numeric_counts.get(outcome, 0) for outcome in _EVE_WIN_OUTCOMES)
        + flag_exposures
    )
    return {
        "delivery_rate": (secure_deliveries + exposures) / total,
        "delivery_failure_rate": delivery_failures / total,
        "qber_abort_rate": qber_aborts / total,
        "chsh_abort_rate": chsh_aborts / total,
        "info_exposure_rate": exposures / total,
        "secure_delivery_rate": secure_deliveries / total,
        "eve_win_rate": eve_wins / total,
    }


def _metric_split_from_steps(steps: list[dict[str, Any]]) -> dict[str, float]:
    flag_exposures = sum(
        1 for step in steps
        if step.get("information_exposed")
        and str(step.get("public_outcome")) == "accepted"
    )
    return _metric_split_from_counts(
        _outcome_counts_from_steps(steps),
        information_exposures=flag_exposures,
    )


_ATOMIC_REPLACE_ATTEMPTS = 25
_ATOMIC_REPLACE_INITIAL_DELAY_S = 0.05
_ATOMIC_REPLACE_MAX_DELAY_S = 1.0


def _replace_with_retry(src: Path, dst: Path) -> None:
    delay = _ATOMIC_REPLACE_INITIAL_DELAY_S
    for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt + 1 >= _ATOMIC_REPLACE_ATTEMPTS:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, _ATOMIC_REPLACE_MAX_DELAY_S)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    replaced = False
    try:
        tmp.write_text(text, encoding="utf-8")
        _replace_with_retry(tmp, path)
        replaced = True
    finally:
        if not replaced and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, default=str))


def _failure_record(
    case_plan: dict[str, Any],
    *,
    error: str,
    traceback_text: str,
) -> dict[str, Any]:
    record = {
        field: case_plan.get(field)
        for field in CASE_PROVENANCE_FIELDS
        if field in case_plan
    }
    record.update({
        "sequence_backend_called": False,
        "sequence_trial_count": 0,
        "status": "failed",
        "error": error,
        "traceback": traceback_text,
    })
    return record


def _write_failed_case(failed_cases_file: Path, record: dict[str, Any]) -> None:
    failed_cases_file.parent.mkdir(parents=True, exist_ok=True)
    with failed_cases_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def _is_eve_win(step: dict) -> bool:
    # Direct/compiled backend steps carry the authoritative per-step flag the
    # reward was computed from; outcome-set matching covers the sequence E91
    # and binary-collision backends, which do not emit it.
    flag = step.get("eve_win")
    if isinstance(flag, bool):
        return flag
    return step.get("public_outcome") in _EVE_WIN_OUTCOMES


def _rolling_hit_rate(steps: list[dict], window: int) -> np.ndarray:
    hits = np.array([1.0 if _is_eve_win(s) else 0.0 for s in steps])
    if len(hits) == 0:
        return np.array([])
    if len(hits) < window:
        return np.array([float(np.mean(hits))])
    kernel = np.ones(window) / window
    return np.convolve(hits, kernel, mode="valid")


def _action_dist(steps: list[dict]) -> dict[str, float]:
    """Distribution of Eve attack types (falls back to target kind for legacy rows)."""
    counts: Counter = Counter()
    for step in steps:
        action = step.get("eve_action") or {}
        label = action.get("attack_type") or action.get("kind", "none")
        counts[str(label)] += 1
    total = len(steps) or 1
    return {k: v / total for k, v in counts.items()}


def _final_hit_rate(steps: list[dict], tail_frac: float = 0.3) -> float:
    tail_start = int(len(steps) * (1 - tail_frac))
    tail = steps[tail_start:]
    if not tail:
        return 0.0
    return float(np.mean([1.0 if _is_eve_win(s) else 0.0 for s in tail]))


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_sequence_runtime_combo(
    game_mode: str,
    eve: str,
    alice: str,
    n_steps: int,
    seed: int,
    *,
    attack_cost: float = 0.05,
    final_window_size: int | None = None,
) -> dict:
    """Run an attack-surface paper-suite combo through graph-route E91 runtime."""

    if game_mode not in {"attack_surface_dos", "attack_surface_mixed", "attack_surface_all"}:
        raise ValueError(
            f"run_sequence_runtime_combo only supports attack_surface_* modes, got {game_mode!r}"
        )
    if eve == "route_aware_oracle" and alice != "cautious_greedy":
        raise ValueError(
            "route-aware oracle Eve is only defined against cautious-greedy Alice"
        )
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    result = run_e91_graph_runtime_game(
        demo_topology_to_ir_dict(PAPER_TOPOLOGY, graph_id="paper_diamond_graph"),
        "alice",
        "bob",
        routes=None,
        game_mode=game_mode,
        eve_algo=eve,
        alice_algo=alice,
        num_steps=n_steps,
        seed=seed,
        final_window_size=(
            min(200, n_steps) if final_window_size is None else final_window_size
        ),
        attack_cost=attack_cost,
    )
    _assert_sequence_backend_result(result)
    return result


def run_combo(game_mode: str, eve: str, alice: str, n_steps: int, seed: int) -> dict:
    if game_mode != "binary_collision":
        return run_sequence_runtime_combo(game_mode, eve, alice, n_steps, seed)
    return run_binary_collision(
        PAPER_TOPOLOGY, "alice", "bob",
        eve_algo=eve,
        alice_algo=alice,
        control_model=game_mode,
        num_steps=n_steps,
        seed=seed,
        final_window_size=min(200, n_steps),
    )


def _suite_case_plan(combo: dict[str, Any], *, seed: int, n_steps: int) -> dict[str, Any]:
    topology_hash = _stable_hash(PAPER_TOPOLOGY)
    route_policy = "paper_diamond_default_routes"
    action_space_hash = _stable_hash({
        "suite": "paper_diamond",
        "combo_key": _combo_key_from_combo(combo),
        "game_mode": combo["game_mode"],
        "route_policy": route_policy,
        "topology_hash": topology_hash,
    })
    config_hash = _stable_hash({
        "suite": "paper_results",
        "combo": combo,
        "num_steps": int(n_steps),
        "route_policy": route_policy,
    })
    return _make_case_plan(
        graph_id="paper_diamond_graph",
        assignment_id="paper_suite",
        combo=combo,
        seed=seed,
        num_steps=n_steps,
        route_policy=route_policy,
        topology_hash=topology_hash,
        action_space_hash=action_space_hash,
        config_hash=config_hash,
    )


def _attach_suite_provenance(result: dict[str, Any], case_plan: dict[str, Any]) -> dict[str, Any]:
    provenance = _case_provenance_from_metrics(
        case_plan,
        _runtime_metrics_from_result(result),
        status="completed",
    )
    result.update(provenance)
    result["case_provenance"] = dict(provenance)
    if isinstance(result.get("summary"), dict):
        result["summary"].update(provenance)
        result["summary"]["case_provenance"] = dict(provenance)
    _validate_case_record(result)
    return result


def run_suite(
    n_steps: int,
    n_seeds: int,
    *,
    allow_failed_cases: bool = False,
    failed_cases_file: Path | None = None,
) -> dict[str, list[dict]]:
    """Run all configured combos in EXPERIMENT_SUITE. Returns dict keyed by combo key."""
    all_results: dict[str, list[dict]] = {}
    for combo in EXPERIMENT_SUITE:
        key   = combo["key"]
        label = _suite_label(combo)
        print(f"\n  [{key}] {label}")
        print(f"    {n_seeds} seed(s) × {n_steps} steps …")
        results = []
        for i in range(n_seeds):
            seed = i * 100 + 7
            case_plan = _suite_case_plan(combo, seed=seed, n_steps=n_steps)
            t0 = time.perf_counter()
            try:
                result = run_combo(combo["game_mode"], combo["eve"], combo["alice"], n_steps, seed)
                result = _attach_suite_provenance(result, case_plan)
                elapsed = time.perf_counter() - t0
                hr = _final_hit_rate(result["steps"])
                print(f"    seed={seed}  eve_win(tail)={hr:.3f}  {elapsed:.1f}s")
                results.append(result)
            except Exception as exc:
                failure = _failure_record(
                    case_plan,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback_text=traceback.format_exc(),
                )
                if failed_cases_file is not None:
                    _write_failed_case(failed_cases_file, failure)
                print(f"    seed={seed} FAILED: {exc}")
                if not allow_failed_cases:
                    raise
        if results:
            all_results[key] = results
        elif not allow_failed_cases:
            raise RuntimeError(f"all seeds failed for {key}")
    return all_results


def run_all(n_steps: int, n_seeds: int) -> dict[str, list[dict]]:
    """Legacy: run all EVE_ALGOS with fixed cautious_greedy Alice in binary_collision."""
    all_results: dict[str, list[dict]] = {}
    for algo in EVE_ALGOS:
        print(f"\n  [{algo}] running {n_seeds} seed(s) × {n_steps} steps …")
        results = []
        for i in range(n_seeds):
            seed = i * 100 + 7
            t0 = time.perf_counter()
            try:
                result = run_combo("binary_collision", algo, ALICE_ALGO, n_steps, seed)
                elapsed = time.perf_counter() - t0
                hr = _final_hit_rate(result["steps"])
                print(f"    seed={seed}  eve_win(tail)={hr:.3f}  {elapsed:.1f}s")
                results.append(result)
            except Exception as exc:
                print(f"    seed={seed} FAILED: {exc}")
        if results:
            all_results[algo] = results
        else:
            print(f"    WARNING: all seeds failed for {algo}, skipping")
    return all_results


# ── Aggregate metrics ─────────────────────────────────────────────────────────

def aggregate_metrics(all_results: dict[str, list[dict]], window: int) -> dict:
    """Generic aggregation — works for both suite keys and legacy algo keys."""
    metrics: dict[str, dict] = {}
    for key, results in all_results.items():
        curves, final_hrs, dists = [], [], []
        split_metrics: dict[str, list[float]] = defaultdict(list)
        q_entries, q_names = None, []
        is_dqn_key = "deep_q" in key
        for result in results:
            steps = result["steps"]
            curves.append(_rolling_hit_rate(steps, window))
            final_hrs.append(_final_hit_rate(steps))
            for metric_key, value in _metric_split_from_steps(steps).items():
                split_metrics[metric_key].append(float(value))
            tail_start = int(len(steps) * 0.7)
            dists.append(_action_dist(steps[tail_start:]))
            if q_entries is None and is_dqn_key:
                raw = result["summary"]["q_tables"]["eve"].get("raw", {})
                q_entries = raw.get("entries", [])
                q_names = result["summary"]["q_tables"]["eve"].get("action_names", [])

        min_len = min((len(c) for c in curves), default=0)
        aligned = np.array([c[:min_len] for c in curves]) if min_len else np.array([])
        all_kinds = sorted({k for d in dists for k in d})
        merged_dist = {
            k: float(np.mean([d.get(k, 0.0) for d in dists]))
            for k in all_kinds
        }
        metrics[key] = {
            "final_hit_rate_mean": float(np.mean(final_hrs)),
            "final_hit_rate_std":  float(np.std(final_hrs)),
            "n_seeds": len(results),
            "rolling_mean": aligned.mean(axis=0).tolist() if min_len else [],
            "rolling_std":  aligned.std(axis=0).tolist() if min_len else [],
            "action_distribution": merged_dist,
            "q_table_entries": q_entries,
            "q_action_names":  q_names,
        }
        for metric_key, values in split_metrics.items():
            metrics[key][metric_key] = float(np.mean(values)) if values else 0.0
            metrics[key][f"{metric_key}_std"] = float(np.std(values)) if values else 0.0
    return metrics


def _write_metrics_files(metrics: dict, dat_dir: Path) -> None:
    """Save full metrics for resume and slim metrics for easier inspection."""
    with open(dat_dir / "metrics_full.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    slim = {
        k: {fk: fv for fk, fv in m.items() if fk not in ("rolling_mean", "rolling_std")}
        for k, m in metrics.items()
    }
    with open(dat_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, default=str)


def _load_resume_metrics(dat_dir: Path) -> tuple[dict | None, bool]:
    """Load saved metrics for --resume-from.

    Returns (metrics, has_rolling_curves). Older runs only have metrics.json,
    which is enough for summary prompts but not convergence plot regeneration.
    """
    full_path = dat_dir / "metrics_full.json"
    if full_path.exists():
        return json.loads(full_path.read_text(encoding="utf-8")), True
    slim_path = dat_dir / "metrics.json"
    if slim_path.exists():
        return json.loads(slim_path.read_text(encoding="utf-8")), False
    return None, False


def _metrics_have_rolling_curves(metrics: dict) -> bool:
    return bool(metrics) and all(
        "rolling_mean" in m and "rolling_std" in m
        for m in metrics.values()
    )


def _metrics_seed_count(metrics: dict) -> int:
    counts = [int(m.get("n_seeds", 0)) for m in metrics.values()]
    return min(counts) if counts else 0


def _print_metric_summary(metrics: dict) -> None:
    combo_by_key = {c["key"]: c for c in EXPERIMENT_SUITE}
    for k, m in metrics.items():
        lbl = _suite_label(combo_by_key[k]) if k in combo_by_key else LABELS.get(k, k)
        hr = m["final_hit_rate_mean"]
        std = m["final_hit_rate_std"]
        print(f"  {lbl:<60}  eve_win={hr:.3f} ± {std:.3f}")


# ── Figures ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}")


def _combos_for_experiment_suite(suite_name: str) -> list[dict]:
    """Return a named experiment suite as detached combo dictionaries."""
    try:
        suite = EXPERIMENT_SUITE_ALIASES[suite_name]
    except KeyError as exc:
        available = ", ".join(sorted(EXPERIMENT_SUITE_ALIASES))
        raise ValueError(
            f"unknown experiment suite {suite_name!r}. Available suites: {available}"
        ) from exc
    return [dict(combo) for combo in suite]


def _combos_for_keys(
    keys: list[str],
    suite: list[dict] | None = None,
) -> list[dict]:
    """Return EXPERIMENT_SUITE entries matching the given keys (preserving order)."""
    available_suite = EXPERIMENT_SUITE if suite is None else suite
    by_key = {c["key"]: c for c in available_suite}
    missing = [key for key in keys if key not in by_key]
    if missing:
        available = ", ".join(sorted(by_key))
        raise ValueError(
            "unknown corpus combo key(s): "
            f"{', '.join(missing)}. Available keys: {available}"
        )
    return [dict(by_key[k]) for k in keys]


def _combos_for_corpus_record(
    combos: list[dict],
    *,
    record_index: int,
    record: dict | None = None,
    reference_max_graphs: int | None = None,
    reference_graph_ids: frozenset[str] | None = None,
) -> list[dict]:
    """Return combos for one accepted corpus record.

    The contracted publication run keeps the learned/main conditions on the
    full corpus while allowing reference controls to run only on a fixed
    synthetic category sample of unique graph IDs.
    """
    if reference_max_graphs is None:
        return combos
    if reference_graph_ids is None:
        reference_enabled = record_index < reference_max_graphs
    else:
        graph_id = str((record or {}).get("graph_id", ""))
        reference_enabled = graph_id in reference_graph_ids
    return [
        combo for combo in combos
        if combo.get("group") != "reference" or reference_enabled
    ]


def _planned_corpus_case_count(
    selected_assignment_count: int,
    combos: list[dict],
) -> int:
    return int(selected_assignment_count) * len(combos)


def _reference_graph_prefix(
    records: list[dict],
    reference_max_graphs: int | None,
) -> frozenset[str] | None:
    if reference_max_graphs is None:
        return None
    if _is_synthetic_corpus(records):
        return _reference_graphs_per_synthetic_category(records, reference_max_graphs)
    selected: list[str] = []
    seen: set[str] = set()
    for record in records:
        graph_id = str(record.get("graph_id", ""))
        if graph_id in seen:
            continue
        if len(selected) >= reference_max_graphs:
            break
        selected.append(graph_id)
        seen.add(graph_id)
    return frozenset(selected)


def _is_synthetic_corpus(records: list[dict]) -> bool:
    return bool(records) and all(
        str(record.get("graph_id", "")).find("_synthetic_") >= 0
        or str(record.get("family", "")).startswith("synthetic_")
        for record in records
    )


def _corpus_graph_identity(record: dict, index: int | None = None) -> str:
    graph_id = record.get("graph_id")
    if graph_id not in (None, ""):
        return str(graph_id)
    topology_path = record.get("topology_path")
    if topology_path not in (None, ""):
        return str(topology_path)
    if index is not None:
        return f"record_{index}"
    return "unknown_graph"


def _assignment_ranking_field(
    records: list[dict],
) -> tuple[str, str, bool]:
    for field, direction in _ASSIGNMENT_RANKING_FIELDS:
        if any(record.get(field) is not None for record in records):
            return field, direction, False
    return "manifest_order", "ascending", True


def _coerce_ranking_value(value: Any) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        text = str(value).strip()
        return text if text else None


def _assignment_sort_key(
    record: dict,
    *,
    ranking_field: str,
    ranking_direction: str,
    manifest_index: int,
) -> tuple:
    if ranking_field == "manifest_order":
        return (manifest_index,)
    value = _coerce_ranking_value(record.get(ranking_field))
    if value is None:
        ranked = (1, 0.0)
    elif isinstance(value, float):
        numeric = -value if ranking_direction == "descending" else value
        ranked = (0, numeric)
    else:
        ranked = (0, value)
    return (*ranked, manifest_index, str(record.get("assignment_id", "")))


def _corpus_assignment_selection_metadata(
    *,
    mode: str,
    effective_rule: str,
    selected_assignments_per_graph: int | None,
    ranking_field: str | None,
    ranking_direction: str | None,
    records_before: list[dict],
    records_after: list[dict],
) -> dict[str, Any]:
    before_graph_ids = {
        _corpus_graph_identity(record, index)
        for index, record in enumerate(records_before)
    }
    selected_assignment_ids_by_graph: dict[str, list[str]] = defaultdict(list)
    for index, record in enumerate(records_after):
        graph_id = _corpus_graph_identity(record, index)
        selected_assignment_ids_by_graph[graph_id].append(
            str(record.get("assignment_id", ""))
        )
    after_graph_ids = set(selected_assignment_ids_by_graph)
    return {
        "assignment_selection": mode,
        "assignment_selection_effective_rule": effective_rule,
        "selected_assignments_per_graph": selected_assignments_per_graph,
        "assignment_ranking_field": ranking_field,
        "assignment_ranking_direction": ranking_direction,
        "graph_count_before_selection": len(before_graph_ids),
        "graph_count_after_selection": len(after_graph_ids),
        "assignment_count_before_selection": len(records_before),
        "assignment_count_after_selection": len(records_after),
        "skipped_graph_count": len(before_graph_ids - after_graph_ids),
        "selected_assignment_ids_by_graph": dict(selected_assignment_ids_by_graph),
    }


def _select_corpus_assignments(
    records: list[dict],
    *,
    assignment_selection: str = ASSIGNMENT_SELECTION_ALL,
) -> tuple[list[dict], dict[str, Any]]:
    """Select corpus assignment records without changing record contents."""
    if assignment_selection == ASSIGNMENT_SELECTION_ALL:
        return list(records), _corpus_assignment_selection_metadata(
            mode=ASSIGNMENT_SELECTION_ALL,
            effective_rule=ASSIGNMENT_SELECTION_ALL,
            selected_assignments_per_graph=None,
            ranking_field=None,
            ranking_direction=None,
            records_before=records,
            records_after=records,
        )
    if assignment_selection != ASSIGNMENT_SELECTION_TOP2_PER_GRAPH:
        raise ValueError(f"unsupported assignment_selection {assignment_selection!r}")

    ranking_field, ranking_direction, used_manifest_order = _assignment_ranking_field(records)
    indexed_by_graph: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, record in enumerate(records):
        indexed_by_graph[_corpus_graph_identity(record, index)].append((index, record))

    selected: list[tuple[int, dict]] = []
    for graph_records in indexed_by_graph.values():
        ordered = sorted(
            graph_records,
            key=lambda pair: _assignment_sort_key(
                pair[1],
                ranking_field=ranking_field,
                ranking_direction=ranking_direction,
                manifest_index=pair[0],
            ),
        )
        selected.extend(ordered[:ASSIGNMENT_SELECTION_TOP2_LIMIT])

    selected.sort(key=lambda pair: pair[0])
    selected_records = [record for _, record in selected]
    effective_rule = (
        ASSIGNMENT_SELECTION_TOP2_MANIFEST_ORDER_RULE
        if used_manifest_order
        else f"top2_per_graph_{ranking_field}"
    )
    metadata = _corpus_assignment_selection_metadata(
        mode=ASSIGNMENT_SELECTION_TOP2_PER_GRAPH,
        effective_rule=effective_rule,
        selected_assignments_per_graph=ASSIGNMENT_SELECTION_TOP2_LIMIT,
        ranking_field=ranking_field,
        ranking_direction=ranking_direction,
        records_before=records,
        records_after=selected_records,
    )
    metadata["assignment_ranking_fallback"] = (
        "manifest_order" if used_manifest_order else None
    )
    return selected_records, metadata


def _synthetic_reference_category(record: dict) -> str:
    family = str(record.get("family") or "").strip()
    if family:
        return family
    graph_id = str(record.get("graph_id") or "")
    marker = "_synthetic_"
    if marker in graph_id:
        tail = graph_id.split(marker, 1)[1]
        return tail.split("_seed_", 1)[0]
    return "unknown"


def _reference_graphs_per_synthetic_category(
    records: list[dict],
    reference_max_graphs: int,
) -> frozenset[str]:
    selected: set[str] = set()
    seen_by_category: dict[str, set[str]] = defaultdict(set)
    for record in records:
        graph_id = str(record.get("graph_id", ""))
        if not graph_id:
            continue
        category = _synthetic_reference_category(record)
        seen = seen_by_category[category]
        if graph_id in seen:
            continue
        if len(seen) >= reference_max_graphs:
            continue
        seen.add(graph_id)
        selected.add(graph_id)
    return frozenset(selected)


def plot_convergence(
    metrics: dict,
    window: int,
    path: Path,
    keys: list[str] | None = None,
    title: str = "Eve convergence (toy scope)",
) -> None:
    """Plot convergence curves. keys selects which combos/algos to plot."""
    if keys is None:
        keys = list(metrics.keys())
    combo_by_key = {c["key"]: c for c in EXPERIMENT_SUITE}
    fig, ax = plt.subplots(figsize=(9, 4))
    for k in keys:
        if k not in metrics:
            continue
        m    = metrics[k]
        if "rolling_mean" not in m or "rolling_std" not in m:
            continue
        mean = np.array(m["rolling_mean"])
        std  = np.array(m["rolling_std"])
        if len(mean) == 0:
            continue
        xs   = np.arange(len(mean)) + window
        if k in combo_by_key:
            combo = combo_by_key[k]
            c   = _suite_color(combo)
            ls  = _suite_linestyle(combo)
            lbl = _suite_label(combo)
        else:
            c   = COLORS.get(k, "#888888")
            ls  = "-"
            lbl = LABELS.get(k, k)
        ax.plot(xs, mean, color=c, linestyle=ls, label=lbl, linewidth=1.8)
        ax.fill_between(xs, mean - std, mean + std, alpha=0.15, color=c)
    ax.set_xlabel("Training step")
    ax.set_ylabel(f"Eve win rate ({window}-step rolling mean)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, path)


def plot_strategy_distribution(
    metrics: dict,
    path: Path,
    keys: list[str] | None = None,
    title: str = "Attack-type distribution — final training phase (toy scope)",
) -> None:
    if keys is None:
        keys = list(metrics.keys())
    present = [k for k in keys if k in metrics]
    all_kinds = _attack_types_in_fixed_order(
        {k2 for k in present for k2 in metrics[k]["action_distribution"]}
    )
    combo_by_key = {c["key"]: c for c in EXPERIMENT_SUITE}

    bar_labels = []
    for k in present:
        if k in combo_by_key:
            c = combo_by_key[k]
            bar_labels.append(f"{_EVE_LABEL[c['eve']]}\nvs {_ALICE_LABEL[c['alice']]}")
        else:
            bar_labels.append(LABELS.get(k, k))

    fig, ax = plt.subplots(figsize=(max(7, len(present) * 1.4), 4))
    bottoms = np.zeros(len(present))
    for kind in all_kinds:
        vals = np.array([metrics[k]["action_distribution"].get(kind, 0.0) for k in present])
        lbl  = "no attack" if kind in {"none", "no_attack"} else kind.replace("_", " ")
        ax.bar(bar_labels, vals, bottom=bottoms, label=lbl,
               color=_ATTACK_TYPE_COLORS.get(kind, "#aaaaaa"),
               edgecolor="white", linewidth=0.5)
        bottoms += vals
    ax.set_ylabel("Fraction of steps (final 30 %)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.tick_params(axis="x", labelsize=7)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    _save(fig, path)


def plot_q_heatmap(metrics: dict, algo: str, path: Path) -> None:
    m = metrics.get(algo)
    if not m or not m.get("q_table_entries"):
        print(f"  skip Q-heatmap ({algo}: no entries)")
        return
    entries = m["q_table_entries"]
    names   = m["q_action_names"]
    if not names:
        return

    state_labels = [str(tuple(e["state"])) for e in entries]
    q_matrix = np.array([e["q"] for e in entries], dtype=float)

    # Keep at most 15 most-informative rows (highest max|Q|)
    if len(entries) > 15:
        importance = np.max(np.abs(q_matrix), axis=1)
        idx = np.argsort(importance)[-15:][::-1]
        q_matrix     = q_matrix[idx]
        state_labels = [state_labels[i] for i in idx]

    short_names = [n[:22] for n in names]
    h = max(4, len(state_labels) * 0.5)
    w = max(6, len(names) * 0.9)
    fig, ax = plt.subplots(figsize=(w, h))
    # Symmetric scale around 0 sized to the data — a fixed ±1 scale saturates
    # once discounted returns exceed single-step rewards (gamma=0.9).
    q_abs_max = float(np.max(np.abs(q_matrix))) if q_matrix.size else 1.0
    scale = max(1.0, q_abs_max)
    im = ax.imshow(q_matrix, aspect="auto", cmap="RdYlGn", vmin=-scale, vmax=scale)
    ax.set_xticks(range(len(short_names)))
    ax.set_xticklabels(short_names, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(state_labels)))
    ax.set_yticklabels(state_labels, fontsize=7)
    combo_by_key = {c["key"]: c for c in EXPERIMENT_SUITE}
    title_lbl = _suite_label(combo_by_key[algo]) if algo in combo_by_key else LABELS.get(algo, algo)
    ax.set_title(f"Q-table — {title_lbl} (top states by |Q|)")
    plt.colorbar(im, ax=ax, label="Q-value")
    fig.tight_layout()
    _save(fig, path)


# ── Corpus condition summary + figures ────────────────────────────────────────
# Corpus case records keep only aggregates (no per-step curves), so the corpus
# training-dynamics view is: full-run win rate vs final-replay-window win rate
# per case, plus per-condition distributions across the graph corpus.

_ALICE_MARKER = {
    "cautious_greedy": "o",
    "random": "s",
    "q_learning": "D",
    "deep_q": "^",
}


def _combo_parts_from_key(combo_key: str) -> tuple[str, str, str]:
    parts = str(combo_key).split("__")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return str(combo_key), "", ""


def _corpus_condition_label(combo_key: str) -> str:
    combo_by_key = {c["key"]: c for c in EXPERIMENT_SUITE}
    if combo_key in combo_by_key:
        combo = combo_by_key[combo_key]
        return f"{_EVE_LABEL[combo['eve']]} vs {_ALICE_LABEL[combo['alice']]}"
    _, eve, alice = _combo_parts_from_key(combo_key)
    return f"{_EVE_LABEL.get(eve, eve)} vs {_ALICE_LABEL.get(alice, alice)}"


def _corpus_condition_ticklabel(combo_key: str) -> str:
    _, eve, alice = _combo_parts_from_key(combo_key)
    return f"{_EVE_LABEL.get(eve, eve)}\nvs {_ALICE_LABEL.get(alice, alice)}"


def _corpus_condition_color(combo_key: str) -> str:
    _, eve, alice = _combo_parts_from_key(combo_key)
    color = _CONDITION_COLORS.get((eve, alice))
    if color is not None:
        return color
    return _EVE_COLORS.get(eve, "#888888")


def _ordered_condition_keys(rows_by_combo: dict[str, list[dict]]) -> list[str]:
    suite_order = [c["key"] for c in EXPERIMENT_SUITE if c["key"] in rows_by_combo]
    extras = sorted(k for k in rows_by_combo if k not in set(suite_order))
    return suite_order + extras


def _corpus_case_rows_by_combo(per_graph: list[dict]) -> dict[str, list[dict]]:
    """Flatten completed corpus case records into per-condition metric rows."""
    rows_by_combo: dict[str, list[dict]] = defaultdict(list)
    for record in per_graph:
        combo_key = str(record.get("combo_key") or record.get("primary_combo_key") or "")
        combo_result = (record.get("combo_results") or {}).get(combo_key) or {}
        if not combo_key or "eve_win_rate" not in combo_result:
            continue
        runtime_metrics = combo_result.get("runtime_metrics") or {}
        counts = runtime_metrics.get("public_outcome_counts") or {}
        split = _metric_split_from_counts(
            counts,
            information_exposures=runtime_metrics.get("information_exposures"),
        )
        if not counts:
            # Legacy records without outcome counts: keep the stored headline
            # rate rather than a degenerate all-zero split.
            split["eve_win_rate"] = float(combo_result["eve_win_rate"])
        final_win_rate = combo_result.get("final_win_rate")
        attack_counts = {
            str(kind): _coerce_nonnegative_int(count)
            for kind, count in (
                runtime_metrics.get("eve_attack_selection_counts") or {}
            ).items()
        }
        no_attack_quality = (
            (runtime_metrics.get("quality_by_attack_type") or {}).get("no_attack") or {}
        )
        rows_by_combo[combo_key].append({
            "graph_id": record.get("graph_id"),
            "assignment_id": record.get("assignment_id"),
            "family": record.get("family"),
            "shortest_hops": record.get("graph_S"),
            "num_routes": record.get("num_routes"),
            "num_steps": record.get("num_steps"),
            "eve_win_curve": combo_result.get("eve_win_curve"),
            "final_win_rate": (
                float(final_win_rate) if final_win_rate is not None else None
            ),
            "route_entropy": runtime_metrics.get("route_entropy"),
            "route_switches": runtime_metrics.get("route_switches"),
            "attack_selection_counts": attack_counts,
            "no_attack_step_count": _coerce_nonnegative_int(no_attack_quality.get("count")),
            "no_attack_delivery_success_rate": (
                float(no_attack_quality["delivery_success_rate"])
                if no_attack_quality.get("delivery_success_rate") is not None
                else None
            ),
            **split,
        })
    return dict(rows_by_combo)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def _summarize_corpus_conditions(rows_by_combo: dict[str, list[dict]]) -> dict:
    conditions: dict[str, dict] = {}
    for combo_key in _ordered_condition_keys(rows_by_combo):
        rows = rows_by_combo[combo_key]
        game_mode, eve, alice = _combo_parts_from_key(combo_key)
        win_rates = [row["eve_win_rate"] for row in rows]
        final_rows = [row for row in rows if row["final_win_rate"] is not None]
        final_rates = [row["final_win_rate"] for row in final_rows]
        deltas = [row["final_win_rate"] - row["eve_win_rate"] for row in final_rows]
        wr_mean, wr_std = _mean_std(win_rates)
        fw_mean, fw_std = _mean_std(final_rates)
        delta_mean, _ = _mean_std(deltas)
        share_sums: dict[str, float] = defaultdict(float)
        share_cases = 0
        for row in rows:
            total = sum(row["attack_selection_counts"].values())
            if total <= 0:
                continue
            share_cases += 1
            for kind, count in row["attack_selection_counts"].items():
                share_sums[kind] += count / total
        entry = {
            "label": _corpus_condition_label(combo_key),
            "game_mode": game_mode,
            "eve": eve,
            "alice": alice,
            "case_count": len(rows),
            "graph_count": len({row["graph_id"] for row in rows}),
            "families": dict(sorted(Counter(
                str(row["family"]) for row in rows
            ).items())),
            "eve_win_rate_mean": wr_mean,
            "eve_win_rate_std": wr_std,
            "eve_win_rate_min": float(min(win_rates)) if win_rates else None,
            "eve_win_rate_max": float(max(win_rates)) if win_rates else None,
            "final_window_win_rate_mean": fw_mean,
            "final_window_win_rate_std": fw_std,
            "final_minus_overall_win_rate_mean": delta_mean,
            "route_entropy_mean": _mean_std([
                float(row["route_entropy"]) for row in rows
                if row.get("route_entropy") is not None
            ])[0],
            "route_switches_mean": _mean_std([
                float(row["route_switches"]) for row in rows
                if row.get("route_switches") is not None
            ])[0],
            "attack_selection_share_mean": {
                kind: share_sums[kind] / share_cases
                for kind in _attack_types_in_fixed_order(set(share_sums))
            } if share_cases else {},
        }
        # Baseline sanity: cases where Eve's *unattacked* steps never delivered
        # indicate a saturated game (Eve wins regardless of action) on that
        # graph/assignment — win rates there carry no strategic signal.
        baseline_rows = [
            row for row in rows
            if row.get("no_attack_delivery_success_rate") is not None
        ]
        entry["no_attack_baseline_case_count"] = len(baseline_rows)
        entry["no_attack_delivery_success_rate_mean"] = _mean_std([
            row["no_attack_delivery_success_rate"] for row in baseline_rows
        ])[0]
        entry["baseline_delivery_zero_case_count"] = sum(
            1 for row in baseline_rows
            if row["no_attack_delivery_success_rate"] == 0.0
        )
        for rate_key in (
            "delivery_rate",
            "delivery_failure_rate",
            "qber_abort_rate",
            "chsh_abort_rate",
            "info_exposure_rate",
            "secure_delivery_rate",
        ):
            entry[f"{rate_key}_mean"] = _mean_std([
                float(row[rate_key]) for row in rows if rate_key in row
            ])[0]
        conditions[combo_key] = entry
    return {
        "definitions": {
            "eve_win_rate": (
                "fraction of steps the runtime scored as an Eve win "
                "(non-accepted public outcome, or accepted delivery with key "
                "exposure); recomputed from public_outcome_counts plus the "
                "information_exposures flag count for backends that do not "
                "emit exposure as its own public outcome"
            ),
            "final_window_win_rate": (
                "Eve win rate over the final replay window reported by the "
                "corpus runner (training-dynamics endpoint)"
            ),
        },
        "conditions": conditions,
    }


def plot_corpus_win_rates(
    rows_by_combo: dict[str, list[dict]],
    path: Path,
    title: str = "Eve win rate by condition across corpus",
) -> None:
    from matplotlib.patches import Patch

    keys = [k for k in _ordered_condition_keys(rows_by_combo) if rows_by_combo.get(k)]
    if not keys:
        print("  skip corpus win-rate figure (no condition rows)")
        return
    rng = np.random.default_rng(0)
    x = np.arange(len(keys), dtype=float)
    width = 0.38
    overall = [[row["eve_win_rate"] for row in rows_by_combo[k]] for k in keys]
    final = [
        [row["final_win_rate"] for row in rows_by_combo[k]
         if row["final_win_rate"] is not None]
        for k in keys
    ]
    colors = [_corpus_condition_color(k) for k in keys]
    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 1.7), 4.5))
    ax.bar(
        x - width / 2,
        [float(np.mean(vals)) if vals else 0.0 for vals in overall],
        width,
        yerr=[float(np.std(vals)) if vals else 0.0 for vals in overall],
        color=colors, capsize=3, edgecolor="white",
    )
    ax.bar(
        x + width / 2,
        [float(np.mean(vals)) if vals else 0.0 for vals in final],
        width,
        yerr=[float(np.std(vals)) if vals else 0.0 for vals in final],
        color=colors, alpha=0.55, hatch="//", capsize=3, edgecolor="white",
    )
    for xi, vals in zip(x, overall):
        jitter = rng.uniform(-width / 3, width / 3, size=len(vals))
        ax.scatter(
            xi - width / 2 + jitter, vals,
            s=9, color="#333333", alpha=0.45, zorder=3, linewidths=0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([_corpus_condition_ticklabel(k) for k in keys], fontsize=7)
    ax.set_ylabel("Eve win rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title)
    ax.legend(
        handles=[
            Patch(facecolor="#bbbbbb", edgecolor="white", label="full run (mean ± std, dots = cases)"),
            Patch(facecolor="#bbbbbb", edgecolor="white", alpha=0.55, hatch="//", label="final window (mean ± std)"),
        ],
        fontsize=8, loc="upper right",
    )
    fig.tight_layout()
    _save(fig, path)


def plot_corpus_learning_delta(
    rows_by_combo: dict[str, list[dict]],
    path: Path,
    title: str = "Within-run learning shift per corpus case",
) -> None:
    keys = [k for k in _ordered_condition_keys(rows_by_combo) if rows_by_combo.get(k)]
    panels: list[tuple[str, list[dict]]] = []
    for key in keys:
        rows = [row for row in rows_by_combo[key] if row["final_win_rate"] is not None]
        if rows:
            panels.append((key, rows))
    if not panels:
        print("  skip corpus learning-delta figure (no final-window rates)")
        return
    # One panel per condition: same-hue overplotting made the conditions
    # nearly indistinguishable when they only differed by marker shape.
    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.3 * ncols + 0.5, 4.3 * nrows + 0.9),
        squeeze=False, sharex=True, sharey=True,
    )
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax, (key, rows) in zip(axes.ravel(), panels):
        _, _eve, alice = _combo_parts_from_key(key)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, zorder=1)
        ax.scatter(
            [row["eve_win_rate"] for row in rows],
            [row["final_win_rate"] for row in rows],
            color=_corpus_condition_color(key),
            marker=_ALICE_MARKER.get(alice, "o"),
            s=26, alpha=0.6, linewidths=0, zorder=2,
        )
        ax.set_title(f"{_corpus_condition_label(key)} (n={len(rows)})", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.supxlabel("Eve win rate — full training run", fontsize=10)
    fig.supylabel("Eve win rate — final replay window", fontsize=10)
    fig.suptitle(
        f"{title}\n(above diagonal = higher win rate late in training)",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, path)


def plot_corpus_attack_mix(
    rows_by_combo: dict[str, list[dict]],
    path: Path,
    title: str = "Eve attack-type mix by condition (mean share of steps)",
) -> None:
    keys = [k for k in _ordered_condition_keys(rows_by_combo) if rows_by_combo.get(k)]
    mean_shares: dict[str, dict[str, float]] = {}
    for key in keys:
        share_sums: dict[str, float] = defaultdict(float)
        cases = 0
        for row in rows_by_combo[key]:
            total = sum(row["attack_selection_counts"].values())
            if total <= 0:
                continue
            cases += 1
            for kind, count in row["attack_selection_counts"].items():
                share_sums[kind] += count / total
        if cases:
            mean_shares[key] = {kind: value / cases for kind, value in share_sums.items()}
    if not mean_shares:
        print("  skip corpus attack-mix figure (no attack selection counts)")
        return
    plot_keys = [k for k in keys if k in mean_shares]
    all_kinds = _attack_types_in_fixed_order({
        kind for shares in mean_shares.values() for kind in shares
    })
    fig, ax = plt.subplots(figsize=(8, max(3.2, len(plot_keys) * 0.65)))
    y = np.arange(len(plot_keys), dtype=float)
    left = np.zeros(len(plot_keys))
    for kind in all_kinds:
        vals = np.array([mean_shares[k].get(kind, 0.0) for k in plot_keys])
        lbl = "no attack" if kind in {"none", "no_attack"} else kind.replace("_", " ")
        ax.barh(
            y, vals, left=left, height=0.62, label=lbl,
            color=_ATTACK_TYPE_COLORS.get(kind, "#aaaaaa"),
            edgecolor="white", linewidth=0.5,
        )
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(
        [_corpus_condition_ticklabel(k).replace("\n", " ") for k in plot_keys],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean fraction of steps (full run)")
    ax.set_title(title)
    ax.legend(
        fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )
    fig.tight_layout()
    _save(fig, path)


def plot_corpus_win_rate_vs_depth(
    rows_by_combo: dict[str, list[dict]],
    path: Path,
    title: str = "Eve win rate vs assignment route depth",
) -> None:
    keys = [k for k in _ordered_condition_keys(rows_by_combo) if rows_by_combo.get(k)]
    rng = np.random.default_rng(0)
    panels: list[tuple[str, list[dict]]] = []
    for key in keys:
        rows = [
            row for row in rows_by_combo[key]
            if row.get("shortest_hops") is not None
        ]
        if rows:
            panels.append((key, rows))
    if not panels:
        print("  skip corpus depth figure (no shortest-hops metadata)")
        return
    # One panel per condition (shared axes) instead of shape-only overplot.
    ncols = min(3, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.9 * ncols + 0.5, 3.9 * nrows + 0.8),
        squeeze=False, sharex=True, sharey=True,
    )
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax, (key, rows) in zip(axes.ravel(), panels):
        _, _eve, alice = _combo_parts_from_key(key)
        hops = np.array([float(row["shortest_hops"]) for row in rows])
        jittered = hops + rng.uniform(-0.18, 0.18, size=len(hops))
        rates = np.array([float(row["eve_win_rate"]) for row in rows])
        color = _corpus_condition_color(key)
        ax.scatter(
            jittered, rates,
            color=color,
            marker=_ALICE_MARKER.get(alice, "o"),
            s=22, alpha=0.55, linewidths=0, zorder=2,
        )
        # Per-hop median guide (only where >= 3 cases support it)
        hop_levels = sorted({int(h) for h in hops})
        med_x = [h for h in hop_levels if np.sum(hops == h) >= 3]
        med_y = [float(np.median(rates[hops == h])) for h in med_x]
        if len(med_x) >= 2:
            ax.plot(
                med_x, med_y,
                color=color, linewidth=2, alpha=0.9, zorder=3,
                label="per-hop median (≥3 cases)",
            )
            ax.legend(fontsize=7, loc="upper left", frameon=False)
        ax.set_title(f"{_corpus_condition_label(key)} (n={len(rows)})", fontsize=9)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
    fig.supxlabel("Assignment shortest-path hops (Alice→Bob)", fontsize=10)
    fig.supylabel("Eve win rate (full run)", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def _resampled_win_curves(rows: list[dict], grid_points: int = 25) -> np.ndarray | None:
    """Stack per-case eve-win curves resampled onto a common run-fraction grid."""
    grid = (np.arange(grid_points) + 1) / grid_points
    curves = []
    for row in rows:
        curve = row.get("eve_win_curve") or {}
        step_end = curve.get("step_end") or []
        rates = curve.get("eve_win_rate") or []
        if len(step_end) < 2 or len(step_end) != len(rates):
            continue
        fractions = np.asarray(step_end, dtype=float) / float(step_end[-1])
        curves.append(np.interp(grid, fractions, np.asarray(rates, dtype=float)))
    if not curves:
        return None
    return np.vstack(curves)


def plot_corpus_learning_curves(
    rows_by_combo: dict[str, list[dict]],
    path: Path,
    title: str = "Eve win rate over training (median across cases, IQR band)",
) -> None:
    keys = [k for k in _ordered_condition_keys(rows_by_combo) if rows_by_combo.get(k)]
    grid_points = 25
    grid = (np.arange(grid_points) + 1) / grid_points
    panels: list[tuple[str, np.ndarray]] = []
    for key in keys:
        stacked = _resampled_win_curves(rows_by_combo[key], grid_points)
        if stacked is not None:
            panels.append((key, stacked))
    if not panels:
        print("  skip corpus learning-curve figure (no eve_win_curve digests)")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for key, stacked in panels:
        color = _corpus_condition_color(key)
        median = np.median(stacked, axis=0)
        q25 = np.quantile(stacked, 0.25, axis=0)
        q75 = np.quantile(stacked, 0.75, axis=0)
        ax.fill_between(grid, q25, q75, color=color, alpha=0.18, linewidth=0)
        ax.plot(
            grid, median, color=color, linewidth=2,
            label=f"{_corpus_condition_label(key)} (n={stacked.shape[0]})",
        )
    ax.set_xlabel("Fraction of training run")
    ax.set_ylabel("Windowed Eve win rate")
    ax.set_xlim(grid[0], 1.0)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.3)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    _save(fig, path)


def _paired_condition_rows(
    rows_by_combo: dict[str, list[dict]],
) -> tuple[str, str, list[tuple[dict, dict]]] | None:
    """Pair cases of the two main conditions by (graph_id, assignment_id)."""
    keys = [k for k in _ordered_condition_keys(rows_by_combo) if rows_by_combo.get(k)]
    if len(keys) != 2:
        main_keys = [
            combo["key"] for combo in EXPERIMENT_SUITE
            if combo.get("group") == "main" and rows_by_combo.get(combo["key"])
        ]
        if len(main_keys) != 2:
            return None
        keys = main_keys
    key_a, key_b = keys
    by_case_a = {
        (row.get("graph_id"), row.get("assignment_id")): row
        for row in rows_by_combo[key_a]
    }
    pairs = [
        (by_case_a[case], row)
        for row in rows_by_combo[key_b]
        if (case := (row.get("graph_id"), row.get("assignment_id"))) in by_case_a
    ]
    if not pairs:
        return None
    return key_a, key_b, pairs


def plot_corpus_condition_paired_delta(
    rows_by_combo: dict[str, list[dict]],
    path: Path,
    title: str = "Per-assignment effect of Alice's policy on Eve's win rate",
) -> None:
    paired = _paired_condition_rows(rows_by_combo)
    if paired is None:
        print("  skip corpus paired-delta figure (need two paired conditions)")
        return
    key_a, key_b, pairs = paired
    label_a = _corpus_condition_label(key_a)
    label_b = _corpus_condition_label(key_b)
    wins_a = np.array([float(a["eve_win_rate"]) for a, _b in pairs])
    wins_b = np.array([float(b["eve_win_rate"]) for _a, b in pairs])
    deltas = wins_b - wins_a

    fig, (ax_scatter, ax_ecdf) = plt.subplots(
        1, 2, figsize=(10.4, 4.9), gridspec_kw={"width_ratios": [1.05, 1]},
    )
    ax_scatter.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1, zorder=1)
    ax_scatter.scatter(
        wins_a, wins_b, s=24, alpha=0.55, linewidths=0, color="#5b6672", zorder=2,
    )
    ax_scatter.set_xlabel(f"Eve win rate — {label_a}")
    ax_scatter.set_ylabel(f"Eve win rate — {label_b}")
    ax_scatter.set_xlim(-0.02, 1.02)
    ax_scatter.set_ylim(-0.02, 1.02)
    ax_scatter.set_aspect("equal")
    ax_scatter.grid(alpha=0.3)
    ax_scatter.set_title(
        f"Same graph+assignment, both conditions (n={len(pairs)})", fontsize=9,
    )

    order = np.sort(deltas)
    ecdf = (np.arange(len(order)) + 1) / len(order)
    ax_ecdf.axvline(0.0, color="#999999", linestyle="--", linewidth=1, zorder=1)
    ax_ecdf.step(order, ecdf, where="post", color="#5b6672", linewidth=2, zorder=2)
    median_delta = float(np.median(deltas))
    ax_ecdf.axvline(
        median_delta, color="#5b6672", linewidth=1, alpha=0.6,
        label=f"median Δ = {median_delta:+.3f}",
    )
    ax_ecdf.set_xlabel(f"Δ Eve win rate\n({label_b} − {label_a})", fontsize=9)
    ax_ecdf.set_ylabel("Fraction of paired cases ≤ Δ")
    ax_ecdf.set_ylim(0, 1.0)
    ax_ecdf.grid(alpha=0.3)
    ax_ecdf.legend(fontsize=8, loc="lower right")
    ax_ecdf.set_title("Right of 0 = Eve does better vs the second condition", fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


# Public-outcome share slots (fractions of steps; secure delivery is Alice's
# success state, the rest are Eve-win components). Okabe-Ito hues, matching
# the attack-mix convention of gray for the "nothing happened" share.
_OUTCOME_SHARE_SLOTS: list[tuple[str, str, str]] = [
    ("secure_delivery_rate", "secure delivery", "#cccccc"),
    ("info_exposure_rate", "information exposure", "#d55e00"),
    ("delivery_failure_rate", "delivery failure", "#e69f00"),
    ("qber_abort_rate", "QBER abort", "#56b4e9"),
    ("chsh_abort_rate", "CHSH abort", "#0072b2"),
]


def plot_corpus_outcome_mix(
    rows_by_combo: dict[str, list[dict]],
    path: Path,
    title: str = "Public outcome mix by condition (mean share of steps)",
) -> None:
    keys = [k for k in _ordered_condition_keys(rows_by_combo) if rows_by_combo.get(k)]
    mean_shares: dict[str, dict[str, float]] = {}
    for key in keys:
        rows = [
            row for row in rows_by_combo[key]
            if any(row.get(field) is not None for field, _l, _c in _OUTCOME_SHARE_SLOTS)
        ]
        if rows:
            mean_shares[key] = {
                field: float(np.mean([float(row.get(field) or 0.0) for row in rows]))
                for field, _label, _color in _OUTCOME_SHARE_SLOTS
            }
    if not mean_shares:
        print("  skip corpus outcome-mix figure (no outcome splits)")
        return
    plot_keys = [k for k in keys if k in mean_shares]
    fig, ax = plt.subplots(figsize=(8, max(3.2, len(plot_keys) * 0.65)))
    y = np.arange(len(plot_keys), dtype=float)
    left = np.zeros(len(plot_keys))
    for field, label, color in _OUTCOME_SHARE_SLOTS:
        vals = np.array([mean_shares[k][field] for k in plot_keys])
        ax.barh(
            y, vals, left=left, height=0.62, label=label,
            color=color, edgecolor="white", linewidth=0.5,
        )
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(
        [_corpus_condition_ticklabel(k).replace("\n", " ") for k in plot_keys],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean fraction of steps (full run)")
    ax.set_title(title)
    ax.legend(
        fontsize=7, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
        frameon=False,
    )
    fig.tight_layout()
    _save(fig, path)


def plot_corpus_route_behavior(
    rows_by_combo: dict[str, list[dict]],
    path: Path,
    title: str = "Alice route behavior by condition",
) -> None:
    keys = [k for k in _ordered_condition_keys(rows_by_combo) if rows_by_combo.get(k)]
    metrics = [
        ("route_entropy", "Route-choice entropy (bits)"),
        ("route_switches", "Route switches per run"),
    ]
    series: dict[str, dict[str, np.ndarray]] = {}
    for key in keys:
        for field, _label in metrics:
            values = [
                float(row[field]) for row in rows_by_combo[key]
                if row.get(field) is not None
            ]
            if values:
                series.setdefault(key, {})[field] = np.sort(np.asarray(values))
    if not series:
        print("  skip corpus route-behavior figure (no route metrics)")
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (field, label) in zip(axes, metrics):
        plotted = False
        for key in keys:
            values = series.get(key, {}).get(field)
            if values is None:
                continue
            ecdf = (np.arange(len(values)) + 1) / len(values)
            ax.step(
                values, ecdf, where="post",
                color=_corpus_condition_color(key), linewidth=2,
                label=f"{_corpus_condition_label(key)} (n={len(values)})",
            )
            plotted = True
        if not plotted:
            ax.axis("off")
            continue
        ax.set_xlabel(label)
        ax.set_ylabel("Fraction of cases ≤ x")
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    _save(fig, path)


def plot_corpus_dt_summary(
    learned_aggs: dict[str, dict],
    route_choice_aggs: dict[str, dict],
    path: Path,
) -> None:
    """Feature importances + fit quality for every pooled cross-graph DT."""
    panels: list[tuple[str, str, dict]] = []
    for emission, aggs in (
        ("Eve attack choice", learned_aggs or {}),
        ("Alice route choice", route_choice_aggs or {}),
    ):
        for combo_key, agg in aggs.items():
            dt_stats = (agg or {}).get("structural_dt") or {}
            if dt_stats.get("top_features"):
                panels.append((
                    f"{emission}\n{_corpus_condition_label(combo_key)}",
                    combo_key,
                    dt_stats,
                ))
    if not panels:
        print("  skip corpus DT summary figure (no fitted DTs)")
        return
    ncols = min(2, len(panels))
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, max(2.8, 2.6 * nrows)),
        squeeze=False,
    )
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for ax, (label, combo_key, dt_stats) in zip(axes.ravel(), panels):
        features = dt_stats["top_features"][:8][::-1]
        names = [str(f["feature"]) for f in features]
        importances = [float(f["importance"]) for f in features]
        ax.barh(
            names, importances,
            color=_corpus_condition_color(combo_key), edgecolor="white",
        )
        ax.set_xlabel("feature importance", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(
            f"{label}\nR²={dt_stats.get('r2_score', 0):.3f}, "
            f"{dt_stats.get('num_rows', 0)} rows, "
            f"{dt_stats.get('num_topologies', 0)} topologies",
            fontsize=8,
        )
    fig.suptitle("Pooled cross-graph decision trees (toy scope)", fontsize=10)
    fig.tight_layout()
    _save(fig, path)


def _render_corpus_figures(rows_by_combo: dict[str, list[dict]], fig_dir: Path) -> None:
    print("  Corpus figures:")
    figure_jobs = [
        (plot_corpus_win_rates, fig_dir / "corpus_win_rates_by_condition.png"),
        (plot_corpus_learning_curves, fig_dir / "corpus_learning_curves.png"),
        (plot_corpus_learning_delta, fig_dir / "corpus_learning_delta.png"),
        (plot_corpus_condition_paired_delta, fig_dir / "corpus_condition_paired_delta.png"),
        (plot_corpus_outcome_mix, fig_dir / "corpus_outcome_mix_by_condition.png"),
        (plot_corpus_attack_mix, fig_dir / "corpus_attack_mix_by_condition.png"),
        (plot_corpus_win_rate_vs_depth, fig_dir / "corpus_win_rate_vs_route_depth.png"),
        (plot_corpus_route_behavior, fig_dir / "corpus_route_behavior.png"),
    ]
    for plot_fn, fig_path in figure_jobs:
        try:
            plot_fn(rows_by_combo, fig_path)
        except Exception as exc:  # a figure failure must not kill a long corpus run
            print(f"  WARNING: {fig_path.name} failed: {exc}")


# ── LLM summary ───────────────────────────────────────────────────────────────

def build_prompt(metrics: dict, n_steps: int, n_seeds: int, window: int) -> str:
    """Build LLM summary prompt. Works with both suite keys and legacy algo keys."""
    combo_by_key = {c["key"]: c for c in EXPERIMENT_SUITE}

    # Group rows by group label for readability
    group_order = ["reference", "main", "baseline"]
    rows_by_group: dict[str, list[str]] = {g: [] for g in group_order}
    rows_other: list[str] = []

    for k, m in metrics.items():
        hr  = m["final_hit_rate_mean"]
        std = m["final_hit_rate_std"]
        d   = m["action_distribution"]
        action_summary = "  ".join(
            f"{kind.replace('_',' ')}={v:.2f}"
            for kind, v in sorted(d.items()) if v > 0.01
        )
        if k in combo_by_key:
            combo = combo_by_key[k]
            lbl   = _suite_label(combo)
            grp   = combo["group"]
            rows_by_group[grp].append(f"  {lbl:<60}  hit={hr:.3f}±{std:.3f}  {action_summary}")
        else:
            lbl = LABELS.get(k, k)
            rows_other.append(f"  {lbl:<60}  hit={hr:.3f}±{std:.3f}  {action_summary}")

    table_parts = []
    for g in group_order:
        if rows_by_group[g]:
            table_parts.append(f"[{g}]\n" + "\n".join(rows_by_group[g]))
    if rows_other:
        table_parts.append("[other]\n" + "\n".join(rows_other))
    table = "\n\n".join(table_parts)

    # Best DQN Eve combo for policy sample
    dqn_key = next((k for k in metrics if "deep_q" in k and metrics[k].get("q_table_entries")), None)
    q_block = "  (no Q-table sample available)"
    if dqn_key:
        m_dqn    = metrics[dqn_key]
        entries  = sorted(
            m_dqn.get("q_table_entries") or [],
            key=lambda entry: max((abs(float(q)) for q in entry.get("q", [])), default=0.0),
            reverse=True,
        )[:8]
        act_names = m_dqn.get("q_action_names") or []
        q_lines  = []
        for entry in entries:
            state  = entry["state"]
            q_vals = entry["q"]
            if act_names and q_vals:
                bi   = int(np.argmax(q_vals))
                best = act_names[bi] if bi < len(act_names) else f"action_{bi}"
                q_lines.append(f"  state={state}  best_action={best}  best_Q={q_vals[bi]:.4f}")
        if q_lines:
            q_block = f"[from combo: {dqn_key}]\n" + "\n".join(q_lines)

    return f"""You are writing a Results section paragraph for a simulation paper.
Scientific scope: TOY SIMULATION — do not claim physical validity or real security.

RULES (strictly enforced):
- Only report facts supported by the numbers in the tables below.
- Cite each claim with its metric, e.g. "hit rate: 0.XX".
- Do not introduce physics, security implications, or conclusions not in the data.
- Do not invent numbers. Do not extrapolate.
- Qualify every claim with "simulation results" or "in simulation".

## Simulation parameters
Topology: 2-route diamond — alice→r0→bob (100 km) and alice→r1→bob (160 km)
Steps per run: {n_steps}  |  Seeds: {n_seeds}  |  Rolling window: {window} steps
Game modes:
  attack_surface_all — all six E91 runtime attacks; Eve reward = DoS + empirical information gain where applicable

## Combo results (mean ± std over final 30 % of training)
{table}

## DQN Eve policy sample (first seed, top 8 states)
{q_block}

## Your task
Write 6–8 sentences for the Results section covering:
1. How DQN Eve performs on the all-six E91 attack surface (cite hit rates).
2. Effect of Alice policy (random, cautious-greedy, DQN) on Eve's hit rate.
3. Reference baselines: random Eve, route-aware oracle Eve (cite numbers).
4. What attack types each learning algorithm converged to (cite distributions).
5. Whether information-bearing actions appear in the learned strategy.
Do not exceed 8 sentences. Cite every number. Use "simulation results".
"""


def _model_timeout(model: str, base: int = 300) -> int:
    """Scale timeout by model size — large/reasoning models need more time."""
    name = model.lower()
    if any(x in name for x in ("70b", "32b", "27b", "30b", "31b")):
        return max(base, 900)
    if any(x in name for x in ("14b",)):
        return max(base, 600)
    return base


def call_llm(prompt: str, model: str, timeout: int = 0) -> str:
    from sequence_game.xai.llm_client import query_model
    effective = timeout if timeout > 0 else _model_timeout(model)
    try:
        resp = query_model(model, prompt, temperature=0.05, timeout=effective)
        return resp.response.strip()
    except Exception as exc:
        return f"[LLM unavailable — {exc}]"


def _is_llm_unavailable(text: str) -> bool:
    return text.startswith("[LLM unavailable")


def _write_llm_failure(path: Path, *, model: str, prompt_kind: str, text: str, elapsed: float) -> None:
    if path.exists():
        try:
            existing_text = path.read_text(encoding="utf-8")
        except OSError:
            existing_text = ""
        if "[LLM unavailable" in existing_text:
            path.unlink()
    error_path = path.with_suffix(path.suffix + ".error.json")
    error_path.write_text(
        json.dumps({
            "model": model,
            "prompt_kind": prompt_kind,
            "elapsed_s": round(elapsed, 3),
            "error": text,
            "retry_note": "Successful LLM markdown was not written; rerun/resume will retry this prompt.",
        }, indent=2, default=str),
        encoding="utf-8",
    )


def _write_llm_markdown_or_error(
    path: Path,
    *,
    model: str,
    prompt_kind: str,
    text: str,
    elapsed: float,
    markdown: str,
) -> bool:
    if _is_llm_unavailable(text):
        _write_llm_failure(path, model=model, prompt_kind=prompt_kind, text=text, elapsed=elapsed)
        print(f" failed; wrote {path.name}.error.json")
        return False
    path.write_text(markdown, encoding="utf-8")
    error_path = path.with_suffix(path.suffix + ".error.json")
    if error_path.exists():
        error_path.unlink()
    return True


def _write_llm_prompt_sidecar(path: Path, prompt: str) -> None:
    path.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")


def _llm_success_path_exists(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return "[LLM unavailable" not in text


# ── Mitigation evaluation ─────────────────────────────────────────────────────

def _build_toy_xai_explanation(key: str, final_window: list[dict]) -> tuple[str, str]:
    """Return (explanation_text, dominant_action_id) from the final replay window."""
    if not final_window:
        return "No simulation data available for this combo.", "unknown"
    from collections import Counter
    action_counts: Counter = Counter()
    for s in final_window:
        action = s.get("eve_action") or {}
        aid = action.get("id") or s.get("eve_action_id") or action.get("kind", "none")
        action_counts[aid] += 1
    dominant_action = action_counts.most_common(1)[0][0]
    win_rate = sum(1 for s in final_window if _is_eve_win(s)) / max(1, len(final_window))
    combo_by_key = {c["key"]: c for c in EXPERIMENT_SUITE}
    lbl = _suite_label(combo_by_key[key]) if key in combo_by_key else LABELS.get(key, key)
    dominant_is_probe = any(
        token in str(dominant_action)
        for token in ("information", "measure")
    )
    dominant_effect = (
        "gaining partial key information (information-exposure attack)."
        if dominant_is_probe
        else "blocking key delivery without information gain (DoS-type attack)."
    )
    explanation = (
        f"[Toy scope — synthetic XAI placeholder for paper evaluation]\n"
        f"In this toy simulation, the {lbl} converged to primarily using "
        f"'{dominant_action}' attacks. Over the final {len(final_window)} steps, "
        f"Eve achieved a win rate (delivery denial or key exposure) of "
        f"{win_rate:.1%} (simulation result). "
        f"The dominant action targets Alice's preferred route, {dominant_effect}"
    )
    return explanation, dominant_action


def run_mitigation_evaluations(
    all_results: dict[str, list[dict]],
    models: list[str],
    out_dir: Path,
    ts: str,
) -> dict:
    """Evaluate mitigation advice quality across models and Eve algorithms.

    For each (algo, model) pair:
    - Build a toy XAI explanation from the final replay window
    - Build a mitigation prompt
    - Query the model
    - Score with faithfulness + mitigation semantic check
    - Save output

    Returns structured dict saved to data/mitigation_evaluations.json.
    """
    from sequence_game.xai.faithfulness import score_faithfulness
    from sequence_game.xai.prompt_builder import build_mitigation_prompt
    from sequence_game.xai.semantic import check_mitigation_semantic_faithfulness

    summaries_dir = out_dir / "summaries"
    summaries_dir.mkdir(exist_ok=True)

    evals: dict = {}  # keyed by algo → model → result

    # Build prompts for all algos first (no LLM)
    algo_prompts: dict[str, tuple[str, str]] = {}
    for algo, results in all_results.items():
        evals[algo] = {}
        final_window = results[0].get("final_window") or [] if results else []
        explanation, dominant_action = _build_toy_xai_explanation(algo, final_window)
        prompt = build_mitigation_prompt(explanation, dominant_action, algo, ctx_type="strategy")
        algo_prompts[algo] = (prompt, dominant_action)

    # Load any existing mitigation results (for resume after crash)
    mit_json = out_dir / "data" / "mitigation_evaluations.json"
    existing_evals: dict = {}
    if mit_json.exists():
        existing_evals = json.loads(mit_json.read_text(encoding="utf-8"))

    # Sweep per model so each model is loaded once for all its prompts
    for model in models:
        print(f"\n  --- model: {model} ---", flush=True)
        for algo, (prompt, dominant_action) in algo_prompts.items():
            existing_entry = existing_evals.get(algo, {}).get(model)
            if existing_entry and "error" not in existing_entry:
                print(f"  [{algo}] mitigation eval → {model} … skipped (exists)")
                evals.setdefault(algo, {})[model] = existing_entry
                continue
            print(f"  [{algo}] mitigation eval → {model} …", flush=True)
            t0 = time.perf_counter()
            response = call_llm(prompt, model, timeout=300)
            elapsed = time.perf_counter() - t0

            if response.startswith("[LLM unavailable"):
                evals[algo][model] = {"error": response, "elapsed_s": round(elapsed, 2)}
                continue

            faith = score_faithfulness(response, prompt)
            sem = check_mitigation_semantic_faithfulness(response, dominant_action)

            entry = {
                "elapsed_s":         round(elapsed, 2),
                "response_len":      len(response),
                "faith_score":       round(faith.score, 4),
                "faith_hallucinated": list(faith.hallucinated or []),
                "sem_passed":        sem.passed,
                "sem_score":         sem.score,
                "sem_violations":    [{"rule": v.rule, "matched": v.matched_text}
                                      for v in sem.violations],
                "response":          response,
                "dominant_action":   dominant_action,
            }
            evals[algo][model] = entry
            print(f"    numeric {faith.score:.2%}  semantic {'PASS' if sem.passed else 'FAIL'}"
                  f"  {elapsed:.1f}s")

    # Save per-model comparison markdown
    _write_mitigation_comparison(evals, models, summaries_dir, ts)
    return evals


def _write_mitigation_comparison(
    evals: dict,
    models: list[str],
    summaries_dir: Path,
    ts: str,
) -> None:
    lines = [
        "# Mitigation Advice Comparison\n",
        f"**Generated:** {ts}  |  **Scope:** toy simulation\n",
        f"Models: {', '.join(f'`{m}`' for m in models)}\n",
        "---\n",
    ]
    for algo in evals:
        lines.append(f"## Eve algorithm: `{algo}`\n")
        for model in models:
            entry = evals[algo].get(model, {})
            if "error" in entry:
                lines.append(f"### {model}\n**Error:** {entry['error']}\n\n")
                continue
            sem_label = "PASS" if entry.get("sem_passed") else "FAIL"
            lines.append(
                f"### `{model}`\n"
                f"numeric faithfulness: **{entry.get('faith_score', 0):.0%}**  |  "
                f"semantic: **{sem_label}**  |  "
                f"time: {entry.get('elapsed_s', 0):.1f}s\n\n"
                f"{entry.get('response', '')}\n\n"
                f"---\n"
            )
    with open(summaries_dir / "mitigation_comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("  → summaries/mitigation_comparison.md")


# ── Entry point ───────────────────────────────────────────────────────────────

def _model_slug(model: str) -> str:
    """Safe filename stem from model name."""
    return model.replace("/", "_").replace(":", "-").replace(" ", "_")


def llm_summaries(
    prompt: str,
    models: list[str],
    out_dir: Path,
    ts: str,
) -> dict[str, str]:
    """Run prompt through each model; return {model: text}. Saves individual files.

    Skips a model if its output file already exists (safe to resume after crash).
    """
    summaries_dir = out_dir / "summaries"
    summaries_dir.mkdir(exist_ok=True)
    results = {}
    for model in models:
        slug = _model_slug(model)
        out_path = summaries_dir / f"{slug}.md"
        if _llm_success_path_exists(out_path):
            print(f"\n  [{model}] skipped (exists) — loading from disk", flush=True)
            results[model] = out_path.read_text(encoding="utf-8")
            continue
        print(f"\n  [{model}] querying…", flush=True)
        t0 = time.perf_counter()
        text = call_llm(prompt, model)
        elapsed = time.perf_counter() - t0
        print(f"    {elapsed:.1f}s  |  {len(text)} chars")
        if _is_llm_unavailable(text):
            _write_llm_failure(
                out_path,
                model=model,
                prompt_kind="results_summary",
                text=text,
                elapsed=elapsed,
            )
            print(f"    failed; wrote summaries/{out_path.name}.error.json")
            continue
        results[model] = text
        md = (
            f"# Results Summary — {model}\n\n"
            f"**Generated:** {ts}  |  **Scope:** toy simulation  |  "
            f"**Time:** {elapsed:.1f}s\n\n"
            f"{text}\n\n"
            f"---\n_Prompt saved in `summary_prompt.txt`._\n"
        )
        out_path.write_text(md, encoding="utf-8")
        error_path = out_path.with_suffix(out_path.suffix + ".error.json")
        if error_path.exists():
            error_path.unlink()
        print(f"    → summaries/{slug}.md")
    return results


def write_comparison(summaries: dict[str, str], out_dir: Path, ts: str) -> None:
    """Write side-by-side comparison markdown."""
    lines = [
        f"# LLM Summary Comparison\n",
        f"**Generated:** {ts}  |  **Scope:** toy simulation\n",
        f"Models compared: {', '.join(f'`{m}`' for m in summaries)}\n",
        "---\n",
    ]
    for model, text in summaries.items():
        lines += [f"## `{model}`\n\n{text}\n\n---\n"]
    with open(out_dir / "summaries" / "comparison.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  → summaries/comparison.md")


def _resolve_manifest_record_path(corpus_root: Any, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    if hasattr(corpus_root, "stable_file_hash"):
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    for candidate in (corpus_root / path, path):
        if candidate.exists():
            return candidate
    return corpus_root / path


def _manifest_file_or_value_hash(corpus_root: Any, raw_path: str | None) -> str:
    stable_file_hash = getattr(corpus_root, "stable_file_hash", None)
    if raw_path and callable(stable_file_hash):
        try:
            return stable_file_hash(raw_path, digest_size=_CASE_HASH_DIGEST_SIZE)
        except FileNotFoundError:
            pass
    path = _resolve_manifest_record_path(corpus_root, raw_path)
    if path is not None and path.exists():
        return _stable_file_hash(path)
    return _stable_hash({"missing_manifest_path": raw_path})


def _corpus_case_config_hash(
    *,
    combo: dict[str, Any],
    dt_refit_config: DynamicDTRefitConfig | None,
    dt_depth: int,
    runtime_backend: str = RUNTIME_BACKEND_SEQUENCE,
    direct_e91_training_light: bool = False,
    compiled_histogram_backend_config: dict[str, Any] | None = None,
) -> str:
    return _stable_hash({
        "combo": combo,
        "dt_refit_config": dt_refit_config.to_dict() if dt_refit_config is not None else None,
        "dt_depth": int(dt_depth),
        "runtime_backend": runtime_backend,
        "direct_e91_training_light": bool(direct_e91_training_light),
        "compiled_histogram_backend_config": (
            dict(compiled_histogram_backend_config or {})
            if runtime_backend == RUNTIME_BACKEND_COMPILED_HISTOGRAM
            else None
        ),
    })


def _corpus_case_action_space_hash(
    *,
    record: dict[str, Any],
    combo: dict[str, Any],
    route_policy: str,
    topology_hash: str,
    assignment_hash: str,
) -> str:
    return _stable_hash({
        "graph_id": record.get("graph_id"),
        "assignment_id": record.get("assignment_id"),
        "combo_key": _combo_key_from_combo(combo),
        "game_mode": combo.get("game_mode"),
        "route_policy": route_policy,
        "topology_hash": topology_hash,
        "assignment_hash": assignment_hash,
        "candidate_route_count": (
            record.get("candidate_route_count")
            or record.get("route_count")
        ),
    })


def _make_corpus_case_plan(
    *,
    record: dict[str, Any],
    corpus_root: Any,
    combo: dict[str, Any],
    seed: int,
    num_steps: int,
    route_policy: str,
    dt_refit_config: DynamicDTRefitConfig | None,
    dt_depth: int,
    runtime_backend: str = RUNTIME_BACKEND_SEQUENCE,
    direct_e91_training_light: bool = False,
    compiled_histogram_backend_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    topology_hash = _manifest_file_or_value_hash(corpus_root, record.get("topology_path"))
    assignment_hash = _manifest_file_or_value_hash(corpus_root, record.get("assignment_path"))
    action_space_hash = _corpus_case_action_space_hash(
        record=record,
        combo=combo,
        route_policy=route_policy,
        topology_hash=topology_hash,
        assignment_hash=assignment_hash,
    )
    config_hash = _corpus_case_config_hash(
        combo=combo,
        dt_refit_config=dt_refit_config,
        dt_depth=dt_depth,
        runtime_backend=runtime_backend,
        direct_e91_training_light=direct_e91_training_light,
        compiled_histogram_backend_config=compiled_histogram_backend_config,
    )
    return _make_case_plan(
        graph_id=str(record.get("graph_id", "")),
        assignment_id=str(record.get("assignment_id", "")),
        combo=combo,
        seed=seed,
        num_steps=num_steps,
        route_policy=route_policy,
        topology_hash=topology_hash,
        action_space_hash=action_space_hash,
        config_hash=config_hash,
        runtime_backend=runtime_backend,
    )


def _load_corpus_manifest_and_source(
    manifest_path: str | None,
    *,
    corpus_db_path: str | Path | None = None,
    corpus_id: str | None = None,
) -> tuple[dict[str, Any], Any, str]:
    if corpus_db_path is not None:
        source = SQLiteGraphCorpusSource.open(corpus_db_path, corpus_id)
        return source.load_manifest(), source, f"sqlite:{source.corpus_id}"

    if not manifest_path:
        raise FileNotFoundError("no corpus manifest path or corpus SQLite DB was provided")
    manifest_p = Path(manifest_path)
    if not manifest_p.exists():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_p.read_text(encoding="utf-8")), manifest_p.parent, manifest_p.name


def _corpus_source_task_spec(corpus_source: Any) -> dict[str, str] | None:
    task_spec = getattr(corpus_source, "task_spec", None)
    if callable(task_spec):
        return task_spec()
    return None


def _corpus_source_from_task(task: dict[str, Any]) -> Any:
    spec = task.get("corpus_source")
    if isinstance(spec, dict) and spec.get("kind") == "sqlite_graph_corpus":
        return SQLiteGraphCorpusSource.open(spec["db_path"], spec.get("corpus_id"))
    return Path(task["corpus_root"])


def _dt_refit_case_output_dir(base_dir: Path, case_id: str) -> Path:
    safe_case_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(case_id))
    return base_dir / safe_case_id


def _apply_corpus_metric_splits(result: dict[str, Any]) -> None:
    combo_results = result.get("combo_results") or {}
    primary_key = result.get("primary_combo_key")
    for combo_key, combo_result in combo_results.items():
        if not isinstance(combo_result, dict):
            continue
        runtime_metrics = combo_result.get("runtime_metrics") or {}
        counts = runtime_metrics.get("public_outcome_counts") or {}
        split = _metric_split_from_counts(
            counts,
            information_exposures=runtime_metrics.get("information_exposures"),
        )
        combo_result.update(split)
        if combo_key == primary_key:
            result.update(split)


def _observed_corpus_action_space_hash(result: dict[str, Any], combo_key: str) -> str:
    combo_result = (result.get("combo_results") or {}).get(combo_key) or {}
    runtime_metrics = combo_result.get("runtime_metrics") or {}
    return _stable_hash({
        "combo_key": combo_key,
        "num_actions": result.get("num_actions"),
        "num_routes": result.get("num_routes"),
        "eve_action_family_counts": result.get("eve_action_family_counts"),
        "enabled_attack_kinds": runtime_metrics.get("enabled_attack_kinds"),
        "attack_surface_scope": runtime_metrics.get("attack_surface_scope"),
    })


def _finalize_corpus_case_result(
    result: dict[str, Any],
    *,
    combo: dict[str, Any],
    case_plan: dict[str, Any],
) -> dict[str, Any]:
    combo_key = _combo_key_from_combo(combo)
    combo_result = (result.get("combo_results") or {}).get(combo_key)
    if not isinstance(combo_result, dict):
        raise CaseValidationError(f"missing combo result for {combo_key}")
    if combo_result.get("error"):
        raise CaseValidationError(f"combo {combo_key} failed: {combo_result['error']}")
    runtime_metrics = combo_result.get("runtime_metrics") or {}
    provenance = _case_provenance_from_metrics(
        case_plan,
        runtime_metrics,
        status="completed",
    )
    _apply_corpus_metric_splits(result)
    result.update(provenance)
    result["case_provenance"] = dict(provenance)
    result["observed_action_space_hash"] = _observed_corpus_action_space_hash(result, combo_key)
    combo_result.update(provenance)
    combo_result["case_provenance"] = dict(provenance)
    combo_result["observed_action_space_hash"] = result["observed_action_space_hash"]
    _validate_case_record(result)
    return result


def _legacy_corpus_worker_args(args: tuple) -> dict[str, Any]:
    dt_refit_config = None
    dt_refit_output_dir = None
    dt_depth = 4
    if len(args) == 5:
        record, corpus_root_str, num_steps, seed, combos = args
    else:
        (
            record,
            corpus_root_str,
            num_steps,
            seed,
            combos,
            *rest,
        ) = args
        if rest:
            dt_refit_config = rest[0]
        if len(rest) > 1 and rest[1] is not None:
            dt_refit_output_dir = Path(rest[1])
        if len(rest) > 2:
            dt_depth = int(rest[2])
    combo = combos[0] if combos else EXPERIMENT_SUITE[0]
    case_plan = _make_case_plan(
        graph_id=str(record.get("graph_id", "")),
        assignment_id=str(record.get("assignment_id", "")),
        combo=combo,
        seed=seed,
        num_steps=num_steps,
        route_policy="legacy_worker_args",
        topology_hash=_stable_hash(record.get("topology_path")),
        action_space_hash=_stable_hash({"combo": combo, "record": _corpus_record_key(record)}),
        config_hash=_stable_hash({"legacy": True, "combo": combo}),
    )
    return {
        "record": record,
        "corpus_root": corpus_root_str,
        "num_steps": num_steps,
        "seed": seed,
        "combo": combo,
        "dt_refit_config": dt_refit_config,
        "dt_refit_output_dir": str(dt_refit_output_dir) if dt_refit_output_dir is not None else None,
        "dt_refit_output_per_case": False,
        "dt_depth": dt_depth,
        "case_plan": case_plan,
    }


def _write_corpus_worker_status(
    task: dict[str, Any],
    status: str,
    *,
    error: str | None = None,
) -> None:
    case_plan = task["case_plan"]
    payload = {
        "status": status,
        "pid": os.getpid(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "case_id": case_plan.get("case_id"),
        "graph_id": case_plan.get("graph_id"),
        "assignment_id": case_plan.get("assignment_id"),
        "combo_key": case_plan.get("combo_key"),
        "seed": case_plan.get("seed"),
        "num_steps": case_plan.get("num_steps"),
        "route_policy": case_plan.get("route_policy"),
    }
    if error is not None:
        payload["error"] = error
    results_db_path = task.get("results_db_path")
    if results_db_path:
        write_worker_status_payload(
            results_db_path,
            payload,
            relative_path=f"corpus_worker_status/{case_plan.get('case_id', 'unknown')}.json",
        )
    status_dir = task.get("worker_status_dir")
    if status_dir:
        _atomic_write_json(
            Path(status_dir) / f"{case_plan.get('case_id', 'unknown')}.json",
            payload,
        )


def _direct_e91_target_active(attack: Any, route: Any) -> bool:
    if attack.attack_kind == "none" or attack.target_kind == "none":
        return False
    if attack.target_kind == "edge":
        return attack.target_id in set(route.edge_ids)
    if attack.target_kind in {"node", "memory"}:
        return attack.target_id in set(route.internal_nodes)
    return False


def _direct_e91_prepared_cache_key(step_input: Any, *, training_light: bool) -> str:
    from sequence_game.protocol.direct_e91_transcript import BACKEND, MODEL_VERSION

    route = step_input.selected_route
    attack = step_input.eve_attack
    payload = {
        "key_version": DIRECT_E91_PREPARED_CACHE_KEY_VERSION,
        "backend": BACKEND,
        "model_version": MODEL_VERSION,
        "selected_route": route.to_dict(),
        "hardware": step_input.hardware.to_dict(),
        "protocol": step_input.protocol.to_dict(),
        "eve_attack": attack.to_dict(),
        "target_active": _direct_e91_target_active(attack, route),
        "training_light": bool(training_light),
    }
    return _stable_hash(payload)


class _DirectE91PreparedTransitionCache:
    def __init__(self) -> None:
        self.entries: dict[str, Any] = {}
        self.hits = 0
        self.misses = 0

    def get_or_prepare(self, step_input: Any, *, training_light: bool) -> Any:
        from sequence_game.protocol import direct_e91_transcript as direct_backend

        key = _direct_e91_prepared_cache_key(
            step_input,
            training_light=training_light,
        )
        prepared = self.entries.get(key)
        if prepared is not None:
            self.hits += 1
            return prepared
        self.misses += 1
        prepared = direct_backend.prepare_direct_e91_transition(
            step_input,
            training_light=training_light,
        )
        self.entries[key] = prepared
        return prepared

    def stats(self) -> dict[str, Any]:
        attempts = self.hits + self.misses
        return {
            "cache_enabled": True,
            "cache_entries": len(self.entries),
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "cache_hit_rate": self.hits / attempts if attempts else 0.0,
            "prepared_cache_key_version": DIRECT_E91_PREPARED_CACHE_KEY_VERSION,
        }


_DIRECT_E91_PREPARED_CACHE = _DirectE91PreparedTransitionCache()
_DIRECT_E91_HARDWARE_CONFIG: Any | None = None
_DIRECT_E91_RUNTIME_CACHE_ENABLED = True
_DIRECT_E91_RUNTIME_TRAINING_LIGHT = False


def _direct_e91_cache_stats() -> dict[str, Any]:
    return _DIRECT_E91_PREPARED_CACHE.stats()


def _direct_e91_cache_delta(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    cache_enabled: bool,
    training_light: bool,
) -> dict[str, Any]:
    hits = int(after.get("cache_hits", 0)) - int(before.get("cache_hits", 0))
    misses = int(after.get("cache_misses", 0)) - int(before.get("cache_misses", 0))
    attempts = hits + misses
    return {
        "cache_enabled": bool(cache_enabled),
        "cache_entries": int(after.get("cache_entries", 0)),
        "cache_hits": hits if cache_enabled else 0,
        "cache_misses": misses if cache_enabled else 0,
        "cache_hit_rate": hits / attempts if cache_enabled and attempts else 0.0,
        "prepared_cache_key_version": DIRECT_E91_PREPARED_CACHE_KEY_VERSION,
        "training_light": bool(training_light),
    }


def _direct_e91_default_hardware_config() -> Any:
    global _DIRECT_E91_HARDWARE_CONFIG
    if _DIRECT_E91_HARDWARE_CONFIG is None:
        from sequence_game.protocol.direct_e91_hardware_adapter import (
            load_direct_e91_hardware_config_from_existing,
        )

        _DIRECT_E91_HARDWARE_CONFIG = load_direct_e91_hardware_config_from_existing()
    return _DIRECT_E91_HARDWARE_CONFIG


def _run_direct_e91_step_with_cache(
    step_input: Any,
    *,
    cache_enabled: bool,
    training_light: bool,
) -> Any:
    from sequence_game.protocol import direct_e91_transcript as direct_backend

    if not cache_enabled:
        return direct_backend.run_direct_e91_transcript_step(
            step_input,
            training_light=training_light,
        )
    prepared = _DIRECT_E91_PREPARED_CACHE.get_or_prepare(
        step_input,
        training_light=training_light,
    )
    return direct_backend.sample_prepared_direct_e91_transition(
        prepared,
        step_input.seed,
        step_input.step_index,
        requested_pairs=step_input.protocol.requested_pairs,
        training_light=training_light,
    )


def _install_corpus_runtime_backend(runtime_backend: str) -> None:
    if runtime_backend == RUNTIME_BACKEND_SEQUENCE:
        corpus_runner_module.run_e91_graph_runtime_game = run_e91_graph_runtime_game
        return
    if runtime_backend == RUNTIME_BACKEND_DIRECT_E91:
        corpus_runner_module.run_e91_graph_runtime_game = (
            run_direct_e91_transcript_runtime_game
        )
        return
    if runtime_backend == RUNTIME_BACKEND_COMPILED_HISTOGRAM:
        corpus_runner_module.run_e91_graph_runtime_game = (
            run_compiled_histogram_runtime_game
        )
        return
    raise ValueError(f"unsupported runtime backend {runtime_backend!r}")


def _attach_direct_cache_diagnostics(
    result: dict[str, Any],
    combo: dict[str, Any],
    diagnostics: dict[str, Any],
) -> None:
    key = _combo_key_from_combo(combo)
    combo_result = (result.get("combo_results") or {}).get(key)
    if not isinstance(combo_result, dict):
        return
    runtime_metrics = combo_result.setdefault("runtime_metrics", {})
    runtime_metrics["direct_e91_prepared_cache"] = dict(diagnostics)
    result["direct_e91_prepared_cache"] = dict(diagnostics)


def _direct_e91_protocol_config(requested_pairs: int) -> Any:
    from sequence_game.corpus import e91_runtime_game as sequence_runtime
    from sequence_game.protocol.direct_e91_transcript import DirectE91ProtocolConfig

    return DirectE91ProtocolConfig(
        requested_pairs=int(requested_pairs),
        qber_abort_threshold=sequence_runtime.DEFAULT_E91_QBER_THRESHOLD,
        min_key_pairs_for_qber_abort=(
            sequence_runtime.DEFAULT_E91_MIN_SIFTED_SAMPLES
        ),
        min_chsh_pairs_for_chsh_abort=16,
    )


def _direct_e91_route_from_runtime_route(
    ir_dict: dict[str, Any],
    route: dict[str, Any],
) -> Any:
    from sequence_game.protocol.direct_e91_transcript import DirectE91Route

    path = tuple(str(node) for node in route["path"])
    edge_ids = tuple(str(edge_id) for edge_id in route.get("edge_ids") or [])
    edge_lengths_raw = route.get("edge_lengths_m") or ()
    if len(edge_ids) != len(path) - 1:
        edge_ids = tuple(_direct_e91_edge_label(u, v) for u, v in zip(path, path[1:]))
    if len(edge_lengths_raw) == len(edge_ids):
        lengths = {
            edge_id: float(length)
            for edge_id, length in zip(edge_ids, edge_lengths_raw)
        }
    else:
        lengths = {}
        for edge_id, u, v in zip(edge_ids, path, path[1:]):
            edge = _direct_e91_edge_between(ir_dict, u, v)
            lengths[edge_id] = float(edge.get("length_m", edge.get("length", 1.0)))

    fidelities: dict[str, float] = {}
    transmittances: dict[str, float] = {}
    losses_db: dict[str, float] = {}
    for edge_id, u, v in zip(edge_ids, path, path[1:]):
        edge = _direct_e91_edge_between(ir_dict, u, v)
        for key in (
            "elementary_pair_fidelity",
            "edge_fidelity",
            "fidelity",
            "polarization_fidelity",
        ):
            if edge.get(key) is not None:
                fidelities[edge_id] = min(1.0, max(0.0, float(edge[key])))
                break
        if edge.get("transmittance") is not None:
            transmittances[edge_id] = min(1.0, max(0.0, float(edge["transmittance"])))
        if edge.get("loss_db") is not None:
            losses_db[edge_id] = float(edge["loss_db"])
    return DirectE91Route(
        route_id=str(route["route_id"]),
        node_path=path,
        edge_ids=edge_ids,
        edge_lengths_m=lengths,
        total_length_m=float(route.get("total_length_m", sum(lengths.values()))),
        edge_fidelities=fidelities,
        edge_transmittances=transmittances,
        edge_loss_db=losses_db,
    )


def _direct_e91_edge_between(ir_dict: dict[str, Any], u: str, v: str) -> dict[str, Any]:
    for edge in ir_dict.get("edges", []):
        if {str(edge.get("u")), str(edge.get("v"))} == {str(u), str(v)}:
            return dict(edge)
    raise ValueError(f"no topology edge for route hop {u!r}->{v!r}")


def _direct_e91_edge_label(u: str, v: str) -> str:
    a, b = sorted((str(u), str(v)))
    return f"{a}-{b}"


def _direct_e91_attack_from_runtime_action(
    action: dict[str, Any],
    *,
    attack_cost: float,
) -> Any:
    from sequence_game.protocol.direct_e91_transcript import DirectE91AttackSpec

    attack_type = str(action.get("attack_type") or "no_attack")
    action_id = str(action.get("id") or attack_type)
    target = str(action.get("target") or "")
    _ = attack_cost
    if attack_type == "no_attack":
        return DirectE91AttackSpec(attack_id=action_id)
    if attack_type == "edge_dos":
        return DirectE91AttackSpec(
            attack_id=action_id,
            attack_kind="edge_added_loss",
            target_kind="edge",
            target_id=target,
            loss_multiplier=0.0,
        )
    if attack_type == "edge_information_probe":
        return DirectE91AttackSpec(
            attack_id=action_id,
            attack_kind="edge_intercept_resend",
            target_kind="edge",
            target_id=target,
            intercept_basis_label="Z",
            intercept_probability=1.0,
            information_probability=1.0,
        )
    if attack_type == "swap_denial":
        return DirectE91AttackSpec(
            attack_id=action_id,
            attack_kind="swap_denial",
            target_kind="node",
            target_id=target,
        )
    if attack_type == "memory_degradation":
        return DirectE91AttackSpec(
            attack_id=action_id,
            attack_kind="memory_degradation",
            target_kind="memory",
            target_id=target,
            memory_depolarizing_probability=0.5,
            memory_dephasing_probability=0.5,
        )
    if attack_type == "repeater_memory_measure_Z":
        return DirectE91AttackSpec(
            attack_id=action_id,
            attack_kind="repeater_memory_measure_Z",
            target_kind="memory",
            target_id=target,
            measurement_probability=1.0,
            information_probability=1.0,
        )
    if attack_type == "repeater_memory_measure_X":
        return DirectE91AttackSpec(
            attack_id=action_id,
            attack_kind="repeater_memory_measure_X",
            target_kind="memory",
            target_id=target,
            measurement_probability=1.0,
            information_probability=1.0,
        )
    raise ValueError(f"unsupported direct E91 runtime action {action!r}")


def _direct_e91_information_metric(result: Any) -> dict[str, Any]:
    return {
        "num_sifted": int(result.sifted_key_count),
        "num_eve_records": int(result.eve_observation_count),
        "mutual_information_bits": max(0.0, float(result.eve_guess_probability) - 0.5),
        "fraction_correct": float(result.eve_guess_probability),
        "metric_source": "direct_e91_transcript.eve_guess_probability",
    }


def _direct_e91_step_record(
    *,
    step_idx: int,
    game_mode: str,
    alice_algo: str,
    eve_algo: str,
    route: dict[str, Any],
    action: dict[str, Any],
    result: Any,
    alice_reward: float,
    eve_reward: float,
    fallback_used: bool,
    active_route_attacked: bool,
    reward_components: dict[str, Any],
    requested_pairs: int,
) -> dict[str, Any]:
    info_payload = _direct_e91_information_metric(result)
    physical_pair_arrival = bool(result.delivered_pair_count > 0)
    delivery_success = result.public_outcome == "accepted"
    quality_metrics = {
        "qber": result.qber_key,
        "fidelity": result.actual_final_bell_fidelity,
        "chsh_S": result.chsh_S_observed,
        "chsh_S_expected": result.chsh_S_expected,
        "physical_pair_arrival": physical_pair_arrival,
        "delivery_success": delivery_success,
        "bell_qualified_delivery": delivery_success,
    }
    return {
        "step": step_idx,
        "mode": game_mode,
        "scope_label": REPEATER_RUNTIME,
        "backend": DIRECT_E91_RUNTIME_BACKEND,
        "claim_status": "implemented_and_testable",
        "alice_policy": alice_algo,
        "eve_policy": eve_algo,
        "alice_route_id": route["route_id"],
        "alice_path": list(route["path"]),
        "eve_action": {
            "id": action["id"],
            "kind": action["kind"],
            "target": action["target"],
            "attack_type": action["attack_type"],
        },
        "runtime_engine": "direct_e91_transcript_step",
        "runtime_attack_applied": result.eve_action if hasattr(result, "eve_action") else result.attack_kind,
        "configured_repeater_attack": None,
        "edge_attack_kind": result.attack_kind if result.target_kind == "edge" else None,
        "node_attack_kind": result.attack_kind if result.target_kind in {"node", "memory"} else None,
        "public_outcome": result.public_outcome,
        "information_exposed": bool(result.information_exposed),
        "secure_delivery": bool(result.secure_delivery),
        "eve_win": bool(result.eve_win),
        "collision": result.public_outcome in {"delivery_failure", "qber_abort", "chsh_abort"},
        "collision_or_delivery_failure": result.public_outcome in {
            "delivery_failure",
            "qber_abort",
            "chsh_abort",
        },
        "alice_reward": float(alice_reward),
        "eve_reward": float(eve_reward),
        "fallback_used": bool(fallback_used),
        "route_features": {
            "hops": route["hop_count"],
            "length": route["total_length"],
            "internal_nodes": list(route.get("internal_nodes") or []),
            "edges": list(route.get("edge_ids") or []),
            "scope_label": REPEATER_RUNTIME,
        },
        "quality_metrics": quality_metrics,
        "delivery_success": quality_metrics["delivery_success"],
        "physical_pair_arrival": physical_pair_arrival,
        "bell_qualified_delivery": delivery_success,
        "delivered_count": int(result.delivered_pair_count),
        "sifted_count": int(result.sifted_key_count),
        "qber": result.qber_key,
        "chsh_S": result.chsh_S_observed,
        "chsh_S_expected": result.chsh_S_expected,
        "fidelity": result.actual_final_bell_fidelity,
        "active_route_attacked": bool(active_route_attacked),
        "memory_intervention_public": None,
        "router_net_summary": {
            "route_state_model": result.route_state_model,
            "elementary_link_count": result.elementary_link_count,
            "bsm_swap_count": result.bsm_swap_count,
            "backend": DIRECT_E91_RUNTIME_BACKEND,
        },
        "information_gain_reward_enabled": False,
        "information_gain_metric": info_payload,
        "eve_information_metric": info_payload,
        "e91_periods_per_step": int(requested_pairs),
        "repeater_memory_pairs_per_step": int(requested_pairs),
        "metrics": {
            "scope_label": REPEATER_RUNTIME,
            "backend": DIRECT_E91_RUNTIME_BACKEND,
            "runtime_engine": "direct_e91_transcript_step",
            **quality_metrics,
            "runtime_attack_applied": result.attack_kind,
            "configured_repeater_attack": None,
            "edge_attack_kind": result.attack_kind if result.target_kind == "edge" else None,
            "node_attack_kind": result.attack_kind if result.target_kind in {"node", "memory"} else None,
            "memory_intervention_public": None,
            "reward_components": reward_components,
            "active_route_attacked": bool(active_route_attacked),
            "information_gain_reward_enabled": False,
            "information_gain_metric": info_payload,
            "e91_periods_per_step": int(requested_pairs),
            "repeater_memory_pairs_per_step": int(requested_pairs),
        },
    }


def _direct_e91_numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _direct_e91_summary(
    *,
    game_mode: str,
    seed: int,
    alice: str,
    bob: str,
    alice_algo: str,
    eve_algo: str,
    routes: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    final_window: list[dict[str, Any]],
    final_window_size: int,
    eve_q: Any,
    alice_q: Any,
    include_alice_q: bool,
    started: float,
    step_seconds: list[float],
    requested_pairs: int,
    cache_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    from sequence_game.corpus import e91_runtime_game as sequence_runtime

    route_ids = [str(step.get("alice_route_id")) for step in steps if step.get("alice_route_id")]
    outcomes = Counter(str(step.get("public_outcome") or "unknown") for step in steps)
    qbers = [float(step["qber"]) for step in steps if step.get("qber") is not None]
    fidelities = [
        float(step["fidelity"]) for step in steps if step.get("fidelity") is not None
    ]
    infos = [
        float((step.get("information_gain_metric") or {}).get("mutual_information_bits") or 0.0)
        for step in steps
    ]
    edge_targets = sorted({
        action["target"] for action in actions if action.get("kind") == "edge"
    })
    node_targets = sorted({
        action["target"] for action in actions if action.get("kind") == "node"
    })
    metrics = {
        "scope_label": REPEATER_RUNTIME,
        "backend": DIRECT_E91_RUNTIME_BACKEND,
        "training_runtime_ms": round((time.perf_counter() - started) * 1000, 3),
        "training_iterations": len(steps),
        "training_execution": (
            "online route/action updates; each step samples direct E91 transcript"
        ),
        "control_model": game_mode,
        "eve_algo": eve_algo,
        "alice_algo": alice_algo,
        "sequence_runtime_executed": False,
        "direct_e91_runtime_executed": True,
        "full_graph_runtime": True,
        "selected_path_runtime": True,
        "runtime_engines": ["direct_e91_transcript_step"],
        "runtime_case_count": len(steps),
        "e91_periods_per_step": int(requested_pairs),
        "repeater_memory_pairs_per_step": int(requested_pairs),
        "reward_model": "public_heralded_service_complement_minus_attack_cost",
        "route_entropy": sequence_runtime._route_entropy(route_ids),
        "route_switches": sum(1 for a, b in zip(route_ids, route_ids[1:]) if a != b),
        "detected_period": sequence_runtime.detect_route_period(route_ids),
        "enabled_attack_kinds": sorted({
            action["attack_type"] for action in actions
            if action["attack_type"] != "no_attack"
        }),
        "attack_surface_scope": "direct_e91_transcript_route_corridor",
        "attack_surface_node_count": len(node_targets),
        "attack_surface_edge_count": len(edge_targets),
        "attack_surface_node_targets": node_targets,
        "attack_surface_edge_targets": edge_targets,
        "information_gain_reward_enabled": False,
        "information_gain_metric_source": "direct_e91_transcript.eve_guess_probability",
        "information_gain_mean_bits": float(np.mean(infos)) if infos else 0.0,
        "information_gain_max_bits": max(infos) if infos else 0.0,
        "information_exposures": sum(1 for step in steps if step.get("information_exposed")),
        "delivery_failures": outcomes.get("delivery_failure", 0),
        "aborts": outcomes.get("qber_abort", 0) + outcomes.get("chsh_abort", 0),
        "public_outcome_counts": dict(sorted(outcomes.items())),
        "qber": float(np.mean(qbers)) if qbers else None,
        "qber_summary": _direct_e91_numeric_summary(qbers),
        "qber_by_eve_win": sequence_runtime._numeric_by_eve_win(steps, "qber"),
        "fidelity_by_eve_win": sequence_runtime._numeric_by_eve_win(steps, "fidelity"),
        "information_gain_by_eve_win": sequence_runtime._information_gain_by_eve_win(steps),
        "quality_by_public_outcome": sequence_runtime._quality_groups(
            steps,
            lambda step: str(step.get("public_outcome") or "unknown"),
        ),
        "quality_by_attack_type": sequence_runtime._quality_groups(
            steps,
            lambda step: str((step.get("eve_action") or {}).get("attack_type") or "unknown"),
        ),
        "fidelity": float(np.mean(fidelities)) if fidelities else None,
        "fidelity_summary": _direct_e91_numeric_summary(fidelities),
        "delivered_pair_count": sum(int(step.get("delivered_count") or 0) for step in steps),
        "eve_attack_selection_counts": dict(sorted(Counter(
            str((step.get("eve_action") or {}).get("attack_type") or "unknown")
            for step in steps
        ).items())),
        "final_replay_window": final_window_size,
        "mean_step_wall_clock_seconds": (
            float(np.mean(step_seconds)) if step_seconds else None
        ),
        "direct_e91_prepared_cache": dict(cache_diagnostics),
        "no_convergence_claim": True,
    }
    eve_q_export = sequence_runtime._bounded_q_export(
        eve_q, sequence_runtime._EVE_Q_EXPORT_MAX_ENTRIES
    )
    alice_q_export = (
        sequence_runtime._bounded_q_export(
            alice_q, sequence_runtime._ALICE_Q_EXPORT_MAX_ENTRIES
        )
        if include_alice_q else None
    )
    return {
        "run_id": f"{game_mode}-{seed}-{len(steps)}",
        "scope_label": REPEATER_RUNTIME,
        "claim_status": "implemented_and_testable",
        "backend": DIRECT_E91_RUNTIME_BACKEND,
        "seed": seed,
        "mode": game_mode,
        "num_steps": len(steps),
        "alice": alice,
        "bob": bob,
        "routes": routes,
        "final_window": final_window,
        "q_tables": {
            "eve": {
                "top_actions": sequence_runtime._q_top_actions(
                    eve_q, actions, export=eve_q_export
                ),
                "raw": eve_q_export,
                "action_names": [action["id"] for action in actions],
                "observation_state": ["previous_action", "previous_public_outcome"],
                "enabled_attack_kinds": metrics["enabled_attack_kinds"],
                "attack_surface_scope": metrics["attack_surface_scope"],
                "control_model": game_mode,
                "reward_model": metrics["reward_model"],
                "information_gain_reward_enabled": False,
            },
            "alice": {
                "top_actions": sequence_runtime._q_top_actions(
                    alice_q,
                    [{"id": route["route_id"], "kind": "route", "target": route["path"]}
                     for route in routes],
                    export=alice_q_export,
                ) if include_alice_q else [],
                "raw": alice_q_export,
                "action_names": [route["route_id"] for route in routes],
                "observation_state": [
                    "previous_route",
                    "previous_public_outcome",
                    "avoided_route_bitset",
                ],
            },
        },
        "metrics": metrics,
    }


def run_direct_e91_transcript_runtime_game(
    ir_dict: dict[str, Any],
    alice: str,
    bob: str,
    *,
    routes: list[dict[str, Any]] | None,
    game_mode: str,
    eve_algo: str,
    alice_algo: str,
    num_steps: int,
    seed: int,
    final_window_size: int = 20,
    e91_periods_per_step: int = 1200,
    repeater_memory_pairs_per_step: int = 8,
    attack_cost: float = 0.05,
    dt_refit_observer: Any = None,
    dt_refit_callback: Any = None,
    direct_e91_prepared_cache_enabled: bool | None = None,
    direct_e91_training_light: bool | None = None,
    progress_every: int = 0,
    progress_label: str | None = None,
) -> dict[str, Any]:
    from sequence_game.corpus import e91_runtime_game as sequence_runtime
    from sequence_game.protocol.direct_e91_transcript import DirectE91StepInput

    _ = repeater_memory_pairs_per_step
    if direct_e91_prepared_cache_enabled is None:
        direct_e91_prepared_cache_enabled = _DIRECT_E91_RUNTIME_CACHE_ENABLED
    if direct_e91_training_light is None:
        direct_e91_training_light = _DIRECT_E91_RUNTIME_TRAINING_LIGHT
    started = time.perf_counter()
    ir = sequence_runtime.TopologyIR.from_dict(ir_dict)
    runtime_routes = sequence_runtime._normalize_routes(ir, alice, bob, routes)
    if not runtime_routes:
        raise ValueError(f"no Alice-Bob routes available for {alice!r}->{bob!r}")
    actions = sequence_runtime._runtime_actions(ir, runtime_routes, game_mode)
    hardware = _direct_e91_default_hardware_config()
    protocol = _direct_e91_protocol_config(e91_periods_per_step)

    rng = np.random.default_rng(seed)
    eve_q = sequence_runtime._make_agent(eve_algo, len(actions))
    alice_q = sequence_runtime._make_agent(alice_algo, len(runtime_routes))
    q_cfg = sequence_runtime.QLearningConfig(alpha=0.2, gamma=0.9)

    avoided_until: dict[str, int] = {}
    previous_eve_action = "start"
    previous_public_outcome = "start"
    previous_route_id = "start"
    steps: list[dict[str, Any]] = []
    step_seconds: list[float] = []
    cache_before = _direct_e91_cache_stats()
    progress_interval = max(0, int(progress_every or 0))
    progress_name = progress_label or f"{game_mode}__{eve_algo}__{alice_algo}"
    eve_win_count = 0

    for step_idx in range(num_steps):
        step_started = time.perf_counter()
        route, route_idx, alice_state, fallback_used = sequence_runtime._select_route(
            runtime_routes,
            alice_algo,
            alice_q,
            previous_route_id,
            previous_public_outcome,
            avoided_until,
            step_idx,
            num_steps,
            rng,
        )
        action, eve_action_idx, eve_state = sequence_runtime._select_action(
            actions,
            route,
            eve_algo,
            eve_q,
            previous_eve_action,
            previous_public_outcome,
            step_idx,
            num_steps,
            rng,
        )
        direct_route = _direct_e91_route_from_runtime_route(ir_dict, route)
        direct_attack = _direct_e91_attack_from_runtime_action(
            action,
            attack_cost=attack_cost,
        )
        step_input = DirectE91StepInput(
            topology=None,
            alice_node_id=alice,
            bob_node_id=bob,
            selected_route=direct_route,
            eve_attack=direct_attack,
            hardware=hardware,
            protocol=protocol,
            seed=seed + 100_000 + step_idx,
            step_index=step_idx,
        )
        direct_result = _run_direct_e91_step_with_cache(
            step_input,
            cache_enabled=direct_e91_prepared_cache_enabled,
            training_light=direct_e91_training_light,
        )
        public_outcome = str(direct_result.public_outcome)
        active_route_attacked = bool(direct_result.target_active)
        # Alice can react only to the authenticated public protocol decision;
        # Eve's private information record is not an Alice observation.
        failure_or_detection = public_outcome != "accepted"
        if direct_result.eve_win:
            eve_win_count += 1
        if failure_or_detection and alice_algo == "cautious_greedy":
            avoided_until[route["route_id"]] = step_idx + 1

        cost = 0.0 if action.get("attack_type") == "no_attack" else float(attack_cost)
        eve_feedback = eve_feedback_from_heralded_service(
            public_outcome=public_outcome,
            requested_pair_count=e91_periods_per_step,
            delivered_pair_count=int(direct_result.delivered_pair_count),
            attack_cost=cost,
            local_observation_kind="attack_instrument_private_record",
            local_observation_count=int(direct_result.eve_observation_count),
        )
        eve_reward = eve_feedback.reward
        alice_reward = 1.0 if public_outcome == "accepted" else -1.0
        if fallback_used:
            alice_reward -= 0.2
        reward_components = {
            "scope_label": REPEATER_RUNTIME,
            "backend": DIRECT_E91_RUNTIME_BACKEND,
            "public_outcome_reward": 1.0 if public_outcome != "accepted" else 0.0,
            "information_exposure": bool(direct_result.information_exposed),
            "secure_delivery": bool(direct_result.secure_delivery),
            "attack_cost": cost,
            "eve_feedback": eve_feedback.to_dict(),
            "target_active_used_for_eve_feedback": False,
            "information_exposure_rewarded": False,
            "metric_source": "direct_e91_transcript",
            "direct_reward_components": dict(direct_result.reward_components),
        }

        next_eve_state = (action["id"], public_outcome)
        if eve_algo in {"q_learning", "deep_q"}:
            eve_q.update(eve_state, eve_action_idx, eve_reward, next_eve_state, q_cfg)
        if alice_algo in {"q_learning", "deep_q"}:
            next_avoid = sequence_runtime._avoidance_signature(
                runtime_routes,
                avoided_until,
                step_idx,
            )
            next_alice_state = (route["route_id"], public_outcome, next_avoid)
            alice_q.update(alice_state, route_idx, alice_reward, next_alice_state, q_cfg)

        record = _direct_e91_step_record(
            step_idx=step_idx,
            game_mode=game_mode,
            alice_algo=alice_algo,
            eve_algo=eve_algo,
            route=route,
            action=action,
            result=direct_result,
            alice_reward=alice_reward,
            eve_reward=eve_reward,
            fallback_used=fallback_used,
            active_route_attacked=active_route_attacked,
            reward_components=reward_components,
            requested_pairs=e91_periods_per_step,
        )
        steps.append(record)
        step_seconds.append(time.perf_counter() - step_started)
        step_num = step_idx + 1
        if progress_interval and (
            step_num % progress_interval == 0 or step_num == num_steps
        ):
            elapsed = time.perf_counter() - started
            cache_now = _direct_e91_cache_stats()
            print(
                "  [case-progress] "
                f"{progress_name} step={step_num}/{num_steps} "
                f"routes={len(runtime_routes)} actions={len(actions)} "
                f"eve_win={eve_win_count / step_num:.2%} "
                f"mean_step={elapsed / step_num:.3f}s "
                f"elapsed={elapsed:.1f}s "
                f"cache={int(cache_now.get('cache_hits', 0))}/"
                f"{int(cache_now.get('cache_misses', 0))} "
                f"last={public_outcome}",
                flush=True,
            )

        if dt_refit_observer is not None:
            dt_refit_observer.record_transition(
                alice_action=route_idx,
                eve_action=eve_action_idx,
                alice_route=route.get("route_id"),
                eve_target=action.get("target"),
                reward=eve_reward,
            )
            tick = dt_refit_observer.observe(
                step=step_idx + 1,
                alice_q=alice_q if alice_algo in {"q_learning", "deep_q"} else None,
                eve_q=eve_q if eve_algo in {"q_learning", "deep_q"} else None,
            )
            if tick is not None and tick.refit_recommended:
                event = dt_refit_observer.refit_event_from_tick(tick)
                if dt_refit_callback is not None:
                    dt_refit_callback(event, {
                        "step": step_idx + 1,
                        "routes": runtime_routes,
                        "actions": actions,
                        "steps": steps,
                        "alice_q": alice_q,
                        "eve_q": eve_q,
                        "alice_algo": alice_algo,
                        "eve_algo": eve_algo,
                        "control_model": game_mode,
                    })
                dt_refit_observer.mark_refit(
                    step=step_idx + 1,
                    alice_q=alice_q if alice_algo in {"q_learning", "deep_q"} else None,
                    eve_q=eve_q if eve_algo in {"q_learning", "deep_q"} else None,
                    reason=tick.refit_reason,
                )

        previous_eve_action = action["id"]
        previous_public_outcome = public_outcome
        previous_route_id = route["route_id"]

    final_window = steps[-min(final_window_size, len(steps)):]
    cache_after = _direct_e91_cache_stats()
    cache_diagnostics = _direct_e91_cache_delta(
        cache_before,
        cache_after,
        cache_enabled=direct_e91_prepared_cache_enabled,
        training_light=direct_e91_training_light,
    )
    summary = _direct_e91_summary(
        game_mode=game_mode,
        seed=seed,
        alice=alice,
        bob=bob,
        alice_algo=alice_algo,
        eve_algo=eve_algo,
        routes=runtime_routes,
        actions=actions,
        steps=steps,
        final_window=final_window,
        final_window_size=final_window_size,
        eve_q=eve_q,
        alice_q=alice_q,
        include_alice_q=alice_algo in {"q_learning", "deep_q"},
        started=started,
        step_seconds=step_seconds,
        requested_pairs=e91_periods_per_step,
        cache_diagnostics=cache_diagnostics,
    )
    return {
        "summary": summary,
        "steps": steps,
        "final_window": final_window,
        "scope_label": REPEATER_RUNTIME,
    }


# ── Compiled-histogram runtime backend ───────────────────────────────────────

_COMPILED_HISTOGRAM_BACKEND_CONFIG: dict[str, Any] = {}
_COMPILED_HISTOGRAM_BACKEND_INSTANCE: Any = None
_COMPILED_HISTOGRAM_BACKEND_POOL: dict[str, Any] = {}
_COMPILED_HISTOGRAM_BACKEND_POOL_METADATA: dict[str, dict[str, Any]] = {}
_COMPILED_HISTOGRAM_RESOLVED_CONFIG_CACHE: dict[Any, tuple[dict[str, Any], dict[str, Any]]] = {}

_COMPILED_HISTOGRAM_CONTROL_CONFIG_KEYS = {
    "rolling_hop_calibration",
    "rolling_hop_calibration_10km",
}


def _compiled_histogram_backend_config_payload(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = _COMPILED_HISTOGRAM_BACKEND_CONFIG if config is None else dict(config)
    return {
        key: value
        for key, value in raw.items()
        if key not in _COMPILED_HISTOGRAM_CONTROL_CONFIG_KEYS
    }


def _compiled_histogram_hop_calibration_spec(
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    raw = _COMPILED_HISTOGRAM_BACKEND_CONFIG if config is None else dict(config)
    spec = raw.get("rolling_hop_calibration")
    if spec is None:
        spec = raw.get("rolling_hop_calibration_10km")
    if spec is None:
        return None
    if spec is True:
        spec = {}
    if not isinstance(spec, dict):
        raise ValueError("rolling_hop_calibration must be a mapping or true")
    out = dict(spec)
    out.setdefault("kind", "hop_based_10km")
    return out


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _compiled_histogram_resolved_config_for_hop(
    hop_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve static or hop-calibrated backend config for a selected route."""

    hop = int(hop_count)
    if hop < 1:
        raise ValueError("compiled histogram route hop count must be >= 1")
    cached = _COMPILED_HISTOGRAM_RESOLVED_CONFIG_CACHE.get(hop)
    if cached is not None:
        return cached

    base = _compiled_histogram_backend_config_payload()
    spec = _compiled_histogram_hop_calibration_spec()
    if spec is None:
        metadata = {
            "config_mode": "static_backend_config",
            "hop_count": hop,
            "hop_calibrated": False,
        }
        resolved = dict(base)
        _COMPILED_HISTOGRAM_RESOLVED_CONFIG_CACHE[hop] = (resolved, metadata)
        return resolved, metadata

    if str(spec.get("kind")) != "hop_based_10km":
        raise ValueError(
            "unsupported compiled histogram rolling_hop_calibration kind "
            f"{spec.get('kind')!r}"
        )
    from scripts.derive_hop_based_rolling_reservation_calibration import (
        HopCalibrationConfig,
        derive_hop_reservation,
    )
    from sequence_game.physical.memory_pool_reservation import route_live_probability

    cal_config = HopCalibrationConfig(
        max_hop=hop,
        stress_max_hop=hop,
        edge_length_km=float(spec.get("edge_length_km", 10.0)),
        accepted_alive_time_s=float(spec.get("accepted_alive_time_s", 0.18)),
        route_availability_target=float(spec.get("route_availability_target", 0.95)),
        chsh_target=float(spec.get("chsh_target", 2.10)),
        age_points_per_edge=int(spec.get("age_points_per_edge", 8)),
        selection_policy=str(spec.get("selection_policy", "youngest_live")),
        hardware_config_dir=str(
            spec.get(
                "hardware_config_dir",
                base.get("hardware_config_dir", "configs/physical"),
            )
        ),
        detector_kind=str(spec.get("detector_kind", base.get("detector_kind", "pnr_snspd"))),
        detector_profile_path=spec.get(
            "detector_profile_path",
            base.get(
                "detector_profile_path",
                str(
                    _ROOT
                    / "configs"
                    / "physical"
                    / "detectors"
                    / "snspd_colangelo_impedance_matched_differential_pnr_1550.json"
                ),
            ),
        ),
        detector_response_path=spec.get(
            "detector_response_path", base.get("detector_response_path")
        ),
        fiber_response_path=spec.get(
            "fiber_response_path", base.get("fiber_response_path")
        ),
        conversion_response_path=spec.get(
            "conversion_response_path",
            base.get(
                "conversion_response_path",
                str(
                    _ROOT
                    / "configs"
                    / "physical"
                    / "conversion"
                    / "cui_dfg_866_to_1558.json"
                ),
            ),
        ),
        multiplexing_controller_path=spec.get(
            "multiplexing_controller_path",
            base.get("multiplexing_controller_path"),
        ),
        allow_wavelength_mismatch=bool(
            spec.get(
                "allow_wavelength_mismatch",
                base.get("allow_wavelength_mismatch", False),
            )
        ),
        allow_ideal_detector_reference=bool(
            spec.get(
                "allow_ideal_detector_reference",
                base.get("allow_ideal_detector_reference", False),
            )
        ),
        fock_cutoff=int(spec.get("fock_cutoff", base.get("fock_cutoff", 2))),
        key_length=int(
            spec.get(
                "key_length",
                spec.get("compile_key_length", base.get("compile_key_length", 8)),
            )
        ),
        c_max=int(spec.get("c_max", 64)),
        rolling_age_support_product_limit=int(
            spec.get(
                "rolling_age_support_product_limit",
                base.get("rolling_age_support_product_limit", 1_000_000),
            )
        ),
        ignore_detected_decay_in_pool=bool(
            spec.get("ignore_detected_decay_in_pool", False)
        ),
    )
    _edge_inputs, _front_meta, plan, edges, payload = derive_hop_reservation(
        hop,
        cal_config,
    )
    resolved = {
        **base,
        "hardware_config_dir": cal_config.hardware_config_dir,
        "detector_kind": cal_config.detector_kind,
        "detector_profile_path": cal_config.detector_profile_path,
        "detector_response_path": cal_config.detector_response_path,
        "fiber_response_path": cal_config.fiber_response_path,
        "conversion_response_path": cal_config.conversion_response_path,
        "allow_wavelength_mismatch": cal_config.allow_wavelength_mismatch,
        "allow_ideal_detector_reference": cal_config.allow_ideal_detector_reference,
        "fock_cutoff": cal_config.fock_cutoff,
        "waiting_protocol": "rolling_alive_pair",
        "rolling_edge_availability": _tuple_tree(payload["rolling_edge_availability"]),
        "rolling_age_distribution_kind": payload["rolling_age_distribution_kind"],
        "rolling_age_distribution_points_s": _tuple_tree(
            payload["rolling_age_distribution_points_s"]
        ),
        "rolling_age_distribution_weights": _tuple_tree(
            payload["rolling_age_distribution_weights"]
        ),
        "rolling_edge_age_distribution_points_s": _tuple_tree(
            payload["rolling_edge_age_distribution_points_s"]
        ),
        "rolling_edge_age_distribution_weights": _tuple_tree(
            payload["rolling_edge_age_distribution_weights"]
        ),
        "rolling_age_evaluation": "moment_reduced",
        "rolling_age_support_product_limit": cal_config.rolling_age_support_product_limit,
        "rolling_accepted_alive_time_s": payload["rolling_accepted_alive_time_s"],
        "rolling_selection_policy": payload["rolling_selection_policy"],
        "rolling_age_distribution_source": payload["rolling_age_distribution_source"],
    }
    if cal_config.multiplexing_controller_path is not None:
        resolved["multiplexing_controller_path"] = cal_config.multiplexing_controller_path
    metadata = {
        "config_mode": "rolling_hop_calibration",
        "hop_calibrated": True,
        "calibration_kind": "hop_based_10km",
        "hop_count": hop,
        "calibration_edge_length_km": cal_config.edge_length_km,
        "runtime_lengths_policy": (
            "rolling reservation resources are derived from the standardized "
            "10 km-per-hop calibration family; the compiled histogram still "
            "evaluates optical propagation at the selected route's actual "
            "fiber vector"
        ),
        "accepted_alive_time_s": cal_config.accepted_alive_time_s,
        "route_availability_target": cal_config.route_availability_target,
        "chsh_target": cal_config.chsh_target,
        "age_points_per_edge": cal_config.age_points_per_edge,
        "selection_policy": cal_config.selection_policy,
        "detector_kind": cal_config.detector_kind,
        "detector_profile_path": cal_config.detector_profile_path,
        "conversion_response_path": cal_config.conversion_response_path,
        "allow_wavelength_mismatch": cal_config.allow_wavelength_mismatch,
        "slot_vector": list(plan.reserved_slots),
        "total_slots": int(sum(plan.reserved_slots)),
        "edge_availabilities": list(plan.edge_availabilities),
        "route_availability": route_live_probability(edges),
        "rolling_age_support_product_avoided": (
            int(cal_config.age_points_per_edge) ** hop
        ),
        "rolling_age_evaluation": "moment_reduced",
    }
    if cal_config.multiplexing_controller_path is not None:
        metadata["multiplexing_controller_path"] = cal_config.multiplexing_controller_path
    _COMPILED_HISTOGRAM_RESOLVED_CONFIG_CACHE[hop] = (resolved, metadata)
    return resolved, metadata


def _compiled_histogram_resolved_config_for_service_route(
    hop_count: int,
    fiber_lengths_m: Sequence[float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a fixed-inventory service profile at the route's actual lengths."""

    hop = int(hop_count)
    lengths_m = tuple(float(value) for value in fiber_lengths_m)
    if hop < 1 or len(lengths_m) != hop:
        raise ValueError("service-profile route needs one positive fiber length per hop")
    if any(value <= 0.0 for value in lengths_m):
        raise ValueError("service-profile route fiber lengths must be positive")
    cache_key = ("repeater_service_profile", hop, lengths_m)
    cached = _COMPILED_HISTOGRAM_RESOLVED_CONFIG_CACHE.get(cache_key)
    if cached is not None:
        return cached

    spec = _compiled_histogram_hop_calibration_spec()
    if spec is None or str(spec.get("kind")) != "repeater_service_profile":
        raise ValueError("active rolling calibration is not a repeater_service_profile")
    base = _compiled_histogram_backend_config_payload()
    profile_path = spec.get(
        "profile_path", base.get("repeater_service_profile_path")
    )
    if not profile_path:
        raise ValueError("repeater_service_profile calibration requires profile_path")

    from sequence_game.physical.repeater_reservation import (
        rolling_config_for_service_profile,
    )
    from sequence_game.physical.repeater_service import (
        load_repeater_service_profile,
    )

    profile = load_repeater_service_profile(profile_path)
    accepted_alive_time_s = float(
        spec.get(
            "accepted_alive_time_s",
            profile.controller_model.get("accepted_output_alive_time_s", 0.18),
        )
    )
    payload = rolling_config_for_service_profile(
        profile,
        [value / 1000.0 for value in lengths_m],
        accepted_alive_time_s=accepted_alive_time_s,
        selection_policy=str(spec.get("selection_policy", "youngest_live")),
        num_age_points=int(spec.get("age_points_per_edge", 8)),
    )
    resolved = {
        **base,
        "repeater_service_profile_path": str(profile_path),
        "waiting_protocol": "rolling_alive_pair",
        "rolling_edge_availability": _tuple_tree(
            payload["rolling_edge_availability"]
        ),
        "rolling_age_distribution_kind": payload[
            "rolling_age_distribution_kind"
        ],
        "rolling_age_distribution_points_s": _tuple_tree(
            payload["rolling_age_distribution_points_s"]
        ),
        "rolling_age_distribution_weights": _tuple_tree(
            payload["rolling_age_distribution_weights"]
        ),
        "rolling_edge_age_distribution_points_s": _tuple_tree(
            payload["rolling_edge_age_distribution_points_s"]
        ),
        "rolling_edge_age_distribution_weights": _tuple_tree(
            payload["rolling_edge_age_distribution_weights"]
        ),
        "rolling_age_evaluation": "moment_reduced",
        "rolling_accepted_alive_time_s": payload[
            "rolling_accepted_alive_time_s"
        ],
        "rolling_memory_slots_per_edge_side": payload[
            "rolling_memory_slots_per_edge_side"
        ],
        "rolling_selection_policy": payload["rolling_selection_policy"],
        "rolling_age_distribution_source": payload[
            "rolling_age_distribution_source"
        ],
    }
    metadata = {
        "config_mode": "repeater_service_profile",
        "hop_calibrated": True,
        "calibration_kind": "route_length_specific_repeater_service",
        "calibration_id": f"h{hop}:{_stable_hash(lengths_m)[:12]}",
        "hop_count": hop,
        "fiber_lengths_m": list(lengths_m),
        "profile_path": str(profile_path),
        "profile_id": profile.profile_id,
        "scope": profile.scope,
        "claim_tier": profile.claim_tier,
        "accepted_alive_time_s": accepted_alive_time_s,
        "age_points_per_edge": int(spec.get("age_points_per_edge", 8)),
        "selection_policy": str(spec.get("selection_policy", "youngest_live")),
        "resource_policy": payload["resource_policy"],
        "slot_vector": [
            int(payload["rolling_memory_slots_per_edge_side"])
        ] * hop,
        "total_slots": int(payload["rolling_memory_slots_per_edge_side"]) * hop,
        "edge_availabilities": list(payload["rolling_edge_availability"]),
        "route_availability": payload["reservation_summary"][
            "route_availability"
        ],
        "route_service": payload["route_service"],
        "bell_quality_without_storage": payload[
            "bell_quality_without_storage"
        ],
        "rolling_age_evaluation": "moment_reduced",
    }
    _COMPILED_HISTOGRAM_RESOLVED_CONFIG_CACHE[cache_key] = (resolved, metadata)
    return resolved, metadata


def _compiled_histogram_backend() -> Any:
    """Per-process compiled-histogram backend singleton (kernel + LRU caches)."""
    global _COMPILED_HISTOGRAM_BACKEND_INSTANCE
    if _COMPILED_HISTOGRAM_BACKEND_INSTANCE is None:
        from sequence_game.corpus.compiled_histogram_backend import (
            CompiledHistogramBackend,
            CompiledHistogramBackendConfig,
        )

        config = CompiledHistogramBackendConfig(
            **_compiled_histogram_backend_config_payload()
        )
        _COMPILED_HISTOGRAM_BACKEND_INSTANCE = CompiledHistogramBackend(config)
    return _COMPILED_HISTOGRAM_BACKEND_INSTANCE


def _compiled_histogram_backend_for_hop(hop_count: int) -> Any:
    spec = _compiled_histogram_hop_calibration_spec()
    if spec is None:
        return _compiled_histogram_backend()

    from sequence_game.corpus.compiled_histogram_backend import (
        CompiledHistogramBackend,
        CompiledHistogramBackendConfig,
    )

    resolved_config, metadata = _compiled_histogram_resolved_config_for_hop(hop_count)
    key = _stable_hash({
        "hop_count": int(hop_count),
        "resolved_backend_config": resolved_config,
    })
    backend = _COMPILED_HISTOGRAM_BACKEND_POOL.get(key)
    if backend is None:
        backend = CompiledHistogramBackend(
            CompiledHistogramBackendConfig(**resolved_config)
        )
        _COMPILED_HISTOGRAM_BACKEND_POOL[key] = backend
        _COMPILED_HISTOGRAM_BACKEND_POOL_METADATA[key] = metadata
    return backend


def _compiled_histogram_backend_for_route(
    route: dict[str, Any],
    ir_dict: dict[str, Any],
) -> Any:
    """Length-specific service backend, or the historical hop backend."""

    spec = _compiled_histogram_hop_calibration_spec()
    if spec is None or str(spec.get("kind")) != "repeater_service_profile":
        return _compiled_histogram_backend_for_hop(int(route["hop_count"]))
    from sequence_game.corpus.compiled_histogram_backend import (
        CompiledHistogramBackend,
        CompiledHistogramBackendConfig,
        fiber_vector_from_route,
    )

    lengths_m = fiber_vector_from_route(ir_dict, route)
    resolved_config, metadata = _compiled_histogram_resolved_config_for_service_route(
        int(route["hop_count"]), lengths_m
    )
    key = _stable_hash({
        "hop_count": int(route["hop_count"]),
        "fiber_lengths_m": lengths_m,
        "resolved_backend_config": resolved_config,
    })
    backend = _COMPILED_HISTOGRAM_BACKEND_POOL.get(key)
    if backend is None:
        backend = CompiledHistogramBackend(
            CompiledHistogramBackendConfig(**resolved_config)
        )
        _COMPILED_HISTOGRAM_BACKEND_POOL[key] = backend
        _COMPILED_HISTOGRAM_BACKEND_POOL_METADATA[key] = metadata
    return backend


def _compiled_histogram_zero_stats() -> dict[str, float]:
    return {
        "kernel_compiles": 0,
        "kernel_cache_hits": 0,
        "kernel_compile_seconds": 0.0,
        "evaluated_cache_hits": 0,
        "evaluated_cache_misses": 0,
        "evaluated_cache_evictions": 0,
        "evaluate_seconds": 0.0,
        "steps_sampled": 0,
        "step_sample_seconds": 0.0,
    }


def _compiled_histogram_aggregate_diagnostics() -> dict[str, Any]:
    backends: list[tuple[str, Any, dict[str, Any]]] = []
    if _COMPILED_HISTOGRAM_BACKEND_INSTANCE is not None:
        backends.append((
            "static",
            _COMPILED_HISTOGRAM_BACKEND_INSTANCE,
            {
                "config_mode": "static_backend_config",
                "hop_calibrated": False,
            },
        ))
    for key, backend in sorted(_COMPILED_HISTOGRAM_BACKEND_POOL.items()):
        backends.append((key, backend, _COMPILED_HISTOGRAM_BACKEND_POOL_METADATA.get(key, {})))

    if not backends:
        return {
            "backend": COMPILED_HISTOGRAM_RUNTIME_BACKEND,
            "backend_pool_enabled": _compiled_histogram_hop_calibration_spec() is not None,
            "backend_count": 0,
            "backend_config": dict(_COMPILED_HISTOGRAM_BACKEND_CONFIG),
            "stats": _compiled_histogram_zero_stats(),
            "hop_calibrations": {},
            "backends": {},
        }

    stats = _compiled_histogram_zero_stats()
    backend_rows: dict[str, Any] = {}
    hop_calibrations: dict[str, Any] = {}
    unsupported_warnings: list[str] = []
    for key, backend, metadata in backends:
        diag = backend.diagnostics()
        for stat_key, value in diag.get("stats", {}).items():
            if isinstance(value, (int, float)):
                stats[stat_key] = stats.get(stat_key, 0.0) + value
        unsupported_warnings.extend(diag.get("unsupported_warnings", []))
        backend_rows[key] = {
            "schema_hash": diag.get("schema_hash"),
            "kernel_cache_size": diag.get("kernel_cache_size"),
            "evaluated_cache_size": diag.get("evaluated_cache_size"),
            "stats": diag.get("stats", {}),
            "calibration": metadata,
            "backend_config": diag.get("backend_config"),
        }
        if metadata.get("hop_calibrated"):
            calibration_key = str(
                metadata.get("calibration_id", metadata.get("hop_count"))
            )
            hop_calibrations[calibration_key] = metadata

    first_diag = backends[0][1].diagnostics()
    return {
        "backend": COMPILED_HISTOGRAM_RUNTIME_BACKEND,
        "backend_scope": first_diag.get("backend_scope"),
        "backend_pool_enabled": _compiled_histogram_hop_calibration_spec() is not None,
        "backend_count": len(backends),
        "generator_version": first_diag.get("generator_version"),
        "compiled_kernel_version": first_diag.get("compiled_kernel_version"),
        "front_matter_hash": first_diag.get("front_matter_hash"),
        "schema_hash": first_diag.get("schema_hash"),
        "kernel_cache_size": sum(
            int(row.get("kernel_cache_size") or 0) for row in backend_rows.values()
        ),
        "evaluated_cache_size": sum(
            int(row.get("evaluated_cache_size") or 0) for row in backend_rows.values()
        ),
        "backend_config": dict(_COMPILED_HISTOGRAM_BACKEND_CONFIG),
        "stats": stats,
        "hop_calibrations": hop_calibrations,
        "backends": backend_rows,
        "unsupported_warnings": sorted(set(unsupported_warnings)),
    }


def _compiled_histogram_stats_delta(
    before: dict[str, Any],
    after: dict[str, Any],
) -> dict[str, float]:
    before_stats = before.get("stats", {})
    after_stats = after.get("stats", {})
    delta: dict[str, float] = {}
    for key, value in after_stats.items():
        if isinstance(value, (int, float)):
            delta[key] = value - float(before_stats.get(key, 0.0) or 0.0)
    return delta


def _reset_compiled_histogram_backend(config: dict[str, Any] | None) -> None:
    global _COMPILED_HISTOGRAM_BACKEND_INSTANCE, _COMPILED_HISTOGRAM_BACKEND_CONFIG
    global _COMPILED_HISTOGRAM_BACKEND_POOL, _COMPILED_HISTOGRAM_BACKEND_POOL_METADATA
    global _COMPILED_HISTOGRAM_RESOLVED_CONFIG_CACHE
    _COMPILED_HISTOGRAM_BACKEND_CONFIG = dict(config or {})
    _COMPILED_HISTOGRAM_BACKEND_INSTANCE = None
    _COMPILED_HISTOGRAM_BACKEND_POOL = {}
    _COMPILED_HISTOGRAM_BACKEND_POOL_METADATA = {}
    _COMPILED_HISTOGRAM_RESOLVED_CONFIG_CACHE = {}


def _compiled_histogram_information_metric(result: Any) -> dict[str, Any]:
    return {
        "num_sifted": int(result.sifted_key_count),
        "num_eve_records": int(result.eve_observation_count),
        "mutual_information_bits": max(0.0, float(result.eve_guess_probability) - 0.5),
        "fraction_correct": float(result.eve_guess_probability),
        "metric_source": "compiled_histogram_transcript.eve_guess_probability",
    }


def _compiled_histogram_step_record(
    *,
    step_idx: int,
    game_mode: str,
    alice_algo: str,
    eve_algo: str,
    route: dict[str, Any],
    action: dict[str, Any],
    result: Any,
    alice_reward: float,
    eve_reward: float,
    fallback_used: bool,
    active_route_attacked: bool,
    reward_components: dict[str, Any],
    requested_pairs: int,
) -> dict[str, Any]:
    info_payload = _compiled_histogram_information_metric(result)
    physical_pair_arrival = bool(result.delivered_pair_count > 0)
    delivery_success = result.public_outcome == "accepted"
    quality_metrics = {
        "qber": result.qber_key,
        # The transcript histogram does not expose a single final-pair
        # fidelity scalar; CHSH/QBER carry the state quality signal.
        "fidelity": None,
        "chsh_S": result.chsh_S_observed,
        "chsh_S_expected": result.chsh_S_expected,
        "physical_pair_arrival": physical_pair_arrival,
        "delivery_success": delivery_success,
        "bell_qualified_delivery": delivery_success,
    }
    return {
        "step": step_idx,
        "mode": game_mode,
        "scope_label": REPEATER_RUNTIME,
        "backend": COMPILED_HISTOGRAM_RUNTIME_BACKEND,
        "claim_status": "implemented_and_testable",
        "alice_policy": alice_algo,
        "eve_policy": eve_algo,
        "alice_route_id": route["route_id"],
        "alice_path": list(route["path"]),
        "eve_action": {
            "id": action["id"],
            "kind": action["kind"],
            "target": action["target"],
            "attack_type": action["attack_type"],
        },
        "runtime_engine": "compiled_histogram_transcript_step",
        "runtime_attack_applied": result.move_kind,
        "configured_repeater_attack": None,
        "edge_attack_kind": result.move_kind if result.target_kind == "edge" else None,
        "node_attack_kind": result.move_kind if result.target_kind in {"node", "memory"} else None,
        "public_outcome": result.public_outcome,
        "information_exposed": bool(result.information_exposed),
        "secure_delivery": bool(result.secure_delivery),
        "eve_win": bool(result.eve_win),
        "collision": result.public_outcome in {"delivery_failure", "qber_abort", "chsh_abort"},
        "collision_or_delivery_failure": result.public_outcome in {
            "delivery_failure",
            "qber_abort",
            "chsh_abort",
        },
        "alice_reward": float(alice_reward),
        "eve_reward": float(eve_reward),
        "fallback_used": bool(fallback_used),
        "route_features": {
            "hops": route["hop_count"],
            "length": route["total_length"],
            "internal_nodes": list(route.get("internal_nodes") or []),
            "edges": list(route.get("edge_ids") or []),
            "scope_label": REPEATER_RUNTIME,
        },
        "quality_metrics": quality_metrics,
        "delivery_success": quality_metrics["delivery_success"],
        "physical_pair_arrival": physical_pair_arrival,
        "bell_qualified_delivery": delivery_success,
        "delivered_count": int(result.delivered_pair_count),
        "sifted_count": int(result.sifted_key_count),
        "qber": result.qber_key,
        "chsh_S": result.chsh_S_observed,
        "chsh_S_expected": result.chsh_S_expected,
        "fidelity": None,
        "active_route_attacked": bool(active_route_attacked),
        "memory_intervention_public": None,
        "router_net_summary": {
            "route_state_model": "compiled_length_parametric_transcript_histogram",
            "elementary_link_count": int(result.hop_count),
            "bsm_swap_count": max(0, int(result.hop_count) - 1),
            "backend": COMPILED_HISTOGRAM_RUNTIME_BACKEND,
            "theta_hash": result.theta_hash,
        },
        "information_gain_reward_enabled": False,
        "information_gain_metric": info_payload,
        "eve_information_metric": info_payload,
        "e91_periods_per_step": int(requested_pairs),
        "repeater_memory_pairs_per_step": int(requested_pairs),
        "metrics": {
            "scope_label": REPEATER_RUNTIME,
            "backend": COMPILED_HISTOGRAM_RUNTIME_BACKEND,
            "runtime_engine": "compiled_histogram_transcript_step",
            **quality_metrics,
            "runtime_attack_applied": result.move_kind,
            "configured_repeater_attack": None,
            "edge_attack_kind": result.move_kind if result.target_kind == "edge" else None,
            "node_attack_kind": result.move_kind if result.target_kind in {"node", "memory"} else None,
            "memory_intervention_public": None,
            "reward_components": reward_components,
            "active_route_attacked": bool(active_route_attacked),
            "information_gain_reward_enabled": False,
            "information_gain_metric": info_payload,
            "e91_periods_per_step": int(requested_pairs),
            "repeater_memory_pairs_per_step": int(requested_pairs),
            "theta_hash": result.theta_hash,
            "schema_hash": result.provenance.get("schema_hash"),
        },
    }


def run_compiled_histogram_runtime_game(
    ir_dict: dict[str, Any],
    alice: str,
    bob: str,
    *,
    routes: list[dict[str, Any]] | None,
    game_mode: str,
    eve_algo: str,
    alice_algo: str,
    num_steps: int,
    seed: int,
    final_window_size: int = 20,
    e91_periods_per_step: int = 1200,
    repeater_memory_pairs_per_step: int = 8,
    attack_cost: float = 0.05,
    dt_refit_observer: Any = None,
    dt_refit_callback: Any = None,
    progress_every: int = 0,
    progress_label: str | None = None,
    **_ignored_backend_kwargs: Any,
) -> dict[str, Any]:
    from sequence_game.corpus import e91_runtime_game as sequence_runtime

    _ = repeater_memory_pairs_per_step
    started = time.perf_counter()
    ir = sequence_runtime.TopologyIR.from_dict(ir_dict)
    runtime_routes = sequence_runtime._normalize_routes(ir, alice, bob, routes)
    if not runtime_routes:
        raise ValueError(f"no Alice-Bob routes available for {alice!r}->{bob!r}")
    actions = sequence_runtime._runtime_actions(ir, runtime_routes, game_mode)

    rng = np.random.default_rng(seed)
    eve_q = sequence_runtime._make_agent(eve_algo, len(actions))
    alice_q = sequence_runtime._make_agent(alice_algo, len(runtime_routes))
    q_cfg = sequence_runtime.QLearningConfig(alpha=0.2, gamma=0.9)

    avoided_until: dict[str, int] = {}
    previous_eve_action = "start"
    previous_public_outcome = "start"
    previous_route_id = "start"
    steps: list[dict[str, Any]] = []
    step_seconds: list[float] = []
    diagnostics_before = _compiled_histogram_aggregate_diagnostics()
    progress_interval = max(0, int(progress_every or 0))
    progress_name = progress_label or f"{game_mode}__{eve_algo}__{alice_algo}"
    eve_win_count = 0

    for step_idx in range(num_steps):
        step_started = time.perf_counter()
        route, route_idx, alice_state, fallback_used = sequence_runtime._select_route(
            runtime_routes,
            alice_algo,
            alice_q,
            previous_route_id,
            previous_public_outcome,
            avoided_until,
            step_idx,
            num_steps,
            rng,
        )
        action, eve_action_idx, eve_state = sequence_runtime._select_action(
            actions,
            route,
            eve_algo,
            eve_q,
            previous_eve_action,
            previous_public_outcome,
            step_idx,
            num_steps,
            rng,
        )
        backend = _compiled_histogram_backend_for_route(route, ir_dict)
        compiled_result = backend.step(
            ir_dict=ir_dict,
            route=route,
            action=action,
            requested_pairs=e91_periods_per_step,
            seed=seed + 100_000 + step_idx,
            step_index=step_idx,
        )
        public_outcome = str(compiled_result.public_outcome)
        active_route_attacked = bool(compiled_result.target_active)
        # Route avoidance is based on Alice's public transcript, not on Eve's
        # private information record or the simulator's target-active flag.
        failure_or_detection = public_outcome != "accepted"
        if compiled_result.eve_win:
            eve_win_count += 1
        if failure_or_detection and alice_algo == "cautious_greedy":
            avoided_until[route["route_id"]] = step_idx + 1

        cost = 0.0 if action.get("attack_type") == "no_attack" else float(attack_cost)
        eve_feedback = eve_feedback_from_heralded_service(
            public_outcome=public_outcome,
            requested_pair_count=e91_periods_per_step,
            delivered_pair_count=int(compiled_result.delivered_pair_count),
            attack_cost=cost,
            local_observation_kind="attack_instrument_private_record",
            local_observation_count=int(compiled_result.eve_observation_count),
        )
        eve_reward = eve_feedback.reward
        alice_reward = 1.0 if public_outcome == "accepted" else -1.0
        if fallback_used:
            alice_reward -= 0.2
        reward_components = {
            "scope_label": REPEATER_RUNTIME,
            "backend": COMPILED_HISTOGRAM_RUNTIME_BACKEND,
            "public_outcome_reward": 1.0 if public_outcome != "accepted" else 0.0,
            "information_exposure": bool(compiled_result.information_exposed),
            "secure_delivery": bool(compiled_result.secure_delivery),
            "attack_cost": cost,
            "eve_feedback": eve_feedback.to_dict(),
            "target_active_used_for_eve_feedback": False,
            "information_exposure_rewarded": False,
            "metric_source": "compiled_histogram_transcript",
            "compiled_reward_components": dict(compiled_result.reward_components),
        }

        next_eve_state = (action["id"], public_outcome)
        if eve_algo in {"q_learning", "deep_q"}:
            eve_q.update(eve_state, eve_action_idx, eve_reward, next_eve_state, q_cfg)
        if alice_algo in {"q_learning", "deep_q"}:
            next_avoid = sequence_runtime._avoidance_signature(
                runtime_routes,
                avoided_until,
                step_idx,
            )
            next_alice_state = (route["route_id"], public_outcome, next_avoid)
            alice_q.update(alice_state, route_idx, alice_reward, next_alice_state, q_cfg)

        record = _compiled_histogram_step_record(
            step_idx=step_idx,
            game_mode=game_mode,
            alice_algo=alice_algo,
            eve_algo=eve_algo,
            route=route,
            action=action,
            result=compiled_result,
            alice_reward=alice_reward,
            eve_reward=eve_reward,
            fallback_used=fallback_used,
            active_route_attacked=active_route_attacked,
            reward_components=reward_components,
            requested_pairs=e91_periods_per_step,
        )
        steps.append(record)
        step_seconds.append(time.perf_counter() - step_started)
        step_num = step_idx + 1
        if progress_interval and (
            step_num % progress_interval == 0 or step_num == num_steps
        ):
            elapsed = time.perf_counter() - started
            stats = _compiled_histogram_aggregate_diagnostics()["stats"]
            print(
                "  [case-progress] "
                f"{progress_name} step={step_num}/{num_steps} "
                f"routes={len(runtime_routes)} actions={len(actions)} "
                f"eve_win={eve_win_count / step_num:.2%} "
                f"mean_step={elapsed / step_num:.3f}s "
                f"elapsed={elapsed:.1f}s "
                f"hist_cache={int(stats['evaluated_cache_hits'])}/"
                f"{int(stats['evaluated_cache_misses'])} "
                f"last={public_outcome}",
                flush=True,
            )

        if dt_refit_observer is not None:
            dt_refit_observer.record_transition(
                alice_action=route_idx,
                eve_action=eve_action_idx,
                alice_route=route.get("route_id"),
                eve_target=action.get("target"),
                reward=eve_reward,
            )
            tick = dt_refit_observer.observe(
                step=step_idx + 1,
                alice_q=alice_q if alice_algo in {"q_learning", "deep_q"} else None,
                eve_q=eve_q if eve_algo in {"q_learning", "deep_q"} else None,
            )
            if tick is not None and tick.refit_recommended:
                event = dt_refit_observer.refit_event_from_tick(tick)
                if dt_refit_callback is not None:
                    dt_refit_callback(event, {
                        "step": step_idx + 1,
                        "routes": runtime_routes,
                        "actions": actions,
                        "steps": steps,
                        "alice_q": alice_q,
                        "eve_q": eve_q,
                        "alice_algo": alice_algo,
                        "eve_algo": eve_algo,
                        "control_model": game_mode,
                    })
                dt_refit_observer.mark_refit(
                    step=step_idx + 1,
                    alice_q=alice_q if alice_algo in {"q_learning", "deep_q"} else None,
                    eve_q=eve_q if eve_algo in {"q_learning", "deep_q"} else None,
                    reason=tick.refit_reason,
                )

        previous_eve_action = action["id"]
        previous_public_outcome = public_outcome
        previous_route_id = route["route_id"]

    final_window = steps[-min(final_window_size, len(steps)):]
    diagnostics_after = _compiled_histogram_aggregate_diagnostics()
    backend_diagnostics = {
        **diagnostics_after,
        "stats_delta": _compiled_histogram_stats_delta(
            diagnostics_before,
            diagnostics_after,
        ),
    }
    summary = _direct_e91_summary(
        game_mode=game_mode,
        seed=seed,
        alice=alice,
        bob=bob,
        alice_algo=alice_algo,
        eve_algo=eve_algo,
        routes=runtime_routes,
        actions=actions,
        steps=steps,
        final_window=final_window,
        final_window_size=final_window_size,
        eve_q=eve_q,
        alice_q=alice_q,
        include_alice_q=alice_algo in {"q_learning", "deep_q"},
        started=started,
        step_seconds=step_seconds,
        requested_pairs=e91_periods_per_step,
        cache_diagnostics={},
    )
    summary["backend"] = COMPILED_HISTOGRAM_RUNTIME_BACKEND
    metrics = summary["metrics"]
    metrics.update({
        "backend": COMPILED_HISTOGRAM_RUNTIME_BACKEND,
        "direct_e91_runtime_executed": False,
        "compiled_histogram_runtime_executed": True,
        "runtime_engines": ["compiled_histogram_transcript_step"],
        "training_execution": (
            "online route/action updates; each step is one multinomial draw from the "
            "cached compiled transcript histogram H(L)"
        ),
        "reward_model": "public_heralded_service_complement_minus_attack_cost",
        "attack_surface_scope": "compiled_histogram_transcript_route_corridor",
        "information_gain_metric_source": (
            "compiled_histogram_transcript.eve_guess_probability"
        ),
        "compiled_histogram_backend": backend_diagnostics,
    })
    metrics.pop("direct_e91_prepared_cache", None)
    return {
        "summary": summary,
        "steps": steps,
        "final_window": final_window,
        "scope_label": REPEATER_RUNTIME,
    }


def _corpus_worker(args: dict[str, Any] | tuple) -> tuple[dict | None, dict | None]:
    """Top-level worker for ProcessPoolExecutor — must be importable at module level."""
    task = args if isinstance(args, dict) else _legacy_corpus_worker_args(args)
    case_plan = task["case_plan"]
    global _DIRECT_E91_RUNTIME_CACHE_ENABLED, _DIRECT_E91_RUNTIME_TRAINING_LIGHT
    previous_runtime_backend = corpus_runner_module.run_e91_graph_runtime_game
    previous_direct_cache_enabled = _DIRECT_E91_RUNTIME_CACHE_ENABLED
    previous_direct_training_light = _DIRECT_E91_RUNTIME_TRAINING_LIGHT
    try:
        _write_corpus_worker_status(task, "started")
        combo = task["combo"]
        runtime_backend = str(task.get("runtime_backend") or RUNTIME_BACKEND_SEQUENCE)
        direct_cache_enabled = bool(
            task.get("direct_e91_prepared_cache_enabled", True)
        )
        direct_training_light = bool(task.get("direct_e91_training_light", False))
        _DIRECT_E91_RUNTIME_CACHE_ENABLED = direct_cache_enabled
        _DIRECT_E91_RUNTIME_TRAINING_LIGHT = direct_training_light
        if runtime_backend == RUNTIME_BACKEND_COMPILED_HISTOGRAM:
            configured = task.get("compiled_histogram_backend_config") or {}
            if configured != _COMPILED_HISTOGRAM_BACKEND_CONFIG or (
                _COMPILED_HISTOGRAM_BACKEND_INSTANCE is None
                and not _COMPILED_HISTOGRAM_BACKEND_POOL
            ):
                _reset_compiled_histogram_backend(configured)
        _install_corpus_runtime_backend(runtime_backend)
        cache_before = _direct_e91_cache_stats()
        result = run_one_corpus_graph(
            task["record"],
            _corpus_source_from_task(task),
            num_steps=int(task["num_steps"]),
            seed=int(task["seed"]),
            combos=[combo],
            dt_refit_config=task.get("dt_refit_config"),
            dt_refit_output_dir=(
                Path(task["dt_refit_output_dir"])
                if task.get("dt_refit_output_dir") is not None
                else None
            ),
            dt_refit_output_per_case=bool(task.get("dt_refit_output_per_case", False)),
            dt_depth=int(task.get("dt_depth", 4)),
            progress_every=int(task.get("progress_every", 0) or 0),
            progress_label=_corpus_task_progress_label(task),
        )
        if runtime_backend == RUNTIME_BACKEND_DIRECT_E91:
            cache_after = _direct_e91_cache_stats()
            _attach_direct_cache_diagnostics(
                result,
                combo,
                _direct_e91_cache_delta(
                    cache_before,
                    cache_after,
                    cache_enabled=direct_cache_enabled,
                    training_light=direct_training_light,
                ),
            )
        if runtime_backend == RUNTIME_BACKEND_COMPILED_HISTOGRAM:
            key = _combo_key_from_combo(combo)
            combo_result = (result.get("combo_results") or {}).get(key)
            diagnostics = _compiled_histogram_aggregate_diagnostics()
            if isinstance(combo_result, dict):
                runtime_metrics = combo_result.setdefault("runtime_metrics", {})
                runtime_metrics.setdefault("compiled_histogram_backend", diagnostics)
                result["compiled_histogram_backend"] = runtime_metrics[
                    "compiled_histogram_backend"
                ]
            else:
                result["compiled_histogram_backend"] = diagnostics
        finalized = _finalize_corpus_case_result(
            result,
            combo=combo,
            case_plan=case_plan,
        )
        _write_corpus_worker_status(task, "completed")
        return finalized, None
    except Exception as exc:
        _write_corpus_worker_status(
            task,
            "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        return None, _failure_record(
            case_plan,
            error=f"{type(exc).__name__}: {exc}",
            traceback_text=traceback.format_exc(),
        )
    finally:
        corpus_runner_module.run_e91_graph_runtime_game = previous_runtime_backend
        _DIRECT_E91_RUNTIME_CACHE_ENABLED = previous_direct_cache_enabled
        _DIRECT_E91_RUNTIME_TRAINING_LIGHT = previous_direct_training_light


def _corpus_record_key(record: dict) -> str:
    return f"{record['graph_id']}/{record.get('assignment_id', '')}"


def _has_corpus_row_payloads(record: dict) -> bool:
    return all(isinstance(record.get(key), list) for key in _CORPUS_ROW_PAYLOAD_KEYS)


def _strip_corpus_row_payloads(record: dict) -> dict:
    return {k: v for k, v in record.items() if k not in _CORPUS_ROW_PAYLOAD_KEYS}


def _load_corpus_checkpoint(checkpoint_file: Path) -> tuple[list[dict], set[str]]:
    """Load only checkpoint records that can support resumed DT fitting."""
    raw_records = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        print("  WARNING: corpus checkpoint is not a list; rerunning corpus simulation")
        return [], set()

    restorable: list[dict] = []
    rejected_count = 0
    for record in raw_records:
        if not isinstance(record, dict) or not _has_corpus_row_payloads(record):
            rejected_count += 1
            continue
        try:
            _validate_case_record(record)
        except CaseValidationError:
            rejected_count += 1
            continue
        restorable.append(record)
    if rejected_count:
        print(
            "  WARNING: "
            f"{rejected_count} checkpoint record(s) are missing DT row payloads "
            "or valid case provenance; "
            "rerunning those records",
            flush=True,
        )
    return restorable, {str(record["case_id"]) for record in restorable}


def _load_corpus_sqlite_checkpoint(db_path: str | Path) -> tuple[list[dict], set[str]]:
    raw_records = load_completed_corpus_case_records(db_path)
    restorable: list[dict] = []
    rejected_count = 0
    for record in raw_records:
        if not isinstance(record, dict) or not _has_corpus_row_payloads(record):
            rejected_count += 1
            continue
        try:
            _validate_case_record(record)
        except CaseValidationError:
            rejected_count += 1
            continue
        restorable.append(record)
    if rejected_count:
        print(
            "  WARNING: "
            f"{rejected_count} SQLite record(s) are missing DT row payloads "
            "or valid case provenance; rerunning those records",
            flush=True,
        )
    return restorable, {str(record["case_id"]) for record in restorable}


def _write_corpus_checkpoint(checkpoint_file: Path, per_graph: list[dict]) -> None:
    # Checkpoints keep structural row payloads because resumed DT fitting needs them.
    _atomic_write_json(checkpoint_file, per_graph)


def _corpus_task_label(task: dict[str, Any]) -> str:
    record = task["record"]
    plan = task["case_plan"]
    return (
        f"{record.get('graph_id')}/{record.get('assignment_id', '')}"
        f" {plan.get('combo_key')} case={str(plan.get('case_id'))[:12]}"
    )


def _corpus_task_progress_label(task: dict[str, Any]) -> str:
    record = task["record"]
    plan = task["case_plan"]
    return (
        f"{record.get('graph_id')}/{record.get('assignment_id', '')}"
        f" case={str(plan.get('case_id'))[:12]}"
    )


def _bounded_in_flight_limit(workers: int) -> int:
    return max(1, int(workers) * 2)


def _run_corpus_tasks_parallel(
    tasks: list[dict[str, Any]],
    *,
    workers: int,
) -> Iterator[tuple[dict[str, Any], dict | None, dict | None]]:
    from concurrent.futures import (
        FIRST_COMPLETED,
        BrokenExecutor,
        ProcessPoolExecutor,
        wait,
    )

    max_in_flight = min(len(tasks), _bounded_in_flight_limit(workers))
    task_iter = iter(tasks)
    heartbeat_interval_s = 60.0
    pool = ProcessPoolExecutor(max_workers=workers)
    in_flight: dict[Any, dict[str, Any]] = {}
    try:
        def _submit_next() -> bool:
            # An abruptly killed worker (e.g. OOM) breaks the whole executor;
            # restart it so one bad case cannot fail the rest of the queue.
            nonlocal pool
            try:
                task = next(task_iter)
            except StopIteration:
                return False
            task["_submitted_at"] = time.perf_counter()
            try:
                future = pool.submit(_corpus_worker, task)
            except BrokenExecutor:
                print("  [pool] process pool broken; restarting workers", flush=True)
                pool.shutdown(wait=False, cancel_futures=True)
                pool = ProcessPoolExecutor(max_workers=workers)
                future = pool.submit(_corpus_worker, task)
            in_flight[future] = task
            print(f"  [submit] {_corpus_task_label(task)}", flush=True)
            return True

        for _ in range(max_in_flight):
            _submit_next()

        while in_flight:
            done, _pending = wait(
                in_flight,
                timeout=heartbeat_interval_s,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                labels = [
                    _corpus_task_label(task)
                    for task in list(in_flight.values())[:max(1, workers)]
                ]
                print(
                    "  [heartbeat] waiting on "
                    f"{len(in_flight)} in-flight case(s): "
                    + "; ".join(labels),
                    flush=True,
                )
                continue
            for future in done:
                task = in_flight.pop(future)
                try:
                    result, failure = future.result()
                except Exception as exc:
                    result = None
                    failure = _failure_record(
                        task["case_plan"],
                        error=f"{type(exc).__name__}: {exc}",
                        traceback_text=traceback.format_exc(),
                    )
                yield task, result, failure
                _submit_next()
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _completed_cases_manifest(per_graph: list[dict[str, Any]]) -> dict[str, Any]:
    entries = [
        {
            field: record.get(field)
            for field in CASE_PROVENANCE_FIELDS
            if field in record
        }
        for record in per_graph
        if record.get("status") == "completed" and record.get("case_id")
    ]
    entries.sort(key=lambda entry: str(entry.get("case_id", "")))
    completed_cases_hash = _stable_hash(entries)
    return {
        "manifest_version": 1,
        "completed_case_count": len(entries),
        "completed_cases_hash": completed_cases_hash,
        "case_ids": [entry["case_id"] for entry in entries],
        "entries": entries,
    }


def _dt_artifacts_match_completed_manifest(
    dat_dir: Path,
    completed_manifest: dict[str, Any],
) -> bool:
    sidecar = dat_dir / "corpus_dt_artifact_manifest.json"
    if not sidecar.exists():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("completed_cases_hash")
        == completed_manifest.get("completed_cases_hash")
    )


def _write_dt_artifact_manifest(dat_dir: Path, completed_manifest: dict[str, Any]) -> None:
    _atomic_write_json(
        dat_dir / "corpus_dt_artifact_manifest.json",
        {
            "manifest_version": 1,
            "completed_cases_hash": completed_manifest.get("completed_cases_hash"),
            "completed_case_count": completed_manifest.get("completed_case_count"),
            "written_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def run_corpus_pipeline(
    manifest_path: str | None,
    models: list[str],
    out_dir: Path,
    ts: str,
    *,
    max_episodes: int = 300,
    seed: int = 42,
    no_llm: bool = False,
    workers: int = 1,
    checkpoint_every: int = 50,
    resume_sim: bool = False,
    dt_refit_config: DynamicDTRefitConfig | None = None,
    dt_depth: int = 4,
    combos: list[dict] | None = None,
    max_graphs: int | None = None,
    reference_max_graphs: int | None = None,
    allow_failed_cases: bool = False,
    runtime_backend: str = RUNTIME_BACKEND_SEQUENCE,
    direct_e91_prepared_cache_enabled: bool = True,
    direct_e91_training_light: bool = False,
    compiled_histogram_backend_config: dict[str, Any] | None = None,
    assignment_selection: str = ASSIGNMENT_SELECTION_ALL,
    experiment_suite_id: str = EXPERIMENT_SUITE_WORKSHOP_PRIMARY_SIX,
    corpus_db_path: str | Path | None = None,
    corpus_id: str | None = None,
    results_db_path: str | Path | None = None,
    write_json_checkpoint: bool | None = None,
) -> dict:
    """Run configured corpus combos, fit pooled DT, call LLM.

    Writes a checkpoint every `checkpoint_every` graphs so progress survives crashes.
    If `resume_sim` is True, loads any existing checkpoint and skips already-completed graphs.
    LLM output files are skipped if they already exist on disk.

    Returns aggregate stats dict; saves per-graph JSON and summaries.
    """
    try:
        manifest, corpus_root, manifest_label = _load_corpus_manifest_and_source(
            manifest_path,
            corpus_db_path=corpus_db_path,
            corpus_id=corpus_id,
        )
    except FileNotFoundError:
        print(f"  WARNING: corpus manifest not found at {manifest_path} - skipping corpus pipeline")
        return {}
    resolved_results_db_path = (
        Path(results_db_path).resolve()
        if results_db_path is not None
        else (Path(corpus_db_path).resolve() if corpus_db_path is not None else None)
    )
    use_json_checkpoint = (
        bool(write_json_checkpoint)
        if write_json_checkpoint is not None
        else resolved_results_db_path is None
    )
    manifest_p = Path(manifest_path) if manifest_path is not None else None
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be >= 1")
    if max_graphs is not None and max_graphs < 1:
        raise ValueError("max_graphs must be >= 1")
    if reference_max_graphs is not None and reference_max_graphs < 0:
        raise ValueError("reference_max_graphs must be >= 0")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if runtime_backend not in {
        RUNTIME_BACKEND_SEQUENCE,
        RUNTIME_BACKEND_DIRECT_E91,
        RUNTIME_BACKEND_COMPILED_HISTOGRAM,
    }:
        raise ValueError(f"unsupported runtime_backend {runtime_backend!r}")
    if assignment_selection not in {
        ASSIGNMENT_SELECTION_ALL,
        ASSIGNMENT_SELECTION_TOP2_PER_GRAPH,
    }:
        raise ValueError(f"unsupported assignment_selection {assignment_selection!r}")
    if (
        assignment_selection == ASSIGNMENT_SELECTION_TOP2_PER_GRAPH
        and reference_max_graphs is not None
    ):
        raise ValueError(
            "top2_per_graph assignment selection runs every selected assignment "
            "through the same experiment hooks; do not combine it with "
            "--corpus-reference-max-graphs"
        )
    run_combos = EXPERIMENT_SUITE if combos is None else combos
    if not run_combos:
        raise ValueError("combos must contain at least one condition")

    if corpus_db_path is None:
        assert manifest_p is not None
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        corpus_root = manifest_p.parent
        manifest_label = manifest_p.name
    accepted = _corpus_records_from_manifest(manifest)
    manifest_graph_ids = {
        str(graph.get("graph_id"))
        for graph in manifest.get("graphs", [])
        if graph.get("graph_id") not in (None, "")
    }
    if max_graphs is not None:
        accepted = accepted[:max_graphs]
    accepted, selection_metadata = _select_corpus_assignments(
        accepted,
        assignment_selection=assignment_selection,
    )
    record_graph_ids_before_selection = set(
        selection_metadata.get("selected_assignment_ids_by_graph", {})
    )
    if max_graphs is None and manifest_graph_ids:
        all_graph_ids_before_selection = manifest_graph_ids | record_graph_ids_before_selection
        zero_assignment_graph_ids = sorted(
            manifest_graph_ids - record_graph_ids_before_selection
        )
        selection_metadata["graph_count_before_selection"] = len(
            all_graph_ids_before_selection
        )
        selection_metadata["zero_assignment_graph_count"] = len(zero_assignment_graph_ids)
        selection_metadata["zero_assignment_graph_ids"] = zero_assignment_graph_ids
        selection_metadata["skipped_graph_count"] = (
            int(selection_metadata.get("skipped_graph_count", 0))
            + len(zero_assignment_graph_ids)
        )
    else:
        selection_metadata["zero_assignment_graph_count"] = 0
        selection_metadata["zero_assignment_graph_ids"] = []
    synthetic_corpus = _is_synthetic_corpus(accepted)
    reference_graph_ids = _reference_graph_prefix(accepted, reference_max_graphs)
    n_combos = len(run_combos)
    selection_metadata.update({
        "experiment_suite_id": experiment_suite_id,
        "experiment_suite_name": experiment_suite_id,
        "experiment_hook_count": n_combos,
        "experiment_hooks": [
            {
                "key": _combo_key_from_combo(combo),
                "game_mode": combo.get("game_mode"),
                "eve": combo.get("eve"),
                "alice": combo.get("alice"),
                "group": combo.get("group"),
            }
            for combo in run_combos
        ],
        "planned_case_count_formula": "selected_assignment_count * experiment_hook_count",
        "planned_case_count_formula_terms": {
            "selected_assignment_count": len(accepted),
            "experiment_hook_count": n_combos,
        },
    })
    route_policy = "all_simple_paths"
    reference_combo_count = sum(1 for combo in run_combos if combo.get("group") == "reference")
    reference_note = (
        (
            f"; reference combos limited to first {reference_max_graphs} "
            "unique graphs per synthetic category"
        )
        if synthetic_corpus
        else f"; reference combos limited to first {reference_max_graphs} unique graphs"
    ) if reference_max_graphs is not None and reference_combo_count else ""
    print(
        f"  Corpus: {len(accepted)} graph/assignment instances × {n_combos} combos "
        f"from {manifest_label} (route_policy={route_policy}{reference_note})"
    )
    if assignment_selection != ASSIGNMENT_SELECTION_ALL:
        print(
            "  Assignment selection: "
            f"{selection_metadata['assignment_selection_effective_rule']} "
            f"({selection_metadata['assignment_count_before_selection']} → "
            f"{selection_metadata['assignment_count_after_selection']} assignments, "
            f"{selection_metadata['graph_count_after_selection']} graphs)"
        )
    if runtime_backend == RUNTIME_BACKEND_DIRECT_E91:
        print(
            "  Direct E91 runtime: "
            f"prepared_cache={'enabled' if direct_e91_prepared_cache_enabled else 'disabled'} "
            f"training_light={bool(direct_e91_training_light)} "
            f"key_version={DIRECT_E91_PREPARED_CACHE_KEY_VERSION}"
        )
    if runtime_backend == RUNTIME_BACKEND_COMPILED_HISTOGRAM:
        _reset_compiled_histogram_backend(compiled_histogram_backend_config)
        calibration_spec = _compiled_histogram_hop_calibration_spec()
        if calibration_spec is None:
            _probe = _compiled_histogram_backend()
            print(
                "  Compiled histogram runtime: "
                f"schema_hash={_probe.schema_hash[:16]} "
                f"evaluated_cache_size={_probe.config.evaluated_cache_size} "
                f"signature_cap={_probe.config.waiting_max_wait_signatures} "
                f"fock_cutoff={_probe.config.fock_cutoff}"
            )
        else:
            payload = _compiled_histogram_backend_config_payload()
            print(
                "  Compiled histogram runtime: "
                "hop_calibrated_rolling_reservation "
                f"kind={calibration_spec.get('kind', 'hop_based_10km')} "
                f"edge_km={float(calibration_spec.get('edge_length_km', 10.0)):g} "
                f"T_alive={float(calibration_spec.get('accepted_alive_time_s', 0.18)):g}s "
                f"age_points={int(calibration_spec.get('age_points_per_edge', 8))} "
                f"evaluated_cache_size={int(payload.get('evaluated_cache_size', 256))} "
                f"fock_cutoff={int(calibration_spec.get('fock_cutoff', payload.get('fock_cutoff', 2)))}"
            )

    dat_dir = out_dir / "data"
    dat_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = dat_dir / "corpus_checkpoint.json"
    failed_cases_file = dat_dir / "failed_cases.jsonl"
    dt_refit_output_dir = dat_dir / "corpus_dt_refit_observer"
    worker_status_dir = dat_dir / "corpus_worker_status"
    if resolved_results_db_path is None:
        worker_status_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"  SQLite corpus results: {resolved_results_db_path}")
    if dt_refit_config is not None:
        (dat_dir / "corpus_dt_refit_config.json").write_text(
            json.dumps(dt_refit_config.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    # ── Load existing checkpoint if resuming ──────────────────────────────────
    per_graph: list[dict] = []
    completed_ids: set[str] = set()
    if resume_sim and resolved_results_db_path is not None:
        per_graph, completed_ids = _load_corpus_sqlite_checkpoint(resolved_results_db_path)
        print(f"  Resuming: loaded {len(per_graph)} completed cases from SQLite")
    elif resume_sim and checkpoint_file.exists():
        per_graph, completed_ids = _load_corpus_checkpoint(checkpoint_file)
        print(f"  Resuming: loaded {len(per_graph)} completed cases from checkpoint")

    tasks: list[dict[str, Any]] = []
    planned_case_ids: set[str] = set()
    planned_case_count = 0
    for record_index, rec in enumerate(accepted):
        record_combos = _combos_for_corpus_record(
            run_combos,
            record_index=record_index,
            record=rec,
            reference_max_graphs=reference_max_graphs,
            reference_graph_ids=reference_graph_ids,
        )
        if not record_combos:
            continue
        for combo in record_combos:
            case_seed = _derive_corpus_run_seed(seed, rec)
            case_plan = _make_corpus_case_plan(
                record=rec,
                corpus_root=corpus_root,
                combo=combo,
                seed=case_seed,
                num_steps=max_episodes,
                route_policy=route_policy,
                dt_refit_config=dt_refit_config,
                dt_depth=dt_depth,
                runtime_backend=runtime_backend,
                direct_e91_training_light=direct_e91_training_light,
                compiled_histogram_backend_config=compiled_histogram_backend_config,
            )
            planned_case_count += 1
            planned_case_ids.add(str(case_plan["case_id"]))
            if str(case_plan["case_id"]) in completed_ids:
                continue
            tasks.append({
                "record": rec,
                "corpus_root": str(corpus_root),
                "corpus_source": _corpus_source_task_spec(corpus_root),
                "num_steps": max_episodes,
                "seed": case_seed,
                "combo": combo,
                "dt_refit_config": dt_refit_config,
                "dt_refit_output_dir": (
                    str(_dt_refit_case_output_dir(dt_refit_output_dir, case_plan["case_id"]))
                    if dt_refit_config is not None
                    else None
                ),
                "dt_refit_output_per_case": True,
                "dt_depth": dt_depth,
                "case_plan": case_plan,
                "worker_status_dir": (
                    str(worker_status_dir) if resolved_results_db_path is None else None
                ),
                "results_db_path": (
                    str(resolved_results_db_path)
                    if resolved_results_db_path is not None
                    else None
                ),
                "runtime_backend": runtime_backend,
                "direct_e91_prepared_cache_enabled": direct_e91_prepared_cache_enabled,
                "direct_e91_training_light": direct_e91_training_light,
                "compiled_histogram_backend_config": dict(
                    compiled_histogram_backend_config or {}
                ),
                "progress_every": 100,
            })
    if resume_sim and per_graph:
        filtered = [
            record for record in per_graph
            if str(record.get("case_id")) in planned_case_ids
        ]
        dropped = len(per_graph) - len(filtered)
        if dropped:
            print(
                f"  Resume checkpoint: ignored {dropped} completed case(s) "
                "not in the current planned case manifest"
            )
        per_graph = filtered
        completed_ids = {str(record["case_id"]) for record in per_graph}
    n_total = planned_case_count
    n_pending = len(tasks)
    selection_metadata.update({
        "planned_case_count": n_total,
        "pending_case_count_at_start": n_pending,
        "planned_case_count_check": _planned_corpus_case_count(
            len(accepted),
            run_combos,
        ) if reference_max_graphs is None else n_total,
    })
    _atomic_write_json(
        dat_dir / "corpus_assignment_selection.json",
        selection_metadata,
    )

    def _write_result_record(record: dict[str, Any], *, payload_source: str = "live_result") -> None:
        if resolved_results_db_path is not None:
            write_corpus_case_record(
                resolved_results_db_path,
                record,
                payload_source=payload_source,
            )

    def _write_checkpoint() -> None:
        if use_json_checkpoint:
            _write_corpus_checkpoint(checkpoint_file, per_graph)

    def _handle_failed_case(task: dict[str, Any], failure: dict[str, Any]) -> None:
        _write_failed_case(failed_cases_file, failure)
        _write_result_record(failure, payload_source="failure")
        _write_checkpoint()
        elapsed = ""
        if task.get("_submitted_at") is not None:
            elapsed = f" elapsed={time.perf_counter() - float(task['_submitted_at']):.1f}s"
        print(
            f"  [{len(per_graph)}/{n_total}] {_corpus_task_label(task)}  "
            f"ERROR: {failure.get('error')}{elapsed}",
            flush=True,
        )
        if not allow_failed_cases:
            raise RuntimeError(
                f"corpus case failed: {_corpus_task_label(task)}; "
                f"traceback written to {failed_cases_file}"
            )

    if n_pending == 0:
        print("  All cases already in checkpoint — skipping simulation")
    elif workers == 1:
        for i, task in enumerate(tasks, 1):
            task["_submitted_at"] = time.perf_counter()
            print(
                f"  [{len(per_graph)+1}/{n_total}] {_corpus_task_label(task)}  "
                f"seed={task['seed']}",
                end="",
                flush=True,
            )
            result, failure = _corpus_worker(task)
            if failure:
                print()
                _handle_failed_case(task, failure)
            elif result is not None:
                per_graph.append(result)
                _write_result_record(result)
                print(
                    f"  eve_win={result['eve_win_rate']:.2%}  "
                    f"combo={result.get('combo_key')}  "
                    f"elapsed={time.perf_counter() - float(task['_submitted_at']):.1f}s"
                )
            if i % checkpoint_every == 0:
                _write_checkpoint()
                label = "checkpoint" if use_json_checkpoint else "sqlite"
                print(f"  [{label}] {len(per_graph)}/{n_total} saved", flush=True)
    else:
        status_target = (
            resolved_results_db_path
            if resolved_results_db_path is not None
            else worker_status_dir
        )
        print(
            f"  (parallel: {workers} workers, checkpoint every {checkpoint_every}; "
            f"worker status: {status_target})"
        )
        done_this_run = 0
        for task, result, failure in _run_corpus_tasks_parallel(tasks, workers=workers):
            done_this_run += 1
            if failure:
                _handle_failed_case(task, failure)
            elif result is not None:
                per_graph.append(result)
                _write_result_record(result)
                elapsed = ""
                if task.get("_submitted_at") is not None:
                    elapsed = f"  elapsed={time.perf_counter() - float(task['_submitted_at']):.1f}s"
                print(
                    f"  [{len(per_graph)}/{n_total}] {_corpus_task_label(task)}  "
                    f"eve_win={result['eve_win_rate']:.2%}{elapsed}",
                    flush=True,
                )
            if done_this_run % checkpoint_every == 0:
                _write_checkpoint()
                label = "checkpoint" if use_json_checkpoint else "sqlite"
                print(f"  [{label}] {len(per_graph)}/{n_total} saved", flush=True)

    # Final checkpoint after all graphs complete
    _write_checkpoint()

    dat_dir = out_dir / "data"
    dat_dir.mkdir(parents=True, exist_ok=True)
    completed_manifest = _completed_cases_manifest(per_graph)
    _atomic_write_json(dat_dir / "corpus_completed_cases_manifest.json", completed_manifest)

    # Save per-graph records (strip pooled DT row payloads to keep file small)
    slim_records = [_strip_corpus_row_payloads(r) for r in per_graph]
    with open(dat_dir / "corpus_per_graph.json", "w", encoding="utf-8") as f:
        json.dump(slim_records, f, indent=2, default=str)
    print(f"  → data/corpus_per_graph.json ({len(per_graph)} cases)")
    print("  → data/corpus_assignment_selection.json")

    route_diagnostics = [
        {
            "graph_id": record.get("graph_id"),
            "assignment_id": record.get("assignment_id"),
            "alice": record.get("alice"),
            "bob": record.get("bob"),
            "sequence_route_diagnostics": record.get("sequence_route_diagnostics"),
        }
        for record in per_graph
        if record.get("sequence_route_diagnostics") is not None
    ]
    with open(dat_dir / "corpus_sequence_route_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(route_diagnostics, f, indent=2, default=str)
    print(
        "  → data/corpus_sequence_route_diagnostics.json "
        f"({len(route_diagnostics)} graphs)"
    )

    # ── Per-condition aggregates + corpus figures (no LLM required) ───────────
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows_by_combo = _corpus_case_rows_by_combo(per_graph)
    if rows_by_combo:
        condition_summary = _summarize_corpus_conditions(rows_by_combo)
        _atomic_write_json(dat_dir / "corpus_condition_summary.json", condition_summary)
        print(
            "  → data/corpus_condition_summary.json "
            f"({len(rows_by_combo)} condition(s))"
        )
        _render_corpus_figures(rows_by_combo, fig_dir)

    # Fit one cross-graph DT per learned condition and per emission type.
    # On resume, load existing DT stats from disk if available (structural_rows
    # are not persisted so refitting from checkpoint data would fail).
    _dt_by_combo_file = dat_dir / "corpus_agg_stats_by_combo.json"
    _dt_emission_file = dat_dir / "corpus_dt_stats_by_emission.json"
    _rc_by_combo_file = dat_dir / "corpus_route_choice_agg_stats_by_combo.json"
    dt_cache_current = _dt_artifacts_match_completed_manifest(dat_dir, completed_manifest)
    if resume_sim and _dt_by_combo_file.exists() and _dt_emission_file.exists() and dt_cache_current:
        print("  Loading DT stats from disk (resume — structural rows not re-fitted)")
        learned_aggs = json.loads(_dt_by_combo_file.read_text(encoding="utf-8"))
        route_choice_aggs = (
            json.loads(_rc_by_combo_file.read_text(encoding="utf-8"))
            if _rc_by_combo_file.exists()
            else {}
        )
    else:
        if resume_sim and _dt_by_combo_file.exists() and not dt_cache_current:
            print("  Re-fitting DT stats (completed case manifest changed)")
        learned_aggs = _fit_learned_condition_dts(
            per_graph,
            combos=run_combos,
            max_depth=dt_depth,
        )
        route_choice_aggs = _fit_learned_alice_route_dts(
            per_graph,
            combos=run_combos,
            max_depth=dt_depth,
        )
        for combo_agg in learned_aggs.values():
            combo_agg["corpus_selection"] = dict(selection_metadata)
        for combo_agg in route_choice_aggs.values():
            combo_agg["corpus_selection"] = dict(selection_metadata)
        if not learned_aggs and not route_choice_aggs:
            print("  WARNING: no learned-condition DT rows produced — skipping DT and LLM synthesis")
            return {}

        if learned_aggs:
            agg = learned_aggs.get(
                per_graph[0].get("primary_combo_key", "") if per_graph else ""
            ) or next(iter(learned_aggs.values()))
            with open(dat_dir / "corpus_agg_stats.json", "w", encoding="utf-8") as f:
                json.dump(agg, f, indent=2, default=str)
            with open(_dt_by_combo_file, "w", encoding="utf-8") as f:
                json.dump(learned_aggs, f, indent=2, default=str)
            print(f"  → data/corpus_agg_stats.json ({agg['condition']['key']})")
            print(f"  → data/corpus_agg_stats_by_combo.json ({len(learned_aggs)} learned Eve conditions)")
        if route_choice_aggs:
            with open(_rc_by_combo_file, "w", encoding="utf-8") as f:
                json.dump(route_choice_aggs, f, indent=2, default=str)
            print(
                "  → data/corpus_route_choice_agg_stats_by_combo.json "
                f"({len(route_choice_aggs)} learned Alice conditions)"
            )
        emission_aggs = {
            "eve_attack_choice": learned_aggs,
            "alice_route_choice": route_choice_aggs,
        }
        with open(_dt_emission_file, "w", encoding="utf-8") as f:
            json.dump(emission_aggs, f, indent=2, default=str)
        _write_dt_artifact_manifest(dat_dir, completed_manifest)
        print("  → data/corpus_dt_stats_by_emission.json")

    try:
        plot_corpus_dt_summary(
            learned_aggs,
            route_choice_aggs,
            fig_dir / "corpus_dt_summary.png",
        )
    except Exception as exc:  # a figure failure must not kill a long corpus run
        print(f"  WARNING: corpus_dt_summary.png failed: {exc}")

    primary_key = per_graph[0].get("primary_combo_key") if per_graph else ""
    if learned_aggs:
        agg = learned_aggs.get(primary_key) or next(iter(learned_aggs.values()))
    else:
        agg = next(iter(route_choice_aggs.values()))

    if no_llm:
        return agg

    # LLM synthesis: explainer + recommender for every learned condition.
    sum_dir = out_dir / "summaries"
    sum_dir.mkdir(parents=True, exist_ok=True)

    # Sweep per model so each model stays loaded for all its prompts.
    # Skip any output file that already exists (safe to resume after crash).
    for model in models:
        print(f"\n  --- model: {model} ---", flush=True)
        for combo_key, combo_agg in learned_aggs.items():
            condition_slug = _model_slug(combo_key)
            rules_text = combo_agg["structural_dt"]["rules_text"]
            model_slug = _model_slug(model)
            out_slug = f"{condition_slug}__{model_slug}"
            explainer_path = sum_dir / f"cross_graph_explainer_{out_slug}.md"
            explainer_prompt = build_cross_graph_explainer_prompt(combo_agg, rules_text)
            _write_llm_prompt_sidecar(explainer_path, explainer_prompt)
            if _llm_success_path_exists(explainer_path):
                print(f"  LLM explainer ({combo_key}, {model}) … skipped (exists)")
            else:
                print(f"  LLM explainer ({combo_key}, {model}) …", end="", flush=True)
                t0 = time.perf_counter()
                explainer_text = call_llm(explainer_prompt, model)
                elapsed = time.perf_counter() - t0
                print(f" {elapsed:.1f}s", end="")
                md = (
                    f"# Cross-Graph Capability Explainer\n\n"
                    f"**Condition:** `{combo_key}`  |  **Model:** `{model}`  |  "
                    f"**Generated:** {ts}  |  **Scope:** toy\n\n"
                    f"{explainer_text}\n"
                )
                if _write_llm_markdown_or_error(
                    explainer_path,
                    model=model,
                    prompt_kind=f"cross_graph_explainer:{combo_key}",
                    text=explainer_text,
                    elapsed=elapsed,
                    markdown=md,
                ):
                    print()

            rec_path = sum_dir / f"graph_recommender_{out_slug}.md"
            rec_prompt = build_graph_recommender_prompt(combo_agg, rules_text)
            _write_llm_prompt_sidecar(rec_path, rec_prompt)
            if _llm_success_path_exists(rec_path):
                print(f"  LLM recommender ({combo_key}, {model}) … skipped (exists)")
            else:
                print(f"  LLM recommender ({combo_key}, {model}) …", end="", flush=True)
                t0 = time.perf_counter()
                rec_text = call_llm(rec_prompt, model)
                elapsed = time.perf_counter() - t0
                print(f" {elapsed:.1f}s", end="")
                md = (
                    f"# Graph Topology Recommender\n\n"
                    f"**Condition:** `{combo_key}`  |  **Model:** `{model}`  |  "
                    f"**Generated:** {ts}  |  **Scope:** toy\n\n"
                    f"{rec_text}\n"
                )
                if _write_llm_markdown_or_error(
                    rec_path,
                    model=model,
                    prompt_kind=f"graph_recommender:{combo_key}",
                    text=rec_text,
                    elapsed=elapsed,
                    markdown=md,
                ):
                    print()

        for combo_key, combo_agg in route_choice_aggs.items():
            condition_slug = _model_slug(combo_key)
            rules_text = combo_agg["structural_dt"]["rules_text"]
            model_slug = _model_slug(model)
            out_slug = f"{condition_slug}__{model_slug}"
            rc_exp_path = sum_dir / f"route_choice_explainer_{out_slug}.md"
            rc_exp_prompt = build_route_choice_explainer_prompt(combo_agg, rules_text)
            _write_llm_prompt_sidecar(rc_exp_path, rc_exp_prompt)
            if _llm_success_path_exists(rc_exp_path):
                print(f"  LLM route-choice explainer ({combo_key}, {model}) … skipped (exists)")
            else:
                print(f"  LLM route-choice explainer ({combo_key}, {model}) …", end="", flush=True)
                t0 = time.perf_counter()
                explainer_text = call_llm(rc_exp_prompt, model)
                elapsed = time.perf_counter() - t0
                print(f" {elapsed:.1f}s", end="")
                md = (
                    f"# Cross-Graph Route-Choice Explainer\n\n"
                    f"**Condition:** `{combo_key}`  |  **Model:** `{model}`  |  "
                    f"**Generated:** {ts}  |  **Scope:** toy\n\n"
                    f"{explainer_text}\n"
                )
                if _write_llm_markdown_or_error(
                    rc_exp_path,
                    model=model,
                    prompt_kind=f"route_choice_explainer:{combo_key}",
                    text=explainer_text,
                    elapsed=elapsed,
                    markdown=md,
                ):
                    print()

            rc_rec_path = sum_dir / f"route_choice_recommender_{out_slug}.md"
            rc_rec_prompt = build_route_choice_recommender_prompt(combo_agg, rules_text)
            _write_llm_prompt_sidecar(rc_rec_path, rc_rec_prompt)
            if _llm_success_path_exists(rc_rec_path):
                print(f"  LLM route-choice recommender ({combo_key}, {model}) … skipped (exists)")
            else:
                print(f"  LLM route-choice recommender ({combo_key}, {model}) …", end="", flush=True)
                t0 = time.perf_counter()
                rec_text = call_llm(rc_rec_prompt, model)
                elapsed = time.perf_counter() - t0
                print(f" {elapsed:.1f}s", end="")
                md = (
                    f"# Alice Route-Choice Topology Recommender\n\n"
                    f"**Condition:** `{combo_key}`  |  **Model:** `{model}`  |  "
                    f"**Generated:** {ts}  |  **Scope:** toy\n\n"
                    f"{rec_text}\n"
                )
                if _write_llm_markdown_or_error(
                    rc_rec_path,
                    model=model,
                    prompt_kind=f"route_choice_recommender:{combo_key}",
                    text=rec_text,
                    elapsed=elapsed,
                    markdown=md,
                ):
                    print()

    return agg


def _learned_suite_combos(combos: list[dict] | None = None) -> list[dict]:
    suite = EXPERIMENT_SUITE if combos is None else combos
    return [combo for combo in suite if combo["eve"] in LEARNED_EVE_ALGOS]


def _learned_alice_suite_combos(combos: list[dict] | None = None) -> list[dict]:
    suite = EXPERIMENT_SUITE if combos is None else combos
    return [combo for combo in suite if combo["alice"] in LEARNED_ALICE_ALGOS]


def _fit_learned_condition_dts(
    per_graph: list[dict],
    *,
    combos: list[dict] | None = None,
    max_depth: int = 4,
) -> dict[str, dict]:
    rows_by_combo: dict[str, list[dict]] = defaultdict(list)
    for record in per_graph:
        for row in record.get("structural_rows", []):
            combo_key = row.get("combo_key")
            if combo_key:
                rows_by_combo[str(combo_key)].append(row)

    aggs: dict[str, dict] = {}
    for combo in _learned_suite_combos(combos):
        key = combo["key"]
        rows = rows_by_combo.get(key, [])
        if not rows:
            print(f"  WARNING: no structural rows for learned condition {key}")
            continue
        combo_records = _records_for_combo(per_graph, combo)
        if not combo_records:
            print(f"  WARNING: no completed records for learned condition {key}")
            continue
        print(f"  Fitting structural DT for {key} on {len(rows)} pooled rows …")
        dt_result = fit_cross_graph_dt(rows, max_depth=max_depth)
        agg = aggregate_corpus_stats(combo_records, dt_result)
        agg["condition"] = {
            "key": key,
            "game_mode": combo["game_mode"],
            "eve": combo["eve"],
            "alice": combo["alice"],
            "group": combo.get("group"),
            "dt_target": "eve_q_rank",
            "policy_scope": "learned_eve_policy",
            "emission_type": "eve_attack_choice",
        }
        print(
            f"  DT[{key}] R²={dt_result.r2_score:.3f} "
            f"({dt_result.num_topologies} topologies, {dt_result.num_rows} rows)"
        )
        aggs[key] = agg
    return aggs


def _fit_learned_alice_route_dts(
    per_graph: list[dict],
    *,
    combos: list[dict] | None = None,
    max_depth: int = 4,
) -> dict[str, dict]:
    rows_by_combo: dict[str, list[dict]] = defaultdict(list)
    for record in per_graph:
        for row in record.get("route_choice_rows", []):
            combo_key = row.get("combo_key")
            if combo_key:
                rows_by_combo[str(combo_key)].append(row)

    aggs: dict[str, dict] = {}
    for combo in _learned_alice_suite_combos(combos):
        key = combo["key"]
        rows = rows_by_combo.get(key, [])
        if not rows:
            print(f"  WARNING: no route-choice rows for learned Alice condition {key}")
            continue
        combo_records = _records_for_combo(per_graph, combo)
        if not combo_records:
            print(f"  WARNING: no completed records for learned Alice condition {key}")
            continue
        print(f"  Fitting route-choice DT for {key} on {len(rows)} pooled rows …")
        dt_result = fit_cross_graph_dt(
            rows,
            max_depth=max_depth,
            target_key="choice_rate",
            feature_names=ROUTE_CHOICE_FEATURE_NAMES,
        )
        agg = aggregate_corpus_stats(combo_records, dt_result)
        agg["condition"] = {
            "key": key,
            "game_mode": combo["game_mode"],
            "eve": combo["eve"],
            "alice": combo["alice"],
            "group": combo.get("group"),
            "dt_target": "alice_route_choice_rate",
            "policy_scope": "learned_alice_route_policy",
            "emission_type": "alice_route_choice",
        }
        print(
            f"  DT[alice_route_choice:{key}] R²={dt_result.r2_score:.3f} "
            f"({dt_result.num_topologies} topologies, {dt_result.num_rows} rows)"
        )
        aggs[key] = agg
    return aggs


def _records_for_combo(per_graph: list[dict], combo: dict) -> list[dict]:
    key = combo["key"]
    records = []
    for record in per_graph:
        combo_result = (record.get("combo_results") or {}).get(key) or {}
        if "eve_win_rate" not in combo_result:
            continue
        projected = {
            k: v for k, v in record.items()
            if k not in {"structural_rows", "route_choice_rows"}
        }
        projected.update({
            "condition_key": key,
            "game_mode": combo["game_mode"],
            "eve_algo": combo["eve"],
            "alice_algo": combo["alice"],
            "combo_group": combo.get("group"),
            "eve_win_rate": combo_result["eve_win_rate"],
            "final_win_rate": combo_result["final_win_rate"],
        })
        records.append(projected)
    return records


def _corpus_records_from_manifest(manifest: dict) -> list[dict]:
    """Return runnable graph/assignment records from v1 or v2 corpus manifests."""
    if manifest.get("accepted"):
        return list(manifest["accepted"])

    graphs_by_id = {graph["graph_id"]: graph for graph in manifest.get("graphs", [])}
    records = []
    for assignment in manifest.get("assignments", []):
        graph = graphs_by_id.get(assignment.get("graph_id"), {})
        if not graph:
            continue
        records.append({
            **graph,
            **assignment,
            "candidate_route_count": assignment.get("route_count"),
        })
    return records


def _derive_corpus_run_seed(base_seed: int, record: dict) -> int:
    """Derive a stable per-assignment seed from the base corpus seed."""
    seed_material = "|".join([
        str(int(base_seed)),
        str(record.get("graph_id", "")),
        str(record.get("assignment_id", "")),
        str(record.get("alice", "")),
        str(record.get("bob", "")),
    ])
    digest = hashlib.blake2b(seed_material.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % (2**32 - 1)


def _latest_output_dir(out_root: Path) -> Path | None:
    if not out_root.exists():
        return None
    candidates = [
        path for path in out_root.iterdir()
        if path.is_dir() and (path / "data").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper results pipeline")
    parser.add_argument("--out-dir", default="results/paper")
    parser.add_argument(
        "--run-id",
        default=None,
        help=(
            "Stable output directory name under --out-dir. Fails if it already exists "
            "unless an explicit resume flag is used."
        ),
    )
    parser.add_argument("--steps",   type=int, default=3000, help="Steps per run")
    parser.add_argument("--seeds",   type=int, default=3,    help="Random seeds")
    parser.add_argument("--window",  type=int, default=50,   help="Rolling avg window")
    parser.add_argument("--llm",     default="phi4:14b",     help="Primary Ollama model")
    parser.add_argument("--no-llm",  action="store_true",    help="Skip all LLM calls")
    parser.add_argument(
        "--compare-llms",
        default=None,
        help=(
            "Comma-separated list of Ollama models to compare in addition to --llm. "
            "E.g. phi4:14b,huihui_ai/qwen3-abliterated:14b,qwq:32b"
        ),
    )
    parser.add_argument(
        "--no-mitigation",
        action="store_true",
        help="Skip mitigation evaluation (saves time when only sim metrics are needed)",
    )
    parser.add_argument(
        "--corpus-manifest",
        default=None,
        metavar="PATH",
        help="Path to corpus_manifest.json from generate_graph_corpus.py. "
             "Enables the cross-graph DQN+DT+LLM pipeline.",
    )
    parser.add_argument(
        "--corpus-db",
        default=None,
        metavar="PATH",
        help=(
            "SQLite DB containing ingested graph_corpora/source_files. When set, "
            "the corpus graph JSON is read from SQLite and completed cases are "
            "written back to the same DB unless --corpus-results-db is provided."
        ),
    )
    parser.add_argument(
        "--corpus-id",
        default=None,
        metavar="ID",
        help=(
            "graph_corpora.corpus_id to read from --corpus-db. Required when the "
            "DB contains more than one graph corpus."
        ),
    )
    parser.add_argument(
        "--corpus-results-db",
        default=None,
        metavar="PATH",
        help="SQLite DB path for corpus run results when different from --corpus-db.",
    )
    parser.add_argument(
        "--corpus-json-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Write the legacy corpus_checkpoint.json. Defaults to off when SQLite "
            "results are enabled and on for legacy file-only runs."
        ),
    )
    parser.add_argument(
        "--corpus-combo-keys",
        default=None,
        metavar="KEYS",
        help=(
            "Comma-separated subset of EXPERIMENT_SUITE keys to run for the corpus "
            "pipeline. Omit for the contracted six-condition publication suite."
        ),
    )
    parser.add_argument(
        "--experiment-suite",
        choices=sorted(EXPERIMENT_SUITE_ALIASES),
        default=EXPERIMENT_SUITE_WORKSHOP_PRIMARY_SIX,
        help=(
            "Named experiment suite for corpus runs when --corpus-combo-keys is "
            "omitted. workshop_primary_six is the six-hook attack_surface_all suite; "
            "workshop_main_two keeps only the two main deep-Q project conditions."
        ),
    )
    parser.add_argument(
        "--assignment-selection",
        choices=[ASSIGNMENT_SELECTION_ALL, ASSIGNMENT_SELECTION_TOP2_PER_GRAPH],
        default=ASSIGNMENT_SELECTION_ALL,
        help=(
            "Corpus assignment selection rule. top2_per_graph selects at most two "
            "preselected route assignments per graph before expanding experiment hooks."
        ),
    )
    parser.add_argument(
        "--corpus-max-graphs",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N accepted corpus graph/assignment records.",
    )
    parser.add_argument(
        "--corpus-reference-max-graphs",
        "--corpus-control-max-graphs",
        dest="corpus_reference_max_graphs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run reference/control group combos only on assignments belonging "
            "to the first N unique graph IDs in the selected corpus; main combos "
            "still run on every selected record."
        ),
    )
    parser.add_argument(
        "--corpus-max-episodes",
        type=int,
        default=300,
        metavar="N",
        help="DQN training steps per corpus graph (default: 300)",
    )
    parser.add_argument(
        "--runtime-backend",
        choices=[
            RUNTIME_BACKEND_SEQUENCE,
            RUNTIME_BACKEND_DIRECT_E91,
            RUNTIME_BACKEND_COMPILED_HISTOGRAM,
        ],
        default=RUNTIME_BACKEND_SEQUENCE,
        help=(
            "Runtime backend for attack_surface corpus cases. Default is sequence; "
            "direct_e91_transcript uses the direct transcript backend; "
            "compiled_histogram samples game turns from the compiled length-parametric "
            "E91 transcript histogram kernel."
        ),
    )
    parser.add_argument(
        "--compiled-histogram-evaluated-cache-size",
        type=int,
        default=256,
        help=(
            "compiled_histogram backend: bounded LRU size for evaluated H(L) "
            "histograms per worker process."
        ),
    )
    parser.add_argument(
        "--compiled-histogram-signature-cap",
        type=int,
        default=512,
        help=(
            "compiled_histogram backend: waiting_max_wait_signatures compute cap "
            "used when compiling/evaluating the histogram kernel."
        ),
    )
    parser.add_argument(
        "--compiled-histogram-fock-cutoff",
        type=int,
        default=3,
        help="compiled_histogram backend: Fock-space cutoff for the elementary instrument.",
    )
    parser.add_argument(
        "--compiled-histogram-detector-kind",
        default="pnr_snspd",
        help="compiled_histogram backend: explicit detector model kind (production default: pnr_snspd).",
    )
    parser.add_argument(
        "--compiled-histogram-detector-profile",
        default=str(
            _ROOT
            / "configs"
            / "physical"
            / "detectors"
            / "snspd_colangelo_impedance_matched_differential_pnr_1550.json"
        ),
        help="compiled_histogram backend: selected detector operating-point JSON.",
    )
    parser.add_argument("--compiled-histogram-detector-response-file", default=None)
    parser.add_argument("--compiled-histogram-fiber-response-file", default=None)
    parser.add_argument(
        "--compiled-histogram-conversion-response-file",
        default=str(
            _ROOT
            / "configs"
            / "physical"
            / "conversion"
            / "cui_dfg_866_to_1558.json"
        ),
        help="compiled_histogram backend: selected frequency-conversion response JSON.",
    )
    parser.add_argument(
        "--compiled-histogram-multiplexing-controller",
        default=None,
        help=(
            "compiled_histogram backend: optional independent time-bin controller JSON; "
            "for h>=2 this requires the rolling hop calibration path"
        ),
    )
    parser.add_argument(
        "--compiled-histogram-repeater-service-profile",
        default=None,
        help=(
            "compiled_histogram backend: opt-in literature-scoped asynchronous "
            "repeater service profile; rolling availability/ages are derived for "
            "each selected route's actual fiber vector with the profile's fixed inventory"
        ),
    )
    parser.add_argument(
        "--compiled-histogram-repeater-service-t-alive-s",
        type=float,
        default=None,
        help=(
            "compiled_histogram repeater-service profile: optional purified/live-pair "
            "staleness cutoff override. By default the profile's controller value is used."
        ),
    )
    parser.add_argument(
        "--compiled-histogram-allow-wavelength-mismatch",
        action="store_true",
        help="diagnostic override only; permit an explicitly reported uncovered wavelength stack.",
    )
    parser.add_argument(
        "--compiled-histogram-allow-ideal-detector-reference",
        action="store_true",
        help="permit ideal_pnr only for an explicitly labelled mathematical reference/test run.",
    )
    parser.add_argument(
        "--compiled-histogram-hop-calibration-10km",
        action="store_true",
        help=(
            "compiled_histogram backend: use the hop-based rolling_alive_pair "
            "reservation calibration derived on the standardized 10 km/link "
            "route family."
        ),
    )
    parser.add_argument(
        "--compiled-histogram-hop-calibration-t-alive-s",
        type=float,
        default=0.18,
        help="compiled_histogram 10 km hop calibration: accepted live-pair cutoff in seconds.",
    )
    parser.add_argument(
        "--compiled-histogram-hop-calibration-age-points",
        type=int,
        default=8,
        help="compiled_histogram 10 km hop calibration: age quadrature points per edge.",
    )
    parser.add_argument(
        "--compiled-histogram-hop-calibration-route-availability",
        type=float,
        default=0.95,
        help="compiled_histogram 10 km hop calibration: target route availability.",
    )
    parser.add_argument(
        "--disable-direct-e91-prepared-cache",
        action="store_true",
        help=(
            "Validation switch for --runtime-backend direct_e91_transcript: "
            "run direct steps without the prepared-transition cache."
        ),
    )
    parser.add_argument(
        "--direct-e91-training-light",
        action="store_true",
        help=(
            "Opt-in direct E91 training-light output mode. Full provenance remains "
            "the default for paper outputs."
        ),
    )
    parser.add_argument(
        "--corpus-seed",
        type=int,
        default=42,
        help="RNG seed for per-graph DQN training (default: 42)",
    )
    parser.add_argument(
        "--corpus-checkpoint-every",
        type=int,
        default=10,
        metavar="N",
        help="write corpus checkpoint every N completed assignments (default: 10)",
    )
    parser.add_argument(
        "--corpus-dt-depth",
        type=int,
        default=4,
        help="max depth for corpus DT refit snapshots and pooled DTs (default: 4)",
    )
    parser.add_argument(
        "--corpus-dt-refit-mode",
        choices=["fixed", "logarithmic", "dynamic", "off"],
        default="dynamic",
        help="corpus learned-combo DT explanation refit schedule (default: dynamic)",
    )
    parser.add_argument("--corpus-dt-observer-interval", type=int, default=50)
    parser.add_argument("--corpus-dt-fixed-refit-interval", type=int, default=50)
    parser.add_argument("--corpus-dt-epsilon", type=float, default=0.10)
    parser.add_argument("--corpus-dt-delta", type=float, default=0.05)
    parser.add_argument("--corpus-dt-policy-drift-threshold", type=float, default=0.05)
    parser.add_argument("--corpus-dt-min-refit-gap", type=int, default=None)
    parser.add_argument("--corpus-dt-max-refit-gap", type=int, default=2000)
    parser.add_argument("--corpus-dt-log-snapshot-count", type=int, default=8)
    parser.add_argument("--corpus-dt-probe-state-count", type=int, default=2048)
    parser.add_argument(
        "--corpus-dt-snapshot-q-entry-budget",
        type=int,
        default=2_097_152,
        help=(
            "cap on probe_states x num_actions Q entries per observer snapshot; "
            "bounds memory on route-dense graphs (default: 2097152)"
        ),
    )
    parser.add_argument(
        "--corpus-dt-final-step-refit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--no-corpus",
        action="store_true",
        help="Skip corpus pipeline even if --corpus-manifest is given",
    )
    parser.add_argument(
        "--no-suite",
        action="store_true",
        help="Skip the small/toy experiment suite, figures, suite LLM summary, and mitigation evaluation.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel workers for corpus graph runs (default: 1). Use os.cpu_count() for max.",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        metavar="PATH",
        help=(
            "Resume an interrupted run from an existing output directory "
            "(e.g. results/paper/20260627T153841Z). Skips suite simulation and "
            "corpus simulation; re-uses checkpoint data and skips already-written LLM files."
        ),
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="Resume the newest existing output directory under --out-dir.",
    )
    parser.add_argument(
        "--resume-run-id",
        default=None,
        metavar="RUN_ID",
        help="Explicitly resume an existing run ID under --out-dir.",
    )
    parser.add_argument(
        "--allow-failed-cases",
        action="store_true",
        help="Continue after failed suite/corpus cases; failures are written to data/failed_cases.jsonl.",
    )
    args = parser.parse_args()

    resume_flags = [bool(args.resume_from), bool(args.resume_latest), bool(args.resume_run_id)]
    if sum(resume_flags) > 1:
        raise SystemExit("--resume-from, --resume-latest, and --resume-run-id are mutually exclusive")

    out_root = Path(args.out_dir)
    resume_from: Path | None = Path(args.resume_from) if args.resume_from else None
    if args.resume_run_id:
        resume_from = out_root / args.resume_run_id
    elif args.resume_latest:
        resume_from = _latest_output_dir(out_root)
        if resume_from is None:
            raise SystemExit(f"--resume-latest found no existing runs under {out_root}")
    elif args.run_id:
        candidate = out_root / args.run_id
        if candidate.exists():
            raise SystemExit(
                f"--run-id already exists: {candidate}. "
                "Use --resume-run-id, --resume-from, or --resume-latest to resume explicitly."
            )

    if resume_from is not None:
        out_dir = resume_from
        if not out_dir.exists():
            raise SystemExit(f"--resume-from path does not exist: {out_dir}")
        ts = out_dir.name  # preserve original timestamp
        print(f"Resuming run from: {out_dir}")
    else:
        ts = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = out_root / ts
    fig_dir = out_dir / "figures"
    dat_dir = out_dir / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    dat_dir.mkdir(parents=True, exist_ok=True)

    compare_models: list[str] = []
    if args.compare_llms:
        compare_models = [m.strip() for m in args.compare_llms.split(",") if m.strip()]
        if args.llm not in compare_models:
            compare_models.insert(0, args.llm)

    try:
        experiment_suite_combos = _combos_for_experiment_suite(args.experiment_suite)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    corpus_combos: list[dict] | None = experiment_suite_combos
    if args.corpus_combo_keys:
        combo_keys = [key.strip() for key in args.corpus_combo_keys.split(",") if key.strip()]
        if not combo_keys:
            raise SystemExit("--corpus-combo-keys was given but no keys were parsed")
        try:
            corpus_combos = _combos_for_keys(combo_keys, suite=experiment_suite_combos)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    corpus_dt_refit_config = DynamicDTRefitConfig(
        mode=args.corpus_dt_refit_mode,
        observer_interval=args.corpus_dt_observer_interval,
        fixed_refit_interval=args.corpus_dt_fixed_refit_interval,
        epsilon=args.corpus_dt_epsilon,
        delta=args.corpus_dt_delta,
        policy_drift_threshold=args.corpus_dt_policy_drift_threshold,
        min_refit_gap=args.corpus_dt_min_refit_gap,
        max_refit_gap=args.corpus_dt_max_refit_gap,
        log_snapshot_count=args.corpus_dt_log_snapshot_count,
        probe_state_count=args.corpus_dt_probe_state_count,
        snapshot_q_entry_budget=args.corpus_dt_snapshot_q_entry_budget,
        final_step_refit=args.corpus_dt_final_step_refit,
    )

    print(f"Paper results pipeline — {ts}")
    has_corpus_source = bool(args.corpus_manifest or args.corpus_db)
    print(f"Output: {out_dir}")
    print(f"Config: steps={args.steps}  seeds={args.seeds}  window={args.window}")
    if compare_models:
        print(f"LLM comparison: {compare_models}")
    if has_corpus_source and not args.no_corpus:
        print(
            "Corpus DT refit: "
            f"mode={corpus_dt_refit_config.mode} "
            f"observer_interval={corpus_dt_refit_config.observer_interval} "
            f"epsilon={corpus_dt_refit_config.epsilon} "
            f"delta={corpus_dt_refit_config.delta} "
            f"max_gap={corpus_dt_refit_config.max_refit_gap}"
        )
        if args.corpus_reference_max_graphs is not None:
            print(
                "Corpus references: "
                f"first {args.corpus_reference_max_graphs} unique graph IDs only"
            )
        print(
            "Corpus runtime backend: "
            f"{args.runtime_backend}"
            + (
                " "
                f"(prepared_cache={'disabled' if args.disable_direct_e91_prepared_cache else 'enabled'}, "
                f"training_light={bool(args.direct_e91_training_light)})"
                if args.runtime_backend == RUNTIME_BACKEND_DIRECT_E91
                else ""
            )
        )
        print(
            "Corpus experiment suite: "
            f"{args.experiment_suite} ({len(corpus_combos)} hook(s)); "
            f"assignment_selection={args.assignment_selection}"
        )

    # ── Run experiments (configured suite) ────────────────────────────────────
    metrics: dict | None = None
    all_results: dict[str, list[dict]] | None = None
    metrics_loaded_from_resume = False
    metrics_have_curves = False

    is_resume = resume_from is not None

    if args.no_suite:
        print("\nSkipping configured experiment suite (--no-suite)")
    elif is_resume:
        metrics, metrics_have_curves = _load_resume_metrics(dat_dir)
        if metrics is not None:
            metrics_loaded_from_resume = True
            print(f"\n=== Loading saved suite metrics (resume) ===")
            if not metrics_have_curves:
                print("  Loaded legacy slim metrics; convergence figures will not be regenerated")

    if not args.no_suite and metrics is None:
        if is_resume:
            print("\n=== Saved suite metrics missing; rerunning experiments ===")
        print(f"\n=== Running experiments ({len(EXPERIMENT_SUITE)} combos) ===")
        all_results = run_suite(
            args.steps,
            args.seeds,
            allow_failed_cases=args.allow_failed_cases,
            failed_cases_file=dat_dir / "failed_cases.jsonl",
        )
        raw = {k: [r["summary"] for r in res] for k, res in all_results.items()}
        with open(dat_dir / "run_summaries.json", "w", encoding="utf-8") as f:
            json.dump(raw, f, indent=2, default=str)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if args.no_suite:
        pass
    elif metrics_loaded_from_resume:
        print("\n=== Saved metrics ===")
    else:
        print("\n=== Aggregating metrics ===")
        assert all_results is not None
        metrics = aggregate_metrics(all_results, args.window)
        _write_metrics_files(metrics, dat_dir)
        metrics_have_curves = _metrics_have_rolling_curves(metrics)
    if metrics is not None:
        _print_metric_summary(metrics)

    # ── Figures (one convergence + strategy plot per group) ────────────────────
    if not args.no_suite and metrics is not None:
        print("\n=== Generating figures ===")
        for group in ("reference", "main", "baseline"):
            grp_keys = [c["key"] for c in EXPERIMENT_SUITE if c["group"] == group and c["key"] in metrics]
            if not grp_keys:
                continue
            mode_label = {
                "baseline":  "binary-collision baselines",
                "reference": "all-surface references",
                "main":      "main all-surface objects",
            }[group]
            if metrics_have_curves:
                plot_convergence(
                    metrics, args.window,
                    fig_dir / f"convergence_{group}.png",
                    keys=grp_keys,
                    title=f"Eve convergence — {mode_label} (toy scope)",
                )
            elif group == "baseline":
                print("  skip convergence figures (saved metrics do not include rolling curves)")
            plot_strategy_distribution(
                metrics,
                fig_dir / f"strategy_distribution_{group}.png",
                keys=grp_keys,
                title=f"Attack distribution — {mode_label} (final 30 %, toy scope)",
            )

        # Q-heatmaps for one DQN combo per group (first available)
        for group in ("reference", "main", "baseline"):
            dqn_key = next(
                (c["key"] for c in EXPERIMENT_SUITE
                 if c["group"] == group and c["eve"] == "deep_q" and c["key"] in metrics),
                None,
            )
            if dqn_key:
                plot_q_heatmap(metrics, dqn_key, fig_dir / f"q_table_heatmap_{group}.png")

    # ── LLM summary ───────────────────────────────────────────────────────────
    if args.no_suite:
        print("\nSkipping suite LLM summary (--no-suite)")
    elif not args.no_llm:
        n_seeds_actual = (
            min(len(v) for v in all_results.values())
            if all_results else _metrics_seed_count(metrics)
        )
        prompt = build_prompt(metrics, args.steps, n_seeds_actual, args.window)
        with open(out_dir / "summary_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt)

        if compare_models:
            # ── Multi-LLM comparison mode ──────────────────────────────────────
            print(f"\n=== LLM comparison ({len(compare_models)} models) ===")
            summaries = llm_summaries(prompt, compare_models, out_dir, ts)
            if summaries:
                write_comparison(summaries, out_dir, ts)
                # Primary model also written to summary.md when available.
                primary_text = summaries.get(args.llm, next(iter(summaries.values())))
                primary_model = args.llm if args.llm in summaries else next(iter(summaries))
                md_primary = (
                    f"# Results Summary\n\n"
                    f"**Model:** `{primary_model}`  |  **Generated:** {ts}  |  "
                    f"**Scope:** toy simulation\n\n"
                    f"{primary_text}\n\n"
                    f"---\n"
                    f"_All claims are simulation results only. "
                    f"See `summaries/comparison.md` for multi-model comparison._\n"
                )
                with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
                    f.write(md_primary)
                print(f"  → summary.md (primary: {primary_model})")
            else:
                print("  WARNING: all LLM summary calls failed; summary.md not written")
        elif is_resume and _llm_success_path_exists(out_dir / "summary.md"):
            print(f"\n=== LLM summary ({args.llm}) ===")
            print("  skipped (summary.md exists)")
        else:
            # ── Single model ───────────────────────────────────────────────────
            print(f"\n=== LLM summary ({args.llm}) ===")
            t0 = time.perf_counter()
            text = call_llm(prompt, args.llm)
            elapsed = time.perf_counter() - t0
            print(f"  response in {elapsed:.1f}s")
            if text.startswith("[LLM unavailable"):
                print(f"  WARNING: {text}")
            md = (
                f"# Results Summary\n\n"
                f"**Model:** `{args.llm}`  |  **Generated:** {ts}  |  "
                f"**Scope:** toy simulation\n\n"
                f"{text}\n\n"
                f"---\n"
                f"_All claims are simulation results only. "
                f"Prompt saved in `summary_prompt.txt` for audit._\n"
            )
            if _write_llm_markdown_or_error(
                out_dir / "summary.md",
                model=args.llm,
                prompt_kind="results_summary",
                text=text,
                elapsed=elapsed,
                markdown=md,
            ):
                print(f"  → summary.md")
    else:
        print("\nSkipping LLM summary (--no-llm)")

    # ── Mitigation evaluation (DQN Eve combos only — XAI-relevant) ────────────
    if args.no_suite:
        print("\nSkipping suite mitigation evaluation (--no-suite)")
    elif not args.no_llm and not args.no_mitigation:
        mit_models = compare_models if compare_models else [args.llm]
        if not all_results:
            existing_mitigation = dat_dir / "mitigation_evaluations.json"
            print(f"\n=== Mitigation evaluation ({len(mit_models)} model(s)) ===")
            if is_resume and existing_mitigation.exists():
                print("  skipped (mitigation_evaluations.json exists)")
            else:
                print("  skipped (resume metrics do not include final replay windows)")
        else:
            # Run mitigation only on DQN Eve combos; random/oracle explanations are trivial
            dqn_results = {k: v for k, v in all_results.items() if "deep_q" in k}
            print(f"\n=== Mitigation evaluation ({len(mit_models)} model(s), {len(dqn_results)} DQN combo(s)) ===")
            mit_evals = run_mitigation_evaluations(dqn_results, mit_models, out_dir, ts)
            with open(dat_dir / "mitigation_evaluations.json", "w", encoding="utf-8") as f:
                json.dump(mit_evals, f, indent=2, default=str)
            print("  → data/mitigation_evaluations.json")
    elif args.no_mitigation:
        print("\nSkipping mitigation evaluation (--no-mitigation)")

    # ── Corpus pipeline ────────────────────────────────────────────────────────
    if has_corpus_source and not args.no_corpus:
        corpus_models = compare_models if compare_models else [args.llm]
        compiled_conversion_response = args.compiled_histogram_conversion_response_file
        if args.compiled_histogram_repeater_service_profile is not None:
            if args.compiled_histogram_hop_calibration_10km:
                raise SystemExit(
                    "--compiled-histogram-repeater-service-profile and "
                    "--compiled-histogram-hop-calibration-10km are mutually exclusive"
                )
            if args.compiled_histogram_multiplexing_controller is not None:
                raise SystemExit(
                    "the repeater service profile already declares its controller/resource model; "
                    "do not combine it with --compiled-histogram-multiplexing-controller"
                )
            historical_default_conversion = (
                _ROOT
                / "configs"
                / "physical"
                / "conversion"
                / "cui_dfg_866_to_1558.json"
            )
            if (
                Path(compiled_conversion_response).resolve()
                == historical_default_conversion.resolve()
            ):
                compiled_conversion_response = str(
                    _ROOT
                    / "configs"
                    / "physical"
                    / "conversion"
                    / "arenskoetter_qfc_854_to_1550.json"
                )
        compiled_histogram_config = {
            "evaluated_cache_size": args.compiled_histogram_evaluated_cache_size,
            "waiting_max_wait_signatures": args.compiled_histogram_signature_cap,
            "fock_cutoff": args.compiled_histogram_fock_cutoff,
            "detector_kind": args.compiled_histogram_detector_kind,
            "detector_profile_path": args.compiled_histogram_detector_profile,
            "detector_response_path": args.compiled_histogram_detector_response_file,
            "fiber_response_path": args.compiled_histogram_fiber_response_file,
            "conversion_response_path": compiled_conversion_response,
            "allow_wavelength_mismatch": bool(
                args.compiled_histogram_allow_wavelength_mismatch
            ),
            "allow_ideal_detector_reference": bool(
                args.compiled_histogram_allow_ideal_detector_reference
            ),
            "require_explicit_device_stack": True,
        }
        if args.compiled_histogram_multiplexing_controller is not None:
            compiled_histogram_config["multiplexing_controller_path"] = (
                args.compiled_histogram_multiplexing_controller
            )
        if args.compiled_histogram_repeater_service_profile is not None:
            compiled_histogram_config["repeater_service_profile_path"] = (
                args.compiled_histogram_repeater_service_profile
            )
            compiled_histogram_config["rolling_hop_calibration"] = {
                "kind": "repeater_service_profile",
                "profile_path": args.compiled_histogram_repeater_service_profile,
                "age_points_per_edge": (
                    args.compiled_histogram_hop_calibration_age_points
                ),
                "selection_policy": "youngest_live",
            }
            if args.compiled_histogram_repeater_service_t_alive_s is not None:
                compiled_histogram_config["rolling_hop_calibration"][
                    "accepted_alive_time_s"
                ] = args.compiled_histogram_repeater_service_t_alive_s
        elif args.compiled_histogram_hop_calibration_10km:
            compiled_histogram_config["rolling_hop_calibration"] = {
                "kind": "hop_based_10km",
                "edge_length_km": 10.0,
                "accepted_alive_time_s": (
                    args.compiled_histogram_hop_calibration_t_alive_s
                ),
                "route_availability_target": (
                    args.compiled_histogram_hop_calibration_route_availability
                ),
                "chsh_target": 2.10,
                "age_points_per_edge": (
                    args.compiled_histogram_hop_calibration_age_points
                ),
                "selection_policy": "youngest_live",
                "detector_kind": args.compiled_histogram_detector_kind,
                "detector_profile_path": args.compiled_histogram_detector_profile,
                "detector_response_path": args.compiled_histogram_detector_response_file,
                "fiber_response_path": args.compiled_histogram_fiber_response_file,
                "conversion_response_path": compiled_conversion_response,
                "allow_wavelength_mismatch": bool(
                    args.compiled_histogram_allow_wavelength_mismatch
                ),
                "allow_ideal_detector_reference": bool(
                    args.compiled_histogram_allow_ideal_detector_reference
                ),
                "fock_cutoff": args.compiled_histogram_fock_cutoff,
            }
            if args.compiled_histogram_multiplexing_controller is not None:
                compiled_histogram_config["rolling_hop_calibration"][
                    "multiplexing_controller_path"
                ] = args.compiled_histogram_multiplexing_controller
        print(f"\n=== Corpus pipeline ({len(corpus_models)} model(s)) ===")
        run_corpus_pipeline(
            args.corpus_manifest,
            corpus_models,
            out_dir,
            ts,
            max_episodes=args.corpus_max_episodes,
            seed=args.corpus_seed,
            no_llm=args.no_llm,
            workers=args.workers,
            checkpoint_every=args.corpus_checkpoint_every,
            resume_sim=is_resume,
            dt_refit_config=corpus_dt_refit_config,
            dt_depth=args.corpus_dt_depth,
            combos=corpus_combos,
            max_graphs=args.corpus_max_graphs,
            reference_max_graphs=args.corpus_reference_max_graphs,
            allow_failed_cases=args.allow_failed_cases,
            runtime_backend=args.runtime_backend,
            direct_e91_prepared_cache_enabled=(
                not args.disable_direct_e91_prepared_cache
            ),
            direct_e91_training_light=args.direct_e91_training_light,
            compiled_histogram_backend_config=compiled_histogram_config,
            assignment_selection=args.assignment_selection,
            experiment_suite_id=args.experiment_suite,
            corpus_db_path=args.corpus_db,
            corpus_id=args.corpus_id,
            results_db_path=args.corpus_results_db,
            write_json_checkpoint=args.corpus_json_checkpoint,
        )
    elif args.no_corpus:
        print("\nSkipping corpus pipeline (--no-corpus)")
    elif not has_corpus_source:
        print("\nNo corpus manifest or corpus DB given; skipping corpus pipeline")

    print(f"\nDone → {out_dir}")


if __name__ == "__main__":
    main()
