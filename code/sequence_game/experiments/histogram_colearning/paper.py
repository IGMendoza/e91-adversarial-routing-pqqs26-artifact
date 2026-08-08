"""Resumable paper-artifact ensemble for histogram bandit co-learning.

This module deliberately does not import the SeQUeNCe runtime.  It composes the
route-specific compiled histogram pool, the partial-feedback learners, and an
artifact layer that keeps physical calibration separate from adversarial game
claims.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
from typing import Any, Callable, Sequence

import matplotlib
import numpy as np
import scipy

from sequence_game.game_theory import exploitability

from .config import DEFAULT_CORPUS_DB_PATH, HistogramCoLearningConfig
from .corpus import HistogramGameCase, load_topology_bench_case
from .learning import run_histogram_colearning
from .physics import HistogramPhysicsPool, histogram_cache_plan


PAPER_ARTIFACT_SCHEMA = "histogram_colearning_paper_v1"
PAPER_BACKEND = "compiled_histogram_transcript"
DEFAULT_PAPER_SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class HistogramPaperCaseSpec:
    """A fixed, reviewable Topology Bench assignment in the paper ensemble."""

    slug: str
    graph_id: str
    assignment_id: str
    structural_regime: str
    rationale: str

    def to_dict(self) -> dict[str, str]:
        return {
            "slug": self.slug,
            "graph_id": self.graph_id,
            "assignment_id": self.assignment_id,
            "structural_regime": self.structural_regime,
            "rationale": self.rationale,
        }


DEFAULT_PAPER_CASES = (
    HistogramPaperCaseSpec(
        slug="short_disjoint",
        graph_id="0002_topology_bench_telecomserbia_seed_na",
        assignment_id="best_route_diverse__3__6",
        structural_regime="two edge/node-disjoint 3-hop routes",
        rationale="short, fully disjoint control with comparable route depths",
    ),
    HistogramPaperCaseSpec(
        slug="medium_disjoint",
        graph_id="0000_topology_bench_spiralight_seed_na",
        assignment_id="best_route_diverse__14__8",
        structural_regime="two edge/node-disjoint 5-hop routes",
        rationale="previously validated 100k co-learning case and medium-depth control",
    ),
    HistogramPaperCaseSpec(
        slug="mixed_bottleneck",
        graph_id="0006_topology_bench_bren_seed_na",
        assignment_id="strongest_node_bottleneck__1__10",
        structural_regime="four overlapping routes spanning 4--9 hops",
        rationale="heterogeneous capacities and shared edge/repeater attack surfaces",
    ),
    HistogramPaperCaseSpec(
        slug="long_composable",
        graph_id="0013_topology_bench_conus30_seed_na",
        assignment_id="best_route_diverse__10__5",
        structural_regime="complete cycle-free long-distance route set",
        rationale=(
            "long-distance composability stress with twelve complete routes "
            "and a maximum observed depth of twenty-one hops"
        ),
    ),
)


ProgressCallback = Callable[[str], None]


def run_histogram_paper_bundle(
    out_dir: str | Path,
    *,
    base_config: HistogramCoLearningConfig | None = None,
    seeds: Sequence[int] = DEFAULT_PAPER_SEEDS,
    corpus_db_path: str | Path = DEFAULT_CORPUS_DB_PATH,
    cases: Sequence[HistogramPaperCaseSpec] = DEFAULT_PAPER_CASES,
    resume: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run/reuse every replicate and write a self-describing artifact bundle."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / "runs"
    run_dir.mkdir(exist_ok=True)
    config = base_config or HistogramCoLearningConfig()
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be a non-empty sequence of unique integers")
    if not cases or len({case.slug for case in cases}) != len(cases):
        raise ValueError("paper cases must have non-empty, unique slugs")

    started_at = _utc_now()
    run_manifest = {
        "schema": PAPER_ARTIFACT_SCHEMA,
        "backend": PAPER_BACKEND,
        "deprecated_sequence_backend_allowed": False,
        "created_at_utc": started_at,
        "corpus_db_path": str(Path(corpus_db_path).resolve()),
        "base_config": config.to_dict(),
        "seeds": list(normalized_seeds),
        "replicates_per_case": len(normalized_seeds),
        "cases": [case.to_dict() for case in cases],
        "scientific_scope": _scientific_scope(),
        "repository": repository_provenance(ignore_paths=[output]),
        "software": software_provenance(),
        "resume_enabled": bool(resume),
    }
    _write_json(output / "config.json", run_manifest)

    payloads: list[dict[str, Any]] = []
    case_records: list[dict[str, Any]] = []
    loaded_cases: list[tuple[HistogramPaperCaseSpec, HistogramGameCase]] = []
    for case_index, spec in enumerate(cases):
        case = load_topology_bench_case(
            corpus_db_path,
            graph_id=spec.graph_id,
            assignment_id=spec.assignment_id,
        )
        structure = case_structure(case)
        case_records.append({**spec.to_dict(), **structure})
        loaded_cases.append((spec, case))
        cache_plan = histogram_cache_plan(case, config)
        _notify(
            progress_callback,
            f"case {case_index + 1}/{len(cases)} {spec.slug}: resolving "
            f"{len(case.routes)} route-specific histogram backends; cache "
            f"preflight {cache_plan['global_route_action_pair_count']} pairs -> "
            f"{cache_plan['canonical_unique_move_count_total']} canonical "
            f"entries, max "
            f"{cache_plan['maximum_canonical_moves_on_one_route']}/route <= "
            f"limit {cache_plan['configured_entries_per_route']}; estimated "
            f"resident cache "
            f"{cache_plan['estimated_full_action_surface_cache_mib']:.1f} MiB",
        )
        physics: HistogramPhysicsPool | None = None
        for replicate_index, seed in enumerate(normalized_seeds):
            run_path = run_dir / spec.slug / f"seed_{seed}.json"
            if resume and run_path.exists():
                payload = json.loads(run_path.read_text(encoding="utf-8"))
                validate_paper_run_payload(
                    payload,
                    spec=spec,
                    config=replace(config, seed=seed),
                    seed=seed,
                )
                _notify(
                    progress_callback,
                    f"case {spec.slug} seed {seed}: reused {run_path}",
                )
            else:
                if physics is None:
                    physics = HistogramPhysicsPool(case, config)
                run_config = replace(config, seed=seed)
                _notify(
                    progress_callback,
                    f"case {spec.slug} seed {seed}: running "
                    f"{run_config.turns:,} histogram turns",
                )
                summary = run_histogram_colearning(
                    case,
                    run_config,
                    physics=physics,
                    progress_callback=(
                        None
                        if progress_callback is None
                        else lambda message, slug=spec.slug, active_seed=seed: _notify(
                            progress_callback,
                            f"{slug}/seed-{active_seed}: {message}",
                        )
                    ),
                )
                payload = summary.to_dict()
                payload["paper_run"] = {
                    "schema": PAPER_ARTIFACT_SCHEMA,
                    "case_slug": spec.slug,
                    "replicate_index": replicate_index,
                    "seed": seed,
                    "structural_regime": spec.structural_regime,
                    "physics_pool_reused_across_seeds": True,
                }
                validate_paper_run_payload(
                    payload,
                    spec=spec,
                    config=run_config,
                    seed=seed,
                )
                _write_json(run_path, payload)
                _notify(progress_callback, f"wrote {run_path}")
            payloads.append(payload)

    tables = build_paper_tables(payloads, case_records=case_records)
    table_paths = write_paper_tables(output / "tables", tables)

    from .paper_plots import write_paper_figures
    from .paper_topology_plots import TopologyFigureCase, write_topology_figures

    figure_paths = write_paper_figures(output / "figures", payloads, tables)
    figure_paths.extend(write_topology_figures(
        output / "figures" / "topologies",
        [
            TopologyFigureCase(
                slug=spec.slug,
                structural_regime=spec.structural_regime,
                case=case,
            )
            for spec, case in loaded_cases
        ],
    ))
    summary = build_bundle_summary(
        payloads,
        case_records=case_records,
        base_config=config,
        seeds=normalized_seeds,
        table_paths=_relative_paths(output, table_paths),
        figure_paths=_relative_paths(output, figure_paths),
        started_at=started_at,
    )
    from .paper_prompts import write_histogram_llm_prompts
    from .paper_xai import write_decision_tree_artifacts

    tree_bundle = write_decision_tree_artifacts(
        output / "decision_trees",
        tables,
        source_schema=PAPER_ARTIFACT_SCHEMA,
        source_backend=PAPER_BACKEND,
    )
    prompt_bundle = write_histogram_llm_prompts(
        output / "llm_prompts",
        tables,
        tree_bundle=tree_bundle,
        source_schema=PAPER_ARTIFACT_SCHEMA,
        source_backend=PAPER_BACKEND,
        total_bandit_turns=int(summary["total_bandit_turns"]),
        turns_per_replicate=int(summary["turns_per_replicate"]),
    )
    summary["decision_tree_artifacts"] = {
        "tree_count": tree_bundle.manifest["tree_count"],
        "validation_scope": tree_bundle.manifest["validation_scope"],
        "generalization_claim": tree_bundle.manifest["generalization_claim"],
        "files": _relative_paths(output, tree_bundle.paths),
    }
    summary["llm_prompt_artifacts"] = {
        "prompt_count": prompt_bundle["manifest"]["prompt_count"],
        "contains_model_outputs": False,
        "files": _relative_paths(output, prompt_bundle["paths"]),
    }
    _write_json(output / "run_summary.json", summary)
    _write_report(output / "report.md", summary)
    artifact_manifest = write_artifact_manifest(output)
    _notify(
        progress_callback,
        f"paper bundle complete: {len(payloads)} runs, "
        f"{len(figure_paths)} figure files, "
        f"{tree_bundle.manifest['tree_count']} decision trees, "
        f"{prompt_bundle['manifest']['prompt_count']} LLM prompt inputs, {output}",
    )
    return {
        "output_dir": str(output),
        "run_summary": summary,
        "artifact_manifest": artifact_manifest,
    }


def validate_paper_run_payload(
    payload: dict[str, Any],
    *,
    spec: HistogramPaperCaseSpec,
    config: HistogramCoLearningConfig,
    seed: int,
) -> None:
    """Reject stale, partial, wrong-backend, or wrong-budget resume records."""

    errors: list[str] = []
    if payload.get("backend") != PAPER_BACKEND:
        errors.append(f"backend={payload.get('backend')!r}")
    if payload.get("deprecated_sequence_backend_executed") is not False:
        errors.append("deprecated_sequence_backend_executed is not false")
    if int(payload.get("turns", -1)) != int(config.turns):
        errors.append(f"turns={payload.get('turns')!r}")
    actual_config = payload.get("config") or {}
    # Older bundles predate this explicit field and used uniform exploration.
    normalized_actual_config = dict(actual_config)
    normalized_actual_config.setdefault("alice_learner_mode", "flat_exp3")
    normalized_actual_config.setdefault("eve_learner_mode", "flat_exp3")
    normalized_actual_config.setdefault("structured_exploration_mode", "uniform")
    normalized_actual_config.setdefault(
        "eve_structured_feature_mode",
        "family_target_topology",
    )
    normalized_actual_config.setdefault(
        "eve_structured_exploration_mode",
        "uniform",
    )
    # Preserve resumability of completed bundles written before the adaptive
    # independent-arm selectors existed. Missing fields mean their current
    # literature defaults; a requested nondefault still fails the exact match.
    adaptive_defaults = HistogramCoLearningConfig().to_dict()
    for key in (
        "tsallis_inf_estimator",
        "alice_tsallis_inf_estimator",
        "eve_tsallis_inf_estimator",
        "alice_tsallis_inf_eta_scale",
        "eve_tsallis_inf_eta_scale",
        "exp3pp_confidence_alpha",
        "alice_exp3pp_exploration_beta",
        "eve_exp3pp_exploration_beta",
        "alice_exp3pp_eta_scale",
        "eve_exp3pp_eta_scale",
    ):
        normalized_actual_config.setdefault(key, adaptive_defaults[key])
    if normalized_actual_config != config.to_dict():
        errors.append("co-learning configuration mismatch")
    if int(actual_config.get("seed", -1)) != int(seed):
        errors.append(f"seed={actual_config.get('seed')!r}")
    if int(actual_config.get("requested_pairs_per_turn", -1)) != int(
        config.requested_pairs_per_turn
    ):
        errors.append("requested_pairs_per_turn mismatch")
    case = payload.get("case") or {}
    if str(case.get("graph_id")) != spec.graph_id:
        errors.append(f"graph_id={case.get('graph_id')!r}")
    if str(case.get("assignment_id")) != spec.assignment_id:
        errors.append(f"assignment_id={case.get('assignment_id')!r}")
    paper = payload.get("paper_run") or {}
    if str(paper.get("case_slug")) != spec.slug:
        errors.append(f"case_slug={paper.get('case_slug')!r}")
    if int(paper.get("seed", -1)) != int(seed):
        errors.append(f"paper seed={paper.get('seed')!r}")
    if not (payload.get("empirical_game") or {}).get("all_cells_observed"):
        errors.append("empirical payoff matrix has unobserved cells")
    empirical = payload.get("empirical_game") or {}
    if config.posthoc_samples_per_cell > 0:
        if empirical.get("sampling_design") != "stratified_independent_posthoc":
            errors.append("stratified post-hoc payoff audit is missing")
        if int(empirical.get("posthoc_samples_per_cell", -1)) != int(
            config.posthoc_samples_per_cell
        ):
            errors.append("post-hoc payoff samples-per-cell mismatch")
    false_signal = payload.get("alice_false_signal_telemetry") or {}
    expected_attribution = {
        "flat_exp3": "fixed_behavior_exp3_reward_correction_v1",
        "tsallis_inf": "fixed_behavior_tsallis_inf_reward_replay_v1",
        "exp3_plus_plus": (
            "fixed_behavior_exp3_plus_plus_reward_replay_v1"
        ),
        "linear_route_exp2": "fixed_behavior_linear_exp2_reward_correction_v1",
        "dag_path_comband": "fixed_behavior_dag_comband_reward_correction_v1",
        "dag_path_exp2": "fixed_behavior_dag_path_exp2_reward_correction_v1",
        "maiti_dag_ftrl": "fixed_behavior_maiti_ftrl_reward_correction_v1",
        "ito_dag_ftrl": "fixed_behavior_ito_ftrl_reward_correction_v1",
    }[config.alice_learner_mode]
    if (false_signal.get("policy_attribution") or {}).get("method") != (
        expected_attribution
    ):
        errors.append("Alice false-signal policy attribution is missing")
    if errors:
        raise ValueError(
            f"invalid resumable paper run for {spec.slug}/seed-{seed}: "
            + "; ".join(errors)
        )


def case_structure(case: HistogramGameCase) -> dict[str, Any]:
    """Scalar route-overlap and distance descriptors for reporting."""

    routes = list(case.routes)
    edge_sets = [set(map(str, row.get("edge_ids") or [])) for row in routes]
    node_sets = [set(map(str, row.get("internal_nodes") or [])) for row in routes]
    pairs = [
        (left, right)
        for left in range(len(routes))
        for right in range(left + 1, len(routes))
    ]
    edge_disjoint = sum(not (edge_sets[left] & edge_sets[right]) for left, right in pairs)
    node_disjoint = sum(not (node_sets[left] & node_sets[right]) for left, right in pairs)
    lengths = [float(row["total_length_m"]) / 1000.0 for row in routes]
    hops = [int(row["hop_count"]) for row in routes]
    edge_lengths = [
        float(value) / 1000.0
        for row in routes
        for value in row.get("edge_lengths_m") or []
    ]
    denominator = len(pairs) or 1
    return {
        "family": case.family,
        "alice": case.alice,
        "bob": case.bob,
        "route_count": len(routes),
        "min_hops": min(hops),
        "maximum_route_depth": max(hops),
        "min_route_distance_km": min(lengths),
        "max_route_distance_km": max(lengths),
        "min_edge_length_km": min(edge_lengths),
        "max_edge_length_km": max(edge_lengths),
        "route_pair_count": len(pairs),
        "edge_disjoint_pair_fraction": edge_disjoint / denominator,
        "node_disjoint_pair_fraction": node_disjoint / denominator,
        "route_universe": dict(case.route_universe),
    }


def build_paper_tables(
    payloads: Sequence[dict[str, Any]],
    *,
    case_records: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build normalized, plotting-ready tables from completed run payloads."""

    if not payloads:
        raise ValueError("paper tables require at least one run payload")
    case_by_slug = {str(row["slug"]): dict(row) for row in case_records}
    runs: list[dict[str, Any]] = []
    learning: list[dict[str, Any]] = []
    family_strategy: list[dict[str, Any]] = []
    route_strategy: list[dict[str, Any]] = []
    route_calibration: list[dict[str, Any]] = []
    payoff_family_samples: list[dict[str, Any]] = []

    seen_calibrations: set[tuple[str, str]] = set()
    for payload in payloads:
        paper = payload["paper_run"]
        slug = str(paper["case_slug"])
        seed = int(paper["seed"])
        record = case_by_slug[slug]
        empirical = payload["empirical_game"]
        learned = empirical.get("learned_profile") or {}
        oracle = empirical.get("oracle") or {}
        false_signal = payload.get("alice_false_signal_telemetry") or {}
        false_signal_policy = false_signal.get("policy_attribution") or {}
        total = int(payload["turns"])
        outcomes = payload.get("outcome_counts") or {}
        run_row = {
            "case_slug": slug,
            "family": record["family"],
            "structural_regime": record["structural_regime"],
            "seed": seed,
            "turns": total,
            "route_count": int(payload["case"]["route_count"]),
            "action_count": len(payload["actions"]),
            "min_hops": record["min_hops"],
            "maximum_route_depth": record["maximum_route_depth"],
            "mean_alice_utility": float(payload["mean_alice_utility"]),
            "final_alice_utility": float(payload["final_alice_utility"]),
            "mean_bell_qualified_rate_hz": float(payload["mean_bell_qualified_rate_hz"]),
            "final_bell_qualified_rate_hz": float(payload["final_bell_qualified_rate_hz"]),
            "mean_eve_public_utility": float(payload["mean_eve_public_utility"]),
            "accepted_fraction": int(outcomes.get("accepted", 0)) / total,
            "delivery_failure_fraction": int(outcomes.get("delivery_failure", 0)) / total,
            "chsh_abort_fraction": int(outcomes.get("chsh_abort", 0)) / total,
            "qber_abort_fraction": int(outcomes.get("qber_abort", 0)) / total,
            "information_observation_rate": float(payload["information_observation_rate"]),
            "target_overlap_rate_diagnostic": float(payload["target_overlap_rate_diagnostic"]),
            "alice_false_signal_rate_all_turns": float(
                false_signal.get("false_signal_rate_all_turns", 0.0)
            ),
            "alice_false_signal_rate_classifiable": _optional_float(
                false_signal.get("false_signal_rate_classifiable_quality_signals")
            ),
            "alice_false_accept_rate_all_turns": float(
                false_signal.get("false_accept_rate_all_turns", 0.0)
            ),
            "alice_false_abort_rate_all_turns": float(
                false_signal.get("false_abort_rate_all_turns", 0.0)
            ),
            "alice_false_signal_reward_bias_per_turn": float(
                false_signal.get("corrected_minus_observed_reward_mean", 0.0)
            ),
            "alice_false_signal_policy_total_variation": float(
                false_signal_policy.get("policy_total_variation", 0.0)
            ),
            "learned_profile_utility": _optional_float(learned.get("alice_public_utility")),
            "oracle_alice_utility": _optional_float(oracle.get("alice_public_utility")),
            "exploitability": _optional_float(learned.get("zero_sum_equivalent_exploitability")),
            "minimum_cell_samples": int(empirical["minimum_cell_samples"]),
            "maximum_cell_standard_error": _optional_float(empirical.get("maximum_cell_standard_error")),
            "payoff_sampling_design": str(empirical.get("sampling_design")),
            "posthoc_samples_per_cell": int(empirical.get("posthoc_samples_per_cell", 0)),
            "posthoc_total_samples": int(empirical.get("posthoc_total_samples", 0)),
            "elapsed_seconds": float(payload["elapsed_seconds"]),
            "turns_per_second": float(payload["turns_per_second"]),
        }
        runs.append(run_row)

        equivalent = empirical.get("zero_sum_equivalent_matrix")
        for point in payload["learning_curve"]:
            gap = None
            if equivalent is not None:
                gap = exploitability(
                    np.asarray(equivalent, dtype=float),
                    np.asarray(point["alice_strategy"], dtype=float),
                    np.asarray(point["eve_strategy"], dtype=float),
                )
            learning.append({
                "case_slug": slug,
                "seed": seed,
                "turn": int(point["turn"]),
                "alice_utility_mean": float(point["alice_utility_mean"]),
                "bell_qualified_rate_hz_mean": float(point["bell_qualified_rate_hz_mean"]),
                "accepted_rate": float(point["accepted_rate"]),
                "posthoc_exploitability": gap,
                "alice_policy_entropy": float(point["alice_policy_entropy"]),
                "eve_policy_entropy": float(point["eve_policy_entropy"]),
                "alice_policy_minimum_action_probability": float(
                    point["alice_policy_minimum_action_probability"]
                ),
                "eve_policy_minimum_action_probability": float(
                    point["eve_policy_minimum_action_probability"]
                ),
                "alice_policy_uniform_coverage_mass": float(
                    point["alice_policy_uniform_coverage_mass"]
                ),
                "eve_policy_uniform_coverage_mass": float(
                    point["eve_policy_uniform_coverage_mass"]
                ),
                "alice_policy_nonmodal_probability_mass": float(
                    point["alice_policy_nonmodal_probability_mass"]
                ),
                "eve_policy_nonmodal_probability_mass": float(
                    point["eve_policy_nonmodal_probability_mass"]
                ),
                "alice_policy_effective_action_fraction": float(
                    point["alice_policy_effective_action_fraction"]
                ),
                "eve_policy_effective_action_fraction": float(
                    point["eve_policy_effective_action_fraction"]
                ),
                "alice_false_signal_rate_all_turns": float(
                    point.get("alice_false_signal_rate_all_turns", 0.0)
                ),
                "alice_false_signal_rate_classifiable": _optional_float(
                    point.get("alice_false_signal_rate_classifiable")
                ),
                "alice_false_accept_rate_all_turns": float(
                    point.get("alice_false_accept_rate_all_turns", 0.0)
                ),
                "alice_false_abort_rate_all_turns": float(
                    point.get("alice_false_abort_rate_all_turns", 0.0)
                ),
                "alice_false_signal_policy_total_variation": float(
                    point.get("alice_false_signal_policy_total_variation", 0.0)
                ),
            })

        actions = list(payload["actions"])
        family_indices = _family_indices(actions)
        eve_strategy = np.asarray(payload["eve_strategy"], dtype=float)
        eve_prior = np.asarray(payload["eve_prior"], dtype=float)
        oracle_eve = np.asarray(oracle.get("eve_strategy") or [], dtype=float)
        for family, indices in family_indices.items():
            family_strategy.append({
                "case_slug": slug,
                "seed": seed,
                "attack_family": family,
                "target_action_count": len(indices),
                "prior_mass": float(eve_prior[indices].sum()),
                "learned_mass": float(eve_strategy[indices].sum()),
                "oracle_mass": (
                    float(oracle_eve[indices].sum())
                    if oracle_eve.size else None
                ),
            })

        alice = np.asarray(payload["alice_strategy"], dtype=float)
        alice_current = np.asarray(payload["alice_current_strategy"], dtype=float)
        alice_prior = np.asarray(payload["alice_prior"], dtype=float)
        oracle_alice = np.asarray(oracle.get("alice_strategy") or [], dtype=float)
        corrected_alice = np.asarray(
            false_signal_policy.get("false_signal_corrected_current_policy") or [],
            dtype=float,
        )
        for route_index, calibration in enumerate(payload["route_calibrations"]):
            route_id = str(calibration["route_id"])
            route_strategy.append({
                "case_slug": slug,
                "seed": seed,
                "route_index": route_index,
                "route_id": route_id,
                "hop_count": int(calibration["hop_count"]),
                "distance_km": float(calibration["total_distance_km"]),
                "prior_mass": float(alice_prior[route_index]),
                "learned_mass": float(alice[route_index]),
                "factual_current_mass": float(alice_current[route_index]),
                "oracle_mass": (
                    float(oracle_alice[route_index])
                    if oracle_alice.size else None
                ),
                "false_signal_corrected_current_mass": (
                    float(corrected_alice[route_index])
                    if corrected_alice.size else None
                ),
            })
            calibration_key = (slug, route_id)
            if calibration_key not in seen_calibrations:
                seen_calibrations.add(calibration_key)
                route_calibration.append({
                    "case_slug": slug,
                    "route_index": route_index,
                    **calibration,
                    "min_edge_length_km": min(calibration["fiber_lengths_m"]) / 1000.0,
                    "max_edge_length_km": max(calibration["fiber_lengths_m"]) / 1000.0,
                })

        matrix = np.asarray(empirical["alice_public_utility_matrix"], dtype=float)
        for route_index in range(matrix.shape[0]):
            for family, indices in family_indices.items():
                payoff_family_samples.append({
                    "case_slug": slug,
                    "seed": seed,
                    "route_index": route_index,
                    "route_id": str(payload["case"]["route_ids"][route_index]),
                    "attack_family": family,
                    "uniform_target_mean_utility": float(np.mean(matrix[route_index, indices])),
                })

    cases = _aggregate_case_rows(runs, case_by_slug)
    payoff_family = _aggregate_rows(
        payoff_family_samples,
        keys=("case_slug", "route_index", "route_id", "attack_family"),
        value="uniform_target_mean_utility",
    )
    return {
        "run_summary": runs,
        "case_summary": cases,
        "learning_curve": learning,
        "eve_family_strategy": family_strategy,
        "alice_route_strategy": route_strategy,
        "route_calibration": route_calibration,
        "payoff_by_attack_family": payoff_family,
    }


def write_paper_tables(
    table_dir: str | Path,
    tables: dict[str, list[dict[str, Any]]],
) -> list[str]:
    output = Path(table_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for name, rows in tables.items():
        path = output / f"{name}.csv"
        _write_csv(path, rows)
        paths.append(str(path))
    return paths


def build_bundle_summary(
    payloads: Sequence[dict[str, Any]],
    *,
    case_records: Sequence[dict[str, Any]],
    base_config: HistogramCoLearningConfig,
    seeds: Sequence[int],
    table_paths: Sequence[str],
    figure_paths: Sequence[str],
    started_at: str,
) -> dict[str, Any]:
    tables = build_paper_tables(payloads, case_records=case_records)
    total_turns = sum(int(payload["turns"]) for payload in payloads)
    total_elapsed = sum(float(payload["elapsed_seconds"]) for payload in payloads)
    total_posthoc_samples = sum(
        int(payload["empirical_game"].get("posthoc_total_samples", 0))
        for payload in payloads
    )
    min_cells = min(
        int(payload["empirical_game"]["minimum_cell_samples"])
        for payload in payloads
    )
    finite_standard_errors = [
        float(value)
        for payload in payloads
        if (
            value := payload["empirical_game"].get(
                "maximum_cell_standard_error"
            )
        ) is not None
    ]
    max_se = max(finite_standard_errors, default=None)
    blockers = sorted({
        str(blocker)
        for payload in payloads
        for route in (payload.get("physics") or {}).get("routes", {}).values()
        for blocker in route.get("device_adherence_blockers", [])
    })
    return {
        "schema": PAPER_ARTIFACT_SCHEMA,
        "backend": PAPER_BACKEND,
        "runtime_engines": ["compiled_histogram_transcript_step"],
        "deprecated_sequence_backend_executed": False,
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "turns_per_replicate": int(base_config.turns),
        "replicate_seeds": list(seeds),
        "case_count": len(case_records),
        "run_count": len(payloads),
        "total_bandit_turns": total_turns,
        "total_posthoc_payoff_samples": total_posthoc_samples,
        "summed_run_elapsed_seconds": total_elapsed,
        "all_empirical_game_cells_observed": all(
            bool(payload["empirical_game"]["all_cells_observed"])
            for payload in payloads
        ),
        "minimum_payoff_cell_samples_across_runs": min_cells,
        "maximum_payoff_cell_standard_error_across_runs": max_se,
        "scientific_scope": _scientific_scope(),
        "case_manifest": list(case_records),
        "case_results": tables["case_summary"],
        "device_adherence_blockers": blockers,
        "tables": list(table_paths),
        "figures": list(figure_paths),
        "interpretation_boundary": (
            "A near-term-composite Alice service model is evaluated under "
            "explicitly modeled adversarial stress. The bundle is a "
            "routing/control experiment, not an authenticated finite-key E91 "
            "security proof."
        ),
        "posthoc_oracle_boundary": (
            "LP oracles and exploitability use the independent stratified "
            "post-learning empirical payoff matrix for validation only; neither "
            "learner sees these samples."
        ),
    }


def write_artifact_manifest(out_dir: str | Path) -> dict[str, Any]:
    root = Path(out_dir)
    manifest_path = root / "artifact_manifest.json"
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path == manifest_path:
            continue
        data = path.read_bytes()
        files.append({
            "path": str(path.relative_to(root)),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    payload = {
        "schema": PAPER_ARTIFACT_SCHEMA,
        "hash_algorithm": "sha256",
        "manifest_self_excluded": True,
        "file_count": len(files),
        "files": files,
    }
    _write_json(manifest_path, payload)
    return payload


def repository_provenance(
    *,
    ignore_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    root_text = run("rev-parse", "--show-toplevel")
    status_args = ["status", "--short"]
    if root_text and ignore_paths:
        root = Path(root_text).resolve()
        exclusions = []
        for ignored in ignore_paths:
            try:
                relative = Path(ignored).resolve().relative_to(root)
            except ValueError:
                continue
            exclusions.append(f":(exclude){relative.as_posix()}/**")
        if exclusions:
            status_args.extend(["--", ".", *exclusions])
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("branch", "--show-current"),
        "git_status_short": run(*status_args),
    }


def software_provenance() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def _aggregate_case_rows(
    runs: Sequence[dict[str, Any]],
    case_by_slug: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = (
        "mean_alice_utility",
        "final_alice_utility",
        "mean_bell_qualified_rate_hz",
        "final_bell_qualified_rate_hz",
        "accepted_fraction",
        "delivery_failure_fraction",
        "chsh_abort_fraction",
        "qber_abort_fraction",
        "information_observation_rate",
        "target_overlap_rate_diagnostic",
        "alice_false_signal_rate_all_turns",
        "alice_false_signal_rate_classifiable",
        "alice_false_accept_rate_all_turns",
        "alice_false_abort_rate_all_turns",
        "alice_false_signal_reward_bias_per_turn",
        "alice_false_signal_policy_total_variation",
        "oracle_alice_utility",
        "exploitability",
        "minimum_cell_samples",
        "maximum_cell_standard_error",
        "elapsed_seconds",
    )
    output = []
    for slug in case_by_slug:
        selected = [row for row in runs if row["case_slug"] == slug]
        record = case_by_slug[slug]
        row = {
            "case_slug": slug,
            "family": record["family"],
            "structural_regime": record["structural_regime"],
            "replicates": len(selected),
            "turns_per_replicate": selected[0]["turns"],
            "route_count": selected[0]["route_count"],
            "action_count": selected[0]["action_count"],
            "min_hops": record["min_hops"],
            "maximum_route_depth": record["maximum_route_depth"],
            "min_route_distance_km": record["min_route_distance_km"],
            "max_route_distance_km": record["max_route_distance_km"],
            "edge_disjoint_pair_fraction": record["edge_disjoint_pair_fraction"],
            "node_disjoint_pair_fraction": record["node_disjoint_pair_fraction"],
        }
        for metric in metrics:
            values = [float(item[metric]) for item in selected if item[metric] is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_sd"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            )
        output.append(row)
    return output


def _aggregate_rows(
    rows: Sequence[dict[str, Any]],
    *,
    keys: Sequence[str],
    value: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = {}
    for row in rows:
        groups.setdefault(tuple(row[key] for key in keys), []).append(float(row[value]))
    output = []
    for group, values in groups.items():
        output.append({
            **dict(zip(keys, group)),
            "replicates": len(values),
            f"{value}_mean": float(np.mean(values)),
            f"{value}_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        })
    return output


def _family_indices(actions: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, action in enumerate(actions):
        groups.setdefault(str(action.get("attack_type") or "unknown"), []).append(index)
    return groups


def _scientific_scope() -> dict[str, str]:
    return {
        "alice_network_physics": "empirical_fit",
        "alice_network_claim_tier": "near_term_composite",
        "repeater_profile": "cui_2026_rolling_purified_near_term_composite",
        "topology_and_link_lengths": "normalized_topology_bench_1_to_50_km_edges",
        "eve_attack_instruments": "modeled_attack",
        "eve_action_cost": "declared_modeled_intervention_cost",
        "service_utility": "derived_saturated_demand_control_payoff",
        "security_claim": "none",
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["case_results"]
    lines = [
        "# Histogram co-learning paper run (100k turns)",
        "",
        f"Backend: `{summary['backend']}`. Deprecated SeQUeNCe runtime executed: "
        f"`{summary['deprecated_sequence_backend_executed']}`.",
        "",
        f"This bundle contains {summary['run_count']} independent learner runs over "
        f"{summary['case_count']} fixed Topology Bench assignments, with "
        f"{summary['turns_per_replicate']:,} simultaneous Alice/Eve turns per replicate "
        f"({summary['total_bandit_turns']:,} total turns), followed by "
        f"{summary['total_posthoc_payoff_samples']:,} stratified payoff-audit draws "
        "that never update either learner.",
        "",
        "## Result overview",
        "",
        "| Case | Routes | Hops | Bell-qualified rate (pairs/s) | Alice utility | Exploitability | Accepted |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_slug']} | {row['route_count']} | "
            f"{row['min_hops']}--{row['maximum_route_depth']} | "
            f"{row['mean_bell_qualified_rate_hz_mean']:.4f} ± "
            f"{row['mean_bell_qualified_rate_hz_sd']:.4f} | "
            f"{row['mean_alice_utility_mean']:.4f} ± "
            f"{row['mean_alice_utility_sd']:.4f} | "
            f"{row['exploitability_mean']:.4f} ± "
            f"{row['exploitability_sd']:.4f} | "
            f"{row['accepted_fraction_mean']:.2%} |"
        )
    lines.extend([
        "",
        "Values are mean ± sample standard deviation across the declared seeds. "
        "Exploitability is a post-hoc validation diagnostic against each run's "
        "independently stratified empirical game; it is not feedback supplied to "
        "either learner.",
        "",
        "## Scientific boundary",
        "",
        summary["interpretation_boundary"],
        "",
        "Alice's route service uses the empirical-fit, near-term-composite Cui/Krutyanskiy "
        "rolling purified profile. Eve's instruments are modeled adversarial stressors and "
        "the action cost is a declared control parameter. The controller uses stationary "
        "saturated demand and does not model a "
        "stateful multi-flow queue, inventory depletion, authentication, privacy amplification, "
        "or a finite-key proof.",
        "",
        "## Interpretability and review artifacts",
        "",
        f"- {summary['decision_tree_artifacts']['tree_count']} compact decision trees "
        "are exported as JSON, rules text, DOT, PNG, and PDF under "
        "`decision_trees/`.",
        "- Their validation scope is descriptive and exploratory: only four fixed "
        "topology assignments were evaluated, so no unseen-network generalization "
        "is claimed.",
        f"- {summary['llm_prompt_artifacts']['prompt_count']} evidence-filled LLM "
        "prompt inputs are under `llm_prompts/`; the bundle contains no model outputs.",
        "- All interpretability inputs come from this histogram-backend run. The "
        "earlier SeQUeNCe-backed `exp3_dynamic_500k` artifacts are not used.",
        "- The `figures/topologies/` directory contains a four-case overview and "
        "one detailed full-topology/candidate-corridor figure per assignment, in "
        "PDF and PNG.",
        "",
        "## Audit checks",
        "",
        f"- All empirical payoff cells observed: `{summary['all_empirical_game_cells_observed']}`.",
        f"- Minimum samples in any payoff cell: `{summary['minimum_payoff_cell_samples_across_runs']}`.",
        f"- Maximum cell standard error: `{summary['maximum_payoff_cell_standard_error_across_runs']:.6f}`.",
        f"- Independent post-hoc payoff samples: `{summary['total_posthoc_payoff_samples']}`.",
        f"- Device-adherence blockers reported by selected backends: "
        f"`{summary['device_adherence_blockers']}`.",
        "- `artifact_manifest.json` supplies SHA-256 hashes for every other bundle file.",
        "",
        "## Reproduction",
        "",
        "```bash",
        ".venv/bin/python scripts/histogram_colearning_paper.py \\",
        "  --out-dir runs/histogram_colearning_paper_100k",
        "```",
        "",
        "The command is resumable: a run is reused only after backend, case, seed, turn "
        "budget, demand, and complete empirical-matrix checks pass.",
    ])
    _atomic_text(path, "\n".join(lines) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _relative_paths(root: Path, paths: Sequence[str]) -> list[str]:
    base = root.resolve()
    return [str(Path(path).resolve().relative_to(base)) for path in paths]


def _notify(callback: ProgressCallback | None, message: str) -> None:
    if callback is not None:
        callback(message)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
