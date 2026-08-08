#!/usr/bin/env python
"""Validation gates for the compiled E91 histogram backend.

This script is a reporting gate, not a new physics model.  It checks that the
compiled fixed-route histogram backend preserves its declared semantics:
normalization, compact/audit marginalization, compiled-vs-reference exactness,
waiting cap diagnostics, QBER observability, small-h SeQUeNCe reference
comparability, and h=10 compact-mode scalability.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.corpus.compiled_histogram_backend import (
    CompiledHistogramBackend,
    CompiledHistogramBackendConfig,
    CompiledHistogramStepResult,
    DEFAULT_CONVERSION_RESPONSE_PATH,
    DEFAULT_DETECTOR_PROFILE_PATH,
    load_histogram_generator,
)

REPORT_VERSION = "compiled_histogram_backend_validation.v1"
REPORT_SECTIONS = (
    "compiled_self_consistency",
    "compact_vs_audit_marginalization",
    "waiting_policy_and_cap_bias",
    "waiting_convergence",
    "protocol_modes",
    "qber_observability",
    "sequence_reference_agreement",
    "scalability_h10",
    "unsupported_physics_registry",
)
SEVERE_CAP_BIAS_CLASSES = {"severe_cap_bias", "catastrophic_cap_bias"}


def classify_cap_bias_factor(factor: float | None) -> str:
    if factor is None:
        return "not_applicable"
    if not math.isfinite(float(factor)):
        return "catastrophic_cap_bias"
    value = float(factor)
    if value <= 1.05:
        return "no_cap_bias"
    if value <= 2.0:
        return "mild_cap_bias"
    if value <= 10.0:
        return "moderate_cap_bias"
    if value <= 1e3:
        return "severe_cap_bias"
    return "catastrophic_cap_bias"


def compact_vs_audit_gate(tv_distance: float, tolerance: float) -> dict[str, Any]:
    return {
        "tv_distance": float(tv_distance),
        "tolerance": float(tolerance),
        "passed": float(tv_distance) <= float(tolerance),
    }


def cell_absolute_validation_allowed(cell: dict[str, Any]) -> bool:
    cap_class = str(
        cell.get("cap_bias_classification")
        or ((cell.get("waiting_cap_diagnostic") or {}).get("cap_bias_classification"))
        or "not_applicable"
    )
    return bool(cell.get("absolute_rate_comparable")) and cap_class not in SEVERE_CAP_BIAS_CLASSES


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _parse_nested_float_sequence(raw: str | None) -> tuple[tuple[float, ...], ...] | None:
    if raw is None or str(raw).strip() == "":
        return None
    return tuple(
        tuple(float(part.strip()) for part in row.split(",") if part.strip())
        for row in str(raw).split(";") if row.strip()
    )


def _backend_config(args: argparse.Namespace) -> CompiledHistogramBackendConfig:
    edge_points = _parse_nested_float_sequence(getattr(args, "rolling_edge_age_distribution_points_s", None))
    edge_weights = _parse_nested_float_sequence(getattr(args, "rolling_edge_age_distribution_weights", None))
    return CompiledHistogramBackendConfig(
        detector_kind=str(args.detector_kind),
        detector_profile_path=args.detector_profile,
        detector_response_path=args.detector_response_file,
        fiber_response_path=args.fiber_response_file,
        conversion_response_path=args.conversion_response_file,
        allow_wavelength_mismatch=bool(args.allow_wavelength_mismatch),
        allow_ideal_detector_reference=bool(args.allow_ideal_detector_reference),
        fock_cutoff=int(args.fock_cutoff),
        waiting_max_attempts=int(args.waiting_max_attempts),
        waiting_max_history_pairs=int(args.waiting_max_history_pairs),
        waiting_max_wait_signatures=int(args.waiting_max_wait_signatures),
        waiting_tail_tolerance=float(args.waiting_tail_tolerance),
        waiting_budget_mode=str(args.waiting_budget_mode),
        waiting_budget_wall_clock_ps=(
            float(args.waiting_budget_wall_clock_ps)
            if args.waiting_budget_wall_clock_ps is not None else None
        ),
        waiting_precision_mode=str(args.waiting_precision_mode),
        waiting_tv_tolerance=float(args.waiting_tv_tolerance),
        waiting_metric_tolerance=float(args.waiting_metric_tolerance),
        waiting_max_refinement_rounds=int(args.waiting_max_refinement_rounds),
        waiting_max_runtime_s=(
            float(args.waiting_max_runtime_s) if args.waiting_max_runtime_s is not None else None
        ),
        waiting_max_memory_mb=(
            float(args.waiting_max_memory_mb) if args.waiting_max_memory_mb is not None else None
        ),
        waiting_protocol=str(args.waiting_protocol),
        rolling_edge_availability=(
            tuple(float(part.strip()) for part in str(args.rolling_edge_availability).split(",") if part.strip())
            if args.rolling_edge_availability is not None else None
        ),
        rolling_age_distribution_kind=str(args.rolling_age_distribution_kind),
        rolling_age_distribution_points_s=tuple(
            float(part.strip()) for part in str(args.rolling_age_distribution_points_s).split(",") if part.strip()
        ),
        rolling_age_distribution_weights=tuple(
            float(part.strip()) for part in str(args.rolling_age_distribution_weights).split(",") if part.strip()
        ),
        rolling_edge_age_distribution_points_s=edge_points,
        rolling_edge_age_distribution_weights=edge_weights,
        rolling_age_evaluation=str(args.rolling_age_evaluation),
        rolling_age_support_product_limit=int(args.rolling_age_support_product_limit),
        rolling_accepted_alive_time_s=(
            float(args.rolling_accepted_alive_time_s)
            if args.rolling_accepted_alive_time_s is not None else None
        ),
        rolling_memory_slots_per_edge_side=int(args.rolling_memory_slots_per_edge_side),
        rolling_selection_policy=str(args.rolling_selection_policy),
        rolling_age_distribution_source=str(args.rolling_age_distribution_source),
        memory_cutoff_ps=(
            float(args.memory_cutoff_ps) if args.memory_cutoff_ps is not None else None
        ),
        compile_key_length=int(args.key_length),
        evaluated_cache_size=int(args.evaluated_cache_size),
        min_key_pairs_for_qber_abort=1,
    )


def _backend(args: argparse.Namespace) -> CompiledHistogramBackend:
    return CompiledHistogramBackend(_backend_config(args))


def _move(generator: Any, kind: str = "none") -> Any:
    if kind == "swap_block":
        return generator.EveMove(
            "swap_block:r1",
            "swap_block",
            target_kind="node",
            target_id="r1",
            target_active=True,
        )
    return generator.EveMove("no_move_baseline", "none", target_active=False)


def _cap_bias_diagnostic(
    backend: CompiledHistogramBackend,
    hop_count: int,
    lengths: Sequence[float],
    prepared: Any,
) -> dict[str, Any] | None:
    if hop_count < 2:
        return None
    if getattr(backend.config, "waiting_protocol", "single_buffer_swap_asap") == "rolling_alive_pair":
        return {
            "hop_count": int(hop_count),
            "fiber_lengths_m": [float(v) for v in lengths],
            "cap_bias_factor": None,
            "cap_bias_classification": "not_applicable",
            "reason": "rolling_alive_pair does not use route-from-scratch waiting history/signature caps",
        }
    generator = load_histogram_generator()
    backend._ensure_ready()  # validation report intentionally inspects frozen theta internals
    route = generator.make_route(int(hop_count), [float(v) for v in lengths])
    attempt_policy = generator.attempt_period_policy(route, backend._front, backend._context)
    edge_periods = {
        edge_id: float(attempt_policy["edge_attempt_periods"][edge_id]["attempt_period_ps"])
        for edge_id in route["edge_ids"]
    }
    budgets, budget_meta = generator.waiting_attempt_budgets(backend._context, edge_periods)
    edge_success = prepared.delivery_model.get("edge_success_probabilities") or {}
    if not edge_success:
        return None
    full_budget_ready = 1.0
    for edge_id in route["edge_ids"]:
        p = float(edge_success.get(edge_id, 0.0))
        k = int(budgets.get(edge_id, 0))
        full_budget_ready *= 1.0 - (1.0 - p) ** k if k > 0 else 0.0
    capped_ready = prepared.delivery_model.get("route_ready_probability_within_waiting_budget")
    factor = (
        full_budget_ready / capped_ready
        if capped_ready and capped_ready > 0.0 else
        (math.inf if full_budget_ready > 0.0 else None)
    )
    waiting_model = prepared.delivery_model.get("waiting_model") or {}
    waiting_distribution = waiting_model.get("waiting_distribution") or {}
    return {
        "hop_count": int(hop_count),
        "fiber_lengths_m": [float(v) for v in lengths],
        "full_budget_route_ready": full_budget_ready,
        "capped_route_ready": capped_ready,
        "cap_bias_factor": factor,
        "cap_bias_classification": classify_cap_bias_factor(factor),
        "waiting_precision_status": waiting_distribution.get("waiting_precision_status"),
        "compute_cap_omitted_route_ready_mass": waiting_distribution.get(
            "compute_cap_omitted_route_ready_mass"
        ),
        "configured_budget_physical_timeout_mass": waiting_distribution.get(
            "configured_budget_physical_timeout_mass"
        ),
        "transcript_tv_distance_bound_vs_configured_budget": waiting_distribution.get(
            "transcript_tv_distance_bound_vs_configured_budget"
        ),
        "attempt_budgets_full": {k: int(v) for k, v in budgets.items()},
        "waiting_budget_meta": _json_ready(budget_meta),
        "cap_reduction_metadata": {
            key: _json_ready(waiting_distribution.get(key))
            for key in (
                "attempt_cutoffs_reduced_below_budget",
                "history_pair_cap",
                "wait_signature_cap",
                "history_count",
                "wait_signature_count",
                "retained_history_mass",
                "omitted_tail_mass",
                "compute_cap_omitted_route_ready_mass",
                "configured_budget_physical_timeout_mass",
                "waiting_precision_status",
                "transcript_tv_distance_bound_vs_configured_budget",
                "tail_exceeds_tolerance",
            )
            if key in waiting_distribution
        },
    }


def _evaluated_array_memory_kib(prepared: Any) -> float:
    arrays = (
        "probabilities",
        "delivered_mask",
        "pair_index",
        "is_key_pair_row",
        "is_key_error_row",
        "is_chsh_row",
        "chsh_pair_index",
        "chsh_same_outcome",
        "eve_index",
        "no_delivery_stage_index",
    )
    nbytes = 0
    for name in arrays:
        value = getattr(prepared, name, None)
        if hasattr(value, "nbytes"):
            nbytes += int(value.nbytes)
    return nbytes / 1024.0


def compiled_self_consistency_report(backend: CompiledHistogramBackend, args: argparse.Namespace) -> dict[str, Any]:
    generator = load_histogram_generator()
    move = _move(generator)
    checks = []
    vectors = {
        1: [[100.0], [125.0]],
        2: [[100.0, 150.0], [80.0, 160.0]],
        3: [[100.0, 150.0, 120.0]],
        5: [[100.0, 120.0, 140.0, 160.0, 180.0]],
    }
    for hop_count, length_vectors in vectors.items():
        if hop_count == 5 and args.skip_h5_reference:
            checks.append({"hop_count": hop_count, "status": "skipped_by_cli", "passed": False})
            continue
        started = time.perf_counter()
        try:
            report = backend.reference_check(hop_count, move, length_vectors, tolerance=args.tv_tolerance)
            report["wall_seconds"] = time.perf_counter() - started
            report["hop_count"] = hop_count
            checks.append(report)
        except Exception as exc:  # validation artifact records, not hides, failures
            checks.append({
                "hop_count": hop_count,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
                "wall_seconds": time.perf_counter() - started,
            })
    return {
        "description": "CompiledRouteKernel TV distance against FrozenHistogramKernel.evaluate.",
        "required_hop_counts": [1, 2, 3],
        "selected_h5_case_required": not args.skip_h5_reference,
        "asymmetric_length_vectors_included": True,
        "checks": checks,
        "passed": all(check.get("passed") for check in checks if check.get("status") != "skipped_by_cli"),
    }


def compact_vs_audit_report(args: argparse.Namespace) -> dict[str, Any]:
    generator = load_histogram_generator()
    front = generator.load_or_build_front_matter(Path(args.hardware_config_dir))
    context = generator.ModelContext(
        physical_path="fock_pnr_snspd",
        source_model="memory_emission",
        detector_kind=str(args.detector_kind),
        fock_cutoff=int(args.fock_cutoff),
        detector_profile_path=args.detector_profile,
        detector_profile=generator.load_json_optional(args.detector_profile),
        detector_response=generator.load_json_optional(args.detector_response_file),
        fiber_response=generator.load_json_optional(args.fiber_response_file),
        conversion_response=generator.load_json_optional(args.conversion_response_file),
        allow_wavelength_mismatch=bool(args.allow_wavelength_mismatch),
        waiting_max_attempts=max(1, min(int(args.waiting_max_attempts), 4)),
        waiting_max_history_pairs=min(int(args.waiting_max_history_pairs), 4096),
        waiting_max_wait_signatures=min(int(args.waiting_max_wait_signatures), 1024),
    )
    move = _move(generator)
    route = generator.make_route(2, [100.0, 150.0])
    production = generator.generate_histogram_case(
        route=route,
        front=front,
        attack=move,
        key_length=int(args.key_length),
        context=context,
    )
    audit = generator.generate_histogram_case(
        route=route,
        front=front,
        attack=move,
        key_length=int(args.key_length),
        context=context,
        emit_detector_click_audit=True,
    )
    projected = generator.rows_with_detector_click_marginalized(audit["rows"], int(args.key_length))
    tv = generator.total_variation_distance(production["rows"], projected)
    prod_metrics = production["metadata"]["expected_metrics"]
    proj_metrics = generator.expected_metrics(projected, generator.ProtocolConfig(requested_pairs=int(args.key_length)))
    invariants = {
        "delivery_probability_delta": generator.optional_delta(
            prod_metrics.get("delivery_probability"), proj_metrics.get("delivery_probability")
        ),
        "expected_qber_delta": generator.optional_delta(
            prod_metrics.get("expected_qber"), proj_metrics.get("expected_qber")
        ),
        "expected_chsh_S_delta": generator.optional_delta(
            prod_metrics.get("expected_chsh_S"), proj_metrics.get("expected_chsh_S")
        ),
        "eve_guess_probability_delta": generator.optional_delta(
            prod_metrics.get("eve_guess_probability"), proj_metrics.get("eve_guess_probability")
        ),
    }
    gate = compact_vs_audit_gate(tv, args.tv_tolerance)
    return {
        "description": "Raw detector records condition branch states, then are marginalized from production rows.",
        "production_row_count": len(production["rows"]),
        "audit_row_count": len(audit["rows"]),
        "projected_audit_row_count": len(projected),
        "tv_gate": gate,
        "metric_invariance_deltas": invariants,
        "passed": gate["passed"] and all(
            value is None or abs(float(value)) <= args.tv_tolerance
            for value in invariants.values()
        ),
    }


def waiting_policy_and_cap_bias_report(
    backend: CompiledHistogramBackend,
    args: argparse.Namespace,
) -> dict[str, Any]:
    generator = load_histogram_generator()
    move = _move(generator)
    representative = {
        3: [100.0, 120.0, 140.0],
        5: [100.0, 120.0, 140.0, 160.0, 180.0],
        10: [100.0 + 5.0 * i for i in range(10)],
    }
    if args.waiting_protocol == "rolling_alive_pair":
        cases = []
        for hop_count, lengths in representative.items():
            started = time.perf_counter()
            prepared = backend.evaluate(hop_count, move, lengths)
            rolling_model = prepared.delivery_model.get("rolling_live_pair_model") or {}
            cases.append({
                "hop_count": hop_count,
                "fiber_lengths_m": lengths,
                "row_count": int(prepared.row_count),
                "delivery_probability": prepared.expected_metrics.get("delivery_probability"),
                "route_availability": rolling_model.get("route_availability"),
                "wall_seconds": time.perf_counter() - started,
                "cap_bias_diagnostic": _cap_bias_diagnostic(backend, hop_count, lengths, prepared),
                "cap_bias_classification": "not_applicable",
            })
        return {
            "description": "Cap-bias diagnostics are not applicable to rolling_alive_pair.",
            "budget_semantics": (
                "rolling_alive_pair uses configured live-pair availability and selected-age distributions; "
                "it does not enumerate route-from-scratch waiting histories"
            ),
            "sweeps": cases,
            "all_h_ge_2_have_cap_bias_factor": False,
            "severe_or_catastrophic_entries": [],
            "passed": True,
        }
    sweeps = []
    for hop_count, lengths in representative.items():
        for history_pairs, signatures in (
            (64, 8),
            (256, 16),
        ):
            cfg_args = argparse.Namespace(**vars(args))
            cfg_args.waiting_max_history_pairs = history_pairs
            cfg_args.waiting_max_wait_signatures = signatures
            cfg_args.evaluated_cache_size = 8
            local_backend = _backend(cfg_args)
            started = time.perf_counter()
            prepared = local_backend.evaluate(hop_count, move, lengths)
            diagnostic = _cap_bias_diagnostic(local_backend, hop_count, lengths, prepared)
            sweeps.append({
                "hop_count": hop_count,
                "waiting_max_history_pairs": history_pairs,
                "waiting_max_wait_signatures": signatures,
                "row_count": int(prepared.row_count),
                "delivery_probability": prepared.expected_metrics.get("delivery_probability"),
                "wall_seconds": time.perf_counter() - started,
                "cap_bias_diagnostic": diagnostic,
                "cap_bias_factor_emitted": diagnostic is not None and diagnostic.get("cap_bias_factor") is not None,
                "cap_bias_classification": (
                    diagnostic.get("cap_bias_classification") if diagnostic else "missing"
                ),
            })
    h_ge_2 = [entry for entry in sweeps if entry["hop_count"] >= 2]
    return {
        "description": "Representative convergence sweep over history-pair/signature caps.",
        "budget_semantics": (
            "fixed attempts or wall-clock budget as configured; h>=3 may be cap-limited by "
            "waiting_max_history_pairs/waiting_max_wait_signatures"
        ),
        "sweeps": sweeps,
        "all_h_ge_2_have_cap_bias_factor": all(entry["cap_bias_factor_emitted"] for entry in h_ge_2),
        "severe_or_catastrophic_entries": [
            entry for entry in sweeps
            if entry["cap_bias_classification"] in SEVERE_CAP_BIAS_CLASSES
        ],
        "passed": all(entry["cap_bias_factor_emitted"] for entry in h_ge_2),
    }


def _waiting_convergence_status(prepared: CompiledHistogramStepResult | Any) -> str:
    cert = getattr(prepared, "waiting_metric_certification", {}) or {}
    delivery = getattr(prepared, "delivery_model", {}) or {}
    if delivery.get("rolling_live_pair_model"):
        return "waiting_model_converged"
    waiting_distribution = ((delivery.get("waiting_model") or {}).get("waiting_distribution") or {})
    precision_status = str(waiting_distribution.get("waiting_precision_status") or "")
    metric_status = str(cert.get("metric_certification_status") or cert.get("status") or "")
    if precision_status in {"waiting_model_bounded", "converged_within_configured_attempt_budget"}:
        if metric_status == "certified":
            return "waiting_model_certified"
        return "waiting_model_bounded_but_metric_uncertified"
    if precision_status == "waiting_model_compute_cap_limited":
        return "waiting_model_compute_cap_limited"
    if precision_status == "waiting_model_requires_user_decision":
        return "waiting_model_requires_user_decision"
    if "unsupported" in precision_status:
        return "waiting_model_unsupported_for_active_features"
    return "waiting_model_requires_user_decision"


def waiting_convergence_report(
    backend: CompiledHistogramBackend,
    args: argparse.Namespace,
) -> dict[str, Any]:
    generator = load_histogram_generator()
    move = _move(generator)
    representative = {
        3: [100.0, 120.0, 140.0],
        5: [100.0, 120.0, 140.0, 160.0, 180.0],
        10: [100.0 + 5.0 * i for i in range(10)],
    }
    cases = []
    for hop_count, lengths in representative.items():
        started = time.perf_counter()
        try:
            prepared = backend.evaluate(hop_count, move, lengths)
            elapsed = time.perf_counter() - started
            delivery = prepared.delivery_model
            waiting_model = delivery.get("waiting_model") or {}
            waiting_distribution = waiting_model.get("waiting_distribution") or {}
            rolling_model = delivery.get("rolling_live_pair_model") or {}
            cert = prepared.waiting_metric_certification
            cases.append({
                "hop_count": hop_count,
                "fiber_lengths_m": lengths,
                "waiting_precision_mode": args.waiting_precision_mode,
                "method_name": (
                    rolling_model.get("status")
                    or waiting_distribution.get("status")
                ),
                "exact_bounded_approximate_status": (
                    "exact"
                    if rolling_model else
                    "bounded" if waiting_distribution.get("compute_cap_omitted_route_ready_mass", 0.0) else "exact"
                ),
                "waiting_protocol": delivery.get("waiting_protocol"),
                "route_availability": rolling_model.get("route_availability"),
                "selected_age_distribution_summary": rolling_model.get("selected_age_distribution"),
                "refinement_rounds": waiting_distribution.get("adaptive_refinement_rounds", []),
                "retained_route_ready_probability": waiting_distribution.get(
                    "route_ready_probability_within_budget"
                ),
                "configured_route_ready_probability": waiting_distribution.get(
                    "configured_attempt_budget_route_ready_probability"
                ),
                "omitted_route_ready_mass": waiting_distribution.get(
                    "compute_cap_omitted_route_ready_mass"
                ),
                "unconditional_tv_bound": cert.get("unconditional_tv_bound"),
                "delivery_probability": prepared.expected_metrics.get("delivery_probability"),
                "expected_qber": prepared.expected_metrics.get("expected_qber_from_histogram"),
                "expected_chsh_S": prepared.expected_metrics.get("expected_chsh_S"),
                "eve_guess_probability": prepared.expected_metrics.get("eve_guess_probability"),
                "metric_certification_status": cert.get("metric_certification_status"),
                "waiting_convergence_status": _waiting_convergence_status(prepared),
                "runtime_seconds": elapsed,
                "array_memory_kib": _evaluated_array_memory_kib(prepared),
            })
        except NotImplementedError as exc:
            cases.append({
                "hop_count": hop_count,
                "fiber_lengths_m": lengths,
                "waiting_convergence_status": "waiting_model_unsupported_for_active_features",
                "error": str(exc),
                "runtime_seconds": time.perf_counter() - started,
            })
        except Exception as exc:
            cases.append({
                "hop_count": hop_count,
                "fiber_lengths_m": lengths,
                "waiting_convergence_status": "waiting_model_requires_user_decision",
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_seconds": time.perf_counter() - started,
            })
    converged = all(case.get("waiting_convergence_status") == "waiting_model_converged" for case in cases)
    certified = all(case.get("waiting_convergence_status") == "waiting_model_certified" for case in cases)
    bounded = all(
        case.get("waiting_convergence_status") in {
            "waiting_model_converged",
            "waiting_model_certified",
            "waiting_model_bounded_but_metric_uncertified",
        }
        for case in cases
    )
    if converged:
        status = "waiting_model_converged"
    elif certified:
        status = "waiting_model_certified"
    elif bounded:
        status = "waiting_model_bounded_but_metric_uncertified"
    elif any(case.get("waiting_convergence_status") == "waiting_model_compute_cap_limited" for case in cases):
        status = "waiting_model_compute_cap_limited"
    elif any(case.get("waiting_convergence_status") == "waiting_model_unsupported_for_active_features" for case in cases):
        status = "waiting_model_unsupported_for_active_features"
    else:
        status = "waiting_model_requires_user_decision"
    return {
        "description": "h=3/h=5/h=10 waiting-support convergence and metric-certification diagnostics.",
        "waiting_precision_mode": args.waiting_precision_mode,
        "cases": cases,
        "waiting_convergence_status": status,
        "passed": status in {
            "waiting_model_converged",
            "waiting_model_certified",
            "waiting_model_bounded_but_metric_uncertified",
        },
    }


def protocol_modes_report(
    backend: CompiledHistogramBackend,
    args: argparse.Namespace,
) -> dict[str, Any]:
    generator = load_histogram_generator()
    move = _move(generator)
    representative = {
        3: [100.0, 120.0, 140.0],
        5: [100.0, 120.0, 140.0, 160.0, 180.0],
        10: [100.0 + 5.0 * i for i in range(10)],
    }
    rolling_cases = []
    for hop_count, lengths in representative.items():
        edge_points = tuple((0.0, 1e-6 * (index + 1)) for index in range(hop_count))
        edge_weights = tuple((0.5, 0.5) for _index in range(hop_count))
        rolling_config = replace(
            backend.config,
            waiting_protocol="rolling_alive_pair",
            rolling_edge_availability=tuple(1.0 for _index in range(hop_count)),
            rolling_age_distribution_kind="discrete",
            rolling_age_distribution_points_s=(0.0,),
            rolling_age_distribution_weights=(1.0,),
            rolling_edge_age_distribution_points_s=edge_points,
            rolling_edge_age_distribution_weights=edge_weights,
            rolling_age_evaluation="moment_reduced",
            rolling_age_support_product_limit=1_000_000,
            rolling_accepted_alive_time_s=hop_count * 1e-6,
            rolling_age_distribution_source="configured_per_edge_validation_smoke",
            rolling_selection_policy="configured_distribution",
        )
        rolling_backend = CompiledHistogramBackend(rolling_config)
        started = time.perf_counter()
        try:
            prepared = rolling_backend.evaluate(hop_count, move, lengths)
            elapsed = time.perf_counter() - started
            delivery = prepared.delivery_model
            rolling_model = delivery.get("rolling_live_pair_model") or {}
            rolling_cases.append({
                "hop_count": hop_count,
                "fiber_lengths_m": lengths,
                "status": "exact_configured_age_distribution",
                "runtime_seconds": elapsed,
                "row_count": prepared.row_count,
                "route_availability": rolling_model.get("route_availability"),
                "edge_availabilities": (rolling_model.get("availability") or {}).get("edge_availability"),
                "heterogeneous_age_distributions_supported": rolling_model.get(
                    "heterogeneous_age_distributions_supported"
                ),
                "age_support_product": rolling_model.get("age_support_product"),
                "selected_age_distribution_summary": rolling_model.get("selected_age_distribution"),
                "delivery_probability": prepared.expected_metrics.get("delivery_probability"),
                "expected_qber": prepared.expected_metrics.get("expected_qber_from_histogram"),
                "expected_chsh_S": prepared.expected_metrics.get("expected_chsh_S"),
                "eve_guess_probability": prepared.expected_metrics.get("eve_guess_probability"),
                "waiting_cap_diagnostics_used": bool(delivery.get("waiting_model")),
                "probability_sum": float(prepared.probabilities.sum()),
            })
        except Exception as exc:
            rolling_cases.append({
                "hop_count": hop_count,
                "fiber_lengths_m": lengths,
                "status": "rolling_alive_pair_evaluation_failed",
                "runtime_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            })
    rolling_passed = all(
        case.get("status") == "exact_configured_age_distribution"
        and not case.get("waiting_cap_diagnostics_used", True)
        and abs(float(case.get("probability_sum", 0.0)) - 1.0) <= 1e-9
        for case in rolling_cases
    )
    return {
        "single_buffer_swap_asap": {
            "status": "existing finite-budget route-from-scratch protocol",
            "selected_for_main_validation": args.waiting_protocol == "single_buffer_swap_asap",
            "precision_mode": args.waiting_precision_mode,
            "note": (
                "h>=3 may be bounded, certified, or compute-cap-limited depending on the waiting "
                "convergence section; this protocol is not equivalent to rolling_alive_pair"
            ),
        },
        "rolling_alive_pair": {
            "status": "exact_configured_age_distribution" if rolling_passed else "rolling_alive_pair_smoke_failed",
            "selected_for_main_validation": args.waiting_protocol == "rolling_alive_pair",
            "protocol_abstraction": (
                "route request consumes configured live elementary pairs; A_i and f_i(t) are supplied "
                "inputs, not derived from event-level memory-pool queueing"
            ),
            "pre_route_decay_and_stale_semantics": (
                "baked into configured availability and selected-age distribution"
            ),
            "cases": rolling_cases,
            "passed": rolling_passed,
        },
    }


def qber_observability_report(backend: CompiledHistogramBackend) -> dict[str, Any]:
    generator = load_histogram_generator()
    move = _move(generator)
    prepared = backend.evaluate(2, move, [100.0, 150.0])
    metrics = prepared.expected_metrics
    key_mass = float(metrics.get("qber_key_probability_mass", 0.0))
    expected_qber = metrics.get("expected_qber_from_histogram")
    schema_fields = {field.name for field in fields(CompiledHistogramStepResult)}
    required_step_fields = {
        "expected_qber_from_histogram",
        "sampled_qber_mean",
        "sampled_qber_trials",
        "sampled_qber_observable",
        "qber_key_mass_expected",
        "qber_estimation_power",
    }
    return {
        "key_mass": key_mass,
        "expected_qber_from_histogram": expected_qber,
        "qber_numerator_probability": metrics.get("qber_numerator_probability"),
        "qber_denominator_probability": metrics.get("qber_denominator_probability"),
        "histogram_gate": {
            "expected_qber_present_when_key_mass_positive": (
                expected_qber is not None if key_mass > 0.0 else expected_qber is None
            ),
            "expected_qber_in_unit_interval": (
                expected_qber is None or 0.0 <= float(expected_qber) <= 1.0
            ),
        },
        "finite_sample_reporting_schema": {
            "required_fields": sorted(required_step_fields),
            "present": sorted(required_step_fields & schema_fields),
            "missing": sorted(required_step_fields - schema_fields),
        },
        "directional_agreement_policy": (
            "comparison harness excludes sampled_qber_mean direction checks when either side has "
            "qber_estimation_power != 'adequate'"
        ),
        "passed": (
            (expected_qber is not None if key_mass > 0.0 else expected_qber is None)
            and (expected_qber is None or 0.0 <= float(expected_qber) <= 1.0)
            and not (required_step_fields - schema_fields)
        ),
    }


def _load_compare_module() -> Any:
    path = REPO_ROOT / "scripts" / "compare_backend_physics.py"
    spec = importlib.util.spec_from_file_location("compare_backend_physics_validation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sequence_reference_agreement_report(args: argparse.Namespace) -> dict[str, Any]:
    if not args.run_sequence_reference:
        return {
            "status": "not_run",
            "passed": False,
            "reason": "use --run-sequence-reference to run the small-h SeQUeNCe fixed-route reference",
            "required_policy": (
                "only cells with absolute_rate_comparable=true and non-severe cap bias count "
                "toward absolute agreement; qualitative cross-family checks are separate"
            ),
        }
    compare = _load_compare_module()
    cmp_args = compare.build_arg_parser().parse_args([
        "--graphs", "braess",
        "--max-routes", str(args.sequence_max_routes),
        "--comparison-mode", "fixed_route_repeater_chain",
        "--sequence-trials", str(args.sequence_trials),
        "--compiled-trials", str(args.compiled_trials),
        "--memory-pairs", str(args.sequence_memory_pairs),
        "--repeater-window-ms", str(args.sequence_repeater_window_ms),
        "--signature-cap", str(args.waiting_max_wait_signatures),
        "--history-pair-cap", str(args.waiting_max_history_pairs),
    ])
    payload = compare.run_comparison(cmp_args)
    cells = payload["cells"]
    absolute_cells = [cell for cell in cells if cell_absolute_validation_allowed(cell)]
    unsupported = [
        {
            "label": cell["label"],
            "route_path": cell["route_path"],
            "reason": cell.get("reason_if_not_comparable"),
        }
        for cell in cells
        if (cell.get("action_semantics") or {}).get("status") == "unsupported_sequence_reference"
    ]
    return {
        "status": "run",
        "comparison_mode": "fixed_route_repeater_chain",
        "cell_count": len(cells),
        "absolute_valid_cell_count_after_cap_filter": len(absolute_cells),
        "agreement": payload["directional_agreement"],
        "action_semantics_summary": {
            cell["label"]: (cell.get("action_semantics") or {}).get("status")
            for cell in cells
        },
        "unsupported_actions": unsupported,
        "artifact_excerpt": {
            "comparison_policy": payload["comparison_policy"].get("comparison_family_policy"),
            "settings": payload["settings"],
        },
        "passed": len(cells) > 0 and all(
            "comparison_family" in cell
            and "absolute_rate_comparable" in cell
            and "action_semantics" in cell
            for cell in cells
        ),
    }


def scalability_h10_report(args: argparse.Namespace) -> dict[str, Any]:
    generator = load_histogram_generator()
    cfg_args = argparse.Namespace(**vars(args))
    cfg_args.waiting_max_attempts = min(int(args.waiting_max_attempts), int(args.h10_max_attempts))
    cfg_args.waiting_max_history_pairs = min(int(args.waiting_max_history_pairs), int(args.h10_history_pair_cap))
    cfg_args.waiting_max_wait_signatures = min(
        int(args.waiting_max_wait_signatures), int(args.h10_signature_cap)
    )
    cfg_args.evaluated_cache_size = max(2, int(args.evaluated_cache_size))
    backend = _backend(cfg_args)
    move = _move(generator)
    lengths = [100.0 + 5.0 * i for i in range(10)]
    cold_started = time.perf_counter()
    prepared = backend.evaluate(10, move, lengths)
    cold_wall = time.perf_counter() - cold_started
    hit_started = time.perf_counter()
    cached = backend.evaluate(10, move, lengths)
    hit_wall = time.perf_counter() - hit_started
    diagnostic = _cap_bias_diagnostic(backend, 10, lengths, prepared)
    stats = backend.diagnostics()["stats"]
    status = "physically_converged"
    if prepared.delivery_model.get("rolling_live_pair_model"):
        status = "exact_configured_age_distribution"
    elif diagnostic and diagnostic.get("cap_bias_classification") in SEVERE_CAP_BIAS_CLASSES:
        status = "cap_limited"
    else:
        route_ready = prepared.delivery_model.get("route_ready_probability_within_waiting_budget")
        if route_ready is not None and float(route_ready) < 1.0:
            status = "deadline_limited"
    return {
        "hop_count": 10,
        "compact_mode": True,
        "audit_schema_required_to_scale": False,
        "fiber_lengths_m": lengths,
        "row_count": int(prepared.row_count),
        "cached_row_count": int(cached.row_count),
        "probability_sum": float(prepared.probabilities.sum()),
        "array_memory_kib": _evaluated_array_memory_kib(prepared),
        "cold_evaluate_wall_seconds": cold_wall,
        "cache_hit_wall_seconds": hit_wall,
        "cache_stats": stats,
        "cache_workflow": (
            "CompiledHistogramBackend uses a bounded LRU evaluated-histogram cache; this is "
            "ring-buffer-like for the paper pipeline, where old route/action entries are evicted "
            "and recompiled/re-evaluated if Alice revisits them after eviction"
        ),
        "cap_bias_diagnostic": diagnostic,
        "result_status": status,
        "passed": (
            abs(float(prepared.probabilities.sum()) - 1.0) <= 1e-9
            and int(prepared.row_count) <= int(args.h10_max_rows)
        ),
    }


def unsupported_physics_registry() -> dict[str, Any]:
    return {
        "physical_repeater_bsm_instrument": {
            "status": "generic_approximation",
            "notes": "compiled path uses ideal Bell projection plus configured scalar swap success/degradation/confusion channels, not a hardware BSM instrument",
        },
        "detector_dead_time_dynamics": {
            "status": "missing",
            "notes": "single-window detector POVM includes efficiency/darks; repeated-window dead-time correlations are not modeled",
        },
        "HOM_wave_packet_model": {
            "status": "missing",
            "notes": "no wave-packet overlap, spectral/temporal distinguishability, or HOM visibility model",
        },
        "polarization_drift_noise": {
            "status": "missing",
            "notes": "fiber polarization_fidelity is not a calibrated photonic polarization channel in the Fock/SNSPD kernel",
        },
        "calibrated_PNR_confusion_matrix": {
            "status": "hook_present_profile_absent_by_default",
            "notes": "profiles may supply pnr_confusion_matrix; Colangelo profile currently uses generic saturated binomial+dark model unless calibrated matrix is provided",
        },
        "endpoint_readout_analysis_rotation_errors": {
            "status": "typed_generic_model_if_supplied",
            "notes": "endpoint readout and analysis rotation models are explicit typed channels when configured; calibrated experimental matrices are not guaranteed",
        },
        "SPDC_multipair_conditioning": {
            "status": "missing_or_fenced",
            "notes": "memory_emission is the scoped production source model; SPDC multipair conditioning is not implemented as the main backend",
        },
        "SeQUeNCe_event_level_scheduling_equivalence": {
            "status": "missing",
            "notes": "compiled backend is event-free analytic; fixed-route SeQUeNCe reference is a small-h validation reference, not event-level equivalence for production h=10+",
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    backend = _backend(args)
    report = {
        "validation_report_version": REPORT_VERSION,
        "claim_under_test": (
            "compiled backend can be called a precise fixed-route quantum route simulation "
            "only inside the explicitly supported Fock/SNSPD memory-emission domain and only "
            "when all gates below pass without severe cap bias"
        ),
        "backend_config": _backend_config(args).to_dict(),
        "acceptance_sections": list(REPORT_SECTIONS),
    }
    sections = {
        "compiled_self_consistency": compiled_self_consistency_report(backend, args),
        "compact_vs_audit_marginalization": compact_vs_audit_report(args),
        "waiting_policy_and_cap_bias": waiting_policy_and_cap_bias_report(backend, args),
        "waiting_convergence": waiting_convergence_report(backend, args),
        "protocol_modes": protocol_modes_report(backend, args),
        "qber_observability": qber_observability_report(backend),
        "sequence_reference_agreement": sequence_reference_agreement_report(args),
        "scalability_h10": scalability_h10_report(args),
        "unsupported_physics_registry": unsupported_physics_registry(),
    }
    report.update(sections)
    hard_gate_sections = [
        "compiled_self_consistency",
        "compact_vs_audit_marginalization",
        "waiting_policy_and_cap_bias",
        "waiting_convergence",
        "qber_observability",
        "scalability_h10",
    ]
    hard_gates_passed = all(sections[name].get("passed", False) for name in hard_gate_sections)
    cap_limited_sweeps = sections["waiting_policy_and_cap_bias"].get("severe_or_catastrophic_entries", [])
    sequence_status = sections["sequence_reference_agreement"].get("status")
    sequence_ran_and_passed = (
        sequence_status == "run"
        and sections["sequence_reference_agreement"].get("passed", False)
    )
    waiting_model_gate_status = "waiting_model_requires_user_decision"
    if sections["waiting_policy_and_cap_bias"].get("passed", False):
        convergence_status = sections["waiting_convergence"].get("waiting_convergence_status")
        if convergence_status in {
            "waiting_model_converged",
            "waiting_model_certified",
            "waiting_model_bounded_but_metric_uncertified",
            "waiting_model_compute_cap_limited",
            "waiting_model_requires_user_decision",
            "waiting_model_unsupported_for_active_features",
        }:
            waiting_model_gate_status = convergence_status
        elif cap_limited_sweeps or sections["scalability_h10"].get("result_status") == "cap_limited":
            waiting_model_gate_status = "waiting_model_compute_cap_limited"
        else:
            sweep_diagnostics = [
                entry.get("cap_bias_diagnostic")
                for entry in sections["waiting_policy_and_cap_bias"].get("sweeps", [])
                if entry.get("cap_bias_diagnostic") is not None
            ]
            has_positive_bound = any(
                float(diag.get("transcript_tv_distance_bound_vs_configured_budget") or 0.0) > 0.0
                for diag in sweep_diagnostics
            )
            waiting_model_gate_status = (
                "waiting_model_bounded" if has_positive_bound else "waiting_model_converged"
            )
    precision_claim_gate_passed = (
        hard_gates_passed
        and sequence_ran_and_passed
        and not cap_limited_sweeps
        and sections["scalability_h10"].get("result_status") != "cap_limited"
    )
    if precision_claim_gate_passed:
        claim_status = "accepted_for_precise_scoped_fixed_route_simulation"
    elif hard_gates_passed:
        claim_status = "instrumentation_passed_precision_claim_not_accepted"
    else:
        claim_status = "not_accepted"
    report["overall_gate_summary"] = {
        "hard_gate_sections": hard_gate_sections,
        "hard_gates_passed": hard_gates_passed,
        "precision_claim_gate_passed": precision_claim_gate_passed,
        "sequence_reference_run": sequence_status == "run",
        "sequence_reference_passed": sequence_ran_and_passed,
        "severe_or_catastrophic_cap_sweep_count": len(cap_limited_sweeps),
        "waiting_model_gate_status": waiting_model_gate_status,
        "unsupported_physics_count": len(sections["unsupported_physics_registry"]),
        "claim_status": claim_status,
    }
    return _json_ready(report)


def write_markdown_summary(report: dict[str, Any], path: Path) -> None:
    summary = report["overall_gate_summary"]
    lines = [
        "# Compiled E91 Histogram Backend Validation Summary",
        "",
        f"Report version: `{report['validation_report_version']}`.",
        "",
        "## Claim Status",
        "",
        f"`{summary['claim_status']}`. This claim is limited to the compact, fixed-route, "
        "memory-emission Fock/SNSPD histogram kernel with the configured caps and device model.",
        f"Precision claim gate passed: `{summary['precision_claim_gate_passed']}`.",
        "",
        "## Gates",
        "",
        f"- Compiled self-consistency: `{report['compiled_self_consistency']['passed']}`",
        f"- Compact vs detector-click audit marginalization: `{report['compact_vs_audit_marginalization']['passed']}`",
        f"- Waiting policy/cap diagnostics emitted: `{report['waiting_policy_and_cap_bias']['passed']}`",
        f"- Waiting convergence diagnostics: `{report['waiting_convergence']['waiting_convergence_status']}`",
        f"- Waiting model gate status: `{summary['waiting_model_gate_status']}`",
        f"- Rolling live-pair protocol smoke: `{report['protocol_modes']['rolling_alive_pair']['status']}`",
        f"- QBER observability contract: `{report['qber_observability']['passed']}`",
        f"- h=10 compact scalability: `{report['scalability_h10']['passed']}`",
        f"- SeQUeNCe fixed-route reference run: `{report['sequence_reference_agreement'].get('status')}`",
        "",
        "## Justified Claims",
        "",
        "- Production compact rows marginalize raw detector-click records after detector-conditioned memory states are formed.",
        "- Expected QBER is an infinite-shot histogram quantity and remains separate from finite-sample sampled QBER.",
        "- Severe/catastrophic waiting cap bias is reported and excludes cells from absolute validation.",
        "- `rolling_alive_pair` is reported separately as an exact configured-age protocol abstraction, not as an approximation to single-buffer waiting.",
        "- h=10 compact evaluation is checked without requiring audit detector-click schema scalability.",
        "",
        "## Remaining Approximate Or Missing Physics",
        "",
    ]
    for name, entry in report["unsupported_physics_registry"].items():
        lines.append(f"- `{name}`: `{entry['status']}` - {entry['notes']}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/tmp/compiled_histogram_backend_validation")
    parser.add_argument("--hardware-config-dir", default=str(REPO_ROOT / "configs" / "physical"))
    parser.add_argument("--detector-kind", default="pnr_snspd")
    parser.add_argument("--detector-profile", default=str(DEFAULT_DETECTOR_PROFILE_PATH))
    parser.add_argument("--detector-response-file", default=None)
    parser.add_argument("--fiber-response-file", default=None)
    parser.add_argument("--conversion-response-file", default=str(DEFAULT_CONVERSION_RESPONSE_PATH))
    parser.add_argument("--allow-wavelength-mismatch", action="store_true")
    parser.add_argument("--allow-ideal-detector-reference", action="store_true")
    parser.add_argument("--fock-cutoff", type=int, default=2)
    parser.add_argument("--key-length", type=int, default=8)
    parser.add_argument("--waiting-max-attempts", type=int, default=2)
    parser.add_argument("--waiting-max-history-pairs", type=int, default=256)
    parser.add_argument("--waiting-max-wait-signatures", type=int, default=16)
    parser.add_argument("--waiting-tail-tolerance", type=float, default=1e-6)
    parser.add_argument("--waiting-budget-mode", default="fixed_attempts")
    parser.add_argument("--waiting-budget-wall-clock-ps", type=float, default=None)
    parser.add_argument(
        "--waiting-precision-mode",
        choices=("fixed_caps", "adaptive_bounded"),
        default="fixed_caps",
    )
    parser.add_argument("--waiting-tv-tolerance", type=float, default=1e-6)
    parser.add_argument("--waiting-metric-tolerance", type=float, default=1e-3)
    parser.add_argument("--waiting-max-refinement-rounds", type=int, default=4)
    parser.add_argument("--waiting-max-runtime-s", type=float, default=None)
    parser.add_argument("--waiting-max-memory-mb", type=float, default=512.0)
    parser.add_argument(
        "--waiting-protocol",
        choices=("single_buffer_swap_asap", "rolling_alive_pair"),
        default="single_buffer_swap_asap",
    )
    parser.add_argument("--rolling-edge-availability", default=None)
    parser.add_argument("--rolling-age-distribution-kind", choices=("discrete",), default="discrete")
    parser.add_argument("--rolling-age-distribution-points-s", default="0")
    parser.add_argument("--rolling-age-distribution-weights", default="1")
    parser.add_argument("--rolling-edge-age-distribution-points-s", default=None)
    parser.add_argument("--rolling-edge-age-distribution-weights", default=None)
    parser.add_argument(
        "--rolling-age-evaluation",
        choices=("cartesian", "moment_reduced"),
        default="cartesian",
    )
    parser.add_argument("--rolling-age-support-product-limit", type=int, default=1_000_000)
    parser.add_argument("--rolling-accepted-alive-time-s", type=float, default=None)
    parser.add_argument("--rolling-memory-slots-per-edge-side", type=int, default=1)
    parser.add_argument("--rolling-selection-policy", default="configured_distribution")
    parser.add_argument("--rolling-age-distribution-source", default="configured")
    parser.add_argument("--memory-cutoff-ps", type=float, default=None)
    parser.add_argument("--evaluated-cache-size", type=int, default=8)
    parser.add_argument("--tv-tolerance", type=float, default=1e-9)
    parser.add_argument("--skip-h5-reference", action="store_true")
    parser.add_argument("--run-sequence-reference", action="store_true")
    parser.add_argument("--sequence-max-routes", type=int, default=2)
    parser.add_argument("--sequence-trials", type=int, default=1)
    parser.add_argument("--compiled-trials", type=int, default=2)
    parser.add_argument("--sequence-memory-pairs", type=int, default=4)
    parser.add_argument("--sequence-repeater-window-ms", type=float, default=80.0)
    parser.add_argument("--h10-max-attempts", type=int, default=1)
    parser.add_argument("--h10-history-pair-cap", type=int, default=256)
    parser.add_argument("--h10-signature-cap", type=int, default=16)
    parser.add_argument("--h10-max-rows", type=int, default=5000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    started = time.perf_counter()
    report = build_report(args)
    report["total_wall_seconds"] = time.perf_counter() - started
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "compiled_backend_validation_report.json"
    md_path = out_dir / "compiled_backend_validation_summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_summary(report, md_path)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(json.dumps(report["overall_gate_summary"], indent=2, sort_keys=True))
    return 0 if report["overall_gate_summary"]["hard_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
