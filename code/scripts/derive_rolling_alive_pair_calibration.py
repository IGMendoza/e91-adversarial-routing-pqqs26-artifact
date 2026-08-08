"""Derive rolling_alive_pair reservation inputs from an honest CHSH target.

This is a calibration helper, not a new histogram physics model. It chooses a
memory-pool reservation policy whose serialized ``A_i`` and ``f_i(t)`` inputs
can be fed to the existing ``rolling_alive_pair`` backend.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.corpus.compiled_histogram_backend import (
    CompiledHistogramBackend,
    CompiledHistogramBackendConfig,
    CompiledHistogramUnsupported,
    DEFAULT_CONVERSION_RESPONSE_PATH,
    DEFAULT_DETECTOR_PROFILE_PATH,
    load_histogram_generator,
)
from sequence_game.physical.memory_pool_reservation import (
    EdgeReservationModel,
    minimal_slots_for_route_availability,
    rolling_alive_pair_config_for_edges,
    route_live_probability,
)


DEFAULT_T_ALIVE_SWEEP_S = (0.010, 0.020, 0.030, 0.040, 0.050, 0.060)
BELL_CLASSICAL_CHSH_BOUND = 2.0


@dataclass(frozen=True)
class ElementaryEdgeInput:
    edge_id: str
    success_probability_per_attempt: float
    attempt_period_s: float
    mean_success_time_s: float
    single_mode_success_probability: float | None = None
    multiplexing_controller_round: dict[str, Any] | None = None
    mean_empty_time_override_s: float | None = None
    initial_memory_age_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "edge_id": self.edge_id,
            "success_probability_per_attempt": self.success_probability_per_attempt,
            "attempt_period_s": self.attempt_period_s,
            "mean_success_time_s": self.mean_success_time_s,
        }
        if self.multiplexing_controller_round is not None:
            payload["single_mode_success_probability"] = self.single_mode_success_probability
            payload["multiplexing_controller_round"] = self.multiplexing_controller_round
            payload["mean_empty_time_override_s"] = self.mean_empty_time_override_s
            payload["initial_memory_age_s"] = self.initial_memory_age_s
        return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return value
    return value


def _to_tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_to_tuple_tree(item) for item in value)
    return value


def parse_float_sequence(raw: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(raw, str):
        values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    else:
        values = tuple(float(value) for value in raw)
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def calibration_lengths_m(hop_count: int, raw_km: str | Sequence[float]) -> list[float]:
    km_values = parse_float_sequence(raw_km)
    if len(km_values) == 1:
        km_values = km_values * hop_count
    if len(km_values) != hop_count:
        raise ValueError("--fiber-length-km must provide one value or one value per hop")
    if any(value <= 0.0 for value in km_values):
        raise ValueError("fiber calibration lengths must be positive")
    return [1000.0 * value for value in km_values]


def _backend_config_from_payload(
    *,
    hardware_config_dir: str,
    detector_kind: str,
    detector_profile_path: str | None,
    detector_response_path: str | None,
    fiber_response_path: str | None,
    conversion_response_path: str | None,
    multiplexing_controller_path: str | None,
    allow_wavelength_mismatch: bool,
    allow_ideal_detector_reference: bool,
    fock_cutoff: int,
    key_length: int,
    payload: dict[str, Any],
    support_product_limit: int,
) -> CompiledHistogramBackendConfig:
    return CompiledHistogramBackendConfig(
        hardware_config_dir=hardware_config_dir,
        detector_kind=detector_kind,
        detector_profile_path=detector_profile_path,
        detector_response_path=detector_response_path,
        fiber_response_path=fiber_response_path,
        conversion_response_path=conversion_response_path,
        multiplexing_controller_path=multiplexing_controller_path,
        allow_wavelength_mismatch=bool(allow_wavelength_mismatch),
        allow_ideal_detector_reference=bool(allow_ideal_detector_reference),
        fock_cutoff=int(fock_cutoff),
        waiting_protocol="rolling_alive_pair",
        rolling_edge_availability=_to_tuple_tree(payload["rolling_edge_availability"]),
        rolling_age_distribution_kind=str(payload["rolling_age_distribution_kind"]),
        rolling_age_distribution_points_s=_to_tuple_tree(payload["rolling_age_distribution_points_s"]),
        rolling_age_distribution_weights=_to_tuple_tree(payload["rolling_age_distribution_weights"]),
        rolling_edge_age_distribution_points_s=_to_tuple_tree(
            payload["rolling_edge_age_distribution_points_s"]
        ),
        rolling_edge_age_distribution_weights=_to_tuple_tree(
            payload["rolling_edge_age_distribution_weights"]
        ),
        rolling_age_evaluation="moment_reduced",
        rolling_accepted_alive_time_s=float(payload["rolling_accepted_alive_time_s"]),
        rolling_selection_policy=str(payload["rolling_selection_policy"]),
        rolling_age_distribution_source=str(payload["rolling_age_distribution_source"]),
        rolling_age_support_product_limit=int(support_product_limit),
        compile_key_length=int(key_length),
    )


def _zero_age_payload(hop_count: int) -> dict[str, Any]:
    return {
        "rolling_edge_availability": [1.0] * hop_count,
        "rolling_age_distribution_kind": "discrete",
        "rolling_age_distribution_points_s": [0.0],
        "rolling_age_distribution_weights": [1.0],
        "rolling_edge_age_distribution_points_s": [[0.0] for _ in range(hop_count)],
        "rolling_edge_age_distribution_weights": [[1.0] for _ in range(hop_count)],
        "rolling_accepted_alive_time_s": 1e-15,
        "rolling_selection_policy": "per_edge_configured_distribution",
        "rolling_age_distribution_source": "zero_age_calibration_probe",
        "backend_schema_compatibility": "per_edge_age_distribution_schema",
        "discretization": {
            "method": "delta_zero_age",
            "support_product": 1,
        },
        "edge_summaries": [
            {
                "reserved_slots": None,
                "edge_availability": 1.0,
                "selected_age_mean_s": 0.0,
                "selected_age_quantiles_s": {"q10": 0.0, "q50": 0.0, "q90": 0.0},
                "discrete_mean_age_s": 0.0,
            }
            for _ in range(hop_count)
        ],
    }


def evaluate_rolling_payload(
    *,
    hardware_config_dir: str,
    detector_kind: str,
    detector_profile_path: str | None,
    detector_response_path: str | None,
    fiber_response_path: str | None,
    conversion_response_path: str | None,
    multiplexing_controller_path: str | None,
    allow_wavelength_mismatch: bool,
    allow_ideal_detector_reference: bool,
    fock_cutoff: int,
    hop_count: int,
    fiber_lengths_m: Sequence[float],
    key_length: int,
    payload: dict[str, Any],
    support_product_limit: int,
) -> dict[str, Any]:
    config = _backend_config_from_payload(
        hardware_config_dir=hardware_config_dir,
        detector_kind=detector_kind,
        detector_profile_path=detector_profile_path,
        detector_response_path=detector_response_path,
        fiber_response_path=fiber_response_path,
        conversion_response_path=conversion_response_path,
        multiplexing_controller_path=multiplexing_controller_path,
        allow_wavelength_mismatch=allow_wavelength_mismatch,
        allow_ideal_detector_reference=allow_ideal_detector_reference,
        fock_cutoff=fock_cutoff,
        key_length=key_length,
        payload=payload,
        support_product_limit=support_product_limit,
    )
    backend = CompiledHistogramBackend(config)
    generator = load_histogram_generator()
    move = generator.EveMove("no_move_baseline", "none", target_active=False)
    evaluated = backend.evaluate(hop_count, move, fiber_lengths_m)
    rolling_model = evaluated.delivery_model.get("rolling_live_pair_model") or {}
    return {
        "row_count": evaluated.row_count,
        "probability_sum": float(evaluated.probabilities.sum()),
        "delivery_probability": evaluated.expected_metrics.get("delivery_probability"),
        "expected_qber": evaluated.expected_metrics.get("expected_qber_from_histogram"),
        "expected_chsh_S": evaluated.expected_metrics.get("expected_chsh_S"),
        "eve_guess_probability": evaluated.expected_metrics.get("eve_guess_probability"),
        "route_availability": rolling_model.get("route_availability"),
        "age_support_product": rolling_model.get("age_support_product"),
        "theta_hash": evaluated.theta_hash,
        "evaluate_seconds": evaluated.evaluate_seconds,
    }


def elementary_edge_inputs(
    *,
    hardware_config_dir: str,
    detector_kind: str,
    detector_profile_path: str | None,
    detector_response_path: str | None,
    fiber_response_path: str | None,
    conversion_response_path: str | None,
    allow_wavelength_mismatch: bool,
    fock_cutoff: int,
    fiber_lengths_m: Sequence[float],
    multiplexing_controller_path: str | None = None,
) -> tuple[list[ElementaryEdgeInput], dict[str, Any]]:
    generator = load_histogram_generator()
    front = generator.load_or_build_front_matter(Path(hardware_config_dir))
    context = generator.ModelContext(
        physical_path="fock_pnr_snspd",
        source_model="memory_emission",
        detector_kind=detector_kind,
        fock_cutoff=int(fock_cutoff),
        detector_profile_path=detector_profile_path,
        detector_profile=generator.load_json_optional(detector_profile_path),
        detector_response=generator.load_json_optional(detector_response_path),
        fiber_response=generator.load_json_optional(fiber_response_path),
        conversion_response=generator.load_json_optional(conversion_response_path),
        multiplexing_controller=generator.load_json_optional(multiplexing_controller_path),
        allow_wavelength_mismatch=bool(allow_wavelength_mismatch),
        waiting_protocol="rolling_alive_pair",
    )
    route = generator.make_route(len(fiber_lengths_m), list(fiber_lengths_m))
    move = generator.EveMove("no_move_baseline", "none", target_active=False)
    attempt_policy = generator.attempt_period_policy(route, front, context)
    memory = front["device_constants"]["memory"]
    detected_decay_lifetime_s = memory.get("detected_decay_lifetime_s")
    out: list[ElementaryEdgeInput] = []
    for edge_id in route["edge_ids"]:
        probs, _meta = generator.fock_elementary_edge_outcome_probabilities(
            edge_id, route, front, move, context
        )
        single_mode_success = float(
            probs.get("psi_plus", 0.0) + probs.get("psi_minus", 0.0)
        )
        success, controller_round = generator.multiplexed_edge_round(
            single_mode_success_probability=single_mode_success,
            edge_id=edge_id,
            route=route,
            front=front,
            context=context,
        )
        period_s = (
            float(attempt_policy["edge_attempt_periods"][edge_id]["attempt_period_ps"])
            * 1e-12
        )
        mean_empty_override_s = None
        initial_memory_age_s = 0.0
        if controller_round is not None:
            herald_s = float(
                attempt_policy["edge_attempt_periods"][edge_id][
                    "deterministic_herald_round_trip_ps"
                ]
            ) * 1e-12
            initial_memory_age_s = herald_s
            preaccept_survival = (
                math.exp(-2.0 * herald_s / float(detected_decay_lifetime_s))
                if detected_decay_lifetime_s is not None
                and float(detected_decay_lifetime_s) > 0.0
                else 1.0
            )
            optical_round_success = float(success)
            success = optical_round_success * preaccept_survival
            round_payload = controller_round["round"]
            expected_herald_s = float(
                round_payload["expected_first_success_herald_time_s"]
            )
            mean_empty_override_s = (
                ((1.0 - success) / success) * period_s + expected_herald_s
                if success > 0.0
                else math.inf
            )
            controller_round = {
                **controller_round,
                "preaccept_memory_survival": {
                    "memory_count": 2,
                    "age_per_memory_s": herald_s,
                    "detected_decay_lifetime_s": detected_decay_lifetime_s,
                    "survival_probability": preaccept_survival,
                    "optical_round_success_probability": optical_round_success,
                    "accepted_round_success_probability": success,
                },
                "reservation_fill_timing": {
                    "mean_empty_time_s": mean_empty_override_s,
                    "formula": "((1-p_accept)/p_accept)*T_round + E[T_herald|accepted]",
                    "successful_round_is_not_charged_the_unused_post-herald_round_tail": True,
                },
            }
        out.append(ElementaryEdgeInput(
            edge_id=edge_id,
            success_probability_per_attempt=success,
            attempt_period_s=period_s,
            mean_success_time_s=(
                mean_empty_override_s
                if mean_empty_override_s is not None
                else (period_s / success) if success > 0.0 else math.inf
            ),
            single_mode_success_probability=(
                single_mode_success if controller_round is not None else None
            ),
            multiplexing_controller_round=controller_round,
            mean_empty_time_override_s=mean_empty_override_s,
            initial_memory_age_s=initial_memory_age_s,
        ))
    detector_profile = generator.detector_profile_metadata(front, context)
    selected_memory_name = front.get("selected_models", {}).get("memory")
    selected_memory = front.get("models", {}).get(selected_memory_name, {})
    declared_multiplexing = selected_memory.get("parameters", {}).get("multiplexing_model")
    front_meta = {
        "front_matter_hash": front.get("front_matter_hash"),
        "detector_kind": detector_kind,
        "resolved_detector_kind": generator.resolved_detector_kind(front, context),
        "detector_profile_path": detector_profile_path,
        "detector_profile_reference": detector_profile.get("reference_name"),
        "detector_response_path": detector_response_path,
        "fiber_response_path": fiber_response_path,
        "conversion_response_path": conversion_response_path,
        "wavelength_compatibility": generator.wavelength_compatibility_report(front, context),
        "memory_coherence_time_s": memory.get("coherence_time_s"),
        "detected_decay_lifetime_s": memory.get("detected_decay_lifetime_s"),
        "storage_model": memory.get("memory_storage_decoherence_model"),
        "memory_process_model": memory.get("memory_process_model"),
        "memory_multiplexing": (
            {
                "declared_model": declared_multiplexing,
                "active_in_compiled_kernel": True,
                "active_operating_mode": "independent_synchronized_time_bin_controller",
                "controller_path": multiplexing_controller_path,
                "controller_hash": generator.stable_hash(context.multiplexing_controller),
            }
            if context.multiplexing_controller is not None
            else {
                "declared_model": declared_multiplexing,
                "active_in_compiled_kernel": False,
                "active_operating_mode": "single_mode_memory_emission",
            }
        ),
        "attempt_period_policy": attempt_policy,
    }
    return out, front_meta


def _reservation_edges(
    edge_inputs: Sequence[ElementaryEdgeInput],
    *,
    accepted_alive_time_s: float,
    detected_decay_lifetime_s: float | None,
    selection_policy: str,
) -> list[EdgeReservationModel]:
    return [
        EdgeReservationModel(
            success_probability_per_attempt=edge.success_probability_per_attempt,
            attempt_period_s=edge.attempt_period_s,
            accepted_alive_time_s=accepted_alive_time_s,
            detected_decay_lifetime_s=detected_decay_lifetime_s,
            reserved_slots=1,
            selection_policy=selection_policy,
            mean_empty_time_override_s=edge.mean_empty_time_override_s,
            initial_memory_age_s=edge.initial_memory_age_s,
            detected_decay_memory_count=(
                2 if edge.multiplexing_controller_round is not None else 1
            ),
        )
        for edge in edge_inputs
    ]


def derive_calibration(args: argparse.Namespace) -> dict[str, Any]:
    hop_count = int(args.h_cal)
    if hop_count < 1:
        raise ValueError("--h-cal must be >= 1")
    fiber_lengths_m = calibration_lengths_m(hop_count, args.fiber_length_km)
    t_alive_sweep = parse_float_sequence(args.t_alive_sweep_s)
    if any(value <= 0.0 for value in t_alive_sweep):
        raise ValueError("--t-alive-sweep-s values must be positive")
    if int(args.num_age_points) < 1:
        raise ValueError("--num-age-points must be >= 1")
    if int(args.c_max) < 1:
        raise ValueError("--c-max must be >= 1")
    if int(args.rolling_age_support_product_limit) < 1:
        raise ValueError("--rolling-age-support-product-limit must be >= 1")

    edge_inputs, front_meta = elementary_edge_inputs(
        hardware_config_dir=args.hardware_config_dir,
        detector_kind=args.detector_kind,
        detector_profile_path=args.detector_profile,
        detector_response_path=args.detector_response_file,
        fiber_response_path=args.fiber_response_file,
        conversion_response_path=args.conversion_response_file,
        multiplexing_controller_path=args.multiplexing_controller,
        allow_wavelength_mismatch=bool(args.allow_wavelength_mismatch),
        fock_cutoff=int(args.fock_cutoff),
        fiber_lengths_m=fiber_lengths_m,
    )
    zero_age_payload = _zero_age_payload(hop_count)
    zero_age = evaluate_rolling_payload(
        hardware_config_dir=args.hardware_config_dir,
        detector_kind=args.detector_kind,
        detector_profile_path=args.detector_profile,
        detector_response_path=args.detector_response_file,
        fiber_response_path=args.fiber_response_file,
        conversion_response_path=args.conversion_response_file,
        multiplexing_controller_path=args.multiplexing_controller,
        allow_wavelength_mismatch=bool(args.allow_wavelength_mismatch),
        allow_ideal_detector_reference=bool(args.allow_ideal_detector_reference),
        fock_cutoff=int(args.fock_cutoff),
        hop_count=hop_count,
        fiber_lengths_m=fiber_lengths_m,
        key_length=int(args.key_length),
        payload=zero_age_payload,
        support_product_limit=int(args.rolling_age_support_product_limit),
    )
    s_target = float(args.chsh_target)
    route_target = float(args.route_availability_target)
    if not (0.0 <= route_target < 1.0):
        raise ValueError("--route-availability-target must be in [0, 1)")
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    zero_s = float(zero_age["expected_chsh_S"])
    zero_status = (
        "zero_age_not_bell_violating" if zero_s <= BELL_CLASSICAL_CHSH_BOUND else
        "zero_age_below_target" if zero_s < s_target else
        "zero_age_passes_target"
    )
    if zero_status == "zero_age_passes_target":
        detected_decay_lifetime_s = (
            None if args.ignore_detected_decay_in_pool
            else front_meta.get("detected_decay_lifetime_s")
        )
        for t_alive_s in t_alive_sweep:
            candidate: dict[str, Any] = {
                "accepted_alive_time_s": float(t_alive_s),
                "status": "not_evaluated",
            }
            base_edges = _reservation_edges(
                edge_inputs,
                accepted_alive_time_s=float(t_alive_s),
                detected_decay_lifetime_s=detected_decay_lifetime_s,
                selection_policy=str(args.selection_policy),
            )
            try:
                plan = minimal_slots_for_route_availability(
                    base_edges,
                    route_target,
                    c_max=int(args.c_max),
                )
            except ValueError as exc:
                candidate.update({
                    "status": "route_availability_unreachable",
                    "reason": str(exc),
                })
                candidates.append(candidate)
                continue
            planned_edges = [
                edge.with_slots(slots)
                for edge, slots in zip(base_edges, plan.reserved_slots)
            ]
            payload = rolling_alive_pair_config_for_edges(
                planned_edges,
                num_age_points=int(args.num_age_points),
            )
            support_product = int(payload["discretization"]["support_product"])
            candidate.update({
                "reservation_plan": plan.to_dict(),
                "total_reserved_slots": int(sum(plan.reserved_slots)),
                "support_product": support_product,
                "rolling_config": payload,
            })
            if support_product > int(args.rolling_age_support_product_limit):
                candidate.update({
                    "status": "support_product_exceeds_limit",
                    "reason": (
                        f"{support_product} > {int(args.rolling_age_support_product_limit)}"
                    ),
                })
                candidates.append(candidate)
                continue
            metrics = evaluate_rolling_payload(
                hardware_config_dir=args.hardware_config_dir,
                detector_kind=args.detector_kind,
                detector_profile_path=args.detector_profile,
                detector_response_path=args.detector_response_file,
                fiber_response_path=args.fiber_response_file,
                conversion_response_path=args.conversion_response_file,
                multiplexing_controller_path=args.multiplexing_controller,
                allow_wavelength_mismatch=bool(args.allow_wavelength_mismatch),
                allow_ideal_detector_reference=bool(args.allow_ideal_detector_reference),
                fock_cutoff=int(args.fock_cutoff),
                hop_count=hop_count,
                fiber_lengths_m=fiber_lengths_m,
                key_length=int(args.key_length),
                payload=payload,
                support_product_limit=int(args.rolling_age_support_product_limit),
            )
            satisfies = (
                float(metrics["expected_chsh_S"]) >= s_target
                and float(metrics["route_availability"]) >= route_target
            )
            candidate.update({
                "status": "satisfies_targets" if satisfies else "below_target",
                "metrics": metrics,
            })
            candidates.append(candidate)
        satisfying = [row for row in candidates if row["status"] == "satisfies_targets"]
        if satisfying:
            selected = min(
                satisfying,
                key=lambda row: (
                    int(row["total_reserved_slots"]),
                    float(row["accepted_alive_time_s"]),
                ),
            )

    report_status = (
        "selected" if selected is not None else
        zero_status if zero_status != "zero_age_passes_target" else
        "no_candidate_satisfied_targets"
    )
    report = {
        "calibration_report_version": 1,
        "status": report_status,
        "calibration_rule": (
            "choose the cheapest rolling_alive_pair reservation policy whose honest no-move "
            "compiled histogram satisfies route availability and CHSH targets at the calibration route"
        ),
        "calibration_inputs": {
            "h_cal": hop_count,
            "fiber_lengths_m": fiber_lengths_m,
            "fiber_length_km": [value / 1000.0 for value in fiber_lengths_m],
            "chsh_target": s_target,
            "bell_classical_chsh_bound": BELL_CLASSICAL_CHSH_BOUND,
            "route_availability_target": route_target,
            "t_alive_sweep_s": list(t_alive_sweep),
            "selection_policy": args.selection_policy,
            "num_age_points": int(args.num_age_points),
            "c_max": int(args.c_max),
            "detector_kind": args.detector_kind,
            "detector_profile_path": args.detector_profile,
            "detector_response_path": args.detector_response_file,
            "fiber_response_path": args.fiber_response_file,
            "conversion_response_path": args.conversion_response_file,
            "allow_wavelength_mismatch": bool(args.allow_wavelength_mismatch),
            "allow_ideal_detector_reference": bool(args.allow_ideal_detector_reference),
            "fock_cutoff": int(args.fock_cutoff),
            "hardware_config_dir": args.hardware_config_dir,
            "rolling_age_support_product_limit": int(args.rolling_age_support_product_limit),
            "selection_objective": "min_total_reserved_slots_then_shortest_T_alive",
        },
        "front_matter": front_meta,
        "elementary_edges": [edge.to_dict() for edge in edge_inputs],
        "zero_age_honest": {
            "status": zero_status,
            "metrics": zero_age,
            "interpretation": (
                "If zero-age CHSH is below the target, storage-age calibration cannot rescue the route; "
                "non-storage noise/process assumptions dominate."
            ),
        },
        "candidates": candidates,
        "selected": selected,
        "notes": [
            "This script derives configured rolling_alive_pair inputs; it does not change backend physics.",
            "The result is exact relative to the finite per-edge age distributions that are serialized.",
            "A 10 km scalar calibration length is a model assumption; shorter/easier links are expected to have at least this much link-transmittance margin, but other noise terms still apply.",
        ],
    }
    if args.multiplexing_controller is not None:
        report["calibration_inputs"]["multiplexing_controller_path"] = (
            args.multiplexing_controller
        )
        report["notes"].append(
            "The controller changes only round cadence/success and reservation-age inputs; the "
            "detector-conditioned one-mode Fock instrument remains unchanged."
        )
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config-dir", default="configs/physical")
    parser.add_argument("--detector-kind", default="pnr_snspd")
    parser.add_argument("--detector-profile", default=str(DEFAULT_DETECTOR_PROFILE_PATH))
    parser.add_argument("--detector-response-file", default=None)
    parser.add_argument("--fiber-response-file", default=None)
    parser.add_argument("--conversion-response-file", default=str(DEFAULT_CONVERSION_RESPONSE_PATH))
    parser.add_argument(
        "--multiplexing-controller",
        default=None,
        help="optional independent time-bin controller JSON; disabled by default",
    )
    parser.add_argument(
        "--allow-wavelength-mismatch",
        action="store_true",
        help="diagnostic override only; strict calibration otherwise rejects uncovered wavelengths",
    )
    parser.add_argument(
        "--allow-ideal-detector-reference",
        action="store_true",
        help="permit ideal_pnr only for an explicitly labelled mathematical reference/test calibration",
    )
    parser.add_argument("--fock-cutoff", type=int, default=2)
    parser.add_argument("--h-cal", type=int, default=12)
    parser.add_argument("--fiber-length-km", default="10")
    parser.add_argument("--chsh-target", type=float, default=2.05)
    parser.add_argument("--route-availability-target", type=float, default=0.95)
    parser.add_argument(
        "--t-alive-sweep-s",
        default=",".join(str(value) for value in DEFAULT_T_ALIVE_SWEEP_S),
        help="comma-separated accepted-live-time candidates in seconds",
    )
    parser.add_argument("--selection-policy", choices=("youngest_live", "random_live"), default="youngest_live")
    parser.add_argument("--num-age-points", type=int, default=2)
    parser.add_argument("--c-max", type=int, default=64)
    parser.add_argument("--rolling-age-support-product-limit", type=int, default=1_000_000)
    parser.add_argument("--key-length", type=int, default=8)
    parser.add_argument(
        "--ignore-detected-decay-in-pool",
        action="store_true",
        help="derive A_i and f_i(t) without the memory detected-decay lifetime in the reservation pool",
    )
    parser.add_argument("--out", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = derive_calibration(args)
    except CompiledHistogramUnsupported as exc:
        raise SystemExit(str(exc)) from exc
    text = json.dumps(_json_ready(report), indent=2, sort_keys=True)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
