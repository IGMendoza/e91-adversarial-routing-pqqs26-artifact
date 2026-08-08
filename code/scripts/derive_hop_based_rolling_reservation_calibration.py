"""Diagnostic target-provisioned rolling reservation study at 10 km per hop.

This historical study solves for however many independent slots are needed to
hit a requested route availability.  It is a ``diagnostic_reference`` and not
a hardware-feasible repeater-v1 calibration.  Production Alice validation uses
``scripts/validate_alice_repeater_v1.py``, whose memory inventory is fixed by a
literature-scoped service profile.  The standardized route family here remains
L_i = 10 km for every elementary hop, with h as the only depth variable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from derive_rolling_alive_pair_calibration import (  # type: ignore
    ElementaryEdgeInput,
    _backend_config_from_payload,
    _json_ready,
    elementary_edge_inputs,
)
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
    edge_live_probability,
    minimal_slots_for_route_availability,
    rolling_alive_pair_config_for_edges,
    route_live_probability,
    selected_pair_age_quantile,
    selected_pair_age_quantiles,
    selected_pair_expected_age_s,
    slot_live_probability,
)


DEFAULT_OUT_JSON = Path("docs/E91_ROLLING_HOP_BASED_CALIBRATION_10KM.json")
DEFAULT_OUT_MD = Path("docs/E91_ROLLING_HOP_BASED_CALIBRATION_10KM.md")
CHSH_CLASSICAL_BOUND = 2.0


@dataclass(frozen=True)
class HopCalibrationConfig:
    max_hop: int = 12
    stress_max_hop: int = 15
    edge_length_km: float = 10.0
    accepted_alive_time_s: float = 0.18
    route_availability_target: float = 0.95
    chsh_target: float = 2.10
    age_points_per_edge: int = 8
    selection_policy: str = "youngest_live"
    hardware_config_dir: str = "configs/physical"
    detector_kind: str = "pnr_snspd"
    detector_profile_path: str | None = str(DEFAULT_DETECTOR_PROFILE_PATH)
    detector_response_path: str | None = None
    fiber_response_path: str | None = None
    conversion_response_path: str | None = str(DEFAULT_CONVERSION_RESPONSE_PATH)
    multiplexing_controller_path: str | None = None
    allow_wavelength_mismatch: bool = False
    allow_ideal_detector_reference: bool = False
    fock_cutoff: int = 2
    key_length: int = 8
    c_max: int = 64
    rolling_age_support_product_limit: int = 1_000_000
    ignore_detected_decay_in_pool: bool = False


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _mean(values: Sequence[float]) -> float:
    return math.fsum(float(value) for value in values) / len(values)


def _reservation_edges(
    edge_inputs: Sequence[ElementaryEdgeInput],
    *,
    accepted_alive_time_s: float,
    detected_decay_lifetime_s: float | None,
    selection_policy: str,
    slots: Sequence[int] | None = None,
) -> list[EdgeReservationModel]:
    if slots is None:
        slots = [1] * len(edge_inputs)
    return [
        EdgeReservationModel(
            success_probability_per_attempt=edge.success_probability_per_attempt,
            attempt_period_s=edge.attempt_period_s,
            accepted_alive_time_s=accepted_alive_time_s,
            detected_decay_lifetime_s=detected_decay_lifetime_s,
            reserved_slots=int(slot),
            selection_policy=selection_policy,
            mean_empty_time_override_s=edge.mean_empty_time_override_s,
            initial_memory_age_s=edge.initial_memory_age_s,
            detected_decay_memory_count=(
                2 if edge.multiplexing_controller_round is not None else 1
            ),
        )
        for edge, slot in zip(edge_inputs, slots)
    ]


def derive_hop_reservation(
    hop_count: int,
    config: HopCalibrationConfig,
) -> tuple[list[ElementaryEdgeInput], dict[str, Any], Any, list[EdgeReservationModel], dict[str, Any]]:
    """Derive the exact minimum-total-slot reservation plan for one hop count."""

    if hop_count < 1:
        raise ValueError("hop_count must be >= 1")
    CompiledHistogramBackend(CompiledHistogramBackendConfig(
        hardware_config_dir=config.hardware_config_dir,
        detector_kind=config.detector_kind,
        detector_profile_path=config.detector_profile_path,
        detector_response_path=config.detector_response_path,
        fiber_response_path=config.fiber_response_path,
        conversion_response_path=config.conversion_response_path,
        multiplexing_controller_path=config.multiplexing_controller_path,
        allow_wavelength_mismatch=config.allow_wavelength_mismatch,
        allow_ideal_detector_reference=config.allow_ideal_detector_reference,
        require_explicit_device_stack=True,
        fock_cutoff=int(config.fock_cutoff),
    )).diagnostics()
    lengths_m = [1000.0 * float(config.edge_length_km)] * int(hop_count)
    edge_inputs, front_meta = elementary_edge_inputs(
        hardware_config_dir=config.hardware_config_dir,
        detector_kind=config.detector_kind,
        detector_profile_path=config.detector_profile_path,
        detector_response_path=config.detector_response_path,
        fiber_response_path=config.fiber_response_path,
        conversion_response_path=config.conversion_response_path,
        allow_wavelength_mismatch=config.allow_wavelength_mismatch,
        fock_cutoff=int(config.fock_cutoff),
        fiber_lengths_m=lengths_m,
        multiplexing_controller_path=config.multiplexing_controller_path,
    )
    detected_decay_lifetime_s = (
        None
        if config.ignore_detected_decay_in_pool
        else front_meta.get("detected_decay_lifetime_s")
    )
    base_edges = _reservation_edges(
        edge_inputs,
        accepted_alive_time_s=float(config.accepted_alive_time_s),
        detected_decay_lifetime_s=detected_decay_lifetime_s,
        selection_policy=config.selection_policy,
    )
    plan = minimal_slots_for_route_availability(
        base_edges,
        float(config.route_availability_target),
        c_max=int(config.c_max),
    )
    planned_edges = _reservation_edges(
        edge_inputs,
        accepted_alive_time_s=float(config.accepted_alive_time_s),
        detected_decay_lifetime_s=detected_decay_lifetime_s,
        selection_policy=config.selection_policy,
        slots=plan.reserved_slots,
    )
    payload = rolling_alive_pair_config_for_edges(
        planned_edges,
        num_age_points=int(config.age_points_per_edge),
    )
    return edge_inputs, front_meta, plan, planned_edges, payload


def _compiled_backend_metrics(
    *,
    hop_count: int,
    fiber_lengths_m: Sequence[float],
    payload: dict[str, Any],
    config: HopCalibrationConfig,
) -> dict[str, Any]:
    backend_config = _backend_config_from_payload(
        hardware_config_dir=config.hardware_config_dir,
        detector_kind=config.detector_kind,
        detector_profile_path=config.detector_profile_path,
        detector_response_path=config.detector_response_path,
        fiber_response_path=config.fiber_response_path,
        conversion_response_path=config.conversion_response_path,
        multiplexing_controller_path=config.multiplexing_controller_path,
        allow_wavelength_mismatch=config.allow_wavelength_mismatch,
        allow_ideal_detector_reference=config.allow_ideal_detector_reference,
        fock_cutoff=int(config.fock_cutoff),
        key_length=int(config.key_length),
        payload=payload,
        support_product_limit=int(config.rolling_age_support_product_limit),
    )
    backend_config = backend_config.__class__(
        **{
            **backend_config.to_dict(),
            "rolling_edge_availability": _tuple_tree(payload["rolling_edge_availability"]),
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
        }
    )
    backend = CompiledHistogramBackend(backend_config)
    generator = load_histogram_generator()
    move = generator.EveMove("no_move_baseline", "none", target_active=False)

    started = time.perf_counter()
    evaluated = backend.evaluate(int(hop_count), move, fiber_lengths_m)
    total_seconds = time.perf_counter() - started
    diagnostics = backend.diagnostics()
    expected = evaluated.expected_metrics
    rolling_model = evaluated.delivery_model.get("rolling_live_pair_model") or {}
    probability_sum = float(evaluated.probabilities.sum())
    return {
        "theta_hash": evaluated.theta_hash,
        "row_count": int(evaluated.row_count),
        "probability_sum": probability_sum,
        "probability_normalization_error": abs(probability_sum - 1.0),
        "delivery_probability": expected.get("delivery_probability"),
        "expected_qber": expected.get(
            "expected_qber_from_histogram",
            expected.get("expected_qber"),
        ),
        "expected_chsh_S": expected.get("expected_chsh_S"),
        "eve_guess_probability": expected.get("eve_guess_probability"),
        "route_availability_from_backend": rolling_model.get("route_availability"),
        "age_support_product_reported": rolling_model.get("age_support_product"),
        "evaluate_seconds": float(evaluated.evaluate_seconds),
        "total_seconds": float(total_seconds),
        "backend_stats": diagnostics["stats"],
    }


def _age_summary(edges: Sequence[EdgeReservationModel]) -> dict[str, Any]:
    rows = []
    for index, edge in enumerate(edges):
        quantiles = selected_pair_age_quantiles(edge, (0.5, 0.9))
        rows.append({
            "edge_index": index,
            "reserved_slots": int(edge.reserved_slots),
            "slot_live_probability": slot_live_probability(edge),
            "edge_availability": edge_live_probability(edge),
            "selected_age_mean_s": selected_pair_expected_age_s(edge),
            "selected_age_q50_s": quantiles["q50"],
            "selected_age_q90_s": quantiles["q90"],
            "selected_age_max_s": float(edge.accepted_alive_time_s),
        })
    means = [row["selected_age_mean_s"] for row in rows]
    q50s = [row["selected_age_q50_s"] for row in rows]
    q90s = [row["selected_age_q90_s"] for row in rows]
    worst = max(rows, key=lambda row: (row["selected_age_q90_s"], row["selected_age_mean_s"]))
    return {
        "per_edge": rows,
        "aggregate": {
            "mean_selected_age_s": _mean(means),
            "mean_q50_s": _mean(q50s),
            "mean_q90_s": _mean(q90s),
            "max_age_s": max(row["selected_age_max_s"] for row in rows),
        },
        "worst_edge_by_q90": worst,
    }


def evaluate_hop(
    hop_count: int,
    config: HopCalibrationConfig,
    *,
    stress: bool = False,
) -> dict[str, Any]:
    edge_inputs, front_meta, plan, edges, payload = derive_hop_reservation(hop_count, config)
    lengths_m = [1000.0 * float(config.edge_length_km)] * int(hop_count)
    metrics = _compiled_backend_metrics(
        hop_count=hop_count,
        fiber_lengths_m=lengths_m,
        payload=payload,
        config=config,
    )
    slots = list(plan.reserved_slots)
    route_availability = route_live_probability(edges)
    expected_chsh = float(metrics["expected_chsh_S"])
    expected_qber = metrics["expected_qber"]
    delivery = float(metrics["delivery_probability"])
    return {
        "h": int(hop_count),
        "stress_row": bool(stress),
        "total_distance_km": float(config.edge_length_km) * int(hop_count),
        "fiber_lengths_m": lengths_m,
        "slot_vector": slots,
        "total_slots": int(sum(slots)),
        "slots_min": int(min(slots)),
        "slots_max": int(max(slots)),
        "slots_mean": _mean(slots),
        "route_availability": route_availability,
        "route_availability_target": float(config.route_availability_target),
        "delivery_probability": delivery,
        "no_live_pair_probability": 1.0 - route_availability,
        "expected_qber": expected_qber,
        "expected_chsh_S": expected_chsh,
        "bell_margin_above_2": expected_chsh - CHSH_CLASSICAL_BOUND,
        "bell_margin_above_target": expected_chsh - float(config.chsh_target),
        "age_summary": _age_summary(edges),
        "support_product_avoided": int(config.age_points_per_edge) ** int(hop_count),
        "row_count": int(metrics["row_count"]),
        "evaluate_seconds": float(metrics["evaluate_seconds"]),
        "total_seconds": float(metrics["total_seconds"]),
        "probability_sum": float(metrics["probability_sum"]),
        "probability_normalization_error": float(metrics["probability_normalization_error"]),
        "reservation_plan": plan.to_dict(),
        "elementary_edges": [edge.to_dict() for edge in edge_inputs],
        "front_matter": front_meta,
        "rolling_config_summary": {
            "rolling_age_evaluation": "moment_reduced",
            "rolling_age_distribution_source": payload["rolling_age_distribution_source"],
            "rolling_selection_policy": payload["rolling_selection_policy"],
            "discretization": payload["discretization"],
            "edge_summaries": payload["edge_summaries"],
        },
        "compiled_metrics": metrics,
        "passes": {
            "route_availability_ge_target": route_availability >= float(config.route_availability_target),
            "chsh_ge_target": expected_chsh >= float(config.chsh_target),
            "qber_zero_within_tolerance": (
                expected_qber is not None and abs(float(expected_qber)) <= 1e-12
            ),
            "probability_normalized": float(metrics["probability_normalization_error"]) <= 1e-10,
            "delivery_matches_route_availability": abs(delivery - route_availability) <= 1e-10,
        },
    }


def derive_hop_calibration(config: HopCalibrationConfig) -> dict[str, Any]:
    if config.max_hop < 1:
        raise ValueError("max_hop must be >= 1")
    if config.stress_max_hop < config.max_hop:
        raise ValueError("stress_max_hop must be >= max_hop")
    if config.edge_length_km <= 0.0:
        raise ValueError("edge_length_km must be > 0")
    if config.accepted_alive_time_s <= 0.0:
        raise ValueError("accepted_alive_time_s must be > 0")
    if not (0.0 <= config.route_availability_target < 1.0):
        raise ValueError("route_availability_target must be in [0, 1)")
    if config.age_points_per_edge < 1:
        raise ValueError("age_points_per_edge must be >= 1")

    primary: list[dict[str, Any]] = []
    stress: list[dict[str, Any]] = []
    started = time.perf_counter()
    for hop in range(1, int(config.max_hop) + 1):
        primary.append(evaluate_hop(hop, config, stress=False))
    for hop in range(int(config.max_hop) + 1, int(config.stress_max_hop) + 1):
        stress.append(evaluate_hop(hop, config, stress=True))
    elapsed = time.perf_counter() - started
    primary_failures = [
        {
            "h": row["h"],
            "passes": row["passes"],
            "route_availability": row["route_availability"],
            "expected_chsh_S": row["expected_chsh_S"],
            "expected_qber": row["expected_qber"],
        }
        for row in primary
        if not all(bool(value) for value in row["passes"].values())
    ]
    h12 = next((row for row in primary if int(row["h"]) == 12), None)
    return {
        "calibration_report_version": 1,
        "status": "diagnostic_reference_pass" if not primary_failures else "diagnostic_reference_failures",
        "scope": "diagnostic_reference",
        "production_replacement": "scripts/validate_alice_repeater_v1.py",
        "calibration_philosophy": (
            "Diagnostic hop-count target provisioning on the standardized family L_i=10 km. "
            "The optimizer is permitted to create up to c_max independent slots per edge; "
            "therefore its output is not a claim about a realizable two-ion repeater node."
        ),
        "fixed_assumptions": {
            "physical_path": "fock_pnr_snspd",
            "source_model": "memory_emission",
            "detector_kind": config.detector_kind,
            "detector_profile_path": config.detector_profile_path,
            "detector_response_path": config.detector_response_path,
            "fiber_response_path": config.fiber_response_path,
            "conversion_response_path": config.conversion_response_path,
            "allow_wavelength_mismatch": config.allow_wavelength_mismatch,
            "allow_ideal_detector_reference": config.allow_ideal_detector_reference,
            "memory_process_model": "none; scalar Cui ion-photon/storage fidelity is metadata only",
            "storage_model": "Gaussian storage coherence from the memory profile",
            "waiting_protocol": "rolling_alive_pair",
            "rolling_age_evaluation": "moment_reduced",
            "selection_policy": config.selection_policy,
            "edge_length_m": 1000.0 * float(config.edge_length_km),
            "accepted_alive_time_s": float(config.accepted_alive_time_s),
            "age_points_per_edge": int(config.age_points_per_edge),
            "route_availability_target": float(config.route_availability_target),
            "chsh_target": float(config.chsh_target),
        },
        "config": config.__dict__,
        "primary_rows": primary,
        "stress_rows": stress,
        "primary_failures": primary_failures,
        "h12_reproduction": h12,
        "elapsed_seconds": elapsed,
        "notes": [
            "A_route is the probability the reserved elementary-link pool has one accepted live pair on every route edge at request time.",
            "A_route is route availability/yield, not Bell-state quality.",
            "Delivery equals A_route in this no-move rolling baseline up to modeled detector, memory-survival, and numerical effects.",
            "The age support product 8^h is avoided by the exact moment_reduced age-hidden evaluator; Cartesian enumeration is not used for the hop sweep.",
            "This calibration is fixed to 10 km per elementary link and should not be interpreted as universal for arbitrary fiber vectors.",
            "The minimum-slot optimizer is not constrained by a literature hardware inventory. Its 45-slot h=12 result is diagnostic, not the repeater-v1 production model.",
            "Use validate_alice_repeater_v1.py for fixed two-ion resources, asynchronous service rate, source-specific Bell errors, and 1--50 km link validation.",
        ],
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if abs(value) >= 1e6 or (0.0 < abs(value) < 1e-4):
            return f"{value:.3e}"
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def _markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    headers = [
        "h",
        "km",
        "slots",
        "sum",
        "A_route",
        "delivery",
        "QBER",
        "S",
        "S-2.10",
        "age_mean_ms",
        "age_q90_worst_ms",
        "8^h avoided",
        "rows",
        "eval_s",
        "norm_err",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        age = row["age_summary"]
        values = [
            row["h"],
            _fmt(row["total_distance_km"], 1),
            "`" + ",".join(str(item) for item in row["slot_vector"]) + "`",
            row["total_slots"],
            _fmt(row["route_availability"], 12),
            _fmt(row["delivery_probability"], 12),
            _fmt(row["expected_qber"], 3),
            _fmt(row["expected_chsh_S"], 12),
            _fmt(row["bell_margin_above_target"], 6),
            _fmt(1000.0 * age["aggregate"]["mean_selected_age_s"], 3),
            _fmt(1000.0 * age["worst_edge_by_q90"]["selected_age_q90_s"], 3),
            f"`{row['support_product_avoided']}`",
            row["row_count"],
            _fmt(row["evaluate_seconds"], 4),
            _fmt(row["probability_normalization_error"], 3),
        ]
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)


def render_markdown(report: dict[str, Any]) -> str:
    config = report["fixed_assumptions"]
    primary_rows = report["primary_rows"]
    primary_max_h = int(primary_rows[-1]["h"]) if primary_rows else 0
    h12 = report.get("h12_reproduction") or {}
    lines = [
        "# Diagnostic E91 target-provisioned rolling study at 10 km",
        "",
        "## Calibration philosophy",
        "",
        report["calibration_philosophy"],
        "",
        "This artifact is a `diagnostic_reference`. Its slot optimizer is not a physical hardware inventory; use `scripts/validate_alice_repeater_v1.py` for the bounded Alice repeater-v1 model.",
        "",
        "`A_route` is the route availability/yield: the probability that every elementary edge has an accepted live pair at route request time. It is not Bell quality. Bell quality is tracked separately through expected CHSH `S` and QBER.",
        "",
        "## Fixed assumptions",
        "",
        f"- elementary-link baseline: {config['edge_length_m'] / 1000.0:g} km per hop",
        f"- waiting protocol: `{config['waiting_protocol']}`",
        f"- age evaluation: `{config['rolling_age_evaluation']}` with {config['age_points_per_edge']} points per edge",
        f"- accepted live-pair cutoff: {config['accepted_alive_time_s']:.3f} s",
        f"- selection policy: `{config['selection_policy']}`",
        f"- route availability target: {config['route_availability_target']:.3f}",
        f"- CHSH target: S >= {config['chsh_target']:.2f}",
        "- corrected Cui semantics: `memory_process_model.kind=none`; ion-photon fidelity is provenance metadata; Gaussian storage coherence is active.",
        "",
        f"## Primary calibrated range: h=1..{primary_max_h}",
        "",
        _markdown_table(primary_rows),
        "",
    ]
    if h12:
        h12_matches_default_baseline = (
            abs(float(config["accepted_alive_time_s"]) - 0.18) <= 1e-12
            and int(h12["total_slots"]) == 45
        )
        lines.extend([
            "## h=12 reproduction",
            "",
            "The h=12 standardized route uses 12 x 10 km = 120 km total fiber.",
            "",
            f"- total slots: {h12['total_slots']}",
            f"- slot vector: `{','.join(str(item) for item in h12['slot_vector'])}`",
            f"- A_route: {h12['route_availability']:.12f}",
            f"- delivery probability: {h12['delivery_probability']:.12f}",
            f"- expected QBER: {h12['expected_qber']}",
            f"- expected CHSH S: {h12['expected_chsh_S']:.12f}",
            "",
            (
                "This reproduces the conservative 0.18 s / 45-slot baseline and "
                "clears `S >= 2.10` under 8-point moment-reduced age quadrature."
                if h12_matches_default_baseline
                else "This h=12 row is regenerated under this artifact's live-pair cutoff "
                "and clears `S >= 2.10` under 8-point moment-reduced age quadrature."
            ),
            "",
        ])
    if report["stress_rows"]:
        lines.extend([
            "## Optional stress extension",
            "",
            "These rows are outside the calibrated production range and are reported only as a depth stress test under the same 10 km/link policy.",
            "",
            _markdown_table(report["stress_rows"]),
            "",
        ])
    if report["primary_failures"]:
        lines.extend([
            "## Primary failures",
            "",
            "At least one primary row failed a calibration invariant:",
            "",
            "```json",
            json.dumps(_json_ready(report["primary_failures"]), indent=2, sort_keys=True),
            "```",
            "",
        ])
    else:
        lines.extend([
            "## Acceptance summary",
            "",
            f"All h=1..{primary_max_h} rows satisfy `A_route >= 0.95`, `S >= 2.10`, QBER numerically zero under pure-dephasing/key-Z assumptions, and probability normalization within tolerance.",
            "",
        ])
    lines.extend([
        "## Scope note",
        "",
        "This is a hop-count calibration at 10 km per elementary link. It deliberately does not provision arbitrary heterogeneous fiber-length vectors. Those should use a separate length-aware provisioning model.",
        "",
    ])
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-hop", type=int, default=12)
    parser.add_argument("--stress-max-hop", type=int, default=15)
    parser.add_argument("--edge-length-km", type=float, default=10.0)
    parser.add_argument("--accepted-alive-time-s", type=float, default=0.18)
    parser.add_argument("--route-availability-target", type=float, default=0.95)
    parser.add_argument("--chsh-target", type=float, default=2.10)
    parser.add_argument("--age-points-per-edge", type=int, default=8)
    parser.add_argument("--selection-policy", choices=("youngest_live", "random_live"), default="youngest_live")
    parser.add_argument("--hardware-config-dir", default="configs/physical")
    parser.add_argument("--detector-kind", default="pnr_snspd")
    parser.add_argument("--detector-profile", default=str(DEFAULT_DETECTOR_PROFILE_PATH))
    parser.add_argument("--detector-response-file", default=None)
    parser.add_argument("--fiber-response-file", default=None)
    parser.add_argument("--conversion-response-file", default=str(DEFAULT_CONVERSION_RESPONSE_PATH))
    parser.add_argument("--multiplexing-controller", default=None)
    parser.add_argument("--allow-wavelength-mismatch", action="store_true")
    parser.add_argument("--allow-ideal-detector-reference", action="store_true")
    parser.add_argument("--fock-cutoff", type=int, default=2)
    parser.add_argument("--key-length", type=int, default=8)
    parser.add_argument("--c-max", type=int, default=64)
    parser.add_argument("--rolling-age-support-product-limit", type=int, default=1_000_000)
    parser.add_argument("--ignore-detected-decay-in-pool", action="store_true")
    parser.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    parser.add_argument("--out-md", default=str(DEFAULT_OUT_MD))
    return parser


def _config_from_args(args: argparse.Namespace) -> HopCalibrationConfig:
    return HopCalibrationConfig(
        max_hop=int(args.max_hop),
        stress_max_hop=int(args.stress_max_hop),
        edge_length_km=float(args.edge_length_km),
        accepted_alive_time_s=float(args.accepted_alive_time_s),
        route_availability_target=float(args.route_availability_target),
        chsh_target=float(args.chsh_target),
        age_points_per_edge=int(args.age_points_per_edge),
        selection_policy=str(args.selection_policy),
        hardware_config_dir=str(args.hardware_config_dir),
        detector_kind=str(args.detector_kind),
        detector_profile_path=args.detector_profile,
        detector_response_path=args.detector_response_file,
        fiber_response_path=args.fiber_response_file,
        conversion_response_path=args.conversion_response_file,
        multiplexing_controller_path=args.multiplexing_controller,
        allow_wavelength_mismatch=bool(args.allow_wavelength_mismatch),
        allow_ideal_detector_reference=bool(args.allow_ideal_detector_reference),
        fock_cutoff=int(args.fock_cutoff),
        key_length=int(args.key_length),
        c_max=int(args.c_max),
        rolling_age_support_product_limit=int(args.rolling_age_support_product_limit),
        ignore_detected_decay_in_pool=bool(args.ignore_detected_decay_in_pool),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = derive_hop_calibration(_config_from_args(args))
    except CompiledHistogramUnsupported as exc:
        raise SystemExit(str(exc)) from exc
    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
