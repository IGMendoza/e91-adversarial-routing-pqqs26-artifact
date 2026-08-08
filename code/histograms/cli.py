"""Command-line parsing and orchestration for histogram corpus generation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .compiled import *
from .devices import *
from .fock import *
from .generation import *
from .models import *
from .models import _json_ready
from .output import *
from .timing import active_multiplexing_controller, attempt_period_policy
from .topology import *

DEFAULT_CLI_DETECTOR_PROFILE = (
    DEFAULT_CONFIG_DIR
    / "detectors"
    / "snspd_colangelo_impedance_matched_differential_pnr_1550.json"
)
DEFAULT_CLI_CONVERSION_RESPONSE = (
    DEFAULT_CONFIG_DIR / "conversion" / "cui_dfg_866_to_1558.json"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--device-front-matter", default=None)
    parser.add_argument("--physical-path", default="fock_pnr_snspd")
    parser.add_argument("--source-model", choices=SOURCE_MODELS, default="memory_emission")
    parser.add_argument("--fock-cutoff", type=int, default=DEFAULT_FOCK_CUTOFF)
    parser.add_argument("--detector-kind", default="pnr_snspd")
    parser.add_argument("--detector-profile", default=str(DEFAULT_CLI_DETECTOR_PROFILE))
    parser.add_argument("--detector-response-file", default=None)
    parser.add_argument("--fiber-response-file", default=None)
    parser.add_argument("--conversion-response-file", default=str(DEFAULT_CLI_CONVERSION_RESPONSE))
    parser.add_argument(
        "--multiplexing-controller",
        default=None,
        help=(
            "optional JSON controller for an independent synchronized time-bin bank; disabled by "
            "default, and unsupported device/resource mappings fail closed"
        ),
    )
    parser.add_argument(
        "--repeater-service-profile",
        default=None,
        help=(
            "optional validated literature-scoped asynchronous repeater service profile; "
            "activates its bounded hardware inventory, state channels, storage model, and rate metadata"
        ),
    )
    parser.add_argument("--key-length", type=int, default=256)
    parser.add_argument("--min-hops", type=int, default=1, help="minimum hop count for batch mode; repeaters = hops - 1")
    parser.add_argument("--max-hops", type=int, default=10, help="maximum hop count for batch mode; repeaters = hops - 1")
    parser.add_argument("--hops", type=int, default=None, help="fixed hop count h; repeaters = h - 1")
    parser.add_argument("--repeaters", type=int, default=None, help="fixed repeater count r; equivalent to --hops r+1")
    parser.add_argument("--fiber-length-symbols", default=None)
    parser.add_argument("--fiber-lengths-m", default=None)
    parser.add_argument(
        "--move-set",
        choices=("all", "edge", "repeater", "projective", "delivery", "none"),
        default="all",
        help="Eve move set for the selected-route network stress test",
    )
    parser.add_argument(
        "--attack-set",
        dest="move_set",
        choices=("all", "edge", "repeater", "projective", "delivery", "none"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--memory-side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--emit-symbolic", action="store_true")
    parser.add_argument("--emit-numeric-evaluator", action="store_true")
    parser.add_argument(
        "--emit-compiled-kernels",
        action="store_true",
        help=(
            "emit a compiled length-parametric kernel artifact (<case>.compiled.npz) per case; "
            "CompiledRouteKernel.load(path).evaluate(fiber_lengths_m) then evaluates the frozen "
            "family at arbitrary length vectors orders of magnitude faster than regeneration"
        ),
    )
    parser.add_argument("--emit-full-bsm-audit", action="store_true")
    parser.add_argument("--max-full-bsm-repeaters", type=int, default=2)
    parser.add_argument(
        "--emit-detector-click-audit",
        action="store_true",
        help=(
            "emit raw optical detector click/count records as transcript coordinates; "
            "default production output marginalizes them after detector-conditioned memory-state construction"
        ),
    )
    parser.add_argument("--emit-approximation-audit", action="store_true")
    parser.add_argument("--max-approximation-audit-repeaters", type=int, default=2)
    parser.add_argument("--waiting-tail-tolerance", type=float, default=1e-6)
    parser.add_argument("--waiting-max-attempts", type=int, default=4096)
    parser.add_argument("--waiting-max-history-pairs", type=int, default=250_000)
    parser.add_argument(
        "--waiting-max-wait-signatures",
        type=int,
        default=2048,
        help=(
            "h>=3 compute cap on distinct grouped wait signatures per case; per-edge attempt cutoffs "
            "are shrunk (and recorded) until the grouped enumeration fits, with the omitted geometric "
            "tail booked as timeout/no_delivery"
        ),
    )
    parser.add_argument(
        "--waiting-budget-mode",
        choices=("fixed_attempts", "fixed_wall_clock_time"),
        default="fixed_attempts",
        help=(
            "h>=2 waiting budget semantics: fixed_attempts gives every elementary link "
            "--waiting-max-attempts attempts; fixed_wall_clock_time derives each link's attempt "
            "budget from --waiting-budget-ps and its attempt period"
        ),
    )
    parser.add_argument(
        "--waiting-budget-ps",
        type=float,
        default=None,
        help="wall-clock route-generation budget in picoseconds; required for --waiting-budget-mode=fixed_wall_clock_time",
    )
    parser.add_argument(
        "--waiting-precision-mode",
        choices=("fixed_caps", "adaptive_bounded"),
        default="fixed_caps",
        help=(
            "h>=3 waiting-support policy: fixed_caps uses the configured caps directly; "
            "adaptive_bounded refines caps until the omitted route-ready TV bound reaches tolerance "
            "or a runtime/memory/refinement limit is hit"
        ),
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
        help=(
            "repeater control protocol: single_buffer_swap_asap is the finite-budget route-from-scratch "
            "model; rolling_alive_pair consumes configured live elementary pairs at route request time"
        ),
    )
    parser.add_argument(
        "--rolling-edge-availability",
        default=None,
        help=(
            "rolling_alive_pair only: scalar A or comma-separated per-edge A_i, the probability an "
            "accepted live elementary pair is available at route request time; default 1.0"
        ),
    )
    parser.add_argument(
        "--rolling-age-distribution-kind",
        choices=("discrete",),
        default="discrete",
        help="rolling_alive_pair only: selected live-pair age distribution kind; first pass supports discrete",
    )
    parser.add_argument(
        "--rolling-age-distribution-points-s",
        default="0",
        help="rolling_alive_pair only: comma-separated selected live-pair ages in seconds",
    )
    parser.add_argument(
        "--rolling-age-distribution-weights",
        default="1",
        help="rolling_alive_pair only: comma-separated probabilities/weights for the selected age points",
    )
    parser.add_argument(
        "--rolling-edge-age-distribution-points-s",
        default=None,
        help=(
            "rolling_alive_pair only: semicolon-separated per-edge age supports, e.g. "
            "'0,1e-6;0,2e-6'; mutually exclusive with non-default shared age support"
        ),
    )
    parser.add_argument(
        "--rolling-edge-age-distribution-weights",
        default=None,
        help=(
            "rolling_alive_pair only: semicolon-separated per-edge age weights, e.g. "
            "'0.5,0.5;0.25,0.75'"
        ),
    )
    parser.add_argument(
        "--rolling-age-evaluation",
        choices=("cartesian", "moment_reduced"),
        default="cartesian",
        help=(
            "rolling_alive_pair only: cartesian is the exact route-age product reference; "
            "moment_reduced exactly folds each independent per-edge age distribution into "
            "the detector-conditioned edge state when age affects only hidden storage dephasing"
        ),
    )
    parser.add_argument("--rolling-age-support-product-limit", type=int, default=1_000_000)
    parser.add_argument("--rolling-accepted-alive-time-s", type=float, default=None)
    parser.add_argument("--rolling-memory-slots-per-edge-side", type=int, default=1)
    parser.add_argument("--rolling-selection-policy", default="configured_distribution")
    parser.add_argument("--rolling-age-distribution-source", default="configured")
    parser.add_argument(
        "--memory-cutoff-ps",
        type=float,
        default=None,
        help=(
            "stale-pair cutoff: waiting histories in which any memory's total represented storage age "
            "exceeds this value are discarded to no_delivery (failure stage stale_pair_cutoff); "
            "protocol policy knob, off by default"
        ),
    )
    parser.add_argument("--fail-on-unsupported-device", action="store_true")
    parser.add_argument(
        "--allow-effective-device-fields",
        action="store_true",
        help=(
            "diagnostic override only: permit a selected detector profile to borrow missing "
            "efficiency/dark/timing/rate fields from repository-wide detector scalars"
        ),
    )
    parser.add_argument(
        "--allow-ideal-detector-reference",
        action="store_true",
        help=(
            "permit ideal_pnr only for an explicitly labelled mathematical reference/test run"
        ),
    )
    parser.add_argument(
        "--allow-wavelength-mismatch",
        action="store_true",
        help=(
            "diagnostic override only: allow source/fiber/detector wavelength mismatches to be emitted "
            "as metadata instead of fail-closed blockers"
        ),
    )
    parser.add_argument(
        "--allow-generic-memory-fidelity-depolarizing",
        action="store_true",
        help=(
            "diagnostic override only: reproduce legacy p_depol=1-memory.fidelity constructor noise; "
            "not calibrated memory physics and still blocked by fail-closed hardware-realism checks"
        ),
    )
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    run_wall_started_ns = time.perf_counter_ns()
    run_cpu_started_ns = time.process_time_ns()
    args = build_arg_parser().parse_args(argv)
    if args.key_length < 1:
        raise SystemExit("--key-length must be >= 1")
    if args.waiting_tail_tolerance < 0.0:
        raise SystemExit("--waiting-tail-tolerance must be >= 0")
    if args.waiting_max_attempts < 1:
        raise SystemExit("--waiting-max-attempts must be >= 1")
    if args.waiting_max_history_pairs < 1:
        raise SystemExit("--waiting-max-history-pairs must be >= 1")
    if args.waiting_max_wait_signatures < 1:
        raise SystemExit("--waiting-max-wait-signatures must be >= 1")
    if args.waiting_tv_tolerance < 0.0:
        raise SystemExit("--waiting-tv-tolerance must be >= 0")
    if args.waiting_metric_tolerance < 0.0:
        raise SystemExit("--waiting-metric-tolerance must be >= 0")
    if args.waiting_max_refinement_rounds < 0:
        raise SystemExit("--waiting-max-refinement-rounds must be >= 0")
    if args.waiting_max_runtime_s is not None and args.waiting_max_runtime_s <= 0.0:
        raise SystemExit("--waiting-max-runtime-s must be positive when supplied")
    if args.waiting_max_memory_mb is not None and args.waiting_max_memory_mb <= 0.0:
        raise SystemExit("--waiting-max-memory-mb must be positive when supplied")
    if args.rolling_accepted_alive_time_s is not None and args.rolling_accepted_alive_time_s < 0.0:
        raise SystemExit("--rolling-accepted-alive-time-s must be nonnegative when supplied")
    if args.rolling_memory_slots_per_edge_side < 1:
        raise SystemExit("--rolling-memory-slots-per-edge-side must be >= 1")
    if len(parse_optional_float_sequence(args.rolling_age_distribution_points_s) or ()) != len(
        parse_optional_float_sequence(args.rolling_age_distribution_weights) or ()
    ):
        raise SystemExit("--rolling-age-distribution-points-s and --rolling-age-distribution-weights must have equal length")
    edge_age_points = parse_optional_nested_float_sequence(args.rolling_edge_age_distribution_points_s)
    edge_age_weights = parse_optional_nested_float_sequence(args.rolling_edge_age_distribution_weights)
    if (edge_age_points is None) != (edge_age_weights is None):
        raise SystemExit("--rolling-edge-age-distribution-points-s requires matching --rolling-edge-age-distribution-weights")
    if edge_age_points is not None:
        shared_points = parse_optional_float_sequence(args.rolling_age_distribution_points_s) or (0.0,)
        shared_weights = parse_optional_float_sequence(args.rolling_age_distribution_weights) or (1.0,)
        if tuple(shared_points) != (0.0,) or tuple(shared_weights) != (1.0,):
            raise SystemExit("use either shared rolling age distribution or per-edge rolling age distribution, not both")
        if len(edge_age_points) != len(edge_age_weights):
            raise SystemExit("per-edge rolling age points/weights must provide the same number of edge rows")
        for index, (points, weights) in enumerate(zip(edge_age_points, edge_age_weights)):
            if len(points) != len(weights):
                raise SystemExit(f"per-edge rolling age row {index} points/weights length mismatch")
    if args.rolling_age_support_product_limit < 1:
        raise SystemExit("--rolling-age-support-product-limit must be >= 1")
    if args.memory_cutoff_ps is not None and args.memory_cutoff_ps <= 0.0:
        raise SystemExit("--memory-cutoff-ps must be positive when supplied")
    if args.waiting_budget_mode == "fixed_wall_clock_time" and (
        args.waiting_budget_ps is None or args.waiting_budget_ps <= 0.0
    ):
        raise SystemExit("--waiting-budget-mode=fixed_wall_clock_time requires a positive --waiting-budget-ps")
    context = build_model_context(args)
    front = load_or_build_front_matter(
        Path(args.hardware_config_dir),
        device_front_matter=Path(args.device_front_matter) if args.device_front_matter else None,
        fail_on_unsupported=args.fail_on_unsupported_device,
    )
    front = apply_repeater_service_front_overrides(front, context)
    service_blockers = repeater_service_activation_blockers(context)
    if service_blockers:
        raise SystemExit(
            "repeater service profile check failed: " + "; ".join(service_blockers)
        )
    empirical_pair_instrument = repeater_elementary_pair_instrument(context)
    try:
        run_resolved_detector_kind = resolved_detector_kind(front, context)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if run_resolved_detector_kind == "ideal_pnr" and not args.allow_ideal_detector_reference:
        raise SystemExit(
            "ideal_pnr is a mathematical reference detector, not a demonstrated device; "
            "use --allow-ideal-detector-reference only for a labelled reference/test run"
        )
    run_detector_profile = detector_profile_metadata(front, context)
    run_detector_max = detector_max_count(context, run_detector_profile)
    run_pnr_meta = pnr_response_model_metadata(
        run_detector_profile,
        run_resolved_detector_kind,
        run_detector_max,
    )
    selected_wavelength_report = wavelength_compatibility_report(front, context)
    run_wavelength_report = (
        {
            "status": "aggregated_empirical_link_stack",
            "issues": [],
            "inactive_repository_stack_report": selected_wavelength_report,
            "semantics": (
                "the demonstrated link rate/state already aggregate source, conversion, "
                "fiber, stabilization, and detection; repository optical components are inactive"
            ),
        }
        if empirical_pair_instrument is not None else selected_wavelength_report
    )
    detector_blockers = (
        []
        if empirical_pair_instrument is not None else
        explicit_detector_operating_point_blockers(front, context)
    )
    if detector_blockers and not args.allow_effective_device_fields:
        details = "; ".join(
            f"{row['effect']} ({row['parameter']}): {row['reason']}"
            for row in detector_blockers
        )
        raise SystemExit(
            "explicit detector operating-point check failed: " + details
            + "; use --allow-effective-device-fields only for a labelled diagnostic composite"
        )
    wavelength_blockers = [
        row for row in run_wavelength_report["issues"]
        if row.get("status") == "missing"
    ]
    if (
        wavelength_blockers
        and not args.allow_wavelength_mismatch
        and not args.fail_on_unsupported_device
    ):
        details = "; ".join(
            f"{row['effect']} ({row['parameter']}): {row['reason']}"
            for row in wavelength_blockers
        )
        raise SystemExit(
            "wavelength compatibility check failed: " + details
            + "; use --allow-wavelength-mismatch only for a labelled diagnostic run"
        )

    controller_model = None
    if context.multiplexing_controller is not None:
        try:
            controller_model = active_multiplexing_controller(context)
            blockers = controller_model.execution_blockers() if controller_model else ()
            if blockers:
                raise NotImplementedError("; ".join(blockers))
            for hops in hop_range(args):
                if hops >= 2 and context.waiting_protocol != "rolling_alive_pair":
                    raise NotImplementedError(
                        "multiplexed h>=2 generation requires calibrated rolling_alive_pair inputs; "
                        "within-round herald offsets are not collapsed into swap-asap round boundaries"
                    )
                attempt_period_policy(
                    make_route(hops, parse_lengths(hops, args.fiber_lengths_m)),
                    front,
                    context,
                )
        except (NotImplementedError, TypeError, ValueError) as exc:
            raise SystemExit(f"multiplexing controller check failed: {exc}") from exc

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    front_path = out_dir / "histogram_device_front_matter.json"
    front_output = dict(front)
    front_output["histogram_model_context"] = context.to_dict()
    front_output["physical_model_context_hash"] = stable_hash(context.to_dict())
    with open(front_path, "w", encoding="utf-8") as handle:
        json.dump(_json_ready(front_output), handle, indent=2, sort_keys=True)
        handle.write("\n")

    cases = []
    warnings = detector_profile_warnings(run_detector_profile, run_resolved_detector_kind, run_detector_max)
    if run_wavelength_report["issues"]:
        warnings.append(
            "wavelength compatibility issues: "
            + "; ".join(issue["effect"] for issue in run_wavelength_report["issues"])
        )
    total_bins = 0
    for hops in hop_range(args):
        path_limitations = physical_path_limitations(front, context, hop_count=hops)
        if args.fail_on_unsupported_device and path_limitations:
            details = "; ".join(
                f"{row.get('effect', row['parameter'])} ({row['parameter']}): {row['classification']}"
                for row in path_limitations
            )
            raise SystemExit(f"--fail-on-unsupported-device blocked {context.physical_path} h={hops}: {details}")
        lengths = parse_lengths(hops, args.fiber_lengths_m)
        symbols = parse_symbols(hops, args.fiber_length_symbols)
        route = make_route(hops, lengths)
        route["fiber_length_symbols"] = symbols
        for attack in moves_for_route(route, args.move_set, args.memory_side):
            case_wall_started_ns = time.perf_counter_ns()
            case_cpu_started_ns = time.process_time_ns()
            kernel = FrozenHistogramKernel(
                front=front,
                context=context,
                hop_count=hops,
                attack=attack,
                key_length=args.key_length,
                fiber_length_symbols=tuple(symbols),
                emit_full_bsm_audit=args.emit_full_bsm_audit,
                max_full_bsm_repeaters=args.max_full_bsm_repeaters,
                emit_detector_click_audit=args.emit_detector_click_audit,
                emit_approximation_audit=args.emit_approximation_audit,
                max_approximation_audit_repeaters=args.max_approximation_audit_repeaters,
            )
            distribution = kernel.evaluate(lengths)
            case = distribution.to_case()
            if args.seed is not None:
                case["metadata"]["optional_finite_sample"] = finite_sample_counts(
                    case["rows"], args.key_length, args.seed
                )
            cid, bins = write_case(out_dir, case)
            compiled_kernel_path = None
            if args.emit_compiled_kernels:
                if args.emit_full_bsm_audit or args.emit_detector_click_audit or args.emit_approximation_audit:
                    warnings.append(
                        f"{cid}: compiled kernel skipped (audit flags change theta; compiled kernels "
                        "are emitted for the default public/detector-record compression only)"
                    )
                else:
                    try:
                        compiled_kernel = CompiledRouteKernel.compile(
                            front=front,
                            context=context,
                            hop_count=hops,
                            attack=attack,
                            key_length=args.key_length,
                            fiber_length_symbols=tuple(symbols),
                        )
                        compiled_kernel_path = str(out_dir / f"{cid}.compiled.npz")
                        compiled_kernel.save(compiled_kernel_path)
                    except NotImplementedError as exc:
                        warnings.append(f"{cid}: compiled kernel skipped ({exc})")
            case_wall_seconds = (time.perf_counter_ns() - case_wall_started_ns) / 1e9
            case_cpu_seconds = (time.process_time_ns() - case_cpu_started_ns) / 1e9
            chain_contract = case["metadata"]["linear_chain_contract"]
            move_contract = case["metadata"]["eve_move_contract"]
            case_length_contract = case["metadata"]["length_contract"]
            cases.append({
                "case_id": cid,
                "compiled_kernel_path": compiled_kernel_path,
                "bin_count": bins,
                "repeater_count": chain_contract["repeater_count"],
                "hop_count": chain_contract["hop_count"],
                "edge_count": chain_contract["edge_count"],
                "move_id": attack.move_id,
                "move_kind": attack.move_kind,
                "target_kind": move_contract["target_kind"],
                "target_index": move_contract["target_index"],
                "target_id": move_contract["target_id"],
                "frozen_theta_hash": case["metadata"]["frozen_theta_hash"],
                "length_contract_status": case_length_contract["length_parametric_status"],
                "rows_probability_status": case_length_contract["rows_probability_status"],
                "fiber_length_symbols": case_length_contract["fiber_length_symbols"],
                "detector_click_record_policy": case["metadata"]["detector_click_record_policy"],
                "runtime_timing": {
                    "wall_seconds": case_wall_seconds,
                    "cpu_seconds": case_cpu_seconds,
                    "scope": "kernel evaluation, case serialization, and optional compiled-kernel emission",
                },
            })
            total_bins += bins
            if not case["metadata"]["validation_checks"]["probabilities_sum_to_one"]:
                warnings.append(f"{cid}: probabilities do not sum to one")
    if args.emit_symbolic:
        warnings.append("symbolic probability emission is partial; matrix-valued endpoint instruments are numeric-evaluator-only")
    pre_manifest_wall_seconds = (time.perf_counter_ns() - run_wall_started_ns) / 1e9
    pre_manifest_cpu_seconds = (time.process_time_ns() - run_cpu_started_ns) / 1e9
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "device_front_matter_path": str(front_path),
        "device_front_matter_hash": front["front_matter_hash"],
        "physical_model_context_hash": stable_hash(context.to_dict()),
        "physical_path": context.physical_path,
        "source_model": context.source_model,
        "detector_kind": context.detector_kind,
        "resolved_detector_kind": run_resolved_detector_kind,
        "pnr_response_model": run_pnr_meta["pnr_response_model"],
        "pnr_response_model_is_calibrated_to_profile": run_pnr_meta[
            "pnr_response_model_is_calibrated_to_profile"
        ],
        "fock_cutoff": context.fock_cutoff,
        "waiting_protocol": context.waiting_protocol,
        "detector_profile_path": context.detector_profile_path,
        "detector_response_loaded": context.detector_response is not None,
        "fiber_response_loaded": context.fiber_response is not None,
        "conversion_response_loaded": context.conversion_response is not None,
        "device_adherence": {
            "detector_operating_point_status": (
                "diagnostic_repository_field_fallback"
                if detector_blockers else "explicit_selected_detector_operating_point"
            ),
            "detector_operating_point_blockers": detector_blockers,
            "allow_effective_device_fields": bool(args.allow_effective_device_fields),
            "allow_ideal_detector_reference": bool(args.allow_ideal_detector_reference),
            "allow_wavelength_mismatch": bool(args.allow_wavelength_mismatch),
        },
        "wavelength_compatibility": run_wavelength_report,
        "cases_generated": cases,
        "total_bin_counts": total_bins,
        "max_hop_count": max(hop_range(args)),
        "max_repeater_count": max(hop_range(args)) - 1,
        "move_set": args.move_set,
        "move_generation_mode": "batch_move_set",
        "route_input_contract": {
            "fixed_repeaters_cli": "--repeaters r sets hop_count h = r + 1",
            "fixed_hops_cli": "--hops h sets repeater_count r = h - 1",
            "batch_hops_cli": "--min-hops/--max-hops sweep hop_count h directly; repeater_count is h - 1",
            "fiber_vector": "[F1, F2, ..., Fh] in edge order e0, e1, ..., e{h-1}",
        },
        "key_length": args.key_length,
        "approximation_audit_enabled": args.emit_approximation_audit,
        "max_approximation_audit_repeaters": args.max_approximation_audit_repeaters,
        "detector_click_record_policy": detector_click_record_policy_metadata(
            emit_detector_click_audit=args.emit_detector_click_audit
        ),
        "symbolic_numeric_mode": "numeric_evaluator" + ("_with_partial_symbolic_delivery_formulas" if args.emit_symbolic else ""),
        "runtime_timing": {
            "wall_seconds_before_manifest_write": pre_manifest_wall_seconds,
            "cpu_seconds_before_manifest_write": pre_manifest_cpu_seconds,
            "case_wall_seconds_total": sum(case["runtime_timing"]["wall_seconds"] for case in cases),
            "case_cpu_seconds_total": sum(case["runtime_timing"]["cpu_seconds"] for case in cases),
            "clock": "time.perf_counter_ns and time.process_time_ns",
            "scope": "CLI validation, front-matter loading, and all case generation before manifest serialization",
        },
        "warnings": warnings,
    }
    if controller_model is not None:
        manifest["multiplexing_controller"] = {
            "status": "active_exact_independent_time_bin_round_reduction",
            "path": args.multiplexing_controller,
            "controller": controller_model.to_dict(),
            "controller_hash": stable_hash(context.multiplexing_controller),
        }
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(_json_ready(manifest), handle, indent=2, sort_keys=True)
        handle.write("\n")
    if args.emit_numeric_evaluator:
        write_numeric_evaluator(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
