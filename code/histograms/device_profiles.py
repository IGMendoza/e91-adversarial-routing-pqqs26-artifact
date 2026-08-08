"""Model-context parsing, wavelength compatibility, and detector profiles."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .device_models import *
from .models import *
def normalize_physical_path(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "fock_pnr_snspd":
        normalized = "fock_pnr_snspd"
    if normalized not in PHYSICAL_PATHS:
        raise SystemExit(f"--physical-path must be one of {PHYSICAL_PATHS}, got {value!r}")
    return normalized


def normalize_detector_kind(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"threshold_snsPD".lower(), "threshold_snspd"}:
        return "threshold_snspd"
    if normalized in {"pnr_snsPD".lower(), "pnr_snspd"}:
        return "pnr_snspd"
    if normalized in {"ideal_pnr", "repository_default"}:
        return normalized
    raise SystemExit(f"--detector-kind must be one of {DETECTOR_KINDS}, got {value!r}")


def load_json_optional(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with open(Path(path), encoding="utf-8") as handle:
        return json.load(handle)


def build_model_context(args: argparse.Namespace | None = None) -> ModelContext:
    from .output import parse_optional_float_sequence, parse_optional_nested_float_sequence

    if args is None:
        return ModelContext()
    physical_path = normalize_physical_path(args.physical_path)
    detector_kind = normalize_detector_kind(args.detector_kind)
    source_model = args.source_model.strip().lower()
    if source_model not in SOURCE_MODELS:
        raise SystemExit(f"--source-model must be one of {SOURCE_MODELS}, got {args.source_model!r}")
    if args.fock_cutoff < 1:
        raise SystemExit("--fock-cutoff must be >= 1")
    return ModelContext(
        physical_path=physical_path,
        source_model=source_model,
        detector_kind=detector_kind,
        fock_cutoff=int(args.fock_cutoff),
        detector_profile_path=args.detector_profile,
        detector_profile=load_json_optional(args.detector_profile),
        detector_response=load_json_optional(args.detector_response_file),
        fiber_response=load_json_optional(args.fiber_response_file),
        conversion_response=load_json_optional(args.conversion_response_file),
        multiplexing_controller=load_json_optional(args.multiplexing_controller),
        repeater_service_profile_path=args.repeater_service_profile,
        repeater_service_profile=load_json_optional(args.repeater_service_profile),
        allow_wavelength_mismatch=bool(args.allow_wavelength_mismatch),
        allow_generic_memory_fidelity_depolarizing=bool(
            args.allow_generic_memory_fidelity_depolarizing
        ),
        waiting_tail_tolerance=float(args.waiting_tail_tolerance),
        waiting_max_attempts=int(args.waiting_max_attempts),
        waiting_max_history_pairs=int(args.waiting_max_history_pairs),
        waiting_max_wait_signatures=int(args.waiting_max_wait_signatures),
        waiting_budget_mode=str(args.waiting_budget_mode),
        waiting_budget_wall_clock_ps=(
            float(args.waiting_budget_ps) if args.waiting_budget_ps is not None else None
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
        rolling_edge_availability=parse_optional_float_sequence(args.rolling_edge_availability),
        rolling_age_distribution_kind=str(args.rolling_age_distribution_kind),
        rolling_age_distribution_points_s=tuple(
            parse_optional_float_sequence(args.rolling_age_distribution_points_s) or (0.0,)
        ),
        rolling_age_distribution_weights=tuple(
            parse_optional_float_sequence(args.rolling_age_distribution_weights) or (1.0,)
        ),
        rolling_edge_age_distribution_points_s=parse_optional_nested_float_sequence(
            args.rolling_edge_age_distribution_points_s
        ),
        rolling_edge_age_distribution_weights=parse_optional_nested_float_sequence(
            args.rolling_edge_age_distribution_weights
        ),
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
    )


def active_wavelength_nm(front: dict[str, Any], context: ModelContext) -> float:
    if context.source_model == "memory_emission":
        return float(front["device_constants"]["memory"]["wavelength_nm"])
    return float(front["device_constants"]["source"]["wavelength_nm"])


def conversion_output_wavelength_nm(context: ModelContext) -> float | None:
    if context.conversion_response is None:
        return None
    for key in (
        "output_wavelength_nm",
        "target_wavelength_nm",
        "to_wavelength_nm",
        "converted_wavelength_nm",
    ):
        value = _float_or_none(context.conversion_response.get(key))
        if value is not None:
            return value
    return None


def wavelength_stages(front: dict[str, Any], context: ModelContext) -> dict[str, Any]:
    source_wavelength = active_wavelength_nm(front, context)
    conversion_output = conversion_output_wavelength_nm(context)
    propagation_wavelength = conversion_output if conversion_output is not None else source_wavelength
    conversion_changes_wavelength = conversion_output is not None and not bool(
        wavelength_close(source_wavelength, conversion_output)
    )
    if context.conversion_response is None:
        conversion_status = "missing_no_frequency_conversion_model"
    elif conversion_output is None:
        conversion_status = (
            "generic_scalar_or_curve_throughput_only_no_wavelength_change_no_noise_or_mode_model"
        )
    else:
        conversion_status = (
            "generic_scalar_or_curve_throughput_with_output_wavelength_no_noise_bandwidth_"
            "temporal_mode_or_polarization_model"
        )
    return {
        "source_wavelength_nm": source_wavelength,
        "conversion_input_wavelength_nm": source_wavelength,
        "conversion_output_wavelength_nm": conversion_output,
        "propagation_wavelength_nm": propagation_wavelength,
        "detector_wavelength_nm": propagation_wavelength,
        "conversion_changes_wavelength": conversion_changes_wavelength,
        "conversion_model_status": conversion_status,
        "conversion_model_limitations": (
            "conversion throughput is a scalar/curve factor only; conversion noise, added photons, "
            "bandwidth, temporal-mode distortion, and polarization effects are not modeled"
            if context.conversion_response is not None else
            "no conversion/filter channel is loaded"
        ),
    }


def _curve_points(curve: dict[str, Any], value_key: str) -> list[tuple[float, float]]:
    if "points" in curve:
        return [(float(p["wavelength_nm"]), float(p[value_key])) for p in curve["points"]]
    wavelengths = curve.get("wavelength_nm") or curve.get("wavelengths_nm")
    values = curve.get(value_key)
    if wavelengths is None or values is None:
        raise ValueError(f"response curve needs wavelength_nm and {value_key!r}")
    return [(float(w), float(v)) for w, v in zip(wavelengths, values)]


def response_curve_wavelength_range(
    curve: dict[str, Any] | None,
    value_key: str,
) -> list[float] | None:
    if curve is None:
        return None
    points = sorted(_curve_points(curve, value_key))
    if not points:
        raise ValueError("empty response curve")
    return [float(points[0][0]), float(points[-1][0])]


def evaluate_response_curve(
    curve: dict[str, Any] | None,
    *,
    wavelength_nm: float,
    value_key: str,
    scalar_default: float,
    quantity: str,
) -> tuple[float, dict[str, Any]]:
    if curve is None:
        return float(scalar_default), {
            "quantity": quantity,
            "wavelength_nm": wavelength_nm,
            "wavelength_dependency": "scalar_flat_response_assumption",
            "value": float(scalar_default),
            "source": "repository scalar config",
        }
    points = sorted(_curve_points(curve, value_key))
    if not points:
        raise ValueError("empty response curve")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    in_range = xs[0] <= wavelength_nm <= xs[-1]
    if wavelength_nm <= xs[0]:
        value = ys[0]
    elif wavelength_nm >= xs[-1]:
        value = ys[-1]
    else:
        value = float(np.interp(wavelength_nm, xs, ys))
    return value, {
        "quantity": quantity,
        "wavelength_nm": wavelength_nm,
        "wavelength_dependency": "interpolated_response_curve",
        "value": value,
        "curve_source": curve.get("reference_name") or curve.get("citation_key") or "response_file",
        "curve_range_nm": [xs[0], xs[-1]],
        "wavelength_in_curve_range": in_range,
        "out_of_range_policy": "clamp_to_nearest_endpoint_for_diagnostics_only",
    }


def detector_profile_metadata(front: dict[str, Any], context: ModelContext) -> dict[str, Any]:
    if context.detector_profile is not None:
        return context.detector_profile
    detector = front["device_constants"]["detector"]
    return {
        "reference_name": detector["model_name"],
        "citation_key": front["mapped_device_parameters"].get("detector.efficiency", {}).get("reference_tag", ""),
        "wavelength_range_nm": "not specified; scalar detector fields are used as flat response",
        "system_detection_efficiency": detector["efficiency"],
        "dark_count_rate_per_s": detector["dark_count_per_s"],
        "timing_jitter_or_resolution_ps": detector["time_resolution_ps"],
        "dead_time_or_max_count_rate_hz": detector["count_rate_hz"],
        "pnr_capability": "not_claimed_by_repository_default",
        "max_resolvable_photon_number": None,
        "values_are_scalar_or_curves": "scalar",
    }


def explicit_detector_operating_point_blockers(
    front: dict[str, Any],
    context: ModelContext,
) -> list[dict[str, Any]]:
    """Fields that would otherwise be borrowed from another detector profile.

    This is intentionally stricter than the diagnostic kernel, which retains
    repository-scalar fallbacks for sensitivity studies.  Production callers
    use these rows to reject cross-device composite operating points.
    """

    from .fock import resolved_detector_kind

    resolved_kind = resolved_detector_kind(front, context)
    if resolved_kind == "ideal_pnr":
        return []
    if context.detector_profile is None:
        return [{
            "effect": "explicit_detector_profile_missing",
            "parameter": "detector_profile_path",
            "status": "missing",
            "reason": (
                "a production histogram must select one detector operating-point profile instead "
                "of inheriting repository-wide detector scalars"
            ),
        }]
    profile = context.detector_profile
    required_profile_fields = {
        "system_detection_efficiency": context.detector_response is not None,
        "dark_count_rate_per_s": False,
        "timing_jitter_or_resolution_ps": profile.get("timing_jitter_response") is not None,
        "dead_time_or_max_count_rate_hz": False,
    }
    return [
        {
            "effect": "detector_profile_field_missing",
            "parameter": f"detector_profile.{field_name}",
            "status": "missing",
            "reason": (
                "strict device adherence forbids filling a missing selected-detector "
                "operating-point field from the repository detector scalar"
            ),
        }
        for field_name, supplied_elsewhere in required_profile_fields.items()
        if profile.get(field_name) is None and not supplied_elsewhere
    ]


def detector_count_rate_model(
    front: dict[str, Any],
    context: ModelContext,
) -> dict[str, Any]:
    """Resolve the selected detector's repetition-rate constraint.

    Demonstrated detector profiles may have reset times far slower than the
    repository-default detector.  Waiting/attempt timing must therefore use
    the selected profile's rate when it is supplied.  Timing-only profiles
    without a rate retain the repository scalar as an explicit fallback.
    Ideal detectors impose no detector-rate constraint.
    """

    if context.detector_kind == "ideal_pnr":
        return {
            "count_rate_hz": None,
            "source": "ideal_detector_override_no_rate_constraint",
            "profile_field": None,
            "mapping_semantics": "ideal detector contributes no repetition-rate lower bound",
        }
    profile = detector_profile_metadata(front, context)
    profile_rate = _float_or_none(profile.get("dead_time_or_max_count_rate_hz"))
    if profile_rate is not None:
        if profile_rate <= 0.0:
            raise ValueError("detector profile dead_time_or_max_count_rate_hz must be positive")
        return {
            "count_rate_hz": profile_rate,
            "source": "selected_detector_profile",
            "profile_field": "dead_time_or_max_count_rate_hz",
            "profile_reference_name": profile.get("reference_name"),
            "profile_scope": profile.get("dead_time_or_max_count_rate_scope"),
            "mapping_semantics": (
                "selected profile maximum count/reset rate mapped to a conservative global "
                "inter-attempt lower bound; per-detector recovery state is not modeled"
            ),
        }
    repository_rate = float(front["device_constants"]["detector"]["count_rate_hz"])
    if repository_rate <= 0.0:
        raise ValueError("detector.count_rate must be positive")
    return {
        "count_rate_hz": repository_rate,
        "source": "repository_scalar_fallback_profile_rate_not_reported",
        "profile_field": "detector.count_rate",
        "profile_reference_name": profile.get("reference_name"),
        "mapping_semantics": (
            "repository detector count rate mapped to a conservative global inter-attempt lower bound "
            "because the selected profile reports no rate; per-detector recovery state is not modeled"
        ),
    }


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detector_wavelength_range_nm(profile: dict[str, Any]) -> list[float] | None:
    raw = profile.get("wavelength_range_nm")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        lo = _float_or_none(raw[0])
        hi = _float_or_none(raw[1])
        if lo is not None and hi is not None:
            return [min(lo, hi), max(lo, hi)]
    scalar = _float_or_none(
        profile.get("operating_wavelength_nm")
        if profile.get("operating_wavelength_nm") is not None else
        profile.get("photon_number_resolution_wavelength_nm")
    )
    if scalar is not None:
        return [scalar, scalar]
    return None


def detector_operating_wavelength_nm(profile: dict[str, Any], resolved_kind: str) -> float | None:
    if resolved_kind == "ideal_pnr":
        return None
    scalar = _float_or_none(
        profile.get("operating_wavelength_nm")
        if profile.get("operating_wavelength_nm") is not None else
        profile.get("photon_number_resolution_wavelength_nm")
    )
    if scalar is not None:
        return scalar
    rng = detector_wavelength_range_nm(profile)
    if rng is not None and abs(rng[0] - rng[1]) <= 1e-9:
        return rng[0]
    return None


def wavelength_in_range(wavelength_nm: float, wavelength_range_nm: list[float] | None) -> bool | None:
    if wavelength_range_nm is None:
        return None
    return wavelength_range_nm[0] - 1e-9 <= wavelength_nm <= wavelength_range_nm[1] + 1e-9


def wavelength_close(left: float | None, right: float | None, tolerance_nm: float = 1.0) -> bool | None:
    if left is None or right is None:
        return None
    return abs(left - right) <= tolerance_nm


def wavelength_compatibility_report(front: dict[str, Any], context: ModelContext) -> dict[str, Any]:
    from .fock import resolved_detector_kind

    stages = wavelength_stages(front, context)
    source_wavelength = stages["source_wavelength_nm"]
    propagation_wavelength = stages["propagation_wavelength_nm"]
    conversion_output = stages["conversion_output_wavelength_nm"]
    conversion_changes_wavelength = stages["conversion_changes_wavelength"]
    fiber = front["device_constants"]["fiber_channel"]
    profile = detector_profile_metadata(front, context)
    resolved_kind = resolved_detector_kind(front, context)
    detector_range = detector_wavelength_range_nm(profile)
    detector_operating = detector_operating_wavelength_nm(profile, resolved_kind)
    fiber_reference = fiber.get("attenuation_reference_wavelength_nm")
    fiber_reference_nm = _float_or_none(fiber_reference)
    conversion_loaded = context.conversion_response is not None
    conversion_input_range = response_curve_wavelength_range(
        context.conversion_response,
        "throughput",
    )
    detector_response_range = response_curve_wavelength_range(
        context.detector_response,
        "efficiency",
    )
    fiber_response_range = response_curve_wavelength_range(
        context.fiber_response,
        "attenuation_db_per_m",
    )
    source_matches_scalar_fiber = (
        None if context.fiber_response is not None
        else wavelength_close(source_wavelength, fiber_reference_nm)
    )
    source_detector_in_range = (
        True if resolved_kind == "ideal_pnr"
        else wavelength_in_range(source_wavelength, detector_range)
    )
    conversion_needed = (
        source_matches_scalar_fiber is False
        or source_detector_in_range is False
    )
    issues: list[dict[str, Any]] = []
    if context.physical_path == "fock_pnr_snspd":
        if conversion_input_range is not None and wavelength_in_range(
            source_wavelength, conversion_input_range
        ) is False:
            issues.append({
                "effect": "conversion_input_wavelength_outside_response",
                "parameter": "conversion_response.wavelengths_nm",
                "status": "missing",
                "reason": (
                    f"conversion throughput response range {conversion_input_range} nm does not include "
                    f"the modeled source wavelength {source_wavelength:g} nm; endpoint clamping is "
                    "diagnostic extrapolation, not a calibrated conversion model"
                ),
            })
        if not conversion_loaded and conversion_needed:
            issues.append({
                "effect": "missing_frequency_conversion_model",
                "parameter": "conversion_response",
                "status": "missing",
                "reason": (
                    "no conversion/filter response file is loaded; source photons are treated as propagating "
                    "and reaching the detector at their source wavelength even though the scalar fiber or "
                    "detector wavelength regime does not match"
                ),
            })
        elif conversion_loaded and conversion_output is None and conversion_needed:
            issues.append({
                "effect": "conversion_response_lacks_required_output_wavelength",
                "parameter": "conversion_response.output_wavelength_nm",
                "status": "missing",
                "reason": (
                    "a conversion/filter response is loaded, but it supplies no output wavelength; the kernel "
                    "therefore cannot route photons from the source wavelength to the scalar fiber/detector "
                    "wavelength regime"
                ),
            })
        elif conversion_loaded and not conversion_changes_wavelength:
            issues.append({
                "effect": "conversion_model_is_scalar_throughput_only",
                "parameter": "conversion_response",
                "status": "generic_approximation",
                "reason": (
                    "conversion/filter response affects throughput but does not supply a distinct output "
                    "wavelength or model conversion noise, added photons, bandwidth, temporal modes, or "
                    "polarization effects"
                ),
            })
        elif conversion_loaded:
            issues.append({
                "effect": "conversion_model_is_scalar_throughput_with_output_wavelength",
                "parameter": "conversion_response",
                "status": "generic_approximation",
                "reason": (
                    "conversion/filter response supplies the wavelength handoff used by the kernel and a "
                    "throughput factor, but conversion noise, added photons, bandwidth, temporal-mode "
                    "distortion, and polarization effects are not modeled"
                ),
            })
        if context.fiber_response is None:
            ref_match = wavelength_close(propagation_wavelength, fiber_reference_nm)
            if ref_match is False:
                issues.append({
                    "effect": "scalar_fiber_attenuation_outside_reference_wavelength",
                    "parameter": "fiber_channel.attenuation",
                    "status": "missing",
                    "reason": (
                        f"scalar fiber attenuation is referenced to {fiber_reference_nm:g} nm but is being evaluated at "
                        f"{propagation_wavelength:g} nm without a wavelength response curve or conversion model"
                    ),
                })
            elif ref_match is None:
                issues.append({
                    "effect": "fiber_attenuation_reference_wavelength_unknown",
                    "parameter": "fiber_channel.attenuation",
                    "status": "missing",
                    "reason": "fiber attenuation is scalar and no reference wavelength is recorded",
                })
        elif wavelength_in_range(propagation_wavelength, fiber_response_range) is False:
            issues.append({
                "effect": "fiber_wavelength_outside_response",
                "parameter": "fiber_response.wavelength_nm",
                "status": "missing",
                "reason": (
                    f"fiber response range {fiber_response_range} nm does not include the modeled "
                    f"propagation wavelength {propagation_wavelength:g} nm; endpoint clamping is "
                    "diagnostic extrapolation"
                ),
            })
        if context.detector_response is not None and wavelength_in_range(
            propagation_wavelength, detector_response_range
        ) is False:
            issues.append({
                "effect": "detector_wavelength_outside_response",
                "parameter": "detector_response.wavelength_nm",
                "status": "missing",
                "reason": (
                    f"detector efficiency response range {detector_response_range} nm does not include "
                    f"the modeled photon wavelength {propagation_wavelength:g} nm; endpoint clamping is "
                    "diagnostic extrapolation"
                ),
            })
        if resolved_kind != "ideal_pnr":
            detector_in_range = wavelength_in_range(propagation_wavelength, detector_range)
            if detector_in_range is False:
                issues.append({
                    "effect": "detector_efficiency_wavelength_mismatch",
                    "parameter": "detector.wavelength_range_nm",
                    "status": "missing",
                    "reason": (
                        f"detector profile range {detector_range} nm does not include the modeled photon wavelength "
                        f"{propagation_wavelength:g} nm"
                    ),
                })
            elif detector_in_range is None:
                issues.append({
                    "effect": "detector_operating_wavelength_unspecified",
                    "parameter": "detector.wavelength_range_nm",
                    "status": "missing",
                    "reason": "detector efficiency is scalar and no operating wavelength/range is recorded",
                })
    blocking_issues = [
        row for row in issues
        if row["status"] == "missing" and not context.allow_wavelength_mismatch
    ]
    if blocking_issues:
        status = "incompatible_missing_conversion_or_wavelength_response"
    elif issues:
        status = (
            "diagnostic_override_wavelength_mismatch"
            if context.allow_wavelength_mismatch else
            "compatible_with_generic_scalar_wavelength_approximations"
        )
    else:
        status = "compatible_or_idealized_for_recorded_wavelength_metadata"
    return {
        "source_photon_wavelength_nm": source_wavelength,
        "active_wavelength_nm": source_wavelength,
        "wavelength_stages": stages,
        "conversion_input_wavelength_nm": stages["conversion_input_wavelength_nm"],
        "propagation_wavelength_nm": propagation_wavelength,
        "detector_wavelength_nm": stages["detector_wavelength_nm"],
        "fiber_attenuation_reference_wavelength_nm": fiber_reference_nm,
        "fiber_attenuation_model": (
            "wavelength_response_curve" if context.fiber_response is not None
            else "scalar_attenuation_assumption"
        ),
        "detector_operating_wavelength_nm": detector_operating,
        "detector_wavelength_range_nm": detector_range,
        "detector_efficiency_model": (
            "idealized_detector" if resolved_kind == "ideal_pnr"
            else "wavelength_response_curve" if context.detector_response is not None
            else "profile_or_repository_scalar_efficiency"
        ),
        "conversion_model_status": (
            stages["conversion_model_status"]
        ),
        "conversion_model_limitations": stages["conversion_model_limitations"],
        "conversion_output_wavelength_nm": conversion_output,
        "conversion_input_response_range_nm": conversion_input_range,
        "fiber_response_range_nm": fiber_response_range,
        "detector_response_range_nm": detector_response_range,
        "conversion_changes_wavelength": conversion_changes_wavelength,
        "wavelength_compatibility_status": status,
        "wavelength_model_calibration_status": (
            "not_hardware_calibrated: scalar attenuation/detector responses are used outside a complete platform wavelength stack"
            if issues else
            "recorded_wavelength_metadata_has_no_detected_stack_mismatch"
        ),
        "diagnostic_override_enabled": context.allow_wavelength_mismatch,
        "issues": issues,
    }


def _truthy_profile_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "demonstrated", "pnr", "capable"}
    return bool(value)


def profile_claims_pnr(profile: dict[str, Any]) -> bool:
    return _truthy_profile_flag(profile.get("pnr_capability"))


def profile_max_resolvable_photon_number(profile: dict[str, Any]) -> int:
    value = profile.get("max_resolvable_photon_number")
    if value is None:
        return 1
    return int(value)


def threshold_coarsening_applied(profile: dict[str, Any], detector_kind: str) -> bool:
    return detector_kind == "threshold_snspd" and profile_claims_pnr(profile)


def validate_pnr_confusion_matrix(
    profile: dict[str, Any],
    max_count: int | None = None,
) -> dict[str, Any] | None:
    matrix = profile.get("pnr_confusion_matrix")
    if matrix is None:
        return None
    inputs = [int(v) for v in matrix.get("input_photon_number", [])]
    rows = matrix.get("reported_count_probabilities")
    if not inputs or rows is None:
        raise ValueError("pnr_confusion_matrix needs input_photon_number and reported_count_probabilities")
    if len(inputs) != len(rows):
        raise ValueError("pnr_confusion_matrix row count must match input_photon_number")
    if inputs != list(range(inputs[-1] + 1)):
        raise ValueError("pnr_confusion_matrix input_photon_number must be contiguous from 0")
    limit = profile_max_resolvable_photon_number(profile) if max_count is None else int(max_count)
    if inputs[-1] < limit:
        raise ValueError("pnr_confusion_matrix must include rows through max_resolvable_photon_number")
    width = limit + 1
    normalized_rows: list[list[float]] = []
    for input_n, raw_row in zip(inputs, rows):
        row = [float(v) for v in raw_row]
        if len(row) != width:
            raise ValueError(
                "pnr_confusion_matrix reported_count_probabilities rows must have "
                "max_resolvable_photon_number + 1 columns"
            )
        if any(v < -1e-12 for v in row):
            raise ValueError("pnr_confusion_matrix probabilities must be nonnegative")
        total = sum(row)
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("pnr_confusion_matrix rows must be stochastic")
        normalized_rows.append([max(0.0, v) for v in row])
    return {
        "input_photon_number": inputs,
        "reported_count_probabilities": normalized_rows,
        "max_input_photon_number": inputs[-1],
        "max_reported_count": limit,
    }


def pnr_response_model_metadata(
    profile: dict[str, Any],
    detector_kind: str,
    max_count: int,
) -> dict[str, Any]:
    matrix = validate_pnr_confusion_matrix(profile, max_count)
    if detector_kind == "ideal_pnr":
        return {
            "pnr_capability": True,
            "max_resolvable_photon_number": max_count,
            "pnr_response_model": "ideal_delta_exact_photon_number",
            "pnr_response_model_is_calibrated_to_profile": False,
            "detector_profile_completeness": "idealized_detector_not_demonstrated_hardware_profile",
            "threshold_coarsening_applied": False,
            "calibrated_pnr_confusion_matrix_available": False,
        }
    if detector_kind == "threshold_snspd":
        return {
            "pnr_capability": profile_claims_pnr(profile),
            "max_resolvable_photon_number": max_count,
            "pnr_response_model": "threshold_click_no_click",
            "pnr_response_model_is_calibrated_to_profile": False,
            "detector_profile_completeness": profile.get("profile_completeness", "profile_without_pnr_confusion_matrix"),
            "threshold_coarsening_applied": threshold_coarsening_applied(profile, detector_kind),
            "calibrated_pnr_confusion_matrix_available": matrix is not None,
        }
    if detector_kind == "pnr_snspd" and matrix is not None:
        return {
            "pnr_capability": True,
            "max_resolvable_photon_number": max_count,
            "pnr_response_model": "profile_calibrated_pnr_confusion_matrix",
            "pnr_response_model_is_calibrated_to_profile": True,
            "detector_profile_completeness": profile.get("profile_completeness", "profile_with_pnr_confusion_matrix"),
            "threshold_coarsening_applied": False,
            "calibrated_pnr_confusion_matrix_available": True,
            "pnr_confusion_matrix_dimensions": {
                "input_rows": len(matrix["input_photon_number"]),
                "reported_count_columns": max_count + 1,
            },
        }
    if detector_kind == "pnr_snspd":
        return {
            "pnr_capability": profile_claims_pnr(profile),
            "max_resolvable_photon_number": max_count,
            "pnr_response_model": "generic_saturated_binomial_detection_plus_poisson_dark_counts",
            "pnr_response_model_is_calibrated_to_profile": False,
            "detector_profile_completeness": profile.get("profile_completeness", "profile_without_pnr_confusion_matrix"),
            "threshold_coarsening_applied": False,
            "calibrated_pnr_confusion_matrix_available": False,
        }
    return {
        "pnr_capability": profile_claims_pnr(profile),
        "max_resolvable_photon_number": max_count,
        "pnr_response_model": "not_applicable",
        "pnr_response_model_is_calibrated_to_profile": False,
        "detector_profile_completeness": profile.get("profile_completeness", "profile_without_pnr_confusion_matrix"),
        "threshold_coarsening_applied": False,
        "calibrated_pnr_confusion_matrix_available": matrix is not None,
    }


def detector_profile_warnings(profile: dict[str, Any], detector_kind: str, max_count: int) -> list[str]:
    meta = pnr_response_model_metadata(profile, detector_kind, max_count)
    warnings: list[str] = []
    name = profile.get("reference_name", "PNR SNSPD profile")
    if (
        profile_claims_pnr(profile)
        and detector_kind == "pnr_snspd"
        and not meta["pnr_response_model_is_calibrated_to_profile"]
    ):
        warnings.append(
            f"{name} demonstrates PNR capability, but no calibrated PNR confusion matrix is supplied; "
            "using generic saturated binomial+dark PNR model."
        )
    if detector_kind == "pnr_snspd" and meta["calibrated_pnr_confusion_matrix_available"]:
        bypassed = [
            field for field in ("system_detection_efficiency", "dark_count_rate_per_s")
            if profile.get(field) is not None
        ]
        warnings.append(
            f"{name} supplies a calibrated pnr_confusion_matrix: reported counts are read directly as "
            "P(reported count | incident photon number), so the matrix must be measured per incident "
            "photon INCLUDING system detection efficiency and in-window dark counts"
            + (
                f"; the profile scalars {', '.join(bypassed)} are bypassed for click statistics"
                if bypassed else ""
            )
            + "; any loaded conversion_noise_rate_per_s background is likewise not added on top of the matrix."
        )
    return warnings
