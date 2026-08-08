#!/usr/bin/env python
"""Generate one-hop Fock/SNSPD detector-profile diagnostics.

This is a diagnostic harness around scripts/generate_e91_game_turn_histograms.py.
It does not define new physics.  It compares detector profiles through the same
memory-emission Fock/SNSPD elementary-link and E91 histogram path used by the
histogram generator.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_e91_game_turn_histograms.py"
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "physical"
DEFAULT_DETECTOR_DIR = DEFAULT_CONFIG_DIR / "detectors"


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_e91_game_turn_histograms", GENERATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hist = load_generator()


SCENARIOS = (
    {
        "scenario_id": "peacoq_threshold",
        "label": "PEACOQ threshold",
        "detector_kind": "repository_default",
        "profile_path": DEFAULT_DETECTOR_DIR / "snspd_peacoq_highspeed_1550_threshold.json",
    },
    {
        "scenario_id": "korzh_threshold_timing",
        "label": "Korzh threshold/timing",
        "detector_kind": "repository_default",
        "profile_path": DEFAULT_DETECTOR_DIR / "snspd_korzh_low_jitter_1550_threshold.json",
    },
    {
        "scenario_id": "colangelo_pnr_generic",
        "label": "Colangelo demonstrated PNR, generic response",
        "detector_kind": "repository_default",
        "profile_path": DEFAULT_DETECTOR_DIR / "snspd_colangelo_impedance_matched_differential_pnr_1550.json",
    },
    {
        "scenario_id": "colangelo_threshold_coarsened",
        "label": "Colangelo threshold-coarsened",
        "detector_kind": "threshold_snspd",
        "profile_path": DEFAULT_DETECTOR_DIR / "snspd_colangelo_impedance_matched_differential_pnr_1550.json",
    },
    {
        "scenario_id": "ideal_pnr",
        "label": "Ideal PNR",
        "detector_kind": "ideal_pnr",
        "profile_path": None,
    },
)


SCENARIO_INTERPRETATIONS = {
    "peacoq_threshold": (
        "PEACOQ high-speed telecom SNSPD profile used as a demonstrated threshold detector; "
        "no PNR capability is claimed."
    ),
    "korzh_threshold_timing": (
        "Korzh low-jitter timing profile used as a threshold detector; efficiency, dark-count, "
        "and dead-time fields fall back to repository defaults when the profile leaves them null."
    ),
    "colangelo_pnr_generic": (
        "Colangelo Detector-A 1550 nm demonstrated-PNR profile with generic saturated "
        "binomial-plus-dark PNR response; no calibrated PNR confusion matrix is supplied."
    ),
    "colangelo_threshold_coarsened": (
        "Colangelo demonstrated-PNR profile intentionally reduced to threshold click/no-click "
        "detector records."
    ),
    "ideal_pnr": "Idealized exact photon-number detector; not a demonstrated hardware profile.",
}


TOL = 1e-9


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def load_profile(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def make_context(detector_kind: str, profile_path: Path | None, profile: dict[str, Any] | None, fock_cutoff: int):
    return hist.ModelContext(
        physical_path="fock_pnr_snspd",
        source_model="memory_emission",
        detector_kind=detector_kind,
        fock_cutoff=fock_cutoff,
        detector_profile_path=str(profile_path) if profile_path else None,
        detector_profile=profile,
    )


def front_for_wavelength(base_front: dict[str, Any], wavelength_nm: float) -> dict[str, Any]:
    front = copy.deepcopy(base_front)
    front["device_constants"]["memory"]["wavelength_nm"] = float(wavelength_nm)
    return front


def no_attack():
    return hist.HistogramAttack(
        attack_id="no_attack_miss",
        attack_kind="none",
        target_active=False,
        histogram_source="no_attack_miss",
    )


def bell_overlap_summary(branches: list[Any]) -> dict[str, Any]:
    labels = ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
    weighted = {label: 0.0 for label in labels}
    branch_rows = []
    for branch in branches:
        overlaps = {
            label: float(np.real(np.trace(hist.bell_density(label) @ branch.rho)))
            for label in labels
        }
        for label, value in overlaps.items():
            weighted[label] += branch.probability * value
        branch_rows.append({
            "probability_given_success": branch.probability,
            "public_bsm_record": branch.public_bsm_record,
            "detector_click_record": branch.detector_click_record,
            "bell_overlaps": overlaps,
            "max_bell_label": max(overlaps, key=overlaps.get),
            "max_bell_overlap": max(overlaps.values()),
        })
    return {
        "weighted_conditional_bell_overlaps": weighted,
        "max_weighted_bell_label": max(weighted, key=weighted.get) if weighted else None,
        "max_weighted_bell_overlap": max(weighted.values()) if weighted else None,
        "branches": branch_rows,
    }


def shannon_entropy(probabilities: list[float]) -> float:
    entropy = 0.0
    for probability in probabilities:
        if probability > 0.0:
            entropy -= probability * float(np.log2(probability))
    return entropy


def raw_record_structure(raw_records: dict[str, float], branches: list[Any]) -> dict[str, Any]:
    return {
        "raw_detector_record_count": len(raw_records),
        "raw_detector_record_entropy": shannon_entropy(list(raw_records.values())),
        "delivered_branch_count": len(branches),
        "delivered_detector_record_count": len({branch.detector_click_record for branch in branches}),
        "has_pnr_resolved_raw_counts": any("=2" in label or "=3" in label for label in raw_records),
    }


def summarize_case(case: dict[str, Any]) -> dict[str, Any]:
    metrics = case["metadata"]["expected_metrics"]
    return {
        "delivery_probability": metrics["delivery_probability"],
        "expected_qber": metrics["expected_qber"],
        "expected_chsh_S": metrics["expected_chsh_S"],
        "probability_sum": case["metadata"]["validation_checks"]["probability_sum"],
        "row_count": len(case["rows"]),
        "key_length": case["metadata"]["key_length"],
    }


def detector_model_interpretation(scenario_id: str, profile: dict[str, Any], resolved_kind: str, pnr_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": SCENARIO_INTERPRETATIONS[scenario_id],
        "resolved_detector_kind": resolved_kind,
        "pnr_capability": pnr_meta["pnr_capability"],
        "detector_is_idealized": resolved_kind == "ideal_pnr",
        "pnr_response_model": pnr_meta["pnr_response_model"],
        "pnr_response_model_is_calibrated_to_profile": pnr_meta[
            "pnr_response_model_is_calibrated_to_profile"
        ],
        "threshold_coarsening_applied": pnr_meta["threshold_coarsening_applied"],
        "detector_profile_completeness": pnr_meta["detector_profile_completeness"],
    }


def run_one(
    *,
    base_front: dict[str, Any],
    scenario: dict[str, Any],
    length_m: float,
    wavelength_nm: float,
    key_length: int,
    fock_cutoff: int,
) -> dict[str, Any]:
    profile = load_profile(scenario["profile_path"])
    context = make_context(scenario["detector_kind"], scenario["profile_path"], profile, fock_cutoff)
    front = front_for_wavelength(base_front, wavelength_nm)
    route = hist.make_route(1, [length_m])
    attack = no_attack()
    data = hist.fock_memory_emission_elementary_link_data("e0", route, front, attack, context)
    case = hist.generate_histogram_case(
        route=route,
        front=front,
        attack=attack,
        key_length=key_length,
        context=context,
    )
    resolved_kind = hist.resolved_detector_kind(front, context)
    profile_meta = hist.detector_profile_metadata(front, context)
    max_count = hist.detector_max_count(context, profile_meta)
    pnr_meta = hist.pnr_response_model_metadata(profile_meta, resolved_kind, max_count)
    branches = data["conditional_success_branches"]
    raw_records = data["raw_detector_record_probabilities"]
    return {
        "scenario_id": scenario["scenario_id"],
        "label": scenario["label"],
        "model_interpretation": detector_model_interpretation(
            scenario["scenario_id"],
            profile_meta,
            resolved_kind,
            pnr_meta,
        ),
        "profile_path": str(scenario["profile_path"]) if scenario["profile_path"] else None,
        "profile_reference_name": profile_meta.get("reference_name"),
        "requested_detector_kind": scenario["detector_kind"],
        "resolved_detector_kind": resolved_kind,
        "wavelength_nm": wavelength_nm,
        "fiber_length_m": length_m,
        "fock_cutoff": fock_cutoff,
        "pnr_capability": pnr_meta["pnr_capability"],
        "max_resolvable_photon_number": max_count,
        "pnr_response_model_metadata": pnr_meta,
        "detector_profile_warnings": hist.detector_profile_warnings(profile_meta, resolved_kind, max_count),
        "elementary_link": {
            "success_probability": data["success_probability"],
            "bsm_outcome_probabilities": data["bsm_outcome_probabilities"],
            "raw_detector_record_probabilities": raw_records,
            "raw_detector_record_structure": raw_record_structure(raw_records, branches),
            "bell_overlap_summary": bell_overlap_summary(branches),
            "metadata": data["metadata"],
        },
        "histogram": summarize_case(case),
    }


def set_active_dark_count(front: dict[str, Any], profile: dict[str, Any] | None, value: float) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    out_front = copy.deepcopy(front)
    out_profile = copy.deepcopy(profile) if profile is not None else None
    if out_profile is not None and out_profile.get("dark_count_rate_per_s") is not None:
        out_profile["dark_count_rate_per_s"] = float(value)
        return out_front, out_profile, "detector_profile.dark_count_rate_per_s"
    out_front["device_constants"]["detector"]["dark_count_per_s"] = float(value)
    return out_front, out_profile, "repository_detector.dark_count_per_s"


def set_active_efficiency(front: dict[str, Any], profile: dict[str, Any] | None, value: float) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    out_front = copy.deepcopy(front)
    out_profile = copy.deepcopy(profile) if profile is not None else None
    if out_profile is not None and out_profile.get("system_detection_efficiency") is not None:
        out_profile["system_detection_efficiency"] = float(value)
        return out_front, out_profile, "detector_profile.system_detection_efficiency"
    out_front["device_constants"]["detector"]["efficiency"] = float(value)
    return out_front, out_profile, "repository_detector.efficiency"


def sweep_point(
    *,
    base_front: dict[str, Any],
    scenario: dict[str, Any],
    profile: dict[str, Any] | None,
    context,
    length_m: float,
    wavelength_nm: float,
    key_length: int,
) -> dict[str, float]:
    front = front_for_wavelength(base_front, wavelength_nm)
    route = hist.make_route(1, [length_m])
    attack = no_attack()
    data = hist.fock_memory_emission_elementary_link_data("e0", route, front, attack, context)
    case = hist.generate_histogram_case(
        route=route,
        front=front,
        attack=attack,
        key_length=key_length,
        context=context,
    )
    return {
        "delivery_probability": case["metadata"]["expected_metrics"]["delivery_probability"],
        "elementary_success_probability": data["success_probability"],
        "psi_plus_probability": data["bsm_outcome_probabilities"].get("psi_plus", 0.0),
        "psi_minus_probability": data["bsm_outcome_probabilities"].get("psi_minus", 0.0),
        "inconclusive_probability": data["bsm_outcome_probabilities"].get("inconclusive", 0.0),
    }


def run_sweeps(
    *,
    base_front: dict[str, Any],
    scenario: dict[str, Any],
    length_m: float,
    wavelength_nm: float,
    key_length: int,
    fock_cutoff: int,
    fiber_sweep_m: list[float],
    dark_count_sweep_per_s: list[float],
    efficiency_sweep: list[float],
) -> dict[str, Any]:
    profile = load_profile(scenario["profile_path"])
    context = make_context(scenario["detector_kind"], scenario["profile_path"], profile, fock_cutoff)
    fiber_points = []
    for value in fiber_sweep_m:
        fiber_points.append({"fiber_length_m": value, **sweep_point(
            base_front=base_front,
            scenario=scenario,
            profile=profile,
            context=context,
            length_m=value,
            wavelength_nm=wavelength_nm,
            key_length=key_length,
        )})

    dark_points = []
    dark_source = None
    for value in dark_count_sweep_per_s:
        dark_front, dark_profile, dark_source = set_active_dark_count(base_front, profile, value)
        dark_context = make_context(scenario["detector_kind"], scenario["profile_path"], dark_profile, fock_cutoff)
        dark_points.append({"dark_count_rate_per_s": value, **sweep_point(
            base_front=dark_front,
            scenario=scenario,
            profile=dark_profile,
            context=dark_context,
            length_m=length_m,
            wavelength_nm=wavelength_nm,
            key_length=key_length,
        )})

    efficiency_points = []
    efficiency_source = None
    for value in efficiency_sweep:
        eff_front, eff_profile, efficiency_source = set_active_efficiency(base_front, profile, value)
        eff_context = make_context(scenario["detector_kind"], scenario["profile_path"], eff_profile, fock_cutoff)
        efficiency_points.append({"detector_efficiency": value, **sweep_point(
            base_front=eff_front,
            scenario=scenario,
            profile=eff_profile,
            context=eff_context,
            length_m=length_m,
            wavelength_nm=wavelength_nm,
            key_length=key_length,
        )})

    return {
        "fiber_length_sweep_m": fiber_points,
        "dark_count_sweep": {
            "varied_parameter": dark_source,
            "points": dark_points,
        },
        "detector_efficiency_sweep": {
            "varied_parameter": efficiency_source,
            "points": efficiency_points,
        },
    }


def ideal_lossless_sanity_check(
    *,
    base_front: dict[str, Any],
    wavelength_nm: float,
    key_length: int,
    fock_cutoff: int,
) -> dict[str, Any]:
    front = front_for_wavelength(base_front, wavelength_nm)
    front["device_constants"]["memory"]["efficiency"] = 1.0
    front["device_constants"]["memory"]["fidelity"] = 1.0
    # The ideal-lossless reference also neutralizes any literature-calibrated
    # emission process noise; this check validates the analyzer, not the node.
    front["device_constants"]["memory"]["memory_process_model"] = {
        "kind": "none",
        "enters_probability": False,
    }
    front["device_constants"]["fiber_channel"]["attenuation_db_per_m"] = 0.0
    front["device_constants"]["detector"]["dark_count_per_s"] = 0.0
    route = hist.make_route(1, [0.0])
    context = make_context("ideal_pnr", None, None, max(2, fock_cutoff))
    attack = no_attack()
    data = hist.fock_memory_emission_elementary_link_data("e0", route, front, attack, context)
    case = hist.generate_histogram_case(
        route=route,
        front=front,
        attack=attack,
        key_length=key_length,
        context=context,
    )
    bell = bell_overlap_summary(data["conditional_success_branches"])
    return {
        "description": "unit memory efficiency/fidelity, zero fiber attenuation, zero dark counts, ideal PNR detector",
        "expected_for_implemented_linear_optics_analyzer": {
            "success_probability": 0.5,
            "psi_plus_probability": 0.25,
            "psi_minus_probability": 0.25,
            "inconclusive_probability": 0.5,
            "successful_branch_bell_overlap": 1.0,
        },
        "observed": {
            "elementary_success_probability": data["success_probability"],
            "bsm_outcome_probabilities": data["bsm_outcome_probabilities"],
            "histogram": summarize_case(case),
            "bell_overlap_summary": bell,
        },
        "checks": {
            "success_probability_matches": abs(data["success_probability"] - 0.5) <= 1e-9,
            "psi_plus_probability_matches": abs(data["bsm_outcome_probabilities"].get("psi_plus", 0.0) - 0.25) <= 1e-9,
            "psi_minus_probability_matches": abs(data["bsm_outcome_probabilities"].get("psi_minus", 0.0) - 0.25) <= 1e-9,
            "inconclusive_probability_matches": abs(data["bsm_outcome_probabilities"].get("inconclusive", 0.0) - 0.5) <= 1e-9,
            "all_successful_branches_are_bell_states": all(
                branch["max_bell_overlap"] >= 1.0 - 1e-9
                for branch in bell["branches"]
            ),
            "frame_corrected_qber_matches": case["metadata"]["expected_metrics"]["expected_qber"] == 0.0,
            "frame_corrected_chsh_matches": abs(
                case["metadata"]["expected_metrics"]["expected_chsh_S"] - (2.0 * np.sqrt(2.0))
            ) <= 1e-9,
        },
    }


def nonincreasing(values: list[float]) -> bool:
    return all(left + TOL >= right for left, right in zip(values, values[1:]))


def nondecreasing(values: list[float]) -> bool:
    return all(left <= right + TOL for left, right in zip(values, values[1:]))


def check_probability_normalization(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_scenario = {
        row["scenario_id"]: abs(row["histogram"]["probability_sum"] - 1.0) <= 1e-7
        for row in results
    }
    return {"passed": all(by_scenario.values()), "by_scenario": by_scenario}


def check_detector_record_alphabets(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["scenario_id"]: row for row in results}
    pnr_records = set(by_id["colangelo_pnr_generic"]["elementary_link"]["raw_detector_record_probabilities"])
    threshold_records = set(
        by_id["colangelo_threshold_coarsened"]["elementary_link"]["raw_detector_record_probabilities"]
    )
    return {
        "passed": pnr_records != threshold_records
        and by_id["colangelo_pnr_generic"]["elementary_link"]["raw_detector_record_structure"][
            "has_pnr_resolved_raw_counts"
        ]
        and not by_id["colangelo_threshold_coarsened"]["elementary_link"]["raw_detector_record_structure"][
            "has_pnr_resolved_raw_counts"
        ],
        "colangelo_pnr_record_count": len(pnr_records),
        "colangelo_threshold_record_count": len(threshold_records),
    }


def check_colangelo_warning(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {row["scenario_id"]: row for row in results}
    warnings = by_id["colangelo_pnr_generic"]["detector_profile_warnings"]
    return {
        "passed": any("no calibrated PNR confusion matrix" in warning for warning in warnings),
        "warnings": warnings,
    }


def check_sweep_monotonicity(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_scenario: dict[str, Any] = {}
    for row in results:
        sweeps = row["sweeps"]
        fiber_delivery = [point["delivery_probability"] for point in sweeps["fiber_length_sweep_m"]]
        efficiency_delivery = [
            point["delivery_probability"] for point in sweeps["detector_efficiency_sweep"]["points"]
        ]
        dark_inconclusive = [
            point["inconclusive_probability"] for point in sweeps["dark_count_sweep"]["points"]
        ]
        by_scenario[row["scenario_id"]] = {
            "fiber_delivery_nonincreasing": nonincreasing(fiber_delivery),
            "efficiency_delivery_nondecreasing": nondecreasing(efficiency_delivery),
            "dark_inconclusive_values": dark_inconclusive,
            "fiber_delivery_values": fiber_delivery,
            "efficiency_delivery_values": efficiency_delivery,
        }
    passed = all(
        values["fiber_delivery_nonincreasing"] and values["efficiency_delivery_nondecreasing"]
        for values in by_scenario.values()
    )
    return {"passed": passed, "by_scenario": by_scenario}


def consistency_checks(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload["scenarios"]
    checks = {
        "ideal_lossless_invariants": {
            "passed": all(payload["ideal_lossless_sanity_check"]["checks"].values()),
            "checks": payload["ideal_lossless_sanity_check"]["checks"],
        },
        "probability_normalization": check_probability_normalization(results),
        "pnr_vs_threshold_raw_record_alphabet": check_detector_record_alphabets(results),
        "colangelo_pnr_generic_warning": check_colangelo_warning(results),
        "sweep_monotonicity": check_sweep_monotonicity(results),
    }
    checks["passed"] = all(check["passed"] for check in checks.values())
    return checks


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "scenario_id",
        "label",
        "model_interpretation",
        "resolved_detector_kind",
        "pnr_response_model",
        "threshold_coarsening_applied",
        "raw_detector_record_entropy",
        "delivered_branch_count",
        "fiber_length_m",
        "delivery_probability",
        "elementary_success_probability",
        "psi_plus_probability",
        "psi_minus_probability",
        "inconclusive_probability",
        "expected_qber",
        "expected_chsh_S",
        "max_weighted_bell_label",
        "max_weighted_bell_overlap",
        "warnings",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            elementary = row["elementary_link"]
            bsm = elementary["bsm_outcome_probabilities"]
            pnr_meta = row["pnr_response_model_metadata"]
            bell = elementary["bell_overlap_summary"]
            structure = elementary["raw_detector_record_structure"]
            writer.writerow({
                "scenario_id": row["scenario_id"],
                "label": row["label"],
                "model_interpretation": row["model_interpretation"]["summary"],
                "resolved_detector_kind": row["resolved_detector_kind"],
                "pnr_response_model": pnr_meta["pnr_response_model"],
                "threshold_coarsening_applied": pnr_meta["threshold_coarsening_applied"],
                "raw_detector_record_entropy": structure["raw_detector_record_entropy"],
                "delivered_branch_count": structure["delivered_branch_count"],
                "fiber_length_m": row["fiber_length_m"],
                "delivery_probability": row["histogram"]["delivery_probability"],
                "elementary_success_probability": elementary["success_probability"],
                "psi_plus_probability": bsm.get("psi_plus", 0.0),
                "psi_minus_probability": bsm.get("psi_minus", 0.0),
                "inconclusive_probability": bsm.get("inconclusive", 0.0),
                "expected_qber": row["histogram"]["expected_qber"],
                "expected_chsh_S": row["histogram"]["expected_chsh_S"],
                "max_weighted_bell_label": bell["max_weighted_bell_label"],
                "max_weighted_bell_overlap": bell["max_weighted_bell_overlap"],
                "warnings": " | ".join(row["detector_profile_warnings"]),
            })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config-dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--fiber-length-m", type=float, default=1000.0)
    parser.add_argument("--wavelength-nm", type=float, default=1550.0)
    parser.add_argument("--key-length", type=int, default=8)
    parser.add_argument("--fock-cutoff", type=int, default=4)
    parser.add_argument("--fiber-sweep-m", default="0,1000,10000")
    parser.add_argument("--dark-count-sweep-per-s", default="0,200,1000")
    parser.add_argument("--efficiency-sweep", default="0.2,0.711,1.0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_front = hist.load_or_build_front_matter(Path(args.hardware_config_dir))
    results = []
    for scenario in SCENARIOS:
        row = run_one(
            base_front=base_front,
            scenario=scenario,
            length_m=args.fiber_length_m,
            wavelength_nm=args.wavelength_nm,
            key_length=args.key_length,
            fock_cutoff=args.fock_cutoff,
        )
        row["sweeps"] = run_sweeps(
            base_front=base_front,
            scenario=scenario,
            length_m=args.fiber_length_m,
            wavelength_nm=args.wavelength_nm,
            key_length=args.key_length,
            fock_cutoff=args.fock_cutoff,
            fiber_sweep_m=parse_float_list(args.fiber_sweep_m),
            dark_count_sweep_per_s=parse_float_list(args.dark_count_sweep_per_s),
            efficiency_sweep=parse_float_list(args.efficiency_sweep),
        )
        results.append(row)
    payload = {
        "diagnostic": "one-hop Fock/SNSPD detector-profile sanity comparison",
        "generator_version": hist.GENERATOR_VERSION,
        "active_wavelength_nm": args.wavelength_nm,
        "fiber_length_m": args.fiber_length_m,
        "key_length": args.key_length,
        "fock_cutoff": args.fock_cutoff,
        "ideal_lossless_sanity_check": ideal_lossless_sanity_check(
            base_front=base_front,
            wavelength_nm=args.wavelength_nm,
            key_length=args.key_length,
            fock_cutoff=args.fock_cutoff,
        ),
        "scenarios": results,
    }
    payload["consistency_checks"] = consistency_checks(payload)
    with open(out_dir / "detector_profile_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(hist._json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_csv(out_dir / "detector_profile_diagnostics_summary.csv", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
