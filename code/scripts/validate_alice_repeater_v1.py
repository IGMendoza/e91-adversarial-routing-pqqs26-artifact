#!/usr/bin/env python
"""Validate the Alice-only asynchronous repeater-v1 histogram stack.

The report keeps three quantities distinct:

* asynchronous controller service time/rate;
* random-time rolling-pool availability with the declared memory inventory;
* conditional delivered-pair quality after edge purification, storage, and
  swap channels.

No Eve move, encoded repeater, or security-rate claim is included.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.corpus.compiled_histogram_backend import (  # noqa: E402
    CompiledHistogramBackend,
    CompiledHistogramBackendConfig,
    DEFAULT_DETECTOR_PROFILE_PATH,
    load_histogram_generator,
)
from sequence_game.physical.repeater_reservation import (  # noqa: E402
    rolling_config_for_service_profile,
)
from sequence_game.physical.repeater_service import (  # noqa: E402
    load_repeater_service_profile,
)


DEFAULT_PROFILE = (
    REPO_ROOT
    / "configs"
    / "physical"
    / "repeater_services"
    / "cui_2026_rolling_purified_near_term_composite.json"
)
DEFAULT_CONVERSION = (
    REPO_ROOT
    / "configs"
    / "physical"
    / "conversion"
    / "cui_dfg_866_to_1558.json"
)
DEFAULT_CASES = (
    (1, 1.0),
    (1, 50.0),
    (2, 10.0),
    (4, 10.0),
    (8, 10.0),
    (12, 1.0),
    (12, 10.0),
    (12, 50.0),
    (16, 50.0),
)


def _tuple_rows(values: Sequence[Sequence[float]]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in values)


def parse_cases(raw: str | None) -> tuple[tuple[int, float], ...]:
    if raw is None:
        return DEFAULT_CASES
    rows: list[tuple[int, float]] = []
    for item in raw.split(","):
        hop_text, separator, length_text = item.strip().partition("x")
        if not separator:
            raise ValueError("cases must use HxKM entries, for example 12x10")
        hops = int(hop_text)
        length = float(length_text)
        if hops < 1 or length <= 0.0:
            raise ValueError("case hop counts and edge lengths must be positive")
        rows.append((hops, length))
    if not rows:
        raise ValueError("at least one validation case is required")
    return tuple(rows)


def backend_config_for_payload(
    *,
    profile_path: Path,
    conversion_path: Path,
    detector_profile_path: Path,
    payload: dict[str, Any],
    key_length: int,
) -> CompiledHistogramBackendConfig:
    return CompiledHistogramBackendConfig(
        detector_kind="pnr_snspd",
        detector_profile_path=str(detector_profile_path),
        conversion_response_path=str(conversion_path),
        repeater_service_profile_path=str(profile_path),
        # The selected PNR SNSPD is characterized at 1550 nm while Cui's DFG
        # output is 1558 nm. Both are C-band, but no measured eight-nanometre
        # efficiency curve is available; retain the scalar operating point as
        # an explicit recorded approximation rather than disabling Cui.
        allow_wavelength_mismatch=True,
        fock_cutoff=2,
        waiting_protocol="rolling_alive_pair",
        rolling_edge_availability=tuple(
            float(value) for value in payload["rolling_edge_availability"]
        ),
        rolling_age_distribution_kind="discrete",
        rolling_age_distribution_points_s=(0.0,),
        rolling_age_distribution_weights=(1.0,),
        rolling_edge_age_distribution_points_s=_tuple_rows(
            payload["rolling_edge_age_distribution_points_s"]
        ),
        rolling_edge_age_distribution_weights=_tuple_rows(
            payload["rolling_edge_age_distribution_weights"]
        ),
        rolling_age_evaluation="moment_reduced",
        rolling_accepted_alive_time_s=float(payload["rolling_accepted_alive_time_s"]),
        rolling_memory_slots_per_edge_side=int(
            payload["rolling_memory_slots_per_edge_side"]
        ),
        rolling_selection_policy=str(payload["rolling_selection_policy"]),
        rolling_age_distribution_source=str(payload["rolling_age_distribution_source"]),
        compile_key_length=int(key_length),
    )


def evaluate_case(
    *,
    profile_path: Path,
    conversion_path: Path,
    detector_profile_path: Path,
    hop_count: int,
    edge_length_km: float,
    accepted_alive_time_s: float,
    age_points: int,
    key_length: int,
) -> dict[str, Any]:
    profile = load_repeater_service_profile(profile_path)
    lengths_km = [float(edge_length_km)] * int(hop_count)
    payload = rolling_config_for_service_profile(
        profile,
        lengths_km,
        accepted_alive_time_s=float(accepted_alive_time_s),
        num_age_points=int(age_points),
    )
    backend = CompiledHistogramBackend(backend_config_for_payload(
        profile_path=profile_path,
        conversion_path=conversion_path,
        detector_profile_path=detector_profile_path,
        payload=payload,
        key_length=key_length,
    ))
    generator = load_histogram_generator()
    move = generator.EveMove("no_move_baseline", "none", target_active=False)
    wall_start = time.perf_counter()
    evaluated = backend.evaluate(
        int(hop_count),
        move,
        [1000.0 * float(edge_length_km)] * int(hop_count),
    )
    wall_seconds = time.perf_counter() - wall_start
    rolling = evaluated.delivery_model["rolling_live_pair_model"]
    service = rolling["repeater_service_model"]["route_service"]
    source_quality = rolling["repeater_service_model"]["bell_quality_without_storage"]
    metrics = evaluated.expected_metrics
    qber = metrics.get("expected_qber_from_histogram", metrics.get("expected_qber"))
    chsh = metrics.get("expected_chsh_S")
    delivery = float(metrics["delivery_probability"])
    rate = float(service["success_rate_hz"])
    return {
        "hop_count": int(hop_count),
        "edge_length_km": float(edge_length_km),
        "total_distance_km": int(hop_count) * float(edge_length_km),
        "resource_inventory": payload["resource_policy"],
        "rolling_route_availability": float(
            payload["reservation_summary"]["route_availability"]
        ),
        "histogram_delivery_probability_per_random_time_request": delivery,
        "asynchronous_service": service,
        "source_equation_quality_before_storage": source_quality,
        "histogram_quality_after_storage": {
            "expected_qber": qber,
            "expected_chsh_S": chsh,
            "chsh_violation": chsh is not None and float(chsh) > 2.0,
        },
        "selected_age_summary_by_edge": payload["edge_summaries"],
        "probability_sum": float(evaluated.probabilities.sum()),
        "row_count": int(evaluated.row_count),
        "theta_hash": evaluated.theta_hash,
        "performance": {
            "evaluate_seconds": float(evaluated.evaluate_seconds),
            "compile_plus_evaluate_wall_seconds": wall_seconds,
            "service_pairs_per_second": rate,
        },
        "checks": {
            "probability_normalized": math.isclose(
                float(evaluated.probabilities.sum()), 1.0, abs_tol=1e-10
            ),
            "delivery_matches_fixed_inventory_availability": math.isclose(
                delivery,
                float(payload["reservation_summary"]["route_availability"]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ),
            "service_rate_positive": rate > 0.0,
            "source_model_end_pair_entangled": bool(
                source_quality["entangled_by_bell_diagonal_ppt_test"]
            ),
            "histogram_chsh_violation": chsh is not None and float(chsh) > 2.0,
            "memory_tau_at_least_mean_service_time": service["memory_regime"] != (
                "mean_service_time_exceeds_tau"
            ),
        },
    }


def validate_alice_repeater_v1(
    *,
    profile_path: Path = DEFAULT_PROFILE,
    conversion_path: Path = DEFAULT_CONVERSION,
    detector_profile_path: Path = DEFAULT_DETECTOR_PROFILE_PATH,
    cases: Sequence[tuple[int, float]] = DEFAULT_CASES,
    accepted_alive_time_s: float | None = None,
    age_points: int = 8,
    key_length: int = 8,
) -> dict[str, Any]:
    started = time.perf_counter()
    profile = load_repeater_service_profile(profile_path)
    resolved_alive_time_s = float(
        accepted_alive_time_s
        if accepted_alive_time_s is not None
        else profile.controller_model.get("accepted_output_alive_time_s", 0.18)
    )
    rows = [
        evaluate_case(
            profile_path=profile_path,
            conversion_path=conversion_path,
            detector_profile_path=detector_profile_path,
            hop_count=hops,
            edge_length_km=length,
            accepted_alive_time_s=resolved_alive_time_s,
            age_points=age_points,
            key_length=key_length,
        )
        for hops, length in cases
    ]
    return {
        "report_version": 1,
        "status": (
            "failed_invariant"
            if not all(
                row["checks"]["probability_normalized"]
                and row["checks"]["delivery_matches_fixed_inventory_availability"]
                and row["checks"]["service_rate_positive"]
                for row in rows
            )
            else (
                "physics_closed_with_bell_qualified_delivery"
                if all(row["checks"]["histogram_chsh_violation"] for row in rows)
                else "physics_closed_with_quality_limitations"
            )
        ),
        "scope": {
            "party": "Alice delivery/control only",
            "eve": "excluded by request",
            "purification": "one noisy elementary-link bilateral phase-parity recurrence round",
            "encoding_or_error_correction": "not implemented",
            "security_claim": "none; CHSH and QBER are diagnostics",
        },
        "profile_path": str(profile_path),
        "conversion_path": str(conversion_path),
        "detector_profile_path": str(detector_profile_path),
        "profile": profile.to_dict(),
        "controller_policy": {
            "accepted_alive_time_s": resolved_alive_time_s,
            "selected_pair_policy": "youngest_live",
            "age_quadrature_points_per_edge": int(age_points),
            "memory_inventory": "fixed by profile, never target-provisioned",
        },
        "cases": rows,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": [
            "Repeaters provide rate advantage through asynchronous generation, storage, and local swapping; they are not noiseless optical amplifiers.",
            "The histogram delivery probability is a random-time rolling-pool availability, while service_pairs_per_second is the controller's end-to-end production rate.",
            "Bell-qualified delivery requires the delivered histogram itself to exceed CHSH S=2; service rate or fidelity alone is insufficient.",
            "The active profile composes literature-demonstrated components and established recurrence physics; the complete multi-node system was not demonstrated.",
        ],
    }


def _format(value: Any, digits: int = 4) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "yes" if value else "no"
    number = float(value)
    if number != 0.0 and (abs(number) < 1e-3 or abs(number) >= 1e4):
        return f"{number:.3e}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Alice repeater-v1 physical validation",
        "",
        f"Status: `{report['status']}`.",
        "",
        "The table separates asynchronous service rate from rolling-pool snapshot availability and delivered-pair quality.",
        "",
        "| hops | km/link | total km | memory slots/side | service Hz | mean service s | pool availability | delivery | Bell F before storage | QBER after storage | CHSH after storage | Bell violation? | memory regime | eval s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :--- | ---: |",
    ]
    for row in report["cases"]:
        service = row["asynchronous_service"]
        quality = row["source_equation_quality_before_storage"]
        after = row["histogram_quality_after_storage"]
        lines.append(
            "| "
            + " | ".join([
                str(row["hop_count"]),
                _format(row["edge_length_km"], 1),
                _format(row["total_distance_km"], 1),
                str(row["resource_inventory"]["memory_slots_per_adjacent_edge_side"]),
                _format(service["success_rate_hz"], 4),
                _format(service["mean_distribution_time_s"], 4),
                _format(row["rolling_route_availability"], 6),
                _format(row["histogram_delivery_probability_per_random_time_request"], 6),
                _format(quality["end_to_end_bell_fidelity"], 5),
                _format(after["expected_qber"], 5),
                _format(after["expected_chsh_S"], 5),
                _format(after["chsh_violation"]),
                str(service["memory_regime"]),
                _format(row["performance"]["evaluate_seconds"], 4),
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Scientific boundary",
        "",
        "- The active profile is a `near_term_composite`, not a demonstrated complete repeater network.",
        "- Resources are fixed by the selected profile: four independent purifier/output lanes per adjacent edge, with no availability-target slot optimizer.",
        "- Each elementary edge consumes two raw pairs in a noisy bilateral phase-parity recurrence; rejection, local-operation time, readout confusion, classical comparison, and sacrificial-pair wait decoherence are active.",
        "- The primary profile retains the detector-conditioned Cui Fock/PNR elementary state, applies one noisy recurrence round, and then applies explicit selected-age Cui memory dephasing.",
        "- No encoded/error-corrected repeater is modeled. A row without S>2 is not counted as successful quantum delivery.",
        "- The Cui raw-lane renewal anchor comes from Supplementary Note 8 author data; the Krutyanskiy/Dhara balanced asynchronous controller is a labelled fixed-resource composition.",
        "- Cui's 1558 nm DFG output uses the selected detector's 1550 nm scalar C-band operating point as an explicit wavelength-response approximation; no hidden ideal detector is used.",
        "",
    ])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--conversion", type=Path, default=DEFAULT_CONVERSION)
    parser.add_argument(
        "--detector-profile", type=Path, default=DEFAULT_DETECTOR_PROFILE_PATH
    )
    parser.add_argument("--cases", default=None, help="comma-separated HxKM cases")
    parser.add_argument(
        "--accepted-alive-time-s",
        type=float,
        default=None,
        help="override the profile controller's accepted purified-pair staleness cutoff",
    )
    parser.add_argument("--age-points", type=int, default=8)
    parser.add_argument("--key-length", type=int, default=8)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = validate_alice_repeater_v1(
        profile_path=args.profile,
        conversion_path=args.conversion,
        detector_profile_path=args.detector_profile,
        cases=parse_cases(args.cases),
        accepted_alive_time_s=args.accepted_alive_time_s,
        age_points=args.age_points,
        key_length=args.key_length,
    )
    markdown = render_markdown(report)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(markdown)
    if args.out_json is None and args.out_md is None:
        print(markdown)
    return 0 if report["status"] != "failed_invariant" else 1


if __name__ == "__main__":
    raise SystemExit(main())
