#!/usr/bin/env python
"""Validate production-hop rolling-pool analytics with a discrete-event oracle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.physical.repeater_reservation import (
    reservation_edges_for_service_profile,
)
from sequence_game.physical.repeater_service import load_repeater_service_profile
from sequence_game.physical.rolling_controller_event import (
    RollingControllerEventConfig,
    rolling_controller_event_resource_estimate,
    simulate_rolling_controller_snapshots,
)


DEFAULT_PROFILE = (
    REPO_ROOT / "configs" / "physical" / "repeater_services"
    / "cui_2026_rolling_purified_near_term_composite.json"
)


def _parse_cases(value: str) -> list[tuple[int, float]]:
    cases = []
    for item in value.split(","):
        hops, length = item.lower().split("x", 1)
        cases.append((int(hops), float(length)))
    if not cases:
        raise ValueError("at least one HxKM case is required")
    return cases


def _case_passed(result: dict) -> tuple[bool, dict]:
    route = result["route_availability"]
    route_tolerance = max(
        0.01,
        3.0 * float(route["batch_standard_error"] or 0.0),
    )
    edge_checks = []
    age_checks = []
    for edge in result["edges"]:
        availability = edge["availability"]
        tolerance = max(
            0.01,
            3.0 * float(availability["batch_standard_error"] or 0.0),
        )
        edge_checks.append(availability["absolute_error"] <= tolerance)
        age = edge["selected_residence_age_s"]
        age_tolerance = max(
            0.001,
            3.0 * float(age["batch_standard_error_of_mean"] or 0.0),
        )
        age_checks.append(age["absolute_mean_error"] <= age_tolerance)
    checks = {
        "route_availability": route["absolute_error"] <= route_tolerance,
        "all_edge_availabilities": all(edge_checks),
        "all_selected_age_means": all(age_checks),
        "route_absolute_tolerance": route_tolerance,
    }
    return all(checks[key] for key in (
        "route_availability", "all_edge_availabilities", "all_selected_age_means"
    )), checks


def run(args: argparse.Namespace) -> dict:
    profile = load_repeater_service_profile(args.profile)
    accepted_alive = float(
        profile.controller_model["accepted_output_alive_time_s"]
    )
    case_specs = _parse_cases(args.cases)
    prepared = []
    for index, (hops, length_km) in enumerate(case_specs):
        edges = reservation_edges_for_service_profile(
            profile,
            [length_km] * hops,
            accepted_alive_time_s=accepted_alive,
        )
        config = RollingControllerEventConfig(
            num_samples=args.samples,
            burn_in_mean_cycles=args.burn_in_cycles,
            mean_sample_interval_s=args.sample_interval_s,
            empty_time_distribution="exponential",
            batch_count=args.batches,
            seed=args.seed + index,
        )
        estimate = rolling_controller_event_resource_estimate(edges, config)
        prepared.append((hops, length_km, edges, config, estimate))

    peak_mib = max(row[4]["estimated_peak_mib"] for row in prepared)
    max_events = max(row[4]["expected_processed_events"] for row in prepared)
    preflight = {
        "cases_execute_sequentially": True,
        "peak_case_estimated_ram_mib": peak_mib,
        "maximum_case_expected_events": max_events,
        "route_attack_cache_entries": 0,
        "ram_limit_mib": args.max_estimated_ram_mib,
        "event_limit": args.max_estimated_events,
    }
    if peak_mib > args.max_estimated_ram_mib:
        raise RuntimeError(
            f"estimated peak {peak_mib:.1f} MiB exceeds explicit limit "
            f"{args.max_estimated_ram_mib:.1f} MiB"
        )
    if max_events > args.max_estimated_events:
        raise RuntimeError(
            f"estimated events {max_events} exceed explicit limit "
            f"{args.max_estimated_events}"
        )

    cases = []
    for hops, length_km, edges, config, estimate in prepared:
        print(
            f"[controller] {hops}x{length_km:g}km: "
            f"{estimate['estimated_peak_mib']:.1f} MiB, "
            f"~{estimate['expected_processed_events']:,} events",
            flush=True,
        )
        result = simulate_rolling_controller_snapshots(edges, config)
        passed, checks = _case_passed(result)
        cases.append({
            "hop_count": hops,
            "edge_length_km": length_km,
            "passed": passed,
            "checks": checks,
            "event_reference": result,
        })
    return {
        "schema": "rolling_controller_production_hop_validation.v1",
        "profile_id": profile.profile_id,
        "profile_path": str(Path(args.profile).resolve()),
        "empty_time_mapping": (
            "profile completion_time_model=empirical_exponential_renewal mapped "
            "to independent exponential empty-slot durations with the adapter's "
            "per-lane mean completion time"
        ),
        "preflight": preflight,
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
        "claim_boundary": (
            "validates stationary rolling-pool availability and selected ages at "
            "production hop counts; does not validate optical states, swap gates, "
            "or a continuous-demand queue"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE))
    parser.add_argument("--cases", default="12x50,16x50")
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--burn-in-cycles", type=float, default=100.0)
    parser.add_argument("--sample-interval-s", type=float, default=None)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-estimated-ram-mib", type=float, default=1024.0)
    parser.add_argument("--max-estimated-events", type=int, default=50_000_000)
    parser.add_argument("--out", default="runs/rolling_controller_event_validation.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {out}; passed={payload['passed']}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
