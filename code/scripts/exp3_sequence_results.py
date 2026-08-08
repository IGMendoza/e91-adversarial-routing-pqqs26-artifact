#!/usr/bin/env python3
"""Reproduce the retained historical Exp3/oracle SeQUeNCe experiment."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["MPLCONFIGDIR"] = str(Path(tempfile.gettempdir()) / "sequence_matplotlib")

from sequence_game.experiments.exp3_sequence import (  # noqa: E402
    Exp3SequenceConfig,
    RouteVerificationConfig,
    run_pipeline,
    run_route_verification,
)
from sequence_game.experiments.exp3_sequence.config import (  # noqa: E402
    ALICE_ACCEPTANCE_RULES,
    DEFAULT_CONDITIONS,
    EXP3_SCHEDULE_MODES,
    default_worker_count,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=("all", "route-verify"),
        help=(
            "all: run payoff/oracle/Exp3/DT/plots; "
            "route-verify: run fixed-route SeQUeNCe CHSH calibration only"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("runs/exp3_sequence"))
    parser.add_argument("--workers", type=int, default=default_worker_count())
    parser.add_argument("--max-graphs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--trials-per-cell", type=int)
    parser.add_argument("--baseline-cache-db-path", type=Path)
    parser.add_argument("--attack-cache-db-path", type=Path)
    parser.add_argument("--attack-payoff-samples-per-route", type=int)
    parser.add_argument("--online-turns", type=int)
    parser.add_argument("--online-final-window", type=int)
    parser.add_argument("--online-step-record-stride", type=int)
    parser.add_argument("--alice-exp3-gamma", type=float)
    parser.add_argument("--eve-exp3-gamma", type=float)
    parser.add_argument(
        "--exp3-schedule-mode",
        choices=sorted(EXP3_SCHEDULE_MODES),
    )
    parser.add_argument("--alice-exp3-eta-c", type=float)
    parser.add_argument("--eve-exp3-eta-c", type=float)
    parser.add_argument("--alice-exp3-t0", type=float)
    parser.add_argument("--eve-exp3-t0", type=float)
    parser.add_argument("--alice-exp3-gamma-max", type=float)
    parser.add_argument("--eve-exp3-gamma-max", type=float)
    parser.add_argument(
        "--alice-acceptance-rule",
        choices=sorted(ALICE_ACCEPTANCE_RULES),
    )
    parser.add_argument("--alice-key-rate-shaping-weight", type=float)
    parser.add_argument(
        "--conditions",
        type=_parse_conditions,
        help=(
            "Comma-separated online condition keys to run. Available: "
            f"{', '.join(condition.key for condition in DEFAULT_CONDITIONS)}"
        ),
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config_kwargs = {
        "out_dir": args.out_dir,
        "workers": args.workers,
    }
    for field in (
        "max_graphs",
        "seed",
        "trials_per_cell",
        "baseline_cache_db_path",
        "attack_cache_db_path",
        "attack_payoff_samples_per_route",
        "online_turns",
        "online_final_window",
        "online_step_record_stride",
        "alice_exp3_gamma",
        "eve_exp3_gamma",
        "exp3_schedule_mode",
        "alice_exp3_eta_c",
        "eve_exp3_eta_c",
        "alice_exp3_t0",
        "eve_exp3_t0",
        "alice_exp3_gamma_max",
        "eve_exp3_gamma_max",
        "alice_acceptance_rule",
        "alice_key_rate_shaping_weight",
        "conditions",
    ):
        value = getattr(args, field)
        if value is not None:
            config_kwargs[field] = value
    config = Exp3SequenceConfig(**config_kwargs)
    if args.command == "route-verify":
        route_config = RouteVerificationConfig.from_exp3_config(config)
        payload = run_route_verification(route_config, progress=not args.quiet)
        print(f"wrote {route_config.out_dir / 'route_verification.json'}")
        print("representative no-attack CHSH sanity by hop:")
        for hops, row in payload["summary"]["hop_sanity_by_hop"].items():
            edge = row["max_passing_edge_length_m"]
            edge_label = "none" if edge is None else f"{edge / 1000.0:.3f} km"
            print(f"  hops={hops}: qualified edge = {edge_label}")
        return

    summary = run_pipeline(config, progress=not args.quiet)
    print(f"wrote {summary['out_dir']}")
    print(f"oracle: {summary['oracle_summary_path']}")
    print(f"exp3: {summary['exp3_summary_path']}")
    print(f"dt: {summary['dt_payload_path']}")
    print("figures:")
    for path in summary["figures"]:
        print(f"  {path}")


def _parse_conditions(value: str):
    by_key = {condition.key: condition for condition in DEFAULT_CONDITIONS}
    keys = [key.strip() for key in value.split(",") if key.strip()]
    if not keys:
        raise argparse.ArgumentTypeError("conditions must not be empty")
    unknown = [key for key in keys if key not in by_key]
    if unknown:
        available = ", ".join(sorted(by_key))
        raise argparse.ArgumentTypeError(
            f"unknown condition(s): {', '.join(unknown)}; available: {available}"
        )
    return tuple(by_key[key] for key in keys)


if __name__ == "__main__":
    main()
