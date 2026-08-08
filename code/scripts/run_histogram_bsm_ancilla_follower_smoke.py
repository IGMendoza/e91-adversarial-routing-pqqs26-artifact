#!/usr/bin/env python3
"""Run the compiled XX/ZZ BSM-ancilla follower smoke experiment."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from sequence_game.experiments.histogram_colearning.bsm_ancilla_follower import (
    BSMAncillaFollowerSmokeConfig,
    run_bsm_ancilla_follower_smoke,
)
from sequence_game.experiments.histogram_colearning.config import (
    ALICE_LEARNER_LINEAR_ROUTE_EXP2,
    ALICE_LEARNER_TSALLIS_INF,
    STRUCTURED_EXPLORATION_G_OPTIMAL,
    HistogramCoLearningConfig,
    REPO_ROOT,
)
from sequence_game.experiments.histogram_colearning.corpus import (
    build_topology_bench_case_for_endpoints,
)
from sequence_game.experiments.histogram_colearning.physics import (
    histogram_cache_plan,
)
from sequence_game.experiments.histogram_colearning.topology_census import (
    available_ram_snapshot,
)
from sequence_game.topology import (
    EdgeRecord,
    NodeRecord,
    TopologyIR,
    TopologyMetadata,
)


OUTPUT = REPO_ROOT / "runs" / "histogram_bsm_ancilla_follower_smoke"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    case = _smoke_case()
    base = HistogramCoLearningConfig(
        turns=80,
        requested_pairs_per_turn=256,
        final_window=40,
        record_every=10,
        posthoc_samples_per_cell=0,
        seed=91,
        game_mode="attack_surface_all",
        age_points_per_edge=2,
        evaluated_cache_size=16,
    )
    learner_configs = {
        "tsallis_inf_rv_eta1": replace(
            base,
            alice_learner_mode=ALICE_LEARNER_TSALLIS_INF,
            alice_tsallis_inf_estimator="rv",
            alice_tsallis_inf_eta_scale=1.0,
        ),
        "linear_exp2_goptimal_eta2": replace(
            base,
            alice_learner_mode=ALICE_LEARNER_LINEAR_ROUTE_EXP2,
            structured_exploration_mode=STRUCTURED_EXPLORATION_G_OPTIMAL,
            alice_exp3_eta_c=2.0,
        ),
    }
    smoke_config = BSMAncillaFollowerSmokeConfig()
    cache = histogram_cache_plan(case, base)
    preflight = {
        "case": case.to_dict(),
        "route_count": len(case.routes),
        "declared_game_action_count": len(case.actions(base.game_mode)),
        "smoke_probe_action_count": 4,
        "nominal_smoke_route_probe_cells": len(case.routes) * 4,
        "canonical_game_cache_entries": int(
            cache["canonical_unique_move_count_total"]
        ),
        "estimated_game_cache_bytes": int(
            cache["estimated_full_action_surface_cache_bytes"]
        ),
        "configured_lru_entries_per_route": base.evaluated_cache_size,
        "available_ram_before_run": available_ram_snapshot(),
        "smoke_config": smoke_config.to_dict(),
        "physics_scope": (
            "ideal commuting XX/ZZ BSM-stabilizer ancilla in the compiled "
            "histogram repeater model"
        ),
    }
    _write_json(OUTPUT / "preflight.json", preflight)
    print(
        "preflight: "
        f"{preflight['route_count']} routes, "
        f"{preflight['smoke_probe_action_count']} quantum probe actions, "
        f"{preflight['nominal_smoke_route_probe_cells']} route/probe cells"
    )
    result = run_bsm_ancilla_follower_smoke(
        case,
        learner_configs,
        smoke_config=smoke_config,
        progress_callback=print,
    )
    _write_json(OUTPUT / "smoke.json", result)
    _write_json(
        OUTPUT / "summary.json",
        {
            key: result[key]
            for key in (
                "schema",
                "case",
                "smoke_config",
                "learner_configs",
                "probe_contract",
                "commissioning",
                "expected_service_invariance",
                "aggregates",
                "elapsed_seconds",
                "information_boundary",
                "scientific_scope",
                "claim_boundary",
            )
        },
    )
    print(f"wrote {OUTPUT / 'summary.json'}")


def _smoke_case():
    ir = TopologyIR(
        nodes={
            node: NodeRecord(node)
            for node in ("alice", "r0", "r1", "bob")
        },
        edges=[
            EdgeRecord(
                "alice-r0",
                "alice",
                "r0",
                10_000.0,
                eve_eligible=True,
            ),
            EdgeRecord(
                "r0-bob",
                "r0",
                "bob",
                10_000.0,
                eve_eligible=True,
            ),
            EdgeRecord(
                "alice-r1",
                "alice",
                "r1",
                30_000.0,
                eve_eligible=True,
            ),
            EdgeRecord(
                "r1-bob",
                "r1",
                "bob",
                30_000.0,
                eve_eligible=True,
            ),
        ],
        metadata=TopologyMetadata("bsm-ancilla-follower-smoke"),
    )
    return build_topology_bench_case_for_endpoints(
        ir_dict=ir.to_dict(),
        graph_id="bsm-ancilla-follower-smoke",
        family="Diamond",
        alice="alice",
        bob="bob",
        corpus_id="diagnostic",
        corpus_db_path="not_applicable",
        assignment_id="alice_bob",
        assignment={},
    )


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
