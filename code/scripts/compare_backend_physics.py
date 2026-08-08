#!/usr/bin/env python
"""Measured physics comparison: compiled-histogram backend vs SeQUeNCe runtime.

Runs fixed (graph, route, Eve move) cells on three reference graphs through
both backends and reports aligned observables side by side. This is a
validation harness, not a boundary of truth: the two backends implement
different physics stacks on top of the same configs/physical device stubs, so
absolute agreement is only meaningful for cells tagged as the same comparison
family. The validation targets are (a) sane, explainable magnitudes for matched
repeater-chain cells and (b) separate qualitative directional stress checks for
explicitly cross-family cells.

Reference graphs:
  braess     4-node Braess diamond, km-scale asymmetric fiber lengths;
  k10        10-node fully connected graph, every fiber exactly 1 km
             (deliberately non-embeddable in R^3 -- a stress shape);
  metro_wan  semi-realistic two-metro quantum-internet topology with
             constrained backbone entry points.

Engine families (the SeQUeNCe runtime dispatches by action kind):
  edge family  -> SeQUeNCe direct-path E91 trial (no entanglement swapping);
                  moves: no_attack, edge_dos, edge_information_probe.
  node family  -> SeQUeNCe fixed repeater chain trial (explicit swaps);
                  moves: repeater baseline (inactive spec), swap_denial,
                  memory_degradation, repeater_memory_measure_Z.
The compiled-histogram backend models a repeater chain for every move in this
harness. Node-family rows are the matched repeater-chain comparison. Edge-family
rows are qualitative cross-family stress checks unless a matching compiled
direct-path E91 backend is added.

Known alignment shims (documented in the output):
  * SeQUeNCe repeater trials need length-aware RSVP/App timing windows; the
    runtime game's fixed 1 us start time asserts out at km scale (recorded as
    a finding), so this script scales start/end/stop with route length.
  * The compiled backend is configured with the SeQUeNCe QBER threshold
    (0.15) and the edge-family min-sifted acceptance gate (20) is applied to
    compiled outcomes post hoc for the aligned-outcome columns.
  * Repeater-chain validation uses an exact pre-measurement state expectation
    at the same Ekert settings on both backends. This avoids confusing an
    under-sampled event transcript with state quality. It is not an executed
    finite-shot Bell test or a security proof.
  * memory_degradation is parameterized differently per backend (SeQUeNCe:
    fidelity multiplier 0.5; compiled: depol 0.5 + dephase 0.5): direction,
    not magnitude, is comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.corpus import e91_runtime_game as sequence_runtime
from sequence_game.corpus.compiled_histogram_backend import (
    COMPILED_HISTOGRAM_BACKEND,
    CompiledHistogramBackend,
    CompiledHistogramBackendConfig,
    DEFAULT_CONVERSION_RESPONSE_PATH,
    DEFAULT_DETECTOR_PROFILE_PATH,
    classify_qber_estimation_power,
    eve_move_from_runtime_action,
    fiber_vector_from_route,
)
from sequence_game.corpus.runner import corpus_candidate_routes
from sequence_game.game_theory import exploitability, solve_zero_sum_lp
from sequence_game.routing.route import make_route
from sequence_game.topology import TopologyIR
from sequence_game.physical.memory_pool_reservation import route_live_probability

COMPARISON_VERSION = "backend_physics_comparison.v4"
GAME_MODE = "attack_surface_mixed"
FIBER_CLASSICAL_SPEED_M_PER_PS = 2e-4  # matches configs/physical light_speed
LOW_DELIVERY_POWER_THRESHOLD = 1e-3
SEVERE_CAP_BIAS_CLASSES = {"severe_cap_bias", "catastrophic_cap_bias"}
MIN_COMPARISON_QBER_OBSERVABLE_TRIALS = 3

SEQUENCE_DIRECT_EDGE_STACK = "sequence_direct_path_e91_no_entanglement_swapping"
SEQUENCE_REPEATER_STACK = "sequence_event_driven_fixed_repeater_chain"
COMPILED_REPEATER_STACK = "compiled_fock_pnr_snspd_repeater_chain_histogram"
COMPILED_DIRECT_STACK_MISSING = "not_implemented_compiled_direct_path_e91"

COMPARISON_MODES = (
    "mixed_runtime",
    "fixed_route_repeater_chain",
    "fixed_route_strategic_matrix",
)
SECURITY_OBSERVABLES = (
    "endpoint_state_expectation",
    "finite_qber_transcript",
)

CAVEATS = [
    "Comparison families are explicit per cell. SeQUeNCe edge trials are "
    "direct-path E91 without entanglement swapping and are qualitative "
    "cross-family checks against the compiled repeater-chain histogram unless "
    "a matching compiled direct-path backend is added. SeQUeNCe node trials are "
    "the matched event-driven repeater-chain comparison.",
    "SeQUeNCe repeater trials require length-aware timing windows; the corpus "
    "runtime game's fixed start_time_ps=1e6 raises an RSVP assertion at km-scale "
    "routes (finding recorded per cell as sequence_error when it occurs).",
    "In endpoint_state_expectation mode, repeater-chain cells compare the same "
    "fixed Ekert CHSH and key-QBER functionals on the actual event-delivered "
    "state and the compiled histogram state. Bell-qualified delivery requires "
    "|S|>2 and QBER<=0.15 on both sides.",
    "The event state witness is evaluated before destructive endpoint measurement "
    "after both swap feed-forward messages arrive. It removes finite-shot sampling "
    "noise, but is not an executed Bell-test transcript or finite-key proof.",
    "memory_degradation parameterizations differ per backend (fidelity multiplier "
    "0.5 vs depol 0.5 + dephase 0.5); compare direction, not magnitude.",
    "SeQUeNCe node trials report mean_actual_fidelity; the compiled transcript "
    "intentionally exposes no single final-pair fidelity scalar.",
    "Waiting/protocol alignment depends on the compiled mode. In the legacy "
    "single_buffer_swap_asap comparison, the compiled backend runs in "
    "fixed_wall_clock_time mode with each requested pair given the per-pair "
    "share (window / memory_pairs) of the SeQUeNCe repeater trial window. In "
    "the rolling_alive_pair comparison, the compiled backend instead uses "
    "configured live-pair availability and hidden selected-age distributions "
    "from the 10 km hop calibration; this is a different protocol abstraction, "
    "so absolute rate agreement with SeQUeNCe event scheduling remains a "
    "small-h sanity check rather than protocol equivalence.",
    "Direct-path information-probe fingerprints differ by construction: the compiled "
    "edge_measure_resend is a fixed-basis intercept-resend whose disturbance "
    "appears in CHSH (infinite-shot S halves; key-basis QBER unchanged), while "
    "the SeQUeNCe edge_information_probe raises sifted-key QBER (~+0.25 "
    "measured) and has no matched event state witness. QBER-direction checks on "
    "these explicitly cross-family probe cells can legitimately disagree.",
]


def classify_cap_bias_factor(value: float | None) -> str:
    """Classify route-ready probability lost to waiting history/signature caps.

    The factor is full-budget route-ready probability divided by the capped
    route-ready probability used by the compiled production histogram. Missing
    values occur for h=1 or failed diagnostics and are kept out of comparable
    agreement filtering.
    """
    if value is None:
        return "not_applicable"
    value = float(value)
    if not math.isfinite(value):
        return "catastrophic_cap_bias"
    if value <= 1.05:
        return "no_cap_bias"
    if value <= 2.0:
        return "mild_cap_bias"
    if value <= 10.0:
        return "moderate_cap_bias"
    if value <= 1e3:
        return "severe_cap_bias"
    return "catastrophic_cap_bias"


def _is_severe_cap_bias(classification: str | None) -> bool:
    return str(classification or "") in SEVERE_CAP_BIAS_CLASSES


def _cap_bias_rank(classification: str | None) -> int:
    order = {
        "not_applicable": 0,
        "no_cap_bias": 1,
        "mild_cap_bias": 2,
        "moderate_cap_bias": 3,
        "severe_cap_bias": 4,
        "catastrophic_cap_bias": 5,
    }
    return order.get(str(classification or "not_applicable"), 0)


def _cell_key(cell: dict[str, Any]) -> str:
    return f"{cell['graph_id']}::{cell['route_id']}::{cell['family']}::{cell['label']}"


def _compiled_expected(cell: dict[str, Any]) -> dict[str, Any]:
    return (cell.get("compiled", {}).get("expected_infinite_shot") or {})


def _cell_cap_bias_classification(cell: dict[str, Any]) -> str:
    diagnostic = cell.get("waiting_cap_diagnostic")
    if not diagnostic:
        diagnostic = _compiled_expected(cell).get("waiting_cap_diagnostic")
    if not diagnostic:
        diagnostic = _compiled_expected(cell).get("uncapped_route_ready_diagnostic")
    return str((diagnostic or {}).get("cap_bias_classification", "not_applicable"))


def _cell_low_delivery_power(cell: dict[str, Any]) -> bool:
    compiled_expected = _compiled_expected(cell)
    candidates = [
        compiled_expected.get("delivery_probability"),
        cell.get("compiled", {}).get("delivered_fraction_mean"),
        cell.get("sequence", {}).get("delivered_fraction_mean"),
    ]
    finite = [
        float(value) for value in candidates
        if value is not None and math.isfinite(float(value))
    ]
    return bool(finite and max(finite) < LOW_DELIVERY_POWER_THRESHOLD)


def comparison_contract_for_cell(route: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    """Declare whether the two engines implement matching physics for a cell."""
    family = str(cell.get("family", ""))
    hop_count = int(route.get("hop_count", max(0, len(route.get("path", [])) - 1)))
    action_semantics = cell.get("action_semantics") or {}
    action_status = str(action_semantics.get("status", "not_declared"))
    if str(cell.get("sequence_reference_mode", "")) == "fixed_route_repeater_chain":
        if hop_count < 2:
            return {
                "comparison_family": "repeater_chain_family",
                "physical_stack_sequence": "sequence_fixed_route_repeater_chain_reference",
                "physical_stack_compiled": COMPILED_REPEATER_STACK,
                "absolute_rate_comparable": False,
                "directional_only": False,
                "reason_if_not_comparable": (
                    "fixed-route repeater-chain reference requires hop_count >= 2"
                ),
                "compiled_direct_path_available": False,
                "requires_entanglement_swapping": True,
                "action_semantics_status": action_status,
            }
        if action_status == "unsupported_sequence_reference":
            return {
                "comparison_family": "repeater_chain_family",
                "physical_stack_sequence": "sequence_fixed_route_repeater_chain_reference",
                "physical_stack_compiled": COMPILED_REPEATER_STACK,
                "absolute_rate_comparable": False,
                "directional_only": False,
                "reason_if_not_comparable": action_semantics.get("reason"),
                "compiled_direct_path_available": False,
                "requires_entanglement_swapping": True,
                "action_semantics_status": action_status,
            }
        return {
            "comparison_family": "repeater_chain_family",
            "physical_stack_sequence": "sequence_fixed_route_repeater_chain_reference",
            "physical_stack_compiled": COMPILED_REPEATER_STACK,
            "absolute_rate_comparable": action_status == "matched",
            "directional_only": action_status == "approximate",
            "reason_if_not_comparable": (
                action_semantics.get("reason") if action_status == "approximate" else None
            ),
            "compiled_direct_path_available": False,
            "requires_entanglement_swapping": True,
            "action_semantics_status": action_status,
        }
    if family == "edge":
        return {
            "comparison_family": "qualitative_cross_family",
            "physical_stack_sequence": SEQUENCE_DIRECT_EDGE_STACK,
            "physical_stack_compiled": COMPILED_REPEATER_STACK,
            "absolute_rate_comparable": False,
            "directional_only": True,
            "reason_if_not_comparable": (
                "SeQUeNCe edge-family cells use direct-path E91 without entanglement "
                "swapping, while the compiled backend currently evaluates the Fock/SNSPD "
                "repeater-chain histogram. No matching compiled direct-path E91 backend "
                "is implemented in this harness."
            ),
            "compiled_direct_path_available": False,
            "requires_entanglement_swapping": False,
        }
    if family == "node":
        if hop_count < 2:
            return {
                "comparison_family": "repeater_chain_family",
                "physical_stack_sequence": SEQUENCE_REPEATER_STACK,
                "physical_stack_compiled": COMPILED_REPEATER_STACK,
                "absolute_rate_comparable": False,
                "directional_only": False,
                "reason_if_not_comparable": (
                    "Repeater-chain comparison requires hop_count >= 2 so at least one "
                    "repeater swap is represented."
                ),
                "compiled_direct_path_available": False,
                "requires_entanglement_swapping": True,
            }
        return {
            "comparison_family": "repeater_chain_family",
            "physical_stack_sequence": SEQUENCE_REPEATER_STACK,
            "physical_stack_compiled": COMPILED_REPEATER_STACK,
            "absolute_rate_comparable": True,
            "directional_only": False,
            "reason_if_not_comparable": None,
            "compiled_direct_path_available": False,
            "requires_entanglement_swapping": True,
        }
    return {
        "comparison_family": "not_comparable",
        "physical_stack_sequence": "unknown",
        "physical_stack_compiled": COMPILED_REPEATER_STACK,
        "absolute_rate_comparable": False,
        "directional_only": False,
            "reason_if_not_comparable": f"unknown comparison cell family {family!r}",
            "compiled_direct_path_available": False,
            "requires_entanglement_swapping": None,
            "action_semantics_status": action_status,
        }


def _metric_value(metrics: dict[str, Any], observable: str) -> Any:
    if observable == "sampled_qber_mean":
        return metrics.get("sampled_qber_mean", metrics.get("qber_mean"))
    return metrics.get(observable)


def _qber_power(metrics: dict[str, Any]) -> str:
    return str(metrics.get("qber_estimation_power", "adequate"))


def _qber_underpowered(*metric_sets: dict[str, Any]) -> bool:
    return any(_qber_power(metrics) != "adequate" for metrics in metric_sets)


def _agreement_summary(checks: list[dict[str, Any]]) -> dict[str, Any]:
    decided = [check for check in checks if check["agreement"] is not None]
    agreeing = [check for check in decided if check["agreement"]]
    return {
        "decided_checks": len(decided),
        "agreeing_checks": len(agreeing),
        "agreement_rate": (len(agreeing) / len(decided)) if decided else None,
        "disagreements": [check for check in decided if not check["agreement"]],
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _tuple_tree(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


def _compiled_device_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "detector_kind": str(getattr(args, "detector_kind", "pnr_snspd")),
        "detector_profile_path": getattr(
            args, "detector_profile", str(DEFAULT_DETECTOR_PROFILE_PATH)
        ),
        "detector_response_path": getattr(args, "detector_response_file", None),
        "fiber_response_path": getattr(args, "fiber_response_file", None),
        "conversion_response_path": getattr(
            args, "conversion_response_file", str(DEFAULT_CONVERSION_RESPONSE_PATH)
        ),
        "allow_wavelength_mismatch": bool(
            getattr(args, "allow_wavelength_mismatch", False)
        ),
        "allow_ideal_detector_reference": bool(
            getattr(args, "allow_ideal_detector_reference", False)
        ),
        "require_explicit_device_stack": True,
    }


class CompiledBackendManager:
    """Own compiled backend instances for static or hop-calibrated comparison."""

    def __init__(self, args: argparse.Namespace, *, per_pair_window_ps: float) -> None:
        self.args = args
        self.per_pair_window_ps = float(per_pair_window_ps)
        self.hop_calibrated = bool(getattr(args, "compiled_histogram_hop_calibration_10km", False))
        self._static_backend: CompiledHistogramBackend | None = None
        self._pool: dict[int, CompiledHistogramBackend] = {}
        self._metadata: dict[int, dict[str, Any]] = {}
        if not self.hop_calibrated:
            self._static_backend = CompiledHistogramBackend(
                self._single_buffer_config(args, per_pair_window_ps=self.per_pair_window_ps)
            )

    @staticmethod
    def _single_buffer_config(
        args: argparse.Namespace,
        *,
        per_pair_window_ps: float,
    ) -> CompiledHistogramBackendConfig:
        return CompiledHistogramBackendConfig(
            **_compiled_device_config(args),
            fock_cutoff=int(args.fock_cutoff),
            waiting_max_wait_signatures=int(args.signature_cap),
            waiting_max_history_pairs=int(args.history_pair_cap),
            waiting_tail_tolerance=float(args.waiting_tail_tolerance),
            waiting_budget_mode="fixed_wall_clock_time",
            waiting_budget_wall_clock_ps=float(per_pair_window_ps),
            memory_cutoff_ps=args.memory_cutoff_ps,
            qber_abort_threshold=float(sequence_runtime.DEFAULT_E91_QBER_THRESHOLD),
            chsh_abort_threshold=(
                2.0 if args.security_observable == "endpoint_state_expectation"
                else -10.0
            ),
            min_key_pairs_for_qber_abort=1,
            evaluated_cache_size=int(args.evaluated_cache_size),
        )

    def _rolling_config_for_hop(self, hop_count: int) -> tuple[CompiledHistogramBackendConfig, dict[str, Any]]:
        from scripts.derive_hop_based_rolling_reservation_calibration import (
            HopCalibrationConfig,
            derive_hop_reservation,
        )

        hop = int(hop_count)
        cal_config = HopCalibrationConfig(
            max_hop=hop,
            stress_max_hop=hop,
            edge_length_km=10.0,
            accepted_alive_time_s=float(self.args.compiled_histogram_hop_calibration_t_alive_s),
            route_availability_target=float(
                self.args.compiled_histogram_hop_calibration_route_availability
            ),
            chsh_target=2.10,
            age_points_per_edge=int(self.args.compiled_histogram_hop_calibration_age_points),
            selection_policy="youngest_live",
            detector_kind=_compiled_device_config(self.args)["detector_kind"],
            detector_profile_path=_compiled_device_config(self.args)["detector_profile_path"],
            detector_response_path=_compiled_device_config(self.args)["detector_response_path"],
            fiber_response_path=_compiled_device_config(self.args)["fiber_response_path"],
            conversion_response_path=_compiled_device_config(self.args)["conversion_response_path"],
            allow_wavelength_mismatch=_compiled_device_config(self.args)["allow_wavelength_mismatch"],
            allow_ideal_detector_reference=_compiled_device_config(self.args)["allow_ideal_detector_reference"],
            fock_cutoff=int(self.args.fock_cutoff),
            key_length=8,
            c_max=int(self.args.compiled_histogram_hop_calibration_c_max),
        )
        _edge_inputs, _front_meta, plan, edges, payload = derive_hop_reservation(
            hop,
            cal_config,
        )
        config = CompiledHistogramBackendConfig(
            **_compiled_device_config(self.args),
            fock_cutoff=int(self.args.fock_cutoff),
            waiting_protocol="rolling_alive_pair",
            rolling_edge_availability=_tuple_tree(payload["rolling_edge_availability"]),
            rolling_age_distribution_kind=payload["rolling_age_distribution_kind"],
            rolling_age_distribution_points_s=_tuple_tree(
                payload["rolling_age_distribution_points_s"]
            ),
            rolling_age_distribution_weights=_tuple_tree(
                payload["rolling_age_distribution_weights"]
            ),
            rolling_edge_age_distribution_points_s=_tuple_tree(
                payload["rolling_edge_age_distribution_points_s"]
            ),
            rolling_edge_age_distribution_weights=_tuple_tree(
                payload["rolling_edge_age_distribution_weights"]
            ),
            rolling_age_evaluation="moment_reduced",
            rolling_age_support_product_limit=1_000_000,
            rolling_accepted_alive_time_s=payload["rolling_accepted_alive_time_s"],
            rolling_selection_policy=payload["rolling_selection_policy"],
            rolling_age_distribution_source=payload["rolling_age_distribution_source"],
            qber_abort_threshold=float(sequence_runtime.DEFAULT_E91_QBER_THRESHOLD),
            chsh_abort_threshold=(
                2.0 if self.args.security_observable == "endpoint_state_expectation"
                else -10.0
            ),
            min_key_pairs_for_qber_abort=1,
            evaluated_cache_size=int(self.args.evaluated_cache_size),
        )
        metadata = {
            "config_mode": "rolling_hop_calibration",
            "calibration_kind": "hop_based_10km",
            "hop_count": hop,
            "calibration_edge_length_km": 10.0,
            "runtime_lengths_policy": (
                "rolling reservation resources are derived from the standardized "
                "10 km-per-hop calibration family; optical propagation is still "
                "evaluated at the selected route's actual fiber vector"
            ),
            "accepted_alive_time_s": cal_config.accepted_alive_time_s,
            "route_availability_target": cal_config.route_availability_target,
            "chsh_target": cal_config.chsh_target,
            "age_points_per_edge": cal_config.age_points_per_edge,
            "selection_policy": cal_config.selection_policy,
            "slot_vector": list(plan.reserved_slots),
            "total_slots": int(sum(plan.reserved_slots)),
            "edge_availabilities": list(plan.edge_availabilities),
            "route_availability": route_live_probability(edges),
            "rolling_age_support_product_avoided": (
                int(cal_config.age_points_per_edge) ** hop
            ),
            "rolling_age_evaluation": "moment_reduced",
        }
        return config, metadata

    def backend_for_route(self, route: dict[str, Any]) -> CompiledHistogramBackend:
        if not self.hop_calibrated:
            assert self._static_backend is not None
            return self._static_backend
        hop_count = int(route.get("hop_count", max(0, len(route.get("path", [])) - 1)))
        backend = self._pool.get(hop_count)
        if backend is None:
            config, metadata = self._rolling_config_for_hop(hop_count)
            backend = CompiledHistogramBackend(config)
            self._pool[hop_count] = backend
            self._metadata[hop_count] = metadata
        return backend

    def diagnostics(self) -> dict[str, Any]:
        if not self.hop_calibrated:
            assert self._static_backend is not None
            diag = self._static_backend.diagnostics()
            diag["backend_pool_enabled"] = False
            diag["hop_calibrations"] = {}
            return diag
        stats = {
            "kernel_compiles": 0,
            "kernel_cache_hits": 0,
            "kernel_compile_seconds": 0.0,
            "evaluated_cache_hits": 0,
            "evaluated_cache_misses": 0,
            "evaluated_cache_evictions": 0,
            "evaluate_seconds": 0.0,
            "steps_sampled": 0,
            "step_sample_seconds": 0.0,
        }
        backends = {}
        warnings: list[str] = []
        first_diag: dict[str, Any] | None = None
        for hop_count, backend in sorted(self._pool.items()):
            diag = backend.diagnostics()
            first_diag = first_diag or diag
            for key, value in diag.get("stats", {}).items():
                if isinstance(value, (int, float)):
                    stats[key] = stats.get(key, 0.0) + value
            warnings.extend(diag.get("unsupported_warnings", []))
            backends[str(hop_count)] = {
                "schema_hash": diag.get("schema_hash"),
                "kernel_cache_size": diag.get("kernel_cache_size"),
                "evaluated_cache_size": diag.get("evaluated_cache_size"),
                "stats": diag.get("stats", {}),
                "calibration": self._metadata.get(hop_count, {}),
                "backend_config": diag.get("backend_config"),
            }
        return {
            "backend": COMPILED_HISTOGRAM_BACKEND,
            "backend_scope": (first_diag or {}).get("backend_scope"),
            "backend_pool_enabled": True,
            "backend_count": len(self._pool),
            "generator_version": (first_diag or {}).get("generator_version"),
            "compiled_kernel_version": (first_diag or {}).get("compiled_kernel_version"),
            "front_matter_hash": (first_diag or {}).get("front_matter_hash"),
            "schema_hash": (first_diag or {}).get("schema_hash"),
            "kernel_cache_size": sum(
                int(row.get("kernel_cache_size") or 0) for row in backends.values()
            ),
            "evaluated_cache_size": sum(
                int(row.get("evaluated_cache_size") or 0) for row in backends.values()
            ),
            "backend_config": {
                "compiled_histogram_hop_calibration_10km": True,
                "compiled_histogram_hop_calibration_t_alive_s": float(
                    self.args.compiled_histogram_hop_calibration_t_alive_s
                ),
                "compiled_histogram_hop_calibration_age_points": int(
                    self.args.compiled_histogram_hop_calibration_age_points
                ),
                "evaluated_cache_size": int(self.args.evaluated_cache_size),
                "fock_cutoff": int(self.args.fock_cutoff),
            },
            "stats": stats,
            "hop_calibrations": {
                str(hop): metadata for hop, metadata in sorted(self._metadata.items())
            },
            "backends": backends,
            "unsupported_warnings": sorted(set(warnings)),
        }


# ── Reference graphs ─────────────────────────────────────────────────────────

def _nodes(node_ids: list[str]) -> dict[str, Any]:
    return {
        node_id: {"node_id": node_id, "roles": ["memory_node", "swap_candidate"]}
        for node_id in node_ids
    }


def _edge(u: str, v: str, length_m: float) -> dict[str, Any]:
    return {"edge_id": f"{u}-{v}", "u": u, "v": v, "length_m": float(length_m)}


def build_braess() -> dict[str, Any]:
    ir_dict = {
        "nodes": _nodes(["n0", "n1", "n2", "n3"]),
        "edges": [
            _edge("n0", "n1", 1200.0),
            _edge("n0", "n2", 2400.0),
            _edge("n1", "n2", 800.0),
            _edge("n1", "n3", 2000.0),
            _edge("n2", "n3", 1600.0),
        ],
        "metadata": {"generator": "manual"},
    }
    return {
        "graph_id": "braess_diamond",
        "description": "Braess diamond, asymmetric km-scale fibers",
        "ir_dict": ir_dict,
        "alice": "n0",
        "bob": "n3",
        "max_routes": 3,
    }


def build_k10() -> dict[str, Any]:
    node_ids = [f"q{i}" for i in range(10)]
    edges = [
        _edge(u, v, 1000.0)
        for u, v in combinations(node_ids, 2)
    ]
    return {
        "graph_id": "k10_uniform_1km",
        "description": "fully connected 10-node graph, every fiber exactly 1 km",
        "ir_dict": {
            "nodes": _nodes(node_ids),
            "edges": edges,
            "metadata": {"generator": "manual"},
        },
        "alice": "q0",
        "bob": "q9",
        # representative depth slices of an otherwise huge route set
        "explicit_paths": [
            ["q0", "q9"],
            ["q0", "q4", "q9"],
            ["q0", "q2", "q6", "q9"],
        ],
    }


def build_metro_wan() -> dict[str, Any]:
    ir_dict = {
        "nodes": _nodes([
            "alice", "a1", "a2", "a3", "b1", "b2", "c1", "c2", "c3", "bob",
        ]),
        "edges": [
            # metro A access ring
            _edge("alice", "a1", 1500.0),
            _edge("alice", "a2", 2200.0),
            _edge("a1", "a2", 1000.0),
            _edge("a2", "a3", 1800.0),
            _edge("a1", "a3", 2500.0),
            # constrained backbone entries: only a3->b1 and a2->b2
            _edge("a3", "b1", 4500.0),
            _edge("a2", "b2", 5500.0),
            _edge("b1", "b2", 3000.0),
            # constrained metro-B entries: only b1->c1 and b2->c2
            _edge("b1", "c1", 4000.0),
            _edge("b2", "c2", 5000.0),
            # metro B access ring
            _edge("c1", "c2", 1200.0),
            _edge("c1", "c3", 2000.0),
            _edge("c2", "c3", 1500.0),
            _edge("c3", "bob", 1000.0),
            _edge("c2", "bob", 1800.0),
            _edge("c1", "bob", 2200.0),
        ],
        "metadata": {"generator": "manual"},
    }
    return {
        "graph_id": "metro_wan_5hop",
        "description": (
            "two-metro quantum-internet WAN with constrained backbone entry "
            "points"
        ),
        "ir_dict": ir_dict,
        "alice": "alice",
        "bob": "bob",
        "max_routes": 3,
    }


GRAPH_BUILDERS = {
    "braess": build_braess,
    "k10": build_k10,
    "metro_wan": build_metro_wan,
}


def select_routes(spec: dict[str, Any]) -> list[dict[str, Any]]:
    if "explicit_paths" in spec:
        ir = TopologyIR.from_dict(spec["ir_dict"])
        return [
            {
                **make_route(ir, tuple(path)).to_dict(),
                "internal_nodes": list(path[1:-1]),
                "edge_lengths_m": [
                    float(ir.edge_between(left, right).length_m)
                    for left, right in zip(path, path[1:])
                ],
            }
            for path in spec["explicit_paths"]
        ]
    routes, _meta = corpus_candidate_routes(
        spec["ir_dict"],
        spec["alice"],
        spec["bob"],
        {},
    )
    # shortest, longest, and a middle alternative
    if len(routes) <= spec.get("max_routes", 3):
        return routes
    picks = [routes[0], routes[len(routes) // 2], routes[-1]]
    seen: set[str] = set()
    unique = []
    for route in picks:
        if route["route_id"] not in seen:
            seen.add(route["route_id"])
            unique.append(route)
    return unique


def strategic_node_target_selection(
    routes: list[dict[str, Any]],
    *,
    limit: int = 2,
) -> dict[str, Any]:
    """Select fixed graph-level node targets using topology only.

    The first target has maximum route support.  Later targets are selected by
    farthest-first Hamming distance between route-incidence vectors.  No
    physics outcome or backend prediction enters the selection.
    """

    if limit < 1:
        raise ValueError("strategic target limit must be >= 1")
    eligible = [route for route in routes if int(route.get("hop_count", 0)) >= 2]
    route_ids = [str(route["route_id"]) for route in eligible]
    nodes = sorted({
        str(node)
        for route in eligible
        for node in (route.get("internal_nodes") or route["path"][1:-1])
    })
    incidence = {
        node: tuple(
            int(node in set(map(str, route.get("internal_nodes") or route["path"][1:-1])))
            for route in eligible
        )
        for node in nodes
    }
    selected: list[str] = []
    if nodes:
        selected.append(sorted(nodes, key=lambda node: (-sum(incidence[node]), node))[0])
    while len(selected) < min(limit, len(nodes)):
        candidates = [node for node in nodes if node not in selected]

        def score(node: str) -> tuple[int, int, str]:
            distance = min(
                sum(left != right for left, right in zip(incidence[node], incidence[other]))
                for other in selected
            )
            return (distance, sum(incidence[node]), node)

        # Stable lexical tie-breaking is expressed by the final min rather
        # than relying on reverse ordering of strings.
        best_score = max((score(node)[:2] for node in candidates), default=(0, 0))
        selected.append(min(
            node for node in candidates if score(node)[:2] == best_score
        ))
    return {
        "selection_rule": (
            "topology-only maximum route support followed by farthest-first "
            "route-incidence Hamming diversity"
        ),
        "requested_target_count": int(limit),
        "eligible_route_ids": route_ids,
        "candidate_node_count": len(nodes),
        "selected_targets": selected,
        "targets": [
            {
                "node": node,
                "route_support_count": int(sum(incidence[node])),
                "route_support_fraction": (
                    sum(incidence[node]) / len(eligible) if eligible else None
                ),
                "route_incidence": {
                    route_id: bool(active)
                    for route_id, active in zip(route_ids, incidence[node])
                },
            }
            for node in selected
        ],
        "physics_or_payoff_used_for_selection": False,
    }


def strategic_actions_for_routes(
    routes: list[dict[str, Any]],
    *,
    target_limit: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = strategic_node_target_selection(routes, limit=target_limit)
    actions = [{
        "id": "repeater_baseline",
        "kind": "node",
        "target": "",
        "attack_type": "no_attack",
        "cost": 0.0,
    }]
    for target in selection["selected_targets"]:
        for attack_type in ("swap_denial", "repeater_memory_measure_Z"):
            actions.append({
                "id": f"{attack_type}:{target}",
                "kind": "node",
                "target": target,
                "attack_type": attack_type,
                "cost": 0.05,
            })
    return actions, selection


# ── Cell definitions ─────────────────────────────────────────────────────────

def fixed_route_repeater_action_semantics(action: dict[str, Any]) -> dict[str, Any]:
    """Reference-mode action support for the fixed-route repeater chain."""
    attack_type = str(action.get("attack_type") or "")
    if attack_type == "no_attack":
        return {
            "status": "matched",
            "sequence_mapping": "RepeaterAttackSpec(kind='none')",
            "compiled_mapping": "EveMove(kind='none')",
            "reason": None,
        }
    if attack_type == "swap_denial":
        return {
            "status": "matched",
            "sequence_mapping": "RepeaterAttackSpec(kind='swap_denial')",
            "compiled_mapping": "EveMove(kind='swap_block')",
            "reason": None,
        }
    if attack_type == "memory_degradation":
        return {
            "status": "approximate",
            "sequence_mapping": (
                "RepeaterAttackSpec(kind='memory_degradation', "
                "memory_fidelity_multiplier=0.5)"
            ),
            "compiled_mapping": (
                "EveMove(kind='memory_degradation', memory_depolarizing_probability=0.5, "
                "memory_dephasing_probability=0.5)"
            ),
            "reason": (
                "both target a repeater memory before swap, but the SeQUeNCe reference "
                "uses a fidelity multiplier while the compiled histogram uses explicit "
                "depolarizing/dephasing probabilities"
            ),
        }
    if attack_type in {"repeater_memory_measure_Z", "repeater_memory_measure_X"}:
        basis = "Z" if attack_type.endswith("_Z") else "X"
        return {
            "status": "matched",
            "sequence_mapping": f"RepeaterAttackSpec(kind='repeater_memory_measurement', basis='{basis}')",
            "compiled_mapping": f"EveMove(kind='{attack_type}')",
            "reason": None,
        }
    if attack_type == "edge_dos":
        return {
            "status": "unsupported_sequence_reference",
            "sequence_mapping": None,
            "compiled_mapping": "EveMove(kind='edge_added_loss')",
            "reason": (
                "the fixed-route SeQUeNCe repeater reference has no matching edge-loss "
                "intervention hook on elementary-link photons in this harness"
            ),
        }
    if attack_type == "edge_information_probe":
        return {
            "status": "unsupported_sequence_reference",
            "sequence_mapping": None,
            "compiled_mapping": "EveMove(kind='edge_measure_resend')",
            "reason": (
                "the fixed-route SeQUeNCe repeater reference has no matching optical "
                "intercept/measure-resend hook on elementary-link photons in this harness"
            ),
        }
    return {
        "status": "unsupported_sequence_reference",
        "sequence_mapping": None,
        "compiled_mapping": None,
        "reason": f"unsupported fixed-route repeater reference action {attack_type!r}",
    }


def _with_fixed_repeater_semantics(cell: dict[str, Any]) -> dict[str, Any]:
    out = dict(cell)
    out["sequence_reference_mode"] = "fixed_route_repeater_chain"
    out["action_semantics"] = fixed_route_repeater_action_semantics(out["action"])
    return out


def cells_for_route(
    route: dict[str, Any],
    *,
    comparison_mode: str = "mixed_runtime",
    strategic_actions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    path = route["path"]
    first_edge = "-".join(sorted((str(path[0]), str(path[1]))))
    internal = [str(node) for node in route.get("internal_nodes") or path[1:-1]]
    mid = internal[len(internal) // 2] if internal else None
    if comparison_mode == "fixed_route_strategic_matrix":
        if mid is None:
            return []
        if not strategic_actions:
            raise ValueError("fixed_route_strategic_matrix requires graph-level actions")
        internal_set = set(internal)
        return [
            _with_fixed_repeater_semantics({
                "family": "fixed_repeater_strategic",
                "label": str(action["id"]),
                "action": dict(action),
                "graph_level_fixed_target": True,
                "target_active_on_route": (
                    str(action.get("attack_type")) == "no_attack"
                    or str(action.get("target") or "") in internal_set
                ),
            })
            for action in strategic_actions
        ]
    if comparison_mode == "fixed_route_repeater_chain":
        if mid is None:
            return []
        fixed_cells = [
            {
                "family": "fixed_repeater",
                "label": "repeater_baseline",
                "action": {"id": "no_attack", "kind": "node", "target": "",
                           "attack_type": "no_attack", "cost": 0.0},
                "compiled_action_override": {"id": "no_attack", "kind": "none", "target": "",
                                             "attack_type": "no_attack", "cost": 0.0},
            },
            {
                "family": "fixed_repeater",
                "label": "edge_dos",
                "action": {"id": f"edge_dos:{first_edge}", "kind": "edge", "target": first_edge,
                           "attack_type": "edge_dos", "cost": 0.05},
            },
            {
                "family": "fixed_repeater",
                "label": "edge_information_probe",
                "action": {"id": f"edge_information_probe:{first_edge}", "kind": "edge",
                           "target": first_edge, "attack_type": "edge_information_probe",
                           "cost": 0.05},
            },
        ]
        for attack_type in ("swap_denial", "memory_degradation", "repeater_memory_measure_Z"):
            fixed_cells.append({
                "family": "fixed_repeater",
                "label": attack_type,
                "action": {"id": f"{attack_type}:{mid}", "kind": "node", "target": mid,
                           "attack_type": attack_type, "cost": 0.05},
            })
        return [_with_fixed_repeater_semantics(cell) for cell in fixed_cells]
    if comparison_mode != "mixed_runtime":
        raise ValueError(f"unknown comparison mode {comparison_mode!r}; expected {COMPARISON_MODES}")
    cells = [
        {
            "family": "edge",
            "label": "no_attack",
            "action": {"id": "no_attack", "kind": "none", "target": "",
                       "attack_type": "no_attack", "cost": 0.0},
        },
        {
            "family": "edge",
            "label": "edge_dos",
            "action": {"id": f"edge_dos:{first_edge}", "kind": "edge", "target": first_edge,
                       "attack_type": "edge_dos", "cost": 0.05},
        },
        {
            "family": "edge",
            "label": "edge_information_probe",
            "action": {"id": f"edge_information_probe:{first_edge}", "kind": "edge",
                       "target": first_edge, "attack_type": "edge_information_probe",
                       "cost": 0.05},
        },
    ]
    if mid is not None:
        # repeater-engine baseline: inactive spec via an off-route target, so
        # the SeQUeNCe side runs the same fixed repeater chain with no attack
        cells.append({
            "family": "node",
            "label": "repeater_baseline",
            "action": {"id": "memory_degradation:__off_route_baseline__", "kind": "node",
                       "target": "__off_route_baseline__",
                       "attack_type": "memory_degradation", "cost": 0.05},
            "compiled_action_override": {"id": "no_attack", "kind": "none", "target": "",
                                         "attack_type": "no_attack", "cost": 0.0},
        })
        for attack_type in ("swap_denial", "memory_degradation", "repeater_memory_measure_Z"):
            cells.append({
                "family": "node",
                "label": attack_type,
                "action": {"id": f"{attack_type}:{mid}", "kind": "node", "target": mid,
                           "attack_type": attack_type, "cost": 0.05},
            })
    return cells


# ── SeQUeNCe side ────────────────────────────────────────────────────────────

def sequence_run_configs(
    route: dict[str, Any],
    *,
    edge_periods: int,
    memory_pairs: int,
    repeater_window_ms: float,
) -> tuple[Any, Any, dict[str, Any]]:
    total_length_m = float(route.get("total_length_m", route.get("total_length", 0.0)))
    one_way_ps = total_length_m / FIBER_CLASSICAL_SPEED_M_PER_PS
    # RSVP setup needs several classical traversals before the reservation
    # start time; the corpus runtime game's fixed 1 us start asserts out at
    # km scale, so scale with route length here.
    start_time_ps = max(1_000_000, int(12.0 * one_way_ps))
    end_time_ps = start_time_ps + int(repeater_window_ms * 1e9)
    stop_time_ps = end_time_ps + int(0.25 * repeater_window_ms * 1e9)
    edge_cfg = sequence_runtime.E91RunConfig(
        num_periods=edge_periods,
        sifting=sequence_runtime.SiftingConfig(
            qber_threshold=sequence_runtime.DEFAULT_E91_QBER_THRESHOLD,
            min_sifted_samples=sequence_runtime.DEFAULT_E91_MIN_SIFTED_SAMPLES,
        ),
    )
    repeater_cfg = sequence_runtime.RepeaterE91RunConfig(
        memory_pairs=memory_pairs,
        sifting=sequence_runtime.SiftingConfig(
            qber_threshold=sequence_runtime.DEFAULT_E91_QBER_THRESHOLD,
            min_sifted_samples=sequence_runtime.DEFAULT_REPEATER_MIN_SIFTED_SAMPLES,
        ),
        request_fidelity=0.01,
        start_time_ps=start_time_ps,
        end_time_ps=end_time_ps,
        stop_time_ps=stop_time_ps,
        swapping_success_prob=1.0,
        swapping_degradation=1.0,
    )
    timing = {
        "start_time_ps": start_time_ps,
        "end_time_ps": end_time_ps,
        "stop_time_ps": stop_time_ps,
        "scaling_rule": "start = max(1 us, 12 x one-way classical traversal)",
    }
    return edge_cfg, repeater_cfg, timing


def run_sequence_cell(
    *,
    ir: TopologyIR,
    route: dict[str, Any],
    cell: dict[str, Any],
    models: tuple[Any, Any, Any, Any],
    trials: int,
    seed: int,
    edge_periods: int,
    memory_pairs: int,
    repeater_window_ms: float,
    security_observable: str = "endpoint_state_expectation",
) -> dict[str, Any]:
    source_model, detector_model, fiber_model, memory_model = models
    edge_cfg, repeater_cfg, timing = sequence_run_configs(
        route,
        edge_periods=edge_periods,
        memory_pairs=memory_pairs,
        repeater_window_ms=repeater_window_ms,
    )
    reference_mode = str(cell.get("sequence_reference_mode", "runtime_dispatch_by_action_kind"))
    budget = edge_periods if cell["family"] == "edge" else memory_pairs
    action_semantics = cell.get("action_semantics")
    if reference_mode == "fixed_route_repeater_chain":
        action_semantics = action_semantics or fixed_route_repeater_action_semantics(cell["action"])
        if action_semantics.get("status") == "unsupported_sequence_reference":
            reason = str(action_semantics.get("reason") or "unsupported fixed-route repeater action")
            return {
                "backend": sequence_runtime.E91_RUNTIME_BACKEND,
                "engine": "fixed_route_repeater_chain_reference",
                "reference_mode": reference_mode,
                "action_semantics": action_semantics,
                "trials_requested": trials,
                "trials_completed": 0,
                "pair_budget_per_trial": budget,
                "outcome_counts": {},
                "acceptance_rate": None,
                "eve_win_rate": None,
                "exposure_rate": None,
                "delivery_rate": None,
                "delivered_fraction_mean": None,
                "delivered_fraction_standard_error": None,
                "delivered_fraction_confidence_interval_95": None,
                "delivered_fraction_confidence_interval_method": None,
                "delivered_fraction_sample_count": 0,
                "qber_mean": None,
                "qber_trials": 0,
                "expected_qber_from_histogram": None,
                "sampled_qber_mean": None,
                "sampled_qber_trials": 0,
                "sampled_qber_observable": False,
                "qber_key_mass_expected": None,
                "qber_estimation_power": "finite_sample_zero_key_trials",
                "fidelity_mean": None,
                "information_bits_mean": None,
                "timing_windows": timing,
                "errors": [],
                "unsupported": True,
                "unsupported_reason": reason,
                "wall_seconds": 0.0,
            }
    outcomes: Counter[str] = Counter()
    qbers: list[float] = []
    fidelities: list[float] = []
    info_bits: list[float] = []
    delivered_fractions: list[float] = []
    bell_qualified_delivered_fractions: list[float] = []
    state_witness_weight = 0
    state_witness_chsh_weighted_sum = 0.0
    state_witness_qber_weighted_sum = 0.0
    delivered_any = 0
    errors: list[str] = []
    started = time.perf_counter()
    for trial in range(trials):
        try:
            transcript, info, _active, runtime_metrics = sequence_runtime._run_selected_action(
                ir=ir,
                route=route,
                action=cell["action"],
                source_model=source_model,
                detector_model=detector_model,
                fiber_model=fiber_model,
                memory_model=memory_model,
                edge_run_config=edge_cfg,
                repeater_run_config=repeater_cfg,
                trial_id=f"cmp_{route['route_id']}_{cell['label']}_{trial}",
                seed=seed + 1000 * trial,
                eve_name=f"eve_cmp_{trial}",
                attack_cost=0.05,
            )
        except Exception as exc:  # recorded as a finding, not swallowed silently
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        outcome = sequence_runtime._public_outcome(transcript, info, cell["action"], GAME_MODE)
        outcomes[outcome] += 1
        delivered = int(transcript.generation_successes)
        delivered_fractions.append(delivered / max(1, budget))
        if delivered > 0:
            delivered_any += 1
        if transcript.qber_estimate is not None:
            qbers.append(float(transcript.qber_estimate))
        if runtime_metrics.get("fidelity") is not None:
            fidelities.append(float(runtime_metrics["fidelity"]))
        info_bits.append(float(info.mutual_information_bits))
        state_witness = transcript.extra.get("endpoint_state_witness")
        if state_witness is not None:
            pair_count = int(state_witness.get("pair_count", 0))
            chsh_s = state_witness.get("fixed_setting_chsh_s")
            state_qber = state_witness.get("key_qber")
            if pair_count > 0 and chsh_s is not None and state_qber is not None:
                state_witness_weight += pair_count
                state_witness_chsh_weighted_sum += pair_count * float(chsh_s)
                state_witness_qber_weighted_sum += pair_count * float(state_qber)
            bell_qualified_delivered_fractions.append(
                delivered / max(1, budget)
                if bool(state_witness.get("bell_qualified")) else 0.0
            )
    completed = sum(outcomes.values())
    elapsed = time.perf_counter() - started
    sampled_qber_mean = float(np.mean(qbers)) if qbers else None
    qber_power = classify_qber_estimation_power(
        expected_key_mass=None,
        sampled_qber_trials=len(qbers),
        adequate_sampled_trials=MIN_COMPARISON_QBER_OBSERVABLE_TRIALS,
    )
    delivery_statistics = _delivery_fraction_statistics(delivered_fractions)
    bell_delivery_statistics = _prefixed_delivery_fraction_statistics(
        bell_qualified_delivered_fractions,
        prefix="bell_qualified_",
    )
    return {
        "backend": sequence_runtime.E91_RUNTIME_BACKEND,
        "engine": (
            "fixed_route_repeater_chain_reference"
            if reference_mode == "fixed_route_repeater_chain" else
            ("edge_e91_trial" if cell["family"] == "edge" else "repeater_e91_trial")
        ),
        "reference_mode": reference_mode,
        "action_semantics": action_semantics,
        "trials_requested": trials,
        "trials_completed": completed,
        "pair_budget_per_trial": budget,
        "outcome_counts": dict(sorted(outcomes.items())),
        "acceptance_rate": outcomes.get("accepted", 0) / completed if completed else None,
        "eve_win_rate": (
            sum(count for outcome, count in outcomes.items() if outcome != "accepted")
            / completed if completed else None
        ),
        "exposure_rate": outcomes.get("information_exposure", 0) / completed if completed else None,
        "delivery_rate": delivered_any / completed if completed else None,
        **delivery_statistics,
        **bell_delivery_statistics,
        "qber_mean": sampled_qber_mean,
        "qber_trials": len(qbers),
        "expected_qber_from_histogram": None,
        "sampled_qber_mean": sampled_qber_mean,
        "sampled_qber_trials": len(qbers),
        "sampled_qber_observable": bool(qbers),
        "qber_key_mass_expected": None,
        "qber_estimation_power": qber_power,
        "fidelity_mean": float(np.mean(fidelities)) if fidelities else None,
        "information_bits_mean": float(np.mean(info_bits)) if info_bits else None,
        "endpoint_state_expected_chsh_s": (
            state_witness_chsh_weighted_sum / state_witness_weight
            if state_witness_weight else None
        ),
        "endpoint_state_expected_qber": (
            state_witness_qber_weighted_sum / state_witness_weight
            if state_witness_weight else None
        ),
        "endpoint_state_witness_pair_count": state_witness_weight,
        "security_observable": security_observable,
        "security_observable_claim_boundary": (
            "pre-measurement state expectation, not an executed finite-shot Bell test"
            if security_observable == "endpoint_state_expectation" else
            "finite Z/X transcript QBER; no event CHSH value"
        ),
        "timing_windows": timing if cell["family"] != "edge" else None,
        "errors": errors,
        "wall_seconds": elapsed,
    }


# ── Compiled side ────────────────────────────────────────────────────────────

def _full_budget_route_ready_diagnostic(
    backend: CompiledHistogramBackend,
    hop_count: int,
    fiber_lengths_m: list[float],
    prepared: Any,
) -> dict[str, Any] | None:
    """Analytic route-ready probability before history/signature cap reduction.

    Diagnostic only: the compiled histogram's delivered mass uses the
    recorded (possibly cap-reduced) attempt cutoffs; at calibrated per-attempt
    success (~1e-3) and h >= 3 the signature caps can truncate the wall-clock
    budget by orders of magnitude. This closed-form Pi(1 - (1-p_i)^{K_i})
    with the selected per-link attempt budgets quantifies that gap. It says
    nothing about the delivered state mixture, only about delivery mass. It is
    not an infinite-time route-ready probability.
    """
    if str(backend.config.waiting_protocol) == "rolling_alive_pair":
        rolling_model = (prepared.delivery_model or {}).get("rolling_live_pair_model") or {}
        return {
            "uncapped_route_ready": rolling_model.get("route_availability"),
            "full_budget_route_ready": rolling_model.get("route_availability"),
            "capped_route_ready": rolling_model.get("route_availability"),
            "cap_bias_factor": None,
            "cap_bias_classification": "not_applicable",
            "diagnostic_semantics": (
                "not applicable: rolling_alive_pair uses configured live-pair "
                "availability and hidden selected-age distributions, not "
                "route-from-scratch waiting history/signature caps"
            ),
        }
    if hop_count < 2:
        return None
    from sequence_game.corpus.compiled_histogram_backend import load_histogram_generator

    generator = load_histogram_generator()
    front = backend._front
    context = backend._context
    route = generator.make_route(hop_count, list(fiber_lengths_m))
    attempt_policy = generator.attempt_period_policy(route, front, context)
    budgets, budget_meta = generator.waiting_attempt_budgets(
        context,
        {
            edge_id: float(attempt_policy["edge_attempt_periods"][edge_id]["attempt_period_ps"])
            for edge_id in route["edge_ids"]
        },
    )
    edge_success = prepared.delivery_model.get("edge_success_probabilities") or {}
    if not edge_success:
        return None
    full_budget_ready = 1.0
    for edge_id in route["edge_ids"]:
        p = float(edge_success.get(edge_id, 0.0))
        k = int(budgets.get(edge_id, 0))
        full_budget_ready *= 1.0 - (1.0 - p) ** k if k > 0 else 0.0
    capped_ready = (prepared.delivery_model or {}).get(
        "route_ready_probability_within_waiting_budget"
    )
    waiting_model = (prepared.delivery_model or {}).get("waiting_model") or {}
    waiting_distribution = waiting_model.get("waiting_distribution") or {}
    factor = (
        full_budget_ready / capped_ready
        if capped_ready and capped_ready > 0.0 else
        (math.inf if full_budget_ready > 0.0 else None)
    )
    return {
        "uncapped_route_ready": full_budget_ready,
        "full_budget_route_ready": full_budget_ready,
        "capped_route_ready": capped_ready,
        "attempt_budgets_full": {k: int(v) for k, v in budgets.items()},
        "attempt_budgets_used_by_context": {k: int(v) for k, v in budgets.items()},
        "waiting_budget_meta": _json_safe(budget_meta),
        "cap_reduction_metadata": {
            key: _json_safe(waiting_distribution.get(key))
            for key in (
                "requested_attempt_budgets",
                "attempt_cutoffs_after_caps",
                "attempt_cutoffs_reduced_below_budget",
                "attempt_budget_per_link",
                "attempt_budgets",
                "left_attempt_cutoff",
                "right_attempt_cutoff",
                "history_pair_cap",
                "wait_signature_cap",
                "signature_cap_shrink_rounds",
                "history_count",
                "wait_signature_count",
                "retained_history_mass",
                "omitted_tail_mass",
                "tail_exceeds_tolerance",
            )
            if key in waiting_distribution
        },
        "cap_bias_factor": factor,
        "cap_bias_classification": classify_cap_bias_factor(factor),
        "diagnostic_semantics": (
            "full_budget_route_ready uses the selected wall-clock/attempt budget before h>=3 "
            "history/signature cap shrinkage; capped_route_ready is the production probability "
            "used by the compiled histogram; the ratio quantifies compute-cap delivery-mass bias"
        ),
    }


def run_compiled_cell(
    *,
    backend: CompiledHistogramBackend,
    ir_dict: dict[str, Any],
    route: dict[str, Any],
    cell: dict[str, Any],
    trials: int,
    seed: int,
    edge_periods: int,
    memory_pairs: int,
) -> dict[str, Any]:
    action = cell.get("compiled_action_override", cell["action"])
    budget = edge_periods if cell["family"] == "edge" else memory_pairs
    min_sifted_gate = (
        sequence_runtime.DEFAULT_E91_MIN_SIFTED_SAMPLES
        if cell["family"] == "edge"
        else sequence_runtime.DEFAULT_REPEATER_MIN_SIFTED_SAMPLES
    )
    outcomes: Counter[str] = Counter()
    aligned_outcomes: Counter[str] = Counter()
    qbers: list[float] = []
    delivered_fractions: list[float] = []
    bell_qualified_delivered_fractions: list[float] = []
    delivered_any = 0
    errors: list[str] = []
    expected: dict[str, Any] | None = None
    theta_hash = None
    started = time.perf_counter()
    for trial in range(trials):
        try:
            result = backend.step(
                ir_dict=ir_dict,
                route=route,
                action=action,
                requested_pairs=budget,
                seed=seed + 1000 * trial,
                step_index=trial,
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        theta_hash = result.theta_hash
        outcomes[result.public_outcome] += 1
        aligned = result.public_outcome
        if aligned == "accepted" and result.sifted_key_count < min_sifted_gate:
            aligned = "delivery_failure"
        aligned_outcomes[aligned] += 1
        delivered_fractions.append(result.delivered_pair_count / max(1, budget))
        if result.delivered_pair_count > 0:
            delivered_any += 1
        if result.qber_key is not None:
            qbers.append(float(result.qber_key))
        if expected is None:
            move = eve_move_from_runtime_action(action, route)[0]
            prepared = backend.evaluate(
                result.hop_count,
                move,
                result.fiber_lengths_m,
            )
            expected_qber_from_histogram = prepared.expected_metrics.get(
                "expected_qber_from_histogram",
                prepared.expected_metrics.get("expected_qber"),
            )
            expected = {
                "delivery_probability": prepared.expected_metrics.get("delivery_probability"),
                "expected_qber": prepared.expected_metrics.get("expected_qber"),
                "expected_qber_from_histogram": expected_qber_from_histogram,
                "qber_key_mass_expected": prepared.expected_metrics.get("qber_key_probability_mass"),
                "qber_error_mass_expected": prepared.expected_metrics.get("qber_error_probability_mass"),
                "qber_numerator_probability": prepared.expected_metrics.get("qber_numerator_probability"),
                "qber_denominator_probability": prepared.expected_metrics.get("qber_denominator_probability"),
                "qber_histogram_observable": prepared.expected_metrics.get("qber_histogram_observable"),
                "qber_diagnostics": prepared.expected_metrics.get("qber_diagnostics"),
                "expected_chsh_S": prepared.expected_metrics.get("expected_chsh_S"),
                "eve_guess_probability": prepared.expected_metrics.get("eve_guess_probability"),
                "uncapped_route_ready_diagnostic": _full_budget_route_ready_diagnostic(
                    backend, result.hop_count, result.fiber_lengths_m, prepared
                ),
            }
            expected["waiting_cap_diagnostic"] = expected["uncapped_route_ready_diagnostic"]
        expected_chsh = (expected or {}).get("expected_chsh_S")
        expected_state_qber = (expected or {}).get("expected_qber_from_histogram")
        bell_qualified = bool(
            expected_chsh is not None
            and abs(float(expected_chsh)) > 2.0
            and expected_state_qber is not None
            and float(expected_state_qber) <= float(backend.config.qber_abort_threshold)
        )
        bell_qualified_delivered_fractions.append(
            result.delivered_pair_count / max(1, budget) if bell_qualified else 0.0
        )
    completed = sum(outcomes.values())
    elapsed = time.perf_counter() - started
    sampled_qber_mean = float(np.mean(qbers)) if qbers else None
    qber_key_mass_expected = None
    qber_error_mass_expected = None
    expected_qber_from_histogram = None
    if expected is not None:
        expected_qber_from_histogram = expected.get(
            "expected_qber_from_histogram",
            expected.get("expected_qber"),
        )
        qber_key_mass_expected = expected.get("qber_key_mass_expected")
        qber_error_mass_expected = expected.get("qber_error_mass_expected")
    qber_power = classify_qber_estimation_power(
        expected_key_mass=qber_key_mass_expected,
        sampled_qber_trials=len(qbers),
        adequate_sampled_trials=MIN_COMPARISON_QBER_OBSERVABLE_TRIALS,
    )
    delivery_statistics = _delivery_fraction_statistics(delivered_fractions)
    bell_delivery_statistics = _prefixed_delivery_fraction_statistics(
        bell_qualified_delivered_fractions,
        prefix="bell_qualified_",
    )
    return {
        "backend": COMPILED_HISTOGRAM_BACKEND,
        "engine": "compiled_histogram_transcript",
        "trials_requested": trials,
        "trials_completed": completed,
        "pair_budget_per_trial": budget,
        "outcome_counts": dict(sorted(outcomes.items())),
        "aligned_outcome_counts": dict(sorted(aligned_outcomes.items())),
        "acceptance_rate": (
            aligned_outcomes.get("accepted", 0) / completed if completed else None
        ),
        "eve_win_rate": (
            sum(count for outcome, count in aligned_outcomes.items() if outcome != "accepted")
            / completed if completed else None
        ),
        "exposure_rate": None,
        "delivery_rate": delivered_any / completed if completed else None,
        **delivery_statistics,
        **bell_delivery_statistics,
        "qber_mean": sampled_qber_mean,
        "qber_trials": len(qbers),
        "expected_qber_from_histogram": expected_qber_from_histogram,
        "sampled_qber_mean": sampled_qber_mean,
        "sampled_qber_trials": len(qbers),
        "sampled_qber_observable": bool(qbers),
        "qber_key_mass_expected": qber_key_mass_expected,
        "qber_error_mass_expected": qber_error_mass_expected,
        "qber_estimation_power": qber_power,
        "expected_infinite_shot": expected,
        "endpoint_state_expected_chsh_s": (
            (expected or {}).get("expected_chsh_S")
        ),
        "endpoint_state_expected_qber": expected_qber_from_histogram,
        "endpoint_state_witness_pair_count": None,
        "security_observable": "compiled_histogram_state_expectation",
        "theta_hash": theta_hash,
        "errors": errors,
        "wall_seconds": elapsed,
    }


def comparison_policy_payload(
    args: argparse.Namespace,
    manager: CompiledBackendManager,
    *,
    per_pair_window_ps: float,
) -> dict[str, Any]:
    backend_diag = manager.diagnostics()
    backend_config = backend_diag.get("backend_config", {})
    if manager.hop_calibrated:
        waiting_budget_mode = "rolling_alive_pair_configured_availability"
        waiting_budget_wall_clock_ps = None
        per_pair_window_policy_ps = None
        per_pair_window_policy_ms = None
        sequence_reference_per_pair_window_ps = float(per_pair_window_ps)
        sequence_reference_per_pair_window_ms = float(per_pair_window_ps) / 1e9
        budget_scope = "per_histogram_draw_route_request_configured_live_pair_pool"
        harmonization_rule = (
            "No SeQUeNCe wall-clock waiting budget is applied to the compiled rolling "
            "histogram; live-pair availability and hidden age distributions are derived "
            "from the standardized 10 km hop calibration."
        )
        waiting_max_attempts = None
        waiting_max_history_pairs = None
        waiting_max_wait_signatures = None
        waiting_tail_tolerance = None
        memory_cutoff_ps = None
        timeout_semantics = (
            "rolling_alive_pair uses configured per-edge live-pair availability "
            "and selected age distributions derived from the 10 km hop calibration; "
            "route-from-scratch timeout/signature cap bias is not applicable"
        )
        h2_model = "rolling live-pair reservation; no two-link repeat-until-success waiting"
        h_ge_3_model = "rolling live-pair reservation; no h>=3 waiting-signature enumeration"
    else:
        config = manager._static_backend.config  # type: ignore[union-attr]
        waiting_budget_mode = str(config.waiting_budget_mode)
        waiting_budget_wall_clock_ps = (
            float(config.waiting_budget_wall_clock_ps)
            if config.waiting_budget_wall_clock_ps is not None else None
        )
        per_pair_window_policy_ps = float(per_pair_window_ps)
        per_pair_window_policy_ms = float(per_pair_window_ps) / 1e9
        sequence_reference_per_pair_window_ps = None
        sequence_reference_per_pair_window_ms = None
        budget_scope = "per_histogram_draw_per_requested_pair"
        harmonization_rule = (
            "per_pair_window_ps = repeater_window_ms * 1e9 / memory_pairs; "
            "compiled samples requested_pairs draws from this per-pair distribution"
        )
        waiting_max_attempts = int(config.waiting_max_attempts)
        waiting_max_history_pairs = int(config.waiting_max_history_pairs)
        waiting_max_wait_signatures = int(config.waiting_max_wait_signatures)
        waiting_tail_tolerance = float(config.waiting_tail_tolerance)
        memory_cutoff_ps = (
            float(config.memory_cutoff_ps) if config.memory_cutoff_ps is not None else None
        )
        timeout_semantics = (
            "finite wall-clock route-generation timeout is physical for this comparison policy; "
            "additional h>=3 history/signature cutoff reduction is a compute-cap artifact and is "
            "reported per cell as cap_bias_factor/classification"
        )
        h2_model = "analytic two-link finite-budget repeat-until-success"
        h_ge_3_model = "grouped swap-asap enumeration within attempt/history/signature caps"
    return {
        "security_observable_policy": {
            "selected": str(args.security_observable),
            "event_repeater_observable": (
                "exact pre-measurement two-memory density-matrix expectation at "
                "the repository Ekert settings"
                if args.security_observable == "endpoint_state_expectation" else
                "finite Z/X transcript QBER without an event CHSH statistic"
            ),
            "compiled_repeater_observable": (
                "fixed-setting expected CHSH and histogram-derived expected key QBER"
            ),
            "bell_qualified_delivery_definition": (
                "delivered pair fraction when abs(S)>2 and expected key QBER<=0.15; "
                "zero otherwise"
            ),
            "finite_transcript_distinction": (
                "state expectation removes setting-count shot noise and is not an "
                "executed Bell-test transcript"
            ),
        },
        "sequence_window_policy": {
            "comparison_mode": str(args.comparison_mode),
            "edge_family": {
                "engine": "direct_path_E91_trial",
                "budget_parameter": "edge_periods",
                "edge_periods": int(args.edge_periods),
                "budget_scope": "per_trial_direct_path_period_count",
                "timeout_or_no_delivery_semantics": (
                    "direct-path trial attempts/sifting periods; no repeater waiting-window "
                    "semantics are modeled in this family"
                ),
            },
            "node_family": {
                "engine": "event_driven_repeater_E91_trial",
                "memory_pairs": int(args.memory_pairs),
                "repeater_window_ms": float(args.repeater_window_ms),
                "budget_scope": "per_trial_window_shared_by_all_requested_memory_pairs",
                "start_stop_policy": "route-dependent start=max(1us,12*one_way_classical_traversal); end=start+repeater_window_ms; stop=end+0.25*window",
                "timeout_or_no_delivery_semantics": (
                    "SeQUeNCe event-scheduler deadline for one repeater trial; route setup and "
                    "generation events are stopped by the configured end/stop times"
                ),
            },
        },
        "compiled_window_policy": {
            "engine": "compiled_length_parametric_histogram",
            "waiting_protocol": (
                "rolling_alive_pair" if manager.hop_calibrated else "single_buffer_swap_asap"
            ),
            "waiting_budget_mode": waiting_budget_mode,
            "waiting_budget_wall_clock_ps": waiting_budget_wall_clock_ps,
            "per_pair_window_ps": per_pair_window_policy_ps,
            "per_pair_window_ms": per_pair_window_policy_ms,
            "sequence_reference_per_pair_window_ps": sequence_reference_per_pair_window_ps,
            "sequence_reference_per_pair_window_ms": sequence_reference_per_pair_window_ms,
            "budget_scope": budget_scope,
            "harmonization_rule": harmonization_rule,
            "hop_calibration": backend_config if manager.hop_calibrated else None,
            "waiting_max_attempts": waiting_max_attempts,
            "waiting_max_history_pairs": waiting_max_history_pairs,
            "waiting_max_wait_signatures": waiting_max_wait_signatures,
            "waiting_tail_tolerance": waiting_tail_tolerance,
            "memory_cutoff_ps": memory_cutoff_ps,
            "timeout_or_no_delivery_semantics": timeout_semantics,
            "h2_waiting_model": h2_model,
            "h_ge_3_waiting_model": h_ge_3_model,
            "uncapped_diagnostic_semantics": (
                "full_budget_route_ready is before history/signature cap reduction for "
                "single_buffer_swap_asap; not applicable to rolling_alive_pair"
            ),
        },
        "cap_bias_policy": {
            "cap_bias_factor_definition": (
                "full_budget_route_ready / capped_route_ready when both exist"
            ),
            "classes": {
                "no_cap_bias": "<= 1.05",
                "mild_cap_bias": "<= 2",
                "moderate_cap_bias": "<= 10",
                "severe_cap_bias": "<= 1e3",
                "catastrophic_cap_bias": "> 1e3 or infinite",
            },
            "excluded_from_comparable_directional_agreement": sorted(SEVERE_CAP_BIAS_CLASSES),
            "low_delivery_power_threshold": LOW_DELIVERY_POWER_THRESHOLD,
        },
        "comparison_family_policy": {
            "comparison_mode": str(args.comparison_mode),
            "fixed_route_repeater_chain": {
                "status": "small_h_reference_mode",
                "sequence_stack": "sequence_fixed_route_repeater_chain_reference",
                "compiled_stack": COMPILED_REPEATER_STACK,
                "supported_actions": {
                    "no_attack": "matched",
                    "swap_denial": "matched",
                    "memory_degradation": "approximate",
                    "repeater_memory_measure_Z": "matched",
                    "repeater_memory_measure_X": "matched",
                    "edge_dos": "unsupported_sequence_reference",
                    "edge_information_probe": "unsupported_sequence_reference",
                },
                "timing_policy": (
                    "length-aware start/end/stop windows from sequence_run_configs; "
                    "route path and per-edge fiber lengths are passed directly to "
                    "FixedRepeaterPath"
                ),
            },
            "fixed_route_strategic_matrix": {
                "status": "small_h_fixed_target_strategic_reference",
                "sequence_stack": "sequence_fixed_route_repeater_chain_reference",
                "compiled_stack": COMPILED_REPEATER_STACK,
                "target_policy": (
                    "graph-level node targets selected from route-incidence "
                    "topology and held fixed across every route"
                ),
                "supported_actions": {
                    "no_attack": "matched",
                    "swap_denial": "matched",
                    "repeater_memory_measure_Z": "matched",
                },
                "information_boundary": (
                    "target selection uses route topology only; no backend "
                    "outcome or physical-quality prediction"
                ),
            },
            "direct_e91_edge_family": {
                "status": "not_currently_absolute_comparable",
                "sequence_stack": SEQUENCE_DIRECT_EDGE_STACK,
                "compiled_stack_required_for_absolute_comparison": COMPILED_DIRECT_STACK_MISSING,
                "compiled_stack_currently_used": COMPILED_REPEATER_STACK,
                "artifact_treatment": (
                    "edge-family cells are emitted as qualitative_cross_family directional "
                    "stress checks unless a matching compiled direct-path backend is added"
                ),
            },
            "repeater_chain_family": {
                "status": "absolute_rate_comparable_when_hop_count_ge_2",
                "sequence_stack": SEQUENCE_REPEATER_STACK,
                "compiled_stack": COMPILED_REPEATER_STACK,
                "required_semantics": (
                    "same selected route path and fiber-length vector; SeQUeNCe fixed "
                    "repeater chain with entanglement swapping; compiled Fock/SNSPD "
                    "repeater-chain histogram"
                ),
            },
            "qualitative_cross_family": {
                "status": "directional_only",
                "absolute_rate_validation": False,
                "summary_field": "qualitative_cross_family_directional_agreement",
            },
        },
    }


def cap_bias_aggregate_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    classes = Counter(_cell_cap_bias_classification(cell) for cell in cells)
    factors = []
    for cell in cells:
        diagnostic = cell.get("waiting_cap_diagnostic") or {}
        factor = diagnostic.get("cap_bias_factor")
        if factor is not None and math.isfinite(float(factor)):
            factors.append(float(factor))
    return {
        "class_counts": dict(sorted(classes.items())),
        "max_finite_cap_bias_factor": max(factors) if factors else None,
        "severe_or_catastrophic_cell_count": sum(
            count for label, count in classes.items() if _is_severe_cap_bias(label)
        ),
        "cell_count": len(cells),
    }


def bell_qualified_observable_parity(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare matched state-expectation CHSH, QBER, and qualified delivery.

    This is intentionally separate from finite-transcript public outcomes.  A
    row is an absolute-validation row only when the existing stack/action/cap
    contract also declares it comparable.
    """

    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for cell in cells:
        sequence = cell.get("sequence", {})
        compiled = cell.get("compiled", {})
        sequence_s = sequence.get("endpoint_state_expected_chsh_s")
        compiled_s = compiled.get("endpoint_state_expected_chsh_s")
        sequence_qber = sequence.get("endpoint_state_expected_qber")
        compiled_qber = compiled.get("endpoint_state_expected_qber")
        sequence_delivery = sequence.get("bell_qualified_delivered_fraction_mean")
        compiled_delivery = compiled.get("bell_qualified_delivered_fraction_mean")
        missing = [
            name for name, value in (
                ("sequence_chsh", sequence_s),
                ("compiled_chsh", compiled_s),
                ("sequence_qber", sequence_qber),
                ("compiled_qber", compiled_qber),
                ("sequence_bell_qualified_delivery", sequence_delivery),
                ("compiled_bell_qualified_delivery", compiled_delivery),
            ) if value is None
        ]
        if missing:
            exclusions.append({
                "cell_key": _cell_key(cell),
                "reason": "missing_matched_state_observable",
                "missing": missing,
            })
            continue
        action_status = str((cell.get("action_semantics") or {}).get("status", "matched"))
        absolute = bool(
            cell.get("absolute_rate_comparable")
            and action_status == "matched"
            and not _is_severe_cap_bias(_cell_cap_bias_classification(cell))
            and not sequence.get("errors")
            and not compiled.get("errors")
        )
        qber_threshold = float(sequence_runtime.DEFAULT_E91_QBER_THRESHOLD)
        row = {
            "cell_key": _cell_key(cell),
            "graph_id": cell.get("graph_id"),
            "route_id": cell.get("route_id"),
            "label": cell.get("label"),
            "absolute_validation_cell": absolute,
            "sequence_expected_chsh_s": float(sequence_s),
            "compiled_expected_chsh_s": float(compiled_s),
            "absolute_chsh_difference": abs(float(sequence_s) - float(compiled_s)),
            "sequence_expected_qber": float(sequence_qber),
            "compiled_expected_qber": float(compiled_qber),
            "absolute_qber_difference": abs(float(sequence_qber) - float(compiled_qber)),
            "sequence_bell_qualified_delivered_fraction": float(sequence_delivery),
            "compiled_bell_qualified_delivered_fraction": float(compiled_delivery),
            "absolute_bell_qualified_delivery_difference": abs(
                float(sequence_delivery) - float(compiled_delivery)
            ),
            "bell_violation_classification_agrees": (
                (abs(float(sequence_s)) > 2.0) == (abs(float(compiled_s)) > 2.0)
            ),
            "qber_gate_classification_agrees": (
                (float(sequence_qber) <= qber_threshold)
                == (float(compiled_qber) <= qber_threshold)
            ),
        }
        rows.append(row)

    absolute_rows = [row for row in rows if row["absolute_validation_cell"]]

    def mean(field: str) -> float | None:
        return (
            float(np.mean([row[field] for row in absolute_rows]))
            if absolute_rows else None
        )

    return {
        "observable": "pre_measurement_endpoint_state_expectation",
        "bell_criterion": "abs(fixed_setting_chsh_s) > 2",
        "qber_criterion": (
            f"expected key QBER <= {sequence_runtime.DEFAULT_E91_QBER_THRESHOLD}"
        ),
        "eligible_cell_count": len(rows),
        "absolute_validation_cell_count": len(absolute_rows),
        "absolute_validation_bell_classification_agreement_rate": (
            float(np.mean([
                row["bell_violation_classification_agrees"] for row in absolute_rows
            ])) if absolute_rows else None
        ),
        "absolute_validation_qber_gate_agreement_rate": (
            float(np.mean([
                row["qber_gate_classification_agrees"] for row in absolute_rows
            ])) if absolute_rows else None
        ),
        "mean_absolute_chsh_difference": mean("absolute_chsh_difference"),
        "mean_absolute_qber_difference": mean("absolute_qber_difference"),
        "mean_absolute_bell_qualified_delivery_difference": mean(
            "absolute_bell_qualified_delivery_difference"
        ),
        "rows": rows,
        "excluded_cells": exclusions,
        "claim_boundary": (
            "same fixed-setting state functional, not finite-shot output parity; "
            "absolute delivery parity additionally depends on controller/timing equivalence"
        ),
    }


# ── Agreement analysis ───────────────────────────────────────────────────────

def _effect_sign(value: float | None, baseline: float | None, tolerance: float) -> str | None:
    if value is None or baseline is None:
        return None
    delta = value - baseline
    if abs(delta) <= tolerance:
        return "flat"
    return "up" if delta > 0 else "down"


def _delivery_fraction_statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "delivered_fraction_mean": None,
            "delivered_fraction_standard_error": None,
            "delivered_fraction_confidence_interval_95": None,
            "delivered_fraction_confidence_interval_method": None,
            "delivered_fraction_sample_count": 0,
        }
    samples = np.asarray(values, dtype=float)
    mean = float(np.mean(samples))
    standard_error = (
        float(np.std(samples, ddof=1) / math.sqrt(samples.size))
        if samples.size > 1 else None
    )
    critical = (
        float(student_t.ppf(0.975, samples.size - 1))
        if standard_error is not None else None
    )
    interval = (
        [
            max(0.0, mean - critical * standard_error),
            min(1.0, mean + critical * standard_error),
        ]
        if standard_error is not None else None
    )
    return {
        "delivered_fraction_mean": mean,
        "delivered_fraction_standard_error": standard_error,
        "delivered_fraction_confidence_interval_95": interval,
        "delivered_fraction_confidence_interval_method": "student_t",
        "delivered_fraction_sample_count": int(samples.size),
    }


def _prefixed_delivery_fraction_statistics(
        values: list[float],
        *,
        prefix: str,
) -> dict[str, Any]:
    return {
        f"{prefix}{key}": value
        for key, value in _delivery_fraction_statistics(values).items()
    }


def _confidence_order(
    left_mean: float,
    right_mean: float,
    left_standard_error: float | None,
    right_standard_error: float | None,
    left_sample_count: int,
    right_sample_count: int,
    *,
    tolerance: float,
) -> dict[str, Any]:
    """Classify a mean difference only when its 95% interval is decisive."""

    delta = float(left_mean - right_mean)
    if (
        left_standard_error is None
        or right_standard_error is None
        or left_sample_count < 2
        or right_sample_count < 2
    ):
        return {
            "decision": "undecided",
            "delta": delta,
            "delta_standard_error": None,
            "confidence_interval_95": None,
            "reason": "at least two independent trials per action are required",
        }
    delta_standard_error = math.sqrt(
        float(left_standard_error) ** 2 + float(right_standard_error) ** 2
    )
    denominator = (
        float(left_standard_error) ** 4 / (left_sample_count - 1)
        + float(right_standard_error) ** 4 / (right_sample_count - 1)
    )
    degrees_of_freedom = (
        delta_standard_error ** 4 / denominator
        if denominator > 0.0 else math.inf
    )
    critical = (
        1.96
        if not math.isfinite(degrees_of_freedom)
        else float(student_t.ppf(0.975, degrees_of_freedom))
    )
    low = delta - critical * delta_standard_error
    high = delta + critical * delta_standard_error
    if low > tolerance:
        decision = "up"
    elif high < -tolerance:
        decision = "down"
    elif low >= -tolerance and high <= tolerance:
        decision = "flat"
    else:
        decision = "undecided"
    return {
        "decision": decision,
        "delta": delta,
        "delta_standard_error": delta_standard_error,
        "welch_degrees_of_freedom": degrees_of_freedom,
        "critical_value": critical,
        "confidence_interval_95": [low, high],
        "reason": (
            "95% Welch t interval lies wholly above/below the practical "
            "tolerance or wholly inside its equivalence band"
            if decision != "undecided"
            else "95% interval crosses a practical-decision boundary"
        ),
    }


def _rank_correlation(
    left: dict[str, float],
    right: dict[str, float],
    tolerance: float,
) -> float | None:
    labels = sorted(set(left) & set(right))
    if len(labels) < 2:
        return None

    def ranks(values: dict[str, float]) -> np.ndarray:
        ordered = sorted(labels, key=lambda label: (values[label], label))
        result: dict[str, float] = {}
        start = 0
        while start < len(ordered):
            stop = start + 1
            anchor = values[ordered[start]]
            while stop < len(ordered) and abs(values[ordered[stop]] - anchor) <= tolerance:
                stop += 1
            average_rank = 0.5 * (start + stop - 1)
            for label in ordered[start:stop]:
                result[label] = average_rank
            start = stop
        return np.asarray([result[label] for label in labels], dtype=float)

    left_ranks = ranks(left)
    right_ranks = ranks(right)
    if np.std(left_ranks) == 0.0 or np.std(right_ranks) == 0.0:
        return None
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def directional_agreement(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Per (route, family): compare each move's effect sign vs the family
    baseline between backends on shared observables."""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for cell in cells:
        key = (cell["graph_id"], cell["route_id"], cell["family"])
        grouped.setdefault(key, []).append(cell)
    checks: list[dict[str, Any]] = []
    severe_cap_bias_cells: dict[str, dict[str, Any]] = {}
    low_delivery_power_cells: dict[str, dict[str, Any]] = {}
    for cell in cells:
        cap_class = _cell_cap_bias_classification(cell)
        if _is_severe_cap_bias(cap_class):
            severe_cap_bias_cells[_cell_key(cell)] = {
                "cell_key": _cell_key(cell),
                "graph_id": cell["graph_id"],
                "route_id": cell["route_id"],
                "family": cell["family"],
                "label": cell["label"],
                "cap_bias_classification": cap_class,
                "cap_bias_factor": (
                    (cell.get("waiting_cap_diagnostic") or {}).get("cap_bias_factor")
                    if cell.get("waiting_cap_diagnostic") else
                    (_compiled_expected(cell).get("waiting_cap_diagnostic") or {}).get("cap_bias_factor")
                ),
            }
        if _cell_low_delivery_power(cell):
            low_delivery_power_cells[_cell_key(cell)] = {
                "cell_key": _cell_key(cell),
                "graph_id": cell["graph_id"],
                "route_id": cell["route_id"],
                "family": cell["family"],
                "label": cell["label"],
                "threshold": LOW_DELIVERY_POWER_THRESHOLD,
                "compiled_delivery_probability": _compiled_expected(cell).get("delivery_probability"),
                "compiled_delivered_fraction_mean": cell.get("compiled", {}).get("delivered_fraction_mean"),
                "sequence_delivered_fraction_mean": cell.get("sequence", {}).get("delivered_fraction_mean"),
            }
    for (graph_id, route_id, family), members in sorted(grouped.items()):
        baseline_label = "no_attack" if family == "edge" else "repeater_baseline"
        baseline = next((m for m in members if m["label"] == baseline_label), None)
        if baseline is None:
            continue
        for member in members:
            if member["label"] == baseline_label:
                continue
            for observable, tolerance in (
                ("delivered_fraction_mean", 0.02),
                ("eve_win_rate", 0.10),
                ("sampled_qber_mean", 0.03),
            ):
                qber_underpowered = (
                    observable == "sampled_qber_mean"
                    and _qber_underpowered(
                        member["sequence"],
                        baseline["sequence"],
                        member["compiled"],
                        baseline["compiled"],
                    )
                )
                seq_sign = _effect_sign(
                    None if qber_underpowered else _metric_value(member["sequence"], observable),
                    None if qber_underpowered else _metric_value(baseline["sequence"], observable),
                    tolerance,
                )
                comp_sign = _effect_sign(
                    None if qber_underpowered else _metric_value(member["compiled"], observable),
                    None if qber_underpowered else _metric_value(baseline["compiled"], observable),
                    tolerance,
                )
                if seq_sign is None or comp_sign is None:
                    agreement = None
                else:
                    agreement = (
                        seq_sign == comp_sign
                        or "flat" in (seq_sign, comp_sign)
                    )
                member_cap = _cell_cap_bias_classification(member)
                baseline_cap = _cell_cap_bias_classification(baseline)
                cap_excluded = _is_severe_cap_bias(member_cap) or _is_severe_cap_bias(baseline_cap)
                low_power_excluded = (
                    _cell_low_delivery_power(member)
                    or _cell_low_delivery_power(baseline)
                )
                qber_power_excluded = bool(qber_underpowered)
                absolute_stack_excluded = not (
                    bool(member.get("absolute_rate_comparable"))
                    and bool(baseline.get("absolute_rate_comparable"))
                )
                qualitative_cross_family = (
                    member.get("comparison_family") == "qualitative_cross_family"
                    and baseline.get("comparison_family") == "qualitative_cross_family"
                )
                non_signal_excluded = bool(cap_excluded or low_power_excluded or qber_power_excluded)
                checks.append({
                    "graph_id": graph_id,
                    "route_id": route_id,
                    "family": family,
                    "comparison_family": member.get("comparison_family", "not_comparable"),
                    "absolute_rate_comparable": bool(member.get("absolute_rate_comparable")),
                    "directional_only": bool(member.get("directional_only")),
                    "move": member["label"],
                    "observable": observable,
                    "sequence_effect": seq_sign,
                    "compiled_effect": comp_sign,
                    "agreement": agreement,
                    "baseline_cap_bias_classification": baseline_cap,
                    "member_cap_bias_classification": member_cap,
                    "excluded_from_comparable_agreement": bool(
                        absolute_stack_excluded or non_signal_excluded
                    ),
                    "excluded_from_absolute_physics_validation": bool(
                        absolute_stack_excluded or non_signal_excluded
                    ),
                    "excluded_from_qualitative_cross_family_agreement": bool(
                        (not qualitative_cross_family) or non_signal_excluded
                    ),
                    "exclusion_reasons": [
                        reason for reason, active in (
                            ("not_absolute_rate_comparable", absolute_stack_excluded),
                            ("severe_or_catastrophic_cap_bias", cap_excluded),
                            ("low_delivery_power", low_power_excluded),
                            ("qber_underpowered", qber_power_excluded),
                        )
                        if active
                    ],
                    "sequence_qber_estimation_power": (
                        _qber_power(member["sequence"])
                        if observable == "sampled_qber_mean" else None
                    ),
                    "compiled_qber_estimation_power": (
                        _qber_power(member["compiled"])
                        if observable == "sampled_qber_mean" else None
                    ),
                })
    all_summary = _agreement_summary(checks)
    absolute_checks = [
        check for check in checks
        if not check.get("excluded_from_absolute_physics_validation", False)
    ]
    absolute_summary = _agreement_summary(absolute_checks)
    qualitative_checks = [
        check for check in checks
        if not check.get("excluded_from_qualitative_cross_family_agreement", False)
    ]
    qualitative_summary = _agreement_summary(qualitative_checks)
    return {
        "definition": (
            "per move vs the same engine-family baseline: sign of the change in "
            "delivered fraction, eve-win rate, and finite-sample sampled QBER; "
            "agreement means same sign or one side within the flat tolerance band. "
            "QBER signs are not decided unless both engines have adequate sampled "
            "QBER estimation power. Absolute physics validation includes only cells "
            "whose sequence and compiled physical stacks are marked absolute-rate "
            "comparable; qualitative cross-family checks are reported separately."
        ),
        "tolerances": {
            "delivered_fraction_mean": 0.02,
            "eve_win_rate": 0.10,
            "sampled_qber_mean": 0.03,
        },
        "low_delivery_power_threshold": LOW_DELIVERY_POWER_THRESHOLD,
        "checks": checks,
        "all_cells_directional_agreement": all_summary,
        "absolute_physics_validation_directional_agreement": absolute_summary,
        "comparable_cells_directional_agreement": absolute_summary,
        "qualitative_cross_family_directional_agreement": qualitative_summary,
        "severe_cap_bias_cells_excluded": list(severe_cap_bias_cells.values()),
        "low_delivery_power_cells_excluded": list(low_delivery_power_cells.values()),
        "cap_bias_class_counts": dict(sorted(Counter(
            _cell_cap_bias_classification(cell) for cell in cells
        ).items())),
        # Backward-compatible summary fields retain the all-cells view.
        "decided_checks": all_summary["decided_checks"],
        "agreeing_checks": all_summary["agreeing_checks"],
        "agreement_rate": all_summary["agreement_rate"],
        "disagreements": all_summary["disagreements"],
    }


def strategic_ordering_parity(
    cells: list[dict[str, Any]],
    *,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """Compare matched action rankings without claiming payoff equivalence.

    This asks the strategic validator's narrow question: on a fixed route, do
    the two backends agree about which *matched* Eve intervention is more
    damaging to delivered service?  Unsupported and approximate action
    mappings are excluded.  The result is an ordering diagnostic, not a Nash
    or distributional parity claim.
    """

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    exclusions: list[dict[str, Any]] = []
    for cell in cells:
        status = str((cell.get("action_semantics") or {}).get("status", "matched"))
        reason = None
        if not bool(cell.get("absolute_rate_comparable")):
            reason = "not_absolute_rate_comparable"
        elif status != "matched":
            reason = f"action_semantics_{status}"
        elif _is_severe_cap_bias(_cell_cap_bias_classification(cell)):
            reason = "severe_or_catastrophic_cap_bias"
        elif cell.get("sequence", {}).get("errors") or cell.get("compiled", {}).get("errors"):
            reason = "backend_error"
        if reason is not None:
            exclusions.append({
                "cell_key": _cell_key(cell),
                "label": cell.get("label"),
                "reason": reason,
            })
            continue
        seq_value = _metric_value(cell.get("sequence", {}), "delivered_fraction_mean")
        cmp_value = _metric_value(cell.get("compiled", {}), "delivered_fraction_mean")
        if seq_value is None or cmp_value is None:
            exclusions.append({
                "cell_key": _cell_key(cell),
                "label": cell.get("label"),
                "reason": "delivery_metric_unavailable",
            })
            continue
        grouped.setdefault((str(cell["graph_id"]), str(cell["route_id"])), []).append(cell)

    route_rows: list[dict[str, Any]] = []
    pair_checks: list[dict[str, Any]] = []
    for (graph_id, route_id), members in sorted(grouped.items()):
        members = sorted(members, key=lambda row: str(row["label"]))
        sequence_values = {
            str(row["label"]): float(row["sequence"]["delivered_fraction_mean"])
            for row in members
        }
        compiled_values = {
            str(row["label"]): float(row["compiled"]["delivered_fraction_mean"])
            for row in members
        }
        sequence_standard_errors = {
            str(row["label"]): _metric_value(
                row["sequence"], "delivered_fraction_standard_error"
            )
            for row in members
        }
        compiled_standard_errors = {
            str(row["label"]): _metric_value(
                row["compiled"], "delivered_fraction_standard_error"
            )
            for row in members
        }
        sequence_sample_counts = {
            str(row["label"]): int(
                row["sequence"].get("delivered_fraction_sample_count", 0)
            )
            for row in members
        }
        compiled_sample_counts = {
            str(row["label"]): int(
                row["compiled"].get("delivered_fraction_sample_count", 0)
            )
            for row in members
        }
        for left_index, left in enumerate(members):
            for right in members[left_index + 1:]:
                left_label = str(left["label"])
                right_label = str(right["label"])
                seq_order = _effect_sign(
                    sequence_values[left_label], sequence_values[right_label], tolerance
                )
                cmp_order = _effect_sign(
                    compiled_values[left_label], compiled_values[right_label], tolerance
                )
                agreement = (
                    None
                    if (seq_order == "flat") != (cmp_order == "flat")
                    else bool(seq_order == cmp_order)
                )
                sequence_confidence = _confidence_order(
                    sequence_values[left_label],
                    sequence_values[right_label],
                    sequence_standard_errors[left_label],
                    sequence_standard_errors[right_label],
                    sequence_sample_counts[left_label],
                    sequence_sample_counts[right_label],
                    tolerance=tolerance,
                )
                compiled_confidence = _confidence_order(
                    compiled_values[left_label],
                    compiled_values[right_label],
                    compiled_standard_errors[left_label],
                    compiled_standard_errors[right_label],
                    compiled_sample_counts[left_label],
                    compiled_sample_counts[right_label],
                    tolerance=tolerance,
                )
                confidence_agreement = (
                    bool(
                        sequence_confidence["decision"]
                        == compiled_confidence["decision"]
                    )
                    if sequence_confidence["decision"] != "undecided"
                    and compiled_confidence["decision"] != "undecided"
                    else None
                )
                check = {
                    "graph_id": graph_id,
                    "route_id": route_id,
                    "left_action": left_label,
                    "right_action": right_label,
                    "sequence_order": seq_order,
                    "compiled_order": cmp_order,
                    "agreement": agreement,
                    "sequence_confidence_order": sequence_confidence,
                    "compiled_confidence_order": compiled_confidence,
                    "confidence_agreement": confidence_agreement,
                }
                pair_checks.append(check)
        seq_min = min(sequence_values.values())
        cmp_min = min(compiled_values.values())
        sequence_best_denial = sorted(
            label for label, value in sequence_values.items()
            if value <= seq_min + tolerance
        )
        compiled_best_denial = sorted(
            label for label, value in compiled_values.items()
            if value <= cmp_min + tolerance
        )
        route_rows.append({
            "graph_id": graph_id,
            "route_id": route_id,
            "matched_actions": sorted(sequence_values),
            "sequence_delivered_fraction": sequence_values,
            "compiled_delivered_fraction": compiled_values,
            "sequence_standard_errors": sequence_standard_errors,
            "compiled_standard_errors": compiled_standard_errors,
            "spearman_rank_correlation_with_tolerance_ties": (
                _rank_correlation(sequence_values, compiled_values, tolerance)
            ),
            "sequence_best_denial_actions": sequence_best_denial,
            "compiled_best_denial_actions": compiled_best_denial,
            "best_denial_set_overlap": bool(
                set(sequence_best_denial) & set(compiled_best_denial)
            ),
            "best_denial_set_exact_match": (
                sequence_best_denial == compiled_best_denial
            ),
        })
    decided = [row for row in pair_checks if row["agreement"] is not None]
    agreeing = sum(bool(row["agreement"]) for row in decided)
    best_overlap = sum(bool(row["best_denial_set_overlap"]) for row in route_rows)
    best_exact = sum(bool(row["best_denial_set_exact_match"]) for row in route_rows)
    confidence_decided = [
        row for row in pair_checks if row["confidence_agreement"] is not None
    ]
    confidence_agreeing = sum(
        bool(row["confidence_agreement"]) for row in confidence_decided
    )
    rank_correlations = [
        float(row["spearman_rank_correlation_with_tolerance_ties"])
        for row in route_rows
        if row["spearman_rank_correlation_with_tolerance_ties"] is not None
    ]
    return {
        "definition": (
            "within each fixed route, pairwise ordering of delivered fractions "
            "for actions with matched semantics and absolute-rate-comparable "
            "stacks; lower delivery is stronger denial"
        ),
        "tolerance": float(tolerance),
        "claim_boundary": (
            "ordering diagnostic only; this is not equality of payoff matrices, "
            "equilibrium strategies, or event distributions"
        ),
        "eligible_route_count": len(route_rows),
        "pairwise_checks": pair_checks,
        "pairwise_agreeing": agreeing,
        "pairwise_decided": len(decided),
        "pairwise_undecided_due_to_one_sided_tie": len(pair_checks) - len(decided),
        "pairwise_agreement_rate": (
            agreeing / len(decided) if decided else None
        ),
        "confidence_pairwise_agreeing": confidence_agreeing,
        "confidence_pairwise_decided": len(confidence_decided),
        "confidence_pairwise_undecided": len(pair_checks) - len(confidence_decided),
        "confidence_pairwise_agreement_rate": (
            confidence_agreeing / len(confidence_decided)
            if confidence_decided else None
        ),
        "mean_route_rank_correlation": (
            float(np.mean(rank_correlations)) if rank_correlations else None
        ),
        "best_denial_route_overlap_count": best_overlap,
        "best_denial_route_exact_match_count": best_exact,
        "best_denial_route_count": len(route_rows),
        "routes": route_rows,
        "excluded_cells": exclusions,
    }


def strategic_payoff_matrix_parity(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare graph-level fixed-target route/action payoff matrices.

    Unlike route-relative midpoint actions, these columns retain one physical
    target across every route.  Off-route targets therefore remain genuine
    misses and the rectangular matrix is a valid hidden-route control game.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, Any]] = []
    for cell in cells:
        if not bool(cell.get("graph_level_fixed_target")):
            continue
        reason = None
        if not bool(cell.get("absolute_rate_comparable")):
            reason = "not_absolute_rate_comparable"
        elif str((cell.get("action_semantics") or {}).get("status")) != "matched":
            reason = "action_semantics_not_matched"
        elif _is_severe_cap_bias(_cell_cap_bias_classification(cell)):
            reason = "severe_or_catastrophic_cap_bias"
        elif cell.get("sequence", {}).get("errors") or cell.get("compiled", {}).get("errors"):
            reason = "backend_error"
        elif _metric_value(cell.get("sequence", {}), "delivered_fraction_mean") is None:
            reason = "sequence_delivery_metric_unavailable"
        elif _metric_value(cell.get("compiled", {}), "delivered_fraction_mean") is None:
            reason = "compiled_delivery_metric_unavailable"
        if reason is not None:
            exclusions.append({
                "cell_key": _cell_key(cell),
                "graph_id": cell.get("graph_id"),
                "route_id": cell.get("route_id"),
                "label": cell.get("label"),
                "reason": reason,
            })
            continue
        grouped.setdefault(str(cell["graph_id"]), []).append(cell)

    graph_rows: list[dict[str, Any]] = []
    for graph_id, members in sorted(grouped.items()):
        route_ids = sorted({str(cell["route_id"]) for cell in members})
        action_labels = sorted({str(cell["label"]) for cell in members})
        by_key = {
            (str(cell["route_id"]), str(cell["label"])): cell
            for cell in members
        }
        missing = [
            {"route_id": route_id, "action": action}
            for route_id in route_ids
            for action in action_labels
            if (route_id, action) not in by_key
        ]
        if missing:
            graph_rows.append({
                "graph_id": graph_id,
                "status": "incomplete_rectangular_matrix",
                "missing_cells": missing,
            })
            continue
        sequence = np.asarray([
            [float(by_key[(route_id, action)]["sequence"]["delivered_fraction_mean"])
             for action in action_labels]
            for route_id in route_ids
        ])
        compiled = np.asarray([
            [float(by_key[(route_id, action)]["compiled"]["delivered_fraction_mean"])
             for action in action_labels]
            for route_id in route_ids
        ])
        costs = np.asarray([
            float(by_key[(route_ids[0], action)]["action"].get("cost", 0.0))
            for action in action_labels
        ])
        sequence_game = sequence + costs[np.newaxis, :]
        compiled_game = compiled + costs[np.newaxis, :]
        sequence_value, sequence_alice, sequence_eve = solve_zero_sum_lp(sequence_game)
        compiled_value, compiled_alice, compiled_eve = solve_zero_sum_lp(compiled_game)
        residual = sequence - compiled
        sequence_standard_errors = np.asarray([
            [float(by_key[(route_id, action)]["sequence"]
                   ["delivered_fraction_standard_error"])
             for action in action_labels]
            for route_id in route_ids
        ])
        compiled_standard_errors = np.asarray([
            [float(by_key[(route_id, action)]["compiled"]
                   ["delivered_fraction_standard_error"])
             for action in action_labels]
            for route_id in route_ids
        ])
        correlation = (
            float(np.corrcoef(sequence.ravel(), compiled.ravel())[0, 1])
            if np.std(sequence) > 0.0 and np.std(compiled) > 0.0 else None
        )
        graph_rows.append({
            "graph_id": graph_id,
            "status": "complete",
            "route_ids": route_ids,
            "action_labels": action_labels,
            "action_costs": costs.tolist(),
            "sequence_delivered_fraction_matrix": sequence.tolist(),
            "compiled_delivered_fraction_matrix": compiled.tolist(),
            "cell_rmse": float(np.sqrt(np.mean(residual ** 2))),
            "cell_mae": float(np.mean(np.abs(residual))),
            "cell_maximum_absolute_error": float(np.max(np.abs(residual))),
            "cell_pearson_correlation": correlation,
            "matrix_sampling_uncertainty": {
                "sequence_maximum_cell_standard_error": float(
                    np.max(sequence_standard_errors)
                ),
                "compiled_maximum_cell_standard_error": float(
                    np.max(compiled_standard_errors)
                ),
                "sequence_trials_per_cell": sorted({
                    int(by_key[(route_id, action)]["sequence"]
                        ["delivered_fraction_sample_count"])
                    for route_id in route_ids for action in action_labels
                }),
                "compiled_trials_per_cell": sorted({
                    int(by_key[(route_id, action)]["compiled"]
                        ["delivered_fraction_sample_count"])
                    for route_id in route_ids for action in action_labels
                }),
            },
            "sequence_equilibrium": {
                "equivalent_value": float(sequence_value),
                "alice_strategy": sequence_alice.tolist(),
                "eve_strategy": sequence_eve.tolist(),
            },
            "compiled_equilibrium": {
                "equivalent_value": float(compiled_value),
                "alice_strategy": compiled_alice.tolist(),
                "eve_strategy": compiled_eve.tolist(),
            },
            "absolute_equilibrium_value_difference": float(
                abs(sequence_value - compiled_value)
            ),
            "sequence_equilibrium_exploitability_in_compiled_game": float(
                exploitability(compiled_game, sequence_alice, sequence_eve)
            ),
            "compiled_equilibrium_exploitability_in_sequence_game": float(
                exploitability(sequence_game, compiled_alice, compiled_eve)
            ),
            "claim_boundary": (
                "point-estimate delivered-fraction control game with fixed "
                "graph-level targets; not the production Bell-qualified-rate "
                "game and not a distributional backend-equivalence claim"
            ),
        })
    complete = [row for row in graph_rows if row.get("status") == "complete"]
    return {
        "definition": (
            "rectangular route x fixed-target action matrices; Eve action "
            "columns retain the same physical node target across routes"
        ),
        "eligible_graph_count": len(complete),
        "graphs": graph_rows,
        "excluded_cells": exclusions,
        "mean_cell_rmse": (
            float(np.mean([row["cell_rmse"] for row in complete]))
            if complete else None
        ),
        "mean_absolute_equilibrium_value_difference": (
            float(np.mean([
                row["absolute_equilibrium_value_difference"] for row in complete
            ])) if complete else None
        ),
        "claim_boundary": (
            "strategic diagnostic for the predeclared small-hop subset only; "
            "does not establish corpus-wide payoff or Nash parity"
        ),
    }


# ── Driver ───────────────────────────────────────────────────────────────────

def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    # Wall-clock waiting alignment: a SeQUeNCe repeater trial retries for a
    # fixed window; give each compiled game-turn pair the per-pair share of
    # the same window so delivered fractions measure comparable time budgets.
    per_pair_window_ps = float(args.repeater_window_ms) * 1e9 / max(1, args.memory_pairs)
    backend_manager = CompiledBackendManager(args, per_pair_window_ps=per_pair_window_ps)
    models = sequence_runtime._default_models()
    cells_out: list[dict[str, Any]] = []
    graphs_out: list[dict[str, Any]] = []
    for graph_key in args.graphs:
        spec = GRAPH_BUILDERS[graph_key]()
        ir = TopologyIR.from_dict(spec["ir_dict"])
        routes = select_routes(spec)
        if args.max_routes is not None:
            routes = routes[: max(1, int(args.max_routes))]
        strategic_actions = None
        target_selection = None
        if args.comparison_mode == "fixed_route_strategic_matrix":
            strategic_actions, target_selection = strategic_actions_for_routes(
                routes,
                target_limit=args.strategic_node_targets,
            )
        graphs_out.append({
            "graph_key": graph_key,
            "graph_id": spec["graph_id"],
            "description": spec["description"],
            "node_count": len(spec["ir_dict"]["nodes"]),
            "edge_count": len(spec["ir_dict"]["edges"]),
            "strategic_target_selection": target_selection,
            "strategic_actions": strategic_actions,
            "routes": [
                {
                    "route_id": route["route_id"],
                    "path": route["path"],
                    "hop_count": route["hop_count"],
                    "fiber_lengths_m": fiber_vector_from_route(spec["ir_dict"], route),
                }
                for route in routes
            ],
        })
        for route in routes:
            route = dict(route)
            route["edge_lengths_m"] = fiber_vector_from_route(spec["ir_dict"], route)
            for cell in cells_for_route(
                route,
                comparison_mode=args.comparison_mode,
                strategic_actions=strategic_actions,
            ):
                label = f"{spec['graph_id']} {route['route_id']} {cell['label']}"
                print(f"[cell] {label}", flush=True)
                sequence_result = run_sequence_cell(
                    ir=ir,
                    route=route,
                    cell=cell,
                    models=models,
                    trials=args.sequence_trials,
                    seed=args.seed,
                    edge_periods=args.edge_periods,
                    memory_pairs=args.memory_pairs,
                    repeater_window_ms=args.repeater_window_ms,
                    security_observable=args.security_observable,
                )
                compiled_result = run_compiled_cell(
                    backend=backend_manager.backend_for_route(route),
                    ir_dict=spec["ir_dict"],
                    route=route,
                    cell=cell,
                    trials=args.compiled_trials,
                    seed=args.seed,
                    edge_periods=args.edge_periods,
                    memory_pairs=args.memory_pairs,
                )
                waiting_cap_diagnostic = (
                    (compiled_result.get("expected_infinite_shot") or {})
                    .get("waiting_cap_diagnostic")
                )
                comparison_contract = comparison_contract_for_cell(route, cell)
                cells_out.append({
                    "graph_id": spec["graph_id"],
                    "route_id": route["route_id"],
                    "route_path": route["path"],
                    "hop_count": route["hop_count"],
                    "fiber_lengths_m": route["edge_lengths_m"],
                    "family": cell["family"],
                    **comparison_contract,
                    "label": cell["label"],
                    "action": cell["action"],
                    "action_semantics": cell.get("action_semantics"),
                    "graph_level_fixed_target": bool(
                        cell.get("graph_level_fixed_target", False)
                    ),
                    "target_active_on_route": cell.get("target_active_on_route"),
                    "sequence": sequence_result,
                    "compiled": compiled_result,
                    "waiting_cap_diagnostic": waiting_cap_diagnostic,
                    "cap_bias_classification": (
                        (waiting_cap_diagnostic or {}).get(
                            "cap_bias_classification",
                            "not_applicable",
                        )
                    ),
                })
    agreement = directional_agreement(cells_out)
    ordering_parity = strategic_ordering_parity(cells_out)
    payoff_matrix_parity = strategic_payoff_matrix_parity(cells_out)
    bell_observable_parity = bell_qualified_observable_parity(cells_out)
    policy = comparison_policy_payload(
        args,
        backend_manager,
        per_pair_window_ps=per_pair_window_ps,
    )
    payload = {
        "comparison_version": COMPARISON_VERSION,
        "purpose": (
            "measured validation comparison between the compiled-histogram backend "
            "and the SeQUeNCe runtime backend on reference graphs; not a boundary "
            "of truth for either physics stack"
        ),
        "caveats": CAVEATS,
        "comparison_policy": policy,
        "settings": {
            "sequence_trials": args.sequence_trials,
            "compiled_trials": args.compiled_trials,
            "edge_periods": args.edge_periods,
            "memory_pairs": args.memory_pairs,
            "repeater_window_ms": args.repeater_window_ms,
            "signature_cap": args.signature_cap,
            "history_pair_cap": args.history_pair_cap,
            "waiting_tail_tolerance": args.waiting_tail_tolerance,
            "memory_cutoff_ps": args.memory_cutoff_ps,
            "fock_cutoff": args.fock_cutoff,
            "compiled_histogram_hop_calibration_10km": bool(
                args.compiled_histogram_hop_calibration_10km
            ),
            "compiled_histogram_hop_calibration_t_alive_s": float(
                args.compiled_histogram_hop_calibration_t_alive_s
            ),
            "compiled_histogram_hop_calibration_age_points": int(
                args.compiled_histogram_hop_calibration_age_points
            ),
            "evaluated_cache_size": int(args.evaluated_cache_size),
            "comparison_mode": args.comparison_mode,
            "security_observable": args.security_observable,
            "strategic_node_targets": int(args.strategic_node_targets),
            "seed": args.seed,
            "game_mode": GAME_MODE,
        },
        "compiled_backend_diagnostics": backend_manager.diagnostics(),
        "graphs": graphs_out,
        "cells": cells_out,
        "cap_bias_summary": cap_bias_aggregate_summary(cells_out),
        "directional_agreement": agreement,
        "strategic_ordering_parity": ordering_parity,
        "strategic_payoff_matrix_parity": payoff_matrix_parity,
        "bell_qualified_observable_parity": bell_observable_parity,
    }
    return payload


def print_tables(payload: dict[str, Any]) -> None:
    print("\n== per-cell comparison (seq = SeQUeNCe runtime, cmp = compiled histogram) ==")
    header = (
        f"{'graph':<18} {'route':<22} {'move':<26} {'fam':<4} "
        f"{'acc seq/cmp':<13} {'evewin seq/cmp':<15} {'dlvr seq/cmp':<14} "
        f"{'sample qber':<14} {'E[S] seq/cmp':<15} {'Bell dlvr':<14} {'cap':<12}"
    )
    print(header)
    print("-" * len(header))
    for cell in payload["cells"]:
        seq, cmp_ = cell["sequence"], cell["compiled"]
        expected = cmp_.get("expected_infinite_shot") or {}
        route_label = "-".join(str(n) for n in cell["route_path"])
        uncapped = (
            expected.get("waiting_cap_diagnostic")
            or expected.get("uncapped_route_ready_diagnostic")
            or {}
        )
        print(
            f"{cell['graph_id']:<18.18} {route_label:<22.22} {cell['label']:<26.26} "
            f"{cell['family']:<4} "
            f"{_fmt(seq['acceptance_rate'], 2)}/{_fmt(cmp_['acceptance_rate'], 2):<8} "
            f"{_fmt(seq['eve_win_rate'], 2)}/{_fmt(cmp_['eve_win_rate'], 2):<9} "
            f"{_fmt(seq['delivered_fraction_mean'])}/{_fmt(cmp_['delivered_fraction_mean']):<7} "
            f"{_fmt(seq['sampled_qber_mean'])}/{_fmt(cmp_['sampled_qber_mean']):<7} "
            f"{_fmt(seq.get('endpoint_state_expected_chsh_s'), 3)}/"
            f"{_fmt(expected.get('expected_chsh_S'), 3):<9} "
            f"{_fmt(seq.get('bell_qualified_delivered_fraction_mean'))}/"
            f"{_fmt(cmp_.get('bell_qualified_delivered_fraction_mean')):<7} "
            f"{str(uncapped.get('cap_bias_classification', '-')):<12.12}"
        )
        if seq["errors"]:
            print(f"    sequence_error x{len(seq['errors'])}: {seq['errors'][0]}")
        if cmp_["errors"]:
            print(f"    compiled_error x{len(cmp_['errors'])}: {cmp_['errors'][0]}")
    agreement = payload["directional_agreement"]
    all_agreement = agreement["all_cells_directional_agreement"]
    comparable = agreement["absolute_physics_validation_directional_agreement"]
    qualitative = agreement["qualitative_cross_family_directional_agreement"]
    print(
        f"\n== directional agreement vs family baselines =="
        f"\n  all cells:        {all_agreement['agreeing_checks']}/{all_agreement['decided_checks']} "
        f"({_fmt(all_agreement['agreement_rate'], 3)})"
        f"\n  absolute validation cells: {comparable['agreeing_checks']}/{comparable['decided_checks']} "
        f"({_fmt(comparable['agreement_rate'], 3)})"
        f"\n  qualitative cross-family cells: {qualitative['agreeing_checks']}/{qualitative['decided_checks']} "
        f"({_fmt(qualitative['agreement_rate'], 3)})"
        f"\n  severe/cat cap cells excluded: {len(agreement['severe_cap_bias_cells_excluded'])}"
        f"\n  low-delivery cells excluded:   {len(agreement['low_delivery_power_cells_excluded'])}"
    )
    for row in all_agreement["disagreements"]:
        print(
            f"  disagreement: {row['graph_id']} {row['route_id']} {row['move']} "
            f"{row['observable']}: seq={row['sequence_effect']} cmp={row['compiled_effect']}"
        )
    print("\ncaveats:")
    for caveat in payload["caveats"]:
        print(f"  - {caveat}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="runs/backend_physics_comparison")
    parser.add_argument(
        "--graphs",
        nargs="+",
        choices=sorted(GRAPH_BUILDERS),
        default=["braess", "k10", "metro_wan"],
    )
    parser.add_argument("--sequence-trials", type=int, default=8)
    parser.add_argument("--compiled-trials", type=int, default=64)
    parser.add_argument("--edge-periods", type=int, default=1200)
    parser.add_argument("--memory-pairs", type=int, default=8)
    parser.add_argument("--repeater-window-ms", type=float, default=60.0)
    parser.add_argument("--signature-cap", type=int, default=256)
    parser.add_argument("--history-pair-cap", type=int, default=250_000)
    parser.add_argument("--waiting-tail-tolerance", type=float, default=1e-6)
    parser.add_argument("--memory-cutoff-ps", type=float, default=None)
    parser.add_argument("--fock-cutoff", type=int, default=3)
    parser.add_argument("--detector-kind", default="pnr_snspd")
    parser.add_argument("--detector-profile", default=str(DEFAULT_DETECTOR_PROFILE_PATH))
    parser.add_argument("--detector-response-file", default=None)
    parser.add_argument("--fiber-response-file", default=None)
    parser.add_argument("--conversion-response-file", default=str(DEFAULT_CONVERSION_RESPONSE_PATH))
    parser.add_argument("--allow-wavelength-mismatch", action="store_true")
    parser.add_argument("--allow-ideal-detector-reference", action="store_true")
    parser.add_argument("--evaluated-cache-size", type=int, default=128)
    parser.add_argument(
        "--compiled-histogram-hop-calibration-10km",
        action="store_true",
        help=(
            "Use hop-based rolling_alive_pair reservation calibration derived "
            "on the standardized 10 km/link route family for the compiled backend."
        ),
    )
    parser.add_argument("--compiled-histogram-hop-calibration-t-alive-s", type=float, default=0.18)
    parser.add_argument("--compiled-histogram-hop-calibration-age-points", type=int, default=8)
    parser.add_argument("--compiled-histogram-hop-calibration-route-availability", type=float, default=0.95)
    parser.add_argument("--compiled-histogram-hop-calibration-c-max", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260704)
    parser.add_argument(
        "--comparison-mode",
        choices=COMPARISON_MODES,
        default="mixed_runtime",
        help=(
            "mixed_runtime preserves the legacy SeQUeNCe dispatch by action family; "
            "fixed_route_repeater_chain runs supported cells through the fixed-route "
            "SeQUeNCe repeater-chain reference; fixed_route_strategic_matrix "
            "uses topology-selected node targets held fixed across every route"
        ),
    )
    parser.add_argument(
        "--security-observable",
        choices=SECURITY_OBSERVABLES,
        default="endpoint_state_expectation",
        help=(
            "endpoint_state_expectation compares exact fixed-setting CHSH/QBER "
            "on the pre-measurement event state and compiled histogram state; "
            "finite_qber_transcript retains the legacy QBER-only outcome alignment"
        ),
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=None,
        help="cap the number of selected routes per graph (smoke runs)",
    )
    parser.add_argument(
        "--strategic-node-targets",
        type=int,
        default=2,
        help=(
            "number of topology-only, graph-level fixed node targets in "
            "fixed_route_strategic_matrix mode"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="small budgets for smoke runs (fewer trials, pairs, shorter windows)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.strategic_node_targets < 1:
        raise ValueError("--strategic-node-targets must be >= 1")
    if args.quick:
        args.sequence_trials = min(args.sequence_trials, 3)
        args.compiled_trials = min(args.compiled_trials, 16)
        args.edge_periods = min(args.edge_periods, 400)
        args.memory_pairs = min(args.memory_pairs, 4)
        args.repeater_window_ms = min(args.repeater_window_ms, 25.0)
        args.history_pair_cap = min(args.history_pair_cap, 25_000)
    started = time.perf_counter()
    payload = run_comparison(args)
    payload["total_wall_seconds"] = time.perf_counter() - started
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "backend_physics_comparison.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=float)
        handle.write("\n")
    print_tables(payload)
    print(f"\nwrote {out_path} ({payload['total_wall_seconds']:.1f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
