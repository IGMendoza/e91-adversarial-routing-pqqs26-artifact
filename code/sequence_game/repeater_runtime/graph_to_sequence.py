"""Compile graph payloads to precise SeQUeNCe repeater-runtime contracts.

The compiler maps graph nodes and edges to the repeater-runtime semantics used
by SeQUeNCe:

* Alice and Bob are endpoint quantum routers selected by the user/GUI.
* Every other graph node is an interim repeater router.
* Every graph edge is a fiber link.
* Multi-hop paths are kept as ordered multi-repeater paths, never compressed
  into a representative repeater.

Compilation and strategy classification are CONTROL_GAME records.  Live route
execution records are REPEATER_RUNTIME when SeQUeNCe successfully runs the
selected route, otherwise BLOCKED records without synthetic metrics.
"""

from __future__ import annotations

import csv
import heapq
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Optional

from ..claims import (
    BLOCKED,
    BLOCKED_PENDING_GRAPH_INTERFACE,
    BLOCKED_PENDING_MAPPING,
    CONTROL_GAME,
    GRAPH_OBJECTIVE_CANDIDATE,
    IMPLEMENTED_AND_TESTABLE,
    IMPLEMENTED_CONTROL_ONLY,
    REPEATER_RUNTIME,
    assert_no_private_public_fields,
)
from ..physical.registry import PhysicalModel
from ..routing.features import route_sort_key
from ..routing.route import Route, enumerate_simple_paths, make_route
from ..topology.ir import EdgeRecord, NodeRecord, TopologyError, TopologyIR, TopologyMetadata
from .repeater_memory_attacks import (
    is_repeater_memory_attack_id,
    parse_structured_repeater_attack,
    structured_attack_runtime_blocker,
)

ALICE_ENDPOINT = "alice_endpoint"
BOB_ENDPOINT = "bob_endpoint"
REPEATER = "repeater"
FIBER_LINK = "fiber_link"

BLOCKED_ALICE_BOB_NOT_SELECTED = "BLOCKED_ALICE_BOB_NOT_SELECTED"
BLOCKED_ALICE_BOB_NOT_DISTINCT = "BLOCKED_ALICE_BOB_NOT_DISTINCT"
BLOCKED_NO_ALICE_BOB_PATH = "BLOCKED_NO_ALICE_BOB_PATH"
BLOCKED_EDGE_LENGTH_MISSING = "BLOCKED_EDGE_LENGTH_MISSING"
BLOCKED_SEQUENCE_TOPOLOGY_BUILD_FAILED = "BLOCKED_SEQUENCE_TOPOLOGY_BUILD_FAILED"
BLOCKED_SEQUENCE_ROUTE_REQUEST_FAILED = "BLOCKED_SEQUENCE_ROUTE_REQUEST_FAILED"

STRATEGY_VALID_MULTI_ROUTE = "STRATEGY_VALID_MULTI_ROUTE"
STRATEGY_DEGENERATE_SINGLE_ROUTE = "STRATEGY_DEGENERATE_SINGLE_ROUTE"
STRATEGY_DEGENERATE_BOTTLENECK = "STRATEGY_DEGENERATE_BOTTLENECK"
STRATEGY_NO_ROUTE = "STRATEGY_NO_ROUTE"
STRATEGY_UNCLASSIFIED = "STRATEGY_UNCLASSIFIED"

GRAPH_OBJECTIVE_BLOCKED_REASON = (
    "graph hardness / operational-entanglement objective is not defined or "
    "mapped in this pass"
)


@dataclass(frozen=True)
class GraphNodeRole:
    role: str
    scope_label: str = CONTROL_GAME

    def __post_init__(self) -> None:
        if self.role not in {ALICE_ENDPOINT, BOB_ENDPOINT, REPEATER}:
            raise ValueError(f"unsupported graph node role {self.role!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "scope_label": self.scope_label}


@dataclass(frozen=True)
class GraphEdgeRole:
    role: str = FIBER_LINK
    scope_label: str = CONTROL_GAME

    def __post_init__(self) -> None:
        if self.role != FIBER_LINK:
            raise ValueError(f"unsupported graph edge role {self.role!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "scope_label": self.scope_label}


@dataclass(frozen=True)
class GraphToSequenceNodeMapping:
    graph_node_id: str
    runtime_node_id: str
    role: str
    scope_label: str = CONTROL_GAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_node_id": self.graph_node_id,
            "runtime_node_id": self.runtime_node_id,
            "role": self.role,
            "scope_label": self.scope_label,
        }


@dataclass(frozen=True)
class GraphToSequenceEdgeMapping:
    graph_edge_id: str
    runtime_edge_id: str
    u: str
    v: str
    runtime_u: str
    runtime_v: str
    length_m: float
    role: str = FIBER_LINK
    scope_label: str = CONTROL_GAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_edge_id": self.graph_edge_id,
            "runtime_edge_id": self.runtime_edge_id,
            "u": self.u,
            "v": self.v,
            "runtime_u": self.runtime_u,
            "runtime_v": self.runtime_v,
            "length_m": self.length_m,
            "role": self.role,
            "scope_label": self.scope_label,
        }


@dataclass(frozen=True)
class GraphToSequenceRouteMapping:
    route_id: str
    runtime_route_id: str
    graph_path: tuple[str, ...]
    runtime_path: tuple[str, ...]
    hop_count: int
    total_length_m: float
    intermediate_repeaters: tuple[str, ...]
    edge_ids: tuple[str, ...]
    edge_lengths_m: tuple[float, ...]
    is_direct: bool
    is_multi_repeater: bool
    internal_node_set: tuple[str, ...]
    edge_set: tuple[str, ...]
    overlap_with_other_routes: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    scope_label: str = CONTROL_GAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "runtime_route_id": self.runtime_route_id,
            "graph_path": list(self.graph_path),
            "runtime_path": list(self.runtime_path),
            "hop_count": self.hop_count,
            "total_length_m": self.total_length_m,
            "intermediate_repeaters": list(self.intermediate_repeaters),
            "edge_ids": list(self.edge_ids),
            "edge_lengths_m": list(self.edge_lengths_m),
            "is_direct": self.is_direct,
            "is_multi_repeater": self.is_multi_repeater,
            "internal_node_set": list(self.internal_node_set),
            "edge_set": list(self.edge_set),
            "overlap_with_other_routes": {
                route_id: {
                    "shared_internal_nodes": list(values["shared_internal_nodes"]),
                    "shared_edge_ids": list(values["shared_edge_ids"]),
                }
                for route_id, values in sorted(self.overlap_with_other_routes.items())
            },
            "scope_label": self.scope_label,
        }


@dataclass(frozen=True)
class GraphRuntimeValidity:
    runtime_supported: bool
    runtime_blocked_code: Optional[str] = None
    runtime_blocked_reason: Optional[str] = None
    scope_label: str = CONTROL_GAME

    def to_dict(self) -> dict[str, Any]:
        scope = CONTROL_GAME if self.runtime_supported else BLOCKED
        return {
            "runtime_supported": self.runtime_supported,
            "runtime_blocked_code": self.runtime_blocked_code,
            "runtime_blocked_reason": self.runtime_blocked_reason,
            "scope_label": scope,
        }


@dataclass(frozen=True)
class GraphStrategyValidity:
    strategy_benchmark_supported: bool
    strategy_label: str
    strategy_blocked_reason: Optional[str] = None
    scope_label: str = CONTROL_GAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_benchmark_supported": self.strategy_benchmark_supported,
            "strategy_label": self.strategy_label,
            "strategy_blocked_reason": self.strategy_blocked_reason,
            "scope_label": self.scope_label,
        }


@dataclass
class GraphToSequenceCompileResult:
    graph_id: str
    alice_graph_node: Optional[str]
    bob_graph_node: Optional[str]
    runtime_supported: bool
    runtime_blocked_code: Optional[str]
    runtime_blocked_reason: Optional[str]
    strategy_benchmark_supported: bool
    strategy_label: str
    strategy_blocked_reason: Optional[str]
    graph_objective_supported: bool
    graph_objective_blocked_reason: str
    node_mappings: list[GraphToSequenceNodeMapping] = field(default_factory=list)
    edge_mappings: list[GraphToSequenceEdgeMapping] = field(default_factory=list)
    candidate_routes: list[GraphToSequenceRouteMapping] = field(default_factory=list)
    selected_route: Optional[GraphToSequenceRouteMapping] = None
    selected_route_mappings: list[GraphToSequenceRouteMapping] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    scope_label: str = CONTROL_GAME
    sequence_topology_config: Optional[dict[str, Any]] = None

    @property
    def runtime_validity(self) -> GraphRuntimeValidity:
        return GraphRuntimeValidity(
            self.runtime_supported,
            self.runtime_blocked_code,
            self.runtime_blocked_reason,
        )

    @property
    def strategy_validity(self) -> GraphStrategyValidity:
        return GraphStrategyValidity(
            self.strategy_benchmark_supported,
            self.strategy_label,
            self.strategy_blocked_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "alice_graph_node": self.alice_graph_node,
            "bob_graph_node": self.bob_graph_node,
            "runtime_supported": self.runtime_supported,
            "runtime_blocked_code": self.runtime_blocked_code,
            "runtime_blocked_reason": self.runtime_blocked_reason,
            "runtime_validity": self.runtime_validity.to_dict(),
            "strategy_benchmark_supported": self.strategy_benchmark_supported,
            "strategy_label": self.strategy_label,
            "strategy_blocked_reason": self.strategy_blocked_reason,
            "strategy_validity": self.strategy_validity.to_dict(),
            "graph_objective_supported": self.graph_objective_supported,
            "graph_objective_blocked_reason": self.graph_objective_blocked_reason,
            "graph_objective": {
                "scope_label": GRAPH_OBJECTIVE_CANDIDATE,
                "graph_objective_supported": self.graph_objective_supported,
                "blocked_reason": self.graph_objective_blocked_reason,
            },
            "node_mappings": [row.to_dict() for row in self.node_mappings],
            "edge_mappings": [row.to_dict() for row in self.edge_mappings],
            "candidate_routes": [row.to_dict() for row in self.candidate_routes],
            "selected_route": (
                self.selected_route.to_dict() if self.selected_route is not None else None
            ),
            "selected_route_mappings": [
                row.to_dict() for row in self.selected_route_mappings
            ],
            "warnings": list(self.warnings),
            "scope_label": self.scope_label,
            "sequence_topology_config": self.sequence_topology_config,
            "information_gain_reward_enabled": False,
        }


def compile_gui_graph_to_sequence_runtime(payload: dict[str, Any]) -> dict[str, Any]:
    """Pure GUI-facing compiler helper; imports no Dash objects."""

    result = compile_graph_to_sequence_runtime(payload)
    payload_dict = result.to_dict()
    payload_dict.update({
        "node_role_table": payload_dict["node_mappings"],
        "edge_fiber_table": payload_dict["edge_mappings"],
        "candidate_route_table": payload_dict["candidate_routes"],
        "selected_alice_bob": {
            "alice": payload_dict["alice_graph_node"],
            "bob": payload_dict["bob_graph_node"],
        },
        "recommended_route_candidates": _recommended_route_candidates(result),
    })
    return payload_dict


def compile_graph_to_sequence_runtime(
        graph: TopologyIR | Mapping[str, Any],
        *,
        alice: Optional[str] = None,
        bob: Optional[str] = None,
        default_edge_length_m: Optional[float] = None,
        graph_id: Optional[str] = None,
        selected_route_id: Optional[str] = None) -> GraphToSequenceCompileResult:
    """Compile a GUI payload or TopologyIR to a SeQUeNCe runtime contract."""

    try:
        normalized = _normalize_graph_payload(
            graph,
            alice=alice,
            bob=bob,
            default_edge_length_m=default_edge_length_m,
            graph_id=graph_id,
        )
    except _CompileBlocked as blocked:
        return _blocked_compile_result(
            graph_id=blocked.graph_id,
            alice=blocked.alice,
            bob=blocked.bob,
            code=blocked.code,
            reason=blocked.reason,
            node_mappings=blocked.node_mappings,
            edge_mappings=blocked.edge_mappings,
            warnings=blocked.warnings,
            strategy_label=blocked.strategy_label,
        )

    ir = normalized["ir"]
    alice_id = normalized["alice"]
    bob_id = normalized["bob"]
    node_mappings = _node_mappings(ir, alice_id, bob_id)
    edge_mappings = _edge_mappings(ir)
    graph_id_value = normalized["graph_id"]
    warnings = list(normalized["warnings"])

    routes = sorted(
        (
            make_route(ir, path)
            for path in enumerate_simple_paths(ir, alice_id, bob_id)
        ),
        key=route_sort_key,
    )
    route_mappings = _attach_route_overlaps([
        _route_mapping(ir, route) for route in routes
    ])
    if not route_mappings:
        return _blocked_compile_result(
            graph_id=graph_id_value,
            alice=alice_id,
            bob=bob_id,
            code=BLOCKED_NO_ALICE_BOB_PATH,
            reason=f"no Alice-Bob path exists between {alice_id!r} and {bob_id!r}",
            node_mappings=node_mappings,
            edge_mappings=edge_mappings,
            warnings=warnings,
            strategy_label=STRATEGY_NO_ROUTE,
        )

    strategy = _classify_strategy(route_mappings)
    try:
        topology_config = _sequence_topology_config(ir, edge_mappings)
        runtime_validity = GraphRuntimeValidity(True)
    except Exception as exc:  # pragma: no cover - defensive fail-closed path
        topology_config = None
        runtime_validity = GraphRuntimeValidity(
            False,
            BLOCKED_SEQUENCE_TOPOLOGY_BUILD_FAILED,
            f"{type(exc).__name__}: {exc}",
        )

    selected = route_mappings[0]
    if selected_route_id is not None:
        selected = next(
            (route for route in route_mappings if route.route_id == selected_route_id),
            route_mappings[0],
        )
        if selected.route_id != selected_route_id:
            warnings.append(
                f"selected_route_id {selected_route_id!r} not found; using deterministic first route"
            )
    return GraphToSequenceCompileResult(
        graph_id=graph_id_value,
        alice_graph_node=alice_id,
        bob_graph_node=bob_id,
        runtime_supported=runtime_validity.runtime_supported,
        runtime_blocked_code=runtime_validity.runtime_blocked_code,
        runtime_blocked_reason=runtime_validity.runtime_blocked_reason,
        strategy_benchmark_supported=strategy.strategy_benchmark_supported,
        strategy_label=strategy.strategy_label,
        strategy_blocked_reason=strategy.strategy_blocked_reason,
        graph_objective_supported=False,
        graph_objective_blocked_reason=GRAPH_OBJECTIVE_BLOCKED_REASON,
        node_mappings=node_mappings,
        edge_mappings=edge_mappings,
        candidate_routes=route_mappings,
        selected_route=selected,
        selected_route_mappings=[selected],
        warnings=warnings,
        sequence_topology_config=topology_config,
    )


def run_sequence_on_compiled_route(
        compile_result: GraphToSequenceCompileResult | Mapping[str, Any],
        route_id: str,
        *,
        seed: int = 1,
        memory_pairs: int = 8,
        attack: str | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run the existing selected-route SeQUeNCe E91/Z-X runtime where supported."""

    result = _coerce_compile_result(compile_result)
    if not result.runtime_supported:
        return _blocked_route_execution(
            result,
            route_id,
            result.runtime_blocked_code or BLOCKED_SEQUENCE_TOPOLOGY_BUILD_FAILED,
            result.runtime_blocked_reason or "compiled graph is not runtime-supported",
        )

    route = _route_by_id(result, route_id)
    if route is None:
        return _blocked_route_execution(
            result,
            route_id,
            BLOCKED_SEQUENCE_ROUTE_REQUEST_FAILED,
            f"route_id {route_id!r} is not in compiled candidate routes",
        )
    attack_id = _attack_id_from_input(attack)
    if is_repeater_memory_attack_id(attack_id):
        structured_attack = parse_structured_repeater_attack(attack_id)
        if structured_attack.target_repeater in {route.runtime_path[0], route.runtime_path[-1]}:
            return _blocked_route_execution(
                result,
                route_id,
                BLOCKED_SEQUENCE_ROUTE_REQUEST_FAILED,
                (
                    f"attack target {structured_attack.target_repeater!r} is an "
                    "Alice/Bob endpoint, not an interior repeater"
                ),
            )
        precheck = structured_attack_runtime_blocker(
            structured_attack,
            selected_path=tuple(route.runtime_path),
        )
        if precheck.get("scope_label") == BLOCKED:
            precheck.update({
                "graph_id": result.graph_id,
                "route_id": route.route_id,
                "runtime_supported": False,
            })
            return precheck
        if precheck.get("active_route_attacked") is False:
            attack_id = "no_attack"
    elif not _is_supported_runtime_attack_id(attack_id):
        from ..eve.attack_capabilities import AttackCapabilityError, blocked_attack_response

        try:
            blocked = blocked_attack_response(attack_id)
        except AttackCapabilityError:
            return _blocked_route_execution(
                result,
                route_id,
                BLOCKED_SEQUENCE_ROUTE_REQUEST_FAILED,
                f"unsupported attack_id {attack_id!r}",
            )
        blocked.update({
            "graph_id": result.graph_id,
            "route_id": route.route_id,
            "runtime_supported": False,
        })
        return blocked

    try:
        from ..protocol.postprocessing import SiftingConfig
        from ..protocol.repeater_trial import (
            RepeaterE91RunConfig,
            run_fixed_repeater_e91_trial,
        )
        from ..sequence_build.repeater_e91_builder import FixedRepeaterPath

        run_config = RepeaterE91RunConfig(
            memory_pairs=memory_pairs,
            sifting=SiftingConfig(qber_threshold=0.5, min_sifted_samples=1),
            request_fidelity=0.01,
            start_time_ps=1_000_000,
            end_time_ps=80_000_000,
            stop_time_ps=100_000_000,
            swapping_success_prob=1.0,
            swapping_degradation=1.0,
        )
        runtime_attack = _runtime_attack_spec(attack_id, route)
        transcript, built = run_fixed_repeater_e91_trial(
            FixedRepeaterPath(
                nodes=tuple(route.runtime_path),
                edge_lengths_m=tuple(route.edge_lengths_m),
            ),
            memory_model=_toy_memory_model(),
            fiber_model=_toy_fiber_model(),
            run_config=run_config,
            trial_id=f"graph_route_{route.route_id}",
            seed=seed,
            attack=runtime_attack,
        )
    except Exception as exc:
        return _blocked_route_execution(
            result,
            route_id,
            BLOCKED_SEQUENCE_ROUTE_REQUEST_FAILED,
            f"{type(exc).__name__}: {exc}",
        )

    if transcript.accepted:
        public_outcome = "accepted"
    elif transcript.abort_reason == "qber_above_threshold":
        public_outcome = "qber_abort"
    else:
        public_outcome = "delivery_failure"
    record = {
        "scope_label": REPEATER_RUNTIME,
        "claim_status": IMPLEMENTED_AND_TESTABLE,
        "graph_id": result.graph_id,
        "route_id": route.route_id,
        "graph_path": list(route.graph_path),
        "runtime_path": list(route.runtime_path),
        "edge_ids": list(route.edge_ids),
        "edge_lengths_m": list(route.edge_lengths_m),
        "alice_graph_node": result.alice_graph_node,
        "bob_graph_node": result.bob_graph_node,
        "delivery_success": transcript.generation_successes > 0,
        "delivered_count": transcript.generation_successes,
        "qber": transcript.qber_estimate,
        "fidelity": transcript.extra.get("mean_actual_fidelity"),
        "amer_style_d_eff_mean": transcript.extra.get("amer_style_d_eff_mean"),
        "public_outcome": public_outcome,
        "abort_reason": transcript.abort_reason,
        "runtime_attack_applied": runtime_attack.to_dict(),
        "active_route_attacked": runtime_attack.is_active,
        "router_net_summary": {
            key: value for key, value in built.summary().items()
            if key != "config_path"
        },
        "information_gain_reward_enabled": False,
        "graph_objective_supported": False,
    }
    assert_no_private_public_fields(record)
    return record


def write_graph_to_sequence_mapping_artifacts(
        examples: Mapping[str, Mapping[str, Any]],
        *,
        contract_path: Path,
        examples_json_path: Path,
        examples_csv_path: Path) -> dict[str, Any]:
    """Write mapping contract and example compile results."""

    compiled = {
        name: compile_gui_graph_to_sequence_runtime(dict(payload))
        for name, payload in examples.items()
    }
    contract = graph_to_sequence_mapping_contract()
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    examples_json_path.parent.mkdir(parents=True, exist_ok=True)
    examples_json_path.write_text(
        json.dumps(compiled, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_examples_csv(compiled, examples_csv_path)
    return {"contract": contract, "examples": compiled}


def graph_to_sequence_mapping_contract() -> dict[str, Any]:
    return {
        "scope_label": CONTROL_GAME,
        "node_roles": [ALICE_ENDPOINT, BOB_ENDPOINT, REPEATER],
        "edge_roles": [FIBER_LINK],
        "runtime_block_codes": [
            BLOCKED_ALICE_BOB_NOT_SELECTED,
            BLOCKED_ALICE_BOB_NOT_DISTINCT,
            BLOCKED_NO_ALICE_BOB_PATH,
            BLOCKED_EDGE_LENGTH_MISSING,
            BLOCKED_SEQUENCE_TOPOLOGY_BUILD_FAILED,
            BLOCKED_SEQUENCE_ROUTE_REQUEST_FAILED,
        ],
        "strategy_labels": [
            STRATEGY_VALID_MULTI_ROUTE,
            STRATEGY_DEGENERATE_SINGLE_ROUTE,
            STRATEGY_DEGENERATE_BOTTLENECK,
            STRATEGY_NO_ROUTE,
            STRATEGY_UNCLASSIFIED,
        ],
        "validity_dimensions": [
            "runtime_supported",
            "strategy_benchmark_supported",
            "graph_objective_supported",
        ],
        "blocked_claims": {
            "graph_objective": GRAPH_OBJECTIVE_BLOCKED_REASON,
            "information_gain_reward": "information_gain_reward_enabled remains false",
            "full_security_claim": "finite-key and Harkness coupling remain blocked",
            "detector_side_channels": (
                "blocked unless detector profiles and public transcript mappings are supplied"
            ),
            "chsh": "CHSH endpoint monitor remains blocked",
            "q_neat": "Q-NEAT remains blocked",
            "xai": "XAI claims remain blocked",
            "new_hardware_adapters": "new hardware adapters remain blocked",
            "new_eve_attack_families": "new Eve attack families remain blocked",
        },
    }


def mapping_example_payloads() -> dict[str, dict[str, Any]]:
    return {
        "direct_alice_bob_edge": {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"u": "a", "v": "b", "length": 10.0}],
            "alice": "a",
            "bob": "b",
        },
        "one_repeater_path": {
            "nodes": [{"id": "a"}, {"id": "r"}, {"id": "b"}],
            "edges": [
                {"u": "a", "v": "r", "length": 10.0},
                {"u": "r", "v": "b", "length": 10.0},
            ],
            "alice": "a",
            "bob": "b",
        },
        "multi_repeater_path": {
            "nodes": [{"id": "a"}, {"id": "r1"}, {"id": "r2"}, {"id": "b"}],
            "edges": [
                {"u": "a", "v": "r1", "length": 10.0},
                {"u": "r1", "v": "r2", "length": 20.0},
                {"u": "r2", "v": "b", "length": 30.0},
            ],
            "alice": "a",
            "bob": "b",
        },
        "two_route_graph": {
            "nodes": [{"id": "a"}, {"id": "r0"}, {"id": "r1"}, {"id": "b"}],
            "edges": [
                {"u": "a", "v": "r0", "length": 10.0},
                {"u": "r0", "v": "b", "length": 10.0},
                {"u": "a", "v": "r1", "length": 10.0},
                {"u": "r1", "v": "b", "length": 10.0},
            ],
            "alice": "a",
            "bob": "b",
        },
        "bottleneck_graph": {
            "nodes": [{"id": "a"}, {"id": "left"}, {"id": "right"}, {"id": "cut"}, {"id": "b"}],
            "edges": [
                {"u": "a", "v": "left", "length": 10.0},
                {"u": "left", "v": "cut", "length": 10.0},
                {"u": "a", "v": "right", "length": 10.0},
                {"u": "right", "v": "cut", "length": 10.0},
                {"u": "cut", "v": "b", "length": 10.0},
            ],
            "alice": "a",
            "bob": "b",
        },
        "disconnected_alice_bob": {
            "nodes": [{"id": "a"}, {"id": "r"}, {"id": "b"}],
            "edges": [{"u": "a", "v": "r", "length": 10.0}],
            "alice": "a",
            "bob": "b",
        },
        "missing_edge_length": {
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"u": "a", "v": "b"}],
            "alice": "a",
            "bob": "b",
        },
    }


class _CompileBlocked(Exception):
    def __init__(self, *,
                 graph_id: str,
                 alice: Optional[str],
                 bob: Optional[str],
                 code: str,
                 reason: str,
                 node_mappings: Optional[list[GraphToSequenceNodeMapping]] = None,
                 edge_mappings: Optional[list[GraphToSequenceEdgeMapping]] = None,
                 warnings: Optional[list[str]] = None,
                 strategy_label: str = STRATEGY_UNCLASSIFIED):
        self.graph_id = graph_id
        self.alice = alice
        self.bob = bob
        self.code = code
        self.reason = reason
        self.node_mappings = node_mappings or []
        self.edge_mappings = edge_mappings or []
        self.warnings = warnings or []
        self.strategy_label = strategy_label


def _normalize_graph_payload(
        graph: TopologyIR | Mapping[str, Any],
        *,
        alice: Optional[str],
        bob: Optional[str],
        default_edge_length_m: Optional[float],
        graph_id: Optional[str]) -> dict[str, Any]:
    if isinstance(graph, TopologyIR):
        ir = graph
        ir.validate()
        alice_id = alice or _single_role_node(ir, "alice")
        bob_id = bob or _single_role_node(ir, "bob")
        graph_id_value = graph_id or _graph_id(ir)
        if not alice_id or not bob_id:
            raise _CompileBlocked(
                graph_id=graph_id_value,
                alice=alice_id,
                bob=bob_id,
                code=BLOCKED_ALICE_BOB_NOT_SELECTED,
                reason="Alice and Bob must both be selected",
                node_mappings=[],
            )
        if alice_id == bob_id:
            raise _CompileBlocked(
                graph_id=graph_id_value,
                alice=alice_id,
                bob=bob_id,
                code=BLOCKED_ALICE_BOB_NOT_DISTINCT,
                reason="Alice and Bob must be distinct graph nodes",
                node_mappings=_node_mappings(ir, alice_id, bob_id),
            )
        return {
            "ir": ir,
            "alice": str(alice_id),
            "bob": str(bob_id),
            "graph_id": graph_id_value,
            "warnings": [],
        }

    payload = dict(graph)
    alice_id = str(alice or payload.get("alice") or "")
    bob_id = str(bob or payload.get("bob") or "")
    graph_id_value = graph_id or str(payload.get("graph_id") or "")
    nodes_payload = list(payload.get("nodes", []))
    node_ids = [
        _node_id_from_payload(node)
        for node in nodes_payload
        if _node_id_from_payload(node)
    ]
    if not graph_id_value:
        graph_id_value = _stable_graph_id_from_parts(node_ids, payload.get("edges", []))
    if not alice_id or not bob_id:
        raise _CompileBlocked(
            graph_id=graph_id_value,
            alice=alice_id or None,
            bob=bob_id or None,
            code=BLOCKED_ALICE_BOB_NOT_SELECTED,
            reason="Alice and Bob must both be selected",
            node_mappings=[
                GraphToSequenceNodeMapping(node_id, node_id, REPEATER)
                for node_id in sorted(node_ids)
            ],
        )
    if alice_id == bob_id:
        pseudo_ir = _ir_from_parts(node_ids, [], graph_id_value)
        raise _CompileBlocked(
            graph_id=graph_id_value,
            alice=alice_id,
            bob=bob_id,
            code=BLOCKED_ALICE_BOB_NOT_DISTINCT,
            reason="Alice and Bob must be distinct graph nodes",
            node_mappings=_node_mappings(pseudo_ir, alice_id, bob_id),
        )

    edge_records: list[EdgeRecord] = []
    warnings: list[str] = []
    for index, edge in enumerate(payload.get("edges", [])):
        edge_id = str(edge.get("edge_id") or edge.get("id") or "")
        if "u" not in edge or "v" not in edge:
            raise _CompileBlocked(
                graph_id=graph_id_value,
                alice=alice_id,
                bob=bob_id,
                code=BLOCKED_SEQUENCE_TOPOLOGY_BUILD_FAILED,
                reason=f"edge at index {index} must contain u and v",
                node_mappings=[
                    GraphToSequenceNodeMapping(node_id, node_id, _role_for_node(node_id, alice_id, bob_id))
                    for node_id in sorted(node_ids)
                ],
                warnings=warnings,
            )
        u = str(edge["u"])
        v = str(edge["v"])
        if not edge_id:
            edge_id = _stable_edge_id(u, v)
        length_value = edge.get("length_m", edge.get("length"))
        if length_value is None:
            if default_edge_length_m is None:
                pseudo_ir = _ir_from_parts(node_ids, [], graph_id_value)
                raise _CompileBlocked(
                    graph_id=graph_id_value,
                    alice=alice_id,
                    bob=bob_id,
                    code=BLOCKED_EDGE_LENGTH_MISSING,
                    reason=f"edge {edge_id!r} is missing length/length_m",
                    node_mappings=_node_mappings(pseudo_ir, alice_id, bob_id),
                    warnings=warnings,
                )
            length_value = default_edge_length_m
            warnings.append(
                f"edge {edge_id!r} used documented default length {default_edge_length_m}"
            )
        edge_records.append(
            EdgeRecord(edge_id, u, v, float(length_value), eve_eligible=True)
        )
    ir = _ir_from_parts(node_ids, edge_records, graph_id_value)
    return {
        "ir": ir,
        "alice": alice_id,
        "bob": bob_id,
        "graph_id": graph_id_value,
        "warnings": warnings,
    }


def _ir_from_parts(node_ids: list[str],
                   edge_records: list[EdgeRecord],
                   graph_id: str) -> TopologyIR:
    nodes = {node_id: NodeRecord(node_id) for node_id in sorted(set(node_ids))}
    ir = TopologyIR(
        nodes=nodes,
        edges=edge_records,
        metadata=TopologyMetadata("graph_to_sequence_payload", {"graph_id": graph_id}),
    )
    ir.validate()
    return ir


def _node_id_from_payload(node: Mapping[str, Any]) -> str:
    value = node.get("id", node.get("node_id"))
    if value is None:
        return ""
    return str(value)


def _single_role_node(ir: TopologyIR, role: str) -> Optional[str]:
    nodes = ir.nodes_with_role(role)
    return nodes[0] if nodes else None


def _node_mappings(ir: TopologyIR,
                   alice: Optional[str],
                   bob: Optional[str]) -> list[GraphToSequenceNodeMapping]:
    return [
        GraphToSequenceNodeMapping(
            graph_node_id=node_id,
            runtime_node_id=node_id,
            role=_role_for_node(node_id, alice, bob),
        )
        for node_id in sorted(ir.nodes)
    ]


def _role_for_node(node_id: str,
                   alice: Optional[str],
                   bob: Optional[str]) -> str:
    if node_id == alice:
        return ALICE_ENDPOINT
    if node_id == bob:
        return BOB_ENDPOINT
    return REPEATER


def _edge_mappings(ir: TopologyIR) -> list[GraphToSequenceEdgeMapping]:
    return [
        GraphToSequenceEdgeMapping(
            graph_edge_id=edge.edge_id,
            runtime_edge_id=edge.edge_id,
            u=edge.u,
            v=edge.v,
            runtime_u=edge.u,
            runtime_v=edge.v,
            length_m=float(edge.length_m),
        )
        for edge in sorted(ir.edges, key=lambda edge: edge.edge_id)
    ]


def _route_mapping(ir: TopologyIR, route: Route) -> GraphToSequenceRouteMapping:
    lengths = []
    for u, v in zip(route.path, route.path[1:]):
        edge = ir.edge_between(u, v)
        if edge is None:
            raise TopologyError(f"route {route.route_id} references missing edge {u}-{v}")
        lengths.append(float(edge.length_m))
    internal = tuple(route.path[1:-1])
    return GraphToSequenceRouteMapping(
        route_id=route.route_id,
        runtime_route_id=route.route_id,
        graph_path=route.path,
        runtime_path=route.path,
        hop_count=route.hop_count,
        total_length_m=route.total_length_m,
        intermediate_repeaters=internal,
        edge_ids=route.edge_ids,
        edge_lengths_m=tuple(lengths),
        is_direct=len(internal) == 0,
        is_multi_repeater=len(internal) > 1,
        internal_node_set=tuple(sorted(internal)),
        edge_set=tuple(sorted(route.edge_ids)),
    )


def _attach_route_overlaps(
        routes: list[GraphToSequenceRouteMapping]
        ) -> list[GraphToSequenceRouteMapping]:
    updated = []
    for route in routes:
        overlaps: dict[str, dict[str, tuple[str, ...]]] = {}
        route_internal = set(route.internal_node_set)
        route_edges = set(route.edge_set)
        for other in routes:
            if other.route_id == route.route_id:
                continue
            overlaps[other.route_id] = {
                "shared_internal_nodes": tuple(
                    sorted(route_internal & set(other.internal_node_set))
                ),
                "shared_edge_ids": tuple(sorted(route_edges & set(other.edge_set))),
            }
        updated.append(replace(route, overlap_with_other_routes=overlaps))
    return updated


def _classify_strategy(routes: list[GraphToSequenceRouteMapping]) -> GraphStrategyValidity:
    if not routes:
        return GraphStrategyValidity(
            False,
            STRATEGY_NO_ROUTE,
            "no Alice-Bob route exists",
        )
    if len(routes) == 1:
        return GraphStrategyValidity(
            False,
            STRATEGY_DEGENERATE_SINGLE_ROUTE,
            "only one Alice-Bob route exists; runtime can run it but route-choice strategy is degenerate",
        )
    for first, second in _route_pairs(routes):
        if set(first.internal_node_set).isdisjoint(second.internal_node_set):
            return GraphStrategyValidity(True, STRATEGY_VALID_MULTI_ROUTE)
    for first, second in _route_pairs(routes):
        if set(first.edge_set).isdisjoint(second.edge_set):
            return GraphStrategyValidity(True, STRATEGY_VALID_MULTI_ROUTE)
    return GraphStrategyValidity(
        False,
        STRATEGY_DEGENERATE_BOTTLENECK,
        "multiple Alice-Bob paths exist, but no internally node-disjoint or edge-disjoint pair exists",
    )


def _route_pairs(routes: list[GraphToSequenceRouteMapping]):
    for i, first in enumerate(routes):
        for second in routes[i + 1:]:
            yield first, second


def _recommended_route_candidates(
        result: GraphToSequenceCompileResult) -> list[dict[str, Any]]:
    if not result.strategy_benchmark_supported:
        return []
    best_pair = None
    best_rank = None
    for first, second in _route_pairs(result.candidate_routes):
        node_disjoint = set(first.internal_node_set).isdisjoint(second.internal_node_set)
        edge_disjoint = set(first.edge_set).isdisjoint(second.edge_set)
        if not node_disjoint and not edge_disjoint:
            continue
        rank = (
            0 if node_disjoint else 1,
            first.total_length_m + second.total_length_m,
            first.hop_count + second.hop_count,
            first.graph_path,
            second.graph_path,
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best_pair = (first, second)
    if best_pair is None:
        return []
    return [
        {
            "route_id": route.route_id,
            "runtime_route_id": route.runtime_route_id,
            "graph_path": list(route.graph_path),
            "total_length_m": route.total_length_m,
            "hop_count": route.hop_count,
        }
        for route in best_pair
    ]


def _sequence_topology_config(
        ir: TopologyIR,
        edge_mappings: list[GraphToSequenceEdgeMapping]) -> dict[str, Any]:
    memory_template = {
        "MemoryArray": {
            "fidelity": 1.0,
            "frequency": 8e7,
            "efficiency": 1.0,
            "coherence_time": -1.0,
            "wavelength": 500,
        },
        "routing": "static",
    }
    nodes = [
        {
            "name": node_id,
            "type": "QuantumRouter",
            "seed": index,
            "memo_size": 16,
            "template": f"memo_{node_id}",
        }
        for index, node_id in enumerate(sorted(ir.nodes))
    ]
    qconnections = [
        {
            "node1": edge.u,
            "node2": edge.v,
            "attenuation": 0.0,
            "distance": edge.length_m,
            "type": "meet_in_the_middle",
            "seed": 100 + index,
            "graph_edge_id": edge.graph_edge_id,
        }
        for index, edge in enumerate(edge_mappings)
    ]
    distances = _all_pairs_shortest_distances(ir)
    cconnections = []
    sorted_nodes = sorted(ir.nodes)
    for i, u in enumerate(sorted_nodes):
        for v in sorted_nodes[i + 1:]:
            distance = distances.get((u, v))
            if distance is None:
                continue
            cconnections.append({
                "node1": u,
                "node2": v,
                "distance": distance,
                "delay": round(distance / 2e-4),
            })
    return {
        "stop_time": 100_000_000,
        "templates": {
            f"memo_{node_id}": dict(memory_template)
            for node_id in sorted(ir.nodes)
        },
        "nodes": nodes,
        "qconnections": qconnections,
        "cconnections": cconnections,
        "graph_runtime_semantics": {
            "scope_label": CONTROL_GAME,
            "node_rule": "Alice/Bob are endpoint QuantumRouters; all other graph nodes are repeater QuantumRouters",
            "edge_rule": "Every graph edge is a fiber qconnection using RouterNetTopo meet-in-the-middle conventions",
        },
    }


def _all_pairs_shortest_distances(ir: TopologyIR) -> dict[tuple[str, str], float]:
    adj: dict[str, list[tuple[str, float]]] = {node: [] for node in ir.nodes}
    for edge in ir.edges:
        adj[edge.u].append((edge.v, float(edge.length_m)))
        adj[edge.v].append((edge.u, float(edge.length_m)))
    distances = {}
    for start in sorted(ir.nodes):
        best = {start: 0.0}
        heap = [(0.0, start)]
        while heap:
            dist, node = heapq.heappop(heap)
            if dist != best[node]:
                continue
            for nbr, weight in adj[node]:
                nd = dist + weight
                if nbr not in best or nd < best[nbr]:
                    best[nbr] = nd
                    heapq.heappush(heap, (nd, nbr))
        for end, dist in best.items():
            if start < end:
                distances[(start, end)] = dist
            elif end < start:
                distances[(end, start)] = dist
    return distances


def _blocked_compile_result(
        *,
        graph_id: str,
        alice: Optional[str],
        bob: Optional[str],
        code: str,
        reason: str,
        node_mappings: list[GraphToSequenceNodeMapping],
        edge_mappings: Optional[list[GraphToSequenceEdgeMapping]] = None,
        warnings: Optional[list[str]] = None,
        strategy_label: str = STRATEGY_UNCLASSIFIED) -> GraphToSequenceCompileResult:
    if strategy_label == STRATEGY_NO_ROUTE:
        strategy_reason = "no Alice-Bob route exists"
    elif strategy_label == STRATEGY_UNCLASSIFIED:
        strategy_reason = "strategy classification unavailable because runtime compilation is blocked"
    else:
        strategy_reason = reason
    return GraphToSequenceCompileResult(
        graph_id=graph_id,
        alice_graph_node=alice,
        bob_graph_node=bob,
        runtime_supported=False,
        runtime_blocked_code=code,
        runtime_blocked_reason=reason,
        strategy_benchmark_supported=False,
        strategy_label=strategy_label,
        strategy_blocked_reason=strategy_reason,
        graph_objective_supported=False,
        graph_objective_blocked_reason=GRAPH_OBJECTIVE_BLOCKED_REASON,
        node_mappings=node_mappings,
        edge_mappings=edge_mappings or [],
        candidate_routes=[],
        selected_route=None,
        selected_route_mappings=[],
        warnings=warnings or [],
        sequence_topology_config=None,
    )


def _route_by_id(result: GraphToSequenceCompileResult,
                 route_id: str) -> Optional[GraphToSequenceRouteMapping]:
    for route in result.candidate_routes:
        if route.route_id == route_id:
            return route
    return None


def _blocked_route_execution(
        result: GraphToSequenceCompileResult,
        route_id: str,
        code: str,
        reason: str) -> dict[str, Any]:
    return {
        "scope_label": BLOCKED,
        "claim_status": BLOCKED_PENDING_MAPPING,
        "blocked": True,
        "runtime_blocked_code": code,
        "blocked_reason": reason,
        "graph_id": result.graph_id,
        "route_id": route_id,
        "runtime_supported": False,
        "information_gain_reward_enabled": False,
    }


def _attack_id_from_input(attack: str | Mapping[str, Any] | None) -> str:
    if attack is None:
        return "no_attack"
    if isinstance(attack, str):
        return "no_attack" if attack in {"none", "no_attack"} else attack
    kind = str(attack.get("kind", attack.get("attack_id", "no_attack")))
    if kind in {"none", "no_attack"}:
        return "no_attack"
    if kind in {"swap_denial", "memory_degradation"}:
        target = str(attack.get("target_node", attack.get("target_repeater", "")))
        return f"{kind}:{target}" if target else kind
    if kind == "repeater_memory_measurement":
        target = str(attack.get("target_node", attack.get("target_repeater", "")))
        basis = str(attack.get("basis", "Z"))
        return f"repeater_memory_measure_{basis}:{target}" if target else kind
    if kind == "repeater_memory_dephasing_probe":
        target = str(attack.get("target_node", attack.get("target_repeater", "")))
        theta = str(attack.get("theta_label", attack.get("theta", "")))
        return f"repeater_memory_dephasing_probe:{target}:{theta}" if target and theta else kind
    return str(attack.get("attack_id", kind))


def _runtime_attack_spec(attack_id: str, route: GraphToSequenceRouteMapping):
    from ..eve.repeater_attacks import RepeaterAttackSpec

    if attack_id in {"none", "no_attack"}:
        return RepeaterAttackSpec(kind="none")
    if attack_id.startswith("swap_denial"):
        target = _target_from_attack_id(attack_id) or _first_repeater(route)
        if target not in route.intermediate_repeaters:
            return RepeaterAttackSpec(kind="none")
        return RepeaterAttackSpec(kind="swap_denial", target_node=target)
    if attack_id.startswith("memory_degradation"):
        target = _target_from_attack_id(attack_id) or _first_repeater(route)
        if target not in route.intermediate_repeaters:
            return RepeaterAttackSpec(kind="none")
        return RepeaterAttackSpec(
            kind="memory_degradation",
            target_node=target,
            memory_fidelity_multiplier=0.5,
        )
    if is_repeater_memory_attack_id(attack_id):
        attack = parse_structured_repeater_attack(attack_id)
        if attack.target_repeater not in route.intermediate_repeaters:
            return RepeaterAttackSpec(kind="none")
        if attack.family == "repeater_memory_measurement":
            return RepeaterAttackSpec(
                kind="repeater_memory_measurement",
                target_node=attack.target_repeater,
                basis=attack.basis,
                attack_id=attack.attack_id,
            )
        if attack.family == "repeater_memory_dephasing_probe":
            return RepeaterAttackSpec(
                kind="repeater_memory_dephasing_probe",
                target_node=attack.target_repeater,
                theta=attack.theta,
                theta_label=attack.theta_label,
                attack_id=attack.attack_id,
            )
    return RepeaterAttackSpec(kind="none")


def _is_supported_runtime_attack_id(attack_id: str) -> bool:
    return (
        attack_id in {"none", "no_attack"}
        or attack_id.startswith("swap_denial")
        or attack_id.startswith("memory_degradation")
        or is_repeater_memory_attack_id(attack_id)
    )


def _target_from_attack_id(attack_id: str) -> str:
    if attack_id.startswith("repeater_memory_dephasing_probe:"):
        parts = attack_id.split(":")
        return parts[1] if len(parts) >= 2 else ""
    if attack_id.startswith("repeater_ancilla_coupling:"):
        parts = attack_id.split(":")
        return parts[1] if len(parts) >= 2 else ""
    if ":" not in attack_id:
        return ""
    return attack_id.split(":", 1)[1]


def _first_repeater(route: GraphToSequenceRouteMapping) -> str:
    return route.intermediate_repeaters[0] if route.intermediate_repeaters else ""


def _coerce_compile_result(
        value: GraphToSequenceCompileResult | Mapping[str, Any]
        ) -> GraphToSequenceCompileResult:
    if isinstance(value, GraphToSequenceCompileResult):
        return value
    # Public API primarily returns dataclasses; mapping support is for simple
    # JSON round-trip call sites.
    routes = [
        GraphToSequenceRouteMapping(
            route_id=row["route_id"],
            runtime_route_id=row.get("runtime_route_id", row["route_id"]),
            graph_path=tuple(row["graph_path"]),
            runtime_path=tuple(row["runtime_path"]),
            hop_count=int(row["hop_count"]),
            total_length_m=float(row["total_length_m"]),
            intermediate_repeaters=tuple(row["intermediate_repeaters"]),
            edge_ids=tuple(row["edge_ids"]),
            edge_lengths_m=tuple(float(v) for v in row["edge_lengths_m"]),
            is_direct=bool(row["is_direct"]),
            is_multi_repeater=bool(row["is_multi_repeater"]),
            internal_node_set=tuple(row["internal_node_set"]),
            edge_set=tuple(row["edge_set"]),
            overlap_with_other_routes={
                other_route_id: {
                    "shared_internal_nodes": tuple(
                        values.get("shared_internal_nodes", [])
                    ),
                    "shared_edge_ids": tuple(values.get("shared_edge_ids", [])),
                }
                for other_route_id, values in row.get(
                    "overlap_with_other_routes", {}
                ).items()
            },
        )
        for row in value.get("candidate_routes", [])
    ]
    selected = None
    if value.get("selected_route"):
        selected_id = value["selected_route"]["route_id"]
        selected = next((route for route in routes if route.route_id == selected_id), None)
    return GraphToSequenceCompileResult(
        graph_id=value["graph_id"],
        alice_graph_node=value.get("alice_graph_node"),
        bob_graph_node=value.get("bob_graph_node"),
        runtime_supported=bool(value["runtime_supported"]),
        runtime_blocked_code=value.get("runtime_blocked_code"),
        runtime_blocked_reason=value.get("runtime_blocked_reason"),
        strategy_benchmark_supported=bool(value["strategy_benchmark_supported"]),
        strategy_label=value["strategy_label"],
        strategy_blocked_reason=value.get("strategy_blocked_reason"),
        graph_objective_supported=bool(value["graph_objective_supported"]),
        graph_objective_blocked_reason=value["graph_objective_blocked_reason"],
        candidate_routes=routes,
        selected_route=selected,
        selected_route_mappings=[selected] if selected is not None else [],
        warnings=list(value.get("warnings", [])),
        sequence_topology_config=value.get("sequence_topology_config"),
    )


def _toy_memory_model() -> PhysicalModel:
    return PhysicalModel("graph_runtime_memory", "memory", "toy", "graph runtime toy memory", {
        "fidelity": 1.0,
        "frequency": 8e7,
        "efficiency": 1.0,
        "coherence_time": -1,
        "wavelength": 500.0,
    })


def _toy_fiber_model() -> PhysicalModel:
    return PhysicalModel("graph_runtime_fiber", "fiber_channel", "toy", "graph runtime toy fiber", {
        "attenuation": 0.0,
        "polarization_fidelity": 1.0,
        "light_speed": 2e-4,
        "frequency": 8e7,
    })


def _graph_id(ir: TopologyIR) -> str:
    value = ir.metadata.params.get("graph_id")
    if value:
        return str(value)
    payload = {
        "nodes": sorted(ir.nodes),
        "edges": [
            [edge.edge_id, edge.u, edge.v, edge.length_m]
            for edge in sorted(ir.edges, key=lambda e: e.edge_id)
        ],
        "metadata": ir.metadata.to_dict(),
    }
    return _stable_hash_id("graph", payload)


def _stable_graph_id_from_parts(nodes: list[str], edges: Any) -> str:
    return _stable_hash_id("graph", {"nodes": sorted(nodes), "edges": edges})


def _stable_edge_id(u: str, v: str) -> str:
    a, b = sorted((u, v))
    return f"edge:{a}--{b}"


def _stable_hash_id(prefix: str, payload: Any) -> str:
    import hashlib

    text = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _write_examples_csv(compiled: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "example",
        "runtime_supported",
        "runtime_blocked_code",
        "strategy_label",
        "strategy_benchmark_supported",
        "selected_route_id",
        "selected_graph_path",
        "selected_hop_count",
        "selected_total_length_m",
        "graph_objective_supported",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for name, result in compiled.items():
            selected = result.get("selected_route") or {}
            writer.writerow({
                "example": name,
                "runtime_supported": result["runtime_supported"],
                "runtime_blocked_code": result["runtime_blocked_code"] or "",
                "strategy_label": result["strategy_label"],
                "strategy_benchmark_supported": result["strategy_benchmark_supported"],
                "selected_route_id": selected.get("route_id", ""),
                "selected_graph_path": json.dumps(selected.get("graph_path", [])),
                "selected_hop_count": selected.get("hop_count", ""),
                "selected_total_length_m": selected.get("total_length_m", ""),
                "graph_objective_supported": result["graph_objective_supported"],
            })


__all__ = [
    "ALICE_ENDPOINT",
    "BOB_ENDPOINT",
    "REPEATER",
    "FIBER_LINK",
    "BLOCKED_ALICE_BOB_NOT_SELECTED",
    "BLOCKED_ALICE_BOB_NOT_DISTINCT",
    "BLOCKED_NO_ALICE_BOB_PATH",
    "BLOCKED_EDGE_LENGTH_MISSING",
    "BLOCKED_SEQUENCE_TOPOLOGY_BUILD_FAILED",
    "BLOCKED_SEQUENCE_ROUTE_REQUEST_FAILED",
    "STRATEGY_VALID_MULTI_ROUTE",
    "STRATEGY_DEGENERATE_SINGLE_ROUTE",
    "STRATEGY_DEGENERATE_BOTTLENECK",
    "STRATEGY_NO_ROUTE",
    "STRATEGY_UNCLASSIFIED",
    "GraphNodeRole",
    "GraphEdgeRole",
    "GraphToSequenceNodeMapping",
    "GraphToSequenceEdgeMapping",
    "GraphToSequenceRouteMapping",
    "GraphToSequenceCompileResult",
    "GraphRuntimeValidity",
    "GraphStrategyValidity",
    "compile_graph_to_sequence_runtime",
    "compile_gui_graph_to_sequence_runtime",
    "graph_to_sequence_mapping_contract",
    "mapping_example_payloads",
    "run_sequence_on_compiled_route",
    "write_graph_to_sequence_mapping_artifacts",
]
