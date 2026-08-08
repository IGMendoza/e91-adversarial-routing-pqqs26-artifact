"""Linear-route construction, contracts, and Eve move expansion."""

from __future__ import annotations

import copy
from typing import Any

from .models import EveMove
def make_route(hops: int, lengths: list[float]) -> dict[str, Any]:
    if hops < 1:
        raise ValueError("hops must be >= 1")
    if len(lengths) != hops:
        raise ValueError("length count must equal hops")
    nodes = ["alice"] + [f"r{i}" for i in range(1, hops)] + ["bob"]
    edge_ids = [f"e{i}" for i in range(hops)]
    return {
        "hops": hops,
        "repeaters": max(0, hops - 1),
        "nodes": nodes,
        "edge_ids": edge_ids,
        "edge_endpoints": {
            edge_id: [left, right]
            for edge_id, left, right in zip(edge_ids, nodes, nodes[1:])
        },
        "edge_lengths_m": {edge_id: float(length) for edge_id, length in zip(edge_ids, lengths)},
    }


def default_fiber_length_symbols(hops: int) -> list[str]:
    return [f"F{i + 1}" for i in range(hops)]


def fiber_length_symbols_for_route(route: dict[str, Any]) -> list[str]:
    return list(route.get("fiber_length_symbols") or default_fiber_length_symbols(int(route["hops"])))


def linear_chain_contract(route: dict[str, Any]) -> dict[str, Any]:
    hops = int(route["hops"])
    repeaters = int(route["repeaters"])
    symbols = fiber_length_symbols_for_route(route)
    display_nodes = ["Alice"] + [f"R{i}" for i in range(1, repeaters + 1)] + ["Bob"]
    if repeaters == 0:
        route_form = f"Alice --{symbols[0]}-- Bob"
    else:
        parts: list[str] = ["Alice"]
        for idx, symbol in enumerate(symbols):
            right = f"R{idx + 1}" if idx < repeaters else "Bob"
            parts.extend([f"--{symbol}--", right])
        route_form = " ".join(parts)
    return {
        "repeater_count": repeaters,
        "hop_count": hops,
        "edge_count": len(route["edge_ids"]),
        "nodes": list(route["nodes"]),
        "display_nodes": display_nodes,
        "edge_ids": list(route["edge_ids"]),
        "edge_endpoints": copy.deepcopy(route["edge_endpoints"]),
        "edge_indexing": "edge e{j} is 0-based and corresponds to fiber vector entry F_{j+1}",
        "node_indexing": "repeater node r{i} is 1-based, with 1 <= i <= repeater_count",
        "fiber_length_symbols": symbols,
        "fiber_vector_order": [
            {
                "vector_index": idx,
                "fiber_symbol": symbol,
                "edge_id": edge_id,
                "left_node": route["edge_endpoints"][edge_id][0],
                "right_node": route["edge_endpoints"][edge_id][1],
                "length_m": float(route["edge_lengths_m"][edge_id]),
            }
            for idx, (symbol, edge_id) in enumerate(zip(symbols, route["edge_ids"]))
        ],
        "route_form": route_form,
    }


def eve_move_target_index(move: EveMove) -> int | None:
    if move.target_kind == "edge" and move.target_id.startswith("e"):
        try:
            return int(move.target_id[1:])
        except ValueError:
            return None
    if move.target_kind in {"node", "memory"} and move.target_id.startswith("r"):
        try:
            return int(move.target_id[1:])
        except ValueError:
            return None
    return None


def eve_move_contract(move: EveMove) -> dict[str, Any]:
    move_payload = move.to_dict()
    move_parameters = {
        key: value
        for key, value in move_payload.items()
        if key not in {"move_id", "move_kind", "target_kind", "target_id", "target_active"}
    }
    return {
        "move_id": move.move_id,
        "move_kind": move.move_kind,
        "target_kind": move.target_kind,
        "target_index": eve_move_target_index(move),
        "target_id": move.target_id,
        "target_active": move.target_active,
        "target_indexing": (
            "edge target_index is 0-based from edge id e{j}; repeater node target_index is "
            "1-based from node id r{i}; none/off-route targets use null when no canonical index exists"
        ),
        "move_parameters": move_parameters,
    }


def length_contract(route: dict[str, Any]) -> dict[str, Any]:
    symbols = fiber_length_symbols_for_route(route)
    lengths = [
        float(route["edge_lengths_m"][edge_id])
        for edge_id in route["edge_ids"]
    ]
    return {
        "runtime_argument": "fiber_lengths_m",
        "required_length": int(route["hops"]),
        "fiber_length_symbols": symbols,
        "fiber_vector_order": [
            {
                "vector_index": idx,
                "fiber_symbol": symbol,
                "edge_id": edge_id,
                "length_m": float(route["edge_lengths_m"][edge_id]),
            }
            for idx, (symbol, edge_id) in enumerate(zip(symbols, route["edge_ids"]))
        ],
        "length_parametric_status": "in_process_evaluator",
        "rows_probability_status": "numeric_rows_evaluated_at_generation_lengths",
        "rows_evaluated_at_lengths_m": lengths,
        "substitution_method": (
            "FrozenHistogramKernel.evaluate(fiber_lengths_m) rebuilds the selected linear-chain route "
            "with the same frozen theta, hop count, Eve move, protocol settings, and model flags; the "
            "emitted rows themselves are numeric values at rows_evaluated_at_lengths_m"
        ),
        "symbolic_status": "not_symbolic",
        "generated_evaluator_module": "histogram_numeric_evaluator.py is emitted only when --emit-numeric-evaluator is passed",
    }


def moves_for_route(route: dict[str, Any], move_set: str, memory_side: str) -> list[EveMove]:
    edge_ids = list(route["edge_ids"])
    repeaters = list(route["nodes"][1:-1])
    moves = [
        EveMove(
            move_id="no_move_baseline",
            move_kind="none",
            target_active=False,
            histogram_source="no_move_baseline",
        )
    ]
    if move_set == "none":
        return moves
    if move_set in {"all", "edge", "delivery"}:
        for edge_id in edge_ids:
            moves.append(EveMove(
                move_id=f"edge_added_loss:{edge_id}",
                move_kind="edge_added_loss",
                target_kind="edge",
                target_id=edge_id,
                target_active=True,
                loss_multiplier=0.0,
                drop_probability=1.0,
            ))
    if move_set in {"all", "edge", "projective"}:
        for edge_id in edge_ids:
            moves.append(EveMove(
                move_id=f"edge_measure_resend:{edge_id}",
                move_kind="edge_measure_resend",
                target_kind="edge",
                target_id=edge_id,
                target_active=True,
                basis_choice="Z",
                edge_measure_basis_label="Z",
            ))
    if move_set in {"all", "repeater", "delivery"}:
        for node_id in repeaters:
            moves.append(EveMove(
                move_id=f"swap_block:{node_id}",
                move_kind="swap_block",
                target_kind="node",
                target_id=node_id,
                target_active=True,
            ))
    if move_set in {"all", "repeater"}:
        for node_id in repeaters:
            moves.append(EveMove(
                move_id=f"memory_degradation:{node_id}:{memory_side}",
                move_kind="memory_degradation",
                target_kind="node",
                target_id=node_id,
                target_active=True,
                memory_side=memory_side,
                target_memory_side=memory_side,
                memory_depolarizing_probability=0.5,
                memory_dephasing_probability=0.0,
            ))
            moves.append(EveMove(
                move_id=f"repeater_memory_dephasing_probe:{node_id}:{memory_side}",
                move_kind="repeater_memory_dephasing_probe",
                target_kind="node",
                target_id=node_id,
                target_active=True,
                memory_side=memory_side,
                target_memory_side=memory_side,
            ))
    if move_set in {"all", "repeater", "projective"}:
        for node_id in repeaters:
            for basis in ("Z", "X"):
                moves.append(EveMove(
                    move_id=f"repeater_memory_measure_{basis}:{node_id}:{memory_side}",
                    move_kind=f"repeater_memory_measure_{basis}",
                    target_kind="node",
                    target_id=node_id,
                    target_active=True,
                    memory_side=memory_side,
                    target_memory_side=memory_side,
                ))
    return moves


attacks_for_route = moves_for_route

