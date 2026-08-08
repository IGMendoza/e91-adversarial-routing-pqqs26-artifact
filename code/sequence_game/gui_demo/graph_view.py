"""Cytoscape graph rendering helpers for the Dash demo."""

from __future__ import annotations

from typing import Any, Optional


def _edge_id(u: str, v: str) -> str:
    return "-".join(sorted((u, v)))


def graph_elements(topology: Optional[dict[str, Any]],
                   alice: Optional[str] = None,
                   bob: Optional[str] = None,
                   current_step: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """Convert GUI topology + replay state into dash-cytoscape elements."""

    if not topology:
        return []
    route_edges: set[str] = set()
    route_nodes: set[str] = set()
    eve_node = None
    eve_edge = None
    if current_step:
        path = current_step.get("alice_path") or []
        route_nodes = set(path)
        route_edges = {_edge_id(u, v) for u, v in zip(path, path[1:])}
        action = current_step.get("eve_action") or {}
        if action.get("kind") in {"node", "swap_denial", "memory_degradation"}:
            eve_node = action.get("target")
        elif action.get("kind") == "edge":
            eve_edge = action.get("target")

    elements: list[dict[str, Any]] = []
    for edge in topology.get("edges", []):
        eid = _edge_id(edge["u"], edge["v"])
        classes = []
        if eid in route_edges:
            classes.append("route-edge")
        if eid == eve_edge:
            classes.append("eve-edge")
        elements.append({
            "data": {
                "id": eid,
                "source": edge["u"],
                "target": edge["v"],
                "label": str(edge.get("length", 1.0)),
            },
            "classes": " ".join(classes),
        })

    for node in topology.get("nodes", []):
        nid = node["id"]
        classes = []
        label = nid
        if nid == alice:
            classes.append("alice")
            label = f"{nid} A"
        if nid == bob:
            classes.append("bob")
            label = f"{nid} B"
        if nid in route_nodes:
            classes.append("route-node")
        if nid == eve_node:
            classes.append("eve-node")
        elements.append({
            "data": {"id": nid, "label": label},
            "position": {"x": float(node.get("x", 0.0)), "y": float(node.get("y", 0.0))},
            "classes": " ".join(classes),
            "selectable": True,
        })
    return elements


CYTOSCAPE_STYLESHEET = [
    {
        "selector": "node",
        "style": {
            "label": "data(label)",
            "background-color": "#7a869a",
            "color": "#edf2f7",
            "font-size": "11px",
            "text-valign": "bottom",
            "text-margin-y": "8px",
            "width": "26px",
            "height": "26px",
            "border-width": 1,
            "border-color": "#44515f",
        },
    },
    {
        "selector": "edge",
        "style": {
            "line-color": "#526173",
            "width": 2,
            "curve-style": "bezier",
            "label": "data(label)",
            "font-size": "9px",
            "color": "#9aa8bb",
        },
    },
    {"selector": ".alice", "style": {"background-color": "#2e9d58", "border-width": 3}},
    {"selector": ".bob", "style": {"background-color": "#2d73d5", "border-width": 3}},
    {"selector": ".route-node", "style": {"border-color": "#f7c766", "border-width": 3}},
    {"selector": ".route-edge", "style": {"line-color": "#f7c766", "width": 5}},
    {
        "selector": ".eve-node",
        "style": {
            "border-color": "#d13b3b",
            "border-width": 5,
            "border-style": "dashed",
        },
    },
    {
        "selector": ".eve-edge",
        "style": {
            "line-color": "#d13b3b",
            "line-style": "dashed",
            "width": 6,
        },
    },
]
