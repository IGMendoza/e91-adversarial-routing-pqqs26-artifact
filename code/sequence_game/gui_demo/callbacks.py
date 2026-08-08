"""Dash callbacks for the GUI demo."""

from __future__ import annotations

import json
from typing import Any

from dash import Input, Output, State, callback_context, dcc, html, no_update

from ..xai.faithfulness import score_faithfulness
from ..xai.global_prompt import build_global_prompt, context_from_steps
from ..xai.prompt_builder import build_mitigation_prompt, build_prompt
from ..xai.semantic import check_mitigation_semantic_faithfulness, check_semantic_faithfulness
from .streaming import clear_stream, get_stream_state, set_cache_meta, start_stream
from .xai_explain import build_context_from_step

from .demo_runner import (
    ACTION_SURFACE_MODELS,
    DemoInputError,
    REPEATER_MOTIF_BLOCK,
    TOY_SCOPE,
    candidate_routes,
    generate_topology,
    get_run_progress,
    run_binary_collision,
    run_repeater_demo,
)
from .graph_view import graph_elements


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _route_text(path: Any) -> str:
    if isinstance(path, str):
        return path
    if not path:
        return "-"
    return " ".join(str(node) for node in path)


def _action_label(action: dict[str, Any] | None) -> tuple[str, str]:
    action = action or {}
    kind = action.get("kind") or ""
    target = action.get("target") or ""
    attack_type = action.get("attack_type") or ""
    raw = action.get("action") or ""
    if raw == "no_attack" or attack_type == "no_attack" or kind == "none":
        return "no_attack", "-"
    if not kind and ":" in raw:
        attack_type, target = raw.split(":", 1)
    return attack_type or kind or raw or "-", target or "-"


def _action_rows(rows: list[dict[str, Any]]) -> list[list[Any]]:
    formatted = []
    for row in rows:
        kind, target = _action_label(row)
        formatted.append([kind, target, row.get("q", "-")])
    return formatted


def _table(headers: list[str], rows: list[list[Any]]) -> html.Table:
    if not rows:
        return html.P("No rows to display.", className="muted")
    return html.Table([
        html.Thead(html.Tr([html.Th(header) for header in headers])),
        html.Tbody([
            html.Tr([html.Td(_format_value(cell)) for cell in row])
            for row in rows
        ]),
    ], className="data-table")


def _kv_table(items: list[tuple[str, Any]]) -> html.Table:
    return _table(["field", "value"], [[key, value] for key, value in items])


def _triggered() -> str:
    if not callback_context.triggered:
        return ""
    return callback_context.triggered[0]["prop_id"].split(".")[0]


def _route_rows(routes: list[dict[str, Any]]) -> list[list[Any]]:
    return [
        [
            rank,
            route["route_id"],
            _route_text(route["path"]),
            route["hop_count"],
            route["total_length"],
            "eligible" if len(route["path"]) >= 3 else "blocked: no repeater",
        ]
        for rank, route in enumerate(routes)
    ]


def _choice_rows(final_window: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    for record in final_window:
        kind, target = _action_label(record.get("eve_action"))
        rows.append([
            record.get("step"),
            _route_text(record.get("alice_path")),
            kind,
            target,
            record.get("public_outcome"),
        ])
    return rows


def _eve_telemetry(run_data: dict[str, Any] | None) -> list[Any]:
    final_window = (run_data or {}).get("final_window") or []
    summary = (run_data or {}).get("summary", {})
    q_tables = summary.get("q_tables", {})
    eve = q_tables.get("eve", {})
    policy = final_window[-1].get("eve_policy") if final_window else "-"
    children: list[Any] = [
        html.H3("Eve telemetry", className="telemetry-title"),
        html.Div("attack type / target label pairs", className="telemetry-subtitle"),
        _kv_table([
            ("policy", policy),
            ("enabled bundle", eve.get("enabled_attack_kinds") or "-"),
            ("private route observed", eve.get("private_route_observed")),
        ]),
    ]
    if policy == "q_learning" and eve.get("top_actions"):
        children.extend([
            html.H5("Q-tabular attack ranking", className="mini-heading"),
            _table(["attack type", "node/edge label", "Q"], _action_rows(eve.get("top_actions", []))),
        ])
    else:
        heading = "Final random sequence" if policy == "random" else "Final revealed sequence"
        children.extend([
            html.H5(heading, className="mini-heading"),
            _table(
                ["step", "Alice route", "attack type", "target", "outcome"],
                _choice_rows(final_window),
            ),
        ])
    return children


def _alice_telemetry(run_data: dict[str, Any] | None) -> list[Any]:
    final_window = (run_data or {}).get("final_window") or []
    summary = (run_data or {}).get("summary", {})
    q_tables = summary.get("q_tables", {})
    alice = q_tables.get("alice", {})
    policy = final_window[-1].get("alice_policy") if final_window else "-"
    children: list[Any] = [
        html.H3("Alice telemetry", className="telemetry-title"),
        html.Div("route choices shown as node sequences", className="telemetry-subtitle"),
        _kv_table([
            ("policy", policy),
            ("observation", alice.get("observation_state") or "-"),
        ]),
    ]
    if policy == "q_learning" and alice.get("top_actions"):
        rows = [
            [_route_text(row.get("target")), row.get("q", "-")]
            for row in alice.get("top_actions", [])
        ]
        children.extend([
            html.H5("Q-tabular route ranking", className="mini-heading"),
            _table(["route", "Q"], rows),
        ])
    else:
        children.extend([
            html.H5("Parallel secret choices / public reveal", className="mini-heading"),
            _table(
                ["step", "Alice route", "Eve attack", "target", "outcome"],
                _choice_rows(final_window),
            ),
        ])
    return children


def _run_status(run_data: dict[str, Any] | None) -> html.Div:
    if not run_data:
        return html.Div("No run yet. Generate topology → select Alice/Bob → press Run.",
                        className="run-status")
    if run_data.get("error"):
        return html.Div(run_data["error"], className="run-status status-error")
    if run_data.get("blocked"):
        return html.Div(run_data.get("blocked_reason", "blocked"), className="run-status status-error")
    summary = run_data.get("summary", {})
    metrics = summary.get("metrics", {})
    mode = summary.get("mode", run_data.get("scope_label", ""))
    if mode in ACTION_SURFACE_MODELS:
        duration = metrics.get("training_runtime_ms")
        steps = metrics.get("training_iterations", summary.get("num_steps"))
        eve_algo   = metrics.get("eve_algo", "?")
        alice_algo = metrics.get("alice_algo", "?")
        fw = run_data.get("final_window") or []
        hits = sum(1 for s in fw if s.get("collision") or s.get("public_outcome") == "delivery_failure")
        hit_rate = f"{hits}/{len(fw)}" if fw else "?"
        return html.Div([
            html.Span("✓ Run complete", style={"fontWeight": "700", "color": "var(--accent)"}),
            html.Span(f"  {eve_algo} Eve vs {alice_algo} Alice  ·  {steps} steps  ·  {_format_value(duration)} ms",
                      style={"marginLeft": "8px"}),
            html.Br(),
            html.Span(f"Replay window: {len(fw)} steps  ·  Eve disruptions in window: {hit_rate}  ·  "
                      "← use play/next/slider to step through",
                      style={"color": "var(--muted)", "fontSize": "11px"}),
        ], className="run-status status-ok")
    duration = metrics.get("runner_runtime_ms")
    return html.Div(
        f"✓ Repeater run complete in {_format_value(duration)} ms. "
        "Replay shows the simulated trial window.",
        className="run-status status-ok",
    )


def _scores_span(faith_score: float, faith_hallucinated: list[str],
                 sem_passed: bool) -> html.Span:
    faith_color = "var(--accent)" if faith_score >= 0.9 else "var(--amber)"
    sem_color = "var(--accent)" if sem_passed else "var(--red)"
    return html.Span([
        html.Span(f"numeric {faith_score:.0%}",
                  style={"color": faith_color, "marginRight": "10px"}),
        html.Span(f"semantic {'PASS' if sem_passed else 'FAIL'}",
                  style={"color": sem_color}),
    ])


def _scores_span_cached(cache_entry: dict[str, Any]) -> html.Span:
    faith_score = cache_entry.get("faith_score", 0)
    sem_passed = cache_entry.get("sem_passed", True)
    faith_color = "var(--accent)" if faith_score >= 0.9 else "var(--amber)"
    sem_color = "var(--accent)" if sem_passed else "var(--red)"
    return html.Span([
        html.Span(f"numeric {faith_score:.0%}",
                  style={"color": faith_color, "marginRight": "10px"}),
        html.Span(f"semantic {'PASS' if sem_passed else 'FAIL'} (cached)",
                  style={"color": sem_color}),
    ])


def _strategy_header_from_meta(header_items: list[dict] | None) -> list[Any]:
    """Re-build Dash header components from the JSON-serializable cache metadata."""
    if not header_items:
        return []
    h = header_items[0]
    return [html.Div([
        html.Span("Strategy summary  ", style={"fontWeight": "700", "color": "var(--muted)"}),
        html.Span(
            f"{h.get('num_steps', 0)} steps · DoS {h.get('dos_fraction', 0):.0%} · "
            f"quality {h.get('quality_fraction', 0):.0%} · "
            f"failure rate {h.get('delivery_failure_rate', 0):.0%}",
            style={"fontSize": "11px", "color": "var(--muted)"},
        ),
    ], style={"marginBottom": "6px"})]


def _render_explanation(text: str, sem_violations: list[Any],
                        hallucinated: list[str],
                        header: list[Any] | None = None) -> list[Any]:
    output: list[Any] = list(header or [])
    output.append(dcc.Markdown(text, className="xai-output-body"))
    if sem_violations:
        output.append(html.Details([
            html.Summary("semantic violations",
                         style={"color": "var(--red)", "cursor": "pointer", "fontSize": "11px"}),
            html.Ul([
                html.Li(f"[{v['rule']}] \"{v['matched_text']}\"",
                        style={"fontSize": "11px", "color": "var(--muted)"})
                for v in sem_violations
            ]),
        ]))
    if hallucinated:
        output.append(html.Details([
            html.Summary("hallucinated numbers",
                         style={"color": "var(--amber)", "cursor": "pointer", "fontSize": "11px"}),
            html.P(", ".join(hallucinated),
                   style={"fontSize": "11px", "color": "var(--muted)"}),
        ]))
    return output


def register_callbacks(app):
    @app.callback(
        Output("topology-store", "data"),
        Output("selection-store", "data"),
        Output("routes-store", "data"),
        Output("run-store", "data"),
        Input("generate-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        Input("num-nodes-input", "value"),
        Input("edge-prob-input", "value"),
        Input("seed-input", "value"),
        Input("edge-length-mode", "value"),
        prevent_initial_call=True,
    )
    def generate_or_reset(_generate_clicks, _reset_clicks, num_nodes, edge_prob, seed, edge_mode):
        if _triggered() == "reset-btn":
            return None, {"alice": None, "bob": None}, [], None
        num_nodes = max(2, min(16, int(num_nodes if num_nodes is not None else 8)))
        edge_prob = max(0.0, min(1.0, float(edge_prob if edge_prob is not None else 0.35)))
        seed = max(0, int(seed if seed is not None else 1))
        topology = generate_topology(
            num_nodes,
            edge_prob,
            seed,
            str(edge_mode or "unit"),
        )
        return topology, {"alice": None, "bob": None}, [], None

    @app.callback(
        Output("selection-store", "data"),
        Input("graph-view", "tapNodeData"),
        State("selection-store", "data"),
        prevent_initial_call=True,
    )
    def select_node(node_data, selection):
        if not node_data:
            return no_update
        node_id = node_data["id"]
        selection = dict(selection or {})
        alice, bob = selection.get("alice"), selection.get("bob")
        if alice is None:
            return {"alice": node_id, "bob": None}
        if bob is None and node_id != alice:
            return {"alice": alice, "bob": node_id}
        return {"alice": node_id, "bob": None}

    @app.callback(
        Output("routes-store", "data"),
        Output("selected-route-dropdown", "options"),
        Output("selected-route-dropdown", "value"),
        Output("warning-box", "children"),
        Input("topology-store", "data"),
        Input("selection-store", "data"),
        Input("mode-select", "value"),
    )
    def update_routes(topology, selection, mode):
        if not topology:
            return [], [], None, "Generate a topology to begin."
        warnings = list(topology.get("warnings", []))
        alice = (selection or {}).get("alice")
        bob = (selection or {}).get("bob")
        if not alice or not bob:
            warnings.append("Select Alice and Bob by clicking two graph nodes.")
            return [], [], None, html.Ul([html.Li(w) for w in warnings])
        try:
            routes = candidate_routes(topology, alice, bob, max_routes=12)
        except DemoInputError as exc:
            routes = []
            warnings.append(str(exc))
        if not routes:
            warnings.append("No candidate route between selected Alice/Bob nodes.")
        options = []
        for route in routes:
            eligible = len(route["path"]) >= 3
            suffix = "selected repeater path" if eligible else "blocked: no repeater"
            options.append({
                "label": f"{' -> '.join(route['path'])} ({route['total_length']:.1f}) - {suffix}",
                "value": route["route_id"],
                "disabled": not eligible,
            })
        value = next((option["value"] for option in options if not option.get("disabled")), None)
        if mode == "repeater_availability_quality":
            if value is None:
                warnings.append("No selected candidate route has an intermediate repeater.")
            warnings.append(REPEATER_MOTIF_BLOCK)
        return routes, options, value, html.Ul([html.Li(w) for w in warnings])

    @app.callback(
        Output("repeater-controls", "style"),
        Output("repeater-controls", "className"),
        Input("mode-select", "value"),
        Input("selected-route-dropdown", "value"),
    )
    def toggle_repeater_controls(mode, selected_route_id):
        if mode == "repeater_availability_quality":
            class_name = "panel-card repeater-controls"
            if not selected_route_id:
                class_name += " attention-card"
            return {"display": "block"}, class_name
        return {"display": "none"}, "panel-card repeater-controls"

    @app.callback(
        Output("run-preview", "children"),
        Input("eve-algo-select", "value"),
        Input("alice-algo-select", "value"),
        Input("steps-select", "value"),
        Input("mode-select", "value"),
    )
    def update_run_preview(eve_algo, alice_algo, steps, mode):
        eve   = eve_algo   or "q_learning"
        alice = alice_algo or "cautious_greedy"
        n     = steps      or 3000
        m     = mode       or "binary_collision"
        return f"Ready → {eve} Eve  ×  {alice} Alice  ·  {n} steps  [{m}]"

    @app.callback(
        Output("attack-bundle-store", "data"),
        Output("attack-bundle-card", "className"),
        Output("attack-bundle-status", "children"),
        Input("mode-select", "value"),
        Input("eve-algo-select", "value"),
        Input("alice-algo-select", "value"),
        Input("topology-store", "data"),
        Input("selection-store", "data"),
    )
    def update_attack_bundle(mode, eve_algo, alice_algo, topology, selection):
        mode_id = str(mode or "binary_collision")
        if mode_id == "repeater_availability_quality":
            selected: list[str] = []
            card_class = "panel-card"
            status = html.Div("Repeater mode uses the repeater attack dropdown.")
            return selected, card_class, status

        model = ACTION_SURFACE_MODELS.get(
            mode_id,
            ACTION_SURFACE_MODELS["binary_collision"],
        )
        selected = list(model["attack_kinds"])
        alice = (selection or {}).get("alice")
        bob = (selection or {}).get("bob")
        non_endpoint_nodes = [
            node["id"] for node in (topology or {}).get("nodes", [])
            if node["id"] not in {alice, bob}
        ]
        edge_count = len((topology or {}).get("edges", []))
        has_actions = bool(selected)
        needs_attention = mode_id in ACTION_SURFACE_MODELS and not has_actions
        card_class = "panel-card attention-card" if needs_attention else "panel-card"

        enabled = ", ".join(selected) if selected else "none selected"
        status = html.Div([
            html.Div(
                f"Model: {model['label']}. "
                f"Eve: {eve_algo or 'q_learning'} | Alice: {alice_algo or 'cautious_greedy'}."
            ),
            html.Div(
                f"Reward model: {model['reward_model']}. Enabled attack types: {enabled}."
            ),
            html.Div(
                f"Expansion: node actions → {len(non_endpoint_nodes)} non-endpoint nodes; "
                f"edge actions → {edge_count} edges."
            ),
        ])
        return selected, card_class, status

    @app.callback(
        Output("run-store", "data"),
        Output("run-status", "children"),
        Input("run-btn", "n_clicks"),
        State("topology-store", "data"),
        State("selection-store", "data"),
        State("routes-store", "data"),
        State("selected-route-dropdown", "value"),
        State("mode-select", "value"),
        State("eve-algo-select", "value"),
        State("alice-algo-select", "value"),
        State("steps-select", "value"),
        State("final-window-input", "value"),
        State("avoidance-horizon-input", "value"),
        State("attack-cost-input", "value"),
        State("attack-bundle-store", "data"),
        State("repeater-attack", "value"),
        State("seed-input", "value"),
        prevent_initial_call=True,
    )
    def run_demo(_clicks, topology, selection, routes, selected_route_id, mode,
                 eve_algo, alice_algo, steps, final_window, avoidance, attack_cost,
                 attack_bundle, repeater_attack, seed):
        if not topology:
            err = {"error": "Generate a topology before running."}
            return err, _run_status(err)
        alice = (selection or {}).get("alice")
        bob = (selection or {}).get("bob")
        if not alice or not bob:
            err = {"error": "Alice/Bob must be selected before run."}
            return err, _run_status(err)
        try:
            if mode == "repeater_availability_quality":
                selected = next((r for r in (routes or []) if r["route_id"] == selected_route_id), None)
                result = run_repeater_demo(
                    topology,
                    alice,
                    bob,
                    list(selected["path"]) if selected else [],
                    attack=str(repeater_attack or "repeater_memory_measure_X"),
                    num_trials=max(1, min(100, int(steps or 20))),
                    seed=int(seed or 1),
                    final_window_size=int(final_window or 20),
                )
            else:
                result = run_binary_collision(
                    topology,
                    alice,
                    bob,
                    eve_algo=str(eve_algo or "q_learning"),
                    alice_algo=str(alice_algo or "cautious_greedy"),
                    control_model=str(mode or "binary_collision"),
                    num_steps=int(steps or 3000),
                    seed=int(seed or 1),
                    avoidance_horizon=int(avoidance or 2),
                    attack_cost=float(attack_cost if attack_cost is not None else 0.05),
                    final_window_size=int(final_window or 20),
                    allowed_attack_kinds=(
                        None if attack_bundle is None else list(attack_bundle)
                    ),
                )
        except Exception as exc:
            result = {"error": str(exc)}
        return result, _run_status(result)

    # ── Replay controls ───────────────────────────────────────────────────────

    @app.callback(
        Output("replay-timer", "disabled"),
        Output("replay-toggle", "children"),
        Input("replay-toggle", "n_clicks"),
        Input("run-store", "data"),
        State("replay-timer", "disabled"),
        prevent_initial_call=True,
    )
    def toggle_replay(_clicks, run_data, disabled):
        if _triggered() == "run-store":
            fw = (run_data or {}).get("final_window") or []
            new_disabled = len(fw) == 0
        else:
            new_disabled = not bool(disabled)
        label = "⏸ pause" if not new_disabled else "▶ play"
        return new_disabled, label

    @app.callback(
        Output("replay-index-store", "data"),
        Output("replay-slider", "value"),
        Output("replay-timer", "disabled", allow_duplicate=True),
        Output("replay-toggle", "children", allow_duplicate=True),
        Input("replay-prev", "n_clicks"),
        Input("replay-next", "n_clicks"),
        Input("replay-slider", "value"),
        Input("replay-timer", "n_intervals"),
        Input("run-store", "data"),
        State("replay-index-store", "data"),
        prevent_initial_call=True,
    )
    def update_replay_index(_prev, _next, slider_value, _intervals, run_data, index):
        trigger = _triggered()
        if trigger == "run-store":
            return 0, 0, True, "▶ play"
        final_window = (run_data or {}).get("final_window") or []
        if not final_window:
            return 0, 0, no_update, no_update
        max_index = len(final_window) - 1
        index = int(index or 0)
        if trigger == "replay-prev":
            index = max(0, index - 1)
        elif trigger == "replay-next":
            index = min(max_index, index + 1)
        elif trigger == "replay-slider":
            index = min(max_index, max(0, int(slider_value or 0)))
        elif trigger == "replay-timer":
            if index >= max_index:
                # Stop animation at end
                return max_index, max_index, True, "▶ play"
            index = index + 1
        return index, index, no_update, no_update

    @app.callback(
        Output("graph-view", "elements"),
        Output("scope-label", "children"),
        Input("topology-store", "data"),
        Input("selection-store", "data"),
        Input("run-store", "data"),
        Input("replay-index-store", "data"),
    )
    def update_graph(topology, selection, run_data, replay_index):
        final_window = (run_data or {}).get("final_window") or []
        step = None
        if final_window:
            step = final_window[min(int(replay_index or 0), len(final_window) - 1)]
        selection = selection or {}
        scope = (run_data or {}).get("scope_label") or (topology or {}).get("scope_label") or TOY_SCOPE
        return graph_elements(topology, selection.get("alice"), selection.get("bob"), step), scope

    @app.callback(
        Output("replay-step-label", "children"),
        Input("run-store", "data"),
        Input("replay-index-store", "data"),
    )
    def update_step_label(run_data, replay_index):
        fw = (run_data or {}).get("final_window") or []
        if not fw:
            return "no run yet"
        idx = min(int(replay_index or 0), len(fw) - 1)
        step = fw[idx]
        outcome = step.get("public_outcome") or "-"
        action  = (step.get("eve_action") or {})
        kind    = action.get("attack_type") or action.get("kind") or "no_attack"
        return f"step {idx + 1}/{len(fw)}  ·  Eve: {kind}  ·  outcome: {outcome}"

    @app.callback(
        Output("eve-telemetry", "children"),
        Output("alice-telemetry", "children"),
        Output("topology-summary", "children"),
        Output("routes-summary", "children"),
        Output("step-details", "children"),
        Output("metrics-summary", "children"),
        Output("replay-slider", "max"),
        Input("topology-store", "data"),
        Input("selection-store", "data"),
        Input("routes-store", "data"),
        Input("run-store", "data"),
        Input("replay-index-store", "data"),
    )
    def update_summaries(topology, selection, routes, run_data, replay_index):
        topology_summary = {
            "nodes": len((topology or {}).get("nodes", [])),
            "edges": len((topology or {}).get("edges", [])),
            "warnings": (topology or {}).get("warnings", []),
            "alice": (selection or {}).get("alice"),
            "bob": (selection or {}).get("bob"),
        }
        final_window = (run_data or {}).get("final_window") or []
        step = final_window[min(int(replay_index or 0), len(final_window) - 1)] if final_window else None
        summary = (run_data or {}).get("summary", {})

        edge_rows = [
            [_format_value((edge["u"], edge["v"])), edge.get("length", 1.0)]
            for edge in (topology or {}).get("edges", [])[:12]
        ]
        topology_panel = [
            html.H4("Topology"),
            _kv_table([
                ("scope", (topology or {}).get("scope_label", TOY_SCOPE)),
                ("nodes", topology_summary["nodes"]),
                ("edges", topology_summary["edges"]),
                ("Alice", topology_summary["alice"]),
                ("Bob", topology_summary["bob"]),
                ("warnings", "; ".join(topology_summary["warnings"]) or "-"),
            ]),
            html.H5("Edges", className="mini-heading"),
            _table(["edge", "length"], edge_rows),
        ]

        routes_panel = [
            html.H4("Candidate routes / priority list"),
            _table(
                ["rank", "route id", "path", "hops", "length", "repeater mode"],
                _route_rows(routes or []),
            ),
        ]

        disabled = summary.get("disabled_attacks") or (run_data or {}).get("disabled_attacks")

        if step:
            step_panel = [
                html.H4("Current replay step"),
                _kv_table([
                    ("step", step.get("step")),
                    ("mode", step.get("mode")),
                    ("Alice policy", step.get("alice_policy")),
                    ("Eve policy", step.get("eve_policy")),
                    ("route", _route_text(step.get("alice_path"))),
                    ("Eve action", step.get("eve_action")),
                    ("public outcome", step.get("public_outcome")),
                    ("collision", step.get("collision")),
                    ("Alice reward", step.get("alice_reward")),
                    ("Eve reward", step.get("eve_reward")),
                    ("fallback used", step.get("fallback_used")),
                ]),
                html.H5("Step metrics", className="mini-heading"),
                _kv_table(list((step.get("metrics") or {}).items())),
            ]
        else:
            step_panel = [
                html.H4("Current replay step"),
                html.P("Run a mode to populate replay records.", className="muted"),
            ]

        metrics = summary.get("metrics", {})
        metrics_panel = [
            html.H4("Final metrics"),
            _kv_table([
                ("run id", summary.get("run_id")),
                ("scope warning", summary.get("scope_warning") or summary.get("limitation")
                 or (run_data or {}).get("blocked_reason")),
                *list(metrics.items()),
            ]),
        ]
        if disabled:
            metrics_panel.extend([
                html.H5("Disabled attacks", className="mini-heading"),
                _table(
                    ["attack", "status", "reason"],
                    [[item["label"], "disabled", item["tooltip"]] for item in disabled],
                ),
            ])
        return (
            _eve_telemetry(run_data),
            _alice_telemetry(run_data),
            topology_panel,
            routes_panel,
            step_panel,
            metrics_panel,
            max(0, len(final_window) - 1),
        )

    # ── XAI — Explain step (streaming + cache) ────────────────────────────────

    @app.callback(
        Output("xai-output", "children"),
        Output("xai-scores", "children"),
        Output("xai-context-store", "data"),
        Input("xai-explain-btn", "n_clicks"),
        Input("xai-regen-btn", "n_clicks"),
        State("run-store", "data"),
        State("replay-index-store", "data"),
        State("routes-store", "data"),
        State("xai-model-select", "value"),
        State("xai-cache-store", "data"),
        prevent_initial_call=True,
    )
    def explain_step(expl_clicks, _regen, run_data, replay_index, routes, model, cache):
        trigger = _triggered()
        if trigger not in ("xai-explain-btn", "xai-regen-btn"):
            return no_update, no_update, no_update

        final_window = (run_data or {}).get("final_window") or []
        if not final_window:
            return [dcc.Markdown("No replay data — run a simulation first.", className="muted")], "", no_update

        idx = min(int(replay_index or 0), len(final_window) - 1)
        model = model or "llama3.1:8b"
        cache_key = f"step_{idx}_{model}"

        # Return from cache unless regenerating
        if trigger != "xai-regen-btn" and cache and cache_key in cache:
            cached = cache[cache_key]
            return (
                _render_explanation(
                    cached["text"],
                    cached.get("sem_violations", []),
                    cached.get("faith_hallucinated", []),
                ),
                _scores_span_cached(cached),
                cached.get("xai_ctx"),
            )

        step = final_window[idx]
        prev_step = final_window[idx - 1] if idx > 0 else None
        try:
            ctx = build_context_from_step(step, prev_step, routes or [], run_data)
            prompt = build_prompt(ctx)
            eve_policy = (run_data or {}).get(
                "eve_policy",
                (run_data or {}).get("summary", {}).get("metrics", {}).get("eve_algo", "unknown"),
            )
            xai_ctx_meta = {
                "type": "step",
                "predicted_action": ctx.predicted_action,
                "eve_policy": eve_policy,
                "prompt": prompt,
                "prev_outcome_is_failure": ctx.policy_explanation.prev_outcome_is_failure,
            }
            start_stream(
                model, prompt, cache_key,
                disable_thinking="qwen3" in model,
                extra_meta=xai_ctx_meta,
            )
        except Exception as exc:
            return [html.P(f"Error: {exc}", style={"color": "var(--red)"})], "", no_update

        return [dcc.Markdown("⟳ Generating...", className="muted")], "", no_update

    # ── XAI — Explain strategy (streaming + cache) ────────────────────────────

    @app.callback(
        Output("xai-output", "children", allow_duplicate=True),
        Output("xai-scores", "children", allow_duplicate=True),
        Output("xai-context-store", "data", allow_duplicate=True),
        Input("xai-strategy-btn", "n_clicks"),
        Input("xai-regen-btn", "n_clicks"),
        State("run-store", "data"),
        State("routes-store", "data"),
        State("xai-model-select", "value"),
        State("xai-cache-store", "data"),
        prevent_initial_call=True,
    )
    def explain_strategy(_strat, _regen, run_data, routes, model, cache):
        trigger = _triggered()
        if trigger not in ("xai-strategy-btn", "xai-regen-btn"):
            return no_update, no_update, no_update

        final_window = (run_data or {}).get("final_window") or []
        if not final_window:
            return [dcc.Markdown("No replay data — run a simulation first.", className="muted")], "", no_update

        model = model or "llama3.1:8b"
        cache_key = f"strategy_{model}"

        if trigger != "xai-regen-btn" and cache and cache_key in cache:
            cached = cache[cache_key]
            return (
                _render_explanation(
                    cached["text"],
                    cached.get("sem_violations", []),
                    cached.get("faith_hallucinated", []),
                    header=_strategy_header_from_meta(cached.get("header_items")),
                ),
                _scores_span_cached(cached),
                cached.get("xai_ctx"),
            )

        try:
            run_meta = run_data or {}
            eve_policy = run_meta.get("eve_policy") or run_meta.get("model") or "unknown"
            ctx = context_from_steps(final_window, routes or [], eve_policy=eve_policy, scope="toy")
            prompt = build_global_prompt(ctx)
            dominant_action = max(
                {s.get("eve_action_id", "unknown") for s in final_window},
                key=lambda a: sum(1 for s in final_window if s.get("eve_action_id") == a),
                default="unknown",
            )
            prev_failure = ctx.delivery_failure_rate >= 0.5
            xai_ctx_meta = {
                "type": "strategy",
                "predicted_action": dominant_action,
                "eve_policy": eve_policy,
                "prompt": prompt,
                "prev_outcome_is_failure": prev_failure,
                "strategy_header": {
                    "num_steps": ctx.num_steps,
                    "dos_fraction": ctx.dos_fraction,
                    "quality_fraction": ctx.quality_fraction,
                    "delivery_failure_rate": ctx.delivery_failure_rate,
                },
            }
            start_stream(
                model, prompt, cache_key,
                disable_thinking="qwen3" in model,
                extra_meta=xai_ctx_meta,
            )
        except Exception as exc:
            return [html.P(f"Error: {exc}", style={"color": "var(--red)"})], "", no_update

        return [dcc.Markdown("⟳ Generating...", className="muted")], "", no_update

    # ── Auto-show cached explanation when navigating between steps ────────────

    @app.callback(
        Output("xai-output", "children", allow_duplicate=True),
        Output("xai-scores", "children", allow_duplicate=True),
        Output("xai-context-store", "data", allow_duplicate=True),
        Input("replay-index-store", "data"),
        State("xai-model-select", "value"),
        State("xai-cache-store", "data"),
        State("run-store", "data"),
        prevent_initial_call=True,
    )
    def show_cached_on_step_change(replay_index, model, cache, run_data):
        if not cache or not run_data:
            return no_update, no_update, no_update
        model = model or "llama3.1:8b"
        cache_key = f"step_{int(replay_index or 0)}_{model}"
        if cache_key not in cache:
            return no_update, no_update, no_update
        cached = cache[cache_key]
        return (
            _render_explanation(
                cached["text"],
                cached.get("sem_violations", []),
                cached.get("faith_hallucinated", []),
            ),
            _scores_span_cached(cached),
            cached.get("xai_ctx"),
        )

    # ── Regenerate button enable/disable ──────────────────────────────────────

    @app.callback(
        Output("xai-regen-btn", "disabled"),
        Input("xai-cache-store", "data"),
        Input("replay-index-store", "data"),
        State("xai-model-select", "value"),
    )
    def toggle_regen_btn(cache, replay_index, model):
        if not cache:
            return True
        model = model or "llama3.1:8b"
        step_key = f"step_{int(replay_index or 0)}_{model}"
        strat_key = f"strategy_{model}"
        return step_key not in cache and strat_key not in cache

    # ── Mitigation advice button toggle ───────────────────────────────────────

    @app.callback(
        Output("xai-mitigate-btn", "disabled"),
        Output("xai-mitigate-hint", "children"),
        Input("xai-context-store", "data"),
    )
    def toggle_mitigate(ctx_data):
        if not ctx_data:
            return True, "Run an explanation first."
        action = ctx_data.get("predicted_action", "unknown")
        ctx_type = ctx_data.get("type", "step")
        return False, f"← advice for {ctx_type} (action: {action})"

    # ── Mitigation advice ─────────────────────────────────────────────────────

    @app.callback(
        Output("xai-output", "children", allow_duplicate=True),
        Output("xai-scores", "children", allow_duplicate=True),
        Input("xai-mitigate-btn", "n_clicks"),
        State("xai-context-store", "data"),
        State("xai-model-select", "value"),
        prevent_initial_call=True,
    )
    def get_mitigation(n_clicks, ctx_data, model):
        if not n_clicks or not ctx_data:
            return no_update, no_update
        explanation = ctx_data.get("explanation", "")
        action      = ctx_data.get("predicted_action", "unknown")
        eve_policy  = ctx_data.get("eve_policy", "unknown")
        ctx_type    = ctx_data.get("type", "step")
        prompt = build_mitigation_prompt(explanation, action, eve_policy, ctx_type)
        try:
            from ..xai.llm_client import query_model
            resp = query_model(model or "llama3.1:8b", prompt, temperature=0.1)
        except Exception as exc:
            return [html.P(f"Mitigation error: {exc}", style={"color": "var(--red)"})], ""

        mit_sem = check_mitigation_semantic_faithfulness(resp.response, action)
        faith = score_faithfulness(resp.response, prompt)
        faith_color = "var(--accent)" if faith.score >= 0.9 else "var(--amber)"
        sem_color = "var(--accent)" if mit_sem.passed else "var(--red)"

        scores = html.Span([
            html.Span("mitigate ", style={"color": "var(--amber)", "marginRight": "6px"}),
            html.Span(f"numeric {faith.score:.0%}",
                      style={"color": faith_color, "marginRight": "10px"}),
            html.Span(f"semantic {'PASS' if mit_sem.passed else 'FAIL'}",
                      style={"color": sem_color}),
        ])
        output = [
            html.Div([
                html.Span("Mitigation advice",
                          style={"fontWeight": "700", "color": "var(--amber)"}),
                html.Span(" · toy simulation scope",
                          style={"color": "var(--muted)", "fontSize": "11px"}),
            ], style={"marginBottom": "6px"}),
            dcc.Markdown(resp.response, className="xai-output-body"),
        ]
        if mit_sem.violations:
            output.append(html.Details([
                html.Summary("semantic violations",
                             style={"color": "var(--red)", "cursor": "pointer", "fontSize": "11px"}),
                html.Ul([
                    html.Li(f"[{v.rule}] \"{v.matched_text}\"",
                            style={"fontSize": "11px", "color": "var(--muted)"})
                    for v in mit_sem.violations
                ]),
            ]))
        return output, scores

    # ── Theme — apply via dropdown (clientside, no round-trip) ───────────────

    app.clientside_callback(
        """
        function(theme) {
            var t = theme || 'midnight';
            var apply = t;
            if (t === 'system') {
                apply = (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches)
                        ? 'midnight' : 'light';
            }
            document.documentElement.setAttribute('data-theme', apply);
            return t;
        }
        """,
        Output("theme-store", "data"),
        Input("theme-select", "value"),
    )

    # ── Combined progress ticker: run + XAI streaming ────────────────────────

    @app.callback(
        Output("run-progress-display", "children"),
        Output("xai-output", "children", allow_duplicate=True),
        Output("xai-scores", "children", allow_duplicate=True),
        Output("xai-cache-store", "data"),
        Output("xai-context-store", "data", allow_duplicate=True),
        Input("progress-ticker", "n_intervals"),
        State("xai-cache-store", "data"),
        State("run-store", "data"),
        State("replay-index-store", "data"),
        State("routes-store", "data"),
        prevent_initial_call=True,
    )
    def poll_progress_and_stream(n_intervals, xai_cache, run_data, replay_index, routes):
        # ── Run progress display ──────────────────────────────────────────────
        prog = get_run_progress()
        if prog.get("running"):
            step, total = prog.get("step", 0), prog.get("total", 0)
            pct = f" [{step}/{total}  {int(step / total * 100)}%]" if total else ""
            run_txt = f"⟳ {prog.get('phase', '')}{pct}"
        else:
            run_txt = ""

        # ── XAI streaming ────────────────────────────────────────────────────
        ss = get_stream_state()
        if not ss["active"] and not ss["done"]:
            return run_txt, no_update, no_update, no_update, no_update

        text = ss["text"]
        cache_key = ss["cache_key"]

        if ss["error"]:
            err_out = [html.P(f"Stream error: {ss['error']}", style={"color": "var(--red)"})]
            clear_stream()
            return run_txt, err_out, "", no_update, no_update

        if ss["active"]:
            # Still streaming: show live token accumulation as markdown
            preview = text or "⟳ Generating..."
            return run_txt, [dcc.Markdown(preview, className="xai-output-body")], "", no_update, no_update

        # Stream is done — run faithfulness checks, save to cache
        meta = ss.get("cache_meta") or {}
        clear_stream()

        if not text:
            return run_txt, [dcc.Markdown("(empty response)", className="muted")], "", no_update, no_update

        try:
            faith = score_faithfulness(text, meta.get("prompt", ""))
            sem = check_semantic_faithfulness(
                text,
                meta.get("prev_outcome_is_failure", False),
                meta.get("predicted_action", ""),
            )
        except Exception as exc:
            faith_score, faith_hallucinated, sem_passed, sem_violations = 0.0, [], True, []
        else:
            faith_score = faith.score
            faith_hallucinated = list(faith.hallucinated or [])
            sem_passed = sem.passed
            sem_violations = [{"rule": v.rule, "matched_text": v.matched_text} for v in sem.violations]

        xai_ctx = {
            "type": meta.get("type", "step"),
            "explanation": text,
            "predicted_action": meta.get("predicted_action", "unknown"),
            "eve_policy": meta.get("eve_policy", "unknown"),
        }

        header_items = None
        if meta.get("strategy_header"):
            h = meta["strategy_header"]
            header_items = [{
                "tag": "strategy_header",
                "num_steps": h.get("num_steps", 0),
                "dos_fraction": h.get("dos_fraction", 0),
                "quality_fraction": h.get("quality_fraction", 0),
                "delivery_failure_rate": h.get("delivery_failure_rate", 0),
            }]

        cache_entry = {
            "text": text,
            "faith_score": faith_score,
            "faith_hallucinated": faith_hallucinated,
            "sem_passed": sem_passed,
            "sem_violations": sem_violations,
            "xai_ctx": xai_ctx,
            "header_items": header_items,
        }
        new_cache = dict(xai_cache or {})
        new_cache[cache_key] = cache_entry

        # Build rendered strategy header if present
        dash_header: list[Any] = []
        if header_items:
            h = header_items[0]
            dash_header = [
                html.Div([
                    html.Span("Strategy summary  ",
                              style={"fontWeight": "700", "color": "var(--muted)"}),
                    html.Span(
                        f"{h['num_steps']} steps · DoS {h['dos_fraction']:.0%} · "
                        f"quality {h['quality_fraction']:.0%} · "
                        f"failure rate {h['delivery_failure_rate']:.0%}",
                        style={"fontSize": "11px", "color": "var(--muted)"},
                    ),
                ], style={"marginBottom": "6px"}),
            ]

        output = _render_explanation(text, sem_violations, faith_hallucinated,
                                     header=dash_header)
        scores = _scores_span(faith_score, faith_hallucinated, sem_passed)
        return run_txt, output, scores, new_cache, xai_ctx
