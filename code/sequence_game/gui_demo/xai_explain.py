"""Build XAI PromptContext from live GUI step data.

Uses build_policy_explanation() to produce the interpretable policy
representation.  For tabular Q-learning this is the actual Q-table row
for the current state.  For future non-interpretable policies (deep Q,
PPO) it would fit a surrogate DT instead — the prompt builder doesn't care.
"""

from __future__ import annotations

from typing import Any, Optional

from ..xai.policy_explainer import build_policy_explanation
from ..xai.prompt_builder import CounterfactualRow, PromptContext


def _bottleneck_scores(routes: list[dict[str, Any]]) -> dict[str, float]:
    total = len(routes) or 1
    scores: dict[str, float] = {}
    for route in routes:
        for node in (route.get("path") or [])[1:-1]:
            scores[node] = scores.get(node, 0.0) + 1.0 / total
    return scores


def build_context_from_step(
    step: dict[str, Any],
    prev_step: Optional[dict[str, Any]],
    routes: list[dict[str, Any]],
    run_data: Optional[dict[str, Any]] = None,
) -> PromptContext:
    """Return a PromptContext from one GUI replay step.

    run_data is passed to retrieve the full Q-table and policy name.
    If absent, falls back to unknown-policy mode.
    """
    eve_action = step.get("eve_action") or {}
    kind   = eve_action.get("kind") or "no_attack"
    target = eve_action.get("target") or ""

    prev_outcome = (prev_step or {}).get("public_outcome", "accepted") if prev_step else "accepted"
    prev_action_raw = ((prev_step or {}).get("eve_action") or {}) if prev_step else {}
    prev_kind   = prev_action_raw.get("kind") or "no_attack"
    prev_target = prev_action_raw.get("target") or ""
    prev_action = f"{prev_kind}_{prev_target}" if prev_target else prev_kind

    # Retrieve Q-table and action names from run_data
    eve_qt = ((run_data or {}).get("summary") or {}).get("q_tables", {}).get("eve") or {}
    q_table_raw  = eve_qt.get("raw")
    action_names = eve_qt.get("action_names") or []
    policy_name  = step.get("eve_policy") or (run_data or {}).get("eve_policy") or "q_learning"

    policy_exp = build_policy_explanation(
        policy_name=policy_name,
        q_table_raw=q_table_raw,
        action_names=action_names,
        prev_action=prev_action,
        prev_outcome=prev_outcome,
    )

    # Override predicted_action with the actual action taken this step
    # (policy_explainer infers from argmax; the GUI knows the real choice)
    actual_action = f"{kind}_{target}" if target else kind
    policy_exp.predicted_action = actual_action

    qm = step.get("quality_metrics") or {}
    outcome = step.get("public_outcome") or "unknown"
    current_route_id = step.get("alice_route_id") or ""

    chosen_row = CounterfactualRow(
        action=actual_action,
        route=current_route_id,
        outcome=outcome,
        fidelity=qm.get("fidelity"),
        qber=qm.get("qber"),
        quality_metric=None,
        is_chosen=True,
    )

    sorted_routes = sorted(routes or [], key=lambda r: r.get("route_id") != current_route_id)
    r0 = sorted_routes[0] if sorted_routes else None
    r1 = sorted_routes[1] if len(sorted_routes) > 1 else None

    bottleneck = _bottleneck_scores(routes or [])

    def _btn(r: Optional[dict]) -> float:
        if not r:
            return 0.5
        return max((bottleneck.get(n, 0.0) for n in (r.get("path") or [])[1:-1]), default=0.5)

    def _hops(r: Optional[dict]) -> int:
        return int(r.get("hop_count") or 1) if r else 1

    alt_rows: list[CounterfactualRow] = []
    if r1:
        alt_path = r1.get("path") or []
        alt_internals = alt_path[1:-1]
        alt_target = alt_internals[0] if alt_internals else (r1.get("route_id") or "alt")
        alt_rows.append(CounterfactualRow(
            action=f"swap_denial_{alt_target}",
            route=r1.get("route_id") or "alt",
            outcome="counterfactual — not simulated",
            fidelity=None, qber=None, quality_metric=None,
        ))

    return PromptContext(
        policy_explanation=policy_exp,
        chosen_row=chosen_row,
        alternative_rows=alt_rows,
        r0_bottleneck=_btn(r0),
        r1_bottleneck=_btn(r1),
        r0_hops=_hops(r0),
        r1_hops=_hops(r1),
    )
