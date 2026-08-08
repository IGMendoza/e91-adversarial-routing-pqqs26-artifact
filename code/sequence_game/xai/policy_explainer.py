"""Policy explainer — produces a PolicyExplanation for the prompt builder.

Dispatch logic:
  interpretable policy  → use the model directly (Q-table rows)
  non-interpretable     → fit surrogate DT, extract decision path

Both branches produce a PolicyExplanation with the same interface.
The prompt builder and semantic checker are completely policy-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .policy_registry import explanation_type, is_interpretable

# "hit"  = binary_collision: Eve's attack hit Alice's route → delivery failed for Alice
# "miss" = binary_collision: Eve missed → Alice delivered successfully (NOT a failure)
_FAILURE_OUTCOMES = {"delivery_failure", "hit", "blocked", "qber_abort"}


@dataclass
class PolicyExplanation:
    text: str                        # formatted block for the prompt
    method: str                      # "q_table" | "random" | "oracle" | "surrogate_dt"
    predicted_action: str
    prev_outcome_is_failure: bool    # drives semantic checker
    dt_fidelity: Optional[float] = None  # None for direct; float for surrogate DT


# ── Q-table direct explanation ────────────────────────────────────────────────

def _explain_q_table(
    q_table_raw: dict[str, Any],
    action_names: list[str],
    prev_action: str,
    prev_outcome: str,
) -> str:
    """Format the Q-table row for the current state."""
    state_key = (prev_action, prev_outcome)
    entries = {tuple(e["state"]): e["q"] for e in q_table_raw.get("entries", [])}
    q_vals = entries.get(state_key, None)

    is_failure = prev_outcome in _FAILURE_OUTCOMES
    state_desc = (
        "delivery_failure (key delivery failed)"
        if is_failure
        else "accepted (key delivery succeeded)"
    )

    lines = [
        f"Current state: (prev_action={prev_action!r}, prev_outcome={prev_outcome!r})",
        f"Previous outcome: {state_desc}",
        "",
    ]

    if q_vals is None:
        lines.append(
            "Q-values: state not yet visited — all actions have Q=0.0 (initial value)."
        )
        greedy = action_names[0] if action_names else "unknown"
    else:
        num_actions = len(q_vals)
        best_idx = max(range(num_actions), key=lambda i: (q_vals[i], -i))
        greedy = action_names[best_idx] if best_idx < len(action_names) else f"action_{best_idx}"
        lines.append("Q-values for each action (higher = Eve prefers this action):")
        for i, (name, val) in enumerate(zip(action_names, q_vals)):
            marker = "  ← greedy choice" if i == best_idx else ""
            lines.append(f"  {name:<35} {val:>7.4f}{marker}")
        lines.append("")
        lines.append(f"Greedy action (argmax Q): {greedy}")

    return "\n".join(lines)


# ── Surrogate DT explanation ──────────────────────────────────────────────────

def _explain_via_surrogate_dt(
    q_table_raw: dict[str, Any],
    action_names: list[str],
    prev_action: str,
    prev_outcome: str,
    r0_features: Any,
    r1_features: Any,
) -> tuple[str, str, Optional[float]]:
    """Fit surrogate DT on Q-table entries, extract path for current state.

    Returns (path_text, predicted_action, fidelity).
    Raises NotImplementedError if q_table_raw has fewer than 2 entries
    (DT needs at least 2 samples).
    """
    from .surrogate import (
        RouteFeatures, build_dataset, encode_state,
        extract_decision_path, fit_surrogate_dt, format_path_text,
    )
    import numpy as np

    entries = q_table_raw.get("entries", [])
    if len(entries) < 2:
        raise NotImplementedError(
            "TODO(scientific): surrogate DT requires ≥2 Q-table entries; "
            f"only {len(entries)} available."
        )

    ds = build_dataset(entries, action_names, r0_features, r1_features)
    dt = fit_surrogate_dt(ds, max_depth=4)
    query = np.array([encode_state(prev_action, prev_outcome) +
                      [r0_features.bottleneck_score, r1_features.bottleneck_score,
                       float(r0_features.hops), float(r1_features.hops)]])
    path = extract_decision_path(dt, ds, query)
    return format_path_text(path), path.predicted_action, path.fidelity


# ── Public entry point ────────────────────────────────────────────────────────

def build_policy_explanation(
    policy_name: str,
    q_table_raw: Optional[dict[str, Any]],
    action_names: list[str],
    prev_action: str,
    prev_outcome: str,
    r0_features: Any = None,
    r1_features: Any = None,
) -> PolicyExplanation:
    """Return a PolicyExplanation for the given policy and state.

    Args:
        policy_name:   key into POLICY_REGISTRY (e.g. "q_learning")
        q_table_raw:   QTable.to_dict() output stored in run_data, or None
        action_names:  ordered list of action name strings
        prev_action:   Eve's previous action
        prev_outcome:  public outcome of the previous step
        r0_features:   RouteFeatures for route 0 (needed for surrogate DT)
        r1_features:   RouteFeatures for route 1 (needed for surrogate DT)
    """
    is_failure = prev_outcome in _FAILURE_OUTCOMES
    etype = explanation_type(policy_name)

    if etype == "random":
        text = (
            f"Policy: uniform random — Eve samples an action without learned structure.\n"
            f"Previous outcome: {'delivery_failure' if is_failure else 'accepted'}."
        )
        return PolicyExplanation(
            text=text, method="random",
            predicted_action=prev_action,
            prev_outcome_is_failure=is_failure,
        )

    if etype == "oracle":
        text = (
            "Policy: route-aware oracle — Eve deterministically targets Alice's\n"
            "current route (privileged observation; not a learned policy).\n"
            f"Previous outcome: {'delivery_failure' if is_failure else 'accepted'}."
        )
        return PolicyExplanation(
            text=text, method="oracle",
            predicted_action=prev_action,
            prev_outcome_is_failure=is_failure,
        )

    if etype == "q_table" and q_table_raw is not None:
        text = _explain_q_table(q_table_raw, action_names, prev_action, prev_outcome)
        # Infer predicted action from Q-table argmax
        entries = {tuple(e["state"]): e["q"] for e in q_table_raw.get("entries", [])}
        q_vals = entries.get((prev_action, prev_outcome), None)
        if q_vals:
            best_idx = max(range(len(q_vals)), key=lambda i: (q_vals[i], -i))
            predicted = action_names[best_idx] if best_idx < len(action_names) else prev_action
        else:
            predicted = prev_action
        return PolicyExplanation(
            text=text, method="q_table",
            predicted_action=predicted,
            prev_outcome_is_failure=is_failure,
        )

    # Fallback: surrogate DT (for non-interpretable policies or missing Q-table)
    if q_table_raw is not None and r0_features is not None and r1_features is not None:
        try:
            text, predicted, fidelity = _explain_via_surrogate_dt(
                q_table_raw, action_names, prev_action, prev_outcome,
                r0_features, r1_features,
            )
            return PolicyExplanation(
                text=text, method="surrogate_dt",
                predicted_action=predicted,
                prev_outcome_is_failure=is_failure,
                dt_fidelity=fidelity,
            )
        except NotImplementedError as exc:
            text = f"[surrogate DT unavailable: {exc}]"
            return PolicyExplanation(
                text=text, method="surrogate_dt_unavailable",
                predicted_action=prev_action,
                prev_outcome_is_failure=is_failure,
            )

    # Last resort: no data at all
    state_desc = "delivery_failure" if is_failure else "accepted"
    text = (
        f"Policy: {policy_name} (no Q-table available in run data).\n"
        f"Previous outcome: {state_desc}."
    )
    return PolicyExplanation(
        text=text, method="unknown",
        predicted_action=prev_action,
        prev_outcome_is_failure=is_failure,
    )
