"""Algorithm registry for XAI pipeline.

Each entry declares whether the policy is already interpretable (Q-table,
rule-based) or requires a surrogate (deep network, ensemble).

The XAI pipeline reads `interpretable` to decide which path to take:
  True  → use the model directly (Q-table rows, oracle rule, etc.)
  False → fit a surrogate DT on the policy's Q-values, extract a path

Both paths produce a PolicyExplanation with the same interface, so the
prompt builder and semantic checker are policy-agnostic.
"""

from __future__ import annotations

from typing import Literal

ExplanationType = Literal["q_table", "random", "oracle", "surrogate_dt"]


POLICY_REGISTRY: dict[str, dict] = {
    "q_learning": {
        "label": "tabular Q-learning",
        "interpretable": True,
        "explanation_type": "q_table",
        "notes": "State space is (prev_action, prev_outcome) — tiny table, directly human-readable.",
    },
    "random": {
        "label": "uniform random policy",
        "interpretable": True,
        "explanation_type": "random",
        "notes": "No learned structure; action is sampled uniformly.",
    },
    "route_aware_oracle": {
        "label": "route-aware oracle",
        "interpretable": True,
        "explanation_type": "oracle",
        "notes": "Deterministic rule: always target Alice's current route. Fully transparent.",
    },
    # ── Non-interpretable policies — surrogate DT used for XAI ──────────────
    "deep_q": {
        "label": "deep Q-network (DQN)",
        "interpretable": False,
        "explanation_type": "surrogate_dt",
        "notes": "MLP function approximator; surrogate DT fitted on enumerated Q-values.",
    },
    # ── Future policies (scaffold) ────────────────────────────────────────
    # "ppo": {
    #     "label": "PPO policy network",
    #     "interpretable": False,
    #     "explanation_type": "surrogate_dt",
    # },
}


def is_interpretable(policy_name: str) -> bool:
    return POLICY_REGISTRY.get(policy_name, {}).get("interpretable", False)


def explanation_type(policy_name: str) -> ExplanationType:
    return POLICY_REGISTRY.get(policy_name, {}).get("explanation_type", "surrogate_dt")
