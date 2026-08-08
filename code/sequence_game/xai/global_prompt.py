"""Global (policy-level) strategy prompt builder.

Local explanations answer: why did Eve choose THIS action at THIS step?
Global explanations answer: what is Eve's overall strategy across all observed steps?

The global prompt summarises the action distribution, dominant state-action
patterns, and how topology shapes the aggregate policy.  No LLM judge or
surrogate DT is required — the prompt is built from the step history alone.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .prompt_builder import ACTION_TYPE, GLOSSARY, _centrality_label, action_type_label


@dataclass
class GlobalStrategyContext:
    eve_policy: str
    num_steps: int
    action_counts: dict[str, int]
    outcome_counts: dict[str, int]
    # (prev_outcome, chosen_action) → count — state-transition summary
    transition_counts: dict[tuple[str, str], int]
    r0_bottleneck: float
    r1_bottleneck: float
    r0_hops: int
    r1_hops: int
    scope: str = "toy"

    @property
    def dominant_action(self) -> str:
        if not self.action_counts:
            return "unknown"
        return max(self.action_counts, key=lambda k: self.action_counts[k])

    @property
    def delivery_failure_rate(self) -> float:
        total = sum(self.outcome_counts.values()) or 1
        # binary_collision uses "hit" (Eve's attack landed = Alice failed) and "miss"
        # attack_surface modes use "delivery_failure" / "delivered" / "information_exposure"
        failures = (
            self.outcome_counts.get("delivery_failure", 0)
            + self.outcome_counts.get("hit", 0)
        )
        return failures / total

    @property
    def dos_fraction(self) -> float:
        total = sum(self.action_counts.values()) or 1
        dos = sum(
            v for k, v in self.action_counts.items()
            # attack_surface: swap_denial_*  |  binary_collision: node_* / edge_*
            if "denial" in k or k.startswith("node_") or k.startswith("edge_")
            or "collision" in k or "dos" in k
        )
        return dos / total

    @property
    def quality_fraction(self) -> float:
        total = sum(self.action_counts.values()) or 1
        qa = sum(
            v for k, v in self.action_counts.items()
            if "degradation" in k or "measure" in k or "dephasing" in k or "probe" in k
        )
        return qa / total


def build_global_prompt(ctx: GlobalStrategyContext) -> str:
    gloss_lines = "\n".join(f"- {k}: {v}" for k, v in GLOSSARY.items())

    # Action distribution table
    total_acts = sum(ctx.action_counts.values()) or 1
    action_rows = "\n".join(
        f"  {act:<35} {cnt:>4}  ({cnt/total_acts:.0%})  [{action_type_label(act)}]"
        for act, cnt in sorted(ctx.action_counts.items(), key=lambda x: -x[1])
    )

    # Outcome distribution
    total_out = sum(ctx.outcome_counts.values()) or 1
    outcome_rows = "\n".join(
        f"  {out:<25} {cnt:>4}  ({cnt/total_out:.0%})"
        for out, cnt in sorted(ctx.outcome_counts.items(), key=lambda x: -x[1])
    )

    # State-transition patterns: (prev_outcome, chosen_action) → count
    trans_rows = "\n".join(
        f"  after {prev_out:<20} → {act:<30} {cnt:>4}×"
        for (prev_out, act), cnt in sorted(
            ctx.transition_counts.items(), key=lambda x: -x[1]
        )[:10]
    )

    r0_label = _centrality_label(ctx.r0_bottleneck)
    r1_label = _centrality_label(ctx.r1_bottleneck)

    dos_pct = f"{ctx.dos_fraction:.0%}"
    qa_pct  = f"{ctx.quality_fraction:.0%}"
    fail_pct = f"{ctx.delivery_failure_rate:.0%}"

    return f"""You are explaining the strategic policy of an adversarial agent (Eve) across an
entire observed game run in a quantum repeater network security game.

## Scope
Scientific scope: {ctx.scope} — policy distribution only; not a security proof.

## Run summary
- Eve policy: {ctx.eve_policy}
- Total steps observed: {ctx.num_steps}
- Alice delivery failure rate: {fail_pct} of steps ended with Alice's key delivery blocked (outcome=hit or delivery_failure)
- DoS actions (swap_denial): {dos_pct} of Eve's moves
- Quality attacks (degradation/probe): {qa_pct} of Eve's moves

## Network topology
- Route r0: alice → r0-repeater → bob  |  hops={ctx.r0_hops}  |  bottleneck_score={ctx.r0_bottleneck:.3f} ({r0_label})
- Route r1: alice → r1-repeater → bob  |  hops={ctx.r1_hops}  |  bottleneck_score={ctx.r1_bottleneck:.3f} ({r1_label})

## Action distribution (all steps)
{action_rows}

## Outcome distribution
{outcome_rows}

## State-transition patterns (top 10)
(what Eve chose after each observed previous outcome)
{trans_rows}

## Glossary
{gloss_lines}

## Task
In 4-6 sentences, describe Eve's OVERALL attack strategy across the full game run.
Your explanation MUST include:
1. Whether Eve primarily relies on DoS attacks, quality attacks, or a mix — and why
   that makes strategic sense given the network topology.
2. How the bottleneck scores shaped which route(s) Eve targeted most.
3. What the state-transition patterns reveal about Eve's adaptive behaviour
   (e.g., does she switch routes after a failed attack? does success reinforce the same attack?).
4. An assessment of strategic effectiveness: given the delivery failure rate and
   action mix, how well did Eve's policy disrupt Alice's key delivery?

Rules:
- Only use facts, numbers, and terms present in the sections above.
- Do not introduce physics concepts beyond what the glossary defines.
- Distinguish clearly between DoS effects (delivery blocked, no info for Eve) and
  quality-attack effects (fidelity degraded, potential info gain for Eve).
- Write for a network security operator, not an AI or physics specialist.
"""


def context_from_steps(
    steps: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    eve_policy: str = "unknown",
    scope: str = "toy",
) -> GlobalStrategyContext:
    """Build a GlobalStrategyContext from GUI final_window step records."""
    action_counts: Counter = Counter()
    outcome_counts: Counter = Counter()
    transition_counts: Counter = Counter()

    prev_outcome = "start"
    for step in steps:
        ea = step.get("eve_action") or {}
        kind   = ea.get("kind", "no_attack")
        target = ea.get("target", "")
        action = f"{kind}_{target}" if target else kind

        outcome = step.get("public_outcome", "unknown")
        action_counts[action] += 1
        outcome_counts[outcome] += 1
        if prev_outcome != "start":
            transition_counts[(prev_outcome, action)] += 1
        prev_outcome = outcome

    # Compute bottleneck scores from candidate routes
    total_routes = len(routes) or 1
    btn: dict[str, float] = {}
    for route in routes:
        for node in (route.get("path") or [])[1:-1]:
            btn[node] = btn.get(node, 0.0) + 1.0 / total_routes

    sorted_routes = sorted(routes, key=lambda r: r.get("route_id", ""))
    r0 = sorted_routes[0] if sorted_routes else None
    r1 = sorted_routes[1] if len(sorted_routes) > 1 else None

    def _btn(r):
        if not r:
            return 0.5
        return max((btn.get(n, 0.0) for n in (r.get("path") or [])[1:-1]), default=0.5)

    def _hops(r):
        return int(r.get("hop_count") or 1) if r else 1

    return GlobalStrategyContext(
        eve_policy=eve_policy,
        num_steps=len(steps),
        action_counts=dict(action_counts),
        outcome_counts=dict(outcome_counts),
        transition_counts=dict(transition_counts),
        r0_bottleneck=_btn(r0),
        r1_bottleneck=_btn(r1),
        r0_hops=_hops(r0),
        r1_hops=_hops(r1),
        scope=scope,
    )
