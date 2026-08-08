"""Build structured LLM prompts from DT path + counterfactual evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

GLOSSARY: dict[str, str] = {
    "QBER": "Quantum Bit Error Rate — fraction of sifted key bits that disagree between Alice and Bob; baseline ~0.0, intercept-resend ~0.24",
    "sifted key": "Bits where Alice and Bob chose the same measurement basis; used to derive the final shared key",
    "entanglement swap": "Operation at a quantum repeater that transfers entanglement across two links; required for end-to-end key delivery",
    "swap denial (DoS attack)": "Eve blocks the entanglement swap at a target repeater → delivery_failure, zero delivered bits. Alice detects the failure but Eve gains no key information.",
    "node collision": "binary_collision mode: Eve blocks entanglement at a specific repeater node on Alice's path → hit outcome, delivery fails. Eve gains no key information.",
    "edge collision": "binary_collision mode: Eve disrupts a specific quantum link (edge) on Alice's path → hit outcome, delivery fails. Eve gains no key information.",
    "memory degradation (quality attack)": "Eve degrades qubit fidelity at a repeater memory → key quality falls (fidelity < 1.0), QBER may rise. Key may still be accepted at lower quality. Eve potentially gains partial information.",
    "delivery_failure": "Alice-to-Bob key delivery failed; Bob received 0 entangled photons after the repeater swap",
    "hit": "binary_collision mode outcome: Eve's attack successfully landed on Alice's chosen route → delivery failed for Alice",
    "miss": "binary_collision mode outcome: Eve's attack missed Alice's route (or Eve chose no attack) → Alice delivered successfully",
    "accepted": "Key delivery succeeded; Alice and Bob share sifted key bits",
    "bottleneck_score": "Fraction of all Alice→Bob routes that pass through a given repeater. Score=1.0 means every path goes through it (single chokepoint); score=0.5 means half of paths use it. Higher = more operationally critical = higher-value attack target.",
    "route overlap": "Number of edges shared between two candidate routes; lower overlap = better resilience to targeted attacks",
    "fidelity": "Closeness of the transmitted quantum state to the ideal Bell state; 1.0 = perfect, lower = noise or degradation attack",
}

# Static map for named attack_surface actions
_NAMED_ACTION_TYPE: dict[str, str] = {
    "no_attack": "passive (no attack)",
    "none": "passive (no attack)",
    "swap_denial_r0": "DoS — blocks delivery on r0, Alice gains no key material",
    "swap_denial_r1": "DoS — blocks delivery on r1, Alice gains no key material",
    "memory_degradation_r0": "quality attack — degrades key fidelity on r0, Eve may gain partial info",
    "memory_degradation_r1": "quality attack — degrades key fidelity on r1, Eve may gain partial info",
}

# Kept for backward compatibility (global_prompt.py imports this)
ACTION_TYPE = _NAMED_ACTION_TYPE


def action_type_label(action: str) -> str:
    """Return a human-readable attack classification for any action name."""
    if action in _NAMED_ACTION_TYPE:
        return _NAMED_ACTION_TYPE[action]
    if action.startswith("node_"):
        target = action[5:]
        return f"node collision at {target} — blocks delivery, Alice gains no key material"
    if action.startswith("edge_"):
        target = action[5:]
        return f"edge collision on {target} — blocks delivery, Alice gains no key material"
    if "denial" in action or "dos" in action:
        return "DoS — blocks delivery"
    if "degradation" in action or "probe" in action or "dephasing" in action:
        return "quality attack — degrades fidelity, potential info gain for Eve"
    if "information_probe" in action:
        return "information probe — Eve intercepts without blocking delivery"
    return "unknown attack type"


@dataclass
class CounterfactualRow:
    action: str
    route: str
    outcome: str
    fidelity: Optional[float]
    qber: Optional[float]
    quality_metric: Optional[float]
    is_chosen: bool = False


@dataclass
class PromptContext:
    policy_explanation: "PolicyExplanation"  # noqa: F821 — imported at call site
    chosen_row: CounterfactualRow
    alternative_rows: list[CounterfactualRow]
    r0_bottleneck: float
    r1_bottleneck: float
    r0_hops: int
    r1_hops: int

    # Convenience shims so existing call sites that read these fields keep working
    @property
    def predicted_action(self) -> str:
        return self.policy_explanation.predicted_action

    @property
    def dt_fidelity(self) -> Optional[float]:
        return self.policy_explanation.dt_fidelity


def build_mitigation_prompt(
    explanation: str,
    predicted_action: str,
    eve_policy: str,
    ctx_type: str = "step",
) -> str:
    """Build a defender-focused mitigation-advice prompt from a prior XAI explanation."""
    action_desc = action_type_label(predicted_action)
    scope = "a specific simulation step" if ctx_type == "step" else "the overall strategy"
    return (
        f"You are a quantum network security advisor reviewing a TOY SIMULATION.\n\n"
        f"Scientific scope: TOY SIMULATION — do not claim physical validity or real security guarantees.\n\n"
        f"## Attack context ({scope})\n"
        f"Eve policy: {eve_policy}\n"
        f"Attack action: {action_desc} (id: {predicted_action})\n"
        f"XAI explanation:\n{explanation[:800]}\n\n"
        f"## Task\n"
        f"In 3–5 concise sentences, suggest concrete network-level defensive measures "
        f"that a researcher or operator could test in this toy simulation to mitigate "
        f"the described attack pattern.\n\n"
        f"Rules:\n"
        f"- Label every suggestion as applying to 'this toy simulation.'\n"
        f"- Focus on actionable parameters: routing diversity, detection thresholds, "
        f"route switching, redundancy.\n"
        f"- Do not invent physics or security claims beyond the described toy model.\n"
        f"- Do not start with disclaimers — answer directly.\n"
        f"- Format: use **bold** for key terms, bullet points where helpful.\n"
        f"- The viewer renders Markdown. Keep under 180 words.\n"
    )


def build_prompt(ctx: PromptContext) -> str:
    gloss_lines = "\n".join(f"- {k}: {v}" for k, v in GLOSSARY.items())
    chosen_str = _format_row(ctx.chosen_row)
    alt_str = "\n".join(_format_row(r) for r in ctx.alternative_rows)
    pe = ctx.policy_explanation
    state_fact = (
        "The PREVIOUS outcome was delivery_failure (key delivery failed)."
        if pe.prev_outcome_is_failure
        else "The PREVIOUS outcome was accepted (key delivery succeeded)."
    )
    chosen_type = action_type_label(pe.predicted_action)
    method_label = {
        "q_table": "tabular Q-learning (directly interpretable — no surrogate needed)",
        "surrogate_dt": f"surrogate decision tree (fidelity={pe.dt_fidelity:.2%})" if pe.dt_fidelity is not None else "surrogate decision tree",
        "random": "uniform random policy",
        "oracle": "route-aware oracle (deterministic rule)",
    }.get(pe.method, pe.method)

    # Network centrality framing
    r0_centrality = _centrality_label(ctx.r0_bottleneck)
    r1_centrality = _centrality_label(ctx.r1_bottleneck)

    return f"""You are explaining the decision of an adversarial agent (Eve) in a quantum repeater network security game.

## Verified current state — read this first, do NOT contradict it
- {state_fact}
- Eve's predicted action: `{pe.predicted_action}`
- Attack type: {chosen_type}
- Explanation method: {method_label}

## Network topology and centrality
- Route r0: alice → r0-repeater → bob  |  hops={ctx.r0_hops}  |  bottleneck_score={ctx.r0_bottleneck:.3f} ({r0_centrality})
- Route r1: alice → r1-repeater → bob  |  hops={ctx.r1_hops}  |  bottleneck_score={ctx.r1_bottleneck:.3f} ({r1_centrality})
- Alice selects a route each step to deliver entangled key material to Bob.
- Eve can attack one repeater per step. Higher bottleneck_score = more routes affected = higher-value target.

## Policy explanation (how the model arrived at this action)
{pe.text}

## Chosen action outcome (from quantum network simulator)
{chosen_str}

## Counterfactual outcomes (simulator results for alternative actions)
{alt_str}

## Glossary
{gloss_lines}

## Task
In 3-5 sentences, explain why Eve chose `{pe.predicted_action}` given the verified state and policy evidence above.
Your explanation MUST include:
1. Whether this is a DoS attack (blocks delivery, no info gain) or a quality attack (degrades fidelity, potential info gain), and why Eve prefers one over the other here.
2. How the network bottleneck scores influence which repeater is the higher-value target.
3. What the policy explanation above reveals about why this action was chosen (Q-values, rule, or DT path — whichever applies).

Rules:
- The verified state section is ground truth — do NOT say the opposite of what it states.
- Only use facts, numbers, and terms present in the sections above.
- Do not introduce quantum physics concepts beyond what the glossary defines.
- Write for a network security operator, not an AI or physics specialist.
- Interpret the decision tree in plain English; do not copy it verbatim.
- Format: use **bold** for key terms, bullet lists where helpful. Keep under 200 words.
- The viewer renders Markdown — feel free to use it.
"""


def _centrality_label(score: float) -> str:
    if score >= 0.9:
        return "single chokepoint — all paths pass through here"
    if score >= 0.6:
        return "high centrality — most paths pass through here"
    if score >= 0.4:
        return "moderate centrality — roughly half of paths use this repeater"
    return "low centrality — minority of paths use this repeater"


def build_cross_graph_explainer_prompt(
    agg_stats: dict,
    dt_rules_text: str,
) -> str:
    """Prompt asking the LLM to explain how graph structure shaped Eve/Alice capabilities."""
    condition = agg_stats.get("condition") or {}
    condition_line = _condition_line(condition)
    n_assign = agg_stats.get("total_assignments", agg_stats.get("total_graphs", "N"))
    n_topo = agg_stats.get("total_topologies", "?")
    fam_lines = []
    for fam, stats in agg_stats.get("families", {}).items():
        fam_lines.append(
            f"  {fam}: {stats['count']} assignments, "
            f"Eve win rate {stats['eve_win_rate_mean']:.2%} ± {stats['eve_win_rate_std']:.2%}, "
            f"N_mean={stats.get('N_mean')}, E_mean={stats.get('E_mean')}"
        )
    families_block = "\n".join(fam_lines) or "  (no family data)"

    top_feats = agg_stats.get("structural_dt", {}).get("top_features", [])
    top_feat_lines = "\n".join(
        f"  {f['feature']}: importance={f['importance']:.3f}" for f in top_feats[:5]
    )
    r2 = agg_stats.get("structural_dt", {}).get("r2_score", 0.0)
    n_dt_topo = agg_stats.get("structural_dt", {}).get("num_topologies", "?")

    rules_block = dt_rules_text[:2000] if len(dt_rules_text) > 2000 else dt_rules_text

    return f"""\
You are an expert in quantum network security and graph theory. The following data
comes from a toy simulation study (NOT a real quantum network) where an RL agent
(Eve) attempts to disrupt quantum key delivery between Alice and Bob.

Condition: {condition_line}

Setup: {n_assign} game assignments (distinct Alice/Bob node-pair instances) across
{n_topo} unique graph topologies drawn from 6 structural families (braess, bottleneck,
ladder, diamond, layered_parallel, bounded_er). Multiple assignments may share a topology.

SCOPE: toy simulation — no physical validity or security guarantees claimed.

## Aggregate results by graph family

{families_block}

## Structural decision tree (pooled from all {n_assign} assignments, {n_dt_topo} unique topologies)
DT R² = {r2:.3f}. Top predictive features:
{top_feat_lines}

DT rules (IF structural_feature conditions → predicted Eve Q-value rank):
{rules_block}

## Glossary
- N (graph_N): size of smallest set of internal nodes that hits every candidate route (node hitting set). Larger N = Alice has more routing diversity = harder for Eve.
- E (graph_E): size of smallest set of edges that hits every candidate route (edge hitting set).
- graph_C_log2: log₂(number of candidate routes from Alice to Bob). Larger = more routing options for Alice.
- graph_S / graph_L: shortest / longest candidate route hop count. graph_L_over_S = ratio.
- route_coverage: fraction of candidate routes that pass through this node or edge. Higher = more valuable target for Eve.
- in_node_hs / in_edge_hs: 1 if this target belongs to a minimum hitting set, 0 otherwise.
- is_node / is_edge: 1 if the action targets a node / edge respectively.
- Eve win rate: fraction of steps where Eve's attack disrupted Alice's delivery.
- Q-value rank: relative preference of an attack action for Eve, rank-normalised to [0,1]. **1.0 = Eve most prefers this target; 0.0 = Eve avoids this target.** High Q-rank actions are effective attack choices.

## Your task
Write a 200-250 word explanation (for a QKD network researcher) of:
1. How graph structural properties (N, E, route coverage, family type) shaped Eve's attack capabilities across graph families. Compare which families were most vs. least vulnerable and explain why the structure accounts for it.
2. What the pooled decision tree reveals about which structural properties most strongly predict high Eve Q-value rank (i.e., effective attack targets).
3. How Alice's routing diversity (reflected in N and E) constrained or enabled Eve's learned strategy.

Format: use **bold** for key terms. Use bullet points where helpful. The viewer renders Markdown.
Do NOT claim physical validity. Do NOT make security guarantees. Describe only what the toy simulation shows.
"""


def build_graph_recommender_prompt(
    agg_stats: dict,
    dt_rules_text: str,
) -> str:
    """Prompt asking the LLM for pro-Eve and pro-Alice graph configuration recommendations."""
    condition = agg_stats.get("condition") or {}
    condition_line = _condition_line(condition)
    n_assign = agg_stats.get("total_assignments", agg_stats.get("total_graphs", "N"))
    n_topo = agg_stats.get("total_topologies", "?")
    fam_lines = []
    for fam, stats in agg_stats.get("families", {}).items():
        fam_lines.append(
            f"  {fam}: {stats['count']} assignments, "
            f"Eve win {stats['eve_win_rate_mean']:.2%}, "
            f"N_mean={stats.get('N_mean')}, E_mean={stats.get('E_mean')}"
        )
    families_block = "\n".join(fam_lines) or "  (no family data)"

    top_feats = agg_stats.get("structural_dt", {}).get("top_features", [])
    top_feat_lines = "\n".join(
        f"  {f['feature']}: importance={f['importance']:.3f}" for f in top_feats[:5]
    )
    r2 = agg_stats.get("structural_dt", {}).get("r2_score", 0.0)
    rules_block = dt_rules_text[:1500] if len(dt_rules_text) > 1500 else dt_rules_text

    return f"""\
You are a quantum network design advisor. The following data comes from a toy simulation
where an RL agent (Eve) attacks quantum key delivery across {n_assign} game assignments
on {n_topo} unique graph topologies spanning 6 structural families.

Condition: {condition_line}

SCOPE: toy simulation — no physical validity or security guarantees claimed.

## Results by family

{families_block}

## Top structural predictors of Eve's attack effectiveness (pooled DT, R²={r2:.3f})
{top_feat_lines}

## DT rules (structural features → Eve Q-value rank)
{rules_block}

## Glossary
- Pro-Eve topology: graph structure that makes Eve's attacks more effective (high win rate).
- Pro-Alice topology: graph structure that makes Alice more resilient (low Eve win rate).
- N (graph_N): minimum node hitting set size. Larger N = Eve must block more nodes simultaneously = harder.
- E (graph_E): minimum edge hitting set size.
- graph_C_log2: log₂(number of candidate routes). Larger = Alice has more routing paths = harder for Eve.
- route_coverage: fraction of routes through a target. Higher = more valuable for Eve to attack.
- Q-value rank: [0,1] preference score for Eve's attack actions. **1.0 = strongly preferred by Eve (effective attack); 0.0 = avoided.** DT leaf values close to 1.0 = conditions where Eve finds effective targets.

## Your task
Write two short recommendation blocks (100-120 words each), formatted as Markdown:

**Pro-Eve topology recommendations**: Describe which graph family and parameter ranges
(N, E, graph_C_log2, route_coverage) most favour Eve in this toy simulation. Cite specific
DT features and the family with the highest win rate. Remember high Q-rank = Eve prefers that action.

**Pro-Alice topology recommendations**: Describe which configurations best protect Alice —
i.e., keep Eve's win rate low — and explain mechanistically why those structural properties
make attacks harder (e.g., higher N means Eve must block more nodes simultaneously, higher graph_C_log2 means more routing diversity).

End with one sentence noting this is a toy simulation scope and real QKD security
depends on physical parameters not modelled here.
"""


def build_route_choice_explainer_prompt(
    agg_stats: dict,
    dt_rules_text: str,
) -> str:
    """Prompt asking the LLM to explain learned Alice route-choice structure."""
    condition = agg_stats.get("condition") or {}
    condition_line = _condition_line(condition)
    n_assign = agg_stats.get("total_assignments", agg_stats.get("total_graphs", "N"))
    n_topo = agg_stats.get("total_topologies", "?")
    fam_lines = []
    for fam, stats in agg_stats.get("families", {}).items():
        fam_lines.append(
            f"  {fam}: {stats['count']} assignments, "
            f"Eve win rate {stats['eve_win_rate_mean']:.2%} ± {stats['eve_win_rate_std']:.2%}, "
            f"N_mean={stats.get('N_mean')}, E_mean={stats.get('E_mean')}"
        )
    families_block = "\n".join(fam_lines) or "  (no family data)"

    top_feats = agg_stats.get("structural_dt", {}).get("top_features", [])
    top_feat_lines = "\n".join(
        f"  {f['feature']}: importance={f['importance']:.3f}" for f in top_feats[:6]
    )
    r2 = agg_stats.get("structural_dt", {}).get("r2_score", 0.0)
    n_dt_topo = agg_stats.get("structural_dt", {}).get("num_topologies", "?")
    rules_block = dt_rules_text[:2000] if len(dt_rules_text) > 2000 else dt_rules_text

    return f"""\
You are an expert in quantum network security, routing, and graph learning. The
following data comes from a toy simulation study (NOT a real quantum network)
where Alice learns/selects routes while Eve attacks.

Condition: {condition_line}

Setup: {n_assign} game assignments across {n_topo} unique graph topologies.
This DT is trained only on the Alice route-choice emission for this condition;
it is not pooled with Eve attack-choice DTs or other experiment designs.

SCOPE: toy simulation — no physical validity or security guarantees claimed.

## Aggregate results by graph family

{families_block}

## Alice route-choice decision tree (pooled from all {n_assign} assignments, {n_dt_topo} unique topologies)
DT R² = {r2:.3f}. Top predictive route/graph features:
{top_feat_lines}

DT rules (IF route/graph structural features → predicted Alice route choice rate):
{rules_block}

## Glossary
- choice_rate: fraction of backend corpus steps where learned Alice selected this candidate route. **1.0 = always selected; 0.0 = never selected.**
- route_hops / route_length: size and length of one candidate route.
- route_length_over_shortest / route_hops_over_shortest: route cost relative to the shortest candidate.
- route_contains_node_hs_member / route_contains_edge_hs_member: whether this route traverses a node/edge in a minimum hitting set.
- N and E: minimum node/edge hitting set sizes across candidate routes; larger values indicate more routing diversity.
- graph_C_log2: log₂(number of candidate Alice-Bob routes).

## Your task
Write a 200-250 word explanation of:
1. Which route and graph features best predicted learned Alice's route choices.
2. Whether Alice favored short/direct routes, routes avoiding bottlenecks, or some mixture.
3. How Eve's attack surface in this same condition may have shaped Alice's learned route selection.

Use **bold** for key terms. Do NOT describe this as an Eve attack-target DT.
Do NOT claim physical validity or security guarantees.
"""


def build_route_choice_recommender_prompt(
    agg_stats: dict,
    dt_rules_text: str,
) -> str:
    """Prompt asking the LLM for graph designs that shape learned Alice routing."""
    condition = agg_stats.get("condition") or {}
    condition_line = _condition_line(condition)
    n_assign = agg_stats.get("total_assignments", agg_stats.get("total_graphs", "N"))
    n_topo = agg_stats.get("total_topologies", "?")
    fam_lines = []
    for fam, stats in agg_stats.get("families", {}).items():
        fam_lines.append(
            f"  {fam}: {stats['count']} assignments, "
            f"Eve win {stats['eve_win_rate_mean']:.2%}, "
            f"N_mean={stats.get('N_mean')}, E_mean={stats.get('E_mean')}"
        )
    families_block = "\n".join(fam_lines) or "  (no family data)"

    top_feats = agg_stats.get("structural_dt", {}).get("top_features", [])
    top_feat_lines = "\n".join(
        f"  {f['feature']}: importance={f['importance']:.3f}" for f in top_feats[:6]
    )
    r2 = agg_stats.get("structural_dt", {}).get("r2_score", 0.0)
    rules_block = dt_rules_text[:1500] if len(dt_rules_text) > 1500 else dt_rules_text

    return f"""\
You are a graph-routing advisor for a toy quantum-network simulation. The data
below describes a learned Alice route-choice DT, not an Eve attack-choice DT.

Condition: {condition_line}

The DT is trained on route_choice rows from {n_assign} assignments and {n_topo}
unique topologies for this experiment condition only.

SCOPE: toy simulation — no physical validity or security guarantees claimed.

## Results by family

{families_block}

## Top predictors of Alice route choice (pooled DT, R²={r2:.3f})
{top_feat_lines}

## DT rules (route/graph structural features → Alice choice_rate)
{rules_block}

## Your task
Write two short recommendation blocks (100-120 words each), formatted as Markdown:

**Route designs Alice tends to select**: Describe the route/graph features associated
with high choice_rate. Mention whether the DT points to shortest routes, bottleneck
membership, route length ratio, or graph-level diversity.

**Route designs that stress Alice's learning**: Describe graph or route structures
that create ambiguous or adversarial route-selection conditions for learned Alice
under this attack surface.

End with one sentence noting this is a toy simulation and real QKD routing depends
on physical parameters not modelled here.
"""


def _condition_line(condition: dict) -> str:
    if not condition:
        return "unspecified learned condition"
    return (
        f"{condition.get('game_mode', 'unknown_mode')} × "
        f"{condition.get('eve', 'unknown_eve')} Eve vs "
        f"{condition.get('alice', 'unknown_alice')} Alice "
        f"(group={condition.get('group', 'unknown')}, target={condition.get('dt_target', 'eve_q_rank')})"
    )


def _format_row(row: CounterfactualRow) -> str:
    fid = f"{row.fidelity:.3f}" if row.fidelity is not None else "N/A"
    qber = f"{row.qber:.4f}" if row.qber is not None else "N/A"
    qm = f"{row.quality_metric:.3f}" if row.quality_metric is not None else "N/A"
    attack_type = action_type_label(row.action)
    tag = "  [CHOSEN]" if row.is_chosen else ""
    return (
        f"  action={row.action} [{attack_type}]  route={row.route}  outcome={row.outcome}"
        f"  fidelity={fid}  QBER={qber}  quality_metric={qm}{tag}"
    )
