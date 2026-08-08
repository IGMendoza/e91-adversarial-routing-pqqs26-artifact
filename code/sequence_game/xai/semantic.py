"""Rule-based semantic consistency checker for LLM explanations.

Catches factual contradictions that numeric faithfulness misses:
a model can hallucinate the causal direction entirely while citing
no invented numbers, scoring 100% numeric faithfulness.

Checks are deterministic regex rules — no LLM judge required:

1. Outcome polarity: did the model contradict the verified previous outcome?
2. Attack type: for DoS actions (swap_denial), did the model correctly
   identify it as blocking delivery rather than an information-gain attack?
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class SemanticViolation:
    rule: str
    matched_text: str


@dataclass
class SemanticCheckResult:
    passed: bool
    score: float
    violations: list[SemanticViolation] = field(default_factory=list)

    @property
    def label(self) -> str:
        return "PASS" if self.passed else "FAIL"


# ── Outcome polarity rules ────────────────────────────────────────────────────

_ACCEPTED_CONTRADICTIONS: list[tuple[str, str]] = [
    (r"following\s+a\s+previous\s+delivery\s+fail", "claimed_prior_delivery_failure"),
    # Exclude "previous outcome was a success (not a failure)" — the "success" before "fail" signals
    # the model is correctly contrasting, not asserting failure.
    (r"previous\s+outcome\s+was\s+(?!.{0,30}(?:success|accepted|not\s+a\s+fail)).{0,25}fail", "claimed_prior_outcome_failure"),
    (r"previous\s+attack\s+.{0,40}fail", "claimed_prior_attack_failure"),
    (r"(?:prior|previous|last)\s+(?:step|round|attempt)\s+.{0,15}fail", "claimed_prior_step_failure"),
    (r"prior\s+delivery\s+fail", "claimed_prior_delivery_failure_alt"),
    (r"had\s+already\s+(?:been\s+)?(?:disrupted|compromised|failed)", "claimed_prior_disruption"),
]

_FAILURE_CONTRADICTIONS: list[tuple[str, str]] = [
    (r"previous\s+outcome\s+was\s+.{0,25}success", "claimed_prior_outcome_success"),
    (r"previous\s+outcome\s+was\s+.{0,25}accept", "claimed_prior_outcome_accepted"),
    (r"prior\s+(?:step|round|attempt)\s+.{0,15}success", "claimed_prior_step_success"),
    (r"previous\s+delivery\s+succeed", "claimed_prior_delivery_success"),
    # catches "previous key delivery was accepted", "previous delivery was accepted"
    (r"previous\s+(?:\w+\s+)?delivery\s+was\s+accept", "claimed_prior_delivery_accepted"),
    # catches "key delivery was accepted", "photon delivery was accepted"
    (r"(?:key|photon)\s+delivery\s+was\s+accept", "claimed_prior_key_delivery_accepted"),
]

# ── Attack-type classification rules ─────────────────────────────────────────
# For swap_denial actions: the response must reflect that this is a DoS attack
# (blocks delivery, no information gain for Eve). Claiming it gives Eve
# information or raises QBER like an intercept-resend contradicts the glossary.

_SWAP_DENIAL_TYPE_VIOLATIONS: list[tuple[str, str]] = [
    # Claiming swap_denial gains Eve information — incorrect; it's a pure DoS.
    # Exclude negated contexts and noun uses:
    #   Lookbehinds: "without " (8), "from " (5), "not " (4), "information " (12 — "information gain" noun).
    (r"(?<!without )(?<!from )(?<!not )(?<!information )\b(?:gain|intercept|eavesdrop|extract)\w*\s+(?!(?:no|zero|not|nothing|any)\b).{0,25}(?:key|information|bits|info)", "swap_denial_info_gain_claim"),
    # Claiming QBER rises due to swap_denial — incorrect; QBER is irrelevant when delivery_failure occurs
    (r"qber.{0,20}(?:rise|rises|rose|increase|high)", "swap_denial_qber_rise_claim"),
    # Confusing swap_denial with memory_degradation ("fidelity drops" due to Eve's swap_denial)
    (r"(?:fidelity|quality).{0,20}(?:drops|degrades|reduces|lowers).{0,30}(?:swap|denial)", "swap_denial_fidelity_claim"),
]


# ── Actions known to be DoS ───────────────────────────────────────────────────
# attack_surface mode: swap_denial_r0 / swap_denial_r1
# binary_collision mode: node_<id> / edge_<id> (both block delivery, no info gain)
_DOS_ACTIONS = {"swap_denial_r0", "swap_denial_r1"}


def _is_dos_action(action: str) -> bool:
    return (
        action in _DOS_ACTIONS
        or action.startswith("node_")
        or action.startswith("edge_")
        or "denial" in action
        or "dos" in action
        or "collision" in action
    )

_NEGATION_WORDS = re.compile(r"\b(?:no|not|never|zero|without|nothing)\b")
# Comparison/preference phrases appearing before the verb in the context window.
# Allows up to 20 chars of intervening words (e.g. "rather than attempting to").
# "risk X gaining" = counterfactual comparison ("alternative actions risk Eve gaining info").
_COMPARISON_BEFORE = re.compile(
    r"\b(?:over|instead\s+of|rather\s+than|prefer(?:red)?\s+to"
    r"|allow(?:s|ed)?\s+\w+\s+to"
    r"|would\s+(?:allow\s+\w+\s+to\s+)?"
    r"|risk(?:s|ed)?\s+\w+\s+"
    r").{0,20}$"
)


def _match_is_negated(matched_text: str, context_before: str = "") -> bool:
    """Return True if the matched snippet is in a negated/comparison context.

    Handles:
    - "gains her no information" — negation INSIDE the match
    - "over gaining partial information" — comparison before the match
    """
    if _NEGATION_WORDS.search(matched_text):
        return True
    if context_before and _COMPARISON_BEFORE.search(context_before):
        return True
    return False


def check_semantic_faithfulness(
    response: str,
    prev_outcome_is_failure: bool,
    predicted_action: str = "",
) -> SemanticCheckResult:
    """Check response for contradictions against verified state facts.

    Args:
        response: LLM response text.
        prev_outcome_is_failure: True if the verified previous outcome was
            delivery_failure; False if it was accepted.
        predicted_action: The DT-predicted action name, used to apply
            attack-type classification checks.

    Returns:
        SemanticCheckResult with pass/fail, score (1.0 = no violations),
        and a list of matched violations.
    """
    resp_lower = response.lower()
    violations: list[SemanticViolation] = []

    # 1. Outcome polarity check
    polarity_patterns = (
        _FAILURE_CONTRADICTIONS if prev_outcome_is_failure else _ACCEPTED_CONTRADICTIONS
    )
    for pattern, rule in polarity_patterns:
        m = re.search(pattern, resp_lower)
        if m:
            violations.append(SemanticViolation(rule=rule, matched_text=m.group()))

    # 2. Attack-type classification check (DoS actions: swap_denial + node/edge collision)
    if _is_dos_action(predicted_action):
        for pattern, rule in _SWAP_DENIAL_TYPE_VIOLATIONS:
            m = re.search(pattern, resp_lower)
            if m:
                ctx_before = resp_lower[max(0, m.start() - 50):m.start()]
                if not _match_is_negated(m.group(), ctx_before):
                    violations.append(SemanticViolation(rule=rule, matched_text=m.group()))

    score = 0.0 if violations else 1.0
    return SemanticCheckResult(passed=len(violations) == 0, score=score, violations=violations)


# ── Mitigation-specific semantic rules ───────────────────────────────────────
# Mitigation responses must: stay within toy-simulation scope, avoid physical
# validity claims, and not contradict the attack description.

_MITIGATION_SCOPE_VIOLATIONS: list[tuple[str, str]] = [
    # Physical validity claim
    (r"\b(?:in\s+(?:reality|practice|real\s+(?:quantum\s+)?networks?)"
     r"|real(?:\s+world)?(?:\s+quantum)?\s+(?:system|network|scenario))",
     "physical_validity_claim"),
    # Security guarantee claim
    (r"\b(?:guarantee[sd]?\s+security|provably\s+secure|information.theoretic"
     r"|unconditionally\s+secure|prevent\s+all\s+attacks?)",
     "security_guarantee_claim"),
    # Contradicting DoS-is-not-info-gain for node/edge collision context
    (r"(?<!no\s)(?<!without\s)eve\s+(?:can|could|would|gains?)\s+.{0,30}"
     r"(?:key\s+material|cryptographic\s+key|secret)",
     "dos_info_gain_contradiction"),
]

_MITIGATION_REQUIRED: list[tuple[str, str]] = [
    # Must mention simulation scope.
    # simulat\w* matches simulation/simulated/simulating/simulator.
    # No trailing \b needed — character class consumes the word suffix.
    (r"\b(?:simulat\w+|toy\s+(?:model|scope|simulation\w*)|in\s+this\s+(?:model|scenario|setup|simulation\w*))\b",
     "missing_simulation_scope"),
    # Must suggest at least one concrete defensive measure
    (r"\b(?:rout(?:e|ing)|redundan|detecti|threshold|switch|divers|monitor|alert|backup)\b",
     "missing_concrete_defense"),
]


def check_mitigation_semantic_faithfulness(
    response: str,
    action: str = "",
) -> SemanticCheckResult:
    """Semantic check for LLM mitigation-advice responses.

    Checks:
    1. No physical-validity or security-guarantee overclaiming.
    2. No contradiction of attack type (DoS ≠ info gain).
    3. Simulation scope is mentioned.
    4. At least one concrete defensive measure is suggested.

    Uses simpler negation detection (only direct negation words, not comparison
    context) to avoid false-negatives on phrases like "routes would guarantee".
    """
    resp_lower = response.lower()
    violations: list[SemanticViolation] = []

    for pattern, rule in _MITIGATION_SCOPE_VIOLATIONS:
        m = re.search(pattern, resp_lower)
        if m:
            matched = m.group()
            # Only suppress if a negation word appears inside the match or
            # immediately before it (within 20 chars), not comparison context.
            ctx_before = resp_lower[max(0, m.start() - 20): m.start()]
            negated = bool(_NEGATION_WORDS.search(matched)) or bool(_NEGATION_WORDS.search(ctx_before))
            if not negated:
                violations.append(SemanticViolation(rule=rule, matched_text=matched))

    for pattern, rule in _MITIGATION_REQUIRED:
        if not re.search(pattern, resp_lower):
            violations.append(SemanticViolation(rule=rule, matched_text="<missing>"))

    score = max(0.0, 1.0 - len(violations) * 0.25)
    return SemanticCheckResult(
        passed=len(violations) == 0,
        score=round(score, 2),
        violations=violations,
    )
