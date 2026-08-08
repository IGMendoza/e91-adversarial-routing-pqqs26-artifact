"""LLM prompt inputs derived from the histogram co-learning paper bundle.

The generated files are review prompts, not LLM responses.  Each prompt keeps
the active histogram backend and the mixed scientific scope visible so a model
cannot silently substitute conclusions from the deprecated SeQUeNCe study.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .paper_xai import PaperDecisionTreeBundle


LLM_PROMPT_SCHEMA = "histogram_colearning_llm_prompt_v1"


def write_histogram_llm_prompts(
    out_dir: str | Path,
    tables: dict[str, list[dict[str, Any]]],
    *,
    tree_bundle: PaperDecisionTreeBundle,
    source_schema: str,
    source_backend: str,
    total_bandit_turns: int,
    turns_per_replicate: int,
) -> dict[str, Any]:
    """Write four global and one-per-case prompts plus a JSONL index."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    prompts = _build_prompts(
        tables,
        tree_bundle=tree_bundle,
        source_backend=source_backend,
        total_bandit_turns=total_bandit_turns,
        turns_per_replicate=turns_per_replicate,
    )
    entries = []
    paths: list[str] = []
    for prompt in prompts:
        path = output / f"{prompt['prompt_id']}.txt"
        path.write_text(str(prompt["prompt"]).rstrip() + "\n", encoding="utf-8")
        paths.append(str(path))
        entries.append({
            "prompt_id": prompt["prompt_id"],
            "purpose": prompt["purpose"],
            "case_slug": prompt.get("case_slug"),
            "file": path.name,
        })

    jsonl_path = output / "prompts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            record = {
                "schema": LLM_PROMPT_SCHEMA,
                "source_schema": source_schema,
                "source_backend": source_backend,
                "deprecated_sequence_backend_data_used": False,
                **prompt,
            }
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    paths.append(str(jsonl_path))

    readme_path = output / "README.md"
    readme_path.write_text(_readme(source_backend, entries), encoding="utf-8")
    paths.append(str(readme_path))
    manifest_path = output / "manifest.json"
    manifest = {
        "schema": LLM_PROMPT_SCHEMA,
        "source_schema": source_schema,
        "source_backend": source_backend,
        "deprecated_sequence_backend_data_used": False,
        "prompt_count": len(prompts),
        "contains_model_outputs": False,
        "total_bandit_turns": int(total_bandit_turns),
        "turns_per_replicate": int(turns_per_replicate),
        "scientific_scope": {
            "alice_network_physics": "empirical_fit_near_term_composite",
            "eve_attack_model": "toy_adversarial_stress",
            "decision_trees": "descriptive_no_generalization_claim",
            "security_claim": "none",
        },
        "prompts": entries,
        "files": [Path(path).name for path in paths],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths.append(str(manifest_path))
    return {"manifest": manifest, "paths": paths}


def _build_prompts(
    tables: dict[str, list[dict[str, Any]]],
    *,
    tree_bundle: PaperDecisionTreeBundle,
    source_backend: str,
    total_bandit_turns: int,
    turns_per_replicate: int,
) -> list[dict[str, Any]]:
    common = _common_scope(
        source_backend,
        total_bandit_turns=total_bandit_turns,
        turns_per_replicate=turns_per_replicate,
    )
    case_rows = tables["case_summary"]
    tree_context = {
        name: {
            "question": artifact["question"],
            "sample_count": artifact["sample_count"],
            "independent_topology_count": artifact["independent_topology_count"],
            "metrics": artifact["metrics"],
            "feature_importances": artifact["feature_importances"],
            "rules_text": artifact["rules_text"],
        }
        for name, artifact in tree_bundle.artifacts.items()
    }
    prompts: list[dict[str, Any]] = [
        {
            "prompt_id": "cross_case_results_explainer",
            "purpose": "Explain the four-case histogram co-learning result without overclaiming.",
            "prompt": f"""{common}

You are reviewing a short quantum-network routing/control experiment. Explain the
main empirical result for a quantum-networking coauthor in 500--700 words.

CASE SUMMARY (mean and sample SD across three learner seeds):
{_json(case_rows)}

DESCRIPTIVE DECISION-TREE DIAGNOSTICS:
{_json(tree_context)}

Address: (1) why the disjoint 3- and 5-hop cases remain close to utility 0.5,
(2) what changes in the mixed bottleneck, (3) why the clean 15-hop service model
can be Bell-qualified while adversarial accepted rate and utility collapse on the
strongly overlapping long case, and (4) what the exploratory tree diagnostics do
and do not add. Separate direct observations from interpretation. Do not call the
15-hop result a failed physical repeater chain when its degradation is caused by
the declared shared-corridor attack game. Do not claim a security proof.
""",
        },
        {
            "prompt_id": "near_term_network_design_interpreter",
            "purpose": "Interpret topology and route-design implications of the short run.",
            "prompt": f"""{common}

Act as a cautious near-term quantum-network design reviewer. Based only on the
evidence below, write a structured design interpretation, not a universal
recommendation.

CASE SUMMARY:
{_json(case_rows)}

ROUTE CALIBRATION:
{_json(tables['route_calibration'])}

DECISION-TREE RULES:
{_json(tree_context)}

Discuss path diversity, shared attack surfaces, hop count, route service rate,
expected CHSH S, expected QBER, and learned routing mass. State explicitly that
only four fixed Topology Bench assignments were sampled, so topology effects are
confounded with assignment identity and require a larger corpus run to validate.
""",
        },
        {
            "prompt_id": "alice_false_signal_interpreter",
            "purpose": "Interpret Alice false-signal rate and policy attribution telemetry.",
            "prompt": f"""{common}

Audit Alice's false-feedback telemetry in this histogram-backend experiment.

CASE SUMMARY:
{_json([_false_signal_fields(row) for row in case_rows])}

PER-RUN TELEMETRY:
{_json([_false_signal_fields(row) for row in tables['run_summary']])}

ALICE ROUTE STRATEGIES:
{_json(tables['alice_route_strategy'])}

Explain the difference between false-signal rate over all turns, false-signal
rate conditional on classifiable quality signals, false accepts, false aborts,
reward bias, and the counterfactual policy total-variation metric. Identify
whether false signals appear frequent and whether the measured policy attribution
is practically large in this run. Treat counterfactual policy attribution as a
model diagnostic, not proof of causal identifiability in a deployed network.
""",
        },
        {
            "prompt_id": "methods_and_limitations_reviewer",
            "purpose": "Review methods, evidence boundaries, and next validation steps.",
            "prompt": f"""{common}

You are a skeptical methods reviewer. Produce a concise review with sections for
strengths, threats to validity, reproducibility, and the next experiment.

CASE SUMMARY:
{_json(case_rows)}

RUN SUMMARY ROWS:
{_json(tables['run_summary'])}

TREE MANIFEST:
{_json(tree_bundle.manifest)}

Check the distinction between online learner turns and independent stratified
post-hoc payoff samples; the interpretation of exploitability; seed replication;
the four-assignment sample limit; stationary saturated demand; missing stateful
multi-flow queue/inventory behavior; modeled Eve instruments/control cost; and absence of
authentication, privacy amplification, finite-key analysis, or a security proof.
Do not import results from any earlier SeQUeNCe-backed study.
""",
        },
    ]
    for case in case_rows:
        slug = str(case["case_slug"])
        prompts.append({
            "prompt_id": f"case_{slug}_explainer",
            "purpose": f"Explain the histogram result for {slug}.",
            "case_slug": slug,
            "prompt": _case_prompt(common, slug, case, tables),
        })
    return prompts


def _case_prompt(
    common: str,
    slug: str,
    case: dict[str, Any],
    tables: dict[str, list[dict[str, Any]]],
) -> str:
    route_strategy = _select(tables["alice_route_strategy"], slug)
    calibration = _select(tables["route_calibration"], slug)
    eve_strategy = _select(tables["eve_family_strategy"], slug)
    payoff = _select(tables["payoff_by_attack_family"], slug)
    return f"""{common}

Explain the `{slug}` result to a quantum-networking coauthor in 350--500 words.

CASE SUMMARY:
{_json(case)}

CLEAN ROUTE CALIBRATION:
{_json(calibration)}

ALICE ROUTE STRATEGY BY SEED:
{_json(route_strategy)}

EVE ATTACK-FAMILY STRATEGY BY SEED:
{_json(eve_strategy)}

INDEPENDENT POST-HOC ALICE UTILITY BY ROUTE AND ATTACK FAMILY:
{_json(payoff)}

Distinguish clean physical calibration from observed adversarial outcomes. Explain
the learned strategies, their seed variation, and the most important structural
feature of this case. Mention false-signal telemetry if material. Label statements
as observation or inference, and do not extrapolate this one assignment to its
whole graph family.
"""


def _common_scope(
    backend: str,
    *,
    total_bandit_turns: int,
    turns_per_replicate: int,
) -> str:
    return f"""EVIDENCE CONTRACT
- Source backend: `{backend}` (new compiled histogram transcript backend).
- Deprecated SeQUeNCe backend data used: false. Do not substitute or compare the
  earlier `exp3_dynamic_500k` SeQUeNCe experiment unless separately supplied.
- Budget: {turns_per_replicate:,} simultaneous Alice/Eve bandit turns per replicate;
  {total_bandit_turns:,} turns total across this bundle.
- Alice network model: empirical-fit, near-term-composite Cui/Krutyanskiy rolling
  purified repeater service, evaluated route by route.
- Eve model: explicitly modeled adversarial stress instruments and declared control costs.
- Security claim: none. This is not an authenticated finite-key E91 proof.
- Decision-tree scope: descriptive for four fixed topology assignments. Any
  leave-one-topology-out score is exploratory; no unseen-network validity is claimed.
- Never present modeled Eve outcomes as measurements from a deployed quantum network."""


def _false_signal_fields(row: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "case_slug",
        "seed",
        "alice_false_signal_rate_all_turns",
        "alice_false_signal_rate_all_turns_mean",
        "alice_false_signal_rate_all_turns_sd",
        "alice_false_signal_rate_classifiable",
        "alice_false_signal_rate_classifiable_mean",
        "alice_false_signal_rate_classifiable_sd",
        "alice_false_accept_rate_all_turns",
        "alice_false_accept_rate_all_turns_mean",
        "alice_false_abort_rate_all_turns",
        "alice_false_abort_rate_all_turns_mean",
        "alice_false_signal_reward_bias_per_turn",
        "alice_false_signal_reward_bias_per_turn_mean",
        "alice_false_signal_policy_total_variation",
        "alice_false_signal_policy_total_variation_mean",
        "alice_false_signal_policy_total_variation_sd",
    )
    return {key: row[key] for key in keep if key in row}


def _select(rows: Sequence[dict[str, Any]], slug: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("case_slug")) == slug]


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def _readme(backend: str, entries: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Histogram co-learning LLM prompts",
        "",
        f"These prompts were generated from `{backend}` artifacts only.",
        "They contain evidence and review tasks, not model outputs.",
        "",
        "Every prompt preserves the mixed scope: Alice's route physics is an "
        "empirical-fit near-term composite, while Eve uses an explicit modeled "
        "adversarial stress model. No finite-key or security proof is claimed.",
        "",
        "## Files",
        "",
    ]
    lines.extend(
        f"- `{entry['file']}` — {entry['purpose']}"
        for entry in entries
    )
    lines.extend([
        "- `prompts.jsonl` — the same prompt inputs with machine-readable metadata.",
        "- `manifest.json` — prompt count, source backend, and scope declarations.",
        "",
    ])
    return "\n".join(lines)
