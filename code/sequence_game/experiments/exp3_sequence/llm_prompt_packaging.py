"""Package per-graph LLM interpretation prompts for Exp3 SeQUeNCe runs."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from sequence_game.xai.cross_graph_dt import fit_cross_graph_dt


GRAPH_TREE_KEY = "graph_oracle_retention"
ACTION_TREE_KEY = "action_oracle_eve_strategy_prob"
DENIAL_TREE_KEY = "action_expected_denial_under_oracle_alice"
ROUTE_TREE_KEY = "route_oracle_alice_strategy_prob"
EXP3_GRAPH_TREE_KEY = "graph_exp3_final_retention"
EXP3_ACTION_TREE_KEY = "action_exp3_eve_empirical_prob"
EXP3_ROUTE_TREE_KEY = "route_exp3_alice_empirical_prob"
PROMPT_KINDS = frozenset({"oracle_minimax", "exp3_learned", "both"})


def package_prompts(
        run_dir: Path,
        *,
        out_dir: Path | None = None,
        top_actions: int = 5,
        top_routes: int = 5,
        limit: int | None = None,
        prompt_kind: str = "oracle_minimax",
) -> dict[str, Any]:
    """Write provider-neutral per-graph prompt records for an Exp3 run."""

    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "llm_prompts"
    if top_actions < 1:
        raise ValueError("top_actions must be >= 1")
    if top_routes < 1:
        raise ValueError("top_routes must be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1 when provided")
    if prompt_kind not in PROMPT_KINDS:
        raise ValueError(f"unsupported prompt_kind {prompt_kind!r}")

    corpus = _read_json(run_dir / "corpus.json")
    dt_payload = _read_json(run_dir / "dt" / "dt_payload.json")
    oracle_summary = _read_json_if_exists(run_dir / "oracle_summary.json", {})
    run_summary = _read_json_if_exists(run_dir / "run_summary.json", {})
    config_payload = _read_json_if_exists(run_dir / "config.json", {})

    graphs = list(corpus.get("graphs", []))
    if limit is not None:
        graphs = graphs[:limit]
    if not graphs:
        raise ValueError(f"no graph records found in {run_dir / 'corpus.json'}")

    trees = dict(dt_payload.get("trees", {}))
    graph_rows = list(dt_payload.get("graph_rows", []))
    action_rows = list(dt_payload.get("action_rows", []))
    route_rows = list(dt_payload.get("route_rows", []))

    graph_rows_by_id = {str(row["graph_id"]): row for row in graph_rows}
    action_rows_by_graph = _group_by_graph(action_rows)
    route_rows_by_graph = _group_by_graph(route_rows)
    family_summary = _family_retention_summary(graph_rows, "oracle_retention")
    exp3_family_summary = _family_retention_summary(graph_rows, "exp3_final_retention")

    graph_tree = _tree_context(trees, GRAPH_TREE_KEY)
    action_tree = _tree_context(trees, ACTION_TREE_KEY)
    denial_tree = _tree_context(trees, DENIAL_TREE_KEY)
    route_tree = _tree_context(trees, ROUTE_TREE_KEY)
    exp3_graph_tree = _tree_context_or_fit(
        trees,
        EXP3_GRAPH_TREE_KEY,
        rows=graph_rows,
        target="exp3_final_retention",
        feature_names=graph_tree["feature_names"],
    )
    exp3_action_tree = _tree_context_or_fit(
        trees,
        EXP3_ACTION_TREE_KEY,
        rows=action_rows,
        target="exp3_eve_empirical_prob",
        feature_names=action_tree["feature_names"],
    )
    exp3_route_tree = _tree_context_or_fit(
        trees,
        EXP3_ROUTE_TREE_KEY,
        rows=route_rows,
        target="exp3_alice_empirical_prob",
        feature_names=route_tree["feature_names"],
    )

    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    prompt_kinds = _resolved_prompt_kinds(prompt_kind)
    for index, graph in enumerate(graphs):
        graph_id = str(graph["graph_id"])
        graph_row = graph_rows_by_id.get(graph_id)
        if graph_row is None:
            raise ValueError(f"DT graph row missing for graph_id={graph_id}")
        oracle = dict(oracle_summary.get(graph_id, {}))
        graph_action_rows = action_rows_by_graph.get(graph_id, [])
        graph_route_rows = route_rows_by_graph.get(graph_id, [])
        actions = _select_top_actions(
            graph_action_rows,
            top_actions=top_actions,
        )
        routes = _select_top_routes(
            graph_route_rows,
            top_routes=top_routes,
        )
        route_lookup = {
            str(route.get("route_id")): route
            for route in graph.get("routes", [])
        }
        context = {
            "index": index,
            "prompt_count": len(graphs),
            "graph": graph,
            "graph_row": graph_row,
            "oracle": oracle,
            "run_summary": run_summary,
            "config": config_payload.get("config", config_payload),
            "family_summary": family_summary,
            "exp3_family_summary": exp3_family_summary,
            "graph_tree": graph_tree,
            "action_tree": action_tree,
            "denial_tree": denial_tree,
            "route_tree": route_tree,
            "exp3_graph_tree": exp3_graph_tree,
            "exp3_action_tree": exp3_action_tree,
            "exp3_route_tree": exp3_route_tree,
            "top_actions": actions,
            "top_routes": routes,
            "action_rows": graph_action_rows,
            "route_rows": graph_route_rows,
            "top_actions_limit": top_actions,
            "top_routes_limit": top_routes,
            "route_lookup": route_lookup,
        }
        for kind in prompt_kinds:
            prompt = build_prompt(context, prompt_kind=kind)
            suffix = "" if prompt_kind == "oracle_minimax" else f".{kind}"
            prompt_path = prompt_dir / f"{_safe_filename(graph_id)}{suffix}.txt"
            prompt_path.write_text(prompt, encoding="utf-8")
            record = {
                "id": f"exp3_sequence_graph_interpretation:{kind}:{graph_id}",
                "prompt_kind": kind,
                "graph_id": graph_id,
                "family": graph.get("family"),
                "prompt_path": str(prompt_path),
                "prompt": prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "suggested_parameters": {
                    "temperature": 0.0,
                    "max_output_tokens": 900,
                },
                "scoring_metadata": _scoring_metadata(
                    graph=graph,
                    graph_row=graph_row,
                    oracle=oracle,
                    graph_tree=(
                        graph_tree if kind == "oracle_minimax"
                        else exp3_graph_tree
                    ),
                    actions=actions,
                    routes=routes,
                    prompt_kind=kind,
                ),
            }
            records.append(record)

    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "prompts.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_run_dir": str(run_dir),
        "out_dir": str(out_dir),
        "prompt_count": len(records),
        "prompt_kind": prompt_kind,
        "prompt_kinds": prompt_kinds,
        "jsonl_path": str(jsonl_path),
        "prompt_dir": str(prompt_dir),
        "inputs": {
            "corpus": str(run_dir / "corpus.json"),
            "dt_payload": str(run_dir / "dt" / "dt_payload.json"),
            "oracle_summary": str(run_dir / "oracle_summary.json"),
            "run_summary": str(run_dir / "run_summary.json"),
            "config": str(run_dir / "config.json"),
        },
        "tree_status": {
            key: {
                "target": tree.get("target"),
                "r2_score": tree.get("r2_score"),
                "num_rows": tree.get("num_rows"),
                "num_topologies": tree.get("num_topologies"),
            }
            for key, tree in trees.items()
        },
        "notes": [
            "Provider-neutral package only; this script does not call any API.",
            "Each JSONL record includes prompt text plus OpenAI-compatible messages.",
            "For Gemini, send record['prompt'] as the user content/contents text.",
            "Prompts are post-hoc, in-sample interpretation tasks over cached simulator outputs.",
            "Prompts do not ask the model to claim a cryptographic security proof.",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def build_prompt(
        context: dict[str, Any],
        *,
        prompt_kind: str = "oracle_minimax",
) -> str:
    if prompt_kind == "oracle_minimax":
        return _build_oracle_prompt(context)
    if prompt_kind == "exp3_learned":
        return _build_exp3_prompt(context)
    raise ValueError(f"unsupported prompt_kind {prompt_kind!r}")


def _build_oracle_prompt(context: dict[str, Any]) -> str:
    graph = context["graph"]
    graph_row = context["graph_row"]
    oracle = context["oracle"]
    graph_tree = context["graph_tree"]
    action_tree = context["action_tree"]
    denial_tree = context["denial_tree"]
    route_tree = context["route_tree"]
    family_summary = context["family_summary"]
    config = context["config"]
    run_summary = context["run_summary"]

    graph_id = str(graph["graph_id"])
    graph_values = _feature_map(graph_row, graph_tree["feature_names"])
    graph_leaf = _matching_leaf(graph_tree, graph_values)
    retention = graph_row.get("oracle_retention")
    family = str(graph.get("family"))
    family_stats = family_summary.get(family, {})

    top_action_blocks = []
    for action in context["top_actions"]:
        action_values = _feature_map(action, action_tree["feature_names"])
        denial_values = _feature_map(action, denial_tree["feature_names"])
        top_action_blocks.append(_format_action_block(
            action,
            probability_key="oracle_eve_strategy_prob",
            probability_label="oracle_eve_strategy_prob",
            selected_label="selected/high-probability oracle action",
            diagnostic_label="diagnostic comparison action",
            action_tree_leaf=_matching_leaf(action_tree, action_values),
            denial_tree_leaf=_matching_leaf(denial_tree, denial_values),
        ))

    top_route_blocks = []
    for route in context["top_routes"]:
        route_id = str(route.get("route_id"))
        route_values = _feature_map(route, route_tree["feature_names"])
        top_route_blocks.append(_format_route_block(
            route,
            route_detail=context["route_lookup"].get(route_id, {}),
            probability_key="oracle_alice_strategy_prob",
            probability_label="oracle_alice_strategy_prob",
            selected_label="selected/high-probability oracle route",
            diagnostic_label="diagnostic comparison route",
            route_tree_leaf=_matching_leaf(route_tree, route_values),
        ))

    edge_lines = _format_edges(graph.get("topology", {}).get("edges", []))
    graph_rule_text = _ascii(str(graph_tree.get("rules_text") or ""))

    return f"""You are interpreting one graph from a SeQUeNCe-based adversarial quantum-network game.

This is an oracle/minimax interpretation prompt. Interpret only the complete-information
minimax and DT evidence below. Do not claim a cryptographic security proof,
deployment guarantee, unmeasured physics, or guarantees beyond these cached
simulator outputs.

## Requested output
Write a concise Markdown response with exactly these sections:
1. **Graph interpretation**: explain the oracle/minimax vulnerability or resilience of this graph.
2. **Decision-tree evidence**: explain the active DT evidence in plain English.
3. **Eve/Alice strategy evidence**: explain selected/high-probability Eve actions and Alice routes, while keeping diagnostic rows separate.
4. **Caveats**: mention DT fidelity/R2, post-hoc in-sample scope, and no cryptographic security proof.
5. **Rubric self-check**: one line each for target awareness, path faithfulness, feature interpretation, and no hallucinated science.

Keep the response under 300 words. Use only facts, labels, and numbers present
in this prompt. If evidence is weak, say it is weak.
When a direct strategy probability and a DT leaf prediction disagree, report
the disagreement instead of forcing them to agree.
Rows with strategy probability > 0 are selected/high-probability rows. Rows
with strategy probability = 0 are diagnostic comparison rows and must not be
described as selected actions or selected routes.

## Experiment scope
- Backend: {run_summary.get("backend", "unknown")}
- Security monitor: {run_summary.get("security_monitor", config.get("security_monitor", "unknown"))}
- Trials per payoff cell: {run_summary.get("trials_per_cell", config.get("trials_per_cell", "unknown"))}
- Online turns per condition: {run_summary.get("online_turns", config.get("online_turns", "unknown"))}
- Interpretation scope: post-hoc, in-sample DT explanation over cached simulator outputs; not a cryptographic security proof.

## Graph identity
- Prompt index: {context["index"] + 1} of {context["prompt_count"]}
- graph_id: {graph_id}
- family: {family}
- alice: {graph.get("alice")}
- bob: {graph.get("bob")}
- route_count: {len(graph.get("routes", []))}
- oracle_status: {oracle.get("status", graph_row.get("oracle_status", "unknown"))}
- oracle_retention target value: {_fmt(retention)}
- oracle_value: {_fmt(oracle.get("value", graph_row.get("oracle_value")))}
- baseline_rate: {_fmt(oracle.get("baseline_rate", graph_row.get("baseline_rate")))}

## Family context
- family_count: {_fmt(family_stats.get("count"))}
- family_retention_mean: {_fmt(family_stats.get("mean_retention"))}
- family_retention_min: {_fmt(family_stats.get("min_retention"))}
- family_retention_max: {_fmt(family_stats.get("max_retention"))}

## Graph structural features
{_format_feature_lines(graph_values)}

## Compact topology edges
{edge_lines}

## Graph retention decision tree
- target: {graph_tree.get("target")}
- R2: {_fmt(graph_tree.get("r2_score"))}
- rows/topologies: {_fmt(graph_tree.get("num_rows"))}/{_fmt(graph_tree.get("num_topologies"))}
- active leaf for this graph: {_format_leaf(graph_leaf)}

Full graph-retention DT rules:
{graph_rule_text}

## Eve action decision-tree evidence
- Eve action tree target: {action_tree.get("target")} ; R2={_fmt(action_tree.get("r2_score"))}
- Expected denial tree target: {denial_tree.get("target")} ; R2={_fmt(denial_tree.get("r2_score"))}
- The rows below are the top actions for this graph by oracle Eve probability, falling back to expected denial when needed.

{chr(10).join(top_action_blocks) or "(no action rows available)"}

## Alice route decision-tree evidence
- Alice route tree target: {route_tree.get("target")} ; R2={_fmt(route_tree.get("r2_score"))}
- {_r2_caution(route_tree, "Alice route tree")}
- The rows below are the top routes for this graph by oracle Alice probability, falling back to clean/worst-case key-rate evidence when needed.

{chr(10).join(top_route_blocks) or "(no route rows available)"}

## Target definitions
- oracle_retention: oracle minimax key-rate value divided by the no-attack baseline rate.
- oracle_eve_strategy_prob: probability assigned to an Eve action by the complete-information minimax oracle.
- action_expected_denial_under_oracle_alice: 1 minus expected key rate when this action is evaluated against oracle Alice.
- oracle_alice_strategy_prob: probability assigned to a route by the complete-information minimax oracle.
"""


def _build_exp3_prompt(context: dict[str, Any]) -> str:
    graph = context["graph"]
    graph_row = context["graph_row"]
    graph_tree = context["exp3_graph_tree"]
    action_tree = context["exp3_action_tree"]
    route_tree = context["exp3_route_tree"]
    family_summary = context["exp3_family_summary"]
    config = context["config"]
    run_summary = context["run_summary"]

    graph_id = str(graph["graph_id"])
    graph_values = _feature_map(graph_row, graph_tree["feature_names"])
    graph_leaf = _matching_leaf(graph_tree, graph_values)
    retention = graph_row.get("exp3_final_retention")
    family = str(graph.get("family"))
    family_stats = family_summary.get(family, {})

    top_action_blocks = []
    for action in _select_top_actions(
        context["action_rows"],
        top_actions=context["top_actions_limit"],
        probability_key="exp3_eve_empirical_prob",
        denial_key="active_hit_accepted_rate",
    ):
        action_values = _feature_map(action, action_tree["feature_names"])
        top_action_blocks.append(_format_action_block(
            action,
            probability_key="exp3_eve_empirical_prob",
            probability_label="learned_eve_strategy_probability",
            selected_label="selected/high-probability learned Eve action",
            diagnostic_label="diagnostic comparison action",
            action_tree_leaf=_matching_leaf(action_tree, action_values),
            denial_tree_leaf=None,
        ))

    top_route_blocks = []
    for route in _select_top_routes(
        context["route_rows"],
        top_routes=context["top_routes_limit"],
        probability_key="exp3_alice_empirical_prob",
    ):
        route_id = str(route.get("route_id"))
        route_values = _feature_map(route, route_tree["feature_names"])
        top_route_blocks.append(_format_route_block(
            route,
            route_detail=context["route_lookup"].get(route_id, {}),
            probability_key="exp3_alice_empirical_prob",
            probability_label="learned_alice_strategy_probability",
            selected_label="selected/high-probability learned Alice route",
            diagnostic_label="diagnostic comparison route",
            route_tree_leaf=_matching_leaf(route_tree, route_values),
        ))

    edge_lines = _format_edges(graph.get("topology", {}).get("edges", []))
    graph_rule_text = _ascii(str(graph_tree.get("rules_text") or ""))

    return f"""You are interpreting one graph from a SeQUeNCe-based adversarial quantum-network game.

This is a learned-strategy Exp3 interpretation prompt. Interpret only the learned time-averaged Exp3 strategy, empirical strategy probabilities, exploitability
diagnostic, and DT evidence below. Do not claim a cryptographic security proof,
deployment guarantee, unmeasured physics, or guarantees beyond these cached
simulator outputs.

## Requested output
Write a concise Markdown response with exactly these sections:
1. **Graph interpretation**: explain the learned Exp3 behavior on this graph.
2. **Decision-tree evidence**: explain the active DT evidence in plain English.
3. **Eve/Alice learned strategy evidence**: explain selected/high-probability learned Eve actions and Alice routes, while keeping diagnostic rows separate.
4. **Caveats**: mention DT fidelity/R2, post-hoc in-sample scope, and no cryptographic security proof.
5. **Rubric self-check**: one line each for target awareness, path faithfulness, feature interpretation, and no hallucinated science.

Keep the response under 300 words. Use only facts, labels, and numbers present
in this prompt. If evidence is weak, say it is weak.
When a direct strategy probability and a DT leaf prediction disagree, report
the disagreement instead of forcing them to agree.
Rows with strategy probability > 0 are selected/high-probability rows. Rows
with strategy probability = 0 are diagnostic comparison rows and must not be
described as selected actions or selected routes.

## Experiment scope
- Backend: {run_summary.get("backend", "unknown")}
- Security monitor: {run_summary.get("security_monitor", config.get("security_monitor", "unknown"))}
- Trials per payoff cell: {run_summary.get("trials_per_cell", config.get("trials_per_cell", "unknown"))}
- Online turns per condition: {run_summary.get("online_turns", config.get("online_turns", "unknown"))}
- Interpretation scope: post-hoc, in-sample DT explanation over cached simulator outputs; not a cryptographic security proof.

## Graph identity
- Prompt index: {context["index"] + 1} of {context["prompt_count"]}
- graph_id: {graph_id}
- family: {family}
- alice: {graph.get("alice")}
- bob: {graph.get("bob")}
- route_count: {len(graph.get("routes", []))}
- learned_final_retention target value: {_fmt(retention)}
- learned_strategy_exploitability_diagnostic: {_fmt(graph_row.get("exp3_exploitability_vs_oracle"))}

## Learned family context
- family_count: {_fmt(family_stats.get("count"))}
- learned_family_retention_mean: {_fmt(family_stats.get("mean_retention"))}
- learned_family_retention_min: {_fmt(family_stats.get("min_retention"))}
- learned_family_retention_max: {_fmt(family_stats.get("max_retention"))}

## Graph structural features
{_format_feature_lines(_feature_map(graph_row, graph_tree["feature_names"]))}

## Compact topology edges
{edge_lines}

## Learned final-retention decision tree
- target: learned_final_retention
- R2: {_fmt(graph_tree.get("r2_score"))}
- rows/topologies: {_fmt(graph_tree.get("num_rows"))}/{_fmt(graph_tree.get("num_topologies"))}
- active leaf for this graph: {_format_leaf(graph_leaf)}

Full learned-final-retention DT rules:
{graph_rule_text}

## Eve learned-strategy decision-tree evidence
- Eve learned-strategy tree target: learned_eve_strategy_probability ; R2={_fmt(action_tree.get("r2_score"))}
- The rows below are top learned Eve empirical strategy rows for this graph. Zero-probability rows are diagnostic comparison rows only.

{chr(10).join(top_action_blocks) or "(no action rows available)"}

## Alice learned-route decision-tree evidence
- Alice learned-route tree target: learned_alice_strategy_probability ; R2={_fmt(route_tree.get("r2_score"))}
- {_r2_caution(route_tree, "Alice learned-route tree")}
- The rows below are top learned Alice empirical strategy rows for this graph. Zero-probability rows are diagnostic comparison rows only.

{chr(10).join(top_route_blocks) or "(no route rows available)"}

## Target definitions
- learned_final_retention: final-window retention reported by the Exp3-vs-Exp3 run for this graph.
- learned_eve_strategy_probability: time-averaged empirical probability assigned to an Eve action by the learned Exp3 strategy.
- learned_alice_strategy_probability: time-averaged empirical probability assigned to a route by the learned Exp3 strategy.
- learned_strategy_exploitability_diagnostic: empirical Nash-gap/exploitability diagnostic for the learned profile on the payoff matrix; lower means closer to equilibrium.
"""


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return _read_json(path)


def _group_by_graph(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["graph_id"])].append(row)
    return dict(grouped)


def _tree_context(
        trees: dict[str, Any],
        key: str,
) -> dict[str, Any]:
    tree = dict(trees.get(key) or {})
    tree["key"] = key
    tree["feature_names"] = list(tree.get("feature_names", []))
    tree["leaves"] = _parse_rules(str(tree.get("rules_text") or ""))
    return tree


def _tree_context_or_fit(
        trees: dict[str, Any],
        key: str,
        *,
        rows: list[dict[str, Any]],
        target: str,
        feature_names: list[str],
) -> dict[str, Any]:
    tree = trees.get(key)
    if tree:
        return _tree_context(trees, key)
    valid = [row for row in rows if row.get(target) is not None]
    if not valid or not feature_names:
        return _tree_context({
            key: {
                "target": target,
                "feature_names": feature_names,
                "num_rows": len(valid),
                "num_topologies": 0,
                "r2_score": None,
                "rules_text": "",
            }
        }, key)
    result = fit_cross_graph_dt(
        valid,
        max_depth=3,
        target_key=target,
        feature_names=feature_names,
    )
    return _tree_context({
        key: {
            "target": target,
            "feature_names": feature_names,
            "num_rows": result.num_rows,
            "num_topologies": result.num_topologies,
            "r2_score": result.r2_score,
            "rules_text": result.rules_text,
        }
    }, key)


def _parse_rules(rules_text: str) -> list[dict[str, Any]]:
    leaves = []
    for block in re.split(r"\n\s*\n", rules_text.strip()):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines or not lines[0].startswith("IF "):
            continue
        condition_text = lines[0][3:].strip()
        conditions = []
        if condition_text != "TRUE":
            for condition in condition_text.split(" AND "):
                match = re.match(r"^(.+?)\s*(<=|>)\s*([-+0-9.eE]+)$", condition)
                if not match:
                    continue
                conditions.append({
                    "feature": match.group(1).strip(),
                    "op": match.group(2),
                    "threshold": float(match.group(3)),
                    "text": condition.strip(),
                })
        value_match = re.search(
            r"avg_[A-Za-z0-9_]+\s*=\s*([-+0-9.eE]+).*?\(n=(\d+)\)",
            " ".join(lines[1:]),
        )
        leaves.append({
            "conditions": conditions,
            "prediction": (
                float(value_match.group(1)) if value_match else None
            ),
            "sample_count": (
                int(value_match.group(2)) if value_match else None
            ),
            "text": _ascii(block),
        })
    return leaves


def _matching_leaf(
        tree: dict[str, Any],
        feature_values: dict[str, Any],
) -> dict[str, Any] | None:
    for leaf in tree.get("leaves", []):
        if all(_condition_matches(condition, feature_values)
               for condition in leaf.get("conditions", [])):
            return leaf
    return None


def _condition_matches(
        condition: dict[str, Any],
        feature_values: dict[str, Any],
) -> bool:
    value = feature_values.get(condition["feature"])
    if value is None:
        return False
    value = float(value)
    threshold = float(condition["threshold"])
    if condition["op"] == "<=":
        return value <= threshold
    return value > threshold


def _feature_map(row: dict[str, Any], feature_names: list[str]) -> dict[str, Any]:
    features = list(row.get("features", []))
    return {
        name: features[index]
        for index, name in enumerate(feature_names)
        if index < len(features)
    }


def _family_retention_summary(
        graph_rows: list[dict[str, Any]],
        value_key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in graph_rows:
        value = row.get(value_key)
        if value is not None:
            grouped[str(row.get("family", "unknown"))].append(float(value))
    return {
        family: {
            "count": len(values),
            "mean_retention": mean(values),
            "min_retention": min(values),
            "max_retention": max(values),
        }
        for family, values in grouped.items()
        if values
    }


def _select_top_actions(
        rows: list[dict[str, Any]],
        *,
        top_actions: int,
        probability_key: str = "oracle_eve_strategy_prob",
        denial_key: str = "action_expected_denial_under_oracle_alice",
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row.get(probability_key) or 0.0),
            -float(row.get(denial_key) or 0.0),
            str(row.get("action_id", "")),
        ),
    )
    return ordered[:top_actions]


def _select_top_routes(
        rows: list[dict[str, Any]],
        *,
        top_routes: int,
        probability_key: str = "oracle_alice_strategy_prob",
) -> list[dict[str, Any]]:
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row.get(probability_key) or 0.0),
            -float(row.get("no_attack_key_rate") or 0.0),
            float(row.get("worst_case_key_rate") or 0.0),
            str(row.get("route_id", "")),
        ),
    )
    return ordered[:top_routes]


def _format_action_block(
        action: dict[str, Any],
        *,
        probability_key: str,
        probability_label: str,
        selected_label: str,
        diagnostic_label: str,
        action_tree_leaf: dict[str, Any] | None,
        denial_tree_leaf: dict[str, Any] | None,
) -> str:
    feature_names = [
        "is_no_attack",
        "is_edge_intercept_resend",
        "is_memory_degradation",
        "target_route_coverage",
        "target_is_bottleneck",
        "num_routes",
        "node_disjoint_paths",
    ]
    features = _feature_map(action, feature_names)
    probability = float(action.get(probability_key) or 0.0)
    role = selected_label if probability > 0.0 else diagnostic_label
    denial_line = (
        f"\n  denial DT leaf: {_format_leaf(denial_tree_leaf)}"
        if denial_tree_leaf is not None else ""
    )
    return (
        f"- action_id={action.get('action_id')} ; "
        f"row_role={role} ; "
        f"{probability_label}={_fmt(action.get(probability_key))} ; "
        f"expected_denial={_fmt(action.get('action_expected_denial_under_oracle_alice'))} ; "
        f"active_hit_sample_count={_fmt(action.get('active_hit_sample_count'))} ; "
        f"active_hit_accepted_rate={_fmt(action.get('active_hit_accepted_rate'))} ; "
        f"active_hit_mean_chsh_s={_fmt(action.get('active_hit_mean_chsh_s'))} ; "
        f"active_hit_mean_qber={_fmt(action.get('active_hit_mean_qber'))}\n"
        f"  features: {_format_inline_features(features)}\n"
        f"  strategy-probability DT leaf: {_format_leaf(action_tree_leaf)}"
        f"{denial_line}"
    )


def _format_route_block(
        route: dict[str, Any],
        *,
        route_detail: dict[str, Any],
        probability_key: str,
        probability_label: str,
        selected_label: str,
        diagnostic_label: str,
        route_tree_leaf: dict[str, Any] | None,
) -> str:
    feature_names = [
        "hop_count",
        "length_m",
        "length_over_shortest",
        "internal_node_count",
        "contains_bottleneck",
        "node_disjoint_paths",
        "mean_overlap_with_other_routes",
    ]
    features = _feature_map(route, feature_names)
    path = " -> ".join(str(node) for node in route_detail.get("path", []))
    probability = float(route.get(probability_key) or 0.0)
    role = selected_label if probability > 0.0 else diagnostic_label
    return (
        f"- route_id={route.get('route_id')} ; "
        f"row_role={role} ; "
        f"path={path or 'unknown'} ; "
        f"{probability_label}={_fmt(route.get(probability_key))} ; "
        f"no_attack_key_rate={_fmt(route.get('no_attack_key_rate'))} ; "
        f"worst_case_key_rate={_fmt(route.get('worst_case_key_rate'))}\n"
        f"  features: {_format_inline_features(features)}\n"
        f"  route DT leaf: {_format_leaf(route_tree_leaf)}"
    )


def _format_edges(edges: list[dict[str, Any]]) -> str:
    if not edges:
        return "- (no edge list available)"
    lines = []
    for edge in edges:
        lines.append(
            "- "
            f"{edge.get('edge_id')}: {edge.get('u')}--{edge.get('v')} "
            f"length_m={_fmt(edge.get('length_m'))} "
            f"eve_eligible={edge.get('eve_eligible')}"
        )
    return "\n".join(lines)


def _format_feature_lines(features: dict[str, Any]) -> str:
    return "\n".join(
        f"- {name}: {_fmt(value)}"
        for name, value in sorted(features.items())
    )


def _format_inline_features(features: dict[str, Any]) -> str:
    return ", ".join(
        f"{name}={_fmt(value)}"
        for name, value in features.items()
    )


def _format_leaf(leaf: dict[str, Any] | None) -> str:
    if leaf is None:
        return "no matching leaf found"
    conditions = leaf.get("conditions", [])
    condition_text = (
        "TRUE"
        if not conditions else
        " AND ".join(str(condition["text"]) for condition in conditions)
    )
    return (
        f"IF {condition_text} -> predicted_avg={_fmt(leaf.get('prediction'))} "
        f"(training_rows={_fmt(leaf.get('sample_count'))})"
    )


def _r2_caution(tree: dict[str, Any], label: str) -> str:
    r2 = tree.get("r2_score")
    if r2 is None:
        return f"{label} R2 is unavailable; treat this tree as diagnostic only."
    if float(r2) < 0.5:
        return (
            f"{label} R2 is {_fmt(r2)}, so treat this route-tree evidence "
            "as weak/low-fidelity and report uncertainty explicitly."
        )
    return f"{label} R2 is {_fmt(r2)}; report this DT fidelity with the caveats."


def _resolved_prompt_kinds(prompt_kind: str) -> list[str]:
    if prompt_kind == "both":
        return ["oracle_minimax", "exp3_learned"]
    return [prompt_kind]


def _scoring_metadata(
        *,
        graph: dict[str, Any],
        graph_row: dict[str, Any],
        oracle: dict[str, Any],
        graph_tree: dict[str, Any],
        actions: list[dict[str, Any]],
        routes: list[dict[str, Any]],
        prompt_kind: str,
) -> dict[str, Any]:
    graph_values = _feature_map(graph_row, graph_tree["feature_names"])
    active_leaf = _matching_leaf(graph_tree, graph_values)
    if prompt_kind == "oracle_minimax":
        action_probability_key = "oracle_eve_strategy_prob"
        route_probability_key = "oracle_alice_strategy_prob"
        required_target_terms = [
            "oracle_retention",
            "oracle_eve_strategy_prob",
            "oracle_alice_strategy_prob",
        ]
    else:
        action_probability_key = "exp3_eve_empirical_prob"
        route_probability_key = "exp3_alice_empirical_prob"
        required_target_terms = [
            "learned_final_retention",
            "learned_eve_strategy_probability",
            "learned_alice_strategy_probability",
            "empirical Nash-gap",
        ]
    return {
        "graph_id": graph.get("graph_id"),
        "family": graph.get("family"),
        "prompt_kind": prompt_kind,
        "oracle_status": oracle.get("status"),
        "oracle_retention": graph_row.get("oracle_retention"),
        "graph_tree_active_leaf": _format_leaf(active_leaf),
        "required_target_terms": required_target_terms,
        "high_probability_action_ids": [
            row.get("action_id")
            for row in actions
            if float(row.get(action_probability_key) or 0.0) > 0.0
        ],
        "high_probability_route_ids": [
            row.get("route_id")
            for row in routes
            if float(row.get(route_probability_key) or 0.0) > 0.0
        ],
        "forbidden_claim_examples": [
            "proves security",
            "real-world guarantee",
            "physically validated",
        ],
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "prompt"


def _ascii(text: str) -> str:
    return (
        text
        .replace("\u2192", "->")
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )
