"""Alice/Bob assignment policies for graph-corpus records."""

from __future__ import annotations

import itertools
import random
import re
from collections import Counter
from dataclasses import dataclass, replace
from typing import Optional

from sequence_game.topology import TopologyIR

from .metrics import (
    RouteStructureRecord,
    all_pairs_shortest_path_data,
    analyze_route_structure,
    connected_components,
    single_source_shortest_path_data,
)

VALID_MULTI_ROUTE = "valid_multi_route"
DEGENERATE_SINGLE_ROUTE = "degenerate_single_route"
DEGENERATE_NODE_BOTTLENECK = "degenerate_node_bottleneck"
DEGENERATE_EDGE_BOTTLENECK = "degenerate_edge_bottleneck"
INVALID_DISCONNECTED = "invalid_disconnected"

POLICY_BEST_ROUTE_DIVERSE = "best_route_diverse"
POLICY_STRONGEST_NODE_BOTTLENECK = "strongest_node_bottleneck"
POLICY_STRONGEST_EDGE_BOTTLENECK = "strongest_edge_bottleneck"
POLICY_LONG_DISTANCE_PAIR = "long_distance_pair"
POLICY_RANDOM_VALID_PAIR = "random_valid_pair"
POLICY_GENERATOR_PREFERRED_PAIR = "generator_preferred_pair"
POLICY_TOP_SAMPLED_PAIR = "top_sampled_pair"

LARGE_GRAPH_SELECTION_POLICY = "policy"
LARGE_GRAPH_SELECTION_TOP_SAMPLED = "top_sampled"

DEFAULT_ASSIGNMENT_POLICIES = (
    POLICY_GENERATOR_PREFERRED_PAIR,
    POLICY_BEST_ROUTE_DIVERSE,
    POLICY_STRONGEST_NODE_BOTTLENECK,
    POLICY_STRONGEST_EDGE_BOTTLENECK,
    POLICY_LONG_DISTANCE_PAIR,
    POLICY_RANDOM_VALID_PAIR,
)


@dataclass(frozen=True)
class AssignmentSearchConfig:
    max_assignments_per_graph: int = 6
    assignment_policies: tuple[str, ...] = DEFAULT_ASSIGNMENT_POLICIES
    include_degenerate_controls: bool = True
    exclude_direct_links: bool = True
    min_ab_hops: int = 2
    max_route_candidates: int = 128
    route_probe_limit: int = 20_000
    exact_hitting_set_max_size: int = 4
    large_graph_node_threshold: int = 16
    large_graph_candidate_node_limit: int = 32
    max_pair_evaluations_large: int = 96
    large_graph_selection_mode: str = LARGE_GRAPH_SELECTION_POLICY
    large_graph_finalist_count: int = 32
    random_candidate_nodes_large: int = 12
    landmark_count_large: int = 8
    max_graph_nodes_for_analysis: int = 10_000
    random_seed: int = 20260627


@dataclass(frozen=True)
class PairEvaluation:
    alice: str
    bob: str
    route_structure: RouteStructureRecord
    score: tuple


@dataclass(frozen=True)
class AssignmentBuildResult:
    assignments: list["GraphAssignmentRecord"]
    assignment_search_mode: str
    pair_evaluations: int
    candidate_node_count: int
    skipped_reason: Optional[str] = None


@dataclass(frozen=True)
class GraphAssignmentRecord:
    assignment_id: str
    graph_id: str
    alice: str
    bob: str
    assignment_policy: str
    strategy_validity: str
    route_count: int
    shortest_hops: Optional[int]
    longest_hops: Optional[int]
    shortest_path_count: int
    node_cut_size: Optional[int]
    edge_cut_size: Optional[int]
    candidate_node_hitting_size: Optional[int]
    candidate_edge_hitting_size: Optional[int]
    route_overlap_summary: dict
    assignment_search_mode: str
    pair_evaluations: int
    candidate_node_count: int
    route_enumeration_truncated: bool
    route_structure: RouteStructureRecord
    assignment_path: str = ""

    def with_assignment_path(self, assignment_path: str) -> "GraphAssignmentRecord":
        return replace(self, assignment_path=assignment_path)

    def to_summary_row(self) -> dict:
        return {
            "assignment_id": self.assignment_id,
            "graph_id": self.graph_id,
            "alice": self.alice,
            "bob": self.bob,
            "assignment_policy": self.assignment_policy,
            "strategy_validity": self.strategy_validity,
            "route_count": self.route_count,
            "shortest_hops": self.shortest_hops,
            "longest_hops": self.longest_hops,
            "shortest_path_count": self.shortest_path_count,
            "node_cut_size": self.node_cut_size,
            "edge_cut_size": self.edge_cut_size,
            "candidate_node_hitting_size": self.candidate_node_hitting_size,
            "candidate_edge_hitting_size": self.candidate_edge_hitting_size,
            "route_overlap_summary": self.route_overlap_summary,
            "assignment_search_mode": self.assignment_search_mode,
            "pair_evaluations": self.pair_evaluations,
            "candidate_node_count": self.candidate_node_count,
            "route_enumeration_truncated": self.route_enumeration_truncated,
            "assignment_path": self.assignment_path,
        }

    def to_dict(self) -> dict:
        return {
            **self.to_summary_row(),
            "route_structure": self.route_structure.to_dict(),
        }


def build_assignments_for_graph(
    graph: TopologyIR,
    graph_id: str,
    config: AssignmentSearchConfig,
) -> AssignmentBuildResult:
    if len(graph.nodes) > config.max_graph_nodes_for_analysis:
        return AssignmentBuildResult(
            assignments=[],
            assignment_search_mode="skipped",
            pair_evaluations=0,
            candidate_node_count=0,
            skipped_reason=(
                f"graph has {len(graph.nodes)} nodes, above "
                f"max_graph_nodes_for_analysis={config.max_graph_nodes_for_analysis}"
            ),
        )

    pairs, search_mode, candidate_node_count = _candidate_pairs(graph, config)
    preferred = _preferred_pair(graph)
    if preferred is not None and preferred not in pairs and preferred[::-1] not in pairs:
        pairs.insert(0, preferred)

    shortest_data_by_source = (
        all_pairs_shortest_path_data(graph)
        if search_mode == "exhaustive"
        else {}
    )
    analysis_pairs = (
        _large_graph_finalist_pairs(graph, pairs, config)
        if search_mode == "sampled"
        and config.large_graph_selection_mode == LARGE_GRAPH_SELECTION_TOP_SAMPLED
        else pairs
    )
    evaluations = []
    shortest_cache: dict[str, dict[str, tuple[int, int]]] = {}
    for alice, bob in analysis_pairs:
        shortest_data = shortest_data_by_source.get(alice) or shortest_cache.get(alice)
        if shortest_data is None:
            shortest_data = single_source_shortest_path_data(graph, alice)
            shortest_cache[alice] = shortest_data
        route_structure = analyze_route_structure(
            graph,
            alice,
            bob,
            max_route_candidates=config.max_route_candidates,
            route_probe_limit=config.route_probe_limit,
            exact_hitting_set_max_size=config.exact_hitting_set_max_size,
            shortest_data=shortest_data,
        )
        evaluations.append(
            PairEvaluation(
                alice=alice,
                bob=bob,
                route_structure=route_structure,
                score=_route_diversity_score(route_structure, alice, bob),
            )
        )

    pair_evaluations = len(pairs) if search_mode == "sampled" else len(evaluations)
    if (
        search_mode == "sampled"
        and config.large_graph_selection_mode == LARGE_GRAPH_SELECTION_TOP_SAMPLED
    ):
        selected = _select_top_sampled_assignments(
            graph_id,
            evaluations,
            config=config,
            search_mode=search_mode,
            pair_evaluations=pair_evaluations,
            candidate_node_count=candidate_node_count,
        )
    else:
        selected = _select_policy_assignments(
            graph_id,
            evaluations,
            preferred_pair=preferred,
            config=config,
            search_mode=search_mode,
            pair_evaluations=pair_evaluations,
            candidate_node_count=candidate_node_count,
        )
    if not selected:
        return AssignmentBuildResult(
            assignments=[],
            assignment_search_mode=search_mode,
            pair_evaluations=pair_evaluations,
            candidate_node_count=candidate_node_count,
            skipped_reason="no Alice/Bob assignment survived configured validity filters",
        )
    return AssignmentBuildResult(
        assignments=selected,
        assignment_search_mode=search_mode,
        pair_evaluations=pair_evaluations,
        candidate_node_count=candidate_node_count,
    )


def classify_strategy_validity(
    route_structure: RouteStructureRecord,
    *,
    assignment_policy: Optional[str] = None,
) -> str:
    if route_structure.route_count == 0:
        return INVALID_DISCONNECTED
    if route_structure.route_count == 1:
        return DEGENERATE_SINGLE_ROUTE
    if (
        assignment_policy == POLICY_STRONGEST_EDGE_BOTTLENECK
        and _cut_is_single_or_zero(route_structure.edge_cut_size)
    ):
        return DEGENERATE_EDGE_BOTTLENECK
    if _cut_is_single_or_zero(route_structure.node_cut_size):
        return DEGENERATE_NODE_BOTTLENECK
    if _cut_is_single_or_zero(route_structure.edge_cut_size):
        return DEGENERATE_EDGE_BOTTLENECK
    return VALID_MULTI_ROUTE


def _select_policy_assignments(
    graph_id: str,
    evaluations: list[PairEvaluation],
    *,
    preferred_pair: Optional[tuple[str, str]],
    config: AssignmentSearchConfig,
    search_mode: str,
    pair_evaluations: int,
    candidate_node_count: int,
) -> list[GraphAssignmentRecord]:
    assignments = []
    used_assignment_ids = set()
    eligible = [
        evaluation
        for evaluation in evaluations
        if _eligible_for_assignment(evaluation.route_structure, config)
    ]
    for policy in config.assignment_policies:
        if policy == POLICY_GENERATOR_PREFERRED_PAIR:
            selected = _select_preferred(eligible, preferred_pair)
        elif policy == POLICY_BEST_ROUTE_DIVERSE:
            selected = _select_best_route_diverse(eligible)
        elif policy == POLICY_STRONGEST_NODE_BOTTLENECK:
            selected = _select_strongest_node_bottleneck(eligible)
        elif policy == POLICY_STRONGEST_EDGE_BOTTLENECK:
            selected = _select_strongest_edge_bottleneck(eligible)
        elif policy == POLICY_LONG_DISTANCE_PAIR:
            selected = _select_long_distance_pair(eligible)
        elif policy == POLICY_RANDOM_VALID_PAIR:
            selected = _select_random_valid_pair(eligible, config.random_seed, graph_id)
        else:
            continue
        if selected is None:
            continue
        validity = classify_strategy_validity(selected.route_structure, assignment_policy=policy)
        if validity != VALID_MULTI_ROUTE and not config.include_degenerate_controls:
            continue
        record = _assignment_record(
            graph_id,
            selected,
            policy,
            validity,
            search_mode=search_mode,
            pair_evaluations=pair_evaluations,
            candidate_node_count=candidate_node_count,
        )
        if record.assignment_id in used_assignment_ids:
            continue
        used_assignment_ids.add(record.assignment_id)
        assignments.append(record)
        if len(assignments) >= config.max_assignments_per_graph:
            break
    return assignments


def _select_top_sampled_assignments(
    graph_id: str,
    evaluations: list[PairEvaluation],
    *,
    config: AssignmentSearchConfig,
    search_mode: str,
    pair_evaluations: int,
    candidate_node_count: int,
) -> list[GraphAssignmentRecord]:
    eligible = [
        evaluation
        for evaluation in evaluations
        if _eligible_for_assignment(evaluation.route_structure, config)
    ]
    ranked = sorted(
        eligible,
        key=_meaningful_assignment_score,
        reverse=True,
    )
    assignments = []
    for rank, evaluation in enumerate(ranked, 1):
        validity = classify_strategy_validity(evaluation.route_structure)
        if validity != VALID_MULTI_ROUTE and not config.include_degenerate_controls:
            continue
        record = _assignment_record(
            graph_id,
            evaluation,
            f"{POLICY_TOP_SAMPLED_PAIR}_{rank}",
            validity,
            search_mode=search_mode,
            pair_evaluations=pair_evaluations,
            candidate_node_count=candidate_node_count,
        )
        assignments.append(record)
        if len(assignments) >= config.max_assignments_per_graph:
            break
    return assignments


def _assignment_record(
    graph_id: str,
    evaluation: PairEvaluation,
    policy: str,
    validity: str,
    *,
    search_mode: str,
    pair_evaluations: int,
    candidate_node_count: int,
) -> GraphAssignmentRecord:
    route_structure = evaluation.route_structure
    assignment_id = _assignment_id(policy, evaluation.alice, evaluation.bob)
    return GraphAssignmentRecord(
        assignment_id=assignment_id,
        graph_id=graph_id,
        alice=evaluation.alice,
        bob=evaluation.bob,
        assignment_policy=policy,
        strategy_validity=validity,
        route_count=route_structure.route_count,
        shortest_hops=route_structure.shortest_hops,
        longest_hops=route_structure.longest_hops,
        shortest_path_count=route_structure.shortest_path_count,
        node_cut_size=route_structure.node_cut_size,
        edge_cut_size=route_structure.edge_cut_size,
        candidate_node_hitting_size=route_structure.candidate_node_hitting_size,
        candidate_edge_hitting_size=route_structure.candidate_edge_hitting_size,
        route_overlap_summary=route_structure.route_overlap_summary,
        assignment_search_mode=search_mode,
        pair_evaluations=pair_evaluations,
        candidate_node_count=candidate_node_count,
        route_enumeration_truncated=route_structure.route_enumeration_truncated,
        route_structure=route_structure,
    )


def _candidate_pairs(
    graph: TopologyIR,
    config: AssignmentSearchConfig,
) -> tuple[list[tuple[str, str]], str, int]:
    if len(graph.nodes) <= config.large_graph_node_threshold:
        return list(itertools.combinations(sorted(graph.nodes), 2)), "exhaustive", len(graph.nodes)

    candidate_nodes = _large_graph_candidate_nodes(graph, config)
    pairs = list(itertools.combinations(candidate_nodes, 2))
    rng = random.Random(config.random_seed + len(graph.nodes) + len(graph.edges))
    rng.shuffle(pairs)
    pairs = pairs[: config.max_pair_evaluations_large]
    return pairs, "sampled", len(candidate_nodes)


def _large_graph_finalist_pairs(
    graph: TopologyIR,
    pairs: list[tuple[str, str]],
    config: AssignmentSearchConfig,
) -> list[tuple[str, str]]:
    if len(pairs) <= config.large_graph_finalist_count:
        return pairs

    adjacency = graph.adjacency()
    sorted_adjacency = {node: sorted(neighbors) for node, neighbors in adjacency.items()}
    degree = {node: len(neighbors) for node, neighbors in adjacency.items()}
    shortest_cache: dict[str, dict[str, tuple[int, int]]] = {}
    scored = []
    for alice, bob in pairs:
        shortest_data = shortest_cache.get(alice)
        if shortest_data is None:
            shortest_data = single_source_shortest_path_data(
                graph,
                alice,
                adjacency=sorted_adjacency,
            )
            shortest_cache[alice] = shortest_data
        if bob not in shortest_data:
            continue
        shortest_hops, shortest_path_count = shortest_data[bob]
        if config.exclude_direct_links and shortest_hops == 1:
            continue
        if shortest_hops < config.min_ab_hops:
            continue
        scored.append((
            _cheap_large_pair_score(
                alice,
                bob,
                shortest_hops,
                shortest_path_count,
                degree,
                config,
            ),
            (alice, bob),
        ))

    if not scored:
        return pairs[: config.large_graph_finalist_count]
    scored.sort(reverse=True)
    return [pair for _score, pair in scored[: config.large_graph_finalist_count]]


def _large_graph_candidate_nodes(graph: TopologyIR, config: AssignmentSearchConfig) -> list[str]:
    adjacency = graph.adjacency()
    degree = {node: len(neighbors) for node, neighbors in adjacency.items()}
    ordered: list[str] = []

    top_n = max(4, config.large_graph_candidate_node_limit // 4)
    ordered.extend(node for node, _ in sorted(degree.items(), key=lambda item: (-item[1], item[0]))[:top_n])
    ordered.extend(node for node, _ in sorted(degree.items(), key=lambda item: (item[1], item[0]))[:top_n])

    rng = random.Random(config.random_seed + len(graph.nodes))
    nodes = sorted(graph.nodes)
    landmarks = rng.sample(nodes, min(config.landmark_count_large, len(nodes)))
    for landmark in landmarks:
        distances = single_source_shortest_path_data(graph, landmark)
        if distances:
            farthest = max(distances.items(), key=lambda item: (item[1][0], item[0]))[0]
            ordered.extend([landmark, farthest])

    for component in connected_components(graph):
        ordered.append(sorted(component)[0])

    random_nodes = rng.sample(nodes, min(config.random_candidate_nodes_large, len(nodes)))
    ordered.extend(random_nodes)
    return _unique_preserve_order(ordered)[: config.large_graph_candidate_node_limit]


def _eligible_for_assignment(route_structure: RouteStructureRecord,
                             config: AssignmentSearchConfig) -> bool:
    if route_structure.route_count == 0:
        return False
    if config.exclude_direct_links and route_structure.shortest_hops == 1:
        return False
    if (
        route_structure.shortest_hops is not None
        and route_structure.shortest_hops < config.min_ab_hops
    ):
        return False
    return True


def _select_preferred(
    evaluations: list[PairEvaluation],
    preferred_pair: Optional[tuple[str, str]],
) -> Optional[PairEvaluation]:
    if preferred_pair is None:
        return None
    for evaluation in evaluations:
        if (evaluation.alice, evaluation.bob) == preferred_pair:
            return evaluation
    return None


def _select_best_route_diverse(evaluations: list[PairEvaluation]) -> Optional[PairEvaluation]:
    valid = [
        evaluation
        for evaluation in evaluations
        if classify_strategy_validity(evaluation.route_structure) == VALID_MULTI_ROUTE
    ]
    return max(valid, key=lambda evaluation: evaluation.score) if valid else None


def _select_strongest_node_bottleneck(
    evaluations: list[PairEvaluation],
) -> Optional[PairEvaluation]:
    if not evaluations:
        return None
    return min(
        evaluations,
        key=lambda evaluation: (
            _cut_sort_value(evaluation.route_structure.node_cut_size),
            -evaluation.route_structure.route_count,
            -(evaluation.route_structure.shortest_hops or 0),
            evaluation.alice,
            evaluation.bob,
        ),
    )


def _select_strongest_edge_bottleneck(
    evaluations: list[PairEvaluation],
) -> Optional[PairEvaluation]:
    if not evaluations:
        return None
    return min(
        evaluations,
        key=lambda evaluation: (
            _cut_sort_value(evaluation.route_structure.edge_cut_size),
            -evaluation.route_structure.route_count,
            -(evaluation.route_structure.shortest_hops or 0),
            evaluation.alice,
            evaluation.bob,
        ),
    )


def _select_long_distance_pair(evaluations: list[PairEvaluation]) -> Optional[PairEvaluation]:
    if not evaluations:
        return None
    return max(
        evaluations,
        key=lambda evaluation: (
            evaluation.route_structure.shortest_hops or -1,
            evaluation.route_structure.route_count,
            evaluation.alice,
            evaluation.bob,
        ),
    )


def _select_random_valid_pair(
    evaluations: list[PairEvaluation],
    seed: int,
    graph_id: str,
) -> Optional[PairEvaluation]:
    valid = [
        evaluation
        for evaluation in evaluations
        if classify_strategy_validity(evaluation.route_structure) == VALID_MULTI_ROUTE
    ]
    pool = valid or evaluations
    if not pool:
        return None
    rng = random.Random(f"{seed}:{graph_id}")
    return pool[rng.randrange(len(pool))]


def _route_diversity_score(route_structure: RouteStructureRecord,
                           alice: str,
                           bob: str) -> tuple:
    return (
        _cut_score_value(route_structure.node_cut_size),
        _cut_score_value(route_structure.edge_cut_size),
        route_structure.route_count,
        route_structure.shortest_path_count,
        route_structure.shortest_hops or 0,
        route_structure.longest_hops or 0,
        alice,
        bob,
    )


def _meaningful_assignment_score(evaluation: PairEvaluation) -> tuple:
    route_structure = evaluation.route_structure
    overlap = route_structure.route_overlap_summary or {}
    return (
        _strategy_validity_rank(classify_strategy_validity(route_structure)),
        route_structure.route_count,
        min(route_structure.shortest_path_count, 1_000_000),
        _cut_quality(route_structure.node_cut_size),
        _cut_quality(route_structure.edge_cut_size),
        route_structure.longest_hops or 0,
        route_structure.shortest_hops or 0,
        -float(overlap.get("mean_edge_jaccard", 0.0) or 0.0),
        -float(overlap.get("mean_internal_node_jaccard", 0.0) or 0.0),
        evaluation.alice,
        evaluation.bob,
    )


def _cheap_large_pair_score(
    alice: str,
    bob: str,
    shortest_hops: int,
    shortest_path_count: int,
    degree: dict[str, int],
    config: AssignmentSearchConfig,
) -> tuple:
    alice_degree = degree.get(alice, 0)
    bob_degree = degree.get(bob, 0)
    target_hops = max(config.min_ab_hops, 4)
    return (
        -abs(shortest_hops - target_hops),
        -shortest_hops,
        min(shortest_path_count, 1_000_000),
        min(alice_degree, bob_degree),
        alice_degree + bob_degree,
        -abs(alice_degree - bob_degree),
        alice,
        bob,
    )


def _strategy_validity_rank(validity: str) -> int:
    return {
        VALID_MULTI_ROUTE: 4,
        DEGENERATE_NODE_BOTTLENECK: 3,
        DEGENERATE_EDGE_BOTTLENECK: 3,
        DEGENERATE_SINGLE_ROUTE: 2,
        INVALID_DISCONNECTED: 1,
    }.get(validity, 0)


def _cut_quality(value: Optional[int]) -> int:
    return value if isinstance(value, int) else -1


def assignment_class_counts(assignments: list[GraphAssignmentRecord]) -> dict[str, int]:
    return dict(Counter(record.strategy_validity for record in assignments))


def _preferred_pair(graph: TopologyIR) -> Optional[tuple[str, str]]:
    alice = graph.metadata.params.get("preferred_alice")
    bob = graph.metadata.params.get("preferred_bob")
    if alice is None or bob is None:
        return None
    return str(alice), str(bob)


def _assignment_id(policy: str, alice: str, bob: str) -> str:
    return f"{_slug(policy)}__{_slug(alice)}__{_slug(bob)}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_") or "none"


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _cut_sort_value(value: Optional[int]) -> int:
    return value if isinstance(value, int) else 1_000_000


def _cut_score_value(value: Optional[int]) -> int:
    return value if isinstance(value, int) else 1_000_000


def _cut_is_single_or_zero(value: Optional[int]) -> bool:
    return isinstance(value, int) and value <= 1
