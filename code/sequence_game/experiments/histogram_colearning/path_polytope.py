"""Bridge complete histogram route cases to an exact DAG path polytope."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from sequence_game.game_theory import (
    BoundedDAGPathPolytope,
    ComBandSchedule,
    DAGComBand,
    DAGPathExp2,
    DirectedPathArc,
)

from .corpus import HistogramGameCase
from .structure import route_resource_embedding


@dataclass(frozen=True)
class HistogramCasePathPolytope:
    """Implicit learner geometry plus the existing physics-pool lookup bridge.

    ``path_to_route_index`` exists only because the current compiled histogram
    cache is keyed by the already calibrated finite route set.  It is not used
    to count, normalize, weight, sample, or update paths inside ``polytope``.
    """

    polytope: BoundedDAGPathPolytope
    path_to_route_index: dict[tuple[str, ...], int]
    route_feature_matrix: np.ndarray
    feature_span_residual: float

    def diagnostics(self, *, horizon: int = 100_000) -> dict[str, Any]:
        schedule = self.polytope.comband_schedule(horizon)
        return {
            "schema": "histogram_case_path_polytope.v1",
            "representation": "finite_source_target_DAG_path_polytope",
            "path_storage_inside_learner": False,
            "cycle_free": True,
            "route_hops_maximum": self.polytope.max_hops,
            "path_count_dynamic_program": self.polytope.path_count,
            "path_length_counts": self.polytope.path_length_counts(),
            "directed_arc_count": len(self.polytope.arcs),
            "node_count": len(self.polytope.nodes),
            "compressed_action_dimension": self.polytope.dimension,
            "uniform_minimum_eigenvalue": (
                self.polytope.uniform_minimum_eigenvalue
            ),
            "histogram_route_resource_rank": int(
                np.linalg.matrix_rank(self.route_feature_matrix)
            ),
            "feature_span_projector_residual": self.feature_span_residual,
            "comband_schedule": schedule.to_dict(),
            "root_reachable_moment_transition_count": int(sum(
                len(plan.arc_indices)
                for plan in self.polytope._hop_moment_plans
            )),
            "physics_adapter_boundary": (
                "sampled node path is mapped to the existing route-indexed "
                "compiled histogram calibration; the learner does not receive "
                "the route table or unselected physics outcomes"
            ),
            "unsupported_scope": (
                "general cyclic graphs and incomplete route subsets whose "
                "directed-arc closure adds undeclared paths"
            ),
        }

    def route_index(self, path: Sequence[str]) -> int:
        try:
            return self.path_to_route_index[tuple(map(str, path))]
        except KeyError as exc:
            raise ValueError(
                "sampled path has no compiled histogram route calibration"
            ) from exc


class HistogramDAGComBand:
    """Route-index adapter around non-enumerative scalar-loss COMBAND.

    Sampling, normalization, and updates remain implicit in the DAG polytope.
    The finite feature table is used only to map a sampled node path to the
    existing route-indexed histogram cache and to serialize comparable policy
    diagnostics.
    """

    def __init__(
        self,
        case: HistogramGameCase,
        *,
        horizon: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.adapter = build_histogram_case_path_polytope(case)
        self.schedule = self.adapter.polytope.comband_schedule(horizon)
        self.learner = DAGComBand(
            self.adapter.polytope,
            eta=self.schedule.eta,
            gamma=self.schedule.gamma,
            rng=rng,
        )
        self.features = self.adapter.route_feature_matrix
        self.k = len(case.routes)
        self.counts = np.zeros(self.k, dtype=float)
        self._last_route_index: int | None = None

    def sample(self) -> int:
        path = self.learner.sample()
        route_index = self.adapter.route_index(path)
        self._last_route_index = route_index
        self.counts[route_index] += 1.0
        return route_index

    def update(self, reward: float) -> None:
        value = float(reward)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("reward must be finite and in [0, 1]")
        self.learner.update(1.0 - value)

    def policy(self, *, turn: int | None = None) -> np.ndarray:
        return self.policy_with_log_weight_offset(
            np.zeros(self.k, dtype=float),
            turn=turn,
        )

    def policy_with_log_weight_offset(
        self,
        offset: np.ndarray,
        *,
        turn: int | None = None,
    ) -> np.ndarray:
        if turn is not None and turn < 1:
            raise ValueError("turn must be >= 1")
        correction = np.asarray(offset, dtype=float)
        if correction.shape != (self.k,) or not np.all(np.isfinite(correction)):
            raise ValueError("offset must be a finite num_routes vector")
        scores = self.features @ self.learner.theta + correction
        exploitation = np.exp(scores - float(np.max(scores)))
        exploitation /= float(np.sum(exploitation))
        gamma = self._gamma_for_turn(turn)
        policy = (
            (1.0 - gamma) * exploitation
            + gamma / self.k
        )
        return policy / float(np.sum(policy))

    def counterfactual_log_weight_correction(
        self,
        reward_delta: float,
    ) -> np.ndarray:
        # Alternate reward r+delta has loss (1-r)-delta.
        theta_correction = self.learner.counterfactual_theta_correction(
            -float(reward_delta)
        )
        return self.features @ theta_correction

    def empirical_strategy(self) -> np.ndarray:
        total = float(np.sum(self.counts))
        if total > 0.0:
            return self.counts / total
        return np.full(self.k, 1.0 / self.k)

    def _gamma_for_turn(self, turn: int | None) -> float:
        return self.schedule.gamma

    @property
    def last_schedule(self) -> ComBandSchedule | None:
        return self.schedule if self.learner.turns else None

    @property
    def last_action_probability(self) -> float | None:
        return self.learner.last_action_probability


class HistogramDAGPathExp2(HistogramDAGComBand):
    """Histogram adapter for anytime reward-maximizing implicit-path Exp2."""

    def __init__(
        self,
        case: HistogramGameCase,
        *,
        rng: np.random.Generator | None = None,
        eta_c: float = 1.0,
        t0: float = 10_000.0,
        gamma_max: float = 0.20,
    ) -> None:
        self.adapter = build_histogram_case_path_polytope(case)
        self.learner = DAGPathExp2(
            self.adapter.polytope,
            eta_c=eta_c,
            t0=t0,
            gamma_max=gamma_max,
            rng=rng,
        )
        self.features = self.adapter.route_feature_matrix
        self.k = len(case.routes)
        self.counts = np.zeros(self.k, dtype=float)
        self._last_route_index: int | None = None

    def update(self, reward: float) -> None:
        self.learner.update(reward)

    def counterfactual_log_weight_correction(
        self,
        reward_delta: float,
    ) -> np.ndarray:
        return self.features @ self.learner.counterfactual_theta_correction(
            float(reward_delta)
        )

    def _gamma_for_turn(self, turn: int | None) -> float:
        selected_turn = self.learner.turns + 1 if turn is None else int(turn)
        return self.learner.schedule_at_turn(selected_turn).gamma

    @property
    def last_schedule(self) -> Any:
        return self.learner.last_schedule


def build_histogram_case_path_polytope(
    case: HistogramGameCase,
) -> HistogramCasePathPolytope:
    """Build an exact implicit polytope when a histogram case admits one.

    Edges are oriented only by their direction in the declared Alice--Bob
    routes.  Opposing orientations therefore produce a directed cycle and are
    rejected.  We also require the bounded DAG closure to reproduce exactly the
    declared route set; this prevents an implicit learner from selecting a path
    for which the histogram backend has no calibrated physics entry.
    """

    if not case.routes:
        raise ValueError("histogram case has no routes")
    route_hops_maximum = max(
        int(route["hop_count"]) for route in case.routes
    )

    arcs_by_pair: dict[tuple[str, str], DirectedPathArc] = {}
    path_to_route_index: dict[tuple[str, ...], int] = {}
    for route_index, route in enumerate(case.routes):
        path = tuple(map(str, route.get("path") or ()))
        edge_ids = tuple(map(str, route.get("edge_ids") or ()))
        if len(path) != len(edge_ids) + 1:
            raise ValueError(
                f"route {route_index} path/edge count is inconsistent"
            )
        if path in path_to_route_index:
            raise ValueError(f"duplicate declared route path {path!r}")
        path_to_route_index[path] = route_index
        for tail, head, edge_id in zip(path, path[1:], edge_ids):
            pair = (tail, head)
            previous = arcs_by_pair.get(pair)
            if previous is not None and previous.arc_id != edge_id:
                raise ValueError(
                    f"directed pair {pair!r} has conflicting edge ids"
                )
            arcs_by_pair[pair] = DirectedPathArc(edge_id, tail, head)

    opposed = sorted({
        tuple(sorted((tail, head)))
        for tail, head in arcs_by_pair
        if (head, tail) in arcs_by_pair
    })
    if opposed:
        raise ValueError(
            f"case {case.graph_id}/{case.assignment_id} has no exact bounded-DAG "
            "path-polytope representation: declared Alice--Bob routes orient "
            f"physical links in both directions {opposed!r}"
        )

    try:
        polytope = BoundedDAGPathPolytope(
            sorted(arcs_by_pair.values(), key=lambda arc: arc.arc_id),
            source=str(case.alice),
            target=str(case.bob),
            max_hops=route_hops_maximum,
        )
    except ValueError as exc:
        raise ValueError(
            f"case {case.graph_id}/{case.assignment_id} has no exact bounded-DAG "
            f"path-polytope representation: {exc}"
        ) from exc
    if polytope.path_count != len(case.routes):
        raise ValueError(
            f"case {case.graph_id}/{case.assignment_id} has {len(case.routes)} "
            f"declared routes but its bounded directed-arc closure has "
            f"{polytope.path_count}; refusing to add undeclared learner actions"
        )

    route_features = np.asarray([
        polytope.feature(tuple(map(str, route["path"])))
        for route in case.routes
    ])
    declared = route_resource_embedding(case).embedding
    path_projector = route_features @ np.linalg.pinv(route_features)
    declared_projector = declared @ declared.T
    residual = float(np.linalg.norm(path_projector - declared_projector, ord=2))
    if residual > 1e-9:
        raise ValueError(
            "implicit edge-incidence features do not preserve the declared "
            f"route-resource span (projector residual {residual:.3e})"
        )
    return HistogramCasePathPolytope(
        polytope=polytope,
        path_to_route_index=path_to_route_index,
        route_feature_matrix=route_features,
        feature_span_residual=residual,
    )
