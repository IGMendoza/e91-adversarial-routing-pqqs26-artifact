"""Non-enumerative adversarial bandit learning over bounded DAG paths.

The scalar-feedback boundary and update follow COMBAND from Cesa-Bianchi and
Lugosi, *Combinatorial Bandits*, JCSS 2012 (Algorithm 1):
https://doi.org/10.1016/j.jcss.2012.01.001

Paths are represented by directed-edge incidence.  Exponential path weights,
uniform exploration, sampling, and first/second moments are evaluated by
dynamic programming over ``(remaining_hops, node)`` states.  The learner never
stores the source--target path set.  This exact polynomial construction is
limited to directed acyclic graphs; it must not be presented as an oracle for
simple paths in a general cyclic graph.

Maiti et al., COLT 2025, give the stronger current result for this exact action
geometry and aggregate bandit feedback.  Their near-minimax algorithm is the
research target, but requires a substantially more involved FTRL optimizer and
graph decomposition than this auditable COMBAND baseline:
https://proceedings.mlr.press/v291/maiti25a.html
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math
from typing import Hashable, Iterable, Sequence

import numpy as np

from .bandit import Exp3SchedulePoint


Node = Hashable


@dataclass(frozen=True)
class DirectedPathArc:
    """One uniquely labelled directed arc in a path-polytope graph."""

    arc_id: str
    tail: Node
    head: Node


@dataclass(frozen=True)
class PathDistributionMoments:
    """Partition function and feature moments of one path distribution."""

    log_partition: float
    mean: np.ndarray
    second_moment: np.ndarray
    backward_log_partitions: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class ComBandSchedule:
    """Fixed-horizon parameters from Cesa-Bianchi--Lugosi Theorem 1.

    The theorem uses binary edge-incidence actions. ``unclipped_gamma`` is
    retained because clipping at one marks a conservative boundary case rather
    than an interior theoretical tuning.
    """

    horizon: int
    action_count: int
    ambient_edge_count: int
    maximum_path_hops: int
    uniform_minimum_nonzero_eigenvalue: float
    eta: float
    gamma: float
    unclipped_gamma: float

    @property
    def gamma_was_clipped(self) -> bool:
        return self.unclipped_gamma > 1.0

    def to_dict(self) -> dict[str, object]:
        return {
            "source": "Cesa-Bianchi and Lugosi, JCSS 2012, Theorem 1",
            "horizon": self.horizon,
            "action_count": self.action_count,
            "ambient_edge_count": self.ambient_edge_count,
            "maximum_path_hops": self.maximum_path_hops,
            "uniform_minimum_nonzero_eigenvalue": (
                self.uniform_minimum_nonzero_eigenvalue
            ),
            "eta": self.eta,
            "gamma": self.gamma,
            "unclipped_gamma": self.unclipped_gamma,
            "gamma_was_clipped": self.gamma_was_clipped,
        }


@dataclass(frozen=True)
class _HopMomentPlan:
    """Root-reachable transitions contributing at one remaining-hop depth."""

    arc_indices: np.ndarray
    tail_indices: np.ndarray
    head_indices: np.ndarray
    group_tails: np.ndarray
    group_starts: np.ndarray


class BoundedDAGPathPolytope:
    """Implicit set of source--target DAG paths with at most ``max_hops``.

    The internal feature basis is the positive-eigenvalue span of the uniform
    path-incidence second moment.  It is computed without path enumeration and
    removes flow-conservation singularities before COMBAND's pseudoinverse.
    """

    def __init__(
        self,
        arcs: Iterable[DirectedPathArc],
        *,
        source: Node,
        target: Node,
        max_hops: int,
    ) -> None:
        arc_rows = tuple(arcs)
        if not arc_rows:
            raise ValueError("path polytope requires at least one directed arc")
        if source == target:
            raise ValueError("source and target must be distinct")
        if max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        if len({arc.arc_id for arc in arc_rows}) != len(arc_rows):
            raise ValueError("directed arc ids must be unique")
        pairs = [(arc.tail, arc.head) for arc in arc_rows]
        if len(set(pairs)) != len(pairs):
            raise ValueError("parallel directed arcs are not supported")
        if any(arc.tail == arc.head for arc in arc_rows):
            raise ValueError("self-loop arcs are not supported")

        nodes = {source, target}
        for arc in arc_rows:
            nodes.update((arc.tail, arc.head))
        node_rows = tuple(sorted(nodes, key=str))
        node_index = {node: index for index, node in enumerate(node_rows)}
        outgoing: list[list[int]] = [[] for _ in node_rows]
        indegree = np.zeros(len(node_rows), dtype=int)
        for index, arc in enumerate(arc_rows):
            outgoing[node_index[arc.tail]].append(index)
            indegree[node_index[arc.head]] += 1
        for rows in outgoing:
            rows.sort(key=lambda index: arc_rows[index].arc_id)
        topological = _topological_order(
            arc_rows=arc_rows,
            node_rows=node_rows,
            node_index=node_index,
            outgoing=outgoing,
            indegree=indegree,
        )

        self.arcs = arc_rows
        self.nodes = node_rows
        self.source = source
        self.target = target
        self.max_hops = int(max_hops)
        self._node_index = node_index
        self._source_index = node_index[source]
        self._target_index = node_index[target]
        self._outgoing = tuple(tuple(rows) for rows in outgoing)
        self._topological_order = topological
        self._arc_index_by_pair = {
            (arc.tail, arc.head): index for index, arc in enumerate(arc_rows)
        }
        grouped_arcs: list[int] = []
        group_starts: list[int] = []
        group_tails: list[int] = []
        for tail_index, tail_arcs in enumerate(self._outgoing):
            if tail_index == self._target_index or not tail_arcs:
                continue
            group_starts.append(len(grouped_arcs))
            group_tails.append(tail_index)
            grouped_arcs.extend(tail_arcs)
        self._grouped_arcs = np.asarray(grouped_arcs, dtype=int)
        self._group_starts = np.asarray(group_starts, dtype=int)
        self._group_tails = np.asarray(group_tails, dtype=int)
        self._grouped_tails_per_arc = np.repeat(
            self._group_tails,
            np.diff(np.append(self._group_starts, len(self._grouped_arcs))),
        )
        self._grouped_heads_per_arc = np.asarray([
            self._node_index[self.arcs[arc_index].head]
            for arc_index in self._grouped_arcs
        ], dtype=int)
        counts = self.path_length_counts()
        self.path_count = int(sum(counts.values()))
        if self.path_count < 1:
            raise ValueError(
                "no source--target path exists inside the declared hop bound"
            )
        self.maximum_path_hops = max(counts)

        zero_raw_scores = np.zeros(len(self.arcs), dtype=float)
        raw_log_partitions = self._backward_log_partitions(zero_raw_scores)
        self._hop_moment_plans = self._build_hop_moment_plans(
            raw_log_partitions
        )
        raw_uniform = self._moments_from_local_features(
            zero_raw_scores,
            np.eye(len(self.arcs), dtype=float),
            log_partition=raw_log_partitions,
        )
        self._admissible_outgoing = tuple(
            tuple(
                tuple(
                    arc_index for arc_index in self._outgoing[node_index]
                    if hops > 0 and math.isfinite(
                        raw_uniform.backward_log_partitions[
                            hops - 1,
                            self._node_index[self.arcs[arc_index].head],
                        ]
                    )
                )
                for node_index in range(len(self.nodes))
            )
            for hops in range(self.max_hops + 1)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(raw_uniform.second_moment)
        tolerance = (
            max(raw_uniform.second_moment.shape)
            * np.finfo(float).eps
            * max(1.0, float(eigenvalues[-1]))
        )
        active = eigenvalues > tolerance
        if not np.any(active):
            raise FloatingPointError("uniform path moment has zero rank")
        self.local_features = eigenvectors[:, active]
        self.dimension = int(np.sum(active))
        self.uniform_moments = self.moments(
            np.zeros(self.dimension, dtype=float)
        )
        compressed_eigenvalues = np.linalg.eigvalsh(
            self.uniform_moments.second_moment
        )
        if compressed_eigenvalues[0] <= 0.0:
            raise FloatingPointError(
                "compressed uniform path moment is not positive definite"
            )
        self.uniform_minimum_eigenvalue = float(compressed_eigenvalues[0])

    def _build_hop_moment_plans(
        self,
        uniform_log_partitions: np.ndarray,
    ) -> tuple[_HopMomentPlan, ...]:
        root_reachable = np.zeros(
            (self.max_hops + 1, len(self.nodes)), dtype=bool
        )
        root_reachable[self.max_hops, self._source_index] = True
        for hops in range(self.max_hops, 0, -1):
            for tail_index in np.flatnonzero(root_reachable[hops]):
                if tail_index == self._target_index:
                    continue
                for arc_index in self._outgoing[tail_index]:
                    head_index = self._node_index[self.arcs[arc_index].head]
                    if math.isfinite(
                        uniform_log_partitions[hops - 1, head_index]
                    ):
                        root_reachable[hops - 1, head_index] = True

        plans: list[_HopMomentPlan] = []
        for hops in range(self.max_hops + 1):
            arc_indices = np.asarray([
                arc_index
                for tail_index in np.flatnonzero(root_reachable[hops])
                if tail_index != self._target_index
                for arc_index in self._outgoing[tail_index]
                if hops > 0 and math.isfinite(
                    uniform_log_partitions[
                        hops - 1,
                        self._node_index[self.arcs[arc_index].head],
                    ]
                )
            ], dtype=int)
            if arc_indices.size:
                tail_indices = np.asarray([
                    self._node_index[self.arcs[arc_index].tail]
                    for arc_index in arc_indices
                ], dtype=int)
                head_indices = np.asarray([
                    self._node_index[self.arcs[arc_index].head]
                    for arc_index in arc_indices
                ], dtype=int)
                group_tails, group_starts = np.unique(
                    tail_indices,
                    return_index=True,
                )
            else:
                tail_indices = np.asarray([], dtype=int)
                head_indices = np.asarray([], dtype=int)
                group_tails = np.asarray([], dtype=int)
                group_starts = np.asarray([], dtype=int)
            plans.append(_HopMomentPlan(
                arc_indices=arc_indices,
                tail_indices=tail_indices,
                head_indices=head_indices,
                group_tails=group_tails,
                group_starts=group_starts,
            ))
        return tuple(plans)

    def path_length_counts(self) -> dict[int, int]:
        """Return exact path counts by hop length without listing paths."""

        exact = np.zeros(
            (self.max_hops + 1, len(self.nodes)), dtype=object
        )
        exact[0, self._target_index] = 1
        for hops in range(1, self.max_hops + 1):
            for node_index in reversed(self._topological_order):
                if node_index == self._target_index:
                    continue
                exact[hops, node_index] = sum(
                    exact[hops - 1, self._node_index[self.arcs[arc].head]]
                    for arc in self._outgoing[node_index]
                )
        return {
            hops: int(exact[hops, self._source_index])
            for hops in range(1, self.max_hops + 1)
            if exact[hops, self._source_index]
        }

    def moments(self, theta: np.ndarray) -> PathDistributionMoments:
        """Exact moments for ``q(path) proportional exp(theta @ feature)``."""

        coefficient = np.asarray(theta, dtype=float)
        if coefficient.shape != (self.dimension,):
            raise ValueError(
                f"theta must have shape ({self.dimension},), got {coefficient.shape}"
            )
        if not np.all(np.isfinite(coefficient)):
            raise ValueError("theta must be finite")
        local_scores = self.local_features @ coefficient
        return self._moments_from_local_features(
            local_scores,
            self.local_features,
        )

    def sample(
        self,
        theta: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> tuple[Node, ...]:
        """Draw one weighted path by backward-partition probabilities."""

        coefficient = np.asarray(theta, dtype=float)
        if coefficient.shape != (self.dimension,):
            raise ValueError(
                f"theta must have shape ({self.dimension},), got {coefficient.shape}"
            )
        if not np.all(np.isfinite(coefficient)):
            raise ValueError("theta must be finite")
        local_scores = self.local_features @ coefficient
        log_partition = self._backward_log_partitions(local_scores)
        path, _feature = self._sample_path_and_feature(
            local_scores,
            log_partition,
            rng=rng,
        )
        return path

    def _sample_path_and_feature(
        self,
        local_scores: np.ndarray,
        log_partition: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> tuple[tuple[Node, ...], np.ndarray]:
        node = self._source_index
        hops_left = self.max_hops
        path: list[Node] = [self.source]
        feature = np.zeros(self.dimension, dtype=float)
        while node != self._target_index:
            candidates = self._admissible_outgoing[hops_left][node]
            if not candidates:
                raise FloatingPointError("weighted path sampler reached a dead end")
            if len(candidates) == 1:
                selected = candidates[0]
            else:
                logits = tuple(
                    local_scores[arc_index]
                    + log_partition[
                        hops_left - 1,
                        self._node_index[self.arcs[arc_index].head],
                    ]
                    for arc_index in candidates
                )
                maximum = max(logits)
                weights = tuple(math.exp(value - maximum) for value in logits)
                threshold = float(rng.random()) * sum(weights)
                cumulative = 0.0
                selected = candidates[-1]
                for candidate, weight in zip(candidates, weights):
                    cumulative += weight
                    if threshold <= cumulative:
                        selected = candidate
                        break
            head = self.arcs[selected].head
            path.append(head)
            feature += self.local_features[selected]
            node = self._node_index[head]
            hops_left -= 1
        return tuple(path), feature

    def feature(self, path: Sequence[Node]) -> np.ndarray:
        """Return the compressed edge-incidence feature of one valid path."""

        arc_indices = self._validate_path(path)
        return np.sum(self.local_features[list(arc_indices)], axis=0)

    def path_probability(self, path: Sequence[Node], theta: np.ndarray) -> float:
        """Evaluate one path probability under the implicit distribution."""

        feature = self.feature(path)
        moments = self.moments(theta)
        return float(math.exp(float(feature @ theta) - moments.log_partition))

    def comband_schedule(self, horizon: int) -> ComBandSchedule:
        """Compute the fixed-horizon tuning stated in COMBAND Theorem 1."""

        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        b = float(self.maximum_path_hops)
        d = float(len(self.arcs))
        lam = self.uniform_minimum_eigenvalue
        raw_gamma = b * math.sqrt(
            math.log(self.path_count)
            / (lam * (b + d * lam) * float(horizon))
        ) if self.path_count > 1 else 0.0
        gamma = min(1.0, raw_gamma)
        eta = gamma * lam / (b * b)
        return ComBandSchedule(
            horizon=int(horizon),
            action_count=self.path_count,
            ambient_edge_count=len(self.arcs),
            maximum_path_hops=self.maximum_path_hops,
            uniform_minimum_nonzero_eigenvalue=lam,
            eta=float(eta),
            gamma=float(gamma),
            unclipped_gamma=float(raw_gamma),
        )

    def _validate_path(self, path: Sequence[Node]) -> tuple[int, ...]:
        row = tuple(path)
        if len(row) < 2 or row[0] != self.source or row[-1] != self.target:
            raise ValueError("path must run from the declared source to target")
        if len(row) - 1 > self.max_hops:
            raise ValueError("path exceeds the declared hop bound")
        if len(set(row)) != len(row):
            raise ValueError("path must be cycle-free")
        try:
            return tuple(
                self._arc_index_by_pair[(tail, head)]
                for tail, head in zip(row, row[1:])
            )
        except KeyError as exc:
            raise ValueError(f"path uses absent directed arc {exc.args[0]!r}") from exc

    def _backward_log_partitions(self, local_scores: np.ndarray) -> np.ndarray:
        table = np.full(
            (self.max_hops + 1, len(self.nodes)), -np.inf, dtype=float
        )
        table[:, self._target_index] = 0.0
        grouped_scores = local_scores[self._grouped_arcs]
        for hops in range(1, self.max_hops + 1):
            path_scores = (
                grouped_scores
                + table[hops - 1, self._grouped_heads_per_arc]
            )
            table[hops, self._group_tails] = np.logaddexp.reduceat(
                path_scores,
                self._group_starts,
            )
        return table

    def _moments_from_local_features(
        self,
        local_scores: np.ndarray,
        local_features: np.ndarray,
        *,
        log_partition: np.ndarray | None = None,
    ) -> PathDistributionMoments:
        feature_count = int(local_features.shape[1])
        if log_partition is None:
            log_partition = self._backward_log_partitions(local_scores)
        means = np.zeros(
            (self.max_hops + 1, len(self.nodes), feature_count), dtype=float
        )
        seconds = np.zeros(
            (
                self.max_hops + 1,
                len(self.nodes),
                feature_count,
                feature_count,
            ),
            dtype=float,
        )
        for hops in range(1, self.max_hops + 1):
            plan = self._hop_moment_plans[hops]
            if not plan.arc_indices.size:
                continue
            grouped_scores = local_scores[plan.arc_indices]
            grouped_features = local_features[plan.arc_indices]
            continuation = log_partition[
                hops - 1, plan.head_indices
            ]
            normalizer = log_partition[
                hops, plan.tail_indices
            ]
            probabilities = np.exp(
                grouped_scores + continuation - normalizer
            )
            tail_means = means[
                hops - 1, plan.head_indices
            ]
            path_means = grouped_features + tail_means
            mean_contributions = probabilities[:, None] * path_means
            means[hops, plan.group_tails] = np.add.reduceat(
                mean_contributions,
                plan.group_starts,
                axis=0,
            )
            path_seconds = (
                seconds[hops - 1, plan.head_indices]
                + grouped_features[:, :, None] * tail_means[:, None, :]
                + tail_means[:, :, None] * grouped_features[:, None, :]
                + grouped_features[:, :, None] * grouped_features[:, None, :]
            )
            second_contributions = (
                probabilities[:, None, None] * path_seconds
            )
            seconds[hops, plan.group_tails] = np.add.reduceat(
                second_contributions,
                plan.group_starts,
                axis=0,
            )
        root_log_partition = float(
            log_partition[self.max_hops, self._source_index]
        )
        if not math.isfinite(root_log_partition):
            raise FloatingPointError("path partition function is zero")
        return PathDistributionMoments(
            log_partition=root_log_partition,
            mean=means[self.max_hops, self._source_index].copy(),
            second_moment=seconds[self.max_hops, self._source_index].copy(),
            backward_log_partitions=log_partition,
        )


class DAGComBand:
    """COMBAND over an implicit bounded-DAG path set and scalar losses."""

    def __init__(
        self,
        polytope: BoundedDAGPathPolytope,
        *,
        eta: float,
        gamma: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        singleton = polytope.path_count == 1
        if (
            not math.isfinite(eta)
            or eta < 0.0
            or (not singleton and eta == 0.0)
        ):
            raise ValueError(
                "eta must be finite and > 0 except for a singleton path set"
            )
        if (
            not math.isfinite(gamma)
            or not 0.0 <= gamma <= 1.0
            or (not singleton and gamma == 0.0)
        ):
            raise ValueError(
                "gamma must be in (0, 1] except for a singleton path set"
            )
        self.polytope = polytope
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.theta = np.zeros(polytope.dimension, dtype=float)
        self.counts: Counter[tuple[Node, ...]] = Counter()
        self.turns = 0
        self._last_path: tuple[Node, ...] | None = None
        self._last_feature: np.ndarray | None = None
        self._last_behavior_second_moment: np.ndarray | None = None
        self._last_behavior_probability: float | None = None
        self._last_exploration: bool | None = None

    def sample(self) -> tuple[Node, ...]:
        exploitation_moments = self.polytope.moments(self.theta)
        behavior_second = (
            (1.0 - self.gamma) * exploitation_moments.second_moment
            + self.gamma * self.polytope.uniform_moments.second_moment
        )
        exploration = bool(self.rng.random() < self.gamma)
        sample_theta = (
            np.zeros(self.polytope.dimension, dtype=float)
            if exploration else self.theta
        )
        sample_moments = (
            self.polytope.uniform_moments
            if exploration else exploitation_moments
        )
        local_scores = self.polytope.local_features @ sample_theta
        path, feature = self.polytope._sample_path_and_feature(
            local_scores,
            sample_moments.backward_log_partitions,
            rng=self.rng,
        )
        exploitation_probability = math.exp(
            float(feature @ self.theta)
            - exploitation_moments.log_partition
        )
        self._last_path = path
        self._last_feature = feature
        self._last_behavior_second_moment = behavior_second
        self._last_behavior_probability = (
            (1.0 - self.gamma) * exploitation_probability
            + self.gamma / self.polytope.path_count
        )
        self._last_exploration = exploration
        self.counts[path] += 1
        self.turns += 1
        return path

    def update(self, loss: float) -> None:
        """Update from only the selected path's aggregate loss in ``[0, 1]``."""

        if self._last_feature is None or self._last_behavior_second_moment is None:
            raise RuntimeError("update() called before sample()")
        value = float(loss)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("loss must be finite and in [0, 1]")
        coefficient_estimate = value * np.linalg.solve(
            self._last_behavior_second_moment,
            self._last_feature,
        )
        self.theta -= self.eta * coefficient_estimate

    def counterfactual_theta_correction(self, loss_delta: float) -> np.ndarray:
        """Exact fixed-behavior coefficient correction for the last sample."""

        if self._last_feature is None or self._last_behavior_second_moment is None:
            raise RuntimeError("correction requested before sample()")
        value = float(loss_delta)
        if not math.isfinite(value):
            raise ValueError("loss_delta must be finite")
        return -self.eta * value * np.linalg.solve(
            self._last_behavior_second_moment,
            self._last_feature,
        )

    def empirical_path_probabilities(self) -> dict[tuple[Node, ...], float]:
        if self.turns == 0:
            return {}
        return {
            path: count / self.turns
            for path, count in sorted(self.counts.items(), key=lambda row: str(row[0]))
        }

    @property
    def last_path(self) -> tuple[Node, ...] | None:
        return self._last_path

    @property
    def last_sample_was_exploration(self) -> bool | None:
        return self._last_exploration

    @property
    def last_action_probability(self) -> float | None:
        return self._last_behavior_probability


class DAGPathExp2(DAGComBand):
    """Anytime reward-maximizing LinearExp2 over an implicit DAG path set.

    This is the non-enumerative counterpart of :class:`LinearExp2`: the
    exponential path distribution and covariance are evaluated by the DAG
    dynamic program, while the update observes only the selected path's scalar
    reward.  Reward maximization subtracts the action-independent unit baseline
    from COMBAND's ``1 - reward`` loss, avoiding estimator variance that cannot
    distinguish paths.
    """

    def __init__(
        self,
        polytope: BoundedDAGPathPolytope,
        *,
        rng: np.random.Generator | None = None,
        eta_c: float = 1.0,
        t0: float = 10_000.0,
        gamma_max: float = 0.20,
    ) -> None:
        if eta_c <= 0.0:
            raise ValueError("eta_c must be > 0")
        if t0 < 0.0:
            raise ValueError("t0 must be >= 0")
        if not 0.0 < gamma_max <= 1.0:
            raise ValueError("gamma_max must be in (0, 1]")
        self.eta_c = float(eta_c)
        self.t0 = float(t0)
        self.gamma_max = float(gamma_max)
        first = self.schedule_at_turn_for_polytope(polytope, 1)
        super().__init__(
            polytope,
            eta=first.eta,
            gamma=first.gamma,
            rng=rng,
        )
        self._last_schedule: Exp3SchedulePoint | None = None

    def schedule_at_turn_for_polytope(
        self,
        polytope: BoundedDAGPathPolytope,
        turn: int,
    ) -> Exp3SchedulePoint:
        if turn < 1:
            raise ValueError("turn must be >= 1")
        eta = self.eta_c * math.sqrt(
            math.log(max(2, polytope.path_count))
            / (polytope.dimension * (float(turn) + self.t0))
        )
        gamma = min(self.gamma_max, polytope.dimension * eta)
        return Exp3SchedulePoint(
            turn=int(turn),
            eta=float(eta),
            gamma=float(gamma),
        )

    def schedule_at_turn(self, turn: int) -> Exp3SchedulePoint:
        return self.schedule_at_turn_for_polytope(self.polytope, turn)

    def sample(self) -> tuple[Node, ...]:
        point = self.schedule_at_turn(self.turns + 1)
        self.eta = point.eta
        self.gamma = point.gamma
        self._last_schedule = point
        return super().sample()

    def update(self, reward: float) -> None:
        """Maximize the sampled path's scalar reward in ``[0, 1]``."""

        if self._last_feature is None or self._last_behavior_second_moment is None:
            raise RuntimeError("update() called before sample()")
        value = float(reward)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("reward must be finite and in [0, 1]")
        coefficient_estimate = value * np.linalg.solve(
            self._last_behavior_second_moment,
            self._last_feature,
        )
        self.theta += self.eta * coefficient_estimate

    def counterfactual_theta_correction(self, reward_delta: float) -> np.ndarray:
        """Exact fixed-behavior coefficient correction for alternate reward."""

        if self._last_feature is None or self._last_behavior_second_moment is None:
            raise RuntimeError("correction requested before sample()")
        value = float(reward_delta)
        if not math.isfinite(value):
            raise ValueError("reward_delta must be finite")
        return self.eta * value * np.linalg.solve(
            self._last_behavior_second_moment,
            self._last_feature,
        )

    @property
    def last_schedule(self) -> Exp3SchedulePoint | None:
        return self._last_schedule


def _topological_order(
    *,
    arc_rows: tuple[DirectedPathArc, ...],
    node_rows: tuple[Node, ...],
    node_index: dict[Node, int],
    outgoing: list[list[int]],
    indegree: np.ndarray,
) -> tuple[int, ...]:
    remaining = indegree.copy()
    queue = deque(
        sorted(
            (index for index, value in enumerate(remaining) if value == 0),
            key=lambda index: str(node_rows[index]),
        )
    )
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for arc_index in outgoing[node]:
            head = node_index[arc_rows[arc_index].head]
            remaining[head] -= 1
            if remaining[head] == 0:
                queue.append(head)
    if len(order) != len(node_rows):
        raise ValueError(
            "non-enumerative cycle-free path learning requires a directed "
            "acyclic graph"
        )
    return tuple(order)
