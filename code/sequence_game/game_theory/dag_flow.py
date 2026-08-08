"""Exact unit-flow geometry and convex FTRL primitives for DAG path bandits.

The classes in this module are deliberately independent of the histogram
backend.  They provide three operations used by the modern aggregate-feedback
learners:

* a source--target unit-flow polytope with Markovian path sampling;
* the centroid graph conversion of Maiti et al. (COLT 2025, Section 3.3);
* a warm-started Newton solver for separable convex regularizers over flows.

The Newton system is solved in the null space of flow conservation.  Its
dimension is the affine dimension of the path polytope rather than the number
of enumerated paths.  Solver residuals are retained so numerical optimization
is an auditable approximation to the papers' exact FTRL argmin.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Hashable, Iterable, Sequence

import numpy as np
from scipy.linalg import null_space

from .path_bandit import DirectedPathArc


Node = Hashable


@dataclass(frozen=True)
class FlowFTRLSolveStats:
    """Numerical certificate for one flow-FTRL minimization."""

    iterations: int
    projected_gradient_inf: float
    newton_decrement_squared_half: float
    objective: float
    converged: bool
    line_search_backtracks: int
    elapsed_seconds: float

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "iterations": self.iterations,
            "projected_gradient_inf": self.projected_gradient_inf,
            "newton_decrement_squared_half": (
                self.newton_decrement_squared_half
            ),
            "objective": self.objective,
            "converged": self.converged,
            "line_search_backtracks": self.line_search_backtracks,
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass(frozen=True)
class FlowRegularizer:
    """A sum of ``-a*sqrt(Mq) - b*log(Mq)`` terms."""

    mapping: np.ndarray
    sqrt_coefficients: np.ndarray
    log_coefficients: np.ndarray

    def __post_init__(self) -> None:
        mapping = np.asarray(self.mapping, dtype=float)
        sqrt = np.asarray(self.sqrt_coefficients, dtype=float)
        log = np.asarray(self.log_coefficients, dtype=float)
        if mapping.ndim != 2 or mapping.shape[0] < 1:
            raise ValueError("regularizer mapping must be a non-empty matrix")
        if sqrt.shape != (mapping.shape[0],) or log.shape != sqrt.shape:
            raise ValueError("regularizer coefficients must match mapping rows")
        if (
            not np.all(np.isfinite(mapping))
            or not np.all(np.isfinite(sqrt))
            or not np.all(np.isfinite(log))
            or np.any(sqrt < 0.0)
            or np.any(log < 0.0)
            or not np.any((sqrt > 0.0) | (log > 0.0))
        ):
            raise ValueError("regularizer entries must be finite and non-negative")
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "sqrt_coefficients", sqrt)
        object.__setattr__(self, "log_coefficients", log)


class DAGUnitFlow:
    """Source--target unit-flow polytope for a directed acyclic graph."""

    def __init__(
        self,
        arcs: Iterable[DirectedPathArc],
        *,
        source: Node,
        target: Node,
    ) -> None:
        arc_rows = tuple(arcs)
        if not arc_rows:
            raise ValueError("unit-flow geometry requires at least one arc")
        if source == target:
            raise ValueError("source and target must differ")
        if len({arc.arc_id for arc in arc_rows}) != len(arc_rows):
            raise ValueError("arc ids must be unique")
        pairs = tuple((arc.tail, arc.head) for arc in arc_rows)
        if len(set(pairs)) != len(pairs):
            raise ValueError("parallel directed arcs are not supported")

        nodes = {source, target}
        for arc in arc_rows:
            nodes.update((arc.tail, arc.head))
        node_rows = tuple(sorted(nodes, key=str))
        node_index = {node: index for index, node in enumerate(node_rows)}
        outgoing: list[list[int]] = [[] for _ in node_rows]
        incoming: list[list[int]] = [[] for _ in node_rows]
        indegree = np.zeros(len(node_rows), dtype=int)
        for edge_index, arc in enumerate(arc_rows):
            tail = node_index[arc.tail]
            head = node_index[arc.head]
            outgoing[tail].append(edge_index)
            incoming[head].append(edge_index)
            indegree[head] += 1
        for rows in (*outgoing, *incoming):
            rows.sort(key=lambda index: arc_rows[index].arc_id)
        topological = _topological_order(
            arcs=arc_rows,
            nodes=node_rows,
            node_index=node_index,
            outgoing=outgoing,
            indegree=indegree,
        )

        source_index = node_index[source]
        target_index = node_index[target]
        prefix = np.zeros(len(node_rows), dtype=object)
        prefix[source_index] = 1
        for node in topological:
            for edge in outgoing[node]:
                prefix[node_index[arc_rows[edge].head]] += prefix[node]
        suffix = np.zeros(len(node_rows), dtype=object)
        suffix[target_index] = 1
        longest = np.full(len(node_rows), -10**9, dtype=int)
        longest[target_index] = 0
        for node in reversed(topological):
            if node == target_index:
                continue
            candidates = [
                longest[node_index[arc_rows[edge].head]] + 1
                for edge in outgoing[node]
                if longest[node_index[arc_rows[edge].head]] >= 0
            ]
            if candidates:
                longest[node] = max(candidates)
            suffix[node] = sum(
                suffix[node_index[arc_rows[edge].head]]
                for edge in outgoing[node]
            )
        path_count = int(suffix[source_index])
        if path_count < 1:
            raise ValueError("DAG contains no source--target path")
        invalid = [
            arc.arc_id
            for edge, arc in enumerate(arc_rows)
            if int(prefix[node_index[arc.tail]]) == 0
            or int(suffix[node_index[arc.head]]) == 0
        ]
        if invalid:
            raise ValueError(
                "every learner arc must lie on a source--target path; invalid "
                f"arcs={invalid!r}"
            )

        equality_nodes = [
            index for index in range(len(node_rows)) if index != target_index
        ]
        equality = np.zeros((len(equality_nodes), len(arc_rows)), dtype=float)
        rhs = np.zeros(len(equality_nodes), dtype=float)
        for row, node in enumerate(equality_nodes):
            equality[row, outgoing[node]] = 1.0
            equality[row, incoming[node]] -= 1.0
            rhs[row] = 1.0 if node == source_index else 0.0
        if np.linalg.matrix_rank(equality) != equality.shape[0]:
            raise ValueError("flow-conservation constraints are rank deficient")
        null = null_space(equality)
        uniform = np.asarray([
            float(
                int(prefix[node_index[arc.tail]])
                * int(suffix[node_index[arc.head]])
                / path_count
            )
            for arc in arc_rows
        ])
        if np.any(uniform <= 0.0):
            raise FloatingPointError("uniform path flow is not strictly positive")
        if not np.allclose(equality @ uniform, rhs, atol=1e-11):
            raise FloatingPointError("uniform path flow violates conservation")

        self.arcs = arc_rows
        self.nodes = node_rows
        self.source = source
        self.target = target
        self.path_count = path_count
        self.maximum_path_hops = int(longest[source_index])
        self.node_index = node_index
        self.source_index = source_index
        self.target_index = target_index
        self.outgoing = tuple(tuple(rows) for rows in outgoing)
        self.incoming = tuple(tuple(rows) for rows in incoming)
        self.topological_order = topological
        self.equality = equality
        self.equality_rhs = rhs
        self.null_basis = null
        self.uniform_flow = uniform
        self.prefix_path_counts = tuple(map(int, prefix))
        self.suffix_path_counts = tuple(map(int, suffix))
        self._arc_index_by_pair = {
            pair: index for index, pair in enumerate(pairs)
        }
        self.tail_incidence = np.zeros(
            (len(node_rows), len(arc_rows)), dtype=float
        )
        for node, rows in enumerate(outgoing):
            self.tail_incidence[node, rows] = 1.0

    @property
    def dimension(self) -> int:
        return int(self.null_basis.shape[1])

    def validate_flow(self, flow: np.ndarray, *, tolerance: float = 1e-9) -> None:
        q = np.asarray(flow, dtype=float)
        if q.shape != (len(self.arcs),) or not np.all(np.isfinite(q)):
            raise ValueError("flow must be a finite edge vector")
        if np.any(q <= 0.0):
            raise ValueError("Markovian learner flow must be strictly positive")
        if not np.allclose(
            self.equality @ q,
            self.equality_rhs,
            atol=tolerance,
            rtol=0.0,
        ):
            raise ValueError("flow violates source--target conservation")

    def node_flow(self, flow: np.ndarray) -> np.ndarray:
        q = np.asarray(flow, dtype=float)
        return self.tail_incidence @ q

    def sample(
        self,
        flow: np.ndarray,
        *,
        rng: np.random.Generator,
    ) -> tuple[tuple[Node, ...], np.ndarray]:
        """Sample a path using the paper's local probabilities ``q(e)/q(v)``."""

        self.validate_flow(flow, tolerance=5e-8)
        q = np.asarray(flow, dtype=float)
        nodes = [self.source]
        incidence = np.zeros(len(self.arcs), dtype=float)
        current = self.source_index
        while current != self.target_index:
            candidates = self.outgoing[current]
            mass = float(np.sum(q[list(candidates)]))
            if not candidates or mass <= 0.0:
                raise FloatingPointError("flow sampler reached a dead end")
            probabilities = q[list(candidates)] / mass
            selected = int(rng.choice(candidates, p=probabilities))
            incidence[selected] = 1.0
            head = self.arcs[selected].head
            nodes.append(head)
            current = self.node_index[head]
        return tuple(nodes), incidence

    def path_incidence(self, path: Sequence[Node]) -> np.ndarray:
        edges = self.path_edge_indices(path)
        incidence = np.zeros(len(self.arcs), dtype=float)
        incidence[list(edges)] = 1.0
        if int(np.sum(incidence)) != len(edges):
            raise ValueError("path repeats a directed arc")
        return incidence

    def path_edge_indices(self, path: Sequence[Node]) -> tuple[int, ...]:
        row = tuple(path)
        if len(row) < 2 or row[0] != self.source or row[-1] != self.target:
            raise ValueError("path must join the declared source and target")
        try:
            edges = tuple(
                self._arc_index_by_pair[(tail, head)]
                for tail, head in zip(row, row[1:])
            )
        except KeyError as exc:
            raise ValueError(f"path uses missing arc {exc.args[0]!r}") from exc
        if len(set(edges)) != len(edges):
            raise ValueError("path repeats a directed arc")
        return edges

    def path_probability(self, path: Sequence[Node], flow: np.ndarray) -> float:
        edges = self.path_edge_indices(path)
        q = np.asarray(flow, dtype=float)
        node_mass = self.node_flow(q)
        probability = 1.0
        for edge in edges:
            tail = self.node_index[self.arcs[int(edge)].tail]
            probability *= q[edge] / node_mass[tail]
        return float(probability)


class ReducedFlowFTRLSolver:
    """Warm-started Newton minimization on the affine unit-flow polytope."""

    def __init__(
        self,
        geometry: DAGUnitFlow,
        *,
        tolerance: float = 1e-9,
        max_iterations: int = 40,
        minimum_argument: float = 1e-14,
    ) -> None:
        if tolerance <= 0.0 or max_iterations < 1 or minimum_argument <= 0.0:
            raise ValueError("invalid reduced Newton solver controls")
        self.geometry = geometry
        self.tolerance = float(tolerance)
        self.max_iterations = int(max_iterations)
        self.minimum_argument = float(minimum_argument)

    def solve(
        self,
        linear: np.ndarray,
        regularizer: FlowRegularizer,
        *,
        warm_start: np.ndarray | None = None,
    ) -> tuple[np.ndarray, FlowFTRLSolveStats]:
        started = time.perf_counter()
        coefficient = np.asarray(linear, dtype=float)
        if coefficient.shape != (len(self.geometry.arcs),):
            raise ValueError("linear coefficient must have one entry per arc")
        if not np.all(np.isfinite(coefficient)):
            raise ValueError("linear coefficient must be finite")
        if regularizer.mapping.shape[1] != len(self.geometry.arcs):
            raise ValueError("regularizer mapping has the wrong edge dimension")
        q = (
            self.geometry.uniform_flow.copy()
            if warm_start is None
            else np.asarray(warm_start, dtype=float).copy()
        )
        self.geometry.validate_flow(q, tolerance=5e-8)
        mapped = regularizer.mapping @ q
        if np.any(mapped <= self.minimum_argument):
            q = self.geometry.uniform_flow.copy()
            mapped = regularizer.mapping @ q
        if np.any(mapped <= self.minimum_argument):
            raise FloatingPointError("regularizer is not interior at uniform flow")

        null = self.geometry.null_basis
        if null.shape[1] == 0:
            value = _flow_objective(coefficient, q, regularizer)
            return q, FlowFTRLSolveStats(
                iterations=0,
                projected_gradient_inf=0.0,
                newton_decrement_squared_half=0.0,
                objective=value,
                converged=True,
                line_search_backtracks=0,
                elapsed_seconds=time.perf_counter() - started,
            )
        mapped_null = regularizer.mapping @ null
        projected_linear = null.T @ coefficient
        backtracks = 0
        converged = False
        residual = math.inf
        decrement_squared_half = math.inf
        iterations = 0
        objective = _flow_objective(coefficient, q, regularizer)

        for iteration in range(1, self.max_iterations + 1):
            iterations = iteration
            mapped = regularizer.mapping @ q
            derivative, curvature = _regularizer_derivatives(mapped, regularizer)
            gradient = projected_linear + mapped_null.T @ derivative
            residual = float(np.max(np.abs(gradient)))
            scale = 1.0 + float(np.max(np.abs(projected_linear), initial=0.0))
            if residual <= self.tolerance * scale:
                converged = True
                break
            hessian = mapped_null.T @ (curvature[:, None] * mapped_null)
            try:
                direction_reduced = np.linalg.solve(hessian, -gradient)
            except np.linalg.LinAlgError as exc:
                raise FloatingPointError("reduced FTRL Hessian is singular") from exc
            direction = null @ direction_reduced
            decrement_squared_half = max(
                0.0, -0.5 * float(gradient @ direction_reduced)
            )
            if decrement_squared_half <= self.tolerance:
                converged = True
                break
            mapped_direction = regularizer.mapping @ direction
            negative = mapped_direction < 0.0
            step = 1.0
            if np.any(negative):
                step = min(
                    step,
                    0.99 * float(np.min(
                        (mapped[negative] - self.minimum_argument)
                        / -mapped_direction[negative]
                    )),
                )
            directional_derivative = float(gradient @ direction_reduced)
            if directional_derivative >= 0.0:
                raise FloatingPointError("Newton direction is not descending")
            accepted = False
            for _ in range(60):
                candidate = q + step * direction
                candidate_mapped = regularizer.mapping @ candidate
                if np.all(candidate_mapped > self.minimum_argument):
                    candidate_objective = _flow_objective(
                        coefficient, candidate, regularizer
                    )
                    if candidate_objective <= (
                        objective + 1e-4 * step * directional_derivative
                    ):
                        q = candidate
                        objective = candidate_objective
                        accepted = True
                        break
                step *= 0.5
                backtracks += 1
            if not accepted:
                raise FloatingPointError("FTRL Newton line search failed")

        mapped = regularizer.mapping @ q
        derivative, _curvature = _regularizer_derivatives(mapped, regularizer)
        final_gradient = projected_linear + mapped_null.T @ derivative
        residual = float(np.max(np.abs(final_gradient)))
        final_hessian = mapped_null.T @ (
            _curvature[:, None] * mapped_null
        )
        try:
            final_direction = np.linalg.solve(final_hessian, -final_gradient)
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError("reduced FTRL Hessian is singular") from exc
        decrement_squared_half = max(
            0.0, -0.5 * float(final_gradient @ final_direction)
        )
        scale = 1.0 + float(np.max(np.abs(projected_linear), initial=0.0))
        converged = (
            converged
            or residual <= self.tolerance * scale
            or decrement_squared_half <= self.tolerance
        )
        if not converged:
            raise FloatingPointError(
                "FTRL optimizer did not converge: projected-gradient residual "
                f"{residual:.3e}, Newton decrement squared / 2 "
                f"{decrement_squared_half:.3e} after {iterations} iterations"
            )
        self.geometry.validate_flow(q, tolerance=5e-8)
        return q, FlowFTRLSolveStats(
            iterations=iterations,
            projected_gradient_inf=residual,
            newton_decrement_squared_half=decrement_squared_half,
            objective=objective,
            converged=True,
            line_search_backtracks=backtracks,
            elapsed_seconds=time.perf_counter() - started,
        )


@dataclass(frozen=True)
class CentroidDAGTransform:
    """Bijection between original DAG paths and Maiti's converted DAG paths."""

    original: DAGUnitFlow
    transformed: DAGUnitFlow
    transformed_arc_original_indices: tuple[tuple[int, ...], ...]
    tree_arc_indices: frozenset[int]
    centroid_components: dict[Node, frozenset[Node]]

    def to_transformed_path(self, path: Sequence[Node]) -> tuple[Node, ...]:
        original_path = tuple(path)
        edges = self.original.path_edge_indices(original_path)
        transformed_nodes: list[Node] = [_flat(original_path[0])]
        segment_start = original_path[0]
        for position, edge in enumerate(edges):
            arc = self.original.arcs[edge]
            if edge in self.tree_arc_indices:
                continue
            self._append_tree_segment(
                transformed_nodes,
                segment_start,
                arc.tail,
            )
            transformed_nodes.append(_flat(arc.head))
            segment_start = arc.head
        self._append_tree_segment(
            transformed_nodes,
            segment_start,
            original_path[-1],
        )
        result = tuple(transformed_nodes)
        self.transformed.path_incidence(result)
        return result

    def _append_tree_segment(
        self,
        transformed_nodes: list[Node],
        start: Node,
        end: Node,
    ) -> None:
        centroid = self._segment_centroid(start, end)
        expected = _flat(start)
        if transformed_nodes[-1] != expected:
            raise FloatingPointError("invalid transformed path segment boundary")
        transformed_nodes.extend((_centroid(centroid), _sharp(end)))

    def _segment_centroid(self, start: Node, end: Node) -> Node:
        candidates = [
            centroid
            for centroid, component in self.centroid_components.items()
            if start in component
            and end in component
            and _tree_path_contains(self.original, start, end, centroid,
                                    self.tree_arc_indices)
        ]
        if len(candidates) != 1:
            raise FloatingPointError(
                f"tree segment {start!r}->{end!r} has {len(candidates)} centroids"
            )
        return candidates[0]

    def to_original_path(self, path: Sequence[Node]) -> tuple[Node, ...]:
        transformed_edges = self.transformed.path_edge_indices(path)
        current = self.original.source
        nodes = [current]
        for transformed_edge in transformed_edges:
            for original_edge in self.transformed_arc_original_indices[
                int(transformed_edge)
            ]:
                arc = self.original.arcs[original_edge]
                if arc.tail != current:
                    raise FloatingPointError("centroid path mapping is discontinuous")
                nodes.append(arc.head)
                current = arc.head
        result = tuple(nodes)
        self.original.path_incidence(result)
        return result

    def diagnostics(self) -> dict[str, int | str | bool]:
        return {
            "schema": "maiti_centroid_dag_transform.v1",
            "source": "Maiti et al., COLT 2025, Section 3.3",
            "original_nodes": len(self.original.nodes),
            "original_edges": len(self.original.arcs),
            "original_paths": self.original.path_count,
            "original_longest_path": self.original.maximum_path_hops,
            "transformed_nodes": len(self.transformed.nodes),
            "transformed_edges": len(self.transformed.arcs),
            "transformed_paths": self.transformed.path_count,
            "transformed_longest_path": self.transformed.maximum_path_hops,
            "path_count_preserved": (
                self.original.path_count == self.transformed.path_count
            ),
        }


@dataclass(frozen=True)
class BoundedLayeredDAGTransform:
    """Bijection between paths of at most ``max_hops`` and a layered DAG."""

    original: DAGUnitFlow
    layered: DAGUnitFlow
    layered_arc_original_indices: tuple[tuple[int, ...], ...]
    max_hops: int

    def to_layered_path(self, path: Sequence[Node]) -> tuple[Node, ...]:
        original = tuple(path)
        edges = self.original.path_edge_indices(original)
        if len(edges) > self.max_hops:
            raise ValueError("path exceeds layered hop bound")
        nodes: list[Node] = [_layer(0, original[0])]
        for hop, node in enumerate(original[1:], start=1):
            nodes.append(_layer(hop, node))
        nodes.append(_bounded_target(self.original.target, self.max_hops))
        result = tuple(nodes)
        self.layered.path_edge_indices(result)
        return result

    def to_original_path(self, path: Sequence[Node]) -> tuple[Node, ...]:
        layered_edges = self.layered.path_edge_indices(path)
        current = self.original.source
        nodes = [current]
        for layered_edge in layered_edges:
            for original_edge in self.layered_arc_original_indices[layered_edge]:
                arc = self.original.arcs[original_edge]
                if arc.tail != current:
                    raise FloatingPointError("layered path mapping is discontinuous")
                nodes.append(arc.head)
                current = arc.head
        result = tuple(nodes)
        self.original.path_edge_indices(result)
        if len(result) - 1 > self.max_hops:
            raise FloatingPointError("layered path exceeded declared hop bound")
        return result

    def diagnostics(self) -> dict[str, int | str | bool]:
        return {
            "schema": "bounded_layered_dag_transform.v1",
            "route_hops_maximum": self.max_hops,
            "raw_dag_path_count": self.original.path_count,
            "bounded_path_count": self.layered.path_count,
            "raw_dag_longest_path": self.original.maximum_path_hops,
            "layered_longest_path": self.layered.maximum_path_hops,
            "layered_nodes": len(self.layered.nodes),
            "layered_edges": len(self.layered.arcs),
            "cycle_free_source_graph": True,
        }


def build_bounded_layered_dag_transform(
    geometry: DAGUnitFlow,
    *,
    max_hops: int,
) -> BoundedLayeredDAGTransform:
    """Encode the declared inclusive hop bound without enumerating paths."""

    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")
    target = _bounded_target(geometry.target, max_hops)
    rows: list[tuple[DirectedPathArc, tuple[int, ...]]] = []
    for hop in range(max_hops):
        for edge, arc in enumerate(geometry.arcs):
            rows.append((
                DirectedPathArc(
                    f"layer::{hop}::{arc.arc_id}",
                    _layer(hop, arc.tail),
                    _layer(hop + 1, arc.head),
                ),
                (edge,),
            ))
    for hop in range(1, max_hops + 1):
        rows.append((
            DirectedPathArc(
                f"layer_terminal::{hop}",
                _layer(hop, geometry.target),
                target,
            ),
            (),
        ))
    rows.sort(key=lambda row: row[0].arc_id)
    rows = _prune_transformed_rows(
        rows,
        source=_layer(0, geometry.source),
        target=target,
    )
    layered = DAGUnitFlow(
        (row[0] for row in rows),
        source=_layer(0, geometry.source),
        target=target,
    )
    return BoundedLayeredDAGTransform(
        original=geometry,
        layered=layered,
        layered_arc_original_indices=tuple(row[1] for row in rows),
        max_hops=int(max_hops),
    )


def build_centroid_dag_transform(geometry: DAGUnitFlow) -> CentroidDAGTransform:
    """Implement the path-preserving graph conversion in Maiti et al. §3.3."""

    prefix = geometry.prefix_path_counts
    tree_edges: set[int] = set()
    parent: dict[Node, Node] = {}
    parent_edge: dict[Node, int] = {}
    for node_index in geometry.topological_order:
        node = geometry.nodes[node_index]
        if node == geometry.source:
            continue
        incoming = geometry.incoming[node_index]
        if not incoming:
            raise ValueError("centroid transform requires source-rooted DAG nodes")
        chosen = min(
            incoming,
            key=lambda edge: (
                -prefix[geometry.node_index[geometry.arcs[edge].tail]],
                geometry.arcs[edge].arc_id,
            ),
        )
        tree_edges.add(chosen)
        parent[node] = geometry.arcs[chosen].tail
        parent_edge[node] = chosen

    adjacency: dict[Node, set[Node]] = {node: set() for node in geometry.nodes}
    for edge in tree_edges:
        arc = geometry.arcs[edge]
        adjacency[arc.tail].add(arc.head)
        adjacency[arc.head].add(arc.tail)
    components: dict[Node, frozenset[Node]] = {}

    def decompose(component: frozenset[Node]) -> None:
        centroid = _tree_centroid(component, adjacency)
        components[centroid] = component
        remaining = set(component)
        remaining.remove(centroid)
        while remaining:
            root = min(remaining, key=str)
            child_component = _component_without(
                root, remaining, adjacency, blocked=centroid
            )
            decompose(frozenset(child_component))
            remaining.difference_update(child_component)

    decompose(frozenset(geometry.nodes))

    transformed_rows: list[tuple[DirectedPathArc, tuple[int, ...]]] = []
    for centroid in sorted(components, key=str):
        component = components[centroid]
        for node in sorted(component, key=str):
            if _is_ancestor(parent, node, centroid):
                mapping = _directed_tree_edges(
                    parent, parent_edge, node, centroid
                )
                transformed_rows.append((
                    DirectedPathArc(
                        f"tree_in::{centroid!s}::{node!s}",
                        _flat(node),
                        _centroid(centroid),
                    ),
                    mapping,
                ))
            if _is_ancestor(parent, centroid, node):
                mapping = _directed_tree_edges(
                    parent, parent_edge, centroid, node
                )
                transformed_rows.append((
                    DirectedPathArc(
                        f"tree_out::{centroid!s}::{node!s}",
                        _centroid(centroid),
                        _sharp(node),
                    ),
                    mapping,
                ))
    for edge, arc in enumerate(geometry.arcs):
        if edge not in tree_edges:
            transformed_rows.append((
                DirectedPathArc(
                    f"non_tree::{arc.arc_id}",
                    _sharp(arc.tail),
                    _flat(arc.head),
                ),
                (edge,),
            ))
    transformed_rows.sort(key=lambda row: row[0].arc_id)
    transformed_rows = _prune_transformed_rows(
        transformed_rows,
        source=_flat(geometry.source),
        target=_sharp(geometry.target),
    )
    transformed = DAGUnitFlow(
        (row[0] for row in transformed_rows),
        source=_flat(geometry.source),
        target=_sharp(geometry.target),
    )
    result = CentroidDAGTransform(
        original=geometry,
        transformed=transformed,
        transformed_arc_original_indices=tuple(row[1] for row in transformed_rows),
        tree_arc_indices=frozenset(tree_edges),
        centroid_components=components,
    )
    if transformed.path_count != geometry.path_count:
        raise FloatingPointError(
            "centroid graph transformation did not preserve path count"
        )
    return result


def _flow_objective(
    linear: np.ndarray,
    flow: np.ndarray,
    regularizer: FlowRegularizer,
) -> float:
    mapped = regularizer.mapping @ flow
    if np.any(mapped <= 0.0):
        return math.inf
    return float(
        linear @ flow
        - regularizer.sqrt_coefficients @ np.sqrt(mapped)
        - regularizer.log_coefficients @ np.log(mapped)
    )


def _regularizer_derivatives(
    mapped: np.ndarray,
    regularizer: FlowRegularizer,
) -> tuple[np.ndarray, np.ndarray]:
    sqrt = regularizer.sqrt_coefficients
    log = regularizer.log_coefficients
    derivative = -sqrt / (2.0 * np.sqrt(mapped)) - log / mapped
    curvature = sqrt / (4.0 * mapped ** 1.5) + log / mapped**2
    return derivative, curvature


def _prune_transformed_rows(
    rows: list[tuple[DirectedPathArc, tuple[int, ...]]],
    *,
    source: Node,
    target: Node,
) -> list[tuple[DirectedPathArc, tuple[int, ...]]]:
    """Drop conversion arcs that cannot participate in an s--t path.

    Section 3.3 states the complete conversion.  Some generated arcs are
    unreachable for a particular declared source/target pair; removing them
    leaves the path bijection and flow polytope unchanged while ensuring every
    numerical variable has positive probability under the interior flow.
    """

    outgoing: dict[Node, list[Node]] = {}
    incoming: dict[Node, list[Node]] = {}
    for arc, _mapping in rows:
        outgoing.setdefault(arc.tail, []).append(arc.head)
        incoming.setdefault(arc.head, []).append(arc.tail)
    reachable = {source}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for head in outgoing.get(node, ()):  # pragma: no branch - tiny helper
            if head not in reachable:
                reachable.add(head)
                queue.append(head)
    co_reachable = {target}
    queue = deque([target])
    while queue:
        node = queue.popleft()
        for tail in incoming.get(node, ()):  # pragma: no branch - tiny helper
            if tail not in co_reachable:
                co_reachable.add(tail)
                queue.append(tail)
    return [
        row for row in rows
        if row[0].tail in reachable and row[0].head in co_reachable
    ]


def _topological_order(
    *,
    arcs: tuple[DirectedPathArc, ...],
    nodes: tuple[Node, ...],
    node_index: dict[Node, int],
    outgoing: list[list[int]],
    indegree: np.ndarray,
) -> tuple[int, ...]:
    remaining = indegree.copy()
    queue = deque(sorted(
        (index for index, value in enumerate(remaining) if value == 0),
        key=lambda index: str(nodes[index]),
    ))
    result: list[int] = []
    while queue:
        node = queue.popleft()
        result.append(node)
        for edge in outgoing[node]:
            head = node_index[arcs[edge].head]
            remaining[head] -= 1
            if remaining[head] == 0:
                queue.append(head)
    if len(result) != len(nodes):
        raise ValueError("modern path learners require a directed acyclic graph")
    return tuple(result)


def _flat(node: Node) -> tuple[str, Node]:
    return ("flat", node)


def _sharp(node: Node) -> tuple[str, Node]:
    return ("sharp", node)


def _centroid(node: Node) -> tuple[str, Node]:
    return ("centroid", node)


def _layer(hop: int, node: Node) -> tuple[str, int, Node]:
    return ("layer", int(hop), node)


def _bounded_target(node: Node, max_hops: int) -> tuple[str, Node, int]:
    return ("bounded_target", node, int(max_hops))


def _is_ancestor(parent: dict[Node, Node], ancestor: Node, node: Node) -> bool:
    current = node
    while True:
        if current == ancestor:
            return True
        if current not in parent:
            return False
        current = parent[current]


def _directed_tree_edges(
    parent: dict[Node, Node],
    parent_edge: dict[Node, int],
    ancestor: Node,
    descendant: Node,
) -> tuple[int, ...]:
    if ancestor == descendant:
        return ()
    reverse: list[int] = []
    current = descendant
    while current != ancestor:
        if current not in parent:
            raise ValueError("requested nodes do not form a directed tree path")
        reverse.append(parent_edge[current])
        current = parent[current]
    return tuple(reversed(reverse))


def _tree_centroid(
    component: frozenset[Node],
    adjacency: dict[Node, set[Node]],
) -> Node:
    if not component:
        raise ValueError("cannot find centroid of an empty component")
    best: tuple[int, str, Node] | None = None
    for candidate in component:
        largest = 0
        remaining = set(component)
        remaining.remove(candidate)
        while remaining:
            root = min(remaining, key=str)
            part = _component_without(
                root, remaining, adjacency, blocked=candidate
            )
            largest = max(largest, len(part))
            remaining.difference_update(part)
        row = (largest, str(candidate), candidate)
        if best is None or row[:2] < best[:2]:
            best = row
    assert best is not None
    return best[2]


def _component_without(
    root: Node,
    allowed: set[Node],
    adjacency: dict[Node, set[Node]],
    *,
    blocked: Node,
) -> set[Node]:
    seen = {root}
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor != blocked and neighbor in allowed and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return seen


def _tree_path_contains(
    geometry: DAGUnitFlow,
    start: Node,
    end: Node,
    candidate: Node,
    tree_edges: frozenset[int],
) -> bool:
    adjacency: dict[Node, set[Node]] = {node: set() for node in geometry.nodes}
    for edge in tree_edges:
        arc = geometry.arcs[edge]
        adjacency[arc.tail].add(arc.head)
        adjacency[arc.head].add(arc.tail)
    parent: dict[Node, Node | None] = {start: None}
    queue = deque([start])
    while queue:
        current = queue.popleft()
        if current == end:
            break
        for neighbor in adjacency[current]:
            if neighbor not in parent:
                parent[neighbor] = current
                queue.append(neighbor)
    if end not in parent:
        return False
    current: Node | None = end
    while current is not None:
        if current == candidate:
            return True
        current = parent[current]
    return False
