"""Modern aggregate-feedback online shortest-path learners.

This module implements the algorithmic objects described in:

* Maiti, Fan, Jamieson, Ratliff, and Farina, COLT 2025;
* Ito, Jamieson, Luo, Maiti, and Tsuchiya, NeurIPS 2025.

Both learners observe only one bounded aggregate loss for their selected path.
They share an exact unit-flow representation and Markovian sampler, but retain
their distinct estimators, regularizers, and schedules.  Maiti's learner uses
the paper's centroid DAG conversion and variable-length auxiliary coordinates.
Ito's learner works directly on the original DAG and offers both regularizers
analyzed in the paper.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np

from .dag_flow import (
    CentroidDAGTransform,
    DAGUnitFlow,
    FlowFTRLSolveStats,
    FlowRegularizer,
    ReducedFlowFTRLSolver,
    build_centroid_dag_transform,
)


ITO_REGULARIZER_TSALLIS = "tsallis"
ITO_REGULARIZER_LOG_BARRIER = "log_barrier"
ITO_REGULARIZERS = frozenset({
    ITO_REGULARIZER_TSALLIS,
    ITO_REGULARIZER_LOG_BARRIER,
})


@dataclass(frozen=True)
class AggregatePathSchedulePoint:
    """Serializable per-round controls for aggregate-feedback FTRL."""

    turn: int
    eta: float
    gamma: float | None = None
    beta: float | None = None
    minimum_eta: float | None = None
    maximum_eta: float | None = None


class _SolverTelemetry:
    def __init__(self) -> None:
        self.solves = 0
        self.iterations = 0
        self.backtracks = 0
        self.elapsed_seconds = 0.0
        self.maximum_residual = 0.0
        self.maximum_newton_decrement_squared_half = 0.0
        self.maximum_iterations = 0

    def observe(self, stats: FlowFTRLSolveStats) -> None:
        self.solves += 1
        self.iterations += stats.iterations
        self.backtracks += stats.line_search_backtracks
        self.elapsed_seconds += stats.elapsed_seconds
        self.maximum_residual = max(
            self.maximum_residual, stats.projected_gradient_inf
        )
        self.maximum_newton_decrement_squared_half = max(
            self.maximum_newton_decrement_squared_half,
            stats.newton_decrement_squared_half,
        )
        self.maximum_iterations = max(self.maximum_iterations, stats.iterations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solve_count": self.solves,
            "mean_newton_iterations": (
                self.iterations / self.solves if self.solves else 0.0
            ),
            "maximum_newton_iterations": self.maximum_iterations,
            "line_search_backtracks": self.backtracks,
            "solver_elapsed_seconds": self.elapsed_seconds,
            "maximum_projected_gradient_residual": self.maximum_residual,
            "maximum_newton_decrement_squared_half": (
                self.maximum_newton_decrement_squared_half
            ),
        }


class MaitiDAGPathFTRL:
    """Maiti et al. 2025 high-probability DAG path learner.

    The implementation follows Algorithm 2 after applying the Section 3.3
    centroid conversion.  Losses lie in ``[0, 1]``, a valid subset of the
    paper's declared aggregate-loss range ``[-1, 1]``.
    """

    def __init__(
        self,
        geometry: DAGUnitFlow,
        *,
        horizon: int,
        failure_probability: float = 0.05,
        eta_scale: float = 1.0,
        implicit_exploration_scale: float = 1.0,
        rng: np.random.Generator | None = None,
        solver_tolerance: float = 1e-6,
        solver_max_iterations: int = 60,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if not 0.0 < failure_probability < 1.0:
            raise ValueError("failure_probability must be in (0, 1)")
        if eta_scale <= 0.0 or implicit_exploration_scale <= 0.0:
            raise ValueError("Maiti tuning scales must be > 0")
        self.original_geometry = geometry
        self.transform: CentroidDAGTransform = build_centroid_dag_transform(
            geometry
        )
        self.geometry = self.transform.transformed
        self.horizon = int(horizon)
        self.failure_probability = float(failure_probability)
        self.eta_scale = float(eta_scale)
        self.implicit_exploration_scale = float(
            implicit_exploration_scale
        )
        self.eta = self.eta_scale / math.sqrt(self.horizon)
        self.gamma = self.implicit_exploration_scale * math.sqrt(
            self.geometry.maximum_path_hops
            * math.log(
                5.0
                * (
                    len(self.geometry.nodes)
                    + len(self.geometry.arcs)
                    + self.geometry.maximum_path_hops
                )
                / self.failure_probability
            )
            / (len(self.geometry.arcs) * self.horizon)
        )
        self.rng = rng if rng is not None else np.random.default_rng()
        self.solver = ReducedFlowFTRLSolver(
            self.geometry,
            tolerance=solver_tolerance,
            max_iterations=solver_max_iterations,
        )
        self._node_mapping = self._internal_node_mapping()
        self._aux_mapping = self._variable_length_auxiliary_mapping()
        maps = [np.eye(len(self.geometry.arcs)), self._node_mapping]
        if self._aux_mapping.shape[0]:
            maps.append(self._aux_mapping)
        self._regularizer_mapping = np.vstack(maps)
        count = self._regularizer_mapping.shape[0]
        self._regularizer = FlowRegularizer(
            self._regularizer_mapping,
            np.ones(count, dtype=float),
            np.zeros(count, dtype=float),
        )
        self._cumulative_linear = np.zeros(len(self.geometry.arcs), dtype=float)
        self._counterfactual_linear_correction = np.zeros_like(
            self._cumulative_linear
        )
        self._flow = self.geometry.uniform_flow.copy()
        self._counterfactual_flow = self._flow.copy()
        self._flow_dirty = True
        self._counterfactual_dirty = True
        self._last_transformed_path: tuple[Any, ...] | None = None
        self._last_original_path: tuple[Any, ...] | None = None
        self._last_incidence: np.ndarray | None = None
        self._last_flow: np.ndarray | None = None
        self._last_loss: float | None = None
        self._last_schedule: AggregatePathSchedulePoint | None = None
        self.turns = 0
        self.counts: Counter[tuple[Any, ...]] = Counter()
        self.solver_telemetry = _SolverTelemetry()
        self.counterfactual_solver_telemetry = _SolverTelemetry()

    def _internal_node_mapping(self) -> np.ndarray:
        rows = [
            self.geometry.tail_incidence[index]
            for index, node in enumerate(self.geometry.nodes)
            if node not in {self.geometry.source, self.geometry.target}
            and self.geometry.outgoing[index]
        ]
        return (
            np.asarray(rows, dtype=float)
            if rows
            else np.zeros((0, len(self.geometry.arcs)), dtype=float)
        )

    def _variable_length_auxiliary_mapping(self) -> np.ndarray:
        longest = np.full(len(self.geometry.nodes), -10**9, dtype=int)
        longest[self.geometry.source_index] = 0
        for node in self.geometry.topological_order:
            if longest[node] < 0:
                continue
            for edge in self.geometry.outgoing[node]:
                head = self.geometry.node_index[self.geometry.arcs[edge].head]
                longest[head] = max(longest[head], longest[node] + 1)
        maximum = int(longest[self.geometry.target_index])
        rows = []
        for level in range(1, maximum):
            row = np.zeros(len(self.geometry.arcs), dtype=float)
            for edge, arc in enumerate(self.geometry.arcs):
                tail = self.geometry.node_index[arc.tail]
                head = self.geometry.node_index[arc.head]
                if longest[tail] < level < longest[head]:
                    row[edge] = 1.0
            if np.any(row):
                rows.append(row)
        return (
            np.asarray(rows, dtype=float)
            if rows
            else np.zeros((0, len(self.geometry.arcs)), dtype=float)
        )

    def _solve_factual(self) -> None:
        if not self._flow_dirty:
            return
        self._flow, stats = self.solver.solve(
            self.eta * self._cumulative_linear,
            self._regularizer,
            warm_start=self._flow,
        )
        self.solver_telemetry.observe(stats)
        self._flow_dirty = False

    def _solve_counterfactual(self) -> None:
        if not self._counterfactual_dirty:
            return
        self._counterfactual_flow, stats = self.solver.solve(
            self.eta
            * (
                self._cumulative_linear
                + self._counterfactual_linear_correction
            ),
            self._regularizer,
            warm_start=self._counterfactual_flow,
        )
        self.counterfactual_solver_telemetry.observe(stats)
        self._counterfactual_dirty = False

    def sample(self) -> tuple[Any, ...]:
        self._solve_factual()
        path, incidence = self.geometry.sample(self._flow, rng=self.rng)
        original = self.transform.to_original_path(path)
        self._last_transformed_path = path
        self._last_original_path = original
        self._last_incidence = incidence
        self._last_flow = self._flow.copy()
        self._last_schedule = AggregatePathSchedulePoint(
            turn=self.turns + 1,
            eta=self.eta,
            gamma=self.gamma,
        )
        self.counts[original] += 1
        self.turns += 1
        return original

    def _estimated_linear(self, loss: float) -> np.ndarray:
        if self._last_incidence is None or self._last_flow is None:
            raise RuntimeError("loss estimator requested before sample()")
        incidence = self._last_incidence
        q = self._last_flow
        edge_estimate = (
            (1.0 + loss) * incidence / (q + self.gamma)
        )
        node_incidence = self._node_mapping @ incidence
        node_flow = self._node_mapping @ q
        node_estimate = (
            (1.0 - loss) * node_incidence / (node_flow + self.gamma)
        )
        result = edge_estimate + self._node_mapping.T @ node_estimate
        if self._aux_mapping.shape[0]:
            aux_incidence = self._aux_mapping @ incidence
            aux_flow = self._aux_mapping @ q
            aux_estimate = 2.0 * aux_incidence / (aux_flow + self.gamma)
            result = result + self._aux_mapping.T @ aux_estimate
        return result

    def update(self, loss: float) -> None:
        value = _bounded_signed_loss(loss)
        self._cumulative_linear += self._estimated_linear(value)
        self._last_loss = value
        self._flow_dirty = True
        self._counterfactual_dirty = True

    def add_counterfactual_loss_correction(self, loss_delta: float) -> None:
        if self._last_incidence is None or self._last_flow is None:
            raise RuntimeError("counterfactual correction requested before sample()")
        delta = float(loss_delta)
        if not math.isfinite(delta):
            raise ValueError("loss_delta must be finite")
        incidence = self._last_incidence
        q = self._last_flow
        correction = delta * incidence / (q + self.gamma)
        if self._node_mapping.shape[0]:
            node_incidence = self._node_mapping @ incidence
            node_flow = self._node_mapping @ q
            correction -= self._node_mapping.T @ (
                delta * node_incidence / (node_flow + self.gamma)
            )
        self._counterfactual_linear_correction += correction
        self._counterfactual_dirty = True

    @property
    def flow(self) -> np.ndarray:
        self._solve_factual()
        return self._flow.copy()

    @property
    def counterfactual_flow(self) -> np.ndarray:
        self._solve_counterfactual()
        return self._counterfactual_flow.copy()

    @property
    def last_schedule(self) -> AggregatePathSchedulePoint | None:
        return self._last_schedule

    def diagnostics(self) -> dict[str, Any]:
        return {
            "algorithm": "Maiti_COLT2025_centroid_Tsallis_FTRL",
            "horizon": self.horizon,
            "failure_probability": self.failure_probability,
            "eta": self.eta,
            "eta_scale": self.eta_scale,
            "implicit_exploration_gamma": self.gamma,
            "implicit_exploration_scale": self.implicit_exploration_scale,
            "centroid_transform": self.transform.diagnostics(),
            "variable_length_auxiliary_coordinates": int(
                self._aux_mapping.shape[0]
            ),
            "solver": self.solver_telemetry.to_dict(),
            "counterfactual_solver": (
                self.counterfactual_solver_telemetry.to_dict()
            ),
        }


class ItoDAGPathFTRL:
    """Ito et al. 2025 best-of-both-worlds DAG path learner."""

    def __init__(
        self,
        geometry: DAGUnitFlow,
        *,
        horizon: int,
        regularizer: str = ITO_REGULARIZER_TSALLIS,
        eta_scale: float = 1.0,
        beta: float = 2.0,
        rng: np.random.Generator | None = None,
        solver_tolerance: float = 1e-6,
        solver_max_iterations: int = 60,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        if regularizer not in ITO_REGULARIZERS:
            raise ValueError(f"unsupported Ito regularizer {regularizer!r}")
        if eta_scale <= 0.0 or beta <= 0.0:
            raise ValueError("Ito eta_scale and beta must be > 0")
        self.geometry = geometry
        self.horizon = int(horizon)
        self.regularizer_mode = regularizer
        self.eta_scale = float(eta_scale)
        self.beta = float(beta)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.solver = ReducedFlowFTRLSolver(
            geometry,
            tolerance=solver_tolerance,
            max_iterations=solver_max_iterations,
        )
        self._cumulative_linear = np.zeros(len(geometry.arcs), dtype=float)
        self._counterfactual_linear_correction = np.zeros_like(
            self._cumulative_linear
        )
        self._cumulative_rho = np.zeros(len(geometry.arcs), dtype=float)
        self._counterfactual_rho_correction = np.zeros_like(
            self._cumulative_rho
        )
        self._flow = geometry.uniform_flow.copy()
        self._counterfactual_flow = self._flow.copy()
        self._flow_dirty = True
        self._counterfactual_dirty = True
        self._last_path: tuple[Any, ...] | None = None
        self._last_incidence: np.ndarray | None = None
        self._last_flow: np.ndarray | None = None
        self._last_loss: float | None = None
        self._last_schedule: AggregatePathSchedulePoint | None = None
        self.turns = 0
        self.counts: Counter[tuple[Any, ...]] = Counter()
        self.solver_telemetry = _SolverTelemetry()
        self.counterfactual_solver_telemetry = _SolverTelemetry()

    def _regularizer(
        self,
        turn: int,
        *,
        counterfactual: bool,
    ) -> tuple[FlowRegularizer, AggregatePathSchedulePoint]:
        identity = np.eye(len(self.geometry.arcs), dtype=float)
        if self.regularizer_mode == ITO_REGULARIZER_TSALLIS:
            eta = self.eta_scale / math.sqrt(turn)
            return (
                FlowRegularizer(
                    identity,
                    np.full(len(self.geometry.arcs), 2.0 / eta),
                    np.full(len(self.geometry.arcs), self.beta),
                ),
                AggregatePathSchedulePoint(
                    turn=turn, eta=eta, beta=self.beta
                ),
            )
        rho = self._cumulative_rho.copy()
        if counterfactual:
            rho += self._counterfactual_rho_correction
        denominator = max(1.0, math.log(self.horizon))
        eta_edges = self.eta_scale / np.sqrt(4.0 + rho / denominator)
        return (
            FlowRegularizer(
                identity,
                np.zeros(len(self.geometry.arcs), dtype=float),
                1.0 / eta_edges,
            ),
            AggregatePathSchedulePoint(
                turn=turn,
                eta=float(np.mean(eta_edges)),
                minimum_eta=float(np.min(eta_edges)),
                maximum_eta=float(np.max(eta_edges)),
            ),
        )

    def _solve_factual(self) -> AggregatePathSchedulePoint:
        regularizer, schedule = self._regularizer(
            self.turns + 1, counterfactual=False
        )
        if self._flow_dirty:
            self._flow, stats = self.solver.solve(
                self._cumulative_linear,
                regularizer,
                warm_start=self._flow,
            )
            self.solver_telemetry.observe(stats)
            self._flow_dirty = False
        return schedule

    def _solve_counterfactual(self) -> None:
        if not self._counterfactual_dirty:
            return
        regularizer, _schedule = self._regularizer(
            self.turns + 1, counterfactual=True
        )
        self._counterfactual_flow, stats = self.solver.solve(
            self._cumulative_linear + self._counterfactual_linear_correction,
            regularizer,
            warm_start=self._counterfactual_flow,
        )
        self.counterfactual_solver_telemetry.observe(stats)
        self._counterfactual_dirty = False

    def sample(self) -> tuple[Any, ...]:
        schedule = self._solve_factual()
        path, incidence = self.geometry.sample(self._flow, rng=self.rng)
        self._last_path = path
        self._last_incidence = incidence
        self._last_flow = self._flow.copy()
        self._last_loss = None
        self._last_schedule = schedule
        self.counts[path] += 1
        self.turns += 1
        return path

    def _estimated_linear(self, loss: float) -> np.ndarray:
        if self._last_incidence is None or self._last_flow is None:
            raise RuntimeError("loss estimator requested before sample()")
        p = self._last_incidence
        q = self._last_flow
        node_p = self.geometry.node_flow(p)
        node_q = self.geometry.node_flow(q)
        tail_p = np.asarray([
            node_p[self.geometry.node_index[arc.tail]]
            for arc in self.geometry.arcs
        ])
        tail_q = np.asarray([
            node_q[self.geometry.node_index[arc.tail]]
            for arc in self.geometry.arcs
        ])
        return loss * (p / q - tail_p / tail_q)

    def _rho(self, loss: float) -> np.ndarray:
        if self._last_incidence is None or self._last_flow is None:
            raise RuntimeError("rho requested before sample()")
        p = self._last_incidence
        q = self._last_flow
        node_q = self.geometry.node_flow(q)
        tail_q = np.asarray([
            node_q[self.geometry.node_index[arc.tail]]
            for arc in self.geometry.arcs
        ])
        return loss**2 * p * (1.0 - q / tail_q) ** 2

    def update(self, loss: float) -> None:
        value = _bounded_loss(loss)
        self._cumulative_linear += self._estimated_linear(value)
        if self.regularizer_mode == ITO_REGULARIZER_LOG_BARRIER:
            self._cumulative_rho += self._rho(value)
        self._last_loss = value
        self._flow_dirty = True
        self._counterfactual_dirty = True

    def add_counterfactual_loss_correction(self, loss_delta: float) -> None:
        if self._last_loss is not None:
            raise RuntimeError(
                "counterfactual correction must be registered before factual update"
            )
        delta = float(loss_delta)
        if not math.isfinite(delta):
            raise ValueError("loss_delta must be finite")
        self._counterfactual_linear_correction += self._estimated_linear(delta)
        if self.regularizer_mode == ITO_REGULARIZER_LOG_BARRIER:
            factual_loss = 0.0  # replaced below by adapter's absolute update
            # The public method below accepts a delta only for parity with
            # Maiti.  Log-barrier rho is quadratic, so callers must use
            # ``add_counterfactual_losses`` with both absolute losses.
            if delta != 0.0:
                raise RuntimeError(
                    "Ito log-barrier counterfactual needs absolute losses"
                )
        self._counterfactual_dirty = True

    def add_counterfactual_losses(
        self,
        factual_loss: float,
        counterfactual_loss: float,
    ) -> None:
        factual = _bounded_loss(factual_loss)
        counterfactual = _bounded_loss(counterfactual_loss)
        delta = counterfactual - factual
        self._counterfactual_linear_correction += self._estimated_linear(delta)
        if self.regularizer_mode == ITO_REGULARIZER_LOG_BARRIER:
            self._counterfactual_rho_correction += (
                self._rho(counterfactual) - self._rho(factual)
            )
        self._counterfactual_dirty = True

    @property
    def flow(self) -> np.ndarray:
        self._solve_factual()
        return self._flow.copy()

    @property
    def counterfactual_flow(self) -> np.ndarray:
        self._solve_counterfactual()
        return self._counterfactual_flow.copy()

    @property
    def last_schedule(self) -> AggregatePathSchedulePoint | None:
        return self._last_schedule

    def diagnostics(self) -> dict[str, Any]:
        return {
            "algorithm": "Ito_NeurIPS2025_BOBW_flow_FTRL",
            "horizon": self.horizon,
            "regularizer": self.regularizer_mode,
            "eta_scale": self.eta_scale,
            "beta": self.beta if self.regularizer_mode == ITO_REGULARIZER_TSALLIS else None,
            "original_nodes": len(self.geometry.nodes),
            "original_edges": len(self.geometry.arcs),
            "path_count": self.geometry.path_count,
            "maximum_path_hops": self.geometry.maximum_path_hops,
            "solver": self.solver_telemetry.to_dict(),
            "counterfactual_solver": (
                self.counterfactual_solver_telemetry.to_dict()
            ),
        }


def _bounded_loss(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("aggregate path loss must be finite and in [0, 1]")
    return result


def _bounded_signed_loss(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError("aggregate path loss must be finite and in [-1, 1]")
    return result
