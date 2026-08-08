"""Histogram adapters for the Maiti and Ito DAG path learners."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from sequence_game.game_theory import (
    DAGUnitFlow,
    ItoDAGPathFTRL,
    MaitiDAGPathFTRL,
    build_bounded_layered_dag_transform,
)

from .corpus import HistogramGameCase
from .path_polytope import build_histogram_case_path_polytope


class _HistogramModernPathAdapter:
    def __init__(self, case: HistogramGameCase) -> None:
        finite_bridge = build_histogram_case_path_polytope(case)
        raw = DAGUnitFlow(
            finite_bridge.polytope.arcs,
            source=finite_bridge.polytope.source,
            target=finite_bridge.polytope.target,
        )
        self.layered = build_bounded_layered_dag_transform(
            raw,
            max_hops=finite_bridge.polytope.max_hops,
        )
        if self.layered.layered.path_count != len(case.routes):
            raise ValueError(
                "bounded layered learner action count does not match histogram routes"
            )
        self.case = case
        self.finite_bridge = finite_bridge
        self.k = len(case.routes)
        self.counts = np.zeros(self.k, dtype=float)
        self._physical_paths = tuple(
            tuple(map(str, route["path"])) for route in case.routes
        )
        self._layered_paths = tuple(
            self.layered.to_layered_path(path) for path in self._physical_paths
        )

    def route_index(self, physical_path: Sequence[str]) -> int:
        return self.finite_bridge.route_index(physical_path)

    def empirical_strategy(self) -> np.ndarray:
        total = float(np.sum(self.counts))
        return (
            self.counts / total
            if total > 0.0
            else np.full(self.k, 1.0 / self.k)
        )

    def _normalize_policy(self, probabilities: np.ndarray) -> np.ndarray:
        values = np.asarray(probabilities, dtype=float)
        total = float(np.sum(values))
        if (
            values.shape != (self.k,)
            or not np.all(np.isfinite(values))
            or np.any(values < 0.0)
            or total <= 0.0
        ):
            raise FloatingPointError("invalid modern path learner policy")
        values /= total
        if abs(float(np.sum(values)) - 1.0) > 1e-10:
            raise FloatingPointError("modern path policy failed normalization")
        return values

    def diagnostics(self) -> dict[str, Any]:
        return {
            "bounded_action_universe": self.layered.diagnostics(),
            "physics_adapter_boundary": (
                "the learner samples an implicit bounded-DAG path; the adapter "
                "maps only that selected path to the existing compiled histogram "
                "route calibration"
            ),
            "route_table_used_inside_ftrl": False,
            "finite_route_table_used_for_reporting": True,
        }


class HistogramMaitiPathFTRL(_HistogramModernPathAdapter):
    """Maiti COLT-2025 learner mapped to histogram route indices."""

    def __init__(
        self,
        case: HistogramGameCase,
        *,
        horizon: int,
        failure_probability: float,
        eta_scale: float,
        implicit_exploration_scale: float,
        reward_loss_baseline: float = 1.0,
        rng: np.random.Generator,
        solver_tolerance: float,
        solver_max_iterations: int,
    ) -> None:
        super().__init__(case)
        if not 0.0 <= reward_loss_baseline <= 1.0:
            raise ValueError("Maiti reward_loss_baseline must be in [0, 1]")
        self.reward_loss_baseline = float(reward_loss_baseline)
        self.learner = MaitiDAGPathFTRL(
            self.layered.layered,
            horizon=horizon,
            failure_probability=failure_probability,
            eta_scale=eta_scale,
            implicit_exploration_scale=implicit_exploration_scale,
            rng=rng,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
        )

    def sample(self) -> int:
        layered_path = self.learner.sample()
        physical_path = self.layered.to_original_path(layered_path)
        route = self.route_index(physical_path)
        self.counts[route] += 1.0
        return route

    def update(self, reward: float) -> None:
        self.learner.update(
            self.reward_loss_baseline - _bounded_reward(reward)
        )

    def update_counterfactual(
        self,
        factual_reward: float,
        corrected_reward: float,
    ) -> None:
        factual_loss = (
            self.reward_loss_baseline - _bounded_reward(factual_reward)
        )
        corrected_loss = (
            self.reward_loss_baseline - _bounded_reward(corrected_reward)
        )
        self.learner.add_counterfactual_loss_correction(
            corrected_loss - factual_loss
        )

    def _policy_for_flow(self, flow: np.ndarray) -> np.ndarray:
        probabilities = np.asarray([
            self.learner.geometry.path_probability(
                self.learner.transform.to_transformed_path(layered_path),
                flow,
            )
            for layered_path in self._layered_paths
        ])
        return self._normalize_policy(probabilities)

    def policy(self, *, turn: int | None = None) -> np.ndarray:
        _validate_optional_turn(turn)
        return self._policy_for_flow(self.learner.flow)

    def counterfactual_policy(self, *, turn: int | None = None) -> np.ndarray:
        _validate_optional_turn(turn)
        return self._policy_for_flow(self.learner.counterfactual_flow)

    @property
    def last_schedule(self) -> Any:
        return self.learner.last_schedule

    def full_diagnostics(self) -> dict[str, Any]:
        return {
            **self.diagnostics(),
            **self.learner.diagnostics(),
            "reward_loss_baseline": self.reward_loss_baseline,
            "reward_to_loss_mapping": (
                f"loss = {self.reward_loss_baseline:g} - public_service_reward"
            ),
        }


class HistogramItoPathFTRL(_HistogramModernPathAdapter):
    """Ito NeurIPS-2025 BOBW learner mapped to histogram route indices."""

    def __init__(
        self,
        case: HistogramGameCase,
        *,
        horizon: int,
        regularizer: str,
        eta_scale: float,
        beta: float,
        rng: np.random.Generator,
        solver_tolerance: float,
        solver_max_iterations: int,
    ) -> None:
        super().__init__(case)
        self.learner = ItoDAGPathFTRL(
            self.layered.layered,
            horizon=horizon,
            regularizer=regularizer,
            eta_scale=eta_scale,
            beta=beta,
            rng=rng,
            solver_tolerance=solver_tolerance,
            solver_max_iterations=solver_max_iterations,
        )

    def sample(self) -> int:
        layered_path = self.learner.sample()
        physical_path = self.layered.to_original_path(layered_path)
        route = self.route_index(physical_path)
        self.counts[route] += 1.0
        return route

    def update(self, reward: float) -> None:
        self.learner.update(1.0 - _bounded_reward(reward))

    def update_counterfactual(
        self,
        factual_reward: float,
        corrected_reward: float,
    ) -> None:
        self.learner.add_counterfactual_losses(
            1.0 - _bounded_reward(factual_reward),
            1.0 - _bounded_reward(corrected_reward),
        )

    def _policy_for_flow(self, flow: np.ndarray) -> np.ndarray:
        probabilities = np.asarray([
            self.learner.geometry.path_probability(path, flow)
            for path in self._layered_paths
        ])
        return self._normalize_policy(probabilities)

    def policy(self, *, turn: int | None = None) -> np.ndarray:
        _validate_optional_turn(turn)
        return self._policy_for_flow(self.learner.flow)

    def counterfactual_policy(self, *, turn: int | None = None) -> np.ndarray:
        _validate_optional_turn(turn)
        return self._policy_for_flow(self.learner.counterfactual_flow)

    @property
    def last_schedule(self) -> Any:
        return self.learner.last_schedule

    def full_diagnostics(self) -> dict[str, Any]:
        return {**self.diagnostics(), **self.learner.diagnostics()}


def _bounded_reward(value: float) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("public service reward must be finite and in [0, 1]")
    return result


def _validate_optional_turn(turn: int | None) -> None:
    if turn is not None and int(turn) < 1:
        raise ValueError("turn must be >= 1")
