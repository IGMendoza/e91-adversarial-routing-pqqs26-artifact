"""Distributional distances to zero-sum equilibrium strategy sets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import linprog


Player = Literal["alice", "eve"]


@dataclass
class EquilibriumSetProjector:
    """Compute total variation to the nearest equilibrium marginal.

    Distance to the full equilibrium polytope remains well-defined when a game
    has redundant routes or attacks and therefore multiple LP solutions.
    """

    game: np.ndarray
    value: float
    numerical_tolerance: float = 1e-9
    _cache: dict[tuple[Player, tuple[float, ...]], float] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        matrix = np.asarray(self.game, dtype=float)
        if matrix.ndim != 2 or min(matrix.shape) < 1:
            raise ValueError("game must be a nonempty matrix")
        if not np.all(np.isfinite(matrix)) or not np.isfinite(self.value):
            raise ValueError("game and value must be finite")
        if self.numerical_tolerance <= 0.0:
            raise ValueError("numerical_tolerance must be positive")
        self.game = matrix

    def total_variation(
        self,
        player: Player,
        strategy: Sequence[float],
    ) -> float:
        """Return half the minimum L1 distance to the player's equilibrium set."""

        if player not in {"alice", "eve"}:
            raise ValueError(f"unsupported player {player!r}")
        expected = self.game.shape[0 if player == "alice" else 1]
        vector = np.asarray(strategy, dtype=float)
        if vector.shape != (expected,) or not np.all(np.isfinite(vector)):
            raise ValueError("strategy has the wrong shape or non-finite values")
        if np.any(vector < -1e-12) or not np.isclose(
            vector.sum(),
            1.0,
            atol=1e-9,
        ):
            raise ValueError("strategy is not a probability distribution")
        vector = np.maximum(vector, 0.0)
        vector /= vector.sum()
        key = (player, tuple(float(value) for value in vector))
        if key not in self._cache:
            self._cache[key] = self._solve(player, vector)
        return self._cache[key]

    def _solve(self, player: Player, strategy: np.ndarray) -> float:
        action_count = strategy.size
        objective = np.concatenate([
            np.zeros(action_count, dtype=float),
            np.ones(action_count, dtype=float),
        ])
        identity = np.eye(action_count, dtype=float)
        absolute_constraints = np.block([
            [identity, -identity],
            [-identity, -identity],
        ])
        absolute_bounds = np.concatenate([strategy, -strategy])
        if player == "alice":
            equilibrium_matrix = -self.game.T
            equilibrium_bound = np.full(
                self.game.shape[1],
                -(self.value - self.numerical_tolerance),
            )
        else:
            equilibrium_matrix = self.game
            equilibrium_bound = np.full(
                self.game.shape[0],
                self.value + self.numerical_tolerance,
            )
        equilibrium_constraints = np.hstack([
            equilibrium_matrix,
            np.zeros(
                (equilibrium_matrix.shape[0], action_count),
                dtype=float,
            ),
        ])
        result = linprog(
            objective,
            A_ub=np.vstack([absolute_constraints, equilibrium_constraints]),
            b_ub=np.concatenate([absolute_bounds, equilibrium_bound]),
            A_eq=np.asarray([
                np.concatenate([
                    np.ones(action_count, dtype=float),
                    np.zeros(action_count, dtype=float),
                ])
            ]),
            b_eq=np.asarray([1.0]),
            bounds=[(0.0, None)] * (2 * action_count),
            method="highs",
        )
        if not result.success:
            raise RuntimeError(
                "equilibrium-set distance linear program failed: "
                f"{result.message}"
            )
        distance = 0.5 * float(result.fun)
        if distance < 10.0 * self.numerical_tolerance:
            return 0.0
        return distance
