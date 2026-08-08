"""Zero-sum routing/interdiction game: RM+ self-play + minimax LP oracle.

The stage game is a matrix ``P`` of shape (num_routes, num_attacks) where
``P[a, e]`` is Alice's payoff (probability the key succeeds) when Alice plays
route ``a`` and Eve plays attack ``e``. It is (constant-sum) zero-sum: Eve's
payoff is the denial ``1 - P[a, e]``.

Design principle (co-learning, not baked in): the players learn *only* from their
own realised regrets via Regret Matching+ self-play; the equilibrium is
discovered, never supplied. :func:`solve_zero_sum_lp` computes the exact minimax
value and optimal mixed strategies and is used strictly as a *validation oracle*
(convergence / exploitability), never fed to the learners.

Asymmetry: Alice is the packing player (spread rate over independent routes),
Eve is the covering player (a small hitting set of nodes/edges). Same no-regret
machinery, dual equilibrium structure. The N-disjoint-routes matrix ``J - I`` is
the self-dual special case (value (N-1)/N, uniform play); a shared-bottleneck
column of zeros is the other extreme (value 0, Eve pure).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


class RegretMatchingPlus:
    """A single-player Regret Matching+ minimizer over a fixed action set.

    The player maximises its own utility. Each round it plays
    :meth:`current_strategy`, then :meth:`observe` the per-action utility vector
    (as if it had committed to each action against the opponent's play) to update
    cumulative clamped regret and the linearly-weighted average strategy.
    """

    def __init__(self, num_actions: int):
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        self.num_actions = num_actions
        self._regret = np.zeros(num_actions)          # RM+ cumulative clamped regret
        self._strategy_sum = np.zeros(num_actions)     # linearly-weighted average
        self._weight_sum = 0.0
        self._iter = 0

    def current_strategy(self) -> np.ndarray:
        pos = np.maximum(self._regret, 0.0)
        total = pos.sum()
        if total > 0:
            return pos / total
        return np.full(self.num_actions, 1.0 / self.num_actions)

    def observe(self, action_utilities: np.ndarray) -> None:
        u = np.asarray(action_utilities, dtype=float)
        if u.shape != (self.num_actions,):
            raise ValueError("action_utilities has wrong shape")
        x = self.current_strategy()
        expected = float(x @ u)
        instantaneous_regret = u - expected
        # RM+: clamp cumulative regret at zero after each update.
        self._regret = np.maximum(self._regret + instantaneous_regret, 0.0)
        # CFR+-style linear averaging (weight by iteration index).
        self._iter += 1
        self._strategy_sum += self._iter * x
        self._weight_sum += self._iter

    def average_strategy(self) -> np.ndarray:
        if self._weight_sum > 0:
            return self._strategy_sum / self._weight_sum
        return np.full(self.num_actions, 1.0 / self.num_actions)


@dataclass(frozen=True)
class SelfPlayResult:
    value: float                    # Alice's key-rate at the average profile
    alice_strategy: np.ndarray      # average route distribution
    eve_strategy: np.ndarray        # average attack distribution
    exploitability: float           # Nash gap of the average profile
    iterations: int


def solve_zero_sum_selfplay(payoff: np.ndarray, *, iterations: int = 5000
                            ) -> SelfPlayResult:
    """Discover the equilibrium by RM+ self-play (no baked-in solution).

    Alice maximises ``P``; Eve maximises denial ``1 - P``. Returns the
    time-averaged mixed strategies, which converge to the minimax equilibrium.
    """

    P = np.asarray(payoff, dtype=float)
    if P.ndim != 2:
        raise ValueError("payoff must be a 2D matrix")
    na, ne = P.shape
    denial = 1.0 - P
    alice = RegretMatchingPlus(na)
    eve = RegretMatchingPlus(ne)
    for _ in range(iterations):
        x = alice.current_strategy()
        y = eve.current_strategy()
        alice.observe(P @ y)          # Alice's utility per route vs Eve's play
        eve.observe(denial.T @ x)     # Eve's denial per attack vs Alice's play
    xbar = alice.average_strategy()
    ybar = eve.average_strategy()
    value = float(xbar @ P @ ybar)
    return SelfPlayResult(
        value=value,
        alice_strategy=xbar,
        eve_strategy=ybar,
        exploitability=exploitability(P, xbar, ybar),
        iterations=iterations,
    )


@dataclass(frozen=True)
class GameCharacterization:
    """Equilibrium key-rate and the graded Eve-win verdict for one graph.

    ``value`` is Alice's key rate at the co-learned equilibrium; ``baseline`` is
    her un-attacked rate. Eve's win is graded (per the model): ``absolute`` when
    the rate vanishes (single point of failure -- a node on every route forces a
    CHSH-abort/denial on every attempt), ``partial`` when the rate is merely
    reduced (QBER or probabilistic CHSH-abort), ``none`` when Alice retains it.
    """

    value: float
    baseline_rate: float
    keyrate_retention: float     # value / baseline
    eve_win: str                 # "absolute" | "partial" | "none"
    exploitability: float


def characterize_equilibrium(payoff: np.ndarray, *, baseline_rate: float,
                             abs_tol: float = 1e-2) -> GameCharacterization:
    """Co-learn the equilibrium and grade Eve's win by Alice's retained key rate.

    ``baseline_rate`` is Alice's un-attacked key rate (the physics rate with Eve
    absent). The equilibrium is discovered by self-play; the verdict follows the
    model's win definition (rate vanished vs merely reduced).
    """

    if baseline_rate <= 0:
        raise ValueError("baseline_rate must be > 0")
    sp = solve_zero_sum_selfplay(payoff)
    value = sp.value
    retention = value / baseline_rate
    if value <= abs_tol:
        verdict = "absolute"
    elif value < baseline_rate - abs_tol:
        verdict = "partial"
    else:
        verdict = "none"
    return GameCharacterization(
        value=value,
        baseline_rate=baseline_rate,
        keyrate_retention=retention,
        eve_win=verdict,
        exploitability=sp.exploitability,
    )


def exploitability(payoff: np.ndarray, alice: np.ndarray, eve: np.ndarray) -> float:
    """Nash gap: best-response advantage available to either player (0 at Nash)."""

    P = np.asarray(payoff, dtype=float)
    x = np.asarray(alice, dtype=float)
    y = np.asarray(eve, dtype=float)
    alice_best_response = float(np.max(P @ y))       # Alice's best key-rate vs eve
    alice_worst_case = float(np.min(x @ P))          # Alice's key-rate vs best eve
    return alice_best_response - alice_worst_case


def solve_zero_sum_lp(payoff: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact minimax value + optimal mixed strategies (VALIDATION ORACLE ONLY).

    Returns ``(value, alice_strategy, eve_strategy)`` where ``value`` is Alice's
    maximin key-rate. Never supplied to the learners; used to check that
    self-play converged to the true equilibrium.
    """

    P = np.asarray(payoff, dtype=float)
    na, ne = P.shape

    # Alice maximin: max v s.t. for every Eve column e, x . P[:, e] >= v; sum x=1.
    # linprog minimises: minimise -v. Vars = [x_0..x_{na-1}, v].
    c = np.concatenate([np.zeros(na), [-1.0]])
    A_ub = np.column_stack([-P.T, np.ones(ne)])       # v - x.P[:,e] <= 0
    b_ub = np.zeros(ne)
    A_eq = np.concatenate([np.ones(na), [0.0]])[None, :]
    b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * na + [(None, None)]
    res_a = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    if not res_a.success:
        raise RuntimeError(f"Alice LP failed: {res_a.message}")
    x = res_a.x[:na]
    value = float(res_a.x[na])

    # Eve minimax: min w s.t. for every Alice row a, y . P[a, :] <= w; sum y=1.
    c2 = np.concatenate([np.zeros(ne), [1.0]])
    A_ub2 = np.column_stack([P, -np.ones(na)])        # y.P[a,:] - w <= 0
    b_ub2 = np.zeros(na)
    A_eq2 = np.concatenate([np.ones(ne), [0.0]])[None, :]
    bounds2 = [(0.0, 1.0)] * ne + [(None, None)]
    res_e = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq, bounds=bounds2)
    if not res_e.success:
        raise RuntimeError(f"Eve LP failed: {res_e.message}")
    y = res_e.x[:ne]
    return value, x, y
