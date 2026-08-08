"""Bandit no-regret co-learning for the asymmetric partial-information game.

Each round Alice picks a route and Eve picks an attack *simultaneously*, and each
player observes **only the realised outcome of their own move** -- never the
opponent's action:

* Alice sees her key-rate outcome (accepted / QBER-abort / CHSH-abort /
  delivery-failure), but not *where* on the route Eve struck.
* Eve sees the authenticated public protocol decision and heralded service
  yield, plus records produced by her own instrument, but not Alice's route or
  the simulator's target-overlap bit.

The flat baseline is Exp3 (exponential weights with importance-weighted
estimates), not the full-information RM+ solver.  LinearExp2 uses the same
scalar bandit boundary while sharing estimates through a declared finite-action
feature span for either player.  The full-information LP/RM+ solver
(:mod:`.zero_sum`) is used only as the oracle to check convergence.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from .zero_sum import exploitability

EXP3_SCHEDULE_CONSTANT = "constant"
EXP3_SCHEDULE_ANYTIME = "anytime"
EXP3_SCHEDULE_MODES = frozenset({
    EXP3_SCHEDULE_CONSTANT,
    EXP3_SCHEDULE_ANYTIME,
})
LINEAR_EXPLORATION_UNIFORM = "uniform"
LINEAR_EXPLORATION_G_OPTIMAL = "g_optimal"
LINEAR_EXPLORATION_MODES = frozenset({
    LINEAR_EXPLORATION_UNIFORM,
    LINEAR_EXPLORATION_G_OPTIMAL,
})
_LINEAR_EXPLORATION_DESIGN_CACHE_MAX_ENTRIES = 32
_LINEAR_EXPLORATION_DESIGN_CACHE: OrderedDict[
    tuple[str, tuple[int, int], str, float, int],
    "LinearExplorationDesign",
] = OrderedDict()


@dataclass(frozen=True)
class Exp3SchedulePoint:
    turn: int
    eta: float
    gamma: float


@dataclass(frozen=True)
class LinearExplorationDesign:
    """Fixed sampling design used to identify linear action rewards.

    The maximum prediction variance is

    ``max_a x_a.T @ M(q)^-1 @ x_a`` with
    ``M(q) = sum_a q_a x_a x_a.T``.

    Kiefer--Wolfowitz equivalence gives the optimum value ``dimension`` for a
    finite full-rank action set.  The ratio reported here therefore measures
    how much worst-case estimator variance remains relative to that optimum.
    This is exploration geometry, not a prior belief about action quality.
    """

    mode: str
    probabilities: np.ndarray
    dimension: int
    maximum_prediction_variance: float
    optimality_ratio: float
    iterations: int
    converged: bool
    tolerance: float

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "probabilities": self.probabilities.tolist(),
            "dimension": int(self.dimension),
            "maximum_prediction_variance": float(
                self.maximum_prediction_variance
            ),
            "theoretical_optimum": float(self.dimension),
            "optimality_ratio": float(self.optimality_ratio),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
            "tolerance": float(self.tolerance),
            "semantics": (
                "topology-feature identifiability design; probabilities do "
                "not encode predicted reward, route quality, or physics"
            ),
        }


def linear_exploration_design(
    features: np.ndarray,
    *,
    mode: str = LINEAR_EXPLORATION_UNIFORM,
    tolerance: float = 1e-6,
    max_iterations: int = 10_000,
) -> LinearExplorationDesign:
    """Construct a uniform or approximately G-optimal linear-bandit design.

    ``g_optimal`` uses the vertex-direction/Frank--Wolfe update with exact
    line search for log-determinant design.  Starting from the uniform design
    keeps every iterate full rank.  At convergence, the equivalence theorem
    certifies that the worst action-prediction variance is within ``tolerance``
    of the information-theoretic optimum ``dimension``.
    """

    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("features must be a non-empty 2D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite")
    if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
        raise ValueError("feature columns must have full rank")
    if mode not in LINEAR_EXPLORATION_MODES:
        raise ValueError(f"unsupported linear exploration mode {mode!r}")
    if tolerance <= 0.0:
        raise ValueError("linear exploration tolerance must be > 0")
    if max_iterations < 1:
        raise ValueError("linear exploration max_iterations must be >= 1")

    feature_digest = hashlib.blake2b(
        np.ascontiguousarray(matrix).view(np.uint8), digest_size=16
    ).hexdigest()
    cache_key = (
        feature_digest,
        (int(matrix.shape[0]), int(matrix.shape[1])),
        mode,
        float(tolerance),
        int(max_iterations),
    )
    cached = _LINEAR_EXPLORATION_DESIGN_CACHE.get(cache_key)
    if cached is not None:
        _LINEAR_EXPLORATION_DESIGN_CACHE.move_to_end(cache_key)
        return LinearExplorationDesign(
            mode=cached.mode,
            probabilities=cached.probabilities.copy(),
            dimension=cached.dimension,
            maximum_prediction_variance=cached.maximum_prediction_variance,
            optimality_ratio=cached.optimality_ratio,
            iterations=cached.iterations,
            converged=cached.converged,
            tolerance=cached.tolerance,
        )

    action_count, dimension = matrix.shape
    probabilities = np.full(action_count, 1.0 / action_count, dtype=float)
    iterations = 0
    converged = mode == LINEAR_EXPLORATION_UNIFORM
    maximum_variance = _maximum_linear_prediction_variance(
        matrix, probabilities
    )
    if mode == LINEAR_EXPLORATION_G_OPTIMAL:
        target = dimension * (1.0 + tolerance)
        for iteration in range(max_iterations + 1):
            leverage = _linear_prediction_variances(matrix, probabilities)
            selected = int(np.argmax(leverage))
            maximum_variance = float(leverage[selected])
            iterations = iteration
            if maximum_variance <= target:
                converged = True
                break
            # Exact line search for log det((1-alpha) M + alpha xx^T).
            # Since maximum_variance > dimension, the denominator is positive.
            alpha = (
                (maximum_variance - dimension)
                / (dimension * (maximum_variance - 1.0))
            )
            probabilities *= 1.0 - alpha
            probabilities[selected] += alpha
        else:  # pragma: no cover - the loop bound above always terminates
            iterations = max_iterations
        maximum_variance = _maximum_linear_prediction_variance(
            matrix, probabilities
        )
        converged = maximum_variance <= target

    probabilities /= probabilities.sum()
    result = LinearExplorationDesign(
        mode=mode,
        probabilities=probabilities,
        dimension=int(dimension),
        maximum_prediction_variance=float(maximum_variance),
        optimality_ratio=float(maximum_variance / dimension),
        iterations=int(iterations),
        converged=bool(converged),
        tolerance=float(tolerance),
    )
    _LINEAR_EXPLORATION_DESIGN_CACHE[cache_key] = result
    _LINEAR_EXPLORATION_DESIGN_CACHE.move_to_end(cache_key)
    while len(_LINEAR_EXPLORATION_DESIGN_CACHE) > (
        _LINEAR_EXPLORATION_DESIGN_CACHE_MAX_ENTRIES
    ):
        _LINEAR_EXPLORATION_DESIGN_CACHE.popitem(last=False)
    return LinearExplorationDesign(
        mode=result.mode,
        probabilities=result.probabilities.copy(),
        dimension=result.dimension,
        maximum_prediction_variance=result.maximum_prediction_variance,
        optimality_ratio=result.optimality_ratio,
        iterations=result.iterations,
        converged=result.converged,
        tolerance=result.tolerance,
    )


def _linear_prediction_variances(
    features: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    moment = features.T @ (probabilities[:, None] * features)
    try:
        solved = np.linalg.solve(moment, features.T)
    except np.linalg.LinAlgError as exc:
        raise FloatingPointError(
            "linear exploration design has singular sampling covariance"
        ) from exc
    values = np.einsum("ij,ji->i", features, solved)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("invalid linear prediction variances")
    return values


def _maximum_linear_prediction_variance(
    features: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    return float(np.max(_linear_prediction_variances(features, probabilities)))


class Exp3:
    """Exp3 adversarial-bandit learner maximizing rewards in [0, 1].

    Constant mode preserves the historical implementation exactly: action
    selection uses the fixed exploration mass ``gamma`` and the log-weight
    update uses ``eta = gamma / K``. Anytime mode keeps the same reward
    maximization convention but uses the supplied player-specific schedule.
    """

    def __init__(self, num_actions: int, *, gamma: float = 0.07,
                 rng: np.random.Generator | None = None,
                 schedule_mode: str = EXP3_SCHEDULE_CONSTANT,
                 eta_c: float = 1.0,
                 t0: float = 10000.0,
                 gamma_max: float = 0.20,
                 prior: np.ndarray | None = None):
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if schedule_mode not in EXP3_SCHEDULE_MODES:
            raise ValueError(f"unsupported Exp3 schedule mode {schedule_mode!r}")
        if eta_c <= 0:
            raise ValueError("eta_c must be > 0")
        if t0 < 0:
            raise ValueError("t0 must be >= 0")
        if not 0 < gamma_max <= 1:
            raise ValueError("gamma_max must be in (0, 1]")
        self.k = num_actions
        self.gamma = gamma
        self.schedule_mode = schedule_mode
        self.eta_c = float(eta_c)
        self.t0 = float(t0)
        self.gamma_max = float(gamma_max)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.prior = _probability_prior(prior, num_actions)
        self._minimum_prior = float(self.prior.min())
        self._logw = np.zeros(num_actions)     # log-weights (overflow-safe)
        self.counts = np.zeros(num_actions)     # empirical action frequency
        self._last_p: np.ndarray | None = None
        self._last_a: int | None = None
        self._last_schedule: Exp3SchedulePoint | None = None

    def schedule_at_turn(self, turn: int) -> Exp3SchedulePoint:
        if turn < 1:
            raise ValueError("turn must be >= 1")
        if self.schedule_mode == EXP3_SCHEDULE_CONSTANT:
            return Exp3SchedulePoint(
                turn=int(turn),
                # For the historical uniform prior this is exactly gamma / K.
                # With a structured prior it keeps eta / min(p_explore) on the
                # same bounded scale under importance weighting.
                eta=float(self.gamma * self._minimum_prior),
                gamma=float(self.gamma),
            )
        effective_actions = 1.0 / self._minimum_prior
        prior_complexity = math.log(effective_actions)
        eta = self.eta_c * math.sqrt(
            prior_complexity
            / (effective_actions * (float(turn) + self.t0))
        )
        gamma = min(self.gamma_max, effective_actions * eta)
        return Exp3SchedulePoint(turn=int(turn), eta=float(eta), gamma=float(gamma))

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
        """Policy after an auditable counterfactual log-weight correction.

        The learner itself is not mutated.  This is useful for fixed-behaviour
        sensitivity analyses: replaying alternative rewards on the actions and
        selection probabilities that were actually observed produces an exact
        additive correction to Exp3's log weights.
        """

        correction = np.asarray(offset, dtype=float)
        if correction.ndim != 1 or correction.size != self.k:
            raise ValueError("offset must be a one-dimensional num_actions vector")
        if not np.all(np.isfinite(correction)):
            raise ValueError("offset entries must be finite")
        point = self.schedule_at_turn(
            int(self.counts.sum()) + 1 if turn is None else int(turn)
        )
        # ``_logw`` stores only learned log-multipliers.  Keeping the prior
        # outside it preserves the legacy zero initialization while making
        # the initial exploitation distribution and exploration floor obey a
        # public, structured action prior.
        adjusted = self._logw + correction
        w = self.prior * np.exp(adjusted - adjusted.max())
        w /= w.sum()
        p = (1.0 - point.gamma) * w + point.gamma * self.prior
        total = float(p.sum())
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("invalid Exp3 policy normalization")
        p = p / total
        floor = point.gamma * self.prior
        if np.any(p < floor - 1e-15):
            raise FloatingPointError("Exp3 policy fell below exploration floor")
        return p

    def sample(self) -> int:
        turn = int(self.counts.sum()) + 1
        point = self.schedule_at_turn(turn)
        p = self.policy(turn=turn)
        a = int(self.rng.choice(self.k, p=p))
        self._register_selection(a, p, point)
        return a

    def select_observed(self, action: int) -> float:
        """Condition Exp3 on an action drawn from its current behavior policy."""

        index = int(action)
        if not 0 <= index < self.k:
            raise IndexError("observed action index is outside Exp3")
        turn = int(self.counts.sum()) + 1
        point = self.schedule_at_turn(turn)
        policy = self.policy(turn=turn)
        self._register_selection(index, policy, point)
        return float(policy[index])

    def _register_selection(
        self,
        action: int,
        policy: np.ndarray,
        point: Exp3SchedulePoint,
    ) -> None:
        a = int(action)
        p = np.asarray(policy, dtype=float)
        self._last_p, self._last_a = p, a
        self._last_schedule = point
        self.counts[a] += 1

    def update(self, reward: float) -> None:
        """Update from the reward in [0, 1] of the last sampled action only."""
        if self._last_a is None:
            raise RuntimeError("update() called before sample()")
        if self._last_schedule is None or self._last_p is None:
            raise RuntimeError("update() called before sample()")
        a = self._last_a
        x_hat = float(reward) / self._last_p[a]      # importance-weighted estimate
        self._logw[a] += self._last_schedule.eta * x_hat

    def counterfactual_log_weight_correction(
        self,
        reward_delta: float,
    ) -> np.ndarray:
        """Exact fixed-behaviour score correction for the last Exp3 sample."""

        if self._last_a is None or self._last_p is None:
            raise RuntimeError("correction requested before sample()")
        if self._last_schedule is None:
            raise RuntimeError("correction requested before sample()")
        correction = np.zeros(self.k, dtype=float)
        correction[self._last_a] = (
            self._last_schedule.eta
            * float(reward_delta)
            / self._last_p[self._last_a]
        )
        return correction

    def empirical_strategy(self) -> np.ndarray:
        total = self.counts.sum()
        if total > 0:
            return self.counts / total
        return np.full(self.k, 1.0 / self.k)

    @property
    def last_schedule(self) -> Exp3SchedulePoint | None:
        return self._last_schedule

    @property
    def last_action_probability(self) -> float | None:
        """Behaviour probability for the last sampled action."""

        if self._last_p is None or self._last_a is None:
            return None
        return float(self._last_p[self._last_a])


class LinearExp2:
    """Adversarial linear bandit over a fixed finite action-feature embedding.

    The learner observes only the scalar reward of its sampled route.  It uses
    the sampling covariance to construct the standard importance-weighted
    linear reward estimate, then shares that estimate across actions through
    their declared pre-play features.  No opponent action, target identity,
    or unplayed payoff cell is an input.

    ``features`` must have one row per action and full column rank.  Uniform
    exploration preserves the original implementation.  G-optimal exploration
    is the more general structured form: it minimizes the worst prediction
    variance of the scalar-feedback estimator without using reward or physics
    information.  The histogram adapter supplies the orthonormal
    left-singular-vector basis supplied by the relevant player adapter.
    """

    def __init__(
        self,
        features: np.ndarray,
        *,
        gamma: float = 0.07,
        rng: np.random.Generator | None = None,
        schedule_mode: str = EXP3_SCHEDULE_ANYTIME,
        eta_c: float = 1.0,
        t0: float = 10000.0,
        gamma_max: float = 0.20,
        exploration_mode: str = LINEAR_EXPLORATION_UNIFORM,
        exploration_tolerance: float = 1e-6,
        exploration_max_iterations: int = 10_000,
    ):
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
            raise ValueError("features must be a non-empty 2D matrix")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("features must be finite")
        if np.linalg.matrix_rank(matrix) != matrix.shape[1]:
            raise ValueError("feature columns must have full rank")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if schedule_mode not in EXP3_SCHEDULE_MODES:
            raise ValueError(f"unsupported Exp2 schedule mode {schedule_mode!r}")
        if eta_c <= 0:
            raise ValueError("eta_c must be > 0")
        if t0 < 0:
            raise ValueError("t0 must be >= 0")
        if not 0 < gamma_max <= 1:
            raise ValueError("gamma_max must be in (0, 1]")
        self.features = matrix.copy()
        self.k, self.dimension = matrix.shape
        self.gamma = float(gamma)
        self.schedule_mode = schedule_mode
        self.eta_c = float(eta_c)
        self.t0 = float(t0)
        self.gamma_max = float(gamma_max)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.exploration_design = linear_exploration_design(
            matrix,
            mode=exploration_mode,
            tolerance=exploration_tolerance,
            max_iterations=exploration_max_iterations,
        )
        self.prior = self.exploration_design.probabilities.copy()
        self._logw = np.zeros(self.k, dtype=float)
        self.counts = np.zeros(self.k, dtype=float)
        self._last_p: np.ndarray | None = None
        self._last_a: int | None = None
        self._last_schedule: Exp3SchedulePoint | None = None
        self._last_covariance_inverse: np.ndarray | None = None

    def schedule_at_turn(self, turn: int) -> Exp3SchedulePoint:
        if turn < 1:
            raise ValueError("turn must be >= 1")
        if self.schedule_mode == EXP3_SCHEDULE_CONSTANT:
            return Exp3SchedulePoint(
                turn=int(turn),
                eta=float(self.gamma / self.dimension),
                gamma=float(self.gamma),
            )
        eta = self.eta_c * math.sqrt(
            math.log(max(2, self.k))
            / (self.dimension * (float(turn) + self.t0))
        )
        gamma = min(self.gamma_max, self.dimension * eta)
        return Exp3SchedulePoint(turn=int(turn), eta=float(eta), gamma=float(gamma))

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
        correction = np.asarray(offset, dtype=float)
        if correction.ndim != 1 or correction.size != self.k:
            raise ValueError("offset must be a one-dimensional num_actions vector")
        if not np.all(np.isfinite(correction)):
            raise ValueError("offset entries must be finite")
        point = self.schedule_at_turn(
            int(self.counts.sum()) + 1 if turn is None else int(turn)
        )
        adjusted = self._logw + correction
        exploitation = np.exp(adjusted - adjusted.max())
        exploitation /= exploitation.sum()
        policy = (
            (1.0 - point.gamma) * exploitation
            + point.gamma * self.prior
        )
        policy /= policy.sum()
        floor = point.gamma * self.prior
        if np.any(policy < floor - 1e-15):
            raise FloatingPointError("LinearExp2 policy fell below exploration floor")
        return policy

    def sample(self) -> int:
        turn = int(self.counts.sum()) + 1
        point = self.schedule_at_turn(turn)
        policy = self.policy(turn=turn)
        action = int(self.rng.choice(self.k, p=policy))
        self._register_selection(action, policy, point)
        return action

    def select_observed(self, action: int) -> float:
        """Condition LinearExp2 on an action drawn from its current policy."""

        index = int(action)
        if not 0 <= index < self.k:
            raise IndexError("observed action index is outside LinearExp2")
        turn = int(self.counts.sum()) + 1
        point = self.schedule_at_turn(turn)
        policy = self.policy(turn=turn)
        self._register_selection(index, policy, point)
        return float(policy[index])

    def _register_selection(
        self,
        action: int,
        policy: np.ndarray,
        point: Exp3SchedulePoint,
    ) -> None:
        covariance = self.features.T @ (policy[:, None] * self.features)
        covariance_inverse = np.linalg.inv(covariance)
        if not np.all(np.isfinite(covariance_inverse)):
            raise FloatingPointError("invalid LinearExp2 sampling covariance")
        self._last_p = policy
        self._last_a = action
        self._last_schedule = point
        self._last_covariance_inverse = covariance_inverse
        self.counts[action] += 1

    def _score_increment(self, reward: float) -> np.ndarray:
        if self._last_a is None or self._last_covariance_inverse is None:
            raise RuntimeError("update() called before sample()")
        if self._last_schedule is None:
            raise RuntimeError("update() called before sample()")
        selected_feature = self.features[self._last_a]
        coefficient_estimate = (
            float(reward) * self._last_covariance_inverse @ selected_feature
        )
        return self._last_schedule.eta * self.features @ coefficient_estimate

    def update(self, reward: float) -> None:
        """Update all route scores from one scalar selected-route reward."""

        self._logw += self._score_increment(reward)

    def counterfactual_log_weight_correction(
        self,
        reward_delta: float,
    ) -> np.ndarray:
        """Exact fixed-behaviour correction to this learner's route scores."""

        return self._score_increment(reward_delta)

    def empirical_strategy(self) -> np.ndarray:
        total = self.counts.sum()
        if total > 0:
            return self.counts / total
        return np.full(self.k, 1.0 / self.k)

    @property
    def last_schedule(self) -> Exp3SchedulePoint | None:
        return self._last_schedule

    @property
    def last_action_probability(self) -> float | None:
        if self._last_p is None or self._last_a is None:
            return None
        return float(self._last_p[self._last_a])


def _probability_prior(prior: np.ndarray | None, size: int) -> np.ndarray:
    if prior is None:
        return np.full(size, 1.0 / size, dtype=float)
    result = np.asarray(prior, dtype=float)
    if result.ndim != 1 or result.size != size:
        raise ValueError("prior must be a one-dimensional num_actions vector")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("prior entries must be finite and strictly positive")
    total = float(result.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("prior must have positive finite mass")
    return result / total


@dataclass(frozen=True)
class BanditSelfPlayResult:
    value: float
    alice_strategy: np.ndarray     # Alice's empirical route distribution
    eve_strategy: np.ndarray       # Eve's empirical attack distribution
    iterations: int


def solve_zero_sum_bandit_selfplay(
        payoff: np.ndarray, *, iterations: int = 40000, gamma: float = 0.07,
        seed: int = 0) -> BanditSelfPlayResult:
    """Co-learn the equilibrium under bandit feedback (Exp3 self-play).

    ``payoff[a, e]`` is Alice's realised key rate for (route a, attack e). Alice's
    bandit reward is that rate and Eve's is its public denial complement
    ``1 - rate``. Neither learner observes the other's action.
    """

    P = np.asarray(payoff, dtype=float)
    if P.ndim != 2:
        raise ValueError("payoff must be a 2D matrix")
    na, ne = P.shape
    rng = np.random.default_rng(seed)
    alice = Exp3(na, gamma=gamma, rng=rng)
    eve = Exp3(ne, gamma=gamma, rng=rng)
    for _ in range(iterations):
        a = alice.sample()
        e = eve.sample()
        rate = P[a, e]
        alice.update(rate)              # Alice: only her key-rate outcome
        eve.update(1.0 - rate)          # Eve: only "hit a real signal" -> denial
    x = alice.empirical_strategy()
    y = eve.empirical_strategy()
    return BanditSelfPlayResult(
        value=float(x @ P @ y),
        alice_strategy=x,
        eve_strategy=y,
        iterations=iterations,
    )


@dataclass(frozen=True)
class ConvergenceTrajectory:
    """Exploitability (Nash gap) of the running average vs iteration.

    The exploitability decaying to 0 is the no-regret convergence proof: when the
    average profile's Nash gap is ``eps`` it is an ``eps``-equilibrium, so
    ``exploitability -> 0`` certifies convergence to the minimax/Nash value (which
    ``value`` approaches). Provided for the learning-curve plots.
    """

    iters: np.ndarray            # checkpoint iteration counts
    value: np.ndarray            # running-average game value at each checkpoint
    exploitability: np.ndarray   # Nash gap of the running average at each checkpoint
    alice_strategy: np.ndarray
    eve_strategy: np.ndarray


def bandit_selfplay_trajectory(
        payoff: np.ndarray, *, iterations: int = 40000, record_every: int = 500,
        gamma: float = 0.07, seed: int = 0) -> ConvergenceTrajectory:
    """Exp3 self-play recording the exploitability trajectory (convergence proof)."""

    P = np.asarray(payoff, dtype=float)
    na, ne = P.shape
    rng = np.random.default_rng(seed)
    alice = Exp3(na, gamma=gamma, rng=rng)
    eve = Exp3(ne, gamma=gamma, rng=rng)
    iters, values, gaps = [], [], []
    for t in range(iterations):
        a = alice.sample()
        e = eve.sample()
        rate = P[a, e]
        alice.update(rate)
        eve.update(1.0 - rate)
        if (t + 1) % record_every == 0:
            x = alice.empirical_strategy()
            y = eve.empirical_strategy()
            iters.append(t + 1)
            values.append(float(x @ P @ y))
            gaps.append(exploitability(P, x, y))
    return ConvergenceTrajectory(
        iters=np.array(iters), value=np.array(values),
        exploitability=np.array(gaps),
        alice_strategy=alice.empirical_strategy(),
        eve_strategy=eve.empirical_strategy())
