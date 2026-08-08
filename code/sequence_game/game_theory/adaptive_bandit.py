"""Best-of-both-worlds independent-arm bandit controls.

These learners deliberately do *not* use route, edge, node, or attack-family
features.  They are stronger finite-arm controls between the historical Exp3
comparator and learners that share observations through topology structure.

The public experiment interface maximizes rewards in ``[0, 1]``.  Both source
papers minimize losses, so the only boundary conversion is ``loss = 1-reward``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


TSALLIS_INF_ESTIMATOR_IW = "iw"
TSALLIS_INF_ESTIMATOR_RV = "rv"
TSALLIS_INF_ESTIMATORS = frozenset({
    TSALLIS_INF_ESTIMATOR_IW,
    TSALLIS_INF_ESTIMATOR_RV,
})


@dataclass(frozen=True)
class TsallisINFSchedulePoint:
    """The anytime learning rate used for one Tsallis-INF decision."""

    turn: int
    eta: float
    gamma: None = None


class TsallisINF:
    """Tsallis-INF with symmetric alpha=1/2 regularization.

    This is Algorithm 1 and the implementation in Section 4.2 of Zimmert and
    Seldin, *Tsallis-INF: An Optimal Algorithm for Stochastic and Adversarial
    Bandits* (JMLR 2021, arXiv:1807.07623).  ``estimator="rv"`` uses the
    paper's reduced-variance estimator and ``eta_t = 4/sqrt(t)``; ``"iw"``
    uses ordinary importance weighting and ``eta_t = 2/sqrt(t)``.

    No horizon, gaps, action-quality prior, or explicit exploration mixture is
    used.  The class also maintains an offline fixed-behaviour replay for the
    histogram false-signal diagnostic.  That replay never changes sampling.
    """

    def __init__(
        self,
        num_actions: int,
        *,
        estimator: str = TSALLIS_INF_ESTIMATOR_RV,
        eta_scale: float = 1.0,
        rng: np.random.Generator | None = None,
        solver_tolerance: float = 1e-12,
        solver_max_iterations: int = 40,
    ) -> None:
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        if estimator not in TSALLIS_INF_ESTIMATORS:
            raise ValueError(f"unsupported Tsallis-INF estimator {estimator!r}")
        if not math.isfinite(eta_scale) or eta_scale <= 0.0:
            raise ValueError("eta_scale must be finite and > 0")
        if not math.isfinite(solver_tolerance) or solver_tolerance <= 0.0:
            raise ValueError("solver_tolerance must be finite and > 0")
        if solver_max_iterations < 1:
            raise ValueError("solver_max_iterations must be >= 1")

        self.k = int(num_actions)
        self.estimator = estimator
        self.eta_scale = float(eta_scale)
        self.rng = rng if rng is not None else np.random.default_rng()
        self.solver_tolerance = float(solver_tolerance)
        self.solver_max_iterations = int(solver_max_iterations)

        self.counts = np.zeros(self.k, dtype=float)
        self._estimated_losses = np.zeros(self.k, dtype=float)
        self._counterfactual_estimated_losses = np.zeros(self.k, dtype=float)
        self._last_policy: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_behavior_probability: float | None = None
        self._last_schedule: TsallisINFSchedulePoint | None = None
        self._pending_update = False
        self._counterfactual_updated = False
        self._counterfactual_factual_reward: float | None = None
        self._last_rv_baseline = np.zeros(self.k, dtype=float)
        self._last_solver_iterations = 0
        self._behavior_conditioned_selection_count = 0

    def schedule_at_turn(self, turn: int) -> TsallisINFSchedulePoint:
        if turn < 1:
            raise ValueError("turn must be >= 1")
        coefficient = (
            4.0 if self.estimator == TSALLIS_INF_ESTIMATOR_RV else 2.0
        )
        return TsallisINFSchedulePoint(
            turn=int(turn),
            eta=float(self.eta_scale * coefficient / math.sqrt(turn)),
        )

    def policy(self, *, turn: int | None = None) -> np.ndarray:
        selected_turn = (
            int(self.counts.sum()) + 1 if turn is None else int(turn)
        )
        point = self.schedule_at_turn(selected_turn)
        policy, _ = self._policy_from_losses(self._estimated_losses, point.eta)
        return policy

    def counterfactual_policy(self, *, turn: int | None = None) -> np.ndarray:
        """Policy from corrected rewards along the factual behavior path."""

        selected_turn = (
            int(self.counts.sum()) + 1 if turn is None else int(turn)
        )
        point = self.schedule_at_turn(selected_turn)
        policy, _ = self._policy_from_losses(
            self._counterfactual_estimated_losses,
            point.eta,
        )
        return policy

    def _policy_from_losses(
        self,
        estimated_losses: np.ndarray,
        eta: float,
    ) -> tuple[np.ndarray, int]:
        """Solve the scalar Tsallis normalization equation.

        With ``m = min_i L_i`` and ``y = m-x``, Section 4.2 gives

        ``w_i = (2 / (eta * (L_i-m+y)))**2``.

        The root lies in ``[2/eta, 2*sqrt(K)/eta]``.  Safeguarded Newton
        iterations preserve that bracket and avoid a dependency on a general
        convex solver in the 100k-turn loop.
        """

        losses = np.asarray(estimated_losses, dtype=float)
        if losses.shape != (self.k,) or not np.all(np.isfinite(losses)):
            raise FloatingPointError("invalid Tsallis-INF cumulative losses")
        if self.k == 1:
            return np.ones(1, dtype=float), 0

        shifted = losses - float(np.min(losses))
        scale = 2.0 / float(eta)
        lower = scale
        upper = scale * math.sqrt(self.k)
        y = 0.5 * (lower + upper)
        iterations = 0

        for iterations in range(1, self.solver_max_iterations + 1):
            denominator = shifted + y
            weights = np.square(scale / denominator)
            mass = float(np.sum(weights))
            error = mass - 1.0
            if abs(error) <= self.solver_tolerance:
                break
            if error > 0.0:
                lower = y
            else:
                upper = y
            derivative = -2.0 * float(
                np.sum(np.square(scale) / np.power(denominator, 3.0))
            )
            candidate = y - error / derivative
            if not lower < candidate < upper or not math.isfinite(candidate):
                candidate = 0.5 * (lower + upper)
            y = candidate

        denominator = shifted + y
        policy = np.square(scale / denominator)
        total = float(np.sum(policy))
        if not math.isfinite(total) or total <= 0.0:
            raise FloatingPointError("invalid Tsallis-INF policy normalization")
        policy /= total
        if not np.all(np.isfinite(policy)) or np.any(policy <= 0.0):
            raise FloatingPointError("Tsallis-INF policy left the simplex interior")
        return policy, iterations

    def sample(self) -> int:
        if self._pending_update:
            raise RuntimeError("sample() called before updating the prior sample")
        turn = int(self.counts.sum()) + 1
        point = self.schedule_at_turn(turn)
        policy, iterations = self._policy_from_losses(
            self._estimated_losses,
            point.eta,
        )
        action = int(self.rng.choice(self.k, p=policy))
        self._register_selection(
            action,
            policy,
            point,
            iterations,
            behavior_probability=float(policy[action]),
        )
        return action

    def sample_from_behavior(self, behavior_policy: np.ndarray) -> int:
        """Sample a controlled behavior policy with a valid loss correction.

        The native Tsallis policy still supplies the RV baseline and the next
        FTRL state. The selected loss correction is divided by the actual
        behavior probability, so its conditional expectation remains the
        full loss vector when an external controller mixes the native policy
        with a containment policy.

        This is an off-policy controller extension. It is not claimed to
        inherit the paper's regret theorem for native Tsallis-INF decisions.
        """

        if self._pending_update:
            raise RuntimeError(
                "sample_from_behavior() called before updating the prior "
                "sample"
            )
        behavior = _probability_vector(
            behavior_policy,
            expected_size=self.k,
            label="Tsallis-INF behavior policy",
        )
        turn = int(self.counts.sum()) + 1
        point = self.schedule_at_turn(turn)
        policy, iterations = self._policy_from_losses(
            self._estimated_losses,
            point.eta,
        )
        action = int(self.rng.choice(self.k, p=behavior))
        self._register_selection(
            action,
            policy,
            point,
            iterations,
            behavior_probability=float(behavior[action]),
        )
        self._behavior_conditioned_selection_count += 1
        return action

    def select_observed(self, action: int) -> float:
        """Condition the learner on an action drawn from its current policy."""

        if self._pending_update:
            raise RuntimeError(
                "select_observed() called before updating the prior sample"
            )
        index = int(action)
        if not 0 <= index < self.k:
            raise IndexError("observed action index is outside Tsallis-INF")
        turn = int(self.counts.sum()) + 1
        point = self.schedule_at_turn(turn)
        policy, iterations = self._policy_from_losses(
            self._estimated_losses,
            point.eta,
        )
        self._register_selection(
            index,
            policy,
            point,
            iterations,
            behavior_probability=float(policy[index]),
        )
        return float(policy[index])

    def _register_selection(
        self,
        action: int,
        policy: np.ndarray,
        point: TsallisINFSchedulePoint,
        iterations: int,
        *,
        behavior_probability: float,
    ) -> None:
        probability = float(behavior_probability)
        if (
            not math.isfinite(probability)
            or probability <= 0.0
            or probability > 1.0
        ):
            raise ValueError(
                "Tsallis-INF behavior probability must lie in (0, 1]"
            )
        self._last_policy = policy
        self._last_action = action
        self._last_behavior_probability = probability
        self._last_schedule = point
        self._last_solver_iterations = iterations
        self._last_rv_baseline = self._rv_baseline(policy, point.eta)
        self.counts[action] += 1.0
        self._pending_update = True
        self._counterfactual_updated = False
        self._counterfactual_factual_reward = None

    def _rv_baseline(self, policy: np.ndarray, eta: float) -> np.ndarray:
        if self.estimator == TSALLIS_INF_ESTIMATOR_IW:
            return np.zeros(self.k, dtype=float)
        return 0.5 * (np.asarray(policy) >= eta * eta).astype(float)

    def _loss_increment(self, reward: float) -> np.ndarray:
        if (
            not self._pending_update
            or self._last_action is None
            or self._last_policy is None
            or self._last_behavior_probability is None
            or self._last_schedule is None
        ):
            raise RuntimeError("update() called before sample()")
        value = _unit_reward(reward)
        loss = 1.0 - value
        action = self._last_action
        probability = float(self._last_behavior_probability)
        baseline = self._last_rv_baseline
        increment = baseline.copy()
        increment[action] += (loss - baseline[action]) / probability
        if not np.all(np.isfinite(increment)):
            raise FloatingPointError("invalid Tsallis-INF loss estimate")
        return increment

    def update_counterfactual(
        self,
        factual_reward: float,
        corrected_reward: float,
    ) -> None:
        """Replay one corrected reward using the factual selection probability."""

        if self._counterfactual_updated:
            raise RuntimeError("counterfactual already updated for this sample")
        factual = _unit_reward(factual_reward)
        self._counterfactual_estimated_losses += self._loss_increment(
            corrected_reward
        )
        self._counterfactual_updated = True
        self._counterfactual_factual_reward = factual

    def update(self, reward: float) -> None:
        factual = _unit_reward(reward)
        increment = self._loss_increment(factual)
        self._estimated_losses += increment
        if self._counterfactual_updated:
            assert self._counterfactual_factual_reward is not None
            if not math.isclose(
                factual,
                self._counterfactual_factual_reward,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "factual reward differs from counterfactual replay input"
                )
        else:
            self._counterfactual_estimated_losses += increment
        self._pending_update = False

    def empirical_strategy(self) -> np.ndarray:
        total = float(np.sum(self.counts))
        if total > 0.0:
            return self.counts / total
        return np.full(self.k, 1.0 / self.k, dtype=float)

    @property
    def last_schedule(self) -> TsallisINFSchedulePoint | None:
        return self._last_schedule

    @property
    def last_action_probability(self) -> float | None:
        if self._last_action is None or self._last_policy is None:
            return None
        return float(self._last_policy[self._last_action])

    @property
    def last_behavior_probability(self) -> float | None:
        return self._last_behavior_probability

    def diagnostics(self) -> dict[str, object]:
        return {
            "estimator": self.estimator,
            "eta_scale": self.eta_scale,
            "last_policy_solver_iterations": self._last_solver_iterations,
            "last_rv_baseline_arm_count": int(
                np.count_nonzero(self._last_rv_baseline)
            ),
            "explicit_exploration_mixture": False,
            "cross_action_information_sharing": False,
            "behavior_conditioned_selection_count": int(
                self._behavior_conditioned_selection_count
            ),
            "last_native_action_probability": (
                self.last_action_probability
            ),
            "last_behavior_action_probability": (
                self.last_behavior_probability
            ),
            "off_policy_importance_correction": (
                "selected loss correction uses the actual behavior-policy "
                "probability; the RV baseline uses the native Tsallis policy"
            ),
            "controlled_behavior_theorem_boundary": (
                "unbiased loss-vector estimate under the declared behavior "
                "mixture; native Tsallis-INF regret theorem not asserted"
            ),
        }


@dataclass(frozen=True)
class Exp3PlusPlusSchedulePoint:
    """Learning and arm-specific exploration state for one EXP3++ turn."""

    turn: int
    eta: float
    gamma: float
    epsilon_min: float
    epsilon_max: float
    maximum_gap_lcb: float
    forced_initialization: bool


class Exp3PlusPlus:
    """Improved EXP3++ with arm-specific LCB gap exploration.

    The playing rule is Algorithm 1 of Seldin and Slivkins (ICML 2014).  Gap
    estimation and ``xi_t(a)`` use Seldin and Lugosi's improved 2017
    parametrization (arXiv:1702.06103, Algorithms 1--2 and Theorem 3): each arm
    is observed once, empirical loss confidence intervals produce a lower
    confidence bound on its gap, and that gap controls an arm-specific
    exploration floor.

    The theorem defaults are ``confidence_alpha=3`` and
    ``exploration_beta=256``.  They are intentionally configurable so a later
    preregistered, payoff-blind calibration can measure finite-time behavior;
    changing them is labelled a sensitivity rather than a literature default.
    """

    def __init__(
        self,
        num_actions: int,
        *,
        confidence_alpha: float = 3.0,
        exploration_beta: float = 256.0,
        eta_scale: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if num_actions < 1:
            raise ValueError("num_actions must be >= 1")
        if not math.isfinite(confidence_alpha) or confidence_alpha < 3.0:
            raise ValueError("confidence_alpha must be finite and >= 3")
        if not math.isfinite(exploration_beta) or exploration_beta <= 0.0:
            raise ValueError("exploration_beta must be finite and > 0")
        if not math.isfinite(eta_scale) or eta_scale <= 0.0:
            raise ValueError("eta_scale must be finite and > 0")

        self.k = int(num_actions)
        self.confidence_alpha = float(confidence_alpha)
        self.exploration_beta = float(exploration_beta)
        self.eta_scale = float(eta_scale)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Randomizing the mandated one-pass initialization removes arm-label
        # ordering while preserving the paper's "play each arm once" rule.
        self._initialization_order = self.rng.permutation(self.k)
        self.counts = np.zeros(self.k, dtype=float)
        self._importance_weighted_losses = np.zeros(self.k, dtype=float)
        self._observed_losses = np.zeros(self.k, dtype=float)
        self._observation_counts = np.zeros(self.k, dtype=float)
        self._counterfactual_importance_weighted_losses = np.zeros(
            self.k, dtype=float
        )
        self._counterfactual_observed_losses = np.zeros(self.k, dtype=float)
        self._counterfactual_observation_counts = np.zeros(self.k, dtype=float)

        self._last_policy: np.ndarray | None = None
        self._last_action: int | None = None
        self._last_schedule: Exp3PlusPlusSchedulePoint | None = None
        self._last_gap_lcb = np.zeros(self.k, dtype=float)
        self._last_epsilon = np.zeros(self.k, dtype=float)
        self._pending_update = False
        self._counterfactual_updated = False
        self._counterfactual_factual_reward: float | None = None

    def _base_eta(self, turn: int) -> float:
        if turn < 1:
            raise ValueError("turn must be >= 1")
        if self.k == 1:
            return 0.0
        return 0.5 * math.sqrt(math.log(self.k) / (turn * self.k))

    def _gap_lcb(
        self,
        *,
        turn: int,
        observed_losses: np.ndarray,
        observation_counts: np.ndarray,
    ) -> np.ndarray:
        if np.any(observation_counts <= 0.0):
            return np.zeros(self.k, dtype=float)
        means = observed_losses / observation_counts
        confidence = np.sqrt(
            (
                self.confidence_alpha * math.log(turn)
                + math.log(self.k)
            )
            / (2.0 * observation_counts)
        )
        upper = np.minimum(1.0, means + confidence)
        lower = np.maximum(0.0, means - confidence)
        gaps = np.maximum(0.0, lower - float(np.min(upper)))
        return np.minimum(1.0, gaps)

    def _policy_details(
        self,
        *,
        turn: int,
        importance_weighted_losses: np.ndarray,
        observed_losses: np.ndarray,
        observation_counts: np.ndarray,
    ) -> tuple[np.ndarray, Exp3PlusPlusSchedulePoint, np.ndarray, np.ndarray]:
        if turn < 1:
            raise ValueError("turn must be >= 1")
        completed = int(np.sum(observation_counts))
        base_eta = self._base_eta(turn)
        eta = self.eta_scale * base_eta

        if completed < self.k:
            action = int(self._initialization_order[completed])
            policy = np.zeros(self.k, dtype=float)
            policy[action] = 1.0
            gaps = np.zeros(self.k, dtype=float)
            epsilon = np.zeros(self.k, dtype=float)
            point = Exp3PlusPlusSchedulePoint(
                turn=int(turn),
                eta=float(eta),
                gamma=0.0,
                epsilon_min=0.0,
                epsilon_max=0.0,
                maximum_gap_lcb=0.0,
                forced_initialization=True,
            )
            return policy, point, gaps, epsilon

        gaps = self._gap_lcb(
            turn=turn,
            observed_losses=observed_losses,
            observation_counts=observation_counts,
        )
        xi = np.full(self.k, np.inf, dtype=float)
        positive = gaps > 0.0
        if np.any(positive):
            xi[positive] = (
                self.exploration_beta
                * math.log(turn)
                / (turn * np.square(gaps[positive]))
            )
        epsilon = np.minimum(
            np.minimum(np.full(self.k, 1.0 / (2.0 * self.k)), base_eta),
            xi,
        )
        if np.any(epsilon < 0.0) or float(np.sum(epsilon)) > 0.5 + 1e-14:
            raise FloatingPointError("invalid EXP3++ exploration mixture")

        scores = -eta * np.asarray(importance_weighted_losses, dtype=float)
        exploitation = np.exp(scores - float(np.max(scores)))
        exploitation /= float(np.sum(exploitation))
        exploration_mass = float(np.sum(epsilon))
        policy = (1.0 - exploration_mass) * exploitation + epsilon
        policy /= float(np.sum(policy))
        if not np.all(np.isfinite(policy)) or np.any(policy <= 0.0):
            raise FloatingPointError("invalid EXP3++ policy")
        if np.any(policy < epsilon - 1e-15):
            raise FloatingPointError("EXP3++ policy fell below an arm floor")
        point = Exp3PlusPlusSchedulePoint(
            turn=int(turn),
            eta=float(eta),
            gamma=exploration_mass,
            epsilon_min=float(np.min(epsilon)),
            epsilon_max=float(np.max(epsilon)),
            maximum_gap_lcb=float(np.max(gaps)),
            forced_initialization=False,
        )
        return policy, point, gaps, epsilon

    def policy(self, *, turn: int | None = None) -> np.ndarray:
        selected_turn = (
            int(self.counts.sum()) + 1 if turn is None else int(turn)
        )
        policy, _, _, _ = self._policy_details(
            turn=selected_turn,
            importance_weighted_losses=self._importance_weighted_losses,
            observed_losses=self._observed_losses,
            observation_counts=self._observation_counts,
        )
        return policy

    def counterfactual_policy(self, *, turn: int | None = None) -> np.ndarray:
        selected_turn = (
            int(self.counts.sum()) + 1 if turn is None else int(turn)
        )
        policy, _, _, _ = self._policy_details(
            turn=selected_turn,
            importance_weighted_losses=(
                self._counterfactual_importance_weighted_losses
            ),
            observed_losses=self._counterfactual_observed_losses,
            observation_counts=self._counterfactual_observation_counts,
        )
        return policy

    def sample(self) -> int:
        if self._pending_update:
            raise RuntimeError("sample() called before updating the prior sample")
        turn = int(self.counts.sum()) + 1
        policy, point, gaps, epsilon = self._policy_details(
            turn=turn,
            importance_weighted_losses=self._importance_weighted_losses,
            observed_losses=self._observed_losses,
            observation_counts=self._observation_counts,
        )
        action = int(self.rng.choice(self.k, p=policy))
        self._last_policy = policy
        self._last_action = action
        self._last_schedule = point
        self._last_gap_lcb = gaps
        self._last_epsilon = epsilon
        self.counts[action] += 1.0
        self._pending_update = True
        self._counterfactual_updated = False
        self._counterfactual_factual_reward = None
        return action

    def _apply_observation(
        self,
        reward: float,
        *,
        importance_weighted_losses: np.ndarray,
        observed_losses: np.ndarray,
        observation_counts: np.ndarray,
    ) -> None:
        if (
            not self._pending_update
            or self._last_action is None
            or self._last_policy is None
        ):
            raise RuntimeError("update() called before sample()")
        value = _unit_reward(reward)
        loss = 1.0 - value
        action = self._last_action
        probability = float(self._last_policy[action])
        importance_weighted_losses[action] += loss / probability
        observed_losses[action] += loss
        observation_counts[action] += 1.0

    def update_counterfactual(
        self,
        factual_reward: float,
        corrected_reward: float,
    ) -> None:
        if self._counterfactual_updated:
            raise RuntimeError("counterfactual already updated for this sample")
        factual = _unit_reward(factual_reward)
        self._apply_observation(
            corrected_reward,
            importance_weighted_losses=(
                self._counterfactual_importance_weighted_losses
            ),
            observed_losses=self._counterfactual_observed_losses,
            observation_counts=self._counterfactual_observation_counts,
        )
        self._counterfactual_updated = True
        self._counterfactual_factual_reward = factual

    def update(self, reward: float) -> None:
        factual = _unit_reward(reward)
        self._apply_observation(
            factual,
            importance_weighted_losses=self._importance_weighted_losses,
            observed_losses=self._observed_losses,
            observation_counts=self._observation_counts,
        )
        if self._counterfactual_updated:
            assert self._counterfactual_factual_reward is not None
            if not math.isclose(
                factual,
                self._counterfactual_factual_reward,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "factual reward differs from counterfactual replay input"
                )
        else:
            self._apply_observation(
                factual,
                importance_weighted_losses=(
                    self._counterfactual_importance_weighted_losses
                ),
                observed_losses=self._counterfactual_observed_losses,
                observation_counts=self._counterfactual_observation_counts,
            )
        self._pending_update = False

    def empirical_strategy(self) -> np.ndarray:
        total = float(np.sum(self.counts))
        if total > 0.0:
            return self.counts / total
        return np.full(self.k, 1.0 / self.k, dtype=float)

    @property
    def last_schedule(self) -> Exp3PlusPlusSchedulePoint | None:
        return self._last_schedule

    @property
    def last_action_probability(self) -> float | None:
        if self._last_action is None or self._last_policy is None:
            return None
        return float(self._last_policy[self._last_action])

    def diagnostics(self) -> dict[str, object]:
        return {
            "confidence_alpha": self.confidence_alpha,
            "exploration_beta": self.exploration_beta,
            "eta_scale": self.eta_scale,
            "forced_initialization_rounds": self.k,
            "last_gap_lcb": self._last_gap_lcb.tolist(),
            "last_arm_exploration": self._last_epsilon.tolist(),
            "cross_action_information_sharing": False,
        }


def _unit_reward(reward: float) -> float:
    value = float(reward)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("reward must be finite and in [0, 1]")
    return value


def _probability_vector(
    values: np.ndarray,
    *,
    expected_size: int,
    label: str,
) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if (
        probabilities.shape != (expected_size,)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or not math.isclose(
            float(np.sum(probabilities)),
            1.0,
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
    ):
        raise ValueError(f"{label} must lie on the declared simplex")
    return probabilities
