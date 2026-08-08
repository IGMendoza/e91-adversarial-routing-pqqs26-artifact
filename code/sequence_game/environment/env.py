"""Eve-vs-network game environment (toy scope).

Glues together: Alice's routing policy, Eve's action, the toy trial generator,
the public observation builder, and an explicit reward config. Lightweight
local API (Gymnasium is not a dependency): ``reset(seed)`` and
``step(action_index) -> (obs, reward, terminated, truncated, info)``.

Information barrier: Eve-visible outputs are the PublicObservation, the reward,
and the action result's public view inside ``info``. Route paths, transcripts,
and bases/outcomes stay in private attributes (``last_transcript``), which are
for metrics/debugging only and must not be fed to learning agents.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from ..claims import ClaimGuardError, validate_information_gain_reward
from ..eve.actions import EveAction
from ..eve.observation import ObservationConfig, PublicObservation, build_observation
from ..protocol.toy_trial import ToyTrialConfig, run_toy_trial
from ..protocol.transcript import PublicTranscript, TrialTranscript
from ..routing.policy import RoutingPolicy
from ..topology.ir import TopologyIR

INITIAL_ACTION_ID = "none"


class EnvironmentError_(ValueError):
    """Invalid environment configuration or usage."""


@dataclass(frozen=True)
class RewardConfig:
    """Eve's reward shaping. Game-design constants, not security metrics."""

    accept_reward: float
    abort_reward: float
    cost_weight: float = 1.0
    # The information-gain term weights Eve's empirical I(K;E|P) (bits) from
    # ``eve.information.compute_information_gain``. It is only meaningful with the
    # SeQUeNCe trial backend (which records Eve's measurements); the toy backend
    # has no such record, so it contributes 0 there.
    information_gain_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_weight < 0:
            raise EnvironmentError_("cost_weight must be >= 0")
        try:
            validate_information_gain_reward(self.information_gain_weight)
        except ClaimGuardError as exc:
            raise EnvironmentError_(str(exc)) from exc

    def to_dict(self) -> dict:
        return {
            "accept_reward": self.accept_reward,
            "abort_reward": self.abort_reward,
            "cost_weight": self.cost_weight,
            "information_gain_weight": self.information_gain_weight,
        }


class EveGameEnv:
    """One Eve decision per step; each step runs one toy protocol trial."""

    def __init__(self, topology: TopologyIR, alice: str, bob: str,
                 routing_policy: RoutingPolicy, actions: list[EveAction],
                 trial_config: ToyTrialConfig, observation_config: ObservationConfig,
                 reward_config: RewardConfig, seed: int):
        if not actions:
            raise EnvironmentError_("need at least one Eve action")
        ids = [a.action_id for a in actions]
        if len(set(ids)) != len(ids):
            raise EnvironmentError_(f"duplicate action ids: {ids}")
        for nid in (alice, bob):
            if nid not in topology.nodes:
                raise EnvironmentError_(f"unknown endpoint {nid!r}")
        self.topology = topology
        self.alice = alice
        self.bob = bob
        self.routing_policy = routing_policy
        self.actions = list(actions)
        self.trial_config = trial_config
        self.observation_config = observation_config
        self.reward_config = reward_config
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._trial_counter = 0
        self.last_transcript: Optional[TrialTranscript] = None  # private, not for agents
        self._last_observation: Optional[PublicObservation] = None

    @property
    def action_ids(self) -> list[str]:
        return [a.action_id for a in self.actions]

    def reset(self, seed: Optional[int] = None) -> PublicObservation:
        if seed is not None:
            self._seed = seed
        self._rng = np.random.default_rng(self._seed)
        self._trial_counter = 0
        self.last_transcript = None
        empty = build_observation(
            _EMPTY_PUBLIC, INITIAL_ACTION_ID, self.observation_config)
        self._last_observation = empty
        return empty

    def step(self, action_index: int
             ) -> tuple[PublicObservation, float, bool, bool, dict[str, Any]]:
        if self._last_observation is None:
            raise EnvironmentError_("call reset() before step()")
        if not 0 <= action_index < len(self.actions):
            raise EnvironmentError_(
                f"action_index {action_index} out of range 0..{len(self.actions) - 1}")
        action = self.actions[action_index]

        route = self.routing_policy.select_route(
            self.topology, self.alice, self.bob, rng=self._rng)
        effect, action_result = action.apply(self.topology, rng=self._rng)

        trial_id = f"trial-{self._trial_counter}"
        self._trial_counter += 1
        transcript = run_toy_trial(self.topology, route, self.trial_config,
                                   effect, self._rng, trial_id)
        self.last_transcript = transcript
        public = transcript.public_view()

        if transcript.accepted:
            reward = self.reward_config.accept_reward
        else:
            reward = self.reward_config.abort_reward
        reward -= self.reward_config.cost_weight * action_result.cost

        observation = build_observation(public, action.action_id,
                                        self.observation_config)
        self._last_observation = observation
        info: dict[str, Any] = {
            "public_transcript": public.to_dict(),
            "action_result": action_result.public_view(),
        }
        return observation, reward, False, False, info


# sentinel public transcript for the pre-first-step observation
_EMPTY_PUBLIC = PublicTranscript(
    trial_id="pre-reset", route_id=None, accepted=None, abort_reason=None,
    qber_estimate=None, latency_ps=None, sifted_count=None)
