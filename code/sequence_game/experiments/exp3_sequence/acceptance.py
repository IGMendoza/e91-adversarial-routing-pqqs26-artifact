"""Alice acceptance/reward rules for cached Exp3 SeQUeNCe turns."""

from __future__ import annotations

from sequence_game.protocol.chsh_core import CLASSICAL_BOUND

from .backend import TurnResult
from .config import Exp3SequenceConfig


def alice_accepts_turn(result: TurnResult, config: Exp3SequenceConfig) -> bool:
    """Return Alice's effective win condition for a cached turn.

    ``cached_protocol`` and ``chsh_and_qber`` preserve the original SeQUeNCe
    transcript decision. ``chsh_only`` treats a sampled CHSH violation as
    Alice's binary win and leaves QBER as a logged diagnostic instead of a hard
    veto.
    """

    if config.alice_acceptance_rule in {"cached_protocol", "chsh_and_qber"}:
        return bool(result.accepted)
    if config.alice_acceptance_rule == "chsh_only":
        return chsh_passes(result)
    raise ValueError(
        f"unsupported Alice acceptance rule {config.alice_acceptance_rule!r}"
    )


def alice_reward_for_turn(result: TurnResult, config: Exp3SequenceConfig) -> float:
    base = 1.0 if alice_accepts_turn(result, config) else 0.0
    shaping = float(config.alice_key_rate_shaping_weight)
    if shaping <= 0:
        return base
    cached_proxy = _clip01(float(result.alice_reward))
    return (1.0 - shaping) * base + shaping * cached_proxy


def chsh_passes(result: TurnResult) -> bool:
    return (
        result.chsh_adequately_sampled is True
        and result.chsh_s is not None
        and abs(float(result.chsh_s)) > CLASSICAL_BOUND
    )


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, value))
