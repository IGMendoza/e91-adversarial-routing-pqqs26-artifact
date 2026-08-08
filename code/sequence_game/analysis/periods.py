"""Deterministic sequence-period helpers for control-game replay."""

from __future__ import annotations

from typing import Optional, Sequence


def detect_route_period(route_ids: Sequence[str], *, max_period: int = 12,
                        min_repetitions: int = 3,
                        suffix_window: int = 60) -> Optional[int]:
    """Return the repeated suffix period length, if a stable suffix exists."""

    if max_period < 1:
        raise ValueError("max_period must be >= 1")
    if min_repetitions < 2:
        raise ValueError("min_repetitions must be >= 2")
    if len(route_ids) < max_period:
        return None
    suffix = list(route_ids[-min(len(route_ids), suffix_window):])
    max_candidate = min(max_period, len(suffix) // min_repetitions)
    for period in range(1, max_candidate + 1):
        pattern = suffix[-period:]
        if suffix[-period * min_repetitions:] == pattern * min_repetitions:
            return period
    return None
