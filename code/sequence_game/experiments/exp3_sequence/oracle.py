"""Complete-information oracle for estimated SeQUeNCe payoff matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sequence_game.game_theory import exploitability, solve_zero_sum_lp, solve_zero_sum_selfplay

from .payoff import PayoffEstimate

ABSOLUTE_EVE_WIN_VALUE_EPS = 1e-9
MEANINGFUL_DEGRADATION_RETENTION = 0.90
BASELINE_EQUIVALENT_RETENTION = 0.99


@dataclass(frozen=True)
class OracleSummary:
    graph_id: str
    value: float
    baseline_rate: float
    retention: float | None
    status: str
    alice_strategy: np.ndarray
    eve_strategy: np.ndarray
    exploitability: float
    rm_value: float
    rm_exploitability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "value": self.value,
            "baseline_rate": self.baseline_rate,
            "retention": self.retention,
            "retention_loss": (
                None if self.retention is None else 1.0 - self.retention
            ),
            "status": self.status,
            "status_thresholds": {
                "absolute_eve_win_value_eps": ABSOLUTE_EVE_WIN_VALUE_EPS,
                "meaningful_degradation_retention_below": (
                    MEANINGFUL_DEGRADATION_RETENTION
                ),
                "baseline_equivalent_retention_at_least": (
                    BASELINE_EQUIVALENT_RETENTION
                ),
            },
            "alice_strategy": self.alice_strategy.tolist(),
            "eve_strategy": self.eve_strategy.tolist(),
            "exploitability": self.exploitability,
            "rm_value": self.rm_value,
            "rm_exploitability": self.rm_exploitability,
        }


def solve_oracle(payoff: PayoffEstimate) -> OracleSummary:
    matrix = np.asarray(payoff.payoff, dtype=float)
    value, alice, eve = solve_zero_sum_lp(matrix)
    rm = solve_zero_sum_selfplay(matrix)
    try:
        no_attack_index = payoff.action_ids.index("no_attack")
    except ValueError:
        no_attack_index = 0
    baseline = float(np.max(matrix[:, no_attack_index]))
    if baseline <= 0.0:
        retention = None
        status = "baseline_delivery_zero"
    else:
        retention = min(1.0, max(0.0, float(value / baseline)))
        if value <= ABSOLUTE_EVE_WIN_VALUE_EPS:
            status = "absolute_eve_win"
        elif retention < MEANINGFUL_DEGRADATION_RETENTION:
            status = "meaningful_degradation"
        elif retention < BASELINE_EQUIVALENT_RETENTION:
            status = "minor_degradation"
        else:
            status = "alice_retains_baseline"
    return OracleSummary(
        graph_id=payoff.graph_id,
        value=float(value),
        baseline_rate=baseline,
        retention=retention,
        status=status,
        alice_strategy=np.asarray(alice, dtype=float),
        eve_strategy=np.asarray(eve, dtype=float),
        exploitability=float(exploitability(matrix, alice, eve)),
        rm_value=float(rm.value),
        rm_exploitability=float(rm.exploitability),
    )
