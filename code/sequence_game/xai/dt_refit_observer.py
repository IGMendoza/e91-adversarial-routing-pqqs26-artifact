"""Dynamic scheduling for expensive DT explanation snapshots.

The observer is intentionally outside the game dynamics: it reads Q-like policy
objects and step records, then recommends when an explanation DT should be
refit. It does not update Q-values, rewards, routes, attacks, or simulator state.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import numpy as np


RefitMode = Literal["fixed", "logarithmic", "dynamic", "off"]


@dataclass(frozen=True)
class DynamicDTRefitConfig:
    mode: RefitMode = "dynamic"
    observer_interval: int = 50
    fixed_refit_interval: int = 50
    epsilon: float = 0.10
    delta: float = 0.05
    policy_drift_threshold: float = 0.05
    min_refit_gap: Optional[int] = None
    max_refit_gap: int = 2000
    log_snapshot_count: int = 8
    q_margin_quantile_low: float = 0.10
    q_drift_quantile_high: float = 0.90
    q_margin_floor: float = 1e-9
    final_step_refit: bool = True
    probe_state_count: int = 2048
    probe_state_seed: int = 42
    # Upper bound on probe_states x num_actions Q entries held per player
    # snapshot. Caps memory on route-dense graphs (~30k Alice actions) where
    # probe_state_count alone would allow multi-GB snapshots.
    snapshot_q_entry_budget: int = 2_097_152

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "logarithmic", "dynamic", "off"}:
            raise ValueError(f"unsupported DT refit mode {self.mode!r}")
        if self.observer_interval < 1:
            raise ValueError("observer_interval must be >= 1")
        if self.fixed_refit_interval < 1:
            raise ValueError("fixed_refit_interval must be >= 1")
        if self.epsilon <= 0.0:
            raise ValueError("epsilon must be > 0")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.policy_drift_threshold < 0.0:
            raise ValueError("policy_drift_threshold must be >= 0")
        if self.min_refit_gap is not None and self.min_refit_gap < 0:
            raise ValueError("min_refit_gap must be >= 0")
        if self.max_refit_gap < 1:
            raise ValueError("max_refit_gap must be >= 1")
        if self.log_snapshot_count < 1:
            raise ValueError("log_snapshot_count must be >= 1")
        if not 0.0 <= self.q_margin_quantile_low <= 1.0:
            raise ValueError("q_margin_quantile_low must be in [0, 1]")
        if not 0.0 <= self.q_drift_quantile_high <= 1.0:
            raise ValueError("q_drift_quantile_high must be in [0, 1]")
        if self.q_margin_floor <= 0.0:
            raise ValueError("q_margin_floor must be > 0")
        if self.probe_state_count < 1:
            raise ValueError("probe_state_count must be >= 1")
        if self.snapshot_q_entry_budget < 1:
            raise ValueError("snapshot_q_entry_budget must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DTObserverTick:
    step: int
    wall_clock_since_last_tick: float
    total_wall_clock: float
    A_A: int
    A_E: int
    m_min: int
    steps_since_last_refit: int
    D_A: Optional[float]
    D_E: Optional[float]
    D_joint: float
    R_A: Optional[float]
    R_E: Optional[float]
    Q_margin_risk: float
    reward_window_mean: Optional[float]
    reward_window_std: Optional[float]
    alice_action_entropy: Optional[float]
    eve_action_entropy: Optional[float]
    alice_route_entropy: Optional[float]
    eve_target_entropy: Optional[float]
    top_k_action_jaccard_A: Optional[float]
    top_k_action_jaccard_E: Optional[float]
    refit_recommended: bool
    refit_reason: str
    probe_state_count_actual_A: int
    probe_state_count_actual_E: int
    probe_state_selection_seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DTRefitEvent:
    refit_step: int
    refit_reason: str
    scope_label: str = "CONTROL_GAME"
    mode: str = "dynamic"
    m_min: int = 0
    D_A: Optional[float] = None
    D_E: Optional[float] = None
    D_joint: float = 0.0
    R_A: Optional[float] = None
    R_E: Optional[float] = None
    Q_margin_risk: float = 0.0
    dt_fit_wall_clock_seconds: Optional[float] = None
    dt_training_sample_count: Optional[int] = None
    dt_depth: Optional[int] = None
    dt_leaf_count: Optional[int] = None
    dt_fidelity_to_Q_policy: Optional[float] = None
    output_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def record_fit(
        self,
        *,
        wall_clock_seconds: float,
        training_sample_count: int,
        dt_depth: int,
        dt_leaf_count: int,
        output_paths: Iterable[str] = (),
        dt_fidelity_to_Q_policy: Optional[float] = None,
    ) -> None:
        self.dt_fit_wall_clock_seconds = float(wall_clock_seconds)
        self.dt_training_sample_count = int(training_sample_count)
        self.dt_depth = int(dt_depth)
        self.dt_leaf_count = int(dt_leaf_count)
        self.output_paths = [str(path) for path in output_paths]
        self.dt_fidelity_to_Q_policy = (
            None if dt_fidelity_to_Q_policy is None else float(dt_fidelity_to_Q_policy)
        )


@dataclass
class _PolicySnapshot:
    policies: dict[tuple[str, ...], int]
    # float32 arrays: snapshots are diagnostics-only and holding Python float
    # lists here costs ~8x the memory on large action spaces.
    q_values: dict[tuple[str, ...], np.ndarray]
    gaps: dict[tuple[str, ...], float]


class DynamicDTRefitObserver:
    """Cheap observer plus refit scheduler for policy explanation snapshots."""

    def __init__(
        self,
        config: DynamicDTRefitConfig | None = None,
        *,
        total_steps: int,
        alice_action_count: int = 0,
        eve_action_count: int = 0,
        clock: Any = time.perf_counter,
    ) -> None:
        self.config = config or DynamicDTRefitConfig()
        self.total_steps = int(total_steps)
        self._clock = clock
        self._start_time = float(clock())
        self._last_tick_time = self._start_time
        self._observer_wall_clock_total = 0.0
        self._last_refit_step = 0
        self._initial_action_counts = {
            "alice": int(alice_action_count),
            "eve": int(eve_action_count),
        }
        self._probe_states: dict[str, list[tuple[str, ...]]] = {"alice": [], "eve": []}
        self._last_refit_snapshot: dict[str, Optional[_PolicySnapshot]] = {
            "alice": None,
            "eve": None,
        }
        self.ticks: list[DTObserverTick] = []
        self.refit_events: list[DTRefitEvent] = []
        self._reward_window: deque[float] = deque(maxlen=max(1, self.config.observer_interval))
        self._alice_actions: deque[str] = deque(maxlen=max(1, self.config.observer_interval))
        self._eve_actions: deque[str] = deque(maxlen=max(1, self.config.observer_interval))
        self._alice_routes: deque[str] = deque(maxlen=max(1, self.config.observer_interval))
        self._eve_targets: deque[str] = deque(maxlen=max(1, self.config.observer_interval))
        self._milestones = logarithmic_milestones(
            self.total_steps,
            self.m_min(alice_action_count, eve_action_count),
            self.config.log_snapshot_count,
        )
        self._used_milestones: set[int] = set()

    @staticmethod
    def sample_action_lower_bound(
        alice_action_count: int,
        eve_action_count: int,
        *,
        epsilon: float = 0.10,
        delta: float = 0.05,
    ) -> int:
        actions = max(1, int(alice_action_count) + int(eve_action_count))
        return int(math.ceil(math.log((2.0 * actions) / delta) / (2.0 * epsilon * epsilon)))

    def m_min(self, alice_action_count: int | None = None, eve_action_count: int | None = None) -> int:
        return self.sample_action_lower_bound(
            self._initial_action_counts["alice"] if alice_action_count is None else alice_action_count,
            self._initial_action_counts["eve"] if eve_action_count is None else eve_action_count,
            epsilon=self.config.epsilon,
            delta=self.config.delta,
        )

    @property
    def min_refit_gap(self) -> int:
        if self.config.min_refit_gap is not None:
            return int(self.config.min_refit_gap)
        return self.m_min()

    def record_transition(
        self,
        *,
        alice_action: Any = None,
        eve_action: Any = None,
        alice_route: Any = None,
        eve_target: Any = None,
        reward: float | None = None,
    ) -> None:
        if reward is not None:
            self._reward_window.append(float(reward))
        if alice_action is not None:
            self._alice_actions.append(str(alice_action))
        if eve_action is not None:
            self._eve_actions.append(str(eve_action))
        if alice_route is not None:
            self._alice_routes.append(str(alice_route))
        if eve_target is not None:
            self._eve_targets.append(str(eve_target))

    def observe(
        self,
        *,
        step: int,
        alice_q: Any = None,
        eve_q: Any = None,
        force: bool = False,
    ) -> DTObserverTick | None:
        step = int(step)
        if not force and step % self.config.observer_interval != 0 and step != self.total_steps:
            return None

        observe_started = float(self._clock())
        A_A = _num_actions(alice_q) or self._initial_action_counts["alice"]
        A_E = _num_actions(eve_q) or self._initial_action_counts["eve"]
        m_min = self.sample_action_lower_bound(
            A_A,
            A_E,
            epsilon=self.config.epsilon,
            delta=self.config.delta,
        )
        if not self._milestones or self._milestones[-1] != self.total_steps:
            self._milestones = logarithmic_milestones(
                self.total_steps,
                m_min,
                self.config.log_snapshot_count,
            )

        alice_diag = self._player_diagnostics("alice", alice_q)
        eve_diag = self._player_diagnostics("eve", eve_q)
        D_A = alice_diag["policy_drift"]
        D_E = eve_diag["policy_drift"]
        R_A = alice_diag["q_margin_risk"]
        R_E = eve_diag["q_margin_risk"]
        D_joint = max(_none_to_zero(D_A), _none_to_zero(D_E))
        q_margin_risk = max(_none_to_zero(R_A), _none_to_zero(R_E))
        steps_since_last_refit = step - self._last_refit_step
        recommended, reason = self._recommend_refit(
            step=step,
            m_min=m_min,
            steps_since_last_refit=steps_since_last_refit,
            D_joint=D_joint,
            q_margin_risk=q_margin_risk,
        )

        tick = DTObserverTick(
            step=step,
            wall_clock_since_last_tick=observe_started - self._last_tick_time,
            total_wall_clock=observe_started - self._start_time,
            A_A=A_A,
            A_E=A_E,
            m_min=m_min,
            steps_since_last_refit=steps_since_last_refit,
            D_A=D_A,
            D_E=D_E,
            D_joint=D_joint,
            R_A=R_A,
            R_E=R_E,
            Q_margin_risk=q_margin_risk,
            reward_window_mean=_mean_or_none(self._reward_window),
            reward_window_std=_std_or_none(self._reward_window),
            alice_action_entropy=_entropy_or_none(self._alice_actions),
            eve_action_entropy=_entropy_or_none(self._eve_actions),
            alice_route_entropy=_entropy_or_none(self._alice_routes),
            eve_target_entropy=_entropy_or_none(self._eve_targets),
            top_k_action_jaccard_A=alice_diag["top_k_jaccard"],
            top_k_action_jaccard_E=eve_diag["top_k_jaccard"],
            refit_recommended=recommended,
            refit_reason=reason,
            probe_state_count_actual_A=len(self._probe_states["alice"]),
            probe_state_count_actual_E=len(self._probe_states["eve"]),
            probe_state_selection_seed=self.config.probe_state_seed,
        )
        observe_finished = float(self._clock())
        self._observer_wall_clock_total += observe_finished - observe_started
        self._last_tick_time = observe_started
        self.ticks.append(tick)
        return tick

    def refit_event_from_tick(self, tick: DTObserverTick) -> DTRefitEvent:
        event = DTRefitEvent(
            refit_step=tick.step,
            refit_reason=tick.refit_reason,
            mode=self.config.mode,
            m_min=tick.m_min,
            D_A=tick.D_A,
            D_E=tick.D_E,
            D_joint=tick.D_joint,
            R_A=tick.R_A,
            R_E=tick.R_E,
            Q_margin_risk=tick.Q_margin_risk,
        )
        self.refit_events.append(event)
        return event

    def mark_refit(
        self,
        *,
        step: int,
        alice_q: Any = None,
        eve_q: Any = None,
        reason: str = "",
    ) -> None:
        self._last_refit_step = int(step)
        self._last_refit_snapshot["alice"] = self._snapshot_for_player("alice", alice_q)
        self._last_refit_snapshot["eve"] = self._snapshot_for_player("eve", eve_q)
        for milestone in self._milestones:
            if milestone <= step:
                self._used_milestones.add(milestone)
        if reason == "logarithmic_milestone_due":
            self._used_milestones.add(step)

    def summary(self) -> dict[str, Any]:
        fit_times = [
            event.dt_fit_wall_clock_seconds
            for event in self.refit_events
            if event.dt_fit_wall_clock_seconds is not None
        ]
        reasons = Counter(event.refit_reason for event in self.refit_events)
        final_done = any(event.refit_step == self.total_steps for event in self.refit_events)
        fixed_equiv = self.total_steps // max(1, self.config.fixed_refit_interval)
        return {
            "total_training_steps": self.total_steps,
            "A_A": self._initial_action_counts["alice"],
            "A_E": self._initial_action_counts["eve"],
            "m_min": self.m_min(),
            "observer_interval": self.config.observer_interval,
            "refit_mode": self.config.mode,
            "refit_count": len(self.refit_events),
            "fixed_50_refit_count_equivalent": fixed_equiv,
            "estimated_refits_avoided": max(0, fixed_equiv - len(self.refit_events)),
            "total_observer_wall_clock": self._observer_wall_clock_total,
            "total_dt_fit_wall_clock": sum(fit_times),
            "mean_dt_fit_wall_clock": statistics.mean(fit_times) if fit_times else 0.0,
            "p50_dt_fit_wall_clock": _quantile(fit_times, 0.50) if fit_times else 0.0,
            "p90_dt_fit_wall_clock": _quantile(fit_times, 0.90) if fit_times else 0.0,
            "refit_steps": [event.refit_step for event in self.refit_events],
            "refit_reasons_count": dict(reasons),
            "final_refit_done": final_done,
            "no_convergence_claim": True,
        }

    def write_outputs(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dynamic_refit_config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        _write_rows_csv(out_dir / "observer_ticks.csv", [tick.to_dict() for tick in self.ticks])
        _write_rows_jsonl(out_dir / "observer_ticks.jsonl", [tick.to_dict() for tick in self.ticks])
        _write_rows_csv(out_dir / "dt_refit_events.csv", [event.to_dict() for event in self.refit_events])
        _write_rows_jsonl(out_dir / "dt_refit_events.jsonl", [event.to_dict() for event in self.refit_events])
        (out_dir / "dt_refit_summary.json").write_text(
            json.dumps(self.summary(), indent=2, default=str),
            encoding="utf-8",
        )

    def _recommend_refit(
        self,
        *,
        step: int,
        m_min: int,
        steps_since_last_refit: int,
        D_joint: float,
        q_margin_risk: float,
    ) -> tuple[bool, str]:
        final_due = step == self.total_steps and self.config.final_step_refit
        if self.config.mode == "off":
            return (True, "final_step") if final_due else (False, "")
        if self.config.mode == "fixed":
            if step % self.config.fixed_refit_interval == 0:
                return True, "fixed_interval"
            return (True, "final_step") if final_due else (False, "")

        if step < m_min and not final_due:
            return False, ""
        if steps_since_last_refit < self.min_refit_gap and not final_due:
            return False, ""

        milestone_due = self._logarithmic_milestone_due(step)
        if self.config.mode == "logarithmic":
            if milestone_due:
                return True, "logarithmic_milestone_due"
            return (True, "final_step") if final_due else (False, "")

        if D_joint >= self.config.policy_drift_threshold:
            return True, "policy_drift"
        if q_margin_risk >= 1.0:
            return True, "q_margin_risk"
        if milestone_due:
            return True, "logarithmic_milestone_due"
        if steps_since_last_refit >= self.config.max_refit_gap:
            return True, "max_refit_gap"
        if final_due:
            return True, "final_step"
        return False, ""

    def _logarithmic_milestone_due(self, step: int) -> bool:
        for milestone in self._milestones:
            if milestone in self._used_milestones:
                continue
            if step >= milestone:
                return True
        return False

    def _player_diagnostics(self, player: str, q_table: Any) -> dict[str, Optional[float]]:
        self._extend_probe_states(player, q_table)
        snapshot = self._snapshot_for_player(player, q_table)
        prior = self._last_refit_snapshot[player]
        if snapshot is None or prior is None:
            return {
                "policy_drift": 0.0 if snapshot is not None else None,
                "q_margin_risk": 0.0 if snapshot is not None else None,
                "top_k_jaccard": 1.0 if snapshot is not None else None,
            }

        common_states = [state for state in snapshot.policies if state in prior.policies]
        if not common_states:
            return {"policy_drift": 0.0, "q_margin_risk": 0.0, "top_k_jaccard": 1.0}
        unchanged = sum(
            1 for state in common_states
            if snapshot.policies[state] == prior.policies[state]
        )
        drift = 1.0 - unchanged / len(common_states)
        q_drifts = []
        gaps = []
        for state in common_states:
            current = snapshot.q_values[state]
            previous = prior.q_values[state]
            n = min(len(current), len(previous))
            if n <= 0:
                continue
            q_drifts.append(float(np.max(np.abs(
                np.asarray(current[:n], dtype=np.float64)
                - np.asarray(previous[:n], dtype=np.float64)
            ))))
            gaps.append(prior.gaps.get(state, 0.0))
        risk = (
            2.0 * _quantile(q_drifts, self.config.q_drift_quantile_high)
            / max(_quantile(gaps, self.config.q_margin_quantile_low), self.config.q_margin_floor)
            if q_drifts and gaps else 0.0
        )
        return {
            "policy_drift": drift,
            "q_margin_risk": risk,
            "top_k_jaccard": _jaccard(
                _top_k_actions(snapshot.policies.values()),
                _top_k_actions(prior.policies.values()),
            ),
        }

    def _probe_capacity(self, player: str, q_table: Any) -> int:
        num_actions = _num_actions(q_table) or self._initial_action_counts.get(player, 0)
        by_budget = self.config.snapshot_q_entry_budget // max(1, num_actions)
        return max(1, min(self.config.probe_state_count, by_budget))

    def _extend_probe_states(self, player: str, q_table: Any) -> None:
        if q_table is None:
            return
        probes = self._probe_states[player]
        capacity = self._probe_capacity(player, q_table)
        if len(probes) >= capacity:
            return
        known = set(probes)
        candidates = [
            state for state in _candidate_states(q_table)
            if state not in known
        ]
        candidates.sort(key=lambda state: _stable_state_key(
            state,
            seed=self.config.probe_state_seed,
            player=player,
        ))
        room = capacity - len(probes)
        probes.extend(candidates[:room])

    def _snapshot_for_player(self, player: str, q_table: Any) -> Optional[_PolicySnapshot]:
        if q_table is None:
            return None
        self._extend_probe_states(player, q_table)
        policies: dict[tuple[str, ...], int] = {}
        q_values: dict[tuple[str, ...], list[float]] = {}
        gaps: dict[tuple[str, ...], float] = {}
        for state in self._probe_states[player]:
            values = _q_values(q_table, state)
            if not values:
                continue
            policies[state] = _best_index(values)
            q_values[state] = np.asarray(values, dtype=np.float32)
            gaps[state] = _action_gap(values)
        return _PolicySnapshot(policies=policies, q_values=q_values, gaps=gaps)


def logarithmic_milestones(total_steps: int, m_min: int, count: int) -> list[int]:
    total_steps = max(1, int(total_steps))
    m_min = max(1, min(int(m_min), total_steps))
    count = max(1, int(count))
    if count == 1 or m_min == total_steps:
        return [total_steps] if m_min == total_steps else [m_min, total_steps]
    values = {
        int(math.ceil(m_min * (total_steps / m_min) ** (i / (count - 1))))
        for i in range(count)
    }
    values.add(m_min)
    values.add(total_steps)
    return sorted(v for v in values if m_min <= v <= total_steps)


_MAX_CANDIDATE_CROSS_PRODUCT = 20_000


def _candidate_states(q_table: Any) -> list[tuple[str, ...]]:
    if q_table is None:
        return []
    states: set[tuple[str, ...]] = set()
    seen = getattr(q_table, "seen_state_tuples", None)
    if callable(seen):
        try:
            states.update(tuple(str(part) for part in state) for state in seen())
        except Exception:
            pass
    raw_q = getattr(q_table, "_q", None)
    if isinstance(raw_q, dict):
        states.update(tuple(str(part) for part in state) for state in raw_q)
    buffer = getattr(q_table, "_buffer", None)
    if isinstance(buffer, list):
        states.update(tuple(str(part) for part in sample[0]) for sample in buffer if sample)
    encoder = getattr(q_table, "_encoder", None)
    if encoder is not None and hasattr(encoder, "all_state_tuples"):
        # The vocabulary cross-product explodes combinatorially on large
        # state spaces; only enumerate it when provably small.
        product_size_fn = getattr(encoder, "vocab_product_size", None)
        product_size = product_size_fn() if callable(product_size_fn) else None
        if product_size is None or product_size <= _MAX_CANDIDATE_CROSS_PRODUCT:
            try:
                states.update(tuple(str(part) for part in state) for state in encoder.all_state_tuples())
            except Exception:
                pass
    return sorted(states)


def _q_values(q_table: Any, state: tuple[str, ...]) -> list[float]:
    try:
        return [float(value) for value in q_table.values(state)]
    except Exception:
        return []


def _num_actions(q_table: Any) -> int:
    if q_table is None:
        return 0
    return int(getattr(q_table, "num_actions", 0) or 0)


def _best_index(values: list[float]) -> int:
    return min(range(len(values)), key=lambda idx: (-values[idx], idx))


def _action_gap(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values, reverse=True)
    return float(ordered[0] - ordered[1])


def _quantile(values: Iterable[float], q: float) -> float:
    vals = sorted(float(value) for value in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = min(max(float(q), 0.0), 1.0) * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _mean_or_none(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return float(statistics.mean(vals)) if vals else None


def _std_or_none(values: Iterable[float]) -> Optional[float]:
    vals = list(values)
    return float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0 if vals else None


def _entropy_or_none(values: Iterable[str]) -> Optional[float]:
    vals = list(values)
    if not vals:
        return None
    total = len(vals)
    counts = Counter(vals)
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def _none_to_zero(value: Optional[float]) -> float:
    return 0.0 if value is None else float(value)


def _stable_state_key(state: tuple[str, ...], *, seed: int, player: str) -> str:
    payload = json.dumps([int(seed), player, list(state)], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _top_k_actions(actions: Iterable[int], k: int = 10) -> set[int]:
    counts = Counter(actions)
    return {
        action for action, _count in
        sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:k]
    }


def _jaccard(left: set[int], right: set[int]) -> Optional[float]:
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def _write_rows_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
