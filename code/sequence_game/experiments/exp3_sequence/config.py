"""Serializable configuration for the retained Exp3 SeQUeNCe experiment."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ALICE_MODES = frozenset({"exp3_bandit", "oracle", "cautious_greedy"})
EVE_MODES = frozenset({"exp3_bandit", "oracle"})
SECURITY_MONITORS = frozenset({"chsh", "qber"})
ACTION_KINDS = frozenset({
    "edge_intercept_resend",
    "memory_degradation",
})
EXP3_SCHEDULE_MODES = frozenset({"constant", "anytime"})
ALICE_ACCEPTANCE_RULES = frozenset({
    "cached_protocol",
    "chsh_and_qber",
    "chsh_only",
})

# SeQUeNCe repeater runtime settings for the fixed corpus. The evaluator scales
# the absolute start time by route length so km-scale RSVP setup does not assert
# out; the window and stop margin remain fixed experiment defaults.
SEQUENCE_START_TIME_PS = 1_000_000
SEQUENCE_END_TIME_PS = 1_001_000_000
SEQUENCE_STOP_TIME_PS = 1_251_000_000
SEQUENCE_SETUP_TRAVERSALS = 12.0
FIBER_CLASSICAL_SPEED_M_PER_PS = 0.0002
CHSH_PAIRS_PER_TRIAL = 350
QBER_PAIRS_PER_TRIAL = 8
QBER_THRESHOLD = 0.15
MIN_KEY_PAIRS = 1
REQUEST_FIDELITY = 0.01
SWAPPING_SUCCESS_PROB = 1.0
SWAPPING_DEGRADATION = 1.0
SEQUENCE_MEMORY_FIDELITY_OVERRIDE = 1.0
TRIALS_PER_CELL = 16
ONLINE_TURNS = 100000
ONLINE_FINAL_WINDOW = 10000
ONLINE_STEP_RECORD_STRIDE = 100
CAUTIOUS_GREEDY_AVOIDANCE_HORIZON = 1
MAX_GRAPHS = 50
MAX_ROUTES = 32
MAX_ROUTE_HOPS = 7
# Short, SeQUeNCe-easy fibers with enough propagation delay to avoid a dense
# retry storm inside the fixed 1 ms simulation window. _length() uses five
# buckets: base + step * {0..4}.
CORPUS_BASE_EDGE_LENGTH_M = 400.0
CORPUS_EDGE_LENGTH_STEP_M = 25.0
BASELINE_PRECOMPUTE_SAMPLES_PER_HOP = 64
BASELINE_PAYOFF_SAMPLES_PER_ROUTE = 16
BASELINE_PRECOMPUTE_SEED = 98_000
BASELINE_CACHE_SQLITE_PATH = Path(__file__).with_name("baselines.sqlite")
ATTACK_PRECOMPUTE_SAMPLES_PER_HOP = 64
ATTACK_PAYOFF_SAMPLES_PER_ROUTE = 16
ATTACK_PRECOMPUTE_SEED = 198_000
ATTACK_CACHE_SQLITE_PATH = Path(__file__).with_name("attack_baselines.sqlite")
CI_HALF_WIDTH_WARN = 0.10
DT_MAX_DEPTH = 3
BASELINE_HEALTH_ROUTES_PER_GRAPH = 3
BASELINE_HEALTH_TRIALS_PER_ROUTE = 16
BASELINE_HEALTH_MIN_ACCEPTED_TRIALS = 1
BASELINE_HEALTH_ACCEPT_RATE_WARN = 0.34
BASELINE_HEALTH_MIN_CHSH_S = 2.10
BASELINE_HEALTH_MIN_DELIVERED_PAIRS = 319
CORPUS_SQLITE_PATH = Path(__file__).with_name("corpus.sqlite")


@dataclass(frozen=True)
class ConditionConfig:
    key: str
    alice: str
    eve: str

    def __post_init__(self) -> None:
        if self.alice not in ALICE_MODES:
            raise ValueError(f"unsupported Alice mode {self.alice!r}")
        if self.eve not in EVE_MODES:
            raise ValueError(f"unsupported Eve mode {self.eve!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONDITIONS = (
    ConditionConfig("exp3_vs_exp3", alice="exp3_bandit", eve="exp3_bandit"),
    ConditionConfig("oracle_eve_vs_exp3_alice", alice="exp3_bandit", eve="oracle"),
    ConditionConfig("exp3_eve_vs_oracle_alice", alice="oracle", eve="exp3_bandit"),
    ConditionConfig(
        "exp3_eve_vs_cautious_greedy_alice",
        alice="cautious_greedy",
        eve="exp3_bandit",
    ),
    ConditionConfig("oracle_vs_oracle", alice="oracle", eve="oracle"),
)


def default_worker_count() -> int:
    count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return max(1, count)


@dataclass(frozen=True)
class Exp3SequenceConfig:
    out_dir: Path = Path("runs/exp3_sequence")
    corpus_db_path: Path = CORPUS_SQLITE_PATH
    max_graphs: int = MAX_GRAPHS
    workers: int = field(default_factory=default_worker_count)
    seed: int = 42
    trials_per_cell: int = TRIALS_PER_CELL
    baseline_cache_db_path: Path = BASELINE_CACHE_SQLITE_PATH
    attack_cache_db_path: Path = ATTACK_CACHE_SQLITE_PATH
    attack_payoff_samples_per_route: int = ATTACK_PAYOFF_SAMPLES_PER_ROUTE
    online_turns: int = ONLINE_TURNS
    online_final_window: int = ONLINE_FINAL_WINDOW
    online_step_record_stride: int = ONLINE_STEP_RECORD_STRIDE
    cautious_greedy_avoidance_horizon: int = CAUTIOUS_GREEDY_AVOIDANCE_HORIZON
    alice_exp3_gamma: float = 0.07
    eve_exp3_gamma: float = 0.07
    exp3_schedule_mode: str = "constant"
    alice_exp3_eta_c: float = 1.0
    eve_exp3_eta_c: float = 1.0
    alice_exp3_t0: float = 10000.0
    eve_exp3_t0: float = 10000.0
    alice_exp3_gamma_max: float = 0.20
    eve_exp3_gamma_max: float = 0.20
    alice_acceptance_rule: str = "chsh_only"
    alice_key_rate_shaping_weight: float = 0.0
    security_monitor: str = "chsh"
    action_kinds: tuple[str, ...] = (
        "edge_intercept_resend",
        "memory_degradation",
    )
    chsh_pairs_per_trial: int = CHSH_PAIRS_PER_TRIAL
    qber_pairs_per_trial: int = QBER_PAIRS_PER_TRIAL
    qber_threshold: float = QBER_THRESHOLD
    min_key_pairs: int = MIN_KEY_PAIRS
    request_fidelity: float = REQUEST_FIDELITY
    start_time_ps: int = SEQUENCE_START_TIME_PS
    end_time_ps: int = SEQUENCE_END_TIME_PS
    stop_time_ps: int = SEQUENCE_STOP_TIME_PS
    sequence_setup_traversals: float = SEQUENCE_SETUP_TRAVERSALS
    swapping_success_prob: float = SWAPPING_SUCCESS_PROB
    swapping_degradation: float = SWAPPING_DEGRADATION
    sequence_memory_fidelity_override: float | None = SEQUENCE_MEMORY_FIDELITY_OVERRIDE
    ci_half_width_warn: float = CI_HALF_WIDTH_WARN
    dt_max_depth: int = DT_MAX_DEPTH
    baseline_health_routes_per_graph: int = BASELINE_HEALTH_ROUTES_PER_GRAPH
    baseline_health_trials_per_route: int = BASELINE_HEALTH_TRIALS_PER_ROUTE
    baseline_health_min_accepted_trials: int = BASELINE_HEALTH_MIN_ACCEPTED_TRIALS
    baseline_health_accept_rate_warn: float = BASELINE_HEALTH_ACCEPT_RATE_WARN
    baseline_health_min_chsh_s: float = BASELINE_HEALTH_MIN_CHSH_S
    baseline_health_min_delivered_pairs: int = BASELINE_HEALTH_MIN_DELIVERED_PAIRS
    conditions: tuple[ConditionConfig, ...] = field(default_factory=lambda: DEFAULT_CONDITIONS)

    def __post_init__(self) -> None:
        if self.max_graphs < 1:
            raise ValueError("max_graphs must be >= 1")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.trials_per_cell < 1:
            raise ValueError("trials_per_cell must be >= 1")
        if self.attack_payoff_samples_per_route < 1:
            raise ValueError("attack_payoff_samples_per_route must be >= 1")
        if self.online_turns < 0:
            raise ValueError("online_turns must be >= 0")
        if self.online_step_record_stride < 1:
            raise ValueError("online_step_record_stride must be >= 1")
        if self.cautious_greedy_avoidance_horizon < 0:
            raise ValueError("cautious_greedy_avoidance_horizon must be >= 0")
        if not 0 < self.alice_exp3_gamma <= 1:
            raise ValueError("alice_exp3_gamma must be in (0, 1]")
        if not 0 < self.eve_exp3_gamma <= 1:
            raise ValueError("eve_exp3_gamma must be in (0, 1]")
        if self.exp3_schedule_mode not in EXP3_SCHEDULE_MODES:
            raise ValueError(f"unsupported EXP3 schedule mode {self.exp3_schedule_mode!r}")
        if self.alice_exp3_eta_c <= 0:
            raise ValueError("alice_exp3_eta_c must be > 0")
        if self.eve_exp3_eta_c <= 0:
            raise ValueError("eve_exp3_eta_c must be > 0")
        if self.alice_exp3_t0 < 0:
            raise ValueError("alice_exp3_t0 must be >= 0")
        if self.eve_exp3_t0 < 0:
            raise ValueError("eve_exp3_t0 must be >= 0")
        if not 0 < self.alice_exp3_gamma_max <= 1:
            raise ValueError("alice_exp3_gamma_max must be in (0, 1]")
        if not 0 < self.eve_exp3_gamma_max <= 1:
            raise ValueError("eve_exp3_gamma_max must be in (0, 1]")
        if self.alice_acceptance_rule not in ALICE_ACCEPTANCE_RULES:
            raise ValueError(
                f"unsupported Alice acceptance rule {self.alice_acceptance_rule!r}"
            )
        if not 0 <= self.alice_key_rate_shaping_weight <= 1:
            raise ValueError("alice_key_rate_shaping_weight must be in [0, 1]")
        if self.security_monitor not in SECURITY_MONITORS:
            raise ValueError(f"unsupported security_monitor {self.security_monitor!r}")
        unknown = sorted(set(self.action_kinds) - ACTION_KINDS)
        if unknown:
            raise ValueError(f"unsupported action kinds {unknown}")
        if self.chsh_pairs_per_trial < 1:
            raise ValueError("chsh_pairs_per_trial must be >= 1")
        if self.qber_pairs_per_trial < 1:
            raise ValueError("qber_pairs_per_trial must be >= 1")
        if not 0 <= self.qber_threshold <= 1:
            raise ValueError("qber_threshold must be in [0, 1]")
        if self.min_key_pairs < 0:
            raise ValueError("min_key_pairs must be >= 0")
        if not 0 <= self.start_time_ps < self.end_time_ps <= self.stop_time_ps:
            raise ValueError("need 0 <= start_time_ps < end_time_ps <= stop_time_ps")
        if self.sequence_setup_traversals < 1:
            raise ValueError("sequence_setup_traversals must be >= 1")
        if not 0 <= self.swapping_success_prob <= 1:
            raise ValueError("swapping_success_prob must be in [0, 1]")
        if not 0 <= self.swapping_degradation <= 1:
            raise ValueError("swapping_degradation must be in [0, 1]")
        if (
                self.sequence_memory_fidelity_override is not None
                and not 0 <= self.sequence_memory_fidelity_override <= 1
        ):
            raise ValueError("sequence_memory_fidelity_override must be in [0, 1]")
        if self.ci_half_width_warn < 0:
            raise ValueError("ci_half_width_warn must be >= 0")
        if self.baseline_health_routes_per_graph < 1:
            raise ValueError("baseline_health_routes_per_graph must be >= 1")
        if self.baseline_health_trials_per_route < 1:
            raise ValueError("baseline_health_trials_per_route must be >= 1")
        if self.baseline_health_min_accepted_trials < 1:
            raise ValueError("baseline_health_min_accepted_trials must be >= 1")
        if not 0 <= self.baseline_health_accept_rate_warn <= 1:
            raise ValueError("baseline_health_accept_rate_warn must be in [0, 1]")
        if self.baseline_health_min_chsh_s < 0:
            raise ValueError("baseline_health_min_chsh_s must be >= 0")
        if self.baseline_health_min_delivered_pairs < 0:
            raise ValueError("baseline_health_min_delivered_pairs must be >= 0")

    @property
    def memory_pairs_per_trial(self) -> int:
        if self.security_monitor == "chsh":
            return int(self.chsh_pairs_per_trial)
        return int(self.qber_pairs_per_trial)

    @property
    def repeater_window_ps(self) -> int:
        return int(self.end_time_ps - self.start_time_ps)

    @property
    def stop_margin_ps(self) -> int:
        return int(self.stop_time_ps - self.end_time_ps)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["out_dir"] = str(self.out_dir)
        data["corpus_db_path"] = str(self.corpus_db_path)
        data["baseline_cache_db_path"] = str(self.baseline_cache_db_path)
        data["attack_cache_db_path"] = str(self.attack_cache_db_path)
        data["exp3_gamma"] = {
            "alice": self.alice_exp3_gamma,
            "eve": self.eve_exp3_gamma,
        }
        data["exp3_schedule"] = {
            "mode": self.exp3_schedule_mode,
            "alice": {
                "eta_c": self.alice_exp3_eta_c,
                "t0": self.alice_exp3_t0,
                "gamma_max": self.alice_exp3_gamma_max,
            },
            "eve": {
                "eta_c": self.eve_exp3_eta_c,
                "t0": self.eve_exp3_t0,
                "gamma_max": self.eve_exp3_gamma_max,
            },
        }
        data["conditions"] = [condition.to_dict() for condition in self.conditions]
        return data
