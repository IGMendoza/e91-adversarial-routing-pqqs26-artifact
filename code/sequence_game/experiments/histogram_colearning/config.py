"""Serializable configuration for compiled-histogram bandit co-learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS_DB_PATH = (
    REPO_ROOT
    / "runs"
    / "graph_corpus_sqlite_v1_topology_bench_real_rank10km_normalized"
    / "graph_corpus.sqlite"
)
DEFAULT_REPEATER_SERVICE_PROFILE_PATH = (
    REPO_ROOT
    / "configs"
    / "physical"
    / "repeater_services"
    / "cui_2026_rolling_purified_near_term_composite.json"
)

ALICE_LEARNER_FLAT_EXP3 = "flat_exp3"
ALICE_LEARNER_TSALLIS_INF = "tsallis_inf"
ALICE_LEARNER_EXP3_PLUS_PLUS = "exp3_plus_plus"
ALICE_LEARNER_LINEAR_ROUTE_EXP2 = "linear_route_exp2"
ALICE_LEARNER_DAG_PATH_COMBAND = "dag_path_comband"
ALICE_LEARNER_DAG_PATH_EXP2 = "dag_path_exp2"
ALICE_LEARNER_MAITI_DAG_FTRL = "maiti_dag_ftrl"
ALICE_LEARNER_ITO_DAG_FTRL = "ito_dag_ftrl"
ALICE_LEARNER_MODES = frozenset({
    ALICE_LEARNER_DAG_PATH_COMBAND,
    ALICE_LEARNER_DAG_PATH_EXP2,
    ALICE_LEARNER_EXP3_PLUS_PLUS,
    ALICE_LEARNER_FLAT_EXP3,
    ALICE_LEARNER_LINEAR_ROUTE_EXP2,
    ALICE_LEARNER_MAITI_DAG_FTRL,
    ALICE_LEARNER_ITO_DAG_FTRL,
    ALICE_LEARNER_TSALLIS_INF,
})
ALICE_LEARNER_ROLES = {
    ALICE_LEARNER_FLAT_EXP3: "frozen_compatibility_comparator",
    ALICE_LEARNER_TSALLIS_INF: "adaptive_independent_arm_control",
    ALICE_LEARNER_EXP3_PLUS_PLUS: "adaptive_independent_arm_control",
    ALICE_LEARNER_LINEAR_ROUTE_EXP2: "working_finite_structured_baseline",
    ALICE_LEARNER_DAG_PATH_COMBAND: "diagnostic_negative_baseline",
    ALICE_LEARNER_DAG_PATH_EXP2: "diagnostic_interim_baseline",
    ALICE_LEARNER_MAITI_DAG_FTRL: "modern_adversarial_structured_target",
    ALICE_LEARNER_ITO_DAG_FTRL: "modern_best_of_both_worlds_structured_target",
}
EVE_LEARNER_FLAT_EXP3 = "flat_exp3"
EVE_LEARNER_TSALLIS_INF = "tsallis_inf"
EVE_LEARNER_EXP3_PLUS_PLUS = "exp3_plus_plus"
EVE_LEARNER_LINEAR_ACTION_EXP2 = "linear_action_exp2"
EVE_LEARNER_MODES = frozenset({
    EVE_LEARNER_EXP3_PLUS_PLUS,
    EVE_LEARNER_FLAT_EXP3,
    EVE_LEARNER_LINEAR_ACTION_EXP2,
    EVE_LEARNER_TSALLIS_INF,
})
EVE_LEARNER_ROLES = {
    EVE_LEARNER_FLAT_EXP3: "frozen_compatibility_comparator",
    EVE_LEARNER_TSALLIS_INF: "adaptive_independent_arm_control",
    EVE_LEARNER_EXP3_PLUS_PLUS: "adaptive_independent_arm_control",
    EVE_LEARNER_LINEAR_ACTION_EXP2: "working_finite_structured_baseline",
}
EVE_STRUCTURED_FEATURE_FAMILY_TARGET_TOPOLOGY = "family_target_topology"
EVE_STRUCTURED_FEATURE_MODES = frozenset({
    EVE_STRUCTURED_FEATURE_FAMILY_TARGET_TOPOLOGY,
})
STRUCTURED_EXPLORATION_UNIFORM = "uniform"
STRUCTURED_EXPLORATION_G_OPTIMAL = "g_optimal"
STRUCTURED_EXPLORATION_MODES = frozenset({
    STRUCTURED_EXPLORATION_UNIFORM,
    STRUCTURED_EXPLORATION_G_OPTIMAL,
})


def alice_learner_role(mode: str) -> str:
    """Return the declared project role of a histogram Alice learner."""
    try:
        return ALICE_LEARNER_ROLES[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported alice_learner_mode {mode!r}") from exc


def eve_learner_role(mode: str) -> str:
    """Return the declared project role of a histogram Eve learner."""
    try:
        return EVE_LEARNER_ROLES[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported eve_learner_mode {mode!r}") from exc


def resolve_structured_exploration(mode: str, requested: str | None) -> str:
    """Resolve CLI exploration without changing historical flat-run configs."""
    alice_learner_role(mode)
    if requested is not None:
        if requested not in STRUCTURED_EXPLORATION_MODES:
            raise ValueError(f"unsupported structured exploration {requested!r}")
        return requested
    if mode == ALICE_LEARNER_LINEAR_ROUTE_EXP2:
        return STRUCTURED_EXPLORATION_G_OPTIMAL
    return STRUCTURED_EXPLORATION_UNIFORM


def resolve_eve_structured_exploration(
    mode: str,
    requested: str | None,
) -> str:
    """Resolve Eve's finite-action exploration geometry from the CLI."""
    eve_learner_role(mode)
    if requested is not None:
        if requested not in STRUCTURED_EXPLORATION_MODES:
            raise ValueError(f"unsupported Eve structured exploration {requested!r}")
        return requested
    if mode == EVE_LEARNER_LINEAR_ACTION_EXP2:
        return STRUCTURED_EXPLORATION_G_OPTIMAL
    return STRUCTURED_EXPLORATION_UNIFORM


@dataclass(frozen=True)
class HistogramCoLearningConfig:
    """One stationary Alice/Eve co-learning experiment.

    ``turns=100_000`` is the production default requested for this experiment.
    Focused correctness runs should override it with a much smaller value.
    """

    turns: int = 100_000
    requested_pairs_per_turn: int = 1_200
    final_window: int = 10_000
    record_every: int = 1_000
    posthoc_samples_per_cell: int = 0
    seed: int = 42
    game_mode: str = "attack_surface_all"
    alice_learner_mode: str = ALICE_LEARNER_FLAT_EXP3
    # The explicit Eve selector makes the runtime genuinely two-player and is
    # extended as new independent/structured Eve learners are introduced.
    eve_learner_mode: str = EVE_LEARNER_FLAT_EXP3
    # Uniform preserves the first structured prototype exactly. G-optimal is
    # an explicit structured-learning refinement, not a route-quality prior.
    structured_exploration_mode: str = STRUCTURED_EXPLORATION_UNIFORM
    # Eve's feature map is the Step 3 topology/family representation selected
    # before online play. Its exploration design is configured independently
    # from Alice's so unilateral experiments change one learner at a time.
    eve_structured_feature_mode: str = (
        EVE_STRUCTURED_FEATURE_FAMILY_TARGET_TOPOLOGY
    )
    eve_structured_exploration_mode: str = STRUCTURED_EXPLORATION_UNIFORM
    exp3_schedule_mode: str = "anytime"
    alice_exp3_gamma: float = 0.07
    eve_exp3_gamma: float = 0.07
    alice_exp3_eta_c: float = 1.0
    eve_exp3_eta_c: float = 1.0
    exp3_t0: float = 10_000.0
    exp3_gamma_max: float = 0.20
    # Best-of-both-worlds independent-arm controls. Unit scales and EXP3++'s
    # confidence constants are the papers' theorem settings; non-unit scales
    # are explicitly labelled finite-time sensitivities.
    tsallis_inf_estimator: str = "rv"
    # Role-specific values permit genuinely symmetric calibration.  ``None``
    # preserves all existing configurations by resolving to the shared field.
    alice_tsallis_inf_estimator: str | None = None
    eve_tsallis_inf_estimator: str | None = None
    alice_tsallis_inf_eta_scale: float = 1.0
    eve_tsallis_inf_eta_scale: float = 1.0
    exp3pp_confidence_alpha: float = 3.0
    alice_exp3pp_exploration_beta: float = 256.0
    eve_exp3pp_exploration_beta: float = 256.0
    alice_exp3pp_eta_scale: float = 1.0
    eve_exp3pp_eta_scale: float = 1.0
    # Literature defaults for the two modern aggregate-feedback DAG learners.
    # Scale parameters exist for preregistered calibration sweeps; they do not
    # encode route quality and default to the papers' stated schedules.
    maiti_failure_probability: float = 0.05
    maiti_eta_scale: float = 1.0
    maiti_implicit_exploration_scale: float = 1.0
    maiti_reward_loss_baseline: float = 1.0
    # Optional for prefix calibration: execute fewer turns while retaining the
    # FTRL eta/gamma schedule of the declared deployment horizon.
    maiti_schedule_horizon: int | None = None
    ito_regularizer: str = "tsallis"
    ito_eta_scale: float = 1.0
    ito_beta: float = 2.0
    path_ftrl_solver_tolerance: float = 1e-6
    path_ftrl_solver_max_iterations: int = 60
    # Production learning starts without a route-quality belief.  Non-zero
    # values are retained only for explicitly labelled sensitivity analyses.
    alice_physics_prior_mix: float = 0.0
    repeater_service_profile_path: Path = DEFAULT_REPEATER_SERVICE_PROFILE_PATH
    accepted_alive_time_s: float | None = None
    age_points_per_edge: int = 8
    allow_known_1550_to_1558_detector_approximation: bool = True
    fock_cutoff: int = 3
    evaluated_cache_size: int = 256

    def __post_init__(self) -> None:
        if self.turns < 1:
            raise ValueError("turns must be >= 1")
        if self.requested_pairs_per_turn < 1:
            raise ValueError("requested_pairs_per_turn must be >= 1")
        if self.final_window < 1:
            raise ValueError("final_window must be >= 1")
        if self.record_every < 1:
            raise ValueError("record_every must be >= 1")
        if self.posthoc_samples_per_cell < 0:
            raise ValueError("posthoc_samples_per_cell must be >= 0")
        if self.game_mode not in {
            "attack_surface_dos",
            "attack_surface_mixed",
            "attack_surface_all",
        }:
            raise ValueError(f"unsupported game_mode {self.game_mode!r}")
        if self.alice_learner_mode not in ALICE_LEARNER_MODES:
            raise ValueError(
                f"unsupported alice_learner_mode {self.alice_learner_mode!r}"
            )
        if self.eve_learner_mode not in EVE_LEARNER_MODES:
            raise ValueError(
                f"unsupported eve_learner_mode {self.eve_learner_mode!r}"
            )
        if self.structured_exploration_mode not in STRUCTURED_EXPLORATION_MODES:
            raise ValueError(
                "unsupported structured_exploration_mode "
                f"{self.structured_exploration_mode!r}"
            )
        if self.eve_structured_feature_mode not in EVE_STRUCTURED_FEATURE_MODES:
            raise ValueError(
                "unsupported eve_structured_feature_mode "
                f"{self.eve_structured_feature_mode!r}"
            )
        if (
            self.eve_structured_exploration_mode
            not in STRUCTURED_EXPLORATION_MODES
        ):
            raise ValueError(
                "unsupported eve_structured_exploration_mode "
                f"{self.eve_structured_exploration_mode!r}"
            )
        if self.exp3_schedule_mode not in {"constant", "anytime"}:
            raise ValueError(
                f"unsupported exp3_schedule_mode {self.exp3_schedule_mode!r}"
            )
        for name, value in (
            ("alice_exp3_gamma", self.alice_exp3_gamma),
            ("eve_exp3_gamma", self.eve_exp3_gamma),
            ("exp3_gamma_max", self.exp3_gamma_max),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.alice_exp3_eta_c <= 0.0 or self.eve_exp3_eta_c <= 0.0:
            raise ValueError("Exp3 eta coefficients must be > 0")
        if self.exp3_t0 < 0.0:
            raise ValueError("exp3_t0 must be >= 0")
        for name, value in (
            ("tsallis_inf_estimator", self.tsallis_inf_estimator),
            ("alice_tsallis_inf_estimator", self.alice_tsallis_inf_estimator),
            ("eve_tsallis_inf_estimator", self.eve_tsallis_inf_estimator),
        ):
            if value is not None and value not in {"iw", "rv"}:
                raise ValueError(f"{name} must be one of {{'iw', 'rv'}}")
        if (
            not math.isfinite(self.exp3pp_confidence_alpha)
            or self.exp3pp_confidence_alpha < 3.0
        ):
            raise ValueError("exp3pp_confidence_alpha must be >= 3")
        for name, value in (
            ("alice_tsallis_inf_eta_scale", self.alice_tsallis_inf_eta_scale),
            ("eve_tsallis_inf_eta_scale", self.eve_tsallis_inf_eta_scale),
            (
                "alice_exp3pp_exploration_beta",
                self.alice_exp3pp_exploration_beta,
            ),
            (
                "eve_exp3pp_exploration_beta",
                self.eve_exp3pp_exploration_beta,
            ),
            ("alice_exp3pp_eta_scale", self.alice_exp3pp_eta_scale),
            ("eve_exp3pp_eta_scale", self.eve_exp3pp_eta_scale),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if not 0.0 < self.maiti_failure_probability < 1.0:
            raise ValueError("maiti_failure_probability must be in (0, 1)")
        if (
            self.maiti_eta_scale <= 0.0
            or self.maiti_implicit_exploration_scale <= 0.0
            or self.ito_eta_scale <= 0.0
            or self.ito_beta <= 0.0
        ):
            raise ValueError("modern path-learner tuning values must be > 0")
        if not 0.0 <= self.maiti_reward_loss_baseline <= 1.0:
            raise ValueError("maiti_reward_loss_baseline must be in [0, 1]")
        if (
            self.maiti_schedule_horizon is not None
            and self.maiti_schedule_horizon < self.turns
        ):
            raise ValueError("maiti_schedule_horizon must be >= turns")
        if self.ito_regularizer not in {"tsallis", "log_barrier"}:
            raise ValueError(f"unsupported ito_regularizer {self.ito_regularizer!r}")
        if self.path_ftrl_solver_tolerance <= 0.0:
            raise ValueError("path_ftrl_solver_tolerance must be > 0")
        if self.path_ftrl_solver_max_iterations < 1:
            raise ValueError("path_ftrl_solver_max_iterations must be >= 1")
        if not 0.0 <= self.alice_physics_prior_mix < 1.0:
            raise ValueError("alice_physics_prior_mix must be in [0, 1)")
        if (
            self.alice_learner_mode in {
                ALICE_LEARNER_DAG_PATH_COMBAND,
                ALICE_LEARNER_DAG_PATH_EXP2,
                ALICE_LEARNER_EXP3_PLUS_PLUS,
                ALICE_LEARNER_LINEAR_ROUTE_EXP2,
                ALICE_LEARNER_MAITI_DAG_FTRL,
                ALICE_LEARNER_ITO_DAG_FTRL,
                ALICE_LEARNER_TSALLIS_INF,
            }
            and self.alice_physics_prior_mix != 0.0
        ):
            raise ValueError(
                "this Alice learner requires the unbiased uniform route prior; "
                "alice_physics_prior_mix must be 0"
            )
        if self.accepted_alive_time_s is not None and self.accepted_alive_time_s <= 0.0:
            raise ValueError("accepted_alive_time_s must be > 0 when supplied")
        if self.age_points_per_edge < 2:
            raise ValueError("age_points_per_edge must be >= 2")
        if self.fock_cutoff < 2:
            raise ValueError("fock_cutoff must be >= 2")
        if self.evaluated_cache_size < 1:
            raise ValueError("evaluated_cache_size must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["repeater_service_profile_path"] = str(
            self.repeater_service_profile_path
        )
        return payload

    @property
    def resolved_alice_tsallis_inf_estimator(self) -> str:
        """Estimator used by Alice after applying the shared-field fallback."""

        return self.alice_tsallis_inf_estimator or self.tsallis_inf_estimator

    @property
    def resolved_eve_tsallis_inf_estimator(self) -> str:
        """Estimator used by Eve after applying the shared-field fallback."""

        return self.eve_tsallis_inf_estimator or self.tsallis_inf_estimator
