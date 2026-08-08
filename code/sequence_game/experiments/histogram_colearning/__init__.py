"""Bandit co-learning over the active compiled histogram physics backend."""

from .config import (
    HistogramCoLearningConfig,
    alice_learner_role,
    eve_learner_role,
    resolve_eve_structured_exploration,
    resolve_structured_exploration,
)
from .corpus import HistogramGameCase, load_topology_bench_case
from .learning import (
    HistogramCoLearningSummary,
    build_histogram_alice_learner,
    build_histogram_eve_learner,
    run_histogram_colearning,
)
from .learner_protocol import (
    BehaviorConditionableScalarFeedbackLearner,
    PlayerLearner,
    ScalarFeedbackLearner,
)
from .path_polytope import (
    HistogramCasePathPolytope,
    HistogramDAGComBand,
    HistogramDAGPathExp2,
    build_histogram_case_path_polytope,
)
from .modern_path_learners import (
    HistogramItoPathFTRL,
    HistogramMaitiPathFTRL,
)
from .modern_path_benchmark import (
    ModernPathBenchmarkSpec,
    benchmark_specs_from_calibration,
    run_modern_path_benchmark,
)
from .physics import (
    HistogramPhysicsPool,
    RoutePhysicsCalibration,
    histogram_cache_plan,
)
from .structure import (
    EVE_FEATURE_ACTION_IDENTITY,
    EVE_FEATURE_FAMILY,
    EVE_FEATURE_FAMILY_BY_TARGET_TOPOLOGY,
    EVE_FEATURE_FAMILY_TARGET_TOPOLOGY,
    EVE_FEATURE_MODES,
    EveActionEmbedding,
    RouteResourceEmbedding,
    eve_action_embedding,
    eve_reward_embedding,
    route_resource_embedding,
    validate_eve_action_contract,
)
from .payoff_structure import analyze_payoff_structure
from .crossplay_validation import two_sided_crossplay_validation
from .topology_census import TopologyCensusConfig, census_topology_bench
from .case_selection import freeze_case_partitions
from .calibration_bank import (
    HistogramPayoffBank,
    PayoffSampleBlock,
    build_histogram_payoff_bank,
)
from .calibration_candidates import (
    LearnerCalibrationCandidate,
    declared_calibration_candidates,
)
from .calibration_cases import (
    FrozenCalibrationCase,
    load_frozen_calibration_cases,
    load_frozen_calibration_specs,
    load_frozen_evaluation_cases,
    load_frozen_evaluation_specs,
    load_frozen_partition_cases,
    load_frozen_partition_specs,
)
from .calibration_game import run_bank_two_player_learning
from .calibration_pipeline import (
    Step7CalibrationPlan,
    run_step7_calibration,
    step7_preflight,
)
from .calibration_artifacts import write_step7_calibration_artifacts
from .prime_probe import (
    PrimeProbePlan,
    PrimeProbeSchedule,
    WITHDRAWAL_STATE_STRATEGIES,
    build_prime_probe_plan,
    build_prime_probe_plans,
    paired_prime_probe_comparison,
    prime_probe_target_family_representatives,
    run_ordered_prime_probe,
    run_prime_probe_withdrawal_state_study,
    run_shuffled_prime_probe_control,
)
from .prime_probe_artifacts import write_prime_probe_artifacts
from .prime_probe_corpus import (
    PrimeProbeDefender,
    build_prime_probe_manifest,
    run_prime_probe_study,
)
from .prime_probe_recovery_corpus import (
    build_withdrawal_state_manifest,
    run_withdrawal_state_corpus_study,
    withdrawal_state_preflight,
)
from .stackelberg_eve import (
    StackelbergCommitment,
    StackelbergCommitmentEvaluation,
    StackelbergOracleResult,
    evaluate_stackelberg_commitment,
    eve_public_payoff_matrix,
    simultaneous_hoeffding_radii,
    solve_stackelberg_oracle,
)
from .stackelberg_eve_runtime import (
    RouteActionRewardEstimator,
    StackelbergExploreCommitConfig,
    run_stackelberg_explore_then_commit,
)
from .stackelberg_eve_corpus import (
    StackelbergDefender,
    build_stackelberg_eve_manifest,
    default_stackelberg_learner_config,
    run_stackelberg_eve_study,
)
from .iris_outlier_artifacts import write_iris_outlier_artifacts
from .stackelberg_steering import (
    StackelbergSteeringConfig,
    feasible_stackelberg_response_margins,
    run_pessimistic_stackelberg_steering,
)
from .stackelberg_steering_corpus import (
    build_stackelberg_steering_manifest,
    run_stackelberg_steering_study,
    select_stackelberg_response_margin,
)
from .stackelberg_steering_artifacts import (
    analyze_stackelberg_steering_results,
    write_stackelberg_steering_artifacts,
)
from .stackelberg_bandit import (
    LearnerConditionedPlanLibrary,
    LearnerHypothesisTracker,
    RouteObservationDesign,
    StackelbergBanditFollowRuntime,
    evaluate_learner_conditioned_prime_probe_plans,
    minimum_route_observation_design,
    run_stackelberg_bandit_follow_identification,
    run_stackelberg_bandit_follow_runtime,
)
from .stackelberg_controller import (
    ControllerPrimeExploitSchedule,
    evaluate_post_follow_controller_plans,
    execute_post_follow_controller_decision,
    run_controller_prime_exploit_continuation,
)
from .stackelberg_controller_plans import (
    ControllerPrimeExploitPlan,
    build_controller_prime_exploit_plans,
    controller_plan_family_census,
    controller_plan_screening_representatives,
    controller_target_family_representatives,
)
from .stackelberg_controller_study import (
    ControllerMatchedStudyConfig,
    run_matched_controller_study,
)
from .quantum_stackelberg_detection import (
    BinomialEvidenceDetectorConfig,
    MatchedClassicalQuantumEndpointDetectors,
    QuantumEndpointEvidence,
    RouteConditionedBinomialEProcess,
)
from .endpoint_evidence_bank import (
    EndpointEvidenceSampleBlock,
    HistogramEndpointEvidenceBank,
    build_histogram_endpoint_evidence_bank,
    validate_endpoint_evidence_bank_against_case,
)
from .matched_payoff_endpoint_bank import (
    MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA,
    build_matched_histogram_payoff_and_endpoint_banks,
)
from .paired_block_evidence import (
    PAIRED_BLOCK_DIAGNOSTIC_SIZES,
    PAIRED_BLOCK_PRIMARY_SIZES,
    PairedBlockSizeBanks,
    derive_paired_block_size_banks,
)
from .endpoint_detection_study import (
    EndpointDetectionStudyConfig,
    run_matched_endpoint_detection_study,
    run_matched_endpoint_detector_continuation,
)
from .endpoint_response import (
    AlarmResponseConfig,
    CommissionedInterventionPosterior,
    CommissionedRobustRoutePolicy,
    PosteriorTrustRegionRoutePolicy,
    RESPONSE_POSTERIOR_TRUST_REGION,
    RESPONSE_SEQUENTIAL_POSTERIOR_TRUST_REGION,
    RESPONSE_UPDATE_NATIVE_ONLY,
    RESPONSE_UPDATE_TSALLIS_BEHAVIOR_IW,
    SequentialInterventionPosterior,
    commissioned_intervention_posterior,
    commissioned_detector_compatible_actions,
    discover_alarm_response_record,
    run_alarm_conditioned_response_continuation,
    solve_commissioned_robust_route_policy,
    solve_posterior_trust_region_route_policy,
)
from .endpoint_response_study import (
    EndpointResponseStudyConfig,
    run_endpoint_response_study_from_detection,
)
from .endpoint_response_calibration import (
    SequentialResponseCalibrationConfig,
    SequentialResponseCandidate,
    run_sequential_response_calibration_from_detection,
    sequential_response_config,
)
from .response_block_milestone import (
    RESPONSE_BLOCK_FREEZE_SCHEMA,
    RESPONSE_BLOCK_MILESTONE_SCHEMA,
    TsallisResponseBlockMilestoneConfig,
    freeze_digest,
    response_result_row,
    select_feedback_response_profiles,
    validate_freeze_manifest,
)
from .bsm_ancilla_follower import (
    BSMAncillaFollowerSmokeConfig,
    run_bsm_ancilla_follower_smoke,
)
from .bsm_ancilla_hidden_route import (
    BSMAncillaHiddenRouteConfig,
    BSMAncillaObservationScenario,
    HiddenRouteLearnerParticleFilter,
    bsm_probe_actions_for_case,
    bsm_probe_cache_plan,
    run_bsm_ancilla_hidden_route_study,
)

__all__ = [
    "HistogramCoLearningConfig",
    "HistogramCoLearningSummary",
    "HistogramGameCase",
    "HistogramCasePathPolytope",
    "HistogramDAGComBand",
    "HistogramDAGPathExp2",
    "HistogramItoPathFTRL",
    "HistogramMaitiPathFTRL",
    "ModernPathBenchmarkSpec",
    "PlayerLearner",
    "ScalarFeedbackLearner",
    "BehaviorConditionableScalarFeedbackLearner",
    "benchmark_specs_from_calibration",
    "build_histogram_alice_learner",
    "build_histogram_eve_learner",
    "run_modern_path_benchmark",
    "HistogramPhysicsPool",
    "RoutePhysicsCalibration",
    "EVE_FEATURE_ACTION_IDENTITY",
    "EVE_FEATURE_FAMILY",
    "EVE_FEATURE_FAMILY_BY_TARGET_TOPOLOGY",
    "EVE_FEATURE_FAMILY_TARGET_TOPOLOGY",
    "EVE_FEATURE_MODES",
    "EveActionEmbedding",
    "RouteResourceEmbedding",
    "analyze_payoff_structure",
    "two_sided_crossplay_validation",
    "build_histogram_case_path_polytope",
    "alice_learner_role",
    "eve_learner_role",
    "eve_action_embedding",
    "eve_reward_embedding",
    "resolve_eve_structured_exploration",
    "resolve_structured_exploration",
    "histogram_cache_plan",
    "load_topology_bench_case",
    "run_histogram_colearning",
    "route_resource_embedding",
    "validate_eve_action_contract",
    "TopologyCensusConfig",
    "census_topology_bench",
    "freeze_case_partitions",
    "FrozenCalibrationCase",
    "HistogramPayoffBank",
    "LearnerCalibrationCandidate",
    "PayoffSampleBlock",
    "Step7CalibrationPlan",
    "build_histogram_payoff_bank",
    "declared_calibration_candidates",
    "load_frozen_calibration_cases",
    "load_frozen_calibration_specs",
    "load_frozen_evaluation_cases",
    "load_frozen_evaluation_specs",
    "load_frozen_partition_cases",
    "load_frozen_partition_specs",
    "run_bank_two_player_learning",
    "run_step7_calibration",
    "step7_preflight",
    "write_step7_calibration_artifacts",
    "PrimeProbeDefender",
    "PrimeProbePlan",
    "PrimeProbeSchedule",
    "WITHDRAWAL_STATE_STRATEGIES",
    "build_prime_probe_manifest",
    "build_prime_probe_plan",
    "build_prime_probe_plans",
    "build_withdrawal_state_manifest",
    "paired_prime_probe_comparison",
    "prime_probe_target_family_representatives",
    "run_ordered_prime_probe",
    "run_prime_probe_withdrawal_state_study",
    "run_prime_probe_study",
    "run_shuffled_prime_probe_control",
    "run_withdrawal_state_corpus_study",
    "withdrawal_state_preflight",
    "write_prime_probe_artifacts",
    "RouteActionRewardEstimator",
    "StackelbergCommitment",
    "StackelbergCommitmentEvaluation",
    "StackelbergExploreCommitConfig",
    "StackelbergOracleResult",
    "evaluate_stackelberg_commitment",
    "eve_public_payoff_matrix",
    "run_stackelberg_explore_then_commit",
    "simultaneous_hoeffding_radii",
    "solve_stackelberg_oracle",
    "StackelbergDefender",
    "build_stackelberg_eve_manifest",
    "default_stackelberg_learner_config",
    "run_stackelberg_eve_study",
    "write_iris_outlier_artifacts",
    "StackelbergSteeringConfig",
    "feasible_stackelberg_response_margins",
    "run_pessimistic_stackelberg_steering",
    "build_stackelberg_steering_manifest",
    "run_stackelberg_steering_study",
    "select_stackelberg_response_margin",
    "analyze_stackelberg_steering_results",
    "write_stackelberg_steering_artifacts",
    "LearnerConditionedPlanLibrary",
    "LearnerHypothesisTracker",
    "RouteObservationDesign",
    "StackelbergBanditFollowRuntime",
    "evaluate_learner_conditioned_prime_probe_plans",
    "minimum_route_observation_design",
    "run_stackelberg_bandit_follow_identification",
    "run_stackelberg_bandit_follow_runtime",
    "ControllerPrimeExploitSchedule",
    "evaluate_post_follow_controller_plans",
    "execute_post_follow_controller_decision",
    "run_controller_prime_exploit_continuation",
    "ControllerPrimeExploitPlan",
    "build_controller_prime_exploit_plans",
    "controller_plan_family_census",
    "controller_plan_screening_representatives",
    "controller_target_family_representatives",
    "ControllerMatchedStudyConfig",
    "run_matched_controller_study",
    "BinomialEvidenceDetectorConfig",
    "MatchedClassicalQuantumEndpointDetectors",
    "QuantumEndpointEvidence",
    "RouteConditionedBinomialEProcess",
    "EndpointEvidenceSampleBlock",
    "HistogramEndpointEvidenceBank",
    "build_histogram_endpoint_evidence_bank",
    "validate_endpoint_evidence_bank_against_case",
    "MATCHED_PAYOFF_ENDPOINT_BANK_SCHEMA",
    "build_matched_histogram_payoff_and_endpoint_banks",
    "PAIRED_BLOCK_DIAGNOSTIC_SIZES",
    "PAIRED_BLOCK_PRIMARY_SIZES",
    "PairedBlockSizeBanks",
    "derive_paired_block_size_banks",
    "EndpointDetectionStudyConfig",
    "run_matched_endpoint_detection_study",
    "run_matched_endpoint_detector_continuation",
    "AlarmResponseConfig",
    "CommissionedInterventionPosterior",
    "CommissionedRobustRoutePolicy",
    "PosteriorTrustRegionRoutePolicy",
    "RESPONSE_POSTERIOR_TRUST_REGION",
    "RESPONSE_SEQUENTIAL_POSTERIOR_TRUST_REGION",
    "RESPONSE_UPDATE_NATIVE_ONLY",
    "RESPONSE_UPDATE_TSALLIS_BEHAVIOR_IW",
    "SequentialInterventionPosterior",
    "commissioned_intervention_posterior",
    "commissioned_detector_compatible_actions",
    "discover_alarm_response_record",
    "run_alarm_conditioned_response_continuation",
    "solve_commissioned_robust_route_policy",
    "solve_posterior_trust_region_route_policy",
    "EndpointResponseStudyConfig",
    "run_endpoint_response_study_from_detection",
    "SequentialResponseCalibrationConfig",
    "SequentialResponseCandidate",
    "run_sequential_response_calibration_from_detection",
    "sequential_response_config",
    "RESPONSE_BLOCK_FREEZE_SCHEMA",
    "RESPONSE_BLOCK_MILESTONE_SCHEMA",
    "TsallisResponseBlockMilestoneConfig",
    "freeze_digest",
    "response_result_row",
    "select_feedback_response_profiles",
    "validate_freeze_manifest",
    "BSMAncillaFollowerSmokeConfig",
    "run_bsm_ancilla_follower_smoke",
    "BSMAncillaHiddenRouteConfig",
    "BSMAncillaObservationScenario",
    "HiddenRouteLearnerParticleFilter",
    "bsm_probe_actions_for_case",
    "bsm_probe_cache_plan",
    "run_bsm_ancilla_hidden_route_study",
]
