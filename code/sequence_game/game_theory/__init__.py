"""Game-theoretic solvers for the routing/interdiction game.

For complement rewards the Alice(routing)-vs-Eve(interdiction) interaction is
a zero-sum covering game. Declared Eve-only action costs admit the equivalent
Alice matrix ``A + cost_E`` because that column offset leaves Alice's best
responses unchanged. This package provides adversarial *co-learning* solvers
(no baked-in equilibrium) plus an exact minimax LP used only as a validation
oracle.
"""

from .bandit import (
    ConvergenceTrajectory,
    Exp3,
    Exp3SchedulePoint,
    LinearExplorationDesign,
    LinearExp2,
    bandit_selfplay_trajectory,
    linear_exploration_design,
    solve_zero_sum_bandit_selfplay,
)
from .adaptive_bandit import (
    Exp3PlusPlus,
    Exp3PlusPlusSchedulePoint,
    TSALLIS_INF_ESTIMATOR_IW,
    TSALLIS_INF_ESTIMATOR_RV,
    TSALLIS_INF_ESTIMATORS,
    TsallisINF,
    TsallisINFSchedulePoint,
)
from .path_bandit import (
    BoundedDAGPathPolytope,
    ComBandSchedule,
    DAGComBand,
    DAGPathExp2,
    DirectedPathArc,
    PathDistributionMoments,
)
from .dag_flow import (
    BoundedLayeredDAGTransform,
    CentroidDAGTransform,
    DAGUnitFlow,
    FlowFTRLSolveStats,
    FlowRegularizer,
    ReducedFlowFTRLSolver,
    build_bounded_layered_dag_transform,
    build_centroid_dag_transform,
)
from .aggregate_path_bandit import (
    AggregatePathSchedulePoint,
    ITO_REGULARIZER_LOG_BARRIER,
    ITO_REGULARIZER_TSALLIS,
    ITO_REGULARIZERS,
    ItoDAGPathFTRL,
    MaitiDAGPathFTRL,
)
from .zero_sum import (
    GameCharacterization,
    RegretMatchingPlus,
    characterize_equilibrium,
    exploitability,
    solve_zero_sum_lp,
    solve_zero_sum_selfplay,
)

__all__ = [
    "BoundedDAGPathPolytope",
    "BoundedLayeredDAGTransform",
    "ComBandSchedule",
    "CentroidDAGTransform",
    "ConvergenceTrajectory",
    "DAGComBand",
    "DAGUnitFlow",
    "DAGPathExp2",
    "DirectedPathArc",
    "FlowFTRLSolveStats",
    "FlowRegularizer",
    "Exp3",
    "Exp3PlusPlus",
    "Exp3PlusPlusSchedulePoint",
    "Exp3SchedulePoint",
    "LinearExplorationDesign",
    "LinearExp2",
    "AggregatePathSchedulePoint",
    "ITO_REGULARIZER_LOG_BARRIER",
    "ITO_REGULARIZER_TSALLIS",
    "ITO_REGULARIZERS",
    "ItoDAGPathFTRL",
    "MaitiDAGPathFTRL",
    "PathDistributionMoments",
    "ReducedFlowFTRLSolver",
    "GameCharacterization",
    "RegretMatchingPlus",
    "TSALLIS_INF_ESTIMATOR_IW",
    "TSALLIS_INF_ESTIMATOR_RV",
    "TSALLIS_INF_ESTIMATORS",
    "TsallisINF",
    "TsallisINFSchedulePoint",
    "bandit_selfplay_trajectory",
    "build_bounded_layered_dag_transform",
    "build_centroid_dag_transform",
    "characterize_equilibrium",
    "exploitability",
    "linear_exploration_design",
    "solve_zero_sum_bandit_selfplay",
    "solve_zero_sum_lp",
    "solve_zero_sum_selfplay",
]
