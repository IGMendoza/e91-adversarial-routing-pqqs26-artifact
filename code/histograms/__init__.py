"""Device-aware E91 transcript histogram generation.

The package exposes small, task-oriented modules while retaining one shared
numerical engine.  Keeping a single engine module is intentional: compiled
kernel caches and NumPy hot paths must not be duplicated across components.
"""

from .compiled import CompiledRouteKernel, verify_compiled_kernel
from .generation import (
    FrozenHistogramKernel,
    TranscriptDistribution,
    evaluate_case_histogram,
    generate_histogram_case,
    sample_transcript_counts,
)
from .models import EveMove, HistogramAttack, ModelContext, ProtocolConfig

__all__ = [
    "CompiledRouteKernel",
    "EveMove",
    "FrozenHistogramKernel",
    "HistogramAttack",
    "ModelContext",
    "ProtocolConfig",
    "TranscriptDistribution",
    "evaluate_case_histogram",
    "generate_histogram_case",
    "sample_transcript_counts",
    "verify_compiled_kernel",
]
