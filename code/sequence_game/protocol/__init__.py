from .postprocessing import (
    AcceptDecision,
    PostprocessingError,
    SiftingConfig,
    apply_postprocessing,
    decide_accept,
    estimate_qber,
    extract_sifted_bits,
    sift_indices,
)
from .chsh_trial import CHSHResult, run_chsh_trial
from .public_record import (
    PUBLIC_STEP_FIELDS,
    PublicStepRecord,
    PublicStepRecordError,
    eve_public_state_from_step,
    make_control_step_record,
)
from .sequence_trial import E91RunConfig, run_e91_trial
from .toy_trial import DISRUPTED_REASON, ToyTrialConfig, route_is_disrupted, run_toy_trial
from .transcript import PRIVATE_FIELDS, PublicTranscript, TrialTranscript

__all__ = [
    "CHSHResult",
    "DISRUPTED_REASON",
    "E91RunConfig",
    "RepeaterE91RunConfig",
    "ToyTrialConfig",
    "route_is_disrupted",
    "run_chsh_trial",
    "run_e91_trial",
    "run_fixed_repeater_e91_trial",
    "run_toy_trial",
    "AcceptDecision",
    "PRIVATE_FIELDS",
    "PostprocessingError",
    "PUBLIC_STEP_FIELDS",
    "PublicTranscript",
    "PublicStepRecord",
    "PublicStepRecordError",
    "SiftingConfig",
    "TrialTranscript",
    "apply_postprocessing",
    "decide_accept",
    "estimate_qber",
    "eve_public_state_from_step",
    "extract_sifted_bits",
    "make_control_step_record",
    "sift_indices",
]


def __getattr__(name):
    if name in {"RepeaterE91RunConfig", "run_fixed_repeater_e91_trial"}:
        from .repeater_trial import RepeaterE91RunConfig, run_fixed_repeater_e91_trial

        values = {
            "RepeaterE91RunConfig": RepeaterE91RunConfig,
            "run_fixed_repeater_e91_trial": run_fixed_repeater_e91_trial,
        }
        return values[name]
    raise AttributeError(name)
