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
from .public_record import (
    PUBLIC_STEP_FIELDS,
    PublicStepRecord,
    PublicStepRecordError,
    eve_public_state_from_step,
    make_control_step_record,
)
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
    # CHSH/legacy-SeQUeNCe-trial and repeater-trial submodules are imported lazily
    # here (rather than at package-import time) because they transitively import
    # real `sequence.*` (SeQUeNCe) modules via ..sequence_build. Code that only
    # needs the pure-Python parts of sequence_game.protocol (postprocessing,
    # toy_trial, public_record, transcript) must be able to `import sequence_game`
    # and its submodules without SeQUeNCe installed at all.
    if name in {"RepeaterE91RunConfig", "run_fixed_repeater_e91_trial"}:
        from .repeater_trial import RepeaterE91RunConfig, run_fixed_repeater_e91_trial

        values = {
            "RepeaterE91RunConfig": RepeaterE91RunConfig,
            "run_fixed_repeater_e91_trial": run_fixed_repeater_e91_trial,
        }
        return values[name]
    if name in {"CHSHResult", "run_chsh_trial"}:
        from .chsh_trial import CHSHResult, run_chsh_trial

        values = {"CHSHResult": CHSHResult, "run_chsh_trial": run_chsh_trial}
        return values[name]
    if name in {"E91RunConfig", "run_e91_trial"}:
        from .sequence_trial import E91RunConfig, run_e91_trial

        values = {"E91RunConfig": E91RunConfig, "run_e91_trial": run_e91_trial}
        return values[name]
    raise AttributeError(name)
