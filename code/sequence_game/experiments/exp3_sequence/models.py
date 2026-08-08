"""Physical-model adjustments for the Exp3 SeQUeNCe experiment."""

from __future__ import annotations

from sequence_game.physical import PhysicalModel

from .config import Exp3SequenceConfig


def apply_sequence_memory_fidelity_override(
        memory_model: PhysicalModel,
        config: Exp3SequenceConfig,
) -> PhysicalModel:
    """Apply the Exp3 runtime memory-fidelity override, if configured.

    The source Cui model remains unchanged on disk. This wrapper records the
    original value and replaces the active SeQUeNCe constructor fidelity so the
    repeater runtime does not compound the Cui benchmark as per-link Bell-state
    preparation noise.
    """

    override = config.sequence_memory_fidelity_override
    if override is None:
        return memory_model
    original = float(memory_model.parameters["fidelity"])
    active = float(override)
    if original == active:
        return memory_model

    parameters = dict(memory_model.parameters)
    parameters["sequence_runtime_original_fidelity"] = original
    parameters["sequence_runtime_fidelity_override"] = active
    parameters["fidelity"] = active
    units = dict(memory_model.units)
    units["sequence_runtime_original_fidelity"] = "dimensionless in [0, 1]"
    units["sequence_runtime_fidelity_override"] = "dimensionless in [0, 1]"
    return PhysicalModel(
        model_name=f"{memory_model.model_name}_sequence_fidelity_{active:g}",
        device_kind=memory_model.device_kind,
        scope=memory_model.scope,
        reference_tag=(
            f"{memory_model.reference_tag}; "
            f"Exp3 SeQUeNCe runtime override fidelity={active:g}"
        ),
        parameters=parameters,
        units=units,
        notes=(
            f"{memory_model.notes} Exp3 SeQUeNCe runtime override: active "
            f"Memory.fidelity is set to {active:g}; source-model fidelity "
            f"{original:g} is retained as sequence_runtime_original_fidelity. "
            "Reason: SeQUeNCe consumes Memory.fidelity as per-elementary-link "
            "Bell-state preparation fidelity and compounds it through swapping."
        ),
    )
