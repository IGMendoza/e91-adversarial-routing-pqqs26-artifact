"""Device adapters: resolved PhysicalModel -> configured SeQUeNCe hardware.

Each adapter turns one *resolved* (all values present), scope-labelled
``PhysicalModel`` into a SeQUeNCe component, mapping registry parameter names
1:1 onto the verified constructor/attribute names of ``sequence.components``.
No parameter value is invented here: the adapter fails closed (via
``require_resolved``) if the model still carries TODO/None values, so an
unresolved literature stub can never silently flow into a build.

Implemented (used by the E91/BBM92 control):

- ``create_source``  -> ``SPDCSource`` (polarization encoding, two output arms).
- ``create_detector``-> ``QSDetectorPolarization`` (two ``Detector``s configured).
- ``create_fiber``   -> ``QuantumChannel`` (loss/latency/polarization fidelity).

Still fail-closed (not used by the bare-fiber E91 design; reserved for a future
repeater extension): ``create_memory`` and ``create_swap_bsm``. For swap/BSM the
registry ``success_probability`` documents the chosen linear-optical baseline but
does not map onto a single SeQUeNCe BSM constructor knob.

This module and the builders are the only places allowed to import ``sequence``.
"""

from __future__ import annotations

from typing import Any, NoReturn

from sequence.components.detector import QSDetectorPolarization
from sequence.components.light_source import SPDCSource
from sequence.components.optical_channel import QuantumChannel
from sequence.kernel.timeline import Timeline
from sequence.utils.encoding import polarization

from ..physical.registry import PhysicalModel

#: Constructor/attribute parameters each adapter reads, verified against the
#: SeQUeNCe source at v0.8.5. Extra keys in a model are ignored; missing keys
#: raise AdapterError so no silent default is used.
_SOURCE_PARAMS = ("frequency", "wavelength", "bandwidth", "mean_photon_num", "phase_error")
_DETECTOR_PARAMS = ("efficiency", "dark_count", "count_rate", "time_resolution")
_FIBER_PARAMS = ("attenuation", "polarization_fidelity", "light_speed", "frequency")

_STUB_REASONS = {
    "memory": ("memory adapter (prompt 14): the bare-fiber E91 design uses no quantum "
               "memories; reserved for a future repeater extension"),
    "swap_bsm": ("swap/BSM adapter (prompt 15): unused by the bare-fiber E91 design, and "
                 "the mapping of registry 'success_probability' onto SeQUeNCe swap "
                 "protocols is an undecided TODO(scientific)"),
}


class AdapterError(ValueError):
    """A PhysicalModel cannot be mapped onto the requested SeQUeNCe component."""


def _resolved_params(model: PhysicalModel, kind: str, needed: tuple[str, ...]) -> dict[str, Any]:
    if model.device_kind != kind:
        raise AdapterError(
            f"expected a {kind!r} model, got device_kind {model.device_kind!r} "
            f"(model {model.model_name!r})")
    model.require_resolved()  # fail closed on TODO/None values or missing reference
    missing = [p for p in needed if p not in model.parameters]
    if missing:
        raise AdapterError(
            f"model {model.model_name!r} (kind {kind}) is missing required parameters "
            f"{missing}; present: {sorted(model.parameters)}")
    return {p: model.parameters[p] for p in needed}


def create_source(model: PhysicalModel, name: str, timeline: Timeline) -> SPDCSource:
    """Build a polarization SPDC entangled-pair source from a 'source' model.

    The registry's single ``wavelength`` maps to the two-arm ``wavelengths`` field
    as ``[wavelength, wavelength]`` (degenerate SPDC); ``encoding_type`` is pinned
    to ``polarization`` so ``SPDCSource.emit`` produces a polarization Bell pair
    (see the polarization branch of ``light_source.SPDCSource.emit``).
    """
    p = _resolved_params(model, "source", _SOURCE_PARAMS)
    wavelength = float(p["wavelength"])
    return SPDCSource(
        name, timeline,
        wavelengths=[wavelength, wavelength],
        frequency=float(p["frequency"]),
        mean_photon_num=float(p["mean_photon_num"]),
        encoding_type=polarization,
        phase_error=float(p["phase_error"]),
        bandwidth=float(p["bandwidth"]),
    )


def create_detector(model: PhysicalModel, name: str,
                    timeline: Timeline) -> QSDetectorPolarization:
    """Build a polarization-measuring detector (PBS + two SPDs) from a 'detector'
    model. Both internal ``Detector``s are configured identically from the model."""
    p = _resolved_params(model, "detector", _DETECTOR_PARAMS)
    qsd = QSDetectorPolarization(name, timeline)
    for idx in range(len(qsd.detectors)):
        qsd.set_detector(
            idx,
            efficiency=float(p["efficiency"]),
            dark_count=float(p["dark_count"]),
            count_rate=float(p["count_rate"]),
            time_resolution=int(p["time_resolution"]),
        )
    return qsd


def create_fiber(model: PhysicalModel, name: str, timeline: Timeline,
                 distance_m: float, *,
                 polarization_fidelity: float | None = None) -> QuantumChannel:
    """Build a fiber ``QuantumChannel`` from a 'fiber_channel' model. ``distance_m``
    comes from the topology edge (loss/delay are derived from it in ``init``).

    ``polarization_fidelity`` overrides the model's value when given. The E91
    builder passes 1.0 for fibers carrying the *entangled* arm: SeQUeNCe applies
    polarization depolarization via ``FreeQuantumState.random_noise``, whose own
    source notes ``TODO: rewrite for entangled states`` -- it sets one qubit of a
    shared pair to a single-qubit state, corrupting the joint state. Disabling
    that one unsupported noise source (fidelity 1.0) keeps loss/delay intact and
    avoids either a crash or an invented entanglement-aware noise model.
    """
    if distance_m < 0:
        raise AdapterError(f"fiber distance_m must be >= 0, got {distance_m}")
    p = _resolved_params(model, "fiber_channel", _FIBER_PARAMS)
    pol_fid = (float(p["polarization_fidelity"]) if polarization_fidelity is None
               else float(polarization_fidelity))
    return QuantumChannel(
        name, timeline,
        attenuation=float(p["attenuation"]),
        distance=float(distance_m),
        polarization_fidelity=pol_fid,
        light_speed=float(p["light_speed"]),
        frequency=float(p["frequency"]),
    )


def _fail(kind: str, model: PhysicalModel) -> NoReturn:
    raise NotImplementedError(
        f"cannot build {kind!r} hardware from model {model.model_name!r} "
        f"(scope={model.scope}): {_STUB_REASONS[kind]}")


def create_memory(model: PhysicalModel, *args: Any, **kwargs: Any) -> NoReturn:
    _fail("memory", model)


def create_swap_bsm(model: PhysicalModel, *args: Any, **kwargs: Any) -> NoReturn:
    _fail("swap_bsm", model)
