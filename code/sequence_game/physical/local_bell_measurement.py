"""Resource-complete local trapped-ion Bell-measurement descriptions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


class LocalBellMeasurementError(ValueError):
    """A local Bell-measurement component is incomplete or unphysical."""


BELL_LABELS = ("phi_plus", "phi_minus", "psi_plus", "psi_minus")
PAULI_LABELS = ("I", "X", "Z", "Y")
_CORRECTION = {
    "phi_plus": "Y",
    "phi_minus": "X",
    "psi_plus": "Z",
    "psi_minus": "I",
}
_FRAME_BITS = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}
_BITS_FRAME = {bits: label for label, bits in _FRAME_BITS.items()}
_BELL_BITS = {
    "phi_plus": (0, 0),
    "phi_minus": (0, 1),
    "psi_plus": (1, 0),
    "psi_minus": (1, 1),
}


def _probability(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise LocalBellMeasurementError(f"{name} must be finite and in [0, 1]")
    return result


def _positive(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise LocalBellMeasurementError(f"{name} must be finite and > 0")
    return result


def _compose_frame(left: str, right: str) -> str:
    lx, lz = _FRAME_BITS[left]
    rx, rz = _FRAME_BITS[right]
    return _BITS_FRAME[(lx ^ rx, lz ^ rz)]


@dataclass(frozen=True)
class LocalBellMeasurementModel:
    """Deterministic two-ion Bell analysis with explicit label confusion.

    The source Bell-preparation fidelity is used as a conservative reciprocal
    Bell-transform classification envelope.  It is not silently relabelled as
    average gate fidelity.  Independent fluorescence assignment errors are
    then composed on the two measured Bell-syndrome bits.
    """

    transform_classification_fidelity: float
    transform_duration_s: float
    readout_fidelity: float
    readout_duration_s: float
    source: tuple[str, ...]
    compatibility: str
    mapping_semantics: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LocalBellMeasurementModel":
        data = dict(value)
        if str(data.get("kind", "")) != "deterministic_two_ion_bell_measurement":
            raise LocalBellMeasurementError(
                "local_swap_model.kind must be deterministic_two_ion_bell_measurement"
            )
        transform = data.get("bell_basis_transform_model")
        readout = data.get("readout_model")
        if not isinstance(transform, Mapping) or not isinstance(readout, Mapping):
            raise LocalBellMeasurementError(
                "local Bell measurement requires transform and readout model objects"
            )
        if str(transform.get("kind", "")) != "reciprocal_bell_preparation_envelope":
            raise LocalBellMeasurementError(
                "bell transform must declare reciprocal_bell_preparation_envelope semantics"
            )
        if str(readout.get("kind", "")) != "independent_symmetric_bit_confusion":
            raise LocalBellMeasurementError(
                "Bell readout must use independent_symmetric_bit_confusion"
            )
        sources = tuple(str(item) for item in data.get("component_sources", ()))
        if not sources:
            raise LocalBellMeasurementError("local Bell measurement needs component_sources")
        compatibility = str(data.get("compatibility", "")).strip()
        mapping = str(transform.get("mapping_semantics", "")).strip()
        if not compatibility or not mapping:
            raise LocalBellMeasurementError(
                "local Bell measurement needs compatibility and mapping semantics"
            )
        return cls(
            transform_classification_fidelity=_probability(
                "bell transform classification fidelity", transform.get("fidelity")
            ),
            transform_duration_s=_positive(
                "bell transform duration", transform.get("duration_s")
            ),
            readout_fidelity=_probability(
                "Bell readout fidelity", readout.get("readout_fidelity")
            ),
            readout_duration_s=_positive(
                "Bell readout duration", readout.get("duration_s")
            ),
            source=sources,
            compatibility=compatibility,
            mapping_semantics=mapping,
        )

    @property
    def duration_s(self) -> float:
        return self.transform_duration_s + self.readout_duration_s

    @property
    def confusion_matrix(self) -> tuple[tuple[float, ...], ...]:
        transform_error = 1.0 - self.transform_classification_fidelity
        transform = [
            [
                self.transform_classification_fidelity
                if actual == intermediate
                else transform_error / 3.0
                for intermediate in BELL_LABELS
            ]
            for actual in BELL_LABELS
        ]
        readout_error = 1.0 - self.readout_fidelity
        readout = []
        for intermediate in BELL_LABELS:
            source_bits = _BELL_BITS[intermediate]
            row = []
            for reported in BELL_LABELS:
                target_bits = _BELL_BITS[reported]
                probability = 1.0
                for source_bit, target_bit in zip(source_bits, target_bits):
                    probability *= (
                        1.0 - readout_error
                        if source_bit == target_bit
                        else readout_error
                    )
                row.append(probability)
            readout.append(row)
        composed = []
        for actual_index in range(4):
            row = []
            for reported_index in range(4):
                row.append(sum(
                    transform[actual_index][middle] * readout[middle][reported_index]
                    for middle in range(4)
                ))
            composed.append(tuple(row))
        return tuple(composed)

    @property
    def frame_error_distribution(self) -> dict[str, float]:
        """Average Pauli-frame error from actual-to-reported Bell labels."""

        out = {label: 0.0 for label in PAULI_LABELS}
        matrix = self.confusion_matrix
        for actual_index, actual in enumerate(BELL_LABELS):
            for reported_index, reported in enumerate(BELL_LABELS):
                relative = _compose_frame(_CORRECTION[actual], _CORRECTION[reported])
                out[relative] += 0.25 * matrix[actual_index][reported_index]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "deterministic_two_ion_bell_measurement",
            "success_probability": 1.0,
            "duration_s": self.duration_s,
            "transform_classification_fidelity": self.transform_classification_fidelity,
            "transform_duration_s": self.transform_duration_s,
            "readout_fidelity": self.readout_fidelity,
            "readout_duration_s": self.readout_duration_s,
            "bell_label_order": list(BELL_LABELS),
            "confusion_matrix": [list(row) for row in self.confusion_matrix],
            "frame_error_distribution": self.frame_error_distribution,
            "component_sources": list(self.source),
            "compatibility": self.compatibility,
            "mapping_semantics": self.mapping_semantics,
        }
