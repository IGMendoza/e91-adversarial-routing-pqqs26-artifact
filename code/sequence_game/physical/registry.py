"""Physical model registry: schema and validation only, no parameter values.

This module defines the *structure* used to carry physical hardware parameters
through the stack. It deliberately contains no physical values, equations, or
device behaviour. Parameter names in configs are expected to mirror the
constructor arguments of the corresponding ``sequence.components`` classes so
that adapters can map them without renaming, but nothing here claims those
parameters are correct or complete for any reference paper.

Scope labels (``scope`` field):

- ``literature``: values must be user-supplied from an explicit reference;
  unresolved TODO values fail closed at validation time.
- ``toy``: explicitly non-physical values used only to exercise wiring.
- ``placeholder``: structure present, values pending a scope decision.
- ``empirical_fit``: values fitted to user-supplied data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

SCOPES = ("literature", "toy", "placeholder", "empirical_fit")

DEVICE_KINDS = ("source", "fiber_channel", "memory", "swap_bsm", "detector")

ParamValue = Union[int, float, str, None]


class PhysicalModelError(ValueError):
    """Invalid physical-model structure or registry contents."""


class UnresolvedParameterError(PhysicalModelError):
    """A model with unresolved (TODO/None) parameters was used where resolved
    values are required."""


def _is_unresolved(value: ParamValue) -> bool:
    return value is None or (isinstance(value, str) and value.upper().startswith("TODO"))


@dataclass(frozen=True)
class PhysicalModel:
    """One named parameter set for one device kind, with explicit scope."""

    model_name: str
    device_kind: str
    scope: str
    reference_tag: str
    parameters: dict[str, ParamValue]
    units: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.model_name:
            raise PhysicalModelError("model_name must be non-empty")
        if self.device_kind not in DEVICE_KINDS:
            raise PhysicalModelError(
                f"unknown device_kind {self.device_kind!r}; expected one of {DEVICE_KINDS}")
        if self.scope not in SCOPES:
            raise PhysicalModelError(
                f"unknown scope {self.scope!r}; expected one of {SCOPES}")
        unknown_units = set(self.units) - set(self.parameters)
        if unknown_units:
            raise PhysicalModelError(
                f"units given for unknown parameters: {sorted(unknown_units)}")

    def unresolved_parameters(self) -> list[str]:
        return sorted(k for k, v in self.parameters.items() if _is_unresolved(v))

    @property
    def is_resolved(self) -> bool:
        if _is_unresolved(self.reference_tag) and self.scope == "literature":
            return False
        return not self.unresolved_parameters()

    def require_resolved(self) -> None:
        problems = []
        unresolved = self.unresolved_parameters()
        if unresolved:
            problems.append(f"unresolved parameters {unresolved}")
        if self.scope == "literature" and _is_unresolved(self.reference_tag):
            problems.append("reference_tag is a TODO placeholder")
        if problems:
            raise UnresolvedParameterError(
                f"model {self.model_name!r} (scope={self.scope}): " + "; ".join(problems))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "device_kind": self.device_kind,
            "scope": self.scope,
            "reference_tag": self.reference_tag,
            "parameters": dict(self.parameters),
            "units": dict(self.units),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicalModel":
        required = {"model_name", "device_kind", "scope", "reference_tag", "parameters"}
        missing = required - set(data)
        if missing:
            raise PhysicalModelError(f"missing required fields: {sorted(missing)}")
        return cls(
            model_name=data["model_name"],
            device_kind=data["device_kind"],
            scope=data["scope"],
            reference_tag=data["reference_tag"],
            parameters=dict(data["parameters"]),
            units=dict(data.get("units", {})),
            notes=data.get("notes", ""),
        )


@dataclass(frozen=True)
class NodeHardwareProfile:
    """Names of the physical models attached to one node role/type."""

    profile_id: str
    source_model: Optional[str] = None
    memory_model: Optional[str] = None
    swap_bsm_model: Optional[str] = None
    detector_model: Optional[str] = None
    notes: str = ""

    def model_refs(self) -> dict[str, str]:
        refs = {
            "source": self.source_model,
            "memory": self.memory_model,
            "swap_bsm": self.swap_bsm_model,
            "detector": self.detector_model,
        }
        return {kind: name for kind, name in refs.items() if name is not None}

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "source_model": self.source_model,
            "memory_model": self.memory_model,
            "swap_bsm_model": self.swap_bsm_model,
            "detector_model": self.detector_model,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeHardwareProfile":
        return cls(**data)


@dataclass(frozen=True)
class EdgeHardwareProfile:
    """Name of the channel model attached to one edge type."""

    profile_id: str
    channel_model: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "channel_model": self.channel_model,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EdgeHardwareProfile":
        return cls(**data)


@dataclass
class PhysicalRegistry:
    """All physical models and hardware profiles for one experiment."""

    models: dict[str, PhysicalModel] = field(default_factory=dict)
    node_profiles: dict[str, NodeHardwareProfile] = field(default_factory=dict)
    edge_profiles: dict[str, EdgeHardwareProfile] = field(default_factory=dict)

    def add_model(self, model: PhysicalModel) -> None:
        if model.model_name in self.models:
            raise PhysicalModelError(f"duplicate model name {model.model_name!r}")
        self.models[model.model_name] = model

    def get_model(self, name: str) -> PhysicalModel:
        try:
            return self.models[name]
        except KeyError:
            raise PhysicalModelError(
                f"unknown physical model {name!r}; known: {sorted(self.models)}") from None

    def add_node_profile(self, profile: NodeHardwareProfile) -> None:
        if profile.profile_id in self.node_profiles:
            raise PhysicalModelError(f"duplicate node profile {profile.profile_id!r}")
        self.node_profiles[profile.profile_id] = profile

    def add_edge_profile(self, profile: EdgeHardwareProfile) -> None:
        if profile.profile_id in self.edge_profiles:
            raise PhysicalModelError(f"duplicate edge profile {profile.profile_id!r}")
        self.edge_profiles[profile.profile_id] = profile

    def validate_structure(self) -> None:
        """Check that every profile reference resolves to a registered model of
        the matching device kind."""
        for profile in self.node_profiles.values():
            for kind, name in profile.model_refs().items():
                model = self.get_model(name)
                if model.device_kind != kind:
                    raise PhysicalModelError(
                        f"node profile {profile.profile_id!r} expects a {kind!r} model "
                        f"but {name!r} has device_kind {model.device_kind!r}")
        for eprofile in self.edge_profiles.values():
            if not eprofile.channel_model:
                raise PhysicalModelError(
                    f"edge profile {eprofile.profile_id!r} has no channel_model")
            model = self.get_model(eprofile.channel_model)
            if model.device_kind != "fiber_channel":
                raise PhysicalModelError(
                    f"edge profile {eprofile.profile_id!r} expects a fiber_channel model "
                    f"but {eprofile.channel_model!r} has device_kind {model.device_kind!r}")

    def validate_for_run(self) -> None:
        """Fail closed before any non-structural use: structure must be valid
        and every registered model must have fully resolved values."""
        self.validate_structure()
        for model in self.models.values():
            model.require_resolved()

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": {name: m.to_dict() for name, m in self.models.items()},
            "node_profiles": {pid: p.to_dict() for pid, p in self.node_profiles.items()},
            "edge_profiles": {pid: p.to_dict() for pid, p in self.edge_profiles.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PhysicalRegistry":
        registry = cls()
        for mdata in data.get("models", {}).values():
            registry.add_model(PhysicalModel.from_dict(mdata))
        for pdata in data.get("node_profiles", {}).values():
            registry.add_node_profile(NodeHardwareProfile.from_dict(pdata))
        for pdata in data.get("edge_profiles", {}).values():
            registry.add_edge_profile(EdgeHardwareProfile.from_dict(pdata))
        return registry
