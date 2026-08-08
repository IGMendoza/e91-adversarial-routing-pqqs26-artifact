"""Hardware and topology adapters for the direct E91 transcript backend."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

from sequence_game.physical.loader import load_physical_model, load_models_from_dir
from sequence_game.physical.registry import PhysicalModel, PhysicalRegistry
from sequence_game.topology.ir import EdgeRecord, TopologyIR

from .direct_e91_transcript import (
    DirectE91HardwareConfig,
    DirectE91HardwareProvenance,
    DirectE91Route,
)

_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs" / "physical"
_CONFIG_FIELDS = (
    "fiber_loss_db_per_km",
    "source_pair_success_probability",
    "detector_efficiency",
    "dark_count_probability",
    "misalignment_probability",
    "elementary_pair_fidelity",
    "swap_success_probability",
    "swap_depolarizing_probability",
    "memory_depolarizing_probability",
    "memory_dephasing_probability",
)


def load_direct_e91_hardware_config_from_existing(
    config_dir: str | Path | None = None,
    *,
    require_resolved: bool = True,
) -> DirectE91HardwareConfig:
    config_path = Path(config_dir) if config_dir is not None else _CONFIG_ROOT
    registry = load_models_from_dir(config_path, require_resolved=require_resolved)
    source_paths = sorted(str(path) for path in config_path.glob("*.json"))
    return direct_e91_hardware_config_from_sequence_runtime_config(
        physical_registry=registry,
        source_paths=source_paths,
    )


def direct_e91_hardware_config_from_sequence_runtime_config(
    runtime_config: Any | None = None,
    *,
    physical_registry: PhysicalRegistry | None = None,
    source_paths: Iterable[str] = (),
) -> DirectE91HardwareConfig:
    values = DirectE91HardwareConfig().to_dict()
    values.pop("provenance", None)
    mapped: set[str] = set()
    unmapped: set[str] = set()
    literature: set[str] = set()
    conversion_notes: dict[str, str] = {}

    registry = physical_registry
    if registry is None:
        registry = load_models_from_dir(_CONFIG_ROOT, require_resolved=True)
        source_paths = sorted(str(path) for path in _CONFIG_ROOT.glob("*.json"))

    model_names: list[str] = []
    for model in registry.models.values():
        model_names.append(model.model_name)
        if model.reference_tag:
            literature.add(str(model.reference_tag))
        extracted = _extract_model_parameters(model)
        for name, value in extracted["mapped_values"].items():
            values[name] = value
            mapped.add(name)
        unmapped.update(extracted["unmapped_parameters"])
        conversion_notes.update(extracted["conversion_notes"])

    runtime_values = _runtime_mapping(runtime_config)
    for name, value in runtime_values["mapped_values"].items():
        values[name] = value
        mapped.add(name)
    unmapped.update(runtime_values["unmapped_parameters"])
    conversion_notes.update(runtime_values["conversion_notes"])

    fallback = sorted(set(_CONFIG_FIELDS) - mapped)
    provenance = DirectE91HardwareProvenance(
        hardware_profile_id="+".join(sorted(model_names)) or "direct_e91_defaults",
        hardware_profile_source_paths=[str(path) for path in source_paths],
        mapped_parameter_names=sorted(mapped),
        fallback_default_names=fallback,
        unmapped_parameter_names=sorted(unmapped),
        literature_source_ids=sorted(literature),
        conversion_notes=conversion_notes,
    )
    return DirectE91HardwareConfig(
        **{name: values[name] for name in _CONFIG_FIELDS},
        provenance=provenance,
    )


def direct_e91_route_from_topology_ir(
    topology: TopologyIR | dict[str, Any],
    node_path: Iterable[str],
    *,
    route_id: str | None = None,
    edge_ids: Iterable[str] | None = None,
) -> DirectE91Route:
    path = tuple(str(node) for node in node_path)
    if len(path) < 2:
        raise ValueError("node_path must include Alice and Bob")
    given_edge_ids = tuple(str(edge_id) for edge_id in edge_ids) if edge_ids is not None else None
    if given_edge_ids is not None and len(given_edge_ids) != len(path) - 1:
        raise ValueError("edge_ids must match route hops")
    route_edges: list[str] = []
    lengths: dict[str, float] = {}
    fidelities: dict[str, float] = {}
    transmittances: dict[str, float] = {}
    losses_db: dict[str, float] = {}
    for index, (u, v) in enumerate(zip(path, path[1:])):
        edge = _edge_between(topology, u, v)
        actual_edge_id = _edge_id(edge)
        if given_edge_ids is not None and given_edge_ids[index] != actual_edge_id:
            raise ValueError(
                f"edge_id {given_edge_ids[index]!r} does not match topology edge "
                f"{actual_edge_id!r} for route hop {u!r}->{v!r}"
            )
        edge_id = actual_edge_id
        route_edges.append(edge_id)
        lengths[edge_id] = extract_edge_length_m(edge)
        fidelity = _extract_edge_fidelity(edge)
        if fidelity is not None:
            fidelities[edge_id] = fidelity
        explicit_loss = extract_edge_transmittance_or_loss(edge)
        if _edge_has_field(edge, "transmittance"):
            transmittances[edge_id] = _clip(explicit_loss["transmittance"])
        elif _edge_has_field(edge, "loss_db"):
            losses_db[edge_id] = float(explicit_loss["loss_db"])
            transmittances[edge_id] = _clip(explicit_loss["transmittance"])
    return DirectE91Route(
        route_id=route_id or _route_id_for_path(path),
        node_path=path,
        edge_ids=tuple(route_edges),
        edge_lengths_m=lengths,
        total_length_m=sum(lengths.values()),
        edge_fidelities=fidelities,
        edge_transmittances=transmittances,
        edge_loss_db=losses_db,
    )


def _edge_between(
    topology: TopologyIR | dict[str, Any],
    u: str,
    v: str,
) -> EdgeRecord | dict[str, Any]:
    if isinstance(topology, TopologyIR):
        edge = topology.edge_between(u, v)
        if edge is None:
            raise ValueError(f"no topology edge for route hop {u!r}->{v!r}")
        return edge
    edges = list(topology.get("edges", []))
    matches = [
        edge
        for edge in edges
        if {str(edge.get("u", "")), str(edge.get("v", ""))} == {u, v}
    ]
    if not matches:
        raise ValueError(f"no topology edge for route hop {u!r}->{v!r}")
    if len(matches) > 1:
        raise ValueError(f"ambiguous topology edge for route hop {u!r}->{v!r}")
    return dict(matches[0])


def _edge_id(edge: EdgeRecord | dict[str, Any]) -> str:
    return str(edge.edge_id if isinstance(edge, EdgeRecord) else edge["edge_id"])


def _edge_has_field(edge: EdgeRecord | dict[str, Any], field_name: str) -> bool:
    return not isinstance(edge, EdgeRecord) and field_name in edge and edge[field_name] is not None


def extract_edge_length_m(edge: EdgeRecord | dict[str, Any]) -> float:
    if isinstance(edge, EdgeRecord):
        return float(edge.length_m)
    if "length_m" in edge:
        return float(edge["length_m"])
    if "length" in edge:
        return float(edge["length"])
    raise ValueError("edge has no length_m or length field")


def extract_edge_transmittance_or_loss(
    edge: EdgeRecord | dict[str, Any],
    hardware: DirectE91HardwareConfig | None = None,
) -> dict[str, float]:
    edge_dict = edge.to_dict() if isinstance(edge, EdgeRecord) else dict(edge)
    length_m = extract_edge_length_m(edge)
    if "transmittance" in edge_dict:
        return {"length_m": length_m, "transmittance": float(edge_dict["transmittance"])}
    if "loss_db" in edge_dict:
        loss_db = float(edge_dict["loss_db"])
        return {"length_m": length_m, "loss_db": loss_db, "transmittance": 10 ** (-loss_db / 10.0)}
    if hardware is None:
        return {"length_m": length_m}
    length_km = length_m / 1000.0
    loss_db = hardware.fiber_loss_db_per_km * length_km
    return {"length_m": length_m, "loss_db": loss_db, "transmittance": 10 ** (-loss_db / 10.0)}


def _extract_edge_fidelity(edge: EdgeRecord | dict[str, Any]) -> float | None:
    edge_dict = edge.to_dict() if isinstance(edge, EdgeRecord) else dict(edge)
    for key in (
        "elementary_pair_fidelity",
        "edge_fidelity",
        "fidelity",
        "polarization_fidelity",
    ):
        if key in edge_dict and edge_dict[key] is not None:
            return _clip(float(edge_dict[key]))
    return None


def extract_detector_parameters(model: PhysicalModel) -> dict[str, Any]:
    return _extract_detector(model)


def extract_memory_parameters(model: PhysicalModel) -> dict[str, Any]:
    return _extract_memory(model)


def extract_swap_parameters(model: PhysicalModel) -> dict[str, Any]:
    return _extract_swap(model)


def extract_source_parameters(model: PhysicalModel) -> dict[str, Any]:
    return _extract_source(model)


def _extract_model_parameters(model: PhysicalModel) -> dict[str, Any]:
    if model.device_kind == "fiber_channel":
        return _extract_fiber(model)
    if model.device_kind == "detector":
        return _extract_detector(model)
    if model.device_kind == "memory":
        return _extract_memory(model)
    if model.device_kind == "swap_bsm":
        return _extract_swap(model)
    if model.device_kind == "source":
        return _extract_source(model)
    return _empty_mapping(unmapped=[f"{model.device_kind}.{name}" for name in model.parameters])


def _extract_fiber(model: PhysicalModel) -> dict[str, Any]:
    mapped: dict[str, float] = {}
    used: set[str] = set()
    notes: dict[str, str] = {}
    params = model.parameters
    units = model.units
    if "attenuation" in params and params["attenuation"] is not None:
        value = float(params["attenuation"])
        unit = units.get("attenuation", "")
        if "dB/m" in unit:
            mapped["fiber_loss_db_per_km"] = value * 1000.0
            notes["fiber_loss_db_per_km"] = "attenuation converted from dB/m to dB/km"
        else:
            mapped["fiber_loss_db_per_km"] = value
            notes["fiber_loss_db_per_km"] = "attenuation used as dB/km"
        used.add("attenuation")
    if "polarization_fidelity" in params and params["polarization_fidelity"] is not None:
        mapped["elementary_pair_fidelity"] = _clip(float(params["polarization_fidelity"]))
        notes["elementary_pair_fidelity"] = "fiber polarization_fidelity used as pair fidelity input"
        used.add("polarization_fidelity")
    return _mapping_result(model, mapped, used, notes)


def _extract_detector(model: PhysicalModel) -> dict[str, Any]:
    mapped: dict[str, float] = {}
    used: set[str] = set()
    notes: dict[str, str] = {}
    params = model.parameters
    units = model.units
    if "efficiency" in params and params["efficiency"] is not None:
        mapped["detector_efficiency"] = _clip(float(params["efficiency"]))
        used.add("efficiency")
    if "dark_count" in params and params["dark_count"] is not None:
        unit = units.get("dark_count", "")
        if "1/s" in unit and "time_resolution" in params and params["time_resolution"] is not None:
            window_s = float(params["time_resolution"]) * 1e-12
            mapped["dark_count_probability"] = _clip(float(params["dark_count"]) * window_s)
            used.update({"dark_count", "time_resolution"})
            notes["dark_count_probability"] = (
                "dark_count converted from 1/s using time_resolution in ps"
            )
        elif "probability" in unit:
            mapped["dark_count_probability"] = _clip(float(params["dark_count"]))
            used.add("dark_count")
            notes["dark_count_probability"] = "dark_count used as a probability"
    return _mapping_result(model, mapped, used, notes)


def _extract_memory(model: PhysicalModel) -> dict[str, Any]:
    mapped: dict[str, float] = {}
    used: set[str] = set()
    notes: dict[str, str] = {}
    params = model.parameters
    if "fidelity" in params and params["fidelity"] is not None:
        fidelity = _clip(float(params["fidelity"]))
        mapped["memory_depolarizing_probability"] = _clip(4.0 * (1.0 - fidelity) / 3.0)
        used.add("fidelity")
        notes["memory_depolarizing_probability"] = (
            "memory fidelity converted through one-qubit depolarizing probability"
        )
    if "decoherence_probability" in params and params["decoherence_probability"] is not None:
        mapped["memory_dephasing_probability"] = _clip(float(params["decoherence_probability"]))
        used.add("decoherence_probability")
    return _mapping_result(model, mapped, used, notes)


def _extract_swap(model: PhysicalModel) -> dict[str, Any]:
    mapped: dict[str, float] = {}
    used: set[str] = set()
    notes: dict[str, str] = {}
    params = model.parameters
    if "success_probability" in params and params["success_probability"] is not None:
        mapped["swap_success_probability"] = _clip(float(params["success_probability"]))
        used.add("success_probability")
    if "swapping_degradation" in params and params["swapping_degradation"] is not None:
        mapped["swap_depolarizing_probability"] = _clip(
            1.0 - float(params["swapping_degradation"])
        )
        used.add("swapping_degradation")
        notes["swap_depolarizing_probability"] = (
            "swapping_degradation converted to depolarizing probability"
        )
    return _mapping_result(model, mapped, used, notes)


def _extract_source(model: PhysicalModel) -> dict[str, Any]:
    mapped: dict[str, float] = {}
    used: set[str] = set()
    notes: dict[str, str] = {}
    params = model.parameters
    for key in ("source_pair_success_probability", "pair_success_probability"):
        if key in params and params[key] is not None:
            mapped["source_pair_success_probability"] = _clip(float(params[key]))
            used.add(key)
            break
    if "source_pair_success_probability" not in mapped and "mean_photon_num" in params:
        mu = float(params["mean_photon_num"])
        mapped["source_pair_success_probability"] = _clip(1.0 - math.exp(-max(0.0, mu)))
        used.add("mean_photon_num")
        notes["source_pair_success_probability"] = (
            "mean_photon_num converted to at-least-one-pair probability"
        )
    return _mapping_result(model, mapped, used, notes)


def _runtime_mapping(runtime_config: Any | None) -> dict[str, Any]:
    if runtime_config is None:
        return _empty_mapping()
    if hasattr(runtime_config, "to_dict"):
        data = runtime_config.to_dict()
    elif isinstance(runtime_config, dict):
        data = dict(runtime_config)
    else:
        data = {
            name: getattr(runtime_config, name)
            for name in dir(runtime_config)
            if not name.startswith("_") and not callable(getattr(runtime_config, name))
        }
    mapped: dict[str, float] = {}
    used: set[str] = set()
    notes: dict[str, str] = {}
    direct_names = {
        "fiber_loss_db_per_km",
        "source_pair_success_probability",
        "detector_efficiency",
        "dark_count_probability",
        "misalignment_probability",
        "elementary_pair_fidelity",
        "swap_success_probability",
        "swap_depolarizing_probability",
        "memory_depolarizing_probability",
        "memory_dephasing_probability",
    }
    for name in direct_names:
        if name in data and data[name] is not None:
            mapped[name] = _clip(float(data[name])) if name != "fiber_loss_db_per_km" else max(0.0, float(data[name]))
            used.add(name)
    for src, dst in (
        ("swapping_success_prob", "swap_success_probability"),
        ("swapping_success_probability", "swap_success_probability"),
    ):
        if src in data and data[src] is not None:
            mapped[dst] = _clip(float(data[src]))
            used.add(src)
    if "swapping_degradation" in data and data["swapping_degradation"] is not None:
        mapped["swap_depolarizing_probability"] = _clip(1.0 - float(data["swapping_degradation"]))
        used.add("swapping_degradation")
        notes["swap_depolarizing_probability"] = (
            "runtime swapping_degradation converted to depolarizing probability"
        )
    unmapped = [f"runtime.{name}" for name in sorted(set(data) - used)]
    return {
        "mapped_values": mapped,
        "unmapped_parameters": unmapped,
        "conversion_notes": notes,
    }


def _mapping_result(
    model: PhysicalModel,
    mapped: dict[str, float],
    used: set[str],
    notes: dict[str, str],
) -> dict[str, Any]:
    unmapped = [
        f"{model.device_kind}.{name}"
        for name in sorted(set(model.parameters) - used)
    ]
    return {
        "mapped_values": mapped,
        "unmapped_parameters": unmapped,
        "conversion_notes": notes,
    }


def _empty_mapping(unmapped: Iterable[str] = ()) -> dict[str, Any]:
    return {
        "mapped_values": {},
        "unmapped_parameters": list(unmapped),
        "conversion_notes": {},
    }


def _route_id_for_path(path: Iterable[str]) -> str:
    return "route:" + "->".join(str(node) for node in path)


def _clip(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
