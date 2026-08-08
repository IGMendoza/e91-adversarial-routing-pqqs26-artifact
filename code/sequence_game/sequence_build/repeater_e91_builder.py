"""Build fixed-path RouterNetTopo networks for repeater-backed E91.

This is the first repeater implementation path. It deliberately starts with an
explicit Alice--repeater--Bob route and the stock SeQUeNCe repeater stack:

- ``RouterNetTopo`` with ``QuantumRouter`` endpoints/repeater and ``BSMNode``s.
- Barrett-Kok elementary generation through
  ``EntanglementGenerationA/B`` (``BarretKokA``/``BarretKokB``).
- Resource-manager entanglement swapping through
  ``EntanglementSwappingA/B``.

The legacy bare-fiber BBM92 builder remains in ``e91_builder.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sequence.components.photon import Photon
from sequence.components.circuit import Circuit
from sequence.constants import BARRET_KOK, SPEED_OF_LIGHT
from sequence.entanglement_management.generation import (
    EntanglementGenerationA,
    EntanglementGenerationB,
)
from sequence.kernel.quantum_manager import QuantumManager
from sequence.network_management.routing import ROUTING_STATIC
from sequence.topology.node import QuantumRouter
from sequence.topology.router_net_topo import RouterNetTopo

from ..eve.attack_formalisms import repeater_memory_dephasing_probe_channel
from ..eve.repeater_attacks import RepeaterAttackSpec
from ..physical.registry import PhysicalModel
from .pre_swap_swapping import install_pre_swap_hook


class RepeaterE91BuildError(ValueError):
    """Invalid fixed-path repeater E91 build configuration."""


TARGET_MEMORY_SIDE = "left"


@dataclass(frozen=True)
class FixedRepeaterPath:
    """Explicit endpoint path and edge lengths for the repeater E91 pass."""

    nodes: tuple[str, ...] = ("alice", "r", "bob")
    edge_lengths_m: tuple[float, ...] = (1000.0, 1000.0)

    def __post_init__(self) -> None:
        if len(self.nodes) < 2:
            raise RepeaterE91BuildError("fixed path needs Alice and Bob endpoints")
        if len(self.edge_lengths_m) != len(self.nodes) - 1:
            raise RepeaterE91BuildError(
                "edge_lengths_m must have one length per consecutive path edge")
        if len(set(self.nodes)) != len(self.nodes):
            raise RepeaterE91BuildError(f"path is not simple: {self.nodes}")
        if any(length < 0 for length in self.edge_lengths_m):
            raise RepeaterE91BuildError("edge lengths must be non-negative")


@dataclass
class BuiltRepeaterE91Line:
    """RouterNetTopo plus metadata needed by the repeater E91 runner."""

    topology: RouterNetTopo
    node_path: tuple[str, ...]
    config_path: Path
    swapping_success_prob: float
    swapping_degradation: float
    attack: RepeaterAttackSpec
    memory_intervention_records: list[dict[str, Any]]
    edge_intercept_records: list[dict[str, Any]]
    pre_swap_hook_installed: bool = False

    @property
    def timeline(self):
        return self.topology.get_timeline()

    @property
    def routers(self) -> dict[str, QuantumRouter]:
        return {
            node.name: node
            for node in self.topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)
        }

    def summary(self) -> dict[str, Any]:
        return {
            "node_path": list(self.node_path),
            "config_path": str(self.config_path),
            "num_routers": len(self.topology.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)),
            "num_bsm_nodes": len(self.topology.get_nodes_by_type(RouterNetTopo.BSM_NODE)),
            "num_qchannels": len(self.topology.get_qchannels()),
            "swapping_success_prob": self.swapping_success_prob,
            "swapping_degradation": self.swapping_degradation,
            "attack": self.attack.to_dict(),
            "pre_swap_memory_hook_registered": _uses_pre_swap_memory_hook(self.attack),
            "edge_intercept_resend_public": {
                "records": len(self.edge_intercept_records),
                "target_edge": self.attack.target_edge,
                "target_u": self.attack.target_u,
                "target_v": self.attack.target_v,
                "implementation": (
                    "bsm_side_single_atom_photon_measure_resend"
                    if self.attack.kind == "edge_intercept_resend"
                    else "none"
                ),
            },
            "pre_swap_memory_intervention_public": {
                "records": len(self.memory_intervention_records),
                "target_memory_side": TARGET_MEMORY_SIDE,
                "selective_measurement_used": (
                    self.attack.kind == "repeater_memory_measurement"
                ),
                "reduced_dephasing_channel_used": (
                    self.attack.kind == "repeater_memory_dephasing_probe"
                ),
                "dephasing_probe_implementation": (
                    "exact_kraus_sampled_reduced_dephasing_channel"
                    if self.attack.kind == "repeater_memory_dephasing_probe"
                    else "none"
                ),
            },
            "stage1_generation": stage1_generation_record(),
        }


def stage1_generation_record() -> dict[str, str]:
    """Exact SeQUeNCe elementary-generation protocol used by RouterNetTopo."""

    protocol = EntanglementGenerationA.get_global_type()
    return {
        "router_topology": "sequence.topology.router_net_topo.RouterNetTopo",
        "router_class": "sequence.topology.node.QuantumRouter",
        "bsm_node_class": "sequence.topology.node.BSMNode",
        "protocol_type": protocol,
        "EntanglementGenerationA": EntanglementGenerationA._registry[protocol].__name__,
        "EntanglementGenerationB": EntanglementGenerationB._registry[protocol].__name__,
        "default_protocol": BARRET_KOK,
    }


def _require_kind(model: PhysicalModel, kind: str) -> None:
    if model.device_kind != kind:
        raise RepeaterE91BuildError(
            f"expected {kind!r} model, got {model.device_kind!r} ({model.model_name})")
    model.require_resolved()


def _memory_array_template(model: PhysicalModel, *, multiplier: float = 1.0,
                           efficiency_override: float | None = None) -> dict[str, Any]:
    _require_kind(model, "memory")
    p = model.parameters
    fidelity = max(0.0, min(1.0, float(p["fidelity"]) * multiplier))
    # efficiency_override lets the SeQUeNCe repeater path substitute the Cui2025
    # multiplexed effective branching ratio for the single-mode per-excitation
    # efficiency (see configs/physical/memory_literature_stub.json
    # multiplexing_model). None keeps the model's single-mode value.
    efficiency = (float(p["efficiency"]) if efficiency_override is None
                  else float(efficiency_override))
    return {
        "fidelity": fidelity,
        "frequency": float(p["frequency"]),
        "efficiency": efficiency,
        "coherence_time": float(p["coherence_time"]),
        "wavelength": int(float(p["wavelength"])),
    }


def _router_templates(path: FixedRepeaterPath, memory_model: PhysicalModel,
                      attack: RepeaterAttackSpec, *,
                      swapping_success_prob: float,
                      swapping_degradation: float,
                      memory_efficiency_override: float | None = None,
                      ) -> dict[str, dict[str, Any]]:
    templates: dict[str, dict[str, Any]] = {}
    for node in path.nodes:
        multiplier = 1.0
        if attack.kind == "memory_degradation" and attack.target_node == node:
            multiplier = attack.memory_fidelity_multiplier
        success = swapping_success_prob
        if attack.kind == "swap_denial" and attack.target_node == node:
            success = 0.0
        templates[f"memo_{node}"] = {
            "MemoryArray": _memory_array_template(
                memory_model, multiplier=multiplier,
                efficiency_override=memory_efficiency_override),
            "routing": ROUTING_STATIC,
            # Upstream v1.0.0 reads these per-node knobs from the template and
            # threads them into EntanglementSwappingA via the rule action args
            # (sequence/topology/node.py, resource_manager.generate_load_rules).
            # This replaces the former submodule patch on rsvp/network_manager.
            "EntanglementSwapping": {
                "swapping_success_prob": success,
                "swapping_degradation": swapping_degradation,
            },
        }
    return templates


def _topology_config(path: FixedRepeaterPath, memory_model: PhysicalModel,
                     fiber_model: PhysicalModel, *, stop_time_ps: int,
                     memory_size: int, seed: int,
                     attack: RepeaterAttackSpec,
                     swapping_success_prob: float = 1.0,
                     swapping_degradation: float = 1.0,
                     memory_efficiency_override: float | None = None,
                     formalism: str | None = None) -> dict[str, Any]:
    _require_kind(fiber_model, "fiber_channel")
    if memory_size < 1:
        raise RepeaterE91BuildError("memory_size must be >= 1")

    f = fiber_model.parameters
    light_speed = float(f.get("light_speed", SPEED_OF_LIGHT))
    attenuation = float(f["attenuation"])

    nodes = [
        {
            "name": node,
            "type": RouterNetTopo.QUANTUM_ROUTER,
            "seed": seed + i,
            "memo_size": memory_size,
            "template": f"memo_{node}",
        }
        for i, node in enumerate(path.nodes)
    ]

    qconnections = []
    for i, (u, v, length) in enumerate(zip(path.nodes, path.nodes[1:], path.edge_lengths_m)):
        qconnections.append({
            "node1": u,
            "node2": v,
            "attenuation": attenuation,
            "distance": length,
            "type": RouterNetTopo.MEET_IN_THE_MID,
            "seed": seed + 100 + i,
        })

    cumulative = [0.0]
    for length in path.edge_lengths_m:
        cumulative.append(cumulative[-1] + length)
    cconnections = []
    for i, u in enumerate(path.nodes):
        for j, v in enumerate(path.nodes[i + 1:], start=i + 1):
            distance = cumulative[j] - cumulative[i]
            delay = round(distance / light_speed) if light_speed > 0 else 0
            cconnections.append({
                "node1": u,
                "node2": v,
                "distance": distance,
                "delay": delay,
            })

    config = {
        "stop_time": stop_time_ps,
        "templates": _router_templates(
            path,
            memory_model,
            attack,
            swapping_success_prob=swapping_success_prob,
            swapping_degradation=swapping_degradation,
            memory_efficiency_override=memory_efficiency_override,
        ),
        "nodes": nodes,
        "qconnections": qconnections,
        "cconnections": cconnections,
    }
    if formalism is not None:
        config["formalism"] = formalism
    return config


# Sentinel config path for in-memory topology builds. RouterNetTopo accepts a
# config dict directly (v1.0.0 Topology._load), so no temp file is written; this
# label keeps BuiltRepeaterE91Line.config_path stable for the debug summary
# (which is stripped before any runtime metrics consumer).
_IN_MEMORY_CONFIG_PATH = Path("<in_memory>")


def build_fixed_repeater_e91_line(
        path: FixedRepeaterPath,
        *,
        memory_model: PhysicalModel,
        fiber_model: PhysicalModel,
        stop_time_ps: int,
        memory_size: int,
        seed: int,
        swapping_success_prob: float = 0.5,
        swapping_degradation: float = 1.0,
        attack: RepeaterAttackSpec = RepeaterAttackSpec(),
        memory_efficiency_override: float | None = None,
) -> BuiltRepeaterE91Line:
    """Build a fixed-path RouterNetTopo and apply swapping/attack knobs.

    ``memory_efficiency_override`` (SeQUeNCe repeater path only) replaces the
    memory model's single-mode per-attempt efficiency with a supplied effective
    value, e.g. the Cui2025 multiplexed effective branching ratio. ``None``
    keeps the model's single-mode efficiency.
    """

    path.__post_init__()
    attack.validate_on_path(path.nodes)
    if not 0 <= swapping_success_prob <= 1:
        raise RepeaterE91BuildError("swapping_success_prob must be in [0, 1]")
    if not 0 <= swapping_degradation <= 1:
        raise RepeaterE91BuildError("swapping_degradation must be in [0, 1]")
    if memory_efficiency_override is not None and not 0 <= memory_efficiency_override <= 1:
        raise RepeaterE91BuildError("memory_efficiency_override must be in [0, 1]")

    config = _topology_config(
        path,
        memory_model,
        fiber_model,
        stop_time_ps=stop_time_ps,
        memory_size=memory_size,
        seed=seed,
        attack=attack,
        swapping_success_prob=swapping_success_prob,
        swapping_degradation=swapping_degradation,
        memory_efficiency_override=memory_efficiency_override,
    )
    previous_formalism = QuantumManager.get_active_formalism()
    try:
        topology = RouterNetTopo(config)
        _apply_fiber_model_to_quantum_channels(topology, fiber_model)
    finally:
        QuantumManager.set_global_manager_formalism(previous_formalism)
    memory_intervention_records: list[dict[str, Any]] = []
    edge_intercept_records: list[dict[str, Any]] = []
    built = BuiltRepeaterE91Line(
        topology=topology,
        node_path=path.nodes,
        config_path=_IN_MEMORY_CONFIG_PATH,
        swapping_success_prob=swapping_success_prob,
        swapping_degradation=swapping_degradation,
        attack=attack,
        memory_intervention_records=memory_intervention_records,
        edge_intercept_records=edge_intercept_records,
    )
    _install_edge_intercept_resend(
        built,
        attack,
        records=edge_intercept_records,
    )
    # Swap-knobs (success/degradation, incl. swap_denial) are applied via the
    # per-node EntanglementSwapping template above. Pre-swap memory attacks are
    # applied by installing a first-party pre-swap hook on the target router,
    # consumed by EntanglementSwappingA_PreSwapHook during the swap. The paired
    # swapping formalism is scoped in run_fixed_repeater_e91_trial around the run.
    pre_swap_hook = _build_pre_swap_memory_hook(
        attack,
        selected_path=path.nodes,
        records=memory_intervention_records,
    )
    if pre_swap_hook is not None:
        for router in built.routers.values():
            install_pre_swap_hook(router, pre_swap_hook)
        built.pre_swap_hook_installed = True
    return built


def _install_edge_intercept_resend(
        built: BuiltRepeaterE91Line,
        attack: RepeaterAttackSpec,
        *,
        records: list[dict[str, Any]],
) -> None:
    if attack.kind != "edge_intercept_resend":
        return

    try:
        edge_index = list(zip(built.node_path, built.node_path[1:])).index(
            (attack.target_u, attack.target_v)
        )
    except ValueError:
        # validate_on_path already accepted the undirected edge. If orientation
        # differs, use the path's Alice-to-Bob orientation for the receive hook.
        pairs = list(zip(built.node_path, built.node_path[1:]))
        edge = frozenset((attack.target_u, attack.target_v))
        edge_index = next(
            index for index, pair in enumerate(pairs)
            if frozenset(pair) == edge
        )
    upstream, downstream = (
        built.node_path[edge_index],
        built.node_path[edge_index + 1],
    )
    bsm_name = f"BSM.{upstream}.{downstream}"
    bsm_node = built.timeline.get_entity_by_name(bsm_name)
    if bsm_node is None:
        raise RepeaterE91BuildError(f"missing BSM node {bsm_name!r}")
    original_receive = bsm_node.receive_qubit

    def receive_with_intercept(src: str, qubit) -> None:
        if src == downstream:
            qubit = _edge_intercept_resend_photon(
                bsm_node,
                qubit,
                attack=attack,
                bsm_name=bsm_name,
                source_node=src,
                records=records,
            )
        original_receive(src, qubit)

    bsm_node.receive_qubit = receive_with_intercept


def _edge_intercept_resend_photon(
        bsm_node,
        photon,
        *,
        attack: RepeaterAttackSpec,
        bsm_name: str,
        source_node: str,
        records: list[dict[str, Any]],
):
    qm = bsm_node.timeline.quantum_manager
    key = photon.quantum_state
    sample = bsm_node.get_generator().random()
    result = qm.run_circuit(_measurement_circuit("Z"), [key], sample)
    outcome = int(result[key])

    resent = Photon(
        f"{attack.attack_id or 'edge_intercept_resend'}.resend{len(records)}",
        bsm_node.timeline,
        wavelength=getattr(photon, "wavelength", 0),
        location=bsm_node,
        encoding_type=photon.encoding_type,
        use_qm=True,
    )
    if outcome:
        circuit = Circuit(1)
        circuit.x(0)
        qm.run_circuit(circuit, [resent.quantum_state])
    resent.is_null = getattr(photon, "is_null", False)
    resent.loss = getattr(photon, "loss", 0.0)

    records.append({
        "attack_id": attack.attack_id,
        "family": attack.kind,
        "target_edge": attack.target_edge,
        "target_u": attack.target_u,
        "target_v": attack.target_v,
        "bsm_node": bsm_name,
        "source_node": source_node,
        "basis": "Z",
        "outcome": outcome,
        "hook_time": bsm_node.timeline.now(),
        "implementation": "bsm_side_single_atom_photon_measure_resend",
    })
    return resent


def _apply_fiber_model_to_quantum_channels(
        topology: RouterNetTopo,
        fiber_model: PhysicalModel,
) -> None:
    """Thread first-party fiber params through RouterNetTopo's QC defaults."""

    params = fiber_model.parameters
    for channel in topology.get_qchannels():
        if "frequency" in params:
            channel.frequency = float(params["frequency"])
        if "light_speed" in params:
            channel.light_speed = float(params["light_speed"])
        if "polarization_fidelity" in params:
            channel.polarization_fidelity = float(params["polarization_fidelity"])


def _uses_pre_swap_memory_hook(attack: RepeaterAttackSpec) -> bool:
    return attack.kind in {
        "repeater_memory_measurement",
        "repeater_memory_dephasing_probe",
    }


def _build_pre_swap_memory_hook(
        attack: RepeaterAttackSpec,
        *,
        selected_path: tuple[str, ...],
        records: list[dict[str, Any]]):
    if not _uses_pre_swap_memory_hook(attack):
        return None
    interior = set(selected_path[1:-1])
    if attack.target_node not in interior:
        return None

    def hook(context) -> None:
        if context.repeater_node_name != attack.target_node:
            return
        target_key = context.left_qstate_key
        target_memory = context.left_memory_name
        if attack.kind == "repeater_memory_measurement":
            _apply_projective_memory_measurement(context, attack, target_key,
                                                 target_memory, records)
        elif attack.kind == "repeater_memory_dephasing_probe":
            _apply_dephasing_probe(context, attack, target_key, target_memory,
                                   records)

    return hook


def _measurement_circuit(basis: str) -> Circuit:
    circuit = Circuit(1)
    if basis == "X":
        circuit.h(0)
    circuit.measure(0)
    return circuit


def _measurement_sample(context) -> float:
    sample = context.swap_protocol.owner.get_generator().random()
    return sample if sample > 0 else 1e-12


def _apply_projective_memory_measurement(
        context,
        attack: RepeaterAttackSpec,
        target_key: int,
        target_memory: str,
        records: list[dict[str, Any]]) -> None:
    result = context.quantum_manager.run_circuit(
        _measurement_circuit(attack.basis),
        [target_key],
        _measurement_sample(context),
    )
    records.append({
        "attack_id": attack.attack_id,
        "family": attack.kind,
        "target_repeater": attack.target_node,
        "target_memory_side": TARGET_MEMORY_SIDE,
        "target_memory": target_memory,
        "target_qstate_key": target_key,
        "basis": attack.basis,
        "outcome": result[target_key],
        "hook_time": context.timeline_time,
        "implementation": "selective_projective_measurement",
    })


def _apply_dephasing_probe(
        context,
        attack: RepeaterAttackSpec,
        target_key: int,
        target_memory: str,
        records: list[dict[str, Any]]) -> None:
    state = context.quantum_manager.get(target_key)
    theta = attack.theta if attack.theta is not None else 0.0
    implementation = "exact_reduced_density_channel"
    kraus_branch = "identity"
    state_array = getattr(state, "state", None)
    if getattr(state_array, "ndim", 1) == 2:
        keys = list(state.keys)
        target_qubit = keys.index(target_key)
        updated = repeater_memory_dephasing_probe_channel(
            state_array,
            theta,
            target_qubit=target_qubit,
        )
        context.quantum_manager.set(keys, updated)
    else:
        phase_flip_probability = (1.0 - math.cos(theta)) / 2.0
        if phase_flip_probability:
            sample = context.swap_protocol.owner.get_generator().random()
            if sample < phase_flip_probability:
                circuit = Circuit(1)
                circuit.z(0)
                context.quantum_manager.run_circuit(circuit, [target_key])
                kraus_branch = "Z"
        implementation = "exact_kraus_sampled_reduced_dephasing_channel"
    records.append({
        "attack_id": attack.attack_id,
        "family": attack.kind,
        "target_repeater": attack.target_node,
        "target_memory_side": TARGET_MEMORY_SIDE,
        "target_memory": target_memory,
        "target_qstate_key": target_key,
        "theta": theta,
        "theta_label": attack.theta_label,
        "hook_time": context.timeline_time,
        "implementation": implementation,
        "kraus_branch": kraus_branch,
    })
