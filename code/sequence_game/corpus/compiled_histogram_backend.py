"""Compiled-histogram runtime backend adapter for the paper pipeline.

Bridges corpus route/action objects to the compiled length-parametric E91
histogram kernel (scripts/generate_e91_game_turn_histograms.py):

    corpus route -> fiber-length vector + positional chain ids
    corpus action -> generator EveMove (chain-local target, canonicalized when
                     the target is off the selected route)
    step          -> one multinomial draw of ``requested_pairs`` transcript
                     records from the cached evaluated histogram H(L), then
                     the same accept/abort/exposure decision layer as the
                     direct-E91 transcript backend.

Two action-surface caches make the per-step cost sampling-only:

* kernel cache: (hop_count, canonical Eve move) -> CompiledRouteKernel,
  compiled once and reused across every route of the same hop count;
* bounded evaluated-histogram LRU: (theta hash, fiber vector, schema) -> prepared
  row arrays, so each distinct (kernel, L) pays one compiled evaluation and
  every subsequent game step is a microsecond-scale multinomial draw.

The kernel dictionary is not an LRU; its finite demand is therefore included in
the histogram experiment's canonical-move cache/RAM preflight.

The generator script is loaded lazily by file path; nothing here imports the
SeQUeNCe runtime. Scope: the compiled kernel's supported stack
(fock_pnr_snspd, memory_emission, pure-dephasing storage); unsupported
configurations raise CompiledHistogramUnsupported with the kernel's reason.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import FunctionType, ModuleType
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_e91_game_turn_histograms.py"
GENERATOR_MODULE_NAME = "generate_e91_game_turn_histograms"
DEFAULT_DETECTOR_PROFILE_PATH = (
    REPO_ROOT
    / "configs"
    / "physical"
    / "detectors"
    / "snspd_colangelo_impedance_matched_differential_pnr_1550.json"
)
DEFAULT_CONVERSION_RESPONSE_PATH = (
    REPO_ROOT
    / "configs"
    / "physical"
    / "conversion"
    / "cui_dfg_866_to_1558.json"
)

COMPILED_HISTOGRAM_BACKEND = "compiled_histogram_transcript"
COMPILED_HISTOGRAM_RUNTIME_VERSION = "compiled_histogram_runtime.v2"
BACKEND_SCOPE = (
    "event-free compiled length-parametric E91 transcript histogram; "
    "no SeQUeNCe runtime, no per-step physics recomputation"
)

_GENERATOR_MODULE: Any = None

# CompiledRouteKernel construction/evaluation temporarily touches generator-
# level instrument caches.  Cold misses are therefore serialized inside one
# process, while prepared-histogram hits and multinomial sampling remain
# concurrent.  Corpus execution obtains parallel cold compilation by running
# several graph-owner processes, each with two learner threads sharing one
# mutable on-demand pool.
_COMPILE_EVALUATE_LOCK = threading.RLock()


def _recursive_python_size_bytes(value: Any) -> int:
    """Approximate cache-owned Python/NumPy bytes without double counting."""

    seen: set[int] = set()

    def visit(item: Any) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(item)
        if isinstance(item, np.ndarray):
            if item.base is not None:
                size += visit(item.base)
            return size
        if isinstance(item, dict):
            return size + sum(
                visit(key) + visit(entry) for key, entry in item.items()
            )
        if isinstance(item, (list, tuple, set, frozenset, OrderedDict)):
            return size + sum(visit(entry) for entry in item)
        if isinstance(item, (FunctionType, ModuleType, type)):
            return size
        attributes = getattr(item, "__dict__", None)
        if isinstance(attributes, dict):
            size += visit(attributes)
        return size

    return int(visit(value))


class CompiledHistogramUnsupported(RuntimeError):
    """Raised when the frozen configuration is outside the compiled kernel's scope."""


def _load_json_config(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    with resolved.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_histogram_generator() -> Any:
    """Load the histogram generator script as a module (cached)."""
    global _GENERATOR_MODULE
    if _GENERATOR_MODULE is not None:
        return _GENERATOR_MODULE
    existing = sys.modules.get(GENERATOR_MODULE_NAME)
    if existing is not None:
        _GENERATOR_MODULE = existing
        return existing
    spec = importlib.util.spec_from_file_location(GENERATOR_MODULE_NAME, GENERATOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load histogram generator from {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[GENERATOR_MODULE_NAME] = module
    spec.loader.exec_module(module)
    _GENERATOR_MODULE = module
    return module


@dataclass(frozen=True)
class CompiledHistogramBackendConfig:
    """Frozen backend configuration; hashed into the cache/schema identity.

    Production defaults name the most complete repository PNR detector and
    the matching 866-nm converter family.  They intentionally fail adherence
    validation until response coverage at the converter's 1558-nm output is
    supplied.  Ideal/reference or wavelength-mismatched runs require explicit
    opt-ins and remain labelled in diagnostics.
    """

    hardware_config_dir: str = str(REPO_ROOT / "configs" / "physical")
    detector_kind: str = "pnr_snspd"
    detector_profile_path: str | None = str(DEFAULT_DETECTOR_PROFILE_PATH)
    detector_response_path: str | None = None
    fiber_response_path: str | None = None
    conversion_response_path: str | None = str(DEFAULT_CONVERSION_RESPONSE_PATH)
    multiplexing_controller_path: str | None = None
    repeater_service_profile_path: str | None = None
    allow_wavelength_mismatch: bool = False
    allow_ideal_detector_reference: bool = False
    require_explicit_device_stack: bool = True
    fock_cutoff: int = 3
    waiting_max_attempts: int = 4096
    waiting_max_history_pairs: int = 250_000
    waiting_max_wait_signatures: int = 512
    waiting_tail_tolerance: float = 1e-6
    waiting_budget_mode: str = "fixed_attempts"
    waiting_budget_wall_clock_ps: float | None = None
    waiting_precision_mode: str = "fixed_caps"
    waiting_tv_tolerance: float = 1e-6
    waiting_metric_tolerance: float = 1e-3
    waiting_max_refinement_rounds: int = 4
    waiting_max_runtime_s: float | None = None
    waiting_max_memory_mb: float | None = 512.0
    waiting_protocol: str = "single_buffer_swap_asap"
    rolling_edge_availability: float | tuple[float, ...] | None = None
    rolling_age_distribution_kind: str = "discrete"
    rolling_age_distribution_points_s: tuple[float, ...] = (0.0,)
    rolling_age_distribution_weights: tuple[float, ...] = (1.0,)
    rolling_edge_age_distribution_points_s: tuple[tuple[float, ...], ...] | None = None
    rolling_edge_age_distribution_weights: tuple[tuple[float, ...], ...] | None = None
    rolling_age_evaluation: str = "cartesian"
    rolling_age_support_product_limit: int = 1_000_000
    rolling_accepted_alive_time_s: float | None = None
    rolling_memory_slots_per_edge_side: int = 1
    rolling_selection_policy: str = "configured_distribution"
    rolling_age_distribution_source: str = "configured"
    memory_cutoff_ps: float | None = None
    compile_key_length: int = 256
    evaluated_cache_size: int = 256
    qber_abort_threshold: float | None = None
    chsh_abort_threshold: float | None = None
    min_key_pairs_for_qber_abort: int = 16
    min_chsh_pairs_for_chsh_abort: int = 16

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "hardware_config_dir": self.hardware_config_dir,
            "detector_kind": self.detector_kind,
            "detector_profile_path": self.detector_profile_path,
            "detector_response_path": self.detector_response_path,
            "fiber_response_path": self.fiber_response_path,
            "conversion_response_path": self.conversion_response_path,
            "allow_wavelength_mismatch": self.allow_wavelength_mismatch,
            "allow_ideal_detector_reference": self.allow_ideal_detector_reference,
            "require_explicit_device_stack": self.require_explicit_device_stack,
            "fock_cutoff": self.fock_cutoff,
            "waiting_max_attempts": self.waiting_max_attempts,
            "waiting_max_history_pairs": self.waiting_max_history_pairs,
            "waiting_max_wait_signatures": self.waiting_max_wait_signatures,
            "waiting_tail_tolerance": self.waiting_tail_tolerance,
            "waiting_budget_mode": self.waiting_budget_mode,
            "waiting_budget_wall_clock_ps": self.waiting_budget_wall_clock_ps,
            "waiting_precision_mode": self.waiting_precision_mode,
            "waiting_tv_tolerance": self.waiting_tv_tolerance,
            "waiting_metric_tolerance": self.waiting_metric_tolerance,
            "waiting_max_refinement_rounds": self.waiting_max_refinement_rounds,
            "waiting_max_runtime_s": self.waiting_max_runtime_s,
            "waiting_max_memory_mb": self.waiting_max_memory_mb,
            "waiting_protocol": self.waiting_protocol,
            "rolling_edge_availability": self.rolling_edge_availability,
            "rolling_age_distribution_kind": self.rolling_age_distribution_kind,
            "rolling_age_distribution_points_s": list(self.rolling_age_distribution_points_s),
            "rolling_age_distribution_weights": list(self.rolling_age_distribution_weights),
            "rolling_edge_age_distribution_points_s": (
                [list(row) for row in self.rolling_edge_age_distribution_points_s]
                if self.rolling_edge_age_distribution_points_s is not None else None
            ),
            "rolling_edge_age_distribution_weights": (
                [list(row) for row in self.rolling_edge_age_distribution_weights]
                if self.rolling_edge_age_distribution_weights is not None else None
            ),
            "rolling_age_evaluation": self.rolling_age_evaluation,
            "rolling_age_support_product_limit": self.rolling_age_support_product_limit,
            "rolling_accepted_alive_time_s": self.rolling_accepted_alive_time_s,
            "rolling_memory_slots_per_edge_side": self.rolling_memory_slots_per_edge_side,
            "rolling_selection_policy": self.rolling_selection_policy,
            "rolling_age_distribution_source": self.rolling_age_distribution_source,
            "memory_cutoff_ps": self.memory_cutoff_ps,
            "compile_key_length": self.compile_key_length,
            "evaluated_cache_size": self.evaluated_cache_size,
            "qber_abort_threshold": self.qber_abort_threshold,
            "chsh_abort_threshold": self.chsh_abort_threshold,
            "min_key_pairs_for_qber_abort": self.min_key_pairs_for_qber_abort,
            "min_chsh_pairs_for_chsh_abort": self.min_chsh_pairs_for_chsh_abort,
        }
        # Keep the historical disabled-path config/cache identity unchanged.
        if self.multiplexing_controller_path is not None:
            payload["multiplexing_controller_path"] = self.multiplexing_controller_path
        if self.repeater_service_profile_path is not None:
            payload["repeater_service_profile_path"] = self.repeater_service_profile_path
        return payload


def edge_label(u: str, v: str) -> str:
    a, b = sorted((str(u), str(v)))
    return f"{a}-{b}"


def fiber_vector_from_route(
    ir_dict: dict[str, Any],
    route: dict[str, Any],
) -> list[float]:
    """Ordered per-hop fiber lengths (m) for a corpus route object.

    Mirrors the direct-E91 adapter: use route['edge_lengths_m'] when it is
    aligned with the hop count, otherwise look up each hop's edge in the
    topology IR edge list.
    """
    path = [str(node) for node in route["path"]]
    hop_count = len(path) - 1
    if hop_count < 1:
        raise ValueError(f"route {route.get('route_id')!r} has no hops")
    raw_lengths = route.get("edge_lengths_m") or ()
    if len(raw_lengths) == hop_count:
        return [float(length) for length in raw_lengths]
    lengths: list[float] = []
    for u, v in zip(path, path[1:]):
        edge = _edge_between(ir_dict, u, v)
        lengths.append(float(edge.get("length_m", edge.get("length", 1.0))))
    return lengths


def _edge_between(ir_dict: dict[str, Any], u: str, v: str) -> dict[str, Any]:
    for edge in ir_dict.get("edges", []):
        if {str(edge.get("u")), str(edge.get("v"))} == {str(u), str(v)}:
            return dict(edge)
    raise ValueError(f"no topology edge for route hop {u!r}->{v!r}")


def eve_move_from_runtime_action(
    action: dict[str, Any],
    route: dict[str, Any],
) -> tuple[Any, bool]:
    """Map a corpus runtime action onto the generator's chain-local EveMove.

    Returns (move, on_route). Edge targets are matched by the sorted
    node-pair label against the route hops and mapped to the positional chain
    edge id e{index}; node targets are matched against internal route nodes
    and mapped to r{position}. Off-route targets are canonicalized to the
    single no-move baseline (physically identical to an inactive move, and it
    keeps the kernel and evaluated caches from splitting on target ids that
    cannot affect the selected chain).
    """
    generator = load_histogram_generator()
    attack_type = str(action.get("attack_type") or "no_attack")
    action_id = str(action.get("id") or attack_type)
    target = str(action.get("target") or "")
    path = [str(node) for node in route["path"]]

    def baseline() -> Any:
        return generator.EveMove(
            move_id="no_move_baseline",
            move_kind="none",
            target_active=False,
            histogram_source="no_move_baseline",
        )

    if attack_type == "no_attack":
        return baseline(), True

    if attack_type in {"edge_dos", "edge_information_probe"}:
        edge_index = None
        for index, (u, v) in enumerate(zip(path, path[1:])):
            if edge_label(u, v) == target:
                edge_index = index
                break
        if edge_index is None:
            return baseline(), False
        chain_edge = f"e{edge_index}"
        if attack_type == "edge_dos":
            return generator.EveMove(
                move_id=f"edge_added_loss:{chain_edge}",
                move_kind="edge_added_loss",
                target_kind="edge",
                target_id=chain_edge,
                target_active=True,
                loss_multiplier=0.0,
                drop_probability=1.0,
            ), True
        return generator.EveMove(
            move_id=f"edge_measure_resend:{chain_edge}",
            move_kind="edge_measure_resend",
            target_kind="edge",
            target_id=chain_edge,
            target_active=True,
            # Match the public corpus action's direct-runtime semantics: the
            # generic information probe has no privileged basis choice and
            # samples X/Z uniformly. Basis-specific memory measurements remain
            # separate actions below.
            basis_choice="random",
            edge_measure_basis_label="random",
            edge_measure_probability=1.0,
            information_probability=1.0,
        ), True

    node_kinds = {
        "swap_denial": "swap_block",
        "memory_degradation": "memory_degradation",
        "repeater_memory_measure_Z": "repeater_memory_measure_Z",
        "repeater_memory_measure_X": "repeater_memory_measure_X",
    }
    if attack_type in node_kinds:
        internal = path[1:-1]
        if target not in internal:
            return baseline(), False
        position = path.index(target)
        chain_node = f"r{position}"
        kind = node_kinds[attack_type]
        kwargs: dict[str, Any] = {
            "move_id": f"{kind}:{chain_node}",
            "move_kind": kind,
            "target_kind": "node",
            "target_id": chain_node,
            "target_active": True,
        }
        if kind == "memory_degradation":
            kwargs.update(
                memory_side="both",
                target_memory_side="both",
                # mirror the corpus action semantics used by the direct-E91
                # backend for the same action id
                memory_depolarizing_probability=0.5,
                memory_dephasing_probability=0.5,
            )
        elif kind in {"repeater_memory_measure_Z", "repeater_memory_measure_X"}:
            kwargs.update(
                memory_side="both",
                target_memory_side="both",
                measurement_probability=1.0,
                information_probability=1.0,
            )
        return generator.EveMove(**kwargs), True

    if attack_type == "repeater_bsm_stabilizer_probe":
        internal = path[1:-1]
        if target not in internal:
            return baseline(), False
        position = path.index(target)
        chain_node = f"r{position}"
        stabilizer = str(action.get("bsm_stabilizer") or "ZZ").upper()
        if stabilizer not in {"ZZ", "XX"}:
            raise ValueError(
                "repeater BSM stabilizer probe requires XX or ZZ"
            )
        theta = float(action.get("dephasing_theta", math.pi / 2.0))
        if not math.isfinite(theta) or not 0.0 <= theta <= math.pi / 2.0:
            raise ValueError(
                "repeater BSM stabilizer probe theta must lie in [0, pi/2]"
            )
        return generator.EveMove(
            move_id=(
                "repeater_bsm_stabilizer_probe:"
                f"{chain_node}:{stabilizer}:{theta:.12g}"
            ),
            move_kind="repeater_bsm_stabilizer_probe",
            target_kind="node",
            target_id=chain_node,
            target_active=True,
            bsm_stabilizer=stabilizer,
            dephasing_theta=theta,
            # The retained record is local BSM parity/activity. It is not an
            # endpoint raw-key guess and therefore is excluded from the
            # information-exposure reward path below.
            information_probability=0.0,
        ), True

    raise ValueError(f"unsupported compiled-histogram runtime action {action_id!r}")


_INFORMATION_EXPOSING_MOVE_KINDS = {
    "edge_measure_resend",
    "repeater_memory_measure_Z",
    "repeater_memory_measure_X",
    "repeater_memory_dephasing_probe",
    "memory_degradation",
}


def classify_qber_estimation_power(
    *,
    expected_key_mass: float | None,
    sampled_qber_trials: int,
    adequate_sampled_trials: int = 1,
) -> str:
    """Classify whether a finite-sample QBER estimate is interpretable.

    ``expected_key_mass`` is the infinite-shot delivered key-basis probability
    mass when known. ``sampled_qber_trials`` is the finite-sample count of
    QBER-observable trials for aggregate reports, or the sampled sifted-key
    count for a single compiled backend step.
    """
    if expected_key_mass is not None and float(expected_key_mass) <= 0.0:
        return "no_key_mass"
    if int(sampled_qber_trials) <= 0:
        return "finite_sample_zero_key_trials"
    if int(sampled_qber_trials) < max(1, int(adequate_sampled_trials)):
        return "low_power"
    return "adequate"


@dataclass
class _EvaluatedHistogram:
    """Row arrays prepared for fast multinomial stepping."""

    theta_hash: str
    fiber_lengths_m: tuple[float, ...]
    probabilities: np.ndarray
    delivered_mask: np.ndarray
    pair_labels: list[str]
    pair_index: np.ndarray
    is_key_pair_row: np.ndarray
    is_key_error_row: np.ndarray
    is_chsh_row: np.ndarray
    chsh_pair_index: np.ndarray
    chsh_same_outcome: np.ndarray
    eve_labels: list[str]
    eve_index: np.ndarray
    no_delivery_stage_labels: list[str]
    no_delivery_stage_index: np.ndarray
    expected_metrics: dict[str, Any]
    delivery_model: dict[str, Any]
    waiting_metric_certification: dict[str, Any]
    evaluate_seconds: float
    row_count: int


def _sample_parallel_eve_mini_histogram(
    prepared: _EvaluatedHistogram,
    *,
    delivered_pair_count: int,
    seed: int,
    step_index: int,
) -> dict[str, int]:
    """Sample Eve labels without refining Alice's transcript histogram."""

    delivered = int(delivered_pair_count)
    if delivered <= 0:
        return {}
    selected = prepared.eve_index >= 0
    if not selected.any() or not prepared.eve_labels:
        raise RuntimeError(
            "parallel Eve observation histogram has no retained labels"
        )
    masses = np.bincount(
        prepared.eve_index[selected],
        weights=prepared.probabilities[selected],
        minlength=len(prepared.eve_labels),
    ).astype(float)
    total = float(np.sum(masses))
    if (
        not math.isfinite(total)
        or total <= 0.0
        or not np.all(np.isfinite(masses))
        or np.any(masses < 0.0)
    ):
        raise FloatingPointError(
            "invalid parallel Eve observation probability mass"
        )
    probabilities = masses / total
    # Independent derived stream: ancillary sampling never consumes or
    # perturbs the Alice transcript RNG stream [seed, step_index].
    rng = np.random.default_rng(
        [int(seed), int(step_index), 0x42534D41]
    )
    counts = rng.multinomial(delivered, probabilities)
    return {
        label: int(value)
        for label, value in zip(prepared.eve_labels, counts)
        if int(value) > 0
    }


@dataclass
class CompiledHistogramStepResult:
    backend: str
    backend_scope: str
    no_sequence_runtime: bool
    route_id: str
    action_id: str
    move_id: str
    move_kind: str
    target_kind: str
    target_id: str
    target_active: bool
    fiber_lengths_m: list[float]
    hop_count: int
    theta_hash: str
    requested_pair_count: int
    delivered_pair_count: int
    delivery_probability_expected: float
    setting_pair_counts: dict[str, int]
    key_pair_count: int
    chsh_pair_count: int
    discarded_setting_pair_count: int
    sifted_key_count: int
    key_error_count: int
    qber_key: float | None
    expected_qber_from_histogram: float | None
    sampled_qber_mean: float | None
    sampled_qber_trials: int
    sampled_qber_observable: bool
    qber_key_mass_expected: float
    qber_error_mass_expected: float
    qber_estimation_power: str
    qber_abort_threshold: float
    chsh_abort_threshold: float
    min_key_pairs_for_qber_abort: int
    min_chsh_pairs_for_chsh_abort: int
    qber_abort: bool
    chsh_S_observed: float | None
    chsh_S_expected: float | None
    chsh_setting_total_counts: dict[str, int]
    chsh_setting_same_counts: dict[str, int]
    chsh_correlations_observed: dict[str, float | None]
    chsh_abort: bool
    public_outcome: str
    no_delivery_stage_counts: dict[str, int]
    eve_observation_count: int
    eve_observation_distribution: dict[str, int]
    eve_guess_probability: float
    information_exposed: bool
    secure_delivery: bool
    eve_win: bool
    reward_components: dict[str, bool]
    provenance: dict[str, Any]

    def to_step_record(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "backend_scope": self.backend_scope,
            "no_sequence_runtime": self.no_sequence_runtime,
            "public_outcome": self.public_outcome,
            "route_id": self.route_id,
            "alice_route_id": self.route_id,
            "eve_action_id": self.action_id,
            "delivered": self.delivered_pair_count > 0,
            "physical_pair_arrival": self.delivered_pair_count > 0,
            "delivery_success": self.public_outcome == "accepted",
            "bell_qualified_delivery": self.public_outcome == "accepted",
            "delivered_count": self.delivered_pair_count,
            "qber_key": self.qber_key,
            "qber": self.qber_key,
            "expected_qber_from_histogram": self.expected_qber_from_histogram,
            "sampled_qber_mean": self.sampled_qber_mean,
            "sampled_qber_trials": self.sampled_qber_trials,
            "sampled_qber_observable": self.sampled_qber_observable,
            "qber_key_mass_expected": self.qber_key_mass_expected,
            "qber_error_mass_expected": self.qber_error_mass_expected,
            "qber_estimation_power": self.qber_estimation_power,
            "qber_abort_threshold": self.qber_abort_threshold,
            "chsh_abort_threshold": self.chsh_abort_threshold,
            "min_key_pairs_for_qber_abort": self.min_key_pairs_for_qber_abort,
            "min_chsh_pairs_for_chsh_abort": self.min_chsh_pairs_for_chsh_abort,
            "chsh_S": self.chsh_S_observed,
            "chsh_S_expected": self.chsh_S_expected,
            "chsh_S_observed": self.chsh_S_observed,
            "chsh_setting_total_counts": dict(
                self.chsh_setting_total_counts
            ),
            "chsh_setting_same_counts": dict(
                self.chsh_setting_same_counts
            ),
            "information_exposed": self.information_exposed,
            "secure_delivery": self.secure_delivery,
            "eve_win": self.eve_win,
            "provenance": dict(self.provenance),
        }


class CompiledHistogramBackend:
    """Per-process compiled-histogram backend with bounded caches."""

    def __init__(self, config: CompiledHistogramBackendConfig | None = None) -> None:
        self.config = config or CompiledHistogramBackendConfig()
        self._generator: Any = None
        self._front: dict[str, Any] | None = None
        self._context: Any = None
        self._protocol: Any = None
        self._schema_hash: str | None = None
        self._front_hash: str | None = None
        self._device_adherence: dict[str, Any] | None = None
        self._kernel_cache: dict[str, Any] = {}
        self._evaluated_cache: "OrderedDict[str, _EvaluatedHistogram]" = OrderedDict()
        self._cache_lock = threading.RLock()
        self._stats = {
            "kernel_compiles": 0,
            "kernel_cache_hits": 0,
            "kernel_compile_seconds": 0.0,
            "evaluated_cache_hits": 0,
            "evaluated_cache_misses": 0,
            "evaluated_cache_evictions": 0,
            "evaluate_seconds": 0.0,
            "steps_sampled": 0,
            "step_sample_seconds": 0.0,
        }
        self._warnings: list[str] = []

    # -- setup ---------------------------------------------------------------

    def _ensure_ready(self) -> Any:
        if self._generator is not None:
            return self._generator
        generator = load_histogram_generator()
        front = generator.load_or_build_front_matter(Path(self.config.hardware_config_dir))
        detector_profile = _load_json_config(self.config.detector_profile_path)
        detector_response = _load_json_config(self.config.detector_response_path)
        fiber_response = _load_json_config(self.config.fiber_response_path)
        conversion_response = _load_json_config(self.config.conversion_response_path)
        multiplexing_controller = _load_json_config(
            self.config.multiplexing_controller_path
        )
        repeater_service_profile = _load_json_config(
            self.config.repeater_service_profile_path
        )
        context = generator.ModelContext(
            physical_path="fock_pnr_snspd",
            source_model="memory_emission",
            detector_kind=self.config.detector_kind,
            fock_cutoff=int(self.config.fock_cutoff),
            detector_profile_path=self.config.detector_profile_path,
            detector_profile=detector_profile,
            detector_response=detector_response,
            fiber_response=fiber_response,
            conversion_response=conversion_response,
            multiplexing_controller=multiplexing_controller,
            repeater_service_profile_path=self.config.repeater_service_profile_path,
            repeater_service_profile=repeater_service_profile,
            allow_wavelength_mismatch=bool(self.config.allow_wavelength_mismatch),
            waiting_max_attempts=int(self.config.waiting_max_attempts),
            waiting_max_history_pairs=int(self.config.waiting_max_history_pairs),
            waiting_max_wait_signatures=int(self.config.waiting_max_wait_signatures),
            waiting_tail_tolerance=float(self.config.waiting_tail_tolerance),
            waiting_budget_mode=str(self.config.waiting_budget_mode),
            waiting_budget_wall_clock_ps=self.config.waiting_budget_wall_clock_ps,
            waiting_precision_mode=str(self.config.waiting_precision_mode),
            waiting_tv_tolerance=float(self.config.waiting_tv_tolerance),
            waiting_metric_tolerance=float(self.config.waiting_metric_tolerance),
            waiting_max_refinement_rounds=int(self.config.waiting_max_refinement_rounds),
            waiting_max_runtime_s=self.config.waiting_max_runtime_s,
            waiting_max_memory_mb=self.config.waiting_max_memory_mb,
            waiting_protocol=str(self.config.waiting_protocol),
            rolling_edge_availability=self.config.rolling_edge_availability,
            rolling_age_distribution_kind=str(self.config.rolling_age_distribution_kind),
            rolling_age_distribution_points_s=tuple(self.config.rolling_age_distribution_points_s),
            rolling_age_distribution_weights=tuple(self.config.rolling_age_distribution_weights),
            rolling_edge_age_distribution_points_s=self.config.rolling_edge_age_distribution_points_s,
            rolling_edge_age_distribution_weights=self.config.rolling_edge_age_distribution_weights,
            rolling_age_evaluation=str(self.config.rolling_age_evaluation),
            rolling_age_support_product_limit=int(self.config.rolling_age_support_product_limit),
            rolling_accepted_alive_time_s=self.config.rolling_accepted_alive_time_s,
            rolling_memory_slots_per_edge_side=int(self.config.rolling_memory_slots_per_edge_side),
            rolling_selection_policy=str(self.config.rolling_selection_policy),
            rolling_age_distribution_source=str(self.config.rolling_age_distribution_source),
            memory_cutoff_ps=self.config.memory_cutoff_ps,
        )
        front = generator.apply_repeater_service_front_overrides(front, context)
        empirical_pair_instrument = generator.repeater_elementary_pair_instrument(context)
        if empirical_pair_instrument is not None:
            resolved_detector_kind = "aggregated_in_empirical_remote_pair"
            wavelength_report = {
                "status": "aggregated_empirical_link_stack",
                "issues": [],
                "semantics": (
                    "the source, converter, fiber, phase stabilization, and detector are "
                    "already included in the demonstrated pair rate/state; repository optical "
                    "components are inactive and are not subjected to a second compatibility check"
                ),
            }
        else:
            try:
                resolved_detector_kind = generator.resolved_detector_kind(front, context)
                wavelength_report = generator.wavelength_compatibility_report(front, context)
            except (TypeError, ValueError) as exc:
                raise CompiledHistogramUnsupported(
                    f"compiled histogram device stack is invalid: {exc}"
                ) from exc

        adherence_blockers: list[dict[str, Any]] = []
        adherence_blockers.extend({
            "effect": "repeater_service_profile_unsupported",
            "parameter": "repeater_service_profile_path",
            "reason": reason,
        } for reason in generator.repeater_service_activation_blockers(context))
        controller_model = None
        if multiplexing_controller is not None:
            try:
                controller_model = generator.active_multiplexing_controller(context)
                controller_blockers = controller_model.execution_blockers() if controller_model else ()
            except (NotImplementedError, TypeError, ValueError) as exc:
                controller_blockers = (str(exc),)
            adherence_blockers.extend({
                "effect": "multiplexing_controller_mapping_unsupported",
                "parameter": "multiplexing_controller_path",
                "reason": reason,
            } for reason in controller_blockers)
        if (
            empirical_pair_instrument is None
            and resolved_detector_kind == "ideal_pnr"
            and not self.config.allow_ideal_detector_reference
        ):
            adherence_blockers.append({
                "effect": "ideal_detector_not_authorized",
                "parameter": "detector_kind",
                "reason": (
                    "ideal_pnr is a mathematical reference detector, not a demonstrated device; "
                    "set allow_ideal_detector_reference=true only for an explicitly labelled "
                    "test/reference calculation"
                ),
            })
        if (
            empirical_pair_instrument is None
            and self.config.require_explicit_device_stack
            and resolved_detector_kind != "ideal_pnr"
        ):
            adherence_blockers.extend(
                generator.explicit_detector_operating_point_blockers(front, context)
            )
        if not self.config.allow_wavelength_mismatch:
            for issue in wavelength_report.get("issues", []):
                if issue.get("status") == "missing":
                    adherence_blockers.append(dict(issue))

        selected_memory_name = front.get("selected_models", {}).get("memory")
        selected_memory = (
            front.get("models", {}).get(selected_memory_name, {})
            if selected_memory_name is not None else {}
        )
        declared_multiplexing = (
            selected_memory.get("parameters", {}).get("multiplexing_model")
            if isinstance(selected_memory, dict) else None
        )

        if empirical_pair_instrument is not None:
            memory_multiplexing = {
                "declared_model": declared_multiplexing,
                "active_in_compiled_kernel": False,
                "active_operating_mode": "empirical_remote_memory_pair",
                "repeater_service_profile": front.get("repeater_service_profile"),
                "elementary_pair_instrument": empirical_pair_instrument,
                "interpretation": (
                    "the demonstrated link's accepted memory-pair state and renewal rate replace "
                    "the repository optical Fock instrument; no Cui or Krutyanskiy optical "
                    "efficiency is multiplied into it"
                ),
            }
        elif repeater_service_profile is not None:
            memory_multiplexing = {
                "declared_model": declared_multiplexing,
                "active_in_compiled_kernel": False,
                "active_operating_mode": "repeater_service_profile_fixed_inventory",
                "repeater_service_profile": front.get("repeater_service_profile"),
                "interpretation": (
                    "the base Cui memory descriptor is superseded for this run by the explicit "
                    "repeater service profile; its aggregate P_link^0 and fixed memory inventory "
                    "are used without activating the Cui repeated-excitation controller. Any "
                    "profile-declared purification lanes are executed by the elementary-link "
                    "density instrument and its reservation adapter."
                ),
            }
        elif controller_model is None:
            memory_multiplexing = {
                "declared_model": declared_multiplexing,
                "active_in_compiled_kernel": False,
                "active_operating_mode": "single_mode_memory_emission",
                "interpretation": (
                    "the compiled optical instrument uses memory.parameters.efficiency per excitation; "
                    "the declared hybrid/time-bin multiplexing enhancement is not silently folded into "
                    "that probability"
                ),
            }
        else:
            memory_multiplexing = {
                "declared_model": declared_multiplexing,
                "active_in_compiled_kernel": True,
                "active_operating_mode": "independent_synchronized_time_bin_controller",
                "controller_path": self.config.multiplexing_controller_path,
                "controller_hash": generator.stable_hash(multiplexing_controller),
                "controller": controller_model.to_dict(),
                "interpretation": (
                    "the one-mode optical instrument remains unchanged; the controller folds iid, "
                    "mode-invariant, independently resourced time bins into a constant-duration round"
                ),
            }

        self._device_adherence = {
            "status": "blocked" if adherence_blockers else (
                "aggregated_empirical_link_stack"
                if empirical_pair_instrument is not None else
                "explicit_reference_override"
                if resolved_detector_kind == "ideal_pnr" else
                "explicit_device_stack_with_recorded_approximations"
            ),
            "resolved_detector_kind": resolved_detector_kind,
            "detector_profile_path": self.config.detector_profile_path,
            "detector_profile_reference": (
                detector_profile.get("reference_name") if detector_profile is not None else None
            ),
            "detector_profile_hash": generator.stable_hash(detector_profile),
            "detector_response_hash": generator.stable_hash(detector_response),
            "fiber_response_hash": generator.stable_hash(fiber_response),
            "conversion_response_path": self.config.conversion_response_path,
            "conversion_response_hash": generator.stable_hash(conversion_response),
            "repeater_service_profile_path": self.config.repeater_service_profile_path,
            "repeater_service_profile_hash": generator.stable_hash(repeater_service_profile),
            "repeater_service_profile": front.get("repeater_service_profile"),
            "memory_multiplexing": memory_multiplexing,
            "allow_ideal_detector_reference": self.config.allow_ideal_detector_reference,
            "require_explicit_device_stack": self.config.require_explicit_device_stack,
            "wavelength_compatibility": wavelength_report,
            "blockers": adherence_blockers,
        }
        if adherence_blockers:
            details = "; ".join(
                f"{row.get('effect', row.get('parameter', 'device_stack'))} "
                f"({row.get('parameter', 'unspecified_parameter')}): "
                f"{row.get('reason', 'unsupported')}"
                for row in adherence_blockers
            )
            raise CompiledHistogramUnsupported(
                "compiled histogram device-adherence check failed: " + details
            )
        protocol = generator.ProtocolConfig(requested_pairs=int(self.config.compile_key_length))
        self._generator = generator
        self._front = front
        self._context = context
        self._protocol = protocol
        self._front_hash = str(front["front_matter_hash"])
        self._schema_hash = generator.stable_hash({
            "transcript_schema": generator.transcript_schema(),
            "generator_version": generator.GENERATOR_VERSION,
            "compiled_kernel_version": generator.COMPILED_KERNEL_VERSION,
            "compiled_histogram_runtime_version": (
                COMPILED_HISTOGRAM_RUNTIME_VERSION
            ),
            "backend_config": self.config.to_dict(),
            "model_context_hash": generator.stable_hash(context.to_dict()),
        })
        return generator

    @property
    def schema_hash(self) -> str:
        self._ensure_ready()
        assert self._schema_hash is not None
        return self._schema_hash

    def _abort_thresholds(self) -> tuple[float, float]:
        self._ensure_ready()
        qber = (
            float(self._protocol.qber_abort_threshold)
            if self.config.qber_abort_threshold is None
            else float(self.config.qber_abort_threshold)
        )
        chsh = (
            float(self._protocol.chsh_abort_threshold)
            if self.config.chsh_abort_threshold is None
            else float(self.config.chsh_abort_threshold)
        )
        return qber, chsh

    # -- kernel / evaluated caches --------------------------------------------

    def kernel_for(self, hop_count: int, move: Any) -> Any:
        generator = self._ensure_ready()
        key = generator.stable_hash({
            "hop_count": int(hop_count),
            "move": move.to_dict(),
            "schema": self.schema_hash,
        })
        with self._cache_lock:
            cached = self._kernel_cache.get(key)
            if cached is not None:
                self._stats["kernel_cache_hits"] += 1
                return cached
        with _COMPILE_EVALUATE_LOCK:
            # A second learner thread may have populated the entry while this
            # thread waited for the generator-level cold-miss lock.
            with self._cache_lock:
                cached = self._kernel_cache.get(key)
                if cached is not None:
                    self._stats["kernel_cache_hits"] += 1
                    return cached
            started = time.perf_counter()
            try:
                kernel = generator.CompiledRouteKernel.compile(
                    front=self._front,
                    context=self._context,
                    hop_count=int(hop_count),
                    attack=move,
                    key_length=int(self.config.compile_key_length),
                )
            except NotImplementedError as exc:
                message = f"compiled histogram backend unsupported for this configuration: {exc}"
                with self._cache_lock:
                    if message not in self._warnings:
                        self._warnings.append(message)
                raise CompiledHistogramUnsupported(message) from exc
            elapsed = time.perf_counter() - started
            with self._cache_lock:
                self._stats["kernel_compiles"] += 1
                self._stats["kernel_compile_seconds"] += elapsed
                self._kernel_cache[key] = kernel
            return kernel

    def evaluate(self, hop_count: int, move: Any, fiber_lengths_m: Sequence[float]) -> _EvaluatedHistogram:
        generator = self._ensure_ready()
        kernel = self.kernel_for(hop_count, move)
        lengths = tuple(float(length) for length in fiber_lengths_m)
        cache_key = generator.stable_hash({
            "theta_hash": kernel.theta_hash,
            "fiber_lengths_m": list(lengths),
            "schema": self.schema_hash,
        })
        with self._cache_lock:
            cached = self._evaluated_cache.get(cache_key)
            if cached is not None:
                self._evaluated_cache.move_to_end(cache_key)
                self._stats["evaluated_cache_hits"] += 1
                return cached
        with _COMPILE_EVALUATE_LOCK:
            with self._cache_lock:
                cached = self._evaluated_cache.get(cache_key)
                if cached is not None:
                    self._evaluated_cache.move_to_end(cache_key)
                    self._stats["evaluated_cache_hits"] += 1
                    return cached
                self._stats["evaluated_cache_misses"] += 1
            started = time.perf_counter()
            try:
                distribution = kernel.evaluate(list(lengths))
            except NotImplementedError as exc:
                message = f"compiled histogram backend unsupported for this configuration: {exc}"
                with self._cache_lock:
                    if message not in self._warnings:
                        self._warnings.append(message)
                raise CompiledHistogramUnsupported(message) from exc
            elapsed = time.perf_counter() - started
            prepared = self._prepare_rows(
                generator,
                kernel.theta_hash,
                lengths,
                distribution,
                elapsed,
            )
            with self._cache_lock:
                self._stats["evaluate_seconds"] += elapsed
                self._evaluated_cache[cache_key] = prepared
                while len(self._evaluated_cache) > max(
                    1,
                    int(self.config.evaluated_cache_size),
                ):
                    self._evaluated_cache.popitem(last=False)
                    self._stats["evaluated_cache_evictions"] += 1
            return prepared

    def _prepare_rows(
        self,
        generator: Any,
        theta_hash: str,
        lengths: tuple[float, ...],
        distribution: Any,
        evaluate_seconds: float,
    ) -> _EvaluatedHistogram:
        rows = distribution.rows
        protocol = self._protocol
        chsh_pair_labels = [f"{a}|{b}" for a, b in protocol.chsh_pairs]
        chsh_pair_position = {label: idx for idx, label in enumerate(chsh_pair_labels)}
        pair_labels: list[str] = []
        pair_position: dict[str, int] = {}
        eve_labels: list[str] = []
        eve_position: dict[str, int] = {}
        stage_labels: list[str] = []
        stage_position: dict[str, int] = {}
        n = len(rows)
        probabilities = np.zeros(n)
        delivered_mask = np.zeros(n, dtype=bool)
        pair_index = np.full(n, -1, dtype=np.int64)
        is_key_pair_row = np.zeros(n, dtype=bool)
        is_key_error_row = np.zeros(n, dtype=bool)
        is_chsh_row = np.zeros(n, dtype=bool)
        chsh_pair_index = np.full(n, -1, dtype=np.int64)
        chsh_same_outcome = np.zeros(n, dtype=bool)
        eve_index = np.full(n, -1, dtype=np.int64)
        stage_index = np.full(n, -1, dtype=np.int64)
        for row_idx, row in enumerate(rows):
            probabilities[row_idx] = float(row["probability"])
            if row["delivery_status"] == "delivered":
                delivered_mask[row_idx] = True
                label = f"{row['alice_setting_label']}|{row['bob_setting_label']}"
                if label not in pair_position:
                    pair_position[label] = len(pair_labels)
                    pair_labels.append(label)
                pair_index[row_idx] = pair_position[label]
                if row["sifting_class"] == "key":
                    is_key_pair_row[row_idx] = True
                    is_key_error_row[row_idx] = generator.is_key_error(row, protocol)
                elif label in chsh_pair_position:
                    is_chsh_row[row_idx] = True
                    chsh_pair_index[row_idx] = chsh_pair_position[label]
                    chsh_same_outcome[row_idx] = (
                        int(row["alice_outcome"]) == int(row["bob_outcome"])
                    )
                eve_label = str(row["eve_observation_record"])
                if eve_label != "none":
                    if eve_label not in eve_position:
                        eve_position[eve_label] = len(eve_labels)
                        eve_labels.append(eve_label)
                    eve_index[row_idx] = eve_position[eve_label]
            else:
                stage = str(row["failure_stage"])
                if stage not in stage_position:
                    stage_position[stage] = len(stage_labels)
                    stage_labels.append(stage)
                stage_index[row_idx] = stage_position[stage]
        return _EvaluatedHistogram(
            theta_hash=theta_hash,
            fiber_lengths_m=lengths,
            probabilities=probabilities,
            delivered_mask=delivered_mask,
            pair_labels=pair_labels,
            pair_index=pair_index,
            is_key_pair_row=is_key_pair_row,
            is_key_error_row=is_key_error_row,
            is_chsh_row=is_chsh_row,
            chsh_pair_index=chsh_pair_index,
            chsh_same_outcome=chsh_same_outcome,
            eve_labels=eve_labels,
            eve_index=eve_index,
            no_delivery_stage_labels=stage_labels,
            no_delivery_stage_index=stage_index,
            expected_metrics=dict(distribution.metadata["expected_metrics"]),
            delivery_model=dict(distribution.metadata["delivery_model"]),
            waiting_metric_certification=dict(
                distribution.metadata.get("waiting_metric_certification", {})
            ),
            evaluate_seconds=evaluate_seconds,
            row_count=n,
        )

    # -- stepping --------------------------------------------------------------

    def step(
        self,
        *,
        ir_dict: dict[str, Any],
        route: dict[str, Any],
        action: dict[str, Any],
        requested_pairs: int,
        seed: int,
        step_index: int,
    ) -> CompiledHistogramStepResult:
        generator = self._ensure_ready()
        lengths = fiber_vector_from_route(ir_dict, route)
        move, on_route = eve_move_from_runtime_action(action, route)
        hop_count = len(lengths)
        parallel_eve_prepared: _EvaluatedHistogram | None = None
        if (
            on_route
            and move.move_kind == "repeater_bsm_stabilizer_probe"
        ):
            # Keep Alice's service histogram byte-for-byte on the no-move
            # marginal. The probe kernel supplies only a parallel conditional
            # mini-histogram for Eve's retained ancilla record. This prevents
            # harmless Eve-label row refinement from changing Alice's seeded
            # multinomial coupling.
            parallel_eve_prepared = self.evaluate(
                hop_count,
                move,
                lengths,
            )
            alice_move = generator.EveMove(
                move_id="no_move_baseline",
                move_kind="none",
                target_active=False,
                histogram_source="no_move_baseline",
            )
            prepared = self.evaluate(hop_count, alice_move, lengths)
            result_theta_hash = parallel_eve_prepared.theta_hash
        else:
            prepared = self.evaluate(hop_count, move, lengths)
            result_theta_hash = prepared.theta_hash
        protocol = self._protocol
        qber_threshold, chsh_threshold = self._abort_thresholds()

        started = time.perf_counter()
        rng = np.random.default_rng([int(seed), int(step_index)])
        counts = rng.multinomial(int(requested_pairs), prepared.probabilities)
        delivered_pair_count = int(counts[prepared.delivered_mask].sum())
        key_pair_count = int(counts[prepared.is_key_pair_row].sum())
        key_error_count = int(counts[prepared.is_key_pair_row & prepared.is_key_error_row].sum())
        chsh_pair_count = int(counts[prepared.is_chsh_row].sum())
        discarded = delivered_pair_count - key_pair_count - chsh_pair_count
        setting_pair_counts: dict[str, int] = {}
        if delivered_pair_count > 0 and prepared.pair_labels:
            pair_selected = prepared.pair_index >= 0
            sums = np.bincount(
                prepared.pair_index[pair_selected],
                weights=counts[pair_selected],
                minlength=len(prepared.pair_labels),
            )
            setting_pair_counts = {
                label: int(value)
                for label, value in zip(prepared.pair_labels, sums)
                if int(value) > 0
            }
        chsh_correlations: dict[str, float | None] = {}
        chsh_s_observed: float | None = None
        chsh_labels = [f"{a}|{b}" for a, b in protocol.chsh_pairs]
        chsh_totals = np.zeros(len(chsh_labels))
        chsh_same = np.zeros(len(chsh_labels))
        selected = prepared.chsh_pair_index >= 0
        if selected.any():
            chsh_totals = np.bincount(
                prepared.chsh_pair_index[selected],
                weights=counts[selected],
                minlength=len(chsh_labels),
            )
            chsh_same = np.bincount(
                prepared.chsh_pair_index[selected & prepared.chsh_same_outcome],
                weights=counts[selected & prepared.chsh_same_outcome],
                minlength=len(chsh_labels),
            )
        correlations: list[float | None] = []
        for label, total, same in zip(chsh_labels, chsh_totals, chsh_same):
            if total > 0:
                value = (2.0 * float(same) - float(total)) / float(total)
            else:
                value = None
            chsh_correlations[label] = value
            correlations.append(value)
        if all(value is not None for value in correlations):
            raw = correlations[0] + correlations[1] + correlations[2] - correlations[3]
            chsh_s_observed = -raw
        eve_observations: dict[str, int] = {}
        eve_selected = prepared.eve_index >= 0
        if eve_selected.any():
            eve_sums = np.bincount(
                prepared.eve_index[eve_selected],
                weights=counts[eve_selected],
                minlength=len(prepared.eve_labels),
            )
            eve_observations = {
                label: int(value)
                for label, value in zip(prepared.eve_labels, eve_sums)
                if int(value) > 0
            }
        if parallel_eve_prepared is not None:
            eve_observations = _sample_parallel_eve_mini_histogram(
                parallel_eve_prepared,
                delivered_pair_count=delivered_pair_count,
                seed=int(seed),
                step_index=int(step_index),
            )
        stage_counts: dict[str, int] = {}
        stage_selected = prepared.no_delivery_stage_index >= 0
        if stage_selected.any():
            stage_sums = np.bincount(
                prepared.no_delivery_stage_index[stage_selected],
                weights=counts[stage_selected],
                minlength=len(prepared.no_delivery_stage_labels),
            )
            stage_counts = {
                label: int(value)
                for label, value in zip(prepared.no_delivery_stage_labels, stage_sums)
                if int(value) > 0
            }

        sifted_key_count = key_pair_count
        qber_key = (key_error_count / sifted_key_count) if sifted_key_count > 0 else None
        expected_qber_from_histogram = prepared.expected_metrics.get(
            "expected_qber_from_histogram",
            prepared.expected_metrics.get("expected_qber"),
        )
        qber_key_mass_expected = float(
            prepared.expected_metrics.get(
                "qber_key_probability_mass",
                prepared.expected_metrics.get("qber_denominator_probability", 0.0),
            )
        )
        qber_error_mass_expected = float(
            prepared.expected_metrics.get(
                "qber_error_probability_mass",
                prepared.expected_metrics.get("qber_numerator_probability", 0.0),
            )
        )
        qber_estimation_power = classify_qber_estimation_power(
            expected_key_mass=qber_key_mass_expected,
            sampled_qber_trials=sifted_key_count,
            adequate_sampled_trials=int(self.config.min_key_pairs_for_qber_abort),
        )
        qber_abort = (
            qber_key is not None
            and sifted_key_count >= int(self.config.min_key_pairs_for_qber_abort)
            and qber_key > qber_threshold
        )
        chsh_abort = (
            not qber_abort
            and chsh_s_observed is not None
            and chsh_pair_count >= int(self.config.min_chsh_pairs_for_chsh_abort)
            and chsh_s_observed < chsh_threshold
        )
        if delivered_pair_count <= 0:
            public_outcome = "delivery_failure"
        elif qber_abort:
            public_outcome = "qber_abort"
        elif chsh_abort:
            public_outcome = "chsh_abort"
        else:
            public_outcome = "accepted"
        eve_metric_source = parallel_eve_prepared or prepared
        eve_guess_probability = float(
            eve_metric_source.expected_metrics.get(
                "eve_guess_probability",
                0.5,
            )
        )
        exposure_capable = (
            bool(move.target_active)
            and move.move_kind in _INFORMATION_EXPOSING_MOVE_KINDS
            and float(move.information_probability) > 0.0
        )
        information_exposed = (
            public_outcome == "accepted"
            and sifted_key_count > 0
            and eve_guess_probability > 0.5 + 1e-12
            and exposure_capable
        )
        secure_delivery = public_outcome == "accepted" and not information_exposed
        eve_win = public_outcome != "accepted" or information_exposed
        reward_components = {
            "delivery_failure": public_outcome == "delivery_failure",
            "qber_abort": bool(qber_abort),
            "chsh_abort": bool(chsh_abort),
            "information_exposure": bool(information_exposed),
            "secure_delivery": bool(secure_delivery),
        }
        with self._cache_lock:
            self._stats["steps_sampled"] += 1
            self._stats["step_sample_seconds"] += time.perf_counter() - started
        provenance = {
            "backend": COMPILED_HISTOGRAM_BACKEND,
            "backend_scope": BACKEND_SCOPE,
            "no_sequence_runtime": True,
            "generator_version": generator.GENERATOR_VERSION,
            "compiled_kernel_version": generator.COMPILED_KERNEL_VERSION,
            "compiled_histogram_runtime_version": (
                COMPILED_HISTOGRAM_RUNTIME_VERSION
            ),
            "theta_hash": result_theta_hash,
            "alice_histogram_theta_hash": prepared.theta_hash,
            "eve_observation_histogram_theta_hash": (
                parallel_eve_prepared.theta_hash
                if parallel_eve_prepared is not None
                else prepared.theta_hash
            ),
            "parallel_eve_observation_histogram": (
                parallel_eve_prepared is not None
            ),
            "parallel_eve_observation_conditioning": (
                "ancilla label counts conditioned on Alice delivered-pair "
                "count; Alice transcript sampled from unchanged no-move "
                "histogram"
                if parallel_eve_prepared is not None
                else "joint compiled histogram"
            ),
            "schema_hash": self.schema_hash,
            "front_matter_hash": self._front_hash,
            "fiber_lengths_m": list(lengths),
            "move_on_route": bool(on_route),
            "off_route_move_canonicalized_to_baseline": not on_route,
            "expected_delivery_probability": float(
                prepared.expected_metrics.get("delivery_probability", 0.0)
            ),
            "expected_chsh_S": prepared.expected_metrics.get("expected_chsh_S"),
            "expected_qber": prepared.expected_metrics.get("expected_qber"),
            "expected_qber_from_histogram": expected_qber_from_histogram,
            "sampled_qber_mean": qber_key,
            "sampled_qber_trials": 1 if qber_key is not None else 0,
            "sampled_qber_observable": qber_key is not None,
            "qber_key_mass_expected": qber_key_mass_expected,
            "qber_error_mass_expected": qber_error_mass_expected,
            "qber_estimation_power": qber_estimation_power,
            "qber_abort_threshold": qber_threshold,
            "chsh_abort_threshold": chsh_threshold,
            "min_key_pairs_for_qber_abort": int(
                self.config.min_key_pairs_for_qber_abort
            ),
            "min_chsh_pairs_for_chsh_abort": int(
                self.config.min_chsh_pairs_for_chsh_abort
            ),
            "qber_diagnostics": prepared.expected_metrics.get("qber_diagnostics"),
            "seed": int(seed),
            "step_index": int(step_index),
        }
        return CompiledHistogramStepResult(
            backend=COMPILED_HISTOGRAM_BACKEND,
            backend_scope=BACKEND_SCOPE,
            no_sequence_runtime=True,
            route_id=str(route["route_id"]),
            action_id=str(action.get("id") or action.get("attack_type") or "no_attack"),
            move_id=str(move.move_id),
            move_kind=str(move.move_kind),
            target_kind=str(move.target_kind),
            target_id=str(move.target_id),
            target_active=bool(move.target_active),
            fiber_lengths_m=list(lengths),
            hop_count=hop_count,
            theta_hash=result_theta_hash,
            requested_pair_count=int(requested_pairs),
            delivered_pair_count=delivered_pair_count,
            delivery_probability_expected=float(
                prepared.expected_metrics.get("delivery_probability", 0.0)
            ),
            setting_pair_counts=setting_pair_counts,
            key_pair_count=key_pair_count,
            chsh_pair_count=chsh_pair_count,
            discarded_setting_pair_count=int(discarded),
            sifted_key_count=sifted_key_count,
            key_error_count=key_error_count,
            qber_key=qber_key,
            expected_qber_from_histogram=expected_qber_from_histogram,
            sampled_qber_mean=qber_key,
            sampled_qber_trials=1 if qber_key is not None else 0,
            sampled_qber_observable=qber_key is not None,
            qber_key_mass_expected=qber_key_mass_expected,
            qber_error_mass_expected=qber_error_mass_expected,
            qber_estimation_power=qber_estimation_power,
            qber_abort_threshold=qber_threshold,
            chsh_abort_threshold=chsh_threshold,
            min_key_pairs_for_qber_abort=int(
                self.config.min_key_pairs_for_qber_abort
            ),
            min_chsh_pairs_for_chsh_abort=int(
                self.config.min_chsh_pairs_for_chsh_abort
            ),
            qber_abort=bool(qber_abort),
            chsh_S_observed=chsh_s_observed,
            chsh_S_expected=prepared.expected_metrics.get("expected_chsh_S"),
            chsh_setting_total_counts={
                label: int(value)
                for label, value in zip(chsh_labels, chsh_totals)
            },
            chsh_setting_same_counts={
                label: int(value)
                for label, value in zip(chsh_labels, chsh_same)
            },
            chsh_correlations_observed=chsh_correlations,
            chsh_abort=bool(chsh_abort),
            public_outcome=public_outcome,
            no_delivery_stage_counts=stage_counts,
            eve_observation_count=int(sum(eve_observations.values())),
            eve_observation_distribution=dict(sorted(eve_observations.items())),
            eve_guess_probability=eve_guess_probability,
            information_exposed=bool(information_exposed),
            secure_delivery=bool(secure_delivery),
            eve_win=bool(eve_win),
            reward_components=reward_components,
            provenance=provenance,
        )

    # -- diagnostics / verification ---------------------------------------------

    def cache_state(self) -> dict[str, Any]:
        """Return constant-cost cache counters without traversing cache objects."""

        self._ensure_ready()
        with self._cache_lock:
            return {
                "kernel_cache_size": len(self._kernel_cache),
                "evaluated_cache_size": len(self._evaluated_cache),
                "stats": dict(self._stats),
            }

    def diagnostics(self) -> dict[str, Any]:
        self._ensure_ready()
        with self._cache_lock:
            evaluated_bytes = _recursive_python_size_bytes(self._evaluated_cache)
            kernel_bytes = _recursive_python_size_bytes(self._kernel_cache)
            combined_bytes = _recursive_python_size_bytes(
                (self._kernel_cache, self._evaluated_cache)
            )
            kernel_cache_size = len(self._kernel_cache)
            evaluated_cache_size = len(self._evaluated_cache)
            stats = dict(self._stats)
            warnings = list(self._warnings)
        return {
            "backend": COMPILED_HISTOGRAM_BACKEND,
            "backend_scope": BACKEND_SCOPE,
            "generator_version": self._generator.GENERATOR_VERSION,
            "compiled_kernel_version": self._generator.COMPILED_KERNEL_VERSION,
            "compiled_histogram_runtime_version": (
                COMPILED_HISTOGRAM_RUNTIME_VERSION
            ),
            "front_matter_hash": self._front_hash,
            "schema_hash": self.schema_hash,
            "backend_config": self.config.to_dict(),
            "device_adherence": self._device_adherence,
            "kernel_cache_size": kernel_cache_size,
            "evaluated_cache_size": evaluated_cache_size,
            "cache_memory": {
                "method": (
                    "recursive sys.getsizeof over cache-owned Python objects "
                    "and NumPy buffers; allocator overhead and shared process "
                    "state are not included"
                ),
                "evaluated_cache_bytes": evaluated_bytes,
                "kernel_cache_bytes": kernel_bytes,
                "combined_cache_bytes_deduplicated": combined_bytes,
                "configured_evaluated_entry_limit": int(
                    self.config.evaluated_cache_size
                ),
            },
            "stats": stats,
            "unsupported_warnings": warnings,
        }

    def reference_check(
        self,
        hop_count: int,
        move: Any,
        fiber_length_vectors: Sequence[Sequence[float]],
        *,
        tolerance: float = 1e-9,
    ) -> dict[str, Any]:
        """Compiled-vs-frozen-reference equivalence report for this backend's theta."""
        generator = self._ensure_ready()
        kernel = self.kernel_for(hop_count, move)
        return generator.verify_compiled_kernel(
            kernel,
            fiber_length_vectors,
            tolerance=tolerance,
        )
