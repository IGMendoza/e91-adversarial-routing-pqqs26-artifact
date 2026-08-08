"""Route-calibrated Cui rolling physics for histogram bandit turns."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from sequence_game.corpus.compiled_histogram_backend import (
    COMPILED_HISTOGRAM_BACKEND,
    COMPILED_HISTOGRAM_RUNTIME_VERSION,
    CompiledHistogramBackend,
    CompiledHistogramBackendConfig,
    CompiledHistogramStepResult,
    eve_move_from_runtime_action,
    fiber_vector_from_route,
)
from sequence_game.physical.repeater_reservation import (
    rolling_config_for_service_profile,
)
from sequence_game.physical.repeater_service import (
    RepeaterServiceProfile,
    load_repeater_service_profile,
)

from .config import HistogramCoLearningConfig
from .corpus import HistogramGameCase


# Empirical cache-retention calibration on the production 15-hop Tatanid route
# with 93 canonical all-attack moves: 16,639,568 recursively counted bytes, or
# 178,920 bytes/move.  Round upward to 192 KiB for a transparent preflight
# estimate.  Runtime diagnostics measure the actual resident cache objects.
CACHE_ENTRY_RAM_ESTIMATE_BYTES = 192 * 1024
CACHE_ENTRY_RAM_ESTIMATE_SOURCE = (
    "2026-07-14 production 15-hop Tatanid route, fock_cutoff=3, "
    "rolling_age_evaluation=moment_reduced; 93 entries measured"
)


@dataclass(frozen=True)
class RoutePhysicsCalibration:
    route_id: str
    hop_count: int
    fiber_lengths_m: tuple[float, ...]
    total_distance_km: float
    service_rate_hz: float
    mean_distribution_time_s: float
    clean_delivery_probability: float
    clean_expected_chsh_s: float | None
    clean_expected_qber: float | None
    route_availability: float
    calculation_mode: str
    exact_for_declared_model: bool
    memory_regime: str
    profile_id: str
    scope: str
    claim_tier: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "hop_count": self.hop_count,
            "fiber_lengths_m": list(self.fiber_lengths_m),
            "total_distance_km": self.total_distance_km,
            "service_rate_hz": self.service_rate_hz,
            "mean_distribution_time_s": self.mean_distribution_time_s,
            "clean_delivery_probability": self.clean_delivery_probability,
            "clean_expected_chsh_s": self.clean_expected_chsh_s,
            "clean_expected_qber": self.clean_expected_qber,
            "route_availability": self.route_availability,
            "calculation_mode": self.calculation_mode,
            "exact_for_declared_model": self.exact_for_declared_model,
            "memory_regime": self.memory_regime,
            "profile_id": self.profile_id,
            "scope": self.scope,
            "claim_tier": self.claim_tier,
        }


def histogram_cache_plan(
    case: HistogramGameCase,
    config: HistogramCoLearningConfig,
) -> dict[str, Any]:
    """Preflight the per-route physics-cache demand for an action surface.

    Global route/action pairs overstate cache demand because every route owns
    its own backend and all off-route actions canonicalize to one no-move
    kernel.  Counting canonical chain-local moves gives the exact number of
    cache entries needed if every declared action is encountered.
    """

    actions = case.actions(config.game_mode)
    unique_by_route: list[int] = []
    for route in case.routes:
        canonical = {
            json.dumps(
                eve_move_from_runtime_action(action, route)[0].to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            )
            for action in actions
        }
        unique_by_route.append(len(canonical))
    configured = int(config.evaluated_cache_size)
    maximum = max(unique_by_route, default=0)
    estimated_bytes = int(sum(unique_by_route) * CACHE_ENTRY_RAM_ESTIMATE_BYTES)
    return {
        "cache_ownership": "one bounded evaluated-histogram LRU per route",
        "route_count": len(case.routes),
        "declared_action_count": len(actions),
        "global_route_action_pair_count": len(case.routes) * len(actions),
        "canonical_unique_move_count_total": int(sum(unique_by_route)),
        "canonical_unique_move_count_per_route": unique_by_route,
        "maximum_canonical_moves_on_one_route": int(maximum),
        "configured_entries_per_route": configured,
        "configured_entry_capacity_all_routes": configured * len(case.routes),
        "sufficient_for_declared_action_surface_without_eviction": (
            configured >= maximum
        ),
        "minimum_entries_per_route_without_eviction": int(maximum),
        "overprovision_factor_at_max_route": (
            configured / maximum if maximum else None
        ),
        "estimated_full_action_surface_cache_bytes": estimated_bytes,
        "estimated_full_action_surface_cache_mib": (
            estimated_bytes / (1024.0 ** 2)
        ),
        "estimated_bytes_per_canonical_move": CACHE_ENTRY_RAM_ESTIMATE_BYTES,
        "ram_estimate_source": CACHE_ENTRY_RAM_ESTIMATE_SOURCE,
        "ram_boundary": (
            "entry counts are exact; the preflight byte estimate is an "
            "empirical planning value, while backend diagnostics report "
            "actual recursively counted resident cache objects"
        ),
    }


class HistogramPhysicsPool:
    """One fixed, route-specific backend per candidate route.

    The rolling reservation distribution depends on the complete fiber vector,
    so routes with the same hop count but different edge lengths must not share
    a backend configuration. Kernels and evaluated histograms remain cached
    inside each backend after the first route/action encounter.
    """

    def __init__(
        self,
        case: HistogramGameCase,
        config: HistogramCoLearningConfig,
    ) -> None:
        self.case = case
        self.config = config
        self.cache_plan = histogram_cache_plan(case, config)
        self.profile = load_repeater_service_profile(
            config.repeater_service_profile_path
        )
        self._backends: list[CompiledHistogramBackend] = []
        self._calibrations: list[RoutePhysicsCalibration] = []
        self._build_route_backends()

    @property
    def calibrations(self) -> tuple[RoutePhysicsCalibration, ...]:
        return tuple(self._calibrations)

    @property
    def reference_service_rate_hz(self) -> float:
        return max(row.service_rate_hz for row in self._calibrations)

    def step(
        self,
        route_index: int,
        action: dict[str, Any],
        *,
        seed: int,
        step_index: int,
    ) -> CompiledHistogramStepResult:
        return self._backends[route_index].step(
            ir_dict=self.case.ir_dict,
            route=self.case.routes[route_index],
            action=action,
            requested_pairs=self.config.requested_pairs_per_turn,
            seed=seed,
            step_index=step_index,
        )

    def diagnostics(self) -> dict[str, Any]:
        backend_rows: dict[str, Any] = {}
        total_stats: dict[str, float] = {}
        total_cache_bytes = 0
        total_evaluated_bytes = 0
        total_kernel_bytes = 0
        generator_versions: set[str] = set()
        compiled_kernel_versions: set[str] = set()
        runtime_versions: set[str] = set()
        for calibration, backend in zip(self._calibrations, self._backends):
            diagnostics = backend.diagnostics()
            generator_versions.add(str(diagnostics["generator_version"]))
            compiled_kernel_versions.add(
                str(diagnostics["compiled_kernel_version"])
            )
            runtime_versions.add(
                str(diagnostics["compiled_histogram_runtime_version"])
            )
            cache_memory = diagnostics.get("cache_memory") or {}
            total_cache_bytes += int(
                cache_memory.get("combined_cache_bytes_deduplicated", 0)
            )
            total_evaluated_bytes += int(
                cache_memory.get("evaluated_cache_bytes", 0)
            )
            total_kernel_bytes += int(cache_memory.get("kernel_cache_bytes", 0))
            stats = diagnostics.get("stats", {})
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    total_stats[key] = total_stats.get(key, 0.0) + float(value)
            backend_rows[calibration.route_id] = {
                "schema_hash": diagnostics.get("schema_hash"),
                "front_matter_hash": diagnostics.get("front_matter_hash"),
                "generator_version": diagnostics.get("generator_version"),
                "compiled_kernel_version": diagnostics.get(
                    "compiled_kernel_version"
                ),
                "compiled_histogram_runtime_version": diagnostics.get(
                    "compiled_histogram_runtime_version"
                ),
                "kernel_cache_size": diagnostics.get("kernel_cache_size"),
                "evaluated_cache_size": diagnostics.get("evaluated_cache_size"),
                "cache_memory": cache_memory,
                "stats": stats,
                "device_adherence_status": (
                    diagnostics.get("device_adherence") or {}
                ).get("status"),
                "device_adherence_blockers": (
                    diagnostics.get("device_adherence") or {}
                ).get("blockers", []),
            }
        if (
            len(generator_versions) != 1
            or len(compiled_kernel_versions) != 1
            or runtime_versions != {COMPILED_HISTOGRAM_RUNTIME_VERSION}
        ):
            raise RuntimeError(
                "route backends disagree on compiled histogram versions"
            )
        return {
            "backend": COMPILED_HISTOGRAM_BACKEND,
            "generator_version": next(iter(generator_versions)),
            "compiled_kernel_version": next(iter(compiled_kernel_versions)),
            "compiled_histogram_runtime_version": next(
                iter(runtime_versions)
            ),
            "profile_id": self.profile.profile_id,
            "profile_scope": self.profile.scope,
            "profile_claim_tier": self.profile.claim_tier,
            "route_calibrations": [row.to_dict() for row in self._calibrations],
            "cache_plan": dict(self.cache_plan),
            "cache_memory": {
                "combined_cache_bytes_sum_across_route_backends": total_cache_bytes,
                "evaluated_cache_bytes_sum_across_route_backends": (
                    total_evaluated_bytes
                ),
                "kernel_cache_bytes_sum_across_route_backends": total_kernel_bytes,
                "combined_cache_mib": total_cache_bytes / (1024.0 ** 2),
                "measurement_boundary": (
                    "sum of per-backend recursive cache estimates; shared "
                    "process state and allocator overhead are excluded"
                ),
            },
            "stats": total_stats,
            "routes": backend_rows,
            "known_detector_approximation": {
                "active": bool(
                    self.config.allow_known_1550_to_1558_detector_approximation
                ),
                "semantics": (
                    "selected scalar PNR detector operating point is measured at "
                    "1550 nm and applied at the Cui 1558 nm converted output; no "
                    "eight-nanometre response curve is registered"
                ),
            },
        }

    def cache_state(self) -> dict[str, Any]:
        """Return cache entry counts and timing counters without a RAM walk."""

        routes: dict[str, Any] = {}
        total_kernel_entries = 0
        total_evaluated_entries = 0
        total_stats: dict[str, float] = {}
        for calibration, backend in zip(self._calibrations, self._backends):
            state = backend.cache_state()
            kernel_entries = int(state["kernel_cache_size"])
            evaluated_entries = int(state["evaluated_cache_size"])
            total_kernel_entries += kernel_entries
            total_evaluated_entries += evaluated_entries
            for key, value in state["stats"].items():
                if isinstance(value, (int, float)):
                    total_stats[key] = total_stats.get(key, 0.0) + float(value)
            routes[calibration.route_id] = {
                "kernel_cache_size": kernel_entries,
                "evaluated_cache_size": evaluated_entries,
            }
        return {
            "kernel_cache_entries": total_kernel_entries,
            "evaluated_cache_entries": total_evaluated_entries,
            "stats": total_stats,
            "routes": routes,
        }

    def _build_route_backends(self) -> None:
        for route in self.case.routes:
            lengths_m = tuple(
                fiber_vector_from_route(self.case.ir_dict, route)
            )
            self._validate_route(lengths_m)
            payload = rolling_config_for_service_profile(
                self.profile,
                [value / 1000.0 for value in lengths_m],
                accepted_alive_time_s=self._accepted_alive_time_s(),
                selection_policy="youngest_live",
                num_age_points=self.config.age_points_per_edge,
            )
            backend_config = CompiledHistogramBackendConfig(
                repeater_service_profile_path=str(
                    self.config.repeater_service_profile_path
                ),
                allow_wavelength_mismatch=bool(
                    self.config.allow_known_1550_to_1558_detector_approximation
                ),
                fock_cutoff=self.config.fock_cutoff,
                evaluated_cache_size=self.config.evaluated_cache_size,
                waiting_protocol="rolling_alive_pair",
                rolling_edge_availability=tuple(
                    float(value)
                    for value in payload["rolling_edge_availability"]
                ),
                rolling_age_distribution_kind=str(
                    payload["rolling_age_distribution_kind"]
                ),
                rolling_age_distribution_points_s=tuple(
                    float(value)
                    for value in payload["rolling_age_distribution_points_s"]
                ),
                rolling_age_distribution_weights=tuple(
                    float(value)
                    for value in payload["rolling_age_distribution_weights"]
                ),
                rolling_edge_age_distribution_points_s=tuple(
                    tuple(float(value) for value in row)
                    for row in payload[
                        "rolling_edge_age_distribution_points_s"
                    ]
                ),
                rolling_edge_age_distribution_weights=tuple(
                    tuple(float(value) for value in row)
                    for row in payload[
                        "rolling_edge_age_distribution_weights"
                    ]
                ),
                rolling_age_evaluation="moment_reduced",
                rolling_accepted_alive_time_s=float(
                    payload["rolling_accepted_alive_time_s"]
                ),
                rolling_memory_slots_per_edge_side=int(
                    payload["rolling_memory_slots_per_edge_side"]
                ),
                rolling_selection_policy=str(
                    payload["rolling_selection_policy"]
                ),
                rolling_age_distribution_source=str(
                    payload["rolling_age_distribution_source"]
                ),
            )
            backend = CompiledHistogramBackend(backend_config)
            baseline, _ = eve_move_from_runtime_action(
                {
                    "id": "no_attack",
                    "kind": "none",
                    "target": "",
                    "attack_type": "no_attack",
                    "cost": 0.0,
                },
                route,
            )
            evaluated = backend.evaluate(len(lengths_m), baseline, lengths_m)
            route_service = payload["route_service"]
            self._backends.append(backend)
            self._calibrations.append(RoutePhysicsCalibration(
                route_id=str(route["route_id"]),
                hop_count=len(lengths_m),
                fiber_lengths_m=lengths_m,
                total_distance_km=float(route_service["total_distance_km"]),
                service_rate_hz=float(route_service["success_rate_hz"]),
                mean_distribution_time_s=float(
                    route_service["mean_distribution_time_s"]
                ),
                clean_delivery_probability=float(
                    evaluated.expected_metrics["delivery_probability"]
                ),
                clean_expected_chsh_s=_optional_float(
                    evaluated.expected_metrics.get("expected_chsh_S")
                ),
                clean_expected_qber=_optional_float(
                    evaluated.expected_metrics.get("expected_qber")
                ),
                route_availability=float(
                    payload["reservation_summary"]["route_availability"]
                ),
                calculation_mode=str(route_service["calculation_mode"]),
                exact_for_declared_model=bool(
                    route_service["exact_for_declared_model"]
                ),
                memory_regime=str(route_service["memory_regime"]),
                profile_id=self.profile.profile_id,
                scope=self.profile.scope,
                claim_tier=self.profile.claim_tier,
            ))

    def _accepted_alive_time_s(self) -> float:
        if self.config.accepted_alive_time_s is not None:
            return float(self.config.accepted_alive_time_s)
        return float(
            self.profile.controller_model.get(
                "accepted_output_alive_time_s", 0.1
            )
        )

    def _validate_route(self, lengths_m: tuple[float, ...]) -> None:
        validity = self.profile.validity
        lower = float(validity["edge_length_min_km"])
        upper = float(validity["edge_length_max_km"])
        outside = [
            value / 1000.0 for value in lengths_m
            if not lower <= value / 1000.0 <= upper
        ]
        if outside:
            raise ValueError(
                f"route edge lengths {outside} km are outside profile range "
                f"[{lower}, {upper}] km"
            )


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
