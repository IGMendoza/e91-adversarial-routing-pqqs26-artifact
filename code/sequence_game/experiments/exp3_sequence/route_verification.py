"""Route-level SeQUeNCe repeater verification for Exp3 corpus calibration."""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from sequence_game.corpus.e91_runtime_game import (
    _default_models,
    _multiplexed_memory_efficiency,
)
from sequence_game.eve.repeater_attacks import RepeaterAttackSpec
from sequence_game.physical import PhysicalModel
from sequence_game.protocol.postprocessing import SiftingConfig
from sequence_game.protocol.repeater_trial import (
    RepeaterE91RunConfig,
    run_fixed_repeater_chsh_trial,
)
from sequence_game.sequence_build.repeater_e91_builder import FixedRepeaterPath

from .config import (
    BASELINE_HEALTH_MIN_CHSH_S,
    BASELINE_HEALTH_MIN_DELIVERED_PAIRS,
    CHSH_PAIRS_PER_TRIAL,
    CORPUS_SQLITE_PATH,
    CORPUS_BASE_EDGE_LENGTH_M,
    CORPUS_EDGE_LENGTH_STEP_M,
    Exp3SequenceConfig,
    FIBER_CLASSICAL_SPEED_M_PER_PS,
    MAX_ROUTE_HOPS,
    QBER_THRESHOLD,
    REQUEST_FIDELITY,
    SEQUENCE_SETUP_TRAVERSALS,
    SEQUENCE_MEMORY_FIDELITY_OVERRIDE,
    SWAPPING_DEGRADATION,
    SWAPPING_SUCCESS_PROB,
)
from .corpus import (
    corpus_summary,
    load_graph_cases_from_sqlite,
    read_corpus_sqlite_metadata,
)
from .io import repo_metadata, write_json
from .models import apply_sequence_memory_fidelity_override
from .runner import _process_context


ROUTE_VERIFY_EDGE_LENGTH_M = CORPUS_BASE_EDGE_LENGTH_M + 4 * CORPUS_EDGE_LENGTH_STEP_M
ROUTE_VERIFY_FREQUENCY_SCALES = (1.0,)
ROUTE_VERIFY_FREQUENCY_EDGE_LENGTH_M = ROUTE_VERIFY_EDGE_LENGTH_M


@dataclass(frozen=True)
class RouteVerificationConfig:
    out_dir: Path = Path("runs/exp3_sequence_route_verify")
    corpus_db_path: Path = CORPUS_SQLITE_PATH
    max_hops: int = MAX_ROUTE_HOPS
    workers: int = 1
    seed: int = 42
    memory_pairs: int = CHSH_PAIRS_PER_TRIAL
    hop_counts: tuple[int, ...] = tuple(range(1, MAX_ROUTE_HOPS + 1))
    edge_lengths_m: tuple[float, ...] = (ROUTE_VERIFY_EDGE_LENGTH_M,)
    frequency_scales: tuple[float, ...] = ROUTE_VERIFY_FREQUENCY_SCALES
    frequency_benchmark_edge_length_m: float = ROUTE_VERIFY_FREQUENCY_EDGE_LENGTH_M
    qber_threshold: float = QBER_THRESHOLD
    request_fidelity: float = REQUEST_FIDELITY
    setup_traversals: float = SEQUENCE_SETUP_TRAVERSALS
    start_time_ps: int = 1_000_000
    window_ps: int = 1_000_000_000
    stop_margin_ps: int = 250_000_000
    swapping_success_prob: float = SWAPPING_SUCCESS_PROB
    swapping_degradation: float = SWAPPING_DEGRADATION
    sequence_memory_fidelity_override: float | None = SEQUENCE_MEMORY_FIDELITY_OVERRIDE
    min_chsh_s: float = BASELINE_HEALTH_MIN_CHSH_S
    min_delivered_pairs: int = BASELINE_HEALTH_MIN_DELIVERED_PAIRS
    length_sweep_frequency_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.max_hops < 1:
            raise ValueError("max_hops must be >= 1")
        if not self.hop_counts:
            raise ValueError("hop_counts must be non-empty")
        if any(hops < 1 for hops in self.hop_counts):
            raise ValueError("hop_counts entries must be >= 1")
        if any(hops > self.max_hops for hops in self.hop_counts):
            raise ValueError("hop_counts entries must be <= max_hops")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        if self.memory_pairs < 1:
            raise ValueError("memory_pairs must be >= 1")
        if not self.edge_lengths_m:
            raise ValueError("edge_lengths_m must be non-empty")
        if any(length <= 0 for length in self.edge_lengths_m):
            raise ValueError("edge_lengths_m entries must be > 0")
        if not self.frequency_scales:
            raise ValueError("frequency_scales must be non-empty")
        if any(scale <= 0 for scale in self.frequency_scales):
            raise ValueError("frequency_scales entries must be > 0")
        if self.frequency_benchmark_edge_length_m <= 0:
            raise ValueError("frequency_benchmark_edge_length_m must be > 0")
        if not 0 <= self.qber_threshold <= 1:
            raise ValueError("qber_threshold must be in [0, 1]")
        if self.setup_traversals < 1:
            raise ValueError("setup_traversals must be >= 1")
        if self.window_ps < 1:
            raise ValueError("window_ps must be >= 1")
        if self.stop_margin_ps < 0:
            raise ValueError("stop_margin_ps must be >= 0")
        if not 0 <= self.swapping_success_prob <= 1:
            raise ValueError("swapping_success_prob must be in [0, 1]")
        if not 0 <= self.swapping_degradation <= 1:
            raise ValueError("swapping_degradation must be in [0, 1]")
        if (
                self.sequence_memory_fidelity_override is not None
                and not 0 <= self.sequence_memory_fidelity_override <= 1
        ):
            raise ValueError("sequence_memory_fidelity_override must be in [0, 1]")
        if self.min_chsh_s < 0:
            raise ValueError("min_chsh_s must be >= 0")
        if self.min_delivered_pairs < 0:
            raise ValueError("min_delivered_pairs must be >= 0")

    @classmethod
    def from_exp3_config(cls, config: Exp3SequenceConfig) -> "RouteVerificationConfig":
        metadata = read_corpus_sqlite_metadata(config.corpus_db_path)
        max_hops = int(metadata.get("max_route_hops", MAX_ROUTE_HOPS))
        return cls(
            out_dir=config.out_dir / "route_verification",
            corpus_db_path=config.corpus_db_path,
            max_hops=max_hops,
            workers=config.workers,
            seed=config.seed,
            memory_pairs=config.chsh_pairs_per_trial,
            hop_counts=tuple(range(1, max_hops + 1)),
            qber_threshold=config.qber_threshold,
            request_fidelity=config.request_fidelity,
            setup_traversals=config.sequence_setup_traversals,
            start_time_ps=config.start_time_ps,
            window_ps=config.repeater_window_ps,
            stop_margin_ps=config.stop_margin_ps,
            swapping_success_prob=config.swapping_success_prob,
            swapping_degradation=config.swapping_degradation,
            sequence_memory_fidelity_override=config.sequence_memory_fidelity_override,
            min_chsh_s=config.baseline_health_min_chsh_s,
            min_delivered_pairs=config.baseline_health_min_delivered_pairs,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["out_dir"] = str(self.out_dir)
        data["corpus_db_path"] = str(self.corpus_db_path)
        data["hop_counts"] = list(self.hop_counts)
        data["edge_lengths_m"] = list(self.edge_lengths_m)
        data["frequency_scales"] = list(self.frequency_scales)
        return data


@dataclass(frozen=True)
class RouteVerificationCase:
    case_id: str
    stage: str
    hops: int
    edge_length_m: float
    memory_frequency_scale: float
    trial_index: int
    seed: int

    @property
    def total_length_m(self) -> float:
        return float(self.hops * self.edge_length_m)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "stage": self.stage,
            "hops": self.hops,
            "repeaters": max(0, self.hops - 1),
            "edge_length_m": self.edge_length_m,
            "total_length_m": self.total_length_m,
            "memory_frequency_scale": self.memory_frequency_scale,
            "trial_index": self.trial_index,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class RouteVerificationResult:
    case: RouteVerificationCase
    public_outcome: str
    accepted: bool
    qualified: bool
    delivered_count: int
    qber: float | None
    chsh_s: float | None
    chsh_adequately_sampled: bool | None
    wall_seconds: float
    simulated_window_seconds: float
    delivered_per_wall_second: float
    delivered_per_simulated_second: float
    timing: dict[str, Any]
    active_models: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "public_outcome": self.public_outcome,
            "accepted": self.accepted,
            "qualified": self.qualified,
            "delivered_count": self.delivered_count,
            "qber": self.qber,
            "chsh_s": self.chsh_s,
            "chsh_adequately_sampled": self.chsh_adequately_sampled,
            "wall_seconds": self.wall_seconds,
            "simulated_window_seconds": self.simulated_window_seconds,
            "delivered_per_wall_second": self.delivered_per_wall_second,
            "delivered_per_simulated_second": self.delivered_per_simulated_second,
            "timing": dict(self.timing),
            "active_models": dict(self.active_models),
            "error": self.error,
        }


TrialRunner = Callable[
    [RouteVerificationCase, RouteVerificationConfig],
    RouteVerificationResult,
]


def build_route_verification_cases(
        config: RouteVerificationConfig,
) -> list[RouteVerificationCase]:
    cases: list[RouteVerificationCase] = []
    trial_index = 0
    for hops in sorted(set(config.hop_counts)):
        for edge_length_m in sorted(config.edge_lengths_m, reverse=True):
            cases.append(RouteVerificationCase(
                case_id=(
                    f"hop_h{hops}_edge{edge_length_m:g}_"
                    f"freq{config.length_sweep_frequency_scale:g}"
                ),
                stage="hop_sanity",
                hops=hops,
                edge_length_m=float(edge_length_m),
                memory_frequency_scale=float(config.length_sweep_frequency_scale),
                trial_index=trial_index,
                seed=config.seed + 10_000 + trial_index,
            ))
            trial_index += 1

    diagnostic_scales = [
        scale for scale in config.frequency_scales
        if float(scale) != float(config.length_sweep_frequency_scale)
    ]
    if not diagnostic_scales:
        return cases

    benchmark_hops = _frequency_benchmark_hops(config.max_hops)
    for hops in benchmark_hops:
        for scale in diagnostic_scales:
            cases.append(RouteVerificationCase(
                case_id=(
                    f"frequency_h{hops}_edge{config.frequency_benchmark_edge_length_m:g}_"
                    f"freq{scale:g}"
                ),
                stage="frequency_benchmark",
                hops=hops,
                edge_length_m=float(config.frequency_benchmark_edge_length_m),
                memory_frequency_scale=float(scale),
                trial_index=trial_index,
                seed=config.seed + 10_000 + trial_index,
            ))
            trial_index += 1
    return cases


def run_route_verification(
        config: RouteVerificationConfig,
        *,
        progress: bool = True,
        trial_runner: TrialRunner | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = build_route_verification_cases(config)
    if progress:
        print(
            "route verification: "
            f"cases={len(cases)} max_hops={config.max_hops} "
            f"edge_lengths_m={list(config.edge_lengths_m)} "
            f"frequency_scales={list(config.frequency_scales)} "
            f"workers={config.workers}",
            flush=True,
        )

    runner = trial_runner or _run_sequence_route_case
    if trial_runner is not None or config.workers == 1 or len(cases) <= 1:
        results = []
        for index, case in enumerate(cases, 1):
            result = runner(case, config)
            results.append(result)
            if progress:
                _print_result(index, len(cases), result)
    else:
        results = _run_parallel(cases, config, progress=progress)

    payload = {
        "status": "complete",
        "config": config.to_dict(),
        "repo": repo_metadata(),
        "scope": {
            "sequence_backend": "repeater_e91_chsh",
            "route_verification_only": True,
            "graph_health_payoff_exp3_oracle_dt_not_run": True,
            "source_model_active_in_repeater_path": False,
            "active_generation_mechanism": "SeQUeNCe BarretKokA/B memory.excite",
            "frequency_benchmark_scope": (
                "diagnostic sensitivity; frequency scales other than 1.0 are "
                "not Cui-calibrated literature settings"
            ),
        },
        "corpus_depth_summary": _corpus_depth_summary(config),
        "results": [result.to_dict() for result in results],
        "summary": summarize_route_verification(results),
        "wall_seconds": time.perf_counter() - started,
    }
    write_json(out_dir / "route_verification.json", payload)
    if progress:
        print(f"wrote {out_dir / 'route_verification.json'}", flush=True)
    return payload


def summarize_route_verification(
        results: list[RouteVerificationResult],
) -> dict[str, Any]:
    length_results = [
        result for result in results
        if result.case.stage in {"hop_sanity", "length_sweep"}
    ]
    thresholds: dict[str, Any] = {}
    for hops in sorted({result.case.hops for result in length_results}):
        rows = [result for result in length_results if result.case.hops == hops]
        passing = [result for result in rows if result.qualified]
        thresholds[str(hops)] = {
            "max_passing_edge_length_m": (
                max(result.case.edge_length_m for result in passing)
                if passing else None
            ),
            "min_tested_edge_length_m": min(result.case.edge_length_m for result in rows),
            "max_tested_edge_length_m": max(result.case.edge_length_m for result in rows),
            "qualified_cases": len(passing),
            "tested_cases": len(rows),
        }

    frequency_rows: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        if result.case.stage != "frequency_benchmark":
            continue
        frequency_rows.setdefault(str(result.case.hops), []).append({
            "edge_length_m": result.case.edge_length_m,
            "memory_frequency_scale": result.case.memory_frequency_scale,
            "qualified": result.qualified,
            "delivered_count": result.delivered_count,
            "wall_seconds": result.wall_seconds,
            "delivered_per_wall_second": result.delivered_per_wall_second,
            "delivered_per_simulated_second": result.delivered_per_simulated_second,
            "chsh_s": result.chsh_s,
            "public_outcome": result.public_outcome,
        })
    for rows in frequency_rows.values():
        rows.sort(key=lambda row: row["memory_frequency_scale"])

    return {
        "length_thresholds_by_hop": thresholds,
        "hop_sanity_by_hop": thresholds,
        "frequency_benchmark_by_hop": frequency_rows,
        "qualified_count": sum(1 for result in results if result.qualified),
        "case_count": len(results),
        "error_count": sum(1 for result in results if result.error),
    }


def _run_parallel(
        cases: list[RouteVerificationCase],
        config: RouteVerificationConfig,
        *,
        progress: bool,
) -> list[RouteVerificationResult]:
    results_by_index: dict[int, RouteVerificationResult] = {}
    mp_context = _process_context()
    with ProcessPoolExecutor(max_workers=config.workers, mp_context=mp_context) as pool:
        futures = {
            pool.submit(_run_sequence_route_case, case, config): index
            for index, case in enumerate(cases)
        }
        for completed, future in enumerate(as_completed(futures), 1):
            index = futures[future]
            result = future.result()
            results_by_index[index] = result
            if progress:
                _print_result(completed, len(cases), result)
    return [results_by_index[index] for index in range(len(cases))]


def _run_sequence_route_case(
        case: RouteVerificationCase,
        config: RouteVerificationConfig,
) -> RouteVerificationResult:
    source_model, _detector_model, fiber_model, memory_model = _default_models()
    exp3_config = Exp3SequenceConfig(
        out_dir=config.out_dir,
        workers=config.workers,
        seed=config.seed,
        chsh_pairs_per_trial=config.memory_pairs,
        qber_threshold=config.qber_threshold,
        request_fidelity=config.request_fidelity,
        start_time_ps=config.start_time_ps,
        end_time_ps=config.start_time_ps + config.window_ps,
        stop_time_ps=config.start_time_ps + config.window_ps + config.stop_margin_ps,
        sequence_setup_traversals=config.setup_traversals,
        swapping_success_prob=config.swapping_success_prob,
        swapping_degradation=config.swapping_degradation,
        sequence_memory_fidelity_override=config.sequence_memory_fidelity_override,
        baseline_health_min_chsh_s=config.min_chsh_s,
        baseline_health_min_delivered_pairs=config.min_delivered_pairs,
    )
    memory_model = apply_sequence_memory_fidelity_override(memory_model, exp3_config)
    base_memory_frequency_hz = float(memory_model.parameters["frequency"])
    memory_frequency_hz = base_memory_frequency_hz * case.memory_frequency_scale
    memory_model = _override_model(
        memory_model,
        parameters={"frequency": memory_frequency_hz},
        suffix=f"freq_x{case.memory_frequency_scale:g}",
        diagnostic_scope=case.memory_frequency_scale != 1.0,
    )
    fiber_model = _override_model(
        fiber_model,
        parameters={
            "frequency": float(fiber_model.parameters["frequency"]) * case.memory_frequency_scale,
        },
        suffix=f"freq_x{case.memory_frequency_scale:g}",
        diagnostic_scope=case.memory_frequency_scale != 1.0,
    )
    memory_efficiency = _multiplexed_memory_efficiency(memory_model)
    started = time.perf_counter()
    path = fixed_route_path(case.hops, case.edge_length_m)
    run_config, timing = _run_config_for_case(
        case,
        config,
        fiber_model,
        memory_efficiency_override=memory_efficiency,
    )
    try:
        transcript, _built = run_fixed_repeater_chsh_trial(
            path,
            memory_model=memory_model,
            fiber_model=fiber_model,
            run_config=run_config,
            trial_id=f"route_verify_{case.case_id}",
            seed=case.seed,
            attack=RepeaterAttackSpec(),
        )
        wall = time.perf_counter() - started
        extra = getattr(transcript, "extra", {}) or {}
        outcome = "accepted" if transcript.accepted else (transcript.abort_reason or "delivery_failure")
        delivered = int(transcript.generation_successes)
        chsh_s = extra.get("chsh_s")
        qualified = (
            bool(transcript.accepted)
            and chsh_s is not None
            and float(chsh_s) >= config.min_chsh_s
            and delivered >= config.min_delivered_pairs
        )
        return RouteVerificationResult(
            case=case,
            public_outcome=str(outcome),
            accepted=bool(transcript.accepted),
            qualified=qualified,
            delivered_count=delivered,
            qber=transcript.qber_estimate,
            chsh_s=chsh_s,
            chsh_adequately_sampled=extra.get("chsh_adequately_sampled"),
            wall_seconds=wall,
            simulated_window_seconds=config.window_ps * 1e-12,
            delivered_per_wall_second=delivered / wall if wall > 0 else 0.0,
            delivered_per_simulated_second=delivered / (config.window_ps * 1e-12),
            timing=timing,
            active_models={
                "source_model_unused_by_repeater_path": True,
                "source_frequency_hz": float(source_model.parameters["frequency"]),
                "memory_original_fidelity": memory_model.parameters.get(
                    "sequence_runtime_original_fidelity",
                    memory_model.parameters.get("fidelity"),
                ),
                "memory_active_fidelity": float(memory_model.parameters["fidelity"]),
                "memory_fidelity_override": memory_model.parameters.get(
                    "sequence_runtime_fidelity_override"),
                "memory_frequency_hz": memory_frequency_hz,
                "memory_efficiency_override": memory_efficiency,
                "fiber_quantum_channel_frequency_hz": float(fiber_model.parameters["frequency"]),
            },
        )
    except Exception as exc:
        wall = time.perf_counter() - started
        return RouteVerificationResult(
            case=case,
            public_outcome="error",
            accepted=False,
            qualified=False,
            delivered_count=0,
            qber=None,
            chsh_s=None,
            chsh_adequately_sampled=None,
            wall_seconds=wall,
            simulated_window_seconds=config.window_ps * 1e-12,
            delivered_per_wall_second=0.0,
            delivered_per_simulated_second=0.0,
            timing=timing,
            active_models={
                "source_model_unused_by_repeater_path": True,
                "source_frequency_hz": float(source_model.parameters["frequency"]),
                "memory_original_fidelity": memory_model.parameters.get(
                    "sequence_runtime_original_fidelity",
                    memory_model.parameters.get("fidelity"),
                ),
                "memory_active_fidelity": float(memory_model.parameters["fidelity"]),
                "memory_fidelity_override": memory_model.parameters.get(
                    "sequence_runtime_fidelity_override"),
                "memory_frequency_hz": memory_frequency_hz,
                "memory_efficiency_override": memory_efficiency,
                "fiber_quantum_channel_frequency_hz": float(fiber_model.parameters["frequency"]),
            },
            error=f"{type(exc).__name__}: {exc}",
        )


def fixed_route_path(hops: int, edge_length_m: float) -> FixedRepeaterPath:
    if hops < 1:
        raise ValueError("hops must be >= 1")
    if edge_length_m <= 0:
        raise ValueError("edge_length_m must be > 0")
    nodes = ["alice"]
    nodes.extend(f"r{i}" for i in range(1, hops))
    nodes.append("bob")
    return FixedRepeaterPath(
        nodes=tuple(nodes),
        edge_lengths_m=tuple(float(edge_length_m) for _ in range(hops)),
    )


def _run_config_for_case(
        case: RouteVerificationCase,
        config: RouteVerificationConfig,
        fiber_model: PhysicalModel,
        *,
        memory_efficiency_override: float | None,
) -> tuple[RepeaterE91RunConfig, dict[str, Any]]:
    light_speed = float(
        fiber_model.parameters.get("light_speed", FIBER_CLASSICAL_SPEED_M_PER_PS)
    )
    one_way_ps = case.total_length_m / light_speed if light_speed > 0 else 0.0
    start_time_ps = max(
        int(config.start_time_ps),
        int(config.setup_traversals * one_way_ps),
    )
    end_time_ps = start_time_ps + int(config.window_ps)
    stop_time_ps = end_time_ps + int(config.stop_margin_ps)
    return (
        RepeaterE91RunConfig(
            memory_pairs=config.memory_pairs,
            sifting=SiftingConfig(
                qber_threshold=config.qber_threshold,
                min_sifted_samples=1,
            ),
            request_fidelity=config.request_fidelity,
            start_time_ps=start_time_ps,
            end_time_ps=end_time_ps,
            stop_time_ps=stop_time_ps,
            swapping_success_prob=config.swapping_success_prob,
            swapping_degradation=config.swapping_degradation,
            memory_efficiency_override=memory_efficiency_override,
        ),
        {
            "start_time_ps": start_time_ps,
            "end_time_ps": end_time_ps,
            "stop_time_ps": stop_time_ps,
            "route_total_length_m": case.total_length_m,
            "route_edge_length_m": case.edge_length_m,
            "route_hops": case.hops,
            "one_way_classical_ps": one_way_ps,
            "setup_traversals": config.setup_traversals,
            "window_ps": config.window_ps,
            "stop_margin_ps": config.stop_margin_ps,
        },
    )


def _override_model(
        model: PhysicalModel,
        *,
        parameters: dict[str, Any],
        suffix: str,
        diagnostic_scope: bool,
) -> PhysicalModel:
    updated = dict(model.parameters)
    updated.update(parameters)
    return PhysicalModel(
        model_name=f"{model.model_name}_{suffix}" if diagnostic_scope else model.model_name,
        device_kind=model.device_kind,
        scope="toy" if diagnostic_scope else model.scope,
        reference_tag=(
            f"{model.reference_tag}; diagnostic scaled parameter {suffix}"
            if diagnostic_scope else model.reference_tag
        ),
        parameters=updated,
        units=dict(model.units),
        notes=(
            f"{model.notes} Diagnostic frequency sensitivity run; not a literature calibration."
            if diagnostic_scope else model.notes
        ),
    )


def _frequency_benchmark_hops(max_hops: int) -> tuple[int, ...]:
    return tuple(sorted({1, max(1, (max_hops + 1) // 2), max_hops}))


def _corpus_depth_summary(config: RouteVerificationConfig) -> dict[str, Any]:
    cases = load_graph_cases_from_sqlite(config.corpus_db_path)
    summary = corpus_summary(cases)
    metadata = read_corpus_sqlite_metadata(config.corpus_db_path)
    return {
        **summary,
        "source": "sqlite",
        "path": str(config.corpus_db_path),
        "metadata": metadata,
        "requested_max_hops": config.max_hops,
        "family_names": sorted({case.family for case in cases}),
    }


def _print_result(index: int, total: int, result: RouteVerificationResult) -> None:
    case = result.case
    s_value = "none" if result.chsh_s is None else f"{float(result.chsh_s):.3f}"
    print(
        f"[{index}/{total}] {case.stage} "
        f"hops={case.hops} edge_km={case.edge_length_m / 1000.0:.3f} "
        f"freq_x={case.memory_frequency_scale:g} "
        f"outcome={result.public_outcome} accepted={int(result.accepted)} "
        f"qualified={int(result.qualified)} delivered={result.delivered_count} "
        f"S={s_value} wall={result.wall_seconds:.1f}s",
        flush=True,
    )
