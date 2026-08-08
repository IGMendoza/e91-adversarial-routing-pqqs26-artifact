"""Opt-in wall-clock and CPU timing without overhead in numerical hot paths."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeMeasurement:
    """Result and elapsed time for one call."""

    result: Any
    wall_seconds: float
    cpu_seconds: float


@dataclass(frozen=True)
class BenchmarkResult:
    """Repeated call timings; ``result`` is from the final measured call."""

    result: Any
    wall_seconds: tuple[float, ...]
    cpu_seconds: tuple[float, ...]

    @property
    def minimum_wall_seconds(self) -> float:
        return min(self.wall_seconds)

    @property
    def median_wall_seconds(self) -> float:
        return statistics.median(self.wall_seconds)

    @property
    def mean_wall_seconds(self) -> float:
        return statistics.fmean(self.wall_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repeats": len(self.wall_seconds),
            "wall_seconds": list(self.wall_seconds),
            "cpu_seconds": list(self.cpu_seconds),
            "minimum_wall_seconds": self.minimum_wall_seconds,
            "median_wall_seconds": self.median_wall_seconds,
            "mean_wall_seconds": self.mean_wall_seconds,
        }


def measure_call(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> RuntimeMeasurement:
    """Measure one call using monotonic wall time and process CPU time."""

    wall_start = time.perf_counter_ns()
    cpu_start = time.process_time_ns()
    result = function(*args, **kwargs)
    return RuntimeMeasurement(
        result=result,
        wall_seconds=(time.perf_counter_ns() - wall_start) / 1e9,
        cpu_seconds=(time.process_time_ns() - cpu_start) / 1e9,
    )


def benchmark_call(
    function: Callable[..., Any],
    /,
    *args: Any,
    repeats: int = 5,
    warmups: int = 1,
    **kwargs: Any,
) -> BenchmarkResult:
    """Benchmark a call with explicit warmups and no instrumentation of the call itself."""

    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if warmups < 0:
        raise ValueError("warmups must be >= 0")
    for _ in range(warmups):
        function(*args, **kwargs)
    samples = [measure_call(function, *args, **kwargs) for _ in range(repeats)]
    return BenchmarkResult(
        result=samples[-1].result,
        wall_seconds=tuple(sample.wall_seconds for sample in samples),
        cpu_seconds=tuple(sample.cpu_seconds for sample in samples),
    )
