"""Manifest-locked case loading for histogram learner calibration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from .config import REPO_ROOT
from .corpus import HistogramGameCase, load_topology_bench_case


DEFAULT_STEP6_CASE_MANIFEST_PATH = (
    REPO_ROOT / "docs" / "topology_bench_step6_census" / "case_manifest.json"
)
STEP6_CASE_MANIFEST_SCHEMA = "histogram_topology_case_manifest.v1"
STEP6_SELECTION_DIGEST = "79ab7e3eb08e3535b9f6dbb945e85425"


@dataclass(frozen=True)
class FrozenCalibrationCase:
    """One calibration assignment selected before physical outcomes were read."""

    case_id: str
    graph_id: str
    assignment_id: str
    family: str
    route_count: int
    eve_action_count: int
    estimated_cache_mib: float
    exact_bounded_dag_compatible: bool
    selection_order: int

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "FrozenCalibrationCase":
        return cls(
            case_id=str(row["case_id"]),
            graph_id=str(row["graph_id"]),
            assignment_id=str(row["assignment_id"]),
            family=str(row["family"]),
            route_count=int(row["route_count"]),
            eve_action_count=int(row["eve_action_count"]),
            estimated_cache_mib=float(row["estimated_cache_mib"]),
            exact_bounded_dag_compatible=bool(
                row["exact_bounded_dag_compatible"]
            ),
            selection_order=int(row["selection_order"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "graph_id": self.graph_id,
            "assignment_id": self.assignment_id,
            "family": self.family,
            "route_count": self.route_count,
            "eve_action_count": self.eve_action_count,
            "estimated_cache_mib": self.estimated_cache_mib,
            "exact_bounded_dag_compatible": self.exact_bounded_dag_compatible,
            "selection_order": self.selection_order,
        }


def load_frozen_calibration_specs(
    manifest_path: str | Path = DEFAULT_STEP6_CASE_MANIFEST_PATH,
) -> tuple[FrozenCalibrationCase, ...]:
    """Read only the frozen calibration partition and verify its identity."""

    return load_frozen_partition_specs(
        "calibration",
        manifest_path=manifest_path,
    )


def load_frozen_evaluation_specs(
    manifest_path: str | Path = DEFAULT_STEP6_CASE_MANIFEST_PATH,
) -> tuple[FrozenCalibrationCase, ...]:
    """Read the untouched evaluation partition only after choices freeze."""

    return load_frozen_partition_specs(
        "evaluation",
        manifest_path=manifest_path,
    )


def load_frozen_partition_specs(
    partition: str,
    *,
    manifest_path: str | Path = DEFAULT_STEP6_CASE_MANIFEST_PATH,
) -> tuple[FrozenCalibrationCase, ...]:
    """Read one frozen Step 6 partition and verify graph disjointness."""

    selected_partition = str(partition)
    if selected_partition not in {"calibration", "evaluation"}:
        raise ValueError("partition must be calibration or evaluation")
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != STEP6_CASE_MANIFEST_SCHEMA:
        raise ValueError(f"unexpected case-manifest schema in {path}")
    if manifest.get("selection_digest_blake2b_128") != STEP6_SELECTION_DIGEST:
        raise ValueError("Step 6 case-manifest selection digest changed")
    partitions = manifest.get("partitions")
    if not isinstance(partitions, dict):
        raise ValueError("case manifest has no partitions")
    rows = partitions.get(selected_partition)
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"case manifest has no {selected_partition} partition"
        )
    specs = tuple(
        sorted(
            (FrozenCalibrationCase.from_dict(row) for row in rows),
            key=lambda row: row.selection_order,
        )
    )
    if len(specs) != 5:
        raise ValueError(
            f"Step 6 {selected_partition} partition must contain five cases"
        )
    if len({row.case_id for row in specs}) != len(specs):
        raise ValueError(
            f"{selected_partition} case ids are not unique"
        )
    if len({row.graph_id for row in specs}) != len(specs):
        raise ValueError(
            f"{selected_partition} graphs are not disjoint"
        )
    return specs


def load_frozen_calibration_cases(
    manifest_path: str | Path = DEFAULT_STEP6_CASE_MANIFEST_PATH,
    *,
    case_ids: Sequence[str] | None = None,
) -> tuple[tuple[FrozenCalibrationCase, HistogramGameCase], ...]:
    """Load the selected cycle-free route universes from the corpus database."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    specs = load_frozen_calibration_specs(path)
    requested = set(case_ids) if case_ids is not None else None
    if requested is not None:
        unknown = requested - {row.case_id for row in specs}
        if unknown:
            raise ValueError(f"unknown calibration case ids: {sorted(unknown)}")
        specs = tuple(row for row in specs if row.case_id in requested)
    corpus = manifest.get("corpus") or {}
    db_path = corpus.get("db_path")
    corpus_id = corpus.get("corpus_id")
    if not db_path or not corpus_id:
        raise ValueError("case manifest lacks corpus identity")
    loaded = []
    for spec in specs:
        case = load_topology_bench_case(
            db_path,
            corpus_id=str(corpus_id),
            graph_id=spec.graph_id,
            assignment_id=spec.assignment_id,
        )
        actions = case.actions("attack_surface_all")
        if len(case.routes) != spec.route_count:
            raise ValueError(
                f"route count changed for {spec.case_id}: "
                f"{len(case.routes)} != {spec.route_count}"
            )
        if len(actions) != spec.eve_action_count:
            raise ValueError(
                f"Eve action count changed for {spec.case_id}: "
                f"{len(actions)} != {spec.eve_action_count}"
            )
        loaded.append((spec, case))
    return tuple(loaded)


def load_frozen_evaluation_cases(
    manifest_path: str | Path = DEFAULT_STEP6_CASE_MANIFEST_PATH,
    *,
    case_ids: Sequence[str] | None = None,
) -> tuple[tuple[FrozenCalibrationCase, HistogramGameCase], ...]:
    """Load evaluation cases only after the caller freezes all choices."""

    return load_frozen_partition_cases(
        "evaluation",
        manifest_path=manifest_path,
        case_ids=case_ids,
    )


def load_frozen_partition_cases(
    partition: str,
    *,
    manifest_path: str | Path = DEFAULT_STEP6_CASE_MANIFEST_PATH,
    case_ids: Sequence[str] | None = None,
) -> tuple[tuple[FrozenCalibrationCase, HistogramGameCase], ...]:
    """Load one manifest-locked route universe from the corpus database."""

    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    specs = load_frozen_partition_specs(
        partition,
        manifest_path=path,
    )
    requested = set(case_ids) if case_ids is not None else None
    if requested is not None:
        unknown = requested - {row.case_id for row in specs}
        if unknown:
            raise ValueError(
                f"unknown {partition} case ids: {sorted(unknown)}"
            )
        specs = tuple(row for row in specs if row.case_id in requested)
    corpus = manifest.get("corpus") or {}
    db_path = corpus.get("db_path")
    corpus_id = corpus.get("corpus_id")
    if not db_path or not corpus_id:
        raise ValueError("case manifest lacks corpus identity")
    loaded = []
    for spec in specs:
        case = load_topology_bench_case(
            db_path,
            corpus_id=str(corpus_id),
            graph_id=spec.graph_id,
            assignment_id=spec.assignment_id,
        )
        actions = case.actions("attack_surface_all")
        if len(case.routes) != spec.route_count:
            raise ValueError(
                f"route count changed for {spec.case_id}: "
                f"{len(case.routes)} != {spec.route_count}"
            )
        if len(actions) != spec.eve_action_count:
            raise ValueError(
                f"Eve action count changed for {spec.case_id}: "
                f"{len(actions)} != {spec.eve_action_count}"
            )
        loaded.append((spec, case))
    return tuple(loaded)
