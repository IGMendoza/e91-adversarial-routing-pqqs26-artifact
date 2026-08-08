#!/usr/bin/env python3
"""Create a length-normalized derivative graph corpus.

This script is intentionally a corpus transform, not a physical model.  It
preserves graph structure and raw-length ordering while replacing edge
``length_m`` values with a bounded empirical-rank scale suitable for structural
stress runs with the compiled E91 histogram backend.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import shutil
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequence_game.corpus.sqlite_store import ingest_graph_corpus_to_sqlite
from sequence_game.topology import TopologyIR


DEFAULT_SOURCE_ROOT = Path("runs") / "graph_corpus_sqlite_v1"
DEFAULT_OUTPUT_ROOT = (
    Path("runs") / "graph_corpus_sqlite_v1_topology_bench_real_rank10km_normalized"
)
DEFAULT_CORPUS_ID = "topology_bench_real_topologies_rank10km_normalized"

TRANSFORM_VERSION = "rank_quantile_10km_median_v1"
EDGE_LENGTH_SEMANTICS = "normalized_rank_quantile_10km"
PHYSICAL_ROLE = "telecom_topology_structure_only"
SCIENTIFIC_SCOPE = "empirical_fit"
NORMALIZATION_SCOPE = "structural_topology_stress"

# Empirical CDF percentile -> normalized fiber length in km.  This gives a
# 10 km median, a 1 km floor, and a rare long-link tail into the 20-50 km band.
DEFAULT_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.000, 1.0),
    (0.010, 1.5),
    (0.050, 2.5),
    (0.100, 4.0),
    (0.250, 7.0),
    (0.500, 10.0),
    (0.750, 13.5),
    (0.900, 17.0),
    (0.950, 22.0),
    (0.980, 32.0),
    (0.995, 45.0),
    (1.000, 50.0),
)


@dataclass(frozen=True)
class NormalizeConfig:
    source_root: Path
    output_root: Path
    corpus_id: str
    db_path: Path
    anchors: tuple[tuple[float, float], ...] = DEFAULT_ANCHORS
    overwrite: bool = False
    write_sqlite: bool = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--corpus-id", default=DEFAULT_CORPUS_ID)
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="SQLite output path. Defaults to <output-root>/graph_corpus.sqlite.",
    )
    parser.add_argument(
        "--anchors",
        default=None,
        help=(
            "Comma-separated percentile:length_km anchors, e.g. "
            "'0:1,0.5:10,0.95:22,1:50'. Defaults to the rank10km profile."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-sqlite",
        action="store_true",
        help="Write the normalized corpus directory but do not ingest SQLite.",
    )
    return parser.parse_args(argv)


def normalize_corpus(config: NormalizeConfig) -> dict[str, Any]:
    source_root = config.source_root.resolve()
    output_root = config.output_root.resolve()
    db_path = config.db_path.resolve()

    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if output_root.exists():
        if not config.overwrite:
            raise FileExistsError(
                f"{output_root} already exists; pass --overwrite to replace it"
            )
        shutil.rmtree(output_root)

    manifest_path = source_root / "corpus_manifest.json"
    manifest = _load_json(manifest_path)
    topology_source_paths = [
        _resolve_display_path(source_root, record["topology_path"])
        for record in manifest.get("graphs", [])
        if isinstance(record, dict) and record.get("topology_path")
    ]
    raw_lengths_m = _collect_edge_lengths_m(topology_source_paths)
    if not raw_lengths_m:
        raise ValueError(f"no edge lengths found under {source_root}")

    sorted_lengths_m = sorted(raw_lengths_m)
    shutil.copytree(source_root, output_root, ignore=_copy_ignore)

    source_display = _display_root(source_root)
    output_display = _display_root(output_root)
    _rewrite_embedded_paths(output_root, source_root, output_root)

    normalization_meta = _normalization_metadata(
        source_root=source_root,
        source_display=source_display,
        output_root=output_root,
        output_display=output_display,
        corpus_id=config.corpus_id,
        anchors=config.anchors,
        raw_lengths_m=raw_lengths_m,
    )

    output_manifest_path = output_root / "corpus_manifest.json"
    output_manifest = _load_json(output_manifest_path)
    graph_stats_by_id: dict[str, dict[str, Any]] = {}
    for graph_record in output_manifest.get("graphs", []):
        if not isinstance(graph_record, dict) or not graph_record.get("topology_path"):
            continue
        topology_path = _resolve_display_path(output_root, graph_record["topology_path"])
        graph_stats = _normalize_topology_file(
            topology_path,
            sorted_lengths_m=sorted_lengths_m,
            anchors=config.anchors,
            normalization_meta=normalization_meta,
        )
        graph_id = str(graph_record.get("graph_id", topology_path.parent.name))
        graph_stats_by_id[graph_id] = graph_stats
        _update_graph_record(graph_record, graph_stats)

        graph_summary_path = topology_path.parent / "graph_summary.json"
        if graph_summary_path.exists():
            summary = _load_json(graph_summary_path)
            _update_graph_record(summary, graph_stats)
            _write_json(summary, graph_summary_path)

    output_manifest["script"] = "scripts/normalize_graph_corpus_lengths.py"
    output_manifest.setdefault("config", {})
    output_manifest["config"]["length_normalization"] = normalization_meta
    output_manifest["counts"] = dict(output_manifest.get("counts", {}))
    output_manifest["counts"]["length_normalized_graphs"] = len(graph_stats_by_id)
    _write_json(output_manifest, output_manifest_path)

    _update_corpus_summary_csv(output_root / "corpus_summary.csv", graph_stats_by_id)
    _rewrite_assignment_summary_csv(output_root / "assignment_summary.csv", source_root, output_root)

    report = _build_report(
        normalization_meta=normalization_meta,
        raw_lengths_m=raw_lengths_m,
        normalized_lengths_m=[
            _normalized_length_m(length_m, sorted_lengths_m, config.anchors)
            for length_m in raw_lengths_m
        ],
        graph_stats_by_id=graph_stats_by_id,
        output_root=output_root,
        db_path=db_path if config.write_sqlite else None,
        corpus_id=config.corpus_id,
    )
    _write_json(report, output_root / "length_normalization_report.json")

    if config.write_sqlite:
        summary = ingest_graph_corpus_to_sqlite(
            output_root,
            db_path,
            corpus_id=config.corpus_id,
            progress=print,
        )
        report["sqlite_ingest"] = summary.to_dict()
        _write_json(report, output_root / "length_normalization_report.json")

    return report


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    return {
        name
        for name in names
        if name == "graph_corpus.sqlite"
        or name.endswith(".sqlite")
        or name.endswith(".sqlite-shm")
        or name.endswith(".sqlite-wal")
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(payload: Any, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _display_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _resolve_display_path(root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, root / path]
    parts = str(raw_path).replace("\\", "/").split("/")
    if root.name in parts:
        index = parts.index(root.name)
        suffix = Path(*parts[index + 1 :])
        candidates.insert(0, root / suffix)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _rewrite_embedded_paths(root: Path, source_root: Path, output_root: Path) -> None:
    for path in sorted(root.rglob("*.json")):
        payload = _load_json(path)
        rewritten = _rewrite_path_strings(payload, source_root, output_root)
        _write_json(rewritten, path)


def _rewrite_path_strings(value: Any, source_root: Path, output_root: Path) -> Any:
    if isinstance(value, dict):
        return {
            key: _rewrite_path_strings(item, source_root, output_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rewrite_path_strings(item, source_root, output_root) for item in value]
    if isinstance(value, str):
        return _rewrite_path_string(value, source_root, output_root)
    return value


def _rewrite_path_string(value: str, source_root: Path, output_root: Path) -> str:
    normalized = value.replace("\\", "/")
    replacements = [
        (_display_root(source_root), _display_root(output_root)),
        (
            str(source_root.resolve()).replace("\\", "/"),
            str(output_root.resolve()).replace("\\", "/"),
        ),
    ]
    for source_prefix, output_prefix in replacements:
        if normalized == source_prefix:
            return output_prefix
        if normalized.startswith(source_prefix + "/"):
            return output_prefix + normalized[len(source_prefix) :]
    return value


def _collect_edge_lengths_m(topology_paths: list[Path]) -> list[float]:
    lengths: list[float] = []
    for path in topology_paths:
        topology = _load_json(path)
        for edge in topology.get("edges", []):
            lengths.append(float(edge["length_m"]))
    return lengths


def _normalize_topology_file(
    topology_path: Path,
    *,
    sorted_lengths_m: list[float],
    anchors: tuple[tuple[float, float], ...],
    normalization_meta: dict[str, Any],
) -> dict[str, Any]:
    topology = _load_json(topology_path)
    raw_lengths_m = [float(edge["length_m"]) for edge in topology.get("edges", [])]
    normalized_lengths_m: list[float] = []

    for edge in topology.get("edges", []):
        original_length_m = float(edge["length_m"])
        normalized_length_m = _normalized_length_m(original_length_m, sorted_lengths_m, anchors)
        edge["length_m"] = normalized_length_m
        edge["notes"] = _append_edge_note(edge.get("notes", ""), original_length_m, normalized_length_m)
        normalized_lengths_m.append(normalized_length_m)

    metadata = dict(topology.get("metadata") or {})
    params = dict(metadata.get("params") or {})
    source_edge_length_semantics = params.get("edge_length_semantics")
    params.update(
        {
            "physical_role": PHYSICAL_ROLE,
            "edge_length_semantics": EDGE_LENGTH_SEMANTICS,
            "scientific_scope": SCIENTIFIC_SCOPE,
            "normalization_scope": NORMALIZATION_SCOPE,
            "length_normalization": {
                **normalization_meta,
                "source_edge_length_semantics": source_edge_length_semantics,
            },
        }
    )
    metadata["params"] = params
    topology["metadata"] = metadata

    TopologyIR.from_dict(topology).validate()
    _write_json(topology, topology_path)

    return {
        "edge_count": len(normalized_lengths_m),
        "raw_length_stats_km": _length_stats_km(raw_lengths_m),
        "normalized_length_stats_km": _length_stats_km(normalized_lengths_m),
    }


def _append_edge_note(existing: str, original_length_m: float, normalized_length_m: float) -> str:
    note = json.dumps(
        {
            "length_normalization": {
                "version": TRANSFORM_VERSION,
                "original_length_m": original_length_m,
                "normalized_length_m": normalized_length_m,
            }
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{existing} | {note}" if existing else note


def _normalized_length_m(
    length_m: float,
    sorted_lengths_m: list[float],
    anchors: tuple[tuple[float, float], ...],
) -> float:
    u = _empirical_percentile(length_m, sorted_lengths_m)
    km = _interpolate_anchors(u, anchors)
    return float(km * 1000.0)


def _empirical_percentile(value: float, sorted_values: list[float]) -> float:
    if len(sorted_values) == 1:
        return 0.5
    lo = bisect.bisect_left(sorted_values, value)
    hi = bisect.bisect_right(sorted_values, value)
    average_index = (lo + hi - 1) / 2.0
    return max(0.0, min(1.0, average_index / (len(sorted_values) - 1)))


def _interpolate_anchors(u: float, anchors: tuple[tuple[float, float], ...]) -> float:
    anchors = tuple(sorted(anchors))
    if u <= anchors[0][0]:
        return anchors[0][1]
    if u >= anchors[-1][0]:
        return anchors[-1][1]
    for (u0, km0), (u1, km1) in zip(anchors, anchors[1:]):
        if u0 <= u <= u1:
            if u1 == u0:
                return km1
            weight = (u - u0) / (u1 - u0)
            return km0 + weight * (km1 - km0)
    return anchors[-1][1]


def _update_graph_record(record: dict[str, Any], graph_stats: dict[str, Any]) -> None:
    record["edge_length_semantics"] = EDGE_LENGTH_SEMANTICS
    record["physical_role"] = PHYSICAL_ROLE
    record["scientific_scope"] = SCIENTIFIC_SCOPE
    record["normalization_scope"] = NORMALIZATION_SCOPE
    record["length_normalization"] = {
        "version": TRANSFORM_VERSION,
        **graph_stats,
    }


def _update_corpus_summary_csv(
    path: Path,
    graph_stats_by_id: dict[str, dict[str, Any]],
) -> None:
    if not path.exists():
        return
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        graph_id = str(row.get("graph_id", ""))
        if graph_id in graph_stats_by_id:
            row["edge_length_semantics"] = EDGE_LENGTH_SEMANTICS
            row["physical_role"] = PHYSICAL_ROLE
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_assignment_summary_csv(path: Path, source_root: Path, output_root: Path) -> None:
    if not path.exists():
        return
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        for key, value in list(row.items()):
            if value:
                row[key] = _rewrite_path_string(value, source_root, output_root)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _normalization_metadata(
    *,
    source_root: Path,
    source_display: str,
    output_root: Path,
    output_display: str,
    corpus_id: str,
    anchors: tuple[tuple[float, float], ...],
    raw_lengths_m: list[float],
) -> dict[str, Any]:
    return {
        "version": TRANSFORM_VERSION,
        "method": "global_empirical_cdf_piecewise_linear_rank_map",
        "scientific_scope": SCIENTIFIC_SCOPE,
        "normalization_scope": NORMALIZATION_SCOPE,
        "edge_length_semantics": EDGE_LENGTH_SEMANTICS,
        "physical_role": PHYSICAL_ROLE,
        "source_corpus_root": source_display,
        "source_corpus_root_abs": str(source_root),
        "output_corpus_root": output_display,
        "output_corpus_root_abs": str(output_root),
        "output_corpus_id": corpus_id,
        "anchor_quantiles_to_km": [
            {"quantile": q, "length_km": km} for q, km in anchors
        ],
        "raw_corpus_length_stats_km": _length_stats_km(raw_lengths_m),
        "original_lengths_preserved_in_edge_notes": True,
        "claim": (
            "Structural empirical-fit length normalization only; not a physical "
            "fiber-distance claim for the original Topology Bench links."
        ),
    }


def _build_report(
    *,
    normalization_meta: dict[str, Any],
    raw_lengths_m: list[float],
    normalized_lengths_m: list[float],
    graph_stats_by_id: dict[str, dict[str, Any]],
    output_root: Path,
    db_path: Path | None,
    corpus_id: str,
) -> dict[str, Any]:
    normalized_stats = _length_stats_km(normalized_lengths_m)
    return {
        "corpus_id": corpus_id,
        "output_root": _display_root(output_root),
        "db_path": _display_root(db_path) if db_path is not None else None,
        "graph_count": len(graph_stats_by_id),
        "edge_count": len(normalized_lengths_m),
        "normalization": normalization_meta,
        "raw_length_stats_km": _length_stats_km(raw_lengths_m),
        "normalized_length_stats_km": normalized_stats,
        "normalized_edge_counts": {
            "lt_1km": sum(1 for value in normalized_lengths_m if value < 1000.0),
            "eq_1km": sum(1 for value in normalized_lengths_m if value == 1000.0),
            "gt_20km": sum(1 for value in normalized_lengths_m if value > 20_000.0),
            "gt_30km": sum(1 for value in normalized_lengths_m if value > 30_000.0),
            "gt_40km": sum(1 for value in normalized_lengths_m if value > 40_000.0),
            "gt_50km": sum(1 for value in normalized_lengths_m if value > 50_000.0),
        },
        "graph_stats": graph_stats_by_id,
    }


def _length_stats_km(lengths_m: list[float]) -> dict[str, float | int | None]:
    if not lengths_m:
        return {"count": 0}
    values = sorted(length / 1000.0 for length in lengths_m)
    return {
        "count": len(values),
        "min": values[0],
        "p01": _nearest_quantile(values, 0.01),
        "p05": _nearest_quantile(values, 0.05),
        "p10": _nearest_quantile(values, 0.10),
        "p25": _nearest_quantile(values, 0.25),
        "median": statistics.median(values),
        "p75": _nearest_quantile(values, 0.75),
        "p90": _nearest_quantile(values, 0.90),
        "p95": _nearest_quantile(values, 0.95),
        "p98": _nearest_quantile(values, 0.98),
        "p995": _nearest_quantile(values, 0.995),
        "max": values[-1],
        "mean": statistics.fmean(values),
    }


def _nearest_quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty quantile input")
    index = round(q * (len(sorted_values) - 1))
    return sorted_values[max(0, min(len(sorted_values) - 1, index))]


def _parse_anchors(raw: str | None) -> tuple[tuple[float, float], ...]:
    if not raw:
        return DEFAULT_ANCHORS
    anchors: list[tuple[float, float]] = []
    for item in raw.split(","):
        q_text, km_text = item.split(":", 1)
        anchors.append((float(q_text), float(km_text)))
    anchors = sorted(anchors)
    if anchors[0][0] != 0.0 or anchors[-1][0] != 1.0:
        raise ValueError("--anchors must include quantile 0 and 1 endpoints")
    if any(q < 0.0 or q > 1.0 for q, _km in anchors):
        raise ValueError("--anchors quantiles must be in [0, 1]")
    if any(km <= 0.0 for _q, km in anchors):
        raise ValueError("--anchors lengths must be positive")
    if any(q0 >= q1 for (q0, _), (q1, _) in zip(anchors, anchors[1:])):
        raise ValueError("--anchors quantiles must be strictly increasing")
    return tuple(anchors)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root
    db_path = args.db_path or (output_root / "graph_corpus.sqlite")
    report = normalize_corpus(
        NormalizeConfig(
            source_root=args.source_root,
            output_root=output_root,
            corpus_id=args.corpus_id,
            db_path=db_path,
            anchors=_parse_anchors(args.anchors),
            overwrite=args.overwrite,
            write_sqlite=not args.no_sqlite,
        )
    )
    stats = report["normalized_length_stats_km"]
    print(
        "normalized corpus written: "
        f"{report['output_root']} corpus_id={report['corpus_id']} "
        f"edges={report['edge_count']} median_km={stats['median']:.3f} "
        f"p95_km={stats['p95']:.3f} max_km={stats['max']:.3f}"
    )
    if report.get("db_path"):
        print(f"sqlite db: {report['db_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
