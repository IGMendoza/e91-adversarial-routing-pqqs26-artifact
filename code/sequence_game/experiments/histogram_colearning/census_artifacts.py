"""Serializable tables and figures for the Topology Bench census."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .case_selection import selection_coordinates


_CENSUS_COLUMNS = (
    "case_id",
    "graph_id",
    "assignment_id",
    "family",
    "assignment_policy",
    "strategy_validity",
    "graph_node_count",
    "graph_edge_count",
    "enumeration_complete",
    "enumeration_status",
    "dfs_expansions",
    "route_count",
    "route_count_lower_bound",
    "route_hops_minimum",
    "route_hops_mean",
    "route_hops_maximum",
    "node_cut_size",
    "edge_cut_size",
    "mean_internal_node_jaccard",
    "mean_edge_jaccard",
    "alice_feature_rank",
    "alice_compression_ratio",
    "eve_action_count",
    "eve_feature_rank",
    "eve_compression_ratio",
    "canonical_cache_entries",
    "estimated_cache_mib",
    "estimated_cache_mib_lower_bound",
    "audit_draws_per_seed",
    "estimated_first_configuration_physics_seconds",
    "exact_bounded_dag_compatible",
    "selection_eligible",
    "selection_exclusion_reasons",
    "payoff_residual_status",
)


def write_census_artifacts(
    census: dict[str, Any],
    case_manifest: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, Any]:
    """Write the complete census, frozen partitions, tables, and figures."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.append(_write_json(output / "census.json", census))
    written.append(_write_json(output / "case_manifest.json", case_manifest))
    written.append(
        _write_csv(output / "census_cases.csv", census["cases"], _CENSUS_COLUMNS)
    )
    selected_rows = _selected_table_rows(census, case_manifest)
    selected_columns = tuple(dict.fromkeys(
        (
            "partition_role",
            "selection_order",
            "maximin_distance",
            *_CENSUS_COLUMNS,
        )
    ))
    written.append(
        _write_csv(output / "selected_cases.csv", selected_rows, selected_columns)
    )
    written.extend(_resource_envelope_figure(census, case_manifest, output))
    written.extend(_selection_map_figure(census, case_manifest, output))
    written.extend(_corpus_composition_figure(census, output))

    manifest = {
        "schema": "histogram_topology_census_artifacts.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "census_schema": census.get("schema"),
        "case_manifest_schema": case_manifest.get("schema"),
        "selection_digest_blake2b_128": case_manifest.get(
            "selection_digest_blake2b_128"
        ),
        "corpus": dict(census.get("corpus") or {}),
        "host_ram_snapshot": dict(census.get("host_ram_snapshot") or {}),
        "files": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "blake2b_128": _file_digest(path),
            }
            for path in written
        ],
    }
    artifact_path = _write_json(output / "artifact_manifest.json", manifest)
    return {
        **manifest,
        "artifact_manifest_path": str(artifact_path),
    }


def _selected_table_rows(
    census: dict[str, Any],
    case_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    census_by_id = {
        str(row["case_id"]): row for row in census.get("cases") or []
    }
    rows: list[dict[str, Any]] = []
    for role, records in case_manifest["partitions"].items():
        for record in records:
            case_id = str(record["case_id"])
            if case_id not in census_by_id:
                raise ValueError(f"selected case {case_id!r} is absent from census")
            rows.append({
                **census_by_id[case_id],
                **record,
                "partition_role": role,
            })
    return rows


def _resource_envelope_figure(
    census: dict[str, Any],
    case_manifest: dict[str, Any],
    output: Path,
) -> list[Path]:
    rows = census["cases"]
    config = census["config"]
    roles = _role_by_case(case_manifest)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)

    groups = (
        (
            "eligible",
            [row for row in rows if row["selection_eligible"]],
            "#2878B5",
            "complete and inside selected-case envelope",
        ),
        (
            "complete_excluded",
            [
                row
                for row in rows
                if row["enumeration_complete"] and not row["selection_eligible"]
            ],
            "#E67E22",
            "complete; outside selected-case envelope",
        ),
        (
            "incomplete",
            [row for row in rows if not row["enumeration_complete"]],
            "#7F8C8D",
            "enumeration stopped at declared census bound",
        ),
    )
    for _, group, color, label in groups:
        if not group:
            continue
        route_counts = [
            max(1, int(row.get("route_count") or row["route_count_lower_bound"]))
            for row in group
        ]
        cache_mib = [
            max(
                0.01,
                float(
                    row.get("estimated_cache_mib")
                    or row["estimated_cache_mib_lower_bound"]
                ),
            )
            for row in group
        ]
        axes[0].scatter(
            route_counts,
            cache_mib,
            s=24,
            alpha=0.66,
            color=color,
            edgecolors="none",
            label=label,
        )
    axes[0].axhline(
        float(config["selected_case_cache_limit_mib"]),
        color="#C0392B",
        linestyle="--",
        linewidth=1.4,
        label="selected-case cache limit",
    )
    _mark_selected(axes[0], rows, roles, x_key="route_count", y_key="estimated_cache_mib")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("complete route count (lower bound when stopped)")
    axes[0].set_ylabel("planned cache memory (MiB)")
    axes[0].set_title("Action-surface memory envelope")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=7.5, loc="best")

    complete = [row for row in rows if row["enumeration_complete"]]
    audit_scatter = axes[1].scatter(
        [row["nominal_route_action_cells"] for row in complete],
        [row["audit_draws_per_seed"] for row in complete],
        c=[row["route_hops_maximum"] for row in complete],
        cmap="viridis",
        s=25,
        alpha=0.7,
        edgecolors="none",
    )
    colorbar = fig.colorbar(audit_scatter, ax=axes[1], pad=0.02)
    colorbar.set_label("maximum route hops")
    axes[1].axhline(
        float(config["selected_case_audit_draw_limit"]),
        color="#C0392B",
        linestyle="--",
        linewidth=1.4,
        label="selected-case audit-draw limit",
    )
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("route × Eve-action cells")
    axes[1].set_ylabel("audit draws per seed")
    axes[1].set_title(
        f"Independent audit demand ({config['audit_samples_per_cell']} draws/cell)"
    )
    _mark_selected(
        axes[1],
        rows,
        roles,
        x_key="nominal_route_action_cells",
        y_key="audit_draws_per_seed",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc="best")
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"resource_envelope.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def _selection_map_figure(
    census: dict[str, Any],
    case_manifest: dict[str, Any],
    output: Path,
) -> list[Path]:
    eligible = [row for row in census["cases"] if row["selection_eligible"]]
    coordinates = selection_coordinates(eligible)
    xy = np.asarray(coordinates["coordinates"])
    roles = _role_by_case(case_manifest)
    fig, ax = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=18,
        color="#BDC3C7",
        alpha=0.55,
        edgecolors="none",
        label="eligible assignment",
    )
    role_format = {
        "development_reference": ("#5D6D7E", "*", 135),
        "calibration": ("#1B9E77", "^", 70),
        "evaluation": ("#D95F02", "s", 62),
    }
    by_id = {str(row["case_id"]): row for row in eligible}
    index_by_id = {
        str(case_id): index for index, case_id in enumerate(coordinates["case_ids"])
    }
    for role, (color, marker, size) in role_format.items():
        case_ids = [case_id for case_id, value in roles.items() if value == role]
        present = [case_id for case_id in case_ids if case_id in index_by_id]
        if not present:
            continue
        points = np.asarray([xy[index_by_id[case_id]] for case_id in present])
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=size,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.6,
            label=role.replace("_", " "),
            zorder=4,
        )
        for case_id, point in zip(present, points):
            row = by_id[case_id]
            ax.annotate(
                str(row["family"]),
                point,
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=6.5,
                color="#283747",
            )
    explained = coordinates["explained_variance_fraction"]
    ax.set_xlabel(f"structural SVD axis 1 ({explained[0]:.1%} variance)")
    ax.set_ylabel(f"structural SVD axis 2 ({explained[1]:.1%} variance)")
    ax.set_title("Frozen partitions in the audited structural feature space")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8, loc="best")
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"case_selection_map.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def _corpus_composition_figure(
    census: dict[str, Any],
    output: Path,
) -> list[Path]:
    rows = census["cases"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    statuses = census["summary"]["enumeration_status_counts"]
    labels = list(statuses)
    values = [statuses[label] for label in labels]
    axes[0].barh(
        [label.replace("_", " ") for label in labels],
        values,
        color=["#2878B5" if label == "complete" else "#7F8C8D" for label in labels],
    )
    for index, value in enumerate(values):
        axes[0].text(value + 2, index, str(value), va="center", fontsize=8)
    axes[0].set_xlabel("assignments")
    axes[0].set_title("Cycle-free route census status")
    axes[0].grid(axis="x", alpha=0.2)

    policies = sorted({str(row["assignment_policy"]) for row in rows})
    eligible = [
        sum(str(row["assignment_policy"]) == policy and row["selection_eligible"] for row in rows)
        for policy in policies
    ]
    excluded = [
        sum(str(row["assignment_policy"]) == policy and not row["selection_eligible"] for row in rows)
        for policy in policies
    ]
    y = np.arange(len(policies))
    axes[1].barh(y, eligible, color="#2878B5", label="eligible")
    axes[1].barh(y, excluded, left=eligible, color="#D5D8DC", label="excluded")
    axes[1].set_yticks(y, [policy.replace("_", " ") for policy in policies], fontsize=7.5)
    axes[1].set_xlabel("assignments")
    axes[1].set_title("Selected-case envelope\nby assignment policy")
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].legend(fontsize=8)
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"corpus_composition.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def _mark_selected(
    ax: Any,
    rows: Sequence[dict[str, Any]],
    roles: dict[str, str],
    *,
    x_key: str,
    y_key: str,
) -> None:
    by_id = {str(row["case_id"]): row for row in rows}
    styles = {
        "development_reference": ("#5D6D7E", "*"),
        "calibration": ("#1B9E77", "^"),
        "evaluation": ("#D95F02", "s"),
    }
    for role, (color, marker) in styles.items():
        records = [by_id[case_id] for case_id, value in roles.items() if value == role]
        records = [row for row in records if row.get(x_key) and row.get(y_key)]
        if records:
            ax.scatter(
                [row[x_key] for row in records],
                [row[y_key] for row in records],
                s=70 if marker != "*" else 125,
                color=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.6,
                zorder=5,
                label=role.replace("_", " "),
            )


def _role_by_case(case_manifest: dict[str, Any]) -> dict[str, str]:
    return {
        str(row["case_id"]): role
        for role, records in case_manifest["partitions"].items()
        for row in records
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    columns: Sequence[str],
) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            rendered = {
                key: (
                    json.dumps(value, sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                )
                for key, value in row.items()
            }
            writer.writerow(rendered)
    return path


def _file_digest(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
