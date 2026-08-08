"""Topology Bench real-topology ingestion for graph-corpus construction.

Topology Bench (Matzner et al., JOCN 2024) distributes 105 real core optical
network topologies as ``.xlsx`` workbooks with one ``Nodes_*`` sheet
(``Node_ID``, ``Latitude``, ``Longitude``, ...) and one ``Edges_*`` sheet
(``Edge_ID``, ``Source``, ``Destination``, ``Computed Length (km)``).

The parser below reads those workbooks with the standard library only
(``zipfile`` + ``xml.etree``), so no Excel dependency is required.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

from sequence_game.topology import EdgeRecord, NodeRecord, TopologyIR

from .ingest import (
    EDGE_LENGTH_PHYSICAL,
    EDGE_LENGTH_SYNTHETIC,
    PHYSICAL_TELECOM_TOPOLOGY,
    _largest_component_from_edges,
    _metadata,
)

SOURCE_TOPOLOGY_BENCH = "topology_bench"

_XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_XLSX_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_XLSX_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

_NODE_ID_HEADERS = ("node_id", "node", "id")
_LAT_HEADERS = ("latitude", "lat")
_LON_HEADERS = ("longitude", "lon", "lng")
_EDGE_SOURCE_HEADERS = ("source", "from", "u")
_EDGE_TARGET_HEADERS = ("destination", "target", "to", "v")
_EDGE_ID_HEADERS = ("edge_id", "edge", "id")
_EDGE_LENGTH_KM_HEADERS = (
    "computed length (km)",
    "length (km)",
    "length_km",
    "distance (km)",
)


def ingest_topology_bench_directory(
    path: Path,
    *,
    source: str = SOURCE_TOPOLOGY_BENCH,
    physical_role: str = PHYSICAL_TELECOM_TOPOLOGY,
    synthetic_edge_length_m: float = 10.0,
    largest_connected_component: bool = True,
) -> list[TopologyIR]:
    """Ingest every Topology Bench ``.xlsx`` workbook under ``path``."""
    graphs = []
    for file_path in sorted(Path(path).glob("*.xlsx")):
        graphs.append(
            ingest_topology_bench_workbook(
                file_path,
                source=source,
                physical_role=physical_role,
                synthetic_edge_length_m=synthetic_edge_length_m,
                largest_connected_component=largest_connected_component,
            )
        )
    return graphs


def ingest_topology_bench_workbook(
    file_path: Path,
    *,
    source: str = SOURCE_TOPOLOGY_BENCH,
    physical_role: str = PHYSICAL_TELECOM_TOPOLOGY,
    synthetic_edge_length_m: float = 10.0,
    largest_connected_component: bool = True,
) -> TopologyIR:
    """Ingest one Topology Bench workbook into a :class:`TopologyIR`."""
    sheets = _read_workbook_sheets(Path(file_path))
    node_rows = _sheet_by_prefix(sheets, "nodes", file_path)
    edge_rows = _sheet_by_prefix(sheets, "edges", file_path)

    node_records = _parse_node_rows(node_rows, file_path)
    raw_edges = _parse_edge_rows(edge_rows, file_path)

    # Keep only edges whose endpoints exist, deduplicate undirected pairs.
    seen_pairs: set[frozenset[str]] = set()
    edges: list[tuple[str, str, Optional[float]]] = []
    for u, v, length_km in raw_edges:
        if u == v or u not in node_records or v not in node_records:
            continue
        pair = frozenset((u, v))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        edges.append((u, v, length_km))

    if largest_connected_component:
        keep, kept_pairs = _largest_component_from_edges(
            set(node_records),
            [(u, v) for u, v, _length in edges],
        )
        kept_pair_set = {frozenset(pair) for pair in kept_pairs}
        node_records = {
            node_id: record for node_id, record in node_records.items() if node_id in keep
        }
        edges = [
            (u, v, length_km)
            for u, v, length_km in edges
            if frozenset((u, v)) in kept_pair_set
        ]

    length_semantics = EDGE_LENGTH_PHYSICAL
    edge_records = []
    for index, (u, v, length_km) in enumerate(edges):
        if length_km is not None and length_km >= 0:
            length_m = float(length_km) * 1000.0
        else:
            length_m = synthetic_edge_length_m
            length_semantics = EDGE_LENGTH_SYNTHETIC
        edge_records.append(EdgeRecord(f"e{index}", u, v, length_m, eve_eligible=True))

    graph = TopologyIR(
        nodes=node_records,
        edges=edge_records,
        metadata=_metadata(
            _topology_family(file_path.stem),
            source=source,
            physical_role=physical_role,
            edge_length_semantics=length_semantics,
            seed=None,
            extra={
                "ingested_from": str(file_path),
                "topology_name": file_path.stem,
                "edge_length_source": "topology_bench_computed_length_km",
            },
        ),
    )
    graph.validate()
    return graph


def _sheet_by_prefix(
    sheets: dict[str, list[list[Optional[str]]]],
    prefix: str,
    file_path: Path,
) -> list[list[Optional[str]]]:
    for name, rows in sheets.items():
        if name.lower().startswith(prefix):
            return rows
    raise ValueError(
        f"{file_path}: no worksheet named {prefix.title()}_*; "
        f"found: {sorted(sheets)}"
    )


def _topology_family(stem: str) -> str:
    match = re.match(r"^TOP_\d+_(?P<name>.+)$", stem)
    name = match.group("name") if match else stem
    return name.lower()


def _parse_node_rows(
    rows: list[list[Optional[str]]],
    file_path: Path,
) -> dict[str, NodeRecord]:
    if not rows:
        raise ValueError(f"{file_path}: empty Nodes sheet")
    header = _header_map(rows[0])
    node_col = _find_column(header, _NODE_ID_HEADERS)
    if node_col is None:
        raise ValueError(f"{file_path}: Nodes sheet is missing a Node_ID column")
    lat_col = _find_column(header, _LAT_HEADERS)
    lon_col = _find_column(header, _LON_HEADERS)

    records: dict[str, NodeRecord] = {}
    for row in rows[1:]:
        node_id = _cell(row, node_col)
        if node_id is None or not str(node_id).strip():
            continue
        node_id = _normalize_id(node_id)
        latitude = _as_optional_float(_cell(row, lat_col))
        longitude = _as_optional_float(_cell(row, lon_col))
        coordinates = (
            (latitude, longitude)
            if latitude is not None and longitude is not None
            else None
        )
        records[node_id] = NodeRecord(node_id, coordinates=coordinates)
    return records


def _parse_edge_rows(
    rows: list[list[Optional[str]]],
    file_path: Path,
) -> list[tuple[str, str, Optional[float]]]:
    if not rows:
        raise ValueError(f"{file_path}: empty Edges sheet")
    header = _header_map(rows[0])
    source_col = _find_column(header, _EDGE_SOURCE_HEADERS)
    target_col = _find_column(header, _EDGE_TARGET_HEADERS)
    if source_col is None or target_col is None:
        raise ValueError(
            f"{file_path}: Edges sheet is missing Source/Destination columns"
        )
    length_col = _find_column(header, _EDGE_LENGTH_KM_HEADERS)

    edges = []
    for row in rows[1:]:
        u = _cell(row, source_col)
        v = _cell(row, target_col)
        if u is None or v is None:
            continue
        if not str(u).strip() or not str(v).strip():
            continue
        length_km = _as_optional_float(_cell(row, length_col))
        edges.append((_normalize_id(u), _normalize_id(v), length_km))
    return edges


def _normalize_id(value: str) -> str:
    text = str(value).strip()
    # Numeric IDs in xlsx cells surface as floats ("1" -> "1.0"); undo that.
    if re.fullmatch(r"-?\d+\.0", text):
        return text[:-2]
    return text


def _header_map(header_row: list[Optional[str]]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, value in enumerate(header_row):
        if value is None:
            continue
        key = str(value).strip().lower()
        if key and key not in mapping:
            mapping[key] = index
    return mapping


def _find_column(header: dict[str, int], candidates: tuple[str, ...]) -> Optional[int]:
    for candidate in candidates:
        if candidate in header:
            return header[candidate]
    return None


def _cell(row: list[Optional[str]], column: Optional[int]) -> Optional[str]:
    if column is None or column >= len(row):
        return None
    return row[column]


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _read_workbook_sheets(file_path: Path) -> dict[str, list[list[Optional[str]]]]:
    """Return ``{sheet_name: rows}`` for every worksheet in the workbook."""
    with zipfile.ZipFile(file_path) as archive:
        shared_strings = _read_shared_strings(archive)
        relationships = _read_workbook_relationships(archive)
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheets: dict[str, list[list[Optional[str]]]] = {}
        for sheet in workbook.iter(f"{{{_XLSX_MAIN_NS}}}sheet"):
            name = sheet.get("name") or ""
            rel_id = sheet.get(f"{{{_XLSX_REL_NS}}}id")
            target = relationships.get(rel_id or "")
            if not target:
                continue
            if target.startswith("/"):
                # Absolute targets are package-root relative.
                target = target.lstrip("/")
            elif not target.startswith("xl/"):
                target = f"xl/{target}"
            sheets[name] = _read_sheet_rows(archive.read(target), shared_strings)
        return sheets


def _read_workbook_relationships(archive: zipfile.ZipFile) -> dict[str, str]:
    try:
        payload = archive.read("xl/_rels/workbook.xml.rels")
    except KeyError:
        return {}
    root = ET.fromstring(payload)
    return {
        rel.get("Id") or "": rel.get("Target") or ""
        for rel in root.iter(f"{{{_XLSX_PKG_REL_NS}}}Relationship")
    }


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        payload = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(payload)
    strings = []
    for item in root.iter(f"{{{_XLSX_MAIN_NS}}}si"):
        strings.append(
            "".join(
                fragment.text or ""
                for fragment in item.iter(f"{{{_XLSX_MAIN_NS}}}t")
            )
        )
    return strings


def _read_sheet_rows(
    payload: bytes,
    shared_strings: list[str],
) -> list[list[Optional[str]]]:
    root = ET.fromstring(payload)
    rows: list[list[Optional[str]]] = []
    for row in root.iter(f"{{{_XLSX_MAIN_NS}}}row"):
        cells: dict[int, Optional[str]] = {}
        fallback_index = 0
        for cell in row.iter(f"{{{_XLSX_MAIN_NS}}}c"):
            reference = cell.get("r")
            column = (
                _column_index(reference)
                if reference is not None
                else fallback_index
            )
            fallback_index = column + 1
            cells[column] = _cell_value(cell, shared_strings)
        width = max(cells) + 1 if cells else 0
        rows.append([cells.get(index) for index in range(width)])
    return rows


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> Optional[str]:
    cell_type = cell.get("t")
    if cell_type == "inlineStr":
        inline = cell.find(f"{{{_XLSX_MAIN_NS}}}is")
        if inline is None:
            return None
        return "".join(
            fragment.text or ""
            for fragment in inline.iter(f"{{{_XLSX_MAIN_NS}}}t")
        )
    value = cell.find(f"{{{_XLSX_MAIN_NS}}}v")
    if value is None or value.text is None:
        return None
    if cell_type == "s":
        try:
            return shared_strings[int(value.text)]
        except (ValueError, IndexError):
            return None
    return value.text


def _column_index(reference: str) -> int:
    index = 0
    for char in reference:
        if not char.isalpha():
            break
        index = index * 26 + (ord(char.upper()) - ord("A") + 1)
    return max(index - 1, 0)
