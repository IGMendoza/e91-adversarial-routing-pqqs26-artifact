"""SQLite storage helpers for corpus run outputs."""

from __future__ import annotations

import json
import hashlib
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from json import JSONDecoder, JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Iterator

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class IngestSummary:
    db_path: Path
    run_root: Path
    data_dir: Path
    checkpoint_records: int = 0
    worker_status_records: int = 0
    json_artifacts: int = 0
    metadata_records: int = 0
    source_file_records: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "run_root": str(self.run_root),
            "data_dir": str(self.data_dir),
            "checkpoint_records": self.checkpoint_records,
            "worker_status_records": self.worker_status_records,
            "json_artifacts": self.json_artifacts,
            "metadata_records": self.metadata_records,
            "source_file_records": self.source_file_records,
        }


@dataclass(frozen=True)
class GraphCorpusIngestSummary:
    db_path: Path
    corpus_root: Path
    corpus_id: str
    source_file_records: int = 0
    manifest_graphs: int | None = None
    manifest_assignments: int | None = None
    total_bytes: int = 0
    indexed_graphs: int = 0
    indexed_assignments: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": str(self.db_path),
            "corpus_root": str(self.corpus_root),
            "corpus_id": self.corpus_id,
            "source_file_records": self.source_file_records,
            "manifest_graphs": self.manifest_graphs,
            "manifest_assignments": self.manifest_assignments,
            "total_bytes": self.total_bytes,
            "indexed_graphs": self.indexed_graphs,
            "indexed_assignments": self.indexed_assignments,
        }


@dataclass(frozen=True)
class SQLiteGraphCorpusSource:
    """Read graph-corpus files from the source_files table."""

    db_path: Path
    corpus_id: str
    root_path: Path | None = None

    @classmethod
    def open(
        cls,
        db_path: str | Path,
        corpus_id: str | None = None,
    ) -> "SQLiteGraphCorpusSource":
        target_db = Path(db_path).resolve()
        resolved_corpus_id = resolve_graph_corpus_id(target_db, corpus_id)
        con = connect_corpus_db(target_db)
        try:
            row = con.execute(
                """
                SELECT root_path
                FROM graph_corpora
                WHERE corpus_id = ?
                """,
                (resolved_corpus_id,),
            ).fetchone()
        finally:
            con.close()
        return cls(
            db_path=target_db,
            corpus_id=resolved_corpus_id,
            root_path=Path(row["root_path"]) if row and row["root_path"] else None,
        )

    def read_bytes(self, raw_path: str | Path) -> bytes:
        relative_path = self._normalize_relative_path(raw_path)
        con = connect_corpus_db(self.db_path)
        try:
            row = con.execute(
                """
                SELECT payload_blob
                FROM source_files
                WHERE source_group = 'graph_corpus'
                  AND source_id = ?
                  AND relative_path = ?
                """,
                (self.corpus_id, relative_path),
            ).fetchone()
            if row is None:
                row = self._unique_suffix_match(con, relative_path)
            if row is None:
                raise FileNotFoundError(
                    f"{raw_path!s} not found in graph corpus {self.corpus_id!r}"
                )
            return bytes(row["payload_blob"])
        finally:
            con.close()

    def load_json(self, raw_path: str | Path) -> Any:
        return json.loads(self.read_bytes(raw_path).decode("utf-8"))

    def load_manifest(self) -> dict[str, Any]:
        payload = self.load_json("corpus_manifest.json")
        if not isinstance(payload, dict):
            raise ValueError(
                f"corpus_manifest.json in {self.corpus_id!r} is not a JSON object"
            )
        return payload

    def stable_file_hash(self, raw_path: str | Path, *, digest_size: int = 16) -> str:
        digest = hashlib.blake2b(digest_size=digest_size)
        digest.update(self.read_bytes(raw_path))
        return digest.hexdigest()

    def task_spec(self) -> dict[str, str]:
        return {
            "kind": "sqlite_graph_corpus",
            "db_path": str(self.db_path),
            "corpus_id": self.corpus_id,
        }

    def _normalize_relative_path(self, raw_path: str | Path) -> str:
        raw = str(raw_path).replace("\\", "/").lstrip("./")
        path = Path(raw_path)
        if path.is_absolute() and self.root_path is not None:
            try:
                return str(path.resolve().relative_to(self.root_path.resolve())).replace("\\", "/")
            except ValueError:
                pass
        if self.root_path is not None:
            for prefix in self._corpus_root_prefixes():
                if raw == prefix:
                    return ""
                if raw.startswith(prefix + "/"):
                    return raw[len(prefix) + 1 :]
            parts = raw.split("/")
            if self.root_path.name in parts:
                index = parts.index(self.root_path.name)
                if index + 1 < len(parts):
                    return "/".join(parts[index + 1 :])
        return raw

    def _corpus_root_prefixes(self) -> tuple[str, ...]:
        if self.root_path is None:
            return ()
        prefixes = {
            str(self.root_path).replace("\\", "/").strip("/"),
            self.root_path.name,
        }
        try:
            prefixes.add(
                str(self.root_path.resolve()).replace("\\", "/").strip("/")
            )
        except OSError:
            pass
        try:
            prefixes.add(
                str(self.root_path.resolve().relative_to(Path.cwd().resolve()))
                .replace("\\", "/")
                .strip("/")
            )
        except (OSError, ValueError):
            pass
        return tuple(sorted((p.lstrip("./") for p in prefixes if p), key=len, reverse=True))

    def _unique_suffix_match(
        self,
        con: sqlite3.Connection,
        relative_path: str,
    ) -> sqlite3.Row | None:
        suffix = "/" + relative_path
        rows = con.execute(
            """
            SELECT payload_blob
            FROM source_files
            WHERE source_group = 'graph_corpus'
              AND source_id = ?
              AND relative_path LIKE ?
            """,
            (self.corpus_id, f"%{suffix}"),
        ).fetchall()
        if len(rows) == 1:
            return rows[0]
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_corpus_data_dir(run_path: str | Path) -> Path:
    """Resolve a run root, outer run collection, or data dir to its data dir."""
    path = Path(run_path).resolve()
    if path.name == "data" and (path / "corpus_checkpoint.json").exists():
        return path
    if (path / "data" / "corpus_checkpoint.json").exists():
        return path / "data"

    candidates = sorted(
        path.glob("*/data/corpus_checkpoint.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].parent

    raise FileNotFoundError(
        f"Could not find data/corpus_checkpoint.json under {path}"
    )


def connect_corpus_db(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=30000")
    initialize_schema(con)
    return con


def resolve_graph_corpus_id(db_path: str | Path, corpus_id: str | None = None) -> str:
    con = connect_corpus_db(db_path)
    try:
        rows = con.execute(
            """
            SELECT corpus_id
            FROM graph_corpora
            ORDER BY corpus_id
            """
        ).fetchall()
    finally:
        con.close()
    ids = [str(row["corpus_id"]) for row in rows]
    if corpus_id:
        if corpus_id not in ids:
            raise ValueError(
                f"graph corpus {corpus_id!r} not found in {db_path}; "
                f"available corpus IDs: {', '.join(ids) or '(none)'}"
            )
        return corpus_id
    if len(ids) == 1:
        return ids[0]
    if not ids:
        raise ValueError(f"no graph corpora are indexed in {db_path}")
    raise ValueError(
        f"{db_path} contains multiple graph corpora; choose one with --corpus-id: "
        + ", ".join(ids)
    )


def initialize_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_info (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS run_metadata (
            key TEXT PRIMARY KEY,
            relative_path TEXT NOT NULL UNIQUE,
            payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
            payload_bytes INTEGER NOT NULL,
            source_mtime REAL NOT NULL,
            ingested_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS corpus_cases (
            case_id TEXT PRIMARY KEY,
            status TEXT,
            graph_id TEXT,
            assignment_id TEXT,
            combo_key TEXT,
            seed INTEGER,
            num_steps INTEGER,
            route_policy TEXT,
            eve_win_rate REAL,
            final_win_rate REAL,
            num_routes INTEGER,
            num_actions INTEGER,
            route_enumeration_complete INTEGER,
            route_enumeration_truncated INTEGER,
            payload_json TEXT CHECK (payload_json IS NULL OR json_valid(payload_json)),
            payload_source TEXT,
            source_checkpoint_path TEXT,
            checkpoint_payload_bytes INTEGER,
            worker_status_path TEXT,
            worker_status_updated_at TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS worker_status (
            case_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            pid INTEGER,
            updated_at TEXT,
            graph_id TEXT,
            assignment_id TEXT,
            combo_key TEXT,
            seed INTEGER,
            num_steps INTEGER,
            route_policy TEXT,
            payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
            relative_path TEXT NOT NULL UNIQUE,
            payload_bytes INTEGER NOT NULL,
            source_mtime REAL NOT NULL,
            ingested_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS json_artifacts (
            relative_path TEXT PRIMARY KEY,
            case_id TEXT,
            artifact_kind TEXT NOT NULL,
            payload_json TEXT NOT NULL CHECK (json_valid(payload_json)),
            payload_bytes INTEGER NOT NULL,
            source_mtime REAL NOT NULL,
            ingested_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS graph_corpora (
            corpus_id TEXT PRIMARY KEY,
            root_path TEXT NOT NULL,
            manifest_relative_path TEXT,
            manifest_graphs INTEGER,
            manifest_assignments INTEGER,
            file_count INTEGER NOT NULL,
            total_bytes INTEGER NOT NULL,
            source_mtime REAL NOT NULL,
            ingested_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS graph_corpus_graphs (
            corpus_id TEXT NOT NULL,
            graph_id TEXT NOT NULL,
            family TEXT,
            source TEXT,
            scope_label TEXT,
            physical_role TEXT,
            node_count INTEGER,
            edge_count INTEGER,
            edge_length_semantics TEXT,
            assignment_count INTEGER,
            topology_relative_path TEXT,
            summary_json TEXT CHECK (summary_json IS NULL OR json_valid(summary_json)),
            ingested_at TEXT NOT NULL,
            PRIMARY KEY (corpus_id, graph_id)
        );

        CREATE TABLE IF NOT EXISTS graph_corpus_assignments (
            corpus_id TEXT NOT NULL,
            graph_id TEXT NOT NULL,
            assignment_id TEXT NOT NULL,
            alice TEXT NOT NULL,
            bob TEXT NOT NULL,
            assignment_policy TEXT,
            strategy_validity TEXT,
            route_count INTEGER,
            shortest_hops INTEGER,
            longest_hops INTEGER,
            shortest_path_count INTEGER,
            node_cut_size INTEGER,
            edge_cut_size INTEGER,
            route_enumeration_truncated INTEGER,
            assignment_relative_path TEXT,
            summary_json TEXT CHECK (summary_json IS NULL OR json_valid(summary_json)),
            ingested_at TEXT NOT NULL,
            PRIMARY KEY (corpus_id, graph_id, assignment_id)
        );

        CREATE TABLE IF NOT EXISTS source_files (
            source_group TEXT NOT NULL,
            source_id TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            file_kind TEXT NOT NULL,
            case_id TEXT,
            graph_id TEXT,
            assignment_id TEXT,
            suffix TEXT NOT NULL,
            payload_blob BLOB NOT NULL,
            source_sha256 TEXT NOT NULL,
            source_size INTEGER NOT NULL,
            source_mtime REAL NOT NULL,
            ingested_at TEXT NOT NULL,
            PRIMARY KEY (source_group, source_id, relative_path)
        );

        CREATE INDEX IF NOT EXISTS idx_corpus_cases_status
            ON corpus_cases(status);
        CREATE INDEX IF NOT EXISTS idx_corpus_cases_combo
            ON corpus_cases(combo_key);
        CREATE INDEX IF NOT EXISTS idx_corpus_cases_graph
            ON corpus_cases(graph_id);
        CREATE INDEX IF NOT EXISTS idx_json_artifacts_case
            ON json_artifacts(case_id);
        CREATE INDEX IF NOT EXISTS idx_json_artifacts_kind
            ON json_artifacts(artifact_kind);
        CREATE INDEX IF NOT EXISTS idx_source_files_case
            ON source_files(case_id);
        CREATE INDEX IF NOT EXISTS idx_source_files_graph
            ON source_files(graph_id);
        CREATE INDEX IF NOT EXISTS idx_source_files_kind
            ON source_files(source_group, file_kind);
        CREATE INDEX IF NOT EXISTS idx_gc_graphs_source
            ON graph_corpus_graphs(source);
        CREATE INDEX IF NOT EXISTS idx_gc_graphs_family
            ON graph_corpus_graphs(family);
        CREATE INDEX IF NOT EXISTS idx_gc_assignments_graph
            ON graph_corpus_assignments(corpus_id, graph_id);
        CREATE INDEX IF NOT EXISTS idx_gc_assignments_hops
            ON graph_corpus_assignments(shortest_hops);
        CREATE INDEX IF NOT EXISTS idx_gc_assignments_validity
            ON graph_corpus_assignments(strategy_validity);
        CREATE INDEX IF NOT EXISTS idx_gc_assignments_policy
            ON graph_corpus_assignments(assignment_policy);
        CREATE INDEX IF NOT EXISTS idx_gc_assignments_pair
            ON graph_corpus_assignments(alice, bob);
        """
    )
    con.execute(
        """
        INSERT INTO schema_info(key, value)
        VALUES ('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value=excluded.value
        """,
        (str(SCHEMA_VERSION),),
    )
    con.commit()


def ingest_run_to_sqlite(
    run_path: str | Path,
    db_path: str | Path | None = None,
    *,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 25,
    refresh_checkpoint: bool = False,
    ingest_source_files: bool = False,
) -> IngestSummary:
    data_dir = resolve_corpus_data_dir(run_path)
    run_root = data_dir.parent
    target_db = Path(db_path).resolve() if db_path is not None else data_dir / "corpus_results.sqlite"

    checkpoint_path = data_dir / "corpus_checkpoint.json"
    con = connect_corpus_db(target_db)
    try:
        metadata_count = ingest_run_metadata(con, data_dir, run_root)
        worker_count = ingest_worker_statuses(con, data_dir, run_root)
        checkpoint_count = ingest_checkpoint_records(
            con,
            checkpoint_path,
            run_root,
            progress=progress,
            progress_every=progress_every,
            refresh_existing=refresh_checkpoint,
        )
        artifact_count = ingest_json_artifacts(con, run_root, data_dir)
        source_file_count = 0
        if ingest_source_files:
            source_file_count = ingest_run_source_files(
                con,
                run_root,
                data_dir,
                target_db,
                progress=progress,
                progress_every=progress_every,
            )
        con.commit()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()

    return IngestSummary(
        db_path=target_db,
        run_root=run_root,
        data_dir=data_dir,
        checkpoint_records=checkpoint_count,
        worker_status_records=worker_count,
        json_artifacts=artifact_count,
        metadata_records=metadata_count,
        source_file_records=source_file_count,
    )


def ingest_graph_corpus_to_sqlite(
    corpus_path: str | Path,
    db_path: str | Path,
    *,
    corpus_id: str | None = None,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 100,
) -> GraphCorpusIngestSummary:
    corpus_root = Path(corpus_path).resolve()
    if not corpus_root.exists():
        raise FileNotFoundError(corpus_root)
    target_db = Path(db_path).resolve()
    resolved_corpus_id = corpus_id or corpus_root.name
    con = connect_corpus_db(target_db)
    try:
        manifest = _load_optional_manifest(corpus_root)
        file_count, total_bytes = _source_file_stats(corpus_root, target_db)
        source_count = ingest_source_files(
            con,
            root=corpus_root,
            source_group="graph_corpus",
            source_id=resolved_corpus_id,
            db_path=target_db,
            progress=progress,
            progress_every=progress_every,
        )
        indexed_graphs, indexed_assignments = index_graph_corpus_structure(
            con,
            resolved_corpus_id,
            manifest,
        )
        con.execute(
            """
            INSERT INTO graph_corpora(
                corpus_id, root_path, manifest_relative_path, manifest_graphs,
                manifest_assignments, file_count, total_bytes, source_mtime,
                ingested_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(corpus_id) DO UPDATE SET
                root_path=excluded.root_path,
                manifest_relative_path=excluded.manifest_relative_path,
                manifest_graphs=excluded.manifest_graphs,
                manifest_assignments=excluded.manifest_assignments,
                file_count=excluded.file_count,
                total_bytes=excluded.total_bytes,
                source_mtime=excluded.source_mtime,
                ingested_at=excluded.ingested_at
            """,
            (
                resolved_corpus_id,
                str(corpus_root),
                "corpus_manifest.json" if (corpus_root / "corpus_manifest.json").exists() else None,
                _manifest_count(manifest, "graphs"),
                _manifest_count(manifest, "assignments"),
                file_count,
                total_bytes,
                corpus_root.stat().st_mtime,
                utc_now_iso(),
            ),
        )
        con.commit()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        con.close()

    return GraphCorpusIngestSummary(
        db_path=target_db,
        corpus_root=corpus_root,
        corpus_id=resolved_corpus_id,
        source_file_records=source_count,
        manifest_graphs=_manifest_count(manifest, "graphs"),
        manifest_assignments=_manifest_count(manifest, "assignments"),
        total_bytes=total_bytes,
        indexed_graphs=indexed_graphs,
        indexed_assignments=indexed_assignments,
    )


def index_graph_corpus_structure(
    con: sqlite3.Connection,
    corpus_id: str,
    manifest: dict[str, Any] | None,
) -> tuple[int, int]:
    """Index manifest graph/assignment rows into structured, queryable tables.

    Re-indexing a corpus replaces its previous rows so removed graphs do not
    linger. Returns ``(graph_rows, assignment_rows)``.
    """
    if manifest is None:
        return 0, 0
    graphs = manifest.get("graphs")
    assignments = manifest.get("assignments")
    if not isinstance(graphs, list):
        graphs = []
    if not isinstance(assignments, list):
        assignments = []

    con.execute(
        "DELETE FROM graph_corpus_graphs WHERE corpus_id = ?", (corpus_id,)
    )
    con.execute(
        "DELETE FROM graph_corpus_assignments WHERE corpus_id = ?", (corpus_id,)
    )

    now = utc_now_iso()
    graph_count = 0
    for record in graphs:
        if not isinstance(record, dict) or not record.get("graph_id"):
            continue
        con.execute(
            """
            INSERT INTO graph_corpus_graphs(
                corpus_id, graph_id, family, source, scope_label, physical_role,
                node_count, edge_count, edge_length_semantics, assignment_count,
                topology_relative_path, summary_json, ingested_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(corpus_id, graph_id) DO UPDATE SET
                family=excluded.family,
                source=excluded.source,
                scope_label=excluded.scope_label,
                physical_role=excluded.physical_role,
                node_count=excluded.node_count,
                edge_count=excluded.edge_count,
                edge_length_semantics=excluded.edge_length_semantics,
                assignment_count=excluded.assignment_count,
                topology_relative_path=excluded.topology_relative_path,
                summary_json=excluded.summary_json,
                ingested_at=excluded.ingested_at
            """,
            (
                corpus_id,
                str(record["graph_id"]),
                _as_text(record.get("family")),
                _as_text(record.get("source")),
                _as_text(record.get("scope_label")),
                _as_text(record.get("physical_role")),
                _as_int(record.get("node_count")),
                _as_int(record.get("edge_count")),
                _as_text(record.get("edge_length_semantics")),
                _as_int(record.get("assignment_count")),
                _as_text(record.get("topology_path")),
                _json_dumps(record),
                now,
            ),
        )
        graph_count += 1

    assignment_count = 0
    for record in assignments:
        if not isinstance(record, dict):
            continue
        if not record.get("graph_id") or not record.get("assignment_id"):
            continue
        if record.get("alice") is None or record.get("bob") is None:
            continue
        con.execute(
            """
            INSERT INTO graph_corpus_assignments(
                corpus_id, graph_id, assignment_id, alice, bob,
                assignment_policy, strategy_validity, route_count,
                shortest_hops, longest_hops, shortest_path_count,
                node_cut_size, edge_cut_size, route_enumeration_truncated,
                assignment_relative_path, summary_json, ingested_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(corpus_id, graph_id, assignment_id) DO UPDATE SET
                alice=excluded.alice,
                bob=excluded.bob,
                assignment_policy=excluded.assignment_policy,
                strategy_validity=excluded.strategy_validity,
                route_count=excluded.route_count,
                shortest_hops=excluded.shortest_hops,
                longest_hops=excluded.longest_hops,
                shortest_path_count=excluded.shortest_path_count,
                node_cut_size=excluded.node_cut_size,
                edge_cut_size=excluded.edge_cut_size,
                route_enumeration_truncated=excluded.route_enumeration_truncated,
                assignment_relative_path=excluded.assignment_relative_path,
                summary_json=excluded.summary_json,
                ingested_at=excluded.ingested_at
            """,
            (
                corpus_id,
                str(record["graph_id"]),
                str(record["assignment_id"]),
                str(record["alice"]),
                str(record["bob"]),
                _as_text(record.get("assignment_policy")),
                _as_text(record.get("strategy_validity")),
                _as_int(record.get("route_count")),
                _as_int(record.get("shortest_hops")),
                _as_int(record.get("longest_hops")),
                _as_int(record.get("shortest_path_count")),
                _as_int(record.get("node_cut_size")),
                _as_int(record.get("edge_cut_size")),
                _as_bool_int(record.get("route_enumeration_truncated")),
                _as_text(record.get("assignment_path")),
                _json_dumps(record),
                now,
            ),
        )
        assignment_count += 1
    return graph_count, assignment_count


def reindex_graph_corpus_structure(
    db_path: str | Path,
    corpus_id: str | None = None,
) -> tuple[int, int]:
    """Rebuild the structured graph/assignment index from a stored manifest."""
    source = SQLiteGraphCorpusSource.open(db_path, corpus_id)
    manifest = source.load_manifest()
    con = connect_corpus_db(db_path)
    try:
        counts = index_graph_corpus_structure(con, source.corpus_id, manifest)
        con.commit()
        return counts
    finally:
        con.close()


def query_graph_corpus_graphs(
    db_path: str | Path,
    *,
    corpus_id: str | None = None,
    source: str | None = None,
    family: str | None = None,
    physical_role: str | None = None,
    edge_length_semantics: str | None = None,
    min_node_count: int | None = None,
    max_node_count: int | None = None,
) -> list[dict[str, Any]]:
    """Query indexed corpus graphs; returns one dict per graph row."""
    clauses = []
    params: list[Any] = []
    for column, value in (
        ("corpus_id", corpus_id),
        ("source", source),
        ("family", family),
        ("physical_role", physical_role),
        ("edge_length_semantics", edge_length_semantics),
    ):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    if min_node_count is not None:
        clauses.append("node_count >= ?")
        params.append(min_node_count)
    if max_node_count is not None:
        clauses.append("node_count <= ?")
        params.append(max_node_count)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    con = connect_corpus_db(db_path)
    try:
        rows = con.execute(
            f"""
            SELECT *
            FROM graph_corpus_graphs
            {where}
            ORDER BY corpus_id, graph_id
            """,
            params,
        ).fetchall()
    finally:
        con.close()
    return [dict(row) for row in rows]


def query_graph_corpus_assignments(
    db_path: str | Path,
    *,
    corpus_id: str | None = None,
    graph_id: str | None = None,
    source: str | None = None,
    family: str | None = None,
    physical_role: str | None = None,
    assignment_policy: str | None = None,
    strategy_validity: str | None = None,
    min_shortest_hops: int | None = None,
    max_shortest_hops: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Query indexed Alice/Bob assignments joined with their graph metadata.

    Each returned dict carries the assignment columns (``alice``, ``bob``,
    ``assignment_policy``, ``strategy_validity``, ``shortest_hops``, ...) plus
    the parent graph's ``family``, ``source``, ``physical_role``,
    ``node_count``, ``edge_count``, and ``edge_length_semantics``.
    """
    clauses = []
    params: list[Any] = []
    for column, value in (
        ("a.corpus_id", corpus_id),
        ("a.graph_id", graph_id),
        ("g.source", source),
        ("g.family", family),
        ("g.physical_role", physical_role),
        ("a.assignment_policy", assignment_policy),
        ("a.strategy_validity", strategy_validity),
    ):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    if min_shortest_hops is not None:
        clauses.append("a.shortest_hops >= ?")
        params.append(min_shortest_hops)
    if max_shortest_hops is not None:
        clauses.append("a.shortest_hops <= ?")
        params.append(max_shortest_hops)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT ?"
        params.append(int(limit))
    con = connect_corpus_db(db_path)
    try:
        rows = con.execute(
            f"""
            SELECT
                a.*,
                g.family, g.source, g.scope_label, g.physical_role,
                g.node_count, g.edge_count, g.edge_length_semantics
            FROM graph_corpus_assignments AS a
            LEFT JOIN graph_corpus_graphs AS g
                ON g.corpus_id = a.corpus_id AND g.graph_id = a.graph_id
            {where}
            ORDER BY a.corpus_id, a.graph_id, a.assignment_id
            {limit_sql}
            """,
            params,
        ).fetchall()
    finally:
        con.close()
    return [dict(row) for row in rows]


def load_graph_corpus_topology(
    db_path: str | Path,
    graph_id: str,
    *,
    corpus_id: str | None = None,
) -> dict[str, Any]:
    """Load the stored ``topology.json`` payload for an indexed corpus graph."""
    source = SQLiteGraphCorpusSource.open(db_path, corpus_id)
    con = connect_corpus_db(db_path)
    try:
        row = con.execute(
            """
            SELECT topology_relative_path
            FROM graph_corpus_graphs
            WHERE corpus_id = ? AND graph_id = ?
            """,
            (source.corpus_id, graph_id),
        ).fetchone()
    finally:
        con.close()
    relative_path = row["topology_relative_path"] if row else None
    if not relative_path:
        relative_path = f"accepted/{graph_id}/topology.json"
    payload = source.load_json(relative_path)
    if not isinstance(payload, dict):
        raise ValueError(
            f"topology payload for graph {graph_id!r} in corpus "
            f"{source.corpus_id!r} is not a JSON object"
        )
    return payload


def ingest_run_metadata(
    con: sqlite3.Connection,
    data_dir: Path,
    run_root: Path,
) -> int:
    count = 0
    for path in sorted(data_dir.glob("*.json")):
        if path.name == "corpus_checkpoint.json":
            continue
        payload = _load_json(path)
        _upsert_run_metadata(con, run_root, path, payload)
        count += 1
    figures_dir = run_root / "figures"
    if figures_dir.exists():
        for path in sorted(figures_dir.glob("*summary.json")):
            payload = _load_json(path)
            _upsert_run_metadata(con, run_root, path, payload)
            count += 1
    con.commit()
    return count


def ingest_worker_statuses(
    con: sqlite3.Connection,
    data_dir: Path,
    run_root: Path,
) -> int:
    status_dir = data_dir / "corpus_worker_status"
    if not status_dir.exists():
        return 0

    count = 0
    for path in sorted(status_dir.glob("*.json")):
        payload = _load_json(path)
        case_id = str(payload.get("case_id") or path.stem)
        _upsert_worker_status(con, run_root, path, case_id, payload)
        _upsert_case_from_worker_status(con, run_root, path, case_id, payload)
        count += 1
    con.commit()
    return count


def ingest_checkpoint_records(
    con: sqlite3.Connection,
    checkpoint_path: Path,
    run_root: Path,
    *,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 25,
    refresh_existing: bool = False,
) -> int:
    if not checkpoint_path.exists():
        return 0

    count = 0
    existing_checkpoint_ids: set[str] = set()
    if not refresh_existing:
        existing_checkpoint_ids = {
            str(row[0])
            for row in con.execute(
                """
                SELECT case_id
                FROM corpus_cases
                WHERE payload_source = 'checkpoint'
                """
            )
        }
    relative_path = _relative_path(run_root, checkpoint_path)
    for record in iter_json_array(checkpoint_path):
        if not isinstance(record, dict):
            continue
        case_id = _case_id_for_record(record)
        if case_id not in existing_checkpoint_ids:
            _upsert_checkpoint_case(con, relative_path, record, case_id=case_id)
        count += 1
        if count % progress_every == 0:
            con.commit()
            if progress is not None:
                progress(f"checkpoint records ingested: {count}")
    con.commit()
    return count


def ingest_json_artifacts(
    con: sqlite3.Connection,
    run_root: Path,
    data_dir: Path,
) -> int:
    skip_dirs = {
        data_dir / "corpus_worker_status",
    }
    count = 0
    for path in sorted(run_root.rglob("*.json")):
        if path == data_dir / "corpus_checkpoint.json":
            continue
        if any(_is_relative_to(path, skip_dir) for skip_dir in skip_dirs):
            continue
        if path.parent == data_dir:
            continue
        if path.parent.name == "figures" and path.name.endswith("summary.json"):
            continue

        payload = _load_json(path)
        _upsert_json_artifact(con, run_root, path, payload)
        count += 1
    con.commit()
    return count


def ingest_run_source_files(
    con: sqlite3.Connection,
    run_root: Path,
    data_dir: Path,
    db_path: Path,
    *,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 100,
) -> int:
    def include(path: Path) -> bool:
        if path == data_dir / "corpus_checkpoint.json":
            return False
        return True

    return ingest_source_files(
        con,
        root=run_root,
        source_group="run_results",
        source_id=run_root.name,
        db_path=db_path,
        include=include,
        progress=progress,
        progress_every=progress_every,
    )


def ingest_source_files(
    con: sqlite3.Connection,
    *,
    root: Path,
    source_group: str,
    source_id: str,
    db_path: Path,
    include: Callable[[Path], bool] | None = None,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 100,
) -> int:
    count = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if _is_sqlite_sidecar(path, db_path):
            continue
        if include is not None and not include(path):
            continue
        _upsert_source_file(con, root, path, source_group, source_id)
        count += 1
        if count % progress_every == 0:
            con.commit()
            if progress is not None:
                progress(f"{source_group} source files ingested: {count}")
    con.commit()
    return count


def upsert_corpus_case_record(
    con: sqlite3.Connection,
    record: dict[str, Any],
    *,
    payload_source: str = "live_result",
    source_path: str = "sqlite_live_result",
) -> str:
    case_id = _case_id_for_record(record)
    _upsert_checkpoint_case(
        con,
        source_path,
        record,
        case_id=case_id,
        payload_source=payload_source,
    )
    return case_id


def write_corpus_case_record(
    db_path: str | Path,
    record: dict[str, Any],
    *,
    payload_source: str = "live_result",
    source_path: str = "sqlite_live_result",
) -> str:
    con = connect_corpus_db(db_path)
    try:
        case_id = upsert_corpus_case_record(
            con,
            record,
            payload_source=payload_source,
            source_path=source_path,
        )
        con.commit()
        return case_id
    finally:
        con.close()


def load_completed_corpus_case_records(
    db_path: str | Path,
    *,
    payload_sources: tuple[str, ...] = ("live_result", "checkpoint"),
) -> list[dict[str, Any]]:
    con = connect_corpus_db(db_path)
    try:
        placeholders = ",".join("?" for _ in payload_sources)
        rows = con.execute(
            f"""
            SELECT payload_json
            FROM corpus_cases
            WHERE status = 'completed'
              AND payload_json IS NOT NULL
              AND payload_source IN ({placeholders})
            ORDER BY updated_at, case_id
            """,
            payload_sources,
        ).fetchall()
    finally:
        con.close()
    records: list[dict[str, Any]] = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        if isinstance(payload, dict):
            records.append(payload)
    return records


def upsert_worker_status_payload(
    con: sqlite3.Connection,
    payload: dict[str, Any],
    *,
    relative_path: str | None = None,
) -> str:
    case_id = str(payload.get("case_id") or _stable_digest(payload))
    resolved_relative_path = relative_path or f"corpus_worker_status/{case_id}.json"
    _upsert_worker_status_payload(con, resolved_relative_path, case_id, payload)
    _upsert_case_from_worker_status_payload(con, resolved_relative_path, case_id, payload)
    return case_id


def write_worker_status_payload(
    db_path: str | Path,
    payload: dict[str, Any],
    *,
    relative_path: str | None = None,
) -> str:
    con = connect_corpus_db(db_path)
    try:
        case_id = upsert_worker_status_payload(
            con,
            payload,
            relative_path=relative_path,
        )
        con.commit()
        return case_id
    finally:
        con.close()


def iter_json_array(path: str | Path, *, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Yield top-level values from a JSON array without loading the whole file."""
    decoder = JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as handle:
        buffer = ""
        started = False
        finished = False

        while not finished:
            if not buffer:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                buffer += chunk

            buffer = buffer.lstrip()
            if not buffer:
                continue

            if not started:
                if buffer[0] != "[":
                    raise ValueError(f"{path} is not a JSON array")
                buffer = buffer[1:]
                started = True
                continue

            buffer = buffer.lstrip()
            if not buffer:
                continue
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue
            if buffer[0] == "]":
                finished = True
                break

            while True:
                try:
                    value, index = decoder.raw_decode(buffer)
                except JSONDecodeError:
                    chunk = handle.read(chunk_size)
                    if not chunk:
                        raise
                    buffer += chunk
                    continue
                yield value
                buffer = buffer[index:]
                break


def _upsert_run_metadata(
    con: sqlite3.Connection,
    run_root: Path,
    path: Path,
    payload: Any,
) -> None:
    payload_json = _json_dumps(payload)
    stat = path.stat()
    relative_path = _relative_path(run_root, path)
    con.execute(
        """
        INSERT INTO run_metadata(
            key, relative_path, payload_json, payload_bytes, source_mtime, ingested_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET
            relative_path=excluded.relative_path,
            payload_json=excluded.payload_json,
            payload_bytes=excluded.payload_bytes,
            source_mtime=excluded.source_mtime,
            ingested_at=excluded.ingested_at
        """,
        (
            path.stem,
            relative_path,
            payload_json,
            len(payload_json.encode("utf-8")),
            stat.st_mtime,
            utc_now_iso(),
        ),
    )


def _upsert_worker_status(
    con: sqlite3.Connection,
    run_root: Path,
    path: Path,
    case_id: str,
    payload: dict[str, Any],
) -> None:
    stat = path.stat()
    _upsert_worker_status_payload(
        con,
        _relative_path(run_root, path),
        case_id,
        payload,
        source_mtime=stat.st_mtime,
    )


def _upsert_worker_status_payload(
    con: sqlite3.Connection,
    relative_path: str,
    case_id: str,
    payload: dict[str, Any],
    *,
    source_mtime: float | None = None,
) -> None:
    payload_json = _json_dumps(payload)
    con.execute(
        """
        INSERT INTO worker_status(
            case_id, status, pid, updated_at, graph_id, assignment_id, combo_key,
            seed, num_steps, route_policy, payload_json, relative_path,
            payload_bytes, source_mtime, ingested_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(case_id) DO UPDATE SET
            status=excluded.status,
            pid=excluded.pid,
            updated_at=excluded.updated_at,
            graph_id=excluded.graph_id,
            assignment_id=excluded.assignment_id,
            combo_key=excluded.combo_key,
            seed=excluded.seed,
            num_steps=excluded.num_steps,
            route_policy=excluded.route_policy,
            payload_json=excluded.payload_json,
            relative_path=excluded.relative_path,
            payload_bytes=excluded.payload_bytes,
            source_mtime=excluded.source_mtime,
            ingested_at=excluded.ingested_at
        """,
        (
            case_id,
            _as_text(payload.get("status")),
            _as_int(payload.get("pid")),
            _as_text(payload.get("updated_at")),
            _as_text(payload.get("graph_id")),
            _as_text(payload.get("assignment_id")),
            _as_text(payload.get("combo_key")),
            _as_int(payload.get("seed")),
            _as_int(payload.get("num_steps")),
            _as_text(payload.get("route_policy")),
            payload_json,
            relative_path,
            len(payload_json.encode("utf-8")),
            float(source_mtime if source_mtime is not None else 0.0),
            utc_now_iso(),
        ),
    )


def _upsert_case_from_worker_status(
    con: sqlite3.Connection,
    run_root: Path,
    path: Path,
    case_id: str,
    payload: dict[str, Any],
) -> None:
    _upsert_case_from_worker_status_payload(
        con,
        _relative_path(run_root, path),
        case_id,
        payload,
    )


def _upsert_case_from_worker_status_payload(
    con: sqlite3.Connection,
    relative_path: str,
    case_id: str,
    payload: dict[str, Any],
) -> None:
    payload_json = _json_dumps(payload)
    con.execute(
        """
        INSERT INTO corpus_cases(
            case_id, status, graph_id, assignment_id, combo_key, seed, num_steps,
            route_policy, payload_json, payload_source, worker_status_path,
            worker_status_updated_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'worker_status', ?, ?, ?)
        ON CONFLICT(case_id) DO UPDATE SET
            status=excluded.status,
            graph_id=COALESCE(corpus_cases.graph_id, excluded.graph_id),
            assignment_id=COALESCE(corpus_cases.assignment_id, excluded.assignment_id),
            combo_key=COALESCE(corpus_cases.combo_key, excluded.combo_key),
            seed=COALESCE(corpus_cases.seed, excluded.seed),
            num_steps=COALESCE(corpus_cases.num_steps, excluded.num_steps),
            route_policy=COALESCE(corpus_cases.route_policy, excluded.route_policy),
            payload_json=CASE
                WHEN corpus_cases.payload_source IN ('checkpoint', 'live_result')
                    THEN corpus_cases.payload_json
                ELSE excluded.payload_json
            END,
            payload_source=CASE
                WHEN corpus_cases.payload_source IN ('checkpoint', 'live_result')
                    THEN corpus_cases.payload_source
                ELSE excluded.payload_source
            END,
            worker_status_path=excluded.worker_status_path,
            worker_status_updated_at=excluded.worker_status_updated_at,
            updated_at=excluded.updated_at
        """,
        (
            case_id,
            _as_text(payload.get("status")),
            _as_text(payload.get("graph_id")),
            _as_text(payload.get("assignment_id")),
            _as_text(payload.get("combo_key")),
            _as_int(payload.get("seed")),
            _as_int(payload.get("num_steps")),
            _as_text(payload.get("route_policy")),
            payload_json,
            relative_path,
            _as_text(payload.get("updated_at")),
            utc_now_iso(),
        ),
    )


def _upsert_checkpoint_case(
    con: sqlite3.Connection,
    relative_checkpoint_path: str,
    record: dict[str, Any],
    *,
    case_id: str | None = None,
    payload_source: str = "checkpoint",
) -> None:
    case_id = case_id or _case_id_for_record(record)
    payload_json = _json_dumps(record)
    con.execute(
        """
        INSERT INTO corpus_cases(
            case_id, status, graph_id, assignment_id, combo_key, seed, num_steps,
            route_policy, eve_win_rate, final_win_rate, num_routes, num_actions,
            route_enumeration_complete, route_enumeration_truncated, payload_json,
            payload_source, source_checkpoint_path, checkpoint_payload_bytes,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(case_id) DO UPDATE SET
            status=excluded.status,
            graph_id=excluded.graph_id,
            assignment_id=excluded.assignment_id,
            combo_key=excluded.combo_key,
            seed=excluded.seed,
            num_steps=excluded.num_steps,
            route_policy=excluded.route_policy,
            eve_win_rate=excluded.eve_win_rate,
            final_win_rate=excluded.final_win_rate,
            num_routes=excluded.num_routes,
            num_actions=excluded.num_actions,
            route_enumeration_complete=excluded.route_enumeration_complete,
            route_enumeration_truncated=excluded.route_enumeration_truncated,
            payload_json=excluded.payload_json,
            payload_source=excluded.payload_source,
            source_checkpoint_path=excluded.source_checkpoint_path,
            checkpoint_payload_bytes=excluded.checkpoint_payload_bytes,
            updated_at=excluded.updated_at
        """,
        (
            case_id,
            _as_text(record.get("status", "completed")),
            _as_text(record.get("graph_id")),
            _as_text(record.get("assignment_id")),
            _as_text(record.get("combo_key")),
            _as_int(record.get("seed", record.get("run_seed"))),
            _as_int(record.get("num_steps")),
            _as_text(record.get("route_policy", record.get("route_selection_policy"))),
            _as_float(record.get("eve_win_rate")),
            _as_float(record.get("final_win_rate")),
            _as_int(record.get("num_routes")),
            _as_int(record.get("num_actions")),
            _as_bool_int(record.get("route_enumeration_complete")),
            _as_bool_int(record.get("route_enumeration_truncated")),
            payload_json,
            payload_source,
            relative_checkpoint_path,
            len(payload_json.encode("utf-8")),
            utc_now_iso(),
        ),
    )


def _upsert_json_artifact(
    con: sqlite3.Connection,
    run_root: Path,
    path: Path,
    payload: Any,
) -> None:
    payload_json = _json_dumps(payload)
    stat = path.stat()
    con.execute(
        """
        INSERT INTO json_artifacts(
            relative_path, case_id, artifact_kind, payload_json, payload_bytes,
            source_mtime, ingested_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(relative_path) DO UPDATE SET
            case_id=excluded.case_id,
            artifact_kind=excluded.artifact_kind,
            payload_json=excluded.payload_json,
            payload_bytes=excluded.payload_bytes,
            source_mtime=excluded.source_mtime,
            ingested_at=excluded.ingested_at
        """,
        (
            _relative_path(run_root, path),
            _case_id_for_artifact(path),
            _artifact_kind(path),
            payload_json,
            len(payload_json.encode("utf-8")),
            stat.st_mtime,
            utc_now_iso(),
        ),
    )


def _upsert_source_file(
    con: sqlite3.Connection,
    root: Path,
    path: Path,
    source_group: str,
    source_id: str,
) -> None:
    payload = path.read_bytes()
    stat = path.stat()
    relative_path = _relative_path(root, path)
    con.execute(
        """
        INSERT INTO source_files(
            source_group, source_id, relative_path, file_kind, case_id, graph_id,
            assignment_id, suffix, payload_blob, source_sha256, source_size,
            source_mtime, ingested_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source_group, source_id, relative_path) DO UPDATE SET
            file_kind=excluded.file_kind,
            case_id=excluded.case_id,
            graph_id=excluded.graph_id,
            assignment_id=excluded.assignment_id,
            suffix=excluded.suffix,
            payload_blob=excluded.payload_blob,
            source_sha256=excluded.source_sha256,
            source_size=excluded.source_size,
            source_mtime=excluded.source_mtime,
            ingested_at=excluded.ingested_at
        """,
        (
            source_group,
            source_id,
            relative_path,
            _source_file_kind(path),
            _case_id_for_artifact(path),
            _graph_id_for_source_file(path),
            _assignment_id_for_source_file(path),
            path.suffix.lower(),
            sqlite3.Binary(payload),
            hashlib.sha256(payload).hexdigest(),
            len(payload),
            stat.st_mtime,
            utc_now_iso(),
        ),
    )


def _case_id_for_record(record: dict[str, Any]) -> str:
    case_id = record.get("case_id")
    if case_id:
        return str(case_id)
    basis = {
        "graph_id": record.get("graph_id"),
        "assignment_id": record.get("assignment_id"),
        "combo_key": record.get("combo_key"),
        "seed": record.get("seed", record.get("run_seed")),
        "num_steps": record.get("num_steps"),
    }
    return _stable_digest(basis)


def _case_id_for_artifact(path: Path) -> str | None:
    parts = path.parts
    if "corpus_dt_refit_observer" in parts:
        index = parts.index("corpus_dt_refit_observer")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _artifact_kind(path: Path) -> str:
    parts = path.parts
    if "corpus_dt_refit_observer" in parts:
        if path.parent.name.startswith("refit_"):
            return f"{path.parent.name}:{path.stem}"
        return f"dt:{path.stem}"
    return path.stem


def _source_file_kind(path: Path) -> str:
    if path.name == "corpus_manifest.json":
        return "corpus_manifest"
    if path.name in {
        "assignment_summary.csv",
        "corpus_summary.csv",
        "pruned_or_skipped.jsonl",
    }:
        return path.stem
    if path.name == "topology.json":
        return "graph_topology"
    if path.name == "graph_summary.json":
        return "graph_summary"
    if "assignments" in path.parts and path.suffix.lower() == ".json":
        return "graph_assignment"
    if "corpus_dt_refit_observer" in path.parts:
        if path.parent.name.startswith("refit_"):
            return f"{path.parent.name}:{path.stem}"
        return f"dt:{path.stem}"
    if "corpus_worker_status" in path.parts:
        return "worker_status"
    return path.stem


def _graph_id_for_source_file(path: Path) -> str | None:
    parts = path.parts
    if "accepted" in parts:
        index = parts.index("accepted")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _assignment_id_for_source_file(path: Path) -> str | None:
    if "assignments" in path.parts and path.suffix.lower() == ".json":
        return path.stem
    return None


def _load_optional_manifest(corpus_root: Path) -> dict[str, Any] | None:
    manifest_path = corpus_root / "corpus_manifest.json"
    if not manifest_path.exists():
        return None
    payload = _load_json(manifest_path)
    return payload if isinstance(payload, dict) else None


def _manifest_count(manifest: dict[str, Any] | None, key: str) -> int | None:
    if manifest is None:
        return None
    value = manifest.get(key)
    if isinstance(value, list):
        return len(value)
    if isinstance(value, int):
        return value
    counts = manifest.get("counts")
    if isinstance(counts, dict):
        count_value = counts.get(key)
        if isinstance(count_value, int):
            return count_value
    return None


def _source_file_stats(root: Path, db_path: Path) -> tuple[int, int]:
    count = 0
    total_bytes = 0
    for path in root.rglob("*"):
        if not path.is_file() or _is_sqlite_sidecar(path, db_path):
            continue
        count += 1
        total_bytes += path.stat().st_size
    return count, total_bytes


def _is_sqlite_sidecar(path: Path, db_path: Path) -> bool:
    resolved = path.resolve()
    db = db_path.resolve()
    return resolved in {
        db,
        Path(str(db) + "-wal"),
        Path(str(db) + "-shm"),
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, separators=(",", ":"), default=str, allow_nan=False)
    except ValueError:
        return json.dumps(_sanitize_json(value), separators=(",", ":"), default=str, allow_nan=False)


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    return value


def _stable_digest(value: Any) -> str:
    import hashlib

    payload = _json_dumps(value)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _relative_path(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _as_bool_int(value: Any) -> int | None:
    if value is None:
        return None
    return 1 if bool(value) else 0
