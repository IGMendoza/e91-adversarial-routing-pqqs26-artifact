#!/usr/bin/env python3
"""Normalize Exp3 SeQUeNCe corpus route fiber-length vectors."""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sequence_game.experiments.exp3_sequence.baseline_cache import (  # noqa: E402
    normalize_case_payload,
    standard_route_length_profiles,
)
from sequence_game.experiments.exp3_sequence.config import (  # noqa: E402
    CORPUS_SQLITE_PATH,
    MAX_ROUTE_HOPS,
)
from sequence_game.experiments.exp3_sequence.corpus import (  # noqa: E402
    corpus_summary,
    graph_case_from_dict,
)


def main() -> None:
    db_path = CORPUS_SQLITE_PATH
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT position, payload_json FROM graph_cases ORDER BY position"
        ).fetchall()
        normalized_payloads = [
            (position, normalize_case_payload(json.loads(payload_json)))
            for position, payload_json in rows
        ]
        cases = [
            graph_case_from_dict(payload)
            for _position, payload in normalized_payloads
        ]
        summary = corpus_summary(cases)
        metadata_updates = {
            **summary,
            "route_length_vectors_normalized": True,
            "route_length_profile": "standard_by_hop_count",
            "route_length_profiles_m": {
                str(hops): list(vector)
                for hops, vector in standard_route_length_profiles(MAX_ROUTE_HOPS).items()
            },
            "route_edge_lengths_are_canonical_for_sequence_runtime": True,
            "normalized_by": "scripts/normalize_exp3_sequence_corpus.py",
            "normalized_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        with con:
            con.executemany(
                """
                UPDATE graph_cases
                SET payload_json = ?
                WHERE position = ?
                """,
                [
                    (json.dumps(payload, sort_keys=True), position)
                    for position, payload in normalized_payloads
                ],
            )
            con.executemany(
                """
                INSERT INTO metadata(key, value_json)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json
                """,
                [
                    (str(key), json.dumps(value, sort_keys=True))
                    for key, value in sorted(metadata_updates.items())
                ],
            )
    finally:
        con.close()

    print(f"normalized {len(rows)} graph cases in {db_path}")
    print(
        "profiles="
        + json.dumps({
            str(hops): list(vector)
            for hops, vector in standard_route_length_profiles(MAX_ROUTE_HOPS).items()
        }, sort_keys=True)
    )


if __name__ == "__main__":
    main()
