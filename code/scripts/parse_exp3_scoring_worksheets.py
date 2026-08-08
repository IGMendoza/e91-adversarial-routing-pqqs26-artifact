#!/usr/bin/env python3
"""Parse rendered Exp3 scoring worksheets into structured records.

Reads one or more ``worksheet_<model>.md`` files produced by
``render_exp3_scoring_worksheet.py`` and extracts one record per graph:
family, prompt_kind, model, the four mechanical objective columns, and the
``understood`` score/reason (parsed as ``score (reason)`` when a reason is
present, plain integer otherwise, or null if still blank). ``understood``
is seeded from Fable 5's independent scoring pass wherever available --
see ``render_exp3_scoring_worksheet.py``'s docstring.

Standalone and idempotent -- safe to re-run any time a worksheet has more
entries filled in. Writes a single JSON array; downstream plotting/pushing
scripts consume this, not the markdown directly.

Usage:
    .venv/bin/python scripts/parse_exp3_scoring_worksheets.py \
      runs/exp3_dynamic_500k/xai_scoring/worksheet_*.md \
      --out runs/exp3_dynamic_500k/xai_scoring/parsed_scores.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any

_HEADER_RE = re.compile(r"^## (?P<graph_id>\S+)\s+·\s+(?P<family>\S+)\s+·\s+(?P<prompt_kind>\S+)")
_ROW_RE = re.compile(
    r"^\|\s*(?P<faith>[^|]+?)\s*\|\s*(?P<scope>[^|]+?)\s*\|\s*(?P<leaf>[^|]+?)\s*\|"
    r"\s*(?P<sections>[^|]+?)\s*\|\s*(?P<duration>[^|]+?)\s*\|"
    r"\s*(?P<understood>[^|]+?)\s*\|\s*$"
)
_SCORE_REASON_RE = re.compile(r"^(?P<score>\d+)\s*(?:\((?P<reason>.*)\))?\s*$")
_FAITH_PCT_RE = re.compile(r"(\d+)%")
_SECTIONS_FRAC_RE = re.compile(r"(\d+)/(\d+)")
_DURATION_S_RE = re.compile(r"([\d.]+)s")


def parse_worksheet(path: Path) -> list[dict[str, Any]]:
    model_match = re.match(r"worksheet_(.+)\.md$", path.name)
    model_slug = model_match.group(1) if model_match else path.stem

    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    # Only the line immediately after the "|---|...|" separator is the real
    # data row -- the table's own header row ("| numeric_faithfulness | ...")
    # also starts with "|" and matches _ROW_RE just as happily (its cells
    # are just column names), so without this flag it gets consumed as a
    # bogus record and the real data row right after it is silently skipped.
    just_saw_separator = False
    for line in path.read_text(encoding="utf-8").splitlines():
        header_match = _HEADER_RE.match(line)
        if header_match:
            current = header_match.groupdict()
            current["model_slug"] = model_slug
            just_saw_separator = False
            continue
        if line.startswith("|---"):
            just_saw_separator = True
            continue
        if current is None or not just_saw_separator or not line.startswith("|"):
            continue
        just_saw_separator = False
        row_match = _ROW_RE.match(line)
        if not row_match:
            continue
        row = row_match.groupdict()

        faith_pct = _FAITH_PCT_RE.search(row["faith"])
        sections_frac = _SECTIONS_FRAC_RE.search(row["sections"])
        duration_s = _DURATION_S_RE.search(row["duration"])
        understood_match = _SCORE_REASON_RE.match(row["understood"])

        record = dict(current)
        record["numeric_faithfulness_pct"] = int(faith_pct.group(1)) if faith_pct else None
        record["scope_honesty"] = "clean" if row["scope"].strip() == "clean" else "flagged"
        record["scope_honesty_detail"] = row["scope"].strip()
        record["dt_branch_cited"] = row["leaf"].strip().startswith("cited")
        record["dt_branch_detail"] = row["leaf"].strip()
        record["sections_present"] = int(sections_frac.group(1)) if sections_frac else None
        record["sections_total"] = int(sections_frac.group(2)) if sections_frac else None
        record["duration_s"] = float(duration_s.group(1)) if duration_s else None
        if understood_match and understood_match.group("score"):
            record["understood_score"] = int(understood_match.group("score"))
            record["understood_reason"] = (understood_match.group("reason") or "").strip() or None
        else:
            record["understood_score"] = None
            record["understood_reason"] = None
        records.append(record)
        current = None
    return records


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    paths: list[Path] = []
    for pattern in args.worksheets:
        matched = sorted(Path(p) for p in glob.glob(pattern))
        if not matched and Path(pattern).exists():
            matched = [Path(pattern)]
        paths.extend(matched)
    if not paths:
        raise SystemExit(f"no worksheet files matched: {args.worksheets}")

    all_records: list[dict[str, Any]] = []
    for path in paths:
        records = parse_worksheet(path)
        all_records.extend(records)
        n_scored = sum(1 for r in records if r["understood_score"] is not None)
        print(f"{path}: {len(records)} graphs, {n_scored} scored")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    print(f"-> {args.out} ({len(all_records)} total records)")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worksheets", nargs="+", help="Worksheet .md file(s) or glob pattern(s).")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON path.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
