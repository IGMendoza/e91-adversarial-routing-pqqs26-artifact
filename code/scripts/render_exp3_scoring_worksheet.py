#!/usr/bin/env python3
"""Render a fillable Markdown scoring worksheet from Exp3 LLM digest responses.

Reads response JSONL records written by ``run_exp3_sequence_llm_prompts.py``,
computes four mechanical (no-LLM-judge) checks per response --
``score_faithfulness`` (numeric groundedness), ``check_scope_overclaim``
(security/physical-validity overclaim), ``check_dt_leaf_citation`` (does the
response cite the correct DT leaf value for this specific graph),
``check_section_completeness`` (how many of the 5 requested sections are
actually present, not binary -- missing 1 of 5 is a different problem than
missing 4 of 5) -- plus the response's generation time (``duration_ms``,
already recorded by the digest script, just surfaced here) -- and writes one
worksheet per model: the model's explanation, immediately followed by a row
showing all objective results plus a single column for the one thing that
can't be computed mechanically: did the reviewer actually understand the
explanation, scored 1-10. That column is seeded from Fable 5's independent
(xhigh, blind-to-Claude) scoring pass wherever a Fable 5 score exists --
Fable 5's numbers are the actual scores here, not Claude's own estimates,
since Fable 5 caught real semantic contradictions (e.g. a low retention
value narrated as "high retention") that Claude's own pass missed. Graphs
Fable 5 hasn't scored yet are left blank for a human to fill in.

A fourth candidate check, "selected vs. diagnostic row confusion", was
considered and dropped: models paraphrase routes/actions in prose rather
than citing literal action_id/route_id strings, so a string-match check
would mostly report "no citation found" rather than a trustworthy signal.
Not forced into an "objective" column just to have one.

Safe to re-run any time more responses land -- merges with anything you've
already typed into ``understood:`` fields rather than clobbering them.

Usage:
    .venv/bin/python scripts/render_exp3_scoring_worksheet.py \
      runs/exp3_dynamic_500k/llm_prompts
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sequence_game.xai.cross_graph_semantic import (
    check_dt_leaf_citation,
    check_scope_overclaim,
    check_section_completeness,
)
from sequence_game.xai.faithfulness import score_faithfulness

_UNDERSTOOD_HEADER_RE = re.compile(r"^## (?P<graph_id>\S+)\s+·\s+\S+\s+·\s+(?P<prompt_kind>\S+)")
# Matches a markdown table row and captures only the LAST cell (understood
# is always the final column, regardless of how many objective columns
# precede it).
_UNDERSTOOD_ROW_RE = re.compile(r"^\|(?:[^|]*\|)+\s*(?P<value>[^|]*?)\s*\|\s*$")


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    prompt_dir = Path(args.prompt_dir)
    responses_dir = prompt_dir / "responses"
    out_dir = Path(args.out_dir) if args.out_dir else prompt_dir.parent / "xai_scoring"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_by_id = _load_prompts_by_id(prompt_dir / "prompts.jsonl")
    response_files = sorted(responses_dir.glob("*.jsonl"))
    if not response_files:
        raise SystemExit(f"no response files found under {responses_dir}")

    fable5_path = args.fable5_scores if args.fable5_scores else out_dir / "fable5_scores.json"
    fable5_by_model = _load_fable5_scores(fable5_path)

    by_model: dict[str, list[dict[str, Any]]] = {}
    for path in response_files:
        for record in _read_jsonl(path):
            by_model.setdefault(str(record.get("model")), []).append(record)

    for model, records in sorted(by_model.items()):
        records = [r for r in records if r.get("prompt_kind") == args.prompt_kind]
        records.sort(key=lambda r: (str(r.get("family")), str(r.get("graph_id"))))
        out_path = out_dir / f"worksheet_{_safe_slug(model)}.md"
        existing_scores = _load_existing_understood_scores(out_path)
        fable5_scores = fable5_by_model.get(model, {})
        text = _render_worksheet(model, records, prompts_by_id, existing_scores, fable5_scores)
        out_path.write_text(text, encoding="utf-8")
        n_ok = sum(1 for r in records if not r.get("error"))
        n_err = len(records) - n_ok
        n_scored = sum(1 for v in existing_scores.values() if v)
        n_fable5 = sum(1 for r in records if str(r.get("graph_id")) in fable5_scores)
        print(
            f"{out_path}  ({n_ok} scorable, {n_err} error records, {n_scored} already scored, preserved, "
            f"{n_fable5} with fable5 estimate)"
        )


def _render_worksheet(
        model: str,
        records: list[dict[str, Any]],
        prompts_by_id: dict[str, dict[str, Any]],
        existing_scores: dict[tuple[str, str], str],
        fable5_scores: dict[str, dict[str, Any]],
) -> str:
    lines = [
        f"# Scoring worksheet — `{model}`",
        "",
        "> `understood` (1-10): did this explanation actually make sense? Seeded",
        "> from Fable 5's independent (xhigh, blind) scoring pass -- these are",
        "> Fable 5's actual scores, not Claude's estimates. Change any of them",
        "> yourself if you disagree; re-running the render script preserves",
        "> whatever's already in this column (including your own edits) and",
        "> only fills in graphs that are still blank.",
        "> Everything else (`numeric_faithfulness`, `scope_honesty`, `dt_branch`,",
        "> `sections_complete`, `duration`) is computed mechanically -- no LLM",
        "> judge, plain string/regex checks against the prompt's ground-truth",
        "> metadata, or values already recorded by the digest script.",
        "> `sections_complete` is a count (e.g. 3/5), not binary.",
        "",
        "> Reason shorthand glossary (from both Claude's and Fable 5's notes):",
        "> - `formulaic`: accurate/complete but just restates condition->value,",
        "> never explains *why* (e.g. never ties disjoint routes to Nash mixing).",
        "> - `drifted (structure)`: ignored the 5 requested section headers.",
        "> - `wrong leaf` / `NOT CITED`: same as a `dt_branch: NOT CITED` flag.",
        "> - `X% faithful, hallucinated ...`: same as `numeric_faithfulness`.",
        "> - `rounding, not verbatim`: a mild faithfulness ding (0.26 for 0.258),",
        "> not fabrication.",
        "> - `conflated` / `reused as`: two different quantities given the same",
        "> number -- a real contradiction, not just imprecision.",
        "> - `backwards` / `X called Y`: a directional contradiction -- e.g. a",
        "> low retention value narrated as \"high retention\" or \"successful",
        "> disruption\". Fable 5 caught more of these than Claude's own pass did.",
        "> - `real overclaim`: the `scope_honesty` flag is genuine, not a",
        "> regex false positive.",
        "> - `explicit diagnostic separation/framing/note/rows`: positive --",
        "> clearly labeled selected (prob>0) vs diagnostic-only (prob=0) rows,",
        "> the one thing deliberately left un-mechanized (too unreliable to",
        "> string-match), rewarded by hand instead.",
        "> - `ties X to Y` / `explains X`: positive -- actually reasons about",
        "> *why* a feature produces an outcome, not just formulaic restating.",
        "",
        "---",
        "",
    ]
    for record in records:
        graph_id = str(record.get("graph_id", "unknown"))
        family = record.get("family", "")
        prompt_kind = str(record.get("prompt_kind", ""))
        prompt_id = record.get("prompt_id", "")
        lines.append(f"## {graph_id}  ·  {family}  ·  {prompt_kind}")
        lines.append("")
        if record.get("error"):
            lines.append(f"**ERROR** — `{record['error'].get('type')}`: {record['error'].get('message')}")
            lines.append("")
            lines.append("(not scorable — rerun the digest script with `--resume` to retry)")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        response_text = str(record.get("response_text", "")).strip()
        lines.append(response_text if response_text else "_(empty response)_")
        lines.append("")

        prompt_record = prompts_by_id.get(str(prompt_id))
        metadata = (prompt_record or {}).get("scoring_metadata") or {}

        if prompt_record is not None:
            faith = score_faithfulness(response_text, str(prompt_record.get("prompt", "")))
            faith_display = f"{faith.score:.0%}"
            if faith.hallucinated:
                faith_display += f" (hallucinated: {', '.join(faith.hallucinated[:5])})"
        else:
            faith_display = "n/a (prompt not found)"

        scope_result = check_scope_overclaim(
            response_text,
            forbidden_claim_examples=tuple(metadata.get("forbidden_claim_examples") or ()),
        )
        scope_display = "clean" if scope_result.clean else f"FLAG: {scope_result.violations[0].rule}"

        leaf_result = check_dt_leaf_citation(response_text, metadata.get("graph_tree_active_leaf"))
        if leaf_result.expected_value is None:
            leaf_display = "n/a"
        elif leaf_result.cited:
            leaf_display = f"cited ({leaf_result.expected_value})"
        else:
            leaf_display = f"NOT CITED (expected {leaf_result.expected_value})"

        sections_result = check_section_completeness(response_text)
        sections_display = f"{len(sections_result.present)}/{sections_result.total}"
        if sections_result.missing:
            sections_display += f" (missing: {', '.join(sections_result.missing)})"

        duration_ms = record.get("duration_ms")
        duration_display = f"{duration_ms / 1000:.1f}s" if isinstance(duration_ms, (int, float)) else "n/a"

        # Fable 5's score is the actual score wherever it exists -- it only
        # ever fills a currently-blank cell, never overwrites something
        # already sitting in the column (a prior Fable 5 fill from an
        # earlier render, or a real human edit).
        existing_value = existing_scores.get((graph_id, prompt_kind), "_")
        fable5_entry = fable5_scores.get(graph_id)
        if existing_value == "_" and fable5_entry:
            understood_value = f"{fable5_entry['score']} ({fable5_entry['reason']})"
        else:
            understood_value = existing_value

        lines.append(
            "| numeric_faithfulness | scope_honesty | dt_branch | sections_complete "
            "| duration | **understood (1-10)** |"
        )
        lines.append("|---|---|---|---|---|---|")
        lines.append(
            f"| {faith_display} | {scope_display} | {leaf_display} | {sections_display} "
            f"| {duration_display} | {understood_value} |"
        )
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _load_existing_understood_scores(path: Path) -> dict[tuple[str, str], str]:
    """Parse a previously-rendered worksheet for already-filled understood values."""
    if not path.exists():
        return {}
    scores: dict[tuple[str, str], str] = {}
    current_key: tuple[str, str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        header_match = _UNDERSTOOD_HEADER_RE.match(line)
        if header_match:
            current_key = (header_match.group("graph_id"), header_match.group("prompt_kind"))
            continue
        if current_key is None or not line.startswith("|"):
            continue
        row_match = _UNDERSTOOD_ROW_RE.match(line)
        if not row_match:
            continue
        value = row_match.group("value").strip()
        if value and value != "_" and not value.startswith("---") and "understood" not in value.lower():
            scores[current_key] = value
            current_key = None
    return scores


def _load_fable5_scores(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Load {model: {graph_id: {"score", "reason"}}} from fable5_scores.json, if present."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for model, entry in payload.items():
        result[model] = {s["graph_id"]: s for s in entry.get("scores", [])}
    return result


def _load_prompts_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return {str(record["id"]): record for record in _read_jsonl(path)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _safe_slug(value: str) -> str:
    allowed = []
    for char in value:
        if char.isalnum() or char in ("-", "_", "."):
            allowed.append(char)
        else:
            allowed.append("_")
    return "".join(allowed).strip("_") or "model"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt_dir",
        type=Path,
        help="Directory containing prompts.jsonl and responses/, e.g. runs/exp3_dynamic_500k/llm_prompts.",
    )
    parser.add_argument("--out-dir", type=Path, help="Defaults to <prompt_dir>/../xai_scoring.")
    parser.add_argument(
        "--fable5-scores",
        type=Path,
        help="JSON file of {model: {scores: [{graph_id, score, reason}]}}. "
        "Defaults to <out_dir>/fable5_scores.json if it exists.",
    )
    parser.add_argument(
        "--prompt-kind",
        choices=("oracle_minimax", "exp3_learned"),
        default="exp3_learned",
        help="Only render this prompt kind. Default: exp3_learned (the firm-commitment set).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
