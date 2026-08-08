#!/usr/bin/env python3
"""Summarize fixed-vs-dynamic DT refit benchmark artifacts with optional LLM calls."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from sequence_game.xai.llm_client import query_model, query_model_streaming


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _read_jsonl_count(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _timing_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import csv

    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                values.append(float(row["last_5_steps_s"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not values:
        return {"tick_count": 0}
    return {
        "tick_count": len(values),
        "last_5_steps_s_mean": sum(values) / len(values),
        "last_5_steps_s_min": min(values),
        "last_5_steps_s_max": max(values),
        "last_5_steps_s_p50": statistics.median(values),
    }


def _run_bundle(run_dir: Path, label: str) -> dict[str, Any]:
    summary = _read_json(run_dir / "summary.json")
    refit_summary = _read_json(run_dir / "dt_refit_summary.json")
    refit_events_count = _read_jsonl_count(run_dir / "dt_refit_events.jsonl")
    observer_ticks_count = _read_jsonl_count(run_dir / "observer_ticks.jsonl")
    snapshot_count = len(list((run_dir / "dt_snapshots").glob("*_dt.json")))

    bundle = {
        "label": label,
        "path": str(run_dir),
        "summary_fields": {
            key: summary.get(key)
            for key in (
                "benchmark",
                "graph_id",
                "assignment_id",
                "alice",
                "bob",
                "shortest_hops",
                "route_selection_policy",
                "route_generation_reason",
                "alice_route_actions",
                "eve_actions",
                "num_steps",
                "seed",
                "dt_refit_mode",
                "dt_refit_count",
                "fixed_50_dt_refit_count_equivalent",
                "estimated_dt_refits_avoided",
                "route_enumeration_wall_s",
                "training_wall_s",
                "pipeline_wall_s_excluding_artifact_write",
                "collision_rate",
                "route_entropy",
                "route_switches",
                "no_convergence_claim",
            )
            if key in summary
        },
        "dt_refit_summary": refit_summary,
        "refit_events_count": refit_events_count,
        "observer_ticks_count": observer_ticks_count,
        "dt_snapshot_json_count": snapshot_count,
        "time_per_50_summary": _timing_summary(run_dir / "time_per_50_ticks.csv"),
        "eve_dt_rules": _read_text(run_dir / "eve_attack_choice_dt_rules.txt"),
        "alice_dt_rules": _read_text(run_dir / "alice_route_choice_dt_rules.txt"),
    }
    if not refit_summary:
        bundle["legacy_artifact_note"] = (
            "No dt_refit_summary.json or dt_refit_events.jsonl was found. "
            "This artifact has final DT outputs and timing ticks, but not observer-era "
            "fixed-interval DT refit event telemetry."
        )
    return bundle


def _build_prompt(fixed: dict[str, Any], dynamic: dict[str, Any]) -> str:
    payload = {
        "fixed_or_legacy_run": fixed,
        "dynamic_run": dynamic,
    }
    return f"""You are summarizing a SNAP AS-733 single-graph DQN/DQN benchmark comparison.

Scientific scope:
- This is a control-game simulation benchmark.
- DT models are post-hoc explanation snapshots of Q-derived behavior.
- Do not claim Q-learning convergence, optimality, physical security, or real ISP behavior.
- If a field is missing, say it is missing; do not infer it.
- Distinguish DQN/MLP learner runtime from DT surrogate refit/runtime.

Task:
Write a concise technical comparison of the two runs:
1. Confirm whether the graph, assignment, route cutoff, action counts, step count, and seed are comparable.
2. Compare DT refit scheduling telemetry where available.
3. Compare total runtime and the last-5-step timing summaries.
4. Compare the final Eve DT rules and Alice route-choice DT rules.
5. State whether the dynamic scheduler reduced DT refits, and whether the final explanations appear materially different.
6. Explicitly state any limitation caused by missing legacy/fixed refit telemetry.

Artifact bundle:
```json
{json.dumps(payload, indent=2, default=str)}
```
"""


def _model_slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "-").replace(" ", "_")


def _model_timeout(model: str, base: int = 300) -> int:
    name = model.lower()
    if any(part in name for part in ("70b", "72b")):
        return max(base, 900)
    if any(part in name for part in ("32b", "31b", "30b", "27b", "26b")):
        return max(base, 600)
    return base


def _query_llm(
    model: str,
    prompt: str,
    *,
    stream: bool,
    timeout: int,
) -> tuple[str, dict[str, Any]]:
    if stream:
        chunks: list[str] = []
        for chunk in query_model_streaming(
            model,
            prompt,
            temperature=0.05,
            timeout=timeout,
        ):
            chunks.append(chunk)
        return "".join(chunks).strip(), {
            "streaming": True,
            "prompt_tokens": None,
            "response_tokens": None,
        }

    response = query_model(
        model,
        prompt,
        temperature=0.05,
        timeout=timeout,
    )
    return response.response.strip(), {
        "streaming": False,
        "prompt_tokens": response.prompt_tokens,
        "response_tokens": response.response_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-run", required=True, help="Fixed-interval or legacy DT artifact run directory.")
    parser.add_argument("--dynamic-run", required=True, help="Dynamic DT refit run directory.")
    parser.add_argument("--out-dir", required=True, help="Directory for prompt and LLM summaries.")
    parser.add_argument("--llm", default="phi4:14b", help="Primary Ollama model.")
    parser.add_argument(
        "--compare-llms",
        default=None,
        help="Comma-separated Ollama models. If omitted, only --llm is queried.",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Ollama streaming generation to avoid proxy idle timeouts.",
    )
    parser.add_argument(
        "--timeout-base",
        type=int,
        default=300,
        help="Base urllib socket timeout in seconds before model-size scaling.",
    )
    parser.add_argument("--no-llm", action="store_true", help="Only write prompt and structured input.")
    args = parser.parse_args()

    fixed_run = Path(args.fixed_run)
    dynamic_run = Path(args.dynamic_run)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_bundle = _run_bundle(fixed_run, "fixed_or_legacy")
    dynamic_bundle = _run_bundle(dynamic_run, "dynamic")
    prompt = _build_prompt(fixed_bundle, dynamic_bundle)

    (out_dir / "dt_refit_comparison_input.json").write_text(
        json.dumps(
            {
                "fixed_or_legacy_run": fixed_bundle,
                "dynamic_run": dynamic_bundle,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    (out_dir / "dt_refit_comparison_prompt.txt").write_text(prompt, encoding="utf-8")

    if args.no_llm:
        print(json.dumps({"out_dir": str(out_dir), "prompt": "dt_refit_comparison_prompt.txt"}, indent=2))
        return

    models = [args.llm]
    if args.compare_llms:
        models = [model.strip() for model in args.compare_llms.split(",") if model.strip()]
        if args.llm not in models:
            models.insert(0, args.llm)

    results: dict[str, Any] = {}
    for model in models:
        timeout = _model_timeout(model, base=args.timeout_base)
        stream_label = "streaming" if args.stream else "non-streaming"
        print(
            f"LLM comparison summary via {model} ({stream_label}, timeout={timeout}s) ...",
            end="",
            flush=True,
        )
        started = time.perf_counter()
        try:
            text, llm_meta = _query_llm(
                model,
                prompt,
                stream=args.stream,
                timeout=timeout,
            )
            elapsed = time.perf_counter() - started
            results[model] = {
                "elapsed_s": elapsed,
                "timeout_s": timeout,
                "output": text,
                **llm_meta,
            }
            print(f" {elapsed:.1f}s", flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            text = f"[LLM unavailable: {exc}]"
            results[model] = {"elapsed_s": elapsed, "error": str(exc), "output": text}
            print(f" ERROR: {exc}", flush=True)

        md = (
            f"# DT Refit Comparison Summary\n\n"
            f"**Model:** `{model}`\n\n"
            f"{text}\n\n"
            f"---\n"
            f"_Prompt: `dt_refit_comparison_prompt.txt`; structured input: "
            f"`dt_refit_comparison_input.json`._\n"
        )
        (out_dir / f"dt_refit_comparison_{_model_slug(model)}.md").write_text(
            md,
            encoding="utf-8",
        )

    (out_dir / "dt_refit_comparison_llm_results.json").write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    print(json.dumps({"out_dir": str(out_dir), "models": models}, indent=2))


if __name__ == "__main__":
    main()
