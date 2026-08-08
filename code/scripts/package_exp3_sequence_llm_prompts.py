#!/usr/bin/env python3
"""Package per-graph LLM prompts for an Exp3 SeQUeNCe result run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sequence_game.experiments.exp3_sequence.llm_prompt_packaging import (  # noqa: E402
    PROMPT_KINDS,
    package_prompts,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/exp3_sequence_paper"),
        help="Completed Exp3 run directory containing corpus/dt/oracle summaries.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to RUN_DIR/llm_prompts.",
    )
    parser.add_argument(
        "--top-actions",
        type=int,
        default=5,
        help="Number of per-graph Eve action rows to include.",
    )
    parser.add_argument(
        "--top-routes",
        type=int,
        default=5,
        help="Number of per-graph Alice route rows to include.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional graph limit for smoke tests.",
    )
    parser.add_argument(
        "--prompt-kind",
        choices=sorted(PROMPT_KINDS),
        default="oracle_minimax",
        help="Which prompt family to emit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = package_prompts(
        args.run_dir,
        out_dir=args.out_dir,
        top_actions=args.top_actions,
        top_routes=args.top_routes,
        limit=args.limit,
        prompt_kind=args.prompt_kind,
    )
    print(f"wrote {manifest['prompt_count']} prompts")
    print(f"jsonl: {manifest['jsonl_path']}")
    print(f"manifest: {Path(manifest['out_dir']) / 'manifest.json'}")
    print(f"text prompts: {manifest['prompt_dir']}")


if __name__ == "__main__":
    main()
