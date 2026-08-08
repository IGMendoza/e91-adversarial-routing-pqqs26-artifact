#!/usr/bin/env python3
"""Write tables, report, and figures for a modern path benchmark artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sequence_game.experiments.histogram_colearning.modern_path_benchmark_analysis import (  # noqa: E402
    write_modern_path_benchmark_analysis,
)


def main(argv: list[str] | None = None) -> list[Path]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    paths = write_modern_path_benchmark_analysis(payload, args.out_dir)
    for path in paths:
        print(path)
    return paths


if __name__ == "__main__":
    main()
