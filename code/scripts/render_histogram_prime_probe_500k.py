#!/usr/bin/env python3
"""Render publication artifacts for the physical Prime-and-Probe study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.prime_probe_artifacts import (  # noqa: E402
    write_prime_probe_artifacts,
)
from sequence_game.experiments.histogram_colearning.prime_probe_corpus import (  # noqa: E402
    DEFAULT_PRIME_PROBE_RUN_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_PRIME_PROBE_RUN_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "artifacts" / "histogram_prime_probe_500k",
    )
    args = parser.parse_args()
    manifest = write_prime_probe_artifacts(args.run_dir, args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
