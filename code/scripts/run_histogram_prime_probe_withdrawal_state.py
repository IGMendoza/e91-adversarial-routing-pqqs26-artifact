#!/usr/bin/env python3
"""Run the five-topology withdrawal-time learner-state sensitivity study."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sequence_game.experiments.histogram_colearning.prime_probe_corpus import (  # noqa: E402
    DEFAULT_CORPUS_RUN_DIR,
)
from sequence_game.experiments.histogram_colearning.prime_probe_recovery_corpus import (  # noqa: E402
    DEFAULT_SOURCE_PRIME_PROBE_RUN_DIR,
    DEFAULT_WITHDRAWAL_STATE_RUN_DIR,
    build_withdrawal_state_manifest,
    run_withdrawal_state_corpus_study,
    withdrawal_state_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-run-dir",
        type=Path,
        default=DEFAULT_CORPUS_RUN_DIR,
    )
    parser.add_argument(
        "--source-prime-probe-run-dir",
        type=Path,
        default=DEFAULT_SOURCE_PRIME_PROBE_RUN_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_WITHDRAWAL_STATE_RUN_DIR,
    )
    parser.add_argument("--graph-workers", type=int, default=5)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Write the frozen manifest and preflight, then stop.",
    )
    args = parser.parse_args()

    corpus_manifest = _read_json(
        args.corpus_run_dir / "run_manifest.json"
    )
    source_manifest = _read_json(
        args.source_prime_probe_run_dir / "run_manifest.json"
    )
    manifest_path = args.output_dir / "run_manifest.json"
    if manifest_path.exists():
        study_manifest = _read_json(manifest_path)
    else:
        study_manifest = build_withdrawal_state_manifest(source_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(manifest_path, study_manifest)
    preflight = withdrawal_state_preflight(
        study_manifest,
        source_run_dir=args.source_prime_probe_run_dir,
    )
    _write_json(args.output_dir / "preflight.json", preflight)
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if args.preflight_only:
        return

    summary = run_withdrawal_state_corpus_study(
        corpus_manifest,
        study_manifest,
        output_dir=args.output_dir,
        graph_workers=args.graph_workers,
        progress_callback=_progress,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _progress(message: str) -> None:
    print(
        f"[{datetime.now().isoformat(timespec='seconds')}] {message}",
        flush=True,
    )


if __name__ == "__main__":
    main()
