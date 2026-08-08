"""Transcript schemas, artifact serialization, and CLI parsing helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .models import *
from .models import _json_ready
from .topology import *
def transcript_schema() -> dict[str, Any]:
    return {
        "fields": [
            "delivery_status",
            "failure_stage",
            "alice_setting_label",
            "bob_setting_label",
            "alice_outcome",
            "bob_outcome",
            "sifting_class",
            "eve_observation_record",
            "effective_pauli_frame",
            "public_bsm_record",
            "detector_click_record",
            "timing_record",
        ],
        "delivery_status": DELIVERY_STATUS,
        "failure_stage": FAILURE_STAGE,
        "sifting_class": SIFTING_CLASS,
        "effective_pauli_frame": {
            label: {"id": FRAME_ID[label], "bits": list(FRAME_BITS[label])}
            for label in FRAME_LABELS
        },
        "observability": {
            "delivery_status": "public game-turn status",
            "failure_stage": "latent diagnostic decomposition emitted for validation; not treated as a public E91 transcript field",
            "alice_setting_label": "Alice public setting label after basis announcement",
            "bob_setting_label": "Bob public setting label after basis announcement",
            "alice_outcome": "Alice reported raw outcome in the generated transcript bin after endpoint memory-readout assignment model",
            "bob_outcome": "Bob reported raw outcome in the generated transcript bin after endpoint memory-readout assignment model",
            "sifting_class": "derived public class from settings",
            "eve_observation_record": "Eve private retained observation record, none for non-observation moves",
            "effective_pauli_frame": "public repeater-swap frame coordinate relevant to endpoint interpretation",
            "detector_click_record": (
                "production coordinate is 'not_emitted' for delivered rows because raw optical detector records "
                "are marginalized after conditioning remote memory states; raw records are emitted only by "
                "--emit-detector-click-audit"
            ),
            "dark_count_cause": "latent, affects elementary heralding probabilities but is not a bin label",
            "timing_record": "marginalized unless a future protocol model declares it public",
            "public_bsm_record": "optional audit coordinate; default coordinate is effective_pauli_frame",
        },
    }


def case_id(route: dict[str, Any], attack: HistogramAttack) -> str:
    safe = attack.move_id.replace(":", "_").replace("/", "_")
    return f"h{route['hops']}_{safe}"


def write_case(out_dir: Path, case: dict[str, Any]) -> tuple[str, int]:
    metadata = case["metadata"]
    rows = case["rows"]
    cid = metadata["case_id"]
    with open(out_dir / f"{cid}.json", "w", encoding="utf-8") as handle:
        json.dump(_json_ready(metadata), handle, indent=2, sort_keys=True)
        handle.write("\n")
    arrays = rows_to_npz_arrays(rows)
    np.savez_compressed(out_dir / f"{cid}.npz", **arrays)
    return cid, len(rows)


def rows_to_npz_arrays(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    alice_labels = sorted({row["alice_setting_label"] for row in rows})
    bob_labels = sorted({row["bob_setting_label"] for row in rows})
    eve_labels = sorted({row["eve_observation_record"] for row in rows})
    bsm_labels = sorted({row["public_bsm_record"] for row in rows})
    click_labels = sorted({row["detector_click_record"] for row in rows})
    timing_labels = sorted({row["timing_record"] for row in rows})
    failure_labels = sorted({row["failure_stage"] for row in rows})
    maps = {
        "alice": {label: idx for idx, label in enumerate(alice_labels)},
        "bob": {label: idx for idx, label in enumerate(bob_labels)},
        "eve": {label: idx for idx, label in enumerate(eve_labels)},
        "bsm": {label: idx for idx, label in enumerate(bsm_labels)},
        "click": {label: idx for idx, label in enumerate(click_labels)},
        "timing": {label: idx for idx, label in enumerate(timing_labels)},
        "failure": {label: idx for idx, label in enumerate(failure_labels)},
    }
    return {
        "delivery_status_id": np.asarray([DELIVERY_STATUS[row["delivery_status"]] for row in rows], dtype=np.int16),
        "failure_stage_id": np.asarray([maps["failure"][row["failure_stage"]] for row in rows], dtype=np.int16),
        "alice_setting_id": np.asarray([maps["alice"][row["alice_setting_label"]] for row in rows], dtype=np.int16),
        "bob_setting_id": np.asarray([maps["bob"][row["bob_setting_label"]] for row in rows], dtype=np.int16),
        "alice_outcome": np.asarray([row["alice_outcome"] for row in rows], dtype=np.int8),
        "bob_outcome": np.asarray([row["bob_outcome"] for row in rows], dtype=np.int8),
        "sifting_class_id": np.asarray([SIFTING_CLASS[row["sifting_class"]] for row in rows], dtype=np.int16),
        "eve_observation_id": np.asarray([maps["eve"][row["eve_observation_record"]] for row in rows], dtype=np.int32),
        "effective_frame_id": np.asarray([FRAME_ID[row["effective_pauli_frame"]] for row in rows], dtype=np.int16),
        "public_bsm_record_id": np.asarray([maps["bsm"][row["public_bsm_record"]] for row in rows], dtype=np.int32),
        "detector_click_record_id": np.asarray([maps["click"][row["detector_click_record"]] for row in rows], dtype=np.int16),
        "timing_record_id": np.asarray([maps["timing"][row["timing_record"]] for row in rows], dtype=np.int16),
        "probability": np.asarray([row["probability"] for row in rows], dtype=np.float64),
        "expected_count": np.asarray([row["expected_count"] for row in rows], dtype=np.float64),
        "alice_setting_labels": np.asarray(alice_labels),
        "bob_setting_labels": np.asarray(bob_labels),
        "eve_observation_labels": np.asarray(eve_labels),
        "public_bsm_record_labels": np.asarray(bsm_labels),
        "detector_click_record_labels": np.asarray(click_labels),
        "timing_record_labels": np.asarray(timing_labels),
        "failure_stage_labels": np.asarray(failure_labels),
    }


def parse_lengths(hops: int, raw: str | None) -> list[float]:
    if raw is None or raw == "":
        return [DEFAULT_LENGTH_M] * hops
    values = [float(part) for part in raw.split(",") if part.strip()]
    if len(values) == 1:
        return values * hops
    if len(values) != hops:
        raise SystemExit("--fiber-lengths-m must provide one length or one length per hop")
    return values


def parse_optional_float_sequence(raw: str | Sequence[float] | None) -> tuple[float, ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw.strip() == "":
            return None
        return tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    return tuple(float(value) for value in raw)


def parse_optional_nested_float_sequence(raw: str | Sequence[Sequence[float]] | None) -> tuple[tuple[float, ...], ...] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        if raw.strip() == "":
            return None
        return tuple(
            tuple(float(part.strip()) for part in row.split(",") if part.strip())
            for row in raw.split(";") if row.strip()
        )
    return tuple(tuple(float(value) for value in row) for row in raw)


def parse_symbols(hops: int, raw: str | None) -> list[str]:
    if raw:
        values = [part.strip() for part in raw.split(",") if part.strip()]
        if len(values) != hops:
            raise SystemExit("--fiber-length-symbols must provide one symbol per hop")
        return values
    return default_fiber_length_symbols(hops)


def hop_range(args: argparse.Namespace) -> list[int]:
    fixed_hops = fixed_hop_count(args)
    if fixed_hops is not None:
        return [fixed_hops]
    return list(range(args.min_hops, args.max_hops + 1))


def fixed_hop_count(args: argparse.Namespace) -> int | None:
    repeaters = getattr(args, "repeaters", None)
    hops = getattr(args, "hops", None)
    if repeaters is not None:
        if repeaters < 0:
            raise SystemExit("--repeaters must be >= 0")
        hops_from_repeaters = int(repeaters) + 1
        if hops is not None and int(hops) != hops_from_repeaters:
            raise SystemExit("--repeaters and --hops are inconsistent: hop_count must equal repeaters + 1")
        return hops_from_repeaters
    if hops is not None:
        if hops < 1:
            raise SystemExit("--hops must be >= 1")
        return int(hops)
    return None


def write_numeric_evaluator(out_dir: Path) -> None:
    generator_path = str(REPO_ROOT / "scripts" / "generate_e91_game_turn_histograms.py")
    text = f'''"""In-process evaluator for generated E91 game-turn histograms.

Each emitted case JSON records the frozen theta family H_{{theta,h,move}}: the
device front matter, model context, hop count, Eve move, and protocol. This
wrapper reloads that family and evaluates it at new fiber-length vectors
without re-running the generator CLI, then optionally samples multinomial
transcript counts. Evaluation is verified against the recorded
frozen_theta_hash, so a result is guaranteed to belong to the emitted family.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

GENERATOR_PATH = Path({generator_path!r})
OUTPUT_DIR = Path(__file__).resolve().parent
FRONT_MATTER_NAME = "histogram_device_front_matter.json"

_GENERATOR = None
_FRONT_CACHE: dict[str, object] = {{}}


def _generator():
    global _GENERATOR
    if _GENERATOR is None:
        spec = importlib.util.spec_from_file_location("e91_histogram_generator", GENERATOR_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _GENERATOR = module
    return _GENERATOR


def _front(out_dir: Path):
    key = str(out_dir)
    if key not in _FRONT_CACHE:
        _FRONT_CACHE[key] = _generator().load_or_build_front_matter(
            out_dir,
            device_front_matter=out_dir / FRONT_MATTER_NAME,
        )
    return _FRONT_CACHE[key]


def load_case_metadata(case_id: str, *, out_dir=None) -> dict:
    out = Path(out_dir) if out_dir is not None else OUTPUT_DIR
    return json.loads((out / f"{{case_id}}.json").read_text(encoding="utf-8"))


def evaluate_histogram(case_id: str, fiber_lengths_m, *, out_dir=None, strict_theta: bool = True) -> dict:
    """Evaluate the case's frozen kernel at fiber_lengths_m; returns {{metadata, rows}}."""
    out = Path(out_dir) if out_dir is not None else OUTPUT_DIR
    generator = _generator()
    distribution = generator.evaluate_case_histogram(
        _front(out),
        load_case_metadata(case_id, out_dir=out),
        fiber_lengths_m,
        strict_theta=strict_theta,
    )
    return distribution.to_case()


def sample_histogram(case_id: str, fiber_lengths_m, shots: int, seed: int, *, out_dir=None) -> dict:
    """Evaluate at fiber_lengths_m and draw a multinomial transcript-count sample."""
    out = Path(out_dir) if out_dir is not None else OUTPUT_DIR
    generator = _generator()
    distribution = generator.evaluate_case_histogram(
        _front(out),
        load_case_metadata(case_id, out_dir=out),
        fiber_lengths_m,
    )
    return generator.sample_transcript_counts(distribution, shots, seed)
'''
    (out_dir / "histogram_numeric_evaluator.py").write_text(text, encoding="utf-8")
