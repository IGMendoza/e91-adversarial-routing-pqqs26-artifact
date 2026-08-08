# Histogram generator package

This directory is the maintained home of the device-aware E91 histogram
generator. The historical script remains at
`scripts/generate_e91_game_turn_histograms.py` as a compatible command-line
entry point and attribute proxy.

## Components

- `models.py`: model, protocol, branch, and transcript value objects.
- `device_models.py`: typed process normalization and memory/swap channels.
- `device_profiles.py`: model context, wavelength compatibility, and detectors.
- `device_effects.py`: effect coverage and fail-closed path limitations.
- `front_matter.py`: physical registry loading and parameter inventory mapping.
- `devices.py`: thin compatibility surface for those device components.
- `topology.py`: linear-route construction, contracts, and Eve move expansion.
- `timing.py`: propagation, waiting, storage-age, and operational timing.
- `quantum.py`: density-matrix channels, measurements, and swap algebra.
- `fock.py`: Fock-space instruments and detector response.
- `routing.py`: elementary-link and repeater-route branch composition.
- `generation.py`: route/case generation and frozen-kernel evaluation.
- `compiled.py`: length-parametric compiled kernels and verification.
- `output.py`: schemas, compact arrays, and artifact output.
- `cli.py`: command-line surface.
- `performance.py`: opt-in wall/CPU measurement and repeatable benchmarks.
- `engine.py`: sub-100-line compatibility aggregator for historical imports.

The literature/controller layer used by the histogram path lives outside this
package in two focused modules:

- `sequence_game/physical/repeater_service.py`: validated asynchronous service
  rates and source-specific Bell-error equations.
- `sequence_game/physical/repeater_reservation.py`: fixed hardware inventory to
  rolling-pool availability/age inputs; it never target-provisions memories.

The implementation lives in these component modules. The device compatibility
surface follows the same state-forwarding pattern as `engine.py`. Fock and compiled caches
are owned by `fock.py` and `compiled.py` respectively; `engine.py` dynamically
forwards historical attributes to their owning module, so it neither duplicates
state nor hides another monolith. This preserves cache identity and keeps the
compiled NumPy path unchanged. New code should import the smallest relevant
component; compatibility consumers may continue importing the script.

## Timing

There are two different timing concepts:

1. Physical timing is part of the modeled result: fiber propagation delay,
   detector windows/cooldown, waiting budgets, memory storage ages, and
   decoherence are exposed from `histograms.timing` and continue to enter the
   histogram engine. With a repeater service profile, metadata also reports the
   asynchronous end-to-end production time/rate separately from random-time
   rolling-pool availability.
2. Operational timing measures software performance. CLI manifests record
   per-case and whole-run wall/CPU durations. `measure_call` and
   `benchmark_call` are opt-in, so ordinary and compiled evaluations do not pay
   per-operation instrumentation overhead.

Run the generator through either entry point:

```bash
.venv/bin/python scripts/generate_e91_game_turn_histograms.py --help
.venv/bin/python -m histograms.cli --help
.venv/bin/python scripts/validate_alice_repeater_v1.py
```

The service-profile path is opt-in. Historical configurations retain their
cache identity and behavior; an active profile is hashed into the model
identity, carries an explicit claim tier, and applies only its declared rate,
memory, elementary-link, and local-swap mappings. No service profile implies no
new physical equations or parameter overrides.
