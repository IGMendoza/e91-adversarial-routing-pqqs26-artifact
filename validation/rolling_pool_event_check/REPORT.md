# Rolling-controller event validation

Date: 2026-07-14

## Question

Does the analytic fixed-slot rolling-pool model consumed by the histogram
backend agree with an independent event-heap realization for two long routes
whose edges are each 50 km?

## Matched process

Each purified output slot alternates independently between:

1. an empty duration drawn from the Cui composite profile's declared
   exponential renewal law, using the profile adapter's per-lane mean
   completion time; and
2. a live duration equal to the minimum of the fixed 0.1 s staleness cutoff
   and the exponential detected-decay time. The two stored memories give the
   effective pair decay lifetime used by the analytic model.

Four fixed purified-output slots are retained per adjacent edge side. At an
independent Poisson snapshot, the controller selects the youngest live pair on
each edge. No resource optimizer changes the slot inventory.

This is exactly the stationary pool process whose availability and selected
ages enter `rolling_alive_pair`. It does not add a continuous-demand queue or
consume pairs: the current profile explicitly declares
`continuous_demand_queue=false`.

## Production run

Command:

```text
.venv/bin/python scripts/validate_rolling_controller_event.py \
  --samples 100000 \
  --out runs/rolling_controller_event_validation.json
```

Both predeclared cases passed:

| Route | Analytic availability | Event estimate | Absolute error | Batch SE |
|---|---:|---:|---:|---:|
| 12 x 50 km | 0.212700 | 0.211530 | 0.001170 | 0.001917 |
| 16 x 50 km | 0.126967 | 0.127370 | 0.000403 | 0.001328 |

For the homogeneous 50 km edge, the analytic selected residence-age mean is
0.0360661 s. The first-edge event estimates were 0.0360537 s and 0.0360140 s
for the two long-route runs. The fixed pre-pool storage age is another 0.000760 s
and is separately retained by the histogram adapter.

## Resource preflight

The cases execute sequentially. The larger case is the peak:

- 64 live-pool slots and at most 64 scheduled heap entries;
- about 3.21 million processed events;
- 27.7 MiB estimated peak array/heap memory;
- zero route-attack or histogram cache entries.

The executable refuses a run only when its explicit user-visible RAM or event
preflight limit is exceeded. Those limits are compute safeguards, not route or
policy priors.

## Claim boundary

This result validates the stationary availability and selected-age inputs of
the rolling controller at production hop counts. It does not independently
validate optical state generation, purification gates, local Bell swaps, or
service behavior under a saturated continuous-demand queue. Those state
channels remain tested in the compiled/reference histogram parity suite; an
under-load queue would require a separately declared controller model.
