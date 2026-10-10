# Validation and runtime records

This folder holds the saved outputs and reproduction commands behind the
technical-validation appendix and the runtime statement in the corpus
evaluation. Commands run from `code/` with the dependencies in
`code/requirements.txt`.

| Paper location | Numbers | Saved output | Status |
|---|---|---|---|
| Appendix, "Comparison with event simulations", rolling-pool table | 12 hops: 0.212700 / 0.211530 / 0.001170 / 0.001917; 16 hops: 0.126967 / 0.127370 / 0.000403 / 0.001328; selected first-edge ages 36.054 / 36.014 ms vs. 36.066 ms analytic | `rolling_pool_event_check/rolling_controller_event_validation.json` | Reproduced from this repository's code |
| Appendix, "Comparison with event simulations", SeQUeNCe comparison | 41 of 80 comparisons resolved, all agreeing; strongest-denial sets agree on 8/8 routes; cell RMSE 0.4125 / 0.5990 / 0.0876 | to be added under `sequence_comparison/` | Report and command included; saved output pending |
| Corpus evaluation, runtime accounting | 32.4 h summed elapsed time; 81.6% learning, 17.4% audit; medians 846 s and 16.5 s; compilation 6.7 h / 4.8 h vs. 22.3 h sampling | `runtime_accounting/runtime_summary.txt` | Recomputed from the bundled logs |

## Rolling-pool event check

```bash
cd code
python scripts/validate_rolling_controller_event.py \
  --samples 100000 \
  --out runs/rolling_controller_event_validation.json
```

The run is seeded (default `--seed 20260714`) and takes about 15 seconds.
Rerunning it from this repository reproduces every value in the saved
output. `REPORT.md` describes the method and pass criteria.

## SeQUeNCe comparison

`sequence_comparison/REPORT.md` contains the full method, results, and
reproduction command (the `compare_backend_physics.py` invocation under
"Reproduction"). The comparison uses a separate 10-km hop calibration rather
than the production component composition, as stated in the appendix. Its
saved output will be added to this folder.

## Runtime accounting

```bash
cd validation/runtime_accounting
python phase_times.py physics_compilation_passA.jsonl physics_compilation_passB.jsonl
```

The two `physics_compilation_*.jsonl` files are the graph-level logs of a
repeat execution of the 500,000-turn corpus configuration, split into two
passes (101 and 4 topologies). Each record splits elapsed time into baseline
calibration, concurrent online learning, and independent audit, and records
cumulative compilation and sampling timers after each phase.
`execution_plan_pass*.json` record the worker configuration (up to four
topology processes, two learner threads per process sharing one histogram
cache). Hours are summed across topologies, and the compilation and sampling
timers are cumulative and can overlap across learner threads.
