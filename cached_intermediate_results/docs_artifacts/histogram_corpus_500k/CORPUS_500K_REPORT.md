# Normalized Real Topologies 500k learner results

Selection digest: `ff0cd3732bfaf3479232759a726ae501`.

Both pairings use 500,000 fresh compiled-histogram transcript turns per topology. Strategic metrics use the independent equal-coverage audit matrix.

| Pairing | Graphs | Median exploitability | Mean exploitability | Median Alice regret | Median Eve regret | Median turns/s |
|---|---:|---:|---:|---:|---:|---:|
| Tsallis-INF / Tsallis-INF | 105 | 0.006187 | 0.008597 | 0.001527 | 0.004527 | 641.6 |
| LinearExp2 / Tsallis-INF | 105 | 0.007890 | 0.010555 | 0.002954 | 0.004446 | 650.7 |

## Paired corpus comparison

The median Linear/Tsallis minus Tsallis/Tsallis exploitability is +0.001110; its finite-corpus topology-bootstrap 95% interval is [+0.000558, +0.001840]. Linear/Tsallis is lower on 22 of 105 topologies, equal on 5.

This is one matched seed per topology. Corpus-level paired differences do not establish repeated-seed uncertainty for an individual graph.

## Figures

- `nash_gap_summary.png`: explicit corpus mean/median Nash gaps and matched-topology differences with finite-topology bootstrap intervals.
- `topology_nash_gap_heatmaps.png`: every selected topology's two final Nash gaps, ranked by Tsallis/Tsallis gap in three consecutive columns.
- `pairing_exploitability.png`: corpus CDFs and paired differences.
- `topology_paired_exploitability.png`: all matched graph results.
- `action_geometry_and_gap.png`: route/action dimensions versus the paired learner difference.
- `learning_dynamics.png`: median policy, reward, and finite-sample decision-mismatch attribution trajectories.
- `service_and_signal.png`: matched service and finite-transcript outcomes.
- `corpus_case_geometry.png`: the frozen action/cache geometry.
