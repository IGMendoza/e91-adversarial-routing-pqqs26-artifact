# Modern path learner 100k comparison

Three matched seeds on the 23-route long composable Topology Bench case; 100,000 online turns and 16 independent audit samples per route--attack cell.

| Learner | Mean Alice utility ± SE | Exploitability ± SE | False-signal rate |
|---|---:|---:|---:|
| Flat Exp3 | 0.5013 ± 0.0011 | 0.0978 ± 0.0015 | 0.3137% |
| LinearExp2 G-optimal | 0.5064 ± 0.0007 | 0.0934 ± 0.0032 | 0.3327% |
| Maiti DAG FTRL | 0.3010 ± 0.0026 | 0.3601 ± 0.0217 | 0.2023% |
| Ito Tsallis | 0.5054 ± 0.0026 | 0.0985 ± 0.0123 | 0.3280% |
| Ito log barrier | 0.5059 ± 0.0020 | 0.1020 ± 0.0025 | 0.3240% |

Ito--Tsallis and Ito--log-barrier preserve mean service utility at the G-optimal LinearExp2 level. Their three-seed exploitability does not establish superiority: Tsallis is statistically noisy and log barrier is slightly worse in point estimate. Maiti at the theorem-scale schedule is not competitive and remains highly exploitable.

Maiti schedule, implicit-exploration, and valid signed-loss-baseline sweeps did not restore parity on disjoint calibration seeds. Retain it as a literature/scaling implementation, not as the active learner.

The edge-additive theorem assumptions remain an approximation to the histogram utility matrix; simultaneous Eve learning uses the adversarial rather than stationary interpretation.
