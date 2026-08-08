# Physical Prime-and-Probe: 500k-turn defender study

Source corpus selection digest: `ff0cd3732bfaf3479232759a726ae501`.

The study contains five predeclared normalized Real Topologies, two Alice defenders, and three learner/outcome-resampling seeds conditional on one shared equal-coverage physical bank per topology. Each ordered intervention is paired with a temporal permutation of exactly the same physical Eve-action multiset.

Eve acts only through declared compiled-histogram instruments. The exploit decision uses the count-valued local probe record; it never receives Alice's policy, weights, route identity, target-active flag, or audit outcomes. The independent audit split is used only for scoring.

| Defender | Runs | Median post-prime target-mass uplift | Probe triggered | Median ordered−shuffled Alice reward | Median audit difference | Joint recovered | Median joint latency among recovered |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tsallis-INF (RV, $\eta{=}1$) | 15 | +0.999661 | 15/15 | +0.011167 | +0.010724 | 4/15 | 96,500 turns |
| LinearExp2 (G-optimal, $\eta{=}2$) | 15 | +0.989346 | 15/15 | -0.002099 | -0.002428 | 1/15 | 114,000 turns |

## Main findings

Priming changes the target-family policy mass by a median +0.999661 for Tsallis-INF and +0.989346 for LinearExp2. Memorex is the important exception visible in the topology figure: the selected target family already carries almost all late-warmup mass, so its continuous uplift is near zero rather than evidence that priming created a new concentration.

Ordered timing lowers Alice's realized reward relative to the shuffled identical-budget control in 12/15 LinearExp2 runs, compared with 4/15 Tsallis-INF runs. The paired medians are -0.002099 and +0.011167, respectively. On these five cases and this declared schedule, the result does not support the hypothesis that the LinearExp2 defender is more resistant to temporal steering.

Joint recovery occurs within 145,000 turns in 4/15 Tsallis-INF runs and 1/15 LinearExp2 runs. Right-censoring is therefore part of the result and is not converted into an inferred recovery time.

## Selected physical intervention plans

| Topology | Routes | Target routes | Exploit instrument | Prime actions | Coverage floor | Prime loss floor | Exploit loss mean |
|---|---:|---:|---|---:|---:|---:|---:|
| Cernet | 35 | 2 | `swap_denial:18` | 2 | 0.5000 | 0.211041 | 0.916688 |
| Hibernia Ireland | 2 | 1 | `swap_denial:4` | 1 | 1.0000 | 0.999583 | 0.787818 |
| Karen | 2 | 1 | `swap_denial:23` | 1 | 1.0000 | 0.999521 | 0.768389 |
| Layer42 | 4 | 2 | `edge_dos:2-6` | 1 | 1.0000 | 0.946991 | 0.834446 |
| Memorex | 8 | 2 | `edge_dos:17-9` | 2 | 0.5000 | 0.226760 | 0.918438 |

## Interpretation boundary

Across the 30 ordered runs, the median count-derived target-mass midpoint error relative to the offline policy diagnostic is -0.000019.

This is a finite-horizon leader/follower experiment, not an assertion that an asymptotic Stackelberg theorem transfers unchanged to the nonstationary, general-sum physical network game. The shuffled control isolates temporal ordering at a fixed action economy; it does not remove stochastic transcript variation. Three resampling seeds per topology quantify finite-run repeatability conditional on the shared bank for these five cases, not device-calibration uncertainty or uncertainty over the full topology population.

The local target-mass interval is conditional on the fixed 64-draw-per-cell feedback-bank calibration: its standard-error term measures the 5,000-turn probe record, while the route-cell minimum/maximum calibration bounds remain fixed. It is not a joint confidence interval over a newly estimated device model.

## Artifact guide

- `prime_probe_phase_protocol`: exact 500k phase schedule, controller decisions, and the online information boundary.
- `prime_probe_network_mechanism`: route-family view of priming alternatives, locally probing the target, exploiting concentration, and withdrawing for recovery.
- `ordered_phase_dynamics`: service and target-family policy response through warmup, prime, probe, exploit, and recovery.
- `timing_matters`: ordered-versus-shuffled outcome under an exactly matched physical action multiset.
- `probe_information`: locally count-identifiable target mass versus offline diagnostics unavailable to Eve.
- `recovery`: sustained recovery to each run's late-warmup service/policy envelope.
- `topology_defender_outcomes`: seed-resolved timing effects and continuous post-prime target-family uplift.
- `selected_bottleneck_physics`: topological coverage and compiled physical loss of each selected plan.
