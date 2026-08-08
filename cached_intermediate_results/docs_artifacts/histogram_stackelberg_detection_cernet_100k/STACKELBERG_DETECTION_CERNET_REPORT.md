# Classical/quantum detection and Alice vulnerability on Cernet

Experiment digest: `4cb48d4ee92b6598a9c375a954d2221b`.

Physical-bank digest: `e466529c437aa4b2ab2912cccb665463`.

The study uses the normalized Cernet topology, 35 complete cycle-free
Alice--Bob routes inside the declared 12-hop scope, 113 declared physical Eve
actions, and one compiled histogram payoff bank. The bank contains 96 feedback
and 96 independent audit outcomes in each of the 3,955 route/action cells.

The strict-steering experiment contains 42 runs: two Alice learners, seven
detection/response conditions, and three learner/outcome-resampling seeds.
Each run has 25,000 clean turns followed by 75,000 turns of one fixed strict
Stackelberg commitment. Six additional 100,000-turn controls run ordinary
Tsallis-INF Eve co-learning against the two Alice learners on the same online
and audit cell distributions.

These seeds measure repeatability conditional on one physical bank. They do
not measure device-calibration uncertainty or variation across topologies.

## What the classical comparison means

The comparison is deliberately matched at the observation boundary:

| Boundary | Alice observes | Detector | Alarm response |
|---|---|---|---|
| Classical binary-service control | selected route plus accepted/rejected service | route-conditioned change \(e\)-process | freeze the current routing policy |
| Quantum service observation | selected route plus delivery failure, QBER abort, CHSH abort, or accepted | the same route-conditioned change \(e\)-process | freeze the current routing policy |

The classical row is a coarsening of the same physical route/action trace. It
is not a separately simulated classical transport network. It answers whether
the extra quantum outcome labels provide detection information beyond binary
service success for this attack.

Classical corruption-robust and combinatorial-bandit work addresses poisoned
rewards or corrupted observations. The present Eve changes the physical
service distribution and reports Alice's resulting outcome accurately.
Consequently, corruption-robust learning is relevant as a comparison but does
not automatically remove this attack.

## Information boundary

Neither detector receives:

- Eve's selected action or commitment;
- target-active diagnostics;
- Eve's private instrument record;
- corrected or counterfactual Alice rewards;
- audit outcomes; or
- simulator-hidden state.

The detectors use separately reserved clean commissioning counts. Their
no-change likelihood is the Dirichlet posterior predictive initialized by
commissioning. The alternative mixture covers route, outcome direction,
declared change fraction, and possible change time. The alarm threshold is
\(\log(1/\alpha)\), with \(\alpha=0.01\).

## Fixed physical commitment

Eve's commitment has the required response margin \(d=0.01\), targets the
three-hop route `15--29--25--24`, and mixes:

| Physical intervention | Probability |
|---|---:|
| `edge_dos:11-24` | 0.094996 |
| `memory_degradation:11` | 0.308001 |
| `memory_degradation:18` | 0.400968 |
| `swap_denial:25` | 0.196035 |

The clean audit best-route reward is 0.999447. Even a fully informed Alice can
obtain at most 0.606595 under the strict commitment. The commitment therefore
removes 39.3% of the clean physical service ceiling before considering online
learning error.

## Detection: no quantum-label advantage for this attack

Both observation regimes have zero alarms in the six clean controls. In the
no-monitoring detect-only traces, the binary and four-outcome detectors have
exactly the same alarm turn in all six attacked runs:

| Alice learner | Detection delays for either observation regime |
|---|---|
| Tsallis-INF (RV, \(\eta=1\)) | 7, 8, 9 turns |
| LinearExp2 (G-optimal, \(\eta=2\)) | 8, 9, 12 turns |

The commitment produces a sufficiently strong accepted/rejected service
change that QBER-versus-CHSH classification adds no onset-detection advantage.
This is a result for this commitment, not a general statement that quantum
integrity telemetry is redundant. An intervention that changes QBER/CHSH
composition without changing binary acceptance would separate the two
observation regimes.

## How vulnerable was Alice?

Mean Alice reward during the final 75,000 turns, divided by the clean audit
best-route reward, is:

| Alice learner | Clean | Ordinary Tsallis Eve | Strict Eve | Strict + freeze |
|---|---:|---:|---:|---:|
| Tsallis-INF | 0.998 | 0.604 | 0.596 | 0.596 |
| LinearExp2 | 0.982 | 0.593 | 0.586 | 0.582 |

Strict Eve is only 0.008 lower than ordinary co-learned Tsallis Eve for
Tsallis Alice and 0.007 lower for LinearExp2 Alice over this 75,000-turn
window. The strict leader is therefore not dramatically more damaging in
realized service than the ordinary co-learning control at this horizon.

Relative to the strict environment's 0.606595 full-information ceiling:

- Tsallis-INF retains 0.595926, leaving about 1.07% of clean-best-route reward
  as Alice adaptation loss;
- LinearExp2 retains 0.586112, leaving about 2.05% as Alice adaptation loss;
- the remaining 39.3% is physical loss imposed by Eve's commitment.

Most of the damage is therefore caused by the changed physical environment,
not Alice choosing far below the best route still available under that
environment.

## Steering and mitigation

The final Stackelberg-target policy mass is:

| Alice learner | Ordinary Tsallis Eve | Strict Eve | Freeze | Restart |
|---|---:|---:|---:|---:|
| Tsallis-INF | 0.5132 | 0.00145 | 0.00016 | 0.3818 |
| LinearExp2 | 0.4520 | 0.00328 | 0.00582 | 0.8181 |

The target route is common under ordinary co-learning, so target-route mass by
itself is not evidence of attack success. Under strict Eve, Alice's 25,000
clean turns prevent the intended route from becoming dominant during the
75,000-turn attacked window.

Freezing Alice's policy at alarm prevents subsequent steering and costs no
sentinel requests. The binary and four-outcome freeze controls are numerically
identical because their alarm turns coincide. Freeze does not improve
immediate service: it preserves the pre-alarm route distribution, including
routes that Eve has just degraded.

Restarting Alice is harmful. It removes the clean-history resistance and lets
the strict commitment steer Alice toward Eve's intended route.

## Classical versus quantum mitigation

For this attack, the response distinction is not detector speed but resource
and diagnostic scope:

- classical binary service outcomes are sufficient for onset detection;
- quantum QBER/CHSH categories provide mechanism-relevant integrity evidence,
  but that extra information is inactive here;
- policy freeze is available to either routing system and prevents learner
  steering without claiming to restore the damaged physical service;
- active quantum sentinel verification consumes entanglement requests, while
  passive service-outcome detection does not; and
- the histogram transcript can detect a route-conditioned distribution change
  but cannot localize its elementary-link, memory, swap, or control-plane
  cause.

## Interpretation boundary

The ordinary co-learning control and strict leader/follower process are
different games. Their realized rewards are directly comparable as physical
service outcomes, but their target-route masses and equilibrium diagnostics
do not have identical meanings.

This single-topology study does not establish performance under lower-margin
commitments, benign device drift, independently rebuilt physical banks,
different attack families, or the full Real Topologies corpus.

## Next controlled extensions

1. Test commitments that preserve binary acceptance while changing QBER/CHSH
   composition; these should exercise genuinely quantum-specific evidence.
2. Replace indefinite freeze with a guarded recovery controller that preserves
   a trusted checkpoint while testing whether the environment has recovered.
3. Solve a detection-constrained Stackelberg frontier.
4. Validate causal localization on a bounded event-level subset.
5. Repeat the comparison across multiple topologies and physical-bank seeds.

## Primary method references

- Gupta, Koren, and Talwar, *Better Algorithms for Stochastic Bandits with
  Adversarial Corruptions*, COLT 2019:
  <https://proceedings.mlr.press/v99/gupta19a.html>
- Balasubramanian et al., *Adversarial Attacks on Combinatorial Multi-Armed
  Bandits*, ICML 2024, including online shortest-path experiments:
  <https://proceedings.mlr.press/v235/balasubramanian24a.html>
- Shin, Ramdas, and Rinaldo, *E-detectors: a nonparametric framework for
  sequential change detection*:
  <https://arxiv.org/abs/2203.03532>

## Artifact guide

- `classical_quantum_detection`: exact classical-control versus quantum
  observation boundary and paired detector delays.
- `alice_vulnerability_comparison`: clean ideal, ordinary co-learning, strict
  Eve, freeze, restart, physical loss, adaptation loss, and target-route mass.
- `detection_performance`: clean false alarms and attacked detection delay.
- `detector_evidence_trajectories`: evidence around attack onset.
- `mitigation_tradeoffs`: monitoring consumption, application yield, residual
  steering, and Eve utility.
- `steering_policy_trajectories`: freeze versus restart policy evolution.
- `study_aggregates.csv`: machine-readable strict-condition aggregates.
- `colearning_controls.csv`: machine-readable ordinary Tsallis-Eve controls.
