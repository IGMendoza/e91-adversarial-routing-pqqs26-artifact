# Extended histogram validation

> Historical validation checkpoint. The strict adoption gate and flat-default
> language below record this run's predeclared analysis. Current learner roles
> are defined by `docs/PROJECT_BASELINE.md` and `docs/LEARNER_ROADMAP.md`.

Date: 2026-07-14
Status: complete

## Questions and claim boundaries

This extension tests two separate questions:

1. Does the compiled histogram backend preserve the route/target strategic
   structure seen in the event backend on a tractable matched subset?
2. Does topology-sharing improve the learned policy at 100,000 turns when flat
   Exp3 and linear route Exp2 are evaluated against the same independent payoff
   audit?

The first question uses **delivered fraction**, not the production
Bell-qualified-rate payoff.  The second uses the production histogram game but
currently covers one 23-route long-path topology and three matched seeds.  The
two validations therefore complement one another; neither is a claim of full
event/histogram distributional equivalence or corpus-wide learner superiority.

## Fixed-target event comparison

### Design

- Graphs: Braess diamond, uniform 1 km K10, and five-hop metro-WAN.
- Routes: all eight selected cycle-free routes, spanning 2--5 hops.
- Fixed graph-level node targets: two per graph.  The first has maximum route
  support; the second is selected by farthest-first Hamming diversity over
  route-incidence vectors.  Selection uses topology only, not backend outcomes
  or predicted route quality.
- Actions: one repeater baseline plus swap denial and repeater-memory
  measurement at each fixed target: five actions per route and 40 matched
  route/action cells in total.
- Sampling: eight SeQUeNCe event trials and 256 compiled-histogram trials per
  cell.  Cell standard errors, 95% Student-t intervals, and pairwise Welch-t
  decisions are retained.
- Compiled waiting model: rolling live-pair availability using the 10 km hop
  calibration and route availability 0.3.

No route was removed for low predicted physical quality.  An off-route target
is evaluated as a real miss.  No cell was removed by a timing or history-cap
condition; all 40 cells have `not_applicable` cap-bias status.

### Results

| Diagnostic | Result |
|---|---:|
| Eligible routes | 8 |
| Point-estimate pairwise orderings decided and agreeing | 65 / 65 |
| Point-estimate comparisons with a one-sided practical tie | 15 |
| 95% uncertainty-decided pairwise orderings agreeing | 41 / 41 |
| Comparisons left uncertainty-limited | 39 |
| Exact strongest-denial-set matches | 8 / 8 |
| Mean tolerance-tied route rank correlation | 0.8888 |

The fixed-target game matrices preserve best-response structure more strongly
than they preserve absolute delivery scale:

| Graph | Cell RMSE | Cell correlation | Event value | Histogram value | Cross-backend exploitability |
|---|---:|---:|---:|---:|---:|
| Braess | 0.4125 | 0.9843 | 0.0500 | 0.0500 | 0 both directions |
| K10 | 0.5990 | 0.9977 | 0.5344 | 0.1972 | 0 both directions |
| Metro-WAN | 0.0876 | 0.9882 | 0.0500 | 0.0500 | 0 both directions |

Thus every backend's equilibrium policy is a zero-exploitability policy in the
other backend's point-estimate control game on these three matrices.  The K10
value difference of 0.3372 and the mean cell RMSE of 0.3664 show why this must
not be described as absolute rate parity.  It is evidence for preserved
route/target control structure on the selected small-hop subset.

### Observable limitation

The memory-measurement implementations do not yet have magnitude parity.  In
this harness, compiled fixed-basis memory measurement places its disturbance in
the CHSH observable while leaving key-basis QBER unchanged; the event backend
raises sampled QBER and does not estimate CHSH.  CHSH abort is disabled for the
acceptance-aligned comparison.  Delivery ordering can therefore agree while
the quantum-disturbance fingerprints remain deliberately different.  The
present event result validates target hit/miss and denial structure, not
interchangeability of the two attack channels.

## Paired learner validation

### Predeclared design

- Flat reference: three existing 100,000-turn Exp3 runs, seeds 42--44.
- Candidate: three fresh 100,000-turn linear route Exp2 runs with the same
  seeds, topology, 23 cycle-free routes, 259 Eve actions, and uniform Alice
  prior.
- Reference oracle: pool only the flat runs' independent stratified post-hoc
  audits.  This produces 768 samples per route/action cell and 4,574,976 audit
  samples in total.  The matrix is never supplied to either learner.
- Paired metrics: total equivalent-game exploitability, Alice and Eve
  best-response gaps, and Alice regret against the flat, structured, and shared
  oracle Eve strategies.
- Adoption rule: retain structured learning only if the upper paired 95%
  confidence bound is below zero for both total exploitability and Alice regret
  against the shared oracle Eve.  Otherwise flat Exp3 remains the default.

The preflight surface contains 5,957 nominal route/action cells but only 2,121
canonical physics entries, at most 93 per route.  The 256-entry per-route LRU is
therefore sufficient; estimated fully resident cache memory is 397.7 MiB,
which is well below the available system memory during the run.

### Results

The adoption gate is **not supported**, so flat Exp3 remains the default.  A
negative delta favors structured learning.

| Paired metric, structured minus flat | Mean delta | Paired 95% t interval |
|---|---:|---:|
| Total reference exploitability | -0.00585 | [-0.04827, 0.03657] |
| Alice best-response gap at joint profile | -0.00533 | [-0.02429, 0.01362] |
| Eve best-response gap at joint profile | -0.00052 | [-0.02404, 0.02300] |
| Alice regret against shared oracle Eve | -0.00339 | [-0.00785, 0.00108] |
| Alice regret against matched flat Eve | -0.00349 | [-0.00660, -0.00038] |
| Alice regret against matched structured Eve | -0.00160 | [-0.00280, -0.00039] |
| Full-run mean Bell-qualified rate (pairs/s) | +0.0990 | [-0.1556, 0.3537] |
| Final-window Bell-qualified rate (pairs/s) | +0.2740 | [0.0497, 0.4983] |

Structured learning reduced Alice's regret against the shared oracle Eve in
all three seeds, but the three-seed interval still crosses zero.  Total
exploitability improved strongly in seed 43 and became slightly worse in seeds
42 and 44; its interval is correspondingly broad.  The learner also reduced
Alice's regret against both matched learned-Eve policies with intervals below
zero, while the Eve-side best-response gap did not improve reliably.

The positive final-window rate difference is useful evidence that topology
sharing accelerates Alice-side learning late in the run, but it is not enough
to replace the primary equilibrium gate.  The full-run rate interval crosses
zero, and the common-audit exploitability result does not demonstrate a robust
joint-profile improvement.  This is a reason to keep the prototype and extend
the seed/topology study, not evidence that the structured action model is
physically or algorithmically invalid.

The pooled reference contains 768 samples per cell (4,574,976 total), has
maximum cell standard error 0.00439, and was generated strictly after the flat
learners completed.  The structured production run completed 300,000 online
turns.  Its first seed, including cold physics construction, took 631.6 s;
warm-cache seeds took 97.7 s and 96.9 s.

Machine-readable paired results and the PNG/PDF validation figure are under
`runs/histogram_colearning_structured_100k_unbiased_routes_3seed/analysis/paired_validation/`.

## What this does and does not validate

Supported if the reported diagnostics remain reproducible:

- fixed physical targets remain hits or misses across hidden Alice routes;
- the compiled and event backends agree on every uncertainty-decided ordering
  and every strongest denial set in the selected 2--5-hop subset;
- the compared point-estimate games have mutually non-exploitable equilibrium
  policies despite different absolute delivery scales.

Not established:

- absolute rate or full histogram distribution parity;
- CHSH/QBER disturbance equivalence for memory measurement;
- event-level parity on the production learner topology containing 15- and 16-edge routes;
- learner superiority beyond three seeds or beyond this topology;
- finite-key security, authenticated classical communication, or Eve's private
  information objective.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python \
  scripts/compare_backend_physics.py \
  --out-dir runs/backend_physics_strategic_matrix_extended \
  --graphs braess k10 metro_wan \
  --comparison-mode fixed_route_strategic_matrix \
  --sequence-trials 8 --compiled-trials 256 \
  --strategic-node-targets 2 --allow-wavelength-mismatch \
  --compiled-histogram-hop-calibration-10km \
  --compiled-histogram-hop-calibration-route-availability 0.3

MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python \
  scripts/histogram_colearning_paper.py \
  --out-dir runs/histogram_colearning_structured_100k_unbiased_routes_3seed \
  --turns 100000 --seeds 42,43,44 --case long_composable \
  --alice-learner linear_route_exp2 \
  --posthoc-samples-per-cell 64 --fresh

MPLCONFIGDIR=/tmp/matplotlib .venv/bin/python \
  scripts/analyze_histogram_learner_validation.py \
  --flat-bundle runs/histogram_colearning_paper_100k_unbiased_routes \
  --structured-bundle \
    runs/histogram_colearning_structured_100k_unbiased_routes_3seed \
  --case long_composable
```
