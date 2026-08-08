# Strict Stackelberg route-steering result

## Experiment

The study evaluated 5 normalized Real Topologies, 180 100k calibration runs, and 30 fresh 500k runs. The selected response margin was `0.01` for both Alice learners on every graph.

## Route induction

The intended Alice route remained an audit-bank best response in `30/30` runs. Its median empirical mass was `0.910445` and the minimum across all graph/learner/seed runs was `0.714594`.

Relative to the zero-margin boundary control, the positive commitment increased intended-route mass by a mean `+0.486926` for Tsallis-INF and `+0.494533` for LinearExp2.

## Leader value

The mean per-turn unconstrained Stackelberg regret was `0.003629` against Tsallis-INF and `0.001787` against LinearExp2. The plotted decomposition is exact up to mean absolute numerical error `0.000e+00`.

The positive response margin deliberately gives up rational-follower leader value. During finite learning, Alice often plays routes that are worse for her than the induced best response, so the signed finite-follower term can temporarily increase Eve's realized value.

## Information boundary

Eve uses the declared histogram feedback payoff surface and physical action costs to choose one stationary commitment. She does not observe Alice's live policy, weights, audit outcomes, or future feedback draws. The audit bank and evaluation seeds are disjoint from commitment construction and margin calibration.

## Figures

- `stackelberg_margin_frontier`: finite-time margin/value tradeoff.
- `stackelberg_strict_evaluation`: route induction and value decomposition.
- `stackelberg_target_route_mass`: target-route mass at `d=0` and `d=0.01`.
- `stackelberg_strict_vs_zero`: paired strict/boundary comparison.
- `stackelberg_physical_commitments`: exact intervention mixtures and target routes.

## Primary sources

- Deng, Schneider, and Sivan, *Strategizing against No-regret Learners*, NeurIPS 2019.
- Zhang, Ma, and Mazumdar, *Learning to Steer Learners in Games*, arXiv:2502.20770, 2025.
