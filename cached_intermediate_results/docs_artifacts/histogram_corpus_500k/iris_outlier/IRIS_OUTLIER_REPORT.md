# Iris outlier analysis

## Result

Iris is rank 1 of 105 by Tsallis/Tsallis Nash gap: `0.058021` versus the corpus median `0.006187`. Its Alice contribution is `0.051134` and its Eve contribution is `0.006887`.

With the same Tsallis-INF Eve and the same frozen Iris physics, LinearExp2 Alice gives gap `0.009551` (rank 40; empirical percentile 62.9). The matched change is `-0.048470`.

## Network mechanism visible in this run

The admitted action set contains all 20 cycle-free Alice--Bob routes within the declared 12-hop scope. Enumeration completed within that scope and no route-quality pruning was used.

Every non-direct route shares the structural corridor edges `1-10`, `17-18`. The direct route is therefore a separate route family, while the other nineteen arms are alternative internal paths between the same two corridor boundaries.

Tsallis-INF Alice concentrated on route r4 (`10-1-34-14-17-18`, 5 hops, 27.83 normalized km) with empirical mass `0.701`. That route has zero audited equilibrium mass.

The audited equilibrium instead uses r18 (`10-1-37-17-18`, 4 hops, 15.13 km) with mass `0.813` and the direct r19 route with mass `0.187`. Tsallis selected the detour only `0.000102` of turns; LinearExp2 selected it `0.762`.

The corresponding audited Eve support is `swap_denial:1` (0.187), `edge_information_probe:10-18` (0.813). This is the expected response to Iris's two effective route families: deny the shared repeater corridor or interrogate the direct edge.

## Does Iris violate Tsallis-INF assumptions?

No formal violation is visible in the experiment contract. Alice has a fixed finite arm set, rewards are bounded, she observes only her selected-arm scalar reward, and Eve samples simultaneously from a policy determined by prior history. The Tsallis-INF source permits adaptive adversarial losses that depend on actions through the preceding turn.

The source guarantee is an expected pseudo-regret bound. The reported Iris value is one seed's audited external regret against the realized opponent-action sequence. It is therefore not a contradiction of the theorem and does not estimate how frequently this lock-in occurs.

The important modeling distinction is structural information. Tsallis-INF treats all 20 route labels as independent arms and cannot transfer an observation from r4 to the structurally similar r18. LinearExp2 represents the 20 routes in the rank-14 span of 57 declared constant/edge/node incidence coordinates. Its selected-route reward update therefore changes the estimates of routes sharing those resources. In this run it finds the audited two-route family. This is evidence that the Iris action geometry is a consequential invariant for Alice.

## Remaining graph-specific validation

Run repeated Iris seeds with the same frozen route/action surface and independent feedback banks. Report the distribution of Alice external regret, the probability of concentrating on r4 rather than r18, and time-to-transfer mass onto the audited support. That experiment distinguishes a rare one-seed realization from a reproducible independent-arm failure mode.

## Figures

- `iris_outlier_network`: full Iris graph, route multiplicity, and exact key corridor routes.
- `iris_outlier_dynamics`: corpus rank, Nash-gap decomposition, and the two Alice learners' cumulative route frequencies.

## Primary learner reference

[Zimmert and Seldin, *Tsallis-INF: An Optimal Algorithm for Stochastic and Adversarial Bandits*, JMLR 22(28), 2021](https://www.jmlr.org/papers/v22/19-753.html).
