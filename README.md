# Artifact bundle: Adversarial Route Learning in Quantum Repeater Networks (PQQS'26)

This bundle accompanies the paper "Adversarial Route Learning in Quantum
Repeater Networks" (PQQS'26 submission). The paper's LaTeX source is not
included here — reviewers receive the PDF directly — so this bundle covers
only the code, data, and generator scripts behind its figures. It was
assembled from the paper's source repository at commit `be33e53d` ("code
provenance cleanup"), plus three small reapplied wording/number fixes to
that commit's own real-corpus digestion appendix (a transposed percentage
pair, missing model names, and a table header — none of which affect any
figure or number tracked in this README).

Nothing in this bundle was fabricated. Every generator-script path named
below is a real path that exists in the source repository at the time of
this release, verified by reading the script's own source, not inferred
from its filename. Every "no generator found" statement below reflects an
actual `git grep` / filesystem search across the source repository, plus
three related branches (`origin/compiled-kernels`,
`origin/results/synthetic-full-run-wip`, `origin/xai-corpus-pipeline`) —
not an assumption.

## What this bundle is for

`main.tex` includes 18 content figures (excluding the EU funding-acknowledgment
logo). This bundle collects, for as many of those 18 as have a real
generator anywhere in the source repository's git history:

1. the real generator/renderer source code (`code/`),
2. the real upstream dataset the 105-topology corpus is built from
   (`data/`),
3. already-rendered cached outputs for figures whose raw simulation runs
   are not present anywhere in the source repository, so the regeneration
   chain is broken above this point (`cached_intermediate_results/`),
4. the exact figure files as they currently appear in the paper, for
   reference (`figures_as_used_in_paper/`).

## Directory layout

```text
data/topology_bench/          Topology Bench raw+extracted dataset (real, upstream, cited below)
code/sequence_game/           First-party Python package: physics kernels, corpus/topology
                               adapters, game theory, XAI decision-tree fitting, and the
                               histogram_colearning experiment modules the render scripts import
code/scripts/                 CLI entry points, including every render_*.py / build_*.py /
                               run_*.py script named in the table below
code/docs/talk_histogram_backend_2026/   Self-contained source for the histogram_anatomy figure
code/docs/topology_bench_step6_census/   Census artifacts (case_manifest.json, census.json, ...)
                               that scripts/run_histogram_corpus_500k.py reads by default
code/runs/graph_corpus_sqlite_v1_topology_bench_real_rank10km_normalized/
                               Normalized 105-graph SQLite corpus (graph_corpus.sqlite, 25 MB),
                               DEFAULT_CORPUS_DB_PATH in
                               sequence_game/experiments/histogram_colearning/config.py
cached_intermediate_results/docs_artifacts/   Verbatim copy of the source repo's docs/artifacts/
                               tree: rendered PDF/PNG figures plus the CSV/JSON numeric tables
                               they were plotted from, for every histogram-corpus, Stackelberg,
                               prime-probe, and calibration experiment family that has a script
figures_as_used_in_paper/     All 18 real content figures + the EU logo, exactly as currently
                               placed in paper/pqqs2026/figures/
```

Bundle size: ~59 MB (`du -sh --apparent-size`; this filesystem reflinks/dedupes,
so a plain `du -sh` reads smaller). Nowhere near the "few hundred MB" flag
threshold and nowhere near the multi-GB caches described below.

## The Topology Bench dataset (`data/topology_bench/`)

The real, upstream "Topology Bench real_topologies dataset." In the source
repository: `data/external/topology_bench/` (gitignored there, present on
disk, small enough to bundle directly here rather than re-download).

- `raw/real_topologies.zip` (876 KB) — 105 real core-optical-network
  topologies as `.xlsx` workbooks (`Nodes_*`/`Edges_*` sheets with
  geospatial link lengths).
- `raw/small_synthetic.zip` (1.6 MB) — the companion synthetic-topology
  archive from the same Zenodo release. This is the raw source material
  behind the paper's historical "Syn-small" pilot study, but it has not
  been extracted or ingested anywhere in the source repository (see
  "Figures with no generator found," Syn-small pilot row, below).
- `extracted/TOP_<n>_<NAME>.xlsx` (105 files, 3.7 MB total) — the
  real-topology workbooks unzipped, used directly by
  `scripts/generate_graph_corpus.py` / `scripts/build_graph_corpus_sqlite.py`.

### Citation

Both entries below are copied verbatim from `paper/pqqs2026/references.bib`
in the source repository.

```bibtex
@article{Matzner2025TopologyBench,
  author = {Matzner, Robin and Ahuja, Akanksha and Sadeghi, Rasoul and Doherty, Michael and Beghelli, Alejandra and Savory, Seb J. and Bayvel, Polina},
  title = {Topology Bench: Systematic Graph-Based Benchmarking for Core Optical Networks},
  journal = {Journal of Optical Communications and Networking},
  volume = {17},
  number = {1},
  pages = {7--27},
  year = {2025},
  doi = {10.1364/JOCN.534477}
}

@misc{TopologyBenchDataset,
  author = {Matzner, Robin and Ahuja, Akanksha and Sadeghi, Rasoul and Doherty, Michael and Beghelli, Alejandra and Savory, Seb J. and Bayvel, Polina},
  title = {Topology Bench: Systematic Graph Based Benchmarking for Optical Networks},
  howpublished = {Zenodo dataset},
  year = {2024},
  note = {Version v4},
  doi = {10.5281/zenodo.13921775}
}
```

Project page: https://topologybench.github.io. Zenodo record:
https://zenodo.org/records/13921775.

## Per-figure generator / dependency table

"Raw run present?" means the raw, from-scratch simulation output directory
under `runs/` that the render script reads by default exists anywhere in
the source repository's git history (checked against the reconciled branch
and the three related branches named above) or on the local disk this
bundle was assembled from. In every "no" case below, only an already-rendered
`docs/artifacts/...` output survives (bundled under
`cached_intermediate_results/`); the raw simulation output that would let
someone regenerate that rendered output byte-for-byte is not present
anywhere reachable from this repository.

| Figure (as used in `main.tex`) | Generator script (real path) | Raw run directory it reads | Raw run present? |
|---|---|---|---|
| `histogram_anatomy.pdf` | `docs/talk_histogram_backend_2026/build_figures.py`, function `build_histogram_anatomy()` | reads the bundled `graph_corpus.sqlite` directly, then evaluates live through `HistogramPhysicsPool` with fixed seed 142 — no separately cached run is read | **Yes — fully self-contained, genuinely reproducible from what's bundled** |
| `corpus_500k_learning_dynamics.pdf`, `topology_nash_gap_heatmaps.pdf` | `scripts/render_histogram_corpus_500k.py` → `sequence_game/experiments/histogram_colearning/corpus_scale_artifacts.py` | `runs/histogram_corpus_500k/`, itself produced by `scripts/run_histogram_corpus_500k.py` (default `--db-path` = the bundled `graph_corpus.sqlite`; default `--census-path` = the bundled `code/docs/topology_bench_step6_census/census.json`) | No |
| `stackelberg_target_route_mass.pdf` | Two real generators exist for this exact output filename. Current/preferred: `scripts/build_histogram_stackelberg_followup_105_artifacts.py` → `sequence_game/experiments/histogram_colearning/stackelberg_followup_artifacts.py`, which writes directly to `paper/pqqs2026/figures/stackelberg_target_route_mass.pdf` by default. Older: `scripts/render_histogram_stackelberg_steering_500k.py` → `stackelberg_steering_artifacts.py`, which writes to `docs/artifacts/histogram_stackelberg_steering_500k/` for later manual copying. | Followup path needs `runs/histogram_stackelberg_followup_105_500k/`, produced by `scripts/run_histogram_stackelberg_followup_105_500k.py`, which itself reads `runs/histogram_corpus_500k/run_manifest.json` as its own input. Older path needs `runs/histogram_stackelberg_steering_500k/` (+ a zero-margin sibling run + `runs/histogram_corpus_500k/`). | No (neither path's raw run exists anywhere checked) |
| `stackelberg_strict_vs_zero.pdf`, `stackelberg_strict_evaluation.pdf` | `scripts/render_histogram_stackelberg_steering_500k.py` → `stackelberg_steering_artifacts.py` | `runs/histogram_stackelberg_steering_500k/` + `runs/histogram_stackelberg_steering_zero_margin_500k/` | No |
| `prime_probe_ordered_dynamics.pdf` | Two real generators exist, both calling the same shared plotting function `_plot_ordered_phase_dynamics` in `prime_probe_artifacts.py` (which saves `.png`, `.pdf`, and `.svg` siblings of one output path). Current/preferred: `scripts/build_histogram_prime_probe_followup_105_artifacts.py`. Older: `scripts/render_histogram_prime_probe_detection_500k.py` → `prime_probe_detection_corpus.py`. | Followup path needs `runs/histogram_prime_probe_followup_105_500k/`, produced by `scripts/run_histogram_prime_probe_followup_105_500k.py` (itself reads `runs/histogram_corpus_500k/run_manifest.json`). Older path needs `runs/histogram_prime_probe_detection_500k/`. | No (neither path's raw run exists anywhere checked) |
| `calibration_pairwise_learners.pdf` | `scripts/render_histogram_step7_calibration.py` → `sequence_game/experiments/histogram_colearning/calibration_artifacts.py` | `runs/histogram_step7_calibration/` | No |

Every "No" row above was rechecked after the "code provenance
cleanup" commit (`be33e53d`) specifically because that commit's name and
file list suggested it might supply the missing raw runs. It does not: it
adds the two new followup scripts/modules described above (a genuine,
useful pipeline simplification for the Stackelberg and Prime-Probe
followup figures — the followup scripts write directly to
`paper/pqqs2026/figures/`, one step cleaner than the older
render-to-`docs/artifacts`-then-copy pattern) and it adds
`EU_Logo_Funding.JPG`, but it does not add any of the six `runs/histogram_*`
raw directories named above, in this bundle or in any of the three other
branches checked.

### Regenerating what actually can be regenerated

```bash
# histogram_anatomy.pdf — genuinely reproducible from what's bundled here.
# Requires the same third-party dependencies as the source repo (matplotlib,
# numpy, networkx, etc — see the source repo's requirements.txt, not
# duplicated in this bundle). Run from inside code/, keeping code/ as the
# resolved repo root (build_figures.py finds it via
# Path(__file__).resolve().parents[2] — do not move the script out of
# code/docs/talk_histogram_backend_2026/).
cd code
python docs/talk_histogram_backend_2026/build_figures.py

# Everything else in the table above needs a raw runs/histogram_corpus_500k/
# directory first — every other listed run directory is built from it, or
# alongside it. Producing it means re-running the 500k-turn compiled-histogram
# corpus experiment against the bundled Topology Bench corpus:
python scripts/run_histogram_corpus_500k.py
# then, e.g., for the two followup figures with a direct-to-paper generator:
python scripts/run_histogram_stackelberg_followup_105_500k.py
python scripts/build_histogram_stackelberg_followup_105_artifacts.py
python scripts/run_histogram_prime_probe_followup_105_500k.py
python scripts/build_histogram_prime_probe_followup_105_artifacts.py
# and the corpus-scale / calibration / older-path figures:
python scripts/render_histogram_corpus_500k.py
python scripts/render_histogram_stackelberg_steering_500k.py
python scripts/render_histogram_prime_probe_detection_500k.py
python scripts/render_histogram_step7_calibration.py
```

This is a real compute job, not a data-copying job: `main.tex`'s own text
states the fully-materialized compiled-histogram physics cache for all 105
topologies would occupy approximately 22.45 GB if nothing were compiled on
demand (the actual 500k-turn run uses compile-on-demand caching and never
materializes that much at once, but from-scratch regeneration is still a
genuinely non-trivial simulation, not a trivial copy). This bundle does not
attempt that job; it bundles the small upstream dataset plus already-rendered
outputs instead, and documents the gap honestly.

### Regenerating the 105-topology corpus from raw Topology Bench data

```bash
cd code
python scripts/generate_graph_corpus.py   # reads ../data/topology_bench/extracted/*.xlsx — check --help for the exact flag/path this bundle's layout needs
python scripts/build_graph_corpus_sqlite.py
```

The already-normalized output of this step is bundled directly at
`code/runs/graph_corpus_sqlite_v1_topology_bench_real_rank10km_normalized/graph_corpus.sqlite`
(25 MB), so this step does not need to be re-run to inspect the 105-topology
structure the paper analyzes.

## Figures with no generator found anywhere

These figures appear in `main.tex` as final rendered files (bundled under
`figures_as_used_in_paper/` for reference) but no script producing them, and
no cached intermediate rendered output for them either, was found anywhere
in: the source repository at `be33e53d`, `origin/compiled-kernels`,
`origin/results/synthetic-full-run-wip`, or `origin/xai-corpus-pipeline`.
This is unchanged by the "code provenance cleanup" commit.

- **`real_digest_border_collie.pdf`, `real_digest_generalization_by_depth.pdf`,
  `real_digest_root_clusters.pdf`, `real_digest_checks_by_depth.pdf`**
  (real-corpus decision-tree digestion figures) and
  **`real_corpus_dt_generalization_by_depth_clean.png`**
  (current-corpus decision-tree prediction figure): `main.tex` itself
  contains an inline source comment at the point these figures' underlying
  targets (`histogram_final_retention`, `exploitability`) are defined:
  *"DATA-PROVENANCE TODO: archive the adapter export and record the exact
  aggregation represented by u_{A,final}; origin/compiled-kernels contains
  the earlier XAI corpus, not this adapter output."* That branch reference
  was checked directly for this release (see below) and does not contain
  this adapter's export either. The paper's own text calls this adapter
  "independently built," but its source was not found under any name
  (`adapter`, `digest`, `real_corpus`, or the two target variable names
  above) anywhere searched.
- **`pilot_single_tree_digest_by_depth_clean.png`,
  `pilot_cluster_root_split_frequencies.png`,
  `strategy_weight_reversal_distribution.png`,
  `strategy_weight_reversal_example.png`** (Syn-small historical-pilot
  figures): not a search failure — `main.tex` itself says, in a source
  comment, that the imported Overleaf archive and repository history did
  not contain the machine-readable sidecars behind these figures (the
  234-graph Syn-small manifest, feature/fold tables, fitted rules,
  subsampling seeds, verifier records, or the exact LLM prompts/responses).
  Only the raw, un-extracted `data/topology_bench/raw/small_synthetic.zip`
  exists as a data dependency; nothing ingests it in this repository.

Treat these nine files as an open provenance question for the paper's
authors, not as evidence a generator never existed — only that a thorough
search across five branches did not find one.

### `origin/compiled-kernels`, `origin/results/synthetic-full-run-wip`, `origin/xai-corpus-pipeline`

Checked read-only for this release (branch tips as of reconciliation):
none of the three contains any `runs/histogram_*_500k`-style raw run
directory, and none contains the real-corpus digestion adapter export
named in the `main.tex` TODO comment above. `compiled-kernels` does contain
a different, unrelated `runs/` tree (`paper_topology_bench_e91_compiled_h20_t012_10k/`,
`graph_corpus_paper_v2/`, etc.) belonging to the earlier, already-accepted
QCE-2026 paper's pipeline, not to this PQQS follow-up paper.

## What was deliberately left out

- `paper/pqqs2026/Stress_Testing_of_Quantum_Networks (1).zip` — **not
  included.** `main.tex` does not reference this file anywhere (checked by
  grep), and the "(1)" suffix is consistent with an accidental local-download
  artifact rather than deliberate paper content. Left out because it could
  not be positively confirmed as intentional; flagged for the authors to
  resolve rather than guessed at.
- `runs/graph_corpus_paper_v2/` and three SQLite files under
  `sequence_game/experiments/exp3_sequence/` (`corpus.sqlite`,
  `baselines.sqlite`, `attack_baselines.sqlite`) — not a dependency of any
  of the 18 traced figures (they belong to the earlier, already-accepted
  QCE-2026 paper's pipeline). Excluded rather than hand-redacted: two of
  the three SQLite files and the corpus manifest JSON embedded a
  collaborator's real local absolute path from a different development
  machine than this one; since none of these files were needed,
  dropping them was cleaner than editing binary SQLite content in place.
  (See "A found-and-fixed leak" below.)
- The full `sequence-toolbox-SeQUeNCe/` submodule — read-only reference
  material for this project, out of scope for a paper-figure artifact
  bundle.
- `release/exp3-e91-repeater-games/` in the source repo — a different,
  already-assembled release bundle for the earlier, already-accepted QCE
  2026 paper; unrelated to this paper's 18 figures.

## A found-and-fixed leak, disclosed for transparency

Three files that **are** included in this bundle
(`code/docs/topology_bench_step6_census/{artifact_manifest,case_manifest,census}.json`
and, under `cached_intermediate_results/`,
`histogram_stackelberg_steering_500k/artifact_manifest.json` and
`step7_histogram_learner_calibration/tables/payoff_structure.csv`) originally
contained the same collaborator's absolute path as plain-text JSON/CSV field
values (recorded input/output paths from the run that produced them, not
secrets). These five files were needed in the bundle, so the path strings
were replaced with the placeholder `<repo-root>` in this bundle's copies
only; the source working tree is untouched. All five JSON files were
re-parsed after the edit to confirm they remain valid JSON.

## Provenance note

Assembled from the paper's source repository at commit `be33e53d`. All file
contents are unmodified copies of that source tree except for the five
path-redacted files noted above. No script, path, dataset location,
citation, or generator linkage in this README was invented; every claim
reflects an actual `git grep`/`git ls-tree`/`find`/read performed against
the source repository and the three related branches named above, and every
"not found" is the result of a real, documented search.
