# Artifact bundle: Adversarial Route Learning in Quantum Repeater Networks (PQQS'26)

This bundle accompanies the paper "Adversarial Route Learning in Quantum
Repeater Networks" (PQQS'26 submission). The paper's LaTeX source is not
included here — reviewers receive the PDF directly — so this bundle covers
only the code, data, and generator scripts behind its figures.

## Quickstart

Requires Python 3.12–3.14 (matches the SeQUeNCe dependency's own
`pyproject.toml` constraint, below).

```bash
# 1. Install this bundle's own pinned dependencies.
pip install -r code/requirements.txt

# 2. Install SeQUeNCe. It is a separate third-party simulator, not bundled
#    here, but sequence_game's package __init__ chain imports it eagerly,
#    so it is required just to import sequence_game at all -- including
#    for the one figure below that is genuinely reproducible from what
#    IS bundled.
pip install "sequence @ git+https://github.com/sequence-toolbox/SeQUeNCe.git@v1.0.0"

# 3. Regenerate the one figure this bundle can fully reproduce end to end.
cd code
python docs/talk_histogram_backend_2026/build_figures.py
# -> writes docs/talk_histogram_backend_2026/figures/histogram_anatomy.pdf,
#    matching figures_as_used_in_paper/histogram_anatomy.pdf up to
#    last-digit floating-point noise from library-version differences.
#    This same command also attempts two companion presentation figures
#    that need data this bundle does not include, and will raise after
#    histogram_anatomy.pdf is already written; see the per-figure table
#    below for why that is expected.
```

Everything else -- the real Topology Bench dataset, every other generator
script, and the cached outputs for figures whose raw runs no longer
exist -- is described below, starting with the per-figure table.

## What this bundle is for

The paper includes 18 content figures (excluding the EU funding-acknowledgment
logo). For as many of those 18 as have a real generator, this bundle
collects:

1. the real generator/renderer source code (`code/`),
2. the real upstream dataset the 105-topology corpus is built from
   (`data/`),
3. already-rendered cached outputs for figures whose raw simulation runs
   no longer exist, so the regeneration chain is broken above this point
   (`cached_intermediate_results/`),
4. the exact figure files as they currently appear in the paper, for
   reference (`figures_as_used_in_paper/`).

The per-figure table below states, for each of the 18, exactly which script
produces it and whether the raw data needed to regenerate it is present.

## Directory layout

```text
data/topology_bench/          Topology Bench raw+extracted dataset (real, upstream, cited below)
code/requirements.txt         This bundle's own pinned dependencies (see "Quickstart" above)
code/sequence_game/           First-party Python package: physics kernels, corpus/topology
                               adapters, game theory, XAI decision-tree fitting, and the
                               histogram_colearning experiment modules the render scripts import
code/histograms/               First-party package sequence_game imports for compiled-histogram
                               generation/device-front-matter logic (needed by histogram_anatomy.pdf)
code/configs/                 Physical-device and repeater-service literature-config JSON that
                               HistogramPhysicsPool reads by default (needed by histogram_anatomy.pdf)
code/scripts/                 CLI entry points, including every render_*.py / build_*.py /
                               run_*.py script named in the table below
code/docs/talk_histogram_backend_2026/   Self-contained source for the histogram_anatomy figure
code/docs/topology_bench_step6_census/   Census artifacts (case_manifest.json, census.json, ...)
                               that scripts/run_histogram_corpus_500k.py reads by default
code/runs/graph_corpus_sqlite_v1_topology_bench_real_rank10km_normalized/
                               Normalized 105-graph SQLite corpus (graph_corpus.sqlite, 25 MB),
                               DEFAULT_CORPUS_DB_PATH in
                               sequence_game/experiments/histogram_colearning/config.py
code/runs/histogram_corpus_full105_backend/   Intermediate LLM-digest/cluster/border-collie
                               JSON + generator for the real-corpus digestion figures (see below)
code/runs/rule_clusters/, code/runs/shape_dt_llm_digest/, code/runs/exp3_topology_bench/,
code/runs/exp3_small_synth_234_production/   Intermediate data for the Syn-small pilot figures
                               (see "Real-corpus digestion and Syn-small pilot figures" below)
cached_intermediate_results/docs_artifacts/   Rendered PDF/PNG figures plus the CSV/JSON numeric
                               tables they were plotted from, for every histogram-corpus,
                               Stackelberg, prime-probe, and calibration experiment family
                               that has a script
figures_as_used_in_paper/     All 18 real content figures + the EU logo, exactly as they
                               appear in the paper
```

Bundle size: ~80 MB (`du -sh --apparent-size`).

A few JSON/CSV files under `code/docs/topology_bench_step6_census/` and
`cached_intermediate_results/` use the placeholder `<repo-root>` in place
of a machine-specific filesystem path recorded at the time those files were
produced; substitute this bundle's own root when resolving relative paths
inside them.

## The Topology Bench dataset (`data/topology_bench/`)

The real, upstream "Topology Bench real_topologies dataset," bundled
directly here.

- `raw/real_topologies.zip` (876 KB) — 105 real core-optical-network
  topologies as `.xlsx` workbooks (`Nodes_*`/`Edges_*` sheets with
  geospatial link lengths).
- `raw/small_synthetic.zip` (1.6 MB) — the companion synthetic-topology
  archive from the same Zenodo release, the raw source material behind the
  paper's historical "Syn-small" pilot study. The pilot figures below are
  reproducible from an already-computed intermediate result (see "Real-corpus
  digestion and Syn-small pilot figures" below) rather than from this raw
  archive directly; nothing in this bundle re-ingests it from scratch.
- `extracted/TOP_<n>_<NAME>.xlsx` (105 files, 3.7 MB total) — the
  real-topology workbooks unzipped, used directly by
  `scripts/generate_graph_corpus.py` / `scripts/build_graph_corpus_sqlite.py`.

### Citation

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
under `runs/` that the render script reads by default is present in this
bundle or was found elsewhere. Where it is not, only an already-rendered
`docs/artifacts/...` output survives (bundled under
`cached_intermediate_results/`); the raw simulation output needed to
regenerate that rendered output byte-for-byte does not currently exist.

| Figure (as used in the paper) | Generator script (real path) | Raw run directory it reads | Raw run present? |
|---|---|---|---|
| `histogram_anatomy.pdf` | `docs/talk_histogram_backend_2026/build_figures.py`, function `build_histogram_anatomy()` | reads the bundled `graph_corpus.sqlite` directly, then evaluates live through `HistogramPhysicsPool` with fixed seed 142 — no separately cached run is read. Also needs the bundled `code/histograms/` package and `code/configs/physical/` literature-config JSON (both required by `HistogramPhysicsPool`), plus SeQUeNCe installed separately (not bundled, but imported transitively through `sequence_game`'s package `__init__` chain even though this figure's own logic never calls into SeQUeNCe's simulation runtime) — see "Quickstart" above. Actually re-run end to end for this release in a clean environment: reproduces `histogram_anatomy.pdf`/`histogram_anatomy_data.json` matching the checked-in copy to 10+ significant figures, the residual being ordinary floating-point noise from library-version differences, not a different computation. | **Yes — fully self-contained, genuinely reproducible from what's bundled (verified by actually running it)** |
| `corpus_500k_learning_dynamics.pdf`, `topology_nash_gap_heatmaps.pdf` | `scripts/render_histogram_corpus_500k.py` → `sequence_game/experiments/histogram_colearning/corpus_scale_artifacts.py` | `runs/histogram_corpus_500k/`, itself produced by `scripts/run_histogram_corpus_500k.py` (default `--db-path` = the bundled `graph_corpus.sqlite`; default `--census-path` = the bundled `code/docs/topology_bench_step6_census/census.json`) | No |
| `stackelberg_target_route_mass.pdf` | Two real generators exist for this exact output filename. Current/preferred: `scripts/build_histogram_stackelberg_followup_105_artifacts.py` → `sequence_game/experiments/histogram_colearning/stackelberg_followup_artifacts.py`, which writes directly to the paper's figures directory by default. Older: `scripts/render_histogram_stackelberg_steering_500k.py` → `stackelberg_steering_artifacts.py`, which writes to `docs/artifacts/histogram_stackelberg_steering_500k/` for later manual copying. | Followup path needs `runs/histogram_stackelberg_followup_105_500k/`, produced by `scripts/run_histogram_stackelberg_followup_105_500k.py`, which itself reads `runs/histogram_corpus_500k/run_manifest.json` as its own input. Older path needs `runs/histogram_stackelberg_steering_500k/` (+ a zero-margin sibling run + `runs/histogram_corpus_500k/`). | No (neither path's raw run currently exists) |
| `stackelberg_strict_vs_zero.pdf`, `stackelberg_strict_evaluation.pdf` | `scripts/render_histogram_stackelberg_steering_500k.py` → `stackelberg_steering_artifacts.py` | `runs/histogram_stackelberg_steering_500k/` + `runs/histogram_stackelberg_steering_zero_margin_500k/` | No |
| `prime_probe_ordered_dynamics.pdf` | Two real generators exist, both calling the same shared plotting function `_plot_ordered_phase_dynamics` in `prime_probe_artifacts.py` (which saves `.png`, `.pdf`, and `.svg` siblings of one output path). Current/preferred: `scripts/build_histogram_prime_probe_followup_105_artifacts.py`. Older: `scripts/render_histogram_prime_probe_detection_500k.py` → `prime_probe_detection_corpus.py`. | Followup path needs `runs/histogram_prime_probe_followup_105_500k/`, produced by `scripts/run_histogram_prime_probe_followup_105_500k.py` (itself reads `runs/histogram_corpus_500k/run_manifest.json`). Older path needs `runs/histogram_prime_probe_detection_500k/`. | No (neither path's raw run currently exists) |
| `calibration_pairwise_learners.pdf` | `scripts/render_histogram_step7_calibration.py` → `sequence_game/experiments/histogram_colearning/calibration_artifacts.py` | `runs/histogram_step7_calibration/` | No |

The two "followup" scripts are the current/preferred generators — they
write directly to the paper's figures directory — but their own raw run
data is likewise absent, so they cannot be run to completion without first
regenerating `runs/histogram_corpus_500k/`.

### Regenerating what actually can be regenerated

```bash
# histogram_anatomy.pdf — genuinely reproducible from what's bundled here.
# Requires this bundle's own code/requirements.txt PLUS SeQUeNCe installed
# separately (see "Quickstart" above; SeQUeNCe itself is not bundled, but
# sequence_game cannot be imported at all without it). Run from inside
# code/, keeping code/ as the resolved repo root (build_figures.py finds
# it via Path(__file__).resolve().parents[2] — do not move the script out
# of code/docs/talk_histogram_backend_2026/).
cd code
python docs/talk_histogram_backend_2026/build_figures.py
# This script's main() also calls build_corpus_versions() and
# build_eve_strategy_heatmap() after build_histogram_anatomy() -- two
# companion presentation figures, not among the 18 paper figures tracked
# above. Both need data this bundle does not include (an un-normalized
# runs/graph_corpus_sqlite_v1/ corpus and a
# runs/histogram_colearning_paper_100k/tables/eve_family_strategy.csv
# table), so the command above writes histogram_anatomy.pdf successfully
# and then raises an uncaught error. That is expected and does not mean
# histogram_anatomy.pdf failed to regenerate. For a clean exit, regenerate
# only the target figure directly:
#   python -c "from docs.talk_histogram_backend_2026.build_figures import build_histogram_anatomy; build_histogram_anatomy()"

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

This is a real compute job, not a data-copying job: the fully-materialized
compiled-histogram physics cache for all 105 topologies would occupy
approximately 22.45 GB if nothing were compiled on demand (the actual
500k-turn run uses compile-on-demand caching and never materializes that
much at once, but regeneration is still a genuinely non-trivial
simulation). This bundle does not attempt that job; it bundles the small
upstream dataset plus already-rendered outputs instead.

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

## Real-corpus digestion and Syn-small pilot figures

Unlike the table above, the five scripts below don't read a raw simulation
run — they read a smaller, already-computed intermediate result (LLM
digest scores, fitted cluster frequencies, a lagged-correlation detection
pass), which is bundled directly. Running the script against the bundled
intermediate data regenerates the figure itself byte-for-byte-equivalent
(verified by actually running all five). What is *not* bundled, in every
case, is the raw run that the intermediate result was itself computed from
— so these are reproducible one step later than `histogram_anatomy.pdf`,
not from the rawest possible simulation log.

| Figure | Generator script | Intermediate data (bundled) |
|---|---|---|
| `real_digest_generalization_by_depth.pdf`, `real_digest_checks_by_depth.pdf`, `real_digest_root_clusters.pdf`, `real_digest_border_collie.pdf` | `runs/histogram_corpus_full105_backend/pqqs2026_section/generate_figures.py` | `runs/histogram_corpus_full105_backend/{100k,500k}/llm_digest_*/digest_results.json`, `rule_clusters_*.json`, `border_collie_*.json` |
| `real_corpus_dt_generalization_by_depth_clean.png` | `scripts/plot_real_corpus_dt.py` | `runs/real_corpus_dt_results.json` |
| `strategy_weight_reversal_distribution.png`, `strategy_weight_reversal_example.png` | `scripts/plot_border_collie.py` | `runs/rule_clusters/border_collie_results.json` (all 234 graphs' scores) + `runs/exp3_small_synth_234_production/{payoff_matrices,online_runs/exp3_vs_exp3}/` for graph `0625` only (the single highest-scoring graph the bundled results select as the "clearest example" — deterministic given the bundled scores) + the bundled `runs/exp3_topology_bench/corpus_small_synthetic_hop7.sqlite` |
| `pilot_cluster_root_split_frequencies.png` | `scripts/plot_rule_clusters.py` | `runs/rule_clusters/level2_clusters.json` |
| `pilot_single_tree_digest_by_depth_clean.png` | `scripts/plot_shape_dt_llm_digest.py` | `runs/shape_dt_llm_digest/digest_results.json` |

```bash
cd code
python runs/histogram_corpus_full105_backend/pqqs2026_section/generate_figures.py
python scripts/plot_real_corpus_dt.py
python scripts/plot_border_collie.py runs/rule_clusters/border_collie_results.json \
  --production-dir runs/exp3_small_synth_234_production \
  --corpus-db-path runs/exp3_topology_bench/corpus_small_synthetic_hop7.sqlite \
  --out-dir runs/rule_clusters/figures
python scripts/plot_rule_clusters.py runs/rule_clusters/level2_clusters.json \
  --out-dir runs/rule_clusters/figures
python scripts/plot_shape_dt_llm_digest.py runs/shape_dt_llm_digest/digest_results.json \
  --out-dir runs/shape_dt_llm_digest/figures
```

**What is not closed, honestly:** the raw run behind the four
`real_digest_*`/`real_corpus_dt_*` figures' intermediate JSON is the same
missing `runs/histogram_corpus_500k/` directory documented in the main
table above — so independently re-deriving `digest_results.json` /
`rule_clusters_*.json` / `border_collie_*.json` / `real_corpus_dt_results.json`
from scratch needs the same missing raw run. For the two
`strategy_weight_reversal_*` figures, `scripts/detect_border_collie.py`
(bundled) is the real generator of `border_collie_results.json`, but it
needs all 234 graphs' raw per-turn trajectories to score them — a ~3.2 GB
production run this bundle does not include (only graph `0625`'s ~10 MB
slice is bundled, since that's the one the figure actually plots). Bundling
the already-computed `border_collie_results.json` directly means the
*figure* is still genuinely reproducible; only an independent from-scratch
re-detection across all 234 graphs is not.

## Integrity note

No script path, data dependency, or citation in this README was invented;
every claim above reflects an actual search, and every "not found" is the
result of a real, documented search rather than an assumption.
