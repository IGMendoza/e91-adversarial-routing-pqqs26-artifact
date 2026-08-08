# Artifact bundle: Adversarial Route Learning in Quantum Repeater Networks (PQQS'26)

This bundle accompanies the paper "Adversarial Route Learning in Quantum
Repeater Networks" (PQQS'26 submission). The paper's LaTeX source is not
included here — reviewers receive the PDF directly — so this bundle covers
only the code, data, and generator scripts behind its figures.

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
cached_intermediate_results/docs_artifacts/   Rendered PDF/PNG figures plus the CSV/JSON numeric
                               tables they were plotted from, for every histogram-corpus,
                               Stackelberg, prime-probe, and calibration experiment family
                               that has a script
figures_as_used_in_paper/     All 18 real content figures + the EU logo, exactly as they
                               appear in the paper
```

Bundle size: ~59 MB (`du -sh --apparent-size`).

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
  archive from the same Zenodo release. This is the raw source material
  behind the paper's historical "Syn-small" pilot study, but it has not
  been extracted or ingested by any script in this bundle (see "Figures
  with no generator found," Syn-small pilot row, below).
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
| `histogram_anatomy.pdf` | `docs/talk_histogram_backend_2026/build_figures.py`, function `build_histogram_anatomy()` | reads the bundled `graph_corpus.sqlite` directly, then evaluates live through `HistogramPhysicsPool` with fixed seed 142 — no separately cached run is read | **Yes — fully self-contained, genuinely reproducible from what's bundled** |
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
# Requires the same third-party dependencies as the source project (matplotlib,
# numpy, networkx, etc — not duplicated in this bundle). Run from inside code/,
# keeping code/ as the resolved repo root (build_figures.py finds it via
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

## Figures with no generator found

These figures appear in the paper as final rendered files (bundled under
`figures_as_used_in_paper/` for reference), but no generator script and no
cached intermediate output for them exists in this bundle or was found
elsewhere.

- **`real_digest_border_collie.pdf`, `real_digest_generalization_by_depth.pdf`,
  `real_digest_root_clusters.pdf`, `real_digest_checks_by_depth.pdf`**
  (real-corpus decision-tree digestion figures) and
  **`real_corpus_dt_generalization_by_depth_clean.png`**
  (current-corpus decision-tree prediction figure): the underlying adapter
  export behind these figures' targets (`histogram_final_retention`,
  `exploitability`) was never archived. Its source was not found under any
  name (`adapter`, `digest`, `real_corpus`, or the two target variable
  names above) anywhere searched.
- **`pilot_single_tree_digest_by_depth_clean.png`,
  `pilot_cluster_root_split_frequencies.png`,
  `strategy_weight_reversal_distribution.png`,
  `strategy_weight_reversal_example.png`** (Syn-small historical-pilot
  figures): the machine-readable sidecars behind these figures (the
  234-graph Syn-small manifest, feature/fold tables, fitted rules,
  subsampling seeds, verifier records, or the exact LLM prompts/responses)
  do not exist in this bundle or were found elsewhere. Only the raw,
  un-extracted `data/topology_bench/raw/small_synthetic.zip` exists as a
  data dependency; nothing ingests it here.

Treat these nine files as an open provenance question, not as evidence a
generator never existed — only that a thorough search did not find one.
