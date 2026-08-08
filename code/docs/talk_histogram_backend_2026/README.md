# histogram_anatomy.pdf generator

This directory contains the generator for `histogram_anatomy.pdf` (the
compiled transcript histogram / sampled endpoint record figure used in the
paper). It reads the bundled `graph_corpus.sqlite` and evaluates live
through `HistogramPhysicsPool` with a fixed seed — no cached run is read.

Regenerate:

```bash
env MPLCONFIGDIR=/tmp/mplconfig-histogram-talk \
  .venv/bin/python docs/talk_histogram_backend_2026/build_figures.py
```

Scientific scope: Alice's network model is an empirical-fit near-term
composite; Eve's attacks are explicit modeled quantum operations with
declared scenario strengths and control costs; no authenticated finite-key
security proof is claimed.
