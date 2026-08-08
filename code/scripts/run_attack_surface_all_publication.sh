#!/usr/bin/env bash
set -euo pipefail

# Contracted publication run:
# - no small suite / no binary collision rerun
# - six attack_surface_all corpus conditions
# - main DQN conditions on the full selected manifest
# - four reference controls on ~34 unique graphs per synthetic family
# - unbounded corpus routes are explicitly enabled
# - dynamic DT refit telemetry and resumable LLM summaries
#
# Windows PowerShell one-liner, fresh/resume aware:
# .\.venv\Scripts\python.exe scripts\paper_results.py $(if (Test-Path 'results\paper\synthetic_full_attack_surface_all_3k_dynamic_llm') {'--resume-run-id'} else {'--run-id'}) synthetic_full_attack_surface_all_3k_dynamic_llm --no-suite --no-llm --corpus-manifest runs\graph_corpus_paper_v2\corpus_manifest.json --corpus-combo-keys 'attack_surface_all__random__random,attack_surface_all__deep_q__random,attack_surface_all__route_aware_oracle__cautious_greedy,attack_surface_all__random__cautious_greedy,attack_surface_all__deep_q__cautious_greedy,attack_surface_all__deep_q__deep_q' --corpus-max-episodes 3000 --corpus-reference-max-graphs 34 --corpus-seed 42 --corpus-checkpoint-every 1 --workers 3 --corpus-dt-refit-mode dynamic --corpus-dt-observer-interval 50 --corpus-dt-epsilon 0.10 --corpus-dt-delta 0.05 --corpus-dt-policy-drift-threshold 0.05 --corpus-dt-max-refit-gap 2000 --corpus-dt-log-snapshot-count 8 --corpus-dt-probe-state-count 2048 --no-mitigation

: "${RUN_ID:=synthetic_full_attack_surface_all_3k_dynamic_llm}"
: "${MANIFEST:=runs/graph_corpus_paper_v2/corpus_manifest.json}"
: "${STEPS:=3000}"
: "${WORKERS:=6}"
: "${REFERENCE_MAX_GRAPHS:=34}"
: "${LLM:=phi4:14b}"
: "${COMPARE_LLMS:=phi4:14b,llama3.1:8b,qwen3:14b,qwen3.6:27b,qwq:32b,deepseek-r1:70b}"
: "${OLLAMA_HOST:=http://localhost:11434}"

export OLLAMA_HOST
if [[ "${OLLAMA_USE_CLOUDFLARE:-0}" != "1" ]]; then
  unset CF_CLIENT_ID
  unset CF_CLIENT_SECRET
fi

run_selector=(--run-id "$RUN_ID")
if [[ -d "results/paper/$RUN_ID" ]]; then
  run_selector=(--resume-run-id "$RUN_ID")
fi

combo_keys=$(
  IFS=,
  printf '%s' \
    "attack_surface_all__random__random,"\
    "attack_surface_all__deep_q__random,"\
    "attack_surface_all__route_aware_oracle__cautious_greedy,"\
    "attack_surface_all__random__cautious_greedy,"\
    "attack_surface_all__deep_q__cautious_greedy,"\
    "attack_surface_all__deep_q__deep_q"
)

cmd=(
  .venv/bin/python scripts/paper_results.py
  "${run_selector[@]}"
  --no-suite
  --corpus-manifest "$MANIFEST"
  --corpus-combo-keys "$combo_keys"
  --corpus-max-episodes "$STEPS"
  --corpus-reference-max-graphs "$REFERENCE_MAX_GRAPHS"
  --corpus-seed 42
  --corpus-checkpoint-every 1
  --workers "$WORKERS"
  --corpus-dt-refit-mode dynamic
  --corpus-dt-observer-interval 50
  --corpus-dt-epsilon 0.10
  --corpus-dt-delta 0.05
  --corpus-dt-policy-drift-threshold 0.05
  --corpus-dt-max-refit-gap 2000
  --corpus-dt-log-snapshot-count 8
  --corpus-dt-probe-state-count 2048
  --llm "$LLM"
  --compare-llms "$COMPARE_LLMS"
  --no-mitigation
)

if [[ -n "${CORPUS_MAX_GRAPHS:-}" ]]; then
  cmd+=(--corpus-max-graphs "$CORPUS_MAX_GRAPHS")
fi

"${cmd[@]}"
