#!/usr/bin/env bash
# Run the whole offline pipeline in order. Each step can also be run on its own.
#   ./run_pipeline.sh            # data -> indexes -> retrieval experiments -> figures -> web assets
#   ./run_pipeline.sh --with-llm # also the generation experiments (needs GROQ_API_KEY in .env)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-.venv/bin/python}

steps=(
  01_acquire 02_profile 03_clean 04_skills 05_chunk_embed 06_index 07_onet_link
  08_eval_retrieval 09_eda
)
[[ "${1:-}" == "--with-llm" ]] && steps+=(10_eval_generation)
steps+=(14_report_figures 13_export_web)

for s in "${steps[@]}"; do
  echo "=== $s ==="
  "$PY" "pipeline/$s.py"
done
echo "Done. Next: python pipeline/11_load_database.py && python pipeline/12_deploy_supabase.py"
