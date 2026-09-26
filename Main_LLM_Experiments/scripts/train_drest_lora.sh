#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
SEED="${SEED:-20260325}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_SLUG="$(printf '%s' "$MODEL_NAME" | tr '/:.' '---' | tr '[:upper:]' '[:lower:]')"
RUN_NAME="${RUN_NAME:-${MODEL_SLUG}_seed_${SEED}}"

"$PYTHON_BIN" train_realistic_pap_rloo.py \
  --model_name "$MODEL_NAME" \
  --train_path data/deterministic_train_diverse_2048.jsonl \
  --output_root runs \
  --run_name "$RUN_NAME" \
  --max_train_samples 2048 \
  --seed "$SEED" \
  --meta_ep_size 32 \
  --lambda_factor 0.9 \
  --num_train_epochs 1 \
  --learning_rate 5e-6 \
  --num_generations 4 \
  --max_prompt_length 512 \
  --max_completion_length 16 \
  --save_steps 20 \
  --logging_steps 1 \
  --temperature 1.0 \
  --top_p 1.0 \
  --torch_dtype auto
