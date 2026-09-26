#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-8B}"
ADAPTER_PATH="${ADAPTER_PATH:-}"
RUN_LABEL="${RUN_LABEL:-eval}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/evals/${RUN_LABEL}}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ARGS=(
  --model_name "$MODEL_NAME"
  --output_dir "$OUTPUT_DIR"
  --suite_output_path "$OUTPUT_DIR/${RUN_LABEL}_suite_report.json"
  --run_label "$RUN_LABEL"
  --dataset_path data/deterministic_test_scaled_main_20260428.jsonl
  --dataset_path data/stochastic_main_test_scaled_main_20260428.jsonl
  --dataset_path data/stochastic_dominance_control_scaled_main_20260428.jsonl
  --dataset_path data/stochastic_zero_cost_control_scaled_main_20260428.jsonl
  --dataset_path data/stochastic_joint_noshift_control_scaled_main_20260428.jsonl
  --torch_dtype auto
)

if [ -n "$ADAPTER_PATH" ]; then
  ARGS+=(--adapter_path "$ADAPTER_PATH")
fi

if [ "$LOAD_IN_4BIT" = "1" ]; then
  ARGS+=(--load_in_4bit)
fi

"$PYTHON_BIN" evaluate_realistic_pap_logprob_suite.py "${ARGS[@]}"
