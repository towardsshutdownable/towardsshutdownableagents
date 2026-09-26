#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" generate_datasets.py
"$PYTHON_BIN" generate_diverse_training_data.py
"$PYTHON_BIN" make_scaled_main_eval_datasets.py
