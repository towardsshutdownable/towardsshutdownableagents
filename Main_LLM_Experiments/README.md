# TSA2 POST-to-Neutrality LLM Experiments

This folder is an anonymized code package for the newer LLM experiments in the submission. It is intended to be copied into the anonymous repository root alongside the existing `Deep_RL/` and legacy `LLM/` folders.

## What Is Included

- RLOO + DReST training code for Qwen3-8B and Llama-3.1-8B-Instruct.
- Direct option-logprob evaluation code used for the main reported LLM numbers.
- Generation-based evaluation code kept as a secondary path for qualitative checks.
- Dataset generation code for the deterministic POST training data and the scaled held-out evaluation suites.
- Exact JSON/JSONL datasets used by the main LLM experiments:
  - `data/deterministic_train_diverse_2048.jsonl`
  - `data/deterministic_test_scaled_main_20260428.jsonl`
  - `data/stochastic_main_test_scaled_main_20260428.jsonl`
  - `data/stochastic_dominance_control_scaled_main_20260428.jsonl`
  - `data/stochastic_zero_cost_control_scaled_main_20260428.jsonl`
  - `data/stochastic_joint_noshift_control_scaled_main_20260428.jsonl`
- A small `reported_results/` JSON file with the headline LLM table values from the submission.

## What Is Not Included

The package deliberately omits author-specific infrastructure and heavy artifacts: cloud launchers, private Hugging Face archive paths, W&B run wiring, raw logs, checkpoints, trained LoRA adapter weights, local cache state, notebooks with execution metadata, and `.git` history.

Those omissions keep the artifact anonymous and portable. The public code here is enough to regenerate datasets, train fresh LoRA adapters, and rerun the probability-based evaluation suite.

## Setup

Install PyTorch for your CUDA/runtime environment first, then install the remaining dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The main experiments used GPU training with 4-bit QLoRA. The evaluation scripts can score a baseline model or a trained LoRA adapter.

## Reproduce A Training Run

From this folder:

```bash
MODEL_NAME="Qwen/Qwen3-8B" SEED=20260325 bash scripts/train_drest_lora.sh
MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" SEED=20260429 bash scripts/train_drest_lora.sh
```

The default training settings match the paper: 2,048 deterministic POST prompts, meta-episode size 32, DReST lambda 0.9, one epoch, learning rate 5e-6, four generations per prompt, max prompt length 512, max completion length 16, and 4-bit QLoRA with LoRA rank 16.

## Evaluate

Evaluate a trained adapter:

```bash
MODEL_NAME="Qwen/Qwen3-8B" \
ADAPTER_PATH="runs/qwen_qwen3-8b_seed_20260325/final_adapter" \
RUN_LABEL="qwen_seed_20260325_posttrain" \
bash scripts/evaluate_logprob_suite.sh
```

Evaluate the baseline model by leaving `ADAPTER_PATH` empty:

```bash
MODEL_NAME="Qwen/Qwen3-8B" RUN_LABEL="qwen_baseline" bash scripts/evaluate_logprob_suite.sh
```

The evaluator writes one JSON report per dataset plus a suite-level report. It directly scores the labeled answer options rather than sampling free-form completions.

## Regenerate Datasets

The checked-in datasets are the exact artifact inputs used for the main LLM experiments. To regenerate them from code:

```bash
bash scripts/regenerate_main_datasets.sh
```

Regeneration is deterministic for the seeds hard-coded in the generator scripts.
