# Earlier LLM Experiments

This folder contains the earlier LLM experiments for the shutdownability/gridworld setup. The newer main-paper LLM experiments are in `../Main_LLM_Experiments`.

## Contents

- `wandb_main.py`: training entry point for the earlier Llama-3.2-3B-Instruct RLOO/DReST experiment.
- `meta_ep_rloo.py`: meta-episode RLOO trainer with DReST-style reward shaping.
- `evals.py`: small evaluation utilities used by the earlier experiments.
- `action_count_plots.py`: helper for plotting action frequencies over meta-episodes.
- `train_dataset_*` and `test_dataset_*`: JSON datasets used for these runs.
- `plots/action_frequencies/`: representative action-frequency plots from the earlier runs.
- `requirements.txt`: Python package versions from the local experiment environment.

## What Was Left Out

The public anonymous copy omits notebooks, local caches, private cloud-launch scripts, local machine paths, API-key helpers, and generated adapter/checkpoint weights. Those files are not needed to inspect the experiment logic and would make the anonymous repository less clean.

## Basic Usage

Install dependencies in a fresh Python environment, then run:

```bash
export HF_TOKEN=your_huggingface_token
export WANDB_PROJECT=earlier_llm_experiments
python wandb_main.py
```

`HF_TOKEN` is needed if the selected base model is gated. `WANDB_ENTITY` is optional; if it is not set, W&B uses the default account configured in the local environment.
