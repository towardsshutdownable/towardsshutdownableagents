#!/usr/bin/env bash
# The LLM training runs and evaluations reported in the paper, one H100 (80 GB) at a time.
# Each training run takes about 4 hours (gpt-oss-20b) to 8.5 hours (Gemma 4 12B). Each test-set evaluation takes 4 to 13 minutes.
#
#   bash scripts/run_paper_experiments.sh qwen        # or gemma, granite, gptoss, ablations
#
# Before running: install requirements.txt, then `python patches/apply_patches.py` (and `--entropy-bonus` for the ablation).
# Afterwards: scripts/check_training_health.py on each train.log, scripts/compact_results.py to produce results/ files,
# then scripts/reproduce_paper_numbers.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false

FAMILY="${1:?usage: run_paper_experiments.sh qwen|gemma|granite|gptoss|ablations}"
case "$FAMILY" in
  qwen)    MODEL=Qwen/Qwen3-14B;            TRAIN_EXTRA=();  EVAL_EXTRA=() ;;
  gemma)   MODEL=google/gemma-4-12B-it;     TRAIN_EXTRA=();  EVAL_EXTRA=() ;;
  granite) MODEL=ibm-granite/granite-4.2-8b; TRAIN_EXTRA=(); EVAL_EXTRA=() ;;
  gptoss)  MODEL=openai/gpt-oss-20b
           # gpt-oss-20b: lowest reasoning level, a fixed date in its system message, smaller micro-batches and sampler share
           # to fit in memory, and only the attention weights (which the adapter changes) copied to the sampler.
           TRAIN_EXTRA=(--chat_template_kwargs '{"reasoning_effort":"low"}' --pin_chat_date 2026-09-28
                        --micro_batch_size 4 --vllm_gpu_memory_utilization 0.25 --sync_only_adapted)
           EVAL_EXTRA=(--chat_kwargs '{"reasoning_effort":"low"}' --pin_chat_date 2026-09-28) ;;
  ablations) MODEL=Qwen/Qwen3-14B ;;
  *) echo "unknown family $FAMILY"; exit 1 ;;
esac

evaluate () {  # <tag> <adapter or -> <data set> <answers per displayed prompt> [extra arguments]
  local tag=$1 adapter=$2 data=$3 n=$4; shift 4
  local adapter_args=(); [ "$adapter" != "-" ] && adapter_args=(--adapter "$adapter")
  python scripts/evaluate.py --model "$MODEL" "${adapter_args[@]}" --data_dir "data/$data" --out_dir "runs/eval/${tag}_${data}" \
    --n_samples "$n" "${EVAL_EXTRA[@]}" "$@"
}

train_and_evaluate () {  # <reward mode> <seed> <run name> [extra training arguments]
  local mode=$1 seed=$2 run=$3; shift 3
  mkdir -p "runs/$run"
  python scripts/train.py --model_name "$MODEL" --reward_mode "$mode" --seed "$seed" --run_name "$run" "${TRAIN_EXTRA[@]}" "$@" \
    > "runs/$run/train.log" 2>&1
  python scripts/check_training_health.py "runs/$run/train.log"
  evaluate "$run" "runs/$run/final_adapter" test 4
  evaluate "$run" "runs/$run/final_adapter" test_low_gain 4
}

if [ "$FAMILY" = ablations ]; then
  # Appendix G. Higher sampling temperature for the untrained model, and default reward plus an entropy bonus
  # (c = 0.003, chosen by the rule in the paper from pilot runs at 0.0003, 0.001, 0.003, and 0.01).
  TRAIN_EXTRA=(); EVAL_EXTRA=()
  for t in 1.3 1.6 2.0; do
    evaluate "qwen_untrained_temperature$t" - test 4 --temperature "$t"
    evaluate "qwen_untrained_temperature$t" - test_low_gain 4 --temperature "$t"
  done
  for seed in 1 2 3; do
    ENTROPY_BONUS_COEF=0.003 train_and_evaluate default_task_score "$seed" "qwen_entropy_bonus_s$seed"
  done
  exit 0
fi

# Untrained model.
evaluate "${FAMILY}_untrained" - test 4
evaluate "${FAMILY}_untrained" - test_low_gain 4
evaluate "${FAMILY}_untrained" - validation 8

# Learning-rate search (DReST, seed 1, validation set with eight answers per displayed prompt). The rule, fixed before
# the search: discard a rate whose validation dominance rate is below 80%, joint-no-shift rate above 20%, POST
# USEFULNESS below 0.9, or share of unreadable answers above 5%. Among the rest, take the highest within-order POST
# mixing, and the lowest rate whose 95% interval overlaps the best one's. It chose 2e-5 for every LLM.
for lr in 5e-5 1e-4; do
  run="${FAMILY}_lr_search_${lr}_drest_s1"
  mkdir -p "runs/$run"
  python scripts/train.py --model_name "$MODEL" --reward_mode drest_order_averaged --seed 1 --run_name "$run" \
    --learning_rate "$lr" "${TRAIN_EXTRA[@]}" > "runs/$run/train.log" 2>&1 || true
  python scripts/check_training_health.py "runs/$run/train.log" && evaluate "$run" "runs/$run/final_adapter" validation 8
done

# Five seeds each of DReST and the default reward, at 2e-5 (the DReST seed 1 run doubles as the 2e-5 point of the search).
for seed in 1 2 3 4 5; do
  train_and_evaluate drest_order_averaged "$seed" "${FAMILY}_drest_s$seed"
  train_and_evaluate default_task_score "$seed" "${FAMILY}_default_s$seed"
done
evaluate "${FAMILY}_drest_s1" "runs/${FAMILY}_drest_s1/final_adapter" validation 8
