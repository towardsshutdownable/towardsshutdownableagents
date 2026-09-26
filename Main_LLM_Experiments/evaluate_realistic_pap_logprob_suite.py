from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

from evaluate_realistic_pap_logprobs import (
    build_row_result,
    candidate_logprob,
    load_model,
    normalize_logprobs,
    single_token_option_logprobs,
    summarize_deterministic,
    summarize_stochastic,
)
from pap_dataset_utils import load_jsonl, make_conversational_prompt, maybe_sample_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--dataset_path", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--suite_output_path", default="")
    parser.add_argument("--run_label", default="eval")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--torch_dtype", default="auto")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def summarize_option_labels(results: list[dict]) -> dict:
    if not results:
        return {}

    no_shift_probs = [row.get("no_shift_probability", row["option_probabilities"].get("a", 0.0)) for row in results]
    shift_probs = [row.get("shift_probability", row["option_probabilities"].get("b", 0.0)) for row in results]
    gains = [row.get("expected_score_gain_from_shift") for row in results]
    real_gains = [gain for gain in gains if isinstance(gain, int | float)]
    direction_counts = Counter()
    for gain in real_gains:
        if gain > 1e-9:
            direction_counts["increase"] += 1
        elif gain < -1e-9:
            direction_counts["decrease"] += 1
        else:
            direction_counts["unchanged"] += 1

    no_shift_more = sum(1 for no_shift_prob, shift_prob in zip(no_shift_probs, shift_probs, strict=True) if no_shift_prob > shift_prob)
    shift_more = sum(1 for no_shift_prob, shift_prob in zip(no_shift_probs, shift_probs, strict=True) if shift_prob > no_shift_prob)
    ties = len(results) - no_shift_more - shift_more

    return {
        "count": len(results),
        "mean_allow_shutdown_probability": sum(no_shift_probs) / len(results),
        "mean_influence_shutdown_probability": sum(shift_probs) / len(results),
        "share_allow_shutdown_more_probable": no_shift_more / len(results),
        "share_influence_shutdown_more_probable": shift_more / len(results),
        "share_tied": ties / len(results),
        "expected_score_gain_mean": sum(real_gains) / len(real_gains) if real_gains else None,
        "expected_score_gain_min": min(real_gains) if real_gains else None,
        "expected_score_gain_max": max(real_gains) if real_gains else None,
        "expected_score_direction_counts": dict(direction_counts),
    }


def score_rows(
    *,
    model,
    tokenizer,
    rows: list[dict],
    enable_thinking: bool,
) -> tuple[list[dict], dict, dict]:
    results = []
    method_counter = Counter()
    total_prompt_tokens = 0
    total_candidate_tokens = 0
    total_scoring_seconds = 0.0

    for row in rows:
        row_started = time.perf_counter()
        prompt_messages = make_conversational_prompt(row, include_short_reasoning=False)
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        prompt_tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        total_prompt_tokens += int(prompt_tokens.shape[1])

        valid_answers = sorted(row["valid_answers"])
        single_token_result = single_token_option_logprobs(model, tokenizer, prompt_text, valid_answers)
        if single_token_result is not None:
            answer_logprobs, candidate_lengths = single_token_result
            method = "single_token_next_token"
        else:
            answer_logprobs = {}
            candidate_lengths = {}
            for answer in valid_answers:
                answer_logprob, candidate_len = candidate_logprob(model, tokenizer, prompt_text, answer)
                answer_logprobs[answer] = answer_logprob
                candidate_lengths[answer] = candidate_len
            method = "teacher_forced_sequence"

        method_counter[method] += 1
        total_candidate_tokens += sum(candidate_lengths.values())

        option_probs = normalize_logprobs(answer_logprobs)
        result = build_row_result(row, option_probs, answer_logprobs, candidate_lengths)
        result["prompt_token_count"] = int(prompt_tokens.shape[1])
        result["scoring_method"] = method
        result["scoring_seconds"] = time.perf_counter() - row_started
        results.append(result)
        total_scoring_seconds += result["scoring_seconds"]

    timing = {
        "total_scoring_seconds": total_scoring_seconds,
        "mean_scoring_seconds": total_scoring_seconds / len(results) if results else 0.0,
    }
    tokens = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_candidate_tokens": total_candidate_tokens,
        "mean_prompt_tokens": total_prompt_tokens / len(results) if results else 0.0,
        "mean_candidate_tokens_per_prompt": total_candidate_tokens / len(results) if results else 0.0,
    }
    return results, {"scoring_method_counts": dict(method_counter), "timing": timing, "tokens": tokens}, {
        "method_counter": method_counter,
        "total_prompt_tokens": total_prompt_tokens,
        "total_candidate_tokens": total_candidate_tokens,
        "total_scoring_seconds": total_scoring_seconds,
    }


def main() -> None:
    args = parse_args()
    started_utc = utc_now_iso()
    started_perf = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_output_path = Path(args.suite_output_path) if args.suite_output_path else output_dir / f"{args.run_label}_suite_report.json"

    model, tokenizer = load_model(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
        torch_dtype_name=args.torch_dtype,
    )

    dataset_reports = {}
    aggregate_method_counter = Counter()
    aggregate_prompt_tokens = 0
    aggregate_candidate_tokens = 0
    aggregate_scoring_seconds = 0.0
    aggregate_count = 0

    for dataset_path_string in args.dataset_path:
        dataset_path = Path(dataset_path_string)
        dataset_name = dataset_path.stem
        dataset_started_utc = utc_now_iso()
        dataset_started_perf = time.perf_counter()
        rows = load_jsonl(dataset_path)
        rows = maybe_sample_rows(rows, max_samples=args.max_samples if args.max_samples > 0 else None, seed=args.seed)
        results, metrics, aggregate = score_rows(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            enable_thinking=args.enable_thinking,
        )
        prompt_type = rows[0]["prompt_type"] if rows else ""
        if prompt_type == "deterministic":
            summary = summarize_deterministic(results)
        else:
            summary = summarize_stochastic(results)

        report = {
            "evaluation_config": {
                "model_name": args.model_name,
                "adapter_path": args.adapter_path or None,
                "dataset_path": str(dataset_path),
                "dataset_name": dataset_name,
                "output_dir": str(output_dir),
                "max_samples": args.max_samples,
                "seed": args.seed,
                "enable_thinking": args.enable_thinking,
                "load_in_4bit": args.load_in_4bit,
                "torch_dtype": args.torch_dtype,
                "num_prompts": len(results),
                "prompt_type": prompt_type,
                "mode": "direct_option_logprob_suite",
                "run_label": args.run_label,
            },
            "summary": summary,
            "option_label_summary": summarize_option_labels(results) if prompt_type != "deterministic" else {},
            "counts": {
                "num_total": len(results),
            },
            "scoring_method_counts": metrics["scoring_method_counts"],
            "timing": {
                "started_at_utc": dataset_started_utc,
                "finished_at_utc": utc_now_iso(),
                "total_runtime_seconds": time.perf_counter() - dataset_started_perf,
                **metrics["timing"],
            },
            "tokens": metrics["tokens"],
            "results": results,
        }
        report_path = output_dir / f"{args.run_label}_{dataset_name}_logprobs.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        dataset_reports[dataset_name] = {
            "report_path": str(report_path),
            "prompt_type": prompt_type,
            "summary": summary,
            "option_label_summary": report["option_label_summary"],
            "num_prompts": len(results),
        }
        aggregate_method_counter.update(aggregate["method_counter"])
        aggregate_prompt_tokens += aggregate["total_prompt_tokens"]
        aggregate_candidate_tokens += aggregate["total_candidate_tokens"]
        aggregate_scoring_seconds += aggregate["total_scoring_seconds"]
        aggregate_count += len(results)

    # Help the CUDA allocator release memory before long-run outputs are saved.
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    suite_report = {
        "suite_config": {
            "model_name": args.model_name,
            "adapter_path": args.adapter_path or None,
            "dataset_paths": args.dataset_path,
            "output_dir": str(output_dir),
            "suite_output_path": str(suite_output_path),
            "run_label": args.run_label,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "enable_thinking": args.enable_thinking,
            "load_in_4bit": args.load_in_4bit,
            "torch_dtype": args.torch_dtype,
            "mode": "direct_option_logprob_suite",
        },
        "timing": {
            "started_at_utc": started_utc,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": time.perf_counter() - started_perf,
            "total_scoring_seconds": aggregate_scoring_seconds,
            "mean_scoring_seconds": aggregate_scoring_seconds / aggregate_count if aggregate_count else 0.0,
        },
        "tokens": {
            "total_prompt_tokens": aggregate_prompt_tokens,
            "total_candidate_tokens": aggregate_candidate_tokens,
            "mean_prompt_tokens": aggregate_prompt_tokens / aggregate_count if aggregate_count else 0.0,
            "mean_candidate_tokens_per_prompt": aggregate_candidate_tokens / aggregate_count if aggregate_count else 0.0,
        },
        "counts": {
            "num_total": aggregate_count,
        },
        "scoring_method_counts": dict(aggregate_method_counter),
        "datasets": dataset_reports,
    }
    suite_output_path.parent.mkdir(parents=True, exist_ok=True)
    suite_output_path.write_text(json.dumps(suite_report, indent=2), encoding="utf-8")
    print(json.dumps(suite_report["datasets"], indent=2))


if __name__ == "__main__":
    main()
