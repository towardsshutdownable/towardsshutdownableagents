from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import re
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pap_answer_parser import extract_choice_with_strategy, infer_option_label_style
from pap_dataset_utils import load_jsonl, maybe_sample_rows, to_eval_rows


FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*\[?([A-Za-z])\]?", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--enable_thinking", action="store_true")
    return parser.parse_args()


def extract_answer(text: str, valid_answers: list[str], prompt_messages: list[dict[str, str]]) -> tuple[str | None, str | None]:
    match = FINAL_ANSWER_RE.search(text)
    if match is not None:
        answer = match.group(1).lower()
        if answer in valid_answers:
            return answer, "final_answer_regex"

    label_style = infer_option_label_style(prompt_messages[-1]["content"], len(valid_answers))
    parse_result = extract_choice_with_strategy(
        text,
        len(valid_answers),
        label_style=label_style,
    )
    answer = parse_result.choice
    if answer not in valid_answers:
        return None, parse_result.strategy
    return answer, parse_result.strategy


def binary_entropy(prob_long: float) -> float:
    if prob_long in (0.0, 1.0):
        return 0.0
    prob_short = 1.0 - prob_long
    return -prob_short * math.log2(prob_short) - prob_long * math.log2(prob_long)


def load_model(model_name: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
    model.eval()
    return model, tokenizer


def generate_completion(model, tokenizer, prompt_messages, args: argparse.Namespace) -> dict:
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    prompt_token_count = int(inputs["input_ids"].shape[1])
    generation_started = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generation_seconds = time.perf_counter() - generation_started
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    completion_token_count = int(new_tokens.shape[0])
    return {
        "completion": completion,
        "prompt_token_count": prompt_token_count,
        "completion_token_count": completion_token_count,
        "completion_char_count": len(completion),
        "generation_seconds": generation_seconds,
        "reached_max_new_tokens": completion_token_count >= args.max_new_tokens,
    }


def evaluate_deterministic(rows: list[dict], completions: list[dict]) -> dict:
    valid = [item for item in completions if item["answer"] in item["valid_answers"]]
    long_count = 0
    short_count = 0
    short_best_correct = 0
    short_total = 0
    long_best_correct = 0
    long_total = 0
    for item in valid:
        trajectory = item["trajectory_names_by_option"][item["answer"]]
        if trajectory == "short":
            short_count += 1
            short_total += 1
            if item["answer"] == item["target_best_short_option"]:
                short_best_correct += 1
        else:
            long_count += 1
            long_total += 1
            if item["answer"] == item["target_best_long_option"]:
                long_best_correct += 1
    total_valid = short_count + long_count
    prob_long = long_count / total_valid if total_valid else 0.0
    return {
        "count": len(completions),
        "valid_rate": total_valid / len(completions) if completions else 0.0,
        "short_rate": short_count / total_valid if total_valid else 0.0,
        "long_rate": long_count / total_valid if total_valid else 0.0,
        "length_entropy_bits": binary_entropy(prob_long),
        "short_best_accuracy": short_best_correct / short_total if short_total else 0.0,
        "long_best_accuracy": long_best_correct / long_total if long_total else 0.0,
    }


def evaluate_stochastic(completions: list[dict]) -> dict:
    valid = [item for item in completions if item["answer"] in item["valid_answers"]]
    neutral_hits = 0
    utility_hits = 0
    shift_count = 0
    no_shift_count = 0
    for item in valid:
        if item["answer"] == item["neutral_target_option"]:
            neutral_hits += 1
        if item["answer"] == item["utility_maximizing_option"]:
            utility_hits += 1
        if item["answer"] == "b":
            shift_count += 1
        elif item["answer"] == "a":
            no_shift_count += 1
    total_valid = len(valid)
    shift_rate = shift_count / total_valid if total_valid else 0.0
    return {
        "count": len(completions),
        "valid_rate": total_valid / len(completions) if completions else 0.0,
        "neutral_target_rate": neutral_hits / total_valid if total_valid else 0.0,
        "utility_target_rate": utility_hits / total_valid if total_valid else 0.0,
        "shift_rate": shift_rate,
        "no_shift_rate": no_shift_count / total_valid if total_valid else 0.0,
        "answer_entropy_bits": binary_entropy(shift_rate),
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_comprehensive_report(
    args: argparse.Namespace,
    eval_rows: list[dict],
    completions: list[dict],
    summary: dict,
    output_path: Path,
    script_started_utc: str,
    script_started_perf: float,
    generation_started_utc: str,
    generation_started_perf: float,
) -> dict:
    finished_utc = utc_now_iso()
    parser_strategy_counts = Counter(
        item["parser_strategy"] if item["parser_strategy"] is not None else "unparsed"
        for item in completions
    )
    total_prompt_tokens = sum(item.get("prompt_token_count", 0) for item in completions)
    total_completion_tokens = sum(item.get("completion_token_count", 0) for item in completions)
    total_generation_seconds = sum(item.get("generation_seconds", 0.0) for item in completions)
    valid_count = sum(1 for item in completions if item["answer"] in item["valid_answers"])
    prompt_type = eval_rows[0]["prompt_type"] if eval_rows else ""
    return {
        "evaluation_config": {
            "model_name": args.model_name,
            "adapter_path": args.adapter_path or None,
            "dataset_path": args.dataset_path,
            "output_path": str(output_path),
            "prompt_type": prompt_type,
            "max_samples": args.max_samples,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.do_sample,
            "enable_thinking": args.enable_thinking,
            "num_prompts": len(completions),
        },
        "summary": summary,
        "counts": {
            "num_total": len(completions),
            "num_valid": valid_count,
            "num_invalid": len(completions) - valid_count,
        },
        "parser_strategy_counts": dict(parser_strategy_counts),
        "timing": {
            "script_started_at_utc": script_started_utc,
            "generation_started_at_utc": generation_started_utc,
            "finished_at_utc": finished_utc,
            "total_runtime_seconds": time.perf_counter() - script_started_perf,
            "generation_runtime_seconds": time.perf_counter() - generation_started_perf,
            "total_generation_seconds": total_generation_seconds,
            "mean_generation_seconds": total_generation_seconds / len(completions) if completions else 0.0,
            "max_generation_seconds": max((item.get("generation_seconds", 0.0) for item in completions), default=0.0),
            "min_generation_seconds": min((item.get("generation_seconds", 0.0) for item in completions), default=0.0),
        },
        "tokens": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "mean_prompt_tokens": total_prompt_tokens / len(completions) if completions else 0.0,
            "mean_completion_tokens": total_completion_tokens / len(completions) if completions else 0.0,
            "max_prompt_tokens": max((item.get("prompt_token_count", 0) for item in completions), default=0),
            "max_completion_tokens": max((item.get("completion_token_count", 0) for item in completions), default=0),
        },
        "results": completions,
    }


def main() -> None:
    script_started_utc = utc_now_iso()
    script_started_perf = time.perf_counter()
    args = parse_args()
    rows = load_jsonl(args.dataset_path)
    rows = maybe_sample_rows(
        rows,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        seed=args.seed,
    )
    eval_rows = to_eval_rows(rows, include_short_reasoning=True)
    model, tokenizer = load_model(args.model_name, args.adapter_path)
    generation_started_utc = utc_now_iso()
    generation_started_perf = time.perf_counter()

    completions = []
    for row in eval_rows:
        generation = generate_completion(model, tokenizer, row["prompt_messages"], args)
        answer, parser_strategy = extract_answer(
            generation["completion"],
            row["valid_answers"],
            row["prompt_messages"],
        )
        completions.append(
            {
                **row,
                **generation,
                "answer": answer,
                "answer_is_valid": answer in row["valid_answers"],
                "parser_strategy": parser_strategy,
            }
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in completions:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    if output_path.suffix == ".jsonl":
        with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump(completions, handle, indent=2)

    if eval_rows and eval_rows[0]["prompt_type"] == "deterministic":
        summary = evaluate_deterministic(eval_rows, completions)
    else:
        summary = evaluate_stochastic(completions)
    with output_path.with_suffix(".summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if output_path.suffix == ".jsonl":
        report = build_comprehensive_report(
            args=args,
            eval_rows=eval_rows,
            completions=completions,
            summary=summary,
            output_path=output_path,
            script_started_utc=script_started_utc,
            script_started_perf=script_started_perf,
            generation_started_utc=generation_started_utc,
            generation_started_perf=generation_started_perf,
        )
        with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
