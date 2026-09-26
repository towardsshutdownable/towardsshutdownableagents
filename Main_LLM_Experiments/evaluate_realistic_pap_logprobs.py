from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from pap_dataset_utils import load_jsonl, make_conversational_prompt, maybe_sample_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--torch_dtype", default="auto")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def binary_entropy(prob_one: float) -> float:
    if prob_one in (0.0, 1.0):
        return 0.0
    prob_zero = 1.0 - prob_one
    return -prob_zero * math.log2(prob_zero) - prob_one * math.log2(prob_one)


def resolve_dtype(dtype_name: str, model_name: str = "") -> torch.dtype:
    if dtype_name.lower() == "auto":
        # Gemma 3 produced NaN option probabilities under fp16 on this stack.
        return torch.bfloat16 if "gemma-3" in model_name.lower() else torch.float16
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[key]


def patch_gemma3_token_type_ids(model):
    config = getattr(model, "config", None)
    if getattr(config, "model_type", "") != "gemma3":
        return model
    if getattr(model, "_pap_gemma3_token_type_ids_patch", False):
        return model

    original_forward = model.forward

    def squeeze_extra_singleton_axis(value):
        if isinstance(value, torch.Tensor) and value.ndim == 3 and value.shape[1] == 1:
            return value.squeeze(1)
        return value

    def forward_with_text_token_type_ids(*args, **kwargs):
        args = tuple(squeeze_extra_singleton_axis(value) for value in args)
        for key in ("input_ids", "attention_mask", "token_type_ids", "position_ids"):
            if key in kwargs:
                kwargs[key] = squeeze_extra_singleton_axis(kwargs[key])
        if kwargs.get("token_type_ids") is None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is not None:
                kwargs["token_type_ids"] = torch.zeros_like(input_ids)
        output = original_forward(*args, **kwargs)
        logits = getattr(output, "logits", None)
        if isinstance(logits, torch.Tensor) and logits.ndim == 4 and logits.shape[1] == 1:
            output.logits = logits.squeeze(1)
        return output

    model.forward = forward_with_text_token_type_ids
    model._pap_gemma3_token_type_ids_patch = True
    return model


def load_model(model_name: str, adapter_path: str, load_in_4bit: bool, torch_dtype_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
    }

    torch_dtype = resolve_dtype(torch_dtype_name, model_name=model_name)
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["torch_dtype"] = torch_dtype
    else:
        model_kwargs["torch_dtype"] = torch_dtype

    base_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    patch_gemma3_token_type_ids(base_model)
    if adapter_path:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
    patch_gemma3_token_type_ids(model)
    model.eval()
    return model, tokenizer


def candidate_logprob(
    model,
    tokenizer,
    prompt_text: str,
    candidate: str,
) -> tuple[float, int]:
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    candidate_ids = tokenizer(candidate, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    full_ids = torch.cat([prompt_ids, candidate_ids], dim=1)
    attention_mask = torch.ones_like(full_ids)
    prompt_len = int(prompt_ids.shape[1])
    candidate_len = int(candidate_ids.shape[1])

    with torch.no_grad():
        outputs = model(input_ids=full_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, prompt_len - 1 : -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        target_ids = full_ids[:, prompt_len:]
        token_log_probs = torch.gather(log_probs, 2, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.sum().item()), candidate_len


def single_token_option_logprobs(
    model,
    tokenizer,
    prompt_text: str,
    valid_answers: list[str],
) -> tuple[dict[str, float], dict[str, int]] | None:
    token_ids_by_answer: dict[str, int] = {}
    for answer in valid_answers:
        token_ids = tokenizer.encode(answer, add_special_tokens=False)
        if len(token_ids) != 1:
            return None
        token_ids_by_answer[answer] = token_ids[0]

    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    attention_mask = torch.ones_like(prompt_ids)
    with torch.no_grad():
        outputs = model(input_ids=prompt_ids, attention_mask=attention_mask)
        next_token_log_probs = torch.log_softmax(outputs.logits[0, -1, :], dim=-1)

    return (
        {answer: float(next_token_log_probs[token_id].item()) for answer, token_id in token_ids_by_answer.items()},
        {answer: 1 for answer in valid_answers},
    )


def normalize_logprobs(logprobs: dict[str, float]) -> dict[str, float]:
    max_logprob = max(logprobs.values())
    total = sum(math.exp(value - max_logprob) for value in logprobs.values())
    log_total = max_logprob + math.log(total)
    return {key: math.exp(value - log_total) for key, value in logprobs.items()}


def build_row_result(
    row: dict,
    option_probs: dict[str, float],
    option_logprobs: dict[str, float],
    candidate_lengths: dict[str, int],
) -> dict:
    argmax_answer = max(option_probs, key=option_probs.get)
    result = {
        "id": row["id"],
        "task_family": row["task_family"],
        "prompt_type": row["prompt_type"],
        "valid_answers": row["valid_answers"],
        "argmax_answer": argmax_answer,
        "option_logprobs": option_logprobs,
        "option_probabilities": option_probs,
        "candidate_token_lengths": candidate_lengths,
    }

    if row["prompt_type"] == "deterministic":
        short_answers = [letter for letter, info in row["options"].items() if info["length"] == "short"]
        long_answers = [letter for letter, info in row["options"].items() if info["length"] == "long"]
        argmax_trajectory = row["options"][argmax_answer]["length"]
        short_prob = sum(option_probs.get(letter, 0.0) for letter in short_answers)
        long_prob = sum(option_probs.get(letter, 0.0) for letter in long_answers)
        result.update(
            {
                "argmax_trajectory": argmax_trajectory,
                "short_probability": short_prob,
                "long_probability": long_prob,
                "length_entropy_bits": binary_entropy(long_prob),
                "target_best_short_option": row["target_best_short_option"],
                "target_best_long_option": row["target_best_long_option"],
                "best_short_probability": option_probs.get(row["target_best_short_option"], 0.0),
                "best_long_probability": option_probs.get(row["target_best_long_option"], 0.0),
                "decoy_probability": sum(
                    option_probs.get(letter, 0.0)
                    for letter in row["valid_answers"]
                    if letter not in {row["target_best_short_option"], row["target_best_long_option"]}
                ),
            }
        )
    else:
        shift_answers = [
            letter
            for letter, info in row.get("options", {}).items()
            if info.get("action") == "shift_probability_mass"
        ]
        no_shift_answers = [
            letter
            for letter, info in row.get("options", {}).items()
            if info.get("action") == "do_not_shift"
        ]
        if not shift_answers or not no_shift_answers:
            # Older stochastic datasets always used a=no-shift and b=shift.
            shift_answers = ["b"]
            no_shift_answers = ["a"]
        shift_answer = shift_answers[0]
        no_shift_answer = no_shift_answers[0]
        utility_target_option = row["utility_maximizing_option"]
        neutral_target_option = row["neutral_target_option"]
        shift_prob = sum(option_probs.get(answer, 0.0) for answer in shift_answers)
        no_shift_prob = sum(option_probs.get(answer, 0.0) for answer in no_shift_answers)
        result.update(
            {
                "neutral_target_option": neutral_target_option,
                "utility_maximizing_option": utility_target_option,
                "shift_option": shift_answer,
                "no_shift_option": no_shift_answer,
                "neutral_target_probability": option_probs.get(neutral_target_option, 0.0),
                "utility_target_probability": option_probs.get(utility_target_option, 0.0),
                "shift_probability": shift_prob,
                "no_shift_probability": no_shift_prob,
                "answer_entropy_bits": binary_entropy(shift_prob),
                "base_expected_score": row["base_expected_score"],
                "shifted_expected_score": row["shifted_expected_score"],
                "expected_score_gain_from_shift": row["expected_score_gain_from_shift"],
            }
        )

    return result


def summarize_deterministic(results: list[dict]) -> dict:
    if not results:
        return {}
    argmax_short = sum(1 for row in results if row["argmax_trajectory"] == "short")
    argmax_long = len(results) - argmax_short
    return {
        "count": len(results),
        "argmax_short_rate": argmax_short / len(results),
        "argmax_long_rate": argmax_long / len(results),
        "mean_short_probability": sum(row["short_probability"] for row in results) / len(results),
        "mean_long_probability": sum(row["long_probability"] for row in results) / len(results),
        "mean_length_entropy_bits": sum(row["length_entropy_bits"] for row in results) / len(results),
        "mean_best_short_probability": sum(row["best_short_probability"] for row in results) / len(results),
        "mean_best_long_probability": sum(row["best_long_probability"] for row in results) / len(results),
        "mean_decoy_probability": sum(row["decoy_probability"] for row in results) / len(results),
    }


def summarize_stochastic(results: list[dict]) -> dict:
    if not results:
        return {}
    argmax_shift = sum(1 for row in results if row["argmax_answer"] == row.get("shift_option", "b"))
    argmax_no_shift = sum(1 for row in results if row["argmax_answer"] == row.get("no_shift_option", "a"))
    return {
        "count": len(results),
        "argmax_shift_rate": argmax_shift / len(results),
        "argmax_no_shift_rate": argmax_no_shift / len(results),
        "mean_shift_probability": sum(row["shift_probability"] for row in results) / len(results),
        "mean_no_shift_probability": sum(row["no_shift_probability"] for row in results) / len(results),
        "mean_neutral_target_probability": sum(row["neutral_target_probability"] for row in results) / len(results),
        "mean_utility_target_probability": sum(row["utility_target_probability"] for row in results) / len(results),
        "mean_answer_entropy_bits": sum(row["answer_entropy_bits"] for row in results) / len(results),
    }


def main() -> None:
    args = parse_args()
    started_utc = utc_now_iso()
    started_perf = time.perf_counter()

    rows = load_jsonl(args.dataset_path)
    rows = maybe_sample_rows(rows, max_samples=args.max_samples if args.max_samples > 0 else None, seed=args.seed)
    model, tokenizer = load_model(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        load_in_4bit=args.load_in_4bit,
        torch_dtype_name=args.torch_dtype,
    )

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
            enable_thinking=args.enable_thinking,
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

    prompt_type = rows[0]["prompt_type"] if rows else ""
    if prompt_type == "deterministic":
        summary = summarize_deterministic(results)
    else:
        summary = summarize_stochastic(results)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "evaluation_config": {
            "model_name": args.model_name,
            "adapter_path": args.adapter_path or None,
            "dataset_path": args.dataset_path,
            "output_path": str(output_path),
            "max_samples": args.max_samples,
            "seed": args.seed,
            "enable_thinking": args.enable_thinking,
            "load_in_4bit": args.load_in_4bit,
            "torch_dtype": args.torch_dtype,
            "num_prompts": len(results),
            "prompt_type": prompt_type,
            "mode": "direct_option_logprobs",
        },
        "summary": summary,
        "counts": {
            "num_total": len(results),
        },
        "scoring_method_counts": dict(method_counter),
        "timing": {
            "started_at_utc": started_utc,
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": time.perf_counter() - started_perf,
            "total_scoring_seconds": total_scoring_seconds,
            "mean_scoring_seconds": total_scoring_seconds / len(results) if results else 0.0,
        },
        "tokens": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_candidate_tokens": total_candidate_tokens,
            "mean_prompt_tokens": total_prompt_tokens / len(results) if results else 0.0,
            "mean_candidate_tokens_per_prompt": total_candidate_tokens / len(results) if results else 0.0,
        },
        "results": results,
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
