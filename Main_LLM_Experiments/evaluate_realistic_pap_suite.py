from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import time
from pathlib import Path

import torch

from evaluate_realistic_pap import (
    evaluate_deterministic,
    evaluate_stochastic,
    extract_answer,
    load_model,
)
from pap_dataset_utils import load_jsonl, maybe_sample_rows, to_eval_rows


@dataclass
class EvalTask:
    dataset_path: str
    output_path: str
    max_samples: int
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save_every", type=int, default=8)
    parser.add_argument("--suite_output_path", default="")
    parser.add_argument("--hardware_profile", default="")
    parser.add_argument("--backend", default="transformers_batched")
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_task_spec(spec: str) -> EvalTask:
    parts = spec.split("|")
    if len(parts) != 7:
        raise ValueError(
            "Task spec must have 7 pipe-delimited fields: "
            "dataset_path|output_path|max_samples|max_new_tokens|temperature|top_p|do_sample"
        )
    dataset_path, output_path, max_samples, max_new_tokens, temperature, top_p, do_sample = parts
    return EvalTask(
        dataset_path=dataset_path,
        output_path=output_path,
        max_samples=int(max_samples),
        max_new_tokens=int(max_new_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
        do_sample=do_sample.strip().lower() in {"1", "true", "yes", "y"},
    )


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    tmp_path.replace(path)


def load_existing_results(output_path: Path) -> list[dict]:
    report_path = output_path.with_suffix(".json")
    if report_path.exists():
        with report_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            return payload["results"]
    if output_path.exists():
        rows: list[dict] = []
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    return []


def generate_batch(
    model,
    tokenizer,
    batch_rows: list[dict],
    *,
    enable_thinking: bool,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> list[dict]:
    prompt_texts = [
        tokenizer.apply_chat_template(
            row["prompt_messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        for row in batch_rows
    ]
    inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)
    prompt_token_counts = [int(value) for value in inputs["attention_mask"].sum(dim=1).tolist()]
    padded_prompt_length = int(inputs["input_ids"].shape[1])

    generation_started = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generation_seconds = time.perf_counter() - generation_started

    completions: list[dict] = []
    for index, row in enumerate(batch_rows):
        new_token_ids = output_ids[index][padded_prompt_length:]
        completion = tokenizer.decode(new_token_ids, skip_special_tokens=True)
        completion_token_count = int(
            len(tokenizer(completion, add_special_tokens=False)["input_ids"])
        )
        answer, parser_strategy = extract_answer(
            completion,
            row["valid_answers"],
            row["prompt_messages"],
        )
        completions.append(
            {
                **row,
                "completion": completion,
                "prompt_token_count": prompt_token_counts[index],
                "completion_token_count": completion_token_count,
                "completion_char_count": len(completion),
                "generation_seconds": generation_seconds / max(len(batch_rows), 1),
                "reached_max_new_tokens": completion_token_count >= max_new_tokens,
                "answer": answer,
                "answer_is_valid": answer in row["valid_answers"],
                "parser_strategy": parser_strategy,
            }
        )
    return completions


def build_task_report(
    *,
    model_name: str,
    adapter_path: str,
    task: EvalTask,
    eval_rows: list[dict],
    completions: list[dict],
    batch_size: int,
    enable_thinking: bool,
    hardware_profile: str,
    backend: str,
    script_started_utc: str,
    script_started_perf: float,
    generation_started_utc: str,
    generation_started_perf: float,
    num_resumed: int,
) -> dict:
    parser_strategy_counts = Counter(
        item["parser_strategy"] if item["parser_strategy"] is not None else "unparsed"
        for item in completions
    )
    total_prompt_tokens = sum(item.get("prompt_token_count", 0) for item in completions)
    total_completion_tokens = sum(item.get("completion_token_count", 0) for item in completions)
    total_generation_seconds = sum(item.get("generation_seconds", 0.0) for item in completions)
    valid_count = sum(1 for item in completions if item["answer"] in item["valid_answers"])
    prompt_type = eval_rows[0]["prompt_type"] if eval_rows else ""
    summary = (
        evaluate_deterministic(eval_rows, completions)
        if prompt_type == "deterministic"
        else evaluate_stochastic(completions)
    )
    return {
        "evaluation_config": {
            "model_name": model_name,
            "adapter_path": adapter_path or None,
            "dataset_path": task.dataset_path,
            "output_path": task.output_path,
            "prompt_type": prompt_type,
            "max_samples": task.max_samples,
            "max_new_tokens": task.max_new_tokens,
            "temperature": task.temperature,
            "top_p": task.top_p,
            "do_sample": task.do_sample,
            "enable_thinking": enable_thinking,
            "batch_size": batch_size,
            "backend": backend,
            "hardware_profile": hardware_profile or None,
            "num_prompts": len(completions),
            "num_resumed": num_resumed,
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
            "finished_at_utc": utc_now_iso(),
            "total_runtime_seconds": time.perf_counter() - script_started_perf,
            "generation_runtime_seconds": time.perf_counter() - generation_started_perf,
            "total_generation_seconds": total_generation_seconds,
            "mean_generation_seconds": total_generation_seconds / len(completions) if completions else 0.0,
        },
        "tokens": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "mean_prompt_tokens": total_prompt_tokens / len(completions) if completions else 0.0,
            "mean_completion_tokens": total_completion_tokens / len(completions) if completions else 0.0,
        },
        "results": completions,
    }


def save_task_outputs(report: dict, output_path: Path) -> None:
    results = report["results"]
    atomic_write_json(output_path.with_suffix(".json"), report)
    atomic_write_json(output_path.with_suffix(".summary.json"), report["summary"])
    atomic_write_jsonl(output_path, results)


def run_task(
    model,
    tokenizer,
    task: EvalTask,
    args: argparse.Namespace,
) -> dict:
    script_started_utc = utc_now_iso()
    script_started_perf = time.perf_counter()

    rows = load_jsonl(task.dataset_path)
    rows = maybe_sample_rows(
        rows,
        max_samples=task.max_samples if task.max_samples > 0 else None,
        seed=args.seed,
    )
    eval_rows = to_eval_rows(rows, include_short_reasoning=True)
    output_path = Path(task.output_path)

    existing_results = load_existing_results(output_path) if args.resume else []
    completed_by_id = {item["id"]: item for item in existing_results}
    num_resumed = len(completed_by_id)
    remaining_rows = [row for row in eval_rows if row["id"] not in completed_by_id]
    generation_started_utc = utc_now_iso()
    generation_started_perf = time.perf_counter()

    for start in range(0, len(remaining_rows), args.batch_size):
        batch_rows = remaining_rows[start : start + args.batch_size]
        batch_completions = generate_batch(
            model,
            tokenizer,
            batch_rows,
            enable_thinking=args.enable_thinking,
            max_new_tokens=task.max_new_tokens,
            do_sample=task.do_sample,
            temperature=task.temperature,
            top_p=task.top_p,
        )
        for item in batch_completions:
            completed_by_id[item["id"]] = item

        completed_results = [completed_by_id[row["id"]] for row in eval_rows if row["id"] in completed_by_id]
        if ((start // args.batch_size) + 1) % max(args.save_every, 1) == 0:
            report = build_task_report(
                model_name=args.model_name,
                adapter_path=args.adapter_path,
                task=task,
                eval_rows=eval_rows,
                completions=completed_results,
                batch_size=args.batch_size,
                enable_thinking=args.enable_thinking,
                hardware_profile=args.hardware_profile,
                backend=args.backend,
                script_started_utc=script_started_utc,
                script_started_perf=script_started_perf,
                generation_started_utc=generation_started_utc,
                generation_started_perf=generation_started_perf,
                num_resumed=num_resumed,
            )
            save_task_outputs(report, output_path)

    completed_results = [completed_by_id[row["id"]] for row in eval_rows if row["id"] in completed_by_id]
    report = build_task_report(
        model_name=args.model_name,
        adapter_path=args.adapter_path,
        task=task,
        eval_rows=eval_rows,
        completions=completed_results,
        batch_size=args.batch_size,
        enable_thinking=args.enable_thinking,
        hardware_profile=args.hardware_profile,
        backend=args.backend,
        script_started_utc=script_started_utc,
        script_started_perf=script_started_perf,
        generation_started_utc=generation_started_utc,
        generation_started_perf=generation_started_perf,
        num_resumed=num_resumed,
    )
    save_task_outputs(report, output_path)
    print(json.dumps({"output_path": task.output_path, "summary": report["summary"]}, indent=2))
    return report


def main() -> None:
    args = parse_args()
    tasks = [parse_task_spec(spec) for spec in args.task]
    model, tokenizer = load_model(args.model_name, args.adapter_path)

    suite_started_utc = utc_now_iso()
    suite_started_perf = time.perf_counter()
    task_reports = [run_task(model, tokenizer, task, args) for task in tasks]

    if args.suite_output_path:
        suite_report = {
            "suite_config": {
                "model_name": args.model_name,
                "adapter_path": args.adapter_path or None,
                "backend": args.backend,
                "batch_size": args.batch_size,
                "hardware_profile": args.hardware_profile or None,
                "resume": args.resume,
                "num_tasks": len(task_reports),
            },
            "timing": {
                "suite_started_at_utc": suite_started_utc,
                "finished_at_utc": utc_now_iso(),
                "total_runtime_seconds": time.perf_counter() - suite_started_perf,
            },
            "tasks": [
                {
                    "output_path": report["evaluation_config"]["output_path"],
                    "dataset_path": report["evaluation_config"]["dataset_path"],
                    "summary": report["summary"],
                }
                for report in task_reports
            ],
        }
        atomic_write_json(Path(args.suite_output_path), suite_report)


if __name__ == "__main__":
    main()
