#!/usr/bin/env python3
"""Sample answers to an evaluation set with vLLM, read each decision, and summarise.

    python scripts/evaluate.py --model Qwen/Qwen3-14B --data_dir data/test --out_dir runs/eval/qwen_untrained_test
    python scripts/evaluate.py --model Qwen/Qwen3-14B --adapter runs/qwen_drest_s1/final_adapter \
        --data_dir data/test --out_dir runs/eval/qwen_drest_s1_test

The defaults are the paper's: four answers per displayed prompt (so eight per scenario, since every scenario
is shown in both option orders), temperature 1.0, top-p 1.0, at most 320 new tokens, thinking mode off where
the chat template has the switch, sampling seed 20260908. For gpt-oss-20b add
    --chat_kwargs '{"reasoning_effort": "low"}' --pin_chat_date 2026-09-28
Adapters are applied through vLLM's LoRA route (gpt-oss-20b needs the vLLM patch in patches/).

Writes <out_dir>/<suite>.samples.jsonl (every answer in full, with its parsed decision), <suite>.summary.json,
and CONFIG.json. Rerunning resumes: displays that already have all their answers are skipped.
Use --dry_run to test the pipeline without a GPU (a fake model writes random final lines).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

LLM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLM_DIR))
from drest_llm.metrics import binary_summary, post_summary  # noqa: E402
from drest_llm.parse import parse_binary, parse_deterministic  # noqa: E402
from drest_llm.scenarios import read_jsonl  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="", help="a LoRA adapter folder (omit for the untrained model)")
    ap.add_argument("--data_dir", required=True, help="data/test, data/test_low_gain, or data/validation")
    ap.add_argument("--suites", default="", help="comma-separated suite names (default: every .jsonl file in data_dir)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_samples", type=int, default=4, help="answers per displayed prompt (the paper used 8 on the validation set)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--chat_kwargs", default="", help='extra chat-template arguments as JSON, e.g. {"reasoning_effort": "low"}')
    ap.add_argument("--pin_chat_date", default="", help="replace the date printed by some chat templates (gpt-oss-20b) with this date")
    ap.add_argument("--max_lora_rank", type=int, default=64)
    ap.add_argument("--batch_displays", type=int, default=512)
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


class FakeLLM:
    """Stands in for vLLM with --dry_run: writes a short rationale and a random permitted final line."""

    def __init__(self, seed):
        self.rng = random.Random(seed)

    def get_tokenizer(self):
        class Tok:
            chat_template = ""

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
                return messages[-1]["content"]
        return Tok()

    def generate(self, prompts, sampling, lora_request=None, use_tqdm=False):
        outs = []
        for prompt in prompts:
            lines = [part.split('"')[0] for part in prompt.split('"Final decision: ')[1:]]
            text = f"I weigh the scores.\n\nFinal decision: {self.rng.choice(lines)}"
            outs.append(type("Out", (), {"outputs": [type("O", (), {"text": text, "token_ids": text.split(), "finish_reason": "stop"})()]})())
        return outs


def main():
    args = parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suites = [s for s in args.suites.split(",") if s] or sorted(p.stem for p in data_dir.glob("*.jsonl"))

    if args.dry_run:
        llm, lora_request = FakeLLM(args.seed), None
        SamplingParams = lambda **kw: kw  # noqa: E731
    else:
        from vllm import LLM, SamplingParams
        llm_kwargs = dict(model=args.model, dtype=args.dtype, seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization,
                          max_model_len=args.max_model_len, enable_prefix_caching=True, trust_remote_code=True)
        from transformers import AutoConfig
        if getattr(AutoConfig.from_pretrained(args.model, trust_remote_code=True), "vision_config", None) is not None:
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0}  # text-only evaluation of a model that can also read images
        lora_request = None
        if args.adapter:
            from vllm.lora.request import LoRARequest
            llm_kwargs.update(enable_lora=True, max_lora_rank=args.max_lora_rank)
            lora_request = LoRARequest("adapter", 1, str(Path(args.adapter).resolve()))
        llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    template = getattr(tokenizer, "chat_template", None) or ""
    chat_kwargs = {"enable_thinking": False} if "enable_thinking" in template else {}
    chat_kwargs.update(json.loads(args.chat_kwargs) if args.chat_kwargs else {})
    if args.pin_chat_date and 'strftime_now("%Y-%m-%d")' in template:
        tokenizer.chat_template = template.replace('strftime_now("%Y-%m-%d")', f'"{args.pin_chat_date}"')
    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens, n=1, seed=None)
    config = {**vars(args), "suites": suites, "chat_kwargs_used": chat_kwargs, "started": datetime.now(timezone.utc).isoformat()}
    (out_dir / "CONFIG.json").write_text(json.dumps(config, indent=2))

    for suite in suites:
        displays = read_jsonl(data_dir / f"{suite}.jsonl")
        by_id = {d["display_id"]: d for d in displays}
        samples_path = out_dir / f"{suite}.samples.jsonl"
        done = defaultdict(int)
        if samples_path.exists():
            for row in read_jsonl(samples_path):
                done[row["display_id"]] += 1
        todo = [(d, k) for d in displays for k in range(done[d["display_id"]], args.n_samples)]
        start_time = time.time()
        with samples_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(todo), args.batch_displays * args.n_samples):
                chunk = todo[start: start + args.batch_displays * args.n_samples]
                prompts = [tokenizer.apply_chat_template(d["messages"], tokenize=False, add_generation_prompt=True, **chat_kwargs) for d, _ in chunk]
                outputs = llm.generate(prompts, sampling, lora_request=lora_request, use_tqdm=False)
                for (d, k), out in zip(chunk, outputs):
                    text = out.outputs[0].text
                    row = {"display_id": d["display_id"], "scenario_id": d["scenario_id"], "suite": suite, "sample_index": k,
                           "completion": text, "finish_reason": out.outputs[0].finish_reason,
                           "n_completion_tokens": len(out.outputs[0].token_ids)}
                    if d["scenario_type"] == "binary":
                        p = parse_binary(text, d)
                        row.update({"decision": p.decision, "strategy": p.strategy, "final_line": p.final_line})
                    else:
                        p = parse_deterministic(text, d)
                        row.update({"position": p.position, "label": p.label, "strategy": p.strategy, "final_line": p.final_line})
                    handle.write(json.dumps(row, ensure_ascii=True) + "\n")
                handle.flush()
        samples = read_jsonl(samples_path)
        summary = post_summary(samples, by_id) if displays[0]["scenario_type"] == "deterministic" else binary_summary(samples, by_id)
        summary["seconds"] = round(time.time() - start_time, 1)
        (out_dir / f"{suite}.summary.json").write_text(json.dumps(summary, indent=2))
        print(suite, json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
