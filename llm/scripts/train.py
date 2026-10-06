#!/usr/bin/env python3
"""Fine-tune one seed of one LLM with the DReST reward or the default reward (RLOO with LoRA, one H100).

The defaults are the paper's settings (Table 6): one pass over the first 2,048 training prompts, 32 answers
per prompt, learning rate 2e-5 held constant after 16 warm-up steps, KL coefficient 0.04, gradient clipping 10,
answers capped at 320 tokens, LoRA rank 32 and alpha 32 on every linear layer, no quantization.
(TRL 1.13 has no prompt-length cap. The longest training prompt is about 400 tokens, and the sampler's
context is 1,280 tokens.)

    python scripts/train.py --model_name Qwen/Qwen3-14B --reward_mode drest_order_averaged --seed 1 --run_name qwen_drest_s1
    python scripts/train.py --model_name Qwen/Qwen3-14B --reward_mode default_task_score   --seed 1 --run_name qwen_default_s1

scripts/run_paper_experiments.sh has the exact command for every model, including the gpt-oss-20b settings.
Outputs in <output_root>/<run_name>/: run_config.json, train_samples.jsonl (every sampled answer, its parsed
choice and reward), checkpoints/, final_adapter/, and train_actions_audit.json. Redirect stdout and stderr to
<run_name>/train.log, which scripts/check_training_health.py reads.

Needs the library patches in patches/ (python patches/apply_patches.py).
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from trl.trainer.rloo_config import RLOOConfig

LLM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLM_DIR))
from drest_llm.scenarios import read_jsonl  # noqa: E402
from drest_llm.trainer import DReSTRLOOTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_name", required=True, help="Qwen/Qwen3-14B, google/gemma-4-12B-it, ibm-granite/granite-4.2-8b, or openai/gpt-oss-20b")
    p.add_argument("--reward_mode", required=True, choices=("drest_order_averaged", "default_task_score", "drest"),
                   help="drest_order_averaged is DReST as used in the paper. drest counts answers in sampler order instead.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run_name", required=True)
    p.add_argument("--output_root", default=str(LLM_DIR / "runs"))
    p.add_argument("--train_path", default=str(LLM_DIR / "data/train/train_v7_want.jsonl"))
    p.add_argument("--max_train_samples", type=int, default=2048, help="use the first N rows (one complete rendering); 0 = all")
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--meta_ep_size", type=int, default=32, help="answers per prompt, all scored as one DReST meta-episode")
    p.add_argument("--lambda_factor", type=float, default=0.9)
    p.add_argument("--default_reward_scale", type=float, default=0.1)
    p.add_argument("--invalid_reward", type=float, default=-1.5)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--lr_scheduler_type", default="constant_with_warmup")
    p.add_argument("--warmup_steps", type=int, default=16, help="in meta-episodes (one weight update each)")
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=0.04, help="KL penalty coefficient against the untrained model")
    p.add_argument("--micro_batch_size", type=int, default=8,
                   help="answers per backward pass. Affects memory only: gradients over the whole meta-episode are summed before each update")
    p.add_argument("--gradient_checkpointing", choices=("on", "off"), default="on")
    p.add_argument("--max_completion_length", type=int, default=320)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0,
                   help="keep at 0 for RL: with dropout the two forward passes over the same answer disagree")
    p.add_argument("--save_meta_episodes", type=int, default=128, help="checkpoint every N meta-episodes")
    p.add_argument("--resume_from_checkpoint", default="", help="a checkpoint folder, or 'latest'")
    p.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.45, help="share of GPU memory for the vLLM sampler (0.25 for gpt-oss-20b)")
    p.add_argument("--vllm_max_model_len", type=int, default=1280)
    p.add_argument("--chat_template_kwargs", default="", help='extra chat-template arguments as JSON, e.g. {"reasoning_effort": "low"} for gpt-oss-20b')
    p.add_argument("--pin_chat_date", default="", help="replace the current date in chat templates that print one (gpt-oss-20b) with this date")
    p.add_argument("--sync_only_adapted", action="store_true",
                   help="gpt-oss-20b: copy into the sampler only the weights the adapter changes (its frozen expert weights stay in MXFP4)")
    p.add_argument("--exact_sampler", action="store_true",
                   help="sample through vLLM's LoRA route instead of merging the adapter into 16-bit weights (robustness check, not the paper's main runs)")
    return p.parse_args()


def end_of_answer_fix(tokenizer, model_name: str) -> None:
    """TRL treats an answer as cut off unless it contains tokenizer.eos_token, and drops cut-off answers from the loss.
    The sampler stops at the token that closes an assistant turn. For Qwen the two coincide. For Gemma 4 they do not,
    which silently gave zero gradient, so set eos_token to the turn-closing token from the chat template."""
    from transformers import GenerationConfig
    try:
        gen_eos = GenerationConfig.from_pretrained(model_name).eos_token_id
    except Exception as exc:  # noqa: BLE001
        print(f"[eos fix] skipped: {exc}", flush=True)
        return
    gen_eos = gen_eos if isinstance(gen_eos, list) else [gen_eos]
    probe = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "PROBE_ANSWER"}], tokenize=False)
    tail = probe[probe.rfind("PROBE_ANSWER") + len("PROBE_ANSWER"):]
    for token_id in gen_eos:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if token and token in tail:
            if token != tokenizer.eos_token:
                print(f"[eos fix] end-of-answer token set to {token!r} ({token_id}), was {tokenizer.eos_token!r}", flush=True)
                tokenizer.eos_token = token
            else:
                print(f"[eos fix] end-of-answer token already {token!r} ({token_id})", flush=True)
            return


def install_adapted_only_sync(peft_model) -> None:
    """gpt-oss-20b. After each update TRL merges the adapter into the base weights and copies every parameter into
    the vLLM sampler. gpt-oss's sampler keeps the frozen expert weights in MXFP4 and cannot take bf16 copies. Merging
    an adapter changes only the base weights of the modules it wraps, so copy only those."""
    import trl.generation.vllm_generation as vg
    trainable = [n for n, p in peft_model.named_parameters() if p.requires_grad]
    if not trainable or not all("lora_" in n for n in trainable):
        raise SystemExit("something other than LoRA weights is trainable, refusing to skip any weights")
    adapted = {n.removeprefix("base_model.model.").replace(".base_layer", "") for n, _ in peft_model.named_parameters() if ".base_layer." in n}
    if not adapted:
        raise SystemExit("no LoRA-wrapped modules found")
    print(f"[adapted-only sync] copying {len(adapted)} tensors of LoRA-wrapped modules "
          f"({', '.join(sorted({n.rsplit('.', 2)[-2] for n in adapted}))})", flush=True)
    original = vg.VLLMGeneration._iter_named_params
    state = {"checked": False}

    def iter_adapted_only(self):
        sent = 0
        for name, param in original(self):  # consume everything so that TRL's merge and unmerge both run
            if name in adapted:
                sent += 1
                yield name, param
        if not state["checked"]:
            if sent != len(adapted):
                raise RuntimeError(f"expected to copy {len(adapted)} tensors, copied {sent}")
            state["checked"] = True

    vg.VLLMGeneration._iter_named_params = iter_adapted_only


def install_exact_sampler(run_name: str) -> None:
    """Robustness check. TRL's colocated sampler has no adapter of its own: after each update TRL merges the adapter
    into the trainer's 16-bit base weights, copies them into vLLM, and unmerges. The merge rounds away part of the
    adapter's change, so the scored answers come from a slightly blurred policy. This option instead hands vLLM the
    adapter itself (vLLM's LoRA route, as in evaluation) and never touches the base weights."""
    import shutil
    import trl.generation.vllm_generation as vg
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file
    from vllm.lora.request import LoRARequest
    root = Path("/dev/shm/drest_exact_sampler") / run_name
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    holder = {"req": None, "n": 0}
    original_llm = vg.LLM

    def llm_with_lora(*a, **k):
        k.update(enable_lora=True, max_lora_rank=64, max_loras=1)
        llm = original_llm(*a, **k)
        generate = llm.generate

        def generate_with_adapter(*ga, **gk):
            if gk.get("lora_request") is None and holder["req"] is not None:
                gk["lora_request"] = holder["req"]
            return generate(*ga, **gk)
        llm.generate = generate_with_adapter
        return llm
    vg.LLM = llm_with_lora

    def sync_weights(self):
        if self.mode != "colocate" or self.enable_sleep_mode:
            raise RuntimeError("the exact sampler is only written for colocate mode without sleep mode")
        holder["n"] += 1
        folder = root / f"copy{holder['n']}"
        folder.mkdir()
        state = {n: t.detach().to("cpu").contiguous() for n, t in get_peft_model_state_dict(self.model).items()}
        save_file(state, str(folder / "adapter_model.safetensors"))
        self.model.peft_config["default"].save_pretrained(str(folder))
        previous, holder["req"] = holder["req"], LoRARequest(f"copy{holder['n']}", holder["n"], str(folder))
        if previous is not None:
            self.llm.llm_engine.remove_lora(previous.lora_int_id)
            shutil.rmtree(previous.lora_path, ignore_errors=True)
        self.llm.reset_prefix_cache()

    vg.VLLMGeneration.sync_weights = sync_weights


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir = Path(args.output_root) / args.run_name
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if args.pin_chat_date:
        if isinstance(tokenizer.chat_template, str) and 'strftime_now("%Y-%m-%d")' in tokenizer.chat_template:
            tokenizer.chat_template = tokenizer.chat_template.replace('strftime_now("%Y-%m-%d")', f'"{args.pin_chat_date}"')
            print(f"[chat date] template date pinned to {args.pin_chat_date}", flush=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    end_of_answer_fix(tokenizer, args.model_name)
    tokenizer.padding_side = "left"

    rows = read_jsonl(args.train_path)
    if args.max_train_samples:
        rows = rows[: args.max_train_samples]
    keep = ("id", "prompt", "option_labels", "score_values", "trajectory_ids", "trajectory_names",
            "max_scores_by_length", "drest_num_lengths", "options", "display")
    dataset = Dataset.from_list([{k: r[k] for k in keep} for r in rows])
    config = vars(args).copy()
    config.update({"run_dir": str(run_dir), "dataset_size": len(dataset)})
    (run_dir / "run_config.json").write_text(json.dumps(config, indent=2))

    dtype = torch.bfloat16 if ("gemma" in args.model_name.lower() or torch.cuda.is_bf16_supported()) else torch.float16
    extra_load = {}
    from transformers import AutoConfig
    quant = getattr(AutoConfig.from_pretrained(args.model_name, trust_remote_code=True), "quantization_config", None)
    if (quant.get("quant_method") if isinstance(quant, dict) else getattr(quant, "quant_method", None)) == "mxfp4":
        from transformers import Mxfp4Config
        extra_load["quantization_config"] = Mxfp4Config(dequantize=True)  # gpt-oss: train on bf16 copies of the expert weights
        print("[mxfp4] expert weights dequantized to bf16 for training", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=dtype, trust_remote_code=True, **extra_load)
    base_model.config.use_cache = False
    base_model.enable_input_require_grads()  # needed for gradient checkpointing to reach the adapter

    resume = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint == "latest":
            found = sorted(checkpoints_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
            resume = found[-1] if found else None
        else:
            resume = Path(args.resume_from_checkpoint)
        if resume is not None and not (resume / "adapter_model.safetensors").exists():
            raise SystemExit(f"Resume checkpoint {resume} has no adapter weights, refusing to continue.")
    if resume is not None:
        # Load the checkpoint's adapter explicitly: the trainer restores the optimizer and data position but not the adapter.
        model = PeftModel.from_pretrained(base_model, str(resume), is_trainable=True)
    else:
        model = get_peft_model(base_model, LoraConfig(task_type="CAUSAL_LM", r=args.lora_rank, lora_alpha=args.lora_alpha,
                                                      lora_dropout=args.lora_dropout, bias="none", target_modules="all-linear"))
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.print_trainable_parameters()
    lora_b = sum(float(p.detach().abs().sum()) for n, p in model.named_parameters() if "lora_B" in n)
    print(f"[adapter] sum of |lora_B| at start = {lora_b:.6g} (0 means an untrained adapter)", flush=True)
    if resume is not None and lora_b == 0:
        raise SystemExit("The resumed adapter has all-zero B matrices, so nothing was loaded.")
    if args.sync_only_adapted and args.exact_sampler:
        raise SystemExit("--sync_only_adapted and --exact_sampler exclude each other")
    if args.sync_only_adapted:
        install_adapted_only_sync(model)
    if args.exact_sampler:
        install_exact_sampler(args.run_name)

    template = getattr(tokenizer, "chat_template", None) or ""
    chat_kwargs = {"enable_thinking": False} if "enable_thinking" in template else {}  # thinking mode off where the template has the switch
    chat_kwargs.update(json.loads(args.chat_template_kwargs) if args.chat_template_kwargs else {})
    training_args = RLOOConfig(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.meta_ep_size // args.micro_batch_size,  # one weight update per meta-episode
        num_generations=args.meta_ep_size,  # one meta-episode is one scoring group
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        max_completion_length=args.max_completion_length,
        save_strategy="steps",
        save_steps=args.save_meta_episodes,
        save_total_limit=6,
        logging_steps=1,
        report_to="none",
        beta=args.beta,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        mask_truncated_completions=True,  # answers cut off at the token cap are dropped from the loss
        router_aux_loss_coef=0.0,  # no mixture-of-experts load-balancing loss (only matters for gpt-oss-20b)
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_max_model_length=args.vllm_max_model_len,
        gradient_checkpointing=args.gradient_checkpointing == "on",
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        shuffle_dataset=False,  # the training file is already shuffled, and rows must be used in order
        **({"chat_template_kwargs": chat_kwargs} if chat_kwargs else {}),
    )
    trainer = DReSTRLOOTrainer(model=model, args=training_args, train_dataset=dataset, meta_ep_size=args.meta_ep_size,
                               lambda_factor=args.lambda_factor, reward_mode=args.reward_mode,
                               default_reward_scale=args.default_reward_scale, invalid_reward=args.invalid_reward,
                               processing_class=tokenizer)
    trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)

    final_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    actions = trainer.actions
    audit = {"n_answers": len(actions), "length_counts": dict(Counter(a["length"] for a in actions)),
             "parse_strategy_counts": dict(Counter(a["strategy"] for a in actions)),
             "unreadable_share": sum(a["position"] is None for a in actions) / max(1, len(actions)),
             "mean_reward": sum(a["reward"] for a in actions) / max(1, len(actions))}
    (run_dir / "train_actions_audit.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
