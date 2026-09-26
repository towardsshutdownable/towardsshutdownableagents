from __future__ import annotations

import argparse
import inspect
import json
from datetime import datetime
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl.trainer.rloo_config import RLOOConfig

from pap_dataset_utils import find_latest_checkpoint, load_jsonl, maybe_sample_rows, to_rloo_dataset_rows
from pap_meta_ep_rloo import PAPMetaRLOOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Qwen/Qwen3-8B")
    parser.add_argument("--init_adapter_path", default="")
    parser.add_argument(
        "--train_path",
        default="data/deterministic_train_diverse_2048.jsonl",
    )
    parser.add_argument("--output_root", default="runs")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--max_train_samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260325)
    parser.add_argument("--meta_ep_size", type=int, default=32)
    parser.add_argument("--lambda_factor", type=float, default=0.9)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--max_completion_length", type=int, default=16)
    parser.add_argument("--save_steps", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", default="")
    parser.add_argument("--report_to", default="none")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--vllm_mode", default="server")
    parser.add_argument("--vllm_model_impl", default="")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.0)
    parser.add_argument(
        "--vllm_max_model_len",
        "--vllm_max_model_length",
        dest="vllm_max_model_len",
        type=int,
        default=0,
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str, model_name: str = "") -> torch.dtype:
    if dtype_name.lower() == "auto":
        # Gemma 3 produced NaNs under fp16 on this stack; use bf16 on A100/H100.
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


def is_gemma3_model(model_name: str) -> bool:
    return "gemma-3" in model_name.lower()


def render_conversations_as_plain_text(dataset_rows: list[dict], tokenizer) -> None:
    for row in dataset_rows:
        row["prompt"] = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=False,
            add_generation_prompt=True,
        )


def resolve_run_dir(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_slug = args.model_name.split("/")[-1].replace(".", "_")
        run_name = f"{timestamp}_{model_slug}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_resume_checkpoint(run_dir: Path, resume_arg: str) -> str | None:
    if not resume_arg:
        return None
    if resume_arg == "latest":
        return find_latest_checkpoint(run_dir / "checkpoints")
    return resume_arg


def filter_rloo_config_kwargs(kwargs: dict) -> dict:
    """Keep the PAP runner usable across small TRL config-version changes."""
    valid_params = inspect.signature(RLOOConfig.__init__).parameters
    return {
        key: value
        for key, value in kwargs.items()
        if key in valid_params and value is not None
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_dir = resolve_run_dir(args)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    rows = load_jsonl(args.train_path)
    rows = maybe_sample_rows(rows, max_samples=args.max_train_samples, seed=args.seed)
    dataset_rows = to_rloo_dataset_rows(rows, include_short_reasoning=False)
    prompt_format = "conversational_messages"
    if is_gemma3_model(args.model_name) and not args.use_vllm:
        # Gemma 3's batched chat-template path can produce generation-time shape
        # issues under TRL. Pre-rendering keeps the same prompt text but lets TRL
        # use the simpler plain-text tokenizer path.
        render_conversations_as_plain_text(dataset_rows, tokenizer)
        prompt_format = "plain_text_chat_rendered_for_gemma3"
    elif is_gemma3_model(args.model_name):
        prompt_format = "conversational_messages_for_gemma3_vllm"
    dataset = Dataset.from_list(dataset_rows)

    run_config = vars(args).copy()
    run_config["resolved_train_path"] = str(Path(args.train_path).resolve())
    run_config["dataset_size"] = len(dataset_rows)
    run_config["prompt_format"] = prompt_format
    run_config["run_dir"] = str(run_dir.resolve())
    run_config["checkpoints_dir"] = str(checkpoints_dir.resolve())
    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2)

    torch_dtype = resolve_dtype(args.torch_dtype, model_name=args.model_name)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    patch_gemma3_token_type_ids(base_model)
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)
    if args.init_adapter_path:
        model = PeftModel.from_pretrained(
            base_model,
            args.init_adapter_path,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",
            r=16,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            target_modules="all-linear",
        )
        model = get_peft_model(base_model, lora_config)
    patch_gemma3_token_type_ids(model)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model.print_trainable_parameters()

    training_kwargs = dict(
        output_dir=str(checkpoints_dir),
        per_device_train_batch_size=args.meta_ep_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=5,
        logging_steps=args.logging_steps,
        report_to=args.report_to,
        beta=0.0,
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        steps_per_generation=1,
        use_vllm=args.use_vllm,
        vllm_mode=args.vllm_mode,
        gradient_checkpointing=True,
        chat_template_kwargs={"enable_thinking": False},
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if args.vllm_model_impl:
        training_kwargs["vllm_model_impl"] = args.vllm_model_impl
    if args.vllm_gpu_memory_utilization > 0:
        training_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    if args.vllm_max_model_len > 0:
        training_kwargs["vllm_max_model_length"] = args.vllm_max_model_len
        training_kwargs["vllm_max_model_len"] = args.vllm_max_model_len
    training_kwargs = filter_rloo_config_kwargs(training_kwargs)
    training_args = RLOOConfig(**training_kwargs)

    trainer = PAPMetaRLOOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        meta_ep_size=args.meta_ep_size,
        lambda_factor=args.lambda_factor,
        processing_class=tokenizer,
    )

    resume_checkpoint = resolve_resume_checkpoint(run_dir, args.resume_from_checkpoint)
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_adapter_dir = run_dir / "final_adapter"
    trainer.save_model(str(final_adapter_dir))
    tokenizer.save_pretrained(final_adapter_dir)

    with (run_dir / "train_actions.json").open("w", encoding="utf-8") as handle:
        json.dump(trainer.actions, handle)


if __name__ == "__main__":
    main()
