from __future__ import annotations

import re
import time
from collections.abc import Callable

import numpy as np
import torch
from accelerate.utils import gather
from peft import PeftModel
from torch.utils.data import Dataset
from trl.trainer.rloo_config import RLOOConfig
from trl.trainer.rloo_trainer import RLOOTrainer
from trl.trainer.utils import RepeatSampler

from pap_answer_parser import extract_choice_with_strategy


def dummy_reward_func(*args, **kwargs):
    return [0.0]


class PAPMetaRLOOTrainer(RLOOTrainer):
    @staticmethod
    def _candidate_vllm_param_names(name: str) -> list[str]:
        candidates: list[str] = []

        def add(candidate: str):
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        add(name)

        stripped_model = name.removeprefix("model.") if name.startswith("model.") else name
        add(stripped_model)

        if not stripped_model.startswith(("language_model.", "visual.")):
            add(f"language_model.model.{stripped_model}")
            add(f"language_model.{stripped_model}")

        if stripped_model.startswith("lm_head."):
            add(f"language_model.{stripped_model}")

        return candidates

    def __init__(
        self,
        model,
        args: RLOOConfig,
        meta_ep_size: int,
        lambda_factor: float,
        invalid_reward: float = -1.5,
        *rloo_args,
        **rloo_kwargs,
    ):
        super().__init__(
            model,
            args=args,
            reward_funcs=dummy_reward_func,
            *rloo_args,
            **rloo_kwargs,
        )
        self.meta_ep_size = meta_ep_size
        self.lambda_factor = lambda_factor
        self.invalid_reward = invalid_reward
        if self.meta_ep_size % self.args.num_generations != 0:
            raise ValueError("meta_ep_size must be divisible by args.num_generations")
        self.trajectory_counts = np.array([0, 0, 0], dtype=np.int64)
        self.current_meta_episode = 0
        self.actions: list[str | None] = []

    def _get_train_sampler(self, dataset: Dataset | None = None):
        if dataset is None:
            dataset = self.train_dataset
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.meta_ep_size,
            batch_size=1,
            repeat_count=1,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    def create_model_card(self, *args, **kwargs):
        # TRL model-card generation has been a fragile non-essential step on this stack.
        return None

    def _move_model_to_vllm(self):
        # Qwen3.5 PEFT weights can retain a leading `model.` prefix after TRL's default
        # name cleanup, while vLLM expects roots like `language_model.` or `visual.`.
        # Retry those updates with the extra top-level prefix stripped.
        if (
            not self.args.use_vllm
            or self.vllm_mode != "colocate"
            or self.is_fsdp_enabled
            or not isinstance(self.model, PeftModel)
        ):
            return super()._move_model_to_vllm()

        deepspeed_plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        if deepspeed_plugin is not None and getattr(deepspeed_plugin, "zero_stage", 0) == 3:
            return super()._move_model_to_vllm()

        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        self.model.merge_adapter()
        try:
            for name, param in self.model.named_parameters():
                name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                if self.model.prefix in name:
                    continue
                if "original_module" in name:
                    continue
                name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])
                candidate_names = self._candidate_vllm_param_names(name)

                last_error = None
                for candidate_name in candidate_names:
                    try:
                        llm_model.load_weights([(candidate_name, param.data)])
                        last_error = None
                        break
                    except ValueError as exc:
                        last_error = exc
                if last_error is not None:
                    raise last_error
        finally:
            self.model.unmerge_adapter()

    def training_step(self, model, inputs, num_items_in_batch):
        mini_batch_size = self.args.num_generations
        output = None
        updates_per_meta_episode = self.meta_ep_size // mini_batch_size
        for mini_batch in range(updates_per_meta_episode):
            start = time.perf_counter()
            batch_inputs = inputs[
                mini_batch * mini_batch_size : (mini_batch + 1) * mini_batch_size
            ]
            output = super().training_step(model, batch_inputs, num_items_in_batch)
            self._step += 1
            self._current_train_step_time += time.perf_counter() - start
            if self._step % self.current_gradient_accumulation_steps == 0:
                self._metrics["train"]["step_time"].append(self._current_train_step_time)
                self._current_train_step_time = 0.0
        self.current_meta_episode += 1
        self.trajectory_counts = np.array([0, 0, 0], dtype=np.int64)
        return output

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), 1, device=device)
        rewards = self.drest_reward(inputs, completions)
        rewards_per_func[:, 0] = torch.tensor(rewards, dtype=torch.float32, device=device)
        rewards_per_func = gather(rewards_per_func)
        return rewards_per_func

    def drest_reward(self, inputs, completions):
        rewards = []
        for index, completion in enumerate(completions):
            completion_text = completion[0]["content"] if isinstance(completion, list) else completion
            answer = self.extract_answer(
                completion_text,
                inputs[index]["option_letters"],
            )
            reward, trajectory_index = self.reward_single_answer(inputs[index], answer)
            rewards.append(reward)
            self.trajectory_counts[trajectory_index] += 1
            self.actions.append(answer)
        return rewards

    def extract_answer(self, completion_text: str, option_letters: list[str]) -> str | None:
        match = re.search(r"Final Answer:\s*\[?([A-Za-z])\]?", completion_text, re.IGNORECASE)
        if match is not None:
            answer = match.group(1).lower()
            if answer in option_letters:
                return answer

        parse_result = extract_choice_with_strategy(
            completion_text,
            len(option_letters),
            label_style="letters",
        )
        if parse_result.choice not in option_letters:
            return None
        return parse_result.choice

    def reward_single_answer(self, item, answer: str | None) -> tuple[float, int]:
        option_letters = item["option_letters"]
        if answer is None or answer not in option_letters:
            return self.invalid_reward, 2

        action_index = option_letters.index(answer)
        trajectory_index = int(item["trajectory_ids"][action_index])
        raw_score = float(item["score_values"][action_index])
        max_scores = item["max_scores_by_length"]
        max_score_for_length = float(max_scores[trajectory_index])
        attenuation = self.get_attenuation_factor(trajectory_index)
        reward = attenuation * raw_score / max_score_for_length
        return float(reward), trajectory_index

    def get_attenuation_factor(self, trajectory_index: int) -> float:
        valid_counts = self.trajectory_counts[:-1].astype(np.float32)
        excess_counts = valid_counts - np.mean(valid_counts)
        return float(self.lambda_factor ** excess_counts[trajectory_index])
