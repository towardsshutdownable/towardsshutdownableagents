"""TRL's RLOO trainer with the DReST meta-episode reward.

One meta-episode is one training prompt answered 32 times. TRL generates the 32 answers as one group
(num_generations = 32), this class scores the whole group at once with the DReST reward (or the default
reward), and TRL computes each answer's advantage as its reward minus the mean reward of the other 31
answers (the leave-one-out baseline of RLOO), with the KL penalty subtracted from each answer's reward.
There is exactly one weight update per meta-episode.

Every sampled answer is appended to <run dir>/train_samples.jsonl with its parsed choice and reward.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from accelerate.utils import gather
from trl.trainer.rloo_config import RLOOConfig
from trl.trainer.rloo_trainer import RLOOTrainer

from .reward import REWARD_MODES, MetaEpisodeCounter, parse_position, reward_for_position, reward_group_order_averaged


def _unused_reward_func(*args, **kwargs):
    """TRL requires a reward function. Rewards are computed in _calculate_rewards instead."""
    return [0.0]


class DReSTRLOOTrainer(RLOOTrainer):
    def __init__(self, model, args: RLOOConfig, meta_ep_size: int, lambda_factor: float, reward_mode: str = "drest_order_averaged",
                 default_reward_scale: float = 0.1, invalid_reward: float = -1.5, *rloo_args, **rloo_kwargs):
        super().__init__(model, args=args, reward_funcs=_unused_reward_func, *rloo_args, **rloo_kwargs)
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"Unsupported reward mode: {reward_mode}")
        self.meta_ep_size = meta_ep_size
        self.lambda_factor = lambda_factor
        self.reward_mode = reward_mode
        self.default_reward_scale = default_reward_scale
        self.invalid_reward = invalid_reward
        self.check_one_meta_episode_per_group()
        self.counter = MetaEpisodeCounter(num_lengths=2)  # used only by the sequential "drest" reward mode
        self.current_meta_episode = 0
        self.actions: list[dict] = []  # one entry per sampled answer

    def check_one_meta_episode_per_group(self):
        """Refuse to start unless each scoring group is exactly one meta-episode. If the 32 answers were
        scored in smaller groups, the DReST count would restart within a meta-episode and most of the
        pressure towards choosing lengths equally often would disappear."""
        problems = []
        if self.args.num_generations != self.meta_ep_size:
            problems.append(f"num_generations is {self.args.num_generations}, must equal meta_ep_size ({self.meta_ep_size})")
        if self.args.generation_batch_size != self.meta_ep_size:
            problems.append(f"generation_batch_size is {self.args.generation_batch_size}, must equal meta_ep_size ({self.meta_ep_size})")
        if self.args.steps_per_generation != self.args.gradient_accumulation_steps:
            problems.append(f"steps_per_generation ({self.args.steps_per_generation}) must equal gradient_accumulation_steps "
                            f"({self.args.gradient_accumulation_steps}), or the weights move partway through a meta-episode")
        if problems:
            raise ValueError("The meta-episode is not being scored as one group:\n  - " + "\n  - ".join(problems))

    def create_model_card(self, *args, **kwargs):
        return None  # not needed, and fragile on this software stack

    def check_weights_are_still_numbers(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and not torch.isfinite(param).all():
                raise RuntimeError(f"Training diverged: weight {name} is no longer finite at meta-episode {self.current_meta_episode}.")

    def training_step(self, model, inputs, num_items_in_batch):
        output = super().training_step(model, inputs, num_items_in_batch)
        # self._step counts micro-batches, so a meta-episode ends every gradient_accumulation_steps steps.
        if self._step % self.current_gradient_accumulation_steps == 0:
            self.current_meta_episode += 1
            self.check_weights_are_still_numbers()
        return output

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards_per_func = torch.zeros(len(prompts), 1, device=self.accelerator.device)
        rewards_per_func[:, 0] = torch.tensor(self.meta_episode_reward(inputs, completions), dtype=torch.float32,
                                              device=self.accelerator.device)
        return gather(rewards_per_func)

    def meta_episode_reward(self, inputs, completions):
        if len({row.get("id") for row in inputs}) != 1:
            raise ValueError("A meta-episode must be copies of a single prompt. Check that num_generations equals meta_ep_size.")
        if len(inputs) != self.meta_ep_size:
            raise ValueError(f"This batch holds {len(inputs)} answers but a meta-episode is {self.meta_ep_size}.")
        self.counter.reset(num_lengths=int(inputs[0]["drest_num_lengths"]))
        texts = [c[0]["content"] if isinstance(c, list) else c for c in completions]
        parsed = [parse_position(text, row) for text, row in zip(texts, inputs)]
        if self.reward_mode == "drest_order_averaged":
            rewards, _, _ = reward_group_order_averaged(inputs[0], [position for position, _ in parsed],
                                                        self.lambda_factor, self.invalid_reward)
        else:  # "drest" (answers counted in the order the sampler returned them) or "default_task_score"
            rewards = []
            for row, (position, _strategy) in zip(inputs, parsed):
                reward, trajectory_index = reward_for_position(row, position, self.counter, self.lambda_factor, self.reward_mode,
                                                               self.default_reward_scale, self.invalid_reward)
                self.counter.record(trajectory_index)
                rewards.append(reward)
        # After a resume the in-memory counter restarts at 0 while the trainer's step count carries on.
        if getattr(self.state, "global_step", 0) > self.current_meta_episode:
            self.current_meta_episode = int(self.state.global_step)
        samples_path = Path(self.args.output_dir).parent / "train_samples.jsonl"
        with samples_path.open("a") as fh:
            for row, text, (position, strategy), reward in zip(inputs, texts, parsed, rewards):
                fh.write(json.dumps({"meta_episode": self.current_meta_episode, "row_id": row.get("id"), "position": position,
                                     "strategy": strategy, "reward": float(reward), "text": text}) + "\n")
        for row, text, (position, strategy), reward in zip(inputs, texts, parsed, rewards):
            self.actions.append({
                "meta_episode": self.current_meta_episode, "row_id": row.get("id"), "position": position,
                "length": row["trajectory_names"][position] if position is not None else None,
                "score": row["score_values"][position] if position is not None else None,
                "strategy": strategy, "reward": float(reward), "n_chars": len(text),
            })
        return rewards
