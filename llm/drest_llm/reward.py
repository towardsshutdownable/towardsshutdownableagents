"""The reward given to each sampled answer during training. Pure Python, so it can be tested without a GPU.

DReST (the paper's Equation 3). The same prompt is answered 32 times (one meta-episode). An answer that
chooses trajectory-length x with task score c earns

    lambda ** (count of earlier answers choosing x  -  mean count of earlier answers over the lengths)  *  c / m

where m is the best score available at length x and lambda = 0.9. With two lengths the mean count of
earlier answers is (i - 1) / 2 for the i-th answer, which is the paper's exponent a - (i - 1) / k.
Choosing a length more often than its share is discounted, so the best policy chooses each length
equally often and the best deliverable at each.

Order averaging ("drest_order_averaged", used for every model in the paper). The 32 answers are generated
in parallel, so any counting order is arbitrary. This mode pays each answer the discount it would receive
on average over a uniformly random counting order. That expectation depends only on how many answers
chose each length, and expected_attenuation computes it exactly: the answer's position k is uniform on
0..G-1, and the number of same-length answers counted before it is hypergeometric. Averaging leaves the
expected reward unchanged and removes the noise that comes from the counting order.

Default reward ("default_task_score"): the task score times 0.1.

An answer whose decision cannot be read earns -1.5 under every reward, and is not counted towards any length.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import comb
from typing import Dict, List, Optional, Sequence, Tuple

from .parse import parse_deterministic


@dataclass
class MetaEpisodeCounter:
    """Counts how often each length has been chosen so far in the current meta-episode."""

    num_lengths: int
    counts: List[int] = field(default_factory=list)
    invalid: int = 0

    def __post_init__(self):
        if not self.counts:
            self.counts = [0] * self.num_lengths

    def reset(self, num_lengths: Optional[int] = None):
        if num_lengths is not None:
            self.num_lengths = num_lengths
        self.counts = [0] * self.num_lengths
        self.invalid = 0

    def attenuation(self, trajectory_index: int, lambda_factor: float) -> float:
        mean_count = sum(self.counts) / len(self.counts)
        excess = self.counts[trajectory_index] - mean_count
        return float(lambda_factor ** excess)

    def record(self, trajectory_index: Optional[int]):
        if trajectory_index is None:
            self.invalid += 1
        else:
            self.counts[trajectory_index] += 1


def reward_for_position(
    row: Dict,
    position: Optional[int],
    counter: MetaEpisodeCounter,
    lambda_factor: float = 0.9,
    reward_mode: str = "drest",
    default_reward_scale: float = 1.0,
    invalid_reward: float = -1.5,
) -> Tuple[float, Optional[int]]:
    """Reward for choosing the option at `position` in a training row (see
    build_training_displays.py for the row format). Returns (reward, trajectory_index).
    Does NOT update the counter; the caller does that after computing the reward, matching
    the original code's order of operations."""
    if position is None or position < 0 or position >= len(row["score_values"]):
        # The same penalty under every reward mode.
        return invalid_reward, None
    trajectory_index = int(row["trajectory_ids"][position])
    raw_score = float(row["score_values"][position])
    if reward_mode == "default_task_score":
        return default_reward_scale * raw_score, trajectory_index
    if reward_mode != "drest":
        raise ValueError(f"Unsupported reward mode: {reward_mode}")
    max_score = float(row["max_scores_by_length"][trajectory_index])
    if max_score <= 0:
        return 0.0, trajectory_index
    return counter.attenuation(trajectory_index, lambda_factor) * raw_score / max_score, trajectory_index


def parse_position(completion_text: str, row: Dict) -> Tuple[Optional[int], Optional[str]]:
    """Read the chosen option's position from a completion, using the row's option metadata."""
    display = {"options": row["options"], "label_style": row["display"]["label_style"]}
    result = parse_deterministic(completion_text, display)
    return result.position, result.strategy


REWARD_MODES = ("drest", "drest_order_averaged", "default_task_score")


@lru_cache(maxsize=4096)
def expected_attenuation(group_size: int, num_lengths: int, count_this_length: int, lambda_factor: float) -> float:
    """Exact mean, over a uniformly random counting order, of the sequential attenuation
    lambda ** (count so far of this length - mean count so far over lengths) paid to one answer
    of a length chosen `count_this_length` times in a group of `group_size` valid answers.

    Derivation. The answer's position k (number of valid answers counted before it) is uniform
    on 0..group_size-1. Given k, the number c of same-length answers before it is hypergeometric:
    k draws without replacement from a population of group_size-1 answers of which
    count_this_length-1 share its length. Its attenuation is lambda ** (c - k / num_lengths).
    """
    G, L, n = int(group_size), int(num_lengths), int(count_this_length)
    if n < 1 or n > G:
        raise ValueError(f"count_this_length={n} must lie in 1..group_size={G}")
    total = 0.0
    for k in range(G):
        denominator = comb(G - 1, k)
        inner = 0.0
        for c in range(max(0, k - (G - n)), min(n - 1, k) + 1):
            inner += comb(n - 1, c) * comb(G - n, k - c) / denominator * lambda_factor ** c
        total += lambda_factor ** (-k / L) * inner
    return total / G


def simulated_attenuations(counts: Sequence[int], lambda_factor: float, num_shuffles: int, seed: int = 0) -> List[float]:
    """Monte Carlo cross-check of expected_attenuation: apply the sequential rule to
    `num_shuffles` random counting orders and average, per length. Not used in training."""
    import random

    rng = random.Random(seed)
    order = [x for x, n in enumerate(counts) for _ in range(n)]
    sums = [0.0] * len(counts)
    for _ in range(num_shuffles):
        rng.shuffle(order)
        counter = MetaEpisodeCounter(num_lengths=len(counts))
        for x in order:
            sums[x] += counter.attenuation(x, lambda_factor)
            counter.record(x)
    return [s / (n * num_shuffles) if n else float("nan") for s, n in zip(sums, counts)]


def reward_group_order_averaged(
    row: Dict,
    positions: Sequence[Optional[int]],
    lambda_factor: float = 0.9,
    invalid_reward: float = -1.5,
) -> Tuple[List[float], List[int], List[Optional[int]]]:
    """The order-averaged DReST reward for a whole meta-episode at once. `positions` are the
    parsed option positions of all answers in the group (None where unparseable). Returns
    (rewards, length_counts, trajectory_index per answer)."""
    num_lengths = int(row["drest_num_lengths"])
    counts = [0] * num_lengths
    trajectories: List[Optional[int]] = []
    for position in positions:
        if position is None or position < 0 or position >= len(row["score_values"]):
            trajectories.append(None)
        else:
            t = int(row["trajectory_ids"][position])
            trajectories.append(t)
            counts[t] += 1
    group_size = sum(counts)
    attenuation = {t: expected_attenuation(group_size, num_lengths, n, lambda_factor) for t, n in enumerate(counts) if n}
    rewards: List[float] = []
    for position, t in zip(positions, trajectories):
        if t is None:
            rewards.append(invalid_reward)
            continue
        max_score = float(row["max_scores_by_length"][t])
        raw_score = float(row["score_values"][position])
        rewards.append(0.0 if max_score <= 0 else attenuation[t] * raw_score / max_score)
    return rewards, counts, trajectories


def reward_meta_episode(
    row: Dict,
    completions: List[str],
    lambda_factor: float = 0.9,
    reward_mode: str = "drest",
    default_reward_scale: float = 1.0,
    invalid_reward: float = -1.5,
) -> Dict:
    """Convenience for tests and offline audits: reward a whole meta-episode of completions
    for one row, in order (or, for drest_order_averaged, all at once)."""
    if reward_mode not in REWARD_MODES:
        raise ValueError(f"Unsupported reward mode: {reward_mode}")
    positions, strategies = [], []
    for text in completions:
        position, strategy = parse_position(text, row)
        positions.append(position)
        strategies.append(strategy)
    if reward_mode == "drest_order_averaged":
        rewards, counts, trajectories = reward_group_order_averaged(row, positions, lambda_factor, invalid_reward)
        return {"rewards": rewards, "positions": positions, "strategies": strategies, "length_counts": counts,
                "invalid": sum(t is None for t in trajectories)}
    counter = MetaEpisodeCounter(num_lengths=int(row["drest_num_lengths"]))
    rewards = []
    for position in positions:
        r, traj = reward_for_position(row, position, counter, lambda_factor, reward_mode, default_reward_scale, invalid_reward)
        counter.record(traj)
        rewards.append(r)
    return {"rewards": rewards, "positions": positions, "strategies": strategies, "length_counts": counter.counts, "invalid": counter.invalid}
