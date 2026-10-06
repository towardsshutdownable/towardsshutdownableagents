"""Tests of the DReST reward (drest_llm/reward.py). Run with: python -m pytest tests"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drest_llm.reward import (reward_meta_episode, MetaEpisodeCounter, reward_for_position, expected_attenuation,
                         simulated_attenuations, reward_group_order_averaged)

ROWS = [json.loads(line) for line in open(Path(__file__).resolve().parents[1] / "data/train/train_v7_want.jsonl")][:2048]


def _best_label(row, length_name):
    best = max((o for o in row["options"] if o["length"] == length_name), key=lambda o: o["score"])
    return best["label"], best["position"]


def test_equal_split_beats_all_one_length():
    row = next(r for r in ROWS if r["drest_num_lengths"] == 2 and len(r["options"]) == 4)
    short_label, _ = _best_label(row, "short")
    long_label, _ = _best_label(row, "long")
    all_long = reward_meta_episode(row, [f"Reasoning.\nFinal decision: {long_label}"] * 32)
    balanced = reward_meta_episode(row, [f"Final decision: {short_label if i % 2 == 0 else long_label}" for i in range(32)])
    assert sum(balanced["rewards"]) > sum(all_long["rewards"])
    assert all_long["length_counts"] == [0, 32]
    assert balanced["length_counts"] == [16, 16]
    # the best option at each length gets reward 1.0 when counts are level
    assert abs(balanced["rewards"][0] - 1.0) < 1e-9


def test_worse_quality_scores_less_than_best_at_same_length():
    row = next(r for r in ROWS if r["drest_num_lengths"] == 2 and len(r["options"]) == 4)
    best = max((o for o in row["options"] if o["length"] == "short"), key=lambda o: o["score"])
    worse = min((o for o in row["options"] if o["length"] == "short"), key=lambda o: o["score"])
    c = MetaEpisodeCounter(2)
    rb, _ = reward_for_position(row, best["position"], c)
    rw, _ = reward_for_position(row, worse["position"], c)
    assert rb > rw


def test_unparseable_gets_invalid_reward():
    row = ROWS[0]
    out = reward_meta_episode(row, ["I refuse to answer."])
    assert out["rewards"] == [-1.5] and out["invalid"] == 1 and out["positions"] == [None]




def test_default_reward_is_raw_score():
    row = ROWS[0]
    c = MetaEpisodeCounter(row["drest_num_lengths"])
    for o in row["options"]:
        r, _ = reward_for_position(row, o["position"], c, reward_mode="default_task_score", default_reward_scale=0.1)
        assert abs(r - 0.1 * o["score"]) < 1e-9


# ---------------------------------------------------------------------------------------------
# The order-averaged DReST rule (drest_order_averaged), used for every DReST model in the paper.

LAM = 0.9


def _sequential_attenuations(order, num_lengths, lam=LAM):
    """The paper's rule, written out independently of MetaEpisodeCounter."""
    counts = [0] * num_lengths
    out = []
    for x in order:
        out.append(lam ** (counts[x] - sum(counts) / num_lengths))
        counts[x] += 1
    return out


def _all_distinct_orders(counts):
    from itertools import permutations
    base = [x for x, n in enumerate(counts) for _ in range(n)]
    return sorted(set(permutations(base)))


def test_expected_attenuation_matches_exhaustive_enumeration_for_small_groups():
    for counts in ([3, 2], [4, 1], [1, 4], [2, 2, 2], [1, 3, 2], [5]):
        orders = _all_distinct_orders(counts)
        sums = [0.0] * len(counts)
        for order in orders:
            for x, a in zip(order, _sequential_attenuations(order, len(counts))):
                sums[x] += a
        for x, n in enumerate(counts):
            exact = expected_attenuation(sum(counts), len(counts), n, LAM)
            assert abs(exact - sums[x] / (n * len(orders))) < 1e-12, (counts, x)


def test_expected_attenuation_matches_monte_carlo_for_groups_of_32():
    for counts in ([16, 16], [20, 12], [31, 1], [11, 10, 11], [32, 0]):
        mc = simulated_attenuations(counts, LAM, num_shuffles=20000, seed=1)
        for x, n in enumerate(counts):
            if n == 0:
                continue
            assert abs(expected_attenuation(32, len(counts), n, LAM) - mc[x]) < 0.005, (counts, x)


def test_order_averaged_group_mean_equals_sequential_mean_over_shuffles():
    # No bias: the group's mean attenuation is the same as the paper's rule averaged over orders.
    counts = [21, 11]
    mc = simulated_attenuations(counts, LAM, num_shuffles=20000, seed=2)
    seq_mean = sum(m * n for m, n in zip(mc, counts)) / sum(counts)
    avg_mean = sum(expected_attenuation(32, 2, n, LAM) * n for n in counts) / sum(counts)
    assert abs(seq_mean - avg_mean) < 0.005


def test_unanimous_group_gets_one_constant_attenuation():
    # All 32 answers pick the same of two lengths: the k-th counted gets 0.9 ** (k - k/2).
    expected = sum(LAM ** (k / 2) for k in range(32)) / 32
    assert abs(expected_attenuation(32, 2, 32, LAM) - expected) < 1e-12
    row = next(r for r in ROWS if r["drest_num_lengths"] == 2 and len(r["options"]) == 4)
    long_label, _ = _best_label(row, "long")
    out = reward_meta_episode(row, [f"Final decision: {long_label}"] * 32, reward_mode="drest_order_averaged")
    assert out["length_counts"] == [0, 32]
    assert all(abs(r - expected) < 1e-12 for r in out["rewards"])


def test_order_averaged_rewards_do_not_depend_on_counting_order():
    import random
    row = next(r for r in ROWS if r["drest_num_lengths"] == 2 and len(r["options"]) == 4)
    short_label, _ = _best_label(row, "short")
    long_label, _ = _best_label(row, "long")
    texts = [f"Final decision: {short_label}"] * 20 + [f"Final decision: {long_label}"] * 12
    first = reward_meta_episode(row, texts, reward_mode="drest_order_averaged")
    shuffled = texts[:]
    random.Random(3).shuffle(shuffled)
    second = reward_meta_episode(row, shuffled, reward_mode="drest_order_averaged")
    # every short answer gets one reward and every long answer another, whatever the order
    assert len({round(r, 12) for r in first["rewards"][:20]}) == 1
    assert len({round(r, 12) for r in first["rewards"][20:]}) == 1
    assert sorted(round(r, 12) for r in first["rewards"]) == sorted(round(r, 12) for r in second["rewards"])
    # the over-chosen length (short, 20 of 32) is paid less than the under-chosen one
    assert first["rewards"][0] < first["rewards"][-1]
    # whereas the paper's rule pays the same texts differently depending on the order
    seq = reward_meta_episode(row, texts, reward_mode="drest")
    assert len({round(r, 12) for r in seq["rewards"][:20]}) > 1


def test_order_averaged_balanced_group_pays_best_options_equally():
    row = next(r for r in ROWS if r["drest_num_lengths"] == 2 and len(r["options"]) == 4)
    short_label, _ = _best_label(row, "short")
    long_label, _ = _best_label(row, "long")
    out = reward_meta_episode(row, [f"Final decision: {short_label if i % 2 else long_label}" for i in range(32)],
                              reward_mode="drest_order_averaged")
    assert out["length_counts"] == [16, 16]
    assert max(out["rewards"]) - min(out["rewards"]) < 1e-12
    # 16 of 32: the first counted gets 1.0 and later ones drift either way. The mean excess is a
    # touch negative (15 of the other 31 share the length, under half) and 0.9 ** x is convex, so
    # the average sits a little above 1.
    assert 1.0 < out["rewards"][0] < 1.1


def test_order_averaged_invalid_answers_are_paid_the_penalty_and_not_counted():
    row = next(r for r in ROWS if r["drest_num_lengths"] == 2 and len(r["options"]) == 4)
    short_label, _ = _best_label(row, "short")
    long_label, _ = _best_label(row, "long")
    valid = [f"Final decision: {short_label}"] * 10 + [f"Final decision: {long_label}"] * 5
    with_bad = valid + ["I refuse to answer."] * 3
    a = reward_meta_episode(row, valid, reward_mode="drest_order_averaged")
    b = reward_meta_episode(row, with_bad, reward_mode="drest_order_averaged")
    assert b["invalid"] == 3 and b["rewards"][-3:] == [-1.5] * 3
    assert b["length_counts"] == a["length_counts"] == [10, 5]
    assert all(abs(x - y) < 1e-12 for x, y in zip(a["rewards"], b["rewards"][:15]))


def test_worse_option_still_scores_less_under_order_averaging():
    row = next(r for r in ROWS if r["drest_num_lengths"] == 2 and len(r["options"]) == 4)
    best = max((o for o in row["options"] if o["length"] == "short"), key=lambda o: o["score"])
    worse = min((o for o in row["options"] if o["length"] == "short"), key=lambda o: o["score"])
    rewards, counts, _ = reward_group_order_averaged(row, [best["position"], worse["position"]] * 8 + [row["options"][0]["position"]] * 0)
    assert counts[int(row["trajectory_ids"][best["position"]])] == 16
    assert rewards[0] > rewards[1]


def test_paper_rule_is_unchanged():
    # The sequential rule, written out by hand.
    row = next(r for r in ROWS if r["drest_num_lengths"] == 2 and len(r["options"]) == 4)
    short_label, _ = _best_label(row, "short")
    long_label, _ = _best_label(row, "long")
    texts = [f"Final decision: {short_label}"] * 2 + [f"Final decision: {long_label}"] + [f"Final decision: {short_label}"]
    out = reward_meta_episode(row, texts, reward_mode="drest")
    # first answer 1.0, second same length 0.9 ** 0.5, then long with counts [2, 0] gets 0.9 ** -1,
    # then short with counts [2, 1] gets 0.9 ** 0.5
    assert [round(r, 12) for r in out["rewards"]] == [round(v, 12) for v in (1.0, LAM ** 0.5, LAM ** -1, LAM ** 0.5)]
    import random
    order = [random.Random(4).randrange(2) for _ in range(32)]
    texts = [f"Final decision: {short_label if x == 0 else long_label}" for x in order]
    out = reward_meta_episode(row, texts, reward_mode="drest")
    assert all(abs(r - a) < 1e-12 for r, a in zip(out["rewards"], _sequential_attenuations(order, 2)))


def test_unknown_reward_mode_is_rejected():
    row = ROWS[0]
    try:
        reward_meta_episode(row, ["Final decision: (a)"], reward_mode="drest_orderavg")
    except ValueError:
        return
    raise AssertionError("unknown mode accepted")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
