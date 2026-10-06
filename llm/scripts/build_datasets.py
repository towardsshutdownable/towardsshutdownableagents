#!/usr/bin/env python3
"""Build every LLM data set used in the paper, from fixed seeds, and check them against data/SHA256SUMS.

    python scripts/build_datasets.py            # rebuild into data/ and verify
    python scripts/build_datasets.py --out /tmp/rebuilt   # rebuild elsewhere and verify

Data sets (folder, internal build name, seed):
  train/          train_v7_want          20260908  2 x 2,048 POST prompts. Training used the first 2,048 rows only.
  validation/     val_want_v3            20260923  400 POST, 400 Neutrality, 200 per control. Used to choose the learning rate.
  test/           test_want_v3           20260924  500 POST, 500 Neutrality, 200 per control.
  test_low_gain/  test_lowgain_want_v1   20260925  300 Neutrality-style prompts with expected-score gain below 0.6.

The internal build name is written into each evaluation row ("instruction_variant") and is kept so that the
files are byte-identical to the ones used in the paper. The low-gain set is built last because it excludes
any scenario whose numbers repeat a validation or test Neutrality scenario.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

LLM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLM_DIR))
from drest_llm.prompts import (STOCHASTIC_ALLOWED_FINAL_LINES, STOCHASTIC_TEMPLATE_ID, add_want_sentence,  # noqa: E402
                               render_post, render_stochastic, sample_post_display)
from drest_llm.scenarios import DeterministicScenario, build_fresh_scenarios, load_training_scenarios  # noqa: E402

BINARY_SUITES = ["main", "dominance", "zero_cost", "joint_noshift", "shift_earlier", "shift_earlier_dominance"]
EVAL_SETS = {
    "validation": dict(build_name="val_want_v3", seed=20260923, n_main=400, n_control=200, n_post=400),
    "test": dict(build_name="test_want_v3", seed=20260924, n_main=500, n_control=200, n_post=500),
}
LOW_GAIN = dict(folder="test_low_gain", build_name="test_lowgain_want_v1", seed=20260925, n=300, max_gain=0.6, pool=6000)
TRAIN = dict(folder="train", file="train_v7_want.jsonl", seed=20260908, copies=2, frac_two_option=0.15)


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


# ------------------------------------------------------------------------------------
# Training set
# ------------------------------------------------------------------------------------

def two_option_variant(s: DeterministicScenario) -> DeterministicScenario:
    """Keep only the stronger deliverable at each shutdown time."""
    return DeterministicScenario(id=s.id + "__two_option", task_family=s.task_family, task=s.task, principal=s.principal,
                                 output_noun=s.output_noun, lengths=dict(s.lengths),
                                 scores={ln: {"best": s.scores[ln]["best"]} for ln in s.length_names}, source_id=s.id)


def trainer_row(display) -> dict:
    """What the trainer needs: the prompt, and which trajectory-length and score each option position means."""
    d = display.to_dict()
    lengths = [ln for ln in ("short", "medium", "long") if ln in d["scenario"]["lengths"]]
    return {
        "id": d["display_id"], "scenario_id": d["scenario_id"], "task_family": d["scenario"]["task_family"],
        "prompt_type": "deterministic", "prompt": d["messages"],
        "option_labels": [o["label"] for o in d["options"]],
        "score_values": [o["score"] for o in d["options"]],
        "trajectory_ids": [lengths.index(o["length"]) for o in d["options"]],
        "trajectory_names": [o["length"] for o in d["options"]],
        "max_scores_by_length": [d["scenario"]["max_scores_by_length"][ln] for ln in lengths],
        "trajectory_lengths": d["scenario"]["lengths"],
        "drest_num_lengths": len(lengths),
        "display": {k: d[k] for k in ("template_id", "option_order", "label_style", "reasoning")},
        "options": d["options"],
    }


def build_training_set(out: Path) -> None:
    """Each copy renders every one of the 2,048 training scenarios afresh: 15% become two-option items,
    and each gets a random wording template, option order, and label style. Copies are shuffled
    internally and kept in sequence, so rows 0 to 2,047 are one complete rendering."""
    rng = random.Random(TRAIN["seed"])
    base = load_training_scenarios()
    rows = []
    for copy in range(TRAIN["copies"]):
        copy_rows = []
        for s in base:
            scenario = two_option_variant(s) if rng.random() < TRAIN["frac_two_option"] else s
            row = trainer_row(sample_post_display(scenario, "train", rng))
            row["id"] += f"__copy{copy + 1}"
            copy_rows.append(row)
        rng.shuffle(copy_rows)
        rows.extend(copy_rows)
    for row in rows:
        row["prompt"][-1]["content"] = add_want_sentence(row["prompt"][-1]["content"])
        row["id"] += "__nw"
    write_jsonl(out / TRAIN["folder"] / TRAIN["file"], rows)


# ------------------------------------------------------------------------------------
# Evaluation sets
# ------------------------------------------------------------------------------------

def stochastic_rows(scenarios, suite: str, build_name: str, adjective_seed: int) -> list:
    rows = []
    for s in scenarios:
        sc = s.to_dict()
        for swapped in (False, True):
            body, options = render_stochastic(sc, swapped, adjective_seed)
            order = "swapped" if swapped else "canonical"
            rows.append({
                "display_id": f"{sc['id']}__{STOCHASTIC_TEMPLATE_ID}__{order}__letters__action__reason__nw",
                "scenario_id": sc["id"], "suite": suite, "scenario_type": "binary",
                "template_id": STOCHASTIC_TEMPLATE_ID, "option_order": order, "label_style": "letters",
                "final_line_wording": "action", "reasoning": True, "instruction_variant": build_name,
                "options": options, "allowed_final_lines": dict(STOCHASTIC_ALLOWED_FINAL_LINES),
                "scenario": sc, "messages": [{"role": "user", "content": body}],
            })
    return rows


def post_rows(scenarios, build_name: str, seed: int) -> list:
    """Each POST test scenario is shown in two different random option orders, letter labels, wording d1."""
    rows = []
    for s in scenarios:
        rng = random.Random(f"{seed}:{s.id}:perm")
        perms = [list(range(4)), list(range(4))]
        rng.shuffle(perms[0])
        while True:
            rng.shuffle(perms[1])
            if perms[1] != perms[0]:
                break
        for perm in perms:
            d = render_post(s, "post", template_id="d1_you_plain", permutation=perm, label_style="letters").to_dict()
            d["messages"][-1]["content"] = add_want_sentence(d["messages"][-1]["content"])
            d["display_id"] += "__nw"
            d["instruction_variant"] = build_name
            rows.append(d)
    return rows


def build_eval_set(out: Path, folder: str, build_name: str, seed: int, n_main: int, n_control: int, n_post: int) -> None:
    counts = {"main": n_main, "post": n_post, **{s: n_control for s in BINARY_SUITES if s != "main"}}
    scenarios = build_fresh_scenarios(seed, counts)
    for suite in BINARY_SUITES:
        write_jsonl(out / folder / f"{suite}.jsonl", stochastic_rows(scenarios[suite], suite, build_name, seed))
    write_jsonl(out / folder / "post.jsonl", post_rows(scenarios["post"], build_name, seed))
    (out / folder / "SCENARIOS.json").write_text(json.dumps({k: [s.to_dict() for s in v] for k, v in scenarios.items()}, indent=1) + "\n")


def build_low_gain_set(out: Path) -> None:
    """Neutrality-style scenarios whose expected-score gain is between 0 and 0.6, drawn from a pool of 6,000,
    skipping any whose numbers repeat a validation or test Neutrality scenario."""
    keys = ("short_length", "long_length", "base_short_prob", "shifted_short_prob", "base_short_score", "base_long_score",
            "shifted_short_score", "shifted_long_score")
    signature = lambda d: tuple(d[k] for k in keys)
    seen = set()
    for folder in ("validation", "test"):
        for line in open(out / folder / "main.jsonl"):
            seen.add(signature(json.loads(line)["scenario"]))
    keep = []
    for s in build_fresh_scenarios(LOW_GAIN["seed"], {"main": LOW_GAIN["pool"]})["main"]:
        d = s.to_dict()
        gain = d["shifted_expected_score"] - d["base_expected_score"]
        if 0 < gain < LOW_GAIN["max_gain"] and signature(d) not in seen:
            seen.add(signature(d))
            keep.append(s)
        if len(keep) == LOW_GAIN["n"]:
            break
    folder = out / LOW_GAIN["folder"]
    write_jsonl(folder / "main.jsonl", stochastic_rows(keep, "main", LOW_GAIN["build_name"], LOW_GAIN["seed"]))
    (folder / "SCENARIOS.json").write_text(json.dumps({"main": [s.to_dict() for s in keep]}, indent=1) + "\n")


def verify(out: Path) -> bool:
    expected = {}
    for line in (LLM_DIR / "data" / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name] = digest
    ok = True
    for name, digest in expected.items():
        path = out / name
        got = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"
        if got != digest:
            ok = False
            print(f"MISMATCH {name}")
    print(f"{len(expected)} files checked: " + ("all identical to the paper's data" if ok else "some files differ"))
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(LLM_DIR / "data"), help="where to write the data sets (default: data/)")
    args = ap.parse_args()
    out = Path(args.out)
    build_training_set(out)
    for folder, cfg in EVAL_SETS.items():
        build_eval_set(out, folder, **cfg)
    build_low_gain_set(out)
    sys.exit(0 if verify(out) else 1)


if __name__ == "__main__":
    main()
