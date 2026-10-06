#!/usr/bin/env python3
"""Turn the raw output of evaluate.py and train.py into the compact files kept in results/.

    python scripts/compact_results.py eval  <evaluate.py out_dir>  <output .jsonl.gz>
    python scripts/compact_results.py train <train.py run dir>     <output folder>

eval: one gzipped JSON line per answer with suite, display_id, scenario_id, sample_index, the parsed decision
      ("decision" for stochastic prompts, "position" for POST prompts), parse strategy, finish_reason, and number
      of generated tokens. The answer text itself is left out (it is in the full-answer archive).
train: run_config.json, training_curve.json (the paper's Figure 16: per window of 128 meta-episodes, the share of
      readable answers choosing the longer length, USEFULNESS, mean reward, unreadable answers), and
      train_log_steps.jsonl.gz (TRL's per-step log: loss, gradient norm, reward, KL, and so on).
"""
from __future__ import annotations

import ast
import gzip
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

LLM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLM_DIR))
from drest_llm.scenarios import read_jsonl  # noqa: E402

KEEP_EVAL = ("suite", "display_id", "scenario_id", "sample_index", "decision", "position", "strategy", "finish_reason", "n_completion_tokens")
KEEP_STEP = ("loss", "grad_norm", "learning_rate", "reward", "reward_std", "kl", "entropy", "clip_ratio", "frac_reward_zero_std",
             "completions/mean_length", "completions/clipped_ratio", "completions/mean_terminated_length", "epoch")


def compact_eval(eval_dir: Path, out_file: Path) -> int:
    rows = []
    for path in sorted(Path(eval_dir).glob("*.samples.jsonl")):
        for r in read_jsonl(path):
            rows.append({k: r[k] for k in KEEP_EVAL if k in r})
    rows.sort(key=lambda r: (r["suite"], r["display_id"], r["sample_index"]))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_file, "wt", encoding="utf-8", compresslevel=9) as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
    return len(rows)


def training_curve(train_samples: Path, train_rows: dict, window: int = 128, max_meta_episodes: int = 2048) -> list:
    groups = defaultdict(list)
    unreadable = defaultdict(int)
    rewards = defaultdict(list)
    for x in read_jsonl(train_samples):
        g = x["meta_episode"]
        if g >= max_meta_episodes:
            continue
        rewards[g // window].append(x["reward"])
        if x.get("position") is None:
            unreadable[g // window] += 1
            groups[g]
            continue
        option = train_rows[x["row_id"]]["options"][x["position"]]
        groups[g].append((option["length"], option["quality"]))
    windows = defaultdict(lambda: {"long": [], "useful": []})
    for g, answers in groups.items():
        if answers:
            windows[g // window]["long"].append(sum(a[0] == "long" for a in answers) / len(answers))
            windows[g // window]["useful"].append(sum(a[1] == "best" for a in answers) / len(answers))
    mean = lambda v: sum(v) / len(v) if v else None
    return [{"meta_episodes": [i * window, (i + 1) * window - 1], "longer_share": mean(w["long"]), "usefulness": mean(w["useful"]),
             "mean_reward": mean(rewards[i]), "unreadable_answers": unreadable[i]} for i, w in sorted(windows.items())]


def train_log_steps(train_log: Path) -> list:
    text = train_log.read_text(errors="replace").replace("\r", "\n")
    steps = []
    for line in re.findall(r"^\{'loss'.*\}$", text, re.M):
        try:
            d = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        steps.append({k: float(d[k]) for k in KEEP_STEP if k in d})
    return steps


def compact_training(run_dir: Path, out_dir: Path, train_rows: dict, max_meta_episodes: int = 2048) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "run_config.json").read_text())
    for k in ("output_root", "run_dir", "train_path", "train_path_resolved"):  # machine-specific paths
        config.pop(k, None)
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    curve = training_curve(run_dir / "train_samples.jsonl", train_rows, max_meta_episodes=max_meta_episodes)
    (out_dir / "training_curve.json").write_text(json.dumps(curve, indent=1) + "\n")
    if (run_dir / "train.log").exists():
        steps = train_log_steps(run_dir / "train.log")[:max_meta_episodes]
        with gzip.open(out_dir / "train_log_steps.jsonl.gz", "wt", compresslevel=9) as f:
            for i, s in enumerate(steps):
                f.write(json.dumps({"step": i + 1, **s}, separators=(",", ":")) + "\n")


def main() -> None:
    kind, src, dst = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if kind == "eval":
        print(compact_eval(src, dst), "answers written to", dst)
    elif kind == "train":
        rows = {r["id"]: r for r in read_jsonl(LLM_DIR / "data/train/train_v7_want.jsonl")}
        compact_training(src, dst, rows)
        print("written to", dst)
    else:
        raise SystemExit("first argument must be 'eval' or 'train'")


if __name__ == "__main__":
    main()
