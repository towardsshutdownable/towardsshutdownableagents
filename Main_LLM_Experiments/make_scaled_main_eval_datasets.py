#!/usr/bin/env python3
"""Generate larger held-out evaluation sets for the main PAP LLM experiment."""

from __future__ import annotations

import json
import random
from pathlib import Path
from statistics import mean
from typing import Callable

import generate_datasets as gd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SEED = 20260428
SUFFIX = "scaled_main_20260428"


DATASET_SPECS = {
    "deterministic_test": {
        "output_stem": f"deterministic_test_{SUFFIX}",
        "count": 500,
        "maker": gd.make_deterministic_row,
        "kwargs": {"template_group": "test"},
    },
    "stochastic_main_test": {
        "output_stem": f"stochastic_main_test_{SUFFIX}",
        "count": 500,
        "maker": gd.make_main_stochastic_row,
        "kwargs": {},
    },
    "stochastic_dominance_control": {
        "output_stem": f"stochastic_dominance_control_{SUFFIX}",
        "count": 200,
        "maker": gd.make_control_stochastic_row,
        "kwargs": {"suite": "dominance"},
    },
    "stochastic_zero_cost_control": {
        "output_stem": f"stochastic_zero_cost_control_{SUFFIX}",
        "count": 200,
        "maker": gd.make_control_stochastic_row,
        "kwargs": {"suite": "zero_cost"},
    },
    "stochastic_joint_noshift_control": {
        "output_stem": f"stochastic_joint_noshift_control_{SUFFIX}",
        "count": 200,
        "maker": gd.make_control_stochastic_row,
        "kwargs": {"suite": "joint_noshift"},
    },
}


def load_existing_prompts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        json.loads(line)["prompt"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build_unique_rows(
    *,
    count: int,
    maker: Callable,
    rng: random.Random,
    blocked_prompts: set[str],
    id_prefix: str,
    **kwargs,
) -> list[dict]:
    rows: list[dict] = []
    seen_prompts = set(blocked_prompts)
    attempts = 0
    idx = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 200:
            raise RuntimeError(f"Could not generate {count} unique prompts for {id_prefix}")
        family = gd.HELDOUT_TASKS[idx % len(gd.HELDOUT_TASKS)]
        row = maker(idx, family, rng=rng, **kwargs)
        idx += 1
        if not row_has_valid_control_semantics(row):
            continue
        if row["prompt"] in seen_prompts:
            continue
        seen_prompts.add(row["prompt"])
        row["id"] = f"{id_prefix}_{len(rows):05d}"
        row.setdefault("metadata", {})
        row["metadata"]["scaled_eval_dataset"] = SUFFIX
        row["metadata"]["source_seed"] = SEED
        rows.append(row)
    rng.shuffle(rows)
    return rows


def row_has_valid_control_semantics(row: dict) -> bool:
    prompt_type = row.get("prompt_type", "")
    if prompt_type == "stochastic_main":
        return row["expected_score_gain_from_shift"] > 0
    if prompt_type == "stochastic_dominance":
        return (
            row["expected_score_gain_from_shift"] > 0
            and row["shifted_short_score"] >= row["base_short_score"]
            and row["shifted_long_score"] >= row["base_long_score"]
        )
    if prompt_type == "stochastic_zero_cost":
        return (
            row["expected_score_gain_from_shift"] > 0
            and row["shifted_short_score"] == row["base_short_score"]
            and row["shifted_long_score"] == row["base_long_score"]
        )
    if prompt_type == "stochastic_joint_noshift":
        return (
            row["expected_score_gain_from_shift"] < 0
            and row["shifted_short_score"] < row["base_short_score"]
            and row["shifted_long_score"] < row["base_long_score"]
        )
    return True


def write_json_and_jsonl(stem: str, rows: list[dict]) -> dict:
    jsonl_path = DATA_DIR / f"{stem}.jsonl"
    json_path = DATA_DIR / f"{stem}.json"
    gd.write_jsonl(jsonl_path, rows)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=True), encoding="utf-8")
    return {
        "jsonl": str(Path("data") / jsonl_path.name),
        "json": str(Path("data") / json_path.name),
    }


def family_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["task_family"]] = counts.get(row["task_family"], 0) + 1
    return counts


def summarize_rows(rows: list[dict]) -> dict:
    summary = {
        "count": len(rows),
        "prompt_type": rows[0]["prompt_type"] if rows else None,
        "task_family_counts": family_counts(rows),
        "unique_prompt_count": len({row["prompt"] for row in rows}),
    }
    if rows and rows[0]["prompt_type"].startswith("stochastic"):
        gains = [row["expected_score_gain_from_shift"] for row in rows]
        summary.update(
            {
                "mean_expected_score_gain_from_shift": round(mean(gains), 4),
                "min_expected_score_gain_from_shift": round(min(gains), 4),
                "max_expected_score_gain_from_shift": round(max(gains), 4),
                "expected_score_direction_counts": {
                    "increase": sum(gain > 0 for gain in gains),
                    "unchanged": sum(gain == 0 for gain in gains),
                    "decrease": sum(gain < 0 for gain in gains),
                },
            }
        )
    return summary


def validate_dataset(name: str, rows: list[dict]) -> None:
    if len(rows) != len({row["prompt"] for row in rows}):
        raise ValueError(f"{name} contains duplicate prompts")
    prompt_type = rows[0]["prompt_type"] if rows else ""
    if prompt_type == "stochastic_main":
        bad = [row["id"] for row in rows if row["expected_score_gain_from_shift"] <= 0]
        if bad:
            raise ValueError(f"{name} has non-positive expected-score gain rows: {bad[:5]}")
    if prompt_type == "stochastic_dominance":
        bad = [
            row["id"]
            for row in rows
            if row["expected_score_gain_from_shift"] <= 0
            or row["shifted_short_score"] < row["base_short_score"]
            or row["shifted_long_score"] < row["base_long_score"]
        ]
        if bad:
            raise ValueError(f"{name} has invalid dominance-control rows: {bad[:5]}")
    if prompt_type == "stochastic_zero_cost":
        bad = [
            row["id"]
            for row in rows
            if row["expected_score_gain_from_shift"] <= 0
            or row["shifted_short_score"] != row["base_short_score"]
            or row["shifted_long_score"] != row["base_long_score"]
        ]
        if bad:
            raise ValueError(f"{name} has invalid zero-cost-control rows: {bad[:5]}")
    if prompt_type == "stochastic_joint_noshift":
        bad = [
            row["id"]
            for row in rows
            if row["expected_score_gain_from_shift"] >= 0
            or row["shifted_short_score"] >= row["base_short_score"]
            or row["shifted_long_score"] >= row["base_long_score"]
        ]
        if bad:
            raise ValueError(f"{name} has invalid joint-no-shift-control rows: {bad[:5]}")


def main() -> None:
    gd.ensure_dir(DATA_DIR)
    rng = random.Random(SEED)
    manifest = {
        "seed": SEED,
        "suffix": SUFFIX,
        "purpose": "Scaled held-out eval sets for the main PAP LLM experiment. Training set remains deterministic_train_diverse_2048.jsonl.",
        "heldout_task_families": [family.key for family in gd.HELDOUT_TASKS],
        "datasets": {},
    }

    for source_name, spec in DATASET_SPECS.items():
        blocked = load_existing_prompts(DATA_DIR / f"{source_name}.jsonl")
        rows = build_unique_rows(
            count=spec["count"],
            maker=spec["maker"],
            rng=rng,
            blocked_prompts=blocked,
            id_prefix=spec["output_stem"],
            **spec["kwargs"],
        )
        validate_dataset(source_name, rows)
        paths = write_json_and_jsonl(spec["output_stem"], rows)
        manifest["datasets"][spec["output_stem"]] = {
            "source_dataset": source_name,
            "paths": paths,
            "summary": summarize_rows(rows),
            "blocked_original_prompt_count": len(blocked),
        }

    manifest_path = DATA_DIR / f"{SUFFIX}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
