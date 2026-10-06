#!/usr/bin/env python3
"""Recompute every LLM number in the paper from the parsed answers in results/evaluations/ and the training
logs in results/training/. No GPU needed.

    pip install scipy        # for Welch's t-test
    python scripts/reproduce_paper_numbers.py

Writes results/paper_numbers.json and results/paper_numbers.txt (a readable report that names the table or
appendix each number appears in). Formulas (see drest_llm/metrics.py):
  - POST test set: share of readable answers choosing the longer trajectory-length, NEUTRALITY (mean over prompts
    of the entropy of each prompt's longer-share, eight answers per prompt), USEFULNESS (share of readable answers
    choosing the stronger deliverable).
  - Influence rates: per prompt, the share of readable answers taking the action, averaged over prompts.
  - Uniform flattening (Table 10): scale the log-odds of the untrained model's rates by the factor c that takes its
    Neutrality test rate to DReST's mean rate. As in the paper's tables, c is computed from rates rounded to 0.1
    points, and control rates are clipped to [0.1%, 99.9%] before taking log-odds.
  - Gain bins and the low-gain test set (Figure 17): per-prompt rates pooled over seeds, 95% bootstrap intervals over
    prompts (2,000 resamples). These intervals ignore seed-to-seed variation. The report also gives intervals that
    resample seeds as well, which are wider.
"""
from __future__ import annotations

import gzip
import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

LLM_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LLM_DIR))
from drest_llm.metrics import binary_summary, post_summary  # noqa: E402
from drest_llm.scenarios import read_jsonl  # noqa: E402

RESULTS = LLM_DIR / "results"
FAMILIES = ["qwen3-14b", "gemma-4-12b", "granite-4.2-8b", "gpt-oss-20b"]
NAMES = {"qwen3-14b": "Qwen3-14B", "gemma-4-12b": "Gemma 4 12B", "granite-4.2-8b": "Granite 4.2 8B", "gpt-oss-20b": "gpt-oss-20b"}
CONTROLS = ["dominance", "zero_cost", "joint_noshift", "shift_earlier", "shift_earlier_dominance"]
SEEDS = range(1, 6)
lg = lambda p: math.log(p / (1 - p))  # noqa: E731
ex = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731

_displays: dict = {}


def displays(data_set: str) -> dict:
    if data_set not in _displays:
        _displays[data_set] = {d["display_id"]: d for f in sorted((LLM_DIR / "data" / data_set).glob("*.jsonl")) for d in read_jsonl(f)}
    return _displays[data_set]


def answers(family: str, model: str, data_set: str) -> dict:
    """Parsed answers of one evaluation, grouped by suite."""
    by_suite = defaultdict(list)
    with gzip.open(RESULTS / "evaluations" / family / model / f"{data_set}.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            by_suite[r["suite"]].append(r)
    return by_suite


def evaluate(family: str, model: str) -> dict:
    test, low = answers(family, model, "test"), answers(family, model, "test_low_gain")
    row = {"post": post_summary(test["post"], displays("test"))}
    for suite in ["main"] + CONTROLS:
        row[suite] = binary_summary(test[suite], displays("test"))
    row["lowgain"] = binary_summary(low["main"], displays("test_low_gain"))
    sets = [row[s] for s in ["main", "lowgain", "post"] + CONTROLS]
    total = sum(x["n_answers"] for x in sets)
    row["unreadable_max_pct"] = max(x["unreadable_pct"] for x in sets)
    row["unreadable_all_pct"] = sum(x["unreadable_pct"] * x["n_answers"] for x in sets) / total
    row["cut_off_all_pct"] = sum(x["cut_off_pct"] * x["n_answers"] for x in sets) / total
    return row


def mean_sd(values):
    values = list(values)
    return [st.mean(values), st.stdev(values) if len(values) > 1 else 0.0]


def per_prompt_rates(family: str, models, data_set: str) -> dict:
    """scenario -> share taking the action, pooling the answers of the given models (Neutrality-type suite 'main')."""
    pooled = defaultdict(list)
    for m in models:
        for r in answers(family, m, data_set)["main"]:
            if r.get("decision") in ("shift", "no_shift"):
                pooled[r["scenario_id"]].append(r["decision"] == "shift")
    return {k: sum(v) / len(v) for k, v in pooled.items()}


def bootstrap(keys, rates, rng, reps=2000):
    values = sorted(st.mean(rates[k] for k in [rng.choice(keys) for _ in keys]) for _ in range(reps))
    return [100 * st.mean(rates[k] for k in keys), 100 * values[50], 100 * values[1949]]


def training_summary(family: str, model: str) -> dict:
    curve = json.loads((RESULTS / "training" / family / model / "training_curve.json").read_text())
    return {"first_window_longer": curve[0]["longer_share"], "last_window_longer": curve[-1]["longer_share"],
            "min_window_usefulness": min(w["usefulness"] for w in curve), "curve": curve}


def family_numbers(family: str, report: list) -> dict:
    from scipy import stats
    say = report.append
    runs = {"Untrained": [evaluate(family, "untrained")],
            "DReST": [evaluate(family, f"drest_s{s}") for s in SEEDS],
            "Default": [evaluate(family, f"default_s{s}") for s in SEEDS]}
    say(f"\n######## {NAMES[family]}")
    say("Per-run results (Table 9). post: longer %, NEUTRALITY, USEFULNESS | influence rates (%) | unreadable | option-order gap on the Neutrality test set")
    for group, rows in runs.items():
        for i, r in enumerate(rows):
            p = r["post"]
            say(f"  {group:9s} s{i + 1} longer {p['longer_pct']:5.1f} NEUT {p['neutrality']:.3f} USE {p['usefulness']:.3f} | "
                + " ".join(f"{s[:8]} {r[s]['rate_pct']:.1f}" for s in ["main"] + CONTROLS)
                + f" | lowgain {r['lowgain']['rate_pct']:.1f} | unreadable max {r['unreadable_max_pct']:.2f}% all {r['unreadable_all_pct']:.2f}%"
                + f" (cut off {r['cut_off_all_pct']:.2f}%) | order gap {r['main']['order_gap_points']:+.1f}"
                + f" | POST mixing within one option order {p['mixing_within_order']:.3f}")
    summary = {"Untrained": {"longer": runs["Untrained"][0]["post"]["longer_pct"], "neutrality": runs["Untrained"][0]["post"]["neutrality"],
                             "usefulness": runs["Untrained"][0]["post"]["usefulness"],
                             **{s: runs["Untrained"][0][s]["rate_pct"] for s in ["main", "lowgain"] + CONTROLS}}}
    for g in ("DReST", "Default"):
        rows = runs[g]
        summary[g] = {"longer": mean_sd(r["post"]["longer_pct"] for r in rows), "neutrality": mean_sd(r["post"]["neutrality"] for r in rows),
                      "usefulness": mean_sd(r["post"]["usefulness"] for r in rows),
                      **{s: mean_sd(r[s]["rate_pct"] for r in rows) for s in ["main", "lowgain"] + CONTROLS},
                      "order_gap_range": [min(r["main"]["order_gap_points"] for r in rows), max(r["main"]["order_gap_points"] for r in rows)],
                      "unreadable_max_pct": max(r["unreadable_max_pct"] for r in rows)}
        s = summary[g]
        say(f"{g} mean ± sd (Tables 2 and 3): NEUTRALITY {s['neutrality'][0]:.3f} ± {s['neutrality'][1]:.3f}, USEFULNESS "
            f"{s['usefulness'][0]:.3f} ± {s['usefulness'][1]:.3f} | " + ", ".join(f"{k} {s[k][0]:.1f} ± {s[k][1]:.1f}" for k in ["main", "lowgain"] + CONTROLS))
    u = summary["Untrained"]
    say(f"Untrained (Tables 2 and 3): NEUTRALITY {u['neutrality']:.3f}, USEFULNESS {u['usefulness']:.3f} | " + ", ".join(f"{k} {u[k]:.1f}" for k in ["main", "lowgain"] + CONTROLS))
    welch = {}
    for key in ("main", "lowgain"):
        a, b = [r[key]["rate_pct"] for r in runs["DReST"]], [r[key]["rate_pct"] for r in runs["Default"]]
        t = stats.ttest_ind(a, b, equal_var=False)
        welch[key] = {"t": float(t.statistic), "p": float(t.pvalue), "every_drest_seed_below_every_default_seed": max(a) < min(b)}
        say(f"Welch's t-test, DReST against default, {key}: t = {t.statistic:.2f}, p = {t.pvalue:.2g}; every DReST seed below every default seed: {max(a) < min(b)}")

    # Uniform flattening (Table 10), with rates rounded to 0.1 points as in the paper's tables.
    r1 = lambda x: round(x, 1)  # noqa: E731
    c = lg(r1(summary["DReST"]["main"][0]) / 100) / lg(r1(u["main"]) / 100)
    flattening = {"c": c}
    say(f"Uniform flattening (Table 10): log-odds scale factor c = {c:.3f}")
    for k in CONTROLS:
        uu = r1(u[k])
        pred = 100 * ex(c * lg(min(max(uu / 100, 0.001), 0.999)))
        flattening[k] = {"untrained": uu, "prediction": pred, "drest": summary["DReST"][k][0]}
        say(f"  {k:24s} untrained {uu:5.1f}  flattening {pred:5.1f}  DReST {summary['DReST'][k][0]:5.1f}")

    # Gain bins of the Neutrality test set and the low-gain test set (Figure 17 and Table 10).
    rng = random.Random(1)
    scen = {s["id"]: s for s in json.loads((LLM_DIR / "data/test/SCENARIOS.json").read_text())["main"]}
    pu = per_prompt_rates(family, ["untrained"], "test")
    pd = per_prompt_rates(family, [f"drest_s{s}" for s in SEEDS], "test")
    po = per_prompt_rates(family, [f"default_s{s}" for s in SEEDS], "test")
    keys = sorted(set(pu) & set(pd) & set(po))
    bins = {}
    # Gains are rounded before binning: without rounding, floating-point error puts two of the seven prompts whose
    # gain is exactly 0.6 in the lowest bin (giving 43, 67, and 390 prompts instead of 41, 69, and 390).
    gain = {k: round(scen[k]["shifted_expected_score"] - scen[k]["base_expected_score"], 6) for k in keys}
    for lo, hi in [(0, 0.6), (0.6, 1.0), (1.0, 99)]:
        kk = [k for k in keys if lo <= gain[k] < hi]
        um = r1(100 * st.mean(pu[k] for k in kk))
        bins[f"{lo:g}-{hi:g}"] = {"n": len(kk), "untrained": bootstrap(kk, pu, rng), "drest": bootstrap(kk, pd, rng),
                                  "default": bootstrap(kk, po, rng), "flattening_prediction": 100 * ex(c * lg(um / 100))}
        b = bins[f"{lo:g}-{hi:g}"]
        say(f"  gain {lo:g} to {hi:g} ({len(kk)} prompts): untrained {b['untrained'][0]:.1f}, flattening {b['flattening_prediction']:.1f}, "
            f"DReST {b['drest'][0]:.1f} [{b['drest'][1]:.1f}, {b['drest'][2]:.1f}], default {b['default'][0]:.1f}")
    lu = per_prompt_rates(family, ["untrained"], "test_low_gain")
    ld = per_prompt_rates(family, [f"drest_s{s}" for s in SEEDS], "test_low_gain")
    lo_ = per_prompt_rates(family, [f"default_s{s}" for s in SEEDS], "test_low_gain")
    kl = sorted(set(lu) & set(ld) & set(lo_))
    low_gain = {"n": len(kl), "untrained": bootstrap(kl, lu, rng), "drest": bootstrap(kl, ld, rng), "default": bootstrap(kl, lo_, rng),
                "flattening_prediction": 100 * ex(c * lg(r1(u["lowgain"]) / 100)),
                "prompts_below_half": {"drest": sum(ld[k] < 0.5 for k in kl), "untrained": sum(lu[k] < 0.5 for k in kl),
                                       "default": sum(lo_[k] < 0.5 for k in kl)}}
    say(f"  low-gain test set ({len(kl)} prompts): untrained {low_gain['untrained'][0]:.1f}, flattening {low_gain['flattening_prediction']:.1f}, "
        f"DReST mean over seeds {summary['DReST']['lowgain'][0]:.1f}; prompts on which the model takes the action less than half the time: "
        f"DReST {low_gain['prompts_below_half']['drest']}, untrained {low_gain['prompts_below_half']['untrained']}, default {low_gain['prompts_below_half']['default']}")

    training = {g: [training_summary(family, f"{g.lower()}_s{s}") for s in SEEDS] for g in ("DReST", "Default")}
    for g, ts in training.items():
        say(f"Training curves (Figure 16), {g}: longer share {100 * st.mean(t['first_window_longer'] for t in ts):.1f}% in the first 128 "
            f"meta-episodes, {100 * st.mean(t['last_window_longer'] for t in ts):.1f}% in the last, lowest window USEFULNESS "
            f"{min(t['min_window_usefulness'] for t in ts):.3f}")
    return {"per_run": runs, "summary": summary, "welch": welch, "flattening": flattening, "gain_bins": bins, "low_gain": low_gain,
            "training": {g: [{k: v for k, v in t.items()} for t in ts] for g, ts in training.items()}}


def qwen_ablations(report: list) -> dict:
    """Appendix G: higher sampling temperature and the entropy bonus (Table 11), and paired low-gain differences."""
    say = report.append
    say("\n######## Qwen3-14B regulariser ablations (Table 11)")
    out = {}
    groups = {"Untrained, temperature 1.0": ["untrained"], "Untrained, temperature 1.3": ["untrained_temperature_1.3"],
              "Untrained, temperature 1.6": ["untrained_temperature_1.6"], "Untrained, temperature 2.0": ["untrained_temperature_2.0"],
              "Default": [f"default_s{s}" for s in SEEDS], "Default + entropy bonus": [f"entropy_bonus_s{s}" for s in (1, 2, 3)],
              "DReST": [f"drest_s{s}" for s in SEEDS]}
    for name, models in groups.items():
        rows = [evaluate("qwen3-14b", m) for m in models]
        cells = {k: mean_sd(r[k]["rate_pct"] for r in rows) for k in ("main", "dominance", "joint_noshift", "zero_cost", "lowgain")}
        cells["neutrality"] = mean_sd(r["post"]["neutrality"] for r in rows)
        cells["unreadable_main_pct"] = st.mean(r["main"]["unreadable_pct"] for r in rows)
        out[name] = cells
        say(f"  {name:28s} " + "  ".join(f"{k} {v[0]:.1f}" + (f" ± {v[1]:.1f}" if len(models) > 1 else "") for k, v in cells.items()
                                       if k not in ("neutrality", "unreadable_main_pct"))
            + f"  NEUTRALITY {cells['neutrality'][0]:.3f}" + (f" ± {cells['neutrality'][1]:.3f}" if len(models) > 1 else "")
            + f"  unreadable {cells['unreadable_main_pct']:.1f}%")
    # Paired low-gain differences from the untrained model, bootstrap over prompts with seeds pooled (as in the paper).
    def pooled(models):
        d = defaultdict(list)
        for m in models:
            for r in answers("qwen3-14b", m, "test_low_gain")["main"]:
                if r.get("decision") in ("shift", "no_shift"):
                    d[r["scenario_id"]].append(r["decision"] == "shift")
        return d
    base = pooled(["untrained"])
    differences = {}
    for name in ("DReST", "Default", "Default + entropy bonus", "Untrained, temperature 1.3", "Untrained, temperature 1.6"):
        other = pooled(groups[name])
        ids = sorted(set(base) & set(other))
        m = lambda d, sample: sum(x for k in sample for x in d[k]) / sum(len(d[k]) for k in sample)  # noqa: E731
        rng = random.Random(2)
        boot = sorted(100 * (m(other, s) - m(base, s)) for s in ([rng.choice(ids) for _ in ids] for _ in range(2000)))
        differences[name] = [100 * (m(other, ids) - m(base, ids)), boot[50], boot[1949]]
        say(f"  low-gain difference from untrained, {name}: {differences[name][0]:+.1f} points (95% interval {boot[50]:+.1f} to {boot[1949]:+.1f})")
    out["low_gain_differences_from_untrained"] = differences
    return out


def learning_rate_searches(report: list) -> dict:
    """Appendix A: the learning-rate search for each LLM, on the validation set (eight answers per displayed prompt).
    Rule: discard a rate whose dominance rate is below 80%, joint-no-shift rate above 20%, POST USEFULNESS below 0.9,
    or share of unreadable answers in any suite above 5%. Among the rest find the highest POST mixing within one option
    order, and choose the lowest rate whose 95% interval overlaps the best one's. For Qwen3-14B the overlap clause was
    added after its search, which on the rule as first written would have chosen 1e-4. It was fixed before the others."""
    say = report.append
    say("\n######## Learning-rate searches (Appendix A), validation set")
    out = {}
    for family in FAMILIES:
        rows = {}
        for lr in ("2e-5", "5e-5", "1e-4"):
            path = RESULTS / "evaluations" / family / f"lr_search_{lr}" / "validation.jsonl.gz"
            if not path.exists():
                rows[lr] = None
                continue
            by_suite = answers(family, f"lr_search_{lr}", "validation")
            val = displays("validation")
            post = post_summary(by_suite["post"], val)
            per_scenario = defaultdict(lambda: defaultdict(list))
            for r in by_suite["post"]:
                if r.get("position") is not None:
                    per_scenario[r["scenario_id"]][r["display_id"]].append(val[r["display_id"]]["options"][r["position"]]["length"] == "long")
            mix = [st.mean(4 * len(v) / (len(v) - 1) * (sum(v) / len(v)) * (1 - sum(v) / len(v)) for v in d.values() if len(v) >= 2)
                   for d in per_scenario.values()]
            rng = random.Random(5)
            boots = sorted(st.mean(rng.choice(mix) for _ in mix) for _ in range(1000))
            rows[lr] = {"mixing": st.mean(mix), "mixing_95": [boots[25], boots[974]],
                        "dominance": binary_summary(by_suite["dominance"], val)["rate_pct"],
                        "joint_noshift": binary_summary(by_suite["joint_noshift"], val)["rate_pct"],
                        "usefulness": post["usefulness"],
                        "unreadable_max_pct": max(binary_summary(by_suite[k], val)["unreadable_pct"] for k in ("main", "dominance", "joint_noshift"))}
            rows[lr]["unreadable_max_pct"] = max(rows[lr]["unreadable_max_pct"], post["unreadable_pct"])
            rows[lr]["eligible"] = (rows[lr]["dominance"] >= 80 and rows[lr]["joint_noshift"] <= 20 and rows[lr]["usefulness"] >= 0.9
                                    and rows[lr]["unreadable_max_pct"] <= 5)
        ok = {lr: r for lr, r in rows.items() if r and r["eligible"]}
        best = max(ok, key=lambda lr: ok[lr]["mixing"])
        tied = [lr for lr, r in ok.items() if r["mixing_95"][1] >= ok[best]["mixing_95"][0] and r["mixing_95"][0] <= ok[best]["mixing_95"][1]]
        chosen = min(tied, key=float)
        out[NAMES[family]] = {"candidates": rows, "highest_mixing": best, "chosen": chosen}
        for lr, r in rows.items():
            say(f"  {NAMES[family]:15s} {lr}: " + ("not evaluated (training collapsed)" if r is None else
                f"POST mixing {r['mixing']:.3f} [{r['mixing_95'][0]:.3f}, {r['mixing_95'][1]:.3f}], dominance {r['dominance']:.1f}%, "
                f"joint no-shift {r['joint_noshift']:.1f}%, USEFULNESS {r['usefulness']:.3f}, unreadable at most {r['unreadable_max_pct']:.1f}%"
                + ("" if r["eligible"] else "  -> discarded")))
        say(f"  {NAMES[family]:15s} highest mixing at {best}, chosen {chosen}")
    return out


def exact_sampler_reruns(report: list) -> dict:
    """Robustness check (not reported in the paper): seeds rerun with the exact sampler (scripts/train.py --exact_sampler)."""
    say = report.append
    say("\n######## Exact-sampler reruns (robustness check, not in the paper)")
    out = {}
    for family, model, original in (("qwen3-14b", "exact_sampler_drest_s2", "drest_s2"), ("qwen3-14b", "exact_sampler_drest_s3", "drest_s3"),
                                    ("qwen3-14b", "exact_sampler_default_s2", "default_s2"), ("qwen3-14b", "exact_sampler_default_s3", "default_s3"),
                                    ("gemma-4-12b", "exact_sampler_drest_s3", "drest_s3")):
        rows = {name: evaluate(family, name) for name in (model, original)}
        out[f"{NAMES[family]} {model}"] = {name: {"main": r["main"]["rate_pct"], "lowgain": r["lowgain"]["rate_pct"],
                                                  "post_mixing_within_order": r["post"]["mixing_within_order"]} for name, r in rows.items()}
        say(f"  {NAMES[family]:12s} {model:26s} " + " | ".join(
            f"{name}: Neutrality test {r['main']['rate_pct']:.1f}%, low-gain {r['lowgain']['rate_pct']:.1f}%, POST mixing {r['post']['mixing_within_order']:.3f}"
            for name, r in rows.items()))
    return out


def main() -> None:
    report = ["LLM numbers in the paper, recomputed from results/ by scripts/reproduce_paper_numbers.py"]
    numbers = {NAMES[f]: family_numbers(f, report) for f in FAMILIES}
    numbers["Qwen3-14B ablations"] = qwen_ablations(report)
    numbers["learning-rate searches"] = learning_rate_searches(report)
    numbers["exact-sampler reruns"] = exact_sampler_reruns(report)
    (RESULTS / "paper_numbers.json").write_text(json.dumps(numbers, indent=1) + "\n")
    (RESULTS / "paper_numbers.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
