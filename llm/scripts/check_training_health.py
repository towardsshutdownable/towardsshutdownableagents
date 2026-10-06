#!/usr/bin/env python3
"""Check from train.log that an RLOO run is actually learning (run it once 16 steps are logged).

    python scripts/check_training_health.py runs/<run_name>/train.log [min_steps, default 16]

Exit 0 and print PASS when healthy, exit 1 and print FAIL with reasons, exit 2 if fewer than min_steps steps are logged.
Checks, over the logged steps:
  - the gradient norm is nonzero on at least half the steps
  - the share of answers flagged as cut off (completions/clipped_ratio) averages below 0.2. Cut-off answers are dropped
    from the loss, so a run in which most answers count as cut off learns nothing. This is what a wrong end-of-answer
    token causes (see end_of_answer_fix in scripts/train.py).
  - the loss is not identically zero
  - the task reward varies within the group (reward_std > 0) on at least 5% of steps
"""
import ast
import re
import sys

log, min_steps = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16
text = open(log, errors="replace").read().replace("\r", "\n")
rows = []
for line in re.findall(r"^\{'loss'.*\}$", text, re.M):
    try:
        rows.append(ast.literal_eval(line))
    except (ValueError, SyntaxError):
        pass
if len(rows) < min_steps:
    print(f"WAIT only {len(rows)} logged steps")
    sys.exit(2)
values = lambda k: [float(r[k]) for r in rows if k in r]  # noqa: E731
grad, clipped, loss, reward_std = values("grad_norm"), values("completions/clipped_ratio"), values("loss"), values("reward_std")
varied = sum(s > 0 for s in reward_std)
fails = []
if grad and sum(g > 0 for g in grad) < len(grad) / 2:
    fails.append(f"gradient zero on {sum(g == 0 for g in grad)} of {len(grad)} steps")
if clipped and sum(clipped) / len(clipped) > 0.2:
    fails.append(f"mean cut-off share {sum(clipped) / len(clipped):.2f} (answers dropped from the loss)")
if loss and all(x == 0 for x in loss):
    fails.append("loss identically zero")
if varied < 0.05 * len(rows):
    fails.append("reward never varies within a group")
summary = (f"{len(rows)} steps, gradient nonzero on {sum(g > 0 for g in grad)}, mean cut-off share "
           f"{(sum(clipped) / len(clipped) if clipped else float('nan')):.3f}, steps with reward variation {varied}")
print(("FAIL " + "; ".join(fails) + " | " if fails else "PASS ") + summary)
sys.exit(1 if fails else 0)
