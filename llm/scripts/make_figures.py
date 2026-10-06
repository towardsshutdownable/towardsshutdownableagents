#!/usr/bin/env python3
"""Draw the paper's two LLM figures (Figures 16 and 17) from results/paper_numbers.json (run scripts/reproduce_paper_numbers.py first).

    pip install matplotlib
    python scripts/make_figures.py [output folder, default results/figures]

  llm_training_curves_four_families.pdf      Figure 16: share of answers choosing the longer trajectory-length, and
                                             USEFULNESS, over training (windows of 128 meta-episodes)
  llm_costly_shift_by_gain_four_families.pdf Figure 17: influence rate by expected-score gain on the Neutrality test set,
                                             and on the low-gain test set, with the uniform-flattening prediction
"""
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LLM_DIR = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else LLM_DIR / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
D = json.loads((LLM_DIR / "results" / "paper_numbers.json").read_text())
FAMILIES = ["Qwen3-14B", "Gemma 4 12B", "Granite 4.2 8B", "gpt-oss-20b"]
C_UNTRAINED, C_DREST, C_DEFAULT, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#e6e6e6"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                     "legend.fontsize": 8.5, "axes.edgecolor": "#888888", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#333333", "pdf.fonttype": 42})
lg = lambda p: math.log(p / (1 - p))  # noqa: E731
ex = lambda z: 1 / (1 + math.exp(-z))  # noqa: E731
r1 = lambda x: round(x, 1)  # noqa: E731


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# Figure 16: training curves, 2 rows (longer share, USEFULNESS) x 4 LLMs
fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.1), sharex=True)
for j, fam in enumerate(FAMILIES):
    for group, colour in (("DReST", C_DREST), ("Default", C_DEFAULT)):
        curves = [t["curve"] for t in D[fam]["training"][group]]
        x = [w["meta_episodes"][1] + 1 for w in curves[0]]
        for i, (key, scale) in enumerate((("longer_share", 100), ("usefulness", 1))):
            ax = axes[i, j]
            for c in curves:
                ax.plot(x, [scale * w[key] for w in c], color=colour, lw=0.8, alpha=0.28)
            ax.plot(x, [scale * sum(c[k][key] for c in curves) / len(curves) for k in range(len(x))], color=colour, lw=2.0, label=group)
    axes[0, j].set_title(fam)
    axes[0, j].set_ylim(50, 101)
    axes[1, j].set_ylim(0.97, 1.0005)
    for i in (0, 1):
        style(axes[i, j])
        axes[i, j].set_xlim(0, 2048)
        axes[i, j].set_xticks([0, 1000, 2000])
        if j:
            axes[i, j].tick_params(labelleft=False)
    axes[1, j].set_xlabel("Meta-episodes")
axes[0, 0].set_ylabel("Longer trajectory-length\n(% of answers)")
axes[1, 0].set_ylabel("USEFULNESS")
axes[0, 0].text(-0.42, 1.02, "(a)", transform=axes[0, 0].transAxes, fontsize=10)
axes[1, 0].text(-0.42, 1.02, "(b)", transform=axes[1, 0].transAxes, fontsize=10)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95), w_pad=0.6, h_pad=0.8)
fig.savefig(OUT / "llm_training_curves_four_families.pdf")
plt.close(fig)

# Figure 17: influence rate by gain, 4 rows (LLMs) x [(a) gain bins, (b) low-gain test set]
fig, axes = plt.subplots(4, 2, figsize=(7.2, 8.3), gridspec_kw={"width_ratios": [3, 1.05]}, sharey=True)
width = 0.26
bars = (("untrained", C_UNTRAINED), ("drest", C_DREST), ("default", C_DEFAULT))
for i, fam in enumerate(FAMILIES):
    v = D[fam]
    axa, axb = axes[i]
    for ax in (axa, axb):
        style(ax)
        ax.axhline(50, color="#888888", lw=0.9, ls=(0, (3, 2)), zorder=1.5)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
    bins = list(v["gain_bins"].items())
    for k, (_, b) in enumerate(bins):
        for o, (key, colour) in enumerate(bars):
            m, lo, hi = b[key]
            axa.bar(k + (o - 1) * width, m, width, color=colour, zorder=2)
            axa.errorbar(k + (o - 1) * width, m, yerr=[[m - lo], [hi - m]], color="#333333", capsize=2.5, lw=1, zorder=3)
        axa.plot([k - 0.6 * width, k + 0.6 * width], [b["flattening_prediction"]] * 2, color="#0b0b0b", lw=2.6, zorder=4, solid_capstyle="butt")
    low = v["low_gain"]
    for o, (key, colour) in enumerate(bars):
        m, lo, hi = low[key]
        axb.bar((o - 1) * width, m, width, color=colour, zorder=2)
        axb.errorbar((o - 1) * width, m, yerr=[[m - lo], [hi - m]], color="#333333", capsize=2.5, lw=1, zorder=3)
    axb.plot([-0.6 * width, 0.6 * width], [low["flattening_prediction"]] * 2, color="#0b0b0b", lw=2.6, zorder=4, solid_capstyle="butt")
    axb.text(0.66, 51.5, "50%", color="#52514e", fontsize=8, ha="right", va="bottom")
    axa.set_xlim(-0.55, 2.55)
    axb.set_xlim(-0.55, 0.7)
    axa.set_ylabel(f"{fam}\nInfluence rate (%)")
    axa.set_xticks(range(3))
    axb.set_xticks([0])
    if i == 0:
        axa.set_title("(a) Neutrality test set, by gain bin")
        axb.set_title("(b) Low-gain test set")
    if i == len(FAMILIES) - 1:
        axa.set_xticklabels([f"0 to 0.6\n({bins[0][1]['n']} scenarios)", f"0.6 to 1.0\n({bins[1][1]['n']} scenarios)",
                             f"above 1.0\n({bins[2][1]['n']} scenarios)"])
        axb.set_xticklabels([f"gain 0.25 to 0.60\n({low['n']} scenarios)"])
        axa.set_xlabel("Expected-score gain from influencing shutdown")
    else:
        axa.set_xticklabels([])
        axb.set_xticklabels([])
handles = [plt.Line2D([], [], color="#0b0b0b", lw=2.6), plt.Rectangle((0, 0), 1, 1, color=C_UNTRAINED),
           plt.Rectangle((0, 0), 1, 1, color=C_DREST), plt.Rectangle((0, 0), 1, 1, color=C_DEFAULT)]
fig.legend(handles, ["Uniform-flattening prediction", "Untrained", "DReST (5 seeds)", "Default (5 seeds)"], loc="upper center", ncol=4,
           frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.975), w_pad=0.8, h_pad=0.9)
fig.savefig(OUT / "llm_costly_shift_by_gain_four_families.pdf")
plt.close(fig)
print("wrote", OUT / "llm_training_curves_four_families.pdf", "and", OUT / "llm_costly_shift_by_gain_four_families.pdf")
