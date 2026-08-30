#!/usr/bin/env python3
"""Purpose-built graphical abstract for JICTASA-2026-018.

Panel A: the contamination finding — 75.2% of the FPB evaluation set appears
verbatim (with identical labels) in the training corpus; the paper evaluates
on the clean remainder + external sets instead.
Panel B: the corrected headline results — weighted F1 for Finistral vs the
best FinGPT adapter per dataset vs FinBERT on the three
exact-match-decontaminated evaluation sets.

Unlike the previous graphical abstract, this shares no image with any body
figure. CPU-only. Usage (repo root):
    python revision/scripts/make_graphical_abstract.py
Writes figures/graphical_abstract.{png,pdf}
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "figures", "graphical_abstract")

BLUE, ORANGE, GREY, GREEN, RED = "#0072B2", "#E69F00", "#999999", "#009E73", "#D55E00"

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.6),
                               gridspec_kw={"height_ratios": [0.72, 1.28]})

# ---------------- Panel A: the contamination finding ----------------
ax1.barh([1], [1699], color=RED, alpha=0.85, height=0.45)
ax1.barh([1], [560], left=[1699], color=GREEN, alpha=0.9, height=0.45)
ax1.set_xlim(0, 2259)
ax1.set_ylim(0.24, 1.92)
ax1.set_yticks([])
ax1.set_xlabel("Financial PhraseBank evaluation sentences (2,259 unique)",
               fontsize=13)
ax1.text(1699 / 2, 1, "1,699 leaked verbatim  (75.2%)\n100% identical gold labels",
         ha="center", va="center", fontsize=14, fontweight="bold", color="white")
ax1.text(1699 + 280, 1, "560\nclean", ha="center", va="center",
         fontsize=13, fontweight="bold", color="white")
ax1.text(1130, 1.63, 'claimed 99.56% "accuracy"  =  memorisation',
         ha="center", fontsize=15, fontweight="bold", color=RED)
ax1.annotate("evaluate here instead\n(+ external datasets)",
             xy=(1979, 0.775), xytext=(1330, 0.66),
             ha="center", va="top",
             fontsize=12.5, fontweight="bold", color=GREEN,
             arrowprops=dict(arrowstyle="->", color=GREEN, lw=2))
ax1.set_title("Finding: 75.2% train/test contamination", fontsize=16,
              fontweight="bold", pad=12)
for s in ("top", "right", "left"):
    ax1.spines[s].set_visible(False)

# ---------------- Panel B: corrected headline results ----------------
datasets = ["FPB-560\n(decontaminated)", "FiQA-SA\n(n=235)", "TFNS\n(n=2,373)"]
finistral = [0.9893, 0.8833, 0.7959]
best_fingpt = [0.9857, 0.8594, 0.8211]
finbert = [0.9648, 0.6147, 0.7329]

x = np.arange(3)
w = 0.26
b1 = ax2.bar(x - w, finistral, w, color=BLUE, label="Finistral-7B-LoRA (ours)")
b2 = ax2.bar(x, best_fingpt, w, color=ORANGE, label="Best FinGPT adapter")
b3 = ax2.bar(x + w, finbert, w, color=GREY, label="FinBERT")
for bars in (b1, b2, b3):
    for r in bars:
        ax2.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015,
                 f"{r.get_height():.3f}", ha="center", fontsize=11)
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, fontsize=13)
ax2.set_ylim(0, 1.30)
ax2.set_ylabel("Weighted F1", fontsize=13)
ax2.set_yticks(np.arange(0, 1.01, 0.2))
ax2.tick_params(axis="y", labelsize=11)
ax2.legend(fontsize=11.5, loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 1.0), columnspacing=2.0)
ax2.set_title("Corrected evaluation: honest, statistically tested results",
              fontsize=16, fontweight="bold", pad=12)
ax2.grid(axis="y", alpha=0.3)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)

fig.tight_layout(h_pad=3.2)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
print(f"wrote {OUT}.png/.pdf")
