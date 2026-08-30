#!/usr/bin/env python3
"""ALTERNATIVE to the zero-based bar charts: dot plot with 95% CIs.

Why this exists: a truncated y-axis on a BAR chart is misleading, because bar
length encodes magnitude. A dot plot encodes value by POSITION, so a zoomed
axis is legitimate — this is the standard way to resolve small differences
honestly. Wilson score intervals (95%) are computed per model from the
observed accuracy and the evaluation-set size, so overlapping intervals show
directly where differences are not meaningful.

CPU-only. Usage (repo root):
    python revision/scripts/make_dotplot_alternative.py
Writes figures/dotplot_comparison.{png,pdf}
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "figures", "dotplot_comparison")

DATASET_TITLES = {"fpb_decontam": "FPB decontaminated (n=560)",
                  "fiqa": "FiQA-SA (decontaminated, n=235)",
                  "tfns": "TFNS (decontaminated, n=2,373)"}
SHORT = {
    "Finistral-7B-LoRA": "Finistral (ours)",
    "Finistral-7B-LoRA (Alpaca prompt)": "Finistral (Alpaca prompt)",
    "Mistral-7B-v0.1-Base": "Mistral-7B (zero-shot)",
    "FinGPT-mt-Llama2-7B-LoRA": "FinGPT Llama-2",
    "FinGPT-Llama-3-8B-LoRA": "FinGPT Llama-3",
    "FinGPT-Falcon-7B-LoRA": "FinGPT Falcon",
    "FinGPT-Bloom-7B1-LoRA": "FinGPT Bloom",
    "FinBERT (ProsusAI)": "FinBERT",
}
ORDER = ["Finistral-7B-LoRA", "FinGPT-mt-Llama2-7B-LoRA", "FinGPT-Falcon-7B-LoRA",
         "FinGPT-Llama-3-8B-LoRA", "FinGPT-Bloom-7B1-LoRA", "FinBERT (ProsusAI)"]
BLUE, GREY = "#0072B2", "#555555"


def wilson(p: float, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion."""
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def main() -> int:
    df = pd.read_csv(os.path.join(REPO, "results_fixed", "stats_summary.csv"))
    datasets = [d for d in DATASET_TITLES if d in set(df["dataset"])]
    fig, axes = plt.subplots(1, len(datasets), figsize=(14.5, 5.0), sharey=True)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds].set_index("model")
        models = [m for m in ORDER if m in sub.index]
        ys = np.arange(len(models))[::-1]
        for y, m in zip(ys, models):
            acc, n = float(sub.loc[m, "accuracy"]), int(sub.loc[m, "n"])
            lo, hi = wilson(acc, n)
            is_ours = m == "Finistral-7B-LoRA"
            c = BLUE if is_ours else GREY
            ax.plot([lo, hi], [y, y], color=c, lw=2.4, alpha=0.85, zorder=2)
            ax.plot([acc], [y], "o", color=c, markersize=11 if is_ours else 8,
                    zorder=3)
            ax.text(hi + 0.008, y, f"{acc:.3f}", va="center", fontsize=10.5,
                    color=c, fontweight="bold" if is_ours else "normal")
        ax.set_yticks(ys)
        ax.set_yticklabels([SHORT[m] for m in models], fontsize=12)
        ax.set_title(DATASET_TITLES[ds], fontsize=13.5, fontweight="bold")
        ax.set_xlabel("Accuracy (95% Wilson CI)", fontsize=12)
        # Zoomed, per-panel range -- legitimate because position, not bar
        # length, encodes the value here.
        vals = [float(sub.loc[m, "accuracy"]) for m in models]
        pad = max(0.04, (max(vals) - min(vals)) * 0.28)
        ax.set_xlim(min(vals) - pad, min(1.005, max(vals) + pad * 1.5))
        ax.grid(axis="x", alpha=0.35)
        ax.tick_params(axis="x", labelsize=10.5)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)

    fig.suptitle("Corrected accuracy with 95% confidence intervals "
                 "(zoomed per panel; overlapping intervals = not distinguishable)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {OUT}.png/.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
