#!/usr/bin/env python3
"""Regenerate the manuscript's comparison bar charts from CORRECTED numbers.

Replaces the contaminated/broken-harness charts (figures/image3.png,
image4.png) with charts drawn from results_fixed/stats_summary.csv:

  figures/image3_corrected.{png,pdf} -- Finistral ([INST] + Alpaca ablation)
      vs the Mistral-7B-v0.1 zero-shot backbone, per dataset.
  figures/image4_corrected.{png,pdf} -- Finistral vs all FinGPT adapters and
      FinBERT, per dataset.

CPU-only. Usage (repo root):
    python revision/scripts/make_comparison_charts.py \
        [--stats results_fixed/stats_summary.csv] [--outdir figures]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Okabe-Ito subset (matches analysis_and_figures.py conventions).
BAR_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
              "#56B4E9", "#F0E442", "#999999"]
DATASET_TITLES = {"fpb_decontam": "FPB decontaminated (n=560)",
                  "fiqa": "FiQA-SA (leakage-free, n=235)",
                  "tfns": "TFNS (leakage-free, n=2,373)"}

GROUPS = {
    "image3_corrected": [
        "Finistral-7B-LoRA",
        "Finistral-7B-LoRA (Alpaca prompt)",
        "Mistral-7B-v0.1-Base",
    ],
    "image4_corrected": [
        "Finistral-7B-LoRA",
        "FinGPT-mt-Llama2-7B-LoRA",
        "FinGPT-Llama-3-8B-LoRA",
        "FinGPT-Falcon-7B-LoRA",
        "FinGPT-Bloom-7B1-LoRA",
        "FinBERT (ProsusAI)",
    ],
}


def chart(df: pd.DataFrame, models: list, out_base: str, title: str):
    datasets = [d for d in DATASET_TITLES if d in set(df["dataset"])]
    present = [m for m in models if m in set(df["model"])]
    if not datasets or not present:
        print(f"skip {out_base}: no matching rows yet")
        return
    fig, axes = plt.subplots(1, len(datasets),
                             figsize=(4.2 * len(datasets), 3.6), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    width = 0.38
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds].set_index("model")
        x = np.arange(len(present))
        acc = [sub.loc[m, "accuracy"] if m in sub.index else np.nan
               for m in present]
        wf1 = [sub.loc[m, "weighted_f1"] if m in sub.index else np.nan
               for m in present]
        ax.bar(x - width / 2, acc, width, label="Accuracy",
               color=BAR_COLORS[0])
        ax.bar(x + width / 2, wf1, width, label="Weighted F1",
               color=BAR_COLORS[1])
        for xi, v in zip(x - width / 2, acc):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
        for xi, v in zip(x + width / 2, wf1):
            if not np.isnan(v):
                ax.text(xi, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace(" (", "\n(").replace("-LoRA", "")
                            for m in present], fontsize=7, rotation=30,
                           ha="right")
        ax.set_title(DATASET_TITLES[ds], fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Score")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=300)
    plt.close(fig)
    print(f"wrote {out_base}.png/.pdf")


def main() -> int:
    repo = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats",
                    default=os.path.join(repo, "results_fixed",
                                         "stats_summary.csv"))
    ap.add_argument("--outdir", default=os.path.join(repo, "figures"))
    args = ap.parse_args()

    df = pd.read_csv(args.stats)
    os.makedirs(args.outdir, exist_ok=True)
    chart(df, GROUPS["image3_corrected"],
          os.path.join(args.outdir, "image3_corrected"),
          "Finistral-7B-LoRA vs Mistral-7B-v0.1 backbone "
          "(leakage-free evaluation, corrected harness)")
    chart(df, GROUPS["image4_corrected"],
          os.path.join(args.outdir, "image4_corrected"),
          "Finistral-7B-LoRA vs FinGPT adapters and FinBERT "
          "(leakage-free evaluation, corrected harness)")
    return 0


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)
