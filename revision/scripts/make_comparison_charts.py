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
                  "fiqa": "FiQA-SA (decontaminated, n=235)",
                  "tfns": "TFNS (decontaminated, n=2,373)"}

SHORT_LABELS = {
    "Finistral-7B-LoRA": "Finistral\n(ours)",
    "Finistral-7B-LoRA (Alpaca prompt)": "Finistral\n(Alpaca prompt)",
    "Mistral-7B-v0.1-Base": "Mistral-7B\n(zero-shot)",
    "FinGPT-mt-Llama2-7B-LoRA": "FinGPT\nLlama-2",
    "FinGPT-Llama-3-8B-LoRA": "FinGPT\nLlama-3",
    "FinGPT-Falcon-7B-LoRA": "FinGPT\nFalcon",
    "FinGPT-Bloom-7B1-LoRA": "FinGPT\nBloom",
    "FinBERT (ProsusAI)": "FinBERT",
}

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
    """One panel per dataset, stacked VERTICALLY so each panel is full width
    and model labels stay legible at print size."""
    datasets = [d for d in DATASET_TITLES if d in set(df["dataset"])]
    present = [m for m in models if m in set(df["model"])]
    if not datasets or not present:
        print(f"skip {out_base}: no matching rows yet")
        return
    fig, axes = plt.subplots(len(datasets), 1,
                             figsize=(11.0, 4.6 * len(datasets)), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    width = 0.34
    for ax, ds in zip(axes, datasets):
        sub = df[df["dataset"] == ds].set_index("model")
        x = np.arange(len(present))
        acc = [sub.loc[m, "accuracy"] if m in sub.index else np.nan
               for m in present]
        wf1 = [sub.loc[m, "weighted_f1"] if m in sub.index else np.nan
               for m in present]
        ax.bar(x - width / 2, acc, width, label="Accuracy", color=BAR_COLORS[0])
        ax.bar(x + width / 2, wf1, width, label="Weighted F1", color=BAR_COLORS[1])
        for xi, v in zip(x - width / 2, acc):
            if not np.isnan(v):
                ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=10.5)
        for xi, v in zip(x + width / 2, wf1):
            if not np.isnan(v):
                ax.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=10.5)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT_LABELS.get(m, m) for m in present],
                           fontsize=12.5)
        ax.set_title(DATASET_TITLES[ds], fontsize=15, fontweight="bold")
        ax.set_ylim(0, 1.14)
        ax.set_yticks(np.arange(0, 1.01, 0.1))
        ax.set_ylabel("Score", fontsize=13)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=12.5, ncol=2, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 0.966))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
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
          "Corrected evaluation: Finistral-7B-LoRA vs the Mistral-7B-v0.1 backbone")
    chart(df, GROUPS["image4_corrected"],
          os.path.join(args.outdir, "image4_corrected"),
          "Corrected evaluation: Finistral-7B-LoRA vs FinGPT adapters and FinBERT")
    return 0


if __name__ == "__main__":
    sys_exit = main()
    raise SystemExit(sys_exit)
