#!/usr/bin/env python3
"""Post-ingest statistics for JICTASA-2026-018 (CPU).

Reads every ``*_predictions.csv`` the fixed harness wrote, then produces:

  results_fixed/all_predictions.csv  -- concatenated audit trail with the
      ``true_label`` column analysis_and_figures.py expects.
  results_fixed/stats_summary.csv    -- per (dataset, model): n, accuracy,
      weighted/macro F1, unparseable rate; plus, versus Finistral-7B-LoRA:
      McNemar b/c/statistic/p and paired-bootstrap 95% CI on the accuracy gap.
  results_fixed/stats_summary.md     -- the same as a readable table, with the
      strongest baseline per dataset and the [INST]-vs-Alpaca template
      ablation called out.

Statistical machinery is imported from ablation_and_seeds.py (mcnemar_test,
bootstrap_accuracy_ci) -- verified side-effect-free imports. Pairing for the
McNemar/bootstrap tests uses the per-example ``index`` column, which is stable
because every model consumed the same frozen CSV in the same order.

Usage (repo root):
    python revision/scripts/stats_report.py [--results_dir results_fixed]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from ablation_and_seeds import bootstrap_accuracy_ci, mcnemar_test  # noqa: E402

FOCUS = "Finistral-7B-LoRA"                    # the model every test compares to
ALPACA = "Finistral-7B-LoRA (Alpaca prompt)"   # the template-ablation row


def load_all(results_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_dir, "*_predictions.csv")))
    if not paths:
        raise SystemExit(f"no *_predictions.csv under {results_dir}")
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        if "model" not in df.columns or "dataset" not in df.columns:
            raise SystemExit(f"{p} lacks model/dataset columns -- "
                             "regenerate with the extended harness")
        frames.append(df)
    allp = pd.concat(frames, ignore_index=True)
    # analysis_and_figures.py's column aliases do not cover 'gold_label'.
    allp = allp.rename(columns={"gold_label": "true_label"})
    print(f"loaded {len(paths)} prediction files, {len(allp)} rows")
    return allp


def per_model_metrics(grp: pd.DataFrame) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    y, p = grp["gold_id"], grp["pred_id"]
    return {
        "n": len(grp),
        "accuracy": accuracy_score(y, p),
        "weighted_f1": f1_score(y, p, labels=[0, 1, 2], average="weighted",
                                zero_division=0),
        "macro_f1": f1_score(y, p, labels=[0, 1, 2], average="macro",
                             zero_division=0),
        "unparseable_rate": float(grp["unparseable"].mean()),
    }


def paired_tests(focus: pd.DataFrame, other: pd.DataFrame) -> dict:
    """McNemar + paired bootstrap CI between two models on one dataset.

    A = Finistral (focus), B = the other model; both signatures are
    (y_true, y_pred_a, y_pred_b) per ablation_and_seeds.py.
    """
    merged = focus.merge(other, on="index", suffixes=("_a", "_b"))
    y_true = merged["gold_id_a"].tolist()
    pred_a = merged["pred_id_a"].tolist()
    pred_b = merged["pred_id_b"].tolist()
    mc = mcnemar_test(y_true, pred_a, pred_b)
    ci = bootstrap_accuracy_ci(y_true, pred_a, pred_b)
    out = {"mcnemar_" + k: v for k, v in mc.items()}
    out.update({"boot_" + k: v for k, v in ci.items()})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default=os.path.join(REPO, "results_fixed"))
    args = ap.parse_args()

    allp = load_all(args.results_dir)
    allp.to_csv(os.path.join(args.results_dir, "all_predictions.csv"), index=False)

    rows, md_lines = [], []
    for dataset, dgrp in allp.groupby("dataset"):
        md_lines.append(f"\n## {dataset}\n")
        md_lines.append("| model | n | acc | wF1 | macroF1 | unparse | "
                        "McNemar p (vs Finistral) | Δacc 95% CI |")
        md_lines.append("|---|---|---|---|---|---|---|---|")
        focus_df = dgrp[dgrp["model"] == FOCUS]
        model_rows = []
        for model, mgrp in dgrp.groupby("model"):
            row = {"dataset": dataset, "model": model}
            row.update(per_model_metrics(mgrp))
            if model != FOCUS and len(focus_df):
                try:
                    row.update(paired_tests(focus_df, mgrp))
                except Exception as exc:
                    row["test_error"] = str(exc)
            model_rows.append(row)
        # strongest baseline = best weighted F1 among non-Finistral rows
        baselines = [r for r in model_rows
                     if r["model"] not in (FOCUS, ALPACA)]
        strongest = max(baselines, key=lambda r: r["weighted_f1"],
                        default=None)
        for row in sorted(model_rows, key=lambda r: -r["weighted_f1"]):
            row["strongest_baseline"] = int(
                strongest is not None and row["model"] == strongest["model"])
            rows.append(row)
            p_txt = (f"{row['mcnemar_p_value']:.4g}"
                     if "mcnemar_p_value" in row else "")
            ci_txt = (f"[{row['boot_lo']:+.4f}, {row['boot_hi']:+.4f}]"
                      if "boot_lo" in row else "")
            star = " **<-- strongest baseline**" if row["strongest_baseline"] else ""
            md_lines.append(
                f"| {row['model']}{star} | {row['n']} | {row['accuracy']:.4f} | "
                f"{row['weighted_f1']:.4f} | {row['macro_f1']:.4f} | "
                f"{row['unparseable_rate']:.4f} | {p_txt} | {ci_txt} |")
        # Template ablation callout
        names = {r["model"] for r in model_rows}
        if FOCUS in names and ALPACA in names:
            f_acc = next(r["accuracy"] for r in model_rows if r["model"] == FOCUS)
            a_acc = next(r["accuracy"] for r in model_rows if r["model"] == ALPACA)
            md_lines.append(
                f"\nTemplate ablation: [INST] {f_acc:.4f} vs Alpaca {a_acc:.4f} "
                f"(Δ = {f_acc - a_acc:+.4f}); see the Alpaca row above for "
                f"McNemar/CI vs the [INST] row.")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.results_dir, "stats_summary.csv"), index=False)
    md = "# Corrected-evaluation statistics\n" + "\n".join(md_lines) + "\n"
    with open(os.path.join(args.results_dir, "stats_summary.md"), "w") as fh:
        fh.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
