#!/usr/bin/env python3
# =============================================================================
# analysis_and_figures.py
# -----------------------------------------------------------------------------
# Publication-quality analysis + figure generation for the Finistral AI
# financial-sentiment study (paper: finistral_grjournals.tex, JICTASA-2026-018).
#
# WHAT THIS SCRIPT PRODUCES (all saved as BOTH .pdf and .png at 300 dpi)
#   (1) Misclassified-examples table with an automatic ERROR-CATEGORY tag
#       (negation / numeric-guidance / conditional-hedge / sarcasm-irony /
#        mixed-sentiment / entity-confusion / other), plus a CSV of
#        representative examples per category.
#   (2) Per-class precision / recall / F1 bar charts (one panel per model).
#   (3) One-vs-rest ROC curves and Precision-Recall curves.
#       -> These REQUIRE soft class scores. This file ships a generation-time
#          hook (capture_answer_logprobs / score_dataframe) that extracts the
#          log-prob the model assigns to the negative/neutral/positive answer
#          tokens, so genuine soft scores exist. See SECTION 4.
#   (4) Learning curves (train vs val loss) read from a HuggingFace
#       Trainer trainer_state.json (the log_history list).
#   (5) A multi-baseline confusion-matrix grid (one heatmap per model).
#
# These five deliverables are explicitly intended to REPLACE the Gradio GUI
# screenshots image1.jpeg / image2.jpeg in the manuscript, which Reviewer 1
# (point 10) asked to be swapped for learning / PR / ROC / ablation curves.
#
# -----------------------------------------------------------------------------
# INPUT CONTRACT
# -----------------------------------------------------------------------------
# Primary input is a *per-example predictions CSV* with (at minimum) the
# columns below. Column names are matched case-insensitively and a small set
# of aliases is accepted (see _COLUMN_ALIASES), so CSVs exported by the
# existing test_final*.ipynb harness work without renaming.
#
#   true_label   gold label   -> "negative" / "neutral" / "positive"  OR  0/1/2
#   pred_label   predicted    -> same domain as true_label
#   sentence     the input text that was classified
#   model        model name   -> e.g. "Finistral-7B-LoRA", "FinGPT-Falcon-7B-LoRA"
#
# OPTIONAL soft-score columns (needed for ROC / PR). If present they are used
# directly; if absent, ROC/PR is skipped for that model UNLESS you produce
# them with the scoring hook in SECTION 4.
#
#   score_negative, score_neutral, score_positive   probabilities in [0,1]
#       (rows for one model should sum to ~1; we renormalise defensively)
#   -- OR --
#   logprob_negative, logprob_neutral, logprob_positive   raw log-probs
#       (softmaxed internally to recover probabilities)
#
# LABEL CONVENTION (matches takala/financial_phrasebank `sentences_allagree`
# and the `_map3` dict in test_final*.ipynb):
#       negative -> 0 , neutral -> 1 , positive -> 2
#
# -----------------------------------------------------------------------------
# USAGE
# -----------------------------------------------------------------------------
#   # Everything from a single predictions CSV:
#   python analysis_and_figures.py --pred-csv predictions.csv --outdir figures/
#
#   # Add learning curves from a Trainer state file:
#   python analysis_and_figures.py --pred-csv predictions.csv \
#       --trainer-state lora-out/checkpoint-XXXX/trainer_state.json \
#       --outdir figures/
#
#   # Restrict the focus model (used for the misclassified table & per-class bars
#   # when a single "headline" model is wanted):
#   python analysis_and_figures.py --pred-csv predictions.csv \
#       --focus-model "Finistral-7B-LoRA" --outdir figures/
#
# To GENERATE the soft scores (so ROC/PR are real, not skipped) see the
# fully-worked, copy-pasteable recipe in SECTION 4's module docstring and the
# score_dataframe() helper.
# =============================================================================

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Matplotlib is imported with the non-interactive Agg backend so the script
# runs headless (CI, servers, Colab batch) and only ever writes files.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# scikit-learn supplies all the metric machinery. Imported at module top so a
# missing dependency fails loudly and immediately rather than mid-run.
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


# =============================================================================
# SECTION 0 -- GLOBAL CONVENTIONS
# =============================================================================

# Canonical 3-class label space. Order is FIXED and matches the FPB feature
# `names` and the notebooks' `_map3` = {"neg":0,"neu":1,"pos":2}. Every figure,
# axis, and confusion matrix below uses THIS ordering so rows/cols are
# comparable across models and against the paper's confusion matrix
# (diagonal 301 / 1385 / 568).
CLASSES: List[str] = ["negative", "neutral", "positive"]
CLASS_TO_ID: Dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
ID_TO_CLASS: Dict[int, str] = {i: c for c, i in CLASS_TO_ID.items()}

# A consistent, colour-blind-friendly palette (Okabe-Ito subset). Used for
# class colours in ROC/PR so the same class is the same colour everywhere.
CLASS_COLORS: Dict[str, str] = {
    "negative": "#D55E00",   # vermillion
    "neutral":  "#999999",   # grey
    "positive": "#009E73",   # bluish-green
}

# Standardised save resolution for print. Reviewer 1 (pt 10) wants
# publication figures; 300 dpi vector+raster covers both LaTeX and reviewers.
SAVE_DPI = 300

# Aliases so we can ingest the column names the notebooks/users might emit
# without forcing a rename step. Keys are the canonical internal names.
_COLUMN_ALIASES: Dict[str, Sequence[str]] = {
    "true_label": ("true_label", "true", "gold", "label", "y_true", "target",
                   "actual", "ground_truth"),
    "pred_label": ("pred_label", "pred", "prediction", "y_pred", "predicted",
                   "output", "model_answer"),
    "sentence":   ("sentence", "text", "input", "news", "snippet"),
    "model":      ("model", "model_name", "pretty", "system", "run"),
}


# =============================================================================
# SECTION 1 -- LOADING & NORMALISING THE PREDICTIONS CSV
# =============================================================================

def _resolve_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Map the canonical names (true_label/pred_label/sentence/model) onto whatever
    the incoming CSV actually called them, case-insensitively, via _COLUMN_ALIASES.

    Returns a dict {canonical_name -> actual_column_name}. Raises a clear error
    if a required column cannot be found, listing what *was* present so the user
    can fix the CSV quickly.
    """
    lower_to_actual = {c.lower(): c for c in df.columns}
    resolved: Dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower_to_actual:
                resolved[canonical] = lower_to_actual[alias.lower()]
                break
    required = ("true_label", "pred_label", "sentence")  # model is optional
    missing = [c for c in required if c not in resolved]
    if missing:
        raise ValueError(
            f"Predictions CSV is missing required column(s) {missing}. "
            f"Columns present: {list(df.columns)}. "
            f"Accepted aliases: "
            f"{ {k: v for k, v in _COLUMN_ALIASES.items() if k in missing} }"
        )
    return resolved


def _to_label_string(value) -> str:
    """
    Coerce a label cell into a canonical class STRING ('negative'/'neutral'/
    'positive'). Accepts:
      * integer ids 0/1/2 (and numpy ints)
      * the short codes 'neg'/'neu'/'pos' produced by extract_sentiment()
      * full words in any case, possibly embedded in a longer answer string
    Anything unrecognised raises so silent label corruption can't happen.
    """
    # Numeric id path (handles ints, floats like 1.0, numpy scalars, "1").
    try:
        as_int = int(float(value))
        if as_int in ID_TO_CLASS:
            return ID_TO_CLASS[as_int]
    except (ValueError, TypeError):
        pass

    s = str(value).strip().lower()
    # Direct exact matches first (fast path).
    if s in CLASS_TO_ID:
        return s
    # Short-code / substring search (mirrors the notebooks' extract_sentiment).
    if s.startswith("neg") or "negative" in s:
        return "negative"
    if s.startswith("pos") or "positive" in s:
        return "positive"
    if s.startswith("neu") or "neutral" in s:
        return "neutral"
    raise ValueError(f"Unrecognised sentiment label: {value!r}")


def load_predictions(pred_csv: str) -> pd.DataFrame:
    """
    Read the per-example predictions CSV and return a tidy, canonicalised
    DataFrame with EXACTLY these columns:

        model | sentence | true_label | pred_label | true_id | pred_id
        [+ score_negative/score_neutral/score_positive if derivable]

    * true_label / pred_label are canonical strings.
    * true_id / pred_id are the 0/1/2 ids (handy for sklearn).
    * If logprob_* columns exist they are softmaxed into score_* probabilities.
      If score_* exist they are renormalised per row to sum to 1.
    """
    df = pd.read_csv(pred_csv)
    cols = _resolve_columns(df)

    out = pd.DataFrame()
    out["sentence"] = df[cols["sentence"]].astype(str)
    out["true_label"] = df[cols["true_label"]].map(_to_label_string)
    out["pred_label"] = df[cols["pred_label"]].map(_to_label_string)
    out["model"] = (
        df[cols["model"]].astype(str) if "model" in cols else "model"
    )
    out["true_id"] = out["true_label"].map(CLASS_TO_ID)
    out["pred_id"] = out["pred_label"].map(CLASS_TO_ID)

    # ---- Soft scores: prefer explicit probabilities, else softmax log-probs --
    score_cols = [f"score_{c}" for c in CLASSES]
    logp_cols = [f"logprob_{c}" for c in CLASSES]
    lower = {c.lower(): c for c in df.columns}

    if all(sc in lower for sc in score_cols):
        probs = df[[lower[sc] for sc in score_cols]].to_numpy(dtype=float)
        # Defensive renormalisation (rows might not be exactly normalised).
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        for i, c in enumerate(CLASSES):
            out[f"score_{c}"] = probs[:, i]
    elif all(lc in lower for lc in logp_cols):
        logp = df[[lower[lc] for lc in logp_cols]].to_numpy(dtype=float)
        # Numerically stable softmax over the 3 answer-token log-probs.
        logp = logp - logp.max(axis=1, keepdims=True)
        ex = np.exp(logp)
        probs = ex / ex.sum(axis=1, keepdims=True)
        for i, c in enumerate(CLASSES):
            out[f"score_{c}"] = probs[:, i]
    # else: no soft scores -> ROC/PR will be skipped for these rows.

    return out


# =============================================================================
# SECTION 2 -- ERROR-CATEGORY HEURISTIC + MISCLASSIFIED-EXAMPLES TABLE
# =============================================================================
#
# Reviewer 1 (pt 7) wants an error analysis with misclassified examples and a
# per-category breakdown. The heuristic below is a TRANSPARENT, rule-based
# tagger -- deliberately not a black box -- so reviewers can audit exactly why
# a sentence got a given tag. Each rule returns a boolean; a sentence can match
# several rules, and we keep ALL matches (a ranked primary tag is also stored).
#
# Categories (financial-sentiment failure modes that are well documented in the
# literature and visible in FPB):
#   negation            -> polarity flipped by a negation cue ("not", "no",
#                          "fail to", "declined") the bag-of-words may miss.
#   numeric_guidance    -> the sentiment hinges on a number / forward guidance
#                          ("EPS of 0.12", "down 4%", "FY25 outlook"), which
#                          models often read as neutral.
#   conditional_hedge   -> hypothetical / hedged framing ("if", "could",
#                          "expects", "subject to") that softens polarity.
#   sarcasm_irony       -> rhetorical / ironic cues (quotes, "so-called",
#                          exclamation) where surface words invert meaning.
#   mixed_sentiment     -> both positive AND negative cues co-occur
#                          ("revenue up but margins fell"), a classic
#                          neutral/positive/negative confusion.
#   entity_confusion    -> multiple companies/tickers, so the sentiment may
#                          attach to the wrong entity.
#   other               -> no rule fired (fallback bucket).
# =============================================================================

# --- Lexicons (lower-cased, word-boundary matched where it matters) ----------

_NEGATION_CUES = [
    r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bnone\b", r"\bwithout\b",
    r"\bfail(?:s|ed|ure)?\b", r"\bdecl(?:ine|ined|ines)\b", r"\bdrop(?:ped|s)?\b",
    r"\bfell\b", r"\bfall(?:s|en)?\b", r"\bcut(?:s|ting)?\b", r"\bloss(?:es)?\b",
    r"\bdownturn\b", r"\bnegative\b", r"\bweak(?:er|ened)?\b", r"n't\b",
]

# Forward-looking / guidance vocabulary frequently mis-read as neutral.
_GUIDANCE_CUES = [
    r"\bguidance\b", r"\boutlook\b", r"\bforecast(?:s|ed)?\b", r"\bexpect(?:s|ed|ation)?\b",
    r"\bestimate(?:s|d)?\b", r"\bprojec(?:t|ts|ted|tion)\b", r"\beps\b",
    r"\bquarter(?:ly)?\b", r"\bfy\d{2,4}\b", r"\bq[1-4]\b", r"\bfull[- ]year\b",
    r"\btarget(?:s|ed)?\b", r"\brevised?\b",
]

# Conditional / hedging connectives that soften a sentence's polarity.
_CONDITIONAL_CUES = [
    r"\bif\b", r"\bcould\b", r"\bmight\b", r"\bmay\b", r"\bwould\b",
    r"\bshould\b", r"\bsubject to\b", r"\bdepend(?:s|ing|ent)?\b",
    r"\bprovided that\b", r"\bunless\b", r"\bpotential(?:ly)?\b",
    r"\blikely\b", r"\bpossible\b", r"\bassuming\b",
]

# Sarcasm / irony surface cues. Genuinely hard to detect; these are coarse
# proxies (scare quotes, "so-called", emphatic punctuation, hype words).
_SARCASM_CUES = [
    r"so-called", r"\bsupposed(?:ly)?\b", r"\bironically\b", r"\bapparently\b",
    r"\bof course\b", r"\bsurprise\b", r'["“”‘’]',  # quote glyphs
    r"!{1,}", r"\bmiracle\b", r"\bgenius\b",
]

# Positive and negative cue lexicons used to detect MIXED sentiment (both fire).
_POS_CUES = [
    r"\bup\b", r"\brose\b", r"\bgrow(?:th|s|ing)?\b", r"\bgain(?:s|ed)?\b",
    r"\bprofit(?:s|able)?\b", r"\brecord\b", r"\bbeat\b", r"\bstrong(?:er)?\b",
    r"\bincrease(?:s|d)?\b", r"\bimprove(?:s|d|ment)?\b", r"\bupgrade(?:d|s)?\b",
    r"\bsurg(?:e|ed|es)\b", r"\bhigher\b", r"\boutperform(?:ed|s)?\b",
]
_NEG_CUES = [
    r"\bdown\b", r"\bfell\b", r"\bloss(?:es)?\b", r"\bdecl(?:ine|ined|ines)\b",
    r"\bdrop(?:ped|s)?\b", r"\bcut(?:s|ting)?\b", r"\bwarn(?:s|ed|ing)?\b",
    r"\bmiss(?:ed|es)?\b", r"\bweak(?:er|ened)?\b", r"\bdowngrade(?:d|s)?\b",
    r"\bplunge(?:d|s)?\b", r"\blower\b", r"\bunderperform(?:ed|s)?\b",
    r"\bbankrupt(?:cy)?\b", r"\blawsuit\b",
]

# A number / percentage / currency amount, e.g. "4%", "EUR 2.3 mn", "0.12".
# FPB famously uses spaced punctuation ("USD 2.3 mn .") so we keep this loose.
_NUMBER_RE = re.compile(
    r"(?:[€$£]|\b(?:eur|usd|gbp|sek|mn|bn|m|bln)\b)|"  # currency / magnitude
    r"\d+(?:[.,]\d+)?\s?%|"                                       # percentages
    r"\b\d+[.,]\d+\b|"                                            # decimals (EPS-like)
    r"\b\d{2,}\b",                                                # bare multi-digit nums
    re.IGNORECASE,
)

# Company / entity proxy: a capitalised multi-word run, a ticker-like token, or
# the corporate-suffix forms common in FPB (Oyj/Oy/Plc/AB/ASA/Corp/Inc/Ltd).
_ENTITY_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z&.\-]+\s){1,}(?:Oyj|Oy|Plc|PLC|AB|ASA|A/S|Corp|Inc|Ltd|"
    r"Group|Holding|Bank|Co)\b|"
    r"\b[A-Z]{2,5}\b"   # ticker-like all-caps token
)


def _count_matches(patterns: Sequence[str], text: str) -> int:
    """Total number of regex hits across a list of patterns (case-insensitive)."""
    return sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in patterns)


def tag_error_categories(sentence: str) -> List[str]:
    """
    Return the list of error-category tags that fire for a sentence. The rules
    are independent boolean detectors; the ORDER in the returned list is the
    priority order used when a single "primary" tag is needed (negation first
    because a missed negation is the most consequential failure, then the
    numeric/guidance and reasoning-style errors, with mixed/entity last, and
    'other' only if nothing fired).
    """
    s = sentence.lower()
    tags: List[str] = []

    # 1) NEGATION: at least one negation cue present.
    if _count_matches(_NEGATION_CUES, s) >= 1:
        tags.append("negation")

    # 2) NUMERIC / GUIDANCE: a number is present, AND/OR guidance vocabulary.
    #    Either signal alone qualifies because both are the "neutral-trap" mode.
    if _NUMBER_RE.search(sentence) or _count_matches(_GUIDANCE_CUES, s) >= 1:
        tags.append("numeric_guidance")

    # 3) CONDITIONAL / HEDGE.
    if _count_matches(_CONDITIONAL_CUES, s) >= 1:
        tags.append("conditional_hedge")

    # 4) SARCASM / IRONY (coarse proxy).
    if _count_matches(_SARCASM_CUES, sentence) >= 1:
        tags.append("sarcasm_irony")

    # 5) MIXED SENTIMENT: both a positive and a negative cue co-occur.
    if _count_matches(_POS_CUES, s) >= 1 and _count_matches(_NEG_CUES, s) >= 1:
        tags.append("mixed_sentiment")

    # 6) ENTITY CONFUSION: two or more distinct entity-like spans.
    if len(set(_ENTITY_RE.findall(sentence))) >= 2:
        tags.append("entity_confusion")

    if not tags:
        tags.append("other")
    return tags


def build_misclassified_table(
    df: pd.DataFrame,
    model: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build the misclassified-examples table for a given model (or all rows if
    `model` is None). Each row of the returned frame is ONE misclassified
    example, annotated with:

        model | sentence | true_label | pred_label
        error_categories (semicolon-joined) | primary_category

    The frame is sorted by primary_category so per-category clusters are
    contiguous and easy to read in the paper's appendix.
    """
    sub = df if model is None else df[df["model"] == model]
    wrong = sub[sub["true_id"] != sub["pred_id"]].copy()
    if wrong.empty:
        # Return an empty, correctly-typed frame so downstream code is safe.
        return pd.DataFrame(
            columns=["model", "sentence", "true_label", "pred_label",
                     "error_categories", "primary_category"]
        )

    cats = wrong["sentence"].map(tag_error_categories)
    wrong["error_categories"] = cats.map(lambda lst: ";".join(lst))
    wrong["primary_category"] = cats.map(lambda lst: lst[0])

    keep = ["model", "sentence", "true_label", "pred_label",
            "error_categories", "primary_category"]
    return wrong[keep].sort_values(["primary_category", "model"]).reset_index(drop=True)


def export_representative_examples(
    mis_table: pd.DataFrame,
    outdir: str,
    per_category: int = 5,
) -> str:
    """
    Export up to `per_category` representative misclassified examples PER
    primary error category to a CSV, plus a small summary of counts.
    Returns the path to the CSV. This directly answers Reviewer 1 pt 7
    ("misclassified examples + per-category").
    """
    os.makedirs(outdir, exist_ok=True)

    # Per-category counts (for the manuscript's error-analysis table).
    counts = (
        mis_table["primary_category"].value_counts().rename_axis("category")
        .reset_index(name="n_misclassified")
        if not mis_table.empty
        else pd.DataFrame(columns=["category", "n_misclassified"])
    )
    counts_path = os.path.join(outdir, "error_category_counts.csv")
    counts.to_csv(counts_path, index=False)

    # Representative sample: first `per_category` rows of each category. We keep
    # them deterministic (head) rather than random so the paper is reproducible.
    reps = (
        mis_table.groupby("primary_category", group_keys=False)
        .head(per_category)
        if not mis_table.empty
        else mis_table
    )
    reps_path = os.path.join(outdir, "misclassified_representative_examples.csv")
    reps.to_csv(reps_path, index=False)

    # Full dump too, for completeness / appendix.
    full_path = os.path.join(outdir, "misclassified_all.csv")
    mis_table.to_csv(full_path, index=False)

    print(f"[error-analysis] {len(mis_table)} misclassified rows; "
          f"category counts -> {counts_path}")
    print(f"[error-analysis] representative examples -> {reps_path}")
    return reps_path


# =============================================================================
# SECTION 3 -- PER-CLASS PRECISION / RECALL / F1 BAR CHARTS
# =============================================================================

def plot_per_class_prf(
    df: pd.DataFrame,
    outdir: str,
    filename: str = "per_class_prf",
) -> Tuple[str, str]:
    """
    Grouped bar chart of per-class precision / recall / F1, one subplot per
    model. Returns the (pdf_path, png_path) it wrote.

    Why per-class (not just weighted F1): the paper's headline weighted-F1 hides
    that a model can score high overall while collapsing on the minority
    'negative'/'positive' classes -- exactly the failure the broken baselines
    exhibit. This chart exposes that per class.
    """
    models = sorted(df["model"].unique())
    n = len(models)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 4.2 * nrows),
                             squeeze=False)

    x = np.arange(len(CLASSES))   # one group per class
    width = 0.26                  # bar width for the 3 metrics

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["model"] == model]
        # labels=[0,1,2] forces sklearn to report ALL three classes even if a
        # broken model never predicts one of them (avoids silently dropping it).
        p, r, f1, _ = precision_recall_fscore_support(
            sub["true_id"], sub["pred_id"], labels=[0, 1, 2], zero_division=0
        )
        ax.bar(x - width, p, width, label="Precision", color="#4C72B0")
        ax.bar(x,         r, width, label="Recall",    color="#DD8452")
        ax.bar(x + width, f1, width, label="F1",       color="#55A868")

        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(CLASSES)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.grid(axis="y", alpha=0.3)
        # Annotate F1 on top of each F1 bar (the metric reviewers care most about).
        for xi, val in zip(x + width, f1):
            ax.text(xi, val + 0.02, f"{val:.2f}", ha="center", va="bottom",
                    fontsize=7)
        if idx == 0:
            ax.legend(loc="lower right", fontsize=8)

    # Hide any unused subplot cells in the grid.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Per-class Precision / Recall / F1 on Financial PhraseBank "
                 "(sentences_allagree)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_fig(fig, outdir, filename)


# =============================================================================
# SECTION 4 -- SOFT SCORES: GENERATION-TIME LOG-PROB HOOK  (for ROC / PR)
# =============================================================================
#
# ROC and Precision-Recall curves need a CONTINUOUS score per class, not just
# the arg-max label. The existing harness only keeps the decoded text, so we
# add a hook that, at generation time, reads the model's log-prob for each of
# the three answer tokens (negative / neutral / positive) and stores them.
#
# IMPORTANT -- this hook also FIXES the bugs that made the baselines score
# below the majority class (see the verified findings):
#   * tokenizer.padding_side = "left"   (decoder-only models MUST left-pad;
#     the notebooks used default right padding, which corrupted batched
#     generation and triggered the "right-padding was detected" warnings).
#   * we score the FIRST answer-token logits directly instead of generating a
#     long ramble with max_length=512 (the notebooks' max_length collided with
#     padding=True and was ignored). No generation-length bug here at all.
#   * an [INST] template option so eval can MATCH the training template
#     ([INST]{input}\n{instruction} [/INST]) instead of the Alpaca prompt the
#     notebooks used -- removing the train/eval template mismatch.
#
# This function is dependency-light at import time: torch / transformers are
# imported INSIDE the function so the plotting half of this file runs on a
# machine with no GPU / no transformers installed.
#
# ---- FULLY-WORKED RECIPE (copy/paste) ---------------------------------------
#
#   from analysis_and_figures import (
#       build_inst_prompt, build_alpaca_prompt, score_dataframe)
#   import pandas as pd
#   from transformers import AutoTokenizer, AutoModelForCausalLM
#   from peft import PeftModel
#
#   tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
#   base = AutoModelForCausalLM.from_pretrained(
#       "mistralai/Mistral-7B-v0.1", device_map="auto", torch_dtype="bfloat16")
#   model = PeftModel.from_pretrained(base, "Ayansk11/Finistral-7B_lora").eval()
#
#   df = pd.read_csv("fpb_eval.csv")          # needs a 'sentence' column
#   scored = score_dataframe(                  # adds logprob_/score_/pred_label
#       df, model, tok,
#       prompt_builder=build_inst_prompt,      # MATCHES training template
#       model_name="Finistral-7B-LoRA")
#   scored.to_csv("predictions_with_scores.csv", index=False)
#   # -> then feed predictions_with_scores.csv to this script's --pred-csv.
# =============================================================================

# The instruction string the training script used (Finistral_Sentiment_analyst.py
# line 37). Kept verbatim so eval prompts can match training exactly.
_INSTRUCTION = ("What is the sentiment of this news? "
                "Please choose an answer from {negative/neutral/positive}")


def build_inst_prompt(sentence: str) -> str:
    """
    Reconstruct the EXACT training-time prompt template from
    Finistral_Sentiment_analyst.py (line 96):
        prompt = f"[INST]{inp}\n{instr} [/INST]"
    Use this builder to evaluate Finistral consistently with how it was trained.
    """
    return f"[INST]{sentence}\n{_INSTRUCTION} [/INST]"


def build_alpaca_prompt(sentence: str) -> str:
    """
    The Alpaca-style prompt the notebooks (test_final*.ipynb make_prompt) used.
    Provided for apples-to-apples reproduction of the *original* (buggy) eval,
    and for baselines whose native template is Instruction/Input/Answer.
    """
    return ("Instruction: What is the sentiment of this news? "
            "Please choose an answer from {negative/neutral/positive}\n"
            f"Input: {sentence}\n"
            "Answer:")


def capture_answer_logprobs(
    model,
    tokenizer,
    prompts: Sequence[str],
    class_words: Sequence[str] = CLASSES,
    batch_size: int = 8,
):
    """
    Core generation-time hook. For each prompt, compute the log-probability the
    model assigns to EACH class word as the next token after the prompt, by:

      1) left-padding the batch (decoder-only correctness -- the key bug fix),
      2) a single forward pass to get the logits at the final (right-most)
         position [this is the position that predicts the first answer token],
      3) log-softmax over the vocabulary,
      4) reading off the log-prob of the first sub-word token of each class word.

    Returns an (N, 3) numpy array of log-probs aligned with `class_words`
    (default order = negative, neutral, positive). We use a forward pass rather
    than .generate() because we only need the distribution over the first answer
    token -- this is faster, deterministic, and immune to the max_length /
    sampling pitfalls in the original notebooks.

    NOTE on tokenisation: we take the FIRST token id of each class word. For
    most tokenizers a leading space matters, so we try both "word" and " word"
    and keep whichever yields a leading in-vocab token; this keeps the three
    class scores comparable.
    """
    import torch  # local import: keeps the plotting path GPU/transformers-free

    # ---- FIX #1: decoder-only models must LEFT-pad for correct batched logits.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Resolve each class word to a single representative token id. --------
    def first_token_id(word: str) -> int:
        # Prefer the space-prefixed form (how a word appears mid-sentence after
        # "[/INST] " or "Answer: "); fall back to the bare form.
        for cand in (" " + word, word):
            ids = tokenizer(cand, add_special_tokens=False)["input_ids"]
            if ids:
                return ids[0]
        # Last-resort: unk id (should never happen for these common words).
        return tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0

    class_token_ids = [first_token_id(w) for w in class_words]

    device = next(model.parameters()).device
    all_logprobs: List[np.ndarray] = []

    model.eval()
    for start in range(0, len(prompts), batch_size):
        batch = list(prompts[start:start + batch_size])
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**enc)
            # logits: (B, T, V). Because we LEFT-padded, the real last token of
            # every sequence sits at position -1 for all rows -> clean to index.
            last_logits = out.logits[:, -1, :]                  # (B, V)
            logprobs = torch.log_softmax(last_logits, dim=-1)   # (B, V)
            # Gather the three class-token log-probs.
            sel = logprobs[:, class_token_ids]                  # (B, 3)
        all_logprobs.append(sel.float().cpu().numpy())

    return np.concatenate(all_logprobs, axis=0)


def score_dataframe(
    df: pd.DataFrame,
    model,
    tokenizer,
    prompt_builder=build_inst_prompt,
    model_name: str = "model",
    sentence_col: str = "sentence",
    batch_size: int = 8,
) -> pd.DataFrame:
    """
    Convenience wrapper: take a DataFrame with a `sentence` column, run the
    log-prob hook, and return a NEW frame with the columns this analysis script
    consumes:

        sentence | model |
        logprob_negative | logprob_neutral | logprob_positive |
        score_negative   | score_neutral   | score_positive   |
        pred_label   (= arg-max of the three scores)

    If the input frame already has a `true_label` column it is carried through,
    giving you a complete predictions CSV ready for --pred-csv.
    """
    prompts = [prompt_builder(s) for s in df[sentence_col].astype(str)]
    logp = capture_answer_logprobs(model, tokenizer, prompts,
                                   batch_size=batch_size)   # (N,3)

    # Softmax the three log-probs into a proper categorical distribution.
    z = logp - logp.max(axis=1, keepdims=True)
    ex = np.exp(z)
    probs = ex / ex.sum(axis=1, keepdims=True)

    out = pd.DataFrame()
    out["sentence"] = df[sentence_col].astype(str).values
    out["model"] = model_name
    if "true_label" in df.columns:
        out["true_label"] = df["true_label"].values
    for i, c in enumerate(CLASSES):
        out[f"logprob_{c}"] = logp[:, i]
        out[f"score_{c}"] = probs[:, i]
    out["pred_label"] = [CLASSES[i] for i in probs.argmax(axis=1)]
    return out


# =============================================================================
# SECTION 5 -- ONE-VS-REST ROC AND PRECISION-RECALL CURVES
# =============================================================================

def _has_scores(df: pd.DataFrame) -> bool:
    """True iff the frame carries usable per-class soft scores."""
    return all(f"score_{c}" in df.columns for c in CLASSES)


def plot_roc_curves(
    df: pd.DataFrame,
    outdir: str,
    filename: str = "roc_curves",
) -> Optional[Tuple[str, str]]:
    """
    One-vs-rest ROC curves (one panel per model, one curve per class) with AUC
    in the legend. Returns (pdf, png) paths, or None if no soft scores exist.

    OvR construction: for class c, positives are rows whose true label == c and
    the score is score_c; this yields the standard 3 one-vs-rest curves.
    Requires the score_* columns (see SECTION 4).
    """
    if not _has_scores(df):
        print("[roc] No per-class scores found -> ROC skipped. "
              "Produce them with score_dataframe() (SECTION 4).")
        return None

    models = sorted(df["model"].unique())
    ncols = min(2, len(models))
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.0 * nrows),
                             squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["model"] == model]
        y_true = sub["true_id"].to_numpy()
        scores = sub[[f"score_{c}" for c in CLASSES]].to_numpy()
        # Binarise the true labels into a (N,3) indicator for OvR.
        y_bin = label_binarize(y_true, classes=[0, 1, 2])

        for i, c in enumerate(CLASSES):
            # A class with no positive examples can't define an ROC -> skip it.
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], scores[:, i])
            ax.plot(fpr, tpr, color=CLASS_COLORS[c], lw=2,
                    label=f"{c} (AUC={auc(fpr, tpr):.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)  # chance line
        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(len(models), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("One-vs-Rest ROC Curves", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_fig(fig, outdir, filename)


def plot_pr_curves(
    df: pd.DataFrame,
    outdir: str,
    filename: str = "pr_curves",
) -> Optional[Tuple[str, str]]:
    """
    One-vs-rest Precision-Recall curves (one panel per model, one curve per
    class) with Average Precision in the legend. Returns (pdf, png) or None.

    PR (not just ROC) is reported because FPB is class-imbalanced (neutral
    ~61.6%); PR curves are more informative than ROC under imbalance for the
    minority negative/positive classes. Requires score_* columns (SECTION 4).
    """
    if not _has_scores(df):
        print("[pr] No per-class scores found -> PR skipped. "
              "Produce them with score_dataframe() (SECTION 4).")
        return None

    models = sorted(df["model"].unique())
    ncols = min(2, len(models))
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 5.0 * nrows),
                             squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // ncols][idx % ncols]
        sub = df[df["model"] == model]
        y_true = sub["true_id"].to_numpy()
        scores = sub[[f"score_{c}" for c in CLASSES]].to_numpy()
        y_bin = label_binarize(y_true, classes=[0, 1, 2])

        for i, c in enumerate(CLASSES):
            if y_bin[:, i].sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(y_bin[:, i], scores[:, i])
            ap = average_precision_score(y_bin[:, i], scores[:, i])
            ax.plot(rec, prec, color=CLASS_COLORS[c], lw=2,
                    label=f"{c} (AP={ap:.3f})")
            # Class prevalence baseline (a no-skill PR classifier sits here).
            ax.axhline(y_bin[:, i].mean(), color=CLASS_COLORS[c],
                       ls=":", lw=1, alpha=0.5)

        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(alpha=0.3)

    for j in range(len(models), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("One-vs-Rest Precision-Recall Curves",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_fig(fig, outdir, filename)


# =============================================================================
# SECTION 6 -- LEARNING CURVES FROM HF TRAINER STATE
# =============================================================================

def load_trainer_log_history(trainer_state_path: str) -> pd.DataFrame:
    """
    Parse a HuggingFace trainer_state.json and return a tidy long-form frame:

        step | epoch | metric | value

    where metric is "train_loss" or "eval_loss". The Trainer writes a
    log_history list of dicts; training-loss entries carry key "loss" and
    eval entries carry key "eval_loss" (logged every eval_steps). We split them
    apart and keep the step/epoch so they can be plotted on a common x-axis.
    """
    with open(trainer_state_path, "r") as fh:
        state = json.load(fh)

    rows: List[dict] = []
    for entry in state.get("log_history", []):
        step = entry.get("step")
        epoch = entry.get("epoch")
        if "loss" in entry:            # training-loss log line
            rows.append({"step": step, "epoch": epoch,
                         "metric": "train_loss", "value": entry["loss"]})
        if "eval_loss" in entry:       # evaluation log line
            rows.append({"step": step, "epoch": epoch,
                         "metric": "eval_loss", "value": entry["eval_loss"]})
    return pd.DataFrame(rows)


def plot_learning_curves(
    trainer_state_path: str,
    outdir: str,
    filename: str = "learning_curves",
) -> Optional[Tuple[str, str]]:
    """
    Train-vs-validation loss curves over optimisation steps, with a secondary
    epoch axis. Returns (pdf, png) paths, or None if the file has no loss logs.

    This is the single most-requested missing figure (Reviewer 1 pt 10): it lets
    a reviewer SEE convergence and check for over-fitting, instead of trusting
    the lone "best val loss 0.1008/0.1009" number quoted in the paper.
    """
    hist = load_trainer_log_history(trainer_state_path)
    if hist.empty:
        print(f"[learning-curves] No loss entries in {trainer_state_path} -> skipped.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    train = hist[hist["metric"] == "train_loss"].dropna(subset=["value"])
    val = hist[hist["metric"] == "eval_loss"].dropna(subset=["value"])

    if not train.empty:
        ax.plot(train["step"], train["value"], color="#4C72B0", lw=1.6,
                alpha=0.85, label="Train loss")
    if not val.empty:
        # Eval is logged sparsely (every eval_steps); markers make it readable.
        ax.plot(val["step"], val["value"], color="#C44E52", lw=2.0,
                marker="o", ms=4, label="Validation loss")
        # Mark and annotate the best (minimum) validation loss -- the number the
        # paper headlines. Showing it on the curve makes the claim auditable.
        best = val.loc[val["value"].idxmin()]
        ax.scatter([best["step"]], [best["value"]], color="black", zorder=5)
        ax.annotate(f"best val = {best['value']:.4f}\n(step {int(best['step'])})",
                    xy=(best["step"], best["value"]),
                    xytext=(10, 18), textcoords="offset points", fontsize=8,
                    arrowprops=dict(arrowstyle="->", lw=0.8))

    ax.set_xlabel("Optimisation step")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss", fontsize=12, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Secondary x-axis in epochs, if epoch info is available, for readability.
    have_epoch = hist["epoch"].notna().any()
    if have_epoch:
        steps = hist.dropna(subset=["epoch", "step"]).sort_values("step")
        if len(steps) >= 2 and steps["step"].max() > steps["step"].min():
            secax = ax.secondary_xaxis(
                "top",
                functions=(
                    # linear step<->epoch mapping using observed values
                    lambda s: np.interp(s, steps["step"], steps["epoch"]),
                    lambda e: np.interp(e, steps["epoch"], steps["step"]),
                ),
            )
            secax.set_xlabel("Epoch")

    fig.tight_layout()
    return _save_fig(fig, outdir, filename)


# =============================================================================
# SECTION 7 -- MULTI-BASELINE CONFUSION-MATRIX GRID
# =============================================================================

def plot_confusion_grid(
    df: pd.DataFrame,
    outdir: str,
    filename: str = "confusion_matrix_grid",
    normalize: bool = False,
) -> Tuple[str, str]:
    """
    A grid of confusion matrices -- one per model -- using a SHARED class order
    (negative, neutral, positive) so cells are directly comparable across
    models. Returns (pdf, png) paths.

    Reviewer 2 (pt 4) explicitly asked for confusion matrices for ALL baselines,
    not just Finistral. normalize=True row-normalises (recall view), which is
    the fairer way to compare models with different per-class supports.
    """
    models = sorted(df["model"].unique())
    n = len(models)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(4.2 * ncols, 3.8 * nrows))
    gs = GridSpec(nrows, ncols, figure=fig)

    for idx, model in enumerate(models):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        sub = df[df["model"] == model]
        cm = confusion_matrix(sub["true_id"], sub["pred_id"], labels=[0, 1, 2])
        disp = cm.astype(float)
        if normalize:
            # Row-normalise; guard against all-zero rows (a class never present).
            row_sums = disp.sum(axis=1, keepdims=True)
            disp = np.divide(disp, row_sums, out=np.zeros_like(disp),
                             where=row_sums != 0)

        ax.imshow(disp, cmap="Blues", vmin=0,
                  vmax=1 if normalize else (disp.max() if disp.max() > 0 else 1))
        ax.set_title(model, fontsize=10, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES, rotation=45, ha="right",
                                                    fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)

        # Cell annotations: counts (or fractions). Auto-contrast the text colour.
        thresh = disp.max() / 2.0 if disp.max() > 0 else 0.5
        for i in range(3):
            for j in range(3):
                txt = (f"{disp[i, j]:.2f}" if normalize
                       else f"{int(cm[i, j])}")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if disp[i, j] > thresh else "black")

    for j in range(n, nrows * ncols):
        fig.add_subplot(gs[j // ncols, j % ncols]).axis("off")

    suffix = " (row-normalised)" if normalize else " (counts)"
    fig.suptitle("Confusion Matrices across Models" + suffix,
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return _save_fig(fig, outdir, filename)


# =============================================================================
# SECTION 8 -- SAVE HELPER + CLI ORCHESTRATION
# =============================================================================

def _save_fig(fig, outdir: str, stem: str) -> Tuple[str, str]:
    """
    Save a figure as BOTH vector PDF (for LaTeX inclusion) and 300-dpi PNG (for
    quick preview / Word), into `outdir`. Returns the two paths and closes the
    figure to free memory. All figures in this script go through here so the
    "PDF/PNG at 300 dpi" requirement is enforced in exactly one place.
    """
    os.makedirs(outdir, exist_ok=True)
    pdf_path = os.path.join(outdir, f"{stem}.pdf")
    png_path = os.path.join(outdir, f"{stem}.png")
    fig.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {pdf_path}  +  {png_path}")
    return pdf_path, png_path


def run_all(
    pred_csv: str,
    outdir: str,
    trainer_state: Optional[str] = None,
    focus_model: Optional[str] = None,
    per_category: int = 5,
) -> None:
    """
    End-to-end driver: load predictions, then emit every figure/table. Each
    block is independent and degrades gracefully (e.g. ROC/PR self-skip when no
    soft scores are present; learning curves skip when no trainer state given).
    """
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== Finistral analysis & figures -> {outdir} ===")
    df = load_predictions(pred_csv)
    print(f"[load] {len(df)} rows, models = {sorted(df['model'].unique())}")

    # (1) Misclassified table + representative per-category examples.
    #     If a focus model is given we tag only that model's errors; otherwise
    #     we tag every model's errors together (model column disambiguates).
    mis = build_misclassified_table(df, model=focus_model)
    export_representative_examples(mis, outdir, per_category=per_category)

    # (2) Per-class precision / recall / F1 bars (all models).
    plot_per_class_prf(df, outdir)

    # (3) ROC + PR (auto-skip if no soft scores in the CSV).
    plot_roc_curves(df, outdir)
    plot_pr_curves(df, outdir)

    # (4) Learning curves (only if a Trainer state file was supplied).
    if trainer_state:
        plot_learning_curves(trainer_state, outdir)
    else:
        print("[learning-curves] --trainer-state not provided -> skipped.")

    # (5) Confusion-matrix grid: both raw-count and row-normalised variants.
    plot_confusion_grid(df, outdir, filename="confusion_matrix_grid_counts",
                        normalize=False)
    plot_confusion_grid(df, outdir, filename="confusion_matrix_grid_normalized",
                        normalize=True)

    print("=== done. These figures are intended to REPLACE image1/image2 "
          "(the Gradio GUI screenshots). ===\n")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Generate publication figures + error analysis for the Finistral
            financial-sentiment study from a per-example predictions CSV.

            The CSV needs: true_label, pred_label, sentence, model
            (case-insensitive; common aliases accepted). Add per-class
            score_*/logprob_* columns -- e.g. via score_dataframe() -- to also
            get ROC and PR curves.
        """),
    )
    p.add_argument("--pred-csv", required=True,
                   help="Path to the per-example predictions CSV.")
    p.add_argument("--outdir", default="figures",
                   help="Directory for output PDFs/PNGs/CSVs (default: figures).")
    p.add_argument("--trainer-state", default=None,
                   help="Path to a HuggingFace trainer_state.json for learning "
                        "curves (optional).")
    p.add_argument("--focus-model", default=None,
                   help="Restrict the misclassified-examples table to this "
                        "model name (optional).")
    p.add_argument("--per-category", type=int, default=5,
                   help="Representative misclassified examples to export per "
                        "error category (default: 5).")
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    try:
        run_all(
            pred_csv=args.pred_csv,
            outdir=args.outdir,
            trainer_state=args.trainer_state,
            focus_model=args.focus_model,
            per_category=args.per_category,
        )
    except Exception as exc:  # surface a clean message, non-zero exit for CI
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
