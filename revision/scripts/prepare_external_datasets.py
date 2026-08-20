#!/usr/bin/env python3
"""Prepare frozen, decontaminated external evaluation sets for JICTASA-2026-018.

Builds data_eval/{fpb_decontam,fiqa,tfns}.csv (schema: sentence,label with
string labels negative/neutral/positive) plus data_eval/PROVENANCE.md.

FiQA-SA and Twitter Financial News Sentiment are BOTH constituent sources of
FinGPT/fingpt-sentiment-train (the fine-tuning corpus), so every external row
is checked against the normalized FinGPT input set and dropped on a match --
the same normalization that produced the canonical 560-sentence decontaminated
FPB set (revision/scripts/measure_leakage_local.py).

CPU-only; requires `datasets`. Run from the repo root:
    python revision/scripts/prepare_external_datasets.py
"""
from __future__ import annotations

import csv
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import date

from datasets import load_dataset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "data_eval")

FPB_DECONTAM_SRC = os.path.join(REPO, "revision", "fpb_decontaminated.csv")

# FiQA-SA: continuous sentiment score in [-1, 1]; thresholds documented in the
# manuscript. FiQA has essentially no neutral band by construction, so the
# neutral bucket is expected to be small.
FIQA_ID = "TheFinAI/fiqa-sentiment-classification"
FIQA_FALLBACK_ID = "pauri32/fiqa-2018"
FIQA_POS_THRESHOLD = 0.1
FIQA_NEG_THRESHOLD = -0.1

TFNS_ID = "zeroshot/twitter-financial-news-sentiment"
TFNS_LABEL_MAP = {0: "negative", 1: "positive", 2: "neutral"}  # Bearish/Bullish/Neutral


def normalize(text: str) -> str:
    """Identical to measure_leakage_local.normalize -- the canonical scheme."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def load_fingpt_normalized() -> set:
    print(">> loading FinGPT/fingpt-sentiment-train (contamination reference) ...",
          flush=True)
    fin = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
    norm = {normalize(x) for x in fin["input"]}
    print(f"   {len(fin)} rows -> {len(norm)} unique normalized inputs", flush=True)
    return norm


def load_fiqa() -> list:
    """Return [(sentence, label)] pooled over ALL FiQA-SA splits.

    Rationale: FinGPT ingested FiQA train data, so the FiQA *test* split is
    itself heavily contaminated (only 51/234 rows survive decontamination).
    Any FiQA sentence absent from the FinGPT corpus was never seen by the
    model regardless of its original split, so we pool train/valid/test and
    let the row-wise decontamination step select the leakage-free subset
    (235 rows). This pooling is disclosed in PROVENANCE.md and the paper.
    """
    def rows_from(ds, sent_col, score_col):
        out = []
        for r in ds:
            score = float(r[score_col])
            if score >= FIQA_POS_THRESHOLD:
                lab = "positive"
            elif score <= FIQA_NEG_THRESHOLD:
                lab = "negative"
            else:
                lab = "neutral"
            out.append((str(r[sent_col]).strip(), lab))
        return out

    def all_splits(repo_id):
        rows = []
        for split in ("train", "valid", "validation", "test"):
            try:
                ds = load_dataset(repo_id, split=split)
            except Exception:
                continue
            cols = ds.column_names
            sent_col = "sentence" if "sentence" in cols else (
                "text" if "text" in cols else cols[0])
            score_col = "score" if "score" in cols else "sentiment_score"
            print(f">> FiQA-SA: {repo_id} {split} split, {len(ds)} rows",
                  flush=True)
            rows.extend(rows_from(ds, sent_col, score_col))
        if not rows:
            raise RuntimeError(f"no splits loadable from {repo_id}")
        return rows

    try:
        return all_splits(FIQA_ID)
    except Exception as exc:  # loader drift -> fallback
        print(f"   primary FiQA loader failed ({exc}); trying {FIQA_FALLBACK_ID}",
              flush=True)
        return all_splits(FIQA_FALLBACK_ID)


def load_tfns() -> list:
    ds = load_dataset(TFNS_ID, split="validation")
    print(f">> TFNS: {TFNS_ID} validation split, {len(ds)} rows", flush=True)
    return [(str(r["text"]).strip(), TFNS_LABEL_MAP[int(r["label"])]) for r in ds]


def decontaminate(rows: list, fingpt_norm: set, name: str):
    kept, dropped = [], 0
    seen = set()
    for sent, lab in rows:
        key = normalize(sent)
        if not key or key in seen:      # drop empties and in-set duplicates
            dropped += 1
            continue
        seen.add(key)
        if key in fingpt_norm:
            dropped += 1
            continue
        kept.append((sent, lab))
    print(f"   {name}: kept {len(kept)}, dropped {dropped} "
          f"(leaked into FinGPT train, duplicate, or empty)", flush=True)
    return kept, dropped


def write_csv(path: str, rows: list):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sentence", "label"])
        w.writerows(rows)


def dist(rows: list) -> str:
    c = Counter(lab for _, lab in rows)
    return " / ".join(f"{k} {c.get(k, 0)}" for k in ("negative", "neutral", "positive"))


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    fingpt_norm = load_fingpt_normalized()

    # 1) FPB decontaminated: copy the canonical 560-row set verbatim.
    fpb_rows = []
    with open(FPB_DECONTAM_SRC, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            fpb_rows.append((r["sentence"], r["label"].strip().lower()))
    assert len(fpb_rows) == 560, f"expected 560 FPB rows, got {len(fpb_rows)}"
    # Sanity: every canonical row must be absent from the FinGPT normalized set.
    leaked = sum(1 for s, _ in fpb_rows if normalize(s) in fingpt_norm)
    assert leaked == 0, f"{leaked} canonical FPB rows still leak -- aborting"
    write_csv(os.path.join(OUT_DIR, "fpb_decontam.csv"), fpb_rows)
    print(f"   fpb_decontam: 560 rows verified leakage-free", flush=True)

    # 2) FiQA-SA
    fiqa_raw = load_fiqa()
    fiqa_rows, fiqa_dropped = decontaminate(fiqa_raw, fingpt_norm, "fiqa")
    write_csv(os.path.join(OUT_DIR, "fiqa.csv"), fiqa_rows)

    # 3) TFNS
    tfns_raw = load_tfns()
    tfns_rows, tfns_dropped = decontaminate(tfns_raw, fingpt_norm, "tfns")
    write_csv(os.path.join(OUT_DIR, "tfns.csv"), tfns_rows)

    with open(os.path.join(OUT_DIR, "PROVENANCE.md"), "w", encoding="utf-8") as fh:
        fh.write(f"""# Evaluation Set Provenance — JICTASA-2026-018

Generated {date.today().isoformat()} by `revision/scripts/prepare_external_datasets.py`
(CPU-only). Uniform schema: `sentence,label` with string labels
negative/neutral/positive. All decontamination uses the same normalization as
`revision/scripts/measure_leakage_local.py` (NFKC fold, lowercase, strip
non-alphanumerics, collapse whitespace) against the
`FinGPT/fingpt-sentiment-train` `input` column ({len(fingpt_norm)} unique
normalized inputs).

## fpb_decontam.csv — {len(fpb_rows)} rows ({dist(fpb_rows)})
Verbatim copy of `revision/fpb_decontaminated.csv`: the Financial PhraseBank
`sentences_allagree` sentences (takala/financial_phrasebank, via the
gtfintechlab parquet mirror) NOT present in the FinGPT training corpus
(1,699 of 2,259 unique sentences = 75.21% were contaminated and removed).
Re-verified leakage-free at generation time (0 matches).

## fiqa.csv — {len(fiqa_rows)} rows ({dist(fiqa_rows)}), {fiqa_dropped} dropped
FiQA-SA, ALL splits pooled (`{FIQA_ID}`, fallback `{FIQA_FALLBACK_ID}`),
then decontaminated row-wise. Pooling rationale (disclosed in the paper):
FinGPT ingested FiQA train data, so the FiQA test split alone is heavily
contaminated (only 51/234 rows survive); any FiQA sentence absent from the
FinGPT corpus was never seen by the model regardless of split. Continuous
sentiment score mapped to labels: score >= {FIQA_POS_THRESHOLD} -> positive,
score <= {FIQA_NEG_THRESHOLD} -> negative, else neutral. FiQA-SA has
essentially no neutral band by construction, so the neutral class is small.
Dropped rows matched a FinGPT training input after normalization (or were
duplicates/empty).

## tfns.csv — {len(tfns_rows)} rows ({dist(tfns_rows)}), {tfns_dropped} dropped
Twitter Financial News Sentiment validation split (`{TFNS_ID}`). Label map:
0 Bearish -> negative, 1 Bullish -> positive, 2 Neutral -> neutral. TFNS train
is a constituent source of the FinGPT corpus; dropped rows matched a FinGPT
training input after normalization (or were duplicates/empty).

## SemEval-2017 Task 5
Not included: no maintained public loader (original distribution requires
registration). Recorded as not delivered in the response to reviewers.
""")
    print(f">> wrote {OUT_DIR}/PROVENANCE.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
