#!/usr/bin/env python3
# =============================================================================
# leakage_analysis.py
#
# Quantifies train/test CONTAMINATION between the corpus Finistral-7B-LoRA was
# fine-tuned on and the corpus it was evaluated on:
#
#   TRAIN corpus : FinGPT/fingpt-sentiment-train          (column "input")
#   EVAL  corpus : takala/financial_phrasebank            (config sentences_allagree,
#                  2264 rows, column "sentence")
#
# Financial PhraseBank (FPB) is one of the four constituent sources that were
# aggregated into FinGPT/fingpt-sentiment-train. As a result, a large fraction
# of the 2264 FPB "sentences_allagree" evaluation sentences appear VERBATIM
# inside the FinGPT training inputs -- with the same gold labels -- so the model
# was scored on data it was trained on. This script measures that overlap at
# three increasingly permissive levels and then writes out a DECONTAMINATED FPB
# test set (the sentences that are genuinely unseen) so the model can be
# re-evaluated honestly.
#
# Three overlap measures:
#   (1) EXACT       : identical raw strings.
#   (2) NORMALIZED  : lowercase, NFKC-unicode-normalize, collapse whitespace,
#                     strip surrounding punctuation (FPB uses spaced punctuation
#                     such as "USD 2.3 mn ." so normalization is essential).
#   (3) FUZZY       : near-duplicate detection. Uses MinHash + LSH (datasketch)
#                     if installed, otherwise falls back to a brute-force
#                     token-set Jaccard with threshold 0.9. Both are blocked /
#                     bucketed so the comparison stays tractable.
#
# Output:
#   - Console report: count + percentage of the 2264 FPB sentences found in the
#     training corpus at each level, with example matched pairs, label-consistency
#     stats, and class-balance of the decontaminated remainder.
#   - CSV: fpb_decontaminated.csv  -- the FPB sentences NOT present in training
#     at the normalized+fuzzy level, ready for a clean re-evaluation.
#
# Design goals: robust (multiple dataset-loading fallbacks), dependency-light
# (only `datasets` is required; `datasketch` is optional), and well commented.
#
# Usage:
#   python leakage_analysis.py
#   python leakage_analysis.py --fuzzy-threshold 0.9 --max-examples 8 \
#                              --out fpb_decontaminated.csv --no-fuzzy
# =============================================================================

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# -----------------------------------------------------------------------------
# Dataset identifiers and the canonical 3-class label map for FPB.
# -----------------------------------------------------------------------------
FINGPT_DATASET = "FinGPT/fingpt-sentiment-train"   # train corpus; column "input"
FINGPT_INPUT_COL = "input"
FINGPT_OUTPUT_COL = "output"

FPB_DATASET = "takala/financial_phrasebank"        # canonical eval corpus
FPB_CONFIG = "sentences_allagree"                  # 2264 rows
FPB_SENTENCE_COL = "sentence"
FPB_LABEL_COL = "label"

# Parquet mirror used as a fallback because the canonical `takala` loader is a
# dataset script and is unsupported by `datasets` >= 4.x. The mirror's row count
# (2264) and class distribution match the canonical config exactly.
FPB_MIRROR = "gtfintechlab/financial_phrasebank_sentences_allagree"
FPB_MIRROR_CONFIGS = ["5768", "78516", "944601"]   # all yield the same 2264 rows

# Canonical takala FPB integer->name map (0=neg, 1=neu, 2=pos).
FPB_ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Maps FinGPT's 9-point output vocabulary onto the FPB 3-class scheme so we can
# check whether a leaked sentence also leaked its (collapsed) gold label.
FINGPT_LABEL_TO_3CLASS = {
    "positive": "positive",
    "moderately positive": "positive",
    "mildly positive": "positive",
    "strong positive": "positive",
    "negative": "negative",
    "moderately negative": "negative",
    "mildly negative": "negative",
    "strong negative": "negative",
    "neutral": "neutral",
}


# =============================================================================
# Normalization
# =============================================================================
_WS_RE = re.compile(r"\s+")
# Leading/trailing run of anything that is not a word char (letter/digit/_).
# This strips surrounding punctuation/quotes/spaces but keeps internal text.
_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$", flags=re.UNICODE)
# For the fuzzy token-set measure: keep only alphanumeric tokens.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text(text: str) -> str:
    """Aggressive normalization for measure (2).

    Steps: NFKC unicode-normalize -> lowercase -> collapse all whitespace runs
    to single spaces -> strip surrounding (edge) punctuation/quotes/spaces.

    This is deliberately strong because FPB renders punctuation with surrounding
    spaces (e.g. "Order intake grew by 40 % year-on-year ." or "USD 2.3 mn ."),
    and trivial loader differences (trailing period, smart quotes, NBSP) should
    not be allowed to hide a genuine verbatim overlap.
    """
    if text is None:
        return ""
    t = unicodedata.normalize("NFKC", str(text))
    t = t.lower()
    t = _WS_RE.sub(" ", t).strip()
    t = _EDGE_PUNCT_RE.sub("", t)
    return t


def token_set(text: str) -> frozenset:
    """Alphanumeric token set for the Jaccard fuzzy measure (3)."""
    return frozenset(_TOKEN_RE.findall(unicodedata.normalize("NFKC", str(text)).lower()))


def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity of two token sets; 0.0 if either is empty."""
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


# =============================================================================
# Dataset loading (robust, with fallbacks)
# =============================================================================
def load_fingpt_inputs() -> Tuple[List[str], List[str]]:
    """Return (inputs, outputs) from FinGPT/fingpt-sentiment-train.

    `outputs` is the raw FinGPT label string (9-point scale); callers collapse
    it to 3 classes via FINGPT_LABEL_TO_3CLASS when needed.
    """
    from datasets import load_dataset  # imported lazily so --help works offline

    print(f"[load] FinGPT train corpus: {FINGPT_DATASET} ...", flush=True)
    ds = load_dataset(FINGPT_DATASET)
    split = ds["train"] if "train" in ds else ds[list(ds.keys())[0]]
    inputs = list(split[FINGPT_INPUT_COL])
    outputs = (
        list(split[FINGPT_OUTPUT_COL])
        if FINGPT_OUTPUT_COL in split.column_names
        else [""] * len(inputs)
    )
    print(f"[load]   -> {len(inputs):,} training rows.", flush=True)
    return inputs, outputs


def load_fpb_allagree() -> Tuple[List[str], List[str]]:
    """Return (sentences, label_names) for FPB sentences_allagree (2264 rows).

    Tries, in order:
      1. canonical `takala/financial_phrasebank` (works on datasets < 4.x);
      2. the `gtfintechlab` parquet mirror (works on datasets >= 4.x).
    Labels are returned as canonical names: negative / neutral / positive.
    """
    from datasets import load_dataset, concatenate_datasets

    # --- Attempt 1: canonical takala loader (script-based) -------------------
    try:
        print(f"[load] FPB eval corpus: {FPB_DATASET} [{FPB_CONFIG}] ...", flush=True)
        ds = load_dataset(FPB_DATASET, FPB_CONFIG)
        split = ds["train"]
        sentences = list(split[FPB_SENTENCE_COL])
        feat = split.features[FPB_LABEL_COL]
        # ClassLabel feature carries the authoritative names.
        names = getattr(feat, "names", None)
        if names:
            labels = [names[i] for i in split[FPB_LABEL_COL]]
        else:
            labels = [FPB_ID2LABEL.get(int(i), str(i)) for i in split[FPB_LABEL_COL]]
        print(f"[load]   -> {len(sentences):,} eval rows (canonical loader).", flush=True)
        return sentences, labels
    except Exception as exc:  # noqa: BLE001 -- any failure -> try the mirror
        print(f"[load]   canonical loader unavailable ({type(exc).__name__}); "
              f"falling back to parquet mirror.", flush=True)

    # --- Attempt 2: gtfintechlab parquet mirror ------------------------------
    last_err: Optional[Exception] = None
    for cfg in FPB_MIRROR_CONFIGS:
        try:
            print(f"[load]   trying mirror {FPB_MIRROR} [{cfg}] ...", flush=True)
            ds = load_dataset(FPB_MIRROR, cfg)
            parts = [ds[s] for s in ds.keys()]            # combine train+test
            alld = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
            sentences = list(alld[FPB_SENTENCE_COL])
            feat = alld.features[FPB_LABEL_COL]
            names = getattr(feat, "names", None)
            if names:
                labels = [names[i] for i in alld[FPB_LABEL_COL]]
            else:
                # Mirror stores plain ints -> use the canonical FPB map.
                labels = [FPB_ID2LABEL.get(int(i), str(i)) for i in alld[FPB_LABEL_COL]]
            print(f"[load]   -> {len(sentences):,} eval rows (mirror {cfg}).", flush=True)
            if len(sentences) != 2264:
                print(f"[load]   WARNING: expected 2264 FPB rows, got {len(sentences)}.",
                      flush=True)
            return sentences, labels
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            print(f"[load]   mirror {cfg} failed ({type(exc).__name__}).", flush=True)

    raise RuntimeError(
        "Could not load Financial PhraseBank sentences_allagree from either the "
        f"canonical loader or the parquet mirror. Last error: {last_err!r}"
    )


# =============================================================================
# Overlap measures
# =============================================================================
def exact_overlap(
    fpb_sentences: Sequence[str],
    train_inputs: Sequence[str],
) -> Set[int]:
    """Measure (1): indices of FPB sentences whose RAW string is in the train set."""
    train_set = set(train_inputs)
    return {i for i, s in enumerate(fpb_sentences) if s in train_set}


def normalized_overlap(
    fpb_norm: Sequence[str],
    train_norm_set: Set[str],
) -> Set[int]:
    """Measure (2): indices of FPB sentences whose NORMALIZED form is in train."""
    return {i for i, n in enumerate(fpb_norm) if n and n in train_norm_set}


def fuzzy_overlap_jaccard(
    fpb_sentences: Sequence[str],
    train_inputs: Sequence[str],
    threshold: float,
    already_matched: Set[int],
) -> Set[int]:
    """Measure (3) fallback: brute-force token-set Jaccard >= threshold.

    Only FPB indices NOT already caught by exact/normalized matching are tested
    (that is the only interesting set, and it keeps the work small). To avoid an
    O(N_fpb * N_train) blow-up we bucket the training corpus by token-length and,
    within a length bucket, by a few "anchor" tokens; a candidate train row is
    only compared if it shares at least one anchor token and a compatible length
    with the FPB sentence. With threshold 0.9 this loses no true matches in
    practice because near-duplicates necessarily share most tokens.
    """
    targets = [i for i in range(len(fpb_sentences)) if i not in already_matched]
    if not targets:
        return set()

    # Pre-tokenize the training corpus and build an inverted index:
    #   token -> list of (train_idx, token_set, n_tokens)
    print(f"[fuzzy] indexing {len(train_inputs):,} training rows for Jaccard ...",
          flush=True)
    train_tokensets: List[frozenset] = [token_set(t) for t in train_inputs]
    inverted: Dict[str, List[int]] = defaultdict(list)
    for idx, ts in enumerate(train_tokensets):
        if not ts:
            continue
        # Index by the rarest-looking proxy: the 4 longest tokens (longer tokens
        # are rarer, which keeps candidate lists short). Any shared rare token is
        # enough to surface a candidate for Jaccard>=0.9.
        for tok in sorted(ts, key=len, reverse=True)[:4]:
            inverted[tok].append(idx)

    matched: Set[int] = set()
    for n, i in enumerate(targets):
        if n and n % 200 == 0:
            print(f"[fuzzy]   checked {n}/{len(targets)} candidate FPB rows ...",
                  flush=True)
        fs = token_set(fpb_sentences[i])
        if not fs:
            continue
        # Collect candidate train rows that share an anchor token.
        cand: Set[int] = set()
        for tok in sorted(fs, key=len, reverse=True)[:6]:
            cand.update(inverted.get(tok, ()))
        # Length pre-filter: |A|/|B| must be >= threshold for Jaccard >= threshold.
        lf = len(fs)
        for cidx in cand:
            ts = train_tokensets[cidx]
            lt = len(ts)
            if lt == 0:
                continue
            lo, hi = (lf, lt) if lf <= lt else (lt, lf)
            if lo / hi < threshold:        # impossible to reach threshold
                continue
            if jaccard(fs, ts) >= threshold:
                matched.add(i)
                break
    return matched


# Optional, faster fuzzy path via datasketch MinHash/LSH ----------------------
def fuzzy_overlap_minhash(
    fpb_sentences: Sequence[str],
    train_inputs: Sequence[str],
    threshold: float,
    already_matched: Set[int],
    num_perm: int = 128,
) -> Set[int]:
    """Measure (3) preferred: MinHash + LSH near-duplicate detection.

    Builds an LSH index over the training corpus' token-set MinHashes, then
    queries each not-yet-matched FPB sentence. Approximates Jaccard >= threshold.
    """
    from datasketch import MinHash, MinHashLSH  # type: ignore

    targets = [i for i in range(len(fpb_sentences)) if i not in already_matched]
    if not targets:
        return set()

    def make_mh(tokens: Iterable[str]) -> "MinHash":
        m = MinHash(num_perm=num_perm)
        for tok in tokens:
            m.update(tok.encode("utf-8"))
        return m

    print(f"[fuzzy] building MinHash-LSH over {len(train_inputs):,} train rows "
          f"(num_perm={num_perm}) ...", flush=True)
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for idx, txt in enumerate(train_inputs):
        ts = token_set(txt)
        if not ts:
            continue
        lsh.insert(f"t{idx}", make_mh(ts))

    matched: Set[int] = set()
    for i in targets:
        ts = token_set(fpb_sentences[i])
        if not ts:
            continue
        if lsh.query(make_mh(ts)):
            matched.add(i)
    return matched


def run_fuzzy(
    fpb_sentences: Sequence[str],
    train_inputs: Sequence[str],
    threshold: float,
    already_matched: Set[int],
) -> Tuple[Set[int], str]:
    """Dispatch to MinHash/LSH if datasketch is present, else Jaccard fallback.

    Returns (matched_indices, backend_name).
    """
    try:
        import datasketch  # noqa: F401
        return (
            fuzzy_overlap_minhash(fpb_sentences, train_inputs, threshold, already_matched),
            "datasketch MinHash/LSH",
        )
    except ImportError:
        print("[fuzzy] datasketch not installed -> using token-set Jaccard fallback.",
              flush=True)
        return (
            fuzzy_overlap_jaccard(fpb_sentences, train_inputs, threshold, already_matched),
            "token-set Jaccard (brute-force)",
        )


# =============================================================================
# Reporting helpers
# =============================================================================
def pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):.2f}%" if total else "n/a"


def build_norm_label_index(
    train_inputs: Sequence[str],
    train_outputs: Sequence[str],
) -> Dict[str, Set[str]]:
    """normalized-train-input -> set of collapsed 3-class labels seen for it.

    Used to verify that a leaked sentence also carried its gold label into the
    training set (label leakage, not just text overlap).
    """
    idx: Dict[str, Set[str]] = defaultdict(set)
    for txt, out in zip(train_inputs, train_outputs):
        n = normalize_text(txt)
        if not n:
            continue
        lab3 = FINGPT_LABEL_TO_3CLASS.get(str(out).strip().lower())
        if lab3:
            idx[n].add(lab3)
    return idx


def print_examples(
    title: str,
    fpb_sentences: Sequence[str],
    matched_idx: Iterable[int],
    norm2train_raw: Optional[Dict[str, str]],
    max_examples: int,
) -> None:
    """Print up to `max_examples` matched FPB sentences (and their train twin)."""
    print(f"\n  Example matches ({title}):")
    shown = 0
    for i in matched_idx:
        if shown >= max_examples:
            break
        fpb = fpb_sentences[i]
        line = f"    [{i}] FPB : {fpb!r}"
        print(line[:200])
        if norm2train_raw is not None:
            twin = norm2train_raw.get(normalize_text(fpb))
            if twin is not None and twin != fpb:
                print(f"         TRAIN: {twin!r}"[:200])
        shown += 1
    if shown == 0:
        print("    (none)")


# =============================================================================
# Main
# =============================================================================
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Quantify FinGPT-train <-> Financial PhraseBank overlap and "
                    "emit a decontaminated FPB test set.")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.9,
                        help="Jaccard / LSH near-duplicate threshold (default 0.9).")
    parser.add_argument("--max-examples", type=int, default=6,
                        help="Number of example matched pairs to print per level.")
    parser.add_argument("--out", default="fpb_decontaminated.csv",
                        help="Output CSV path for the decontaminated FPB test set.")
    parser.add_argument("--no-fuzzy", action="store_true",
                        help="Skip the (slower) fuzzy near-duplicate measure.")
    args = parser.parse_args(argv)

    t0 = time.time()
    print("=" * 78)
    print(" FinGPT-train  <->  Financial PhraseBank (sentences_allagree)")
    print(" Train/test contamination analysis")
    print("=" * 78)

    # --- Load both corpora ---------------------------------------------------
    train_inputs, train_outputs = load_fingpt_inputs()
    fpb_sentences, fpb_labels = load_fpb_allagree()
    n_fpb = len(fpb_sentences)

    # FPB class balance (for context; majority class drives the "baseline" floor).
    fpb_dist = Counter(fpb_labels)
    print("\n[info] FPB sentences_allagree class balance:")
    for lab in ("negative", "neutral", "positive"):
        c = fpb_dist.get(lab, 0)
        print(f"        {lab:<9}: {c:5d}  ({pct(c, n_fpb)})")
    maj_lab, maj_c = fpb_dist.most_common(1)[0]
    print(f"        majority-class ('{maj_lab}') accuracy floor = {pct(maj_c, n_fpb)}")

    # --- Pre-compute normalized forms ---------------------------------------
    print("\n[prep] normalizing both corpora ...", flush=True)
    fpb_norm = [normalize_text(s) for s in fpb_sentences]
    train_norm_set: Set[str] = set()
    norm2train_raw: Dict[str, str] = {}   # one representative raw train string per norm
    for txt in train_inputs:
        n = normalize_text(txt)
        if n:
            train_norm_set.add(n)
            norm2train_raw.setdefault(n, txt)

    # --- (1) EXACT -----------------------------------------------------------
    print("\n" + "-" * 78)
    print(" MEASURE 1: EXACT string match")
    print("-" * 78)
    exact_idx = exact_overlap(fpb_sentences, train_inputs)
    print(f"  FPB sentences found verbatim in training inputs: "
          f"{len(exact_idx):4d} / {n_fpb}  ({pct(len(exact_idx), n_fpb)})")
    print_examples("exact", fpb_sentences, sorted(exact_idx), None, args.max_examples)

    # --- (2) NORMALIZED ------------------------------------------------------
    print("\n" + "-" * 78)
    print(" MEASURE 2: NORMALIZED match (lowercase, NFKC, collapse ws, strip edge punct)")
    print("-" * 78)
    norm_idx = normalized_overlap(fpb_norm, train_norm_set)
    print(f"  FPB sentences found (normalized) in training inputs: "
          f"{len(norm_idx):4d} / {n_fpb}  ({pct(len(norm_idx), n_fpb)})")
    print(f"  (normalized match adds {len(norm_idx - exact_idx)} beyond exact)")
    print_examples("normalized", fpb_sentences, sorted(norm_idx),
                   norm2train_raw, args.max_examples)

    # --- Label-leakage check on the normalized matches -----------------------
    print("\n  Label-leakage check (does the leaked sentence also carry its gold label?):")
    norm_label_index = build_norm_label_index(train_inputs, train_outputs)
    label_consistent = 0
    label_present = 0
    per_class_match: Counter = Counter()
    per_class_total: Counter = Counter()
    for i in norm_idx:
        gold = fpb_labels[i]
        per_class_total[gold] += 1
        train_labs = norm_label_index.get(fpb_norm[i])
        if train_labs:
            label_present += 1
            if gold in train_labs:
                label_consistent += 1
                per_class_match[gold] += 1
    if norm_idx:
        print(f"        of {len(norm_idx)} normalized matches, {label_present} carry a "
              f"FinGPT label; {label_consistent} ({pct(label_consistent, len(norm_idx))}) "
              f"match the FPB gold label exactly.")
        for lab in ("negative", "neutral", "positive"):
            tot = per_class_total.get(lab, 0)
            if tot:
                print(f"          {lab:<9}: {per_class_match.get(lab,0):4d}/{tot:<4d} "
                      f"label-consistent matches  (match rate {pct(per_class_match.get(lab,0), tot)})")

    # --- (3) FUZZY -----------------------------------------------------------
    fuzzy_idx: Set[int] = set()
    fuzzy_backend = "skipped"
    if not args.no_fuzzy:
        print("\n" + "-" * 78)
        print(f" MEASURE 3: FUZZY near-duplicate (threshold {args.fuzzy_threshold})")
        print("-" * 78)
        already = exact_idx | norm_idx
        fuzzy_new, fuzzy_backend = run_fuzzy(
            fpb_sentences, train_inputs, args.fuzzy_threshold, already)
        fuzzy_idx = already | fuzzy_new
        print(f"  Backend: {fuzzy_backend}")
        print(f"  FPB sentences matched at fuzzy level (cumulative): "
              f"{len(fuzzy_idx):4d} / {n_fpb}  ({pct(len(fuzzy_idx), n_fpb)})")
        print(f"  (fuzzy adds {len(fuzzy_new)} beyond exact+normalized)")
        print_examples("fuzzy-only new", fpb_sentences, sorted(fuzzy_new),
                       None, args.max_examples)
    else:
        print("\n[fuzzy] skipped (--no-fuzzy).")

    # --- Summary -------------------------------------------------------------
    contaminated = exact_idx | norm_idx | fuzzy_idx   # union of all levels
    print("\n" + "=" * 78)
    print(" SUMMARY OF OVERLAP")
    print("=" * 78)
    print(f"  total FPB eval sentences          : {n_fpb}")
    print(f"  (1) exact match                   : {len(exact_idx):4d}  ({pct(len(exact_idx), n_fpb)})")
    print(f"  (2) normalized match              : {len(norm_idx):4d}  ({pct(len(norm_idx), n_fpb)})")
    if not args.no_fuzzy:
        print(f"  (3) fuzzy (>= {args.fuzzy_threshold}) cumulative   : "
              f"{len(fuzzy_idx):4d}  ({pct(len(fuzzy_idx), n_fpb)})")
    print(f"  CONTAMINATED (union of levels)    : {len(contaminated):4d}  "
          f"({pct(len(contaminated), n_fpb)})")
    print(f"  CLEAN / unseen remainder          : {n_fpb - len(contaminated):4d}  "
          f"({pct(n_fpb - len(contaminated), n_fpb)})")

    # --- Build & save the decontaminated FPB test set ------------------------
    # "Decontaminated" = NOT present in training at the normalized OR fuzzy level.
    # (Exact is a strict subset of normalized, so the union below is the right set.)
    contaminated_for_decon = norm_idx | fuzzy_idx
    clean_idx = [i for i in range(n_fpb) if i not in contaminated_for_decon]

    out_path = os.path.abspath(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sentence", "label"])
        for i in clean_idx:
            writer.writerow([fpb_sentences[i], fpb_labels[i]])

    clean_dist = Counter(fpb_labels[i] for i in clean_idx)
    n_clean = len(clean_idx)
    print("\n" + "=" * 78)
    print(" DECONTAMINATED FPB TEST SET")
    print("=" * 78)
    print(f"  removed (normalized|fuzzy contaminated) : {n_fpb - n_clean}")
    print(f"  retained (truly unseen)                 : {n_clean}  ({pct(n_clean, n_fpb)})")
    print(f"  saved to                                : {out_path}")
    print(f"  class balance of decontaminated set:")
    for lab in ("negative", "neutral", "positive"):
        c = clean_dist.get(lab, 0)
        print(f"        {lab:<9}: {c:4d}  ({pct(c, n_clean)})")
    if n_clean:
        cmaj_lab, cmaj_c = clean_dist.most_common(1)[0]
        print(f"  majority-class ('{cmaj_lab}') floor on clean set = {pct(cmaj_c, n_clean)}")

    print("\n[done] elapsed %.1fs" % (time.time() - t0))
    print("\nInterpretation: re-evaluate Finistral-7B-LoRA (and all baselines) on")
    print(f"  '{os.path.basename(out_path)}' to obtain a contamination-free accuracy/F1.")
    print("  A large drop from the reported 99.56% on this disjoint remainder confirms")
    print("  the headline number was inflated by train/test contamination.")
    return 0


# -----------------------------------------------------------------------------
# Contract entry point for ablation_and_seeds.py (see its SECTION 1).
#
# Returns the frozen decontaminated FPB eval set WITHOUT any network access by
# reading the committed CSV produced by this script / measure_leakage_local.py.
# The canonical copy lives at revision/fpb_decontaminated.csv (560 rows); an
# identical copy sits at the repo root.
# -----------------------------------------------------------------------------
def build_decontaminated_fpb() -> Tuple[List[str], List[int]]:
    """Return (sentences, label_ids) for the decontaminated FPB set.

    label_ids use the canonical 3-class convention {negative:0, neutral:1,
    positive:2} shared by ablation_and_seeds.py and eval_harness_fixed.py.
    Raises FileNotFoundError with remediation instructions if no committed
    decontaminated CSV can be found.
    """
    label2id = {name: i for i, name in FPB_ID2LABEL.items()}  # invert 0/1/2 map
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "revision", "fpb_decontaminated.csv"),
        os.path.join(here, "fpb_decontaminated.csv"),
        os.path.join(here, "..", "fpb_decontaminated.csv"),  # revision/scripts copy
    ]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        raise FileNotFoundError(
            "No decontaminated FPB CSV found (looked for %s). Run "
            "`python leakage_analysis.py` or "
            "`python revision/scripts/measure_leakage_local.py` first."
            % ", ".join(candidates))
    sentences: List[str] = []
    labels: List[int] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lab = row["label"].strip().lower()
            if lab not in label2id:
                raise ValueError("unrecognized label %r in %s" % (row["label"], path))
            sentences.append(row["sentence"])
            labels.append(label2id[lab])
    if not (500 <= len(sentences) <= 600):
        raise ValueError(
            "decontaminated set has %d rows; expected ~560 -- refusing to "
            "return a possibly contaminated or truncated set (%s)"
            % (len(sentences), path))
    return sentences, labels


if __name__ == "__main__":
    sys.exit(main())
