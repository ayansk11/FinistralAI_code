#!/usr/bin/env python3
"""Near-duplicate audit of the evaluation sets against the FinGPT training corpus.

Answers R1.3's near-duplicate question beyond exact/normalized matching:
for every sentence in each frozen evaluation set, compute the maximum
token-set Jaccard similarity against any FinGPT training input, and report
counts at thresholds 0.7 / 0.8 / 0.9 plus the offenders.

Method: normalized token-set Jaccard with an inverted-index blocking scheme
(candidates = training rows sharing >= 1 of the eval sentence's 4 longest
tokens, plus a token-count ratio prefilter), mirroring leakage_analysis.py's
brute-force fuzzy fallback. CPU-only, deterministic, no network beyond the
datasets download cache.

Usage (repo root):  python revision/scripts/near_duplicate_audit.py
Writes: data_eval/near_duplicate_audit.json + .md
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

from datasets import load_dataset

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_JSON = os.path.join(REPO, "data_eval", "near_duplicate_audit.json")
OUT_MD = os.path.join(REPO, "data_eval", "near_duplicate_audit.md")
THRESHOLDS = (0.7, 0.8, 0.9)


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokens(text: str) -> frozenset:
    return frozenset(normalize(text).split())


def main() -> int:
    print(">> loading FinGPT training inputs ...", flush=True)
    fin = load_dataset("FinGPT/fingpt-sentiment-train", split="train")["input"]
    train_tok = []
    seen = set()
    for s in fin:
        n = normalize(s)
        if n in seen:
            continue
        seen.add(n)
        train_tok.append(frozenset(n.split()))
    print(f"   {len(train_tok)} unique normalized training inputs", flush=True)

    # Inverted index: token -> train row ids, restricted to each row's 4
    # longest tokens (a pair with Jaccard >= 0.7 shares most tokens, so
    # sharing one of the 4 longest is near-certain).
    index = defaultdict(list)
    for i, ts in enumerate(train_tok):
        for tok in sorted(ts, key=len, reverse=True)[:4]:
            index[tok].append(i)

    results = {}
    for name in ("fpb_decontam", "fiqa", "tfns"):
        path = os.path.join(REPO, "data_eval", f"{name}.csv")
        rows = [r["sentence"] for r in csv.DictReader(open(path, encoding="utf-8"))]
        counts = {str(t): 0 for t in THRESHOLDS}
        max_sims = []
        offenders = []
        for sent in rows:
            ets = tokens(sent)
            if not ets:
                max_sims.append(0.0)
                continue
            cands = set()
            for tok in sorted(ets, key=len, reverse=True)[:4]:
                cands.update(index.get(tok, ()))
            best = 0.0
            lo, hi = len(ets) * 0.5, len(ets) * 2.0  # ratio prefilter for J>=0.5
            for ci in cands:
                cts = train_tok[ci]
                if not (lo <= len(cts) <= hi):
                    continue
                inter = len(ets & cts)
                if inter == 0:
                    continue
                j = inter / (len(ets) + len(cts) - inter)
                if j > best:
                    best = j
            max_sims.append(best)
            for t in THRESHOLDS:
                if best >= t:
                    counts[str(t)] += 1
            if best >= 0.7:
                offenders.append({"sentence": sent[:140], "max_jaccard": round(best, 3)})
        n = len(rows)
        results[name] = {
            "n": n,
            "counts": counts,
            "pct": {k: round(100 * v / n, 2) for k, v in counts.items()},
            "mean_max_jaccard": round(sum(max_sims) / n, 4),
            "offenders": sorted(offenders, key=lambda o: -o["max_jaccard"])[:25],
        }
        print(f"   {name}: n={n}  >=0.7: {counts['0.7']}  >=0.8: {counts['0.8']}"
              f"  >=0.9: {counts['0.9']}  mean(max J)={results[name]['mean_max_jaccard']}",
              flush=True)

    json.dump(results, open(OUT_JSON, "w"), indent=2)
    with open(OUT_MD, "w") as fh:
        fh.write("# Near-duplicate audit (token-set Jaccard vs FinGPT training inputs)\n\n"
                 "| Eval set | n | J >= 0.7 | J >= 0.8 | J >= 0.9 | mean max-J |\n|---|---|---|---|---|---|\n")
        for name, r in results.items():
            fh.write(f"| {name} | {r['n']} | {r['counts']['0.7']} ({r['pct']['0.7']}%) | "
                     f"{r['counts']['0.8']} ({r['pct']['0.8']}%) | "
                     f"{r['counts']['0.9']} ({r['pct']['0.9']}%) | {r['mean_max_jaccard']} |\n")
    print(f">> wrote {OUT_JSON} and {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
