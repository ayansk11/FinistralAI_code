#!/usr/bin/env python3
"""Generate data_eval/fpb_disjoint_indices.txt for eval_harness_fixed.py.

The indices refer to the DETERMINISTIC (sentence-sorted) ordering produced by
eval_harness_fixed.load_fpb_allagree(), which is identical whether the takala
loader or the gtfintechlab mirror serves the data. An index is emitted when
the sentence's normalized form appears in the canonical 560-row decontaminated
set (revision/fpb_decontaminated.csv). Asserts exactly 560 indices.

CPU-only. Run from the repo root:
    python revision/scripts/make_disjoint_ids.py
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import unicodedata

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_PATH = os.path.join(REPO, "data_eval", "fpb_disjoint_indices.txt")
DECONTAM_CSV = os.path.join(REPO, "revision", "fpb_decontaminated.csv")


def normalize(text: str) -> str:
    """Same scheme as measure_leakage_local.py (the canonical 560-set)."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def main() -> int:
    spec = importlib.util.spec_from_file_location(
        "eval_harness_fixed", os.path.join(REPO, "eval_harness_fixed.py"))
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)

    import csv
    clean_norms = set()
    with open(DECONTAM_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            clean_norms.add(normalize(row["sentence"]))
    assert len(clean_norms) == 560, f"expected 560 unique, got {len(clean_norms)}"

    sentences, _ = harness.load_fpb_allagree(
        os.environ.get("HF_TOKEN"), disjoint_ids=None)

    indices, seen = [], set()
    for i, s in enumerate(sentences):
        key = normalize(s)
        if key in clean_norms and key not in seen:  # first occurrence only
            seen.add(key)
            indices.append(i)
    assert len(indices) == 560, (
        f"matched {len(indices)} of 560 canonical sentences -- ordering or "
        f"normalization drift; refusing to write a partial index file")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(str(i) for i in indices) + "\n")
    print(f"wrote {len(indices)} indices -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
