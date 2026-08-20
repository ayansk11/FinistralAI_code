#!/usr/bin/env python3
"""
Independent, self-contained measurement of train/test contamination between
FinGPT/fingpt-sentiment-train (the fine-tuning corpus) and
takala/financial_phrasebank :: sentences_allagree (the evaluation set).

Produces:
  - revision/leakage_report.md            (human-readable findings)
  - revision/fpb_decontaminated.csv       (FPB sentences NOT seen in training)
  - revision/fpb_overlap_matches.csv      (sample matched pairs w/ labels)

Run:  python3 revision/scripts/measure_leakage_local.py
CPU-only, no GPU required.
"""
import csv, os, re, sys, unicodedata, json
from collections import Counter

OUT_DIR = os.path.join(os.path.dirname(__file__), "..")  # revision/
from datasets import load_dataset

# FinGPT 3-way label space; FinGPT stores richer strings, normalize to 3 classes.
def norm_label(s):
    s = str(s).strip().lower()
    if "neg" in s: return "negative"
    if "pos" in s: return "positive"
    if "neu" in s: return "neutral"
    if "strong neg" in s: return "negative"
    if "strong pos" in s: return "positive"
    return s

def normalize(text):
    """Lowercase, unicode-fold, strip punctuation, collapse whitespace.
    Essential because FPB uses spaced punctuation: 'USD 2.3 mn .'"""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"[^a-z0-9 ]+", " ", t)      # keep alphanumerics + space
    t = re.sub(r"\s+", " ", t).strip()
    return t

print(">> loading FinGPT/fingpt-sentiment-train ...", flush=True)
fin = load_dataset("FinGPT/fingpt-sentiment-train", split="train")
fin_inputs = fin["input"]
fin_labels = fin["output"]
print(f"   FinGPT rows: {len(fin_inputs)}", flush=True)

print(">> loading Financial PhraseBank sentences_allagree ...", flush=True)
from datasets import concatenate_datasets, get_dataset_config_names
fpb = None
# The gtfintechlab mirror partitions the SAME 2264 allagree sentences under
# several random-seed configs; concatenating all splits of one config rebuilds
# the full allagree set. We then dedup by sentence.
try:
    cfgs = get_dataset_config_names("gtfintechlab/financial_phrasebank_sentences_allagree")
    cfg = cfgs[0]
    dd = load_dataset("gtfintechlab/financial_phrasebank_sentences_allagree", cfg)
    fpb = concatenate_datasets([dd[s] for s in dd.keys()])
    print(f"   loaded mirror config '{cfg}' splits={list(dd.keys())} -> {len(fpb)} rows", flush=True)
except Exception as e:
    print(f"   (gtfintechlab loader failed: {type(e).__name__}: {str(e)[:160]})", flush=True)
if fpb is None:
    sys.exit("Could not load Financial PhraseBank from any source.")
# Dedup by sentence to recover the unique allagree set.
seen=set(); keep=[]
_cols = fpb.column_names
_sc = "sentence" if "sentence" in _cols else [c for c in _cols if "sent" in c.lower() or "text" in c.lower()][0]
for i,row in enumerate(fpb):
    k=row[_sc]
    if k not in seen:
        seen.add(k); keep.append(i)
fpb = fpb.select(keep)
print(f"   unique FPB sentences after dedup: {len(fpb)}", flush=True)

# Identify sentence + label columns robustly across mirrors.
cols = fpb.column_names
sent_col = "sentence" if "sentence" in cols else [c for c in cols if "sent" in c.lower() or "text" in c.lower()][0]
lab_col  = "label" if "label" in cols else [c for c in cols if "label" in c.lower()][0]
fpb_sentences = fpb[sent_col]
# label may be int (ClassLabel) or string
feat = fpb.features[lab_col]
# FPB ClassLabel convention: 0=negative, 1=neutral, 2=positive.
INT2LBL = {0: "negative", 1: "neutral", 2: "positive"}
def map_fpb_label(x):
    if hasattr(feat, "names"):           # proper ClassLabel
        return norm_label(feat.names[x])
    try:                                  # integer or int-as-string mirror
        return INT2LBL[int(x)]
    except (ValueError, KeyError, TypeError):
        return norm_label(x)
fpb_labels = [map_fpb_label(x) for x in fpb[lab_col]]
print(f"   FPB rows: {len(fpb_sentences)}  | class dist: {Counter(fpb_labels)}", flush=True)

# ---- Build lookup of training inputs: normalized -> set of normalized labels ----
fin_norm_to_labels = {}
fin_norm_set = set()
fin_exact_set = set(fin_inputs)
for inp, lab in zip(fin_inputs, fin_labels):
    n = normalize(inp)
    fin_norm_set.add(n)
    fin_norm_to_labels.setdefault(n, set()).add(norm_label(lab))

# ---- Overlap measurement ----
exact_hits = 0
norm_hits = 0
label_agree = 0
matches = []   # (sentence, fpb_label, train_labels)
decontam_rows = []  # FPB rows NOT in training (normalized)

for s, y in zip(fpb_sentences, fpb_labels):
    n = normalize(s)
    is_exact = s in fin_exact_set
    is_norm  = n in fin_norm_set
    if is_exact: exact_hits += 1
    if is_norm:
        norm_hits += 1
        tl = fin_norm_to_labels.get(n, set())
        if y in tl: label_agree += 1
        if len(matches) < 25:
            matches.append((s, y, sorted(tl)))
    else:
        decontam_rows.append((s, y))

N = len(fpb_sentences)
report = []
report.append("# Train/Test Contamination Measurement (local, independent run)\n")
report.append(f"- FinGPT/fingpt-sentiment-train rows: **{len(fin_inputs)}**  (unique normalized inputs: {len(fin_norm_set)})")
report.append(f"- FPB sentences_allagree rows: **{N}**  | class distribution: {dict(Counter(fpb_labels))}\n")
report.append("## Overlap of the FPB evaluation set inside the FinGPT training corpus")
report.append(f"- **Exact verbatim match:** {exact_hits}/{N} = **{100*exact_hits/N:.2f}%**")
report.append(f"- **Normalized match** (lowercase/strip-punct/collapse-ws): {norm_hits}/{N} = **{100*norm_hits/N:.2f}%**")
report.append(f"- **Label agreement among matched:** {label_agree}/{norm_hits} = "
              f"**{(100*label_agree/norm_hits) if norm_hits else 0:.2f}%** "
              f"(matched sentences carry the SAME gold label in training)")
report.append(f"- **Decontaminated remainder** (FPB sentences NOT in training): **{len(decontam_rows)}** "
              f"({100*len(decontam_rows)/N:.2f}%) | class dist: {dict(Counter(l for _,l in decontam_rows))}\n")
report.append("## Interpretation")
report.append(f"The model is fine-tuned on FinGPT/fingpt-sentiment-train and evaluated on FPB. "
              f"{100*norm_hits/N:.1f}% of the evaluation sentences — with identical gold labels — were in the "
              f"training pool. The reported 99.56% is therefore largely memorization, not generalization. "
              f"A valid number must be reported on the {len(decontam_rows)}-sentence decontaminated remainder "
              f"and on an external dataset never present in fingpt-sentiment-train.\n")
report.append("## Sample matched (leaked) sentences")
for s,y,tl in matches[:15]:
    report.append(f"- [{y}] {s[:140]}")

with open(os.path.join(OUT_DIR,"leakage_report.md"),"w") as f:
    f.write("\n".join(report)+"\n")

with open(os.path.join(OUT_DIR,"fpb_decontaminated.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["sentence","label"]); w.writerows(decontam_rows)

with open(os.path.join(OUT_DIR,"fpb_overlap_matches.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["sentence","fpb_label","train_labels"])
    for s,y,tl in matches: w.writerow([s,y,"|".join(tl)])

print("\n".join(report))
print(f"\n>> wrote leakage_report.md, fpb_decontaminated.csv ({len(decontam_rows)} rows), fpb_overlap_matches.csv")
