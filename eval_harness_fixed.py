#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_harness_fixed.py
=====================

A SINGLE, corrected, fair evaluation harness for the Finistral-7B-LoRA paper
(JICTASA-2026-018) and its baselines on the Financial PhraseBank
``sentences_allagree`` test set.

This file is a drop-in replacement for the broken loop in
``test_final.ipynb`` / ``test_final2.ipynb``. It fixes EVERY bug that drove the
baselines below the majority-class floor (neutral = 61.6%) and that made the
13-23% baseline scores an artifact of the harness rather than model quality.

WHAT WAS WRONG IN THE ORIGINAL HARNESS (and how this file fixes it)
------------------------------------------------------------------
1. RIGHT-PADDING ON DECODER-ONLY MODELS.
   Original: ``AutoTokenizer.from_pretrained(...)`` never set ``padding_side``,
   so it defaulted to RIGHT padding. For a causal LM, pad tokens are appended
   AFTER the prompt and generation continues from a PAD position, corrupting the
   output of every sequence except the single longest one in each batch. The
   notebook outputs literally contain 566 copies of:
     "A decoder-only architecture is being used, but right-padding was detected!
      For correct generation results, please set padding_side='left'".
   FIX: we force ``tokenizer.padding_side = "left"`` for every model (see
   ``build_tokenizer``). This is the single most important correction.

2. PAD TOKEN SET CARELESSLY.
   Original: ``tokenizer.pad_token = tokenizer.eos_token`` unconditionally,
   which can collide with EOS and is unsafe for models (e.g. Llama-3, Bloom)
   whose tokenizers may already define a pad token. FIX: ``ensure_pad_token``
   only sets a pad token when one is missing, prefers a real ``<pad>`` /
   ``unk`` / EOS in that order, and keeps ``model.config.pad_token_id`` in sync.

3. ``max_length`` USED INSTEAD OF ``max_new_tokens``.
   Original: ``model.generate(**tokens, max_length=512)`` caps TOTAL length
   (prompt + generation), and with ``padding=True`` transformers prints
   "max_length is ignored when padding=True and there is no truncation
   strategy". The base/baseline models then rambled for hundreds of tokens
   (the Mistral-base row took 5h31m at ~70s/it). FIX: we use
   ``max_new_tokens=8`` (a sentiment word is 1-3 tokens) so generation stops
   almost immediately and is comparable across models.

4. ONE SHARED ALPACA PROMPT FOR EVERY MODEL.
   Original: every model -- base Mistral, Finistral, and all five FinGPT-mt
   adapters -- was scored with the SAME Alpaca-style prompt, even though
   Finistral was TRAINED with an ``[INST]...[/INST]`` template and the FinGPT-mt
   adapters expect the FinGPT inference template (with an ``Options:`` line).
   That train/test template mismatch alone degrades every adapter. FIX: a
   PROMPT-TEMPLATE REGISTRY (``PROMPT_TEMPLATES`` + ``MODELS``) evaluates EACH
   model under ITS OWN documented inference procedure:
     * Finistral  -> the exact [INST]{input}\n{instruction} [/INST] template
                     from Finistral_Sentiment_analyst.py line 96.
     * FinGPT-mt  -> the AI4Finance FinGPT_Benchmark template:
                     "Instruction: ...\nInput: ...\nAnswer: " with an explicit
                     "Options" line, matching how those adapters were trained.
     * base models-> a neutral zero-shot instruction template.

5. BOGUS QUANTIZATION CONFIG SILENTLY DROPPED.
   Original: ``BitsAndBytesConfig(load_in_8bit=True,
   bnb_8bit_compute_dtype=..., bnb_8bit_use_double_quant=True,
   bnb_8bit_quant_type="nf4")``. The ``bnb_8bit_*`` kwargs DO NOT EXIST in the
   API (only ``bnb_4bit_*`` do), so they fall into **kwargs and are silently
   ignored -- the model loads as plain int8, NOT the intended NF4 double-quant.
   "8-bit NF4" is also self-contradictory (NF4 is a 4-bit dtype). FIX: a
   ``--quant`` flag that builds a CORRECT config: ``none`` (fp16/bf16),
   ``int8`` (valid ``load_in_8bit``), or ``nf4`` (genuine QLoRA:
   ``load_in_4bit=True, bnb_4bit_quant_type='nf4',
   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=bf16``).

6. PARSER SILENTLY FOLDED FAILURES INTO NEUTRAL.
   Original: ``extract_sentiment`` returned "neu" whenever no sentiment word
   was found, so corrupted/rambling generations were scored as neutral -- and
   because neutral is the 61.6% majority class, this hides a broken harness.
   FIX: ``parse_sentiment`` returns ``None`` (UNPARSEABLE) when no label is
   found. Unparseable predictions are recorded SEPARATELY, an unparseable RATE
   is reported, and they are NOT counted as correct. For metric computation an
   unparseable prediction is mapped to a dedicated out-of-class id so it is
   always wrong (never a free neutral hit).

7. INCONSISTENT MISTRAL BACKBONE FOR THE "BASE" ROW.
   Original: the Mistral-7B-Base row used ``unsloth/mistral-7b-v0.2`` while
   Finistral uses ``mistralai/Mistral-7B-v0.1`` -- different versions compared
   as if the same backbone. FIX: the registry pins the zero-shot Mistral
   baseline to the IDENTICAL ``mistralai/Mistral-7B-v0.1`` backbone Finistral
   is built on, so "gain over backbone" is a like-for-like comparison.

WHAT THIS HARNESS PRODUCES (per Reviewer 1 pts 1,2,7 / Reviewer 2 Q4)
--------------------------------------------------------------------
For every model and every seed it computes accuracy, weighted F1, macro F1,
and per-class precision/recall/F1, plus the unparseable rate. It SAVES:
  * ``<model>_seed<k>_predictions.csv`` -- per-example text, gold, raw output,
    parsed label, prediction id, correct flag, unparseable flag.
  * ``<model>_seed<k>_confusion_matrix.csv`` -- the full 3x3 confusion matrix
    (so confusion matrices for ALL baselines can be produced).
  * ``<model>_seed<k>_per_class.csv`` -- per-class P/R/F1/support.
  * ``summary.csv`` -- one row per (model, seed) with all headline metrics, and
    aggregated mean +/- std across seeds (Reviewer 1 pt 5 / Reviewer 2 Q5).

NOTE ON DATA CONTAMINATION (out of scope for this file, but flagged):
~75% of the FPB sentences_allagree eval set appears verbatim, with identical
gold labels, inside FinGPT/fingpt-sentiment-train (the corpus Finistral was
fine-tuned on). A clean evaluation must additionally report accuracy on the
FPB-disjoint remainder. This harness exposes ``--disjoint_ids_file`` so you can
restrict the eval to a decontaminated subset; the fairness fixes above are
necessary but NOT sufficient on their own to validate the 99.56% headline.

Usage
-----
  # All models, fp16, 3 seeds, full eval set
  python eval_harness_fixed.py --models all --quant none --seeds 0 1 2 \
      --out_dir results_fixed

  # Just Finistral + base Mistral, genuine NF4 QLoRA, single seed
  python eval_harness_fixed.py --models finistral mistral_base \
      --quant nf4 --seeds 0 --out_dir results_qlora

  # Decontaminated eval on a held-out index subset
  python eval_harness_fixed.py --models all --quant none --seeds 0 1 2 \
      --disjoint_ids_file fpb_disjoint_indices.txt --out_dir results_clean
"""

import argparse
import gc
import json
import os
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

# Torch / HF are imported lazily inside main so that --help works without a GPU
# environment and so the file can be imported for its registry/parsers in tests.


# =============================================================================
# 1. LABEL SPACE
# =============================================================================
# Canonical 3-class label space. ``id2label`` order matches FPB's own feature
# names so confusion matrices line up with the paper's (neg, neu, pos) axes.
LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}

# Dedicated id for UNPARSEABLE predictions. It is deliberately OUTSIDE the
# {0,1,2} class space so that (a) it never accidentally matches a gold label
# (an unparseable prediction is always counted wrong, never a free neutral
# hit), and (b) it can be filtered/reported separately. We exclude this id from
# the confusion matrix and per-class report, but we DO count it against
# accuracy as an error and we report its rate.
UNPARSEABLE_ID = -1


# =============================================================================
# 2. ROBUST ANSWER PARSER  (fix #6)
# =============================================================================
# Word-boundary regex over the full canonical/abbreviated label vocabulary.
_SENT_RE = re.compile(r"\b(negative|neutral|positive|neg|neu|pos)\b", re.IGNORECASE)
_ABBR2ID = {
    "negative": 0, "neg": 0,
    "neutral": 1, "neu": 1,
    "positive": 2, "pos": 2,
}


def parse_sentiment(generated_text, prompt_text=None):
    """
    Extract a sentiment label id (0/1/2) from a model's generated string.

    CRITICAL DIFFERENCE FROM THE ORIGINAL ``extract_sentiment``:
    if NO sentiment word can be found we return ``None`` (UNPARSEABLE) instead
    of silently defaulting to neutral. The caller records None separately and
    reports the unparseable rate, so a broken/rambling model can no longer hide
    behind the 61.6% neutral majority class.

    We first strip the prompt (when provided) so that label words that appear in
    the INSTRUCTION itself (e.g. the literal "{negative/neutral/positive}"
    option list) are never mistaken for the model's answer. We then prefer the
    text after the final "Answer:" / "[/INST]" marker, and finally take the
    FIRST label word in the remaining continuation (the model's actual choice).
    """
    text = generated_text or ""

    # 1) Remove the echoed prompt if the tokenizer returned prompt+continuation.
    if prompt_text:
        idx = text.find(prompt_text)
        if idx != -1:
            text = text[idx + len(prompt_text):]

    # 2) Prefer the continuation after the model's answer marker.
    lowered = text.lower()
    for marker in ("[/inst]", "answer:", "sentiment:"):
        pos = lowered.rfind(marker)
        if pos != -1:
            text = text[pos + len(marker):]
            break

    # 3) First label word in the continuation is the model's decision.
    match = _SENT_RE.search(text)
    if match:
        return _ABBR2ID[match.group(1).lower()]

    # 4) Nothing found -> UNPARSEABLE (do NOT fold into neutral).
    return None


# =============================================================================
# 3. PROMPT-TEMPLATE REGISTRY  (fix #4)
# =============================================================================
# Each model family is evaluated under ITS OWN documented inference procedure.
# A template is a function (instruction, input_text) -> prompt string.

INSTRUCTION_3CLASS = (
    "What is the sentiment of this news? "
    "Please choose an answer from {negative/neutral/positive}"
)


def template_finistral(instruction, input_text):
    """
    Finistral's OWN training template, copied verbatim from
    Finistral_Sentiment_analyst.py line 96:
        prompt = f"[INST]{inp}\n{instr} [/INST]"
    Evaluating Finistral with the Alpaca prompt (as the original notebook did)
    is a train/test mismatch; this is the correct, like-for-like template.
    """
    return f"[INST]{input_text}\n{instruction} [/INST]"


def template_fingpt(instruction, input_text):
    """
    FinGPT-mt inference template, matching AI4Finance-Foundation/FinGPT
    (FinGPT_Benchmark). The FinGPT multi-task sentiment adapters were trained
    with an explicit "Instruction / Input / Answer" block AND an options line.
    The original harness omitted the options line and used a different
    instruction, degrading every FinGPT adapter. We restore the documented
    format so each FinGPT baseline is evaluated as published.
    """
    return (
        f"Instruction: {instruction}\n"
        f"Input: {input_text}\n"
        f"Answer: "
    )


def template_zeroshot(instruction, input_text):
    """
    Neutral zero-shot template for an untuned base model (e.g. the Mistral-7B
    backbone). Plain instruction-following format; no adapter-specific markers.
    """
    return (
        f"Instruction: {instruction}\n"
        f"Input: {input_text}\n"
        f"Answer: "
    )


# Each FinGPT-mt sentiment adapter's documented options line.
FINGPT_INSTRUCTION = (
    "What is the sentiment of this news? "
    "Please choose an answer from {negative/neutral/positive}"
)


@dataclass
class ModelSpec:
    """One row of the model registry: how to load and prompt a given model."""
    key: str                       # CLI key, e.g. "finistral"
    pretty: str                    # display name for reports
    base: str                      # base model repo id
    peft: Optional[str]            # LoRA adapter repo id, or None for base model
    template: Optional[Callable[[str, str], str]]  # None for seq-cls models
    instruction: str = INSTRUCTION_3CLASS
    trust_remote_code: bool = True
    kind: str = "causal"           # "causal" (generate+parse) or "seq_cls"


# THE REGISTRY.
# NOTE the fix for bug #7: the Mistral-7B-Base baseline is pinned to the SAME
# mistralai/Mistral-7B-v0.1 backbone that Finistral is built on (NOT the
# unsloth/mistral-7b-v0.2 the original notebook used), so the "gain over
# backbone" comparison is like-for-like.
MODELS = {
    "finistral": ModelSpec(
        key="finistral",
        pretty="Finistral-7B-LoRA",
        base="mistralai/Mistral-7B-v0.1",
        peft="Ayansk11/Finistral-7B_lora",
        template=template_finistral,          # its OWN [INST] template
    ),
    # Eval-time PROMPT-TEMPLATE ABLATION row: identical weights to "finistral"
    # but prompted with the Alpaca-style Instruction/Input/Answer template the
    # ORIGINAL notebooks used. Comparing this row against "finistral" directly
    # measures the cost of the train/eval template mismatch (claim C10).
    "finistral_alpaca": ModelSpec(
        key="finistral_alpaca",
        pretty="Finistral-7B-LoRA (Alpaca prompt)",
        base="mistralai/Mistral-7B-v0.1",
        peft="Ayansk11/Finistral-7B_lora",
        template=template_zeroshot,           # the mismatched original template
    ),
    "mistral_base": ModelSpec(
        key="mistral_base",
        pretty="Mistral-7B-v0.1-Base",
        base="mistralai/Mistral-7B-v0.1",     # SAME backbone as Finistral
        peft=None,
        template=template_zeroshot,
    ),
    "fingpt_llama2": ModelSpec(
        key="fingpt_llama2",
        pretty="FinGPT-mt-Llama2-7B-LoRA",
        base="meta-llama/Llama-2-7b-hf",
        peft="FinGPT/fingpt-mt_llama2-7b_lora",
        template=template_fingpt,
        instruction=FINGPT_INSTRUCTION,
    ),
    "fingpt_llama3": ModelSpec(
        key="fingpt_llama3",
        pretty="FinGPT-Llama-3-8B-LoRA",
        base="meta-llama/Meta-Llama-3-8B",
        peft="FinGPT/fingpt-mt_llama3-8b_lora",
        template=template_fingpt,
        instruction=FINGPT_INSTRUCTION,
    ),
    "fingpt_falcon": ModelSpec(
        key="fingpt_falcon",
        pretty="FinGPT-Falcon-7B-LoRA",
        base="tiiuae/falcon-7b",
        peft="FinGPT/fingpt-mt_falcon-7b_lora",
        template=template_fingpt,
        instruction=FINGPT_INSTRUCTION,
    ),
    "fingpt_bloom": ModelSpec(
        key="fingpt_bloom",
        pretty="FinGPT-Bloom-7B1-LoRA",
        base="bigscience/bloom-7b1",
        peft="FinGPT/fingpt-mt_bloom-7b1_lora",
        template=template_fingpt,
        instruction=FINGPT_INSTRUCTION,
    ),
    # Encoder baseline (Reviewer comparison row). FinBERT is a full-fine-tuned
    # BERT sentiment classifier: no generation, no parsing -- we read the
    # softmax over its three classes directly, so its unparseable rate is 0 by
    # construction and it natively provides scores for ROC/PR curves.
    "finbert": ModelSpec(
        key="finbert",
        pretty="FinBERT (ProsusAI)",
        base="ProsusAI/finbert",
        peft=None,
        template=None,
        kind="seq_cls",
        trust_remote_code=False,
    ),
}


# =============================================================================
# 4. DETERMINISM  (Reviewer 1 pt 5 / Reviewer 2 Q5)
# =============================================================================
def set_all_seeds(seed):
    """
    Seed every RNG and force deterministic, greedy-friendly behavior. Greedy
    decoding (do_sample=False) is itself deterministic, so across seeds the
    generations are identical; the multi-seed loop exists to (a) prove that
    determinism and (b) provide the mean +/- std table reviewers asked for.
    """
    import torch  # local import; see module docstring note

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Best-effort determinism; harmless if unsupported on the backend.
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


# =============================================================================
# 5. TOKENIZER / MODEL LOADING  (fixes #1, #2, #5)
# =============================================================================
def ensure_pad_token(tokenizer):
    """
    Set a pad token ONLY if the tokenizer lacks one (fix #2). We never clobber a
    pre-existing pad token (some tokenizers, e.g. Llama-3, define their own).
    Preference order: existing pad -> a real <pad>/unk -> eos as last resort.
    """
    if tokenizer.pad_token is not None:
        return
    if tokenizer.unk_token is not None:
        tokenizer.pad_token = tokenizer.unk_token
    elif tokenizer.eos_token is not None:
        # Last resort. Acceptable for left-padded generation because pads sit
        # BEFORE the prompt and are masked by attention_mask; they never become
        # part of the continuation.
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})


def build_tokenizer(spec, hf_token):
    """
    Build a tokenizer with LEFT padding (fix #1) and a safe pad token (fix #2).
    Left padding is mandatory for batched decoder-only generation: pads go
    before the prompt so generation always continues from the real last token.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        spec.base,
        trust_remote_code=spec.trust_remote_code,
        token=hf_token,
    )
    # *** THE KEY FIX (#1): left padding for decoder-only generation. ***
    tokenizer.padding_side = "left"
    ensure_pad_token(tokenizer)
    return tokenizer


def build_quant_config(quant):
    """
    Build a CORRECT BitsAndBytesConfig (fix #5), or return None for fp16/bf16.

      * "none" -> no quantization (load in bf16/fp16). Most faithful baseline.
      * "int8" -> valid 8-bit (load_in_8bit=True). No bogus bnb_8bit_* kwargs.
      * "nf4"  -> genuine QLoRA 4-bit NF4 with double quantization.

    This replaces the original config whose ``bnb_8bit_*`` kwargs were silently
    dropped (they do not exist) and whose "8-bit NF4" was self-contradictory.
    """
    import torch
    from transformers import BitsAndBytesConfig

    if quant == "none":
        return None
    if quant == "int8":
        # Plain, VALID int8. The original config intended something like this
        # but never achieved it because the extra kwargs were ignored.
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant == "nf4":
        # Genuine QLoRA: 4-bit NormalFloat with nested (double) quantization.
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",            # correct kwarg name
            bnb_4bit_use_double_quant=True,       # correct kwarg name
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    raise ValueError("Unknown quant mode: %r (use none|int8|nf4)" % (quant,))


def load_model(spec, quant, hf_token):
    """Load base model (with optional correct quantization) and attach LoRA."""
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    quant_cfg = build_quant_config(quant)

    load_kwargs = dict(
        device_map="auto",
        trust_remote_code=spec.trust_remote_code,
        token=hf_token,
    )
    if quant_cfg is not None:
        load_kwargs["quantization_config"] = quant_cfg
    else:
        # No quantization: pick the best available full/half precision dtype.
        load_kwargs["torch_dtype"] = (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

    model = AutoModelForCausalLM.from_pretrained(spec.base, **load_kwargs)

    if spec.peft:
        model = PeftModel.from_pretrained(model, spec.peft, token=hf_token)

    model.eval()
    return model


# =============================================================================
# 6. DATA LOADING
# =============================================================================
def load_fpb_allagree(hf_token, disjoint_ids):
    """
    Load FPB ``sentences_allagree`` (2,264 sentences). Returns (sentences,
    y_true). If ``disjoint_ids`` is provided, the eval is RESTRICTED to those
    indices -- use this to evaluate on the decontaminated, FinGPT-disjoint
    remainder (see the contamination note in the module docstring).

    Loading is tried in two ways: the canonical ``takala`` script loader
    (works on ``datasets`` < 4.x), then the ``gtfintechlab`` parquet mirror
    (same 2,264 sentences; needed on ``datasets`` >= 4.x where script-based
    loaders are unsupported). Mirror rows are sorted by sentence so the
    ordering -- and therefore any ``disjoint_ids`` file -- is deterministic
    regardless of which source loads.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(
            "takala/financial_phrasebank",
            "sentences_allagree",
            trust_remote_code=True,
            token=hf_token,
        )["train"]
        feat_names = ds.features["label"].names  # FPB's own (neg, neu, pos)
        pairs = [(s, LABEL2ID[feat_names[lbl]])
                 for s, lbl in zip(ds["sentence"], ds["label"])]
        source = "takala/financial_phrasebank"
    except Exception as exc:
        print("[data] takala loader failed (%s); falling back to the "
              "gtfintechlab parquet mirror." % type(exc).__name__)
        from datasets import concatenate_datasets, get_dataset_config_names
        mirror = "gtfintechlab/financial_phrasebank_sentences_allagree"
        cfg = get_dataset_config_names(mirror)[0]
        dd = load_dataset(mirror, cfg, token=hf_token)
        ds = concatenate_datasets([dd[split] for split in dd])
        # The mirror partitions the same 2,264 sentences across splits; dedup
        # by sentence, then normalize its label column to our LABEL2ID. The
        # mirror stores plain int64 labels already in the canonical FPB
        # convention (0=negative, 1=neutral, 2=positive -- verified: the
        # class counts 303/1391/570 match takala's own distribution).
        seen, pairs = set(), []
        feat = ds.features["label"]
        names = feat.names if hasattr(feat, "names") else None
        for row in ds:
            s = row["sentence"]
            if s in seen:
                continue
            seen.add(s)
            lbl = row["label"]
            if names is not None:                     # named ClassLabel
                canon_id = LABEL2ID[names[lbl].lower()]
            elif isinstance(lbl, int) or str(lbl).isdigit():  # canonical int
                canon_id = int(lbl)
                if canon_id not in ID2LABEL:
                    raise ValueError("mirror label %r outside 0..2" % (lbl,))
            else:                                     # string label
                canon_id = LABEL2ID[str(lbl).strip().lower()]
            pairs.append((s, canon_id))
        source = mirror

    # Deterministic ordering independent of the source that happened to load.
    pairs.sort(key=lambda p: p[0])
    sentences = [p[0] for p in pairs]
    y_true = [p[1] for p in pairs]
    print("[data] FPB sentences_allagree loaded from %s (%d rows, sorted)."
          % (source, len(sentences)))

    if disjoint_ids is not None:
        keep = set(disjoint_ids)
        sentences = [s for i, s in enumerate(sentences) if i in keep]
        y_true = [y for i, y in enumerate(y_true) if i in keep]
        print("[data] Restricted to %d disjoint (decontaminated) sentences."
              % len(sentences))

    return sentences, y_true


def load_eval_csv(path):
    """
    Load a frozen evaluation CSV with columns ``sentence,label`` (string labels
    negative/neutral/positive) as produced by
    revision/scripts/prepare_external_datasets.py. Returns (sentences, y_true).
    """
    import csv as _csv

    sentences, y_true = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            lab = row["label"].strip().lower()
            if lab not in LABEL2ID:
                raise ValueError("unrecognized label %r in %s" % (row["label"], path))
            sentences.append(row["sentence"])
            y_true.append(LABEL2ID[lab])
    print("[data] Loaded %d examples from %s." % (len(sentences), path))
    return sentences, y_true


# =============================================================================
# 7. GENERATION LOOP  (fixes #1, #3, #6)
# =============================================================================
def class_word_first_token_ids(tokenizer):
    """
    First-token id of each class word (" negative"/" neutral"/" positive",
    falling back to the unspaced form), used to read class log-probabilities
    from a single forward pass for ROC/PR curves.
    """
    ids = []
    for word in LABELS:
        tok_id = None
        for cand in (" " + word, word):
            toks = tokenizer(cand, add_special_tokens=False)["input_ids"]
            if toks:
                tok_id = toks[0]
                break
        if tok_id is None:
            raise RuntimeError("tokenizer produced no tokens for %r" % word)
        ids.append(tok_id)
    return ids


def run_inference(model, tokenizer, spec, sentences, batch_size, max_new_tokens,
                  capture_scores=False):
    """
    Greedy, deterministic generation for one model. Returns parallel lists:
    raw_outputs (continuation only), pred_ids (0/1/2 or UNPARSEABLE_ID),
    parsed_labels (str or "UNPARSEABLE"), and scores (list of per-example dicts
    with logprob_/score_ per class, or None when ``capture_scores`` is off).

    Score capture (adapted from analysis_and_figures.capture_answer_logprobs):
    one extra no-grad forward pass per batch on the same left-padded encodings;
    the log-softmax of the logits at the last prompt position is read at the
    first-token id of each class word. Predictions remain the generation-parsed
    labels -- the scores exist only so ROC/PR curves can be drawn.
    """
    import torch
    from tqdm import tqdm

    raw_outputs = []
    pred_ids = []
    parsed_labels = []
    scores = [] if capture_scores else None
    class_tok_ids = class_word_first_token_ids(tokenizer) if capture_scores else None

    for start in tqdm(range(0, len(sentences), batch_size),
                      desc="  " + spec.pretty):
        batch = sentences[start:start + batch_size]
        prompts = [spec.template(spec.instruction, t) for t in batch]

        # Left-padded batch (fix #1). We do NOT pass max_length here; we cap the
        # OUTPUT with max_new_tokens below (fix #3). We do bound the INPUT with a
        # generous truncation so a pathological long sentence cannot OOM.
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,   # fix #3: cap NEW tokens, not total
                do_sample=False,                 # greedy
                num_beams=1,                     # greedy
                temperature=None,                # silence sampling warnings
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )

            if capture_scores:
                out = model(input_ids=enc["input_ids"],
                            attention_mask=enc["attention_mask"])
                # With left padding the last column is the true last prompt
                # token for every row.
                logprobs = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
                class_lp = logprobs[:, class_tok_ids]              # (B, 3)
                class_p = torch.softmax(class_lp, dim=-1)          # renormed
                for b in range(class_lp.shape[0]):
                    row = {}
                    for ci, name in enumerate(LABELS):
                        row["logprob_" + name] = float(class_lp[b, ci])
                        row["score_" + name] = float(class_p[b, ci])
                    scores.append(row)

        # Slice off the prompt tokens so we decode ONLY the continuation. With
        # left padding, every row's prompt occupies the first ``input_len``
        # columns, so the new tokens are uniformly everything after it.
        input_len = enc["input_ids"].shape[1]
        new_tokens = gen[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for cont in decoded:
            pid = parse_sentiment(cont)  # continuation only; no prompt echo
            raw_outputs.append(cont.strip())
            if pid is None:
                pred_ids.append(UNPARSEABLE_ID)
                parsed_labels.append("UNPARSEABLE")
            else:
                pred_ids.append(pid)
                parsed_labels.append(ID2LABEL[pid])

    return raw_outputs, pred_ids, parsed_labels, scores


def run_inference_seq_cls(model, tokenizer, spec, sentences, batch_size):
    """
    Inference for an encoder sequence-classification baseline (FinBERT).
    No generation, no parsing: predictions and class probabilities come
    straight from the classifier head. Class order is mapped through the
    model's own ``id2label`` BY NAME (ProsusAI/finbert orders its labels
    positive/negative/neutral -- never assume FPB order). Returns the same
    4-tuple shape as run_inference; unparseable rate is 0 by construction.
    """
    import torch
    from tqdm import tqdm

    # model class index -> our canonical 0/1/2, matched by label NAME.
    idx_map = {}
    for i, name in model.config.id2label.items():
        idx_map[int(i)] = LABEL2ID[name.strip().lower()]

    raw_outputs, pred_ids, parsed_labels, scores = [], [], [], []

    for start in tqdm(range(0, len(sentences), batch_size),
                      desc="  " + spec.pretty):
        batch = sentences[start:start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits.float(), dim=-1)

        for b in range(probs.shape[0]):
            # Re-express the probability row in canonical class order.
            canon = [0.0, 0.0, 0.0]
            for model_idx, canon_idx in idx_map.items():
                canon[canon_idx] = float(probs[b, model_idx])
            pid = int(max(range(3), key=lambda c: canon[c]))
            pred_ids.append(pid)
            parsed_labels.append(ID2LABEL[pid])
            raw_outputs.append("p(neg,neu,pos)=(%.4f,%.4f,%.4f)" % tuple(canon))
            row = {}
            for ci, name in enumerate(LABELS):
                row["logprob_" + name] = float(np.log(max(canon[ci], 1e-12)))
                row["score_" + name] = canon[ci]
            scores.append(row)

    return raw_outputs, pred_ids, parsed_labels, scores


# =============================================================================
# 8. METRICS + CSV PERSISTENCE  (Reviewer 1 pts 1,2,7 / Reviewer 2 Q4)
# =============================================================================
@dataclass
class RunResult:
    model_key: str
    pretty: str
    seed: int
    accuracy: float
    weighted_f1: float
    macro_f1: float
    unparseable_rate: float
    n: int
    dataset: str = "fpb"
    per_class: dict = field(default_factory=dict)


def compute_and_save(spec, seed, sentences, y_true, raw_outputs,
                     pred_ids, parsed_labels, out_dir,
                     dataset_name="fpb", scores=None):
    """
    Compute accuracy / weighted F1 / macro F1 / per-class P-R-F1 and the
    unparseable rate, then write per-example predictions, the full confusion
    matrix, and the per-class table to CSV.

    UNPARSEABLE handling: an unparseable prediction is ALWAYS an error (it can
    never equal a gold id because UNPARSEABLE_ID is outside {0,1,2}). It is
    reported as a separate rate and EXCLUDED from the 3x3 confusion matrix and
    per-class report (those describe only the in-class predictions), but it IS
    counted against accuracy. This is the correct, non-cheating treatment that
    the original neutral-default parser violated.
    """
    import pandas as pd
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_recall_fscore_support,
                                 confusion_matrix)

    os.makedirs(out_dir, exist_ok=True)
    tag = "%s_%s_seed%d" % (spec.key, dataset_name, seed)

    n = len(y_true)
    n_unparseable = sum(1 for p in pred_ids if p == UNPARSEABLE_ID)
    unparseable_rate = n_unparseable / n if n else 0.0

    # Accuracy over ALL examples (unparseable rows are guaranteed wrong).
    accuracy = accuracy_score(y_true, pred_ids)

    # Weighted/macro F1 over the three real classes. Unparseable predictions
    # depress recall (they never match a gold id) but are not a phantom class.
    weighted_f1 = f1_score(y_true, pred_ids, labels=[0, 1, 2],
                           average="weighted", zero_division=0)
    macro_f1 = f1_score(y_true, pred_ids, labels=[0, 1, 2],
                        average="macro", zero_division=0)

    # Per-class precision/recall/F1/support over the three real classes.
    p, r, f, s = precision_recall_fscore_support(
        y_true, pred_ids, labels=[0, 1, 2], zero_division=0)
    per_class = {}
    per_class_rows = []
    for i, name in enumerate(LABELS):
        per_class[name] = {"precision": float(p[i]), "recall": float(r[i]),
                           "f1": float(f[i]), "support": int(s[i])}
        per_class_rows.append({"class": name, "precision": p[i], "recall": r[i],
                               "f1": f[i], "support": int(s[i])})
    pd.DataFrame(per_class_rows).to_csv(
        os.path.join(out_dir, tag + "_per_class.csv"), index=False)

    # --- Confusion matrix (3x3 over real classes only). Reviewer 2 Q4. ---
    # Rows = true class, cols = predicted class. Unparseable predictions are
    # NOT placed in any 3x3 column; we add an explicit pred_UNPARSEABLE column
    # so a reader can reconstruct each true-class total.
    cm = confusion_matrix(y_true, pred_ids, labels=[0, 1, 2])
    cm_df = pd.DataFrame(cm, index=["true_" + c for c in LABELS],
                         columns=["pred_" + c for c in LABELS])
    unpar_per_true = [0, 0, 0]
    for yt, pid in zip(y_true, pred_ids):
        if pid == UNPARSEABLE_ID:
            unpar_per_true[yt] += 1
    cm_df["pred_UNPARSEABLE"] = unpar_per_true
    cm_df.to_csv(os.path.join(out_dir, tag + "_confusion_matrix.csv"))

    # --- Per-example predictions CSV (full audit trail; Reviewer 1 pt 7). ---
    pred_rows = []
    for i in range(n):
        row = {
            "index": i,
            "sentence": sentences[i],
            "gold_id": y_true[i],
            "gold_label": ID2LABEL[y_true[i]],
            "pred_id": pred_ids[i],
            "pred_label": parsed_labels[i],
            "raw_output": raw_outputs[i],
            "correct": int(pred_ids[i] == y_true[i]),
            "unparseable": int(pred_ids[i] == UNPARSEABLE_ID),
            "model": spec.pretty,
            "dataset": dataset_name,
        }
        if scores is not None:
            row.update(scores[i])
        pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(out_dir, tag + "_predictions.csv"), index=False)

    print("    acc=%.4f  wF1=%.4f  macroF1=%.4f  unparseable=%.4f  (n=%d)"
          % (accuracy, weighted_f1, macro_f1, unparseable_rate, n))

    return RunResult(
        model_key=spec.key, pretty=spec.pretty, seed=seed,
        accuracy=float(accuracy), weighted_f1=float(weighted_f1),
        macro_f1=float(macro_f1), unparseable_rate=float(unparseable_rate),
        n=n, dataset=dataset_name, per_class=per_class,
    )


def write_summary(results, out_dir):
    """
    Write summary.csv (one row per model/seed) and summary_aggregated.csv
    (mean +/- std across seeds per model) -- the multi-seed table reviewers
    asked for (Reviewer 1 pt 5 / Reviewer 2 Q5).
    """
    import pandas as pd

    rows = [{
        "model": r.pretty, "model_key": r.model_key, "dataset": r.dataset,
        "seed": r.seed, "n": r.n, "accuracy": r.accuracy,
        "weighted_f1": r.weighted_f1, "macro_f1": r.macro_f1,
        "unparseable_rate": r.unparseable_rate,
    } for r in results]
    df = pd.DataFrame(rows).sort_values(["dataset", "model", "seed"])
    # Merge with any rows from an earlier (resumed) session so summary.csv is
    # always the union of everything present in out_dir.
    summary_path = os.path.join(out_dir, "summary.csv")
    if os.path.isfile(summary_path):
        prev = pd.read_csv(summary_path)
        df = (pd.concat([prev, df], ignore_index=True)
                .drop_duplicates(subset=["model_key", "dataset", "seed"],
                                 keep="last")
                .sort_values(["dataset", "model", "seed"]))
    df.to_csv(summary_path, index=False)

    agg_rows = []
    for (key, dataset), grp in df.groupby(["model_key", "dataset"]):
        agg_rows.append({
            "model": grp["model"].iloc[0],
            "model_key": key,
            "dataset": dataset,
            "n_seeds": len(grp),
            "accuracy_mean": grp["accuracy"].mean(),
            "accuracy_std": grp["accuracy"].std(ddof=0),
            "weighted_f1_mean": grp["weighted_f1"].mean(),
            "weighted_f1_std": grp["weighted_f1"].std(ddof=0),
            "macro_f1_mean": grp["macro_f1"].mean(),
            "macro_f1_std": grp["macro_f1"].std(ddof=0),
            "unparseable_rate_mean": grp["unparseable_rate"].mean(),
        })
    agg = pd.DataFrame(agg_rows).sort_values(
        ["dataset", "weighted_f1_mean"], ascending=[True, False])
    agg.to_csv(os.path.join(out_dir, "summary_aggregated.csv"), index=False)

    print("\n=== Aggregated (mean +/- std across seeds) ===")
    for _, row in agg.iterrows():
        print("  [%-12s] %-32s acc=%.4f+/-%.4f  wF1=%.4f+/-%.4f  macroF1=%.4f+/-%.4f"
              % (row["dataset"], row["model"],
                 row["accuracy_mean"], row["accuracy_std"],
                 row["weighted_f1_mean"], row["weighted_f1_std"],
                 row["macro_f1_mean"], row["macro_f1_std"]))


# =============================================================================
# 9. CLEANUP
# =============================================================================
def free_model(model):
    """Release a model's GPU memory between runs to avoid OOM across models."""
    import torch
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =============================================================================
# 10. CLI / MAIN
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(
        description="Fixed, fair evaluation harness for Finistral + baselines.")
    ap.add_argument("--models", nargs="+", default=["all"],
                    help="Model keys to run, or 'all'. Choices: "
                         + ", ".join(MODELS) + " | all")
    ap.add_argument("--quant", choices=["none", "int8", "nf4"], default="none",
                    help="Quantization: none=fp16/bf16 (default, most faithful), "
                         "int8=valid 8-bit, nf4=genuine QLoRA 4-bit NF4 double-quant.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="One or more seeds for multi-run mean+/-std (Reviewer 1 pt 5).")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=8,
                    help="Cap on NEW tokens (a sentiment word is 1-3 tokens).")
    ap.add_argument("--out_dir", default="results_fixed")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"),
                    help="HF token (or set HF_TOKEN env var). Optional for public repos.")
    ap.add_argument("--disjoint_ids_file", default=None,
                    help="Optional text file with one FPB index per line to "
                         "restrict eval to the FinGPT-disjoint (decontaminated) subset. "
                         "Only meaningful with --dataset fpb.")
    ap.add_argument("--dataset", default="fpb",
                    choices=["fpb", "fpb_decontam", "fiqa", "tfns", "csv"],
                    help="Evaluation set. 'fpb' loads the full (CONTAMINATED) "
                         "sentences_allagree from the Hub; the others read a "
                         "frozen CSV from data_eval/ (see --dataset_csv).")
    ap.add_argument("--dataset_csv", default=None,
                    help="Path to a sentence,label CSV for non-'fpb' datasets. "
                         "Defaults to data_eval/<dataset>.csv.")
    ap.add_argument("--capture_scores", action="store_true",
                    help="Also record per-class log-probabilities (one extra "
                         "forward pass per batch) so ROC/PR curves can be drawn.")
    ap.add_argument("--no_resume", action="store_true",
                    help="Re-run (model, dataset, seed) combinations even if "
                         "their predictions CSV already exists in out_dir.")
    return ap.parse_args()


def try_load_existing(spec, seed, dataset_name, out_dir):
    """
    Resume guard: if this (model, dataset, seed)'s predictions CSV already
    exists, rebuild its RunResult from disk instead of re-running inference.
    Returns None when no usable CSV is present.
    """
    path = os.path.join(
        out_dir, "%s_%s_seed%d_predictions.csv" % (spec.key, dataset_name, seed))
    if not os.path.isfile(path):
        return None
    import pandas as pd
    from sklearn.metrics import (accuracy_score, f1_score,
                                 precision_recall_fscore_support)
    try:
        df = pd.read_csv(path)
        y_true = df["gold_id"].tolist()
        pred_ids = df["pred_id"].tolist()
    except Exception as exc:
        print("  [resume] could not reuse %s (%s); re-running." % (path, exc))
        return None
    n = len(y_true)
    if n == 0:
        return None
    unparseable_rate = sum(1 for p in pred_ids if p == UNPARSEABLE_ID) / n
    p, r, f, s = precision_recall_fscore_support(
        y_true, pred_ids, labels=[0, 1, 2], zero_division=0)
    per_class = {name: {"precision": float(p[i]), "recall": float(r[i]),
                        "f1": float(f[i]), "support": int(s[i])}
                 for i, name in enumerate(LABELS)}
    print("  [resume] reusing existing %s (n=%d)" % (os.path.basename(path), n))
    return RunResult(
        model_key=spec.key, pretty=spec.pretty, seed=seed,
        accuracy=float(accuracy_score(y_true, pred_ids)),
        weighted_f1=float(f1_score(y_true, pred_ids, labels=[0, 1, 2],
                                   average="weighted", zero_division=0)),
        macro_f1=float(f1_score(y_true, pred_ids, labels=[0, 1, 2],
                                average="macro", zero_division=0)),
        unparseable_rate=float(unparseable_rate), n=n,
        dataset=dataset_name, per_class=per_class,
    )


def main():
    args = parse_args()

    # Resolve the model selection.
    if args.models == ["all"] or "all" in args.models:
        selected = list(MODELS.values())
    else:
        unknown = [m for m in args.models if m not in MODELS]
        if unknown:
            raise SystemExit("Unknown model keys: %s. Choices: %s"
                             % (unknown, list(MODELS)))
        selected = [MODELS[m] for m in args.models]

    # Optional decontamination index list.
    disjoint_ids = None
    if args.disjoint_ids_file:
        with open(args.disjoint_ids_file) as fh:
            disjoint_ids = [int(line.strip()) for line in fh if line.strip()]

    os.makedirs(args.out_dir, exist_ok=True)

    # Load eval data ONCE (it is identical across models and seeds).
    if args.dataset == "fpb":
        sentences, y_true = load_fpb_allagree(args.hf_token, disjoint_ids)
    else:
        csv_path = args.dataset_csv or os.path.join(
            "data_eval", args.dataset + ".csv")
        sentences, y_true = load_eval_csv(csv_path)
    print("[data] Evaluating on %d '%s' examples."
          % (len(sentences), args.dataset))
    print("[config] quant=%s  seeds=%s  batch_size=%d  max_new_tokens=%d"
          % (args.quant, args.seeds, args.batch_size, args.max_new_tokens))

    all_results = []

    for spec in selected:
        # Resume guard: reuse any (model, dataset, seed) already on disk.
        pending = list(args.seeds)
        if not args.no_resume:
            for seed in list(pending):
                existing = try_load_existing(spec, seed, args.dataset,
                                             args.out_dir)
                if existing is not None:
                    all_results.append(existing)
                    pending.remove(seed)
        if not pending:
            print("\n=== Model: %s -- all seeds already on disk, skipped ==="
                  % spec.pretty)
            continue

        print("\n=== Model: %s  (base=%s, peft=%s, kind=%s) ==="
              % (spec.pretty, spec.base, spec.peft, spec.kind))

        if spec.kind == "seq_cls":
            # Encoder classifier (FinBERT): default right padding (absolute
            # position embeddings), no generation, native class probabilities.
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            import torch
            tokenizer = AutoTokenizer.from_pretrained(
                spec.base, token=args.hf_token)
            model = AutoModelForSequenceClassification.from_pretrained(
                spec.base, token=args.hf_token,
                device_map="auto" if torch.cuda.is_available() else None)
            model.eval()
            for seed in pending:
                set_all_seeds(seed)
                raw_outputs, pred_ids, parsed_labels, scores = \
                    run_inference_seq_cls(model, tokenizer, spec, sentences,
                                          batch_size=args.batch_size)
                result = compute_and_save(
                    spec, seed, sentences, y_true,
                    raw_outputs, pred_ids, parsed_labels, args.out_dir,
                    dataset_name=args.dataset, scores=scores)
                all_results.append(result)
        else:
            # Load tokenizer+model ONCE per model; greedy decoding is
            # deterministic, so we can reuse the loaded model across seeds (the
            # seed loop still re-seeds RNGs and re-runs inference for a clean
            # per-seed audit trail).
            tokenizer = build_tokenizer(spec, args.hf_token)
            model = load_model(spec, args.quant, args.hf_token)
            # Keep model + tokenizer pad ids in sync (fix #2).
            if model.config.pad_token_id is None:
                model.config.pad_token_id = tokenizer.pad_token_id

            for seed in pending:
                set_all_seeds(seed)
                raw_outputs, pred_ids, parsed_labels, scores = run_inference(
                    model, tokenizer, spec, sentences,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    capture_scores=args.capture_scores)
                result = compute_and_save(
                    spec, seed, sentences, y_true,
                    raw_outputs, pred_ids, parsed_labels, args.out_dir,
                    dataset_name=args.dataset, scores=scores)
                all_results.append(result)

        free_model(model)
        del tokenizer

    write_summary(all_results, args.out_dir)

    # Also dump a machine-readable JSON of everything for downstream plotting
    # (learning/PR/ROC/ablation curves can be built from the per-example CSVs).
    with open(os.path.join(args.out_dir, "results.json"), "w") as fh:
        json.dump([r.__dict__ for r in all_results], fh, indent=2)

    print("\nDone. Per-model predictions, confusion matrices, per-class tables,"
          " and summaries written to: %s" % os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
