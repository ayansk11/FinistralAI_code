# Session Log — Finistral-7B-LoRA Revision (JICTASA-2026-018)

**Date:** 23 Jun 2026
**Project:** `Finistral AI: Financial Sentiment Analyst` — GR Journals / Kryoni JMS, Major Revision (Round 2)
**Manuscript ID:** JICTASA-2026-018
**Authors:** Zainab Mirza¹, Ayan Javeed Shaikh²·*, Fazal Jalil Parkar³
**Repo:** `/Users/ayansk11/Desktop/FinistralAI_code`

This document records everything that happened in this working session: the reviewer
comments that triggered it, the investigation, the keystone finding, every artifact
produced, the manuscript changes, an incident where the project folder was moved
mid-session, the journal-portal guidance, and what remains to be done.

---

## 1. Starting point & goal

The paper reports **99.56% accuracy / 0.996 weighted-F1** for a LoRA-adapted
Mistral-7B on Financial PhraseBank, and claims it beats five FinGPT baselines (which
score a bizarre **13–23%**) plus FinBERT. Two reviewers returned a major revision:

- **Reviewer 1** — 13 points. Headline concerns: (1) validate the extreme accuracy,
  (2) implausibly low baselines, (3) **possible data leakage**, (4) limited novelty,
  (5) no statistical validation, (6) missing ablation, (7) weak error analysis,
  (8) single-benchmark evaluation, (9) grammar, (10) replace GUI screenshots with real
  curves, (11) deeper discussion, (12) 2024–2026 literature, (13) release scripts.
- **Reviewer 2** — 5 questions, all circling the same suspicion: how was FinGPT-train /
  Financial PhraseBank **overlap** prevented, why are the baselines so low, were they
  run with correct inference procedures, can we see **confusion matrices** for all
  baselines, and can results be **replicated across seeds / on external datasets**.

Both reviewers independently zeroed in on the same root worry: the numbers are too
good and the baselines too bad to be real. **They were right.**

**Goal of the session:** investigate, prove or disprove the leakage, and produce an
honest, defensible revision — code fixes, corrected manuscript, response letter, and a
plan.

---

## 2. The keystone finding — measured data contamination

The single most important result of the session, **measured locally and reproducibly**
(`revision/scripts/measure_leakage_local.py`, CPU-only):

> **1,699 of 2,259 unique Financial PhraseBank `sentences_allagree` evaluation
> sentences (75.21%) appear *verbatim* inside the `FinGPT/fingpt-sentiment-train`
> corpus the adapter was fine-tuned on — and 100% of those carry the identical gold
> label.** The remaining **560 sentences (24.79%)** are genuinely unseen.

Datasets used:
- `FinGPT/fingpt-sentiment-train` — 76,772 rows (30,209 unique normalized inputs).
- Financial PhraseBank `sentences_allagree` — 2,264 lines, 2,259 unique
  (class distribution: neutral 1,386 · positive 570 · negative 303).

Why this is decisive: `FinGPT/fingpt-sentiment-train` **aggregates Financial
PhraseBank as one of its constituent sources**. The training script does only a
`train_test_split(test_size=0.05, seed=42)`, so ~1,614 of the leaked sentences land
directly in the 95% fine-tuning partition — then the paper evaluates on the *full*
2,264. This is textbook train-on-test. The measured class distribution (303/1,386/570)
matches the paper's own confusion-matrix diagonal, confirming it is the same eval set.

**Conclusion:** the 99.56% is a memorization artifact, not generalization. It must be
reported only as a *contamination diagnostic*, with honest numbers coming from the
560-sentence decontaminated remainder + external datasets.

Decontaminated test set saved: `revision/fpb_decontaminated.csv` (560 sentences).
Sample leaked pairs for the appendix: `revision/fpb_overlap_matches.csv`.

---

## 3. The baselines are a *separate*, independent bug

The 13–23% baseline scores (below the 61.6% neutral majority-class floor — a level no
working model produces) are **not** evidence the models are weak. They are artifacts of
a broken evaluation harness:

- **Right-padding on decoder-only models** (566 `padding_side=left` warnings in the
  committed notebook output) — generation conditioned on trailing pad tokens.
- **`generate(..., max_length=512)` with `padding=True`** — `max_length` silently
  ignored; models ramble (the base row took 5h31m).
- **A single Alpaca prompt** applied to every model, mismatched to each FinGPT
  adapter's native template *and* to Finistral's own `[INST]` training template.
- **A silently-dropped quantization config** — invalid `BitsAndBytesConfig` kwargs.
- **A neutral-defaulting parser** that scores corrupted generations as `neutral`.

So the paper simultaneously **inflated** its own model (contamination) and
**deflated** the baselines (harness bugs). Both must be fixed; neither alone explains
the paper. The "+78 F1 over baselines" claim is withdrawn.

---

## 4. Multi-agent verification workflow

A background **Workflow** (dynamic, multi-agent) had been dispatched to adversarially
verify the leakage + broken-baseline diagnoses, refresh literature, and draft corrected
artifacts. It completed during the session:

- **10 agents**, ~942k subagent tokens, 142 tool uses, ~52 min wall-clock.
- **Verify phase (4 findings):** the contamination claim came back **CONFIRMED (high
  confidence)** — one verifier went beyond the brief and *actually downloaded both
  datasets*, independently measuring 75.22% overlap (1,703/2,264) with 100% label
  agreement. The counter-claim ("no leakage, 99.56% is legitimate") was **REFUTED**.
- **Build phase (5 artifacts):** 4 corrected scripts + a literature set of **23 verified
  2024–2026 citations** (financial LLMs, PEFT variants, and benchmark-contamination
  papers).
- **Letter:** a full point-by-point response-to-reviewers draft.

I then **independently re-measured** the overlap locally (§2) and got **1,699/2,259 =
75.21%** — confirming the workflow's number, and I standardized all documents on the
locally-reproduced figures (2,259 unique of 2,264 raw; 560 clean).

---

## 5. Code-vs-paper audit — 14 discrepancies

Reading the actual training script (`Finistral_Sentiment_analyst.py`) against the
manuscript surfaced 14 mismatches (full list = "Corrected Claims Table" C1–C14 in
`revision/RESPONSE_TO_REVIEWERS.md`). The most material:

| Paper claims | Code actually does |
|---|---|
| "We utilise **Unsloth** (2× faster, 70% less VRAM)" | plain `transformers`+`peft`+`Trainer`; `gradient_checkpointing` + `enable_xformers_memory_efficient_attention()` — **no Unsloth anywhere** |
| "load backbone in **8-bit NF4** (`load_in_8bit=True`)" | `torch_dtype=bfloat16`, **no quantization** |
| LoRA on "**every** attention + feed-forward layer" | `target_modules=["q_proj","v_proj"]` **only** |
| LoRA **dropout 0.10** | `lora_dropout=0.05` |
| **≈9M** trainable (0.2%) | q/v-only, r=16 → **6.82M (0.094%)** |
| prompt = Alpaca "Instruction/Input/Answer" | training prompt = `[INST]{input}\n{instruction} [/INST]`; the eval notebooks used Alpaca → **train/eval template mismatch** |
| GGML 4-bit adapter = **168 MB** | a real 4-bit variant is *smaller* than the 83.9 MB fp16 — number is wrong |
| val loss **0.1008** (text) vs **0.1009** (table) | inconsistent; standardized to 0.1009 |
| "**No** AI-assisted technology used" | inconsistent with tooling actually used — corrected to a truthful disclosure |

---

## 6. Manuscript rewrite (honest reframing)

`finistral_grjournals.tex` was rewritten for the parts that need **no new experiments**,
and now **compiles clean (13 pp)**; it grew from 630 → 687 lines. Changes:

- **Abstract** — discloses the 75.2% overlap, relabels 99.56% as *in-distribution*,
  promises decontaminated + external results as primary evidence; contribution reframed
  as **(i) an efficient reproducible LoRA recipe + (ii) a contamination/evaluation case
  study**. Fixed 0.2% → 0.094% (6.82M).
- **Novelty statement** — dropped "first / state-of-the-art"; now the efficient recipe +
  the 75.2% contamination finding.
- **Research Questions** — RQ1 reframed around *leakage-free* comparison; added **RQ3**
  on how much apparent performance is overlap vs. generalization.
- **Contributions** — replaced "state-of-the-art performance" with the contamination &
  evaluation case study; fixed param counts and adapter sizes.
- **Related Work** — added benchmark-contamination literature and framed FPB-in-FinGPT
  overlap as a first-class measurement.
- **Preprocessing** — corrected prompt template to the actual `[INST]` format and
  disclosed the train/eval mismatch; parser no longer defaults to "neutral".
- **Base Model** — removed the false Unsloth + "8-bit NF4" claims; describes the real
  `transformers`+`peft`+`Trainer` / bf16 / gradient-checkpointing / xFormers stack.
- **LoRA Adaptation** — corrected to `q_proj`/`v_proj` only.
- **New `\section`/`\label{sec:contamination}`** — the Data Contamination Analysis
  section (referenced from 10 places) reporting the overlap statistics + decontamination
  procedure.
- **Placeholder macro** `\rerun{...}` (renders red) marks the **12 numbers still needing
  the GPU re-runs** — the only figures not yet honest.
- Fixed a pre-existing `\captionsetup` LaTeX bug so the document builds.

---

## 7. Incident — project folder moved mid-session

While editing, Bash reported *"Working directory … was deleted"*. Investigation showed
the folder had been **moved** (by Finder/user) from `Desktop/FinistralAI_code` into
`Desktop/Work/FinistralAI_code`, taking the `revision/` artifacts with it intact. Two
older copies exist elsewhere (`Desktop/Things That Might Help You…/` and `Downloads/`)
but neither has the GR Journals manuscript, so they were left untouched. The folder was
subsequently moved **back** to the original `Desktop/FinistralAI_code` path, which is
now the single canonical copy. Two abstract edits attempted during the move errored and
did **not** apply (re-applied cleanly afterward). No work was lost.

---

## 8. Journal-portal guidance (Kryoni JMS)

- **"Should I accept the revision?"** → Yes. For an author, *Accept Revision* means
  accepting the revise-and-resubmit assignment (it unlocks the file-upload stage); it is
  **not** the journal accepting the paper. "No" on the confirmation just cancels the
  dialog. Recommended: accept, then immediately post a status/extension note, then upload
  once the re-runs are done. (The action is the user's to take in their own account.)
- **Extension / status message** drafted for the *Revision Submission* discussion thread
  (Subject + body), naming the contamination re-analysis and requesting a short
  extension to **7 July 2026** (with an optional interim 30 Jun partial to respect the
  June-issue timeline). The submission is already past the 12 Jun due date.
- **Cover letter** drafted → `revision/COVER_LETTER.md` (1-page summary of changes for
  the formal `..._Cover_Letter.docx` upload).

Three distinct documents, not to be confused:
1. Extension message → discussion thread (communication only).
2. Cover Letter (`COVER_LETTER.md`) → formal upload, File Type *Cover Letter*.
3. Reviewer Response File (`RESPONSE_TO_REVIEWERS.md`) → formal upload, the point-by-point
   replies the editor explicitly requested.

---

## 9. Deliverables produced (all under `revision/`)

| File | What it is | Status |
|---|---|---|
| `RESPONSE_TO_REVIEWERS.md` | Point-by-point reply to all 13 R1 + 5 R2 comments + the C1–C14 Corrected Claims Table | ✅ drafted |
| `COVER_LETTER.md` | 1-page resubmission cover letter | ✅ drafted |
| `REVISION_PLAN.md` | Sequenced plan: reviewer point → action → script → hardware → acceptance criterion (Phases 0–8) | ✅ |
| `README.md` | Index of the revision package | ✅ |
| `leakage_report.md` | Human-readable contamination measurement | ✅ |
| `fpb_decontaminated.csv` | The 560 leakage-free FPB sentences (the honest test set) | ✅ |
| `fpb_overlap_matches.csv` | Sample leaked (sentence, label) pairs | ✅ |
| `literature.json` | 23 verified 2024–2026 references | ✅ |
| `_verify_findings.json` | Raw multi-agent verification record (audit trail) | ✅ |
| `scripts/measure_leakage_local.py` | Reproduces the 75.2% overlap + builds decontaminated set (**CPU, verified working**) | ✅ |
| `scripts/leakage_analysis.py` | Fuller overlap analysis (exact + normalized + MinHash/LSH) | ✅ (CPU) |
| `scripts/eval_harness_fixed.py` | Corrected eval: left-padding, `max_new_tokens`, per-model prompts, strict parser, consistent backbone | ⏳ needs GPU |
| `scripts/ablation_and_seeds.py` | 5-seed runs (mean±std + significance) + ablation grid | ⏳ needs GPU |
| `scripts/analysis_and_figures.py` | Learning/PR/ROC/ablation curves + per-baseline confusion matrices | ⏳ needs GPU |
| `../finistral_grjournals.tex` | Honestly rewritten manuscript, compiles clean (13 pp), 12 `\rerun{}` markers left | ✅ partial |

---

## 10. What remains (all GPU-gated) — from `REVISION_PLAN.md`

- **Phase 1 (keystone):** run `eval_harness_fixed.py` on `fpb_decontaminated.csv` (with
  the `[INST]` template) **and** on external leakage-free sets (FiQA-SA, Twitter Financial
  News Sentiment, SemEval-2017 Task 5) → the first honest accuracy numbers.
- **Phase 2:** re-run all baselines (incl. FinBERT) under native inference; expect FinGPT
  baselines to recover to ≈0.84–0.86; emit per-baseline confusion matrices (R2.4).
- **Phase 3:** 5-seed mean ± std + McNemar / bootstrap significance tests (R1.5, R2.5).
- **Phase 4:** ablation grid — rank, α, dropout, target modules, `[INST]` vs Alpaca,
  epochs (R1.6); ablation **curves** replace the GUI screenshots (R1.10).
- **Phase 5–6:** error-analysis table + real figures (learning/PR/ROC/confusion).
- **Phase 8:** fill the 12 `\rerun{}` markers, recompile, sync the `.docx`, bundle scripts
  into the public repo, and submit via Kryoni JMS before the deadline.

**Compute estimate:** 1× A100/40GB (or L4/A10). Eval ≈ minutes/dataset; 5-seed retrain
≈ 5 × ~2h; full ablation ≈ 0.5–1 GPU-day. Stack pins: PyTorch 2.2, Transformers 4.39,
PEFT 0.10, bitsandbytes 0.43; set `padding_side='left'` and `set_seed`.

---

## 11. Key numbers — quick reference

| Quantity | Value |
|---|---|
| FinGPT training rows | 76,772 (30,209 unique) |
| FPB `sentences_allagree` | 2,264 lines / 2,259 unique |
| **Verbatim overlap (leakage)** | **1,699 / 2,259 = 75.21%** |
| **Label agreement on matches** | **100%** |
| **Decontaminated remainder** | **560 (24.79%)** — neutral 353 · negative 77 · positive 130 |
| Original (contaminated) headline | 99.56% acc / 0.996 F1 |
| Baseline scores (broken harness) | 13–23% (below 61.6% majority floor) |
| LoRA config | r=16, α=32, dropout=0.05, targets q_proj+v_proj |
| Trainable params | **6.82M (0.094%)** — not 9M/0.2% |
| Backbone | `mistralai/Mistral-7B-v0.1`, bf16, no quantization |
| Code-vs-paper mismatches | 14 (C1–C14) |
| Verified new citations | 23 (2024–2026) |
| Manuscript | 687 lines, compiles, 13 pp, 12 `\rerun{}` markers |

---

## 12. One-line status

**Evidence, scripts, response letter, cover letter, revision plan, and the honest
manuscript rewrite are DONE. The only thing left is the GPU compute** (Phases 1–4) to
produce the real decontaminated/external/multi-seed/ablation numbers that fill the 12
`\rerun{}` placeholders — then fill, recompile, and submit.
