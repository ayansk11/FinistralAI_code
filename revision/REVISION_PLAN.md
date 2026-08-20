# Finistral-7B-LoRA — Revision Plan (JICTASA-2026-018)

Sequenced plan to take the paper from "major revision" to resubmission. Maps every
reviewer point to a concrete action, the script that does it, the hardware needed,
and an acceptance criterion. Items are ordered so that each unblocks the next.

**Root cause (measured, not assumed):** 75.2% of the Financial PhraseBank
`sentences_allagree` evaluation set (1,699 / 2,259 unique sentences) appears
*verbatim, with identical gold labels*, in the FinGPT training corpus the adapter
was fine-tuned on. The 99.56% headline is a contamination artifact. The
sub-majority-class baselines (13–23%) are a *separate*, independent problem: a
broken evaluation harness. Both must be fixed; neither alone explains the paper.

---

## Status legend
- ✅ **DONE** in this revision pass (committed under `revision/`)
- 🔬 **NEEDS GPU** — requires an A100/L4-class GPU re-run; produces the numbers in
  the `\rerun{}` placeholders in `finistral_grjournals.tex`
- ✍️ **WRITING** — prose/figure work once 🔬 numbers exist

---

## Phase 0 — Evidence (DONE)
| # | Action | Artifact | Status |
|---|--------|----------|--------|
| 0.1 | Measure train/test overlap locally | `scripts/measure_leakage_local.py`, `leakage_report.md` | ✅ 75.21% verbatim, 100% label-match |
| 0.2 | Build decontaminated FPB test set | `fpb_decontaminated.csv` (560 sentences) | ✅ |
| 0.3 | Sample of leaked pairs for the appendix | `fpb_overlap_matches.csv` | ✅ |
| 0.4 | Audit code vs manuscript | this plan + letter "Corrected Claims Table" | ✅ 14 mismatches found |
| 0.5 | Honest manuscript rewrite (no-new-data parts) | `../finistral_grjournals.tex` (compiles, 13 pp) | ✅ |
| 0.6 | Response-to-reviewers letter | `RESPONSE_TO_REVIEWERS.md` | ✅ |
| 0.7 | Verified 2024–2026 citations | `literature.json` (23 refs) | ✅ |

## Phase 1 — Decontaminated + external evaluation 🔬 (the keystone numbers)
Unblocks: R1.1, R1.3, R1.8, R2.1, R2.5, and Table `tab:corrected-results`.

1. **Re-run Finistral on the 560-sentence decontaminated FPB set** using the
   `[INST]` template that matches training. → `scripts/eval_harness_fixed.py`
   - Acceptance: accuracy reported with 95% CI; expect a drop from 99.56% toward
     the ~0.80–0.90 range. Whatever it is, it is the honest FPB number.
2. **Re-run on external, leakage-free datasets** *not* in `fingpt-sentiment-train`:
   - FiQA-SA (held-out test), Twitter Financial News Sentiment (`zeroshot/twitter-financial-news-sentiment`) test split, SemEval-2017 Task 5.
   - Verify non-overlap with training first (reuse `measure_leakage_local.py` logic).
   - Acceptance: external accuracy + weighted-F1; this is the primary generalisation claim.
3. **Decision point:** if decontaminated/external accuracy is competitive → "efficient
   reproducible recipe" framing holds. If it collapses → the paper becomes primarily a
   contamination case study (still publishable, already framed that way in the rewrite).

## Phase 2 — Fix and re-run ALL baselines 🔬
Unblocks: R1.2, R2.2, R2.3, R2.4, Table 7 replacement.

4. Run every baseline under its **native** inference procedure via
   `scripts/eval_harness_fixed.py` (left-padding, `max_new_tokens`, per-model prompt,
   strict parser, consistent `mistralai/Mistral-7B-v0.1` backbone for the base row).
   - Baselines: Mistral-7B-v0.1 zero-shot, FinGPT-mt-Llama2-7B-LoRA, FinGPT-Llama-3-8B-LoRA,
     FinGPT-Falcon-7B-LoRA, FinGPT-Bloom-7B1-LoRA, FinBERT (`ProsusAI/finbert`).
   - Acceptance: FinGPT baselines recover to ≈0.84–0.86 (published range), *not* 13–23%.
     If any stays below the 61.6% majority floor, the harness is still wrong — debug before reporting.
5. Emit **per-baseline confusion matrices** (R2.4) via `scripts/analysis_and_figures.py`.

## Phase 3 — Statistical validation 🔬
Unblocks: R1.5, R2.5.

6. Retrain + re-evaluate across **5 seeds** (`scripts/ablation_and_seeds.py`); report
   mean ± std for accuracy and weighted-F1 on decontaminated + external sets.
7. **Significance tests** for every comparison: McNemar (paired predictions) and/or
   bootstrap CIs. No comparative claim without one.

## Phase 4 — Ablations 🔬
Unblocks: R1.6, replaces GUI screenshots (R1.10).

8. `scripts/ablation_and_seeds.py` grid on the decontaminated set:
   - LoRA rank r ∈ {8, 16, 32, 64}; α ∈ {16, 32, 64}; dropout ∈ {0.0, 0.05, 0.1}
   - target modules: {q,v} vs {q,k,v,o} vs +MLP
   - prompt template: `[INST]` vs Alpaca (quantifies the train/eval mismatch)
   - epochs / checkpoints: {1, 2, 3, 4}
   - Acceptance: ablation **curves** (not screenshots) for each axis.

## Phase 5 — Error analysis ✍️🔬
Unblocks: R1.7.

9. From decontaminated + external predictions, build a misclassified-examples table
   (sentence, gold, pred, raw model output) and a per-category breakdown. Discuss
   negation, hedged guidance, mixed-signal sentences. → `scripts/analysis_and_figures.py`.

## Phase 6 — Figures ✍️🔬
Unblocks: R1.10.

10. Generate and insert: training/validation **learning curves**, per-class **PR** and
    **ROC** curves, **ablation curves**, per-baseline **confusion matrices**. Move the
    two Gradio screenshots (`figures/image1.jpeg`, `image2.jpeg`) to an appendix or cut.
    Regenerate `image3/image4` bar charts from corrected numbers. → `scripts/analysis_and_figures.py`.

## Phase 7 — Text polish ✍️ (mostly DONE)
- R1.4 novelty moderated ✅ · R1.9 language pass (do a final proofread) ⏳ ·
  R1.11/R1.12 discussion + 2024–26 lit ✅ (extend with `literature.json` as needed) ·
  R1.13 scripts released ✅ · AI-disclosure corrected ✅.

## Phase 8 — Fill placeholders & finalize ✍️
11. Replace all 11 `\rerun{}` markers in `finistral_grjournals.tex` with Phase 1–4 numbers.
12. Recompile (`pdflatex` ×2), confirm 0 undefined refs.
13. Update `Finistral_AI_GRJournals.docx` to match (or submit the LaTeX PDF if the journal accepts it).
14. Bundle `revision/scripts/*` + `fpb_decontaminated.csv` into the public repo; cite the commit in R1.13.
15. Submit revised manuscript + `RESPONSE_TO_REVIEWERS.md` + corrected files via the Kryoni JMS portal **before the June issue deadline** (Editorial Manager flagged Response/Review due dates as overdue — request a short extension citing the scope of the contamination re-analysis).

---

## Hardware / environment
- **GPU runs (Phases 1–4):** 1× A100/40GB (or L4/A10) is sufficient; Mistral-7B in bf16
  inference fits comfortably. Big Red 200 (already used) is ideal. Est. compute:
  eval ≈ minutes/dataset; 5-seed retrain ≈ 5 × ~2h; full ablation ≈ 0.5–1 GPU-day.
- **Pins:** PyTorch 2.2, Transformers 4.39, PEFT 0.10, datasets ≥2.x, bitsandbytes 0.43
  (the paper's stack). Set `padding_side='left'`, `set_seed`, and log versions.
- **CPU-only (done):** contamination measurement runs anywhere (`measure_leakage_local.py`).

## Acceptance criteria for resubmission (self-check)
- [ ] No claim depends on the contaminated split; 99.56% appears only as a diagnostic.
- [ ] Every baseline ≥ majority-class floor under the corrected harness.
- [ ] All headline numbers carry mean ± std + a significance test.
- [ ] At least one external (non-FinGPT) dataset reported.
- [ ] Zero `\rerun{}` markers remain; manuscript compiles clean.
- [ ] Every code-vs-text claim in the "Corrected Claims Table" (letter) is satisfied.
- [ ] Scripts in the public repo reproduce every number end-to-end.

## Open decisions for the authors
- **Extension:** the JMS portal shows Response/Review due 12 Jun 2026 (overdue). Decide
  whether to request an extension now vs. submit a partial revision + timeline.
- **External datasets:** confirm the three proposed (FiQA-SA / TFNS / SemEval-2017 T5)
  or substitute a freshly hand-labelled set.
- **Framing if accuracy collapses:** confirm comfort with the "contamination case study"
  emphasis (already the spine of the rewrite).
