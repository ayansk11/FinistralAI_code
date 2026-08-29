# Cover Letter — Resubmission of JICTASA-2026-018

**Manuscript:** "Finistral AI: Financial Sentiment Analyst"
**Authors:** Zainab Mirza, Ayan Javeed Shaikh, Fazal Jalil Parkar
**Journal:** Journal of Information and Communications Technology and Applications (GR Journals)
**Submission type:** Major Revision (Round 2)
**Date:** 28 August 2026

---

Dear Editor and Reviewers,

We are grateful for the opportunity to revise our manuscript, and we thank both
reviewers for a rigorous and constructive review. Their central concern was correct,
and acting on it has materially improved the paper. We also apologise for the delay
in returning this revision: the contamination finding described below obliged us to
rebuild our evaluation from the ground up — decontaminating the test data,
reconstructing the harness, and re-running every model — and we judged that doing
this properly was worth the additional time.

**The headline change.** Following Reviewer 1 (point 3) and Reviewer 2 (question 1),
we performed a direct data-overlap audit between our training corpus
(`FinGPT/fingpt-sentiment-train`) and our evaluation set (Financial PhraseBank,
`sentences_allagree`). We found that **75.2% of the evaluation sentences (1,699 of
2,259 unique) appear verbatim — with the identical gold label in 100% of cases —
inside the corpus the model was fine-tuned on.** The originally reported 99.56%
accuracy is therefore a data-contamination artifact rather than a measure of
generalisation, and we have withdrawn the state-of-the-art claim accordingly. We also
determined that the unusually low baseline scores were caused by defects in our
evaluation harness, not by weakness of those models, and have withdrawn the
"+78 F1 over baselines" claim.

**How the paper has changed.** The revised manuscript is reframed honestly as
(i) an efficient, fully reproducible LoRA recipe for financial sentiment analysis,
and (ii) a contamination-and-evaluation case study. Specifically, we have:

1. Added a **Data Contamination Analysis** section reporting the overlap statistics,
   the train/test split mechanics, and a documented decontamination procedure.
2. **Re-evaluated** on an exact-match-decontaminated subset of Financial PhraseBank
   (560 sentences, with a released near-duplicate audit bounding residual overlap)
   and on two external datasets decontaminated row-by-row
   against the training corpus — FiQA-SA (n = 235) and Twitter Financial News
   Sentiment (n = 2,373) — with full dataset provenance released.
3. **Rebuilt the evaluation harness** (left-padding for decoder-only models,
   `max_new_tokens`-bounded greedy decoding, per-model native prompt templates, a
   consistent `Mistral-7B-v0.1` backbone, a strict non-defaulting label parser) and
   added **per-baseline confusion matrices**, plus **FinBERT** as an additional
   baseline evaluated through its own classification head.
4. Added **per-example statistical validation** for every comparison — McNemar tests
   and paired-bootstrap 95% confidence intervals — computed from released
   per-example prediction files (evaluation itself is deterministic by design;
   training-seed replication is released as a runnable protocol and noted as
   deferred).
5. Added an evaluation-time **prompt-template ablation** (training template vs the
   original mismatched Alpaca template) quantifying the disclosed protocol defect;
   the full training-time ablation grid is released as runnable code and flagged as
   future work rather than claimed as done.
6. Expanded the **error analysis** with representative misclassified examples and a
   per-category breakdown.
7. Replaced the contaminated comparison figures with **quantitative figures from the
   corrected evaluation** (per-class precision/recall/F1, ROC and PR curves,
   per-baseline confusion matrices); the original run's learning curves could not be
   faithfully regenerated and the recorded validation-loss table stands in for them,
   which the manuscript states plainly.
8. **Corrected every text-vs-code-vs-artifact discrepancy** identified during
   revision — including re-verifying the LoRA configuration against the published
   adapter file itself (all seven projection modules, 41.94 M trainable parameters,
   0.58% of the backbone; the adapter's file sizes confirm the count exactly) — plus
   the training framework, precision/quantisation, dropout, adapter-export precision,
   prompt template, and validation loss.
9. Added **recent 2024–2026 literature** on financial LLMs, reasoning-enhanced LLMs,
   financial instruction tuning, PEFT methods, and benchmark data contamination.
10. **Released everything** — training script, corrected harness, leakage analysis,
    external-dataset preparation with provenance, statistics pipeline, figure
    generators, frozen evaluation sets, and the original flawed notebooks (for
    transparency) — publicly at `github.com/ayansk11/FinistralAI_code`, and provided
    a **truthful AI-assistance disclosure**.

We recognise that these corrections substantially lower our headline numbers. We
believe the honest, decontaminated results and the contamination case study are a
more valuable and trustworthy contribution than the original inflated figure, and we
are grateful to the reviewers for guiding us there. A detailed point-by-point response
to every reviewer comment is provided in the accompanying **Reviewer Response File**.

We hope the revised manuscript now meets the journal's standards and look forward to
your assessment.

Sincerely,
Ayan Javeed Shaikh, on behalf of all authors
ayshaikh@iu.edu
