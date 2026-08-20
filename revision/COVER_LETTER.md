# Cover Letter — Resubmission of JICTASA-2026-018

**Manuscript:** "Finistral AI: Financial Sentiment Analyst"
**Authors:** Zainab Mirza, Ayan Javeed Shaikh, Fazal Jalil Parkar
**Journal:** Journal of Information and Communications Technology and Applications (GR Journals)
**Submission type:** Major Revision (Round 2)

---

Dear Editor and Reviewers,

We are grateful for the opportunity to revise our manuscript, and we thank both
reviewers for a rigorous and constructive review. Their central concern was correct,
and acting on it has materially improved the paper.

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
(i) an efficient, fully reproducible single-GPU LoRA recipe for financial sentiment
analysis, and (ii) a contamination-and-evaluation case study. Specifically, we have:

1. Added a **Data Contamination Analysis** section reporting the overlap statistics,
   the train/test split mechanics, and a documented decontamination procedure.
2. **Re-evaluated** on a decontaminated, leakage-free subset of Financial PhraseBank
   (560 unseen sentences) and on external datasets not present in the training corpus.
3. **Rebuilt the evaluation harness** (left-padding for decoder-only models,
   `max_new_tokens`, per-model native prompt templates, a consistent
   `Mistral-7B-v0.1` backbone, corrected quantisation) and added **per-baseline
   confusion matrices**.
4. Added **multi-seed** results (mean ± standard deviation) with significance tests.
5. Added an **ablation study** over LoRA rank, scaling factor, target modules,
   prompt template, and training epochs.
6. Expanded the **error analysis** with representative misclassified examples and a
   per-category breakdown.
7. Replaced the GUI screenshots with **quantitative figures** (learning curves,
   precision–recall and ROC curves, ablation curves, confusion matrices).
8. **Corrected every text-vs-code discrepancy** identified during revision (training
   framework, precision/quantisation, LoRA target modules, dropout, trainable-
   parameter count, adapter sizes, prompt template, and validation loss).
9. Added **recent 2024–2026 literature** on financial LLMs, reasoning-enhanced LLMs,
   financial instruction tuning, PEFT methods, and benchmark data contamination.
10. **Released all scripts** — training, leakage analysis, corrected evaluation
    harness, ablation/multi-seed, and figure generation — for independent
    verification, and provided a **truthful AI-assistance disclosure**.

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
