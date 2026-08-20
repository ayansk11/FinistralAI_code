# Response to Reviewers — Manuscript JICTASA-2026-018

**"Finistral AI: Financial Sentiment Analyst"** (Mirza, Shaikh, Parkar)
Submitted to GR Journals — Major Revision

---

## Preamble: Summary of Major Changes

We thank both reviewers for their exceptionally careful and technically rigorous reading. Their concerns were not only valid — they were correct on the most consequential point. Acting on Reviewer 1's point 3 and Reviewer 2's question 1, we conducted a direct data-overlap analysis between our training corpus (`FinGPT/fingpt-sentiment-train`) and our evaluation set (`takala/financial_phrasebank`, `sentences_allagree`). The result is unambiguous and we report it plainly:

**1,699 of the 2,259 unique evaluation sentences (75.2%) appear verbatim — with the identical gold label in 100% of cases — inside the corpus the model was fine-tuned on** (locally reproduced via `revision/scripts/measure_leakage_local.py`; the original AllAgree file lists 2,264 lines, 2,259 unique). The reported 99.56% accuracy is therefore a data-contamination artifact, not a measure of generalization. We were wrong to present it as a state-of-the-art result, and we have removed that claim from the abstract, title framing, and body of the paper.

We have also confirmed, by reproducing the harness and inspecting the committed notebook outputs, that the baseline scores (13–23%, below the 61.6% majority-class floor) are artifacts of a broken evaluation harness — right-padding on decoder-only models, a `max_length`/`padding=True` collision, a single shared prompt mismatched to each model's native template, a silently-dropped quantization config, and inconsistent Mistral backbones. The "+78 F1 over baselines" claim is consequently invalid and has been withdrawn.

The revised manuscript is no longer a SOTA-accuracy paper. It is reframed honestly as **an efficient, fully reproducible LoRA recipe and an evaluation-and-contamination case study** for financial sentiment analysis. The major changes are:

1. **New Section on Data Contamination** (Reviewer 1.3, Reviewer 2.1): reports the 75.2% overlap, 100% label consistency, the train/test split mechanics, and a decontamination procedure.
2. **Decontaminated re-evaluation**: accuracy and weighted-F1 reported on the disjoint 560-sentence remainder *and* on external, leakage-free datasets.
3. **Fixed evaluation harness** (Reviewer 1.2, Reviewer 2.2/2.3/2.4): `padding_side='left'`, `max_new_tokens`, per-model native prompts, consistent `Mistral-7B-v0.1` backbone, corrected quantization; per-baseline confusion matrices added.
4. **Multi-seed evaluation** with mean ± std and significance testing (Reviewer 1.5, Reviewer 2.5).
5. **Ablation study** over LoRA rank, alpha, target modules, prompt template, and epochs (Reviewer 1.6).
6. **Expanded error analysis** with misclassified examples and per-category breakdown (Reviewer 1.7).
7. **New quantitative figures** (learning curves, PR/ROC, ablation curves) replacing the GUI screenshots (Reviewer 1.10).
8. **Corrected text-vs-code mismatches** throughout (Unsloth, "8-bit NF4", LoRA targets, dropout, param count, adapter sizes, prompt template, val loss).
9. **Expanded 2024–2026 literature** (Reviewer 1.12) and a **truthful AI-assistance disclosure**.
10. **Released scripts**: `eval_harness_fixed.py`, `leakage_analysis.py`, `ablation_and_seeds.py`, `analysis_and_figures.py` (Reviewer 1.13).

We recognize that these corrections substantially lower our headline numbers. We believe the honest, decontaminated result and the contamination case study are a more useful contribution than the original inflated figure, and we are grateful to the reviewers for steering us there.

---

## Corrected Claims Table

| # | Original claim (as submitted) | Status | Revised wording / action |
|---|---|---|---|
| C1 | "99.56% accuracy, state-of-the-art on Financial PhraseBank" | **Withdrawn** | "On a decontaminated, disjoint test set, Finistral-7B-LoRA attains [corrected acc/F1, expected ~0.80–0.90 range]; the original 99.56% reflects 75.2% train/test overlap and is not a generalization result." |
| C2 | "+78 weighted-F1 over FinGPT baselines / outperforms by large margins" | **Withdrawn** | "Under a corrected harness, baselines score near their published ~0.84–0.86 accuracy; differences are modest and reported with significance tests." |
| C3 | "Outperforms FinBERT by +3–6 accuracy points" | **Softened** | Either evaluated under an identical correct protocol, or stated explicitly as a literature-reported comparison with citation and matched split. |
| C4 | "We utilise Unsloth … 2× faster, ~70% less VRAM" | **Removed** | Describe actual stack: `transformers` + `peft` + HF `Trainer` + gradient checkpointing + xFormers memory-efficient attention. `\cite{unsloth2023}` removed. |
| C5 | "Backbone loaded in 8-bit NF4 (load_in_8bit=True)" during training | **Corrected** | "Training used bf16 full-precision weights with no quantization." The impossible "8-bit NF4" phrasing removed. |
| C6 | "LoRA injected into every attention-projection and feed-forward linear layer" | **Re-verified — original targets confirmed, count corrected** | The published adapter's `adapter_config.json` (Ayansk11/Finistral-7B_lora) targets all seven linear modules (q/k/v/o/gate/up/down), so the original submission's *coverage* description was correct. (An intermediate revision draft "corrected" this to q/v-only based on a training-script variant that did not match the released artifact; the artifact is authoritative and the released training script has been reconciled to it.) |
| C7 | "LoRA dropout = 0.10" (Table 4) | **Corrected** | 0.05 (matches the published `adapter_config.json` and code). |
| C8 | "Trainable parameters ≈ 9M (0.2%)" | **Corrected** | 41,943,040 ≈ 41.94M (≈ 0.58% of 7.24B). Verified two independent ways from the released files: fp16 `adapter_model.safetensors` = 41.94M × 2 B = 83.9 MB; fp32 GGML export = 41.94M × 4 B = 167.8 MB. (The old 9M figure was arithmetically impossible: a 9M fp16 adapter would be ~18 MB, not 83.9 MB.) |
| C9 | "4-bit GGML adapter = 168 MB; fp16 adapter = 83.9 MB" | **Corrected** | The 168 MB `ggml-adapter-model.bin` exists but is an **fp32** export (167.8 MB = 41.94M × 4 B), not 4-bit; the "4-bit" label and the sub-100 ms CPU-latency claim are withdrawn. |
| C10 | "Sentences wrapped exactly as in the training script" (Instruction/Input/Answer) | **Corrected** | Training used `[INST]{input}\n{instruction} [/INST] {output}`; the train/eval template mismatch is disclosed and the eval re-run under the `[INST]` template. |
| C11 | "Best val loss 0.1008" vs Table 6 "0.1009" | **Corrected** | Single value from training logs used consistently. |
| C12 | "All baselines evaluated with identical truncation + greedy decoding" | **Corrected** | Harness was broken (right-padding, `max_length` collision, neutral-default parser); fixed and re-run; honest protocol described. |
| C13 | "Mistral-7B-Base" baseline = Finistral's backbone | **Corrected** | Base baseline re-run on identical `mistralai/Mistral-7B-v0.1` (was `unsloth/mistral-7b-v0.2`). |
| C14 | "No AI-assisted technology used in writing" | **Corrected** | Truthful disclosure consistent with `generate_docx.py` and drafting tools used. |

---

# Reviewer 1

## R1.1 — Validate the extreme 99.56% accuracy

**We acknowledge this concern and, on investigation, found that the reviewer's suspicion was justified.** We performed a direct cross-match between the 2,259 unique `sentences_allagree` evaluation sentences and the 76,772-row `FinGPT/fingpt-sentiment-train` corpus on which the adapter was fine-tuned. **1,699 evaluation sentences (75.2%) appear verbatim in the training inputs, with the identical 3-class gold label in 100% of cases.** Reproducing the training script's own `train_test_split(test_size=0.05, seed=42)`, ~1,614 of those leaked sentences fall in the 95% fine-tuning partition — i.e., the model was trained on the exact (sentence, label) pairs it was later scored on.

The 99.56% is therefore not validated as a generalization result; it is invalidated as a contamination artifact. **Action:** we have (a) added a dedicated contamination section reporting these overlap statistics; (b) re-evaluated on the disjoint 560-sentence remainder never seen in training; (c) re-evaluated on external leakage-free datasets (below). We report those corrected numbers as the paper's results and have removed the 99.56% SOTA framing from the abstract and title. We released `leakage_analysis.py` so the overlap is independently reproducible.

## R1.2 — Implausibly low baselines; provide implementation details

**Acknowledged; the baselines are not a fair measure of those models.** Inspecting the committed notebook outputs and reproducing the harness, the sub-majority-class scores (13–23%, below the 61.6% neutral floor) trace to concrete harness defects:

- **Right-padding on decoder-only models** — the output stream contains 566 `please set padding_side=left` warnings (Falcon/Llama-3/Bloom); generation is conditioned on trailing pad tokens.
- **`generate(**tokens, max_length=512)` with `padding=True`** — `max_length` is silently ignored; models ramble for hundreds of tokens (the base row took 5h31m at ~70s/it).
- **A single Alpaca prompt** applied to every model, mismatched to each FinGPT adapter's native `Instruction/Input/Answer + Options` template (and to Finistral's own `[INST]` training template).
- **A silently-dropped quantization config** — `bnb_8bit_*` kwargs are not valid `BitsAndBytesConfig` parameters and resolve to plain int8.
- **A neutral-defaulting parser** that scores corrupted generations as `neu`.

**Action:** we rebuilt the harness (`eval_harness_fixed.py`): `padding_side='left'`, `max_new_tokens=8`, each baseline run under its own published inference procedure, a single consistent `Mistral-7B-v0.1` backbone, and corrected quantization. We report the corrected baseline numbers (expected near the published ~0.84–0.86 accuracy / ~0.84–0.92 F1) with full per-model configuration in an appendix.

## R1.3 — Possible data leakage; duplicate/near-duplicate analysis

**Confirmed — this is the central finding.** `FinGPT/fingpt-sentiment-train` aggregates four sources, one of which is Financial PhraseBank itself; the FinGPT replication notes explicitly state that for FPB "all data in the train part were used in finetuning." Our normalized exact-match and fuzzy (Jaccard ≥ 0.9) analysis quantifies it: 1,699 verbatim of 2,259 unique (75.2%), 100% label-consistent, uniform across classes (pos 77.2% / neu 74.6% / neg 74.6%); 560 sentences remain genuinely unseen.

**Action:** we added the full overlap analysis (exact + normalized + MinHash/LSH), report accuracy/F1 on the decontaminated disjoint remainder, and validate on truly held-out external sets (R1.8/R2.5). `leakage_analysis.py` is released.

## R1.4 — Limited novelty (LoRA on existing backbone)

**We accept that the original framing over-claimed novelty.** With the SOTA claim withdrawn, we no longer position the work as a methodological breakthrough. **Action:** we reframe the contribution as (a) a *fully reproducible, efficient* LoRA recipe (~41.94M trainable params, ~0.58%, bf16, single-node) and (b) a *contamination-and-evaluation case study* demonstrating how FinGPT-train/FPB overlap and harness defects produce simultaneously inflated and deflated numbers across the financial-LLM literature. We position this honestly as a cautionary, reproducibility-focused contribution rather than a novel architecture.

## R1.5 — No statistical validation; multi-seed mean+std+significance

**Acknowledged; the original was a single run.** **Action:** `ablation_and_seeds.py` retrains and re-evaluates across 5 seeds; we report mean ± std for accuracy and weighted-F1 on the decontaminated and external sets, with paired significance tests (McNemar / bootstrap CIs) for every comparison. No comparative claim is made without a significance test.

## R1.6 — Missing ablation (rank, alpha, prompt, epochs, PEFT configs)

**Acknowledged.** **Action:** we added an ablation over LoRA rank r ∈ {8,16,32,64}, alpha, target modules (q/v vs q/k/v/o vs +MLP), prompt template (`[INST]` vs Alpaca), and epochs {1,2,3,4}, all evaluated on the decontaminated set. Results are reported as ablation curves (replacing GUI screenshots).

## R1.7 — Insufficient error analysis

**Acknowledged.** **Action:** we added a per-category error breakdown and a table of misclassified examples (sentence, gold, predicted, model output) drawn from the decontaminated/external evaluations, with discussion of failure modes (negation, hedged guidance, mixed-signal sentences).

## R1.8 — Limited benchmark; more financial datasets

**Acknowledged, and essential given the contamination.** **Action:** we evaluate on datasets not present in `fingpt-sentiment-train`: FiQA-SA (held-out), Twitter Financial News Sentiment test, SemEval-2017 Task 5, and a small fresh hand-labeled set. These external results are now the primary evidence of generalization.

## R1.9 — Language/grammar

**Acknowledged.** **Action:** the manuscript has undergone a full language and consistency pass.

## R1.10 — Replace GUI screenshots with learning/PR/ROC/ablation curves

**Acknowledged.** **Action:** `analysis_and_figures.py` generates training/validation learning curves, per-class PR and ROC curves, ablation curves, and per-baseline confusion matrices. The Gradio screenshots (image1/image2) are moved to an appendix or removed; image3/image4 bar charts are regenerated from corrected numbers.

## R1.11 — Deeper discussion of *why* it outperforms

**Acknowledged — and the honest answer reframes this entirely.** The original apparent outperformance was driven by (i) contamination inflating Finistral and (ii) a broken harness deflating baselines, not by genuine model superiority. **Action:** the discussion now explains these mechanisms directly. Any residual, statistically-significant advantage on the decontaminated/external sets is discussed in terms of task-matched instruction tuning, not SOTA accuracy.

## R1.12 — Recent 2024–2026 literature

**Acknowledged.** **Action:** we expanded related work with recent financial-LLM, reasoning-LLM, financial instruction-tuning, and PEFT references (e.g., FinLoRA, arXiv:2505.19819, and contemporaneous QLoRA/financial-sentiment benchmarks), and use their published FPB numbers as sanity checks for our corrected baselines.

## R1.13 — Release training + eval scripts

**Acknowledged and done.** **Action:** we release `Finistral_Sentiment_analyst.py` (training), `eval_harness_fixed.py`, `leakage_analysis.py`, `ablation_and_seeds.py`, and `analysis_and_figures.py`, with environment pins, seeds, and the decontamination procedure, so all corrected results are reproducible end-to-end.

---

# Reviewer 2

## R2.1 — How was FinGPT-train / FPB overlap prevented?

**Honestly: it was not.** No de-duplication was performed, and FPB is a constituent source of `FinGPT/fingpt-sentiment-train`. As reported under R1.3, 75.2% of the evaluation set leaked into training with identical labels. **Action:** we added the overlap analysis, decontaminated the test set, re-evaluated on the disjoint remainder and on external sets, and documented the procedure in `leakage_analysis.py`. We do not attempt to minimize this; it is the paper's central correction.

## R2.2 — Why do FinGPT baselines do dramatically worse than published?

**Because our harness was broken, not because the models are weak.** As detailed in R1.2, right-padding, the `max_length` collision, a mismatched shared prompt, dropped quantization, and a neutral-default parser drove scores below the majority-class floor — an outcome no functioning model produces on FPB-AllAgree. **Action:** under the corrected harness we expect the FinGPT baselines to recover to their published ~0.84–0.86 accuracy; we report the corrected numbers and the corresponding shrinkage of Finistral's apparent margin.

## R2.3 — Were baselines evaluated per their original inference procedures?

**No — and this was a methodological error.** All models were run with one shared Alpaca prompt rather than each adapter's native template, and on an inconsistent backbone (`unsloth/mistral-7b-v0.2` for the "base" row vs `mistralai/Mistral-7B-v0.1` for Finistral). **Action:** each baseline is now evaluated with its own official prompt/inference procedure and on the consistent `Mistral-7B-v0.1` backbone, documented per model.

## R2.4 — Provide confusion matrices for all baselines

**Acknowledged.** **Action:** `analysis_and_figures.py` outputs per-baseline confusion matrices (and PR/ROC) under the corrected harness; these are included for every model in Table 5, not only for Finistral.

## R2.5 — Replicate across seeds + test on external datasets

**Acknowledged.** **Action:** as in R1.5/R1.8, we report 5-seed mean ± std with significance tests, and we add external leakage-free datasets (FiQA-SA held-out, Twitter Financial News test, SemEval-2017 Task 5, fresh hand-labeled) as the primary generalization evidence.

---

## Closing

We are grateful to both reviewers. Their scrutiny surfaced a contamination problem and a harness problem that, together, fully account for the implausible numbers in the original submission. The revised paper reports lower, honest, decontaminated results; reframes the contribution as an efficient and reproducible recipe plus an evaluation/contamination case study; corrects every text-vs-code discrepancy; and releases all scripts for independent verification. We hope this constitutes the kind of rigorous, transparent revision the reviewers' comments called for.

---

Key repository files referenced: `/Users/ayansk11/Desktop/FinistralAI_code/finistral_grjournals.tex`, `/Users/ayansk11/Desktop/FinistralAI_code/Finistral_Sentiment_analyst.py`, `/Users/ayansk11/Desktop/FinistralAI_code/test_final.ipynb`, `/Users/ayansk11/Desktop/FinistralAI_code/test_final2.ipynb`, `/Users/ayansk11/Desktop/FinistralAI_code/inference_test.ipynb`, `/Users/ayansk11/Desktop/FinistralAI_code/generate_docx.py`. New scripts to be released: `eval_harness_fixed.py`, `leakage_analysis.py`, `ablation_and_seeds.py`, `analysis_and_figures.py`.