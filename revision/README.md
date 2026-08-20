# Revision package — Finistral-7B-LoRA (JICTASA-2026-018)

Everything produced for the GR Journals major revision. The central finding:
**75.2% of the Financial PhraseBank evaluation set (1,699/2,259 sentences, identical
labels) is verbatim in the FinGPT training corpus** — the 99.56% headline is a
contamination artifact. The weak baselines are a *separate* broken-harness problem.

## What's here
| File | What it is |
|------|------------|
| `RESPONSE_TO_REVIEWERS.md` | Point-by-point reply to all 13 R1 comments + 5 R2 questions, with a "Corrected Claims Table" of 14 code-vs-paper fixes. |
| `COVER_LETTER.md` | 1-page resubmission cover letter (summary of changes). |
| `REVISION_PLAN.md` | Sequenced plan: reviewer point → action → script → hardware → acceptance criterion. Start here to execute. |
| `leakage_report.md` | Human-readable contamination measurement (locally reproduced). |
| `fpb_decontaminated.csv` | The 560 leakage-free FPB sentences — the honest test set. |
| `fpb_overlap_matches.csv` | Sample of leaked (sentence, label) pairs for the appendix. |
| `literature.json` | 23 verified 2024–2026 references (financial LLMs, PEFT, contamination). |
| `_verify_findings.json` | Raw multi-agent verification record (audit trail). |
| `scripts/` | Released code (see below). |

## scripts/
| Script | Purpose | Runs on |
|--------|---------|---------|
| `measure_leakage_local.py` | Reproduces the 75.2% overlap + builds the decontaminated set. **Verified working.** | CPU |
| `leakage_analysis.py` | Fuller overlap analysis (exact + normalized + MinHash/LSH fuzzy). | CPU |
| `eval_harness_fixed.py` | Corrected evaluation: left-padding, `max_new_tokens`, per-model prompts, strict parser, consistent backbone. | GPU |
| `ablation_and_seeds.py` | 5-seed runs (mean±std + significance) and the ablation grid. | GPU |
| `analysis_and_figures.py` | Learning/PR/ROC/ablation curves + per-baseline confusion matrices. | GPU |

## Manuscript
`../finistral_grjournals.tex` — honestly rewritten and **compiles clean (13 pp)**.
The 12 `\rerun{...}` markers (red in the PDF) are the only numbers still needing the
GPU re-runs in `REVISION_PLAN.md` Phases 1–4. A pre-existing `\captionsetup` bug
(unrelated to content) was also fixed so it builds.

## Next action
Open `REVISION_PLAN.md`, Phase 1 — run `eval_harness_fixed.py` on `fpb_decontaminated.csv`
and one external dataset to get the first honest accuracy number.
