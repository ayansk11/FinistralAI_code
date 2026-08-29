# Paper Creation Process Record — JICTASA-2026-018

**Manuscript:** Finistral AI: An Efficient Financial-Sentiment LoRA Adapter and a Train/Test Contamination Case Study
**Authors:** Zainab Mirza, Ayan Javeed Shaikh, Fazal Jalil Parkar
**Record covers:** original submission (March 2026) → major-revision completion (August 2026)
**Prepared:** 21 August 2026, per the ARS academic-pipeline Stage 6 protocol. This record documents the human–AI collaboration honestly, including AI errors, per the no-inflation rule.

---

## 1. Paper Creation Journey

| Phase | Period | What happened |
|---|---|---|
| Original submission | up to Mar 2026 | Paper submitted claiming 99.56% accuracy / SOTA on Financial PhraseBank, with baselines at 13–23%. |
| Reviews received | Jun 2026 | Two reviewers, 18 points total, both converging on the same suspicion: the numbers were too good and the baselines too bad to be real. **They were right.** |
| Contamination discovery | 23 Jun 2026 | AI-assisted audit measured **75.2% verbatim train/test overlap** (1,699/2,259 sentences, 100% label agreement) between the FinGPT training corpus and the FPB evaluation set, plus five independent defects in the evaluation harness that had crushed the baselines. Manuscript honestly rewritten; 26 result placeholders awaited GPU re-runs. |
| Stall | Jul–mid-Aug 2026 | GPU phases unexecuted; deadline (12 Jun, extended to 7 Jul) lapsed; editor sent four reminders. |
| Artifact re-verification | 20 Aug 2026 | Cross-checking the *published* adapter's own files revealed the June "correction" of the LoRA configuration was itself wrong: the released adapter trains **all seven linear modules (41.94M params, 0.58%)**, not q/v-only (6.82M) — proven by two independent file-size computations (83.9 MB fp16; 167.8 MB fp32 GGML). |
| Corrected evaluation | 20–21 Aug 2026 | All 24 model×dataset measurements completed on IU Big Red 200 / Quartz (A100 + H100) through a severe Lustre filesystem outage, using fail-fast retry chains and a venv/weights tarball workaround. Exact-match decontamination, external datasets (FiQA-SA n=235, TFNS n=2,373), per-example McNemar + bootstrap statistics, template ablation, error analysis, latency benchmark. |
| Internal review | 21 Aug 2026 | ARS pipeline: Stage 2.5 integrity gate (18/18 citations verified against arXiv/ACL/Crossref; 42 table cells mechanically checked) → 5-reviewer simulated panel + Devil's Advocate + rebuttal audit → unanimous *minor revision* with a 30-item roadmap → all P0/selected P1 items applied → Stage 4.5 fresh verification PASS. |
| Final state | 21 Aug 2026 | 18-page manuscript, zero placeholders, near-duplicate audit added, two appendices (complete statistics; consolidated corrections), three portal-ready DOCX files. One week ahead of the committed 28 Aug date. |

## 2. Key Decisions and Turning Points

1. **Honesty over rescue (June).** When the contamination was measured, the paper was reframed from a SOTA claim into a recipe + contamination case study, and the "+78 F1" claim withdrawn — rather than attempting to defend the indefensible number.
2. **The artifact as authority (20 Aug).** When training script, manuscript, and published adapter disagreed, the published artifact's own bytes were treated as ground truth; both the original paper *and* the first revision were corrected against it.
3. **Pragmatic scope (20 Aug).** Eval-only re-runs plus per-example paired statistics were chosen over a multi-week retraining program, with deferred items disclosed plainly instead of claimed.
4. **Outlasting the filesystem (20–21 Aug).** A cluster-wide Lustre pathology (imports hanging in `cl_sync_io_wait`) was diagnosed empirically — read() fast, mmap() hung; many-small-files worse than one-large-file — and defeated with single-tarball staging to node-local disk, after several failed cheaper attempts.
5. **Letting the panel cut (21 Aug).** The simulated panel's findings — including a factually wrong p-value claim and a macro-F1 omission that reversed one headline — were accepted and fixed rather than argued away.

## 3. Integrity and Failure-Mode Audit Log

- **Citations:** 18/18 verified against primary sources (16 exact fetches; 2 via Crossref canonical metadata). Zero fabricated references.
- **Reported data:** every number in the results tables and prose mechanically traced to released CSVs (42 Table-8 cells, 72 Appendix-A values, 4 quoted p-values, contamination and near-duplicate figures). Slurm logs of the producing jobs are preserved in-repo.
- **7-mode failure checklist (final):** implementation bugs — not suspected (harness independently validated); hallucinated results — not suspected (full audit trail); shortcut reliance — disclosed as the paper's own subject; bug-as-insight — not suspected (the TFNS template anomaly reported as an open finding); methodology fabrication — not suspected (methods reconciled to artifact bytes); frame-lock — mitigated by panel review and explicit no-significance statements; citation hallucination — cleared.
- **Blocking events:** none overridden; every gate passed on evidence.

## 4. Collaboration Quality Evaluation (honest, evidence-cited; 1–100)

| Dimension | Score | Evidence |
|---|---|---|
| Scientific integrity | 93 | The revision's entire purpose was self-correction; the correction of the correction (adapter config) was volunteered, not extracted; overclaims in the draft letter were fixed when caught. Deduction: the original submission's false claims (Unsloth, "8-bit NF4", "no AI used") should never have existed. |
| Evidence discipline | 90 | Every number traced to artifacts; file-size arithmetic used as independent verification; near-duplicate audit run rather than asserted. Deduction: the June q/v "correction" trusted a script over the artifact for two months. |
| Critical engagement | 85 | A genuinely adversarial panel was run and its findings applied; the user explicitly demanded "honest responses" and got a critical read that named the efficiency-framing weakness. Deduction: several defects (wrong p-value claim, macro-F1 omission) were introduced by the AI and only caught by the AI's own later review layer — single-layer drafting was insufficiently self-checked. |
| Efficiency | 74 | 24 measurements, statistics, figures, and a full revision executed in ~36 hours despite a cluster outage. Deductions: ~6 hours lost to retry churn before the mmap diagnosis; two staging strategies made things worse before the tarball insight; a monitor false-alarmed on queued jobs. |
| Transparency | 95 | Corrections consolidated in a manuscript appendix; deferred work labeled deferred; AI assistance disclosed in the paper; this record itself lists AI errors. |
| Reproducibility | 92 | Frozen evaluation sets with provenance, deterministic evaluation, released harness/audit/statistics scripts, public repo. Deduction: training-run variance unmeasured (disclosed). |

**No aggregate score is computed** (per protocol: no hidden scalar).

## 5. AI Self-Reflection

Errors made by the AI assistant during this collaboration, in the open:

1. **The wrong correction (June):** "corrected" the LoRA target-module claim to q/v-only based on the repository's training script, without checking the published adapter — propagating a new error into the abstract, contributions, and hyperparameter table for two months. Lesson: released artifacts outrank source trees.
2. **Overstated response letter (June draft):** preamble bullets claimed multi-seed evaluation, a full ablation grid, and figures before they existed. Caught only at the Stage-3 rebuttal audit. Lesson: letters must be written from delivered state, not planned state.
3. **Self-ingestion bug:** the statistics script globbed its own output file as input, double-counting rows; caught because an n of 1,120 looked wrong. Lesson: output artifacts must not match input patterns.
4. **A wrong significance sentence:** "all p < 1e-8" written where one baseline's p was 2.1×10⁻³ — introduced during results-filling, caught by four of five panel reviewers.
5. **Infrastructure over-engineering:** two node-local staging strategies (full venv, torch-only) both hung on the copy step itself before the correct single-tarball pattern; roughly six hours and several cluster jobs were spent learning what an HPC veteran might have tried first.
6. **Monitor blind spot:** a stall detector treated a queued (PENDING) job as a hung one, producing a false alarm.

What went right is visible in the deliverables; what went wrong is listed here so the next collaboration starts smarter.

## 6. Collaboration Depth (advisory observer)

**Pattern:** high delegation intensity with punctuated, well-timed human vigilance — the user delegated execution but personally: approved every Duo authentication, chose the compute strategy twice (h100-single suggestion; dual-cluster lanes), demanded an unvarnished quality assessment ("give me honest responses"), and made the scope decision (pragmatic vs full program). **Zone 2** (productive delegation with retained oversight): the human held the judgment reins at every gate that mattered — portal actions, spending compute, scope, and final acceptance — while ceding mechanics. Advisory note: the densest vigilance came late (review stages); earlier-stage spot-checks of AI-drafted claims would have caught the letter overstatements sooner.

## 7. Deliverables Inventory

**For the portal:** `JICTASA-2026-018-Finistral_AI_Revised.docx` · `..._Cover_Letter.docx` · `..._Response_to_Reviewers.docx` (+ authoritative `finistral_grjournals.pdf`, 18 pp).
**Public artifacts:** github.com/ayansk11/FinistralAI_code (manuscript source, harness, audits, frozen data, statistics, figures, slurm logs) · huggingface.co/Ayansk11/Finistral-7B_lora (adapter).
**Evidence chain:** `data_eval/PROVENANCE.md`, `near_duplicate_audit.{json,md}`, `results_fixed/stats_summary.csv`, `revision/leakage_report.md`, `revision/_verify_findings.json`.

---

*End of record. Generated under ARS academic-pipeline v3.20.1 Stage 6; scores are uninflated per Anti-Pattern #7.*
