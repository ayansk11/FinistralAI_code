# Editor Reply — Kryoni JMS Discussion Thread (JICTASA-2026-018)

**Where to post:** Peer Review Discussion → thread "JICTASA-2026-018 — Revision in
progress: data-contamination …" (the thread you opened 24 Jun).

**Draft (paste as-is or edit):**

---

Dear Editorial Manager,

We sincerely apologize for the delay in returning JICTASA-2026-018, and thank
you for your patience through the reminders. Acting on both reviewers' central
concern, our re-examination uncovered substantial train/test overlap between
the FinGPT training corpus and the Financial PhraseBank evaluation split
(75.2% of evaluation sentences appear verbatim in the fine-tuning data). This
required rebuilding our evaluation harness from scratch, decontaminating the
test set, and re-running our model and all baselines on leakage-free and
external datasets under each model's correct inference procedure.

That re-evaluation is now in its final stage. We will upload the complete
revision — revised manuscript, point-by-point response to both reviewers, and
cover letter — by **28 August 2026**.

We are grateful for the reviewers' rigor: their concerns were correct and have
materially improved the paper's honesty and value.

Kind regards,
Ayan Javeed Shaikh, on behalf of all authors

---

**Note:** posting this in the portal is your action (Claude does not act in the
journal portal). After posting, the remaining schedule is in the approved plan:
GPU run → results ingest → manuscript fill → letters + DOCX → submit 28 Aug.
