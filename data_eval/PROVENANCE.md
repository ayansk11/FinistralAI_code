# Evaluation Set Provenance — JICTASA-2026-018

Generated 2026-08-20 by `revision/scripts/prepare_external_datasets.py`
(CPU-only). Uniform schema: `sentence,label` with string labels
negative/neutral/positive. All decontamination uses the same normalization as
`revision/scripts/measure_leakage_local.py` (NFKC fold, lowercase, strip
non-alphanumerics, collapse whitespace) against the
`FinGPT/fingpt-sentiment-train` `input` column (30209 unique
normalized inputs).

## fpb_decontam.csv — 560 rows (negative 77 / neutral 353 / positive 130)
Verbatim copy of `revision/fpb_decontaminated.csv`: the Financial PhraseBank
`sentences_allagree` sentences (takala/financial_phrasebank, via the
gtfintechlab parquet mirror) NOT present in the FinGPT training corpus
(1,699 of 2,259 unique sentences = 75.21% were contaminated and removed).
Re-verified leakage-free at generation time (0 matches).

## fiqa.csv — 235 rows (negative 76 / neutral 12 / positive 147), 938 dropped
FiQA-SA, ALL splits pooled (`TheFinAI/fiqa-sentiment-classification`, fallback `pauri32/fiqa-2018`),
then decontaminated row-wise. Pooling rationale (disclosed in the paper):
FinGPT ingested FiQA train data, so the FiQA test split alone is heavily
contaminated (only 51/234 rows survive); any FiQA sentence absent from the
FinGPT corpus was never seen by the model regardless of split. Continuous
sentiment score mapped to labels: score >= 0.1 -> positive,
score <= -0.1 -> negative, else neutral. FiQA-SA has
essentially no neutral band by construction, so the neutral class is small.
Dropped rows matched a FinGPT training input after normalization (or were
duplicates/empty).

## tfns.csv — 2373 rows (negative 344 / neutral 1557 / positive 472), 15 dropped
Twitter Financial News Sentiment validation split (`zeroshot/twitter-financial-news-sentiment`). Label map:
0 Bearish -> negative, 1 Bullish -> positive, 2 Neutral -> neutral. TFNS train
is a constituent source of the FinGPT corpus; dropped rows matched a FinGPT
training input after normalization (or were duplicates/empty).

## SemEval-2017 Task 5
Not included: no maintained public loader (original distribution requires
registration). Recorded as not delivered in the response to reviewers.
