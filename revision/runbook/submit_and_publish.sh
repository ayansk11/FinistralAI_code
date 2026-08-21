#!/bin/bash
# =============================================================================
# One-shot, socket-independent driver — run on a BigRed200 login node:
#
#   cd /N/scratch/ayshaikh/FinistralAI_code && \
#     curl -sL https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook/submit_and_publish.sh | bash
#
# Submits the (resumable) eval sweep, then a dependent CPU job that pushes
# results_fixed/ to a GitHub branch when the sweep ends — success OR failure —
# so the local machine can pull results without any Mac->cluster SSH.
# =============================================================================
set -euo pipefail

REPO=/N/scratch/ayshaikh/FinistralAI_code
RAW=https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook
cd "$REPO"
mkdir -p logs/slurm

# Refresh both job scripts from GitHub main (the rsync'd copy may be stale).
curl -sL "$RAW/run_evals_bigred200.sh"   > revision/runbook/run_evals_bigred200.sh
curl -sL "$RAW/publish_results_job.sh"   > /N/scratch/ayshaikh/publish_results_job.sh

EVAL_JID=$(sbatch --parsable --partition=gpu --qos=allocated --time=02:00:00 \
  revision/runbook/run_evals_bigred200.sh)
echo "eval job:    $EVAL_JID"

PUB_JID=$(sbatch --parsable --dependency=afterany:"$EVAL_JID" \
  /N/scratch/ayshaikh/publish_results_job.sh)
echo "publish job: $PUB_JID (afterany:$EVAL_JID)"

squeue -u ayshaikh -o '%.10i %.20j %.8T %.14E %.12R'
