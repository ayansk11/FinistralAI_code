#!/bin/bash
# =============================================================================
# Quartz H100 variant of submit_and_publish.sh — run on a Quartz login node:
#
#   cd /N/scratch/ayshaikh/FinistralAI_code && \
#     curl -sL https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook/submit_and_publish_quartz.sh | bash
#
# Same flow: resumable eval sweep on h100-single, then a dependent publish
# job that pushes results_fixed/ to a GitHub branch on completion (success
# or failure). The publish job rides h100-single too (tiny 15-min CPU job;
# avoids guessing Quartz's CPU partition names).
# =============================================================================
set -euo pipefail

REPO=/N/scratch/ayshaikh/FinistralAI_code
RAW=https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook
cd "$REPO"
mkdir -p logs/slurm

curl -sL "$RAW/run_evals_bigred200.sh" > revision/runbook/run_evals_bigred200.sh
curl -sL "$RAW/publish_results_job.sh" > /N/scratch/ayshaikh/publish_results_job.sh

EVAL_JID=$(sbatch --parsable --partition=h100-single --qos=allocated --time=02:00:00 \
  --export=ALL,PY_MODULE=python/gpu/3.11.5,VENV=/N/scratch/ayshaikh/venv-finistral-qz \
  revision/runbook/run_evals_bigred200.sh)
echo "eval job:    $EVAL_JID (h100-single)"

PUB_JID=$(sbatch --parsable --partition=h100-single --qos=allocated \
  --dependency=afterany:"$EVAL_JID" /N/scratch/ayshaikh/publish_results_job.sh)
echo "publish job: $PUB_JID (afterany:$EVAL_JID)"

squeue -u ayshaikh -o '%.10i %.20j %.8T %.14E %.12R' | head -8
