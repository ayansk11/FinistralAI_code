#!/bin/bash
# =============================================================================
# Staggered retry chain — run on a BigRed200 login node:
#
#   curl -sL https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook/submit_retry_chain.sh | bash
#
# Rationale: fresh torch imports from /N/scratch have been hanging on COMPUTE
# nodes since ~15:00 (login nodes unaffected) — cluster-side Lustre
# degradation. Each eval attempt fails fast (~3 min) while the filesystem is
# sick, and the resume guard makes attempts after a successful one nearly
# free. So: schedule 6 attempts at hourly offsets to catch the recovery
# without babysitting, then publish whatever exists at the end.
# =============================================================================
set -euo pipefail

REPO=/N/scratch/ayshaikh/FinistralAI_code
RAW=https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook
cd "$REPO"
mkdir -p logs/slurm

curl -sL "$RAW/run_evals_bigred200.sh" > revision/runbook/run_evals_bigred200.sh
curl -sL "$RAW/publish_results_job.sh" > /N/scratch/ayshaikh/publish_results_job.sh

LAST=""
for H in 1 2 3 4 5 6; do
  JID=$(sbatch --parsable --partition=gpu --qos=allocated --time=02:00:00 \
    --begin="now+${H}hours" revision/runbook/run_evals_bigred200.sh)
  echo "attempt +${H}h: job $JID"
  LAST=$JID
done

PUB=$(sbatch --parsable --partition=general --dependency=afterany:"$LAST" \
  /N/scratch/ayshaikh/publish_results_job.sh)
echo "final publisher: $PUB (afterany:$LAST)"
squeue -u ayshaikh -o '%.10i %.20j %.8T %.14E %.20S' | grep -i finistral
