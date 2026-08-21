#!/bin/bash
# =============================================================================
# Dependent CPU job: publish results_fixed/ to GitHub branch results-run-<id>.
# Runs afterany:<eval-job>, so it pushes whatever results exist even if the
# eval attempt failed fast — partial results still let local analysis proceed.
#
# Uses the cluster's own GitHub SSH key (registered as "BigRed200" on the
# ayansk11 account). known_hosts is redirected to scratch because the home
# NFS is over quota (writes there fail with EDQUOT).
# =============================================================================
#SBATCH --job-name=finistral-publish
#SBATCH --account=r01510
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/N/scratch/ayshaikh/FinistralAI_code/logs/slurm/publish-%j.out

set -euo pipefail

RESULTS_SRC=/N/scratch/ayshaikh/FinistralAI_code/results_fixed
# git object writes on /N/scratch hit ESTALE (job 8025162); use node-local /tmp.
WORK="/tmp/finpub-${SLURM_JOB_ID:-$$}"
GITSSH="ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/N/scratch/ayshaikh/.known_hosts"

test -d "$RESULTS_SRC" || { echo "no results_fixed/ to publish"; exit 1; }

rm -rf "$WORK"
GIT_SSH_COMMAND="$GITSSH" git clone --depth 1 git@github.com:ayansk11/FinistralAI_code.git "$WORK"
cd "$WORK"

BRANCH="results-run-${SLURM_JOB_ID:-manual}"
git checkout -b "$BRANCH"
rm -rf results_fixed
cp -r "$RESULTS_SRC" results_fixed
# Include the slurm logs for the eval attempts (diagnostic provenance).
mkdir -p results_fixed/slurm_logs
cp /N/scratch/ayshaikh/FinistralAI_code/logs/slurm/finistral-eval-*.out results_fixed/slurm_logs/ 2>/dev/null || true

git add -f results_fixed
git -c user.name="Ayan Javeed Shaikh (BigRed200)" -c user.email="ayshaikh@iu.edu" \
    commit -m "Eval results from BigRed200 (results_fixed snapshot, job ${SLURM_JOB_ID:-manual})"
GIT_SSH_COMMAND="$GITSSH" git push -u origin "$BRANCH"
echo "PUBLISHED branch $BRANCH"
cd /
rm -rf "$WORK"
