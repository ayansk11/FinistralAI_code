#!/bin/bash
# =============================================================================
# Manual results publisher — run directly on a login node (BR200 or Quartz):
#
#   curl -sL https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook/publish_now.sh | bash
#
# Clones to the node-local /tmp — NOT scratch — because git object writes on
# /N/scratch hit "Stale file handle" (ESTALE) errors (observed in job
# 8025162's log; the SSH auth and clone were fine, git add was not). Results
# are copied from scratch with plain reads, which have been reliable.
# =============================================================================
set -euo pipefail

SRC=/N/scratch/ayshaikh/FinistralAI_code
GITSSH="ssh -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/N/scratch/ayshaikh/.known_hosts"
WORK="/tmp/finpub-$$"
BRANCH="results-run-$(hostname -s)-$(date +%s)"

echo "== results inventory on scratch =="
ls "$SRC"/results_fixed/*_predictions.csv 2>/dev/null | wc -l
ls -la "$SRC"/results_fixed/latency.json 2>/dev/null || echo "(no latency.json)"

rm -rf "$WORK"
GIT_SSH_COMMAND="$GITSSH" git clone --depth 1 git@github.com:ayansk11/FinistralAI_code.git "$WORK"
cd "$WORK"
git checkout -b "$BRANCH"
rm -rf results_fixed
cp -r "$SRC/results_fixed" results_fixed
mkdir -p results_fixed/slurm_logs
cp "$SRC"/logs/slurm/finistral-eval-*.out results_fixed/slurm_logs/ 2>/dev/null || true

git add -f results_fixed
git -c user.name="Ayan Javeed Shaikh" -c user.email="ayshaikh@iu.edu" \
    commit -m "Eval results snapshot ($BRANCH)"
GIT_SSH_COMMAND="$GITSSH" git push -u origin "$BRANCH"
echo "PUBLISHED $BRANCH"
cd /
rm -rf "$WORK"
