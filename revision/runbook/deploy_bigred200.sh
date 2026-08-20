#!/bin/bash
# ============================================================
# Mac-side driver for the BigRed200 eval — deploy / setup / submit /
# status / fetch. Uses the `bigred200` alias from ~/.ssh/config
# (ControlMaster auto, ControlPersist 4h): open ONE interactive
# `ssh bigred200` first (Duo), then every command here rides that
# socket with no further auth.
#
#   bash revision/runbook/deploy_bigred200.sh deploy   # rsync repo -> /N/scratch
#   bash revision/runbook/deploy_bigred200.sh setup    # build venv (login node, ~10 min)
#   bash revision/runbook/deploy_bigred200.sh submit   # sbatch the eval sweep
#   bash revision/runbook/deploy_bigred200.sh status   # squeue + tail of latest log
#   bash revision/runbook/deploy_bigred200.sh fetch    # pull results_fixed.tar.gz + unpack
# ============================================================
set -euo pipefail

HOST=bigred200
REMOTE_DIR=/N/scratch/ayshaikh/FinistralAI_code
LOCAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

case "${1:-}" in
  deploy)
    rsync -av --delete \
      --exclude .git --exclude __pycache__ --exclude results_smoke \
      --exclude '*.pdf' --exclude '*.docx' --exclude '*.ipynb' \
      --exclude '.claude*' --exclude '.DS_Store' --exclude '~$*' \
      --exclude results_fixed --exclude 'results_fixed.tar.gz' \
      "$LOCAL_DIR/" "$HOST:$REMOTE_DIR/"
    ssh "$HOST" "ls $REMOTE_DIR/data_eval/"
    ;;
  setup)
    ssh "$HOST" "cd $REMOTE_DIR && bash revision/runbook/setup_venv_bigred200.sh"
    ;;
  submit)
    ssh "$HOST" "cd $REMOTE_DIR && mkdir -p logs/slurm && \
      sbatch --partition=gpu --qos=allocated revision/runbook/run_evals_bigred200.sh"
    ;;
  status)
    ssh "$HOST" "squeue -u ayshaikh -o '%.10i %.20j %.8T %.10P %.10M %.12R'; \
      cd $REMOTE_DIR && ls -t logs/slurm/ 2>/dev/null | head -3 && \
      tail -n 25 \$(ls -t logs/slurm/finistral-eval-*.out 2>/dev/null | head -1) 2>/dev/null || true"
    ;;
  fetch)
    scp "$HOST:$REMOTE_DIR/results_fixed.tar.gz" "$LOCAL_DIR/"
    tar xzf "$LOCAL_DIR/results_fixed.tar.gz" -C "$LOCAL_DIR"
    echo "results_fixed/ unpacked:"
    ls "$LOCAL_DIR/results_fixed/" | head
    ;;
  *)
    echo "usage: $0 {deploy|setup|submit|status|fetch}" >&2
    exit 2
    ;;
esac
