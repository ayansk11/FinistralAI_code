#!/bin/bash
# =============================================================================
# TARBALL lane — run on a Quartz login node:
#
#   curl -sL https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook/tarball_lane_quartz.sh | bash
#
# Builds the venv at /tmp/finvenv on the LOGIN node (local disk, no quota,
# no Lustre), packs it into ONE tarball on scratch, and submits the eval
# with VENV_TARBALL so the compute node untars it to its own /tmp — turning
# every Python import into a node-local read. Single-large-file scratch
# reads are the one access pattern proven reliable throughout this outage.
# =============================================================================
set -euo pipefail

REPO=/N/scratch/ayshaikh/FinistralAI_code
RAW=https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook
TARBALL=/N/scratch/ayshaikh/finvenv.tar.gz

if [ ! -s "$TARBALL" ]; then
  echo "== building venv at /tmp/finvenv on login node (~10 min) =="
  module load python/gpu/3.11.5
  rm -rf /tmp/finvenv
  python3 -m venv /tmp/finvenv
  # shellcheck disable=SC1091
  source /tmp/finvenv/bin/activate
  pip install -q -U pip wheel setuptools
  pip install -q torch==2.4.1
  pip install -q transformers==4.44.2 peft==0.11.1 accelerate==0.33.0 \
      bitsandbytes==0.43.3 datasets==2.20.0 scikit-learn pandas tqdm sentencepiece
  python -c "import torch, transformers; print('VENV OK', torch.__version__)"
  deactivate
  echo "== packing tarball to scratch =="
  tar -czf "$TARBALL" -C /tmp finvenv
  ls -lh "$TARBALL"
else
  echo "== tarball already exists: $(ls -lh "$TARBALL" | awk '{print $5}') =="
fi

cd "$REPO"
mkdir -p logs/slurm
curl -sL "$RAW/run_evals_bigred200.sh" > revision/runbook/run_evals_bigred200.sh
curl -sL "$RAW/publish_results_job.sh" > /N/scratch/ayshaikh/publish_results_job.sh

EVAL_JID=$(sbatch --parsable --partition=h100-single --qos=allocated --time=02:00:00 \
  --export=ALL,PY_MODULE=python/gpu/3.11.5,VENV_TARBALL="$TARBALL" \
  revision/runbook/run_evals_bigred200.sh)
echo "eval job:    $EVAL_JID (h100-single, tarball venv)"

PUB_JID=$(sbatch --parsable --partition=h100-single --qos=allocated \
  --dependency=afterany:"$EVAL_JID" /N/scratch/ayshaikh/publish_results_job.sh)
echo "publish job: $PUB_JID (afterany:$EVAL_JID)"
squeue -u ayshaikh -o '%.10i %.20j %.8T %.12R' | grep -i finistral
