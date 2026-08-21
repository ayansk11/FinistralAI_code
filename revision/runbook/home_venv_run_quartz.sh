#!/bin/bash
# =============================================================================
# HOME-VENV lane — run on a Quartz login node:
#
#   curl -sL https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook/home_venv_run_quartz.sh | bash
#
# Rationale (evidence-based): for ~10h, fresh Python imports from /N/scratch
# have hung on EVERY compute node tried (A100/H100/V100, both clusters), while
# the user's LLMRL jobs — whose vllm imports come from HOME NFS (~/.local) —
# started and ran fine during the same window. Large-file reads from scratch
# also work (those jobs stream checkpoints from scratch). Conclusion: the
# pathology is specific to scratch's metadata/mmap path for many-small-file
# Python imports. So: build the venv on Quartz HOME (which has space, unlike
# the over-quota BR200 home), keep HF model cache on scratch, and run.
# =============================================================================
set -euo pipefail

REPO=/N/scratch/ayshaikh/FinistralAI_code
RAW=https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook
VENV="$HOME/venv-finistral-home"

# Home space guard: need ~6GB for torch + HF stack.
AVAIL_KB=$(df -k "$HOME" | awk 'NR==2 {print $4}')
if [ "${AVAIL_KB:-0}" -lt 8000000 ]; then
  echo "FATAL: <8GB free in $HOME (${AVAIL_KB}KB) -- clear space first"; exit 1
fi

if [ ! -f "$VENV/bin/activate" ]; then
  echo "== building home venv (~10 min) =="
  module load python/gpu/3.11.5
  python3 -m venv "$VENV"
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
  pip install -q -U pip wheel setuptools
  pip install -q torch==2.4.1
  pip install -q transformers==4.44.2 peft==0.11.1 accelerate==0.33.0 \
      bitsandbytes==0.43.3 datasets==2.20.0 scikit-learn pandas tqdm sentencepiece
  python -c "import torch, transformers; print('HOME VENV OK', torch.__version__)"
else
  echo "== home venv already exists =="
fi

cd "$REPO"
mkdir -p logs/slurm
curl -sL "$RAW/run_evals_bigred200.sh" > revision/runbook/run_evals_bigred200.sh
curl -sL "$RAW/publish_results_job.sh" > /N/scratch/ayshaikh/publish_results_job.sh

EVAL_JID=$(sbatch --parsable --partition=h100-single --qos=allocated --time=02:00:00 \
  --export=ALL,PY_MODULE=python/gpu/3.11.5,VENV="$VENV" \
  revision/runbook/run_evals_bigred200.sh)
echo "eval job:    $EVAL_JID (h100-single, HOME venv)"

PUB_JID=$(sbatch --parsable --partition=h100-single --qos=allocated \
  --dependency=afterany:"$EVAL_JID" /N/scratch/ayshaikh/publish_results_job.sh)
echo "publish job: $PUB_JID (afterany:$EVAL_JID)"
squeue -u ayshaikh -o '%.10i %.20j %.8T %.12R' | grep -i finistral
