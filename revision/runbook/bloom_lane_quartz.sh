#!/bin/bash
# =============================================================================
# BLOOM lane — run on a Quartz login node:
#
#   curl -sL https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook/bloom_lane_quartz.sh | bash
#
# Last missing measurements: fingpt_bloom x {fiqa, tfns}. Bloom hung at MODEL
# LOAD (exit 124) even with the tarball venv, because it is the one baseline
# stored as legacy pytorch .bin shards — a different many-small-read path
# that trips the same Lustre pathology. Fix: same single-large-file trick,
# applied to the weights — pack Bloom's HF cache dirs into one plain tar on
# scratch, untar to node-local /tmp in the job, point HF_HOME there.
# =============================================================================
set -euo pipefail

REPO=/N/scratch/ayshaikh/FinistralAI_code
RAW=https://raw.githubusercontent.com/ayansk11/FinistralAI_code/main/revision/runbook
CACHE=/N/scratch/ayshaikh/hf_cache
BLOOM_TAR=/N/scratch/ayshaikh/bloom_cache.tar

if [ ! -s "$BLOOM_TAR" ]; then
  echo "== packing Bloom cache dirs to one tar (weights don't compress; plain tar) =="
  tar -cf "$BLOOM_TAR" -C "$CACHE" \
    hub/models--bigscience--bloom-7b1 \
    hub/models--FinGPT--fingpt-mt_bloom-7b1_lora
  ls -lh "$BLOOM_TAR"
fi

cd "$REPO"
mkdir -p logs/slurm

cat > /N/scratch/ayshaikh/bloom_job.sh <<'JOB'
#!/bin/bash
#SBATCH --job-name=finistral-bloom
#SBATCH --account=r01510
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/N/scratch/ayshaikh/FinistralAI_code/logs/slurm/finistral-bloom-%j.out
set -uo pipefail
module load python/gpu/3.11.5

rm -rf /tmp/finvenv /tmp/hf_bloom
timeout 600 tar -xzf /N/scratch/ayshaikh/finvenv.tar.gz -C /tmp \
  || { echo "FATAL: venv untar failed"; exit 3; }
mkdir -p /tmp/hf_bloom
timeout 900 tar -xf /N/scratch/ayshaikh/bloom_cache.tar -C /tmp/hf_bloom \
  || { echo "FATAL: bloom cache untar failed"; exit 3; }
trap 'rm -rf /tmp/finvenv /tmp/hf_bloom' EXIT

source /tmp/finvenv/bin/activate
export HF_HOME=/tmp/hf_bloom
export TMPDIR=/N/scratch/ayshaikh/tmp

for f in /N/scratch/ayshaikh/.hf_token; do
  [ -s "$f" ] && HF_TOKEN="$(tr -d '[:space:]' < "$f")" && export HF_TOKEN
done

cd /N/scratch/ayshaikh/FinistralAI_code
timeout 180 python -c "import torch; print('torch OK', torch.cuda.is_available())" \
  || { echo "FATAL: torch import failed"; exit 3; }

for D in fiqa tfns; do
  echo "########## fingpt_bloom / $D ##########"
  timeout 2400 python -u eval_harness_fixed.py --models fingpt_bloom --dataset "$D" \
    --quant none --seeds 0 --capture_scores \
    --batch_size 8 --max_new_tokens 8 --out_dir results_fixed \
    || echo "WARNING: fingpt_bloom/$D failed (exit $?)"
done
JOB

curl -sL "$RAW/publish_results_job.sh" > /N/scratch/ayshaikh/publish_results_job.sh

EVAL_JID=$(sbatch --parsable --partition=h100-single --qos=allocated /N/scratch/ayshaikh/bloom_job.sh)
echo "bloom job:   $EVAL_JID (h100-single, local venv + local weights)"
PUB_JID=$(sbatch --parsable --partition=h100-single --qos=allocated \
  --dependency=afterany:"$EVAL_JID" /N/scratch/ayshaikh/publish_results_job.sh)
echo "publish job: $PUB_JID (afterany:$EVAL_JID)"
squeue -u ayshaikh -o '%.10i %.20j %.8T %.12R' | grep -i finistral
