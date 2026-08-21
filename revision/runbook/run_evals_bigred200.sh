#!/bin/bash
# =============================================================================
# JICTASA-2026-018 corrected evaluation sweep — IU BigRed200 (A100, Slurm).
#
# Submit from the repo root on the cluster (partition/qos at submit time, per
# the account's convention — Quartz rejects qos-as-directive):
#
#   BigRed200 A100:  sbatch --partition=gpu --qos=allocated \
#                      revision/runbook/run_evals_bigred200.sh
#   Quartz H100:     sbatch --partition=hopper --qos=hopper \
#                      --export=ALL,PY_MODULE=python/gpu/3.11.5,VENV=/N/scratch/ayshaikh/venv-finistral-qz \
#                      revision/runbook/run_evals_bigred200.sh
#
# One-time prep:
#   1. bash revision/runbook/setup_venv_bigred200.sh       (login node, ~10 min)
#   2. HF read token in ~/.hf_token (chmod 600) — gated meta-llama repos
#      require the licenses accepted on huggingface.co first.
#
# Resumable: finished (model, dataset) pairs are skipped on resubmit.
# Expected wall time on one A100: ~2-3 h for the full 8x3 sweep.
# =============================================================================
#SBATCH --job-name=finistral-eval
#SBATCH --account=r01510
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/finistral-eval-%j.out
#SBATCH --error=logs/slurm/finistral-eval-%j.err

set -uo pipefail

# --- Environment (BigRed200 is Cray/SLES: cray-python; Quartz overrides via
#     --export=ALL,PY_MODULE=python/gpu/3.11.5) ---
module load "${PY_MODULE:-cray-python/3.11.7}"
VENV="${VENV:-/N/scratch/ayshaikh/venv-finistral}"
# shellcheck disable=SC1090
source "$VENV/bin/activate"

# All caches on scratch — never the home NFS (quota + EIO history there).
export HF_HOME=/N/scratch/ayshaikh/hf_cache
export TMPDIR=/N/scratch/ayshaikh/tmp
mkdir -p "$HF_HOME" "$TMPDIR"

# Token lookup order: env var, scratch, then home. The home NFS is over
# quota (writes fail with EDQUOT), so /N/scratch/ayshaikh/.hf_token is the
# canonical location.
for f in "${HF_TOKEN_FILE:-}" /N/scratch/ayshaikh/.hf_token "$HOME/.hf_token"; do
  if [ -z "${HF_TOKEN:-}" ] && [ -n "$f" ] && [ -s "$f" ]; then
    HF_TOKEN="$(tr -d '[:space:]' < "$f")"
    export HF_TOKEN
  fi
done
case "${HF_TOKEN:-}" in
  hf_????????????????*) : ;;  # plausible token
  *) echo "No usable HF token: put a read token in /N/scratch/ayshaikh/.hf_token" >&2
     exit 1 ;;
esac

cd "${SLURM_SUBMIT_DIR:-$PWD}"
test -f data_eval/fpb_decontam.csv || { echo "data_eval/ missing — run deploy first"; exit 1; }
mkdir -p logs/slurm

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# NOTE: node-local staging (whole venv, then torch-only) was tried and
# REJECTED -- both hung on `cp -r` itself, worse than the original problem
# (Lustre's metadata server appears to be the real bottleneck for
# many-small-file trees; torch's own package has thousands of files
# including bundled C++ headers, not just a few large binaries). Plain,
# unmodified execution from the network venv has demonstrably worked
# (every first attempt on both clusters succeeded and did real multi-model
# inference), so we do that -- just fail fast if this attempt hits
# unhealthy filesystem state, rather than hang silently for hours.
timeout 180 python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
  || { echo "FATAL: torch import timed out/failed (possible stuck Lustre I/O on this node) -- resubmit."; exit 3; }

MODELS="finistral finistral_alpaca mistral_base fingpt_llama2 fingpt_llama3 fingpt_falcon fingpt_bloom finbert"
DATASETS="fpb_decontam fiqa tfns"

for M in $MODELS; do
  for D in $DATASETS; do
    echo "########## $M / $D ##########"
    # Per-combo cap (25 min): generous for real work (largest combo is ~2.4k
    # rows) but bounds a stuck-I/O or stuck-download combo so the loop moves
    # on instead of consuming the entire job's wall clock. A timed-out combo
    # simply has no predictions CSV yet, so the resume guard retries it on
    # the next submission.
    timeout 1500 python -u eval_harness_fixed.py --models "$M" --dataset "$D" \
      --quant none --seeds 0 --capture_scores \
      --batch_size 8 --max_new_tokens 8 --out_dir results_fixed \
      || echo "WARNING: $M/$D timed out or failed (exit $?) -- will retry on next resubmit."
  done
done

# --- Latency micro-benchmark (fills the manuscript's deployment claim) ---
python - <<'EOF'
import json, time, statistics, csv, importlib.util
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

spec = importlib.util.spec_from_file_location('eh', 'eval_harness_fixed.py')
eh = importlib.util.module_from_spec(spec); spec.loader.exec_module(eh)

tok = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1')
tok.padding_side = 'left'
if tok.pad_token is None:
    tok.pad_token = tok.unk_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-v0.1', torch_dtype=torch.bfloat16, device_map='auto')
model = PeftModel.from_pretrained(model, 'Ayansk11/Finistral-7B_lora')
model.eval()

sents = [r['sentence'] for r in csv.DictReader(open('data_eval/fpb_decontam.csv'))][:55]
spec_f = eh.MODELS['finistral']
times = []
for i, s in enumerate(sents):
    enc = tok(spec_f.template(spec_f.instruction, s), return_tensors='pt').to(model.device)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**enc, max_new_tokens=8, do_sample=False, num_beams=1,
                       pad_token_id=tok.pad_token_id)
    torch.cuda.synchronize()
    if i >= 5:
        times.append((time.perf_counter() - t0) * 1000)
res = {'gpu': torch.cuda.get_device_name(0), 'dtype': 'bfloat16', 'batch_size': 1,
       'max_new_tokens': 8, 'decoding': 'greedy', 'n': len(times),
       'median_ms': round(statistics.median(times), 1),
       'p95_ms': round(sorted(times)[int(0.95 * len(times))], 1)}
json.dump(res, open('results_fixed/latency.json', 'w'), indent=2)
print(res)
EOF

tar czf results_fixed.tar.gz results_fixed/
echo "Done. From the Mac:  bash revision/runbook/deploy_bigred200.sh fetch"
