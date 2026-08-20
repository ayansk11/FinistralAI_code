#!/bin/bash
# =============================================================================
# JICTASA-2026-018 corrected evaluation sweep — IU Big Red 200 (Slurm)
#
# Usage (from the repo root on the cluster, after one-time setup below):
#   sbatch revision/runbook/run_evals_bigred200.sh
#
# One-time setup:
#   1. Accept the meta-llama/Llama-2-7b-hf and meta-llama/Meta-Llama-3-8B
#      licenses on huggingface.co while logged in.
#   2. export HF_TOKEN=hf_...   (a READ token; or put it in ~/.bashrc)
#   3. python -m venv ~/finistral-env && source ~/finistral-env/bin/activate
#      pip install torch --index-url https://download.pytorch.org/whl/cu121
#      pip install transformers==4.44.2 peft==0.11.1 accelerate==0.33.0 \
#          bitsandbytes==0.43.3 datasets==2.20.0 scikit-learn pandas tqdm \
#          sentencepiece
#
# Runs are resumable: finished (model, dataset) combinations are skipped, so
# resubmitting the job after a timeout continues where it left off.
# =============================================================================
#SBATCH -J finistral-eval
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o finistral-eval-%j.out

set -euo pipefail

module load python 2>/dev/null || true
module load cudatoolkit 2>/dev/null || true
source ~/finistral-env/bin/activate

: "${HF_TOKEN:?Set HF_TOKEN to a HuggingFace read token (gated Llama repos)}"

cd "$SLURM_SUBMIT_DIR"
test -f data_eval/fpb_decontam.csv || { echo "data_eval/ missing"; exit 1; }

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

MODELS="finistral finistral_alpaca mistral_base fingpt_llama2 fingpt_llama3 fingpt_falcon fingpt_bloom finbert"
DATASETS="fpb_decontam fiqa tfns"

for M in $MODELS; do
  for D in $DATASETS; do
    echo "########## $M / $D ##########"
    python eval_harness_fixed.py --models "$M" --dataset "$D" \
      --quant none --seeds 0 --capture_scores \
      --batch_size 8 --max_new_tokens 8 --out_dir results_fixed
  done
done

# Latency micro-benchmark (see the Colab notebook cell 6 for the same logic).
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
echo "Done. Copy results_fixed.tar.gz back to the local repo root:"
echo "  scp <user>@bigred200.uits.iu.edu:$PWD/results_fixed.tar.gz ."
