#!/bin/bash
# ============================================================
# One-time venv build for the Finistral eval — IU BigRed200 LOGIN node.
#
# BigRed200 is Cray/SLES: use cray-python/3.11.7 (the RHEL8 python from
# Quartz crashes on libffi here — see hierarchical-planner-executor's
# setup_eval_bigred.sh for the precedent this follows).
#
# Run on a login node (pip only, no GPU), ~10 min:
#   bash revision/runbook/setup_venv_bigred200.sh
#
# Quartz variant: PY_MODULE=python/gpu/3.11.5 VENV=/N/scratch/ayshaikh/venv-finistral-qz \
#   bash revision/runbook/setup_venv_bigred200.sh
# ============================================================
set -uo pipefail

VENV="${VENV:-/N/scratch/ayshaikh/venv-finistral}"
export TMPDIR=/N/scratch/ayshaikh/tmp
mkdir -p "$TMPDIR"

module load "${PY_MODULE:-cray-python/3.11.7}"
echo "=== interpreter ==="; which python3; python3 --version
echo "VENV=$VENV"

python3 -m venv "$VENV"
# shellcheck disable=SC1090
source "$VENV/bin/activate"
python -m pip install -U pip wheel setuptools

# torch first (pulls the cu12 wheel), then the pinned HF stack.
pip install torch==2.4.1
pip install transformers==4.44.2 peft==0.11.1 accelerate==0.33.0 \
    bitsandbytes==0.43.3 datasets==2.20.0 scikit-learn pandas tqdm sentencepiece

echo ""
echo "=== critical import check ==="
python - <<'PY'
mods = ["torch", "transformers", "peft", "datasets", "sklearn", "pandas", "tqdm"]
ok = True
for m in mods:
    try:
        mod = __import__(m)
        print(f"  ok   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        ok = False
        print(f"  FAIL {m}: {e}")
import torch
print(f"  cuda wheels present: {torch.version.cuda}")
print(">>> FINISTRAL EVAL VENV READY" if ok else ">>> VENV INCOMPLETE — fix FAILs above")
PY

echo ""
echo ">>> venv: $VENV"
echo ">>> Next: put your HF read token in ~/.hf_token (chmod 600), then submit:"
echo "    sbatch --partition=gpu --qos=allocated-gpu revision/runbook/run_evals_bigred200.sh"
