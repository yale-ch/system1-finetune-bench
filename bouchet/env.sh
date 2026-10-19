# Sourced by every job (and by bin/bouchet on the login node). Paths for Bouchet; override any of them
# in bouchet/env.local.sh (not in git).
export S1_HOME=${S1_HOME:-/nfs/roberts/pi/pi_rs2668/system1-finetune-bench}
export HF_HOME=${S1_HF_HOME:-$S1_HOME/hf_cache}   # not ${HF_HOME:-...}: ~/.bashrc points HF_HOME at the nearly full project fileset
export UV_CACHE_DIR=${UV_CACHE_DIR:-$S1_HOME/uv_cache}
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# no hosted model APIs in this workflow: fail loudly if anything tries
export GEMINI_API_KEY= GOOGLE_API_KEY= TYPESAFE_API_KEY= JEV_API_KEY= OPENAI_API_KEY= ANTHROPIC_API_KEY=
if command -v module >/dev/null 2>&1; then module purge >/dev/null 2>&1; module load CUDA/12.8.0 >/dev/null 2>&1 || true; fi
if [ -f "$(dirname "${BASH_SOURCE[0]}")/env.local.sh" ]; then source "$(dirname "${BASH_SOURCE[0]}")/env.local.sh"; fi
true
