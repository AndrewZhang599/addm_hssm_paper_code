#!/bin/bash
#SBATCH --job-name=addm_hier
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#
# Hierarchical aDDM for ONE monkey.  Sessions are random intercepts on
# eta, kappa, a, b, x0.  Submit twice -- once per monkey -- to fit both;
# there is no code path that pools the two monkeys into one model.
#
# Everything is driven by environment variables so one file covers the smoke
# test and the production run:
#
#   MONKEY_DIR   required. .../two_monkey_42_session/monkey_c  (or monkey_k)
#   N_SESSIONS   fit only the first N sessions          (default: all)
#   SESSIONS     explicit space-separated session names (overrides N_SESSIONS)
#   OUT_DIR      where the .nc lands  (default: /oscar/scratch/$USER/addm_hier)
#   DRAWS TUNE CHAINS CORES TARGET_ACCEPT SEED
#   CHAIN_METHOD auto|parallel|vectorized|sequential   (default: auto)
#   GPUS         convenience: same as passing --gres=gpu:N to sbatch (default 1)
#   PARAMS       which params get (1|session)  (default: eta kappa a b x0)
#   TAG          extra string in the output filename
#   DRY_RUN=1    build + check + time a gradient, then stop
#   OVERWRITE=1  replace an existing .nc
#   PYTHON       interpreter (default: the HSSM checkout venv, which is the only
#                one carrying hssm>=0.4 with hssm.addm -- the conda envs are 0.2.x)
#
# --- GPU quota: which partition to submit to ---------------------------------
#
#   partition  account                  QoS             GPUs/user
#   gpu        (default)                norm-gpu                2   <- default here
#   gpu-he     carney-frankmj-condo2    carney-gcondo2         75
#
# The default `gpu` partition caps you at TWO GPUs in total, so two 2-GPU jobs
# (one per monkey) will serialize. To run both monkeys concurrently at 2 GPUs
# each, submit against the lab condo:
#
#   sbatch --partition=gpu-he --account=carney-frankmj-condo2 --gres=gpu:2 \
#     --export=ALL,MONKEY_DIR=$MD/monkey_c,CHAINS=2 monkey_hier_addm.sbatch
#
# Smoke test (3 sessions, tiny):
#   sbatch --time=1:00:00 \
#     --export=ALL,MONKEY_DIR=$MD/monkey_c,N_SESSIONS=3,DRAWS=50,TUNE=50,CHAINS=1 \
#     monkey_hier_addm.sbatch
#
# Production, both monkeys (two independent jobs):
#   for m in monkey_c monkey_k; do
#     sbatch --export=ALL,MONKEY_DIR=$MD/$m monkey_hier_addm.sbatch
#   done
#
# --- running 2 MCMC chains at the same time ---------------------------------
#
# numpyro's chain_method decides this, and pymc 6.2 hides it (see the shim in
# monkey_hier_fit.force_chain_method).  Two ways to get concurrent chains:
#
#   (a) TWO GPUs, one chain each ("parallel").  Chains run on separate devices
#       and coordinate over NCCL:
#         sbatch --gres=gpu:2 \
#           --export=ALL,MONKEY_DIR=$MD/monkey_c,CHAINS=2,CHAIN_METHOD=parallel \
#           monkey_hier_addm.sbatch
#       This script sets NCCL_P2P_DISABLE=1 automatically whenever it sees more
#       than one GPU -- without it, sampling stalls at 100% on Oscar.
#
#   (b) ONE GPU, both chains vmapped onto it ("vectorized").  No NCCL, so no
#       stall risk, and the A6000 is far from saturated at ~10k trials:
#         sbatch --export=ALL,MONKEY_DIR=$MD/monkey_c,CHAINS=2,CHAIN_METHOD=vectorized \
#           monkey_hier_addm.sbatch
#
# CHAIN_METHOD=auto (the default) picks (a) when the allocation has at least as
# many GPUs as chains, and (b) otherwise.  So `--gres=gpu:2 CHAINS=2` just works,
# and a 1-GPU job stops silently degrading to sequential.

set -euo pipefail

SCRIPT_DIR="/users/azhan378/data/azhang/addm_hssm_paper_code/scripts"
PY="${PYTHON:-/oscar/data/frankmj/azhang/HSSMSpine/repos/HSSM/.venv/bin/python}"

if [[ ! -x "$PY" ]]; then
    echo "Interpreter not found or not executable: $PY" >&2
    exit 2
fi

if [[ -z "${MONKEY_DIR:-}" ]]; then
    echo "MONKEY_DIR is required, e.g. --export=ALL,MONKEY_DIR=.../monkey_c" >&2
    exit 2
fi
MONKEY_DIR="${MONKEY_DIR%/}"
MONKEY="$(basename "$MONKEY_DIR")"

OUT_DIR="${OUT_DIR:-/oscar/scratch/${USER}/addm_hier}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# GPUS is a convenience for people who forget that #SBATCH --gres cannot read an
# env var.  If it asks for more GPUs than this allocation actually has, resubmit
# ourselves with the right --gres rather than silently running under-provisioned.
if [[ -n "${GPUS:-}" && -n "${SLURM_JOB_ID:-}" ]]; then
    have=$(echo "${CUDA_VISIBLE_DEVICES:-}" | awk -F, '{print NF}')
    if [[ "$GPUS" -gt "${have:-0}" ]]; then
        echo "GPUS=$GPUS but this allocation has ${have:-0}; resubmitting with --gres=gpu:$GPUS"
        exec sbatch --gres=gpu:"$GPUS" --export=ALL,GPUS= "$0"
    fi
fi

# --output/--error are /dev/null above so a single log file can carry a name
# built from MONKEY, which Slurm cannot interpolate into #SBATCH directives.
JOB_ID="${SLURM_JOB_ID:-local$$}"
LOG="${LOG_DIR}/${MONKEY}_${JOB_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=== $(date -Is)  job ${JOB_ID}  ${MONKEY}  on $(hostname) ==="
echo "monkey_dir : $MONKEY_DIR"
echo "out_dir    : $OUT_DIR"
echo "log        : $LOG"

# Slurm has already set CUDA_VISIBLE_DEVICES from --gres; leave it alone.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

N_GPU=$(echo "${CUDA_VISIBLE_DEVICES:-}" | awk -F, '{print ($0=="" ? 0 : NF)}')
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}  (${N_GPU} GPU(s))"

# Multi-GPU sampling coordinates over NCCL, which is where HSSM+numpyro has been
# seen to stall at 100% on Oscar. Peer-to-peer transport is the culprit; disable
# it whenever more than one device is in play. (monkey_hier_fit.py sets this too,
# belt-and-braces, since it must land before jax initialises.)
if [[ "$N_GPU" -gt 1 ]]; then
    export NCCL_P2P_DISABLE=1
    echo "NCCL_P2P_DISABLE=1  (set because ${N_GPU} GPUs are visible)"
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

ARGS=(--monkey-dir "$MONKEY_DIR" --out-dir "$OUT_DIR")
[[ -n "${SESSIONS:-}"      ]] && ARGS+=(--sessions ${SESSIONS})
[[ -z "${SESSIONS:-}" && -n "${N_SESSIONS:-}" ]] && ARGS+=(--n-sessions "$N_SESSIONS")
[[ -n "${DRAWS:-}"         ]] && ARGS+=(--draws "$DRAWS")
[[ -n "${TUNE:-}"          ]] && ARGS+=(--tune "$TUNE")
[[ -n "${CHAINS:-}"        ]] && ARGS+=(--chains "$CHAINS")
[[ -n "${CORES:-}"         ]] && ARGS+=(--cores "$CORES")
[[ -n "${TARGET_ACCEPT:-}" ]] && ARGS+=(--target-accept "$TARGET_ACCEPT")
[[ -n "${SEED:-}"          ]] && ARGS+=(--seed "$SEED")
[[ -n "${PARAMS:-}"        ]] && ARGS+=(--params ${PARAMS})
[[ -n "${TAG:-}"           ]] && ARGS+=(--tag "$TAG")
[[ -n "${CHAIN_METHOD:-}"  ]] && ARGS+=(--chain-method "$CHAIN_METHOD")
[[ "${DRY_RUN:-0}"   == "1" ]] && ARGS+=(--dry-run)
[[ "${OVERWRITE:-0}" == "1" ]] && ARGS+=(--overwrite)

echo "cmd        : $PY ${SCRIPT_DIR}/monkey_hier_fit.py ${ARGS[*]}"
echo "---"

srun "$PY" "${SCRIPT_DIR}/monkey_hier_fit.py" "${ARGS[@]}"
rc=$?

echo "--- $(date -Is)  exit ${rc} ---"
ls -lh "$OUT_DIR"/*.nc 2>/dev/null || true
exit $rc
