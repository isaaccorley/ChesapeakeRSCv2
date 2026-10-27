#!/bin/bash
#SBATCH --job-name=rsc-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gpus-per-node=1

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset directory}"
: "${CKPT:?Set CKPT to an existing checkpoint}"
uv run --frozen --no-sync python dwr_eval.py \
  --manifest "$DATA_ROOT/index/manifest.parquet" --checkpoint "$CKPT" \
  --dataset "${DATASET:-core}" --split "${SPLIT:-test}" --bands "${BANDS:-rgb}" \
  --batch_size "${BATCH_SIZE:-16}" --num_workers "${WORKERS:-8}" "$@"
