#!/bin/bash
#SBATCH --job-name=rsc-targets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset directory}"
uv run --frozen --no-sync python dataset/create_targets.py \
  --root "${PATCHES_ROOT:-$DATA_ROOT/patches}" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "${NSHARDS:-1}" --workers "${WORKERS:-8}"
