#!/bin/bash
#SBATCH --job-name=rsc-index-labels
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=6G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset directory}"
uv run --frozen --no-sync python dataset/index_label_tiles.py \
  --labels_dir "$DATA_ROOT/labels/t3" --out "$DATA_ROOT/index/tiles" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "${NSHARDS:-1}"
