#!/bin/bash
#SBATCH --job-name=rsc-extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset directory}"
uv run --frozen --no-sync python dataset/extract_patches.py \
  --patches "${PATCHES:-$DATA_ROOT/index/patches_v2.parquet}" \
  --naip "${NAIP:-$DATA_ROOT/index/naip_items.parquet}" \
  --labels_dir "${LABELS:-$DATA_ROOT/labels/t3}" --out "${OUT:-$DATA_ROOT/patches}" \
  --shard "${SLURM_ARRAY_TASK_ID:-0}" --nshards "${NSHARDS:-1}" \
  --workers "${WORKERS:-8}" --limit "${LIMIT:-0}"
