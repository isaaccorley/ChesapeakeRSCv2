#!/bin/bash
#SBATCH --job-name=rsc-warm-start
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --gpus-per-node=1

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset directory}"
: "${NAME:?Set NAME to the new experiment name}"
: "${CKPT:?Set CKPT to an existing checkpoint}"
uv run --frozen --no-sync python train.py \
  --manifest "$DATA_ROOT/index/manifest.parquet" --experiment_name "$NAME" \
  --dataset "${DATASET:-core}" --bands "${BANDS:-rgb}" \
  --num_epochs "${EPOCHS:-50}" --tmax "${EPOCHS:-50}" \
  --batch_size "${BATCH_SIZE:-16}" --num_workers "${WORKERS:-8}" \
  --model "${MODEL:-unet}" --backbone "${BACKBONE:-resnet18}" \
  --resume_from "$CKPT" "$@"
