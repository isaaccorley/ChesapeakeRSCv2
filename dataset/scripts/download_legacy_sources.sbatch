#!/bin/bash
#SBATCH --job-name=rsc-download-legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset directory}"
mkdir -p "$DATA_ROOT/index" "$DATA_ROOT/labels/t2"
uv run --frozen --no-sync python dataset/catalog_naip.py --pairs md:2018 --out "$DATA_ROOT/index/naip_items_md2018.parquet"
uv run --frozen --no-sync python dataset/download_labels.py --years 2018 --states Maryland --out_dir "$DATA_ROOT/labels/t2" --workers "${WORKERS:-4}"
