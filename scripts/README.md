# Training and evaluation on Slurm

Submit jobs from the repository root. The scripts use its uv environment and accept the data directory through `DATA_ROOT`. Set your cluster's account and partition with `sbatch` options.

```sh
uv sync --locked
mkdir -p logs
export DATA_ROOT=/absolute/path/to/chesapeakersc-v2
NAME=core_rgb BANDS=rgb EPOCHS=100 \
  sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/train.sbatch
```

| Script | Required variables | Purpose |
| --- | --- | --- |
| `train.sbatch` | `DATA_ROOT`, `NAME` | Train a new model |
| `train_resume.sbatch` | `DATA_ROOT`, `NAME`, `CKPT` | Start from checkpoint weights with a new optimizer and schedule |
| `dwr_eval.sbatch` | `DATA_ROOT`, `CKPT` | Measure road and canopy recall, including distance-weighted recall |

All three accept `DATASET`, `BANDS`, `BATCH_SIZE`, and `WORKERS`. Training also accepts `EPOCHS`, `MODEL`, and `BACKBONE`; evaluation accepts `SPLIT`. Additional Python arguments can follow the batch-script path.

```sh
CKPT=/absolute/path/to/model.ckpt BANDS=rgb SPLIT=test_de \
  sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION scripts/dwr_eval.sbatch
```

The scripts request one GPU and write logs under `logs/` in the submission directory. Create that directory before submitting. If submitting from elsewhere, set `REPO_ROOT` to the checkout and pass absolute `--output` and `--error` paths. Jobs use `uv run --no-sync`, so install dependencies before submission.

Dataset jobs are documented in [`dataset/scripts/`](../dataset/scripts/README.md).
