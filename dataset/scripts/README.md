# Dataset jobs on Slurm

Install the dataset environment and submit from the repository root:

```sh
uv sync --locked --group dataset
mkdir -p logs
export DATA_ROOT=/absolute/path/to/chesapeakersc-v2
sbatch --account=YOUR_ACCOUNT --partition=YOUR_CPU_PARTITION dataset/scripts/download_sources.sbatch
```

| Script | Operation |
| --- | --- |
| `download_sources.sbatch` | Download core labels and catalog NAIP imagery |
| `download_legacy_sources.sbatch` | Download Maryland 2018 inputs for the optional legacy subset |
| `index_label_tiles.sbatch` | Index county label rasters |
| `extract_patches.sbatch` | Extract aligned image and label patches |
| `create_targets.sbatch` | Convert land-cover labels to training targets |

Indexing, extraction, and target creation support zero-based contiguous arrays. Set `NSHARDS` to the number of array tasks:

```sh
NSHARDS=8 sbatch --array=0-7 --account=YOUR_ACCOUNT \
  --partition=YOUR_CPU_PARTITION dataset/scripts/index_label_tiles.sbatch
```

Use the same pattern for extraction and target creation. Submit each stage only after its inputs are complete; matching, sampling, quality checks, and splits use the commands in the [dataset guide](../README.md).

`WORKERS` controls concurrent downloads or patch processing. Extraction also accepts `LIMIT`, `PATCHES`, `NAIP`, `LABELS`, and `OUT`; target creation accepts `PATCHES_ROOT`. Path overrides should be absolute. The scripts use the installed environment without syncing it on compute nodes. Set `REPO_ROOT` if the submission directory is not the checkout, and adjust log destinations with `sbatch --output` and `--error`.
