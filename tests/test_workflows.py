"""Check county batching, package training, and portable batch launchers."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import lightning
import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin
from torch.utils.data import DataLoader, Dataset

from chesapeakersc.datamodules import ChesapeakeRSCv2DataModule
from chesapeakersc.modules import ChesapeakeRSCv2Task

ROOT = Path(__file__).resolve().parents[1]
BATCH_SCRIPTS = [*ROOT.glob("scripts/*.sbatch"), *ROOT.glob("dataset/scripts/*.sbatch")]


class LocalSamples(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


def test_county_index_cli_shards(tmp_path):
    labels = tmp_path / "labels"
    labels.mkdir()
    inventory = tmp_path / "inventory.csv"
    names = [
        "balt_24005_lulc_2021_2024-Edition.tif",
        "alle_24001_lulc_2021_2024-Edition.tif",
    ]
    with inventory.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=["name", "state", "year"])
        writer.writeheader()
        for name in names:
            writer.writerow({"name": name, "state": "Maryland", "year": "2021"})
            with rasterio.open(
                labels / name,
                "w",
                driver="GTiff",
                count=1,
                width=32,
                height=32,
                dtype="uint8",
                crs="EPSG:32618",
                transform=from_origin(500000, 4500000, 1, 1),
            ) as dst:
                dst.write(np.full((32, 32), 20, dtype="uint8"), 1)
    output = tmp_path / "indexes"
    for shard in range(2):
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "dataset/index_label_tiles.py"),
                "--labels_dir",
                str(labels),
                "--inventory",
                str(inventory),
                "--out",
                str(output),
                "--shard",
                str(shard),
                "--nshards",
                "2",
            ],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        )
        assert len(list(output.glob("*.parquet"))) == shard + 1
    for name in names:
        rows = gpd.read_parquet(output / Path(name).with_suffix(".parquet"))
        assert rows.c20.tolist() == [1024]
        assert rows.n_valid.tolist() == [1024]
        assert rows.state_abbr.tolist() == ["md"]


@pytest.mark.parametrize(
    "script", BATCH_SCRIPTS, ids=lambda p: str(p.relative_to(ROOT))
)
def test_batch_launcher_arguments(script, tmp_path):
    """Intercept uv so each job can be checked without submitting or running it."""
    binary = tmp_path / "bin"
    binary.mkdir()
    fake_uv = binary / "uv"
    fake_uv.write_text(
        f"#!{sys.executable}\nimport json, os, sys\nprint(json.dumps([os.getcwd(), sys.argv[1:]]))\n"
    )
    fake_uv.chmod(0o755)
    data = tmp_path / "data with spaces"
    env = dict(
        os.environ,
        PATH=f"{binary}:{os.environ['PATH']}",
        REPO_ROOT=str(ROOT),
        DATA_ROOT=str(data),
        NAME="test",
        CKPT="/tmp/checkpoint with spaces.ckpt",
        SLURM_ARRAY_TASK_ID="1",
        NSHARDS="2",
    )
    result = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        cwd, args = json.loads(line)
        assert cwd == str(ROOT)
        assert args[:4] == ["run", "--frozen", "--no-sync", "python"]
        assert (ROOT / args[4]).is_file()
        assert any(str(data) in arg for arg in args)
        if "--checkpoint" in args:
            assert args[args.index("--checkpoint") + 1] == env["CKPT"]
        if "--shard" in args:
            assert args[args.index("--shard") + 1] == "1"
            assert args[args.index("--nshards") + 1] == "2"


def test_model_training_step_and_normalization(tmp_path):
    torch.set_num_threads(2)
    task = ChesapeakeRSCv2Task(
        model="unet",
        backbone="resnet18",
        weights=False,
        in_channels=3,
        num_classes=2,
        num_filters=64,
        loss="ce",
        ignore_index=255,
        lr=1e-3,
        tmax=1,
        class_weights=None,
    )
    image = torch.full((3, 64, 64), 255.0)
    mask = torch.zeros((64, 64), dtype=torch.long)
    mask[20:40] = 1
    mask[:2] = 255
    dm = ChesapeakeRSCv2DataModule()
    normalized = dm.aug({"image": image[None], "mask": mask[None]})
    torch.testing.assert_close(normalized["image"], torch.ones_like(image[None]))
    torch.testing.assert_close(normalized["mask"], mask[None])
    samples = [{"image": normalized["image"][0], "mask": mask} for _ in range(2)]
    loader = DataLoader(LocalSamples(samples), batch_size=2)
    trainer = lightning.Trainer(
        accelerator="cpu",
        devices=1,
        fast_dev_run=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(task, train_dataloaders=loader, val_dataloaders=loader)
    assert trainer.global_step == 1
    assert torch.isfinite(trainer.callback_metrics["train_loss"])


def test_invalid_cutout_raises_value_error():
    with pytest.raises(ValueError, match="Unknown cutout"):
        ChesapeakeRSCv2DataModule(cutout="unknown")
