"""Check relocated commands and model imports without external data."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from chesapeakersc.datamodules import ChesapeakeRSCv2DataModule
from chesapeakersc.modules import ChesapeakeRSCv2Task

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [
    *sorted((ROOT / "dataset").glob("*.py")),
    *sorted((ROOT / "release").glob("*.py")),
    ROOT / "train.py",
    ROOT / "dwr_eval.py",
]


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_cli_help_from_other_directory(script, tmp_path):
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_default_inventory_from_other_directory(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "dataset/download_labels.py"),
            "--states",
            "NoSuchState",
            "--out_dir",
            str(tmp_path / "labels"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "0 files" in result.stdout


def test_manifest_rejects_incomplete_legacy_arguments(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "dataset/build_manifest.py"),
            "--meta",
            "unused",
            "--splits",
            "unused",
            "--patches_root",
            "unused",
            "--legacy_root",
            "unused",
            "--out",
            "unused",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "must be provided together" in result.stderr


def test_renamed_model_package_forward():
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
        tmax=50,
        class_weights=None,
    ).eval()
    with torch.no_grad():
        output = task(torch.zeros(1, 3, 64, 64))
    assert output.shape == (1, 2, 64, 64)
    assert ChesapeakeRSCv2DataModule.mean == 0
    assert ChesapeakeRSCv2DataModule.std == 255
