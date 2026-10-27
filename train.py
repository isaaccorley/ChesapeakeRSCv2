"""Train a semantic segmentation model on ChesapeakeRSC v2.

Example:
  uv run python train.py --manifest /path/to/manifest.parquet --bands rgb \
      --model unet --backbone resnet18 --experiment_name core_rgb
"""

import argparse
import os
import warnings

import lightning
import torch

from chesapeakersc.datamodules import ChesapeakeRSCv2DataModule
from chesapeakersc.modules import ChesapeakeRSCv2Task

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
torch.set_float32_matmul_precision("medium")

IGNORE_INDEX = 255


def setup_argparse() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train a ChesapeakeRSC v2 segmentation model."
    )
    p.add_argument("--manifest", required=True, help="Path to index/manifest.parquet")
    p.add_argument("--dataset", choices=["core", "legacy", "both"], default="core")
    p.add_argument("--bands", choices=["rgb", "rgbn"], default="rgbn")
    p.add_argument(
        "--test_split", choices=["test", "test_de", "test_wv"], default="test"
    )
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument(
        "--model",
        choices=["unet", "unet++", "deeplabv3+", "upernet", "segformer", "fcn"],
        default="unet",
    )
    p.add_argument("--num_epochs", type=int, default=150)
    p.add_argument("--num_filters", type=int, default=64)
    p.add_argument("--backbone", default="resnet18")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--tmax", type=int, default=50)
    p.add_argument("--experiment_name", required=True)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument(
        "--cutout", choices=["small", "medium", "large", "multi"], default=None
    )
    p.add_argument("--differentiate_tree_canopy_over_roads", action="store_true")
    p.add_argument(
        "--resume_from",
        default=None,
        help=(
            "Load checkpoint weights with a new optimizer and --tmax-length cosine schedule."
        ),
    )
    return p


def main(args: argparse.Namespace) -> None:
    in_channels = 3 if args.bands == "rgb" else 4
    num_classes = 3 if args.differentiate_tree_canopy_over_roads else 2

    dm = ChesapeakeRSCv2DataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        differentiate_tree_canopy_over_roads=args.differentiate_tree_canopy_over_roads,
        cutout=args.cutout,
        test_split=args.test_split,
        manifest=args.manifest,
        dataset=args.dataset,
        bands=args.bands,
    )
    task = ChesapeakeRSCv2Task(
        model=args.model,
        backbone=args.backbone,
        weights=True,
        in_channels=in_channels,
        num_classes=num_classes,
        num_filters=args.num_filters,
        loss="ce",
        ignore_index=IGNORE_INDEX,
        lr=args.lr,
        tmax=args.tmax,
        class_weights=None,
    )
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        missing, unexpected = task.load_state_dict(ckpt["state_dict"], strict=True)
        prior_epochs = ckpt.get("epoch", -1) + 1
        assert not missing and not unexpected, (
            f"checkpoint/task mismatch: missing={missing} unexpected={unexpected}"
        )
        print(
            f"Warm-started weights from {args.resume_from} ({prior_epochs} prior epochs); "
            f"starting a fresh {args.num_epochs}-epoch cosine cycle (total training "
            f"{prior_epochs + args.num_epochs} epoch-equivalents).",
            flush=True,
        )
    trainer = lightning.Trainer(
        accelerator="gpu",
        precision="16-mixed",
        devices=1,
        min_epochs=args.num_epochs,
        max_epochs=args.num_epochs,
        log_every_n_steps=50,
        default_root_dir=os.path.join("logs", args.experiment_name),
    )
    trainer.fit(task, dm)
    trainer.test(task, dm, ckpt_path="best")


if __name__ == "__main__":
    main(setup_argparse().parse_args())
