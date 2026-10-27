"""Export RGB and RGBN checkpoints to ONNX and compare outputs with PyTorch.

Example:
  uv run --group release python release/export_onnx.py \
      --rgb_ckpt /path/to/rgb.ckpt \
      --rgbn_ckpt /path/to/rgbn.ckpt \
      --out_dir release/models
"""

import argparse
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from chesapeakersc.modules import ChesapeakeRSCv2Task

PATCH_SIZE = 1024


def export_one(ckpt_path: str, bands: str, out_path: str) -> None:
    in_channels = 3 if bands == "rgb" else 4
    task = ChesapeakeRSCv2Task(
        model="unet",
        backbone="resnet18",
        weights=False,
        in_channels=in_channels,
        num_classes=2,
        num_filters=64,
        loss="ce",
        ignore_index=255,
        lr=1e-3,
        tmax=50,
        class_weights=None,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    missing, unexpected = task.load_state_dict(ckpt["state_dict"], strict=True)
    assert not missing and not unexpected, (
        f"checkpoint/task mismatch: missing={missing} unexpected={unexpected}"
    )
    model = task.model.eval()

    dummy = torch.randn(1, in_channels, PATCH_SIZE, PATCH_SIZE)
    with torch.no_grad():
        torch_out = model(dummy)

    torch.onnx.export(
        model,
        (dummy,),
        out_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
    )

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])

    # Verify at the export shape, a different batch size, and a different (32-multiple)
    # spatial size, to confirm the dynamic axes actually work end-to-end.
    for shape in [
        (1, in_channels, PATCH_SIZE, PATCH_SIZE),
        (3, in_channels, PATCH_SIZE, PATCH_SIZE),
        (2, in_channels, 512, 1536),
    ]:
        x = torch.randn(*shape)
        with torch.no_grad():
            torch_out = model(x)
        (onnx_out,) = sess.run(None, {"image": x.numpy()})
        max_diff = np.abs(onnx_out - torch_out.numpy()).max()
        print(f"{out_path}: shape={shape} max_diff_vs_torch={max_diff:.2e}")
        assert max_diff < 1e-3, (
            f"ONNX output diverges from PyTorch by {max_diff} at shape {shape}"
        )

    print(
        f"{out_path}: in_channels={in_channels} size={os.path.getsize(out_path) / 1e6:.1f} MB"
    )


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    export_one(
        args.rgb_ckpt, "rgb", os.path.join(args.out_dir, "unet-resnet18-rgb.onnx")
    )
    export_one(
        args.rgbn_ckpt, "rgbn", os.path.join(args.out_dir, "unet-resnet18-rgbn.onnx")
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rgb_ckpt", required=True)
    p.add_argument("--rgbn_ckpt", required=True)
    p.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "models"))
    main(p.parse_args())
