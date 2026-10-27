"""Write STAC MLM v1.5.1 metadata with assets pinned to a model repository commit."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import segmentation_models_pytorch as smp
import torch

REPO = "isaaccorley/chesapeakersc-v2"
# Chesapeake Bay watershed states covered by the core training data.
BBOX = [-80.52, 36.54, -74.69, 42.75]
METRICS = {
    "rgb": {"recall": 0.756, "precision": 0.826, "f1": 0.790},
    "rgbn": {"recall": 0.790, "precision": 0.781, "f1": 0.785},
}
BANDS = {
    "rgb": ["red", "green", "blue"],
    "rgbn": ["red", "green", "blue", "nir"],
}


def multihash_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "1220" + h.hexdigest()


def item(bands: str, export_dir: str, revision: str) -> dict:
    stem = f"unet-resnet18-{bands}"
    band_names = BANDS[bands]
    n = len(band_names)
    base = f"https://huggingface.co/{REPO}/resolve/{revision}"
    onnx_path = os.path.join(export_dir, f"{stem}.onnx")
    pt_path = os.path.join(export_dir, f"{stem}.pt")
    m = METRICS[bands]
    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [
            "https://stac-extensions.github.io/mlm/v1.5.1/schema.json",
            "https://stac-extensions.github.io/version/v1.2.0/schema.json",
            "https://stac-extensions.github.io/classification/v2.0.0/schema.json",
            "https://stac-extensions.github.io/file/v2.1.0/schema.json",
            "https://stac-extensions.github.io/raster/v1.1.0/schema.json",
        ],
        "id": f"chesapeakersc-v2-{stem}",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    [BBOX[0], BBOX[1]],
                    [BBOX[2], BBOX[1]],
                    [BBOX[2], BBOX[3]],
                    [BBOX[0], BBOX[3]],
                    [BBOX[0], BBOX[1]],
                ]
            ],
        },
        "bbox": BBOX,
        "properties": {
            "title": f"ChesapeakeRSC v2 U-Net ResNet-18 ({bands.upper()}) road segmentation",
            "description": (
                "Binary road segmentation under tree-canopy occlusion from 0.6 m NAIP aerial "
                f"imagery ({bands.upper()}). U-Net with an ImageNet-pretrained ResNet-18 encoder, "
                "trained on the ChesapeakeRSC v2 core split (1024x1024 px patches at native "
                "resolution) from Chesapeake Bay watershed states."
            ),
            "datetime": None,
            "start_datetime": "2021-01-01T00:00:00Z",
            "end_datetime": "2022-12-31T23:59:59Z",
            "license": "MIT",
            "version": "1",
            "keywords": [
                "road",
                "semantic-segmentation",
                "naip",
                "aerial-imagery",
                "tree-canopy-occlusion",
            ],
            "mlm:name": f"chesapeakersc-v2-{stem}",
            "mlm:architecture": "UNet",
            "mlm:tasks": ["semantic-segmentation"],
            "mlm:framework": "PyTorch",
            "mlm:framework_version": torch.__version__.split("+")[0],
            "mlm:accelerator": "cuda",
            "mlm:accelerator_constrained": False,
            "mlm:batch_size_suggestion": 16,
            "mlm:memory_size": os.path.getsize(pt_path),
            "mlm:total_parameters": sum(
                p.numel()
                for p in smp.Unet(
                    "resnet18", encoder_weights=None, in_channels=n, classes=2
                ).parameters()
            ),
            "mlm:pretrained": True,
            "mlm:pretrained_source": "ImageNet (torchvision ResNet-18 weights via segmentation_models_pytorch)",
            "mlm:input": [
                {
                    "name": "image",
                    "bands": [{"name": b} for b in band_names],
                    "input": {
                        "shape": [-1, n, -1, -1],
                        "dim_order": ["batch", "bands", "height", "width"],
                        "data_type": "float32",
                    },
                    "description": "Height and width must be multiples of 32. Trained on 1024x1024 px chips at 0.6 m GSD.",
                    "value_scaling": [
                        {"type": "scale", "value": 255.0} for _ in band_names
                    ],
                    "resize_type": None,
                    "pre_processing_function": None,
                }
            ],
            "mlm:output": [
                {
                    "name": "logits",
                    "tasks": ["semantic-segmentation"],
                    "result": {
                        "shape": [-1, 2, -1, -1],
                        "dim_order": ["batch", "channel", "height", "width"],
                        "data_type": "float32",
                    },
                    "classification:classes": [
                        {
                            "name": "background",
                            "value": 0,
                            "description": "Non-road pixels",
                        },
                        {
                            "name": "road",
                            "value": 1,
                            "description": "Road pixels, visible or occluded by tree canopy",
                            "color_hint": "FFFFFF",
                        },
                    ],
                    "post_processing_function": None,
                    "description": "Per-pixel class logits; argmax over the channel axis gives the class map.",
                }
            ],
            "mlm:hyperparameters": {
                "epochs": 100,
                "batch_size": 16,
                "optimizer": "AdamW",
                "learning_rate": 0.001,
                "scheduler": "cosine",
                "loss": "cross_entropy",
            },
            "eval:core_test_road_recall": m["recall"],
            "eval:core_test_road_precision": m["precision"],
            "eval:core_test_road_f1": m["f1"],
        },
        "links": [
            {
                "rel": "cite-as",
                "href": "https://doi.org/10.48550/arXiv.2401.06762",
                "title": "Seeing the Roads Through the Trees: A Benchmark for Modeling Spatial Dependencies with Aerial Imagery",
            },
            {
                "rel": "derived_from",
                "href": f"https://huggingface.co/datasets/{REPO}",
                "title": "ChesapeakeRSC v2 dataset",
            },
        ],
        "assets": {
            "model": {
                "href": f"{base}/{stem}.onnx",
                "type": "application/octet-stream; application=onnx",
                "title": f"ONNX model ({bands.upper()}), dynamic batch/height/width",
                "roles": ["mlm:model", "mlm:weights"],
                "mlm:artifact_type": "onnx",
                "raster:bands": [{"name": b} for b in band_names],
                "file:size": os.path.getsize(onnx_path),
                "file:checksum": multihash_sha256(onnx_path),
            },
            "checkpoint": {
                "href": f"{base}/{stem}.pt",
                "type": "application/octet-stream; framework=pytorch",
                "title": f"PyTorch state_dict for smp.Unet(resnet18, in_channels={n}, classes=2)",
                "roles": ["mlm:model", "mlm:weights"],
                "mlm:artifact_type": "torch.save",
                "raster:bands": [{"name": b} for b in band_names],
                "file:size": os.path.getsize(pt_path),
                "file:checksum": multihash_sha256(pt_path),
            },
            "source-code": {
                "href": "https://github.com/isaaccorley/ChesapeakeRSCv2",
                "type": "text/html",
                "title": "Training and evaluation code",
                "roles": ["code"],
            },
            "readme": {
                "href": f"{base}/README.md",
                "type": "text/markdown",
                "title": "Model card",
                "roles": ["metadata"],
            },
        },
    }


def main(a: argparse.Namespace) -> None:
    for bands in ("rgb", "rgbn"):
        out = os.path.join(a.export_dir, f"unet-resnet18-{bands}.stac-item.json")
        with open(out, "w") as f:
            json.dump(item(bands, a.export_dir, a.revision), f, indent=2)
            f.write("\n")
        print("wrote", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--export_dir", default=str(Path(__file__).resolve().parent / "models")
    )
    p.add_argument(
        "--revision",
        required=True,
        help="HF model repo commit sha to pin asset hrefs to",
    )
    main(p.parse_args())
