"""Create training targets from raw land-cover labels and image validity.

Targets use 0 for background, 1 for road, 2 for canopy-covered road, and 255 for
label nodata or pixels whose image bands are all zero."""

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio

IGNORE = 255
TARGET_PROFILE = dict(
    compress="zstd", zstd_level=9, predictor=1, nodata=IGNORE, count=1, dtype="uint8"
)


def create_target(image_path: str) -> tuple[str, int, int]:
    """Write one target and return its status and ignored-pixel counts."""
    label_path = image_path.replace("_image.tif", "_mask.tif")
    target_path = image_path.replace("_image.tif", "_target.tif")
    if os.path.exists(target_path):
        return "skip", 0, 0
    with rasterio.open(label_path) as source:
        labels = source.read(1)
        profile = dict(source.profile, **TARGET_PROFILE)
    with rasterio.open(image_path) as source:
        img_nodata = source.read().max(axis=0) == 0
    target = np.zeros_like(labels)
    target[labels == 20] = 1
    target[labels == 23] = 2
    target[(labels == 0) | img_nodata] = IGNORE
    temporary_path = target_path + ".tmp.tif"
    with rasterio.open(temporary_path, "w", **profile) as destination:
        destination.write(target, 1)
    os.replace(temporary_path, target_path)
    return "ok", int((target == IGNORE).sum()), int(img_nodata.sum())


def main(args: argparse.Namespace) -> None:
    files = sorted(glob.glob(os.path.join(args.root, "*", "*_image.tif")))
    files = [source for i, source in enumerate(files) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(files)} patches", flush=True)
    started = time.time()
    counts = {"ok": 0, "skip": 0}
    ignored_pixels = image_nodata_pixels = 0
    with ThreadPoolExecutor(args.workers) as executor:
        for i, (status, patch_ignored, patch_image_nodata) in enumerate(
            executor.map(create_target, files), 1
        ):
            counts[status] += 1
            ignored_pixels += patch_ignored
            image_nodata_pixels += patch_image_nodata
            if i % 5000 == 0 or i == len(files):
                print(
                    f"{i}/{len(files)} {counts} ignore_px={ignored_pixels} img_nodata_px={image_nodata_pixels} {time.time() - started:.0f}s",
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    main(parser.parse_args())
