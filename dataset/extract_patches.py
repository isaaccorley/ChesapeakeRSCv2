"""Extract aligned NAIP images and county land-cover labels.

Each 1024-pixel patch uses the NAIP UTM pixel grid and native resolution. Labels
are resampled with nearest-neighbor interpolation and retain their original
class values. Existing image/label pairs are skipped; metadata is appended per shard."""

import argparse
import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.merge
import rasterio.windows
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")
os.environ.setdefault("VSI_CACHE", "FALSE")
PATCH = 1024
PROFILE = dict(
    driver="GTiff",
    dtype="uint8",
    width=PATCH,
    height=PATCH,
    compress="zstd",
    zstd_level=9,
    predictor=2,
    tiled=True,
    blockxsize=256,
    blockysize=256,
)


def output_paths(root, patch_id):
    directory = os.path.join(root, f"{patch_id // 1000:03d}")
    return (
        directory,
        os.path.join(directory, f"{patch_id:06d}_image.tif"),
        os.path.join(directory, f"{patch_id:06d}_mask.tif"),
    )


def extract(row, crs, naip, label_path, root) -> dict[str, Any]:
    """Write one aligned image/label pair and return its extraction record."""
    directory, image_path, label_output = output_paths(root, row.patch_id)
    if os.path.exists(image_path) and os.path.exists(label_output):
        return dict(patch_id=int(row.patch_id), status="cached")
    os.makedirs(directory, exist_ok=True)
    imagery_items = [naip[i] for i in row.naip_ids.split(",")]
    epsg = imagery_items[0]["epsg"]
    cx, cy = row.geometry.centroid.coords[0]
    ux, uy = Transformer.from_crs(crs, f"EPSG:{epsg}", always_xy=True).transform(cx, cy)
    sources = [rasterio.open(item["href"]) for item in imagery_items]
    try:
        resolution = sources[0].res[0]
        half = PATCH * resolution / 2
        source_transform = sources[0].transform
        x0 = (
            source_transform.c
            + np.floor((ux - half - source_transform.c) / resolution) * resolution
        )
        y1 = (
            source_transform.f
            + np.ceil((uy + half - source_transform.f) / -resolution) * -resolution
        )
        bounds = (x0, y1 - PATCH * resolution, x0 + PATCH * resolution, y1)
        if len(sources) == 1:
            window = rasterio.windows.from_bounds(*bounds, transform=source_transform)
            image = sources[0].read(
                window=window,
                boundless=True,
                fill_value=0,
                out_shape=(sources[0].count, PATCH, PATCH),
            )
            transform = rasterio.transform.from_bounds(*bounds, PATCH, PATCH)
        else:
            mosaic = rasterio.merge.merge(
                sources, bounds=bounds, res=resolution, nodata=0
            )
            assert mosaic is not None  # No destination path: merge returns an array.
            image, transform = mosaic
            image = image[:, :PATCH, :PATCH]
    finally:
        for source in sources:
            source.close()
    with (
        rasterio.open(label_path) as label_source,
        WarpedVRT(
            label_source,
            crs=f"EPSG:{epsg}",
            transform=transform,
            width=PATCH,
            height=PATCH,
            resampling=Resampling.nearest,
            nodata=0,
        ) as aligned_labels,
    ):
        labels = aligned_labels.read(1)
    profile = dict(PROFILE, crs=f"EPSG:{epsg}", transform=transform)
    temporary_path = image_path + ".tmp.tif"
    with rasterio.open(
        temporary_path, "w", count=image.shape[0], **profile
    ) as destination:
        destination.write(image)
    os.replace(temporary_path, image_path)
    temporary_path = label_output + ".tmp.tif"
    with rasterio.open(
        temporary_path, "w", count=1, nodata=0, **dict(profile, predictor=1)
    ) as destination:
        destination.write(labels, 1)
    os.replace(temporary_path, label_output)
    counts = np.bincount(labels.ravel(), minlength=256)
    return dict(
        patch_id=int(row.patch_id),
        tile_id=row.tile_id,
        status="ok",
        epsg=int(epsg),
        gsd=float(resolution),
        bounds=[float(b) for b in bounds],
        naip_ids=row.naip_ids,
        naip_dates=row.naip_dates,
        leaf_state=row.leaf_state,
        label_year=int(row.label_year),
        fips=row.fips,
        state=row.state_abbr,
        stratum=row.stratum,
        role=row.role,
        n_items=len(imagery_items),
        img_nodata_frac=float((image.max(0) == 0).mean()),
        road_px=int(counts[20]),
        tc_road_px=int(counts[23]),
        mask_nodata_px=int(counts[0]),
        bytes=os.path.getsize(image_path) + os.path.getsize(label_output),
    )


def extract_with_retry(row, crs, naip, labels, root):
    label_path = labels[(row.fips, int(row.label_year))]
    for attempt in range(4):
        started = time.time()
        try:
            record = extract(row, crs, naip, label_path, root)
            record["sec"] = round(time.time() - started, 2)
            return record
        except Exception as error:  # noqa: BLE001 - network errors of many types; retry then report
            error_message = f"{type(error).__name__}: {str(error)[:200]}"
            time.sleep(2**attempt)
    return dict(patch_id=int(row.patch_id), status="fail", error=error_message)


def main(args: argparse.Namespace) -> None:
    patches = gpd.read_parquet(args.patches)
    patches = patches[patches.patch_id % args.nshards == args.shard]
    if args.limit:
        patches = patches.head(args.limit)
    naip = (
        gpd.read_parquet(args.naip)
        .set_index("id")[["href", "epsg", "datetime"]]
        .to_dict("index")
    )
    with open(args.inventory) as stream:
        labels = {
            (record["fips"], int(record["year"])): os.path.join(
                args.labels_dir, record["name"]
            )
            for record in csv.DictReader(stream)
        }
    rows = list(patches.itertuples())
    crs = patches.crs
    os.makedirs(os.path.join(args.out, "meta"), exist_ok=True)
    metadata_path = os.path.join(
        args.out, "meta", f"shard_{args.shard:03d}_of_{args.nshards:03d}.jsonl"
    )
    print(
        f"shard {args.shard}/{args.nshards}: {len(rows)} patches, {args.workers} workers -> {metadata_path}",
        flush=True,
    )
    started = time.time()
    n_ok = n_fail = n_cached = 0
    with (
        open(metadata_path, "a") as metadata_stream,
        ThreadPoolExecutor(args.workers) as executor,
    ):
        for i, record in enumerate(
            executor.map(
                lambda row: extract_with_retry(row, crs, naip, labels, args.out), rows
            ),
            1,
        ):
            metadata_stream.write(json.dumps(record) + "\n")
            n_ok += record["status"] == "ok"
            n_fail += record["status"] == "fail"
            n_cached += record["status"] == "cached"
            if i % 100 == 0 or i == len(rows):
                elapsed = time.time() - started
                print(
                    f"{i}/{len(rows)} ok={n_ok} cached={n_cached} fail={n_fail} {elapsed:.0f}s {i / elapsed:.2f} patch/s",
                    flush=True,
                )
                metadata_stream.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", required=True)
    parser.add_argument("--naip", required=True)
    parser.add_argument(
        "--inventory",
        default=str(
            Path(__file__).resolve().parent / "provenance/county_inventory.csv"
        ),
    )
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--nshards", type=int, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    main(parser.parse_args())
