"""Join extraction records to patches and flag missing pixels and acquisition issues.

Flags remain in the output. The manifest builder excludes patches with flag_drop."""

import argparse
import glob
import json

import geopandas as gpd
import pandas as pd


def main(args: argparse.Namespace) -> None:
    records = []
    for path in sorted(glob.glob(f"{args.meta_dir}/*.jsonl")):
        with open(path) as stream:
            records.extend(json.loads(line) for line in stream)
    extractions = pd.DataFrame(records)
    extractions = extractions[extractions.status == "ok"].drop_duplicates(
        "patch_id", keep="last"
    )
    patches = gpd.read_parquet(args.patches)
    metadata_columns = [
        "patch_id",
        "epsg",
        "gsd",
        "bounds",
        "n_items",
        "img_nodata_frac",
        "road_px",
        "tc_road_px",
        "mask_nodata_px",
        "bytes",
        "sec",
    ]
    quality = patches.merge(
        extractions[metadata_columns], on="patch_id", how="left", validate="1:1"
    )
    missing = quality.epsg.isna().sum()
    # Match label vintage by NAIP program year; acquisition can extend into the next year.
    naip = (
        pd.read_parquet(args.naip, columns=["id", "year"])
        .set_index("id")["year"]
        .astype(int)
    )
    quality["naip_year"] = quality["naip_ids"].str.split(",").str[0].map(naip)
    expected = quality["state_abbr"].map(
        {
            "de": 2021,
            "md": 2021,
            "va": 2021,
            "ny": 2021,
            "dc": 2021,
            "pa": 2022,
            "wv": 2022,
        }
    )
    quality["naip_year_expected"] = expected
    quality["flag_img_nodata"] = quality.img_nodata_frac > 0.001
    quality["flag_mask_nodata"] = quality.mask_nodata_px > 0.01 * 1024 * 1024
    quality["flag_straddle"] = quality.n_items > 1
    quality["flag_multi_date"] = quality.naip_dates.str.contains(",")
    quality["flag_leaf_off"] = quality.leaf_state == "leaf_off"
    quality["flag_transitional"] = quality.leaf_state == "transitional"
    quality["flag_year_gap"] = quality.naip_year != quality.naip_year_expected
    quality["flag_no_road"] = (quality.road_px + quality.tc_road_px) == 0
    # Drop patches above 5% nodata. Target masks ignore smaller missing areas per pixel.
    quality["flag_drop"] = (quality.img_nodata_frac > 0.05) | (
        quality.mask_nodata_px > 0.05 * 1024 * 1024
    )
    quality["flag_any"] = quality[
        [
            column
            for column in quality.columns
            if column.startswith("flag_") and column not in ("flag_no_road",)
        ]
    ].any(axis=1)
    quality = quality.drop(
        columns=[
            column
            for column in quality.columns
            if column.startswith("c") and column[1:].isdigit()
        ]
    )
    gpd.GeoDataFrame(quality, geometry="geometry", crs=patches.crs).to_parquet(args.out)
    print(f"{len(quality)} patches, {missing} without extraction meta -> {args.out}")
    flags = [column for column in quality.columns if column.startswith("flag_")]
    print(quality[flags].sum().to_string())
    print("\nflag_any by role:")
    print(quality.groupby("role").flag_any.mean().round(3).to_string())
    print(
        "\nbytes total GB:",
        round(quality.bytes.sum() / 1e9, 1),
        " sec/patch mean:",
        round(quality.sec.mean(), 2),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta_dir", required=True)
    parser.add_argument("--patches", required=True)
    parser.add_argument("--naip", required=True)
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
