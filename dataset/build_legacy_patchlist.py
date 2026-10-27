"""Match original ChesapeakeRSC windows to Maryland 2018 labels and NAIP imagery.

Preserve original patch IDs so the v1 split lists can be reused. Keep windows
with valid county labels, matching imagery, and a single image projection."""

import argparse
import csv
import os
from contextlib import ExitStack
from pathlib import Path

import geopandas as gpd
import rasterio
import rasterio.windows

LEAF_RANK = {"leaf_on": 0, "transitional": 1, "leaf_off": 2}


def main(args: argparse.Namespace) -> None:
    windows = gpd.read_file(args.gpkg)
    with open(args.inventory) as stream:
        inventory = [
            row
            for row in csv.DictReader(stream)
            if row["state"] == "Maryland" and row["year"] == "2018"
        ]
    with ExitStack() as stack:
        county_rasters = {
            row["fips"]: stack.enter_context(
                rasterio.open(os.path.join(args.labels_dir, row["name"]))
            )
            for row in inventory
        }
        fips = []
        for geometry in windows.geometry:
            cx, cy = geometry.centroid.coords[0]
            hit = None
            for path, source in county_rasters.items():
                bounds = source.bounds
                if (
                    bounds.left <= cx <= bounds.right
                    and bounds.bottom <= cy <= bounds.top
                ):
                    row, column = source.index(cx, cy)
                    if (
                        source.read(
                            1,
                            window=rasterio.windows.Window.from_slices(
                                (row, row + 1), (column, column + 1)
                            ),
                        )[0, 0]
                        != 0
                    ):
                        hit = path
                        break
            fips.append(hit)
    windows["fips"] = fips
    print("windows", len(windows), "no county:", sum(path is None for path in fips))

    naip = gpd.read_parquet(args.naip).to_crs(windows.crs)
    matches = gpd.sjoin(
        windows[["idx", "geometry"]],
        naip[["id", "datetime", "leaf_state", "epsg", "geometry"]],
        how="left",
        predicate="intersects",
    )
    summary = matches.groupby("idx").agg(
        naip_ids=(
            "id",
            lambda values: ",".join(
                sorted(value for value in values if isinstance(value, str))
            ),
        ),
        n_naip=("id", lambda values: int(values.notna().sum())),
        naip_dates=(
            "datetime",
            lambda values: ",".join(
                sorted(
                    set(str(value)[:10] for value in values if isinstance(value, str))
                )
            ),
        ),
        leaf_state=(
            "leaf_state",
            lambda values: max(
                (value for value in values if isinstance(value, str)),
                key=LEAF_RANK.__getitem__,
                default="none",
            ),
        ),
        n_epsg=("epsg", lambda values: int(values.nunique())),
    )
    windows = windows.merge(summary, left_on="idx", right_index=True)
    windows = windows[
        (windows.n_naip > 0) & (windows.n_epsg == 1) & windows.fips.notna()
    ].copy()
    windows["patch_id"] = windows["idx"].astype(int)
    windows["tile_id"] = "v1_" + windows["idx"].astype(str)
    windows["label_year"] = 2018
    windows["state_abbr"] = "md"
    windows["stratum"] = "legacy"
    windows["role"] = "legacy_v1"
    columns = [
        "patch_id",
        "tile_id",
        "fips",
        "state_abbr",
        "label_year",
        "naip_ids",
        "n_naip",
        "naip_dates",
        "leaf_state",
        "n_epsg",
        "stratum",
        "role",
        "geometry",
    ]
    legacy_patches = gpd.GeoDataFrame(
        windows[columns], geometry="geometry", crs=windows.crs
    )
    legacy_patches.to_parquet(args.out)
    print(len(legacy_patches), "legacy patches ->", args.out)
    print(legacy_patches.leaf_state.value_counts().to_string())
    print(
        "straddle:",
        int((legacy_patches.n_naip > 1).sum()),
        " counties:",
        legacy_patches.fips.nunique(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpkg", required=True, help="Path to the original v1 patches.gpkg"
    )
    parser.add_argument(
        "--inventory",
        default=str(
            Path(__file__).resolve().parent / "provenance/county_inventory.csv"
        ),
    )
    parser.add_argument("--labels_dir", required=True)
    parser.add_argument("--naip", required=True)
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
