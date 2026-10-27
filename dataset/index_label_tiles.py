"""Count land-cover classes on a fixed grid in one county or an inventory of counties."""

import argparse
import csv
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

TILE_SIZE = 614
INVENTORY = Path(__file__).parent / "provenance/county_inventory.csv"
STATE_CODES = {
    "Delaware": "de",
    "Maryland": "md",
    "New York": "ny",
    "Pennsylvania": "pa",
    "Virginia": "va",
    "West Virginia": "wv",
    "District of Columbia": "dc",
}


def index_county(cog: str | Path, state: str, output: str | Path) -> None:
    """Write class counts and geometry for nonempty 614-pixel label tiles.

    The 1 m label grid approximates the footprint of a 1024-pixel NAIP patch
    at 0.6 m. Extraction later aligns each patch to the NAIP pixel grid.
    """
    name = Path(cog).name
    match = re.fullmatch(r"(\w+)_(\d{5})_lulc_(\d{4})_2024-Edition\.tif", name)
    if match is None:
        raise ValueError(f"Unrecognized county label filename: {name}")
    _, fips, year = match.groups()
    records = []
    with rasterio.open(cog) as source:
        transform = source.transform
        row_count = -(-source.height // TILE_SIZE)
        column_count = -(-source.width // TILE_SIZE)
        for row in range(row_count):
            height = min(TILE_SIZE, source.height - row * TILE_SIZE)
            strip = source.read(
                1,
                window=Window.from_slices(
                    (row * TILE_SIZE, row * TILE_SIZE + height), (0, source.width)
                ),
            )
            for column in range(column_count):
                tile = strip[:, column * TILE_SIZE : (column + 1) * TILE_SIZE]
                counts = np.bincount(tile.ravel(), minlength=256)
                if counts[0] == tile.size:
                    continue
                left, top = transform * (column * TILE_SIZE, row * TILE_SIZE)
                right, bottom = transform * (
                    (column + 1) * TILE_SIZE,
                    (row + 1) * TILE_SIZE,
                )
                record = {
                    "fips": fips,
                    "state_abbr": state,
                    "label_year": int(year),
                    "row": row,
                    "col": column,
                    "n_valid": int(tile.size - counts[0]),
                    "geometry": box(left, bottom, right, top),
                }
                for value in np.nonzero(counts)[0]:
                    record[f"c{value}"] = int(counts[value])
                records.append(record)
        crs = source.crs
    if not records:
        raise ValueError(f"No valid label pixels in {cog}")
    tiles = gpd.GeoDataFrame(
        pd.DataFrame(records).fillna(0), geometry="geometry", crs=crs
    )
    for column in tiles.columns:
        if column.startswith("c") and column[1:].isdigit():
            tiles[column] = tiles[column].astype("int32")
    tiles["tile_id"] = (
        fips
        + "_"
        + year
        + "_"
        + tiles["row"].astype(str)
        + "_"
        + tiles["col"].astype(str)
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    tiles.to_parquet(Path(output))
    print(f"{name}: {len(tiles)} tiles -> {output}", flush=True)


def main(args: argparse.Namespace) -> None:
    if args.cog:
        index_county(args.cog, args.state, args.out)
        return
    years = set(args.years.split(","))
    with open(args.inventory) as stream:
        counties = [row for row in csv.DictReader(stream) if row["year"] in years]
    # Sharding supports Slurm arrays without parsing CSV in shell scripts.
    for county in counties[args.shard :: args.nshards]:
        source = Path(args.labels_dir) / county["name"]
        output = Path(args.out) / Path(county["name"]).with_suffix(".parquet")
        index_county(source, STATE_CODES[county["state"]], output)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    inputs = result.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--cog", help="Index one county label raster")
    inputs.add_argument("--labels_dir", help="Index rasters listed in the inventory")
    result.add_argument(
        "--state", choices=sorted(STATE_CODES.values()), help="Required with --cog"
    )
    result.add_argument(
        "--out",
        required=True,
        help="Output file for --cog, or directory for --labels_dir",
    )
    result.add_argument("--inventory", type=Path, default=INVENTORY)
    result.add_argument(
        "--years", default="2021,2022", help="Comma-separated label years"
    )
    result.add_argument("--shard", type=int, default=0)
    result.add_argument("--nshards", type=int, default=1)
    return result


if __name__ == "__main__":
    cli = parser()
    args = cli.parse_args()
    if args.cog and not args.state:
        cli.error("--state is required with --cog")
    if args.nshards < 1 or not 0 <= args.shard < args.nshards:
        cli.error("Require --nshards >= 1 and 0 <= --shard < --nshards")
    main(args)
