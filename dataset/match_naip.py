"""Match county label tiles to NAIP footprints from the corresponding program year.

Record image IDs, acquisition dates, leaf state, and class fractions for sampling.
Tiles without imagery remain in the index with n_naip equal to zero."""

import argparse
import glob

import geopandas as gpd
import numpy as np
import pandas as pd

T3_YEAR = {
    "de": 2021,
    "md": 2021,
    "va": 2021,
    "ny": 2021,
    "dc": 2021,
    "pa": 2022,
    "wv": 2022,
}
LEAF_RANK = {"leaf_on": 0, "transitional": 1, "leaf_off": 2}


def main(args: argparse.Namespace) -> None:
    county_indexes = [
        gpd.read_parquet(path)
        for path in sorted(glob.glob(f"{args.tiles_dir}/*.parquet"))
    ]
    tiles = pd.concat(county_indexes, ignore_index=True)
    class_columns = [
        column
        for column in tiles.columns
        if column.startswith("c") and column[1:].isdigit()
    ]
    tiles[class_columns] = tiles[class_columns].fillna(0).astype("int32")
    tiles = gpd.GeoDataFrame(tiles, geometry="geometry", crs=county_indexes[0].crs)
    print("tiles", len(tiles), "counties", tiles.fips.nunique(), flush=True)

    naip = gpd.read_parquet(args.naip)
    naip["year"] = naip["year"].astype(int)
    naip = naip.to_crs(tiles.crs)
    tiles["naip_year"] = tiles["state_abbr"].map(T3_YEAR)

    # restrict to the T3 year per tile before joining; DC rows lack naip:state so match by geometry+year
    summaries = []
    for year, year_tiles in tiles.groupby("naip_year"):
        year_imagery = naip[naip.year == year][
            ["id", "datetime", "leaf_state", "epsg", "gsd", "geometry"]
        ]
        matches = gpd.sjoin(
            gpd.GeoDataFrame(year_tiles[["tile_id", "geometry"]], crs=tiles.crs),
            year_imagery,
            how="left",
            predicate="intersects",
        )
        summary = matches.groupby("tile_id").agg(
            naip_ids=(
                "id",
                lambda values: ",".join(
                    sorted(x for x in values if isinstance(x, str))
                ),
            ),
            n_naip=("id", lambda values: int(values.notna().sum())),
            naip_dates=(
                "datetime",
                lambda values: ",".join(
                    sorted(set(str(x)[:10] for x in values if isinstance(x, str)))
                ),
            ),
            leaf_state=(
                "leaf_state",
                lambda values: max(
                    (x for x in values if isinstance(x, str)),
                    key=LEAF_RANK.__getitem__,
                    default="none",
                ),
            ),
            n_epsg=("epsg", lambda values: int(values.nunique())),
            gsd=("gsd", "max"),
        )
        summaries.append(summary)
    tiles = tiles.merge(
        pd.concat(summaries), left_on="tile_id", right_index=True, how="left"
    )

    valid_pixels = tiles["n_valid"].clip(lower=1)

    def class_count(column):
        return tiles.get(column, 0)

    tiles["frac_road"] = class_count("c20") / valid_pixels
    tiles["frac_tc_road"] = class_count("c23") / valid_pixels
    tiles["frac_impervious"] = (
        class_count("c20") + class_count("c21") + class_count("c22")
    ) / valid_pixels
    tiles["frac_nodata"] = 1 - tiles["n_valid"] / (614 * 614)
    tiles["has_road"] = (class_count("c20") + class_count("c23")) > 0
    tiles = gpd.GeoDataFrame(tiles, geometry="geometry", crs=county_indexes[0].crs)
    tiles.to_parquet(args.out)

    print(tiles.groupby(["state_abbr", "leaf_state"]).size().unstack(fill_value=0))
    print(
        "no NAIP:",
        int((tiles.n_naip == 0).sum()),
        " multi-epsg:",
        int((tiles.n_epsg > 1).sum()),
    )
    print(
        "has_road:",
        int(tiles.has_road.sum()),
        " tc_road>0:",
        int((tiles.frac_tc_road > 0).sum()),
        " tc_road>=0.5%:",
        int((tiles.frac_tc_road >= 0.005).sum()),
    )
    print(
        "frac_road quantiles:",
        np.round(tiles.frac_road.quantile([0.5, 0.9, 0.99]).to_numpy(), 4),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles_dir", required=True)
    parser.add_argument("--naip", required=True)
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
