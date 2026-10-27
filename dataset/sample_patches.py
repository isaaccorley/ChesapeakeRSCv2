"""Sample patches by canopy coverage, road density, leaf state, and county.

A1/A2/A3 are canopy-road bins starting at 0.5%, 0.8%, and 1.8%. B selects urban
roads, C selects sparse roads, D selects negatives, and O covers other tiles.
Delaware and West Virginia are separate holdouts. Quotas and seeds preserve the
original benchmark sampling procedure."""

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd

TRAIN_STATES = ["md", "va", "pa", "ny", "dc"]
# (stratum, leaf filter) -> quota for the train pool
QUOTA = {
    ("A1", "on"): 12000,
    ("A2", "on"): 20000,
    ("A3", "on"): 13000,
    ("A", "off"): 8000,
    ("B", "on"): 15000,
    ("B", "off"): 2000,
    ("C", "on"): 20000,
    ("D", "on"): 10000,
    ("O", "on"): 12000,
}
HOLDOUT_QUOTA = {
    "de": {
        ("A", "on"): 824,
        ("B", "on"): 932,
        ("C", "on"): 1109,
        ("O", "on"): 2581,
        ("D", "on"): 1000,
    },
    "wv": {
        ("A", "any"): 2000,
        ("B", "any"): 500,
        ("C", "any"): 1500,
        ("O", "any"): 1000,
        ("D", "any"): 1000,
    },
}
FLOOR = 100


def stratify(tiles):
    strata = pd.Series("O", index=tiles.index)
    strata[tiles.frac_tc_road >= 0.005] = "A1"
    strata[tiles.frac_tc_road >= 0.008] = "A2"
    strata[tiles.frac_tc_road >= 0.018] = "A3"
    strata[
        (strata == "O") & (tiles.frac_impervious >= 0.15) & (tiles.frac_road >= 0.03)
    ] = "B"
    strata[(strata == "O") & tiles.has_road & (tiles.frac_road < 0.01)] = "C"
    strata[(strata == "O") & ~tiles.has_road] = "D"
    return strata


def draw(pool, quota, rng):
    """Distribute a target quota across counties, capped by available patches."""
    if quota >= len(pool):
        return pool
    available = pool.groupby("fips").size()
    county_quotas = np.maximum(
        FLOOR, (available / available.sum() * quota).round()
    ).clip(upper=available)
    # Rebalance county quotas after applying the floor and availability caps.
    for _ in range(5):
        county_quotas = np.minimum(
            available,
            np.maximum(
                np.minimum(FLOOR, available),
                (county_quotas * quota / county_quotas.sum()).round(),
            ),
        )
    parts = [
        county_rows.sample(
            int(county_quotas[county]), random_state=int(rng.integers(1 << 31))
        )
        for county, county_rows in pool.groupby("fips")
    ]
    return pd.concat(parts)


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    tiles = gpd.read_parquet(args.candidates)
    tiles = tiles[
        (tiles.frac_nodata < 0.05) & (tiles.n_naip > 0) & (tiles.n_epsg == 1)
    ].copy()
    tiles["stratum"] = stratify(tiles)
    tiles["leaf"] = tiles.leaf_state.map({"leaf_on": "on"}).fillna("off")
    selected = []

    def select_group(pool, quota, role):
        sampled = draw(pool, quota, rng).copy()
        sampled["role"] = role
        selected.append(sampled)

    training_tiles = tiles[tiles.state_abbr.isin(TRAIN_STATES)]
    for (stratum, leaf), quota in QUOTA.items():
        eligible = training_tiles.stratum.str.startswith(stratum) & (
            (training_tiles.leaf == leaf) if leaf != "any" else True
        )
        select_group(
            training_tiles[
                eligible
                & ~training_tiles.tile_id.isin(
                    pd.concat(selected).tile_id if selected else []
                )
            ],
            quota,
            "train_pool",
        )
    for state, quotas in HOLDOUT_QUOTA.items():
        holdout_tiles = tiles[tiles.state_abbr == state]
        for (stratum, leaf), quota in quotas.items():
            eligible = holdout_tiles.stratum.str.startswith(stratum) & (
                (holdout_tiles.leaf == leaf) if leaf != "any" else True
            )
            select_group(holdout_tiles[eligible], quota, f"holdout_{state}")

    patches = gpd.GeoDataFrame(pd.concat(selected), geometry="geometry", crs=tiles.crs)
    assert patches.tile_id.is_unique
    patches["patch_id"] = np.arange(len(patches))
    patches = patches.sample(frac=1, random_state=args.seed).reset_index(
        drop=True
    )  # shuffle so shards are mixed
    patches.to_parquet(args.out)
    print(len(patches), "patches ->", args.out)
    print(pd.crosstab([patches.role, patches.stratum], patches.leaf, margins=True))
    print(patches.groupby("state_abbr").size())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
