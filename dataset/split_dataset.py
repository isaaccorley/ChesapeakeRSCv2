"""Assign whole counties to train, validation, and test splits.

Target shares are 80/10/10, with seeded tie-breaking by county size. Delaware
and West Virginia remain whole-state holdouts. Border flags mark patches within
the requested distance of a patch in another split."""

import argparse

import geopandas as gpd
import numpy as np
import pandas as pd


def assign_counties(patches, shares, rng):
    sizes = (
        patches.groupby("fips")
        .size()
        .sample(frac=1, random_state=int(rng.integers(1 << 31)))
    )
    sizes = sizes.sort_values(ascending=False, kind="stable")
    target = {split_name: share * len(patches) for split_name, share in shares.items()}
    filled = {split_name: 0 for split_name in shares}
    assignments = {}
    for fips, count in sizes.items():
        # put the county where the remaining deficit (relative to target) is largest
        split_name = max(
            shares, key=lambda name: (target[name] - filled[name]) / target[name]
        )
        assignments[fips] = split_name
        filled[split_name] += count
    return assignments


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    patches = gpd.read_parquet(args.patches)
    split = pd.Series("", index=patches.index)
    training_patches = patches[patches.role == "train_pool"]
    county_splits = assign_counties(
        training_patches, {"train": 0.8, "val": 0.1, "test": 0.1}, rng
    )
    split[training_patches.index] = training_patches.fips.map(county_splits)
    split[patches.role == "holdout_de"] = "test_de"
    split[patches.role == "holdout_wv"] = "test_wv"
    patches["split"] = split

    print(patches.groupby("split").size().to_string())
    print("\ncounties per split:", patches.groupby("split").fips.nunique().to_dict())
    print("\nstratum share by split:")
    print(
        pd.crosstab(patches.split, patches.stratum, normalize="index")
        .round(3)
        .to_string()
    )

    # cross-split adjacency within the train pool
    buffered = patches[patches.split.isin(["train", "val", "test"])][
        ["patch_id", "split", "geometry"]
    ].copy()
    buffered["geometry"] = buffered.geometry.buffer(args.buffer)
    neighbors = gpd.sjoin(
        buffered,
        patches[patches.split.isin(["train", "val", "test"])][
            ["patch_id", "split", "geometry"]
        ],
        predicate="intersects",
    )
    neighbors = neighbors[
        (neighbors.patch_id_left != neighbors.patch_id_right)
        & (neighbors.split_left != neighbors.split_right)
    ]
    border_ids = sorted(set(neighbors.patch_id_left) | set(neighbors.patch_id_right))
    patches["flag_border"] = patches.patch_id.isin(border_ids)
    print(
        f"\npatches within {args.buffer} m of a different split: {len(border_ids)} -> flag_border"
    )
    patches[["patch_id", "split", "flag_border"]].to_parquet(args.out)
    print("->", args.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--buffer", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
