"""Build a training manifest for the core dataset and optional legacy subset.

Core rows exclude patches with more than 5% image or label nodata. The manifest
records image and target paths, split, dataset, state, leaf state, stratum, and role.
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def core_rows(meta_path, splits_path, patches_root) -> pd.DataFrame:
    """Combine core split assignments with the patches retained by quality checks."""
    metadata = pd.read_parquet(
        meta_path,
        columns=[
            "patch_id",
            "state_abbr",
            "leaf_state",
            "stratum",
            "role",
            "flag_drop",
        ],
    )
    splits = pd.read_parquet(splits_path)
    manifest = metadata.merge(splits, on="patch_id", validate="1:1")
    manifest = manifest[~manifest.flag_drop].copy()
    patch_id = manifest.patch_id.astype(int)
    shard = (patch_id // 1000).map(lambda value: f"{value:03d}")
    manifest["image_path"] = (
        patches_root
        + "/"
        + shard
        + "/"
        + patch_id.astype(str).str.zfill(6)
        + "_image.tif"
    )
    manifest["target_path"] = (
        patches_root
        + "/"
        + shard
        + "/"
        + patch_id.astype(str).str.zfill(6)
        + "_target.tif"
    )
    manifest["dataset"] = "core"
    return manifest[
        [
            "patch_id",
            "image_path",
            "target_path",
            "split",
            "dataset",
            "state_abbr",
            "leaf_state",
            "stratum",
            "role",
        ]
    ]


def legacy_rows(patches_root, splits_dir, legacy_patchlist_path) -> pd.DataFrame:
    """List re-extracted v1 patches with their original split and leaf state."""
    # Preserve acquisition-derived leaf state for leaf-on evaluation.
    leaf_state_of = pd.read_parquet(
        legacy_patchlist_path, columns=["patch_id", "leaf_state"]
    ).set_index("patch_id")["leaf_state"]
    split_of = {}
    for name in ("train", "val", "test"):
        ids = {
            int(value)
            for value in (Path(splits_dir) / f"{name}_idxs.txt").read_text().split()
        }
        for patch_id in ids:
            split_of[patch_id] = name
    rows = []
    for shard in sorted(os.listdir(patches_root)):
        subdir = os.path.join(patches_root, shard)
        if not os.path.isdir(subdir):
            continue
        for filename in os.listdir(subdir):
            if not filename.endswith("_image.tif"):
                continue
            patch_id = int(filename.split("_")[0])
            if patch_id not in split_of:
                continue  # dropped in build_legacy_patchlist.py (no NAIP match / multi-UTM)
            target_path = os.path.join(subdir, f"{patch_id:06d}_target.tif")
            if not os.path.exists(target_path):
                continue
            rows.append(
                dict(
                    patch_id=patch_id,
                    image_path=os.path.join(subdir, filename),
                    target_path=target_path,
                    split=split_of[patch_id],
                    dataset="legacy",
                    state_abbr="md",
                    leaf_state=leaf_state_of.get(patch_id),
                    stratum="legacy",
                    role="legacy_v1",
                )
            )
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    core = core_rows(args.meta, args.splits, args.patches_root)
    manifest = core
    if args.legacy_root:
        legacy = legacy_rows(
            args.legacy_root, args.legacy_splits, args.legacy_patchlist
        )
        manifest = pd.concat([core, legacy], ignore_index=True)
    manifest.to_parquet(args.out)
    print(f"{len(manifest)} rows -> {args.out}")
    print(pd.crosstab(manifest.dataset, manifest.split))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--patches_root", required=True)
    parser.add_argument(
        "--legacy_root", help="Optional directory of re-extracted v1 patches"
    )
    parser.add_argument(
        "--legacy_splits",
        default=str(Path(__file__).resolve().parent / "provenance/v1_splits"),
    )
    parser.add_argument("--legacy_patchlist", help="Required with --legacy_root")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if bool(args.legacy_root) != bool(args.legacy_patchlist):
        parser.error("--legacy_root and --legacy_patchlist must be provided together")
    main(args)
