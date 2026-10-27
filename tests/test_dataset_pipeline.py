"""Exercise dataset creation and packaging with local rasters."""

import tarfile
from argparse import Namespace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from chesapeakersc.datasets import ChesapeakeRSCv2
from dataset import (
    build_manifest,
    check_quality,
    create_targets,
    extract_patches,
    index_label_tiles,
    match_naip,
    sample_patches,
    split_dataset,
)
from release import pack_shards

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("mosaic", [False, True])
def test_local_build_to_release(tmp_path, mosaic):
    labels = tmp_path / "labels"
    tiles = tmp_path / "tiles"
    labels.mkdir()
    tiles.mkdir()
    label_path = labels / "balt_24005_lulc_2021_2024-Edition.tif"
    label = np.full((614, 614), 10, dtype="uint8")
    label[299:314, :] = 20
    label[299:314, 200:400] = 23
    with rasterio.open(
        label_path,
        "w",
        driver="GTiff",
        height=614,
        width=614,
        count=1,
        dtype="uint8",
        crs="EPSG:32618",
        transform=from_origin(500000, 4500000, 1, 1),
        nodata=0,
    ) as dst:
        dst.write(label, 1)

    image_path = tmp_path / "naip.tif"
    image = np.full((4, 1028, 1028), 100, dtype="uint8")
    image[:, :8] = 0
    with rasterio.open(
        image_path,
        "w",
        driver="GTiff",
        height=1028,
        width=1028,
        count=4,
        dtype="uint8",
        crs="EPSG:32618",
        transform=from_origin(499999.4, 4500000.6, 0.6, 0.6),
    ) as dst:
        dst.write(image)
        bounds = dst.bounds

    naip = tmp_path / "naip.parquet"
    imagery = [
        {
            "id": "local-naip",
            "year": 2021,
            "datetime": "2021-07-01T00:00:00Z",
            "leaf_state": "leaf_on",
            "epsg": 32618,
            "gsd": 0.6,
            "href": str(image_path),
            "geometry": box(*bounds),
        }
    ]
    if mosaic:
        imagery.append({**imagery[0], "id": "overlapping-naip"})
    gpd.GeoDataFrame(imagery, crs="EPSG:32618").to_parquet(naip)
    index_label_tiles.main(
        Namespace(
            cog=str(label_path),
            state="md",
            out=tiles / "county.parquet",
        )
    )
    candidates = tmp_path / "candidates.parquet"
    match_naip.main(Namespace(tiles_dir=tiles, naip=naip, out=candidates))
    patches = tmp_path / "patches.parquet"
    sample_patches.main(Namespace(candidates=candidates, out=patches, seed=0))
    patch_root = tmp_path / "patches"
    extract_patches.main(
        Namespace(
            patches=patches,
            naip=naip,
            inventory=ROOT / "dataset/provenance/county_inventory.csv",
            labels_dir=labels,
            out=str(patch_root),
            shard=0,
            nshards=1,
            workers=1,
            limit=0,
        )
    )
    meta = tmp_path / "meta.parquet"
    check_quality.main(
        Namespace(
            meta_dir=patch_root / "meta",
            patches=patches,
            naip=naip,
            out=meta,
        )
    )
    assert bool(pd.read_parquet(meta).iloc[0].flag_straddle) == mosaic
    splits = tmp_path / "splits.parquet"
    split_dataset.main(Namespace(patches=meta, out=splits, buffer=1000, seed=0))
    create_targets.main(Namespace(root=patch_root, shard=0, nshards=1, workers=1))
    manifest = tmp_path / "manifest.parquet"
    build_manifest.main(
        Namespace(
            meta=meta,
            splits=splits,
            patches_root=str(patch_root),
            legacy_root=None,
            out=manifest,
        )
    )

    records = pd.read_parquet(manifest)
    assert len(records) == 1
    assert records.iloc[0].dataset == "core"
    assert records.iloc[0].leaf_state == "leaf_on"
    with rasterio.open(records.iloc[0].target_path) as target:
        assert set(np.unique(target.read(1))) == {0, 1, 2, 255}
    rgb = ChesapeakeRSCv2(str(manifest), bands="rgb")[0]
    rgbn = ChesapeakeRSCv2(
        str(manifest),
        bands="rgbn",
        differentiate_tree_canopy_over_roads=True,
    )[0]
    assert rgb["image"].shape == (3, 1024, 1024)
    assert rgbn["image"].shape == (4, 1024, 1024)
    assert 2 not in rgb["mask"].unique()
    assert 2 in rgbn["mask"].unique()
    assert 255 in rgb["mask"].unique()

    archives = tmp_path / "archives"
    pack_shards.main(Namespace(patches_dir=patch_root, out_dir=archives, workers=1))
    assert [path.name for path in archives.iterdir()] == ["shard-000.tar"]
    with tarfile.open(archives / "shard-000.tar") as archive:
        files = {member.name for member in archive.getmembers() if member.isfile()}
        assert files == {
            "000/000000_image.tif",
            "000/000000_mask.tif",
            "000/000000_target.tif",
        }
        for name in files:
            stream = archive.extractfile(name)
            assert stream is not None
            with stream:
                assert stream.read() == (patch_root / name).read_bytes()


def test_manifest_filters_missing_data(tmp_path):
    meta = tmp_path / "meta.parquet"
    splits = tmp_path / "splits.parquet"
    pd.DataFrame(
        {
            "patch_id": [0, 1],
            "state_abbr": ["md", "md"],
            "leaf_state": ["leaf_on", "leaf_off"],
            "stratum": ["A1", "A1"],
            "role": ["train_pool", "train_pool"],
            "flag_drop": [False, True],
        }
    ).to_parquet(meta)
    pd.DataFrame({"patch_id": [0, 1], "split": ["train", "test"]}).to_parquet(splits)
    result = build_manifest.core_rows(meta, splits, str(tmp_path / "patches"))
    assert result.patch_id.tolist() == [0]
    assert result.image_path.iloc[0].endswith("000/000000_image.tif")


def test_legacy_manifest_preserves_splits_and_leaf_state(tmp_path):
    patch_dir = tmp_path / "patches/000"
    patch_dir.mkdir(parents=True)
    (patch_dir / "000007_image.tif").touch()
    (patch_dir / "000007_target.tif").touch()
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    for split in ("train", "val", "test"):
        (split_dir / f"{split}_idxs.txt").write_text("7\n" if split == "val" else "")
    patchlist = tmp_path / "legacy.parquet"
    pd.DataFrame({"patch_id": [7], "leaf_state": ["transitional"]}).to_parquet(
        patchlist
    )
    result = build_manifest.legacy_rows(patch_dir.parent, split_dir, patchlist)
    assert result.split.tolist() == ["val"]
    assert result.leaf_state.tolist() == ["transitional"]
    assert result.dataset.tolist() == ["legacy"]
