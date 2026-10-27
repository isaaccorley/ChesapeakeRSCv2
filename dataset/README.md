# Build the dataset

These commands create ChesapeakeRSC v2 from NAIP aerial imagery and Chesapeake Bay Land Use/Land Cover 2024 Edition labels. The core release uses 2021 imagery for DC, Delaware, Maryland, New York, and Virginia, and 2022 imagery for Pennsylvania and West Virginia.

Run commands from the repository root. Install the dataset dependencies and choose an absolute data directory:

```sh
uv sync --locked --group dataset
export DATA_ROOT=/absolute/path/to/chesapeakersc-v2
mkdir -p "$DATA_ROOT/index" "$DATA_ROOT/labels/t3"
```

Downloads require internet access and `curl`. Label rasters are downloaded locally; extraction reads NAIP windows from remote imagery. The [source inventory](provenance/README.md) lists the county rasters and original v1 split IDs.

## Download and index

Download the labels, save the NAIP catalog, and index each county's label tiles:

```sh
uv run --group dataset python dataset/download_labels.py --out_dir "$DATA_ROOT/labels/t3"
uv run --group dataset python dataset/catalog_naip.py --out "$DATA_ROOT/index/naip_items.parquet"
uv run --group dataset python dataset/index_label_tiles.py \
  --labels_dir "$DATA_ROOT/labels/t3" --out "$DATA_ROOT/index/tiles"
```

`download_labels.py` resumes partial downloads and checks file sizes against the inventory. Use `--states Maryland --years 2021` for a smaller download. To index one raster, use `index_label_tiles.py --cog FILE --state md --out FILE.parquet`.

## Select patches

Match the label tiles to imagery from the required program year, then sample by road coverage, canopy coverage, county, and leaf state:

```sh
uv run --group dataset python dataset/match_naip.py \
  --tiles_dir "$DATA_ROOT/index/tiles" --naip "$DATA_ROOT/index/naip_items.parquet" \
  --out "$DATA_ROOT/index/candidates_t3.parquet"
uv run --group dataset python dataset/sample_patches.py \
  --candidates "$DATA_ROOT/index/candidates_t3.parquet" --out "$DATA_ROOT/index/patches_v2.parquet"
```

Sampling excludes candidates with at least 5% label nodata, no matching imagery, or imagery spanning multiple UTM zones. Quotas are defined in `sample_patches.py`; `--seed` defaults to 0.

## Extract images and labels

Each image patch contains four bands and 1024 × 1024 pixels on the native NAIP grid. County labels are resampled to that grid with nearest-neighbor interpolation.

```sh
uv run --group dataset python dataset/extract_patches.py \
  --patches "$DATA_ROOT/index/patches_v2.parquet" --naip "$DATA_ROOT/index/naip_items.parquet" \
  --labels_dir "$DATA_ROOT/labels/t3" --out "$DATA_ROOT/patches" --shard 0 --nshards 1
uv run --group dataset python dataset/create_targets.py --root "$DATA_ROOT/patches"
```

Use `--limit 10` for a trial extraction. To distribute extraction across workers or nodes, use the same `--nshards N` and distinct `--shard` values from 0 to N−1. Existing image/label pairs and target files are skipped; use a fresh output directory when changing extraction settings. [Batch examples](scripts/README.md) cover downloads, indexing, extraction, and target creation.

Patches are stored in numbered directories, such as `patches/000/000000_image.tif`. Each patch has three GeoTIFFs:

| Suffix | Contents |
| --- | --- |
| `_image.tif` | RGB and near-infrared bands, uint8 |
| `_mask.tif` | Original land-cover codes, including road (20), canopy over road (23), and nodata (0) |
| `_target.tif` | Background (0), road (1), canopy over road (2), and ignored pixels (255) |

Target pixels are ignored where labels are missing or all image bands are zero. Extraction records are appended to `patches/meta/`.

## Check quality and assign splits

```sh
uv run --group dataset python dataset/check_quality.py \
  --patches "$DATA_ROOT/index/patches_v2.parquet" --meta_dir "$DATA_ROOT/patches/meta" \
  --naip "$DATA_ROOT/index/naip_items.parquet" --out "$DATA_ROOT/index/patches_meta.parquet"
uv run --group dataset python dataset/split_dataset.py \
  --patches "$DATA_ROOT/index/patches_meta.parquet" --out "$DATA_ROOT/index/splits.parquet"
```

Quality checks flag missing pixels, imagery boundaries, acquisition dates, and leaf state. County splits target an 80/10/10 allocation across training, validation, and testing. Delaware and West Virginia are separate `test_de` and `test_wv` holdouts. `flag_border` identifies patches near another split; it does not remove them.

Leaf state comes from acquisition month: May–September is `leaf_on`, October is `transitional`, and November–April is `leaf_off`.

## Training manifest

Build the manifest after extraction, target creation, and splitting:

```sh
uv run --group dataset python dataset/build_manifest.py \
  --meta "$DATA_ROOT/index/patches_meta.parquet" --splits "$DATA_ROOT/index/splits.parquet" \
  --patches_root "$DATA_ROOT/patches" --out "$DATA_ROOT/index/manifest.parquet"
```

The manifest records image and target paths, split, state, leaf state, and sampling stratum. It excludes core patches with more than 5% image or label nodata. The same command can rebuild paths for a downloaded release: point `--meta` and `--splits` at its index files and `--patches_root` at the extracted shard directory.

## Rebuild the legacy subset

The optional legacy subset re-extracts the original ChesapeakeRSC windows. It requires the original `patches.gpkg`, Maryland 2018 labels, and a Maryland 2018 NAIP catalog.

```sh
uv run --group dataset python dataset/download_labels.py \
  --years 2018 --states Maryland --out_dir "$DATA_ROOT/labels/t2"
uv run --group dataset python dataset/catalog_naip.py \
  --pairs md:2018 --out "$DATA_ROOT/index/naip_items_md2018.parquet"
uv run --group dataset python dataset/build_legacy_patchlist.py \
  --gpkg /path/to/v1/patches.gpkg --labels_dir "$DATA_ROOT/labels/t2" \
  --naip "$DATA_ROOT/index/naip_items_md2018.parquet" --out "$DATA_ROOT/index/patches_legacy.parquet"
```

Run `extract_patches.py` with those legacy inputs and a separate `patches_legacy` output directory, then run `create_targets.py` there. Add `--legacy_root` and `--legacy_patchlist` to `build_manifest.py` to combine the subsets. The builder uses the original IDs in `provenance/v1_splits/`; `--legacy_splits` selects another split directory.

The public dataset contains core patches only. See the [release guide](../release/README.md) for packaging and upload.
