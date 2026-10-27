# Export and publish

Run release commands from the repository root after installing their dependencies:

```sh
uv sync --locked --group release
```

[`models/README.md`](models/README.md) and [`dataset/README.md`](dataset/README.md) are the Hugging Face cards. Their image paths resolve within each Hugging Face repository; upload the PNG figures from the root `assets/` directory to `assets/` alongside each card. The STAC items in `models/` describe the published weights and pin asset URLs to a Hugging Face commit.

## Model exports

```sh
uv run --group release python release/export_onnx.py \
  --rgb_ckpt /path/to/rgb.ckpt --rgbn_ckpt /path/to/rgbn.ckpt --out_dir release/models
```

The exporter writes RGB and RGBN ONNX files and compares their logits with PyTorch at several batch and image sizes. The corresponding `.pt` files must contain plain model state dictionaries, as shown in the [model card](models/README.md).

Upload the weights, card, and figures:

```sh
uv run --group release hf upload isaaccorley/chesapeakersc-v2 release/models .
uv run --group release hf upload isaaccorley/chesapeakersc-v2 assets assets --include '*.png'
```

Use the weights upload's commit to generate STAC metadata:

```sh
uv run --group release python release/make_stac_mlm.py \
  --export_dir release/models --revision MODEL_COMMIT_SHA
```

The generator reads both `.onnx` and `.pt` files and records their sizes and SHA-256 checksums. Upload the resulting STAC files after generation. Regenerate them whenever weights change.

## Dataset archives

Stage the core-only manifest, index files, and card in a separate directory:

```sh
export RELEASE_DIR=/absolute/path/to/dataset-release
mkdir -p "$RELEASE_DIR/index" "$RELEASE_DIR/assets"
cp release/dataset/README.md "$RELEASE_DIR/README.md"
cp assets/*.png "$RELEASE_DIR/assets/"
cp /path/to/index/*.parquet "$RELEASE_DIR/index/"
uv run --group release python release/pack_shards.py \
  --patches_dir /path/to/patches --out_dir "$RELEASE_DIR/patches_tar"
```

Each numbered patch directory becomes a tar archive, retaining paths such as `000/000000_image.tif`. The packer skips extraction metadata and existing archives. Use an empty output directory when repackaging changed patches. Include only core patches and a core-only manifest in the public release.

Authenticate with Hugging Face, then upload the staged files:

```sh
uv run --group release hf auth login
uv run --group release python release/upload_dataset.py \
  --release_dir "$RELEASE_DIR" --repo_id isaaccorley/chesapeakersc-v2
```

The upload includes `patches_tar/*`, `index/*`, `assets/*.png`, and `README.md`. Users can rebuild manifest paths with `dataset/build_manifest.py` after extracting the shards.
