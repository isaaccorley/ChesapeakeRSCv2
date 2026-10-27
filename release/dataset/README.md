---
license: cc0-1.0
tags:
  - remote-sensing
  - semantic-segmentation
  - road-segmentation
  - naip
  - aerial-imagery
---

# ChesapeakeRSC v2

ChesapeakeRSC v2 extends [Seeing the Roads Through the Trees](https://arxiv.org/abs/2401.06762), a benchmark for road segmentation under tree canopy. It combines NAIP imagery at its native 0.6 m resolution with CC0 Chesapeake Bay LULC 2024 Edition labels across multiple states.

[Models](https://huggingface.co/isaaccorley/chesapeakersc-v2) · [Dataset and training code](https://github.com/isaaccorley/ChesapeakeRSCv2)

![NAIP imagery with visible and canopy-covered road labels](assets/hero_gallery.png)

Examples across states and seasons. Blue marks visible roads; red marks roads under tree canopy.

![Aerial imagery, ground-truth road labels, and predicted road probabilities](assets/qualitative_predictions.png)

Image, ground truth, and predicted road probability for five examples.

## Contents

The 128 archives in `patches_tar/shard-{000..127}.tar` contain 126,701 patches. Each patch has three ZSTD-compressed GeoTIFFs at 1024 × 1024 pixels and 0.6 m resolution.

| File | Contents |
| --- | --- |
| `{id}_image.tif` | Four RGBN image bands |
| `{id}_mask.tif` | Raw land-cover labels |
| `{id}_target.tif` | Background (0), visible road (1), canopy-covered road (2), and ignored pixels (255) |
| `index/manifest.parquet` | Quality-filtered image and target paths, split, state, leaf state, stratum, and dataset membership |

Supporting indexes include `splits.parquet`, `patches_meta.parquet`, `patches_v2.parquet`, `naip_items.parquet`, `naip_items_md2018.parquet`, and `candidates_t3.parquet`. They retain county, acquisition-date, sampling, and quality-control records.

The release contains the core v2 subset. For the optional legacy Maryland subset, see the build instructions in the code repository. Use `dataset/build_manifest.py` to set manifest paths to your local extraction directory.

## Loading a shard

```python
import io
import tarfile

import rasterio

with tarfile.open("shard-000.tar") as tar:
    member = next(item for item in tar.getmembers() if item.name.endswith("_image.tif"))
    stream = tar.extractfile(member)
    assert stream is not None
    with stream:
        image_bytes = stream.read()

with rasterio.open(io.BytesIO(image_bytes)) as src:
    image = src.read()  # (4, 1024, 1024), bands = R, G, B, NIR
```

## Citation

```bibtex
@article{robinson2024seeing,
  title={Seeing the roads through the trees: A benchmark for modeling spatial dependencies with aerial imagery},
  author={Robinson, Caleb and Corley, Isaac and Ortiz, Anthony and Dodhia, Rahul and Ferres, Juan M Lavista and Najafirad, Peyman},
  journal={arXiv preprint arXiv:2401.06762},
  year={2024}
}
```
