---
license: mit
pipeline_tag: image-segmentation
library_name: onnxruntime
datasets:
  - isaaccorley/chesapeakersc-v2
tags:
  - remote-sensing
  - semantic-segmentation
  - road-segmentation
  - onnx
  - naip
  - aerial-imagery
---

# ChesapeakeRSC v2 road segmentation models

[Dataset](https://huggingface.co/datasets/isaaccorley/chesapeakersc-v2) · [Training and export code](https://github.com/isaaccorley/ChesapeakeRSCv2)

![Aerial imagery, ground-truth road labels, and predicted road probabilities](assets/qualitative_predictions.png)

Image, ground truth, and predicted road probability for five examples. Ground-truth labels distinguish visible roads (blue) from roads under tree canopy (red).

![NAIP imagery with visible and canopy-covered road labels](assets/hero_gallery.png)

Dataset examples across states and seasons.

## Architecture

These binary road segmentation models use U-Net with an ImageNet-pretrained ResNet-18 encoder. They are trained on [ChesapeakeRSC v2](https://huggingface.co/datasets/isaaccorley/chesapeakersc-v2), which extends [Seeing the Roads Through the Trees](https://arxiv.org/abs/2401.06762). The training imagery is NAIP at its native 0.6 m resolution. Labels include roads obscured by tree canopy, so the models use surrounding image context to predict those road pixels.

| File | Bands | Input channels |
|---|---|---|
| `unet-resnet18-rgb.onnx` | R, G, B | 3 |
| `unet-resnet18-rgbn.onnx` | R, G, B, NIR | 4 |
| `unet-resnet18-rgb.pt` | R, G, B | 3 |
| `unet-resnet18-rgbn.pt` | R, G, B, NIR | 4 |

The `.pt` files contain state dictionaries for the same `segmentation_models_pytorch` U-Net architecture. Load them in PyTorch with:

```python
import torch
import segmentation_models_pytorch as smp

model = smp.Unet(
    encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=2
)
model.load_state_dict(
    torch.load("unet-resnet18-rgb.pt", map_location="cpu", weights_only=True)
)
model.eval()
```

RGB accepts three-band imagery; RGBN also uses the near-infrared band. Both ONNX exports accept any batch size and image dimensions that are multiples of 32, as required by the encoder's five downsampling stages. Outputs are `[batch, 2, H, W]` logits, with class 0 for background and class 1 for road. Apply `argmax(dim=1)` to obtain a class mask.

## Intended use

The models support road extraction and local fine-tuning with imagery at approximately 0.5–1 m per pixel. Evaluation covers 0.6 m NAIP imagery in the U.S. Mid-Atlantic; validate predictions for other sensors, resolutions, and regions. Label errors and seasonal differences can affect accuracy, including whether canopy-covered roads are visible in an image.

## ONNX inference

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession(
    "unet-resnet18-rgb.onnx", providers=["CPUExecutionProvider"]
)

image = np.random.rand(1, 3, 1024, 1024).astype(np.float32)  # Example input in [0, 1].
(logits,) = sess.run(None, {"image": image})
pred = logits.argmax(axis=1)  # [1, 1024, 1024], 0=background, 1=road
```

Divide actual uint8 image values by 255 before inference, matching the training normalization (mean 0, standard deviation 255).

## Training data

The core dataset contains 126,354 patches after quality filtering. Of 126,701 extracted patches, 347 were excluded because more than 5% of their imagery or labels were missing. Each patch is 1024 × 1024 pixels of 2021–2022 NAIP imagery at 0.6 m resolution, without resizing or cropping. Labels come from the CC0 Chesapeake Bay LULC 2024 Edition.

County-level splits contain 90,648 training, 11,349 validation, and 11,345 test patches. Delaware's 6,385 patches and West Virginia's 6,627 patches form separate out-of-region test sets. The main test split uses held-out counties from the DC, MD, NY, PA, and VA pool.

Training used AdamW with cosine annealing for 100 epochs and a batch size of 16. The ResNet-18 encoder was initialized with ImageNet weights.

## Evaluation

The following pixel-level scores (%) measure the road class on the held-out core test split.

| Model | Recall | Precision | F1 |
|---|---|---|---|
| RGB | 75.6 | 82.6 | 79.0 |
| RGBN | 79.0 | 78.1 | 78.5 |

RGBN has higher road recall, while RGB has higher precision and F1 on this split. The road class includes visible and canopy-covered pixels.

## Citation

Please cite the original ChesapeakeRSC paper:

```bibtex
@article{robinson2024seeing,
  title={Seeing the roads through the trees: A benchmark for modeling spatial dependencies with aerial imagery},
  author={Robinson, Caleb and Corley, Isaac and Ortiz, Anthony and Dodhia, Rahul and Ferres, Juan M Lavista and Najafirad, Peyman},
  journal={arXiv preprint arXiv:2401.06762},
  year={2024}
}
```
