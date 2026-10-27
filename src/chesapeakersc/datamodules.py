"""Load dataset splits and apply normalization and training augmentations."""

import kornia.augmentation as K
import torch
from torchgeo.datamodules.geo import NonGeoDataModule

from .datasets import ChesapeakeRSCv2

CUTOUT_CONFIGS = {
    "small": [K.RandomErasing(scale=(0.02, 0.06), ratio=(0.3, 3.3), value=0.0, p=0.5)],
    "medium": [K.RandomErasing(scale=(0.10, 0.25), ratio=(0.3, 3.3), value=0.0, p=0.5)],
    "large": [K.RandomErasing(scale=(0.25, 0.50), ratio=(0.3, 3.3), value=0.0, p=0.75)],
    "multi": [K.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2.0), value=0.0, p=0.7)]
    * 3,
}


class ChesapeakeRSCv2DataModule(NonGeoDataModule):
    mean = torch.tensor(0.0)
    std = torch.tensor(255.0)

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        differentiate_tree_canopy_over_roads: bool = False,
        cutout: str | None = None,
        test_split: str = "test",
        **kwargs,
    ) -> None:
        super().__init__(ChesapeakeRSCv2, batch_size, num_workers, **kwargs)
        self.differentiate_tree_canopy_over_roads = differentiate_tree_canopy_over_roads
        self.test_split = test_split

        if cutout and cutout not in CUTOUT_CONFIGS:
            raise ValueError(
                f"Unknown cutout '{cutout}'. Choose from: {list(CUTOUT_CONFIGS)}"
            )

        cutout_layers = CUTOUT_CONFIGS[cutout] if cutout else []

        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            *cutout_layers,
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            keepdim=True,
            data_keys=None,
        )
        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), keepdim=True, data_keys=None
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = ChesapeakeRSCv2(
                split="train",
                differentiate_tree_canopy_over_roads=self.differentiate_tree_canopy_over_roads,
                **self.kwargs,
            )
        if stage in ("fit", "validate"):
            self.val_dataset = ChesapeakeRSCv2(
                split="val",
                differentiate_tree_canopy_over_roads=self.differentiate_tree_canopy_over_roads,
                **self.kwargs,
            )
        if stage == "test":
            self.test_dataset = ChesapeakeRSCv2(
                split=self.test_split,
                differentiate_tree_canopy_over_roads=self.differentiate_tree_canopy_over_roads,
                **self.kwargs,
            )
