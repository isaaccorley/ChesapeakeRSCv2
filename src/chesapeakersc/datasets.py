"""Load ChesapeakeRSC v2 image and target pairs from a training manifest.

dataset/build_manifest.py records the paths and splits. Targets created by
dataset/create_targets.py use 0 for background, 1 for road, 2 for canopy-covered road,
and 255 for ignored pixels. RGB reads the first three bands; RGBN reads all four.
"""

import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import torch
from matplotlib.colors import ListedColormap
from torchgeo.datasets import NonGeoDataset

IGNORE_INDEX = 255


class ChesapeakeRSCv2(NonGeoDataset):
    def __init__(
        self,
        manifest,
        split="train",
        dataset="core",
        bands="rgbn",
        differentiate_tree_canopy_over_roads=False,
        transforms=None,
        leaf_state=None,
    ):
        assert split in ("train", "val", "test", "test_de", "test_wv")
        assert bands in ("rgb", "rgbn")
        assert dataset in ("core", "legacy", "both")
        self.bands = bands
        self.n_channels = 3 if bands == "rgb" else 4
        self.differentiate_tree_canopy_over_roads = differentiate_tree_canopy_over_roads
        self.transforms = transforms

        df = pd.read_parquet(manifest) if isinstance(manifest, str) else manifest
        df = df[df.split == split]
        if dataset != "both":
            df = df[df.dataset == dataset]
        if leaf_state is not None:
            df = df[df.leaf_state == leaf_state]
        self.df = df.reset_index(drop=True)

        colors = [(0, 0, 0), (1, 1, 1)]
        if differentiate_tree_canopy_over_roads:
            colors.append((1, 0, 0))
        self._cmap = ListedColormap(colors)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[index]
        with rasterio.open(row.image_path) as f:
            image = f.read()[: self.n_channels]
            image = torch.from_numpy(image).float()
        with rasterio.open(row.target_path) as f:
            mask = f.read(1)
            mask = torch.from_numpy(mask.astype("int64"))
        if not self.differentiate_tree_canopy_over_roads:
            mask = torch.where(mask == 2, torch.tensor(1), mask)
        sample = {"image": image, "mask": mask}
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def plot(self, sample, show_titles=True, suptitle=None):
        img = sample["image"].numpy().transpose(1, 2, 0)
        mask = sample["mask"].numpy()
        n_cols = 3 if "prediction" in sample else 2
        fig, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        axs[0].imshow(img[:, :, :3].astype("uint8"))
        valid = mask != IGNORE_INDEX
        m = mask.copy()
        m[~valid] = 0
        axs[1].imshow(
            m, vmin=0, vmax=self._cmap.N - 1, cmap=self._cmap, interpolation="none"
        )
        if "prediction" in sample:
            axs[2].imshow(
                sample["prediction"].numpy(),
                vmin=0,
                vmax=self._cmap.N - 1,
                cmap=self._cmap,
                interpolation="none",
            )
        if show_titles:
            axs[0].set_title("Image")
            axs[1].set_title("Target")
            if "prediction" in sample:
                axs[2].set_title("Prediction")
        for ax in axs:
            ax.axis("off")
        if suptitle:
            plt.suptitle(suptitle)
        plt.tight_layout()
        return fig
