"""Train and evaluate road segmentation models with PyTorch Lightning."""

import warnings
from typing import Literal, TypedDict

import segmentation_models_pytorch as smp
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchgeo.datasets import RGBBandsMissingError, unbind_samples
from torchgeo.models import FCN, get_weight
from torchgeo.tasks import SemanticSegmentation
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, FBetaScore, Precision, Recall
from torchmetrics.wrappers import ClasswiseWrapper
from torchvision.models._api import WeightsEnum

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


class MetricOptions(TypedDict):
    task: Literal["multiclass"]
    num_classes: int
    ignore_index: int | None


class ChesapeakeRSCv2Task(SemanticSegmentation):
    def __init__(self, *args, tmax=50, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hparams["tmax"] = tmax

    def configure_optimizers(self):
        tmax = int(self.hparams.get("tmax", 50))
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"])
        scheduler = CosineAnnealingLR(optimizer, T_max=tmax, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }

    def configure_metrics(self) -> None:
        num_classes: int = self.hparams["num_classes"]
        ignore_index: int | None = self.hparams["ignore_index"]
        kw: MetricOptions = {
            "task": "multiclass",
            "num_classes": num_classes,
            "ignore_index": ignore_index,
        }
        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    **kw, average="micro", multidim_average="global"
                ),
                "OverallPrecision": Precision(
                    **kw, average="micro", multidim_average="global"
                ),
                "OverallRecall": Recall(
                    **kw, average="micro", multidim_average="global"
                ),
                "OverallF1Score": FBetaScore(
                    **kw, beta=1.0, average="micro", multidim_average="global"
                ),
                "Accuracy": ClasswiseWrapper(
                    Accuracy(**kw, average="none", multidim_average="global")
                ),
                "Precision": ClasswiseWrapper(
                    Precision(**kw, average="none", multidim_average="global")
                ),
                "Recall": ClasswiseWrapper(
                    Recall(**kw, average="none", multidim_average="global")
                ),
                "F1Score": ClasswiseWrapper(
                    FBetaScore(
                        **kw, beta=1.0, average="none", multidim_average="global"
                    )
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def configure_models(self) -> None:
        model: str = self.hparams["model"]
        backbone: str = self.hparams["backbone"]
        weights = self.weights
        in_channels: int = self.hparams["in_channels"]
        num_classes: int = self.hparams["num_classes"]

        common = dict(
            encoder_name=backbone,
            encoder_weights="imagenet" if weights is True else None,
            in_channels=in_channels,
            classes=num_classes,
        )
        if model == "unet":
            self.model = smp.Unet(**common)
        elif model == "unet++":
            self.model = smp.UnetPlusPlus(**common)
        elif model == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(**common)
        elif model == "upernet":
            self.model = smp.UPerNet(**common)
        elif model == "segformer":
            self.model = smp.Segformer(**common)
        elif model == "fcn":
            self.model = FCN(
                in_channels=in_channels,
                classes=num_classes,
                num_filters=self.hparams["num_filters"],
            )
        else:
            raise ValueError(
                f"Model type '{model}' is not supported in v2's task "
                "(unet, unet++, deeplabv3+, upernet, segformer, fcn)."
            )

        if model != "fcn" and weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            encoder = self.model.encoder
            assert isinstance(encoder, torch.nn.Module)
            encoder.load_state_dict(state_dict)

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        y_hat_hard = torch.softmax(y_hat, dim=1).argmax(dim=1)
        self.train_metrics(y_hat_hard, y)
        self.log_dict(dict(self.train_metrics.compute()))
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, on_epoch=True)
        y_hat_hard = torch.softmax(y_hat, dim=1).argmax(dim=1)
        self.val_metrics(y_hat_hard, y)
        self.log_dict(dict(self.val_metrics.compute()), on_epoch=True)

        datamodule = getattr(self.trainer, "datamodule", None)
        plot = getattr(datamodule, "plot", None)
        experiment = getattr(self.logger, "experiment", None)
        add_figure = getattr(experiment, "add_figure", None)
        if batch_idx < 10 and callable(plot) and callable(add_figure):
            plot_batch = dict(batch)
            plot_batch["prediction"] = y_hat_hard
            for k in ("image", "mask", "prediction"):
                plot_batch[k] = plot_batch[k].cpu()
            sample = unbind_samples(plot_batch)[0]
            try:
                fig = plot(sample)
                add_figure(f"image/{batch_idx}", fig, global_step=self.global_step)
                import matplotlib.pyplot as plt

                plt.close(fig)
            except RGBBandsMissingError:
                pass

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch["image"], batch["mask"]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        y_hat_hard = torch.softmax(y_hat, dim=1).argmax(dim=1)
        self.test_metrics(y_hat_hard, y)
        self.log_dict(dict(self.test_metrics.compute()), on_epoch=True)
