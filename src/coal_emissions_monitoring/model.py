from typing import Any, Dict, Tuple
from lightning import LightningModule
import torch
from loguru import logger

from coal_emissions_monitoring.constants import EMISSIONS_CATEGORIES
from coal_emissions_monitoring.ml_utils import preds_n_targets_to_categories


class CoalEmissionsModel(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        emissions_quantiles: Dict[float, float] = None,
        category_weights: Dict[int, float] = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.emissions_quantiles = emissions_quantiles
        self.category_weights = category_weights
        self.train_step_preds = []
        self.train_step_targets = []
        self.val_step_preds = []
        self.val_step_targets = []
        self.test_step_preds = []
        self.test_step_targets = []

        if self.emissions_quantiles is None:
            logger.warning(
                "No emissions quantiles provided, "
                "so emissions quantile metrics will not be calculated."
            )

    def loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # find the emissions categories in the targets
        _, targets_cat = preds_n_targets_to_categories(
            preds=preds,
            targets=targets,
            quantiles=self.emissions_quantiles,
            rescale=True,
        )
        # assign weights to each sample based on the categories
        weights = torch.tensor(
            [self.category_weights[int(cat)] for cat in targets_cat],
            dtype=torch.float,
            device=self.device,
        )
        # calculate the weighted mean squared error loss
        return (
            torch.nn.functional.mse_loss(
                preds.to(self.device), targets.to(self.device), reduction="none"
            )
            * weights
        ).mean()

    def forward(self, x):
        preds = self.model(x).squeeze(-1)
        # apply sigmoid to force predictions to be between 0 and 2
        preds = torch.nn.functional.sigmoid(preds) * 2
        return preds

    def calculate_categorical_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        metrics = dict()
        # convert emissions to categories
        preds_cat, targets_cat = preds_n_targets_to_categories(
            preds=preds,
            targets=targets,
            quantiles=self.emissions_quantiles,
            rescale=True,
        )
        # calculate emissions recall per category
        for cat in EMISSIONS_CATEGORIES.keys():
            metrics[f"{EMISSIONS_CATEGORIES[cat]}_category_recall"] = (
                (targets_cat[targets_cat == cat] == preds_cat[targets_cat == cat])
                .float()
                .mean()
            )
        return metrics

    def calculate_all_metrics(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate metrics for a batch of predictions and targets.

        Args:
            preds (torch.Tensor): predictions
            targets (torch.Tensor): targets

        Returns:
            Dict[str, float]: metrics
        """
        metrics = dict()
        # calculate mean squared error loss
        metrics["loss"] = self.loss(preds, targets)
        # calculate emissions vs no-emissions accuracy
        metrics["on_off_acc"] = ((preds > 0) == (targets > 0)).float().mean()
        if self.emissions_quantiles is not None:
            # calculate emissions quantile metrics
            preds_cat, targets_cat = preds_n_targets_to_categories(
                preds=preds,
                targets=targets,
                quantiles=self.emissions_quantiles,
                rescale=True,
            )
            # calculate emissions quantile aggregate accuracy
            metrics["agg_quant_acc"] = (preds_cat == targets_cat).float().mean()
            # calculate emissions recall per category
            metrics.update(self.calculate_categorical_metrics(preds, targets))
        return metrics

    def shared_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        stage: str,
    ):
        if len(batch["image"].shape) == 0:
            # avoid iteration over a 0-d array error
            return dict()
        metrics = dict()
        x, y = batch["image"], batch["target"]
        x, y = x.float().to(self.device), y.float().to(self.device)
        # forward pass (calculate predictions)
        y_pred = self(x)
        # add predictions and targets to list for epoch-level metrics
        preds = y_pred.detach().cpu().float().numpy()
        preds = getattr(self, f"{stage}_step_preds") + list(preds)
        setattr(self, f"{stage}_step_preds", preds)
        targets = y.detach().cpu().float().numpy()
        targets = getattr(self, f"{stage}_step_targets") + list(targets)
        setattr(self, f"{stage}_step_targets", targets)
        # clip the targets to the range [0, 2] since the model is trained
        # on emissions values in this range
        y = torch.clamp(y, 0, 2)
        # calculate metrics for the current batch
        metrics = self.calculate_all_metrics(preds=y_pred, targets=y)
        metrics = {
            (f"{stage}_{k}" if k != "loss" or stage != "train" else k): v
            for k, v in metrics.items()
        }
        # log metrics that don't need to be aggregated over the epoch
        for k, v in metrics.items():
            if k == "loss":
                self.log(k, v, on_step=True, prog_bar=True)
            elif "_category_recall" not in k:
                self.log(k, v, on_step=False, on_epoch=True, prog_bar=True)
        return metrics

    def shared_on_epoch_end(self, stage: str):
        # calculate epoch-level metrics
        preds = getattr(self, f"{stage}_step_preds")
        targets = getattr(self, f"{stage}_step_targets")
        metrics = self.calculate_categorical_metrics(
            preds=torch.tensor(preds), targets=torch.tensor(targets)
        )
        metrics = {f"{stage}_{k}": v for k, v in metrics.items()}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)
        # reset lists of predictions and targets
        setattr(self, f"{stage}_step_preds", [])
        setattr(self, f"{stage}_step_targets", [])

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="test")

    def on_train_epoch_end(self):
        self.shared_on_epoch_end(stage="train")

    def on_validation_epoch_end(self):
        self.shared_on_epoch_end(stage="val")

    def on_test_epoch_end(self):
        self.shared_on_epoch_end(stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=3
            ),
            "monitor": "val_loss",
        }
