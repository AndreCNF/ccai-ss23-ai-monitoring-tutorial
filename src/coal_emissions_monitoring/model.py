from typing import Any, Dict
from lightning import LightningModule
import torch
from loguru import logger

from coal_emissions_monitoring.constants import EMISSIONS_CATEGORIES


def emissions_to_category(
    emissions: float, quantiles: Dict[float, float], rescale: bool = False
) -> int:
    """
    Convert emissions to a category based on quantiles. The quantiles are
    calculated from the training data. Here's how the categories are defined:
    - 0: no emissions
    - 1: low emissions
    - 2: medium emissions
    - 3: high emissions
    - 4: very high emissions

    Args:
        emissions (float): emissions value
        quantiles (Dict[float, float]): quantiles to use for categorization
        rescale (bool): whether to rescale emissions to the original range,
            using the 99th quantile as the maximum value

    Returns:
        int: category
    """
    if rescale:
        emissions = emissions * quantiles[0.99]
    if emissions <= 0:
        return 0
    elif emissions <= quantiles[0.3]:
        return 1
    elif emissions > quantiles[0.3] and emissions <= quantiles[0.6]:
        return 2
    elif emissions > quantiles[0.6] and emissions <= quantiles[0.99]:
        return 3
    else:
        return 4


class CoalEmissionsModel(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        emissions_quantiles: Dict[float, float] = None,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.emissions_quantiles = emissions_quantiles
        self.loss = torch.nn.MSELoss()
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

    def forward(self, x):
        preds = self.model(x).squeeze()
        # apply ReLU to ensure predictions are non-negative
        preds = torch.nn.functional.relu(preds)
        return preds

    def calculate_metrics(
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
            targets_cat = torch.tensor(
                [
                    emissions_to_category(y_i, self.emissions_quantiles, rescale=True)
                    for y_i in targets
                ]
            ).to(self.device)
            preds_cat = torch.tensor(
                [
                    emissions_to_category(
                        y_pred_i, self.emissions_quantiles, rescale=True
                    )
                    for y_pred_i in preds
                ]
            ).to(self.device)
            # calculate emissions quantile aggregate accuracy
            metrics["agg_quant_acc"] = (preds_cat == targets_cat).float().mean()
            # calculate emissions recall per category
            for cat in EMISSIONS_CATEGORIES.keys():
                metrics[f"{EMISSIONS_CATEGORIES[cat]}_category_recall"] = (
                    (targets_cat[targets_cat == cat] == preds_cat[targets_cat == cat])
                    .float()
                    .mean()
                )
        return metrics

    def shared_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        stage: str,
    ):
        metrics = dict()
        x, y = batch["image"], batch["target"]
        x, y = x.float().to(self.device), y.float().to(self.device)
        y_pred = self(x)
        metrics = self.calculate_metrics(preds=y_pred, targets=y)
        metrics = {
            (f"{stage}_{k}" if k != "loss" or stage != "train" else k): v
            for k, v in metrics.items()
        }
        for k, v in metrics.items():
            if k == "loss":
                self.log(k, v, on_step=True, prog_bar=True)
            elif "_category_recall" not in k:
                self.log_dict(metrics, on_epoch=True, prog_bar=True)
        return metrics

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        return self.shared_step(batch, batch_idx, stage="test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=3
            ),
            "monitor": "val_loss",
        }
