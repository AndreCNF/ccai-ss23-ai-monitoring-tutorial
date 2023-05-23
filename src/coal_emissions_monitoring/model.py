from typing import Any, Dict
from lightning import LightningModule
import torch
from loguru import logger

from coal_emissions_monitoring.constants import EMISSIONS_CATEGORIES


def emissions_to_category(emissions: float, quantiles: Dict[float, float]) -> int:
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

    Returns:
        int: category
    """
    if emissions <= 0:
        return 0
    elif emissions <= quantiles[0.3]:
        return 1
    elif emissions > quantiles[0.3] and emissions <= quantiles[0.6]:
        return 2
    elif emissions > quantiles[0.6] and emissions <= quantiles[0.95]:
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

    def shared_step(
        self,
        batch: Dict[str, Any],
        batch_idx: int,
        stage: str,
    ):
        x, y = batch["image"], batch["target"]
        x, y = x.float().to(self.device), y.float().to(self.device)
        y_pred = self(x)
        # calculate mean squared error loss
        loss = self.loss(y_pred, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        # calculate emissions vs no-emissions accuracy
        on_off_acc = ((y_pred > 0) == (y > 0)).float().mean()
        self.log(f"{stage}_on_off_acc", on_off_acc, prog_bar=True)
        if self.emissions_quantiles is not None:
            # calculate emissions quantile metrics
            y_cat = torch.tensor(
                [emissions_to_category(y_i, self.emissions_quantiles) for y_i in y]
            ).to(self.device)
            y_pred_cat = torch.tensor(
                [
                    emissions_to_category(y_pred_i, self.emissions_quantiles)
                    for y_pred_i in y_pred
                ]
            ).to(self.device)
            # calculate emissions quantile aggregate accuracy
            agg_quant_acc = (y_cat == y_pred_cat).float().mean()
            self.log(f"{stage}_agg_quant_acc", agg_quant_acc, prog_bar=True)
            # calculate emissions quantile per-category accuracy
            for cat in EMISSIONS_CATEGORIES.keys():
                acc = (y_cat[y_cat == cat] == y_pred_cat[y_cat == cat]).float().mean()
                self.log(
                    f"{stage}_{EMISSIONS_CATEGORIES[cat]}_category_acc",
                    acc,
                    prog_bar=True,
                )
        return loss

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
