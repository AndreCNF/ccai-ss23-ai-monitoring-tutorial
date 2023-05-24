from typing import Any, Dict, Tuple
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


def preds_n_targets_to_categories(
    preds: torch.Tensor,
    targets: torch.Tensor,
    quantiles: Dict[float, float],
    rescale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert emissions to a category based on quantiles. The quantiles are
    calculated from the training data. Here's how the categories are defined:
    - 0: no emissions
    - 1: low emissions
    - 2: medium emissions
    - 3: high emissions
    - 4: very high emissions

    Args:
        preds (torch.Tensor): emissions predictions
        targets (torch.Tensor): emissions targets
        quantiles (Dict[float, float]): quantiles to use for categorization
        rescale (bool): whether to rescale emissions to the original range,
            using the 99th quantile as the maximum value

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of predictions and targets
    """
    preds_cat = torch.tensor(
        [
            emissions_to_category(y_pred_i, quantiles, rescale=rescale)
            for y_pred_i in preds
        ]
    ).to(preds.device)
    targets_cat = torch.tensor(
        [emissions_to_category(y_i, quantiles, rescale=rescale) for y_i in targets]
    ).to(targets.device)
    return preds_cat, targets_cat


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
