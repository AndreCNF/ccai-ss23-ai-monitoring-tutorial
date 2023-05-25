from typing import Any, Dict
from lightning import LightningModule
import torch
import torchmetrics


class CoalEmissionsModel(LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        preds = self.model(x).squeeze(-1)
        return preds

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
        # calculate the cross entropy loss
        metrics["loss"] = self.loss(preds, targets)
        # apply sigmoid to the predictions to get a value between 0 and 1
        preds = torch.sigmoid(preds)
        # calculate emissions vs no-emissions accuracy
        metrics["accuracy"] = ((preds > 0) == (targets > 0)).float().mean()
        # calculate balanced accuracy, which accounts for class imbalance
        metrics["balanced_accuracy"] = torchmetrics.functional.balanced_accuracy(
            preds=preds, target=targets
        )
        # calculate recall and precision
        metrics["recall"] = torchmetrics.functional.recall(
            preds=preds, target=targets, average="macro"
        )
        metrics["precision"] = torchmetrics.functional.precision(
            preds=preds, target=targets, average="macro"
        )
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
        # calculate metrics for the current batch
        metrics = self.calculate_all_metrics(preds=y_pred, targets=y)
        metrics = {
            (f"{stage}_{k}" if k != "loss" or stage != "train" else k): v
            for k, v in metrics.items()
        }
        # log metrics
        for k, v in metrics.items():
            if k == "loss":
                self.log(k, v, on_step=True, prog_bar=True)
            else:
                self.log(k, v, on_step=False, on_epoch=True, prog_bar=True)
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
