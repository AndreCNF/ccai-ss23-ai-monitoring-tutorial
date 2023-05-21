from typing import Any, Dict
from lightning import LightningModule
import torch


class CoalEmissionsModel(LightningModule):
    def __init__(self, model: torch.nn.Module, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x).squeeze()

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
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        # calculate emissions vs no-emissions accuracy
        acc = ((y_pred > 0) == (y > 0)).float().mean()
        self.log(f"{stage}_acc", acc, on_epoch=True, prog_bar=True)
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
                optimizer, mode="min", factor=0.1, patience=5
            ),
            "monitor": "val_loss",
        }
