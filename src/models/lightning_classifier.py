import torch
import pytorch_lightning as pl

import torch.nn.functional as F

from torch.optim.lr_scheduler import ReduceLROnPlateau

class LightningClassifier(pl.LightningModule):
    """
    Wrapper class for any model that inherits from nn.Module. 
    Makes it compatible with PyTorch Lightning.
    """

    def __init__(self, model):
        super(LightningClassifier, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('test_loss', loss)
        self.log('test_acc', (y_hat.argmax(dim=1) == y).float().mean().item())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = ReduceLROnPlateau(optimizer, patience=2)
        return optimizer