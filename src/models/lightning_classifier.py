import torch
import pytorch_lightning as pl

import torch.nn.functional as F
import torchmetrics

from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


class LightningClassifier(pl.LightningModule):
    """
    Wrapper class for any model that inherits from nn.Module. 
    Makes it compatible with PyTorch Lightning.
    """

    def __init__(self, model, num_classes):
        super(LightningClassifier, self).__init__()
        self.model = model
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_acc_step', self.train_acc, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.val_acc(y_hat, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        self.test_acc(y_hat, y)

        self.log('test_loss', loss, sync_dist=True)
        self.log('test_acc', self.test_acc, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-3, weight_decay=0.3)
        scheduler = CosineAnnealingLR(optimizer, 80000)
        return [optimizer], [scheduler]