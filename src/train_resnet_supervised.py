import os
import sys

from torch.nn import Linear

# Current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['DATA_PATH'] = dir_path + '/data'

import torch
import pytorch_lightning as pl

from torchvision.models import resnet18
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils import get_device
from models import LightningPredictor
from processing import svhn_train, svhn_test


train_loader = DataLoader(svhn_train, batch_size=32, shuffle=True, pin_memory=True)
test_loader = DataLoader(svhn_test, batch_size=32, shuffle=False, pin_memory=True)


model = resnet18(pretrained=True)
model.fc = Linear(model.fc.in_features, 10)  # SVHN has 10 classes

predictor = LightningPredictor(model)

trainer = pl.Trainer(
    max_epochs=10,
    default_root_dir='./logs',
    accelerator=get_device(as_str=True),
    callbacks=[
        ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
        LearningRateMonitor("epoch"),
    ],
)

trainer.fit(predictor, train_loader)