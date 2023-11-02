import os
import sys
# Current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['DATA_PATH'] = dir_path + '/data'

import torch
import pytorch_lightning as pl

from torchvision.models import vit_b_16
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils import get_device
from models import LightningPredictor
from processing import rotated_svhn_train, rotated_svhn_test


train_loader = DataLoader(rotated_svhn_train, batch_size=32, shuffle=True, pin_memory=True)
test_loader = DataLoader(rotated_svhn_test, batch_size=32, shuffle=False, pin_memory=True)


# One class per rotation
vit = vit_b_16(num_classes=4)
_ = vit.to(get_device())

predictor = LightningPredictor(vit)

trainer = pl.Trainer(
    max_epochs=10, 
    default_root_dir='./logs/vit/',
    accelerator=get_device(as_str=True),
    callbacks=[
        ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
        LearningRateMonitor("epoch"),
    ],
)

trainer.fit(predictor, train_loader)

