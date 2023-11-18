import os
import sys

import torchmetrics

# Current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['DATA_PATH'] = dir_path + '/data'
os.environ['LOG_PATH']  = dir_path + '/logs'
os.environ['CHECKPOINT_PATH'] = dir_path + '/checkpoints'

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import get_device
from models import LightningClassifier
from processing import rotated_svhn_train, rotated_svhn_val, svhn_train, svhn_test, rotated_svhn_test

# ------------- Self-supervised pretraining -------------

num_workers = 2
test_loader = DataLoader(rotated_svhn_test, batch_size=32, shuffle=False, pin_memory=True)

# Load pretrained model
vit = vit_b_16(num_classes=4)
classifier = LightningClassifier.load_from_checkpoint(os.environ['CHECKPOINT_PATH'] + "/vit/epoch=230-step=52899.ckpt", model=vit, num_classes=4)

logger = TensorBoardLogger(os.environ['LOG_PATH'] + '/vit/tb_logs/pretrain', name="vit")

self_supervised_trainer = pl.Trainer(
    max_epochs=300,
    num_nodes=2,
    gradient_clip_val=1,
    logger=True,
    default_root_dir=os.environ['LOG_PATH'] + '/vit',
    accelerator=get_device(as_str=True),
    callbacks=[
        # checkpoint_callback,
        LearningRateMonitor("epoch"),
    ],
)

results = self_supervised_trainer.test(classifier, test_loader)

print("Supervised pre-training results:")
print(results)


