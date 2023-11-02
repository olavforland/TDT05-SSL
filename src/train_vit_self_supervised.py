import os
import sys
# Current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['DATA_PATH'] = dir_path + '/data'
os.environ['LOG_PATH']  = dir_path + '/logs'
os.environ['WEIGHTS_PATH'] = dir_path + '/weights'

import torch
import pytorch_lightning as pl

from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from utils import get_device
from models import LightningClassifier
from processing import rotated_svhn_train, rotated_svhn_test


# Split train_loader into train and val
N = len(rotated_svhn_train)
split = [int(0.9 * N), int(0.1 * N)]
train_dataset, val_dataset = random_split(rotated_svhn_train, split)

# Use 1% of training data for finetuning, rest for pretraining
pretrain_split = [int(0.99 * split[0]), int(0.01 * split[0])]
pretrain_dataset, finetune_dataset = random_split(train_dataset, pretrain_split)

pretrain_loader = DataLoader(pretrain_dataset, batch_size=32, shuffle=True, pin_memory=True)
finetune_loader = DataLoader(finetune_dataset, batch_size=32, shuffle=True, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)
test_loader = DataLoader(rotated_svhn_test, batch_size=32, shuffle=False, pin_memory=True)


# One class per rotation
vit = vit_b_16(num_classes=4)
_ = vit.to(get_device())

classifier = LightningClassifier(vit)

trainer = pl.Trainer(
    max_epochs=30, 
    default_root_dir=os.environ['LOG_PATH'] + '/vit',
    accelerator=get_device(as_str=True),
    callbacks=[
        ModelCheckpoint(dirpath=os.environ['WEIGHT_PATH'], save_weights_only=True, mode="min", monitor="val_loss"),
        LearningRateMonitor("epoch"),
    ],
)

trainer.fit(classifier, pretrain_loader, val_loader)

