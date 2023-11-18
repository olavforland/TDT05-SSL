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


# ------------- Supervised finetuning -------------

# Split train_loader into train and val
# N = len(svhn_train)
# split = [int(0.01 * N), int(0.2 * N), N - (int(0.01 * N) + int(0.2 * N))]
split = [0.01, 0.2, 0.79]
train_dataset, val_dataset, *_ = random_split(svhn_train, split)

train_loader = DataLoader(train_dataset, batch_size=32, num_workers=num_workers, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, num_workers=num_workers, shuffle=False, pin_memory=True)

test_loader = DataLoader(svhn_test, batch_size=32, shuffle=False, pin_memory=True)

# Load pretrained model
vit = vit_b_16(num_classes=4)
classifier = LightningClassifier.load_from_checkpoint(os.environ['CHECKPOINT_PATH'] + "/vit/epoch=230-step=52899.ckpt", model=vit, num_classes=10)


# Freeze all layers
for param in classifier.model.parameters():
    param.requires_grad = False

# Replace head
head = classifier.model.heads.head
classifier.model.heads.head = nn.Linear(in_features=head.in_features, out_features=10, bias=True)

supervised_checkpoint = ModelCheckpoint(dirpath=os.environ['CHECKPOINT_PATH'], save_weights_only=True, mode="min", monitor="val_loss")
logger = TensorBoardLogger(os.environ['LOG_PATH'] + '/vit/tb_logs/finetune', name="vit")

# Train supervised
supervised_trainer = pl.Trainer(
    max_epochs=10,
    num_nodes=2,
    logger=logger,
    default_root_dir=os.environ['LOG_PATH'],
    accelerator=get_device(as_str=True),
    callbacks=[
        supervised_checkpoint,
        LearningRateMonitor("epoch"),
    ],
)
supervised_trainer.fit(classifier, train_loader, val_loader)

# Retrieve best model weights
classifier = LightningClassifier.load_from_checkpoint(supervised_checkpoint.best_model_path, model=classifier.model, num_classes=10)

results = supervised_trainer.test(classifier, test_loader)

print("Supervised fine-tuning results:")
print(results)


