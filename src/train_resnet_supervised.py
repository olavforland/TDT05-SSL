import os
import sys

from torch.nn import Linear

# Current directory
dir_path = os.path.dirname(os.path.realpath(__file__))
os.environ['DATA_PATH'] = dir_path + '/data'
os.environ['LOG_PATH'] = dir_path + '/logs'
os.environ['CHECKPOINT_PATH'] = dir_path + '/checkpoints'

import pytorch_lightning as pl

from torchvision.models import resnet18
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import get_device
from models import LightningClassifier
from processing import svhn_train, svhn_test, svhn_val

batch_size = 32
train_loader = DataLoader(svhn_train, batch_size=batch_size, num_workers=8, shuffle=True, pin_memory=True)
val_loader = DataLoader(svhn_val, batch_size=batch_size, num_workers=8, pin_memory=True)
test_loader = DataLoader(svhn_test, batch_size=batch_size, pin_memory=True)

model = resnet18(weights=None)
num_classes = 10
model.fc = Linear(model.fc.in_features, num_classes)  # SVHN has 10 classes

classifier = LightningClassifier(model, num_classes)

checkpoint_callback = ModelCheckpoint(dirpath=os.environ['CHECKPOINT_PATH'] + "/resnet/", save_weights_only=True, mode="min", monitor="val_loss")
logger = TensorBoardLogger(os.environ['LOG_PATH'] + '/resnet/tb_logs', name="resnet")

supervised_trainer = pl.Trainer(
    max_epochs=30,
    num_nodes=2,
    logger=logger,
    default_root_dir=os.environ['LOG_PATH'] + '/resnet/',
    accelerator=get_device(as_str=True),
    callbacks=[
        checkpoint_callback,
        LearningRateMonitor("epoch"),
    ]
)

supervised_trainer.fit(classifier, train_loader, val_loader)

# Retrieve best model weights
classifier = LightningClassifier.load_from_checkpoint(checkpoint_callback.best_model_path, model=classifier.model, num_classes=num_classes)

results = supervised_trainer.test(classifier, test_loader)

print("Supervised training results:")
print(results)