import os
import torchvision.transforms as transforms
from torchvision.datasets import SVHN

from .custom_datasets import RotatedSVHN

# Transformations needed to be compatible with vit_b_16 model and ResNet from torchvision
# Found @ https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.vit_b_16
# and @ https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html#torchvision.models.resnet18
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    rotated_svhn_train = RotatedSVHN(root=os.environ['DATA_PATH'], split='train', transform=transform, download=True)
    rotated_svhn_test = RotatedSVHN(root=os.environ['DATA_PATH'], split='test', transform=transform, download=True)
    svhn_train = SVHN(root=os.environ['DATA_PATH'], split='test', transform=transform, download=True)
    svhn_test = SVHN(root=os.environ['DATA_PATH'], split='test', transform=transform, download=True)
except KeyError:
    raise Exception('DATA_PATH environment variable not set. Please set it to the path where the SVHN dataset is stored.')
except:
    raise Exception('Error loading SVHN dataset.')