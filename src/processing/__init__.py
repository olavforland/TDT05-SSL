import os
import torchvision.transforms as transforms

from .custom_datasets import RotatedSVHN

# Transformations needed to be compatible with vit_b_16 model from torchvision 
# Found @ https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html#torchvision.models.vit_b_16
vit_b_16_transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    rotated_svhn_train = RotatedSVHN(root=os.environ['DATA_PATH'], split='train', transform=vit_b_16_transform, download=True)
    rotated_svhn_test = RotatedSVHN(root=os.environ['DATA_PATH'], split='test', transform=vit_b_16_transform, download=True)

except:
    raise Exception('DATAPATH environment variable not set. Please set it to the path where the SVHN dataset is stored.')