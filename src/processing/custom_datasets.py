import torch
import torchvision.transforms as transforms
from torchvision.datasets import SVHN

class RotatedSVHN(SVHN):
    """
    Wrapper class for SVHN dataset that returns rotated images.
    """
    def __init__(self, root, split, transform, download):
        super().__init__(root=root, split=split, transform=transform, download=download)
        self.transform = transform
        self.rotations = [0, 90, 180, 270]

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        
        rotation_label = torch.randint(0, len(self.rotations), size=(1,)).item()
        rotation_angle = self.rotations[rotation_label]
        
        rotated_img = transforms.functional.rotate(img, rotation_angle)

        return rotated_img, rotation_label


    def __len__(self):
        return super().__len__()
