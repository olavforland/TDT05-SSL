import torch
from torch.utils.data import random_split


def get_device(as_str=False):
    device_str = 'cpu'
    if torch.cuda.is_available():
        device_str = 'cuda'
    elif torch.backends.mps.is_available():
        device_str = 'mps'
    print('Using device:', device_str)
    if as_str:
        return device_str
    return torch.device(device_str)


