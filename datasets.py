import numpy as np
from torch.utils.data import Dataset
import random

def transform(image, label):
    tmp = random.random()
    if tmp > 0.875:
        # vertical flipping
        image, label = image[:, ::-1], label[:, ::-1]
    elif tmp > 0.75:
        # horizontal flipping
        image, label = image[:, :, ::-1], label[:, :, ::-1]
    elif tmp > 0.625:
        # vertical and horizontal flipping
        image, label = image[:, :-1, ::-1], label[:, :-1, ::-1]
    elif tmp > 0.5:
        image, label = np.rot90(image, 1, axes=(1, 2)), np.rot90(label, 1, axes=(1, 2))
    elif tmp > 0.375:
        image, label = np.rot90(image, 3, axes=(1, 2)), np.rot90(label, 3, axes=(1, 2))
    elif tmp > 0.25:
        image, label = np.rot90(image, 1, axes=(1, 2))[:, ::-1], np.rot90(label, 1, axes=(1, 2))[:, ::-1]
    elif tmp > 0.125:
        image, label = np.rot90(image, 3, axes=(1, 2))[:, ::-1], np.rot90(label, 3, axes=(1, 2))[:, ::-1]
    return image, label

class AxialNPY(Dataset):
    def __init__(self, recon_low_vertical, recon_low_horizontal, recon_low, augmentation=False):
        self.recon_low_vertical = np.load(recon_low_vertical) if isinstance(recon_low_vertical, str) else recon_low_vertical.copy()
        self.recon_low_horizontal = np.load(recon_low_horizontal) if isinstance(recon_low_horizontal, str) else recon_low_horizontal.copy()
        self.recon_low = np.load(recon_low) if isinstance(recon_low, str) else recon_low.copy()
        self.augmentation = augmentation
        self.l = self.recon_low_vertical.shape[0]

    def __len__(self):
        return self.l * 2

    def __getitem__(self, idx):
        if idx < self.l:
            image = self.recon_low_vertical[idx]
            label = self.recon_low[idx]
        else:
            # resolution enhancemnt always in horizontal direction
            # rotate the horizontally down- upscaled images by 90 degree
            image = self.recon_low_horizontal[idx - self.l]
            label = self.recon_low[idx-self.l]
            image = np.rot90(image).copy()
            label = np.rot90(label).copy()
        image = np.expand_dims(image, (0))
        label = np.expand_dims(label, (0))

        if self.augmentation:
            transform(image, label)
        return image, label

class SaggitalNPY(Dataset):
    def __init__(self, recon_low):
        self.recon_low = np.load(recon_low) if isinstance(recon_low, str) else recon_low.copy()
        self.l = self.recon_low.shape[2]

    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        image = self.recon_low[:, :, idx]
        image = np.expand_dims(image, (0))
        return image

class CoronalNPY(Dataset):
    def __init__(self, recon_low):
        self.recon_low = np.load(recon_low) if isinstance(recon_low, str) else recon_low.copy()
        self.l = self.recon_low.shape[1]

    def __len__(self):
        return self.l

    def __getitem__(self, idx):
        image = self.recon_low[:, idx, :]
        image = np.expand_dims(image, (0))
        return image
