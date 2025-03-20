import os
import torch
import numpy as np
import torch.nn as nn

from os.path import join
from PIL import Image
from torch.utils.data import Dataset


class LandDataset(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        super().__init__()
        self.img_path = [join(img_path, img) for img in os.listdir(img_path)
                         if img.endswith('.tif')]
        self.mask_path = [join(mask_path, mask) for mask in os.listdir(mask_path)
                          if mask.endswith('.tif')]
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = np.array(Image.open(self.img_path[idx]).convert('RGB'))
        mask = np.array(Image.open(self.mask_path[idx]), dtype=np.uint8)
        mask[mask != 0] = 1

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']

        # return img, mask, self.img_path[idx], self.mask_path[idx]
        return img, mask

if __name__ == '__main__':
    img_path = r'E:\Datasets\CAS_Landslide\Hokkaido\img\test'
    mask_path = r'E:\Datasets\CAS_Landslide\Hokkaido\mask\test'
    dataset = LandDataset(img_path, mask_path)
    img, mask, img_path, mask_path = dataset[0]
    print(img.shape)
    print(mask.shape)
    print(img_path)
    print(mask_path)