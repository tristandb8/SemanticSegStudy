import os
import shutil
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, preprocessor=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.preprocessor = preprocessor
        self.augment = augment
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self.apply_augmentation(image, mask)

        if self.preprocessor:
            image = self.preprocessor(image)
            mask = torch.from_numpy(np.array(mask)).long()
        
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(), size=(128, 128), mode='nearest').squeeze(0).squeeze(0).long()

        return image, mask

    def apply_augmentation(self, image, mask):
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # Random color jitter
        if random.random() > 0.5:
            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))
        
        return image, mask