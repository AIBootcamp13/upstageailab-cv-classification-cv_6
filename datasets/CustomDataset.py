__all__ = ['CustomImageDataset']
import os
from PIL import Image
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A


class CustomImageDataset(Dataset):
    def __init__(self, csv, path, transform=None):
        self.df = pd.read_csv(csv)
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = row["ID"]
        target = row["target"]
        
        img = np.array(Image.open(os.path.join(self.path, name)))
        
        if self.transform:
            img = self.transform(img)


        return img, target, name
