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
        self.df = pd.read_csv(csv).values
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        name, target = self.df[idx]
        img = np.array(Image.open(os.path.join(self.path, name)))
        
        if self.transform:
            if isinstance(self.transform, A.BasicTransform) or isinstance(self.transform, A.Compose):
                img = self.transform(image=img)["image"]
            elif isinstance(self.transform, transforms.Compose):
                img = Image.fromarray(img)
                img = self.transform(img)


        return img, target
