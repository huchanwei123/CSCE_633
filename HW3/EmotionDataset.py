import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None, resize=False):
        self.X = images
        self.y = labels
        self.transforms = transforms
        self.resize = resize
    
    def __len__(self):
        return (len(self.X))

    def __getitem__(self, i):
        data = self.X[i, :]
        if self.resize:
            data = np.asarray(data).astype(np.uint8).reshape(48, 48)

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data
