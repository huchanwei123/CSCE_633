import cv2
import numpy as np
import matplotlib.pyplot as plt
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
            data = np.asarray(data).astype(np.float32).reshape(1, 48, 48)
            #data = np.asarray(data).astype(np.float32).reshape(48, 48, 1)
        if self.transforms:
            data = self.transforms(data)
        
        #data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        #data = np.moveaxis(data, 2, 0)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data
        
    def imshow(self):
        # show random examples for each class
        pass
        #plt.imshow(np.transpose(npimg, (1, 2, 0)))
        #plt.show()
