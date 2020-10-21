import pdb
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

EMOTION_MAP = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
               4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

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
            
        if self.transforms:
            data = self.transforms(data)      

        #data = cv2.cvtColor(data, cv2.COLOR_GRAY2RGB)
        #data = np.moveaxis(data, 2, 0)

        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

    def data_distribution(self):
        # Help calculate data distribution
        unique_label = set(self.y)
        total_samples = 0
        for i in unique_label:
            samples_of_class = np.sum(self.y == i)
            total_samples += samples_of_class
            print('%s: %d samples' % (EMOTION_MAP[i], samples_of_class))
        assert total_samples == len(self.X)
        print('=> Total %d samples\n' % total_samples)
    
    def show_random_img(self, spc):
        # spc: samples per class
        # show random examples for each class
        class_num = len(set(self.y))
        # initial figure
        
        fig, axs = plt.subplots(spc, class_num,
                                figsize=(15, spc*2))
        
        for c in range(class_num):
            idx = np.where(self.y == c)[0]
            random_img = self.X[np.random.choice(idx, spc)]
            axs[0, c].set_title(EMOTION_MAP[c])
            for i in range(spc):
                img = random_img[i].reshape(48, 48)
                axs[i, c].axis('off')
                axs[i, c].imshow(img, cmap='gray', vmin=0, vmax=255)
