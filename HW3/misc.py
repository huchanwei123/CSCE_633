import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import hog

# read the csv file
def read_data(data_dir):
    # start time 
    t0 = time.process_time()
    
    train_data = pd.read_csv(os.path.join(data_dir, 'Q2_Train_Data.csv'))
    val_data = pd.read_csv(os.path.join(data_dir, 'Q2_Validation_Data.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'Q2_Test_Data.csv'))

    # get data and labels
    # training data
    train_img = train_data.iloc[:, 1].apply(lambda x: x.split())
    train_img = [list(map(int, train_img[i])) for i in range(len(train_data))]
    train_img = np.asarray(train_img, dtype=np.float32)
    
    # HOG
    train_img_HOG = np.zeros(np.shape(train_img))
    for i in range(len(train_img)):
        img = np.asarray(train_img[i]).reshape(48, 48)
        fd, data = hog(img, orientations=4, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True)
        train_img_HOG[i] = np.asarray(data).astype(np.float32).reshape(48*48)
        print('Training HOG done')
    train_img = np.asarray(train_img_HOG, dtype=np.float32)
    train_label = train_data.iloc[:, 0]

    # validation data
    val_img = val_data.iloc[:, 1].apply(lambda x: x.split())
    val_img = [list(map(int, val_img[i])) for i in range(len(val_data))]
    val_img = np.asarray(val_img, dtype=np.float32)
    # HOG
    val_img_HOG = np.zeros(np.shape(val_img))
    for i in range(len(val_img)):
        img = np.asarray(val_img[i]).reshape(48, 48)
        fd, data = hog(img, orientations=4, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True)
        val_img_HOG[i] = np.asarray(data).astype(np.float32).reshape(48*48)
    val_img = np.asarray(val_img_HOG, dtype=np.float32)
    val_label = val_data.iloc[:, 0]

    # testing data
    test_img = test_data.iloc[:, 1].apply(lambda x: x.split())
    test_img = [list(map(int, test_img[i])) for i in range(len(test_data))]
    test_img = np.asarray(test_img, dtype=np.float32)
    # HOG
    test_img_HOG = np.zeros(np.shape(test_img))
    for i in range(len(test_img)):
        img = np.asarray(test_img[i]).reshape(48, 48)
        fd, data = hog(img, orientations=4, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True)
        test_img_HOG[i] = np.asarray(data).astype(np.float32).reshape(48*48)
    test_img = np.asarray(test_img_HOG, dtype=np.float32)
    test_label = test_data.iloc[:, 0]

    print('Data Processing Done. Time elapsed: %.2f sec\n' 
                                  % (time.process_time()-t0))

    return train_img, train_label, val_img, val_label, test_img, test_label

# plot the accuracy curve
def plot_acc(curve_list, curve_label):
    assert len(curve_list) == len(curve_label)
    
    data_len = len(curve_list)
    for i in range(data_len):
        plt.plot(range(len(curve_list[i])), curve_list[i], label=curve_label[i])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.legend()
    plt.show()
    