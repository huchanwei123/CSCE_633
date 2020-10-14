# CSCE 633 - Machine Learning
# HW3 - Question 2

import cv2
import csv
import pdb
import torch
import numpy as np

def read_csv(csv_path):
    # create a dictionary
    dataset = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = int(row['emotion'])
            pixel = np.array(row['pixels'].split()).astype(np.uint8)

            # convert to 48x48 image
            img = np.reshape(pixel, (48, 48))

            # append to dataset
            # dataset[n][0] is the emotion label
            # dataset[n][1] is the image
            dataset.append([label, img])

            # convert to numpy array
            np_dataset = np.asarray(dataset)

    return np_dataset

read_csv('../../hw3_data/Q2_Train_Data.csv')
