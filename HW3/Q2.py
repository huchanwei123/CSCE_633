# CSCE 633 - Machine Learning
# HW3 - Question 2
import os
import cv2
import csv
import pdb
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from EmotionDataset import EmotionDataset

EMOTION_MAP = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# read the csv file
def read_data(data_dir):
    train_data = pd.read_csv(os.path.join(data_dir, 'Q2_Train_Data.csv'))
    val_data = pd.read_csv(os.path.join(data_dir, 'Q2_Validation_Data.csv'))
    test_data = pd.read_csv(os.path.join(data_dir, 'Q2_Test_Data.csv'))

    # get data and labels
    # training data
    train_img = train_data.iloc[:, 1].apply(lambda x: x.split())
    train_img = [list(map(int, train_img[i])) for i in range(len(train_data))]
    train_img = np.asarray(train_img, dtype=np.float32)
    train_label = train_data.iloc[:, 0]

    # validation data
    val_img = val_data.iloc[:, 1].apply(lambda x: x.split())
    val_img = [list(map(int, val_img[i])) for i in range(len(val_data))]
    val_img = np.asarray(val_img, dtype=np.float32)
    val_label = val_data.iloc[:, 0]

    # testing data
    test_img = test_data.iloc[:, 1].apply(lambda x: x.split())
    test_img = [list(map(int, test_img[i])) for i in range(len(test_data))]
    test_img = np.asarray(test_img, dtype=np.float32)
    test_label = test_data.iloc[:, 0]

    print('Data Processing Done...\n')

    return train_img, train_label, val_img, val_label, test_img, test_label

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # layers
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.output_size)

        # activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train(net, trainloader, optimizer, criterion):
    running_loss = 0
    for data in trainloader:
        # data pixels and labels to GPU if available
        inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
        # set the parameter gradients to zero
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # propagate the loss backward
        loss.backward()
        # update the gradients
        optimizer.step()

        running_loss += loss.item()
    return running_loss

if __name__ == '__main__':
    # read data first
    data_dir = '../../hw3_data'
    train_img, train_label, val_img, val_label, test_img, test_label = read_data(data_dir)

    train_data = EmotionDataset(train_img, train_label)
    val_data = EmotionDataset(val_img, val_label)
    test_data = EmotionDataset(test_img, test_label)
    
    # dataloaders
    trainloader = DataLoader(train_data, batch_size=128, shuffle=True)
    valloader = DataLoader(val_data, batch_size=128, shuffle=True)
    testloader = DataLoader(test_data, batch_size=128, shuffle=True)

    # start doing training
    INPUT_SIZE = 48*48
    HIDDEN_SIZE = 4096
    OUTPUT_SIZE = 7
    fnn = FNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    optimizer = optim.Adam(fnn.parameters())

    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fnn = fnn.to(device)
    criterion = criterion.to(device)

    for epoch in range(10):
        running_loss = train(fnn, trainloader, optimizer, criterion)
        train_acc = test(fnn, trainloader)
        val_acc = test(fnn, valloader)
        print('[Epoch {}] loss: {} | training acc: {} | val acc: {}'.format(epoch + 1, running_loss/len(trainloader), train_acc, val_acc))
        
