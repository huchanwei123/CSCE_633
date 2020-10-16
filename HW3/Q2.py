# CSCE 633 - Machine Learning
# HW3 - Question 2
import os
import pdb
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

# import my function
from EmotionDataset import EmotionDataset
from model import FNN, CNN

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

def plot_acc(curve_list, curve_label):
    assert len(curve_list) == len(curve_label)
    
    data_len = len(curve_list)
    for i in range(data_len):
        plt.plot(range(len(curve_list[i])), curve_list[i], label=curve_label[i])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    
    plt.legend()
    plt.show()

def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            #pdb.set_trace()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def train(net, trainloader, optimizer, criterion):
    net.train()
    running_loss = 0
    for data in trainloader:
        #pdb.set_trace()
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
    data_dir = './'
    # start doing training
    INPUT_SIZE = 48*48
    HIDDEN_SIZE = 2048
    OUTPUT_SIZE = 7
    MAX_EPOCH = 200
    net = 'CNN'
    resize = net =='CNN'
    train_img, train_label, val_img, val_label, test_img, test_label = read_data(data_dir)

    train_data = EmotionDataset(train_img, train_label, resize=resize)
    val_data = EmotionDataset(val_img, val_label, resize=resize)
    test_data = EmotionDataset(test_img, test_label, resize=resize)
    
    # dataloaders
    trainloader = DataLoader(train_data, batch_size=64, shuffle=True)
    valloader = DataLoader(val_data, batch_size=64, shuffle=True)
    testloader = DataLoader(test_data, batch_size=64, shuffle=True)

    if net == 'FNN':
        model = FNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    elif net == 'CNN':
        model = CNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        #model = models.alexnet()
    else:
        raise ValueError('{} not supported yet!'.format(net))

    #optimizer = optim.Adam(model.parameters())
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)

    Acc_train, Acc_val = [], []

    for epoch in range(MAX_EPOCH):
        running_loss = train(model, trainloader, optimizer, criterion)
        Acc_train.append(test(model, trainloader))
        Acc_val.append(test(model, valloader))
        print('[Epoch %d] loss: %.3f, training acc: %.3f, val acc: %.3f' % \
              (epoch + 1, running_loss/len(trainloader), Acc_train[epoch], Acc_val[epoch]))
    
    test_acc = test(model, testloader)
    print('Testing accuracy = %.3f' % test_acc)
        
    # plot 
    plot_acc([Acc_train, Acc_val], ['Training', 'Validation'])