# CSCE 633 - Machine Learning
# HW3 - Question 2
import os
import pdb
import time
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import pytorch_model_summary as pms

# import my function
from EmotionDataset import EmotionDataset
from model import FNN, CNN
from misc import read_data, plot_acc

# read data first
data_dir = './'
# start doing training
HIDDEN_SIZE = 4096
MAX_EPOCH = 1000
batch_size = 64
net = 'FNN'
phase = 'test'
set_ = 'set4_HOG'
checkpoint = os.path.join('./model', net)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
def test(net, testloader):
    net.eval()
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

def train(net, trainloader, valloader, optimizer, criterion):
    Acc_train, Acc_val, train_loss = [], [], []
    best_val_acc = 0
    net.train()

    running_loss = 0
    for data in trainloader:
        # data pixels and labels to GPU if available
        inputs, labels = data[0].to(device, non_blocking=True), data[1].to(device, non_blocking=True)
    
        # set the parameter gradients to zero
        optimizer.zero_grad()
        outputs = net(inputs)
    
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    train_loss.append(running_loss/len(trainloader))
    
    return running_loss

if __name__ == '__main__':

    resize = net =='CNN'
    train_img, train_label, val_img, val_label, test_img, test_label = read_data(data_dir)

    train_data = EmotionDataset(train_img, train_label, resize=resize)
    # show random sample
    #train_data.show_random_img(spc=2)
    #train_data.data_distribution()
    
    val_data = EmotionDataset(val_img, val_label, resize=resize)
    test_data = EmotionDataset(test_img, test_label, resize=resize)
    
    # dataloaders
    trainloader = DataLoader(train_data, batch_size, shuffle=True)
    valloader = DataLoader(val_data, batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size, shuffle=True)

    if net == 'FNN':
        model = FNN(HIDDEN_SIZE)
        print(pms.summary(model, torch.zeros((1, 2304)), 
                      show_input=True, show_hierarchical=False))
    elif net == 'CNN':
        model = CNN()
        print(pms.summary(model, torch.zeros((1, 1, 48, 48)), 
                      show_input=True, show_hierarchical=False))
    else:
        raise ValueError('{} not supported yet!'.format(net))
    
    optimizer = optim.Adam(model.parameters())    
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
                
    criterion = nn.CrossEntropyLoss()

    model = model.to(device)
    criterion = criterion.to(device)
    
    t0 = time.process_time()
    Acc_train, Acc_val, train_loss = [], [], []
    best_val_acc = 0
    best_epoch = 0
    
    if phase == 'train':    
        for epoch in range(1, MAX_EPOCH+1):
            # start traning for one epoch
            running_loss = train(model, trainloader, valloader, optimizer, criterion)
            # get the average loss
            train_loss.append(running_loss/len(trainloader))
            
            # test the result
            train_acc = test(model, trainloader)
            val_acc = test(model, valloader)
            Acc_train.append(train_acc)
            Acc_val.append(val_acc)
            
            # store the model and print out result
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                PATH = os.path.join(checkpoint, set_+'_best.pt')
                torch.save(model.state_dict(), PATH)
                print('[Epoch %d] loss: %.3f, training acc: %.3f, val acc: %.3f -> Model saved!' % \
                  (epoch, running_loss/len(trainloader), train_acc, val_acc))
            else:
                print('[Epoch %d] loss: %.3f, training acc: %.3f, val acc: %.3f' % \
              (epoch, running_loss/len(trainloader), train_acc, val_acc))
            
        print('Training Done. Best validation acc = %.3f at epoch %d.\n Time elapsed: %.2f sec\n' 
                                      % (best_val_acc, best_epoch, time.process_time()-t0))
        
        test_acc = test(model, testloader)
        print('Testing accuracy = %.3f' % test_acc)
            
        # plot 
        plot_acc([Acc_train, Acc_val], ['Training', 'Validation'])
        
        plt.plot(range(MAX_EPOCH), train_loss)
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.show()
        
        # save loss
        df = pd.DataFrame(list(zip(Acc_train, Acc_val, train_loss)), 
                          columns=["train_acc", "val_acc", "train_loss"])
        df.to_csv(checkpoint + '/' + set_ + '_stats.csv', index=False)
    elif phase == 'test':
        PATH = os.path.join(checkpoint, set_+'_best.pt')
        model.load_state_dict(torch.load(PATH))
        test_acc = test(model, testloader)
        print('Testing accuracy = %.3f' % test_acc)
    else:
        raise ValueError('Wrong phase!')