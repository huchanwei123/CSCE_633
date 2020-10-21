# -*- coding: utf-8 -*-
"""
Description:
    Neural Network Architecture
"""
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, hidden_size):
        super(FNN, self).__init__()
        self.hidden_size = hidden_size

        # layers
        self.fc1 = nn.Sequential(     
            nn.Linear(in_features = 2304, 
                      out_features = 4096,),
            nn.ReLU(),
            #nn.SELU(),
            nn.BatchNorm1d(4096),
        )
        
        self.fc2 = nn.Sequential(     
            nn.Linear(in_features = 4096, 
                      out_features = 4096,),
            #nn.SELU(),
            nn.ReLU(),
            nn.BatchNorm1d(4096),
        )
        
        self.fc3 = nn.Sequential(     
            nn.Linear(in_features = 4096, 
                      out_features = 7,),
            #nn.SELU(),
            nn.ReLU(),
            nn.BatchNorm1d(7),
            nn.Dropout(),
        )

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Sequential(         # input shape: (1, 48, 48)
            nn.Conv2d(in_channels = 1, 
                      out_channels = 16, 
                      kernel_size = 3,
                      stride = 1,
                      padding = 1,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),  # output shape: (16, 24, 24) 
            nn.BatchNorm2d(16),
            nn.Dropout2d(),
        )
        
        self.conv2 = nn.Sequential(         # input shape: (16, 24, 24)
            nn.Conv2d(in_channels = 16, 
                      out_channels = 32, 
                      kernel_size = 3,
                      stride = 1,
                      padding = 1,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),  # output shape: (64, 12, 12) 
            nn.BatchNorm2d(32),
            nn.Dropout2d(),
        )
        
        self.conv3 = nn.Sequential(         # input shape: (64, 12, 12)
            nn.Conv2d(in_channels = 32, 
                      out_channels = 64, 
                      kernel_size = 3,
                      stride = 1,
                      padding = 1,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),  # output shape: (64, 6, 6) 
            nn.BatchNorm2d(64),
            nn.Dropout2d(),
        )
        
        
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 24 * 24, 256),
            nn.ReLU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        
        self.fc3 = nn.Sequential(
            nn.Linear(64, 7),
            nn.ReLU(),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x