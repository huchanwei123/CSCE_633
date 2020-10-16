# -*- coding: utf-8 -*-
"""
Description:
    Neural Network Architecture
"""
import torch.nn as nn

class FNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # layers
        self.fc1 = nn.Sequential(     
            nn.Linear(in_features = self.input_size, 
                      out_features = self.hidden_size,),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
            #nn.Dropout(),
        )
        
        self.fc2 = nn.Sequential(     
            nn.Linear(in_features = self.hidden_size, 
                      out_features = self.hidden_size,),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
            #nn.Dropout(),
        )
        
        self.fc3 = nn.Sequential(     
            nn.Linear(in_features = self.hidden_size, 
                      out_features = self.hidden_size,),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
            #nn.Dropout(),
        )

        self.fc4 = nn.Sequential(     
            nn.Linear(in_features = self.hidden_size, 
                      out_features = self.output_size,),
            nn.ReLU(),
            #nn.BatchNorm1d(self.hidden_size),
            #nn.Dropout(),
        )

        # activation
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # dropout
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.conv1 = nn.Sequential(         # input shape: (1, 48, 48)
            nn.Conv2d(in_channels = 1, 
                      out_channels = 16, 
                      kernel_size = 7,
                      stride = 1,
                      padding = 3,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),  # output shape: (16, 24, 24) 
            nn.BatchNorm2d(16),
            #nn.Dropout2d(),
            #nn.Dropout(),
        )
        
        self.conv2 = nn.Sequential(         # input shape: (16, 24, 24)
            nn.Conv2d(in_channels = 16, 
                      out_channels = 64, 
                      kernel_size = 7,
                      stride = 1,
                      padding = 3,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),  # output shape: (64, 12, 12) 
            nn.BatchNorm2d(64),
            #nn.Dropout2d(),
            #nn.Dropout(),
        )
        
        self.conv3 = nn.Sequential(         # input shape: (64, 12, 12)
            nn.Conv2d(in_channels = 64, 
                      out_channels = 64, 
                      kernel_size = 7,
                      stride = 1,
                      padding = 3,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),  # output shape: (64, 6, 6) 
            nn.BatchNorm2d(64),
            #nn.Dropout2d(),
            #nn.Dropout(),
        )
        
        self.conv4 = nn.Sequential(         # input shape: (64, 6, 6)
            nn.Conv2d(in_channels = 64, 
                      out_channels = 64, 
                      kernel_size = 7,
                      stride = 1,
                      padding = 3,),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),  # output shape: (64, 3, 3) 
            #nn.Dropout2d(),
            #nn.Dropout(),
        )
        
        self.fc1 = nn.Linear(64 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 7)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.dropout(x)
        return x