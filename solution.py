
# import libraries
import numpy as np
import random
# PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class mlp(nn.Module):

  def __init__(self,
               time_periods, n_classes):
        super(mlp, self).__init__()
        self.time_periods = time_periods
        self.n_classes = n_classes
        # WRITE CODE HERE
         
        self.flatten_layer = nn.Flatten()
        self.fc1 = nn.Linear(in_features=self.time_periods*3,out_features=100)
        self.fc2 = nn.Linear(in_features=100,out_features=100)
        self.fc3 = nn.Linear(in_features=100,out_features=100)
        self.fc4 = nn.Linear(in_features=100,out_features=self.n_classes)
        #self.softmax = nn.Softmax()

  def forward(self, x):
    # WRITE CODE HERE
    x = self.flatten_layer(x)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    x = F.log_softmax(x)
    
    return x
  
# # WRITE CODE HERE

class cnn(nn.Module):

  def __init__(self, time_periods, n_sensors, n_classes):
        super(cnn, self).__init__()
        self.time_periods = time_periods
        self.n_sensors = n_sensors
        self.n_classes = n_classes

        # WRITE CODE HERE
        self.kernel_size = 10

        self.conv1 = nn.Conv1d(self.n_sensors, 100, self.kernel_size, stride=1,padding='valid')
        self.conv2 = nn.Conv1d(100, 100, self.kernel_size, stride=1, padding='valid')
        self.maxpool1 = nn.MaxPool1d(3)
        self.conv3 = nn.Conv1d(100, 160, self.kernel_size, stride=1, padding='valid')
        self.conv4 = nn.Conv1d(160, 160, self.kernel_size, stride=1, padding='valid')
        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(160,self.n_classes)
        
        

  def forward(self, x):
        # Reshape the input to (batch_size, n_sensors, time_periods)
        # WRITE CODE HERE
        x = x.reshape(x.shape[0],self.n_sensors,self.time_periods)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv3(x))
        x_last = F.relu(self.conv4(x))

        x_avg = self.avgpool1(x_last)
        x = self.dropout(x_avg)
        x = x.squeeze() # squeeze out the time dimension that is 1

        x = self.fc(x)
        x = F.log_softmax(x)
        # Layers
        # WRITE CODE HERE
        
        return x
