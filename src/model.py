import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1, bias=False)
        self.gn1   = nn.GroupNorm(4, 32)

        self.conv2 = nn.Conv2d(32,32,3,1, 1, bias=False)
        self.gn2   = nn.GroupNorm(4, 32)
        self.pool1 = nn.MaxPool2d(2,2)
      

        self.conv3 = nn.Conv2d(32,64,3, 1, 1, bias=False)
        self.gn3   = nn.GroupNorm(8, 64)
        self.conv4 = nn.Conv2d(64,64,3, 1, 1, bias=False)
        self.gn4   = nn.GroupNorm(8, 64)
        self.pool2 = nn.MaxPool2d(2,2)
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64*1*1, 64)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(64, num_classes)
        

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))

        x = F.relu(self.gn2(self.conv2(x)))
        x = self.pool1(x)

        x = F.relu(self.gn3(self.conv3(x)))
        x = F.relu(self.gn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)







        return x



        
        

        
