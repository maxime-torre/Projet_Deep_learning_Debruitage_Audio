import os

from pathlib import Path
from sklearn.model_selection import train_test_split

from modules import normalizeDividingByMax

from noise import noise_data
from spectrograms import compute_spectrograms, load_spectrograms_to_tensor
from mask import compute_masks_into_tensor, compute_binary_mask, compute_soft_mask

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LogSoftmax
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader, TensorDataset
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = []
        for i in range(x.shape[0]):
            self.data.append([x[i], y[i]])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1]

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(5, 3, 3)
        self.conv2 = nn.Conv2d(3, 6, 3)
        self.conv3 = nn.Conv2d(6, 9, 3)
        self.conv4 = nn.Conv2d(9, 9, 3)
        self.conv5 = nn.ConvTranspose2d(9, 6, 3)
        self.conv6 = nn.ConvTranspose2d(6, 3, 3)
        self.conv7 = nn.ConvTranspose2d(3, 1, 3)

        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = MaxPool2d(kernel_size=2, stride=2)
        self.pool5 = MaxPool2d(kernel_size=2, stride=2)
        self.pool6 = MaxPool2d(kernel_size=2, stride=2)
        self.pool7 = MaxPool2d(kernel_size=2, stride=2)

        self.logSoftMax = LogSoftmax(dim=1)

    def forward(self, x):
        print(x.shape)
        x = self.conv1(x)
        x = self.logSoftMax(x)
        x = self.pool1(x)
        print(x.shape)
        x = self.conv2(x)
        x = self.logSoftMax(x)
        x = self.pool2(x)
        print(x.shape)
        x = self.conv3(x)
        x = self.logSoftMax(x)
        x = self.pool3(x)
        print(x.shape)
        x = self.conv4(x)
        x = self.logSoftMax(x)
        x = self.pool4(x)
        print(x.shape)
        x = self.conv5(x)
        x = self.logSoftMax(x)
        x = self.pool5(x)
        print(x.shape)
        x = self.conv6(x)
        x = self.logSoftMax(x)
        x = self.pool6(x)
        print(x.shape)
        x = self.conv7(x)
        x = self.logSoftMax(x)
        x = self.pool7(x)
        print(x.shape)
        print()
        return x
        
    
test = torch.rand(5,1025,251)
test2 = torch.rand(5,1025,251)

# set the device we will be using to train the model
device = torch.device("cpu")    
model = Net().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 5
EPOCHS = 10

train = CustomDataset(test, test2)

train_dataset = TensorDataset(train)

trainDataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
