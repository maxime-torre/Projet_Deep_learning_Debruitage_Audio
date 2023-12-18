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
from torch.nn import MaxPool2d, MaxUnpool2d
from torch.utils.data import DataLoader, TensorDataset
import glob
# import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Param_stft():
    def __init__(self) -> None:
        self.n_fft = 2048
        self.n_window = 600
        self.hop_length = 256
        self.window = "hann"

class Paths():
    def __init__(self) -> None:
        self.data_path = Path.cwd() / "data"

        self.raw_folder = self.data_path / "raw_data" # dossier des fichiers audios de base
        self.raw_cut_folder = self.data_path / "raw_data_cut" # dossier des fichiers audios de taille normées
        self.babble_file = self.data_path / "babble_16k.wav"  # fichier de bruit
        self.noised_folder = self.data_path / "noised_data"  # fichiers audio bruités
        self.only_noise_folder = self.data_path / "only_noise" # bruits générés aléatoirement
        self.spectrogram_folder = self.data_path / "spectros" # spectrogrammes

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = []
        for i in range(x.shape[0]):
            self.data.append([x[i], y[i]])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][0][None,:,:], self.data[idx][1][None,:,:]

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1) # output: 28x28x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, output_padding=(0,1))
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, output_padding=(0,1))
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=(1,1))
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, 1, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        print(x.shape)
        xe11 = self.relu(self.e11(x))
        xe12 = self.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)
        print(xp1.shape)

        xe21 = self.relu(self.e21(xp1))
        xe22 = self.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)
        print(xp2.shape)

        xe31 = self.relu(self.e31(xp2))
        xe32 = self.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)
        print(xp3.shape)

        xe41 = self.relu(self.e41(xp3))
        xe42 = self.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)
        print(xp4.shape)

        xe51 = self.relu(self.e51(xp4))
        xe52 = self.relu(self.e52(xe51))
        print(xe52.shape)
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = self.relu(self.d11(xu11))
        xd12 = self.relu(self.d12(xd11))
        print(xd12.shape)

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = self.relu(self.d21(xu22))
        xd22 = self.relu(self.d22(xd21))
        print(xd22.shape)

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = self.relu(self.d31(xu33))
        xd32 = self.relu(self.d32(xd31))
        print(xd32.shape)

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = self.relu(self.d41(xu44))
        xd42 = self.relu(self.d42(xd41))
        print(xd42.shape)

        # Output layer
        out = self.outconv(xd42)
        print(out.shape)
        print()

        return out
    
param_stft = Param_stft()
paths = Paths()

wanted_snr = [10]

process_data = False
train = False
test = True


normalizer = normalizeDividingByMax
compute_mask = compute_binary_mask

def create_folders():
    if not os.path.exists(paths.noised_folder):
        os.makedirs(paths.noised_folder)
        for i in range(-10, 11):
            os.makedirs(paths.noised_folder / f"{i}")

    if not os.path.exists(paths.only_noise_folder):
        os.makedirs(paths.only_noise_folder)
    
    if not os.path.exists(paths.raw_cut_folder):
        os.makedirs(paths.raw_cut_folder)
    
    if not os.path.exists(paths.spectrogram_folder):
        os.makedirs(paths.spectrogram_folder)




if __name__ == '__main__':

    create_folders()

    if process_data:
        print("adding noise to data - ", end="")
        noise_data(paths)
        print("OK")

        print("Computing spectrograms - ", end="")
        compute_spectrograms(paths, param_stft)
        print("OK")

    if train:
        print("Loading spectrograms into a tensor - ", end="")
        X = load_spectrograms_to_tensor(wanted_snr, paths)
        X = X.astype(np.float32)
        # X_normalized = normalizer(X)
        print("OK")

        print("Computing masks into a tensor - ", end="")
        Y = compute_masks_into_tensor(wanted_snr, paths, compute_mask, param_stft)
        print("OK")

        print(X.shape)
        print(Y.shape)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

        net = UNet()
        # print(net)

            # Initialisation
        # set the device we will be using to train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("cuda" if torch.cuda.is_available() else "cpu")

        model = UNet().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        # define training hyperparameters
        INIT_LR = 1e-3
        BATCH_SIZE = 5
        EPOCHS = 10

        Ytrain = torch.from_numpy(Ytrain)
        Ytest = torch.from_numpy(Ytest)
        # Ensure X and Y tensors are of the same type
        Xtrain = torch.from_numpy(Xtrain)
        Xtest = torch.from_numpy(Xtest)

        # Création des datasets
        # print("---------------------Xtrain----------------------")
        # print(Xtrain)
        # print(Xtrain.shape)
        # print("---------------------Ytrain----------------------")
        # print(Ytrain)
        # print(Ytrain.shape)
        # print("---------------------Xtest----------------------")
        # print(Xtest)
        # print(Xtest.shape)
        # print("---------------------Ytest----------------------")
        # print(Ytest)
        # print(Ytest.shape)


        train = CustomDataset(Xtrain, Ytrain)
        test = CustomDataset(Xtest, Ytest)

        # print("uuuuuuuuuuuuuuuuuuuuuuuuuuu",train[1][0].shape)
        # print(f"train len : {train.__len__()}")
        # print(f"test len : {test.__len__()}")
        # Now create your datasets
        # train_dataset = TensorDataset(Xtrain, Ytrain)
        # val_dataset = TensorDataset(Xtest, Ytest)
        
        # train_dataset = TensorDataset(Xtrain, Ytrain)
        # val_dataset = TensorDataset(Xtest, Ytest)

        trainDataLoader = DataLoader(train, shuffle=True, batch_size=BATCH_SIZE)
        
        # testDataLoader = DataLoader(test, batch_size=BATCH_SIZE)

        # Entraînement
        for epoch in range(EPOCHS):
            print(f"epoch : {epoch}")
            model.train()
            for data, target in trainDataLoader:  # Supposons que train_loader est votre DataLoader
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # Sauvegarde du modèle
        torch.save(model.state_dict(), 'model_path.pth')

    if test:
        print("test")
