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

        x = self.conv1(x)
        x = self.logSoftMax(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.logSoftMax(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.logSoftMax(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.logSoftMax(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.logSoftMax(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.logSoftMax(x)
        x = self.pool6(x)

        x = self.conv7(x)
        x = self.logSoftMax(x)
        x = self.pool7(x)

        return x
    
param_stft = Param_stft()
paths = Paths()

wanted_snr = [10]

process_data = False
create_tensors = True


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

    if create_tensors:
        print("Loading spectrograms into a tensor - ", end="")
        X = load_spectrograms_to_tensor(wanted_snr, paths)
        # X_normalized = normalizer(X)
        print("OK")

        print("Computing masks into a tensor - ", end="")
        Y = compute_masks_into_tensor(wanted_snr, paths, compute_mask, param_stft)
        print("OK")

        print(X.shape)
        print(Y.shape)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

    net = Net()
    print(net)

    # Initialisation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    BATCH_SIZE = 5
    num_epochs = 100

    Ytrain = torch.abs(torch.from_numpy(Ytrain))
    Ytest = torch.abs(torch.from_numpy(Ytest))
    # Ensure X and Y tensors are of the same type
    Xtrain = torch.abs(torch.from_numpy(Xtrain))
    Xtest = torch.abs(torch.from_numpy(Xtest))

    # Création des datasets
    print("---------------------Xtrain----------------------")
    print(Xtrain)
    print(Xtrain.shape)
    print("---------------------Ytrain----------------------")
    print(Ytrain)
    print(Ytrain.shape)
    print("---------------------Xtest----------------------")
    print(Xtest)
    print(Xtest.shape)
    print("---------------------Ytest----------------------")
    print(Ytest)
    print(Ytest.shape)


    # Now create your datasets
    train_dataset = TensorDataset(Xtrain, Ytrain)
    val_dataset = TensorDataset(Xtest, Ytest)
    
    train_dataset = TensorDataset(Xtrain, Ytrain)
    val_dataset = TensorDataset(Xtest, Ytest)

    trainDataLoader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)
    valDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # Entraînement
    for epoch in range(num_epochs):
        for data, target in trainDataLoader:  # Supposons que train_loader est votre DataLoader
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    # Sauvegarde du modèle
    torch.save(model.state_dict(), 'model_path.pth')


