import os
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split


from modules import normalizeDividingByMax

from noise import noise_data
from spectrograms import compute_spectrograms, load_spectrograms_to_tensor
from mask import compute_masks_into_tensor, compute_binary_mask, compute_soft_mask, save_mask
from models.unet import UNet

import torch
import torch.nn as nn
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
        self.generated_mask = self.data_path / "generated_mask" # spectrogrammes

class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.data = []
        for i in range(x.shape[0]):
            self.data.append([x[i], y[i]])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx][0][None,:,:], self.data[idx][1][None,:,:]

    
param_stft = Param_stft()
paths = Paths()




process_data = False
train = True
test = False


normalizer = normalizeDividingByMax
compute_mask = compute_binary_mask
model_we_are_using = UNet()

wanted_snr = [10]
TEST_SIZE = 0.2
INIT_LR = 5e-3
BATCH_SIZE = 5
EPOCHS = 10

# Chemin vers le fichier de modèle
model_name = 'model'
model_path = Path.cwd() / f'{model_name}.pth'


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

    if train or test:
        print("Loading spectrograms into a tensor - ", end="")
        X = load_spectrograms_to_tensor(wanted_snr, paths)
        X = X.astype(np.float32)
        X_normalized = normalizer(X)
        print("OK")

        print("Computing masks into a tensor - ", end="")
        Y = compute_masks_into_tensor(wanted_snr, paths, compute_mask, param_stft)
        print("OK")

        print(X.shape)
        print(Y.shape)

        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=TEST_SIZE, shuffle=False)
        # Xval, Xtest, Yval, Ytest = train_test_split(Xtest, Ytest, test_size=0.5)

        # TODO A dégager après la ligne du dessus décommenter
        Xval, Yval = Xtest, Ytest

        net = UNet()

        if train:
            epoch_loss = []
            epoch_val_loss = []

            # Initialisation
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cpu")


            model = UNet().to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            # define training hyperparameters

            # Transforme X et Y en objet Torch
            Xtrain = torch.from_numpy(Xtrain)
            Xval = torch.from_numpy(Xval)
            Xtest = torch.from_numpy(Xtest)

            Ytrain = torch.from_numpy(Ytrain)
            Yval = torch.from_numpy(Yval)
            Ytest = torch.from_numpy(Ytest)

            # # Création des datasets
            # print("---------------------Xtrain----------------------")
            # # print(Xtrain)
            # print(Xtrain.shape)
            # print("---------------------Ytrain----------------------")
            # # print(Ytrain)
            # print(Ytrain.shape)
            # print("---------------------Xval----------------------")
            # # print(Xval)
            # print(Xval.shape)
            # print("---------------------Yval----------------------")
            # # print(Yval)
            # print(Yval.shape)
            # print("---------------------Xtest----------------------")
            # # print(Xtest)
            # print(Xtest.shape)
            # print("---------------------Ytest----------------------")
            # # print(Ytest)
            # print(Ytest.shape)


            train = CustomDataset(Xtrain, Ytrain)
            val = CustomDataset(Xval, Yval)

            trainDataLoader = DataLoader(train, shuffle=True, batch_size=BATCH_SIZE)
            valDataLoader = DataLoader(val, batch_size=BATCH_SIZE)

            # Entraînement
            for epoch in range(EPOCHS):
                print(f"epoch : {epoch}")
                model.train()
                total_loss = 0
                for data, target in trainDataLoader:
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
            
                average_loss = total_loss / len(trainDataLoader)
                epoch_loss.append(average_loss)
                # print(f"Loss : {epoch + 1} = {epoch_loss[-1]}")

                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for data, target in valDataLoader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        total_val_loss += criterion(output, target).item()
                average_val_loss = total_val_loss / len(valDataLoader)
                epoch_val_loss.append(average_val_loss)
                # print(f"Val loss : {epoch + 1} = {epoch_val_loss[-1]}")

            # Sauvegarde du modèle
            torch.save(model.state_dict(), model_path)

            plt.figure()
            plt.plot(epoch_loss, label="Train loss")
            plt.plot(epoch_val_loss, label="Validation loss")
            plt.legend()
            plt.grid(True)
            plt.title("Train et Validation loss en fonction des epochs")
            plt.show()

        if test:
            print("test")

            # Définir le périphérique
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # device = torch.device("cpu")
            test_dataset = CustomDataset(Xtest, Ytest)
            
            # Charger le modèle
            model = net.load_model(model_path, device)

            # Préparer les données de test (à remplir avec vos données)
            testDataLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

            # Testez le modèle
            outputs = net.test_model(model, testDataLoader, device)

            # Sauvegarder les mask

            # On récupère les noms des fichiers de test
            wrong, names = train_test_split(os.listdir(paths.spectrogram_folder / f"{wanted_snr[0]}"), test_size=TEST_SIZE, shuffle=False)

            output_path = paths.generated_mask / model_name / f"{wanted_snr[0]}"
            save_mask(outputs, output_path, names)




