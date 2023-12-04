import os

from pathlib import Path

from modules import normalizeDividingByMax

from noise import noise_data
from spectrograms import compute_spectrograms, load_spectrograms_to_tensor
from mask import compute_masks_into_tensor, compute_binary_mask, compute_soft_mask


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


param_stft = Param_stft()
paths = Paths()

wanted_snr = [10]

process_data = False
create_tensors = False


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
