import os
import numpy as np


from pathlib import Path
from add_background_noise import add_background_noise
from load_spectrograms_to_tensor import load_spectrograms_to_tensor
from modules import normalizeDividingByMax
import librosa

from mask_binary import create_mask


audio_folder = Path.cwd() / "data" / "raw_data" # chemin du dossier contenant les fichiers flac
audio_folder_cut = Path.cwd() / "data" / "raw_data_cut" # chemin du dossier contenant les fichiers flac
noise_file = Path.cwd() / "data" / "babble_16k.wav"  # chemin du fichier de bruit
output_folder = Path.cwd() / "data" / "noised_data"  # dossier de sortie pour les fichiers audio bruités
output_noise_folder = Path.cwd() / "data" / "only_noise" # dossier de sortie pour stocker les bruits générés aléatoirement

n_fft = 2048
n_window = 600
hop_length = 256
window = "hann"

do_noise_data = False
create_tensors = True

wanted_snr = [10]

normalizer = normalizeDividingByMax


def noise_data():

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        for i in range(-10, 11):
            os.makedirs(output_folder / f"{i}")

    if not os.path.exists(output_noise_folder):
        os.makedirs(output_noise_folder)
    
    if not os.path.exists(audio_folder_cut):
        os.makedirs(audio_folder_cut)

    

    # Traiter tous les fichiers .flac dans le dossier
    i=0
    for file in os.listdir(audio_folder):
        if file.endswith('.flac'):

            SNRvalue = -10 + i%21
            full_audio_path = audio_folder / file
            full_output_folder_path = output_folder / f"{SNRvalue}"
            
            # Appeler la fonction pour ajouter un bruit de fond
            add_background_noise(full_audio_path, noise_file, full_output_folder_path, output_noise_folder, audio_folder_cut, snr_level = SNRvalue)
        
        i += 1

def create_masks_into_tensor(wanted_snr):

    masks = []

    for snr in wanted_snr:
        noised_data_path = os.path.join(output_folder, str(snr))

    #     # Vérifier si c'est bien un dossier
        if os.path.isdir(noised_data_path):
            noised_audio_files = [f for f in os.listdir(noised_data_path) if f.endswith('.flac')]

            for noised_audio_file in noised_audio_files:
                raw_path = os.path.join(audio_folder_cut, noised_audio_file)
                noised_audio_path = os.path.join(noised_data_path, noised_audio_file)
                noise_path = os.path.join(output_noise_folder, noised_audio_file)
                
                y_raw, sr= librosa.load(raw_path, sr=None)
                y_signal, sr = librosa.load(noised_audio_path, sr=None)
                y_noise, sr = librosa.load(noise_path, sr=None)

                masks.append(create_mask(y_raw, y_noise, y_signal, n_fft, n_window, hop_length, window))

    tensor = np.stack(masks)
    return tensor

if __name__ == '__main__':
    if do_noise_data:
        noise_data()

    tensor = None
    if create_tensors:

        X = load_spectrograms_to_tensor(wanted_snr)
        # X_normalized = normalizer(X)
    
        Y = create_masks_into_tensor(wanted_snr)

    print(X.shape)
    print(Y.shape)
