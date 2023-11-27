import os
import numpy as np
import soundfile as sf
import pandas as pd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import os
import pickle
from pathlib import Path

# Nous allons maintenant enregistrer les données du spectrogramme dans un fichier JSON

n_fft = 2048
n_window = 600
hop_length = 256
window = "hann"


base_path = Path.cwd() / "data" / "noised_data"
spectrogram_path = Path.cwd() / "data" / "Spectros"

os.makedirs(spectrogram_path, exist_ok=True)

def save_amplitude_spectrogram_to_pickle(y, sr, pickle_file_path):
    """
    Save the amplitude spectrogram data to a pickle file.
    """
    # Calcul du STFT (Short-Time Fourier Transform)
    D = librosa.stft(y)

    # Conversion en amplitude
    S = np.abs(D)

    # Sauvegarde dans un fichier pickle
    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(S, pickle_file)

# Processus pour convertir les fichiers .flac en spectrogrammes et les enregistrer en format pickle
for folder in range(-10, 11):
    current_folder_path = os.path.join(base_path, str(folder))
    current_spectrogram_folder = os.path.join(spectrogram_path, str(folder))
    os.makedirs(current_spectrogram_folder, exist_ok=True)

    for file in os.listdir(current_folder_path):
        if file.endswith('.flac'):
            file_path = os.path.join(current_folder_path, file)
            y, sr = librosa.load(file_path, sr=None)  # Charger sans changer le taux d'échantillonnage

            pickle_file_path = os.path.join(current_spectrogram_folder, file.replace('.flac', '.pkl'))

            stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_window, window='hann')

            # Conversion en amplitude
            stft_amplitude = np.abs(stft)
            with open(pickle_file_path, 'wb') as pickle_file:
                pickle.dump(stft_amplitude, pickle_file)





            # save_amplitude_spectrogram_to_pickle(y, sr, pickle_file_path)

"Les spectrogrammes ont été générés et enregistrés sous forme de fichiers pickle dans le dossier 'spectrogrammes'."