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

base_path = Path.cwd() / "data" / "noised_data"
spectrogram_path = Path.cwd() / "data" / "Spectros"

os.makedirs(spectrogram_path, exist_ok=True)

def save_spectrogram_to_pickle(y, sr, pickle_file_path):
    """
    Save the spectrogram data to a pickle file.
    """
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    with open(pickle_file_path, 'wb') as pickle_file:
        pickle.dump(S_dB, pickle_file)

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
            save_spectrogram_to_pickle(y, sr, pickle_file_path)

"Les spectrogrammes ont été générés et enregistrés sous forme de fichiers pickle dans le dossier 'spectrogrammes'."