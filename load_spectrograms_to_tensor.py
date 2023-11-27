import os
import pickle
import numpy as np
from pathlib import Path

def load_spectrograms_to_tensor(wanted_snr):
    """
    Charge tous les spectrogrammes depuis les sous-dossiers d'un dossier spécifié, 
    les place dans un tenseur et renvoie ce tenseur.
    """
    spectrogram_folder = Path.cwd() / "data" / "Spectros"
    
    spectrograms = []


    # Parcourir tous les sous-dossiers
    for snr in wanted_snr:
        subdir_path = os.path.join(spectrogram_folder, str(snr))

    #     # Vérifier si c'est bien un dossier
        if os.path.isdir(subdir_path):
            spectrogram_files = [f for f in os.listdir(subdir_path) if f.endswith('.pkl')]

            for file in spectrogram_files:
                file_path = os.path.join(subdir_path, file)
                with open(file_path, 'rb') as pickle_file:
                    spectrogram = pickle.load(pickle_file)
                    spectrograms.append(spectrogram)

    tensor = np.stack(spectrograms)
    return tensor
