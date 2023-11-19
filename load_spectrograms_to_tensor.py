import os
import pickle
import numpy as np

def load_spectrograms_to_tensor(spectrogram_folder):
    """
    Charge tous les spectrogrammes depuis les sous-dossiers d'un dossier spécifié, 
    les place dans un tenseur et renvoie ce tenseur.
    """
    spectrograms = []

    # Parcourir tous les sous-dossiers
    for subdir in os.listdir(spectrogram_folder):
        subdir_path = os.path.join(spectrogram_folder, subdir)

        # Vérifier si c'est bien un dossier
        if os.path.isdir(subdir_path):
            spectrogram_files = [f for f in os.listdir(subdir_path) if f.endswith('.pkl')]

            for file in spectrogram_files:
                file_path = os.path.join(subdir_path, file)
                with open(file_path, 'rb') as pickle_file:
                    spectrogram = pickle.load(pickle_file)
                    spectrograms.append(spectrogram)

    tensor = np.stack(spectrograms)
    return tensor

# Exemple d'utilisation
# spectrogram_folder = r"C:\Users\torre\Documents\Sicom 3A\Projet_Parole_Audio_SICOM_3A\data\Spectros"
# spectrogram_tensor = load_spectrograms_to_tensor(spectrogram_folder)

# print(spectrogram_tensor)
# print(len(spectrogram_tensor))
