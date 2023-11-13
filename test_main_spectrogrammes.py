import matplotlib.pyplot as plt
import pickle
import numpy as np
import librosa

def display_spectrogram_from_pickle(pickle_file_path):
    """
    Load a spectrogram from a pickle file and display it.
    """
    with open(pickle_file_path, 'rb') as pickle_file:
        S_dB = pickle.load(pickle_file)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()

# Exemple de chemin vers un fichier pickle de spectrogramme
example_pickle_path = r'C:\Users\torre\Documents\Sicom 3A\Projet_Parole_Audio_SICOM_3A\data\Spectros\-2\25.pkl' # Remplacer par le chemin réel

# Afficher le spectrogramme
display_spectrogram_from_pickle(example_pickle_path)

"Le spectrogramme a été chargé à partir du fichier pickle et affiché."
