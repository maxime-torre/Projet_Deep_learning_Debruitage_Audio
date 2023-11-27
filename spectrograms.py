import os
import numpy as np
import librosa
import librosa.display
import numpy as np
import os
import pickle


def compute_spectrograms(paths, param_stft):
    """
    Calcul les spectrogrammes pour chaque sons se situant dans le dossier "noised_data"
    et les enregistres au format pickle dans le dossier "spectros".

    :param paths: Objet contenant tous les paths utiles et génériques.
    :param param_stft: Objet contenant les paramètres du calcul de la stft.
    """

    for folder in range(-10, 11):
        current_folder_path = os.path.join(paths.noised_folder, str(folder))
        current_spectrogram_folder = os.path.join(paths.spectrogram_folder, str(folder))
        os.makedirs(current_spectrogram_folder, exist_ok=True)

        for file in os.listdir(current_folder_path):
            if file.endswith('.flac'):
                file_path = os.path.join(current_folder_path, file)
                y, sr = librosa.load(file_path, sr=None)  # Charger sans changer le taux d'échantillonnage

                pickle_file_path = os.path.join(current_spectrogram_folder, file.replace('.flac', '.pkl'))

                stft = librosa.stft(y, n_fft=param_stft.n_fft, hop_length=param_stft.hop_length, win_length=param_stft.n_window, window=param_stft.window)

                # Conversion en amplitude
                stft_amplitude = np.abs(stft)
                with open(pickle_file_path, 'wb') as pickle_file:
                    pickle.dump(stft_amplitude, pickle_file)


def load_spectrograms_to_tensor(wanted_snr, paths):
    """
    Charge tous les spectrogrammes stockés en mémoire et les retourne 
    sous la forme d'un tenseur (nb de spectro, dim X, dim Y).

    :param wanted_snr: Liste d'entier correspondant au snr que nous voulons utiliser.
    :param paths: Objet contenant tous les paths utiles et génériques.
    """

    spectrograms = []

    # Parcourir tous les sous-dossiers
    for snr in wanted_snr:
        subdir_path = os.path.join(paths.spectrogram_folder, str(snr))

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