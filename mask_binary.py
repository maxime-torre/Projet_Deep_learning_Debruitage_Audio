import numpy as np
from pathlib import Path
import pickle
import librosa

def calculate_amplitude_spectrogram(audio_path):
    """
    Calculer le spectrogramme d'amplitude pour un fichier audio donné.

    :param audio_path: Chemin vers le fichier audio.
    :return: Spectrogramme d'amplitude.
    """
    # Charger le fichier audio
    y, sr = librosa.load(audio_path, sr=None)

    # Calculer le STFT
    D = librosa.stft(y)

    # Convertir en amplitudes
    S = np.abs(D)

    return S

def mask_binary(noisy_spectrogram_pkl, noise_audio_path, mask_type='soft'):
    """
    Appliquer un masque binaire ou souple sur un spectrogramme bruité.

    :param noisy_spectrogram_pkl: Chemin vers le fichier pickle contenant le spectrogramme bruité.
    :param noise_audio_path: Chemin vers le fichier audio de bruit.
    :param mask_type: Type de masque à utiliser ('soft' ou 'binary').
    :return: Spectrogramme avec masque appliqué.
    """
    # Charger le spectrogramme bruité depuis le fichier pickle
    with open(noisy_spectrogram_pkl, 'rb') as file:
        noisy_spectrogram = pickle.load(file)

    # Charger le fichier audio de bruit et calculer son spectrogramme
    noise_spectrogram = calculate_amplitude_spectrogram(noise_audio_path)

    if mask_type == 'binary':
        # Estimation du masque binaire
        mask = np.abs(noisy_spectrogram) > np.abs(noise_spectrogram)
        return mask * noisy_spectrogram
    elif mask_type == 'soft':
        # Estimation du masque souple
        soft_mask = np.abs(noisy_spectrogram) / (np.abs(noisy_spectrogram) + np.abs(noise_spectrogram))
        soft_mask = np.clip(soft_mask, 0, 1)
        return soft_mask * noisy_spectrogram
    else:
        raise ValueError("Invalid mask type. Choose 'soft' or 'binary'.")


# Exemple d'utilisation
noise_audio_path =  Path.cwd() / "data" / "only_noise" / "6.flac"# Remplacer par le chemin réel
noise_spectrogram = Path.cwd() / "data" / "Spectros" / "0" / "6.pkl"
soft_mask = mask_binary(noise_spectrogram, noise_audio_path)

print(soft_mask)