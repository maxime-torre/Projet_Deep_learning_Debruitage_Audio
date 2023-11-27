import numpy as np
from pathlib import Path
import pickle
import librosa 

def calculate_amplitude_spectrogram(audio_path, N, H):
    """
    Calculer le spectrogramme d'amplitude pour un fichier audio donné.

    :param audio_path: Chemin vers le fichier audio.
    :return: Spectrogramme d'amplitude.
    """
    # Charger le fichier audio
    y, sr = librosa.load(audio_path, sr=None)
    print(len(y))

    # Calculer le STFT
    D = librosa.stft(y, n_fft=N, hop_length=H, win_length=N, window='hann')

    # Convertir en amplitudes
    S = np.abs(D)

    return S

def mask_binary(raw_data_audio, noise_audio_path, N, H, mask_type='soft'):
    """
    Appliquer un masque binaire ou souple sur un spectrogramme bruité.

    :param raw_data_audio: Chemin vers le fichier audio sans bruit.
    :param noise_audio_path: Chemin vers le fichier audio de bruit.
    :param mask_type: Type de masque à utiliser ('soft' ou 'binary').
    :return: Spectrogramme avec masque appliqué.
    """
    # Charger le fichier audio de bruit et calculer son spectrogramme
    audio_spectrogram = calculate_amplitude_spectrogram(raw_data_audio, N, H)
    noise_spectrogram = calculate_amplitude_spectrogram(noise_audio_path, N, H)

    if mask_type == 'binary':
        # Estimation du masque binaire
        mask = np.abs(audio_spectrogram) > np.abs(noise_spectrogram)
        return mask
    elif mask_type == 'soft':
        # Estimation du masque souple
        soft_mask = np.abs(audio_spectrogram) / (np.abs(audio_spectrogram) + np.abs(noise_spectrogram))
        soft_mask = np.clip(soft_mask, 0, 1)
        return soft_mask 
    else:
        raise ValueError("Invalid mask type. Choose 'soft' or 'binary'.")
    
#test
N = 2048
H = 1024
noise_audio_path =  Path.cwd() / "data" / "only_noise" / "20.flac"# Remplacer par le chemin réel
audio_raw_data_path = Path.cwd() / "data" / "raw_data_cut" / "20.flac"
soft_mask = mask_binary(audio_raw_data_path, noise_audio_path, N, H)
signal_and_noise = Path.cwd() / "data" / "noised_data" / "10" / "20.flac"
