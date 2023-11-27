import numpy as np
from pathlib import Path
import pickle
import librosa 
import cmath


def calculate_amplitude_spectrogram(audio_path, N, N_window, H):
    """
    Calculer le spectrogramme d'amplitude pour un fichier audio donné.

    :param audio_path: Chemin vers le fichier audio.
    :return: Spectrogramme d'amplitude.
    """
    # Charger le fichier audio
    y, sr = librosa.load(audio_path, sr=None)
    print(len(y))

    # Calculer le STFT
    D = librosa.stft(y, n_fft=N, hop_length=H, win_length=N_window, window='hann')

    # Convertir en amplitudes
    S = np.abs(D)

    return S

def mask_binary(raw_data_audio, noise_audio_path, N, N_window, H, mask_type='soft'):
    """
    Appliquer un masque binaire ou souple sur un spectrogramme bruité.

    :param raw_data_audio: Chemin vers le fichier audio sans bruit.
    :param noise_audio_path: Chemin vers le fichier audio de bruit.
    :param mask_type: Type de masque à utiliser ('soft' ou 'binary').
    :return: Spectrogramme avec masque appliqué.
    """
    # Charger le fichier audio de bruit et calculer son spectrogramme
    audio_spectrogram = calculate_amplitude_spectrogram(raw_data_audio, N, N_window, H)
    noise_spectrogram = calculate_amplitude_spectrogram(noise_audio_path, N, N_window, H)

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

def create_mask(raw, noise, signal, n_fft, n_window, hop_length, window):
    stft_raw = librosa.stft(raw, n_fft=n_fft, hop_length=hop_length, win_length=n_window, window=window)
    stft_noise = librosa.stft(noise, n_fft=n_fft, hop_length=hop_length, win_length=n_window, window=window)
    stft_signal = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=n_window, window='hann')

    mod_raw = np.abs(stft_raw)
    mod_noise = np.abs(stft_noise)

    phase = np.zeros_like(stft_signal)
    for i in range(stft_signal.shape[0]):
        for j in range(stft_signal.shape[1]):
            phase[i, j] = cmath.phase(stft_signal[i, j])

    mask = mod_raw > mod_noise

    return mask * phase