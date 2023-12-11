import numpy as np
import librosa 
import cmath
import os


def compute_binary_mask(raw, noise, param_stft):
    """
    Calcul le mask binaire pour un signal donnée, ainsi que sa version bruité et le
    bruit lui même (tout au format temporelle)

    :param raw: Série temporelle du signal originale
    :param noise: Série temporelle du signal bruit
    :param signal: Série temporelle du signal originale + bruit
    :param param_stft: Objet contenant les paramètres du calcul de la stft.
    """
    
    stft_raw = librosa.stft(raw, n_fft=param_stft.n_fft, hop_length=param_stft.hop_length, win_length=param_stft.n_window, window=param_stft.window)
    stft_noise = librosa.stft(noise, n_fft=param_stft.n_fft, hop_length=param_stft.hop_length, win_length=param_stft.n_window, window=param_stft.window)

    mod_raw = np.abs(stft_raw)**2
    mod_noise = np.abs(stft_noise)**2

    mask = mod_raw > mod_noise

    return mask


def compute_soft_mask(raw, noise, param_stft):
    """
    Calcul le soft mask pour un signal donnée, ainsi que sa version bruité et le
    bruit lui même (tout au format temporelle)

    :param raw: Série temporelle du signal originale
    :param noise: Série temporelle du signal bruit
    :param signal: Série temporelle du signal originale + bruit
    :param param_stft: Objet contenant les paramètres du calcul de la stft.
    """
    
    stft_raw = librosa.stft(raw, n_fft=param_stft.n_fft, hop_length=param_stft.hop_length, win_length=param_stft.n_window, window=param_stft.window)
    stft_noise = librosa.stft(noise, n_fft=param_stft.n_fft, hop_length=param_stft.hop_length, win_length=param_stft.n_window, window=param_stft.window)

    mod_raw = np.abs(stft_raw)**2
    mod_noise = np.abs(stft_noise)**2

    mask = mod_raw / (mod_raw + mod_noise)

    # TODO Est-ce vraiment utile, on devrait pas garder pour des questions de normes ?
    mask = np.clip(mask, 0, 1)

    return mask


def compute_masks_into_tensor(wanted_snr, paths, compute_mask, param_stft):
    """
    Calcul les mask pour chaque snr désiré et les retourne sous la forme
    d'un tenseur (nb de mask, dim X, dim Y).

    :param wanted_snr: Liste d'entier correspondant au snr que nous voulons utiliser.
    :param paths: Objet contenant tous les paths utiles et génériques.
    :param compute_mask: Fonction qui calcule le mask
    :param param_stft: Objet contenant les paramètres du calcul de la stft.
    """
    masks = []

    for snr in wanted_snr:
        noised_data_path = os.path.join(paths.noised_folder, str(snr))

    #     # Vérifier si c'est bien un dossier
        if os.path.isdir(noised_data_path):
            noised_audio_files = [f for f in os.listdir(noised_data_path) if f.endswith('.flac')]

            for noised_audio_file in noised_audio_files:
                raw_path = os.path.join(paths.raw_cut_folder, noised_audio_file)
                noise_path = os.path.join(paths.only_noise_folder, noised_audio_file)
                
                y_raw, sr= librosa.load(raw_path, sr=None)
                y_noise, sr = librosa.load(noise_path, sr=None)

                masks.append(compute_mask(y_raw, y_noise, param_stft))

    tensor = np.stack(masks)
    return tensor