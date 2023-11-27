import os
import numpy as np
import random
import soundfile as sf
from modules import SNRin_dB, power

import librosa
# from librosa import load, resample
from pathlib import Path

def add_background_noise(audio_path, noise_path, output_folder, output_noise_folder, audio_folder_cut, snr_level=10):
    """
    Ajoute un bruit de fond de cafétaria à un fichier audio et sauvegarde le nouveau fichier.

    :param audio_path: Chemin vers le fichier audio d'origine.
    :param noise_path: Chemin vers le fichier audio de bruit de fond.
    :param output_folder: Dossier pour enregistrer l'audio modifié.
    :param snr_level: Niveau de rapport signal/bruit souhaité.
    """
    # Charger les fichiers audio
    y, sr = librosa.load(audio_path, sr=None)
    originalNoise, srNoise = librosa.load(noise_path, sr=None)

    if sr != srNoise:
        originalNoise = librosa.resample(originalNoise, orig_sr=srNoise, target_sr=sr)

    # TODO Faire 4 secondes de signal
    if len(y) < 4 * sr:
        yResized = np.zeros(4*sr)
        yResized[:len(y)] = y
    
    else:
        yResized = y[:4*sr]

    # Sélection puis somme de deux parties du bruit cafet de la taille du fichier audio y
    n1 = random.randint(0, len(originalNoise) - len(yResized))
    n2 = random.randint(0, len(originalNoise) - len(yResized))
    noise1 = originalNoise[n1:n1+len(yResized)]
    noise2 = originalNoise[n2:n2+len(yResized)]
    
    noise = noise1 + noise2


    # Calculer la puissance du signal et du bruit
    y_power = power(yResized)
    noise_power = power(noise)

    # Calculer le facteur de mise à l'échelle pour atteindre le niveau de SNR désiré
    scale_factor = (y_power / noise_power) * (10 ** (-snr_level / 10))
    scale_noise = noise * np.sqrt(scale_factor)

    # Ajouter le bruit au signal d'origine
    combined_signal = yResized + scale_noise

    # Enregistrer le nouveau fichier audio
    output_path = os.path.join(output_folder, os.path.basename(audio_path))
    output_noise_path = os.path.join(output_noise_folder, os.path.basename(audio_path))
    audio_folder_cut_path = os.path.join(audio_folder_cut, os.path.basename(audio_path))

    sf.write(audio_folder_cut_path, yResized, sr)
    sf.write(output_path, combined_signal, sr)
    sf.write(output_noise_path, scale_noise, sr)

    print(f"Le fichier audio modifié a été enregistré sous : {output_path}")
    print(SNRin_dB(yResized, np.sqrt(scale_factor)*noise))

