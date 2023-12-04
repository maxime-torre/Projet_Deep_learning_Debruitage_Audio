import os
import numpy as np
import random
import soundfile as sf
from modules import SNRin_dB, power

import librosa
from pathlib import Path


def add_background_noise(paths, filename, snr_level=10):
    """
    Ajoute un bruit aléatoire de cafétaria à un fichier audio et sauvegarde le nouveau fichier,
    le bruit crée dans "only_noise" et la version tronqué du signal de base dans "raw_data_cut"

    :param paths: Objet contenant tous les paths utiles et génériques.
    :param filename: Nom du fichier audio originale, par exemple "123.flac".
    :param snr_level: Niveau de rapport signal/bruit souhaité.
    """
    # Charger le fichier audio
    y, sr = librosa.load(paths.raw_folder / filename, sr=None)
    originalNoise, srNoise = librosa.load(paths.babble_file, sr=None)

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
    noised_signal = yResized + scale_noise

    # Enregistrer les nouveaux fichiers audios
    noised_folder = paths.noised_folder / f"{snr_level}"

    noised_path = os.path.join(noised_folder, filename)
    only_noise_path = os.path.join(paths.only_noise_folder, filename)
    raw_cut_path = os.path.join(paths.raw_cut_folder, filename)

    sf.write(noised_path, noised_signal, sr)
    sf.write(only_noise_path, scale_noise, sr)
    sf.write(raw_cut_path, yResized, sr)

    # print(f"Le fichier audio modifié a été enregistré sous : {noised_path}")
    # print(SNRin_dB(yResized, np.sqrt(scale_factor)*noise))

def noise_data(paths):
    """
    Ajoute un bruit aléatoire de cafétaria sur tous les signaux stockés dans le dossier "raw_data"

    :param paths: Objet contenant tous les paths utiles et génériques.
    """
    # Traiter tous les fichiers .flac dans le dossier
    i=0
    for file in os.listdir(paths.raw_folder):
        if file.endswith('.flac'):

            SNRvalue = -10 + i%21
            
            add_background_noise(paths, file, snr_level = SNRvalue)
        i += 1