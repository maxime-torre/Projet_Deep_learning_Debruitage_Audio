import os
from librosa import load, resample
import numpy as np
import random
import soundfile as sf
from modules import SNR_dB, power

def add_background_noise(audio_path, noise_path, output_folder, snr_level=10):
    """
    Ajoute un bruit de fond de cafétaria à un fichier audio et sauvegarde le nouveau fichier.

    :param audio_path: Chemin vers le fichier audio d'origine.
    :param noise_path: Chemin vers le fichier audio de bruit de fond.
    :param output_folder: Dossier pour enregistrer l'audio modifié.
    :param snr_level: Niveau de rapport signal/bruit souhaité.
    """
    # Charger les fichiers audio
    y, sr = load(audio_path, sr=None)
    originalNoise, srNoise = load(noise_path, sr=None)

    if sr != srNoise:
        originalNoise = resample(originalNoise, orig_sr=srNoise, target_sr=sr)

    # Sélection puis somme de deux parties du bruit cafet de la taille du fichier audio y
    n1 = random.randint(0, len(originalNoise) - len(y))
    n2 = random.randint(0, len(originalNoise) - len(y))
    noise1 = originalNoise[n1:n1+len(y)]
    noise2 = originalNoise[n2:n2+len(y)]
    
    noise = noise1 + noise2


    # Calculer la puissance du signal et du bruit
    y_power = power(y)
    noise_power = power(noise)

    # Calculer le facteur de mise à l'échelle pour atteindre le niveau de SNR désiré
    scale_factor = (y_power / noise_power) * (10 ** (-snr_level / 10))

    # Ajouter le bruit au signal d'origine
    combined_signal = y + noise * np.sqrt(scale_factor)

    # Enregistrer le nouveau fichier audio
    output_path = os.path.join(output_folder, os.path.basename(audio_path))
    sf.write(output_path, combined_signal, sr)

    print(f"Le fichier audio modifié a été enregistré sous : {output_path}")
    print(SNR_dB(y, np.sqrt(scale_factor)*noise))


# Spécifier les chemins des fichiers et dossiers
audio_folder = r"C:\Users\torre\Documents\Sicom 3A\Projet_Parole_Audio_SICOM_3A\raw_data"  # chemin du dossier contenant vos fichiers flac
noise_file = "C:\\Users\\torre\\Documents\\Sicom 3A\\Projet_Parole_Audio_SICOM_3A\\babble_16k.wav"  # chemin du fichier de bruit
output_folder = "C:\\Users\\torre\\Documents\\Sicom 3A\\Projet_Parole_Audio_SICOM_3A\\noised_data"  # dossier de sortie pour les fichiers audio bruités

# Créer le dossier de sortie s'il n'existe pas
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Traiter tous les fichiers .flac dans le dossier
for file in os.listdir(audio_folder):
    if file.endswith('.flac'):
        # Chemin complet du fichier audio
        full_audio_path = os.path.join(audio_folder, file)

        # Appeler la fonction pour ajouter un bruit de fond
        add_background_noise(full_audio_path, noise_file, output_folder, snr_level = random.randint(-10,10))
