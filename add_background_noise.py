import os
from pydub import AudioSegment
import soundfile as sf
import numpy as np

def add_background_noise(audio_path, noise_path, output_folder, snr_level=10):
    """
    Ajoute un bruit de fond à un fichier audio et sauvegarde le nouveau fichier.

    :param audio_path: Chemin vers le fichier audio d'origine.
    :param noise_path: Chemin vers le fichier audio de bruit de fond.
    :param output_folder: Dossier pour enregistrer l'audio modifié.
    :param snr_level: Niveau de rapport signal/bruit souhaité.
    """
    # Charger les fichiers audio
    data, samplerate = sf.read(audio_path)
    noise, _ = sf.read(noise_path)

    # Assurer que le bruit peut couvrir l'audio
    while len(noise) < len(data):
        noise = np.concatenate([noise, noise])

    # Tronquer le bruit pour qu'il ait la même longueur que l'audio d'origine
    noise = noise[:len(data)]

    # Calculer la puissance du signal et du bruit
    signal_power = np.sum(data ** 2)
    noise_power = np.sum(noise ** 2)

    # Calculer le facteur de mise à l'échelle pour atteindre le niveau de SNR désiré
    scale_factor = (signal_power / noise_power) * (10 ** (-snr_level / 10))
    noise_scaled = noise * np.sqrt(scale_factor)

    # Ajouter le bruit au signal d'origine
    combined_signal = data + noise_scaled

    # S'assurer que les valeurs sont dans les limites pour éviter la distorsion
    combined_signal = np.clip(combined_signal, -1.0, 1.0)

    # Enregistrer le nouveau fichier audio
    output_path = os.path.join(output_folder, os.path.basename(audio_path))
    sf.write(output_path, combined_signal, samplerate)

    print(f"Le fichier audio modifié a été enregistré sous : {output_path}")


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
        add_background_noise(full_audio_path, noise_file, output_folder)
