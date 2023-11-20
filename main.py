import os

from pathlib import Path
from add_background_noise import add_background_noise
from load_spectrograms_to_tensor import load_spectrograms_to_tensor
from modules import normalizeDividingByMax


do_noise_data = True
create_tensor = False


normalizer = normalizeDividingByMax


def noise_data():
    audio_folder = Path.cwd() / "data" / "raw_data" # chemin du dossier contenant les fichiers flac
    noise_file = Path.cwd() / "data" / "babble_16k.wav"  # chemin du fichier de bruit
    output_folder = Path.cwd() / "data" / "noised_data"  # dossier de sortie pour les fichiers audio bruités
    output_noise_folder = Path.cwd() / "data" / "only_noise" # dossier de sortie pour stocker les bruits générés aléatoirement

    # Créer le dossier de sortie s'il n'existe pas
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        for i in range(-10, 11):
            os.makedirs(output_folder / f"{i}")

    if not os.path.exists(output_noise_folder):
        os.makedirs(output_noise_folder)


    # Traiter tous les fichiers .flac dans le dossier
    i=0
    for file in os.listdir(audio_folder):
        if file.endswith('.flac'):

            SNRvalue = -10 + i%21
            full_audio_path = audio_folder / file
            full_output_folder_path = output_folder / f"{SNRvalue}"
            
            # Appeler la fonction pour ajouter un bruit de fond
            add_background_noise(full_audio_path, noise_file, full_output_folder_path, output_noise_folder, snr_level = SNRvalue)
        
        i += 1



if __name__ == '__main__':
    if do_noise_data:
        noise_data()

    tensor = None
    if create_tensor:
        spectrogram_path = Path.cwd() / "data" / "Spectros"

        tensor = load_spectrograms_to_tensor(spectrogram_path)

        normalizedTensor = normalizer(tensor)
    