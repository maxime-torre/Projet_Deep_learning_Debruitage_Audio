import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Audio, display
import os

# Spécifier les chemins des fichiers et dossiers
audio_folder = r"C:\Users\torre\Documents\Sicom 3A\Projet_Parole_Audio_SICOM_3A\data\raw_data"  # chemin du dossier contenant vos fichiers flac
noise_file = "C:\\Users\\torre\\Documents\\Sicom 3A\\Projet_Parole_Audio_SICOM_3A\\data\\babble_16k.wav"  # chemin du fichier de bruit
output_folder = "C:\\Users\\torre\\Documents\\Sicom 3A\\Projet_Parole_Audio_SICOM_3A\\data\\noised_data"  # dossier de sortie pour les fichiers audio bruités

# Traiter tous les fichiers .flac dans le dossier
for file in os.listdir(audio_folder):
    if file.endswith('.flac'):
        # Chemin complet du fichier audio
        full_audio_path = os.path.join(audio_folder, file)
        break

# 1. Charger un fichier audio
y, sr = librosa.load(full_audio_path)

# 2. Calculer la STFT
N = 2048
H = 1024

# Analyse STFT
stft_result = librosa.stft(y, n_fft=N, hop_length=H, win_length=N, window='hann')

# 3. Calculer le spectrogramme de puissance
D = np.abs(stft_result)**2

print(D)

# 4. Afficher le spectrogramme de puissance sur une échelle logarithmique
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, hop_length=H, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogramme de puissance')
plt.tight_layout()
plt.show()