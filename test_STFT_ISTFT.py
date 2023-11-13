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

# Transformation (dans ce cas, simplement l'identité)
transformed_stft = np.copy(stft_result)

# Synthèse ISTFT
reconstructed_signal = librosa.istft(transformed_stft, hop_length=H, win_length=N, window='hann')

# Afficher le signal original et le signal reconstruit
plt.figure(figsize=(12, 6))

time_vector_original = np.linspace(0, len(y)/sr, num=len(y))
time_vector_reconstructed = np.linspace(0, len(reconstructed_signal)/sr, num=len(reconstructed_signal))


plt.plot(y[8000:8050], linewidth = 4)
plt.plot(reconstructed_signal[8000:8050])
plt.show()

# Écouter le signal reconstruit
display(Audio(reconstructed_signal, rate=sr))