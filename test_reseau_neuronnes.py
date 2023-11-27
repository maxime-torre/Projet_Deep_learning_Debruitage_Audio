import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf
from pathlib import Path
import librosa
import librosa.display
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from sklearn.model_selection import train_test_split
from IPython.display import Audio, display


from load_spectrograms_to_tensor import load_spectrograms_to_tensor
from main import create_masks_into_tensor
from mask_binary import  calculate_amplitude_spectrogram, mask_binary

## Objectif débruité ce signal 
# signal_and_noise = Path.cwd() / "data" / "noised_data" / "10" / "20.flac" <-- signal à débruiter

# Charger les données
wanted_snr = [10]
N = 2048
H = 1024

X = load_spectrograms_to_tensor(wanted_snr) # Signals + noise tensor data
Y = create_masks_into_tensor(wanted_snr)    # Masques tensor data

# Diviser les données en ensembles d'entraînement et de test
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2)

# Construction du modèle
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(Xtrain.shape[1], 1)),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(50, activation='relu'),
    Dense(Ytrain.shape[1], activation='sigmoid')  # La sortie a la même taille que le masque Y
])

# Compilation du modèle
model.compile(optimizer='adam', loss='mean_squared_error')

# Entraînement du modèle
model.fit(Xtrain, Ytrain, epochs=10, batch_size=32, validation_data=(Xtest, Ytest))

# Chargement et traitement d'un signal spécifique pour débruitage
# Remplacer ceci par votre propre méthode de chargement et traitement du signal
signal_and_noise_path = Path.cwd() / "data" / "noised_data" / "10" / "20.flac"
signal_to_denoise, sr = librosa.load(signal_and_noise_path) # Charger et transformer le signal 'signal_and_noise_path'
spectrogram_to_denoise = calculate_amplitude_spectrogram(signal_to_denoise, N , H)

# Prédiction du masque pour le signal spécifique
predicted_mask = model.predict(signal_to_denoise)

# Appliquer le masque pour obtenir le signal débruité
# À compléter en fonction de votre méthode de traitement du signal
denoised_spectrogram = spectrogram_to_denoise*predicted_mask

#Retour au signal temporel avec l'ISTFT
reconstructed_signal = librosa.istft(denoised_spectrogram, hop_length=H, win_length=N, window='hann')
print(f"len(reconstructed_signal) : {len(reconstructed_signal)}")
display(Audio(reconstructed_signal, rate=sr))
