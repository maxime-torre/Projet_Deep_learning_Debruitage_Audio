import os
import numpy as np
import soundfile as sf
import pandas as pd

NOISY_DIR = "C:\\Users\\torre\\Documents\\Sicom 3A\\Projet_Parole_Audio_SICOM_3A\\noised_data"
CLEAN_DIR = "C:\\Users\\torre\\Documents\\Sicom 3A\\Projet_Parole_Audio_SICOM_3A\\raw_data"

def load_audio_files(noisy_dir, clean_dir, file_limit=10):
    # Récupérer les noms de fichiers dans les dossiers donnés
    noisy_files = [f for f in os.listdir(noisy_dir) if f.endswith('.flac')]
    clean_files = [f for f in os.listdir(clean_dir) if f.endswith('.flac')]

    # Assurez-vous que les fichiers sont correctement appariés, cela dépend de la façon dont vos fichiers sont organisés.
    # Cela suppose que les fichiers dans les deux dossiers sont dans le même ordre et correspondent exactement.
    noisy_files.sort()
    clean_files.sort()

    data_pairs = []
    # Modifier la boucle pour qu'elle s'arrête après un certain nombre de fichiers
    for i in range(min(file_limit, len(noisy_files), len(clean_files))):
        noisy_file = noisy_files[i]
        clean_file = clean_files[i]

        # Charger les fichiers audio
        noisy, _ = sf.read(os.path.join(noisy_dir, noisy_file))
        clean, _ = sf.read(os.path.join(clean_dir, clean_file))
        
        # Ajouter à notre liste de tuples
        data_pairs.append((noisy, clean))

    return data_pairs

data_pairs = load_audio_files(NOISY_DIR, CLEAN_DIR)

def pad_audio_data(data, target_length):
    """Rembourrer les données audio pour qu'elles aient toutes la même taille."""
    padded_data = []

    for item in data:
        # Si l'item est plus court que la longueur cible, on le rembourre avec des zéros
        if len(item) < target_length:
            padded_item = np.pad(item, (0, target_length - len(item)), 'constant', constant_values=(0,))
        else:
            padded_item = item[:target_length]  # Sinon, on coupe l'item à la longueur cible

        padded_data.append(padded_item)

    return np.array(padded_data)

# Trouver la longueur maximale dans vos données audio
max_length = max(max(len(pair[0]), len(pair[1])) for pair in data_pairs)

# Après le chargement des fichiers, avant de diviser les données en ensembles d'entraînement et de test:
noisy_data = [pair[0] for pair in data_pairs]
clean_data = [pair[1] for pair in data_pairs]

# Appliquer le rembourrage / coupe pour obtenir des enregistrements audio uniformes
noisy_data = pad_audio_data(noisy_data, max_length)
clean_data = pad_audio_data(clean_data, max_length)

# Conversion en np.array pour faciliter la manipulation
noisy_data = np.array(noisy_data)
clean_data = np.array(clean_data)

# Calcul de l'indice de séparation pour 80% des données
split_idx = int(0.8 * len(noisy_data))

# Séparation des données en ensembles d'entraînement et de test
x_train_noisy, x_test_noisy = noisy_data[:split_idx], noisy_data[split_idx:]
y_train_clean, y_test_clean = clean_data[:split_idx], clean_data[split_idx:]

# Conversion en DataFrame si nécessaire pour votre traitement ultérieur
x_train_noisy = pd.DataFrame(x_train_noisy)
y_train_clean = pd.DataFrame(y_train_clean)
x_test_noisy = pd.DataFrame(x_test_noisy)
y_test_clean = pd.DataFrame(y_test_clean)

