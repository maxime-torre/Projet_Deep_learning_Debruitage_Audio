import os
import shutil
import numpy as np

# Chemin du dossier source
source_path = r"C:\Users\torre\Documents\Sicom 3A\Projet_Parole_Audio_SICOM_3A\LibriSpeech\dev-clean"

# Chemin du dossier de destination
destination_path = r"C:\Users\torre\Documents\Sicom 3A\Projet_Parole_Audio_SICOM_3A\LibriSpeech\combined_flac"

# Créer le dossier de destination s'il n'existe pas
if not os.path.exists(destination_path):
    os.makedirs(destination_path)

# Initialiser le numéro de séquence pour le renommage
sequence_number = 1

# Fonction pour déplacer et renommer les fichiers
def move_and_rename_files(root, file):
    global sequence_number  # Utiliser la variable globale pour la séquence
    # Chemin complet du fichier source
    full_file_path = os.path.join(root, file)
    
    # Nouveau nom de fichier avec séquence
    new_filename = f"{sequence_number}.flac"
    sequence_number += 1
    
    # Chemin complet du fichier de destination
    destination_file_path = os.path.join(destination_path, new_filename)
    
    # Déplacer et renommer le fichier
    shutil.move(full_file_path, destination_file_path)

# Parcourir tous les dossiers et sous-dossiers à partir du chemin source
for current_path, subfolders, files in os.walk(source_path):
    for file in files:
        if file.endswith('.flac'):
            move_and_rename_files(current_path, file)

print(f"Tous les fichiers .flac ont été déplacés et renommés dans '{destination_path}'.")
