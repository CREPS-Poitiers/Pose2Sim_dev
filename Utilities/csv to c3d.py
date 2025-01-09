import pandas as pd
import ezc3d
import numpy as np

def csv_to_c3d_with_transform(csv_file, output_c3d):
    """
    Convertit un fichier CSV contenant des coordonnées X, Y, Z en fichier C3D,
    applique une rotation de 90° autour de l'axe X et ajoute 1 mètre (1000 mm) à Z.
    
    :param csv_file: Chemin vers le fichier CSV.
    :param output_c3d: Chemin de sortie du fichier C3D.
    """
    # Charger le fichier CSV
    try:
        data = pd.read_csv(csv_file, sep=None, engine='python', header=None, skiprows=1)
        print(f"Fichier CSV chargé : {data.shape[0]} frames, {data.shape[1]} colonnes.")
    except Exception as e:
        print("Erreur lors du chargement du fichier :", e)
        return

    # Vérification du nombre de colonnes
    num_markers = data.shape[1] // 3
    if data.shape[1] % 3 != 0:
        raise ValueError("Le fichier CSV doit contenir un nombre de colonnes multiple de 3 (X, Y, Z).")

    # Initialisation des points (4, nMarkers, nFrames)
    num_frames = data.shape[0]
    points = np.zeros((4, num_markers, num_frames))

    # Matrice de rotation 90° autour de l'axe X
    rotation_matrix_x = np.array([
        [-1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ])

    # Remplir les données : X, Y, Z pour chaque frame et chaque marqueur
    for frame_idx in range(num_frames):
        for marker_idx in range(num_markers):
            x = data.iloc[frame_idx, marker_idx * 3 + 0]  # X
            y = data.iloc[frame_idx, marker_idx * 3 + 1]  # Y
            z = data.iloc[frame_idx, marker_idx * 3 + 2]  # Z

            # Appliquer la rotation autour de l'axe X
            rotated = rotation_matrix_x @ np.array([x, y, z])
            
            # Ajouter 1 mètre (1000 mm) à la coordonnée Z
            rotated[2] += 1  

            # Insérer dans la structure
            points[0, marker_idx, frame_idx] = rotated[0]  # X
            points[1, marker_idx, frame_idx] = rotated[1]  # Y
            points[2, marker_idx, frame_idx] = rotated[2]  # Z
            points[3, marker_idx, frame_idx] = 0  # Résidu par défaut à 0

    # Création d'un fichier C3D
    c3d = ezc3d.c3d()
    c3d['data']['points'] = points

    # Ajouter les labels des marqueurs
    labels = [f"Marker_{i+1}" for i in range(num_markers)]
    c3d['parameters']['POINT']['LABELS']['value'] = labels
    c3d['parameters']['POINT']['UNITS']['value'] = ['m']

    # Définir la fréquence d'échantillonnage (par exemple, 100 Hz)
    c3d['parameters']['POINT']['RATE']['value'] = [100]

    # Sauvegarder le fichier C3D
    c3d.write(output_c3d)
    print(f"Fichier C3D généré avec succès : {output_c3d}")


# Paramètres d'entrée et de sortie
csv_file = r"C:\Users\fdelap01\motionagformer\demo\output\squatlab_34\keypoints_3D.csv"
output_c3d = r"C:\Users\fdelap01\motionagformer\demo\output\squatlab_34\output_keypoints_rotated.c3d"

# Exécuter la conversion avec transformation
csv_to_c3d_with_transform(csv_file, output_c3d)
