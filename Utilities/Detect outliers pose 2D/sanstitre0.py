import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Mapping des keypoints avec les bons IDs du modèle HALPE_26
HALPE_26_KEYPOINTS = {
    0: "Nose", 17: "Head", 18: "Neck",
    5: "LShoulder", 7: "LElbow", 9: "LWrist",
    6: "RShoulder", 8: "RElbow", 10: "RWrist",
    11: "LHip", 13: "LKnee", 15: "LAnkle",
    12: "RHip", 14: "RKnee", 16: "RAnkle",
    20: "LBigToe", 22: "LSmallToe", 24: "LHeel",
    21: "RBigToe", 23: "RSmallToe", 25: "RHeel",
    19: "Hip"
}

def compute_velocity(positions, frame_numbers):
    """ Calcule la vitesse à partir des positions en utilisant des différences discrètes. """
    velocities = np.diff(positions) / np.diff(frame_numbers)
    velocities = np.append(velocities, 0)  # Ajouter 0 pour conserver la même taille
    return velocities

def plot_keypoints_and_velocity(folder_path):
    """
    Trace pour chaque keypoint les positions X/Y et l'indice de confiance sur un premier graphique,
    puis les vitesses X/Y et l'indice de confiance sur un second graphique.

    Arguments :
    - folder_path : Chemin vers le dossier contenant les fichiers JSON.
    """
    # Initialisation des dictionnaires pour stocker les positions X, Y et les indices de confiance
    keypoints_x = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    keypoints_y = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    keypoints_confidence = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    frame_numbers = []

    # Parcourt tous les fichiers JSON du dossier
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                people = data.get("people", [])
                
                if people:
                    pose_keypoints = people[0].get("pose_keypoints_2d", [])
                    if pose_keypoints:
                        # Stocke les coordonnées X, Y et l'indice de confiance pour chaque keypoint
                        for keypoint_id, keypoint_name in HALPE_26_KEYPOINTS.items():
                            x = pose_keypoints[keypoint_id * 3]       # Coordonnée X
                            y = pose_keypoints[keypoint_id * 3 + 1]   # Coordonnée Y
                            c = pose_keypoints[keypoint_id * 3 + 2]   # Indice de confiance
                            keypoints_x[keypoint_name].append(x)
                            keypoints_y[keypoint_name].append(y)
                            keypoints_confidence[keypoint_name].append(c)
                        frame_number = int(filename.split('_')[-1].split('.')[0])
                        frame_numbers.append(frame_number)

    # Convertir en numpy array
    frame_numbers = np.array(frame_numbers)

    # Tracer pour chaque keypoint
    for keypoint_name in HALPE_26_KEYPOINTS.values():
        # Récupérer les données
        x = np.array(keypoints_x[keypoint_name])
        y = np.array(keypoints_y[keypoint_name])
        confidence = np.array(keypoints_confidence[keypoint_name])

        # Calcul des vitesses
        velocity_x = compute_velocity(x, frame_numbers)
        velocity_y = compute_velocity(y, frame_numbers)

        ## PREMIER GRAPH : Position X/Y et Indice de Confiance
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(frame_numbers, x, 'r-', label='X Position')
        ax1.plot(frame_numbers, y, 'b-', label='Y Position')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Pixel Position', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        ax2 = ax1.twinx()
        ax2.plot(frame_numbers, confidence, 'g-', label='Confidence')
        ax2.set_ylabel('Confidence Score (0-1)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0, 1)

        fig.suptitle(f"{keypoint_name} - Position and Confidence")
        fig.tight_layout()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.grid()
        plt.show()

        ## SECOND GRAPH : Vitesse X/Y et Indice de Confiance
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(frame_numbers, velocity_x, 'r-', label='X Velocity')
        ax1.plot(frame_numbers, velocity_y, 'b-', label='Y Velocity')
        ax1.set_xlabel('Frame Number')
        ax1.set_ylabel('Velocity (Pixels/frame)', color='black')
        ax1.tick_params(axis='y', labelcolor='black')

        ax2 = ax1.twinx()
        ax2.plot(frame_numbers, confidence, 'g-', label='Confidence')
        ax2.set_ylabel('Confidence Score (0-1)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0, 1)

        fig.suptitle(f"{keypoint_name} - Velocity and Confidence")
        fig.tight_layout()
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.grid()
        plt.show()

# Chemin vers ton dossier
folder_path = r"C:\ProgramData\Projets_Florian\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 1_traitement_new calib\Trial_2\pose-associated\sync_20241024_113719-CAMERA01_json"

# Appel de la fonction
plot_keypoints_and_velocity(folder_path)
