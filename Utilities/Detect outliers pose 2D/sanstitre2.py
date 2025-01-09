import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Mapping des keypoints avec les bons IDs du modèle HALPE_26
# Mapping des keypoints avec les bons IDs du modèle HALPE_26
HALPE_26_KEYPOINTS = {
    0: "Nose",
    1: "LEye",
    2: "REye",
    3: "LEar",
    4: "REar",
    5: "LShoulder",
    6: "RShoulder",
    7: "LElbow",
    8: "RElbow",
    9: "LWrist",
    10: "RWrist",
    11: "LHip",
    12: "RHip",
    13: "LKnee",
    14: "Rknee",
    15: "LAnkle",
    16: "RAnkle",
    17: "Head",
    18: "Neck",
    19: "Hip",
    20: "LBigToe",
    21: "RBigToe",
    22: "LSmallToe",
    23: "RSmallToe",
    24: "LHeel",
    25: "RHeel"
}


def detect_and_correct_outliers(positions, frame_numbers, window_size=20, threshold=2):
    """
    Détecte les outliers sur les vitesses locales et les corrige en interpolant les positions.
    Retourne les positions corrigées, les vitesses et les seuils.
    """
    positions = np.array(positions)
    velocities = np.gradient(positions, frame_numbers)  # Vitesses locales
    
    # Moyenne glissante et écart-type pour les vitesses
    mean_velocity = pd.Series(velocities).rolling(window=window_size, center=True).mean()
    std_velocity = pd.Series(velocities).rolling(window=window_size, center=True).std()

    # Détection des frames outliers
    outliers = np.abs(velocities - mean_velocity) > threshold * std_velocity
    clean_positions = positions.copy()
    clean_positions[outliers] = np.nan  # Suppression des outliers

    # # Interpolation des positions manquantes
    # clean_positions = pd.Series(clean_positions).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    return velocities, clean_positions, mean_velocity, std_velocity

def process_keypoints_with_outlier_correction(folder_path, threshold=2):
    """
    Charge les JSON, détecte et corrige les outliers sur les vitesses locales.
    Affiche les vitesses et les positions brutes/corrigées.
    """
    keypoints_x = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    keypoints_y = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    frame_numbers = []

    # Chargement des JSON
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                people = data.get("people", [])
                if people:
                    pose_keypoints = people[0].get("pose_keypoints_2d", [])
                    if pose_keypoints:
                        for keypoint_id, keypoint_name in HALPE_26_KEYPOINTS.items():
                            x = pose_keypoints[keypoint_id * 3]
                            y = pose_keypoints[keypoint_id * 3 + 1]
                            keypoints_x[keypoint_name].append(x)
                            keypoints_y[keypoint_name].append(y)
                        frame_number = int(filename.split('_')[-1].split('.')[0])
                        frame_numbers.append(frame_number)

    frame_numbers = np.array(frame_numbers)

    # Traitement par keypoint
    for keypoint_name in HALPE_26_KEYPOINTS.values():
        x = np.array(keypoints_x[keypoint_name])
        y = np.array(keypoints_y[keypoint_name])

        # Détection et correction des outliers
        velocity_x, x_corrected, mean_vel_x, std_vel_x = detect_and_correct_outliers(x, frame_numbers, threshold=threshold)
        velocity_y, y_corrected, mean_vel_y, std_vel_y = detect_and_correct_outliers(y, frame_numbers, threshold=threshold)

        ## GRAPH 1 : Vitesses locales X avec moyennes glissantes et seuils
        plt.figure(figsize=(10, 6))
        plt.plot(frame_numbers, velocity_x, 'b-', label='Velocity X (Raw)')
        plt.plot(frame_numbers, mean_vel_x, 'g-', label='Sliding Mean')
        plt.plot(frame_numbers, mean_vel_x + threshold * std_vel_x, 'orange', linestyle='--', label=f'+{threshold} Std Dev')
        plt.plot(frame_numbers, mean_vel_x - threshold * std_vel_x, 'orange', linestyle='--', label=f'-{threshold} Std Dev')
        plt.title(f"{keypoint_name} - Local Velocity X with Sliding Thresholds")
        plt.legend()
        plt.grid()
        plt.show()

        ## GRAPH 2 : Vitesses locales Y avec moyennes glissantes et seuils
        plt.figure(figsize=(10, 6))
        plt.plot(frame_numbers, velocity_y, 'b-', label='Velocity Y (Raw)')
        plt.plot(frame_numbers, mean_vel_y, 'g-', label='Sliding Mean')
        plt.plot(frame_numbers, mean_vel_y + threshold * std_vel_y, 'orange', linestyle='--', label=f'+{threshold} Std Dev')
        plt.plot(frame_numbers, mean_vel_y - threshold * std_vel_y, 'orange', linestyle='--', label=f'-{threshold} Std Dev')
        plt.title(f"{keypoint_name} - Local Velocity Y with Sliding Thresholds")
        plt.legend()
        plt.grid()
        plt.show()

        ## GRAPH 3 : Positions X et Y brutes vs corrigées
        plt.figure(figsize=(10, 6))
        plt.plot(frame_numbers, x, 'r-', alpha=0.5, label='Raw X')
        plt.plot(frame_numbers, x_corrected, 'b-', label='Corrected X')
        plt.plot(frame_numbers, y, 'orange', alpha=0.5, label='Raw Y')
        plt.plot(frame_numbers, y_corrected, 'green', label='Corrected Y')
        plt.title(f"{keypoint_name} - Positions (Raw vs Corrected)")
        plt.legend()
        plt.grid()
        plt.show()

# Chemin vers ton dossier JSON
folder_path = r"C:\ProgramData\Projets_Florian\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 1_traitement_new calib\Trial_2_short_milieu\pose-associated\Cut_sync_20241024_113719-CAMERA01_json"

# Appel de la fonction principale avec un seuil (threshold)
process_keypoints_with_outlier_correction(folder_path, threshold=1)
