import os
import json
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

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


def gaussian_moving_average_filter(data, sigma=2):
    """ Moyenne mobile pondérée par une gaussienne pour lisser les données. """
    return gaussian_filter1d(data, sigma=sigma, mode='nearest')

def remove_dynamic_outliers(data, window_size=20, threshold=2):
    """ Élimine les outliers dynamiques (± 2 écarts-types dans une fenêtre glissante). """
    clean_data = data.copy()
    series = pd.Series(data)
    
    rolling_mean = series.rolling(window=window_size, center=True).mean()
    rolling_std = series.rolling(window=window_size, center=True).std()

    for i in range(len(series)):
        if abs(series[i] - rolling_mean[i]) > threshold * rolling_std[i]:
            clean_data[i] = rolling_mean[i]
    return clean_data

def process_keypoints_with_filtering(folder_path):
    """
    Traite les fichiers JSON pour chaque keypoint :
    - Affiche les positions brutes avec l'indice de confiance.
    - Applique un filtrage gaussien et supprime les outliers sur les positions.
    - Calcule les vitesses brutes et filtrées.
    - Affiche 4 graphiques pour chaque keypoint.
    """
    keypoints_x = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    keypoints_y = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    keypoints_confidence = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    frame_numbers = []

    # Chargement des données JSON
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
                            c = pose_keypoints[keypoint_id * 3 + 2]
                            keypoints_x[keypoint_name].append(x)
                            keypoints_y[keypoint_name].append(y)
                            keypoints_confidence[keypoint_name].append(c)
                        frame_number = int(filename.split('_')[-1].split('.')[0])
                        frame_numbers.append(frame_number)

    frame_numbers = np.array(frame_numbers)

    # Traitement pour chaque keypoint
    for keypoint_name in HALPE_26_KEYPOINTS.values():
        x = np.array(keypoints_x[keypoint_name])
        y = np.array(keypoints_y[keypoint_name])
        confidence = np.array(keypoints_confidence[keypoint_name])

        # Filtrage des positions
        x_filtered = remove_dynamic_outliers(gaussian_moving_average_filter(x, sigma=2))
        y_filtered = remove_dynamic_outliers(gaussian_moving_average_filter(y, sigma=2))

        # Vitesses brutes
        velocity_x_raw = np.gradient(x, frame_numbers)
        velocity_y_raw = np.gradient(y, frame_numbers)

        # Vitesses filtrées
        velocity_x_filtered = np.gradient(x_filtered, frame_numbers)
        velocity_y_filtered = np.gradient(y_filtered, frame_numbers)

        ## GRAPH 1 : Positions brutes avec indice de confiance
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(frame_numbers, x, 'r-', label='Raw X')
        ax1.plot(frame_numbers, y, 'b-', label='Raw Y')
        ax2 = ax1.twinx()
        ax2.plot(frame_numbers, confidence, 'g-', label='Confidence', alpha=0.7)
        ax2.set_ylabel('Confidence (0-1)', color='green')
        plt.title(f"{keypoint_name} - Raw Positions and Confidence")
        ax1.legend()
        plt.show()

        ## GRAPH 2 : Positions filtrées
        plt.figure(figsize=(10, 6))
        plt.plot(frame_numbers, x_filtered, 'r-', label='Filtered X')
        plt.plot(frame_numbers, y_filtered, 'b-', label='Filtered Y')
        plt.title(f"{keypoint_name} - Filtered Positions")
        plt.legend()
        plt.show()

        ## GRAPH 3 : Vitesses brutes avec indice de confiance
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(frame_numbers, velocity_x_raw, 'r-', label='Raw Velocity X')
        ax1.plot(frame_numbers, velocity_y_raw, 'b-', label='Raw Velocity Y')
        ax2 = ax1.twinx()
        ax2.plot(frame_numbers, confidence, 'g-', label='Confidence', alpha=0.7)
        ax2.set_ylabel('Confidence (0-1)', color='green')
        plt.title(f"{keypoint_name} - Raw Velocity and Confidence")
        ax1.legend()
        plt.show()

        ## GRAPH 4 : Vitesses filtrées
        plt.figure(figsize=(10, 6))
        plt.plot(frame_numbers, velocity_x_filtered, 'r-', label='Filtered Velocity X')
        plt.plot(frame_numbers, velocity_y_filtered, 'b-', label='Filtered Velocity Y')
        plt.title(f"{keypoint_name} - Filtered Velocities")
        plt.legend()
        plt.show()


# Chemin vers ton dossier
folder_path = r"C:\ProgramData\Projets_Florian\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 1_traitement_new calib\Trial_2_short_depart\pose-associated\Cut_sync_20241024_113719-CAMERA01_json"

# Appel de la fonction
process_keypoints_with_filtering(folder_path)
