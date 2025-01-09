# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 10:30:30 2024

@author: fdelap01
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from numpy.polynomial.polynomial import Polynomial

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

def detect_outliers_with_model(positions, frame_numbers, method="savgol", degree=3, window_size=11, segment_size=100, threshold=2):
    """
    Détecte et supprime les outliers en utilisant Savitzky-Golay ou polynômes par portion.
    """
    positions = np.array(positions)
    valid_mask = ~np.isnan(positions)  # Masque des valeurs valides
    x = frame_numbers[valid_mask]
    y = positions[valid_mask]

    fitted_positions = np.full_like(positions, np.nan, dtype=np.float64)

    if method == "savgol":
        # Application de Savitzky-Golay
        fitted_positions = savgol_filter(positions, window_length=window_size, polyorder=degree, mode='interp')
    elif method == "segment_poly":
        # Polynômes par portion
        for start in range(0, len(frame_numbers), segment_size):
            end = min(start + segment_size, len(frame_numbers))  # Limite `end` à la taille maximale
            segment_mask = (frame_numbers >= frame_numbers[start]) & (frame_numbers < frame_numbers[end - 1])
            segment_x = frame_numbers[segment_mask & valid_mask]
            segment_y = positions[segment_mask & valid_mask]

            if len(segment_x) > degree:  # Nécessite au moins degree+1 points
                poly_coeffs = Polynomial.fit(segment_x, segment_y, degree)
                fitted_positions[segment_mask] = poly_coeffs(frame_numbers[segment_mask])
    else:
        raise ValueError("Invalid method. Use 'savgol' or 'segment_poly'.")

    # Détection des outliers
    residuals = positions - fitted_positions
    std_dev = np.nanstd(residuals)
    outliers = np.abs(residuals) > threshold * std_dev

    # Suppression des outliers
    cleaned_positions = positions.copy()
    cleaned_positions[outliers] = np.nan

    return cleaned_positions, fitted_positions, outliers


def process_positions(folder_path, method="savgol", degree=3, window_size=11, segment_size=100, threshold=2):
    """
    Charge les JSON, applique la méthode spécifiée (Savitzky-Golay ou polynômes par portion) pour détecter les outliers.
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

        # Application de la méthode choisie
        x_cleaned, x_fitted, x_outliers = detect_outliers_with_model(
            x, frame_numbers, method=method, degree=degree, window_size=window_size, segment_size=segment_size, threshold=threshold
        )
        y_cleaned, y_fitted, y_outliers = detect_outliers_with_model(
            y, frame_numbers, method=method, degree=degree, window_size=window_size, segment_size=segment_size, threshold=threshold
        )

        ## GRAPH : Positions brutes, nettoyées, et ajustées
        plt.figure(figsize=(10, 6))
        plt.plot(frame_numbers, x, 'r-', alpha=0.5, label='Raw X')
        plt.plot(frame_numbers, x_fitted, 'b-', label=f'Fitted X ({method})')
        plt.plot(frame_numbers, x_cleaned, 'g-', label='Cleaned X')
        plt.plot(frame_numbers, y, 'orange', alpha=0.5, label='Raw Y')
        plt.plot(frame_numbers, y_fitted, 'purple', label=f'Fitted Y ({method})')
        plt.plot(frame_numbers, y_cleaned, 'green', label='Cleaned Y')
        #plt.scatter(frame_numbers[x_outliers], x[x_outliers], color='red', label='Outliers X', zorder=5)
        #plt.scatter(frame_numbers[y_outliers], y[y_outliers], color='purple', label='Outliers Y', zorder=5)
        plt.title(f"{keypoint_name} - Positions with {method} Outlier Detection")
        plt.legend()
        plt.grid()
        plt.show()

# Chemin vers ton dossier JSON
folder_path = r"C:\ProgramData\Projets_Florian\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 2_traitement_new calib\Trial_2\pose-associated\Cut_HideObj_20241024_132404-CAMERA03_json"

# Appel de la fonction principale avec Savitzky-Golay
process_positions(folder_path, method="savgol", degree=3, window_size=51, threshold=1)

# Appel de la fonction principale avec polynômes par portion
#process_positions(folder_path, method="segment_poly", degree=3, segment_size=50, threshold=1)


