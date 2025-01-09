# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:21:07 2024

@author: fdelap01
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline

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


def interpolate_with_neighbors(cleaned_positions, frame_numbers, window_size=10, method="spline"):
    """
    Interpole les points manquants (NaN) en utilisant des points voisins avant et après le segment d'outliers.
    Respecte les premières et dernières frames stabilisées.
    """
    interpolated_positions = cleaned_positions.copy()
    nan_indices = np.where(np.isnan(cleaned_positions))[0]  # Indices des NaN

    if len(nan_indices) == 0:
        return interpolated_positions  # Pas d'outliers

    # Identifier les plages continues de NaN
    nan_segments = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0] + 1)

    for segment in nan_segments:
        start_nan = segment[0]
        end_nan = segment[-1]

        # Vérifier si le segment touche les premières ou dernières frames stabilisées
        if start_nan < window_size // 20 or end_nan > len(cleaned_positions) - window_size // 20:
            continue  # Ignorer les segments stabilisés

        # Prendre des points valides avant et après
        start_valid = max(0, start_nan - window_size)
        end_valid = min(len(frame_numbers), end_nan + window_size + 1)

        valid_mask = ~np.isnan(cleaned_positions[start_valid:end_valid])
        valid_positions = cleaned_positions[start_valid:end_valid][valid_mask]
        valid_frames = frame_numbers[start_valid:end_valid][valid_mask]

        if len(valid_positions) < 2:
            continue  # Pas assez de points valides pour interpoler

        # Choisir la méthode d'interpolation
        if method == "linear":
            interpolator = interp1d(valid_frames, valid_positions, kind="linear", fill_value="extrapolate")
        elif method == "spline":
            interpolator = CubicSpline(valid_frames, valid_positions, bc_type="natural")
        else:
            raise ValueError("Méthode inconnue pour l'interpolation. Utilisez 'linear' ou 'spline'.")

        # Interpoler pour les indices NaN dans le segment
        interpolated_positions[segment] = interpolator(frame_numbers[segment])

    return interpolated_positions


def detect_outliers_with_savgol(positions, frame_numbers, degree=3, window_size=11, threshold=2):
    """
    Détecte les outliers en utilisant Savitzky-Golay et retourne une courbe nettoyée.
    Les premières et dernières `window_size // 2` frames sont remplacées par la même valeur moyenne respective.
    """
    positions = np.array(positions)
    valid_mask = ~np.isnan(positions)  # Masque des valeurs valides
    half_window = window_size // 20

    if valid_mask.sum() < degree + 1:
        # Si pas assez de points pour ajuster, retourner les données telles quelles
        return positions, positions.copy(), np.zeros_like(positions, dtype=bool), 0

    # Stabiliser les premières frames : Remplacement par une valeur constante
    smoothed_positions = positions.copy()
    if half_window <= len(positions):  # Vérifier que la taille des données est suffisante
        start_mean = np.nanmean(positions[0:half_window])  # Moyenne des premières `half_window` frames
        smoothed_positions[0:half_window] = start_mean  # Remplacement constant pour la moitié initiale

    # Stabiliser les dernières frames : Remplacement par une valeur constante
    if half_window <= len(positions):  # Vérifier que la taille des données est suffisante
        end_mean = np.nanmean(positions[-half_window:])  # Moyenne des dernières `half_window` frames
        smoothed_positions[-half_window:] = end_mean  # Remplacement constant pour la moitié finale

    # Appliquer Savitzky-Golay uniquement sur les frames restantes
    smoothed_positions[valid_mask] = savgol_filter(
        smoothed_positions[valid_mask],
        window_length=min(window_size, valid_mask.sum()),
        polyorder=degree,
        mode='interp'
    )

    # Détection des outliers uniquement pour les frames centrales
    residuals = positions - smoothed_positions
    std_dev = np.nanstd(residuals)
    outliers = np.zeros_like(positions, dtype=bool)
    outliers[half_window:-half_window] = np.abs(residuals[half_window:-half_window]) > threshold * std_dev

    # Créer la courbe nettoyée
    cleaned_positions = positions.copy()
    cleaned_positions[outliers] = np.nan

    return smoothed_positions, cleaned_positions, outliers, std_dev


def interpolate_with_fitted_curve(cleaned_positions, fitted_positions):
    """
    Interpole les points manquants (NaN) dans la courbe nettoyée en utilisant la courbe fitted.
    """
    interpolated_positions = cleaned_positions.copy()

    # Remplir les NaN avec les valeurs de la courbe fitted
    nan_mask = np.isnan(cleaned_positions)
    interpolated_positions[nan_mask] = fitted_positions[nan_mask]

    return interpolated_positions

def apply_sliding_window_average(data, window_size=10):
    """
    Applique une moyenne glissante sur les données, ajustant dynamiquement la fenêtre aux extrémités.
    """
    data = np.array(data)
    smoothed_data = np.full_like(data, np.nan, dtype=np.float64)  # Initialiser avec NaN

    for i in range(len(data)):
        start = max(0, i - window_size // 2)  # Début de la fenêtre
        end = min(len(data), i + window_size // 2 + 1)  # Fin de la fenêtre
        window_values = data[start:end]  # Valeurs dans la fenêtre
        smoothed_data[i] = np.nanmean(window_values)  # Moyenne en ignorant les NaN

    return smoothed_data


def process_keypoints_and_update_json(folder_path, detection_params):
    """
    Processus principal pour détecter les outliers, nettoyer et interpoler uniquement les points manquants.
    """
    keypoints_x = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    keypoints_y = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    keypoints_conf = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    frame_numbers = []
    json_files = []
    
    # Chargement des JSON
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.json'):
            filepath = os.path.join(folder_path, filename)
            json_files.append(filepath)
            with open(filepath, 'r') as f:
                data = json.load(f)
                people = data.get("people", [])
                if people:
                    pose_keypoints = people[0].get("pose_keypoints_2d", [])
                    if pose_keypoints:
                        for keypoint_id, keypoint_name in HALPE_26_KEYPOINTS.items():
                            x = pose_keypoints[keypoint_id * 3]
                            y = pose_keypoints[keypoint_id * 3 + 1]
                            conf = pose_keypoints[keypoint_id * 3 + 2]
                            keypoints_x[keypoint_name].append(x)
                            keypoints_y[keypoint_name].append(y)
                            keypoints_conf[keypoint_name].append(conf)
                        frame_number = int(filename.split('_')[-1].split('.')[0])
                        frame_numbers.append(frame_number)

    frame_numbers = np.array(frame_numbers)

    # Traitement par keypoint
    for keypoint_name in HALPE_26_KEYPOINTS.values():
        x = np.array(keypoints_x[keypoint_name])
        y = np.array(keypoints_y[keypoint_name])
        confidence = np.array(keypoints_conf[keypoint_name])

        # Calcul des moyennes des premières et dernières frames
        half_window = detection_params["window_size"] // 20

        # Moyenne des premières frames
        if len(x[:half_window]) > 0:
            start_mean_x = np.nanmean(x[:half_window])
            x[:half_window] = start_mean_x
        if len(y[:half_window]) > 0:
            start_mean_y = np.nanmean(y[:half_window])
            y[:half_window] = start_mean_y

        # Moyenne des dernières frames
        if len(x[-half_window:]) > 0:
            end_mean_x = np.nanmean(x[-half_window:])
            x[-half_window:] = end_mean_x
        if len(y[-half_window:]) > 0:
            end_mean_y = np.nanmean(y[-half_window:])
            y[-half_window:] = end_mean_y

        # Étape 1 : Détection des outliers
        x_smoothed, x_cleaned, x_outliers, _ = detect_outliers_with_savgol(
            x, frame_numbers,
            degree=detection_params["degree"],
            window_size=detection_params["window_size"],
            threshold=detection_params["threshold"]
        )
        y_smoothed, y_cleaned, y_outliers, _ = detect_outliers_with_savgol(
            y, frame_numbers,
            degree=detection_params["degree"],
            window_size=detection_params["window_size"],
            threshold=detection_params["threshold"]
        )

        # Étape 2 : Interpolation basée sur les voisins (pour les NaN centraux uniquement)
        x_interpolated = interpolate_with_neighbors(x_cleaned, frame_numbers, window_size=10, method="spline")
        y_interpolated = interpolate_with_neighbors(y_cleaned, frame_numbers, window_size=10, method="spline")
        
        # Étape 3 : Appliquer la moyenne glissante
        x_interpolated = apply_sliding_window_average(x_interpolated, window_size=20)
        y_interpolated = apply_sliding_window_average(y_interpolated, window_size=20)


        ## GRAPH 2 : Raw vs final interpolated positions (X and Y) + Confidence
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Premier axe (positions)
        ax1.plot(frame_numbers, x, 'r-', alpha=0.5, label='Raw X (Stabilized)')
        ax1.plot(frame_numbers, x_interpolated, 'b-', label='Final X (Cleaned + Interpolated)')
        ax1.plot(frame_numbers, y, 'orange', alpha=0.5, label='Raw Y (Stabilized)')
        ax1.plot(frame_numbers, y_interpolated, color='purple', label='Final Y (Cleaned + Interpolated)')
        ax1.set_xlabel("Frame Number")
        ax1.set_ylabel("Position (pixels)")
        ax1.legend(loc="upper left")
        ax1.grid()

        # Deuxième axe (confiance)
        ax2 = ax1.twinx()
        ax2.plot(frame_numbers, confidence, 'k--', alpha=0.7, label='Confidence')
        ax2.set_ylabel("Confidence")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right")

        plt.title(f"{folder_path} \n - {keypoint_name} - Final Interpolation (X and Y) + Confidence")
        plt.show()
        
        
        # Mise à jour des fichiers JSON avec les positions interpolées
        for idx, json_file in enumerate(json_files):
            if idx >= len(x_interpolated) or idx >= len(y_interpolated):
                print(f"Skipping frame {idx}: Index exceeds interpolated data size")
                continue  # Ignorer les frames dépassant la taille des données interpolées
        
            with open(json_file, 'r') as f:
                data = json.load(f)
        
            people = data.get("people", [])
            if people:
                pose_keypoints = people[0].get("pose_keypoints_2d", [])
                if len(pose_keypoints) >= (keypoint_id + 1) * 3:                    
                    pose_keypoints[keypoint_id * 3] = float(x_interpolated[idx])
                    pose_keypoints[keypoint_id * 3 + 1] = float(y_interpolated[idx])
                    people[0]["pose_keypoints_2d"] = pose_keypoints
                    
                else:
                    print(f"Skipping frame {idx}: Inconsistent keypoints length")
            data["people"] = people
            
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=4)
            

def process_all_cameras(parent_folder, detection_params):
    """
    Parcourt tous les dossiers de caméras dans un dossier parent et applique le traitement à chacun.
    """
    for camera_folder in sorted(os.listdir(parent_folder)):
        folder_path = os.path.join(parent_folder, camera_folder)
        if os.path.isdir(folder_path):
            print(f"Processing camera folder: {camera_folder}")
            process_keypoints_and_update_json(folder_path, detection_params)

# Chemin vers ton dossier JSON
parent_folder = r"C:\ProgramData\Projets_Florian\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 1_traitement_new calib\Trial_2_short_milieu\pose-associated"

# Paramètres pour la détection des outliers
detection_params = {
    "degree": 3,
    "window_size": 81,
    "threshold": 0.5
}

# Lancer le traitement sur tous les dossiers de caméras
process_all_cameras(parent_folder, detection_params)
