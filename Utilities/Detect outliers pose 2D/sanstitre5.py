import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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

def detect_outliers_with_savgol(positions, frame_numbers, degree=3, window_size=11, threshold=2):
    """
    Détecte les outliers en utilisant Savitzky-Golay et retourne une courbe nettoyée.
    """
    positions = np.array(positions)
    valid_mask = ~np.isnan(positions)  # Masque des valeurs valides

    # Appliquer Savitzky-Golay
    fitted_positions = savgol_filter(positions, window_length=window_size, polyorder=degree, mode='interp')

    # Détection des outliers
    residuals = positions - fitted_positions
    std_dev = np.nanstd(residuals)
    outliers = np.abs(residuals) > threshold * std_dev

    # Création de la courbe nettoyée
    cleaned_positions = positions.copy()
    cleaned_positions[outliers] = np.nan

    return fitted_positions, cleaned_positions, outliers, std_dev

def interpolate_with_fitted_curve(cleaned_positions, fitted_positions):
    """
    Interpole les points manquants (NaN) dans la courbe nettoyée en utilisant la courbe fitted.
    """
    interpolated_positions = cleaned_positions.copy()

    # Remplir les NaN avec les valeurs de la courbe fitted
    nan_mask = np.isnan(cleaned_positions)
    interpolated_positions[nan_mask] = fitted_positions[nan_mask]

    return interpolated_positions


def process_keypoints(folder_path, detection_params):
    """
    Processus principal pour détecter les outliers, nettoyer et interpoler uniquement les points manquants.
    """
    keypoints_x = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    keypoints_y = {name: [] for name in HALPE_26_KEYPOINTS.values()}
    keypoints_conf = {name: [] for name in HALPE_26_KEYPOINTS.values()}
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

        # Étape 1 : Détection des outliers
        x_fitted, x_cleaned, x_outliers, _ = detect_outliers_with_savgol(
            x, frame_numbers,
            degree=detection_params["degree"],
            window_size=detection_params["window_size"],
            threshold=detection_params["threshold"]
        )
        y_fitted, y_cleaned, y_outliers, _ = detect_outliers_with_savgol(
            y, frame_numbers,
            degree=detection_params["degree"],
            window_size=detection_params["window_size"],
            threshold=detection_params["threshold"]
        )

        # Étape 2 : Interpolation basée sur la courbe fitted
        x_interpolated = interpolate_with_fitted_curve(x_cleaned, x_fitted)
        y_interpolated = interpolate_with_fitted_curve(y_cleaned, y_fitted)

        ## GRAPH 1 : Raw, fitted, cleaned positions (X and Y) + Outlier Indicators
        plt.figure(figsize=(12, 8))
        plt.plot(frame_numbers, x, 'r-', alpha=0.5, label='Raw X')
        plt.plot(frame_numbers, x_fitted, 'b-', label='Fitted X (Outlier Detection)')
        plt.plot(frame_numbers, x_cleaned, 'g-', label='Cleaned X (Outliers Removed)')
        plt.plot(frame_numbers, y, 'orange', alpha=0.5, label='Raw Y')
        plt.plot(frame_numbers, y_fitted, color='purple', label='Fitted Y (Outlier Detection)')
        plt.plot(frame_numbers, y_cleaned, color='green', label='Cleaned Y (Outliers Removed)')

        # Lignes verticales pour les outliers détectés en X (rouge)
        for frame in frame_numbers[x_outliers]:
            plt.axvline(frame, color='red', linestyle='--', alpha=0.3, linewidth=0.8, label='_nolegend_')

        # Lignes verticales pour les outliers détectés en Y (bleu)
        for frame in frame_numbers[y_outliers]:
            plt.axvline(frame, color='blue', linestyle='--', alpha=0.3, linewidth=0.8, label='_nolegend_')

        plt.title(f"{keypoint_name} - Outlier Detection (X and Y)")
        plt.legend()
        plt.grid()
        plt.show()

        ## GRAPH 2 : Raw vs final interpolated positions (X and Y) + Confidence
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Premier axe (positions)
        ax1.plot(frame_numbers, x, 'r-', alpha=0.5, label='Raw X')
        ax1.plot(frame_numbers, x_interpolated, 'b-', label='Final X (Cleaned + Interpolated)')
        ax1.plot(frame_numbers, y, 'orange', alpha=0.5, label='Raw Y')
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

        plt.title(f"{keypoint_name} - Final Interpolation (X and Y) + Confidence")
        plt.show()

# Chemin vers ton dossier JSON
folder_path = r"C:\ProgramData\Projets_Florian\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 2_traitement_new calib\Trial_2\pose-associated\Cut_HideObj_20241024_132404-CAMERA03_json"

# Paramètres pour la détection des outliers
detection_params = {
    "degree": 3,
    "window_size": 31,
    "threshold": 1
}

# Appel de la fonction principale
process_keypoints(folder_path, detection_params)
