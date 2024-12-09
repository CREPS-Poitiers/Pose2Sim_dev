# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:12:33 2024

@author: fdelap01
"""

import ezc3d
import numpy as np

def rotation_matrix(angles):
    """
    Create a combined rotation matrix from rotation angles around x, y, and z axes.
    """
    x_angle, y_angle, z_angle = np.radians(angles)

    # Rotation around X-axis
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(x_angle), -np.sin(x_angle)],
        [0, np.sin(x_angle), np.cos(x_angle)]
    ])

    # Rotation around Y-axis
    rot_y = np.array([
        [np.cos(y_angle), 0, np.sin(y_angle)],
        [0, 1, 0],
        [-np.sin(y_angle), 0, np.cos(y_angle)]
    ])

    # Rotation around Z-axis
    rot_z = np.array([
        [np.cos(z_angle), -np.sin(z_angle), 0],
        [np.sin(z_angle), np.cos(z_angle), 0],
        [0, 0, 1]
    ])

    # Combine rotations: Z * Y * X
    return rot_z @ rot_y @ rot_x

def resolve_duplicate_labels(existing_labels, new_labels):
    """
    Resolve duplicate marker labels by appending a suffix (_2, _3, etc.) to duplicates.
    """
    updated_labels = []
    label_counts = {label: 1 for label in existing_labels}  # Track existing labels

    for label in new_labels:
        if label in label_counts:  # It's a duplicate
            label_counts[label] += 1
            updated_labels.append(f"{label}_{label_counts[label]}")
        else:  # No conflict
            label_counts[label] = 1
            updated_labels.append(label)

    return updated_labels

def combine_and_rotate_c3d(output_file, c3d_files, rotation_angles):
    """
    Combine up to 5 C3D files, resolve duplicate marker names, and apply separate 3D rotations to their markers.

    Parameters:
    output_file (str): Path to save the combined and rotated C3D file.
    c3d_files (list of str): List of up to 5 C3D file paths (None values will be ignored).
    rotation_angles (list of tuples): List of rotation angles (x_angle, y_angle, z_angle) for each file.

    Returns:
    None
    """
    combined_points = []
    combined_labels = []
    n_frames = None  # To track the minimum number of frames across all files

    for i, c3d_file in enumerate(c3d_files):
        if not c3d_file:  # Skip if the file is None
            continue

        # Load the C3D file
        c3d = ezc3d.c3d(c3d_file)

        # Extract marker data and labels
        points = c3d['data']['points']  # (4, nMarkers, nFrames)
        labels = c3d['parameters']['POINT']['LABELS']['value']

        # Update the minimum number of frames
        if n_frames is None:
            n_frames = points.shape[2]
        else:
            n_frames = min(n_frames, points.shape[2])

        # Apply rotation to the markers
        rot_matrix = rotation_matrix(rotation_angles[i])
        xyz_points = points[:3]  # Extract XYZ coordinates
        rotated_points = np.einsum('ij,jkf->ikf', rot_matrix, xyz_points)  # Apply rotation
        points[:3] = rotated_points

        # Validate and fix points
        if points.shape[0] == 3:  # If only X, Y, Z are present
            points = np.vstack([points, np.ones((1, points.shape[1], points.shape[2]))])
        points = points[:, :, :n_frames]  # Ensure frame consistency

        # Resolve duplicate labels
        labels = resolve_duplicate_labels(combined_labels, labels)

        # Append data
        combined_points.append(points)
        combined_labels.extend(labels)

    # Combine all points
    combined_points = np.concatenate(combined_points, axis=1)

    # Create a new C3D structure
    combined_c3d = ezc3d.c3d()

    # Add combined marker data
    combined_c3d['data']['points'] = combined_points

    # Update labels and units
    combined_c3d['parameters']['POINT']['LABELS']['value'] = combined_labels
    combined_c3d['parameters']['POINT']['UNITS']['value'] = ['mm']

    # Set frame rate and analog data from the first file
    first_c3d = ezc3d.c3d(c3d_files[0])
    combined_c3d['parameters']['POINT']['RATE']['value'] = first_c3d['parameters']['POINT']['RATE']['value']
    combined_c3d['data']['analogs'] = first_c3d['data']['analogs']

    # Write the combined and rotated C3D to a file
    combined_c3d.write(output_file)

    print(f"Combined and rotated C3D file created successfully: {output_file}")


# Chemins vers les fichiers C3D d'entrée
c3d_file1 = r"C:\ProgramData\Projets_Florian\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 2_traitement_new calib\Trial_1\terrain_basket_transformed.c3d"
c3d_file2 = r"C:\ProgramData\Projets_Florian\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 2_traitement_new calib\Trial_1\pose-3d\Trial_1_0-203_filt_butterworth.c3d"
c3d_file3 = r"C:\ProgramData\Projets_Florian\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 2_traitement_new calib\Trial_1\pose-3d-custom\Trial_1_0-203_filt_butterworth.c3d"
c3d_file4 = None
c3d_file5 = None

c3d_files = [c3d_file1, c3d_file2, c3d_file3, c3d_file4, c3d_file5]

# Chemin vers le fichier C3D combiné et tourné
output_file = r"C:\ProgramData\Projets_Florian\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 2_traitement_new calib\Trial_1\combined_results.c3d"

# Angles de rotation (en degrés) pour chaque fichier
rotation_angles = [
    (0, 0, 0),  # Rotation pour le premier fichier
    (90, 0, 90),   # Rotation pour le second fichier
    (90, 0, 90),   # Rotation pour le troisième fichier
    (90, 0, 90),
    (90, 0, 90)# Rotation pour le troisième fichier
]

# Combine les fichiers et applique des rotations spécifiques
combine_and_rotate_c3d(output_file, c3d_files, rotation_angles)

