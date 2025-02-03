# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 08:12:33 2024

@author: F.Delaplace

###########################################################################
## COMBINING AND ROTATING MULTIPLE C3D FILES for Pose2Sim final results  ##
###########################################################################

This script provides functionality for combining multiple C3D files, resolving 
duplicate marker labels, and applying separate 3D rotations to each file's marker 
coordinates. It is designed for motion capture data processing and enables the 
generation of a single, coherent C3D file from multiple sources with differing 
orientations.

Key Features:
1. **Rotation Matrix Generation:** Constructs a combined rotation matrix for 
   rotations around x, y, and z axes.
2. **Duplicate Marker Label Resolution:** Ensures unique marker labels by 
   appending suffixes (_2, _3, etc.) to duplicate labels.
3. **Incremental File Identification:** Automatically identifies and processes 
   files with specific naming patterns (e.g., P1_filt_butterworth.c3d).
4. **Customizable Rotations:** Applies user-defined rotation angles to each 
   C3D file's marker coordinates.
5. **Combination of Marker Data:** Merges marker data from multiple C3D files 
   into a single output, ensuring consistency in frame count and data structure.
6. **Flexible Configuration:** Supports both single-trial and batch processing, 
   with parameters provided via a configuration dictionary (config_dict).
7. **Final Output:** Generates a combined and rotated C3D file in the 
   `results-3d` directory for further analysis.

Expected Input:
- C3D files containing motion capture data.
- A configuration dictionary specifying file paths, rotation angles, and 
  project details.

Output:
- A single combined and rotated C3D file (`combined_results.c3d`) stored in the 
  `results-3d` directory.

This script is modular and can be adapted for various motion capture workflows 
that require combining and reorienting multiple datasets.
"""

import ezc3d
import numpy as np
import os
import re


def rotation_matrix(angles):
    """
    Create a combined rotation matrix from rotation angles around x, y, and z axes.

    Parameters:
    angles (tuple or list): Rotation angles in degrees for x, y, and z axes.

    Returns:
    numpy.ndarray: A 3x3 rotation matrix.
    """
    # Convert angles from degrees to radians
    x_angle, y_angle, z_angle = np.radians(angles)

    # Rotation matrix around the X-axis
    rot_x = np.array([
        [1, 0, 0],
        [0, np.cos(x_angle), -np.sin(x_angle)],
        [0, np.sin(x_angle), np.cos(x_angle)]
    ])

    # Rotation matrix around the Y-axis
    rot_y = np.array([
        [np.cos(y_angle), 0, np.sin(y_angle)],
        [0, 1, 0],
        [-np.sin(y_angle), 0, np.cos(y_angle)]
    ])

    # Rotation matrix around the Z-axis
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

    Parameters:
    existing_labels (list): List of existing marker labels.
    new_labels (list): List of new marker labels to check for duplicates.

    Returns:
    list: Updated list of marker labels with resolved duplicates.
    """
    updated_labels = []
    label_counts = {label: 1 for label in existing_labels}  # Track occurrences of existing labels

    for label in new_labels:
        if label in label_counts:  # If label is a duplicate
            label_counts[label] += 1
            updated_labels.append(f"{label}_{label_counts[label]}")
        else:  # If label is unique
            label_counts[label] = 1
            updated_labels.append(label)

    return updated_labels

def get_incremental_c3d_files(folder_path, max_participants):
    """
    Retrieve C3D files with a flexible naming pattern up to a maximum number of participants.

    Parameters:
    folder_path (str): Path to the folder containing the C3D files.
    max_participants (int): Maximum number of participants to consider.

    Returns:
    list: A list of paths to C3D files sorted by participant ID.
    """
    file_pattern = re.compile(r"Trial_\d+_P(\d+)_.*_filt_butterworth\.c3d")  # Match P1, P2, etc.
    file_list = os.listdir(folder_path)

    # Filter files matching the pattern and within max_participants
    matching_files = []
    for file_name in file_list:
        match = file_pattern.match(file_name)
        if match:
            participant_id = int(match.group(1))
            if participant_id <= max_participants:
                matching_files.append(os.path.join(folder_path, file_name))

    # Sort files by participant ID
    matching_files.sort(key=lambda x: int(file_pattern.match(os.path.basename(x)).group(1)))

    return matching_files



def extract_from_folder(folder_path):
    """
    Extract the first C3D file matching the expected naming pattern from the folder.

    Parameters:
    folder_path (str): Path to the folder containing the C3D files.

    Returns:
    str: Path to the first matching C3D file.

    Raises:
    FileNotFoundError: If no matching file is found.
    """
    # Iterate over files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file matches the expected pattern
        if file_name.endswith("_filt_butterworth.c3d"):
            c3d_file = os.path.join(folder_path, file_name)
            return c3d_file

    # Raise an error if no matching file is found
    raise FileNotFoundError("No matching C3D file found in the folder.")



def combine_and_rotate_c3d(config_dict):
    """
    Combine up to 5 C3D files, resolve duplicate marker names, and apply separate 3D rotations to their markers.

    Parameters:
    config_dict (dict): Configuration dictionary containing project and rotation settings.

    Returns:
    None: Saves the combined and rotated C3D file to disk.
    """
    # Extract project directories from the config dictionary
    project_dir = config_dict.get('project').get('project_dir')
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))  # Default session directory
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()  # Adjust for single trial

    # Define subdirectories
    multi_person = config_dict.get('project').get('multi_person')
    pose3d_custom_dir = os.path.join(project_dir, 'pose-3d-custom')
    pose3d_dir = os.path.join(project_dir, 'pose-3d')
    results_dir = os.path.join(project_dir, 'results-3d')

    # Retrieve rotation angles for each component
    rotation_person = config_dict.get('genResults').get('combine').get('rotation_person')
    rotation_court = config_dict.get('genResults').get('combine').get('rotation_court')
    rotation_custom = config_dict.get('genResults').get('combine').get('rotation_custom')

    # Debug prints for extracted values
    print("rotation_person:", rotation_person)
    print("rotation_court:", rotation_court)
    print("rotation_custom:", rotation_custom)

    # Define output file path
    output_file = os.path.join(results_dir, 'combined_results.c3d')
        
    if multi_person:
        # Nombre maximum de participants
        max_participants = config_dict.get('genResults').get('combine').get('max_participants', 2)
    
        # Récupération des fichiers participants
        c3d_files = get_incremental_c3d_files(pose3d_dir, max_participants)
    
        # Ajout du fichier "court" si présent
        court_c3d_path = os.path.join(results_dir, 'sport_court.c3d')
        if os.path.exists(court_c3d_path):
            c3d_files.append(court_c3d_path)
    
        # Ajout du fichier "custom person" si présent
        if os.path.isdir(pose3d_custom_dir):
            custom_person_c3d_path = extract_from_folder(pose3d_custom_dir)
            c3d_files.append(custom_person_c3d_path)
    else:
        # Mode single-person
        if os.path.isdir(pose3d_custom_dir):
            c3d_files = [
                extract_from_folder(pose3d_dir),
                os.path.join(results_dir, 'sport_court.c3d'),
                extract_from_folder(pose3d_custom_dir)
            ]
        else:
            c3d_files = [
                extract_from_folder(pose3d_dir),
                os.path.join(results_dir, 'sport_court.c3d')
            ]
    
    # Initialisation des angles de rotation pour chaque fichier
    rotation_angles = [tuple(rotation_person)] * (len(c3d_files) - 2)  # Participants
    if os.path.exists(court_c3d_path):
        rotation_angles.append(tuple(rotation_court))
    if os.path.isdir(pose3d_custom_dir):
        rotation_angles.append(tuple(rotation_custom))
    
    combined_points = []
    combined_labels = []
    n_frames = None  # Minimum number of frames parmi tous les fichiers


    for i, c3d_file in enumerate(c3d_files):
        if not c3d_file:  # Skip if no file
            continue

        # Load the C3D file
        c3d = ezc3d.c3d(c3d_file)

        # Extract marker data and labels
        points = c3d['data']['points']  # (4, nMarkers, nFrames)
        labels = c3d['parameters']['POINT']['LABELS']['value']

        # Update the minimum number of frames
        n_frames = min(n_frames, points.shape[2]) if n_frames else points.shape[2]

        # Apply rotation
        rot_matrix = rotation_matrix(rotation_angles[i])
        xyz_points = points[:3]  # Extract XYZ coordinates
        rotated_points = np.einsum('ij,jkf->ikf', rot_matrix, xyz_points)  # Rotate points
        points[:3] = rotated_points

        # Validate and pad points to ensure frame consistency
        if points.shape[0] == 3:  # Add a fourth dimension for homogeneity
            points = np.vstack([points, np.ones((1, points.shape[1], points.shape[2]))])
        points = points[:, :, :n_frames]

        # Resolve duplicate labels
        labels = resolve_duplicate_labels(combined_labels, labels)

        # Store points and labels
        combined_points.append(points)
        combined_labels.extend(labels)

    # Combine points from all files
    combined_points = np.concatenate(combined_points, axis=1)

    # Create a new C3D structure for the combined data
    combined_c3d = ezc3d.c3d()

    # Add marker data
    combined_c3d['data']['points'] = combined_points

    # Update labels and units
    combined_c3d['parameters']['POINT']['LABELS']['value'] = combined_labels
    combined_c3d['parameters']['POINT']['UNITS']['value'] = ['mm']

    # Set frame rate and analog data based on the first file
    first_c3d = ezc3d.c3d(c3d_files[0])
    combined_c3d['parameters']['POINT']['RATE']['value'] = first_c3d['parameters']['POINT']['RATE']['value']
    combined_c3d['data']['analogs'] = first_c3d['data']['analogs']

    # Save the combined and rotated C3D file
    combined_c3d.write(output_file)
    print(f"Combined and rotated C3D file created successfully: {output_file}")
