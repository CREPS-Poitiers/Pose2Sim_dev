# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:31:00 2024

@author: fdelap01
"""

import numpy as np
import ezc3d

def transform_points(points, origin, x_direction):
    """
    Transform points to a new coordinate system defined by an origin and an x-axis direction.

    Parameters:
    points (numpy.ndarray): Array of points (N, 3) to transform.
    origin (list): Origin of the new coordinate system (1, 3).
    x_direction (list): A point defining the x-axis direction (1, 3).

    Returns:
    numpy.ndarray: Transformed points.
    """
    origin = np.array(origin, dtype=np.float64)
    x_direction = np.array(x_direction, dtype=np.float64)

    # Compute the new x-axis
    x_axis = (x_direction - origin)
    x_axis /= np.linalg.norm(x_axis)

    # Compute the new z-axis (vertical)
    z_axis = np.array([0, 0, 1], dtype=np.float64)

    # Compute the new y-axis as perpendicular to x and z
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Transformation matrix
    transformation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # Translate points to the new origin
    translated_points = points - origin

    # Apply rotation
    transformed_points = np.dot(translated_points, transformation_matrix)

    return transformed_points


# Load an empty c3d structure
c3d = ezc3d.c3d()

# Define marker names
marker_names = [
    # Périmètre du terrain
    'Corner_Bas_Gauche', 'Corner_Bas_Droit', 'Corner_Haut_Gauche', 'Corner_Haut_Droit',
    # Panneau et arceau
    'Planche_Bas_Gauche', 'Planche_Bas_Droit', 'Planche_Haut_Gauche', 'Planche_Haut_Droit',
    'Panier_Centre', 
    # Zone intérieure
    'Zone_Bas_Gauche', 'Zone_Bas_Droit', 'Zone_Haut_Gauche', 'Zone_Haut_Droit',
    # Rectangle sur la planche
    'Planche_Rectangle_Bas_Gauche', 'Planche_Rectangle_Bas_Droit',
    'Planche_Rectangle_Haut_Gauche', 'Planche_Rectangle_Haut_Droit'
]

# Define fixed marker positions
marker_positions = np.array([
    # Périmètre du terrain
    [0, 0, 0], [11000, 0, 0], [0, 15000, 0], [11000, 15000, 0],
    # Panneau et arceau
    [1200, 6600, 2900], [1200, 8400, 2900], [1200, 6600, 3950], [1200, 8400, 3950],
    [1575, 7500, 3050], 
    # Zone intérieure
    [0, 5050, 0], [5800, 5050, 0], [0, 9950, 0], [5800, 9950, 0],
    # Rectangle sur la planche
    [1200, 7205, 3050], [1200, 7795, 3050], [1200, 7205, 3500], [1200, 7795, 3500]
])

# Add points around the hoop
hoop_center = [1575, 7500, 3050]
hoop_radius = 225  # mm
num_hoop_points = 8

for i in range(num_hoop_points):
    angle = 2 * np.pi * i / num_hoop_points
    x = hoop_center[0] + hoop_radius * np.cos(angle)
    y = hoop_center[1] + hoop_radius * np.sin(angle)
    z = hoop_center[2]
    marker_positions = np.vstack([marker_positions, [x, y, z]])
    marker_names.append(f'Arceau_Point_{i + 1}')

# Add points for the non-charge semi-circle
non_charge_radius = 1250  # Radius of non-charge semi-circle
non_charge_center = [1575, 7500, 0]
num_non_charge_points = 10

for i in range(num_non_charge_points):
    angle = -np.pi / 2 + (np.pi / (num_non_charge_points - 1)) * i
    x = non_charge_center[0] + non_charge_radius * np.cos(angle)
    y = non_charge_center[1] + non_charge_radius * np.sin(angle)
    z = 0  # Ground level
    marker_positions = np.vstack([marker_positions, [x, y, z]])
    marker_names.append(f'Non_Charge_Point_{i + 1}')

# Add points for the 3-point arc
three_points_radius = 6750  # Radius of 3-point arc
three_points_center = [1575, 7500, 0]
num_three_point_points = 20

for i in range(num_three_point_points):
    angle = -np.pi / 2 + (np.pi / (num_three_point_points - 1)) * i
    x = three_points_center[0] + three_points_radius * np.cos(angle)
    y = three_points_center[1] + three_points_radius * np.sin(angle)
    z = 0  # Ground level
    marker_positions = np.vstack([marker_positions, [x, y, z]])
    marker_names.append(f'Three_Point_Arc_{i + 1}')

# Add points to connect the 3-point line to the baseline
marker_positions = np.vstack([marker_positions, [0, 900, 0], [0, 14100, 0]])
marker_names.extend(['Three_Point_Base_Left', 'Three_Point_Base_Right'])

# Apply transformation
# origin = [0, 0, 0]  # Define the new origin
# x_direction = [11000, 0, 0]  # Define the x-axis direction
origin = [0, 0, 0]  # Define the new origin
x_direction = [11000, 0, 0]  # Define the x-axis direction
transformed_positions = transform_points(marker_positions, origin, x_direction)

# Fill the parameters for POINT
c3d['parameters']['POINT']['UNITS']['value'] = ['mm']
c3d['parameters']['POINT']['RATE']['value'] = [100]  # Frame rate for points
c3d['parameters']['POINT']['LABELS']['value'] = marker_names

# Prepare marker data
num_frames_points = 719  # nombre de frames for the points
num_markers = len(marker_names)
points = np.zeros((4, num_markers, num_frames_points))  # (X, Y, Z, Validity)
for i, pos in enumerate(transformed_positions):
    for frame in range(num_frames_points):
        points[0:3, i, frame] = pos  # X, Y, Z coordinates

c3d['data']['points'] = points

# Write the data
output_path = r"C:\ProgramData\Projets_Florian\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 1_traitement_new calib\Trial_6\terrain_basket_transformed.c3d"
c3d.write(output_path)

print(f'Fichier C3D créé avec succès : {output_path}')

