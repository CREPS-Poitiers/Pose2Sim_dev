
# -*- coding: utf-8 -*-
"""
###########################################################################
# GENERATE 3D SPORT COURTS AND EXPORT TO C3D FILE                         #
###########################################################################

Created on Mon Nov 25 2024

@author: F.Delaplace

### Description:
This script generates 3D representations of various sports courts (e.g., basketball, tennis, handball) and saves them 
as `.c3d` files, which are compatible with motion analysis software such as OpenSim. It allows customization of the 
court's coordinate system and includes features to transform marker positions into a specified coordinate frame.

### Key Features:
1. **Court Creation**:
   - Supports predefined court types including:
     - Basketball (3x3 and 5x5 variants)
     - Tennis (singles and doubles lines)
     - Handball (with goal areas, penalty spots, and arcs)
   - Automatically generates marker positions and labels for key points on the court.

2. **Coordinate Transformation**:
   - Transforms marker positions to a user-defined coordinate system based on an origin and x-axis direction.

3. **Dynamic Frame Number Detection**:
   - Automatically determines the number of frames for the `.c3d` file based on files in the 3D pose directory.

4. **C3D File Export**:
   - Exports marker positions and frame data to a `.c3d` file with customizable parameters such as frame rate and units.

5. **Custom Court Configuration**:
   - Reads configuration data from a `Config.toml` file, allowing dynamic specification of court type, origin, and x-axis direction.

### Inputs:
- A `Config.toml` file with the following parameters:
  - `genResults.court.type_sport`: Type of court to generate (e.g., "basket3x3", "basket5x5", "tennis", "handball").
  - `genResults.court.origin`: Origin of the new coordinate system (list of 3 floats).
  - `genResults.court.x_axis`: Direction of the x-axis in the new coordinate system (list of 3 floats).

- Predefined marker positions and names for each court type.

### Outputs:
- A `.c3d` file containing:
  - Marker positions in 3D (in millimeters).
  - Frame-by-frame data for the specified number of frames.
  - Marker labels for easy identification.

### How to Use:
1. Update the `Config.toml` file with the desired court type, origin, and x-axis direction.
2. Run the script.
3. The generated `.c3d` file will be saved in the specified output directory.

### Applications:
- Biomechanical motion analysis.
- Sports science research.
- Simulation and visualization of sports court setups in 3D environments.
"""

import numpy as np
import ezc3d
import os
import re

def extract_number_from_folder(folder_path):
    """
    Extract a number from filenames in the folder that match a specific pattern.

    Parameters:
    folder_path (str): Path to the folder containing files.

    Returns:
    int: Extracted number from the matching filename.

    Raises:
    FileNotFoundError: If no matching file is found in the folder.
    """
    # Iterate through all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file matches the expected format
        if file_name.endswith("_filt_butterworth.c3d"):
            # Use a regular expression to extract the number
            match = re.search(r'-(\d+)_filt_butterworth\.c3d$', file_name)
            if match:
                return int(match.group(1))
    # Raise an error if no valid file is found
    raise FileNotFoundError("No matching file found in the folder.")

def transform_points(points, origin, x_direction):
    """
    Transform 3D points to a new coordinate system defined by an origin and an x-axis direction.

    Parameters:
    points (numpy.ndarray): Array of points (N, 3) to transform.
    origin (list): Origin of the new coordinate system (x, y, z).
    x_direction (list): A point defining the x-axis direction (x, y, z).

    Returns:
    numpy.ndarray: Transformed points in the new coordinate system.
    """
    # Convert the origin and x_direction to numpy arrays for calculations
    origin = np.array(origin, dtype=np.float64)
    x_direction = np.array(x_direction, dtype=np.float64)

    # Compute the unit vector for the x-axis
    x_axis = (x_direction - origin)
    x_axis /= np.linalg.norm(x_axis)

    # Define the z-axis as the vertical axis
    z_axis = np.array([0, 0, 1], dtype=np.float64)

    # Compute the y-axis using the cross product of z-axis and x-axis
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    # Create a transformation matrix from the x, y, and z axes
    transformation_matrix = np.vstack([x_axis, y_axis, z_axis]).T

    # Translate the points to the new origin
    translated_points = points - origin

    # Transform the points using the transformation matrix
    transformed_points = np.dot(translated_points, transformation_matrix)

    return transformed_points

def create_basket3x3():
    """
    Generate marker names and positions for a 3x3 basketball court.

    The function defines the positions of key markers on the court, including
    corners, hoop-related markers, and additional markers for the court's
    three-point arc and restricted areas.

    Returns:
    tuple: A list of marker names and a numpy array of marker positions (x, y, z).
    """
    # Define initial marker names for the court
    marker_names = [
        'Corner_Bottom_Left', 'Corner_Bottom_Right', 'Corner_Top_Left', 'Corner_Top_Right',
        'Backboard_Bottom_Left', 'Backboard_Bottom_Right', 'Backboard_Top_Left', 'Backboard_Top_Right',
        'Hoop_Center', 'Restricted_Area_Bottom_Left', 'Restricted_Area_Bottom_Right',
        'Restricted_Area_Top_Left', 'Restricted_Area_Top_Right',
        'Backboard_Rect_Bottom_Left', 'Backboard_Rect_Bottom_Right',
        'Backboard_Rect_Top_Left', 'Backboard_Rect_Top_Right'
    ]

    # Define initial marker positions in millimeters
    marker_positions = np.array([
        [0, 0, 0], [11000, 0, 0], [0, 15000, 0], [11000, 15000, 0],  # Court corners
        [1200, 6600, 2900], [1200, 8400, 2900], [1200, 6600, 3950], [1200, 8400, 3950],  # Backboard corners
        [1575, 7500, 3050],  # Hoop center
        [0, 5050, 0], [5800, 5050, 0], [0, 9950, 0], [5800, 9950, 0],  # Restricted area corners
        [1200, 7205, 3050], [1200, 7795, 3050], [1200, 7205, 3500], [1200, 7795, 3500]  # Backboard rectangle
    ])

    # Add points around the hoop
    hoop_center = [1575, 7500, 3050]  # Center of the hoop
    hoop_radius = 225  # Radius of the hoop in mm
    num_hoop_points = 8  # Number of points around the hoop

    # Generate evenly distributed points around the hoop
    for i in range(num_hoop_points):
        angle = 2 * np.pi * i / num_hoop_points
        x = hoop_center[0] + hoop_radius * np.cos(angle)
        y = hoop_center[1] + hoop_radius * np.sin(angle)
        z = hoop_center[2]
        marker_positions = np.vstack([marker_positions, [x, y, z]])
        marker_names.append(f'Hoop_Point_{i + 1}')

    # Add points for the non-charge semi-circle
    non_charge_radius = 1250  # Radius of the non-charge area in mm
    non_charge_center = [1575, 7500, 0]  # Center of the semi-circle
    num_non_charge_points = 10  # Number of points for the semi-circle

    # Generate points for the non-charge semi-circle
    for i in range(num_non_charge_points):
        angle = -np.pi / 2 + (np.pi / (num_non_charge_points - 1)) * i
        x = non_charge_center[0] + non_charge_radius * np.cos(angle)
        y = non_charge_center[1] + non_charge_radius * np.sin(angle)
        z = 0  # Ground level
        marker_positions = np.vstack([marker_positions, [x, y, z]])
        marker_names.append(f'Non_Charge_Point_{i + 1}')

    # Add points to connect the 3-point line to the baseline
    marker_positions = np.vstack([marker_positions, [0, 900, 0], [0, 14100, 0]])
    marker_names.extend(['Three_Point_Base_Left', 'Three_Point_Base_Right'])

    # Add points for the 3-point arc with intersections at the baseline
    three_points_radius = 6750  # Radius of the 3-point arc in mm
    three_points_center = [1575, 7500, 0]  # Center of the 3-point arc
    marker_positions, marker_names = add_three_point_arc_with_intersections(
        marker_positions, marker_names, three_points_radius, three_points_center, "Left"
    )

    return marker_names, marker_positions



def add_three_point_arc_with_intersections(marker_positions, marker_names, three_points_radius, three_points_center, side):
    """
    Add points for the 3-point arc, including intersections with the sidelines.

    Parameters:
    marker_positions (numpy.ndarray): Existing marker positions.
    marker_names (list): Existing marker names.
    three_points_radius (float): Radius of the 3-point arc.
    three_points_center (list): Center of the 3-point arc.
    side (str): 'Left' or 'Right'.

    Returns:
    tuple: Updated marker positions (numpy.ndarray) and marker names (list).
    """
    # Define sideline intersection points at ground level
    sideline_points = [[0, 900, 0], [0, 14100, 0]]  # Sideline y-coordinates
    intersection_points = []

    for sideline_point in sideline_points:
        # Calculate the y-offset from the center of the 3-point arc
        y_offset = sideline_point[1] - three_points_center[1]
        # Ensure the sideline intersects the arc
        if abs(y_offset) > three_points_radius:
            raise ValueError("Sideline point is outside the 3-point arc radius.")

        # Calculate the x-coordinate of the intersection point
        x = three_points_center[0] + np.sqrt(three_points_radius**2 - y_offset**2)
        intersection_points.append([x, sideline_point[1], 0])  # Append the 3D point

    # Add evenly spaced points along the arc between the intersections
    num_three_point_points = 20  # Number of points along the arc
    angle_start = np.arcsin((intersection_points[0][1] - three_points_center[1]) / three_points_radius)
    angle_end = np.arcsin((intersection_points[1][1] - three_points_center[1]) / three_points_radius)
    angles = np.linspace(angle_start, angle_end, num_three_point_points)

    for angle in angles:
        # Compute x, y coordinates for each point along the arc
        x = three_points_center[0] + three_points_radius * np.cos(angle)
        y = three_points_center[1] + three_points_radius * np.sin(angle)
        z = 0  # Ground level
        marker_positions = np.vstack([marker_positions, [x, y, z]])
        marker_names.append(f'{side}_Three_Point_Arc_{len(marker_names) + 1}')  # Append marker name

    # Add the intersection points to the marker list
    for i, intersection_point in enumerate(intersection_points):
        marker_positions = np.vstack([marker_positions, intersection_point])
        marker_names.append(f'{side}_Three_Point_Intersection_{i + 1}')  # Name the intersections

    return marker_positions, marker_names

def create_basket5x5():
    """
    Generate marker names and positions for a basketball court (5x5),
    based on the 3x3 version, with symmetry around the center line at 14 m.
    Returns marker names and positions.
    """
    marker_names = []  # List to store marker names
    marker_positions = np.empty((0, 3))  # Array to store marker positions

    # Define court dimensions
    court_length = 28000  # Total length of the court (28 m)
    court_width = 15000   # Total width of the court (15 m)
    half_court_length = court_length / 2  # Half-court line position

    # Generate markers for the left side using the 3x3 logic
    left_marker_names, left_marker_positions = create_basket3x3()

    # Adjust the positions for the left side markers
    for i, position in enumerate(left_marker_positions):
        # No offset needed for the left side markers
        left_marker_positions[i][0] += 0
        marker_names.append(f"Left_{left_marker_names[i]}")  # Prefix 'Left_' to the marker name
        marker_positions = np.vstack([marker_positions, left_marker_positions[i]])

    # Generate markers for the right side by mirroring the left side
    for i, position in enumerate(left_marker_positions):
        # Mirror the x-coordinate around the court center
        mirrored_position = [
            court_length - position[0],  # Flip x-axis
            position[1],                # Keep y-coordinate
            position[2]                 # Keep z-coordinate
        ]
        marker_names.append(f"Right_{left_marker_names[i]}")  # Prefix 'Right_' to the marker name
        marker_positions = np.vstack([marker_positions, mirrored_position])

    # Add markers for the center line
    center_line_left = [half_court_length, 0, 0]  # Left end of the center line
    center_line_right = [half_court_length, court_width, 0]  # Right end of the center line
    marker_positions = np.vstack([marker_positions, center_line_left, center_line_right])
    marker_names.extend(["Center_Line_Left", "Center_Line_Right"])

    # Add markers for the central circle
    center_circle_radius = 1800  # Radius of the central circle (1.8 m)
    center_circle_center = [half_court_length, court_width / 2, 0]  # Center of the circle
    num_circle_points = 20  # Number of points to define the circle

    for i in range(num_circle_points):
        # Calculate the x, y coordinates of the points on the circle
        angle = 2 * np.pi * i / num_circle_points
        x = center_circle_center[0] + center_circle_radius * np.cos(angle)
        y = center_circle_center[1] + center_circle_radius * np.sin(angle)
        z = 0  # Ground level
        marker_positions = np.vstack([marker_positions, [x, y, z]])
        marker_names.append(f'Center_Circle_Point_{i + 1}')  # Name the points sequentially

    return marker_names, marker_positions

def create_tennis():
    """
    Generate marker names and positions for a tennis court with full lines for singles and doubles,
    zones for services, and markers under the net at specific intersections.
    Includes correct net height (91.4 cm at the center, 107 cm at the sides).
    Returns marker names and positions.
    """
    marker_names = [
        # Court perimeter
        'Baseline_Left', 'Baseline_Right',  # Baselines at each end
        'Singles_Sideline_Left_Net', 'Singles_Sideline_Left_Back',  # Singles sidelines (left side)
        'Singles_Sideline_Right_Net', 'Singles_Sideline_Right_Back',  # Singles sidelines (right side)
        'Doubles_Sideline_Left_Net', 'Doubles_Sideline_Left_Back',  # Doubles sidelines (left side)
        'Doubles_Sideline_Right_Net', 'Doubles_Sideline_Right_Back',  # Doubles sidelines (right side)
        # Net markers
        'Net_Top_Left', 'Net_Top_Right', 'Net_Bottom_Left', 'Net_Bottom_Right', 'Net_Top_Center',
        # Markers below the net
        'Net_Under_Center', 'Net_Under_Left_Doubles', 'Net_Under_Right_Doubles',
        'Net_Under_Left_Singles', 'Net_Under_Right_Singles',
        # Service box markers
        'Service_Line_Left_Front', 'Service_Line_Left_Back',  # Left service box
        'Service_Line_Right_Front', 'Service_Line_Right_Back',  # Right service box
        'T_Center_Left', 'T_Center_Right'  # T-Center markers for service areas
    ]

    # Tennis court dimensions in millimeters
    court_length = 23770  # Total length of the court
    singles_width = 8230  # Width of the singles court
    doubles_width = 10970  # Width of the doubles court
    net_center_height = 914  # Net height at the center
    net_side_height = 1070  # Net height at the sides
    service_box_distance = 6400  # Distance from net to the service line
    t_width = 2440  # Width of the T line at the center

    # Define marker positions using court dimensions
    marker_positions = np.array([
        # Baseline markers
        [0, 0, 0], [court_length, 0, 0],  # Left and right baselines
        # Singles sideline markers
        [0, singles_width / 2, 0], [0, -singles_width / 2, 0],  # Left singles sideline
        [court_length, singles_width / 2, 0], [court_length, -singles_width / 2, 0],  # Right singles sideline
        # Doubles sideline markers
        [0, doubles_width / 2, 0], [0, -doubles_width / 2, 0],  # Left doubles sideline
        [court_length, doubles_width / 2, 0], [court_length, -doubles_width / 2, 0],  # Right doubles sideline
        # Net corners and center
        [court_length / 2, doubles_width / 2, net_side_height],  # Top left corner of net
        [court_length / 2, -doubles_width / 2, net_side_height],  # Top right corner of net
        [court_length / 2, doubles_width / 2, 0],  # Bottom left corner of net
        [court_length / 2, -doubles_width / 2, 0],  # Bottom right corner of net
        [court_length / 2, 0, net_center_height],  # Top center of net
        # Markers below the net
        [court_length / 2, 0, 0],  # Center below net
        [court_length / 2, doubles_width / 2, 0],  # Below left doubles sideline
        [court_length / 2, -doubles_width / 2, 0],  # Below right doubles sideline
        [court_length / 2, singles_width / 2, 0],  # Below left singles sideline
        [court_length / 2, -singles_width / 2, 0],  # Below right singles sideline
        # Service box markers (left side)
        [court_length / 2 - service_box_distance, singles_width / 2, 0],  # Left front service line
        [court_length / 2 - service_box_distance, -singles_width / 2, 0],  # Left back service line
        # Service box markers (right side)
        [court_length / 2 + service_box_distance, singles_width / 2, 0],  # Right front service line
        [court_length / 2 + service_box_distance, -singles_width / 2, 0],  # Right back service line
        # T-Center markers
        [court_length / 2 - service_box_distance, 0, 0],  # T Center (left side)
        [court_length / 2 + service_box_distance, 0, 0]  # T Center (right side)
    ])

    return marker_names, marker_positions

def create_handball():
    """
    Generate marker names and positions for a handball court, including:
    - Court perimeter
    - Goals (4 corners for each goal)
    - 6 m, 7 m (penalty), and 9 m lines
    - Quarter arcs around the goalposts
    Returns marker names and positions.
    """
    marker_names = [
        # Court perimeter
        'Corner_Bottom_Left', 'Corner_Bottom_Right',
        'Corner_Top_Left', 'Corner_Top_Right',
        # Left goal posts and corners
        'Left_Goal_Post_Left_Base', 'Left_Goal_Post_Right_Base',
        'Left_Goal_Post_Left_Top', 'Left_Goal_Post_Right_Top',
        # Right goal posts and corners
        'Right_Goal_Post_Left_Base', 'Right_Goal_Post_Right_Base',
        'Right_Goal_Post_Left_Top', 'Right_Goal_Post_Right_Top',
        # Center line
        'Center_Line_Left', 'Center_Line_Right',
        # 6 m lines
        'Left_6m_Line_Left', 'Left_6m_Line_Right',
        'Right_6m_Line_Left', 'Right_6m_Line_Right',
        # 7 m penalty lines
        'Left_7m_Line_Left', 'Left_7m_Line_Right',
        'Right_7m_Line_Left', 'Right_7m_Line_Right',
        # 9 m lines
        'Left_9m_Line_Left', 'Left_9m_Line_Right',
        'Right_9m_Line_Left', 'Right_9m_Line_Right',
        # Quarter arcs (6 m and 9 m)
        *[f'Left_Goal_6m_Arc_Left_{i}' for i in range(1, 11)],
        *[f'Left_Goal_6m_Arc_Right_{i}' for i in range(1, 11)],
        *[f'Left_Goal_9m_Arc_Left_{i}' for i in range(1, 11)],
        *[f'Left_Goal_9m_Arc_Right_{i}' for i in range(1, 11)],
        *[f'Right_Goal_6m_Arc_Left_{i}' for i in range(1, 11)],
        *[f'Right_Goal_6m_Arc_Right_{i}' for i in range(1, 11)],
        *[f'Right_Goal_9m_Arc_Left_{i}' for i in range(1, 11)],
        *[f'Right_Goal_9m_Arc_Right_{i}' for i in range(1, 11)],
    ]

    # Handball court dimensions (in mm)
    court_length = 40000  # Total court length (40 m)
    court_width = 20000   # Total court width (20 m)
    goal_width = 3000     # Goal width (3 m)
    goal_height = 2000    # Goal height (2 m)
    six_meter_distance = 6000  # 6 m line distance
    seven_meter_distance = 7000  # 7 m (penalty) line distance
    nine_meter_distance = 9000  # 9 m line distance
    penalty_line_width = 1000  # Penalty line width
    arc_points = 10  # Number of points for quarter arcs

    # Initialize marker positions
    marker_positions = []

    # Define court perimeter
    marker_positions.extend([
        [0, 0, 0], [court_length, 0, 0],  # Bottom corners
        [0, court_width, 0], [court_length, court_width, 0],  # Top corners
    ])

    # Define left goal (4 corners)
    left_goal_center_x = 0
    left_goal_center_y = court_width / 2
    marker_positions.extend([
        [left_goal_center_x, left_goal_center_y - goal_width / 2, 0],  # Left post base
        [left_goal_center_x, left_goal_center_y + goal_width / 2, 0],  # Right post base
        [left_goal_center_x, left_goal_center_y - goal_width / 2, goal_height],  # Left post top
        [left_goal_center_x, left_goal_center_y + goal_width / 2, goal_height],  # Right post top
    ])

    # Define right goal (4 corners)
    right_goal_center_x = court_length
    right_goal_center_y = court_width / 2
    marker_positions.extend([
        [right_goal_center_x, right_goal_center_y - goal_width / 2, 0],  # Left post base
        [right_goal_center_x, right_goal_center_y + goal_width / 2, 0],  # Right post base
        [right_goal_center_x, right_goal_center_y - goal_width / 2, goal_height],  # Left post top
        [right_goal_center_x, right_goal_center_y + goal_width / 2, goal_height],  # Right post top
    ])

    # Add center line
    marker_positions.extend([
        [court_length / 2, 0, 0], [court_length / 2, court_width, 0],  # Center line
    ])

    # Add 6 m lines
    marker_positions.extend([
        [left_goal_center_x + six_meter_distance, left_goal_center_y - goal_width / 2, 0],
        [left_goal_center_x + six_meter_distance, left_goal_center_y + goal_width / 2, 0],
        [right_goal_center_x - six_meter_distance, right_goal_center_y - goal_width / 2, 0],
        [right_goal_center_x - six_meter_distance, right_goal_center_y + goal_width / 2, 0],
    ])

    # Add 7 m penalty lines
    marker_positions.extend([
        [left_goal_center_x + seven_meter_distance, left_goal_center_y - penalty_line_width / 2, 0],
        [left_goal_center_x + seven_meter_distance, left_goal_center_y + penalty_line_width / 2, 0],
        [right_goal_center_x - seven_meter_distance, right_goal_center_y - penalty_line_width / 2, 0],
        [right_goal_center_x - seven_meter_distance, right_goal_center_y + penalty_line_width / 2, 0],
    ])

    # Add 9 m lines
    marker_positions.extend([
        [left_goal_center_x + nine_meter_distance, left_goal_center_y - goal_width / 2, 0],
        [left_goal_center_x + nine_meter_distance, left_goal_center_y + goal_width / 2, 0],
        [right_goal_center_x - nine_meter_distance, right_goal_center_y - goal_width / 2, 0],
        [right_goal_center_x - nine_meter_distance, right_goal_center_y + goal_width / 2, 0],
    ])

    # Add quarter arcs for 6 m and 9 m
    for radius, prefix in [(six_meter_distance, "6m"), (nine_meter_distance, "9m")]:
        for i in range(arc_points):
            angle = np.pi / 2 * i / (arc_points - 1)
            # Left goal arcs
            x = left_goal_center_x + radius * np.cos(angle)
            y = left_goal_center_y - goal_width / 2 - radius * np.sin(angle)
            if y < 0:  # Stop at sideline
                break
            marker_positions.append([x, y, 0])
            x = left_goal_center_x + radius * np.cos(angle)
            y = left_goal_center_y + goal_width / 2 + radius * np.sin(angle)
            if y > court_width:  # Stop at sideline
                break
            marker_positions.append([x, y, 0])

            # Right goal arcs
            x = right_goal_center_x - radius * np.cos(angle)
            y = right_goal_center_y - goal_width / 2 - radius * np.sin(angle)
            if y < 0:  # Stop at sideline
                break
            marker_positions.append([x, y, 0])
            x = right_goal_center_x - radius * np.cos(angle)
            y = right_goal_center_y + goal_width / 2 + radius * np.sin(angle)
            if y > court_width:  # Stop at sideline
                break
            marker_positions.append([x, y, 0])

    return marker_names, np.array(marker_positions)

def create_court(config_dict):
    """
    Create a court of a specified type and save it as a C3D file.
    
    Parameters:
    config_dict (dict): Configuration dictionary containing the project directory, court type, 
                        coordinate system details, and triangulation settings.
    
    Outputs:
    - A C3D file representing the court, stored in the 'results-3d' directory.
    """
    # Extract the project directory from the configuration
    project_dir = config_dict.get('project').get('project_dir')
    
    # Determine the session directory: batch or single trial
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()
    
    # Set up directories for output files
    results_dir = os.path.join(project_dir, 'results-3d')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"The directory {results_dir} has been created.")
    
    # Path to save the resulting C3D file
    output_c3d_path = os.path.join(results_dir, 'sport_court.c3d')
    
    # Extract court specifications from the configuration
    court_type = config_dict.get('genResults').get('court').get('type_sport')
    print("Court type:", court_type)
    origin = config_dict.get('genResults').get('court').get('origin')
    x_axis_direction = config_dict.get('genResults').get('court').get('x_axis')

    # Determine the number of frames based on existing 3D pose files
    try:
        num_frames = extract_number_from_folder(os.path.join(project_dir, 'pose-3d'))
    except FileNotFoundError as e:
        print(e)
        return

    # Generate court markers based on the specified type
    if court_type == "basket3x3":
        marker_names, marker_positions = create_basket3x3()
    elif court_type == "basket5x5":
        marker_names, marker_positions = create_basket5x5()
    elif court_type == "tennis":
        marker_names, marker_positions = create_tennis()
    elif court_type == "handball":
        marker_names, marker_positions = create_handball()
    else:
        raise ValueError(f"Unsupported court type: {court_type}")

    # Transform marker positions to the new coordinate system
    transformed_positions = transform_points(marker_positions, origin, x_axis_direction)

    # Initialize the C3D structure
    c3d = ezc3d.c3d()
    c3d['parameters']['POINT']['UNITS']['value'] = ['mm']
    c3d['parameters']['POINT']['RATE']['value'] = [100]  # Frame rate
    c3d['parameters']['POINT']['LABELS']['value'] = marker_names

    # Populate the C3D data with marker positions
    num_markers = len(marker_names)
    points = np.zeros((4, num_markers, num_frames))  # (X, Y, Z, residual) for each marker
    for marker_index, position in enumerate(transformed_positions):
        for frame_index in range(num_frames):
            points[0:3, marker_index, frame_index] = position

    c3d['data']['points'] = points  # Assign the points data to the C3D structure

    # Write the C3D file to the results directory
    c3d.write(output_c3d_path)
    print(f'C3D file successfully created: {output_c3d_path}')

# # Example usage
# create_court(
#     court_type="basket5x5",
#     origin=[0, 0, 0],
#     x_direction=[1, 0, 0],
#     num_frames=1,
#     output_path=r"C:\ProgramData\Projets_Florian\Videos_Markerless\Manip_3x3_ext_24-10-2024\sport_court.c3d"
# )
