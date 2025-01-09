#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
###########################################################################
##                      CAMERA CALIBRATION                               ##
###########################################################################

This script provides tools to calibrate cameras for motion capture, 
biomechanical analysis, or computer vision applications. It supports a wide 
range of features, including calibration file conversion and intrinsic and 
extrinsic camera calibration calculations.

### MAIN FEATURES:
1. **Calibration File Conversion**:
   - Supported formats: Qualisys (.qca.txt), Vicon (.xcp), OpenCap (.pickle), 
     EasyMocap (.yml), and Biocv (.calib).
   - Converts data into a universal `.toml` format.

2. **Intrinsic Calibration**:
   - Two methods available: 
     - **Checkerboard** (classic method).
     - **Charuco board** (new functionality, much more precise and accurate).
   - Automatically detects corners/markers and interpolates Charuco corners.
   - Extracts frames from videos automatically if needed.

3. **Extrinsic Calibration**:
   - Two methods available:
     - **Board**: Uses a physical calibration board, which should be large 
       enough to be detected when laid on the floor. Not recommended due to 
       accuracy limitations.
     - **Scene**: Involves manually clicking any point of known coordinates 
       in the scene. Usually more accurate if points are spread out across the volume.
   - Determines spatial relationships between cameras using boards or 
     pre-measured scene points.
   - Allows for manual point selection if automatic detection fails.

### INPUTS:
- **For file conversion**: Calibration files in supported formats.
- **For intrinsic calibration**: Checkerboard/Charuco images or videos 
  (approximately 30 images recommended).
- **For extrinsic calibration**: Images or videos showing a known board 
  or 3D scene points.

### OUTPUTS:
- A calibration file in `.toml` format containing:
  - Intrinsic parameters (camera matrix and distortion coefficients).
  - Extrinsic parameters (rotation and translation vectors).
- Diagnostic information, such as residual calibration errors in pixels or millimeters.

### DEVELOPMENTS:
- Additional features and improvements have been developed by F. Delaplace:
  - Integration of the **Charuco board** method for intrinsic calibration.
  - Enhanced extrinsic calibration options for improved accuracy.

### DEPENDENCIES:
- OpenCV
- NumPy
- Pandas
- Matplotlib
- tqdm
- lxml
- PIL

### NOTES:
- Intrinsic calibration requires at least 10 quality images for accurate results.
- The script supports manual interaction for corner detection if automated methods fail.
- Extrinsic calibration supports visualizing reprojection errors for verification.

### AUTHORSHIP INFORMATION:
- Original Author: David Pagnon
- Additional Developments: F. Delaplace
- Version: 0.9.4
- License: BSD 3-Clause License
"""


# TODO: DETECT WHEN WINDOW IS CLOSED
# TODO: WHEN 'Y', CATCH IF NUMBER OF IMAGE POINTS CLICKED NOT EQUAL TO NB OBJ POINTS


## INIT
from Pose2Sim.common import world_to_camera_persp, rotate_cam, quat2mat, euclidean_distance, natural_sort_key, zup2yup

import os
import logging
import pickle
import numpy as np
import pandas as pd
os.environ["OPENCV_LOG_LEVEL"]="FATAL"
import cv2
import glob
import toml
import re
from lxml import etree
import warnings
import matplotlib.pyplot as plt
from mpl_interactions import zoom_factory, panhandler
from PIL import Image
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
from tqdm import tqdm

# Mapping of Aruco marker dictionaries to OpenCV constants.
# This allows for easy selection of Aruco marker types for detection and calibration tasks.
ARUCO_DICT_MAPPING = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6": cv2.aruco.DICT_6X6_250,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL
}

def calib_qca_fun(file_to_convert_path, binning_factor=1):
    """
    Converts a Qualisys `.qca.txt` calibration file to a `.toml` format.
    - Converts camera view to object view using rotation transformations.
    - Applies the Rodrigues formula for rotation matrices.
    
    Parameters:
        file_to_convert_path (str): Path of the `.qca.txt` file.
        binning_factor (int): Scaling factor for pixel binning (default=1).

    Returns:
        ret (list[float]): Residual reprojection error in millimeters.
        C (list[str]): Camera names.
        S (list[list[float]]): Image sizes as [width, height].
        D (list[np.array]): Distortion coefficients.
        K (list[np.array]): Intrinsic parameters as 3x3 matrices.
        R (list[np.array]): Extrinsic rotation matrices.
        T (list[np.array]): Extrinsic translation vectors.
    """
    logging.info(f'Converting {file_to_convert_path} to .toml calibration file...')
    
    # Read the Qualisys .qca file and extract relevant calibration parameters.
    ret, C, S, D, K, R, T = read_qca(file_to_convert_path, binning_factor)
    
    # Convert extrinsic parameters to align with the world coordinate system.
    # Transforms rotation and translation matrices from camera to world view.
    RT = [world_to_camera_persp(r, t) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    # Apply a rotation of 180° about the x-axis.
    # This aligns cameras correctly in the 3D coordinate system.
    RT = [rotate_cam(r, t, ang_x=np.pi, ang_y=0, ang_z=0) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]
    T = [rt[1] for rt in RT]

    # Convert rotation matrices to Rodrigues vectors for compact representation.
    R = [np.array(cv2.Rodrigues(r)[0]).flatten() for r in R]
    T = np.array(T)  # Convert translation vectors to numpy arrays.
      
    return ret, C, S, D, K, R, T


def read_qca(qca_path, binning_factor):
    """
    Reads a Qualisys `.qca.txt` calibration file and extracts calibration data.
    This function parses the XML file structure to extract intrinsic and extrinsic
    camera parameters.

    Parameters:
        qca_path (str): Path to the `.qca.txt` file.
        binning_factor (int): Scaling factor for pixel binning (default=1).

    Returns:
        ret (list[float]): Residual reprojection errors in millimeters.
        C (list[str]): Camera names.
        S (list[list[float]]): Image sizes as [width, height].
        D (list[np.array]): Distortion coefficients.
        K (list[np.array]): Intrinsic matrices as 3x3 numpy arrays.
        R (list[np.array]): Extrinsic rotation matrices.
        T (list[np.array]): Extrinsic translation vectors.
    """
    # Parse the XML structure of the .qca file.
    root = etree.parse(qca_path).getroot()
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    res = []  # Stores video resolution for each camera.
    vid_id = []  # Identifies video cameras.

    # Extract basic camera information, including residual errors and serial numbers.
    for i, tag in enumerate(root.findall('cameras/camera')):
        ret.append(float(tag.attrib.get('avg-residual')))
        C.append(tag.attrib.get('serial'))
        res.append(int(tag.attrib.get('video_resolution')[:-1]) if tag.attrib.get('video_resolution') else 1080)
        if tag.attrib.get('model') in ('Miqus Video', 'Miqus Video UnderWater', 'none'):
            vid_id.append(i)
    
    # Compute image sizes based on the field of view and binning factor.
    for i, tag in enumerate(root.findall('cameras/camera/fov_video')):
        w = (float(tag.attrib.get('right')) - float(tag.attrib.get('left')) + 1) / binning_factor / (1080 / res[i])
        h = (float(tag.attrib.get('bottom')) - float(tag.attrib.get('top')) + 1) / binning_factor / (1080 / res[i])
        S.append([w, h])
    
    # Extract intrinsic parameters, including distortion coefficients and intrinsic matrices.
    for i, tag in enumerate(root.findall('cameras/camera/intrinsic')):
        k1 = float(tag.get('radialDistortion1')) / 64 / binning_factor
        k2 = float(tag.get('radialDistortion2')) / 64 / binning_factor
        p1 = float(tag.get('tangentalDistortion1')) / 64 / binning_factor
        p2 = float(tag.get('tangentalDistortion2')) / 64 / binning_factor
        D.append(np.array([k1, k2, p1, p2]))
        
        # Calculate focal lengths and principal points, considering binning and resolution.
        fu = float(tag.get('focalLengthU')) / 64 / binning_factor / (1080 / res[i])
        fv = float(tag.get('focalLengthV')) / 64 / binning_factor / (1080 / res[i])
        cu = (float(tag.get('centerPointU')) / 64 / binning_factor - float(root.findall('cameras/camera/fov_video')[i].attrib.get('left'))) / (1080 / res[i])
        cv = (float(tag.get('centerPointV')) / 64 / binning_factor - float(root.findall('cameras/camera/fov_video')[i].attrib.get('top'))) / (1080 / res[i])
        K.append(np.array([fu, 0., cu, 0., fv, cv, 0., 0., 1.]).reshape(3, 3))
    
    # Extract extrinsic parameters: rotation matrices and translation vectors.
    for tag in root.findall('cameras/camera/transform'):
        tx = float(tag.get('x')) / 1000
        ty = float(tag.get('y')) / 1000
        tz = float(tag.get('z')) / 1000
        r11 = float(tag.get('r11'))
        r12 = float(tag.get('r12'))
        r13 = float(tag.get('r13'))
        r21 = float(tag.get('r21'))
        r22 = float(tag.get('r22'))
        r23 = float(tag.get('r23'))
        r31 = float(tag.get('r31'))
        r32 = float(tag.get('r32'))
        r33 = float(tag.get('r33'))

        # Transpose rotation matrix for consistency.
        R.append(np.array([r11, r12, r13, r21, r22, r23, r31, r32, r33]).reshape(3, 3).T)
        T.append(np.array([tx, ty, tz]))
    
    # Sort cameras by natural order based on serial numbers.
    C_vid = [C[v] for v in vid_id]
    C_vid_id = [C_vid.index(c) for c in sorted(C_vid, key=natural_sort_key)]
    C_id = [vid_id[c] for c in C_vid_id]
    C = [C[c] for c in C_id]
    ret = [ret[c] for c in C_id]
    S = [S[c] for c in C_id]
    D = [D[c] for c in C_id]
    K = [K[c] for c in C_id]
    R = [R[c] for c in C_id]
    T = [T[c] for c in C_id]
    
    return ret, C, S, D, K, R, T


def calib_optitrack_fun(config_dict, binning_factor=1):
    """
    Placeholder function for handling OptiTrack calibration file conversion.
    Currently raises an exception as the implementation details are
    unavailable and must be referenced from external documentation.

    Parameters:
        config_dict (dict): Configuration dictionary, typically read from a Config.toml file.
        binning_factor (int): Binning factor for scaling; not used here but kept for consistency.

    Raises:
        NameError: Indicates users should check the README for OptiTrack calibration details.

    Outputs:
        This function does not return any values. It only raises an exception.
    """
    logging.warning('See Readme.md to retrieve Optitrack calibration values.')
    # Raises an exception to prevent further execution without proper configuration details.
    raise NameError("See Readme.md to retrieve Optitrack calibration values.")


def calib_vicon_fun(file_to_convert_path, binning_factor=1):
    """
    Converts a Vicon `.xcp` calibration file to a `.toml` format.
    Handles transformations from camera view to object view using 
    the Rodrigues formula for rotations.

    Parameters:
        file_to_convert_path (str): Path to the `.xcp` file containing Vicon calibration data.
        binning_factor (int): Binning factor for scaling; always 1 for Vicon calibration.

    Returns:
        ret (list[float]): Residual reprojection error in millimeters.
        C (list[str]): Camera names.
        S (list[list[float]]): Image sizes as [width, height].
        D (list[np.array]): Distortion coefficients.
        K (list[np.array]): Intrinsic parameters as 3x3 matrices.
        R (list[np.array]): Extrinsic rotation matrices.
        T (list[np.array]): Extrinsic translation vectors.
    """
    logging.info(f'Converting {file_to_convert_path} to .toml calibration file...')
    
    # Reads the Vicon file and extracts relevant calibration data.
    ret, C, S, D, K, R, T = read_vicon(file_to_convert_path)
    
    # Convert extrinsic parameters from world coordinates to camera coordinates.
    RT = [world_to_camera_persp(r, t) for r, t in zip(R, T)]
    R = [rt[0] for rt in RT]  # Extract rotation matrices after transformation.
    T = [rt[1] for rt in RT]  # Extract translation vectors after transformation.

    # Converts rotation matrices into Rodrigues vectors for compact representation.
    R = [np.array(cv2.Rodrigues(r)[0]).flatten() for r in R]
    T = np.array(T)  # Converts the list of translation vectors into a numpy array.
    
    # Return the calibration parameters in a structured format.
    return ret, C, S, D, K, R, T


def read_vicon(vicon_path):
    '''
    Reads a Vicon .xcp calibration file and extracts intrinsic and extrinsic calibration parameters.
    The function processes XML-like data to produce camera parameters for further analysis.

    Parameters:
        vicon_path (str): Path to the `.xcp` calibration file.

    Returns:
        ret (list[float]): Residual reprojection error in millimeters.
        C (list[str]): Camera names.
        S (list[list[float]]): Image sizes as [width, height].
        D (list[np.array]): Distortion coefficients.
        K (list[np.array]): Intrinsic parameters as 3x3 matrices.
        R (list[np.array]): Extrinsic rotation matrices.
        T (list[np.array]): Extrinsic translation vectors.
    '''

    # Parse the .xcp file and get the root of the XML tree structure.
    root = etree.parse(vicon_path).getroot()

    # Initialize variables to store calibration data.
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []
    vid_id = []

    # Process camera names and image sizes.
    for i, tag in enumerate(root.findall('Camera')):
        # Camera name from DEVICEID attribute.
        C.append(tag.attrib.get('DEVICEID'))
        # Image size as SENSOR_SIZE.
        S.append([float(t) for t in tag.attrib.get('SENSOR_SIZE').split()])
        # Residual reprojection error from the first KeyFrame's WORLD_ERROR attribute.
        ret.append(float(tag.findall('KeyFrames/KeyFrame')[0].attrib.get('WORLD_ERROR')))
        # Track indices for cameras with video types.
        vid_id.append(i)

    # Process intrinsic parameters: distortion coefficients and intrinsic matrices.
    for cam_elem in root.findall('Camera'):
        # Attempt to extract distortion coefficients from VICON_RADIAL or VICON_RADIAL2 attributes.
        try:
            dist = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('VICON_RADIAL2').split()[3:5]
        except:
            dist = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('VICON_RADIAL').split()
        # Append distortion coefficients to the list, adding tangential coefficients as 0.0.
        D.append([float(d) for d in dist] + [0.0, 0.0])

        # Focal lengths and principal points.
        fu = float(cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('FOCAL_LENGTH'))
        fv = fu / float(cam_elem.attrib.get('PIXEL_ASPECT_RATIO'))  # Adjust for pixel aspect ratio.
        cam_center = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('PRINCIPAL_POINT').split()
        cu, cv = [float(c) for c in cam_center]

        # Construct intrinsic matrix and append it.
        K.append(np.array([fu, 0., cu, 0., fv, cv, 0., 0., 1.]).reshape(3, 3))

    # Process extrinsic parameters: rotation matrices and translation vectors.
    for cam_elem in root.findall('Camera'):
        # Extract quaternion-based orientation and convert it to a rotation matrix.
        rot = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('ORIENTATION').split()
        R_quat = [float(r) for r in rot]
        R_mat = quat2mat(R_quat, scalar_idx=3)  # Convert quaternion to rotation matrix.
        R.append(R_mat)

        # Extract and scale translation vectors (convert mm to meters).
        trans = cam_elem.findall('KeyFrames/KeyFrame')[0].attrib.get('POSITION').split()
        T.append([float(t) / 1000 for t in trans])

    # Reorganize camera names and their parameters by natural order for video cameras.
    C_vid_id = [v for v in vid_id if ('VIDEO' or 'Video') in root.findall('Camera')[v].attrib.get('TYPE')]
    C_vid = [root.findall('Camera')[v].attrib.get('DEVICEID') for v in C_vid_id]
    C = sorted(C_vid, key=natural_sort_key)  # Sort camera names naturally.
    
    # Create an index mapping sorted camera names back to their original order.
    C_id_sorted = [i for v_sorted in C for i, v in enumerate(root.findall('Camera')) if v.attrib.get('DEVICEID') == v_sorted]
    
    # Reorganize all other parameters to match the sorted camera order.
    S = [S[c] for c in C_id_sorted]
    D = [D[c] for c in C_id_sorted]
    K = [K[c] for c in C_id_sorted]
    R = [R[c] for c in C_id_sorted]
    T = [T[c] for c in C_id_sorted]

    # Return all calibration parameters as structured lists.
    return ret, C, S, D, K, R, T

def read_intrinsic_yml(intrinsic_path):
    '''
    Reads an intrinsic .yml calibration file and extracts intrinsic parameters for each camera.

    Parameters:
        intrinsic_path (str): Path to the .yml file containing intrinsic parameters.

    Returns:
        N (list[str]): Camera names.
        S (list[list[float]]): Image sizes as [width, height].
        K (list[np.array]): Intrinsic matrices (3x3 arrays).
        D (list[np.array]): Distortion coefficients (radial and tangential).
    
    Note:
        Image size is estimated as twice the optical center position. This should be adjusted
        in the resulting `.toml` file if necessary.
    '''

    # Open the intrinsic .yml file using OpenCV's FileStorage.
    intrinsic_yml = cv2.FileStorage(intrinsic_path, cv2.FILE_STORAGE_READ)

    # Determine the number of cameras from the 'names' node.
    cam_number = intrinsic_yml.getNode('names').size()

    # Initialize lists for camera names, sizes, intrinsic matrices, and distortion coefficients.
    N, S, D, K = [], [], [], []

    for i in range(cam_number):
        # Read the camera name.
        name = intrinsic_yml.getNode('names').at(i).string()
        N.append(name)

        # Read the intrinsic matrix for the camera.
        K.append(intrinsic_yml.getNode(f'K_{name}').mat())

        # Read the distortion coefficients and exclude the last value (e.g., k3).
        D.append(intrinsic_yml.getNode(f'dist_{name}').mat().flatten()[:-1])

        # Calculate the image size as twice the optical center position.
        S.append([K[i][0, 2] * 2, K[i][1, 2] * 2])

    # Return camera names, sizes, intrinsic matrices, and distortion coefficients.
    return N, S, K, D


def read_extrinsic_yml(extrinsic_path):
    '''
    Reads an extrinsic .yml calibration file and extracts extrinsic parameters for each camera.

    Parameters:
        extrinsic_path (str): Path to the .yml file containing extrinsic parameters.

    Returns:
        N (list[str]): Camera names.
        R (list[np.array]): Extrinsic rotations as Rodrigues vectors.
        T (list[np.array]): Extrinsic translations.
    '''

    # Open the extrinsic .yml file using OpenCV's FileStorage.
    extrinsic_yml = cv2.FileStorage(extrinsic_path, cv2.FILE_STORAGE_READ)

    # Determine the number of cameras from the 'names' node.
    cam_number = extrinsic_yml.getNode('names').size()

    # Initialize lists for camera names, rotations, and translations.
    N, R, T = [], [], []

    for i in range(cam_number):
        # Read the camera name.
        name = extrinsic_yml.getNode('names').at(i).string()
        N.append(name)

        # Read the extrinsic rotation as a Rodrigues vector.
        R.append(extrinsic_yml.getNode(f'R_{name}').mat().flatten())

        # Read the extrinsic translation vector.
        T.append(extrinsic_yml.getNode(f'T_{name}').mat().flatten())

    # Return camera names, rotations, and translations.
    return N, R, T


def calib_easymocap_fun(files_to_convert_paths, binning_factor=1):
    '''
    Converts EasyMocap .yml calibration files into structured calibration data.

    Parameters:
        files_to_convert_paths (tuple[str, str]): Paths to the 'intri.yml' and 'extri.yml' calibration files.
        binning_factor (int): Binning factor for pixel averaging (always 1 for EasyMocap).

    Returns:
        ret (list[float]): Placeholder for residual reprojection error (set to NaN for all cameras).
        C (list[str]): Camera names.
        S (list[list[float]]): Image sizes as [width, height].
        D (list[np.array]): Distortion coefficients.
        K (list[np.array]): Intrinsic matrices (3x3 arrays).
        R (list[np.array]): Extrinsic rotations (Rodrigues vectors).
        T (list[np.array]): Extrinsic translations.
    '''

    # Unpack paths for intrinsic and extrinsic .yml files.
    extrinsic_path, intrinsic_path = files_to_convert_paths

    # Read intrinsic parameters from the corresponding .yml file.
    C, S, K, D = read_intrinsic_yml(intrinsic_path)

    # Read extrinsic parameters from the corresponding .yml file.
    _, R, T = read_extrinsic_yml(extrinsic_path)

    # Initialize reprojection errors as NaN (not available in EasyMocap files).
    ret = [np.nan] * len(C)

    # Return structured calibration data.
    return ret, C, S, D, K, R, T


def calib_biocv_fun(files_to_convert_paths, binning_factor=1):
    '''
    Convert bioCV calibration files into structured calibration data.

    Parameters:
        files_to_convert_paths (list[str]): Paths to the calibration files (without extensions).
        binning_factor (int): Always 1 for bioCV calibration.

    Returns:
        ret (list[float]): Residual reprojection error in mm (placeholder values set to NaN).
        C (list[str]): Camera names.
        S (list[list[float]]): Image sizes as [width, height].
        D (list[np.array]): Distortion coefficients.
        K (list[np.array]): Intrinsic matrices (3x3 arrays).
        R (list[np.array]): Extrinsic rotations (Rodrigues vectors).
        T (list[np.array]): Extrinsic translations.
    '''
    logging.info(f'Converting {[os.path.basename(f) for f in files_to_convert_paths]} to .toml calibration file...')

    # Initialize lists for outputs
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []

    # Loop through each calibration file
    for i, f_path in enumerate(files_to_convert_paths):
        with open(f_path) as f:
            calib_data = f.read().split('\n')  # Read and split the file content by lines
            
            # Residual error is not provided; use NaN as placeholder
            ret += [np.nan]

            # Assign camera names based on the order in the list
            C += [f'cam_{str(i).zfill(2)}']

            # Read image size
            S += [[int(calib_data[0]), int(calib_data[1])]]

            # Read distortion coefficients (first 4 values from the last second-to-last line)
            D += [[float(d) for d in calib_data[-2].split(' ')[:4]]]

            # Read intrinsic matrix (lines 2-4)
            K += [np.array([k.strip().split(' ') for k in calib_data[2:5]], np.float32)]

            # Read rotation and translation matrix (lines 6-8)
            RT = np.array([k.strip().split(' ') for k in calib_data[6:9]], np.float32)

            # Convert rotation matrix to Rodrigues vector
            R += [cv2.Rodrigues(RT[:, :3])[0].squeeze()]

            # Translation vector, converted to meters
            T += [RT[:, 3] / 1000]

    return ret, C, S, D, K, R, T


def calib_opencap_fun(files_to_convert_paths, binning_factor=1):
    '''
    Convert OpenCap calibration files into structured calibration data.

    Parameters:
        files_to_convert_paths (list[str]): Paths to the .pickle calibration files.
        binning_factor (int): Always 1 for OpenCap calibration.

    Returns:
        ret (list[float]): Residual reprojection error in mm (placeholder values set to NaN).
        C (list[str]): Camera names.
        S (list[list[float]]): Image sizes as [width, height].
        D (list[np.array]): Distortion coefficients.
        K (list[np.array]): Intrinsic matrices (3x3 arrays).
        R (list[np.array]): Extrinsic rotations (Rodrigues vectors).
        T (list[np.array]): Extrinsic translations.
    '''
    logging.info(f'Converting {[os.path.basename(f) for f in files_to_convert_paths]} to .toml calibration file...')

    # Initialize lists for outputs
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []

    # Loop through each calibration file
    for i, f_path in enumerate(files_to_convert_paths):
        with open(f_path, 'rb') as f_pickle:
            # Load calibration data from the pickle file
            calib_data = pickle.load(f_pickle)

            # Residual error is not provided; use NaN as placeholder
            ret += [np.nan]

            # Assign camera names based on the order in the list
            C += [f'cam_{str(i).zfill(2)}']

            # Read image size (convert to [width, height])
            S += [list(calib_data['imageSize'].squeeze()[::-1])]

            # Read distortion coefficients (excluding k3)
            D += [list(calib_data['distortion'][0][:-1])]

            # Read intrinsic matrix
            K += [calib_data['intrinsicMat']]

            # Read rotation and translation
            R_cam = calib_data['rotation']
            T_cam = calib_data['translation'].squeeze()

            # Convert camera frame to world frame
            R_w, T_w = world_to_camera_persp(R_cam, T_cam)

            # Apply a rotation of -Pi/2 around the x-axis and Pi around the z-axis
            R_w_90, T_w_90 = rotate_cam(R_w, T_w, ang_x=-np.pi/2, ang_y=0, ang_z=np.pi)

            # Convert back to camera frame
            R_c_90, T_c_90 = world_to_camera_persp(R_w_90, T_w_90)

            # Convert rotation matrix to Rodrigues vector
            R += [cv2.Rodrigues(R_c_90)[0].squeeze()]

            # Translation vector, converted to meters
            T += [T_cam / 1000]

    return ret, C, S, D, K, R, T


def calib_calc_fun(calib_dir, intrinsics_config_dict, extrinsics_config_dict):
    '''
    Calibrates intrinsic and extrinsic parameters
    from images or videos of a checkerboard or Charuco board,
    or retrieves them from an existing calibration file.

    **Inputs:**
    - `calib_dir`: Directory containing intrinsic and extrinsic folders, each populated with camera directories.
    - `intrinsics_config_dict`: Dictionary containing intrinsics calibration settings:
        - `overwrite_intrinsics` (bool): Determines whether to recalculate intrinsic parameters.
        - `intrinsics_method` (str): Calibration method ('charuco' for ArUco + chessboard, or 'chessboard').
        - `show_detection_intrinsics` (bool): Whether to display detection during intrinsics calibration.
        - `intrinsics_extension` (str): File format for input videos/images (e.g., 'mp4', 'jpg').
        - `extract_every_N_sec` (float): Frame extraction frequency for videos (e.g., 0.2 seconds).
        - `intrinsics_corners_nb` (list[int]): Number of internal corners [rows, cols] for the board.
        - `intrinsics_square_size` (float): Square size in mm for calibration patterns.
        - `intrinsics_aruco_size` (float): Marker size in mm (often 0.75 * square size).
        - `intrinsics_aruco_dict` (str): ArUco dictionary type (e.g., 'DICT_6X6_250').
    - `extrinsics_config_dict`: Dictionary containing extrinsics calibration settings:
        - `calculate_extrinsics` (bool): Whether to calculate extrinsic parameters.
        - `extrinsics_method` (str): Calibration method ('board', 'scene', or 'keypoints').
            - `board`: Uses a checkerboard/Charuco board laid on the floor (not recommended for accuracy).
            - `scene`: Requires manual labeling of known 3D points in the scene.
            - `keypoints`: Uses pose estimation of a moving subject in the scene (requires synchronized cameras).
        - `moving_cameras` (bool): Placeholder for future functionality.
        - `extrinsics_extension` (str): File format for input videos/images (e.g., 'mp4').
        - `extrinsics_corners_nb` (list[int]): Number of corners [rows, cols] for the extrinsics board.
        - `extrinsics_square_size` (float): Square size in mm (or dimensions for rectangular patterns).
        - `object_coords_3d` (list[list[float]]): Known 3D coordinates in meters for the 'scene' method.

    **Outputs:**
    - `ret`: Residual reprojection error (list[float]) in pixels.
    - `C`: List of camera names (list[str]).
    - `S`: List of image sizes (list[list[float]]).
    - `D`: List of distortion coefficients (list[np.array]).
    - `K`: List of intrinsic matrices (list[np.array], 3x3).
    - `R`: List of extrinsic rotations (list[np.array], Rodrigues vectors).
    - `T`: List of extrinsic translations (list[np.array]).

    **Configuration File Integration:**
    The `.toml` configuration file defines two modes:
    - `convert`: Converts calibration files from other software (e.g., Qualisys, OpenCap).
    - `calculate`: Performs calibration based on images/videos of a calibration pattern.

    Intrinsics:
    - The method is specified (`charuco` or `chessboard`).
    - Board dimensions (`intrinsics_corners_nb`) and square size (`intrinsics_square_size`) are used.
    - Frame extraction frequency (`extract_every_N_sec`) is applied for videos.

    Extrinsics:
    - Methods (`board`, `scene`, or `keypoints`) determine how calibration is performed.
    - For `scene`, 3D coordinates (`object_coords_3d`) must be provided.

    Example:
    - The function reads the configuration file and extracts relevant parameters for calibration.
    - If existing calibration files are found and `overwrite_intrinsics` is `False`, the parameters are reused.
    - Extrinsics are calculated only if `calculate_extrinsics` is `True`.

    '''

    # Retrieve parameters for intrinsics and extrinsics calibration
    overwrite_intrinsics = intrinsics_config_dict.get('overwrite_intrinsics')
    calculate_extrinsics = extrinsics_config_dict.get('calculate_extrinsics')

    # Attempt to retrieve existing calibration file if overwrite is disabled
    try:
        calib_file = glob.glob(os.path.join(calib_dir, f'Calib*.toml'))[0]  # Locate the calibration file
    except:
        pass  # If no file found, continue to calculate intrinsics

    if not overwrite_intrinsics and 'calib_file' in locals():
        logging.info(f'\nPreexisting calibration file found: \'{calib_file}\'.')
        logging.info(f'\nRetrieving intrinsic parameters from file. Set "overwrite_intrinsics" to true in Config.toml to recalculate them.')
        
        calib_data = toml.load(calib_file)  # Load calibration data from the TOML file

        # Extract intrinsic parameters from the calibration file
        ret, C, S, D, K, R, T = [], [], [], [], [], [], []
        for cam in calib_data:
            if cam != 'metadata':  # Skip metadata section
                ret += [0.0]  # Placeholder for reprojection error
                C += [calib_data[cam]['name']]  # Camera name
                S += [calib_data[cam]['size']]  # Image size
                K += [np.array(calib_data[cam]['matrix'])]  # Intrinsic matrix
                D += [calib_data[cam]['distortions']]  # Distortion coefficients
                R += [[0.0, 0.0, 0.0]]  # Placeholder for rotation
                T += [[0.0, 0.0, 0.0]]  # Placeholder for translation
        nb_cams_intrinsics = len(C)  # Count the number of cameras with intrinsics
    else:
        # Calculate intrinsic parameters
        logging.info(f'\nCalculating intrinsic parameters...')
        ret, C, S, D, K, R, T = calibrate_intrinsics(calib_dir, intrinsics_config_dict)
        nb_cams_intrinsics = len(C)  # Count the number of cameras with intrinsics

    # Calculate extrinsic parameters if enabled
    if calculate_extrinsics:
        logging.info(f'\nCalculating extrinsic parameters...')
        
        # Verify consistency in the number of cameras for intrinsics and extrinsics
        nb_cams_extrinsics = len(next(os.walk(os.path.join(calib_dir, 'extrinsics')))[1])
        if nb_cams_intrinsics != nb_cams_extrinsics:
            raise Exception(f'Error: The number of cameras is not consistent:\n'
                            f'Found {nb_cams_intrinsics} cameras based on the number of intrinsic folders or on calibration file data,\n'
                            f'and {nb_cams_extrinsics} cameras based on the number of extrinsic folders.')

        # Perform extrinsics calibration
        ret, C, S, D, K, R, T = calibrate_extrinsics(calib_dir, extrinsics_config_dict, C, S, K, D)
    else:
        logging.info(f'\nExtrinsic parameters won\'t be calculated. Set "calculate_extrinsics" to true in Config.toml to calculate them.')

    return ret, C, S, D, K, R, T



def calibrate_intrinsics(calib_dir, intrinsics_config_dict):
    '''
    Calculate intrinsic parameters
    from images or videos of a checkerboard or a Charuco board.
    Extract frames, then detect corners, then calibrate.

    INPUTS:
    - calib_dir: directory containing intrinsic and extrinsic folders, each populated with camera directories
    - intrinsics_config_dict: dictionary of intrinsics parameters:
        - overwrite_intrinsics: (bool) whether to overwrite existing intrinsics
        - intrinsics_method: (str) method of calibration ('charuco' or 'chessboard')
        - show_detection_intrinsics: (bool) show detected corners for debugging
        - intrinsics_extension: (str) file extension of input files ('mp4', 'png', etc.)
        - extract_every_N_sec: (float) frame extraction interval from video (if input is a video)
        - intrinsics_corners_nb: (list[int]) number of corners [rows, cols]
        - intrinsics_square_size: (float) size of squares in mm
        - intrinsics_marker_size: (float) size of ArUco markers in mm
        - intrinsics_aruco_dict: (str) dictionary type for ArUco markers (e.g., 'DICT_6X6_250')

    OUTPUTS:
    - ret: (list[float]) residual reprojection error
    - C: (list[str]) camera names
    - S: (list[list[float]]) image sizes [width, height]
    - D: (list[np.ndarray]) distortion coefficients
    - K: (list[np.ndarray]) intrinsic matrices
    - R: (list[np.ndarray]) extrinsic rotations (default zeros)
    - T: (list[np.ndarray]) extrinsic translations (default zeros)
    '''

    # Get the list of camera directories inside 'intrinsics'
    try:
        intrinsics_cam_listdirs_names = next(os.walk(os.path.join(calib_dir, 'intrinsics')))[1]
    except StopIteration:
        logging.exception(f'Error: No {os.path.join(calib_dir, "intrinsics")} folder found.')
        raise Exception(f'Error: No {os.path.join(calib_dir, "intrinsics")} folder found.')

    # Extract parameters from the configuration dictionary
    intrinsics_method = intrinsics_config_dict.get('intrinsics_method')
    intrinsics_extension = intrinsics_config_dict.get('intrinsics_extension')
    extract_every_N_sec = intrinsics_config_dict.get('extract_every_N_sec')
    show_detection_intrinsics = intrinsics_config_dict.get('show_detection_intrinsics')
    intrinsics_corners_nb = intrinsics_config_dict.get('intrinsics_corners_nb')
    intrinsics_square_size = intrinsics_config_dict.get('intrinsics_square_size') / 1000  # Convert to meters
    intrinsics_aruco_size = intrinsics_config_dict.get('intrinsics_aruco_size') / 1000  # Convert to meters
    intrinsics_aruco_dict = intrinsics_config_dict.get('intrinsics_aruco_dict')

    # Initialize lists to store calibration outputs
    ret, C, S, D, K, R, T = [], [], [], [], [], [], []

    # Process each camera directory
    for i, cam in enumerate(intrinsics_cam_listdirs_names):
        # Prepare 3D object points based on the board configuration
        objp = np.zeros((intrinsics_corners_nb[0] * intrinsics_corners_nb[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:intrinsics_corners_nb[0], 0:intrinsics_corners_nb[1]].T.reshape(-1, 2)
        objp[:, :2] *= intrinsics_square_size
        objpoints, imgpoints = [], []

        logging.info(f'\nCamera {cam}:')

        # Get all files in the camera directory with the specified extension
        img_vid_files = glob.glob(os.path.join(calib_dir, 'intrinsics', cam, f'*.{intrinsics_extension}'))
        frames_raw_dir = os.path.join(calib_dir, 'intrinsics', cam, "frames_raw")
        frames_annoted_dir = os.path.join(calib_dir, 'intrinsics', cam, "frames_annoted")

        # Create necessary directories if they don't exist
        os.makedirs(frames_raw_dir, exist_ok=True)
        os.makedirs(frames_annoted_dir, exist_ok=True)

        # Raise an error if no input files are found
        if len(img_vid_files) == 0:
            logging.exception(f'The folder {os.path.join(calib_dir, "intrinsics", cam)} does not exist or does not contain any files with extension .{intrinsics_extension}.')
            raise ValueError(f'The folder {os.path.join(calib_dir, "intrinsics", cam)} does not exist or does not contain any files with extension .{intrinsics_extension}.')

        # Sort input files numerically (useful if filenames contain numbers)
        img_vid_files = sorted(img_vid_files, key=lambda c: [int(n) for n in re.findall(r'\d+', c)])

        # Extract frames if the input is a video
        try:
            cap = cv2.VideoCapture(img_vid_files[0])
            if not cap.isOpened():
                raise
            extract_frames(img_vid_files[0], frames_raw_dir, extract_every_N_sec)
        except:
            pass

        # Get list of raw frames for calibration
        image_files = [os.path.join(frames_raw_dir, f) for f in os.listdir(frames_raw_dir) if f.endswith(".png")]
        image_files.sort()

        ### Charuco Method ###
        if intrinsics_method == "charuco":
            aruco_dict = ARUCO_DICT_MAPPING[intrinsics_aruco_dict]
            aruco_dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
            board = cv2.aruco.CharucoBoard((intrinsics_corners_nb[1], intrinsics_corners_nb[0]), intrinsics_square_size, intrinsics_aruco_size, aruco_dictionary)
            board.setLegacyPattern(True)
            params = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dictionary, params)

            all_charuco_corners, all_charuco_ids = [], []

            with tqdm(total=len(image_files), desc="Intrinsics calibration", unit="image") as pbar:
                for image_file in image_files:
                    image = cv2.imread(image_file)
                    image_copy = image.copy()
                    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(image, dictionary=aruco_dictionary, parameters=params)

                    if marker_ids is not None and len(marker_ids) > 0:
                        cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
                        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                        charuco_retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)

                        if charuco_retval and charuco_corners is not None and len(charuco_corners) >= 6:
                            cv2.aruco.drawDetectedCornersCharuco(image_copy, charuco_corners, charuco_ids)

                            if show_detection_intrinsics:
                                annotated_filename = os.path.join(frames_annoted_dir, os.path.basename(image_file))
                                cv2.imwrite(annotated_filename, image_copy)

                            all_charuco_corners.append(charuco_corners)
                            all_charuco_ids.append(charuco_ids)

                    pbar.update(1)

            if len(all_charuco_corners) > 0 and len(all_charuco_ids) > 0:
                retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                    all_charuco_corners, all_charuco_ids, board, image.shape[1::-1], None, None
                )
                ret.append(retval)
                C.append(cam)
                S.append([image.shape[1], image.shape[0]])
                D.append(dist_coeffs[0])
                K.append(camera_matrix)
                R.append([0.0, 0.0, 0.0])
                T.append([0.0, 0.0, 0.0])

                logging.info(f"Calibration completed for camera {cam}. Intrinsics error: {np.around(retval, decimals=3)} px.")
            else:
                logging.info(f"Calibration error: No valid corners detected for camera {cam}.")
        
        ### Chessboard Method ###
        elif intrinsics_method == "chessboard":
            for img_path in image_files:
                imgp_confirmed = findCorners(img_path, frames_annoted_dir, intrinsics_corners_nb, objp=objp, show=show_detection_intrinsics)
                if isinstance(imgp_confirmed, np.ndarray):
                    imgpoints.append(imgp_confirmed)
                    objpoints.append(objp)

            if len(imgpoints) < 10:
                logging.info(f"Only {len(imgpoints)} valid frames for camera {cam}. Results may be inaccurate.")
            
            img = cv2.imread(image_files[0])
            objpoints = np.array(objpoints)
            ret_cam, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1::-1], None, None)
            
            ret.append(ret_cam)
            C.append(cam)
            S.append([img.shape[1], img.shape[0]])
            D.append(dist[0])
            K.append(mtx)
            R.append([0.0, 0.0, 0.0])
            T.append([0.0, 0.0, 0.0])

            logging.info(f'Calibration error: {np.around(ret_cam, decimals=3)} px for camera {cam}.')

    return ret, C, S, D, K, R, T

def calibrate_extrinsics(calib_dir, extrinsics_config_dict, C, S, K, D):
    '''
    Calibrates extrinsic parameters
    from an image or the first frame of a video
    of a checkerboard or measured clues on the scene.

    INPUTS:
    - calib_dir: directory containing intrinsic and extrinsic folders, each populated with camera directories
    - extrinsics_config_dict: dictionary of extrinsics parameters:
        - extrinsics_method: (str) method of calibration ('board', 'scene', or 'keypoints')
        - calculate_extrinsics: (bool) flag to calculate extrinsics
        - show_detection_extrinsics: (bool) show detected points for debugging
        - extrinsics_extension: (str) file extension of input files ('mp4', 'png', etc.)
        - extrinsics_corners_nb: (list[int]) number of corners [rows, cols]
        - extrinsics_square_size: (float) size of squares in mm
        - extrinsics_marker_size: (float) size of ArUco markers in mm
        - extrinsics_aruco_dict: (str) dictionary type for ArUco markers
        - object_coords_3d: (list[list[float]]) 3D coordinates of known points in the scene

    OUTPUTS:
    - ret: (list[float]) reprojection errors for extrinsics
    - R: (list[np.ndarray]) extrinsic rotations (Rodrigues format)
    - T: (list[np.ndarray]) extrinsic translations
    '''

    # Ensure the 'extrinsics' folder exists in the calibration directory
    try:
        extrinsics_cam_listdirs_names = next(os.walk(os.path.join(calib_dir, 'extrinsics')))[1]
    except StopIteration:
        logging.exception(f'Error: No {os.path.join(calib_dir, "extrinsics")} folder found.')
        raise Exception(f'Error: No {os.path.join(calib_dir, "extrinsics")} folder found.')

    # Extract method of extrinsics calibration
    extrinsics_method = extrinsics_config_dict.get('extrinsics_method')
    ret, R, T = [], [], []

    # Handle board and scene calibration methods
    if extrinsics_method in ('board', 'scene'):

        # Define 3D object points
        if extrinsics_method == 'board':
            extrinsics_corners_nb = extrinsics_config_dict.get('board').get('extrinsics_corners_nb')
            extrinsics_square_size = extrinsics_config_dict.get('board').get('extrinsics_square_size') / 1000  # Convert to meters
            object_coords_3d = np.zeros((extrinsics_corners_nb[0] * extrinsics_corners_nb[1], 3), np.float32)
            object_coords_3d[:, :2] = np.mgrid[0:extrinsics_corners_nb[0], 0:extrinsics_corners_nb[1]].T.reshape(-1, 2)
            object_coords_3d[:, :2] *= extrinsics_square_size
        elif extrinsics_method == 'scene':
            object_coords_3d = np.array(extrinsics_config_dict.get('scene').get('object_coords_3d'), np.float32)

        # Save reference 3D coordinates to a .trc file
        calib_output_path = os.path.join(calib_dir, 'Object_points.trc')
        trc_write(object_coords_3d, calib_output_path)

        # Process each camera directory in 'extrinsics'
        for i, cam in enumerate(extrinsics_cam_listdirs_names):
            logging.info(f'\nCamera {cam}:')

            # Retrieve file extension and reprojection flag
            extrinsics_extension = (extrinsics_config_dict.get('board').get('extrinsics_extension') 
                                    if extrinsics_method == 'board' else 
                                    extrinsics_config_dict.get('scene').get('extrinsics_extension'))
            show_reprojection_error = (extrinsics_config_dict.get('board').get('show_reprojection_error') 
                                       if extrinsics_method == 'board' else 
                                       extrinsics_config_dict.get('scene').get('show_reprojection_error'))

            # Get input image or video files
            img_vid_files = glob.glob(os.path.join(calib_dir, 'extrinsics', cam, f'*.{extrinsics_extension}'))
            if not img_vid_files:
                logging.exception(f'The folder {os.path.join(calib_dir, "extrinsics", cam)} does not exist or does not contain any files with extension .{extrinsics_extension}.')
                raise ValueError(f'The folder {os.path.join(calib_dir, "extrinsics", cam)} does not exist or does not contain any files with extension .{extrinsics_extension}.')
            img_vid_files = sorted(img_vid_files, key=lambda c: [int(n) for n in re.findall(r'\d+', c)])

            # Read the first image or extract the first frame from the video
            img = cv2.imread(img_vid_files[0])
            if img is None:
                cap = cv2.VideoCapture(img_vid_files[0])
                _, img = cap.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Detect points based on the selected method
            if extrinsics_method == 'board':
                imgp, objp = findCorners(img_vid_files[0], extrinsics_corners_nb, objp=object_coords_3d, show=show_reprojection_error)
                if not imgp:
                    raise ValueError('No corners found. Try enabling "show_detection_extrinsics" or switching to "scene" method.')

            elif extrinsics_method == 'scene':
                imgp, objp = imgp_objp_visualizer_clicker(img, imgp=[], objp=object_coords_3d, img_path=img_vid_files[0])
                if not imgp:
                    raise ValueError('No points clicked. Ensure to click on image points corresponding to object_coords_3d.')

            # Calculate extrinsics using PnP
            mtx, dist = np.array(K[i]), np.array(D[i])
            _, r, t = cv2.solvePnP(np.array(objp), imgp, mtx, dist)

            # Refine extrinsics
            rvec_refined, tvec_refined = cv2.solvePnPRefineVVS(np.array(objp), imgp, mtx, dist, r, t)
            r, t = rvec_refined.flatten(), tvec_refined.flatten()

            # Project object points to the image plane for verification
            proj_obj = np.squeeze(cv2.projectPoints(objp, r, t, mtx, dist)[0])

            # Visualize reprojection error
            if show_reprojection_error:
                img = cv2.imread(img_vid_files[0])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                for o in proj_obj:
                    center = (int(o[0]), int(o[1]))
                    cv2.circle(img, center, 8, (0, 0, 255), -1)
                for i in imgp:
                    cv2.drawMarker(img, (int(i[0][0]), int(i[0][1])), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
                cv2.imshow(f"Reprojection Error - Camera {cam}", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            # Calculate reprojection error
            imgp_to_objreproj_dist = [np.linalg.norm(proj_obj[n] - imgp[n]) for n in range(len(proj_obj))]
            rms_px = np.sqrt(np.mean(np.square(imgp_to_objreproj_dist)))
            ret.append(rms_px)
            R.append(r)
            T.append(t)

    elif extrinsics_method == 'keypoints':
        raise NotImplementedError('Keypoints-based calibration is not yet implemented.')

    else:
        raise ValueError('Invalid value for extrinsics_method.')

    return ret, C, S, D, K, R, T


def findCorners(img_path, img_annoted_path, corner_nb, objp=[], show=True):
    '''
    Find corners in the photo of a checkerboard.
    Press 'Y' to accept detection, 'N' to dismiss this image, 'C' to click points by hand.
    Left click to add a point, right click to remove the last point.
    Use mouse wheel to zoom in and out and to pan.
    
    Make sure that: 
    - the checkerboard is surrounded by a white border
    - rows != lines, and row is even if lines is odd (or conversely)
    - it is flat and without reflections
    - corner_nb correspond to _internal_ corners
    
    INPUTS:
    - img_path: path to image (or video)
    - img_annoted_path: path to save annotated image
    - corner_nb: [H, W] internal corners in checkerboard: list of two integers [4, 7]
    - objp: (optional) array [3D corner coordinates]
    - show: (optional) choose whether to show corner detections

    OUTPUTS:
    - imgp_confirmed: array of [[2D corner coordinates]]
    - If `objp` is provided: objp_confirmed: array of [3D corner coordinates]
    '''

    # Criteria for corner refinement: stop after 30 iterations or if error is less than 0.001px
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Read the input image or extract the first frame from the video
    img = cv2.imread(img_path)
    if img is None:  # If the file is a video
        cap = cv2.VideoCapture(img_path)
        ret, img = cap.read()

    # Convert the image to grayscale for corner detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

    # Detect checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, corner_nb, None)

    # If corners are found
    if ret is True:
        # Refine corner coordinates to subpixel accuracy
        imgp = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        logging.info(f'{os.path.basename(img_path)}: Corners found.')

        # Draw the detected corners on the image
        cv2.drawChessboardCorners(img, corner_nb, imgp, ret)

        # Add corner indices to specific key corners for better visualization
        for i, corner in enumerate(imgp):
            if i in [0, corner_nb[0]-1, corner_nb[0]*(corner_nb[1]-1), corner_nb[0]*corner_nb[1]-1]:
                x, y = corner.ravel()
                cv2.putText(img, str(i+1), (int(x)-5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 7)
                cv2.putText(img, str(i+1), (int(x)-5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)

        if show:  # If visualization is enabled
            # Reset previous variables for manual adjustments
            for var_to_delete in ['imgp_confirmed', 'objp_confirmed']:
                if var_to_delete in globals():
                    del globals()[var_to_delete]

            # Open visualizer to confirm or adjust corners
            imgp_objp_confirmed = imgp_objp_visualizer_clicker(img, imgp=imgp, objp=objp, img_path=img_path)

            # Save the annotated image
            annotated_filename = os.path.join(img_annoted_path, os.path.basename(img_path))
            cv2.imwrite(annotated_filename, img)
        else:
            # If no manual adjustment, save directly
            imgp_objp_confirmed = imgp
            annotated_filename = os.path.join(img_annoted_path, os.path.basename(img_path))
            cv2.imwrite(annotated_filename, img)

    # If corners are not found
    else:
        logging.info(f'{os.path.basename(img_path)}: Corners not found. To label them by hand, set "show_detection_intrinsics" to true in the Config.toml file.')
        if show:
            # Open visualizer to manually label points
            imgp_objp_confirmed = imgp_objp_visualizer_clicker(img, imgp=[], objp=objp, img_path=img_path)
        else:
            # No corners found, return an empty list
            imgp_objp_confirmed = []

    return imgp_objp_confirmed



def imgp_objp_visualizer_clicker(img, imgp=[], objp=[], img_path=''):
    '''
    Shows image img. 
    If imgp is given, displays them in green
    If objp is given, can be displayed in a 3D plot if 'C' is pressed.
    If img_path is given, just uses it to name the window

    If 'Y' is pressed, closes all and returns confirmed imgp and (if given) objp
    If 'N' is pressed, closes all and returns nothing
    If 'C' is pressed, allows clicking imgp by hand. If objp is given:
        Displays them in 3D as a helper. 
        Left click to add a point, right click to remove the last point.
        Press 'H' to indicate that one of the objp is not visible on image
        Closes all and returns imgp and objp if all points have been clicked
    Allows for zooming and panning with middle click
    
    INPUTS:
    - img: image opened with openCV
    - optional: imgp: detected image points, to be accepted or not. Array of [[2d corner coordinates]]
    - optionnal: objp: array of [3d corner coordinates]
    - optional: img_path: path to image

    OUTPUTS:
    - imgp_confirmed: image points that have been correctly identified. array of [[2d corner coordinates]]
    - only if objp!=[]: objp_confirmed: array of [3d corner coordinates]
    '''
    global old_image_path
    old_image_path = img_path
                                 
    def on_key(event):
        '''
        Handles key press events:
        'Y' to return imgp, 'N' to dismiss image, 'C' to click points by hand.
        Left click to add a point, 'H' to indicate it is not visible, right click to remove the last point.
        '''

        global imgp_confirmed, objp_confirmed, objp_confirmed_notok, scat, ax_3d, fig_3d, events, count
        
        if event.key == 'y':
            # If 'y', close all
            # If points have been clicked, imgp_confirmed is returned, else imgp
            # If objp is given, objp_confirmed is returned in addition
            if 'scat' not in globals() or 'imgp_confirmed' not in globals():
                imgp_confirmed = imgp
                objp_confirmed = objp
            else:
                imgp_confirmed = np.array([imgp.astype('float32') for imgp in imgp_confirmed])
                objp_confirmed = objp_confirmed
            # OpenCV needs at leas 4 correspondance points to calibrate
            if len(imgp_confirmed) < 6:
                objp_confirmed = []
                imgp_confirmed = []
            # close all, del all global variables except imgp_confirmed and objp_confirmed
            plt.close('all')
            if len(objp) == 0:
                if 'objp_confirmed' in globals():
                    del objp_confirmed

        if event.key == 'n' or event.key == 'q':
            # If 'n', close all and return nothing
            plt.close('all')
            imgp_confirmed = []
            objp_confirmed = []

        if event.key == 'c':
            # TODO: RIGHT NOW, IF 'C' IS PRESSED ANOTHER TIME, OBJP_CONFIRMED AND IMGP_CONFIRMED ARE RESET TO []
            # We should reopen a figure without point on it
            img_for_pointing = cv2.imread(old_image_path)
            if img_for_pointing is None:
                cap = cv2.VideoCapture(old_image_path)
                ret, img_for_pointing = cap.read()
            img_for_pointing = cv2.cvtColor(img_for_pointing, cv2.COLOR_BGR2RGB)
            ax.imshow(img_for_pointing)
            # To update the image
            plt.draw()

            if 'objp_confirmed' in globals():
                del objp_confirmed
            # If 'c', allows retrieving imgp_confirmed by clicking them on the image
            scat = ax.scatter([],[],s=100,marker='+',color='g')
            plt.connect('button_press_event', on_click)
            # If objp is given, display 3D object points in black
            if len(objp) != 0 and not plt.fignum_exists(2):
                fig_3d = plt.figure()
                fig_3d.tight_layout()
                fig_3d.canvas.manager.set_window_title('Object points to be clicked')
                ax_3d = fig_3d.add_subplot(projection='3d')
                plt.rc('xtick', labelsize=5)
                plt.rc('ytick', labelsize=5)
                for i, (xs,ys,zs) in enumerate(np.float32(objp)):
                    ax_3d.scatter(xs,ys,zs, marker='.', color='k')
                    ax_3d.text(xs,ys,zs,  f'{str(i+1)}', size=10, zorder=1, color='k') 
                set_axes_equal(ax_3d)
                ax_3d.set_xlabel('X')
                ax_3d.set_ylabel('Y')
                ax_3d.set_zlabel('Z')
                if np.all(objp[:,2] == 0):
                    ax_3d.view_init(elev=-90, azim=0)
                fig_3d.show()

        if event.key == 'h':
            # If 'h', indicates that one of the objp is not visible on image
            # Displays it in red on 3D plot
            if len(objp) != 0  and 'ax_3d' in globals():
                count = [0 if 'count' not in globals() else count+1][0]
                if 'events' not in globals():
                    # retrieve first objp_confirmed_notok and plot 3D
                    events = [event]
                    objp_confirmed_notok = objp[count]
                    ax_3d.scatter(*objp_confirmed_notok, marker='o', color='r')
                    fig_3d.canvas.draw()
                elif count == len(objp)-1:
                    # if all objp have been clicked or indicated as not visible, close all
                    objp_confirmed = np.array([[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0])[:-1]
                    imgp_confirmed = np.array(np.expand_dims(scat.get_offsets(), axis=1), np.float32) 
                    plt.close('all')
                    for var_to_delete in ['events', 'count', 'scat', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
                        if var_to_delete in globals():
                            del globals()[var_to_delete]
                else:
                    # retrieve other objp_confirmed_notok and plot 3D
                    events.append(event)
                    objp_confirmed_notok = objp[count]
                    ax_3d.scatter(*objp_confirmed_notok, marker='o', color='r')
                    fig_3d.canvas.draw()
            else:
                pass


    def on_click(event):
        '''
        Detect click position on image
        If right click, last point is removed
        '''
        
        global imgp_confirmed, objp_confirmed, objp_confirmed_notok, scat, ax_3d, fig_3d, events, count, xydata
        
        # Left click: Add clicked point to imgp_confirmed
        # Display it on image and on 3D plot
        if event.button == 1: 
            # To remember the event to cancel after right click
            if 'events' in globals():
                events.append(event)
            else:
                events = [event]

            # Add clicked point to image
            xydata = scat.get_offsets()
            new_xydata = np.concatenate((xydata,[[event.xdata,event.ydata]]))
            scat.set_offsets(new_xydata)
            imgp_confirmed = np.expand_dims(scat.get_offsets(), axis=1)    
            plt.draw()

            # Add clicked point to 3D object points if given
            if len(objp) != 0:
                count = [0 if 'count' not in globals() else count+1][0]
                if count==0:
                    # retrieve objp_confirmed and plot 3D
                    objp_confirmed = [objp[count]]
                    ax_3d.scatter(*objp[count], marker='o', color='g')
                    fig_3d.canvas.draw()
                elif count == len(objp)-1:
                    # close all
                    plt.close('all')
                    # retrieve objp_confirmed
                    objp_confirmed = np.array([[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0])
                    imgp_confirmed = np.array(imgp_confirmed, np.float32)
                    # delete all
                    for var_to_delete in ['events', 'count', 'scat', 'scat_3d', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
                        if var_to_delete in globals():
                            del globals()[var_to_delete]
                else:
                    # retrieve objp_confirmed and plot 3D
                    objp_confirmed = [[objp[count]] if 'objp_confirmed' not in globals() else objp_confirmed+[objp[count]]][0]
                    ax_3d.scatter(*objp[count], marker='o', color='g')
                    fig_3d.canvas.draw()
                

        # Right click: 
        # If last event was left click, remove last point and if objp given, from objp_confirmed
        # If last event was 'H' and objp given, remove last point from objp_confirmed_notok
        elif event.button == 3: # right click
            if 'events' in globals():
                # If last event was left click: 
                if 'button' in dir(events[-1]):
                    if events[-1].button == 1: 
                        # Remove lastpoint from image
                        new_xydata = scat.get_offsets()[:-1]
                        scat.set_offsets(new_xydata)
                        plt.draw()
                        # Remove last point from imgp_confirmed
                        imgp_confirmed = imgp_confirmed[:-1]
                        if len(objp) != 0:
                            if count >= 0: 
                                count -= 1
                            # Remove last point from objp_confirmed
                            objp_confirmed = objp_confirmed[:-1]
                            # remove from plot 
                            if len(ax_3d.collections) > len(objp):
                                ax_3d.collections[-1].remove()
                                fig_3d.canvas.draw()
                            
                # If last event was 'h' key
                elif events[-1].key == 'h':
                    if len(objp) != 0:
                        if count >= 1: count -= 1
                        # Remove last point from objp_confirmed_notok
                        objp_confirmed_notok = objp_confirmed_notok[:-1]
                        # remove from plot  
                        if len(ax_3d.collections) > len(objp):
                            ax_3d.collections[-1].remove()
                            fig_3d.canvas.draw()                
    

    def set_axes_equal(ax):
        '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc.
        From https://stackoverflow.com/questions/13685386/how-to-set-the-equal-aspect-ratio-for-all-axes-x-y-z

        Input
        ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    # Write instructions
    cv2.putText(img, 'Type "Y" to accept point detection.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'Type "Y" to accept point detection.', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, 'If points are wrongfully (or not) detected:', (20, 43), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'If points are wrongfully (or not) detected:', (20, 43), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '- type "N" to dismiss this image,', (20, 66), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '- type "N" to dismiss this image,', (20, 66), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '- type "C" to click points by hand (beware of their order).', (20, 89), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '- type "C" to click points by hand (beware of their order).', (20, 89), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ', (20, 112), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '   left click to add a point, right click to remove it, "H" to indicate it is not visible. ', (20, 112), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, '   Confirm with "Y", cancel with "N".', (20, 135), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, '   Confirm with "Y", cancel with "N".', (20, 135), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    cv2.putText(img, 'Use mouse wheel to zoom in and out and to pan', (20, 158), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 7, lineType = cv2.LINE_AA)
    cv2.putText(img, 'Use mouse wheel to zoom in and out and to pan', (20, 158), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,0,0), 2, lineType = cv2.LINE_AA)    
    
    # Put image in a matplotlib figure for more controls
    plt.rcParams['toolbar'] = 'None'
    fig, ax = plt.subplots()
    fig = plt.gcf()
    fig.canvas.manager.set_window_title(os.path.basename(img_path))
    ax.axis("off")
    for corner in imgp:
        x, y = corner.ravel()
        cv2.drawMarker(img, (int(x),int(y)), (128,128,128), cv2.MARKER_CROSS, 10, 2)
    ax.imshow(img)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.tight_layout()
    
    # Allow for zoom and pan in image
    zoom_factory(ax)
    ph = panhandler(fig, button=2)

    # Handles key presses to Accept, dismiss, or click points by hand
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.draw()
    plt.show(block=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.rcParams['toolbar'] = 'toolmanager'

    for var_to_delete in ['events', 'count', 'scat', 'fig_3d', 'ax_3d', 'objp_confirmed_notok']:
        if var_to_delete in globals():
            del globals()[var_to_delete]

    if 'imgp_confirmed' in globals() and 'objp_confirmed' in globals():
        return imgp_confirmed, objp_confirmed
    elif 'imgp_confirmed' in globals() and not 'objp_confirmed' in globals():
        return imgp_confirmed
    else:
        return


def extract_frames(video_path, extract_path, extract_every_N_sec=1, overwrite_extraction=False):
    '''
    Extract frames from video 
    if has not been done yet or if overwrite==True
    
    INPUT:
    - video_path: path to video whose frames need to be extracted
    - extract_every_N_sec: extract one frame every N seconds (can be <1)
    - overwrite_extraction: if True, overwrite even if frames have already been extracted
    
    OUTPUT:
    - extracted frames in folder
    '''
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    # Ouvre la vidéo pour lire les frames
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Nombre total de frames dans la vidéo
    
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"Extracting frames from video {os.path.basename(video_path)}...")

    frame_count = 0
    saved_count = 0
    
    # Barre de progression avec tqdm pour montrer l'avancement de l'extraction
    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sauvegarde toutes les frames selon l'intervalle défini
            if frame_count % (fps*extract_every_N_sec) == 0:
                frame_filename = os.path.join(extract_path, f"frame_{saved_count:04d}.png")
                cv2.imwrite(frame_filename, frame)
                saved_count += 1

            frame_count += 1
            pbar.update(1)

    # Libération des ressources et fermeture de la vidéo
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"Extraction completed. {saved_count} images saved into {extract_path}.")

def trc_write(object_coords_3d, trc_path):
    '''
    Make Opensim compatible trc file from a dataframe with 3D coordinates

    INPUT:
    - object_coords_3d: list of 3D point lists
    - trc_path: output path of the trc file

    OUTPUT:
    - trc file with 2 frames of the same 3D points
    
    '''

    #Header
    DataRate = CameraRate = OrigDataRate = 1
    NumFrames = 2
    NumMarkers = len(object_coords_3d)
    keypoints_names = np.arange(NumMarkers)
    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + os.path.basename(trc_path), 
            'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames', 
            '\t'.join(map(str,[DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, NumFrames])),
            'Frame#\tTime\t' + '\t\t\t'.join(str(k) for k in keypoints_names) + '\t\t',
            '\t\t'+'\t'.join([f'X{i+1}\tY{i+1}\tZ{i+1}' for i in range(NumMarkers)])]
    
    # Zup to Yup coordinate system
    object_coords_3d = pd.DataFrame([np.array(object_coords_3d).flatten(), np.array(object_coords_3d).flatten()])
    object_coords_3d = zup2yup(object_coords_3d)
    
    #Add Frame# and Time columns
    object_coords_3d.index = np.array(range(0, NumFrames)) + 1
    object_coords_3d.insert(0, 't', object_coords_3d.index / DataRate)

    #Write file
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line+'\n') for line in header_trc]
        object_coords_3d.to_csv(trc_o, sep='\t', index=True, header=None, lineterminator='\n')

    return trc_path


def toml_write(calib_path, C, S, D, K, R, T):
    '''
    Writes calibration parameters to a .toml file

    INPUTS:
    - calib_path: path to the output calibration file: string
    - C: camera name: list of strings
    - S: image size: list of list of floats
    - D: distorsion: list of arrays of floats
    - K: intrinsic parameters: list of 3x3 arrays of floats
    - R: extrinsic rotation: list of arrays of floats (Rodrigues)
    - T: extrinsic translation: list of arrays of floats

    OUTPUTS:
    - a .toml file cameras calibrations
    '''

    with open(os.path.join(calib_path), 'w+') as cal_f:
        for c in range(len(C)):
            cam=f'[{C[c]}]\n'
            name = f'name = "{C[c]}"\n'
            size = f'size = [ {S[c][0]}, {S[c][1]}]\n' 
            mat = f'matrix = [ [ {K[c][0,0]}, 0.0, {K[c][0,2]}], [ 0.0, {K[c][1,1]}, {K[c][1,2]}], [ 0.0, 0.0, 1.0]]\n'
            dist = f'distortions = [ {D[c][0]}, {D[c][1]}, {D[c][2]}, {D[c][3]}]\n' 
            rot = f'rotation = [ {R[c][0]}, {R[c][1]}, {R[c][2]}]\n'
            tran = f'translation = [ {T[c][0]}, {T[c][1]}, {T[c][2]}]\n'
            fish = f'fisheye = false\n\n'
            cal_f.write(cam + name + size + mat + dist + rot + tran + fish)
        meta = '[metadata]\nadjusted = false\nerror = 0.0\n'
        cal_f.write(meta)


def recap_calibrate(ret, calib_path, calib_full_type):
    '''
    Print a log message giving calibration results. Also stored in User/logs.txt.

    OUTPUT:
    - Message in console
    '''
    
    calib = toml.load(calib_path)
    
    ret_m, ret_px = [], []
    for c, cam in enumerate(calib.keys()):
        if cam != 'metadata':
            f_px = calib[cam]['matrix'][0][0]
            Dm = euclidean_distance(calib[cam]['translation'], [0,0,0])
            if calib_full_type in ['convert_qualisys', 'convert_vicon','convert_opencap', 'convert_biocv']:
                ret_m.append( np.around(ret[c], decimals=3) )
                ret_px.append( np.around(ret[c] / (Dm*1000) * f_px, decimals=3) )
            elif calib_full_type=='calculate':
                ret_px.append( np.around(ret[c], decimals=3) )
                ret_m.append( np.around(ret[c]*Dm*1000 / f_px, decimals=3) )
                
    logging.info(f'\n--> Residual (RMS) calibration errors for each camera are respectively {ret_px} px, \nwhich corresponds to {ret_m} mm.\n')
    logging.info(f'Calibration file is stored at {calib_path}.')

def calibrate_cams_all(config_dict):
    '''
    Either converts a preexisting calibration file, 
    or calculates calibration from scratch (from a board or from points).
    Stores calibration in a .toml file and prints a recap.
    
    INPUTS:
    - config_dict: configuration dictionary containing project details and calibration parameters

    OUTPUT:
    - a .toml camera calibration file
    '''

    # Extract project directory from the configuration dictionary
    project_dir = config_dict.get('project').get('project_dir')
    # Locate the calibration directory (searches for a directory containing 'Calib' or 'calib')
    calib_dir = [os.path.join(project_dir, c) for c in os.listdir(project_dir) if ('Calib' in c or 'calib' in c)][0]
    # Determine the type of calibration (either 'convert' or 'calculate')
    calib_type = config_dict.get('calibration').get('calibration_type')

    # Handle 'convert' type calibration
    if calib_type == 'convert':
        convert_filetype = config_dict.get('calibration').get('convert').get('convert_from')
        try:
            # Identify file extensions and specific parameters for each supported file type
            if convert_filetype == 'qualisys':
                convert_ext = '.qca.txt'
                file_to_convert_path = glob.glob(os.path.join(calib_dir, f'*{convert_ext}*'))[0]
                binning_factor = config_dict.get('calibration').get('convert').get('qualisys').get('binning_factor')
            elif convert_filetype == 'optitrack':
                file_to_convert_path = ['']  # Optitrack has no specific file
                binning_factor = 1
            elif convert_filetype == 'vicon':
                convert_ext = '.xcp'
                file_to_convert_path = glob.glob(os.path.join(calib_dir, f'*{convert_ext}'))[0]
                binning_factor = 1
            elif convert_filetype == 'opencap':  # OpenCap calibration files
                convert_ext = '.pickle'
                file_to_convert_path = sorted(glob.glob(os.path.join(calib_dir, f'*{convert_ext}')))
                binning_factor = 1
            elif convert_filetype == 'easymocap':  # EasyMocap calibration files
                convert_ext = '.yml'
                file_to_convert_path = sorted(glob.glob(os.path.join(calib_dir, f'*{convert_ext}')))
                binning_factor = 1
            elif convert_filetype == 'biocv':  # BioCV calibration files
                convert_ext = '.calib'
                file_to_convert_path = sorted(glob.glob(os.path.join(calib_dir, f'*{convert_ext}')))
                binning_factor = 1
            elif convert_filetype in ('anipose', 'freemocap'):  # Skip conversion for these file types
                logging.info(f'\n--> No conversion needed for AniPose or FreeMocap. Calibration skipped.\n')
                return
            else:
                raise NameError(f'Calibration conversion from {convert_filetype} is not supported.') from None

            # Ensure the file exists
            assert file_to_convert_path != []
        except:
            raise NameError(f'No file with {convert_ext} extension found in {calib_dir}.')

        # Prepare output file path and function arguments
        calib_output_path = os.path.join(calib_dir, f'Calib_{convert_filetype}.toml')
        calib_full_type = '_'.join([calib_type, convert_filetype])
        args_calib_fun = [file_to_convert_path, binning_factor]

    # Handle 'calculate' type calibration
    elif calib_type == 'calculate':
        intrinsics_config_dict = config_dict.get('calibration').get('calculate').get('intrinsics')
        intrinsics_method = intrinsics_config_dict.get('intrinsics_method')

        extrinsics_config_dict = config_dict.get('calibration').get('calculate').get('extrinsics')
        extrinsics_method = extrinsics_config_dict.get('extrinsics_method')

        # Prepare output file path and function arguments
        calib_output_path = os.path.join(calib_dir, f'Calib_{extrinsics_method}.toml')
        calib_full_type = calib_type
        args_calib_fun = [calib_dir, intrinsics_config_dict, extrinsics_config_dict]

    else:
        logging.info('Wrong calibration_type in Config.toml')
        return

    # Map calibration functions to their respective keys
    calib_mapping = {
        'convert_qualisys': calib_qca_fun,
        'convert_optitrack': calib_optitrack_fun,
        'convert_vicon': calib_vicon_fun,
        'convert_opencap': calib_opencap_fun,
        'convert_easymocap': calib_easymocap_fun,
        'convert_biocv': calib_biocv_fun,
        'calculate': calib_calc_fun,
    }
    # Retrieve the appropriate calibration function
    calib_fun = calib_mapping[calib_full_type]

    # Perform calibration using the selected function
    ret, C, S, D, K, R, T = calib_fun(*args_calib_fun)

    # Write the calibration results to a .toml file
    toml_write(calib_output_path, C, S, D, K, R, T)

    # Print a summary of the calibration process
    recap_calibrate(ret, calib_output_path, calib_full_type)

