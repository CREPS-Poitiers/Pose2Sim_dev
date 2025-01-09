#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## POSE2SIM                                                              ##
###########################################################################

This repository offers a way to perform markerless kinematics, and gives an 
example workflow from an Openpose input to an OpenSim result.

This script is the main entry point for the Pose2Sim pipeline. 
It orchestrates the different steps required for markerless motion capture and biomechanical analysis
It offers tools for: 
    - directory and video classification
    - camera calibration
    - synchronization
    - pose estimation (human via rtmlib or custom object via DLC)
    - person tracking
    - robust triangulation
    - filtering
    - marker augmentation
    - result generation
    - OpenSim scaling and inverse kinematics

It has been tested on Windows, Linux and MacOS, and works for any Python version >= 3.9

Installation: 
# Open Anaconda prompt. Type:
# - conda create -n Pose2Sim python=3.9
# - conda activate Pose2Sim
# - conda install -c opensim-org opensim -y
# - pip install Pose2Sim

Usage: 
from Pose2Sim import Pose2Sim
Pose2Sim.classification()
Pose2Sim.calibration()
Pose2Sim.poseEstimation()
Pose2Sim.synchronization()
Pose2Sim.personAssociation()
Pose2Sim.triangulation()
Pose2Sim.filtering()
Pose2Sim.genResults
Pose2Sim.markerAugmentation() (optional)

# Then run OpenSim (see Readme.md)
'''


## INIT
import toml
import os
import time
from copy import deepcopy
import logging, logging.handlers
from datetime import datetime
import numpy as np
import itertools
import copy


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## FUNCTIONS
def setup_logging(session_dir):
    '''
    Sets up a logging system for the current session.
    
    - Creates a log file in the specified `session_dir`, named `logs.txt`.
    - Uses a timed rotating file handler to create a new log file every 7 days.
    - Streams logs to both the log file and the console.
    
    Parameters:
        session_dir (str): Directory where the log file will be saved.
    '''
    logging.basicConfig(
        format='%(message)s', 
        level=logging.INFO,
        handlers=[
            logging.handlers.TimedRotatingFileHandler(
                os.path.join(session_dir, 'logs.txt'), when='D', interval=7
            ),
            logging.StreamHandler()
        ]
    )

    
def recursive_update(dict_to_update, dict_with_new_values):
    '''
    Updates a dictionary with values from another dictionary, preserving nested structures.
    
    - Unlike the standard `.update()` method, this function recursively updates keys within 
      nested dictionaries without overwriting the entire structure.
    - If a key exists in both dictionaries and the corresponding value is another dictionary, 
      it performs a recursive update on that nested dictionary.
    - If a key exists only in `dict_with_new_values` or the values are not dictionaries, 
      it directly adds or replaces the key-value pair in `dict_to_update`.

    Parameters:
        dict_to_update (dict): The dictionary to be updated.
        dict_with_new_values (dict): The dictionary containing new values to update.

    Returns:
        dict: The updated dictionary with merged values.

    Example:
        dict_to_update = {'key': {'key_1': 'val_1', 'key_2': 'val_2'}}
        dict_with_new_values = {'key': {'key_1': 'val_1_new'}}
        Result:
        {'key': {'key_1': 'val_1_new', 'key_2': 'val_2'}}
    '''
    for key, value in dict_with_new_values.items():
        if key in dict_to_update and isinstance(value, dict) and isinstance(dict_to_update[key], dict):
            # Recursively update nested dictionaries
            dict_to_update[key] = recursive_update(dict_to_update[key], value)
        else:
            # Update or add new key-value pairs
            dict_to_update[key] = value

    return dict_to_update

def determine_level(config_dir):
    '''
    Determines the hierarchy level of the configuration directory.
    
    - **Level 1:** The function is called at the trial folder level, where individual experiments are stored.
    - **Level 2:** The function is called at the root folder level, containing multiple trial folders.
    
    Process:
    - Iterates through the directory structure starting from `config_dir`.
    - Checks for the presence of `Config.toml` files in the directory tree.
    - Calculates the depth of directories with `Config.toml` files.
    - Raises an error if no `Config.toml` files are found.

    Parameters:
        config_dir (str): The directory path to evaluate.

    Returns:
        int: The calculated level (1 or 2) based on the directory structure.

    Raises:
        FileNotFoundError: If no `Config.toml` files are found in the given directory tree.
    '''
    len_paths = [len(root.split(os.sep)) for root, dirs, files in os.walk(config_dir) if 'Config.toml' in files]
    if len_paths == []:
        raise FileNotFoundError('You need a Config.toml file in each trial or root folder.')
    level = max(len_paths) - min(len_paths) + 1
    return level


def read_config_files(config):
    '''
    Reads configuration files from the specified directory or dictionary.
    
    - Supports two levels:
      - **Trial level (Level 1):** Reads a single trial's configuration.
      - **Root level (Level 2):** Reads configurations for multiple trials within a root folder.
    - Combines parameters from the root-level configuration (`Config.toml`) and trial-level configurations.
    - Ensures consistency by merging parameters recursively using `recursive_update`.

    Parameters:
        config (str, dict, or None): 
            - Path to the configuration directory.
            - A dictionary representing the configuration.
            - `None` to use the current directory.

    Returns:
        tuple: 
            - `level` (int): The determined level (1 or 2).
            - `config_dicts` (list of dict): List of dictionaries containing the merged configuration parameters.

    Raises:
        ValueError: If the project directory is not specified in the configuration dictionary.
    '''
    if type(config) == dict:
        # Directly use the provided dictionary for Level 2.
        level = 2
        config_dicts = [config]
        if config_dicts[0].get('project').get('project_dir') is None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_PROJECT_DIRECTORY>"})')
    else:
        # Use the provided directory or the current one if `config` is None.
        config_dir = ['.' if config is None else config][0]
        level = determine_level(config_dir)
        
        # If called at the trial level
        if level == 1:
            try:
                # Handle batch processing (multiple trials)
                session_config_dict = toml.load(os.path.join(config_dir, '..', 'Config.toml'))
                trial_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
                session_config_dict = recursive_update(session_config_dict, trial_config_dict)
            except:
                # Handle single trial
                session_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
            session_config_dict.get("project").update({"project_dir": config_dir})
            config_dicts = [session_config_dict]
        
        # If called at the root level
        if level == 2:
            session_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
            config_dicts = []
            # Iterate over trials and merge their configurations with the root configuration
            for root, dirs, files in os.walk(config_dir):
                if 'Config.toml' in files and root != config_dir:
                    trial_config_dict = toml.load(os.path.join(root, files[0]))
                    # Use a deep copy to avoid modifying the root configuration across iterations
                    temp_dict = deepcopy(session_config_dict)
                    temp_dict = recursive_update(temp_dict, trial_config_dict)
                    temp_dict.get("project").update({"project_dir": os.path.join(config_dir, os.path.relpath(root))})
                    # Exclude trials marked for exclusion in the configuration
                    if os.path.basename(root) not in temp_dict.get("project").get('exclude_from_batch'):
                        config_dicts.append(temp_dict)

    return level, config_dicts


def classification(config=None):
    '''
    Classifies videos from an acquisition session into the appropriate directory structure.
    
    **Objective:**
    - Organize raw video files into a structured folder named `_traitement`, 
      created in the parent directory of the active folder.
    - Ensures proper folder and file naming conventions are followed for processing.

    **File Organization Requirements:**
    - Place all raw videos in the active directory alongside a `Config.toml` file.
    - For retaining the same intrinsic calibration:
        - Include the `intrinsics` folder and the `Calib.toml` file.
    - For retaining both intrinsic and extrinsic calibration:
        - Include a `calibration` folder containing subfolders `intrinsics` and `extrinsics`, 
          as well as the `Calib.toml` file.

    Parameters:
        config (dict or None): 
            - A dictionary containing configuration parameters.
            - If `None`, the function uses the current directory as the configuration directory.
    
    Workflow:
    - Reads configuration files using `read_config_files`, determining whether the function 
      is called at the root or trial level.
    - Sets up logging to track the classification process.
    - Calls `classification_run` to perform the actual classification task, 
      passing the prepared configuration dictionary.
    
    '''
    
    # Import the classification function
    from Pose2Sim.classification import classification_run

    # Determine the level (root:2, trial:1) and read configuration files
    level, config_dicts = read_config_files(config)
    config_dict = config_dicts[0]

    # Validate the configuration dictionary
    if isinstance(config, dict):
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') is None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                              config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging for the session
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)    
    currentDateAndTime = datetime.now()

    # Log the start of the classification process
    logging.info("\n---------------------------------------------------------------------")
    logging.info("Folders classification")
    logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
    logging.info("---------------------------------------------------------------------\n")
    
    # Measure the execution time of the classification
    start = time.time()
    
    # Run the classification process
    classification_run(config_dict)
    
    # Log the time taken for classification
    end = time.time()
    logging.info(f'\nClassification took {end-start:.2f} s.\n')

def calibration(config=None):
    '''
    Performs camera calibration using various methods, including checkerboards, Charuco boards, or Qualisys files.
    
    **Objective:**
    - Calibrate cameras either intrinsically (internal parameters like focal length) or extrinsically 
      (relative positions and orientations between cameras).
    - Offers a new intrinsic calibration method using Charuco boards, which provides higher precision compared 
      to traditional checkerboards.

    **Configuration:**
    - The calibration method is determined based on the settings provided in the `Config.toml` file.
    - Supports the following calibration types:
        - Intrinsic calibration
        - Extrinsic calibration
        - Both intrinsic and extrinsic calibration

    Parameters:
        config (dict or None): 
            - A dictionary containing configuration parameters.
            - Path to the configuration directory for a trial, participant, or session.
            - If `None`, the current directory is used as the configuration directory.

    Workflow:
    - Reads the configuration files using `read_config_files` and determines the directory level (root or trial).
    - Validates the presence of calibration directories in the session folder.
    - Sets up logging to track the calibration process.
    - Calls `calibrate_cams_all` to execute the calibration based on the provided configuration.

    Notes:
    - The Charuco board method is highlighted as a recent addition for intrinsic calibration, offering improved accuracy.
    '''
    # Import the calibration function (includes Charuco board support)
    from Pose2Sim.calibration_dev import calibrate_cams_all

    # Determine the level and read configuration files
    level, config_dicts = read_config_files(config)
    config_dict = config_dicts[0]

    # Define the session directory based on the level
    try:
        session_dir = os.path.realpath([os.getcwd() if level == 2 else os.path.join(os.getcwd(), '..')][0])
        [os.path.join(session_dir, c) for c in os.listdir(session_dir) if 'calib' in c.lower() and not c.lower().endswith('.py')][0]
    except:
        # Fallback to the current working directory if no calibration directory is found
        session_dir = os.path.realpath(os.getcwd())
    config_dict.get("project").update({"project_dir": session_dir})

    # Set up logging for the session
    setup_logging(session_dir)  
    currentDateAndTime = datetime.now()

    # Identify the calibration directory and log the process
    calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) 
                 if os.path.isdir(os.path.join(session_dir, c)) and 'calib' in c.lower()][0]
    logging.info("\n---------------------------------------------------------------------")
    logging.info("Camera calibration")
    logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
    logging.info(f"Calibration directory: {calib_dir}")
    logging.info("---------------------------------------------------------------------\n")
    
    # Measure the execution time of the calibration process
    start = time.time()
    
    # Perform the calibration
    calibrate_cams_all(config_dict)
    
    # Log the time taken for calibration
    end = time.time()
    logging.info(f'\nCalibration took {end-start:.2f} s.\n')


def poseEstimation(config=None):
    '''
    Estimates human poses using RTMLib and optionally DeepLabCut for custom object detection.
    
    **Objective:**
    - Perform pose estimation for individuals in video sequences, generating 2D keypoints for further processing.
    - Includes recent developments to support DeepLabCut for custom object detection tasks, in addition to RTMLib.

    **Configuration:**
    - The function supports flexible configuration inputs:
        - A dictionary containing configuration parameters.
        - A path to the configuration directory for a trial, participant, or session.
        - `None`, which defaults to the current working directory.

    **Workflow:**
    1. Reads the configuration files and determines whether the function is called at the root or trial level.
    2. Validates the configuration dictionary to ensure the `project_dir` key is correctly specified.
    3. Sets up a logging system for the session to track the pose estimation process.
    4. Batch processes all trials in the session, calling `rtm_estimator` for each trial.

    **Notes:**
    - The `frame_range` parameter can limit the process to a specific range of frames. If not specified, the function processes all frames.
    - RTMLib is used for human pose estimation, but the function is extensible to support other methods like DeepLabCut for detecting objects such as sports equipment.
    
    - **Synchronization Considerations:**
        - If camera synchronization is based on the vertical acceleration of a specific point (e.g., markerless motion tracking), 
          pose estimation must be completed **before synchronization** to extract the necessary acceleration data.
        - For other synchronization methods (e.g., manual sync, audio, or GPS-based sync), synchronization should be performed **before pose estimation**.

    Parameters:
        config (dict or None): 
            - A dictionary of configuration parameters.
            - Path to the configuration directory.
            - If `None`, the function assumes the current directory.

    Raises:
        ValueError: If the `project_dir` key is not specified in the configuration dictionary.
    '''
    from Pose2Sim.poseEstimation import rtm_estimator  # Import the pose estimation function

    # Determine the level and read configuration files
    level, config_dicts = read_config_files(config)

    # Validate the configuration dictionary
    if isinstance(config, dict):
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') is None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging for the session
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()  # Measure execution time for each trial
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if not frame_range else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        # Log information about the current trial
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Pose estimation for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")

        # Perform pose estimation using RTMLib
        rtm_estimator(config_dict)
        
        # Log the execution time for the trial
        end = time.time()
        elapsed = end - start
        logging.info(f'\nPose estimation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    
def synchronization(config=None):
    '''
    Synchronizes cameras if necessary, supporting various synchronization methods.
    
    **Objective:**
    - Align video streams from multiple cameras to ensure temporal consistency during analysis.
    - Supports different synchronization methods to accommodate various experimental setups.

    **Synchronization Methods:**
    1. **Vertical acceleration-based (default in Pose2Sim):**
       - Requires pose estimation to extract acceleration data for synchronization.
       - Ideal for scenarios where markerless motion tracking is used.
    2. **Manual synchronization:**
       - Users visually inspect videos and align each camera stream to a reference frame.
       - Useful for cases where automated synchronization methods are not applicable.
    3. **Audio-based synchronization:**
       - Relies on matching audio signals across camera recordings.
       - Not recommended for outdoor environments due to potential noise interference.
    4. **Qualisys synchronization file:**
       - Imports pre-synchronized data from Qualisys systems, bypassing manual or automated syncing.
    5. **GPS-based synchronization (in development):**
       - Uses timestamps from GPS data to align camera streams.
       - Promising for outdoor settings or large-scale experiments.

    **Configuration:**
    - The function supports flexible configuration inputs:
        - A dictionary containing configuration parameters.
        - A path to the configuration directory for a trial, participant, or session.
        - `None`, which defaults to the current working directory.

    Parameters:
        config (dict or None): 
            - A dictionary of configuration parameters.
            - Path to the configuration directory.
            - If `None`, the function assumes the current directory.

    Workflow:
    1. Reads configuration files and determines the level (root or trial).
    2. Validates the configuration dictionary to ensure the `project_dir` key is correctly specified.
    3. Sets up logging to track the synchronization process.
    4. Calls `synchronize_cams_all` to perform the synchronization for each trial.

    Raises:
        ValueError: If the `project_dir` key is not specified in the configuration dictionary.

    Notes:
    - Ensure that pose estimation is completed **before synchronization** for acceleration-based syncing.
    - Other methods (manual, audio, GPS) can be performed **before pose estimation** if needed.
    '''
    # Import the synchronization function
    from Pose2Sim.synchronization_dev import synchronize_cams_all

    # Determine the level and read configuration files
    level, config_dicts = read_config_files(config)

    # Validate the configuration dictionary
    if type(config) == dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') is None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging for the session
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)    

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()  # Measure execution time for each trial
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))

        # Log the synchronization details
        logging.info("\n---------------------------------------------------------------------")
        logging.info("Camera synchronization")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")
        
        # Perform synchronization
        synchronize_cams_all(level, config_dict)
    
        # Log the execution time for the synchronization
        end = time.time()
        elapsed = end - start
        logging.info(f'\nSynchronization took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

def personAssociation(config=None):
    '''
    Tracks one or multiple persons of interest across video streams.
    
    **Objective:**
    - Associates detected 2D keypoints to track specific individuals consistently across frames and views.
    - Enables robust tracking by leveraging calibration data and optimizing parameters for accuracy.

    **Configuration:**
    - The function supports the following input configurations:
        - A dictionary containing configuration parameters.
        - A path to the configuration directory for a trial, participant, or session.
        - `None`, which defaults to the current working directory.
    - Requires a valid calibration file for accurate person association.

    **Workflow:**
    1. Reads configuration files to determine the directory level (root or trial).
    2. Validates the presence of configuration dictionaries.
    3. Sets up a logging system for the session to monitor the process.
    4. Batch processes all trials in the session, iterating through combinations of parameters if enabled.
    5. Calls `track_2d_all` to perform person association for each trial.

    **Parameter Optimization:**
    - If `iterative_optimal_association` is enabled, the function tests multiple parameter combinations, including:
        - Likelihood thresholds for associating keypoints.
        - Reprojection error thresholds to filter outliers.
        - The keypoint used for tracking (e.g., head, hip).
        - Minimum number of cameras required for triangulation.
    - Logs all combinations tested and their respective results.

    **Notes:**
    - The function adjusts and resets parameters dynamically to avoid impacting subsequent trials.
    - Parameter optimization allows for fine-tuning the tracking process to maximize accuracy.

    Parameters:
        config (dict or None): 
            - A dictionary of configuration parameters.
            - Path to the configuration directory.
            - If `None`, the function assumes the current directory.

    Raises:
        ValueError: If no valid configuration files are found.

    **Dependencies:**
    - Relies on the function `track_2d_all` to perform person association for each configuration.

    '''
    from Pose2Sim.personAssociation_dev import track_2d_all

    # Determine the level and read configuration files
    level, config_dicts = read_config_files(config)

    if not config_dicts:
        raise ValueError("No valid configuration files found. Please provide a valid configuration.")

    # Set up logging for the session
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()  # Measure execution time
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        # Log details about the current trial
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Associating persons for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")

        # Retrieve parameters from the configuration
        iterative_optimal_association = config_dict.get('personAssociation', {}).get('iterative_optimal_association', False)
        likelihood_threshold_value = config_dict['personAssociation']['likelihood_threshold_association']
        reproj_error_threshold_value = config_dict['personAssociation']['single_person']['reproj_error_threshold_association']
        tracked_keypoint_value = config_dict['personAssociation']['single_person']['tracked_keypoint']
        min_cameras_value = config_dict['triangulation']['min_cameras_for_triangulation']

        # Adjust and generate parameter combinations if enabled
        if iterative_optimal_association:
            if not isinstance(likelihood_threshold_value, list):
                likelihood_threshold_value = [likelihood_threshold_value,likelihood_threshold_value, likelihood_threshold_value]
            if not isinstance(reproj_error_threshold_value, list):
                reproj_error_threshold_value = [reproj_error_threshold_value, reproj_error_threshold_value, reproj_error_threshold_value]
            if not isinstance(tracked_keypoint_value, list):
                tracked_keypoint_value = [tracked_keypoint_value]
            if not isinstance(min_cameras_value, list):
                min_cameras_value = [min_cameras_value, min_cameras_value, min_cameras_value]

            # Generate combinations for parameter testing
            likelihood_values = [round(x, 2) for x in np.arange(
                likelihood_threshold_value[0], 
                likelihood_threshold_value[2] + likelihood_threshold_value[1],  # Ajuste avec le pas
                likelihood_threshold_value[1]
            )]

            reproj_values = [int(x) for x in range(
                reproj_error_threshold_value[0],
                reproj_error_threshold_value[2] + reproj_error_threshold_value[1],  # Ajuste avec le pas
                reproj_error_threshold_value[1]
            )]

            min_cameras_values = [int(x) for x in range(
                min_cameras_value[0], 
                min_cameras_value[2] + min_cameras_value[1], 
                min_cameras_value[1]
            )]
            
            param_combinations = list(itertools.product(likelihood_values, reproj_values, tracked_keypoint_value, min_cameras_values))

        else:
            # Single execution without combinations
            param_combinations = [(likelihood_threshold_value, reproj_error_threshold_value, tracked_keypoint_value, min_cameras_value)]

        # Log parameter ranges and combinations
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Number of combinations to test: {len(param_combinations)}")
        logging.info(f"Likelihood threshold range: {likelihood_threshold_value}")
        logging.info(f"Reprojection error threshold range: {reproj_error_threshold_value}")
        logging.info(f"Tracked keypoints: {tracked_keypoint_value}")
        logging.info(f"Minimum cameras for triangulation range: {min_cameras_value}")
        logging.info("---------------------------------------------------------------------\n")

        # Perform tracking for each parameter combination
        for likelihood, reproj, tracked_keypoint, min_cameras in param_combinations:
            config_dict['personAssociation']['likelihood_threshold_association'] = likelihood
            config_dict['personAssociation']['single_person']['reproj_error_threshold_association'] = reproj
            config_dict['personAssociation']['single_person']['tracked_keypoint'] = tracked_keypoint
            config_dict['triangulation']['min_cameras_for_triangulation'] = min_cameras

            logging.info(f"Testing with likelihood_threshold_association={likelihood}, reproj_error_threshold_association={reproj}, tracked_keypoint={tracked_keypoint}, min_cameras_for_triangulation={min_cameras}")
            track_2d_all(config_dict)

        # Reset parameters to their original values
        config_dict['personAssociation']['likelihood_threshold_association'] = likelihood_threshold_value
        config_dict['personAssociation']['single_person']['reproj_error_threshold_association'] = reproj_error_threshold_value
        config_dict['personAssociation']['single_person']['tracked_keypoint'] = tracked_keypoint_value
        config_dict['triangulation']['min_cameras_for_triangulation'] = min_cameras_value

        # Log execution time for the trial
        end = time.time()
        elapsed = end - start
        logging.info(f'\nAssociating persons took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')


def triangulation(config=None):
    '''
    Performs robust triangulation of 2D points to calculate 3D coordinates.
    
    **Objective:**
    - Converts 2D keypoints detected in multiple camera views into 3D coordinates.
    - Utilizes camera calibration data to ensure geometric accuracy.
    - Provides parameter optimization options for enhanced robustness.

    **Configuration:**
    - The function supports flexible configuration inputs:
        - A dictionary containing configuration parameters.
        - A path to the configuration directory for a trial, participant, or session.
        - `None`, which defaults to the current working directory.
    - Requires calibration files to compute 3D points accurately.

    **Workflow:**
    1. Reads the configuration files and determines the directory level (root or trial).
    2. Validates the presence of configuration dictionaries.
    3. Sets up logging to monitor the triangulation process.
    4. Iterates over all trials in the session, testing parameter combinations if `iterative_optimal_triangulation` is enabled.
    5. Calls `triangulate_all` to perform the triangulation for each trial.

    **Parameter Optimization:**
    - If `iterative_optimal_triangulation` is enabled, the function tests various combinations of:
        - **Reprojection error thresholds:** Filters points based on the error between observed and projected 2D coordinates.
        - **Likelihood thresholds:** Ensures only points with high confidence are included.
        - **Minimum cameras:** Specifies the minimum number of cameras required for triangulating a point.
    - Logs parameter combinations and their results for debugging and performance tracking.

    **Notes:**
    - Default behavior runs triangulation with a single set of parameters unless optimization is enabled.
    - Parameter ranges are adjusted dynamically if scalar values are provided.

    Parameters:
        config (dict or None): 
            - A dictionary of configuration parameters.
            - Path to the configuration directory.
            - If `None`, the function assumes the current directory.

    Raises:
        ValueError: If no valid configuration files are found.

    **Dependencies:**
    - Calls `triangulate_all` to process the triangulation for each configuration.

    '''
    from Pose2Sim.triangulation_dev import triangulate_all

    # Determine the level and read configuration files
    level, config_dicts = read_config_files(config)

    if not config_dicts:
        raise ValueError("No valid configuration files found. Please provide a valid configuration.")

    # Set up logging for the session
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()  # Measure execution time for each trial
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        # Log details about the current trial
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Triangulation of 2D points for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")

        # Retrieve parameters from the configuration
        iterative_optimal_triangulation = config_dict.get('triangulation', {}).get('iterative_optimal_triangulation', False)
        reproj_error_threshold_value = config_dict['triangulation']['reproj_error_threshold_triangulation']
        likelihood_threshold_value = config_dict['triangulation']['likelihood_threshold_triangulation']
        min_cameras_value = config_dict['triangulation']['min_cameras_for_triangulation']

        # Adjust and generate parameter combinations if enabled
        if iterative_optimal_triangulation:
            # Ensure values are converted to ranges if not already lists
            if not isinstance(reproj_error_threshold_value, list):
                reproj_error_threshold_value = [reproj_error_threshold_value,reproj_error_threshold_value, reproj_error_threshold_value]
            if not isinstance(likelihood_threshold_value, list):
                likelihood_threshold_value = [likelihood_threshold_value,likelihood_threshold_value, likelihood_threshold_value]
            if not isinstance(min_cameras_value, list):
                min_cameras_value = [min_cameras_value, min_cameras_value, min_cameras_value]

            reproj_values = [int(x) for x in range(
                reproj_error_threshold_value[0], 
                reproj_error_threshold_value[2] + reproj_error_threshold_value[1], 
                reproj_error_threshold_value[1]
            )]
            likelihood_values = [round(x, 2) for x in np.arange(
                likelihood_threshold_value[0], 
                likelihood_threshold_value[2] + likelihood_threshold_value[1], 
                likelihood_threshold_value[1]
            )]
            min_cameras_values = [int(x) for x in range(
                min_cameras_value[0], 
                min_cameras_value[2] + min_cameras_value[1], 
                min_cameras_value[1]
            )]
            
            param_combinations = list(itertools.product(reproj_values, likelihood_values, min_cameras_values))

        else:
            # Single execution without combinations
            param_combinations = [(reproj_error_threshold_value, likelihood_threshold_value, min_cameras_value)]

        # Log parameter ranges and combinations
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Number of combinations to test: {len(param_combinations)}")
        logging.info(f"Reprojection error threshold range: {reproj_error_threshold_value}")
        logging.info(f"Likelihood threshold range: {likelihood_threshold_value}")
        logging.info(f"Minimum cameras for triangulation range: {min_cameras_value}")
        logging.info("---------------------------------------------------------------------\n")

        # Perform triangulation for each parameter combination
        for reproj, likelihood, min_cameras in param_combinations:
            config_dict['triangulation']['reproj_error_threshold_triangulation'] = reproj
            config_dict['triangulation']['likelihood_threshold_triangulation'] = likelihood
            config_dict['triangulation']['min_cameras_for_triangulation'] = min_cameras

            logging.info(f"Testing with reproj_error_threshold_triangulation={reproj}, likelihood_threshold_triangulation={likelihood}, min_cameras_for_triangulation={min_cameras}")
            triangulate_all(config_dict)

        # Reset parameters to their original values
        config_dict['triangulation']['reproj_error_threshold_triangulation'] = reproj_error_threshold_value
        config_dict['triangulation']['likelihood_threshold_triangulation'] = likelihood_threshold_value
        config_dict['triangulation']['min_cameras_for_triangulation'] = min_cameras_value

        # Log execution time for the trial
        end = time.time()
        elapsed = end - start
        logging.info(f'\nTriangulation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

def filtering(config=None):
    '''
    Filters 3D coordinate data stored in TRC (Track Row Coordinate) files to improve data quality.
    
    **Objective:**
    - Applies filtering techniques to smooth and clean 3D marker data, reducing noise or artifacts from the motion capture process.
    - Ensures the filtered data is ready for subsequent analyses, such as marker augmentation or biomechanical modeling.

    **Configuration:**
    - The function supports flexible configuration inputs:
        - A dictionary containing configuration parameters.
        - A path to the configuration directory for a trial, participant, or session.
        - `None`, which defaults to the current working directory.

    **Workflow:**
    1. Reads the configuration files to determine the directory level (root or trial).
    2. Validates the configuration dictionary to ensure the `project_dir` key is specified.
    3. Sets up a logging system to track the filtering process.
    4. Batch processes all trials in the session by calling `filter_all` for each trial.

    **Notes:**
    - Filtering techniques are typically applied to reduce high-frequency noise or interpolate missing data points.
    - The function logs key details, including the project directory and frame range being processed.

    Parameters:
        config (dict or None): 
            - A dictionary of configuration parameters.
            - Path to the configuration directory.
            - If `None`, the function assumes the current directory.

    Raises:
        ValueError: If the `project_dir` key is not specified in the configuration dictionary.

    **Dependencies:**
    - Relies on the `filter_all` function to perform the filtering on TRC files.
    '''
    from Pose2Sim.filtering import filter_all  # Import the filtering function

    # Determine the level and read configuration files
    level, config_dicts = read_config_files(config)

    # Validate the configuration dictionary
    if type(config) == dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') is None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging for the session
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)

    # Batch process all trials
    for config_dict in config_dicts:
        currentDateAndTime = datetime.now()  # Capture the current timestamp
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        # Log details about the current trial
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Filtering 3D coordinates for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}\n")
        logging.info("---------------------------------------------------------------------\n")

        # Perform filtering using the filter_all function
        filter_all(config_dict)

        # Log the completion of filtering
        logging.info('\n')


def genResults(config=None):
    """
    Generates results by creating a sports court and combining C3D files into a single output file.

    **Objective:**
    - Automates the generation of results by creating a virtual sports court and consolidating motion capture data 
      (stored in C3D files) into a unified output for further analysis.
    - Handles both the visualization of the experimental setup (e.g., sports court) and the integration of recorded data.

    **Workflow:**
    1. Reads the configuration files to determine the directory level (root or trial).
    2. Validates the configuration dictionary to ensure the `project_dir` key is specified.
    3. Sets up logging to monitor the result generation process.
    4. For each trial:
        - Creates a sports court based on the trial's configuration.
        - Combines and rotates C3D files into a single file.
        - Logs the status and handles errors gracefully.

    **Steps:**
    - **Step 1: Creating Court**:
        - Uses the `create_court` function to visualize the experimental environment or field layout based on the configuration.
    - **Step 2: Combining and Rotating C3D Files**:
        - Merges multiple C3D files into a single output file.
        - Rotates the C3D data if specified in the configuration, ensuring proper alignment of the 3D data.

    **Configuration:**
    - The function supports flexible configuration inputs:
        - A dictionary containing configuration parameters.
        - A path to the configuration directory for a trial, participant, or session.
        - `None`, which defaults to the current working directory.

    Parameters:
        config (dict or None): 
            - A dictionary of configuration parameters.
            - Path to the configuration directory.
            - If `None`, the function assumes the current directory.

    **Error Handling:**
    - Logs detailed error messages if issues arise during court creation or C3D file processing.
    - Continues processing the next trial even if an error occurs for a specific sequence.

    Raises:
        ValueError: If the `project_dir` key is not specified in the configuration dictionary.

    **Dependencies:**
    - Relies on `create_court` to set up the sports court visualization.
    - Relies on `combine_and_rotate_c3d` to consolidate and adjust C3D files.

    **Example Output:**
    - A generated sports court visualization.
    - A single consolidated C3D file for each trial.

    """
    from Pose2Sim.Utilities.create_court import create_court  # Import court creation utility
    from Pose2Sim.Utilities.combine_c3d import combine_and_rotate_c3d  # Import C3D file utility

    # Determine the level and read configuration files
    level, config_dicts = read_config_files(config)

    # Validate the configuration dictionary
    if type(config) == dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') is None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging for the session
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)

    # Batch process all trials
    for config_dict in config_dicts:
        currentDateAndTime = datetime.now()  # Capture the current timestamp
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        # Log details about the current trial
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Starting genResults for sequence: {seq_name}")
        logging.info(f"Processing {frames}.")
        logging.info(f"Timestamp: {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")

        try:
            # Create court
            logging.info(f"Step 1: Creating court for {seq_name}.")
            create_court(config_dict)
            logging.info(f"Court created successfully for {seq_name}.")

            # Combine and rotate C3D files
            logging.info(f"Step 2: Combining and rotating C3D files for {seq_name}.")
            combine_and_rotate_c3d(config_dict)
            logging.info(f"C3D files combined and rotated successfully for {seq_name}.")

        except Exception as e:
            # Log error details and proceed with the next trial
            logging.error(f"Error during genResults for {seq_name}: {e}", exc_info=True)
            continue

        # Log completion of result generation for the trial
        logging.info(f"genResults completed for sequence: {seq_name}\n")
        logging.info("---------------------------------------------------------------------\n")


def markerAugmentation(config=None):
    '''
    Augments 3D TRC (Track Row Coordinate) data by estimating the positions of additional markers.

    **Objective:**
    - Enhances the motion capture dataset by interpolating and estimating the positions of 43 additional virtual markers.
    - Improves the completeness and usability of biomechanical data for further analysis, such as inverse kinematics or detailed movement studies.

    **Workflow:**
    1. Reads configuration files to determine the directory level (root or trial).
    2. Validates the configuration dictionary to ensure the `project_dir` key is specified.
    3. Sets up logging to monitor the augmentation process.
    4. Processes all trials in the session using the `augmentTRC` function to estimate additional marker positions.

    **Notes:**
    - The function logs detailed information, including the trial name, frame range, and elapsed time for the augmentation process.
    - Assumes the TRC files contain the necessary baseline markers to compute the positions of the virtual markers.

    **Configuration:**
    - The function supports flexible configuration inputs:
        - A dictionary containing configuration parameters.
        - A path to the configuration directory for a trial, participant, or session.
        - `None`, which defaults to the current working directory.

    Parameters:
        config (dict or None): 
            - A dictionary of configuration parameters.
            - Path to the configuration directory.
            - If `None`, the function assumes the current directory.

    Raises:
        ValueError: If the `project_dir` key is not specified in the configuration dictionary.

    **Dependencies:**
    - Relies on the `augmentTRC` function to compute and add virtual markers to the TRC files.

    '''
    from Pose2Sim.markerAugmentation import augmentTRC  # Import the marker augmentation function
    level, config_dicts = read_config_files(config)

    # Validate the configuration dictionary
    if type(config) == dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') is None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging for the session
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()  # Measure execution time for each trial
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        # Log details about the current trial
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Augmentation process for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")

        # Perform marker augmentation using augmentTRC
        augmentTRC(config_dict)

        # Log completion and execution time for the trial
        end = time.time()
        elapsed = end - start 
        logging.info(f'\nMarker augmentation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')



def opensimProcessing(config=None):
    '''
    Uses OpenSim to run scaling based on a static trc pose
    and inverse kinematics based on a trc motion file.
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''
    
    raise NotImplementedError('This has not been implemented yet. \nPlease see README.md for further explanation')
    
    # # TODO
    # from Pose2Sim.opensimProcessing import opensim_processing_all
    
    # # Determine the level at which the function is called (root:2, trial:1)
    # level, config_dicts = read_config_files(config)

    # if type(config)==dict:
    #     config_dict = config_dicts[0]
    #     if config_dict.get('project').get('project_dir') == None:
    #         raise ValueError('Please specify the project directory in config_dict:\n \
    #                          config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # # Set up logging
    # session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    # setup_logging(session_dir)    

    # # Batch process all trials
    # for config_dict in config_dicts:
    #     currentDateAndTime = datetime.now()
    #     start = time.time()
    #     project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
    #     seq_name = os.path.basename(project_dir)
    #     frame_range = config_dict.get('project').get('frame_range')
    #     frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

    #     logging.info("\n---------------------------------------------------------------------")
    #     # if static_file in project_dir: 
    #     #     logging.info(f"Scaling model with <STATIC TRC FILE>.")
    #     # else:
    #     #     logging.info(f"Running inverse kinematics <MOTION TRC FILE>.")
    #     logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
    #     logging.info(f"OpenSim output directory: {project_dir}")
    #     logging.info("---------------------------------------------------------------------\n")
   
    #     opensim_processing_all(config_dict)
    
    #     end = time.time()
    #     elapsed = end-start 
    #     # if static_file in project_dir: 
    #     #     logging.info(f'Model scaling took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')
    #     # else:
    #     #     logging.info(f'Inverse kinematics took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')


def runAll(config=None, do_classification=True, do_calibration=True, do_poseEstimation=True, do_synchronization=True, do_personAssociation=True, do_triangulation=True, do_filtering=True, do_markerAugmentation=False, do_opensimProcessing=False):
    '''
    Run all functions at once. Beware that Synchronization, personAssociation, and markerAugmentation are not always necessary, 
    and may even lead to worse results. Think carefully before running all.
    '''


    # Set up logging
    level, config_dicts = read_config_files(config)
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)  

    currentDateAndTime = datetime.now()
    start = time.time()

    logging.info("\n\n=====================================================================")
    logging.info(f"RUNNING ALL.")
    logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
    logging.info(f"Project directory: {session_dir}\n")
    logging.info("=====================================================================")

    if do_classification:
        logging.info("\n\n=====================================================================")
        logging.info('Running classification...')
        logging.info("=====================================================================")
        calibration(config)
    else: 
        logging.info("\n\n=====================================================================")
        logging.info('Skipping classification.')
        logging.info("=====================================================================")
        
    if do_calibration:
        logging.info("\n\n=====================================================================")
        logging.info('Running calibration...')
        logging.info("=====================================================================")
        calibration(config)
    else: 
        logging.info("\n\n=====================================================================")
        logging.info('Skipping calibration.')
        logging.info("=====================================================================")

    if do_synchronization:
        logging.info("\n\n=====================================================================")
        logging.info('Running synchronization...')
        logging.info("=====================================================================")
        synchronization(config)
    else:
        logging.info("\n\n=====================================================================")
        logging.info('Skipping synchronization.')
        logging.info("=====================================================================")
        
    if do_poseEstimation:
        logging.info("\n\n=====================================================================")
        logging.info('Running pose estimation...')
        logging.info("=====================================================================")
        poseEstimation(config)
    else:
        logging.info("\n\n=====================================================================")
        logging.info('Skipping pose estimation.')
        logging.info("=====================================================================")

    if do_personAssociation:
        logging.info("\n\n=====================================================================")
        logging.info('Running person association...')
        logging.info("=====================================================================")
        personAssociation(config)
    else:
        logging.info("\n\n=====================================================================")
        logging.info('Skipping person association.')
        logging.info("=====================================================================")

    if do_triangulation:
        logging.info("\n\n=====================================================================")
        logging.info('Running triangulation...')
        logging.info("=====================================================================")
        triangulation(config)
    else:
        logging.info("\n\n=====================================================================")
        logging.info('Skipping triangulation.')
        logging.info("=====================================================================")
        
    if do_filtering:
        logging.info("\n\n=====================================================================")
        logging.info('Running filtering...')
        logging.info("=====================================================================")
        filtering(config)
    else:
        logging.info("\n\n=====================================================================")
        logging.info('Skipping filtering.')
        logging.info("=====================================================================")

    if do_markerAugmentation:
        logging.info("\n\n=====================================================================")
        logging.info('Running marker augmentation.')
        logging.info("=====================================================================")
        markerAugmentation(config)
    else:
        logging.info("\n\n=====================================================================")
        logging.info('Skipping marker augmentation.')
        logging.info("\n\n=====================================================================")
    
    # if do_opensimProcessing:
    #     logging.info("\n\n=====================================================================")
    #     logging.info('Running opensim processing.')
    #     logging.info("=====================================================================")
    #     opensimProcessing(config)
    # else:
    #     logging.info("\n\n=====================================================================")
    #     logging.info('Skipping opensim processing.')
    #     logging.info("=====================================================================")

    end = time.time()
    elapsed = end-start 
    logging.info(f'\nRUNNING ALL FUNCTIONS TOOK  {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')