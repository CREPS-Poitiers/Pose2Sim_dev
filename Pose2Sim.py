#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
###########################################################################
## POSE2SIM                                                              ##
###########################################################################

This repository offers a way to perform markerless kinematics, and gives an 
example workflow from an Openpose input to an OpenSim result.

It offers tools for:
- Cameras calibration,
- 2D pose estimation,
- Camera synchronization,
- Tracking the person of interest,
- Robust triangulation,
- Filtration, 
- Marker augmentation,
- OpenSim scaling and inverse kinematics

It has been tested on Windows, Linux and MacOS, and works for any Python version >= 3.9

Installation: 
# Open Anaconda prompt. Type:
# - conda create -n Pose2Sim python=3.9
# - conda activate Pose2Sim
# - conda install -c opensim-org opensim -y
# - pip install Pose2Sim

Usage: 
# First run Pose estimation and organize your directories (see Readme.md)
from Pose2Sim import Pose2Sim
Pose2Sim.calibration()
Pose2Sim.poseEstimation()
Pose2Sim.synchronization()
Pose2Sim.personAssociation()
Pose2Sim.triangulation()
Pose2Sim.filtering()
Pose2Sim.markerAugmentation()
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
    Create logging file and stream handlers
    '''

    logging.basicConfig(format='%(message)s', level=logging.INFO, 
        handlers = [logging.handlers.TimedRotatingFileHandler(os.path.join(session_dir, 'logs.txt'), when='D', interval=7), logging.StreamHandler()])
    
    
def recursive_update(dict_to_update, dict_with_new_values):
    '''
    Update nested dictionaries without overwriting existing keys in any level of nesting
    
    Example: 
    dict_to_update = {'key': {'key_1': 'val_1', 'key_2': 'val_2'}}
    dict_with_new_values = {'key': {'key_1': 'val_1_new'}}
    returns {'key': {'key_1': 'val_1_new', 'key_2': 'val_2'}}
    while dict_to_update.update(dict_with_new_values) would return {'key': {'key_1': 'val_1_new'}}
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
    Determine the level at which the function is called.
    Level = 1: Trial folder
    Level = 2: Root folder
    '''

    len_paths = [len(root.split(os.sep)) for root,dirs,files in os.walk(config_dir) if 'Config.toml' in files]
    if len_paths == []:
        raise FileNotFoundError('You need a Config.toml file in each trial or root folder.')
    level = max(len_paths) - min(len_paths) + 1
    return level


def read_config_files(config):
    '''
    Read Root and Trial configuration files, 
    and output a dictionary with all the parameters.
    '''

    if type(config)==dict:
        level = 2 # log_dir = os.getcwd()
        config_dicts = [config]
        if config_dicts[0].get('project').get('project_dir') == None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_PROJECT_DIRECTORY>"})')
    else:
        # if launched without an argument, config == None, else it is the path to the config directory
        config_dir = ['.' if config == None else config][0]  
        level = determine_level(config_dir)
        
        # Trial level
        if level == 1: # Trial
            try:
                # if batch
                session_config_dict = toml.load(os.path.join(config_dir, '..','Config.toml'))
                trial_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
                session_config_dict = recursive_update(session_config_dict,trial_config_dict)
            except:
                # if single trial
                session_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
            session_config_dict.get("project").update({"project_dir":config_dir})
            config_dicts = [session_config_dict]
        
        # Root level
        if level == 2:
            session_config_dict = toml.load(os.path.join(config_dir, 'Config.toml'))
            config_dicts = []
            # Create config dictionaries for all trials of the participant
            for (root,dirs,files) in os.walk(config_dir):
                if 'Config.toml' in files and root != config_dir:
                    trial_config_dict = toml.load(os.path.join(root, files[0]))
                    # deep copy, otherwise session_config_dict is modified at each iteration within the config_dicts list
                    temp_dict = deepcopy(session_config_dict)
                    temp_dict = recursive_update(temp_dict,trial_config_dict)
                    temp_dict.get("project").update({"project_dir":os.path.join(config_dir, os.path.relpath(root))})
                    if not os.path.basename(root) in temp_dict.get("project").get('exclude_from_batch'):
                        config_dicts.append(temp_dict)

    return level, config_dicts


def classification(config=None):
    '''
    ***
    OBJECTIF DE LA FONCTION
    ***
    Classer toutes les vidéos d'une session d'acquisition avec la bonne 
    structure de dossier et les bons noms de dossier et de vidéos dans un 
    dossier "_traitement" qui est créé dans le dossier parent du dossier actif.
    
    ORGANISATION DES FICHIERS DANS LE DOSSIER A TRAITER
    ***
        - Placez toutes vos vidéos brutes dans le dossier et un fichier Config.toml
        - Dans le cas où vous gardez la même calibration intrinsèque, placez le dossier intrinsics et le fichier Calib.toml
        - Dans le cas où vous gardez la même calibration (intrinsèque et extrinsèque), placez le dossier calibration 
            contenant les sous-dossiers intrinsics et extrinsics et le fichier Calib.toml
    
    """
    '''
    
    #Changement of the function called for new calibration method with charuco
    from Pose2Sim.classification import classification_run

    # Determine the level at which the function is called (root:2, trial:1)
    level, config_dicts = read_config_files(config)
    config_dict = config_dicts[0]

    if type(config)==dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') == None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                              config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)    
    currentDateAndTime = datetime.now()
    
    # Run classification
    logging.info("\n---------------------------------------------------------------------")
    logging.info("Folders classification")
    logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
    logging.info("---------------------------------------------------------------------\n")
    start = time.time()
    
    classification_run(config_dict)
    
    end = time.time()
    logging.info(f'\nClassification took {end-start:.2f} s.\n')

def calibration(config=None):
    '''
    Cameras calibration from checkerboards or from qualisys files.
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''
    
    #Changement of the function called for new calibration method with charuco
    from Pose2Sim.calibration_dev import calibrate_cams_all

    level, config_dicts = read_config_files(config)
    config_dict = config_dicts[0]
    try:
        session_dir = os.path.realpath([os.getcwd() if level==2 else os.path.join(os.getcwd(), '..')][0])
        [os.path.join(session_dir, c) for c in os.listdir(session_dir) if 'calib' in c.lower() and not c.lower().endswith('.py')][0]
    except:
        session_dir = os.path.realpath(os.getcwd())
    config_dict.get("project").update({"project_dir":session_dir})

    # Set up logging
    setup_logging(session_dir)  
    currentDateAndTime = datetime.now()
    
    # Run calibration
    calib_dir = [os.path.join(session_dir, c) for c in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, c)) and  'calib' in c.lower()][0]
    logging.info("\n---------------------------------------------------------------------")
    logging.info("Camera calibration")
    logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
    logging.info(f"Calibration directory: {calib_dir}")
    logging.info("---------------------------------------------------------------------\n")
    start = time.time()
    
    calibrate_cams_all(config_dict)
    
    end = time.time()
    logging.info(f'\nCalibration took {end-start:.2f} s.\n')


def poseEstimation(config=None):
    '''
    Estimate pose using RTMLib
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''
    
    from Pose2Sim.poseEstimation import rtm_estimator # The name of the function might change

    level, config_dicts = read_config_files(config)

    if isinstance(config, dict):
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') is None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if not frame_range else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Pose estimation for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")
    
        rtm_estimator(config_dict)
        
        end = time.time()
        elapsed = end - start 
        logging.info(f'\nPose estimation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')
    

def synchronization(config=None):
    '''
    Synchronize cameras if needed.
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''   
    
    # Import the function
    from Pose2Sim.synchronization_dev import synchronize_cams_all

    # Determine the level at which the function is called (root:2, trial:1)
    level, config_dicts = read_config_files(config)

    if type(config)==dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') == None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)    

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))

        logging.info("\n---------------------------------------------------------------------")
        logging.info("Camera synchronization")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")
        
        synchronize_cams_all(level, config_dict)
    
        end = time.time()
        elapsed = end-start 
        logging.info(f'\nSynchronization took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')
    


def personAssociation(config=None):
    '''
    Tracking one or several persons of interest.
    Needs a calibration file.
    
    config can be a dictionary,
    or the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case the config directory is the current one.
    '''
    
    from Pose2Sim.personAssociation_dev import track_2d_all

    # Determine the level at which the function is called (root:2, trial:1)
    level, config_dicts = read_config_files(config)

    if not config_dicts:
        raise ValueError("No valid configuration files found. Please provide a valid configuration.")

    # Set up logging
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Associating persons for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")

        # Get configuration parameters specific to this trial
        iterative_optimal_association = config_dict.get('personAssociation', {}).get('iterative_optimal_association', False)
        likelihood_threshold_value = config_dict['personAssociation']['likelihood_threshold_association']
        reproj_error_threshold_value = config_dict['personAssociation']['single_person']['reproj_error_threshold_association']
        tracked_keypoint_value = config_dict['personAssociation']['single_person']['tracked_keypoint']
        min_cameras_value = config_dict['triangulation']['min_cameras_for_triangulation']

        # Adjust parameters if they are scalar values
        if iterative_optimal_association:
            if not isinstance(likelihood_threshold_value, list):
                likelihood_threshold_value = [likelihood_threshold_value, likelihood_threshold_value]
            if not isinstance(reproj_error_threshold_value, list):
                reproj_error_threshold_value = [reproj_error_threshold_value, reproj_error_threshold_value]
            if not isinstance(tracked_keypoint_value, list):
                tracked_keypoint_value = [tracked_keypoint_value]
            if not isinstance(min_cameras_value, list):
                min_cameras_value = [min_cameras_value, min_cameras_value]

            # Generate parameter combinations
            likelihood_values = [round(x, 2) for x in np.arange(likelihood_threshold_value[1], likelihood_threshold_value[0] - 0.1, -0.1)]
            reproj_values = [int(x) for x in range(reproj_error_threshold_value[1], reproj_error_threshold_value[0] - 1, -10)]
            min_cameras_values = [int(x) for x in range(min_cameras_value[0], min_cameras_value[1] + 1)]
            param_combinations = list(itertools.product(likelihood_values, reproj_values, tracked_keypoint_value, min_cameras_values))

        else:
            # No combinations: single execution
            param_combinations = [(likelihood_threshold_value, reproj_error_threshold_value, tracked_keypoint_value, min_cameras_value)]

        # Log parameters and number of combinations
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Number of combinations to test: {len(param_combinations)}")
        logging.info(f"Likelihood threshold range: {likelihood_threshold_value}")
        logging.info(f"Reprojection error threshold range: {reproj_error_threshold_value}")
        logging.info(f"Tracked keypoints: {tracked_keypoint_value}")
        logging.info(f"Minimum cameras for triangulation range: {min_cameras_value}")
        logging.info("---------------------------------------------------------------------\n")

        # Iterate through parameter combinations
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

        end = time.time()
        elapsed = end - start
        logging.info(f'\nAssociating persons took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')



# def personAssociation(config=None):
#     '''
#     Tracking one or several persons of interest.
#     Needs a calibration file.
    
#     config can be a dictionary,
#     or the directory path of a trial, participant, or session,
#     or the function can be called without an argument, in which case the config directory is the current one.
#     '''
    
#     from Pose2Sim.personAssociation import track_2d_all


#     # Determine the level at which the function is called (root:2, trial:1)
#     level, config_dicts = read_config_files(config)

#     if not config_dicts:
#         raise ValueError("No valid configuration files found. Please provide a valid configuration.")

#     # Set up logging
#     session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
#     setup_logging(session_dir)

#     # Batch process all trials
#     for config_dict in config_dicts:
#         start = time.time()
#         currentDateAndTime = datetime.now()
#         project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
#         seq_name = os.path.basename(project_dir)
#         frame_range = config_dict.get('project').get('frame_range')
#         frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

#         logging.info("\n---------------------------------------------------------------------")
#         logging.info(f"Associating persons for {seq_name}, for {frames}.")
#         logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
#         logging.info(f"Project directory: {project_dir}")
#         logging.info("---------------------------------------------------------------------\n")

#         # Get configuration parameters specific to this trial
#         iterative_optimal_association = config_dict.get('personAssociation', {}).get('iterative_optimal_association', False)
#         likelihood_threshold_association = config_dict.get('personAssociation', {}).get('likelihood_threshold_association', 1)
#         reproj_error_threshold_association = config_dict.get('personAssociation', {}).get('single_person', {}).get('reproj_error_threshold_association', 100)

#         # Generate parameter combinations if iterative_optimal_association is True
#         param_combinations = []
#         if iterative_optimal_association:
#             likelihood_values = [round(x, 2) for x in np.linspace(likelihood_threshold_association, 0, 11)]  # From 1 to 0 in steps of 0.1
#             reproj_values = [int(x) for x in np.linspace(reproj_error_threshold_association, 0, 11)]  # From reproj_error_threshold_association to 0
#             param_combinations = list(itertools.product(likelihood_values, reproj_values))
#         else:
#             param_combinations = [(likelihood_threshold_association, reproj_error_threshold_association)]

#         # Iterate through parameter combinations
#         for likelihood, reproj in param_combinations:
#             config_dict['personAssociation']['likelihood_threshold_association'] = likelihood
#             config_dict['personAssociation']['single_person']['reproj_error_threshold_association'] = reproj

#             logging.info(f"Testing with likelihood_threshold_association={likelihood} and reproj_error_threshold_association={reproj}")
#             track_2d_all(config_dict)

#         end = time.time()
#         elapsed = end - start
#         logging.info(f'\nAssociating persons took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')

    
    
    
# def personAssociation(config=None):
#     '''
#     Tracking one or several persons of interest.
#     Needs a calibration file.
    
#     config can be a dictionary,
#     or a the directory path of a trial, participant, or session,
#     or the function can be called without an argument, in which case it the config directory is the current one.
#     '''
    
#     from Pose2Sim.personAssociation import track_2d_all

#     # Determine the level at which the function is called (root:2, trial:1)
#     level, config_dicts = read_config_files(config)

#     if type(config)==dict:
#         config_dict = config_dicts[0]
#         if config_dict.get('project').get('project_dir') == None:
#             raise ValueError('Please specify the project directory in config_dict:\n \
#                              config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

#     # Set up logging
#     session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
#     setup_logging(session_dir)    

#     # Batch process all trials
#     for config_dict in config_dicts:
#         start = time.time()
#         currentDateAndTime = datetime.now()
#         project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
#         seq_name = os.path.basename(project_dir)
#         frame_range = config_dict.get('project').get('frame_range')
#         frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

#         logging.info("\n---------------------------------------------------------------------")
#         logging.info(f"Associating persons for {seq_name}, for {frames}.")
#         logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
#         logging.info(f"Project directory: {project_dir}")
#         logging.info("---------------------------------------------------------------------\n")
    
#         track_2d_all(config_dict)
    
#         end = time.time()
#         elapsed = end-start 
#         logging.info(f'\nAssociating persons took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')
    


def triangulation(config=None):
    '''
    Robust triangulation of 2D points coordinates.
    
    config can be a dictionary,
    or the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''

    from Pose2Sim.triangulation_dev import triangulate_all


    # Determine the level at which the function is called (root:2, trial:1)
    level, config_dicts = read_config_files(config)

    if not config_dicts:
        raise ValueError("No valid configuration files found. Please provide a valid configuration.")

    # Set up logging
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)

    # Batch process all trials
    for config_dict in config_dicts:
        start = time.time()
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Triangulation of 2D points for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")

        # Get configuration parameters specific to this trial
        iterative_optimal_triangulation = config_dict.get('triangulation', {}).get('iterative_optimal_triangulation', False)
        reproj_error_threshold_value = config_dict['triangulation']['reproj_error_threshold_triangulation']
        likelihood_threshold_value = config_dict['triangulation']['likelihood_threshold_triangulation']
        min_cameras_value = config_dict['triangulation']['min_cameras_for_triangulation']

        # Vérification des formats selon iterative_optimal_triangulation
        if iterative_optimal_triangulation:
            # Convert single values to ranges if needed
            if not isinstance(reproj_error_threshold_value, list):
                reproj_error_threshold_value = [reproj_error_threshold_value, reproj_error_threshold_value]
            if not isinstance(likelihood_threshold_value, list):
                likelihood_threshold_value = [likelihood_threshold_value, likelihood_threshold_value]
            if not isinstance(min_cameras_value, list):
                min_cameras_value = [min_cameras_value, min_cameras_value]

            # Génération des valeurs pour les tests
            reproj_values = [int(x) for x in range(reproj_error_threshold_value[1], reproj_error_threshold_value[0] - 1, -5)]
            likelihood_values = [round(x, 2) for x in np.arange(likelihood_threshold_value[1], likelihood_threshold_value[0] - 0.1, -0.1)]
            min_cameras_values = [int(x) for x in range(min_cameras_value[0], min_cameras_value[1] + 1)]
            param_combinations = list(itertools.product(reproj_values, likelihood_values, min_cameras_values))

        else:
            # Pas de combinaisons : une seule exécution
            param_combinations = [(reproj_error_threshold_value, likelihood_threshold_value, min_cameras_value)]

        # Affichage des paramètres et du nombre de combinaisons
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Number of combinations to test: {len(param_combinations)}")
        logging.info(f"Reprojection error threshold range: {reproj_error_threshold_value}")
        logging.info(f"Likelihood threshold range: {likelihood_threshold_value}")
        logging.info(f"Minimum cameras for triangulation range: {min_cameras_value}")
        logging.info("---------------------------------------------------------------------\n")

        # Iterate through parameter combinations
        for reproj, likelihood, min_cameras in param_combinations:
            config_dict['triangulation']['reproj_error_threshold_triangulation'] = reproj
            config_dict['triangulation']['likelihood_threshold_triangulation'] = likelihood
            config_dict['triangulation']['min_cameras_for_triangulation'] = min_cameras

            logging.info(f"Testing with reproj_error_threshold_triangulation={reproj}, likelihood_threshold_triangulation={likelihood}, min_cameras_for_triangulation={min_cameras}")
            triangulate_all(config_dict)

        # Réinitialisation des valeurs originales
        config_dict['triangulation']['reproj_error_threshold_triangulation'] = reproj_error_threshold_value
        config_dict['triangulation']['likelihood_threshold_triangulation'] = likelihood_threshold_value
        config_dict['triangulation']['min_cameras_for_triangulation'] = min_cameras_value

        end = time.time()
        elapsed = end - start
        logging.info(f'\nTriangulation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')






    
# def triangulation(config=None):
#     '''
#     Robust triangulation of 2D points coordinates.
    
#     config can be a dictionary,
#     or a the directory path of a trial, participant, or session,
#     or the function can be called without an argument, in which case it the config directory is the current one.
#     '''

#     from Pose2Sim.triangulation import triangulate_all

#     # Determine the level at which the function is called (root:2, trial:1)
#     level, config_dicts = read_config_files(config)

#     if type(config)==dict:
#         config_dict = config_dicts[0]
#         if config_dict.get('project').get('project_dir') == None:
#             raise ValueError('Please specify the project directory in config_dict:\n \
#                              config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

#     # Set up logging
#     session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
#     setup_logging(session_dir)  

#     # Batch process all trials
#     for config_dict in config_dicts:
#         start = time.time()
#         currentDateAndTime = datetime.now()
#         project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
#         seq_name = os.path.basename(project_dir)
#         frame_range = config_dict.get('project').get('frame_range')
#         frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

#         logging.info("\n---------------------------------------------------------------------")
#         logging.info(f"Triangulation of 2D points for {seq_name}, for {frames}.")
#         logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
#         logging.info(f"Project directory: {project_dir}")
#         logging.info("---------------------------------------------------------------------\n")
        
#         triangulate_all(config_dict)
    
#         end = time.time()
#         elapsed = end-start 
#         logging.info(f'\nTriangulation took {time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))}.\n')
 
    
def filtering(config=None):
    '''
    Filter trc 3D coordinates.
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''

    from Pose2Sim.filtering import filter_all

    # Determine the level at which the function is called (root:2, trial:1)
    level, config_dicts = read_config_files(config)

    if type(config)==dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') == None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    # Set up logging
    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)

    # Batch process all trials
    for config_dict in config_dicts:
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]
    
        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Filtering 3D coordinates for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}\n")
        logging.info("---------------------------------------------------------------------\n")
    
        filter_all(config_dict)
        
        logging.info('\n')


def markerAugmentation(config=None):
    '''
    Augment trc 3D coordinates. 
    Estimate the position of 43 additional markers.
    
    config can be a dictionary,
    or a the directory path of a trial, participant, or session,
    or the function can be called without an argument, in which case it the config directory is the current one.
    '''

    from Pose2Sim.markerAugmentation import augmentTRC
    level, config_dicts = read_config_files(config)

    if type(config) == dict:
        config_dict = config_dicts[0]
        if config_dict.get('project').get('project_dir') is None:
            raise ValueError('Please specify the project directory in config_dict:\n \
                             config_dict.get("project").update({"project_dir":"<YOUR_TRIAL_DIRECTORY>"})')

    session_dir = os.path.realpath(os.path.join(config_dicts[0].get('project').get('project_dir'), '..'))
    setup_logging(session_dir)

    for config_dict in config_dicts:
        start = time.time()
        currentDateAndTime = datetime.now()
        project_dir = os.path.realpath(config_dict.get('project').get('project_dir'))
        seq_name = os.path.basename(project_dir)
        frame_range = config_dict.get('project').get('frame_range')
        frames = ["all frames" if frame_range == [] else f"frames {frame_range[0]} to {frame_range[1]}"][0]

        logging.info("\n---------------------------------------------------------------------")
        logging.info(f"Augmentation process for {seq_name}, for {frames}.")
        logging.info(f"On {currentDateAndTime.strftime('%A %d. %B %Y, %H:%M:%S')}")
        logging.info(f"Project directory: {project_dir}")
        logging.info("---------------------------------------------------------------------\n")

        augmentTRC(config_dict)

        end = time.time()
        elapsed = end-start 
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