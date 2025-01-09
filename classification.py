# -*- coding: utf-8 -*-
"""
Script for classifying videos from an acquisition session into a structured folder format.

Author: fdelaplace
Date: November 17, 2024

**Objective:**
The script organizes raw video files from an acquisition session into a processing folder (`_traitement`) with the appropriate directory structure and naming conventions. It also handles calibration data if available.

**Key Features:**
1. **Video Classification**:
   - Videos are sorted by acquisition timestamp (excluding seconds) and camera name.
   - Videos are grouped into `Trial_X` directories with subfolders for raw videos (`videos_raw`).

2. **Calibration Data Handling**:
   - Detects and copies calibration data (`intrinsics` and `extrinsics` folders, and relevant `.toml` files) into the `_traitement` directory.
   - Supports scenarios where only `intrinsics` or no calibration data is available.

3. **Dynamic Folder Creation**:
   - Creates structured subfolders for calibration (`calibration/intrinsics` and `calibration/extrinsics`).
   - Generates trial directories (`Trial_X`) for organizing videos.

4. **Error Handling**:
   - Ensures consistency between the number of cameras and videos.
   - Alerts if critical files like `Config.toml` are missing.

5. **Progress Visualization**:
   - Uses `tqdm` to provide progress bars for long operations, such as file copying and folder creation.

6. **Logging**:
   - Logs the classification process, including camera count, number of trials, and working directory.

**Usage:**
- Place raw videos and configuration files in the working directory.
- Run the script to classify and structure the files into the `_traitement` folder.
- Ensure videos follow a specific naming convention with timestamps and camera identifiers (e.g., `20240911_123456-CAMERA01.MP4`).

**Dependencies:**
- Python standard libraries (`os`, `shutil`, `datetime`, `logging`).
- `tqdm` for progress bars.
- `Pathlib` for path manipulations.

"""

import shutil
import os
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm  # Pour ajouter des barres de chargement

def copy_folder_progress(source, destination):
    """
    Copies a folder from 'source' to 'destination' with a progress bar.

    :param source: Path to the source directory to be copied.
    :param destination: Path to the target directory where the files and folders will be copied.
    """
    try:
        # Check if the source directory exists
        if not os.path.exists(source):
            print(f"Le dossier source {source} n'existe pas.")  # Warn if source doesn't exist
            return
        
        # Create the destination directory if it does not exist
        if not os.path.exists(destination):
            os.makedirs(destination)  # Ensure destination directory exists

        # Count total files in the source directory for the progress bar
        total_files = sum([len(files) for _, _, files in os.walk(source)])

        # Use tqdm to display a progress bar during the copy process
        with tqdm(total=total_files, desc="Copie des fichiers", unit="fichier") as pbar:
            for root, dirs, files in os.walk(source):  # Traverse the source directory
                # Compute relative path for the current directory
                rel_path = os.path.relpath(root, source)
                dest_dir = os.path.join(destination, rel_path)
                
                # Create the corresponding directory in the destination
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                # Copy each file and update the progress bar
                for file in files:
                    src_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_dir, file)
                    shutil.copy2(src_file, dest_file)  # Preserve file metadata
                    pbar.update(1)  # Increment progress bar

        print(f"Copie terminée de {source} à {destination}")  # Success message

    except Exception as e:
        # Log the error message in case of failure
        print(f"Erreur lors de la copie : {e}")


def classification_run(config_dict):
    """
    ***
    FUNCTION OBJECTIVE
    ***
    This function organizes all videos from an acquisition session into a standardized folder structure 
    to facilitate data processing and analysis. It also ensures that calibration files 
    (intrinsic and extrinsic) and necessary configuration files are correctly copied to their designated locations. 
    Videos are sorted by their timestamp and corresponding camera.

    The data is then transferred into a "_traitement" folder, created in the parent directory of the current working folder, 
    following a well-defined folder structure:
    
    **FOLDER STRUCTURE IN "_traitement"**
        - **Calibration** (if applicable):
            - `intrinsics/`: Intrinsic calibration data.
            - `extrinsics/`: Extrinsic calibration data.
        - **Trials**:
            - `Trial_1/` -> `videos_raw/`: Raw videos from the first trial.
            - `Trial_2/` -> `videos_raw/`: Raw videos from the second trial, and so on.

    **PREREQUISITES FOR USE**
        - Place all your raw videos in a folder along with a `Config.toml` file.
        - If intrinsic calibration is being reused, include an `intrinsics` folder with the `Calib.toml` file.
        - If full calibration (intrinsic and extrinsic) is being reused, include a `calibration` folder 
          containing `intrinsics` and `extrinsics` subfolders, along with the `Calib.toml` file.

    ***
    FUNCTION EXECUTION
    ***
    The function performs the following steps:
        1. Checks for video files and calibration folders in the current directory.
        2. Sorts the videos by timestamp and identifies unique cameras.
        3. Calculates the number of trials based on the number of cameras and videos.
        4. Creates the "_traitement" folder and organizes videos into appropriate subfolders (Trials, Calibration).
        5. Copies the required files and folders while ensuring consistent renaming.
    """

    print("====================")
    print("File classification...")
    print("--------------------\n\n")  

    # 1. Define working directory and video format
    path = os.getcwd()
    videoFormat = config_dict['pose']['vid_img_extension']

    # 2. Create the "_traitement" folder in the parent directory
    traitement_path = os.path.join(os.path.dirname(path), os.path.basename(path) + "_traitement")
    if not os.path.exists(traitement_path):
        os.mkdir(traitement_path)
        print(f"Création du dossier de traitement : {traitement_path}")
    
    # 3. Retrieve video file names in the working directory
    filenames = [file for file in os.listdir(path) if file.endswith(videoFormat)]

    # Sort videos by timestamp and camera number
    filenames = sorted(filenames, key=lambda x: (
        datetime.strptime(x[:13], "%Y%m%d_%H%M"),  # Extract date and time
        int(x[13:15]) // 10,                      # Extract seconds range
        x[x.find("CAMERA"):x.find("CAMERA") + 8]  # Sort by camera name
    ))

    # Count total files and identify unique cameras
    nbfiles = len(filenames)
    camList = [filenames[file][filenames[file].find("CAMERA"):filenames[file].find("CAMERA")+8] for file in range(nbfiles)]
    camNames = list(set(camList))  # Unique cameras
    nbcam = len(camNames)

    print(f"Nombre total de fichiers : {nbfiles}")
    print(f"Nombre de caméras uniques : {nbcam}")

    # Copy the Config.toml file if present
    config_file = os.path.join(path, 'Config.toml')
    if os.path.exists(config_file):
        shutil.copy(config_file, os.path.join(traitement_path, 'Config.toml'))
        print("Le fichier Config.toml a été copié dans le dossier de traitement.")
    else:
        print("ATTENTION : Aucun fichier 'Config.toml' trouvé dans le dossier actif.")

    # 4. Check for calibration folders
    """
    2 - Vérification des dossiers calibration et intrinsics
    If a calibration or intrinsics folder is present, it will be used for the processing.
    """
    calibFolderPath = os.path.join(path, "calibration")
    calibVerifIntFolderPath = os.path.join(path, "intrinsics")
    userCalib, userCalibInt = False, False

    if os.path.exists(calibFolderPath):
        print("Un dossier calibration a été trouvé.")
        userCalib = True  # Use calibration folder

    elif os.path.exists(calibVerifIntFolderPath):
        print("Un dossier intrinsics a été trouvé.")
        userCalibInt = True  # Use intrinsics folder

    # Identify calibration files
    calib_files = ['Calib.toml', 'Calib_scene.toml', 'calib.toml']
    found_calib_file = None
    for calib_file in calib_files:
        if os.path.exists(os.path.join(path, calib_file)):
            found_calib_file = calib_file
            print(f"Le fichier {calib_file} a été trouvé dans le dossier actif.")
            break

    # 5. Calculate number of trials
    """
    3 - Sort trials
    Determine the number of trials based on the number of cameras and videos.
    """
    if userCalib == False and userCalibInt == False:
        nbtrials = (nbfiles - nbcam * 2) / nbcam
    elif userCalib == False and userCalibInt == True:
        nbtrials = (nbfiles - nbcam) / nbcam
    else:
        nbtrials = (nbfiles) / nbcam

    # Check trial validity
    if nbtrials % 1 == 0:
        nbtrials = int(nbtrials)
        print(f"Nombre d'essais (trials) trouvés : {nbtrials}")

        # Create Trial folders
        print("Création des dossiers Trial...")
        for trial in tqdm(range(1, nbtrials + 1), desc="Création des dossiers Trial"):
            trial_path = os.path.join(traitement_path, f"Trial_{trial}")
            if not os.path.exists(trial_path):
                os.mkdir(trial_path)
            if not os.path.exists(os.path.join(trial_path, "videos_raw")):
                os.mkdir(os.path.join(trial_path, "videos_raw"))

        # Create a new calibration folder if calibration data is provided
        new_calib_folder = os.path.join(traitement_path, "calibration")
        
        if userCalib:
            """
            If a calibration folder exists, classify videos into trials and copy the calibration folder.
            """
            print("Classement des vidéos dans les dossiers Trial.")
            for acq in tqdm(range(nbtrials), desc="Classification des essais"):
                for cam in range(nbcam):
                    # Source and destination paths
                    src = os.path.join(path, filenames[nbcam * acq + cam])
                    original_name = filenames[nbcam * acq + cam]
                    new_name = original_name[:13] + "-" + original_name[original_name.find("CAMERA"):original_name.find("CAMERA") + 8] + ".MP4"
                    dest_dir = os.path.join(traitement_path, f"Trial_{acq + 1}", "videos_raw")
                    dest = os.path.join(dest_dir, new_name)

                    # Ensure the destination folder exists and copy the file
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(src, dest)

            # Copy the calibration folder
            copy_folder_progress(calibFolderPath, new_calib_folder)

            # Copy calibration file if found
            if found_calib_file and not os.path.exists(os.path.join(calibFolderPath, found_calib_file)):
                shutil.copy(os.path.join(path, found_calib_file), os.path.join(traitement_path, "calibration", found_calib_file))
                print(f"Le fichier {found_calib_file} a été copié dans le dossier calibration du traitement.")

        elif userCalibInt:
            """
            If only the intrinsics folder is found, create extrinsics subfolders and manage trials.
            """
            print("Le dossier intrinsics a été trouvé. Création des extrinsics et gestion des trials.")
            if not os.path.exists(new_calib_folder):
                os.mkdir(new_calib_folder)

            # Copy the intrinsics folder
            copy_folder_progress(calibVerifIntFolderPath, os.path.join(new_calib_folder, "intrinsics"))

            # Create extrinsics subfolders
            calibExtFolderPath = os.path.join(new_calib_folder, "extrinsics")
            os.mkdir(calibExtFolderPath)
            for n in tqdm(range(1, nbcam + 1), desc="Création des sous-dossiers extrinsèques"):
                folder_name = f"ext_cam{n:02}"  # Format: ext_cam01, ext_cam02, etc.
                os.mkdir(os.path.join(calibExtFolderPath, folder_name))

            # Copy extrinsic videos and classify trials
            for cam in tqdm(range(nbcam), desc="Classification des vidéos extrinsèques"):
                shutil.copy(os.path.join(path, filenames[cam]),
                            os.path.join(calibExtFolderPath, f"ext_cam{cam+1:02}", filenames[cam]))

            for acq in tqdm(range(nbtrials), desc="Classification des essais"):
                for cam in range(nbcam):
                    src = os.path.join(path, filenames[nbcam * acq + cam])
                    original_name = filenames[nbcam * acq + cam]
                    new_name = original_name[:13] + "-" + original_name[original_name.find("CAMERA"):original_name.find("CAMERA") + 8] + ".MP4"
                    dest_dir = os.path.join(traitement_path, f"Trial_{acq + 1}", "videos_raw")
                    dest = os.path.join(dest_dir, new_name)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(src, dest)

            # Copy calibration file
            if found_calib_file:
                shutil.copy(os.path.join(path, found_calib_file), os.path.join(traitement_path, "calibration", found_calib_file))
                print(f"Le fichier {found_calib_file} a été copié dans le dossier calibration du traitement.")

        else:
            """
            If no calibration is found, handle full video management for intrinsics, extrinsics, and trials.
            """
            print("Aucune calibration trouvée. Gestion complète des vidéos (intrinsics, extrinsics, trials).")
            os.mkdir(new_calib_folder)

            # Create intrinsics and extrinsics subfolders
            calibIntFolderPath = os.path.join(new_calib_folder, "intrinsics")
            os.mkdir(calibIntFolderPath)
            for n in tqdm(range(1, nbcam + 1), desc="Création des sous-dossiers intrinsics"):
                folder_name = f"int_cam{n:02}"
                os.mkdir(os.path.join(calibIntFolderPath, folder_name))

            calibExtFolderPath = os.path.join(new_calib_folder, "extrinsics")
            os.mkdir(calibExtFolderPath)
            for n in tqdm(range(1, nbcam + 1), desc="Création des sous-dossiers extrinsèques"):
                folder_name = f"ext_cam{n:02}"
                os.mkdir(os.path.join(calibExtFolderPath, folder_name))

            # Classify videos into intrinsics, extrinsics, and trials
            for cam in tqdm(range(nbcam), desc="Classification des vidéos intrinsics"):
                shutil.copy(os.path.join(path, filenames[cam]),
                            os.path.join(calibIntFolderPath, f"int_cam{cam+1:02}", filenames[cam]))

            for cam in tqdm(range(nbcam), desc="Classification des vidéos extrinsics"):
                shutil.copy(os.path.join(path, filenames[nbcam + cam]),
                            os.path.join(calibExtFolderPath, f"ext_cam{cam+1:02}", filenames[nbcam + cam]))

            for acq in tqdm(range(nbtrials), desc="Classification des essais"):
                for cam in range(nbcam):
                    src = os.path.join(path, filenames[nbcam * acq + cam])
                    original_name = filenames[nbcam * acq + cam]
                    new_name = original_name[:13] + "-" + original_name[original_name.find("CAMERA"):original_name.find("CAMERA") + 8] + ".MP4"
                    dest_dir = os.path.join(traitement_path, f"Trial_{acq + 1}", "videos_raw")
                    dest = os.path.join(dest_dir, new_name)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(src, dest)

            # Copy calibration file if found
            if found_calib_file:
                shutil.copy(os.path.join(path, found_calib_file), os.path.join(traitement_path, "calibration", found_calib_file))
                print(f"Le fichier {found_calib_file} a été copié dans le dossier calibration du traitement.")

    else:
        print("ERROR - Nombre de fichiers vidéo non cohérent avec le nombre de caméras.")

    # 6. Final summary and log creation
    os.chdir(traitement_path)
    print("\n\n--------------------")
    logging.info(f'Dossier de travail : {traitement_path}')
    logging.info(f'Nombre de caméras trouvées : {nbcam}')
    logging.info(f'Nombre d\'acquisitions classées : {nbtrials}')
    print("\nClassification des enregistrements fait avec succès.")
    print("====================")

# Function call example:
# classification_run(config_dict)
