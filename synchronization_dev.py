# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 10:53:59 2024

@author: fdelaplace
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
#########################################
## SYNCHRONIZE CAMERAS                 ##
#########################################

    Post-synchronize your cameras in case they are not natively synchronized.

    For each camera, computes mean vertical speed for the chosen keypoints, 
    and find the time offset for which their correlation is highest. 

    Depending on the analysed motion, all keypoints can be taken into account, 
    or a list of them, or the right or left side.
    All frames can be considered, or only those around a specific time (typically, 
    the time when there is a single participant in the scene performing a clear vertical motion).
    Has also been successfully tested for synchronizing random walkswith random walks.

    Keypoints whose likelihood is too low are filtered out; and the remaining ones are 
    filtered with a butterworth filter.

    INPUTS: 
    - json files from each camera folders
    - a Config.toml file
    - a skeleton model

    OUTPUTS: 
    - synchronized json files for each camera
'''


## INIT
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import json
import os
import glob
import fnmatch
import time
from datetime import datetime
import re
import shutil
from anytree import RenderTree
from anytree.importer import DictImporter
import logging
import toml
import tomlkit
from tqdm import tqdm
from Pose2Sim.common import sort_stringlist_by_last_number
from Pose2Sim.skeletons import *
from pathlib import Path
import ffmpeg
from tqdm import tqdm
import keyboard  # Bibliothèque pour écouter les touches
import screeninfo
import sys
import threading
from queue import Queue

#CHANGER ICI si on veut la skelly de base ou ameliorée avec le filtrage (skelly_synchronize ou skelly_synchronize_dev)
from skelly_synchronize import skelly_synchronize_dev as sync


## AUTHORSHIP INFORMATION
__author__ = "David Pagnon, HunMin Kim"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


# FUNCTIONS
def convert_json2pandas(json_files, likelihood_threshold=0.6):
    '''
    Convert a list of JSON files to a pandas DataFrame.

    INPUTS:
    - json_files: list of str. Paths of the the JSON files.
    - likelihood_threshold: float. Drop values if confidence is below likelihood_threshold.
    - frame_range: select files within frame_range.

    OUTPUTS:
    - df_json_coords: dataframe. Extracted coordinates in a pandas dataframe.
    '''

    nb_coord = 25 # int(len(json_data)/3)
    json_coords = []
    for j_p in json_files:
        with open(j_p) as j_f:
            try:
                json_data = json.load(j_f)['people'][0]['pose_keypoints_2d']
                # remove points with low confidence
                json_data = np.array([[json_data[3*i],json_data[3*i+1],json_data[3*i+2]] if json_data[3*i+2]>likelihood_threshold else [0.,0.,0.] for i in range(nb_coord)]).ravel().tolist()
            except:
                # print(f'No person found in {os.path.basename(json_dir)}, frame {i}')
                json_data = [np.nan] * 25*3
        json_coords.append(json_data)
    df_json_coords = pd.DataFrame(json_coords)

    return df_json_coords


def drop_col(df, col_nb):
    '''
    Drops every nth column from a DataFrame.

    INPUTS:
    - df: dataframe. The DataFrame from which columns will be dropped.
    - col_nb: int. The column number to drop.

    OUTPUTS:
    - dataframe: DataFrame with dropped columns.
    '''

    idx_col = list(range(col_nb-1, df.shape[1], col_nb)) 
    df_dropped = df.drop(idx_col, axis=1)
    df_dropped.columns = range(df_dropped.columns.size)
    return df_dropped


def vert_speed(df, axis='y'):
    '''
    Calculate the vertical speed of a DataFrame along a specified axis.

    INPUTS:
    - df: dataframe. DataFrame of 2D coordinates.
    - axis: str. The axis along which to calculate speed. 'x', 'y', or 'z', default is 'y'.

    OUTPUTS:
    - df_vert_speed: DataFrame of vertical speed values.
    '''

    axis_dict = {'x':0, 'y':1, 'z':2}
    df_diff = df.diff()
    df_diff = df_diff.fillna(df_diff.iloc[1]*2)
    df_vert_speed = pd.DataFrame([df_diff.loc[:, 2*k + axis_dict[axis]] for k in range(int(df_diff.shape[1] / 2))]).T # modified ( df_diff.shape[1]*2 to df_diff.shape[1] / 2 )
    df_vert_speed.columns = np.arange(len(df_vert_speed.columns))
    return df_vert_speed


def interpolate_zeros_nans(col, kind):
    '''
    Interpolate missing points (of value nan)

    INPUTS:
    - col: pandas column of coordinates
    - kind: 'linear', 'slinear', 'quadratic', 'cubic'. Default 'cubic'

    OUTPUTS:
    - col_interp: interpolated pandas column
    '''
    
    mask = ~(np.isnan(col) | col.eq(0)) # true where nans or zeros
    idx_good = np.where(mask)[0]
    try: 
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind=kind, bounds_error=False)
        col_interp = np.where(mask, col, f_interp(col.index))
        return col_interp 
    except:
        # print('No good values to interpolate')
        return col


def time_lagged_cross_corr(camx, camy, lag_range, show=True, ref_cam_id=0, cam_id=1):
    '''
    Compute the time-lagged cross-correlation between two pandas series.

    INPUTS:
    - camx: pandas series. Coordinates of reference camera.
    - camy: pandas series. Coordinates of camera to compare.
    - lag_range: int or list. Range of frames for which to compute cross-correlation.
    - show: bool. If True, display the cross-correlation plot.
    - ref_cam_id: int. The reference camera id.
    - cam_id: int. The camera id to compare.

    OUTPUTS:
    - offset: int. The time offset for which the correlation is highest.
    - max_corr: float. The maximum correlation value.
    '''

    if isinstance(lag_range, int):
        lag_range = [-lag_range, lag_range]

    import hashlib
    print(repr(list(camx)), repr(list(camy)))
    hashlib.md5(pd.util.hash_pandas_object(camx).values).hexdigest()
    hashlib.md5(pd.util.hash_pandas_object(camy).values).hexdigest()

    pearson_r = [camx.corr(camy.shift(lag)) for lag in range(lag_range[0], lag_range[1])]
    offset = int(np.floor(len(pearson_r)/2)-np.argmax(pearson_r))
    if not np.isnan(pearson_r).all():
        max_corr = np.nanmax(pearson_r)

        if show:
            f, ax = plt.subplots(2,1)
            # speed
            camx.plot(ax=ax[0], label = f'Reference: camera #{ref_cam_id}')
            camy.plot(ax=ax[0], label = f'Compared: camera #{cam_id}')
            ax[0].set(xlabel='Frame', ylabel='Speed (px/frame)')
            ax[0].legend()
            # time lagged cross-correlation
            ax[1].plot(list(range(lag_range[0], lag_range[1])), pearson_r)
            ax[1].axvline(np.ceil(len(pearson_r)/2) + lag_range[0],color='k',linestyle='--')
            ax[1].axvline(np.argmax(pearson_r) + lag_range[0],color='r',linestyle='--',label='Peak synchrony')
            plt.annotate(f'Max correlation={np.round(max_corr,2)}', xy=(0.05, 0.9), xycoords='axes fraction')
            ax[1].set(title=f'Offset = {offset} frames', xlabel='Offset (frames)',ylabel='Pearson r')
            
            plt.legend()
            f.tight_layout()
            plt.show()
    else:
        max_corr = 0
        offset = 0
        if show:
            # print('No good values to interpolate')
            pass

    return offset, max_corr


def extract_camera_number(filename):
    """Extract the camera number from the filename, assuming format like CAMERA01.mp4"""
    match = re.search(r"CAMERA(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0  # Si le nom n'a pas de numéro de caméra identifiable


def synchroMosaique(trial_folder):
    # Récupérer les fichiers vidéos .MP4 et les trier par numéro de caméra
    video_files = [f for f in os.listdir(trial_folder) if f.lower().endswith(".mp4")]
    video_files.sort(key=extract_camera_number)  # Trier par numéro de caméra

    nbVideos = len(video_files)

    # Vérification de l'existence des vidéos
    if nbVideos == 0:
        print(f"No videos found in {trial_folder}")
        return

    # Calculer les dimensions de la mosaïque
    dimOverlay = 2
    while nbVideos / dimOverlay > dimOverlay:
        dimOverlay += 1

    # # Lire le fichier synchronization_debug.toml pour obtenir les lags
    # toml_file = os.path.join(trial_folder, "debug_files", "synchronization_debug.toml")
    # if not os.path.exists(toml_file):
    #     print(f"Error: synchronization_debug.toml not found in {trial_folder}")
    #     return

    # # Charger les données TOML
    # with open(toml_file, 'r') as f:
    #     toml_data = toml.load(f)

    # # Extraire les lags dans l'ordre et les stocker dans une liste
    # lag_dict = toml_data.get("Lag_dictionary", {})
    # lag_values = [float(lag_dict[key].replace('"', '')) for key in sorted(lag_dict.keys())]

    # Construire la commande FFmpeg pour créer la mosaïque
    ffmpeg_cmd = "ffmpeg"

    # Ajouter les vidéos en entrée
    for vid in video_files:
        ffmpeg_cmd += f" -i {os.path.join(trial_folder, vid)}"

    # Créer le filtre pour la mosaïque
    filter_complex = f' -filter_complex "nullsrc=size=1920x1080 [base];'

    # Ajouter les vidéos redimensionnées
    for i in range(nbVideos):
        filter_complex += f"[{i}:v] setpts=PTS-STARTPTS, scale={int(1920/dimOverlay)}x{int(1080/dimOverlay)} [v{i}];"

    # Positionner les vidéos dans la mosaïque
    xinc = 1920 // dimOverlay
    yinc = 1080 // dimOverlay
    vidInc = 0

    for y in range(dimOverlay):
        ypos = y * yinc
        for x in range(dimOverlay):
            xpos = x * xinc
            if vidInc < nbVideos:
                if vidInc == 0:
                    filter_complex += f"[base][v{vidInc}] overlay=shortest=1:x={xpos}:y={ypos} [tmp{vidInc}];"
                elif vidInc == nbVideos - 1:
                    filter_complex += f"[tmp{vidInc-1}][v{vidInc}] overlay=shortest=1:x={xpos}:y={ypos}\" "
                else:
                    filter_complex += f"[tmp{vidInc-1}][v{vidInc}] overlay=shortest=1:x={xpos}:y={ypos} [tmp{vidInc}];"
                vidInc += 1

    # Enregistrer la mosaïque sans texte
    mosaic_path = os.path.join(trial_folder, "debug_files", "SyncVideos.mp4")
    ffmpeg_cmd += filter_complex + f" -c:v libx264 {mosaic_path} -y"

    # Rediriger la sortie de FFmpeg pour éviter l'affichage des informations inutiles
    if os.name == 'nt':  # Windows
        ffmpeg_cmd += " > NUL 2>&1"
    else:  # Linux/Mac
        ffmpeg_cmd += " > /dev/null 2>&1"

    # Exécuter la commande FFmpeg pour créer la mosaïque
    print(f"Creating mosaic for {trial_folder}")
    os.system(ffmpeg_cmd)

    # Vérifier si le fichier de la mosaïque a été créé
    if not os.path.exists(mosaic_path):
        print(f"Error: The video {mosaic_path} was not created.")
        return

    # Ajouter le texte des caméras et des lags
    output_with_text_path = os.path.join(trial_folder, "debug_files", "SyncVideos_with_text.mp4")
    text_commands = ""

    vidInc = 0
    for y in range(dimOverlay):
        ypos = y * yinc + 10  # Position verticale du texte pour CAMERA
        for x in range(dimOverlay):
            xpos = x * xinc + 30  # Position horizontale du texte
            if vidInc < nbVideos:
                #lag_value = lag_values[vidInc] if vidInc < len(lag_values) else "N/A"
                # Première ligne pour 'CAMERA X'
                text_commands += f"drawtext=text='CAMERA {vidInc + 1}':fontcolor=black:fontsize=24:x={xpos}:y={ypos},"
                # Seconde ligne pour 'Lag: 0.000'
                #text_commands += f"drawtext=text='Lag: {lag_value:.4f}s':fontcolor=black:fontsize=24:x={xpos}:y={ypos + 30},"
                vidInc += 1


    # Ajouter le texte pour demander de sauvegarder ou non
    text_commands += "drawtext=text='Do you want to keep this mosaic? Type Y to save or N to delete':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=h-th-50"

    # Exécuter la commande FFmpeg pour ajouter le texte
    ffmpeg_text_cmd = f"ffmpeg -i {mosaic_path} -vf \"{text_commands}\" -c:v libx264 {output_with_text_path} -y"

    # Rediriger la sortie de FFmpeg pour éviter l'affichage des informations inutiles
    if os.name == 'nt':  # Windows
        ffmpeg_text_cmd += " > NUL 2>&1"
    else:  # Linux/Mac
        ffmpeg_text_cmd += " > /dev/null 2>&1"

    print(f"Adding text to mosaic for {trial_folder}")
    os.system(ffmpeg_text_cmd)

    # Vérifier si la vidéo avec texte a été créée
    if not os.path.exists(output_with_text_path):
        print(f"Error: The video {output_with_text_path} was not created.")
        return

    # Ouvrir la vidéo pour prévisualisation
    os.system(f"start {output_with_text_path}")  # Ouvre la vidéo sur Windows

    # Écouter les touches 'y' ou 'n' pour sauvegarder ou supprimer la vidéo
    print("Press 'y' to save the video, or 'n' to delete it. (You can press these keys while the video is playing.)")

    while True:
        if keyboard.is_pressed('y'):
            print(f"Video {output_with_text_path} has been saved.")
            print("Closing the video window.")
            os.system("taskkill /im vlc.exe /f")
            os.remove(mosaic_path)  # Supprimer l'ancienne vidéo sans texte
            break
        elif keyboard.is_pressed('n'):
            print("Closing the video window.")
            os.system("taskkill /im vlc.exe /f")
            os.remove(output_with_text_path)
            os.remove(mosaic_path)
            print(f"Video {output_with_text_path} has been deleted.")
            break
        
        
        
        
        
"""
========================================================
____________Fonctions pour synchro manual_______________
========================================================
"""

# Ouvre une vidéo et retourne l'objet vidéo
def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : impossible d'ouvrir la vidéo {video_path}")
        return None
    return cap

# Récupère la taille de l'écran pour forcer la fenêtre à cette taille
def get_screen_size():
    screen = screeninfo.get_monitors()[0]
    return screen.width, screen.height

# Ajuster la taille de la fenêtre au maximum de l'écran
screen_width, screen_height = get_screen_size()

# Trouve la vidéo avec le moins de frames pour la définir comme référence
def find_reference_video(video_paths):
    min_frames = float('inf')
    reference_path = None
    for video_path in video_paths:
        video = open_video(video_path)
        if video:
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames < min_frames:
                min_frames = total_frames
                reference_path = video_path
            video.release()
    return reference_path, min_frames

# Variables globales pour le zoom et le déplacement
zoom_scale = 1.0
pan_x, pan_y = 0, 0
dragging = False
start_x, start_y = 0, 0

# Callback pour gérer les événements de la souris pour le zoom et le déplacement
def mouse_callback(event, x, y, flags, param):
    global pan_x, pan_y, dragging, start_x, start_y, zoom_scale
    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_x, start_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        # Inverse le déplacement par rapport au pointeur
        pan_x -= (x - start_x)
        pan_y -= (y - start_y)
        start_x, start_y = x, y
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom_in(x, y)
        else:
            zoom_out(x, y)


# Fonction de zoom avant sans zoom initial
def zoom_in(center_x, center_y):
    global zoom_scale, pan_x, pan_y
    zoom_scale += 0.1
    pan_x = int((pan_x + center_x) * 1.1 - center_x)
    pan_y = int((pan_y + center_y) * 1.1 - center_y)

def zoom_out(center_x, center_y):
    global zoom_scale, pan_x, pan_y
    zoom_scale = max(zoom_scale - 0.1, 0.1)
    pan_x = int((pan_x + center_x) / 1.1 - center_x)
    pan_y = int((pan_y + center_y) / 1.1 - center_y)



def preload_frames(video, center_frame, range_frames=50):
    """
    Précharge les frames autour d'une frame donnée dans un intervalle spécifié.
    
    Args:
        video: Objet cv2.VideoCapture de la vidéo.
        center_frame: Frame centrale autour de laquelle les frames sont préchargées.
        range_frames: Nombre de frames à charger avant et après la frame centrale.
    
    Returns:
        dict: Dictionnaire avec les numéros de frames comme clés et les frames comme valeurs.
    """
    preloaded_frames = {}
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Définir la plage de frames à précharger
    start_frame = max(0, center_frame - range_frames)
    end_frame = min(total_frames, center_frame + range_frames + 1)

    for i in range(start_frame, end_frame):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if ret:
            preloaded_frames[i] = frame
    return preloaded_frames



def navigate_frames(video, window_name="Select Reference Frame", start_frame=0):
    global zoom_scale, pan_x, pan_y
    zoom_scale, pan_x, pan_y = 1.0, 0, 0  # Initialiser sans zoom initial
    frame_number = start_frame
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    selected_frame = None

    # Initialiser le préchargement en arrière-plan
    preload_lock = threading.Lock()
    preload_queue = {}
    stop_preloading = threading.Event()

    def preload_frames():
        """Précharge les frames autour de la frame courante en arrière-plan."""
        while not stop_preloading.is_set():
            with preload_lock:
                preload_start = max(0, frame_number - 10)
                preload_end = min(total_frames, frame_number + 10)
                for i in range(preload_start, preload_end):
                    if i not in preload_queue:
                        video.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = video.read()
                        if ret:
                            preload_queue[i] = frame
            time.sleep(0.1)  # Pause légère pour éviter de saturer le CPU

    # Lancer le thread de préchargement
    preload_thread = threading.Thread(target=preload_frames, daemon=True)
    preload_thread.start()

    # Variables pour gérer le clic par clic et l'appui prolongé
    key_hold_time = 0.2  # Délai pour le clic par clic (en secondes)
    fast_scroll_threshold = 0.5  # Temps avant de commencer le défilement rapide (en secondes)
    fast_scroll_interval = 0.05  # Intervalle entre les défilements rapides (en secondes)

    last_key_time = time.time()  # Temps de la dernière touche appuyée
    key_pressed_time = None  # Temps du début de l'appui continu

    # Création de la fenêtre pour la sélection de la frame de référence
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, screen_width, screen_height)
    cv2.setMouseCallback(window_name, mouse_callback)  # Attacher le callback de souris

    while True:
        # Vérifier si la frame actuelle est préchargée
        with preload_lock:
            frame = preload_queue.get(frame_number, None)

        if frame is None:
            print(f"Frame {frame_number} en cours de chargement...")
            time.sleep(0.05)  # Attendre un court instant pour laisser le thread de préchargement travailler
            continue

        # Calculer le temps correspondant en secondes
        time_in_seconds = frame_number / fps

        # Appliquer le zoom et le déplacement avec affichage dynamique du texte
        zoomed_frame = apply_zoom_and_pan(frame, screen_height // 2, frame_number=frame_number, time_in_seconds=time_in_seconds)
        cv2.imshow(window_name, zoomed_frame)

        key = cv2.waitKey(1)
        current_time = time.time()

        # Gérer l'appui sur les touches
        if keyboard.is_pressed('q'):
            print("Programme arrêté par l'utilisateur.")
            break
        elif keyboard.is_pressed('space'):
            selected_frame = frame.copy()  # Capture la frame de référence
            print(f"Frame sélectionnée : {frame_number}")
            break

        # Gestion du clic par clic et de l'appui prolongé
        elif keyboard.is_pressed('left'):
            if key_pressed_time is None:
                key_pressed_time = current_time  # Début de l'appui
            elif current_time - key_pressed_time > fast_scroll_threshold:
                # Défilement rapide après un certain temps d'appui
                frame_number = max(0, frame_number - 1)
                time.sleep(fast_scroll_interval)
            elif current_time - last_key_time >= key_hold_time:
                # Clic par clic
                frame_number = max(0, frame_number - 1)
                last_key_time = current_time
        elif keyboard.is_pressed('right'):
            if key_pressed_time is None:
                key_pressed_time = current_time  # Début de l'appui
            elif current_time - key_pressed_time > fast_scroll_threshold:
                # Défilement rapide après un certain temps d'appui
                frame_number = min(total_frames - 1, frame_number + 1)
                time.sleep(fast_scroll_interval)
            elif current_time - last_key_time >= key_hold_time:
                # Clic par clic
                frame_number = min(total_frames - 1, frame_number + 1)
                last_key_time = current_time
        elif keyboard.is_pressed('up'):
            # Clic par clic
            frame_number = min(total_frames - 1, frame_number + 100)
            last_key_time = current_time
        elif keyboard.is_pressed('down'):
            frame_number = max(0, frame_number - 100)
            last_key_time = current_time
        else:
            key_pressed_time = None  # Réinitialiser si la touche n'est plus appuyée

        # Gestion des autres touches pour zoom
        if key == ord('+') or key == ord('='):
            zoom_in(screen_width // 4, screen_height // 4)
        elif key == ord('-'):
            zoom_out(screen_width // 4, screen_height // 4)

    # Arrêter le préchargement et nettoyer
    stop_preloading.set()
    preload_thread.join()
    cv2.destroyWindow(window_name)

    return selected_frame, frame_number


def calculate_min_frames(video_paths, offsets, start_frame, end_frame):
    """
    Calculer la durée minimale des vidéos après application des offsets.
    """
    min_frames = int(end_frame - start_frame)  # Calcul initial de la durée
    actual_min_frames = min_frames  # Initialisation de la durée minimale effective

    # Trouver le plus grand offset négatif
    min_offset = min(offsets.values())  # Offset le plus négatif

    for video_path in video_paths:
        video_name = os.path.basename(video_path).split('.')[0]  # Obtenir le nom sans extension
        offset = int(offsets.get(video_name, 0))  # Récupérer l'offset par vidéo
        
        # Calculer les frames disponibles après découpage initial
        frames_to_cut = abs(min_offset) + offset  # Nombre de frames à couper au début
        effective_duration = min_frames - frames_to_cut  # Durée restante
        actual_min_frames = min(actual_min_frames, effective_duration)

    return actual_min_frames



def synchronize_videos(video_paths, video_save):
    sync_data = {}

    # Demande à l'utilisateur pour un découpage personnalisé
    custom_cut = ask_for_custom_cut()

    # Initialisation du début et de la fin si un découpage personnalisé est souhaité
    start_frame, end_frame = 0, None  # Valeurs par défaut
    
    # Détermine la vidéo de référence comme celle avec le moins de frames, convertie en entier
    reference_path, min_frames = find_reference_video(video_paths)
    min_frames = int(min_frames)  # Assurer que c'est un entier
    video_paths.remove(reference_path)
    video_paths.insert(0, reference_path)  # Place la vidéo de référence en première position
    
    if custom_cut:
        reference_video = open_video(video_paths[0])  # Utilise la première vidéo comme référence pour définir l'intervalle
        if reference_video is not None:
            print(f"Sélection de la plage de découpage pour la vidéo de référence : {video_paths[0]}")
            start_frame, end_frame = select_start_end_frames(reference_video)
            reference_video.release()
            print(f"Plage définie : début à {start_frame}, fin à {end_frame}")
    else:
        # Si aucun découpage personnalisé, utiliser la dernière frame comme end_frame
        reference_video = open_video(video_paths[0])
        if reference_video is not None:
            end_frame = int(reference_video.get(cv2.CAP_PROP_FRAME_COUNT))  # Dernière frame de la vidéo
            reference_video.release()
            


    # Ajuste min_frames en fonction du découpage défini, s'il est personnalisé
    if end_frame is not None:
        min_frames = end_frame - start_frame  # Mettre à jour min_frames pour correspondre à l'intervalle personnalisé
    
    offsets = {}  # Pour stocker les offsets de chaque vidéo

    max_negative_offset = 0  # Suivi du plus grand offset négatif
    for idx, video_path in enumerate(video_paths):
        video = open_video(video_path)
        
        if video is None:
            continue

        video_name = os.path.basename(video_path).split('.')[0]

        if idx == 0:
            # La première vidéo (avec le moins de frames) devient la référence
            print(f"Définition de la frame de référence pour la vidéo : {video_path}")
            reference_window_name = "Reference Video"
            reference_frame, reference_image = navigate_frames(video, reference_window_name)
            offsets[video_name] = 0  # La caméra de référence a un décalage de 0
            if cv2.getWindowProperty(reference_window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(reference_window_name)
        else:
            # Synchronisation des autres vidéos en comparant avec la frame de référence
            print(f"Synchronisation de la vidéo {video_path} sur la frame de référence {reference_image}")
            offset = synchronize_with_reference(video, reference_frame, reference_image, video_name)
            offsets[video_name] = offset  # Stocke l'offset pour chaque vidéo

            # Mettre à jour le plus grand offset négatif
            if offset < max_negative_offset:
                max_negative_offset = offset

        video.release()

    # Calculer le début de chaque vidéo en fonction de l'offset négatif le plus important
    cut_start = abs(max_negative_offset)  # Décalage initial basé sur l'offset négatif le plus grand
    print(f"Découpage initial (cut_start) : {cut_start} frames.")

    # Calculer la durée minimale effective pour chaque vidéo
    actual_min_frames = calculate_min_frames(video_paths, offsets, start_frame, end_frame)
    print(f"Durée finale après découpage : {actual_min_frames} frames.")

    # Couper et sauvegarder chaque vidéo
    for video_path in video_paths:
        video_name = os.path.basename(video_path).split('.')[0]
        offset = offsets[video_name]
        output_path = os.path.join(video_save, f"{video_name}_synchronized.mp4")
        
        # Calculer les frames à couper au début
        start_frame_adjusted = cut_start + offset  # Décalage initial ajusté pour chaque vidéo
        final_frame_count = cut_video(video_path, output_path, start_frame_adjusted, actual_min_frames)
        
        sync_data[video_name] = {"offset": offset, "final_frame_count": final_frame_count}
        print(f"Vidéo synchronisée enregistrée pour {video_name} à {output_path}")

    # Enregistre les décalages et le nombre de frames coupées dans un fichier .toml
    if not os.path.exists(os.path.join(video_save, "debug_files")):
        os.mkdir(os.path.join(video_save, "debug_files"))
    toml_path = os.path.join(video_save, "debug_files", "synchronization_debug.toml")
    with open(toml_path, "w") as toml_file:
        toml.dump(sync_data, toml_file)
    print(f"Les décalages de synchronisation et le nombre de frames ont été enregistrés dans {toml_path}")



def synchronize_with_reference(video, reference_frame, reference_frame_number, video_name):
    global zoom_scale, pan_x, pan_y
    zoom_scale, pan_x, pan_y = 1.0, 0, 0  # Réinitialiser les valeurs de zoom et de déplacement pour la comparaison
    compare_window = "Comparison Frame"

    # Création de la fenêtre fixe pour la frame de référence sans interaction
    reference_window = "Reference Frame"
    cv2.namedWindow(reference_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(reference_window, screen_width // 2, screen_height // 2)
    cv2.moveWindow(reference_window, 0, 0)

    # Redimensionner la frame de référence pour l'adapter sans déformation
    ref_height, ref_width = reference_frame.shape[:2]
    scaling_factor = min(screen_height // 2 / ref_height, screen_width // 2 / ref_width)
    resized_reference = cv2.resize(reference_frame, (int(ref_width * scaling_factor), int(ref_height * scaling_factor)), interpolation=cv2.INTER_LINEAR)
    cv2.imshow(reference_window, resized_reference)  # Afficher la frame de référence fixe sans numéro de frame

    # Création de la fenêtre de comparaison avec zoom et déplacement
    cv2.namedWindow(compare_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(compare_window, screen_width // 2, screen_height // 2)
    cv2.moveWindow(compare_window, screen_width // 2, 0)
    cv2.setMouseCallback(compare_window, mouse_callback)  # Interaction uniquement sur la frame de comparaison

    frame_number = reference_frame_number  # Initialiser la frame actuelle à la même que la référence
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Gestion du préchargement des frames
    preload_lock = threading.Lock()
    preload_queue = {}
    stop_preloading = threading.Event()

    def preload_frames():
        while not stop_preloading.is_set():
            with preload_lock:
                preload_start = max(0, frame_number - 10)
                preload_end = min(total_frames, frame_number + 10)
                for i in range(preload_start, preload_end):
                    if i not in preload_queue:
                        video.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = video.read()
                        if ret:
                            preload_queue[i] = frame
            threading.Event().wait(0.1)  # Pause légère pour éviter la surcharge CPU

    preload_thread = threading.Thread(target=preload_frames, daemon=True)
    preload_thread.start()

    # Variables pour gérer le clic par clic et l'appui prolongé
    key_hold_time = 0.2  # Délai pour le clic par clic (en secondes)
    fast_scroll_threshold = 0.5  # Temps avant de commencer le défilement rapide (en secondes)
    fast_scroll_interval = 0.05  # Intervalle entre les défilements rapides (en secondes)

    last_key_time = time.time()  # Temps de la dernière touche appuyée
    key_pressed_time = None  # Temps du début de l'appui continu

    while True:
        with preload_lock:
            current_frame = preload_queue.get(frame_number, None)

        if current_frame is None:
            print(f"Frame {frame_number} en cours de chargement...")
            time.sleep(0.05)  # Attendre un court instant pour laisser le thread de préchargement travailler
            continue

        # Calculer le temps correspondant en secondes
        fps = video.get(cv2.CAP_PROP_FPS)
        time_in_seconds = frame_number / fps

        # Appliquer le zoom et le déplacement sur la frame actuelle
        zoomed_current = apply_zoom_and_pan(current_frame, screen_height // 2, frame_number=frame_number, time_in_seconds=time_in_seconds)

        # Afficher la frame de comparaison
        cv2.imshow(compare_window, zoomed_current)

        key = cv2.waitKey(1)
        current_time = time.time()

        # Gérer l'appui sur les touches
        if keyboard.is_pressed('q'):
            print("Programme arrêté par l'utilisateur.")
            break
        elif keyboard.is_pressed('space'):
            offset = frame_number - reference_frame_number
            print(f"{video_name} : Décalage de {offset} frames par rapport à la référence")
            break

        # Gestion du clic par clic et de l'appui prolongé
        elif keyboard.is_pressed('left'):
            if key_pressed_time is None:
                key_pressed_time = current_time  # Début de l'appui
            elif current_time - key_pressed_time > fast_scroll_threshold:
                # Défilement rapide après un certain temps d'appui
                frame_number = max(0, frame_number - 1)
                time.sleep(fast_scroll_interval)
            elif current_time - last_key_time >= key_hold_time:
                # Clic par clic
                frame_number = max(0, frame_number - 1)
                last_key_time = current_time
        elif keyboard.is_pressed('right'):
            if key_pressed_time is None:
                key_pressed_time = current_time  # Début de l'appui
            elif current_time - key_pressed_time > fast_scroll_threshold:
                # Défilement rapide après un certain temps d'appui
                frame_number = min(total_frames - 1, frame_number + 1)
                time.sleep(fast_scroll_interval)
            elif current_time - last_key_time >= key_hold_time:
                # Clic par clic
                frame_number = min(total_frames - 1, frame_number + 1)
                last_key_time = current_time
        elif keyboard.is_pressed('up'):
            # Clic par clic
            frame_number = min(total_frames - 1, frame_number + 100)
            last_key_time = current_time
        elif keyboard.is_pressed('down'):
            frame_number = max(0, frame_number - 100)
            last_key_time = current_time
        else:
            key_pressed_time = None  # Réinitialiser si la touche n'est plus appuyée

        # Gestion des autres touches pour zoom
        if key == ord('+') or key == ord('='):
            zoom_in(screen_width // 4, screen_height // 4)
        elif key == ord('-'):
            zoom_out(screen_width // 4, screen_height // 4)

    # Arrêter le préchargement
    stop_preloading.set()
    preload_thread.join()

    cv2.destroyWindow(reference_window)
    cv2.destroyWindow(compare_window)
    return offset


# Appliquer le zoom et le déplacement sur une frame sans déformation aux bords
def apply_zoom_and_pan(frame, target_height, frame_number=None, time_in_seconds=None):
    global zoom_scale, pan_x, pan_y

    # Redimensionner l'image selon le facteur de zoom
    height, width = frame.shape[:2]
    zoomed_width, zoomed_height = int(width * zoom_scale), int(height * zoom_scale)
    resized_frame = cv2.resize(frame, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)
    
    # Calculer les limites pour le déplacement sans déformer l'image aux bords
    x_start = max(0, min(pan_x, zoomed_width - width))
    y_start = max(0, min(pan_y, zoomed_height - height))
    x_end = x_start + width
    y_end = y_start + height
    displayed_frame = resized_frame[y_start:y_end, x_start:x_end]

    # Ajuste la taille pour correspondre à la fenêtre sans déformation
    scaling_factor = target_height / displayed_frame.shape[0]
    final_width = int(displayed_frame.shape[1] * scaling_factor)
    displayed_frame = cv2.resize(displayed_frame, (final_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Ajouter le numéro de frame et le temps en haut à gauche de la zone visible
    if frame_number is not None and time_in_seconds is not None:
        cv2.putText(displayed_frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        cv2.putText(displayed_frame, f"Time: {time_in_seconds:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    return displayed_frame

# Fonction pour interroger l'utilisateur en yes/no
def ask_for_custom_cut():
    response = input("Voulez-vous définir un nouveau début et une nouvelle fin ? (yes/no) : ")
    return response.lower() == "yes"

# Fonction de sélection des frames de début et de fin sur une seule vidéo
def select_start_end_frames(video):
    print("Sélection de la frame de début.")
    _, start_frame = navigate_frames(video, "Definir Frame Debut")  # Récupérer uniquement le numéro de la frame

    # Affiche directement la dernière frame pour sélectionner la fin
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, last_frame = video.read()

    if ret:
        print("Sélection de la frame de fin.")
        _, end_frame = navigate_frames(video, "Definir Frame Fin", start_frame=total_frames - 1)  # Récupérer uniquement le numéro de la frame

    return start_frame, end_frame


# Couper une vidéo en fonction de l'offset et de la durée minimale, avec une barre de progression
def cut_video(video_path, output_path, offset, min_frames):
    video = open_video(video_path)
    if video is None:
        return 0

    # Obtenir les propriétés de la vidéo
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Définir le writer pour enregistrer la vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Aller à la première frame nécessaire en fonction de l'offset
    start_frame = max(0, offset)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Initialiser le compteur de frames et la barre de progression
    frame_count = 0
    with tqdm(total=min_frames, desc=f"Enregistrement de {output_path}", unit="frame") as pbar:
        for i in range(min_frames):
            ret, frame = video.read()
            if not ret:
                break
            out.write(frame)
            frame_count += 1
            pbar.update(1)  # Mise à jour de la barre de progression
    
    # Libérer les ressources
    video.release()
    out.release()

    return frame_count  # Retourne le nombre de frames écrites
        
        
"""
==========================================================================
_____________________________Fonction principale__________________________
=========================================================================="""
        

def synchronize_cams_all(level, config_dict):
    '''
    Post-synchronize your cameras in case they are not natively synchronized.

    For each camera, computes mean vertical speed for the chosen keypoints, 
    and find the time offset for which their correlation is highest. 

    Depending on the analysed motion, all keypoints can be taken into account, 
    or a list of them, or the right or left side.
    All frames can be considered, or only those around a specific time (typically, 
    the time when there is a single participant in the scene performing a clear vertical motion).
    Has also been successfully tested for synchronizing random walkswith random walks.

    Keypoints whose likelihood is too low are filtered out; and the remaining ones are 
    filtered with a butterworth filter.

    INPUTS: 
    - json files from each camera folders
    - a Config.toml file
    - a skeleton model

    OUTPUTS: 
    - synchronized json files for each camera
    '''
    
    # Get parameters from Config.toml
    project_dir = config_dict.get('project').get('project_dir')
    pose_dir = os.path.realpath(os.path.join(project_dir, 'pose'))
    pose_model = config_dict.get('pose').get('pose_model')
    multi_person = config_dict.get('project').get('multi_person')
    fps =  config_dict.get('project').get('frame_rate')
    frame_range = config_dict.get('project').get('frame_range')
    synchronization_type = config_dict.get('synchronization').get('synchronization_type')
    display_sync_plots = config_dict.get('synchronization').get('display_sync_plots')
    keypoints_to_consider = config_dict.get('synchronization').get('keypoints_to_consider')
    approx_time_maxspeed = config_dict.get('synchronization').get('approx_time_maxspeed') 
    time_range_around_maxspeed = config_dict.get('synchronization').get('time_range_around_maxspeed')

    likelihood_threshold = config_dict.get('synchronization').get('likelihood_threshold')
    filter_cutoff = int(config_dict.get('synchronization').get('filter_cutoff'))
    filter_order = int(config_dict.get('synchronization').get('filter_order'))

    # Determine frame rate
    video_dir = os.path.join(project_dir, 'videos_raw')
    vid_img_extension = config_dict['pose']['vid_img_extension']
    video_files = glob.glob(os.path.join(video_dir, '*'+vid_img_extension))

    if fps == 'auto': 
        try:
            cap = cv2.VideoCapture(video_files[0])
            cap.read()
            if cap.read()[0] == False:
                raise
            fps = int(cap.get(cv2.CAP_PROP_FPS))
        except:
            fps = 60  
    lag_range = time_range_around_maxspeed*fps # frames

    if synchronization_type == 'move':
        # Warning if multi_person
        if multi_person:
            logging.warning('\nYou set your project as a multi-person one: make sure you set `approx_time_maxspeed` and `time_range_around_maxspeed` at times where one single person is in the scene, or you may get inaccurate results.')
            do_synchro = input('Do you want to continue? (y/n)')
            if do_synchro.lower() not in ["y","yes"]:
                logging.warning('Synchronization cancelled.')
                return
            else:
                logging.warning('Synchronization will be attempted.\n')
        
        # Retrieve keypoints from model
        try: # from skeletons.py
            model = eval(pose_model)
        except:
            try: # from Config.toml
                model = DictImporter().import_(config_dict.get('pose').get(pose_model))
                if model.id == 'None':
                    model.id = None
            except:
                raise NameError('Model not found in skeletons.py nor in Config.toml')
        keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
        keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]
    
        # List json files
        try:
            pose_listdirs_names = next(os.walk(pose_dir))[1]
            os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
        except:
            raise ValueError(f'No json files found in {pose_dir} subdirectories. Make sure you run Pose2Sim.poseEstimation() first.')
        pose_listdirs_names = sort_stringlist_by_last_number(pose_listdirs_names)
        json_dirs_names = [k for k in pose_listdirs_names if 'json' in k]
        json_dirs = [os.path.join(pose_dir, j_d) for j_d in json_dirs_names] # list of json directories in pose_dir
        json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names]
        json_files_names = [sort_stringlist_by_last_number(j) for j in json_files_names]
        nb_frames_per_cam = [len(fnmatch.filter(os.listdir(os.path.join(json_dir)), '*.json')) for json_dir in json_dirs]
        cam_nb = len(json_dirs)
        cam_list = list(range(cam_nb))
        
        # frame range selection
        f_range = [[0, min([len(j) for j in json_files_names])] if frame_range==[] else frame_range][0]
        # json_files_names = [[j for j in json_files_cam if int(re.split(r'(\d+)',j)[-2]) in range(*f_range)] for json_files_cam in json_files_names]
    
        # Determine frames to consider for synchronization
        if isinstance(approx_time_maxspeed, list): # search around max speed
            approx_frame_maxspeed = [int(fps * t) for t in approx_time_maxspeed]
            nb_frames_per_cam = [len(fnmatch.filter(os.listdir(os.path.join(json_dir)), '*.json')) for json_dir in json_dirs]
            search_around_frames = [[int(a-lag_range) if a-lag_range>0 else 0, int(a+lag_range) if a+lag_range<nb_frames_per_cam[i] else nb_frames_per_cam[i]+f_range[0]] for i,a in enumerate(approx_frame_maxspeed)]
            logging.info(f'Synchronization is calculated around the times {approx_time_maxspeed} +/- {time_range_around_maxspeed} s.')
        elif approx_time_maxspeed == 'auto': # search on the whole sequence (slower if long sequence)
            search_around_frames = [[f_range[0], f_range[0]+nb_frames_per_cam[i]] for i in range(cam_nb)]
            logging.info('Synchronization is calculated on the whole sequence. This may take a while.')
        else:
            raise ValueError('approx_time_maxspeed should be a list of floats or "auto"')
        
        if keypoints_to_consider == 'right':
            logging.info(f'Keypoints used to compute the best synchronization offset: right side.')
        elif keypoints_to_consider == 'left':
            logging.info(f'Keypoints used to compute the best synchronization offset: left side.')
        elif isinstance(keypoints_to_consider, list):
            logging.info(f'Keypoints used to compute the best synchronization offset: {keypoints_to_consider}.')
        elif keypoints_to_consider == 'all':
            logging.info(f'All keypoints are used to compute the best synchronization offset.')
        logging.info(f'These keypoints are filtered with a Butterworth filter (cut-off frequency: {filter_cutoff} Hz, order: {filter_order}).')
        logging.info(f'They are removed when their likelihood is below {likelihood_threshold}.\n')
    
        # Extract, interpolate, and filter keypoint coordinates
        logging.info('Synchronizing...')
        df_coords = []
        b, a = signal.butter(filter_order/2, filter_cutoff/(fps/2), 'low', analog = False) 
        json_files_names_range = [[j for j in json_files_cam if int(re.split(r'(\d+)',j)[-2]) in range(*frames_cam)] for (json_files_cam, frames_cam) in zip(json_files_names,search_around_frames)]
        json_files_range = [[os.path.join(pose_dir, j_dir, j_file) for j_file in json_files_names_range[j]] for j, j_dir in enumerate(json_dirs_names)]
        
        if np.array([j==[] for j in json_files_names_range]).any():
            raise ValueError(f'No json files found within the specified frame range ({frame_range}) at the times {approx_time_maxspeed} +/- {time_range_around_maxspeed} s.')
        
        for i in range(cam_nb):
            df_coords.append(convert_json2pandas(json_files_range[i], likelihood_threshold=likelihood_threshold))
            df_coords[i] = drop_col(df_coords[i],3) # drop likelihood
            if keypoints_to_consider == 'right':
                kpt_indices = [i for i,k in zip(keypoints_ids,keypoints_names) if k.startswith('R') or k.startswith('right')]
                kpt_indices = np.sort(np.concatenate([np.array(kpt_indices)*2, np.array(kpt_indices)*2+1]))
                df_coords[i] = df_coords[i][kpt_indices]
            elif keypoints_to_consider == 'left':
                kpt_indices = [i for i,k in zip(keypoints_ids,keypoints_names) if k.startswith('L') or k.startswith('left')]
                kpt_indices = np.sort(np.concatenate([np.array(kpt_indices)*2, np.array(kpt_indices)*2+1]))
                df_coords[i] = df_coords[i][kpt_indices]
            elif isinstance(keypoints_to_consider, list):
                kpt_indices = [i for i,k in zip(keypoints_ids,keypoints_names) if k in keypoints_to_consider]
                kpt_indices = np.sort(np.concatenate([np.array(kpt_indices)*2, np.array(kpt_indices)*2+1]))
                df_coords[i] = df_coords[i][kpt_indices]
            elif keypoints_to_consider == 'all':
                pass
            else:
                raise ValueError('keypoints_to_consider should be "all", "right", "left", or a list of keypoint names.\n\
                                If you specified keypoints, make sure that they exist in your pose_model.')
            
            df_coords[i] = df_coords[i].apply(interpolate_zeros_nans, axis=0, args = ['linear'])
            df_coords[i] = df_coords[i].bfill().ffill()
            df_coords[i] = pd.DataFrame(signal.filtfilt(b, a, df_coords[i], axis=0))
    
    
        # Compute sum of speeds
        df_speed = []
        sum_speeds = []
        for i in range(cam_nb):
            df_speed.append(vert_speed(df_coords[i]))
            sum_speeds.append(abs(df_speed[i]).sum(axis=1))
            # nb_coord = df_speed[i].shape[1]
            # sum_speeds[i][ sum_speeds[i]>vmax*nb_coord ] = 0
            
            # # Replace 0 by random values, otherwise 0 padding may lead to unreliable correlations
            # sum_speeds[i].loc[sum_speeds[i] < 1] = sum_speeds[i].loc[sum_speeds[i] < 1].apply(lambda x: np.random.normal(0,1))
            
            sum_speeds[i] = pd.DataFrame(signal.filtfilt(b, a, sum_speeds[i], axis=0)).squeeze()
    
    
        # Compute offset for best synchronization:
        # Highest correlation of sum of absolute speeds for each cam compared to reference cam
        ref_cam_id = nb_frames_per_cam.index(min(nb_frames_per_cam)) # ref cam: least amount of frames
        ref_frame_nb = len(df_coords[ref_cam_id])
        lag_range = int(ref_frame_nb/2)
        cam_list.pop(ref_cam_id)
        offset = []
        for cam_id in cam_list:
            offset_cam_section, max_corr_cam = time_lagged_cross_corr(sum_speeds[ref_cam_id], sum_speeds[cam_id], lag_range, show=display_sync_plots, ref_cam_id=ref_cam_id, cam_id=cam_id)
            offset_cam = offset_cam_section - (search_around_frames[ref_cam_id][0] - search_around_frames[cam_id][0])
            if isinstance(approx_time_maxspeed, list):
                logging.info(f'--> Camera {ref_cam_id} and {cam_id}: {offset_cam} frames offset ({offset_cam_section} on the selected section), correlation {round(max_corr_cam, 2)}.')
            else:
                logging.info(f'--> Camera {ref_cam_id} and {cam_id}: {offset_cam} frames offset, correlation {round(max_corr_cam, 2)}.')
            offset.append(offset_cam)
        offset.insert(ref_cam_id, 0)
    
        # rename json files according to the offset and copy them to pose-sync
        sync_dir = os.path.abspath(os.path.join(pose_dir, '..', 'pose-sync'))
        os.makedirs(sync_dir, exist_ok=True)
        for d, j_dir in enumerate(json_dirs):
            os.makedirs(os.path.join(sync_dir, os.path.basename(j_dir)), exist_ok=True)
            for j_file in json_files_names[d]:
                j_split = re.split(r'(\d+)',j_file)
                j_split[-2] = f'{int(j_split[-2])-offset[d]:06d}'
                if int(j_split[-2]) > 0:
                    json_offset_name = ''.join(j_split)
                    shutil.copy(os.path.join(pose_dir, os.path.basename(j_dir), j_file), os.path.join(sync_dir, os.path.basename(j_dir), json_offset_name))
    
        # if display_sync_plots== True:
        #     sync_video_folder_path = os.path.join(project_dir,"videos")
        #     synchroMosaique(sync_video_folder_path)
            
        logging.info(f'Synchronized json files saved in {sync_dir}.')

    elif synchronization_type == 'sound':
        logging.info("====================")
        print("Videos Synchronization...")
        logging.info("--------------------\n\n")  
        
        if level==1:
            #Récupération des chemins de l'essai à traiter
            path_folder=os.path.dirname(project_dir)
            
            raw_video_folder_path = Path(os.path.join(project_dir, 'videos_raw'))
            sync_video_folder_path = Path(os.path.join(project_dir, 'videos'))        
   
            #Si dossier de vidéos synchronisées vide ou inexistant...
            if not os.path.exists(sync_video_folder_path) or os.listdir(sync_video_folder_path)==[] :
                
                #Création du dossier si inexistant
                if not os.path.exists(os.path.join(path_folder,"videos")):os.mkdir(os.path.join(path_folder,"videos"))
                
                #Fonction de synchronisation
                sync.synchronize_videos_from_audio(raw_video_folder_path=raw_video_folder_path,
                                                    synchronized_video_folder_path = sync_video_folder_path,
                                                    video_handler="deffcode",
                                                    create_debug_plots_bool=display_sync_plots)
                if display_sync_plots== True:
                    synchroMosaique(sync_video_folder_path)
            
        if level==2:    
            #Récupération du path de travail général
            path_folder=os.path.dirname(project_dir)

            folders = os.listdir(path_folder)
            
            #Récupération du nombre et des noms des essais à traiter
            nbtrials=0
            trialname = []
            for i in range(0,len(folders)):
                if "Trial" in folders[i]:
                    nbtrials += 1
                    trialname.append(folders[i])
    

            for trial in range(nbtrials) :
                
                #Récupération des chemins de l'essai à traiter
                raw_video_folder_path = Path(os.path.join(path_folder,trialname[trial],"videos_raw"))
                sync_video_folder_path = Path(os.path.join(path_folder,trialname[trial],"videos"))        
       
                #Si dossier de vidéos synchronisées vide ou inexistant...
                if not os.path.exists(sync_video_folder_path) or os.listdir(sync_video_folder_path)==[] :
                    
                    #Création du dossier si inexistant
                    if not os.path.exists(os.path.join(path_folder,"Trial_"+str(trial+1),"videos")):os.mkdir(os.path.join(path_folder,"Trial_"+str(trial+1),"videos"))
                    
                    #Fonction de synchronisation
                    sync.synchronize_videos_from_audio(raw_video_folder_path=raw_video_folder_path,
                                                        synchronized_video_folder_path = sync_video_folder_path,
                                                        video_handler="deffcode",
                                                        create_debug_plots_bool=display_sync_plots)
                    if display_sync_plots== True:
                        synchroMosaique(sync_video_folder_path)
        
        logging.info("Vidéos synchronisées avec succès.")
        
    elif synchronization_type == 'manual':
        logging.info("====================")
        print("Videos Synchronization...")
        logging.info("--------------------\n\n")  
        
        if level==1:
            #Récupération des chemins de l'essai à traiter
            path_folder=os.path.dirname(project_dir)
            
            raw_video_folder_path = Path(os.path.join(project_dir, 'videos_raw'))
            sync_video_folder_path = Path(os.path.join(project_dir, 'videos'))      
            
            video_files = sorted([os.path.join(raw_video_folder_path, f) for f in os.listdir(raw_video_folder_path) if f.lower().endswith(".mp4")])

   
            #Si dossier de vidéos synchronisées vide ou inexistant...
            if not os.path.exists(sync_video_folder_path) or os.listdir(sync_video_folder_path)==[] :
                
                #Création du dossier si inexistant
                if not os.path.exists(os.path.join(path_folder,"videos")):os.mkdir(os.path.join(path_folder,"videos"))
                
                #Fonction de synchronisation
                synchronize_videos(video_files,sync_video_folder_path)
                cv2.destroyAllWindows()
                
                if display_sync_plots== True:
                    synchroMosaique(sync_video_folder_path)
            
        if level==2:    
            #Récupération du path de travail général
            path_folder=os.path.dirname(project_dir)

            folders = os.listdir(path_folder)
            
            #Récupération du nombre et des noms des essais à traiter
            nbtrials=0
            trialname = []
            for i in range(0,len(folders)):
                if "Trial" in folders[i]:
                    nbtrials += 1
                    trialname.append(folders[i])
    

            for trial in range(nbtrials) :
                
                #Récupération des chemins de l'essai à traiter
                raw_video_folder_path = Path(os.path.join(path_folder,trialname[trial],"videos_raw"))
                sync_video_folder_path = Path(os.path.join(path_folder,trialname[trial],"videos"))        
                
                video_files = sorted([os.path.join(raw_video_folder_path, f) for f in os.listdir(raw_video_folder_path) if f.endswith(".MP4")])

                #Si dossier de vidéos synchronisées vide ou inexistant...
                if not os.path.exists(sync_video_folder_path) or os.listdir(sync_video_folder_path)==[] :
                    
                    #Création du dossier si inexistant
                    if not os.path.exists(os.path.join(path_folder,"Trial_"+str(trial+1),"videos")):os.mkdir(os.path.join(path_folder,"Trial_"+str(trial+1),"videos"))
                    
                    #Fonction de synchronisation
                    synchronize_videos(video_files,sync_video_folder_path)
                    cv2.destroyAllWindows()
                    
                    if display_sync_plots== True:
                        synchroMosaique(sync_video_folder_path)
        
        logging.info("Vidéos synchronisées avec succès.")        
        
        
    else: 
        logging.info("Vidéos déjà synchronisées.")
    
    logging.info("\n\n--------------------")
    logging.info("Synchronisation des vidéos faites avec succès.")
    logging.info("====================")