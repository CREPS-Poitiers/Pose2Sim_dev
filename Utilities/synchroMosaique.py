# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 13:19:01 2024

@author: fdelaplace
"""
import shutil
import os
from pathlib import Path
import time
from datetime import datetime
import cv2
import logging
import ffmpeg
import keyboard
import re

def extract_camera_number(filename):
    """Extract the camera number from the filename, assuming format like CAMERA01.mp4"""
    match = re.search(r"CAMERA(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0  # Si le nom n'a pas de numéro de caméra identifiable

def synchroMosaique(trial_folder):
    # Récupérer les fichiers vidéos .MP4 et les trier par numéro de caméra
    video_files = [f for f in os.listdir(trial_folder) if f.endswith(".mp4")]
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
               
                # Première ligne pour 'CAMERA X'
                text_commands += f"drawtext=text='CAMERA {vidInc + 1}':fontcolor=black:fontsize=24:x={xpos}:y={ypos},"
               
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

# Appeler la fonction en collant le chemin d'accès
synchroMosaique(r"C:\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 3_traitement\Trial_1")
