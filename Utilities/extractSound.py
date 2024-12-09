# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:40:50 2024

@author: fdelap01
"""

import os
from moviepy.editor import VideoFileClip

def extract_audio_from_videos(folder_path):
    # Parcourt tous les fichiers du dossier
    for filename in os.listdir(folder_path):
        if filename.endswith(".MP4"):  # Vérifie si le fichier est une vidéo MP4
            video_path = os.path.join(folder_path, filename)
            audio_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.mp3")
            
            # Charge la vidéo et extrait l'audio
            with VideoFileClip(video_path) as video:
                audio = video.audio
                if audio is not None:
                    # Enregistre l'audio en MP3
                    audio.write_audiofile(audio_path, codec='mp3')
                    print(f"Audio extrait pour : {filename}")
                else:
                    print(f"Aucun audio trouvé dans : {filename}")
        else:
            print(f"Fichier ignoré (non MP4) : {filename}")

# Chemin du dossier contenant les vidéos
folder_path = r"C:\Videos_Markerless\Manip_gymnase_traité\Trial_12\videos_raw"
extract_audio_from_videos(folder_path)
