# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:42:14 2024

@author: fdelap01
"""
import cv2
import os
import keyboard
import screeninfo
import toml
from tqdm import tqdm

# Ouvre une vidéo et retourne l'objet vidéo
def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : impossible d'ouvrir la vidéo {video_path}")
        return None
    return cap

# Récupère la taille de l'écran pour redimensionner les vidéos affichées
def get_screen_size():
    screen = screeninfo.get_monitors()[0]
    return screen.width, screen.height

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

# Navigue dans une vidéo pour définir la frame de référence
def navigate_frames(video, window_name="Video", start_frame=0, max_width=640):
    frame_number = start_frame
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)  # Récupère les FPS pour calculer le temps
    selected_frame = None
    
    while True:
        # Aller à la frame spécifiée
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = video.read()
        
        if not ret:
            print("Fin de la vidéo atteinte ou erreur de lecture")
            break
        
        # Calculer le temps correspondant en secondes
        time_in_seconds = frame_number / fps
        # Affichage du texte en blanc avec des dimensions plus grandes
        cv2.putText(frame, f"Frame: {frame_number}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        cv2.putText(frame, f"Time: {time_in_seconds:.2f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)
        
        # Redimensionne si la vidéo dépasse la largeur maximale autorisée
        scaling_factor = max_width / frame.shape[1]
        frame_resized = cv2.resize(frame, (max_width, int(frame.shape[0] * scaling_factor)), interpolation=cv2.INTER_LINEAR)
        
        # Affiche la frame redimensionnée
        cv2.imshow(window_name, frame_resized)
        
        cv2.waitKey(0)
        
        # Commandes pour naviguer instantanément
        if keyboard.is_pressed('q'):
            print("Programme arrêté par l'utilisateur.")
            exit(0)
        elif keyboard.is_pressed('space'):
            selected_frame = frame_number
            print(f"Frame sélectionnée : {frame_number}")
            break
        elif keyboard.is_pressed('left'):
            frame_number = max(0, frame_number - 1)
        elif keyboard.is_pressed('right'):
            frame_number = min(total_frames - 1, frame_number + 1)
        elif keyboard.is_pressed('up'):
            frame_number = min(total_frames - 1, frame_number + 100)
        elif keyboard.is_pressed('down'):
            frame_number = max(0, frame_number - 100)

    return selected_frame, frame_resized

# Synchronise les vidéos, coupe en fonction de la vidéo de référence, et enregistre les décalages dans un fichier .toml
def synchronize_videos(video_paths, video_save):
    screen_width, screen_height = get_screen_size()
    max_width = screen_width // 2 - 20
    sync_data = {}

    # Détermine la vidéo de référence comme celle avec le moins de frames
    reference_path, min_frames = find_reference_video(video_paths)
    video_paths.remove(reference_path)
    video_paths.insert(0, reference_path)  # Place la vidéo de référence en première position

    for idx, video_path in enumerate(video_paths):
        video = open_video(video_path)
        
        if video is None:
            continue

        video_name = os.path.basename(video_path).split('.')[0]

        if idx == 0:
            # La première vidéo (avec le moins de frames) devient la référence
            print(f"Définition de la frame de référence pour la vidéo : {video_path}")
            reference_frame, reference_image = navigate_frames(video, "Référence Vidéo", max_width=max_width)
            sync_data[video_name] = {"offset": 0, "final_frame_count": min_frames}  # La caméra de référence a un décalage de 0
            cv2.destroyWindow("Référence Vidéo")
        else:
            # Synchronisation des autres vidéos en comparant avec la frame de référence
            print(f"Synchronisation de la vidéo {video_path} sur la frame de référence {reference_frame}")
            offset = synchronize_with_reference(video, reference_frame, reference_image, max_width, video_name)
            sync_data[video_name] = {"offset": offset}
        
        video.release()

    # Couper et sauvegarder chaque vidéo en fonction du décalage et de la durée de la vidéo de référence
    for video_path in video_paths:
        video_name = os.path.basename(video_path).split('.')[0]
        offset = sync_data[video_name]["offset"]
        output_path = os.path.join(video_save, f"{video_name}_synchronized.mp4")
        
        # Coupe la vidéo et enregistre le nombre de frames coupées dans `sync_data`
        final_frame_count = cut_video(video_path, output_path, offset, min_frames)
        sync_data[video_name]["final_frame_count"] = final_frame_count
        print(f"Vidéo synchronisée enregistrée pour {video_name} à {output_path}")

    # Enregistre les décalages et le nombre de frames coupées dans un fichier .toml dans le dossier `video_save`
    toml_path = os.path.join(video_save, "synchronization_data.toml")
    with open(toml_path, "w") as toml_file:
        toml.dump(sync_data, toml_file)
    print(f"Les décalages de synchronisation et le nombre de frames ont été enregistrés dans {toml_path}")

# Synchronise une vidéo avec la frame de référence et renvoie le décalage en frames
def synchronize_with_reference(video, reference_frame, reference_image, max_width, video_name):
    frame_number = reference_frame
    window_name = "Synchronisation"
    offset = 0  # Écart par rapport à la référence

    while True:
        # Aller à la frame actuelle dans la vidéo
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, current_frame = video.read()
        
        if not ret:
            print("Fin de la vidéo atteinte ou erreur de lecture")
            break
        
        # Affiche le numéro de la frame actuelle et le temps
        fps = video.get(cv2.CAP_PROP_FPS)
        time_in_seconds = frame_number / fps
        cv2.putText(current_frame, f"Frame: {frame_number}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        cv2.putText(current_frame, f"Time: {time_in_seconds:.2f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 5)
        
        # Redimensionne la vidéo actuelle pour la même largeur que l'image de référence
        scaling_factor = max_width / current_frame.shape[1]
        current_resized = cv2.resize(current_frame, (max_width, int(current_frame.shape[0] * scaling_factor)), interpolation=cv2.INTER_LINEAR)

        # Combine les images de référence et la vidéo actuelle
        combined_frame = cv2.hconcat([reference_image, current_resized])
        cv2.imshow(window_name, combined_frame)
        
        # Commandes de navigation
        cv2.waitKey(0)
        
        if keyboard.is_pressed('q'):
            print("Programme arrêté par l'utilisateur.")
            exit(0)
        elif keyboard.is_pressed('space'):
            offset = frame_number - reference_frame  # Calcul de l'écart par rapport à la référence
            print(f"{video_name} : Décalage de {offset} frames par rapport à la référence")
            break
        elif keyboard.is_pressed('left'):
            frame_number = max(0, frame_number - 1)
        elif keyboard.is_pressed('right'):
            frame_number = min(int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, frame_number + 1)
        elif keyboard.is_pressed('up'):
            frame_number = min(int(video.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, frame_number + 100)
        elif keyboard.is_pressed('down'):
            frame_number = max(0, frame_number - 100)

    cv2.destroyWindow(window_name)
    return offset  # Retourne le décalage en frames pour l'enregistrement dans le fichier


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

# Utilisation
video_folder = r"C:\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 1_traitement\Trial_2\videos_raw"
video_save = r"C:\Videos_Markerless\Manip_3x3_ext_24-10-2024\Manip 1_traitement\Trial_2\videos"
video_files = sorted([os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.endswith(".MP4")])

synchronize_videos(video_files, video_save)
cv2.destroyAllWindows()
