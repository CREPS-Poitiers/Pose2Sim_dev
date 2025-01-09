# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 17:06:17 2024

@author: fdelap01
"""
import cv2
import os
import argparse

def play_and_cut_videos_in_folder(folder_path):
    # Récupérer toutes les vidéos du dossier (filtrées par extension)
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mov')]
    if not video_files:
        print(f"Aucune video trouvee dans le dossier : {folder_path}")
        return

    # Sélectionner une vidéo pour définir les frames de découpage
    reference_video = os.path.join(folder_path, video_files[0])
    print(f"Video de reference : {reference_video}")
    
    # Sélectionner les frames de découpage
    start_frame, end_frame, fps, frame_width, frame_height = play_and_select_frames(reference_video)

    # Vérifier si des frames valides ont été sélectionnées
    if start_frame is None or end_frame is None:
        print("Aucune decoupe effectuee.")
        return

    # Découper toutes les vidéos dans le dossier avec les mêmes frames
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        output_path = os.path.join(folder_path, f"Cut_{video_file}")
        save_cut_video(video_path, output_path, start_frame, end_frame, fps, frame_width, frame_height)

def play_and_select_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible de lire la video {video_path}")
        return None, None, None, None, None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Récupérer les dimensions de l'écran pour ajuster la taille de la vidéo
    screen_width = 1280
    screen_height = 720
    scale = min(screen_width / frame_width, screen_height / frame_height)
    resized_width = int(frame_width * scale)
    resized_height = int(frame_height * scale)

    # Définir les styles de texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_frame = 1
    font_scale_text = 0.5
    font_color_frame = (0, 255, 0)
    font_color_text = (255, 255, 255)
    thickness_frame = 2
    thickness_text = 1

    start_frame = None
    end_frame = None
    paused = False
    current_frame = 0

    instructions = [
        "Espace : Pause/Reprendre",
        "S : Selectionner la frame de debut",
        "E : Selectionner la frame de fin",
        "V : Valider la selection",
        "Fleche droite : Avancer de 100 frames",
        "Fleche gauche : Reculer de 100 frames",
        "Q : Quitter sans sauvegarder"
    ]

    while True:
        if not paused:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                current_frame = 0  # Revenir au début si la fin est atteinte
                cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = cap.read()

            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Redimensionner la frame pour qu'elle s'adapte à l'écran
        resized_frame = cv2.resize(frame, (resized_width, resized_height))

        # Ajouter le numéro de la frame
        cv2.putText(resized_frame, f"Frame: {current_frame}/{total_frames}", (10, 50),
                    font, font_scale_frame, font_color_frame, thickness_frame)

        # Ajouter les consignes
        for i, text in enumerate(instructions):
            cv2.putText(resized_frame, text, (10, 100 + i * 20),
                        font, font_scale_text, font_color_text, thickness_text)

        # Afficher la vidéo
        cv2.imshow("Video Player", resized_frame)

        # Gérer les commandes clavier
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q'):  # Quitter
            print("Quitter sans sauvegarder.")
            cap.release()
            cv2.destroyAllWindows()
            return None, None, None, None, None
        elif key == ord(' '):  # Pause/Reprendre
            paused = not paused
        elif key == ord('s'):  # Sélectionner la frame de début
            start_frame = current_frame
            print(f"Frame de debut selectionnee : {start_frame}")
        elif key == ord('e'):  # Sélectionner la frame de fin
            end_frame = current_frame
            print(f"Frame de fin selectionnee : {end_frame}")
        elif key == ord('v'):  # Valider la sélection
            if start_frame is not None and end_frame is not None and start_frame < end_frame:
                print("Selection validee.")
                cap.release()
                cv2.destroyAllWindows()
                return start_frame, end_frame, fps, frame_width, frame_height
            else:
                print("Erreur : Selection invalide. Assurez-vous que les frames de debut et de fin sont definies et coherentes.")
        elif key == 2555904:  # Flèche droite (avancer de 100 frames)
            current_frame = min(current_frame + 100, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            print(f"Avance a la frame : {current_frame}")
        elif key == 2424832:  # Flèche gauche (reculer de 100 frames)
            current_frame = max(current_frame - 100, 0)
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            print(f"Recule a la frame : {current_frame}")

        # Vérifier si la fenêtre a été fermée
        if cv2.getWindowProperty("Video Player", cv2.WND_PROP_VISIBLE) < 1:
            print("Fenetre fermee. Arret du programme.")
            cap.release()
            cv2.destroyAllWindows()
            return None, None, None, None, None

    cap.release()
    cv2.destroyAllWindows()
    return start_frame, end_frame, fps, frame_width, frame_height

def save_cut_video(video_path, output_path, start_frame, end_frame, fps, frame_width, frame_height):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur : Impossible de lire la video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"Sauvegarde de la video de la frame {start_frame} a {end_frame}...")

    for i in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video coupee sauvegardee sous : {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Découper des vidéos dans un dossier avec les mêmes frames.")
    parser.add_argument("folder_path", type=str, help="Chemin du dossier contenant les videos .mp4 a traiter")
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"Erreur : Le chemin specifie n'est pas un dossier valide : {args.folder_path}")
    else:
        play_and_cut_videos_in_folder(args.folder_path)
