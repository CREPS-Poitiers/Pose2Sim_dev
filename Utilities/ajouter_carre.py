# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 16:29:00 2024

@author: fdelap01
"""

import cv2
import os
import argparse
from tkinter import Tk, Canvas
from PIL import Image, ImageTk

# Variables globales
rectangles = []  # Liste des rectangles
current_start = None
drawing = False

def start_draw(event):
    """Début du dessin d'un rectangle."""
    global current_start, drawing
    current_start = (event.x, event.y)  # Sauvegarde la position initiale
    drawing = True

def update_draw(event):
    """Mettre à jour le rectangle en cours."""
    global rectangles, current_start
    if drawing:
        canvas.delete("preview")
        canvas.create_rectangle(
            current_start[0],
            current_start[1],
            event.x,
            event.y,
            outline="red",
            width=2,
            tag="preview",
        )

def end_draw(event):
    """Ajouter le rectangle à la liste une fois terminé."""
    global rectangles, current_start, drawing
    if drawing:
        rectangles.append((current_start[0], current_start[1], event.x, event.y))
        canvas.create_rectangle(
            current_start[0],
            current_start[1],
            event.x,
            event.y,
            outline="red",
            width=2,
            tag=f"rect_{len(rectangles)}",
        )
        canvas.delete("preview")
        current_start = None
        drawing = False

def remove_last(event):
    """Supprimer le dernier rectangle ajouté."""
    global rectangles
    if rectangles:
        rectangles.pop()  # Supprime le dernier rectangle
        canvas.delete(f"rect_{len(rectangles) + 1}")  # Supprime visuellement

def process_videos(folder_path):
    # Obtenir tous les fichiers .mp4 et .MP4
    video_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.mp4')]

    if not video_files:
        print("Aucune vidéo trouvée dans le dossier.")
        return

    # Parcourir chaque fichier vidéo
    for video_file in video_files:
        video_path = os.path.join(folder_path, video_file)
        output_path = os.path.join(folder_path, f"HideObj_{video_file}")

        # Charger la première frame avec OpenCV
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if not ret:
            print(f"Erreur : Impossible de lire la vidéo {video_file}.")
            cap.release()
            continue

        # Dimensions de l'image
        image_height, image_width, _ = frame.shape

        # Interface graphique Tkinter
        global rectangles
        rectangles = []  # Réinitialiser les rectangles pour chaque vidéo
        root = Tk()
        root.title(f"Sélectionnez les rectangles pour {video_file}")

        # Taille de la fenêtre en largeur
        window_width = root.winfo_screenwidth()  # Largeur de l'écran
        scale = window_width / image_width  # Calcul du facteur d'échelle pour ajuster à la largeur

        # Redimensionner l'image
        resized_width = int(image_width * scale)
        resized_height = int(image_height * scale)

        # Convertir la frame en une image compatible avec Tkinter
        resized_frame = cv2.resize(frame, (resized_width, resized_height))
        resized_image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        photo_image = ImageTk.PhotoImage(resized_image)

        # Créer une fenêtre de la largeur de l'écran et afficher l'image
        global canvas
        canvas = Canvas(root, width=resized_width, height=resized_height)
        canvas.pack()
        canvas.create_image(0, 0, anchor="nw", image=photo_image)

        # Associer les événements
        canvas.bind("<Button-1>", start_draw)  # Début du rectangle
        canvas.bind("<B1-Motion>", update_draw)  # Mise à jour pendant le glissement
        canvas.bind("<ButtonRelease-1>", end_draw)  # Fin du rectangle
        canvas.bind("<Button-3>", remove_last)  # Clic droit pour supprimer le dernier rectangle

        # Lancer la boucle Tkinter
        root.mainloop()

        # Vérifier si des rectangles ont été définis
        if not rectangles:
            print(f"Pas de rectangle défini pour {video_file}, passage à la suivante.")
            cap.release()
            continue

        # Redimensionner les coordonnées des rectangles pour la vidéo originale
        scaled_rectangles = [
            (
                int(x1 / scale),
                int(y1 / scale),
                int(x2 / scale),
                int(y2 / scale),
            )
            for (x1, y1, x2, y2) in rectangles
        ]

        print(f"Rectangles définis (échelle native) : {scaled_rectangles}")

        # Réinitialiser la capture vidéo et préparer la sortie
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Ajouter les rectangles à toutes les frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Ajouter tous les rectangles
            for (x1, y1, x2, y2) in scaled_rectangles:
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 0, 0),
                    -1,
                )
            out.write(frame)

        cap.release()
        out.release()
        print(f"Vidéo générée avec les rectangles : {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ajoute des carrés noirs sur des vidéos.")
    parser.add_argument(
        "folder_path",
        type=str,
        help="Chemin du dossier contenant les vidéos .mp4 à traiter.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder_path):
        print(f"Le chemin spécifié n'est pas un dossier valide : {args.folder_path}")
    else:
        process_videos(args.folder_path)
