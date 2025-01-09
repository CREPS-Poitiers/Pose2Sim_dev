# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 11:16:04 2024

@author: fdelap01
"""

import cv2
import os

def images_to_video(image_folder, output_video, fps=120):
    """
    Crée une vidéo à partir d'images dans un dossier.

    :param image_folder: Chemin vers le dossier contenant les images.
    :param output_video: Chemin du fichier de sortie pour la vidéo MP4.
    :param fps: Nombre d'images par seconde dans la vidéo (par défaut 30).
    """
    # Liste des fichiers image triés par ordre alphabétique
    images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images.sort()

    if not images:
        print("Aucune image trouvée dans le dossier !")
        return

    # Obtenir les dimensions de la première image pour la taille de la vidéo
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        print(f"Impossible de lire l'image : {first_image_path}")
        return
    height, width, layers = first_image.shape

    # Définir le codec et créer l'objet VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour le format MP4
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # Ajouter chaque image à la vidéo
    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Impossible de lire l'image : {image_path}, saut de cette image.")
            continue
        video_writer.write(frame)

    # Libérer les ressources
    video_writer.release()
    print(f"Vidéo enregistrée avec succès : {output_video}")

if __name__ == "__main__":
    # Chemin vers le dossier contenant les images et chemin de sortie pour la vidéo
    image_folder = r"C:\Users\fdelap01\motionagformer\demo\output\Basket_Cam08\pose"  # Remplacer par votre chemin
    output_video = r"C:\Users\fdelap01\motionagformer\demo\output\Basket_Cam08\video.mp4"  # Remplacer par le chemin de sortie
    
    # Créer la vidéo
    images_to_video(image_folder, output_video, fps=120)
