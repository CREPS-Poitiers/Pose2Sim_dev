import os
import cv2

# Dimensions en cm pour différents formats de papier (de A4 à A0), en mode paysage
paper_sizes = {
    "A4": (29.7, 21.0),
    "A3": (42.0, 29.7),
    "A2": (59.4, 42.0),
    "A1": (84.1, 59.4),
    "A0": (118.9, 84.1)
}

def find_optimal_grid(usable_width_cm, usable_height_cm, max_square_size_mm=40, min_square_size_mm=30):
    """
    Trouver la disposition optimale pour la grille Charuco en fonction des dimensions de la page et de la marge.
    La taille des carreaux doit être un entier en millimètres, avec une taille optimale proche de 40 mm si possible.
    
    :param usable_width_cm: Largeur utilisable en cm (sans marges).
    :param usable_height_cm: Hauteur utilisable en cm (sans marges).
    :param max_square_size_mm: Taille maximale du carreau en mm.
    :param min_square_size_mm: Taille minimale du carreau en mm.
    :return: Nombre de carrés en X, Nombre de carrés en Y, et taille des carrés en mm.
    """
    # Convertir les dimensions en millimètres
    usable_width_mm = usable_width_cm * 10
    usable_height_mm = usable_height_cm * 10

    # Initialiser les meilleures configurations
    best_squares_x, best_squares_y, best_square_size_mm = 0, 0, 0
    best_area_used = 0

    # Parcourir toutes les tailles possibles de carreaux en partant des plus grandes
    for square_size_mm in range(max_square_size_mm, min_square_size_mm - 1, -1):
        squares_x = int(usable_width_mm // square_size_mm)
        squares_y = int(usable_height_mm // square_size_mm)
        
        # Calculer l'espace utilisé avec cette configuration
        width_used = squares_x * square_size_mm
        height_used = squares_y * square_size_mm
        area_used = width_used * height_used

        # Choisir la meilleure configuration qui maximise l'utilisation de l'espace
        if area_used > best_area_used:
            best_squares_x = squares_x
            best_squares_y = squares_y
            best_square_size_mm = square_size_mm
            best_area_used = area_used

    return best_squares_x, best_squares_y, best_square_size_mm

def create_and_save_new_board(paper_format="A3", dpi=300, margin_cm=2.0, max_square_size_mm=40, min_square_size_mm=30):
    """
    Crée une grille Charuco optimisée en fonction du format de papier et des marges, en utilisant les carrés de taille optimale.
    :param paper_format: Format du papier (A4, A3, A2, A1, A0).
    :param dpi: Résolution de l'image pour l'impression (dots per inch).
    :param margin_cm: Taille des marges en cm.
    :param max_square_size_mm: Taille maximale des carrés en mm.
    :param min_square_size_mm: Taille minimale des carrés en mm.
    """
    # Vérifier que le format de papier est supporté
    if paper_format not in paper_sizes:
        raise ValueError(f"Format de papier non supporté : {paper_format}. Choisissez parmi A4, A3, A2, A1, A0.")
    
    # Dimensions du papier en cm
    width_cm, height_cm = paper_sizes[paper_format]
    
    # Dimensions utilisables avec les marges
    usable_width_cm = width_cm - 2 * margin_cm
    usable_height_cm = height_cm - 2 * margin_cm
    
    # Trouver la grille optimale
    squares_x, squares_y, square_length_mm = find_optimal_grid(usable_width_cm, usable_height_cm, max_square_size_mm, min_square_size_mm)
    
    # Créer un dictionnaire Aruco et un plateau Charuco avec la taille optimale
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length_mm / 10, (square_length_mm * 0.75) / 10, dictionary)

    # Dimensions en pixels pour l'image
    width_px = int(width_cm * dpi / 2.54)  # Conversion cm en pixels
    height_px = int(height_cm * dpi / 2.54)
    
    # Générer l'image avec une marge
    img = board.generateImage((width_px, height_px), marginSize=int(margin_cm * dpi / 2.54))
    
    # Sauvegarder l'image générée
    file_name = f'charuco_{squares_x}x{squares_y}_{square_length_mm}mm.png'
    cv2.imwrite(file_name, img)
    
    print(f"Plateau Charuco optimisé sauvegardé sous : {file_name}, format : {paper_format}, taille des carrés : {square_length_mm} mm")




# Exécuter la fonction pour créer et sauvegarder le plateau Charuco
create_and_save_new_board(paper_format="A3", dpi=300, margin_cm=2.0, max_square_size_mm=50, min_square_size_mm=30)
