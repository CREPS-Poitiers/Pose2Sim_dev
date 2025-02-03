import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk,filedialog, messagebox
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tomlkit

# Global variables
step = 1
config_values = {}
instructions = []
extrinsic_points = []
question_widgets = []


# Chemin du fichier config.toml
CONFIG_PATH = None

def browse_config_file():
    global CONFIG_PATH
    CONFIG_PATH = filedialog.askopenfilename(
        title="Sélectionnez le fichier config.toml",
        filetypes=[("Fichiers TOML", "*.toml")],
    )
    if CONFIG_PATH:
        instructions_box.config(state=tk.NORMAL)
        instructions_box.insert(tk.END, f"Fichier choisi : {CONFIG_PATH}\n")
        instructions_box.config(state=tk.DISABLED)
    else:
        messagebox.showwarning("Aucun fichier", "Veuillez sélectionner un fichier config.toml valide.")

def validate_config_file():
    """
    Valide le fichier sélectionné, masque les boutons et passe directement à la classification.
    """
    global CONFIG_PATH
    if CONFIG_PATH:
        # Ajouter un message dans les instructions
        instructions_box.config(state=tk.NORMAL)
        instructions_box.insert(tk.END, "Fichier config validé !\n\n")
        instructions_box.insert(tk.END, "Démarrage de la classification...\n")
        instructions_box.config(state=tk.DISABLED)

        # Masquer les widgets liés à cette étape
        for widget in question_widgets:
            widget.pack_forget()
        question_widgets.clear()

        # Lancer directement la classification
        setup_classification()
    else:
        messagebox.showwarning("Aucun fichier", "Veuillez sélectionner un fichier config.toml.")


def update_toml_value(file_path, section_hierarchy, key, new_value):
    """
    Met à jour une valeur dans un fichier TOML en conservant la structure, les commentaires et les indentations.

    :param file_path: Chemin vers le fichier TOML.
    :param section_hierarchy: Liste représentant la hiérarchie des sections. Par exemple, ["calibration", "calculate", "intrinsics"].
    :param key: Nom de la variable à mettre à jour.
    :param new_value: Nouvelle valeur à assigner à la variable.
    """
    try:
        # Charger le fichier TOML
        with open(file_path, "r") as file:
            toml_content = tomlkit.parse(file.read())

        # Naviguer dans la hiérarchie des sections
        section = toml_content
        for sub_section in section_hierarchy:
            if sub_section in section:
                section = section[sub_section]
            else:
                raise KeyError(f"La section '{sub_section}' n'existe pas dans le fichier.")

        # Mettre à jour la valeur
        if key in section:
            section[key] = new_value
        else:
            raise KeyError(f"La clé '{key}' n'existe pas dans la section '{' > '.join(section_hierarchy)}'.")

        # Sauvegarder les modifications dans le fichier
        with open(file_path, "w") as file:
            file.write(tomlkit.dumps(toml_content))

        print(f"Le fichier '{file_path}' a été mis à jour avec succès.")
        print(f"Section : {' > '.join(section_hierarchy)} | {key} = {new_value}")

    except FileNotFoundError:
        print(f"Erreur : Le fichier '{file_path}' est introuvable.")
    except KeyError as e:
        print(f"Erreur : {str(e)}")
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {str(e)}")



def initialize_instructions():
    """Initialize the guide instructions."""
    instructions.extend([
        "Vous voici dans le guide d'utilisation de Pose2Sim.",
        "Première étape : Organisation du dossier de traitement des vidéos.",
        "Déposez toutes vos vidéos dans un dossier en vrac avec un fichier config.toml.\n"
    ])

# Function to update the instructions display
def update_instructions(new_text):
    instructions.append(new_text)
    instructions_box.config(state=tk.NORMAL)
    instructions_box.insert(tk.END, new_text + "\n")
    instructions_box.see(tk.END)  # Scroll to the latest text
    instructions_box.config(state=tk.DISABLED)

# Function to show questions dynamically
def show_question(question_text, options, next_step_func):
    question_label.config(text=question_text)
    question_label.pack(pady=10)
    for i, option in enumerate(options):
        radios[i].config(text=option, value=option)
        radios[i].pack(anchor="w", padx=50)
    validate_button.config(text="Valider", command=next_step_func)
    validate_button.pack(pady=20)

# Function to hide the question
def hide_question():
    question_label.pack_forget()
    for radio in radios:
        radio.pack_forget()
    validate_button.pack_forget()

def setup_classification():
    """
    Demande si l'utilisateur souhaite effectuer la classification.
    """
    title_label.config(text="Étape 1 : Organisation du dossier de traitement")
    update_instructions("Souhaitez-vous effectuer la classification des fichiers ?")
    show_question(
        "Souhaitez-vous effectuer la classification des fichiers ?",
        ["Oui", "Non"],
        handle_classification_decision
    )

def handle_classification_decision():
    """
    Gère la réponse de l'utilisateur concernant la nécessité de faire la classification.
    """
    user_response = intrinsic_var.get()
    if user_response == "Oui":
        # Passer à la configuration pour vérifier si la calibration intrinsèque est faite
        update_instructions("Vous avez choisi d'effectuer la classification des fichiers.")
        ask_intrinsics_status()
    else:
        # Passer directement à l'étape suivante
        update_instructions("Vous avez choisi de ne pas effectuer la classification. Passons à l'étape suivante.")
        hide_question()
        validate_button.config(text="Suivant", command=setup_calibration)
        validate_button.pack(pady=20)

def ask_intrinsics_status():
    """
    Demande si l'utilisateur a déjà effectué la calibration intrinsèque.
    """
    hide_question()
    update_instructions("Avez-vous déjà effectué la calibration intrinsèque des caméras ?")
    show_question(
        "Avez-vous déjà effectué la calibration intrinsèque des caméras ?",
        ["Oui", "Non"],
        handle_classification_response
    )

def handle_classification_response():
    """
    Affiche des consignes spécifiques selon l'état de la calibration intrinsèque.
    """
    user_response = intrinsic_var.get()
    if user_response == "Oui":
        update_instructions("✅ Calibration intrinsèque déjà effectuée.")
        update_instructions(
            "Assurez-vous d'avoir placé le dossier `intrinsics` et le fichier `Calib.toml` dans le dossier contenant les vidéos en vrac.\n"
        )
    else:
        update_instructions("❌ Calibration intrinsèque non effectuée.")
        update_instructions("Déposez toutes vos vidéos de calibration en vrac dans le même dossier.\n")
    hide_question()
    setup_classification_environment()

def setup_classification_environment():
    """
    Affiche les consignes pour configurer l'environnement et lancer la classification.
    """
    update_instructions("Lancez un prompt conda et activez votre environnement Pose2Sim :")
    update_instructions("```bash\nconda activate Pose2Sim\n```\n")
    update_instructions("Allez dans le bon dossier avec :")
    update_instructions("```bash\ncd path_to_your_videos\n```\n")
    update_instructions("Démarrez un shell Python interactif avec :")
    update_instructions("```bash\nipython\n```\n")
    update_instructions("Importez Pose2Sim et exécutez la classification :")
    update_instructions("```python\nfrom Pose2Sim import Pose2Sim\nPose2Sim.classification()\n```\n")
    update_instructions("Votre environnement et votre dossier de traitement sont maintenant configurés. Vous pouvez continuer avec les prochaines étapes.")
    update_instructions("=============================================================")
    validate_button.config(text="Suivant", command=setup_calibration)
    validate_button.pack(pady=20)


# Step 2: Calibration
def setup_calibration():
    hide_question()
    title_label.config(text="Étape 2 : Calibration")
    show_question(
        "Souhaitez-vous effectuer une calibration intrinsèque ?",
        ["Oui", "Non"],
        handle_calibration_response
    )

def handle_calibration_response():
    user_response = intrinsic_var.get()
    if user_response == "Oui":
        update_instructions("✅ Vous avez choisi de faire une calibration intrinsèque.")
        show_question(
            "Préférez-vous utiliser Charuco ou Chessboard ?",
            ["Charuco", "Chessboard"],
            handle_calibration_method
        )
    else:
        update_instructions("❌ Vous avez choisi de ne pas effectuer de calibration intrinsèque.\n")
        hide_question()
        validate_button.config(text="Suivant", command=setup_extrinsic_calibration)
        validate_button.pack(pady=20)

def handle_calibration_method():
    user_response = intrinsic_var.get()
    config_values["intrinsics_method"] = user_response.lower()
    update_instructions(f"Vous avez choisi la méthode {user_response} pour la calibration intrinsèque.")
    ask_calibration_details()

def ask_calibration_details():
    hide_question()
    update_instructions("Complétez les paramètres suivants pour la calibration intrinsèque :")

    labels = [
        "extract_every_N_sec (secondes) :",
        "intrinsics_corners_nb (par ex. [6,9]) :",
        "intrinsics_square_size (mm) :",
        "intrinsics_aruco_size (mm) (seulement pour Charuco) :"
    ]
    defaults = ["0.2", "[6,9]", "40", "30"]
    inputs = []

    for i, label in enumerate(labels):
        label_widget = tk.Label(root, text=label, font=("Arial", 10))
        label_widget.pack(anchor="w", padx=20)
        question_widgets.append(label_widget)

        entry = tk.Entry(root, font=("Arial", 10))
        entry.insert(0, defaults[i])
        entry.pack(anchor="w", padx=40)
        question_widgets.append(entry)
        inputs.append(entry)

    def save_calibration_details():
        """
        Enregistre les détails de la calibration intrinsèque dans le fichier config.toml.
        """
        # Récupération des valeurs des entrées
        config_values["extract_every_N_sec"] = inputs[0].get()
        config_values["intrinsics_corners_nb"] = inputs[1].get()
        config_values["intrinsics_square_size"] = inputs[2].get()
        config_values["intrinsics_aruco_size"] = inputs[3].get()
    
        update_instructions("Paramètres enregistrés. Mise à jour du fichier de configuration...")
    
        # Masquer uniquement les widgets liés à la question en cours
        for widget in question_widgets:
            widget.pack_forget()
        question_widgets.clear()
    
        # Mise à jour du fichier config.toml ligne par ligne
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["calibration", "calculate", "intrinsics"],
            key="overwrite_intrinsics",
            new_value=True
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["calibration", "calculate", "intrinsics"],
            key="intrinsics_method",
            new_value=config_values.get("intrinsics_method", "charuco")
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["calibration", "calculate", "intrinsics"],
            key="extract_every_N_sec",
            new_value=float(config_values["extract_every_N_sec"])
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["calibration", "calculate", "intrinsics"],
            key="intrinsics_corners_nb",
            new_value=eval(config_values["intrinsics_corners_nb"])
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["calibration", "calculate", "intrinsics"],
            key="intrinsics_square_size",
            new_value=float(config_values["intrinsics_square_size"])
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["calibration", "calculate", "intrinsics"],
            key="intrinsics_aruco_size",
            new_value=float(config_values["intrinsics_aruco_size"])
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["calibration", "calculate", "intrinsics"],
            key="intrinsics_aruco_dict",
            new_value="DICT_6X6_250"  # Pour Charuco uniquement
        )
    
        update_instructions("Les paramètres ont été automatiquement ajoutés au fichier config.toml.")
        update_instructions("=============================================================")
    
        # Passage à l'étape suivante (calibration extrinsèque)
        validate_button.config(text="Suivant", command=setup_extrinsic_calibration)
        validate_button.pack(pady=20)
        question_widgets.append(validate_button)


    validate_button.config(text="Valider", command=save_calibration_details)
    validate_button.pack(pady=20)

    
def setup_extrinsic_calibration():
    hide_question()
    title_label.config(text="Étape 3 : Calibration Extrinsèque")
    show_question(
        "Souhaitez-vous effectuer une calibration extrinsèque ?", 
        ["Oui", "Non"], 
        handle_extrinsic_response
    )

def handle_extrinsic_response():
    user_response = intrinsic_var.get()
    if user_response == "Oui":
        update_instructions("✅ Vous avez choisi de faire une calibration extrinsèque.")
        input_extrinsic_points()
    else:
        update_instructions("❌ Vous avez choisi de ne pas effectuer de calibration extrinsèque.\n")
        validate_button.config(text="Suivant", command=setup_calibration_instructions)
        # setup_calibration_instructions()
        validate_button.pack(pady=20)


def input_extrinsic_points():
    hide_question()
    update_instructions("Définissez les coordonnées 3D de vos points en mètres.")
    update_instructions("Entrez les valeurs pour X, Y, Z. Une nouvelle ligne apparaîtra automatiquement lorsque vous commencez à remplir une ligne existante.")

    point_entries = []  # Liste pour stocker les tuples (x, y, z)
    question_widgets.clear()  # Vider les widgets de questions précédents

    # Cadre principal pour organiser les entrées et le bouton
    points_frame = tk.Frame(root)
    points_frame.pack(anchor="w", padx=20, pady=5)
    question_widgets.append(points_frame)  # Ajout au suivi des widgets

    def add_point_entry():
        """Ajouter une nouvelle ligne de saisie pour X, Y, Z."""
        frame = tk.Frame(points_frame)
        frame.pack(anchor="w", padx=20, pady=5)

        # Labels et zones de texte pour X, Y, Z
        x_label = tk.Label(frame, text="X (mètres):", font=("Arial", 10))
        x_label.pack(side=tk.LEFT, padx=5)
        x_entry = tk.Entry(frame, width=10)
        x_entry.pack(side=tk.LEFT, padx=5)

        y_label = tk.Label(frame, text="Y (mètres):", font=("Arial", 10))
        y_label.pack(side=tk.LEFT, padx=5)
        y_entry = tk.Entry(frame, width=10)
        y_entry.pack(side=tk.LEFT, padx=5)

        z_label = tk.Label(frame, text="Z (mètres):", font=("Arial", 10))
        z_label.pack(side=tk.LEFT, padx=5)
        z_entry = tk.Entry(frame, width=10)
        z_entry.pack(side=tk.LEFT, padx=5)

        # Ajouter les entrées à la liste des points
        point_entries.append((x_entry, y_entry, z_entry))

        # Ajouter une nouvelle ligne si nécessaire
        x_entry.bind("<KeyRelease>", lambda _: check_last_entry())
        y_entry.bind("<KeyRelease>", lambda _: check_last_entry())
        z_entry.bind("<KeyRelease>", lambda _: check_last_entry())

    def check_last_entry():
        """Ajouter une nouvelle ligne lorsque la dernière ligne commence à être remplie."""
        if point_entries:
            last_x, last_y, last_z = point_entries[-1]
            if last_x.get() or last_y.get() or last_z.get():
                add_point_entry()

    def visualize_points():
        """Afficher le graphique 3D et afficher les boutons Modifier/Confirmer."""
        global extrinsic_points  # Utilisation de la variable globale
        extrinsic_points = []

        for x, y, z in point_entries:
            if x.get() and y.get() and z.get():  # Ne sauvegarder que les lignes complètes
                extrinsic_points.append([float(x.get()), float(y.get()), float(z.get())])

        # Générer un graphique 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            [p[0] for p in extrinsic_points],
            [p[1] for p in extrinsic_points],
            [p[2] for p in extrinsic_points],
            c="r", marker="o"
        )
        ax.set_xlabel("X (mètres)")
        ax.set_ylabel("Y (mètres)")
        ax.set_zlabel("Z (mètres)")
        plt.title("Visualisation des Points 3D")
        plt.show(block=True)

        # Transition vers les boutons Modifier/Confirmer
        visualize_button.pack_forget()
        modify_button.pack(pady=10)
        confirm_button.pack(pady=10)

    def modify_points():
        """Revenir à l'état initial pour permettre les modifications."""
        modify_button.pack_forget()
        confirm_button.pack_forget()
        visualize_button.pack(pady=20)

    def confirm_points():
        """Confirmer les points saisis, désafficher les widgets et mettre à jour la configuration."""
        # Désafficher tous les widgets liés à cette étape
        for widget in question_widgets:
            widget.pack_forget()
        question_widgets.clear()
    
        # Afficher les coordonnées confirmées
        update_instructions("Coordonnées confirmées :")
        for point in extrinsic_points:
            update_instructions(f" - {point}")
    
        # Enregistrer les points dans config_values
        config_values["object_coords_3d"] = extrinsic_points
    
        # Mise à jour du fichier config.toml
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["calibration", "calculate", "extrinsics", "scene"],
            key="show_reprojection_error",
            new_value=True
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["calibration", "calculate", "extrinsics", "scene"],
            key="extrinsics_extension",
            new_value="mp4"
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["calibration", "calculate", "extrinsics", "scene"],
            key="object_coords_3d",
            new_value=config_values["object_coords_3d"]
        )
    
        update_instructions("Les points et paramètres de la calibration extrinsèque ont été enregistrés automatiquement dans le fichier config.toml.")
        update_instructions("=============================================================")
    
        # Passer à l'étape suivante
        setup_calibration_instructions()


    # Ajouter la première ligne de saisie
    add_point_entry()

    # Ajouter le bouton Visualiser les points
    visualize_button = tk.Button(root, text="Visualiser les points", font=("Arial", 12), bg="lightblue", command=visualize_points)
    visualize_button.pack(pady=20)
    question_widgets.append(visualize_button)

    # Ajouter les boutons Modifier et Confirmer
    modify_button = tk.Button(root, text="Modifier", font=("Arial", 12), bg="lightblue", command=modify_points)
    confirm_button = tk.Button(root, text="Confirmer", font=("Arial", 12), bg="lightgreen", command=confirm_points)
    question_widgets.extend([modify_button, confirm_button])
                                        


def confirm_extrinsic_points():
    """Confirm the extrinsic points and proceed."""
    update_instructions("Coordonnées confirmées :")
    for point in extrinsic_points:
        update_instructions(f" - {point}")

    config_values["object_coords_3d"] = extrinsic_points
    validate_button.config(text="Suivant", command=final_step)
    validate_button.pack(pady=20)
    
def setup_calibration_instructions():
    hide_question()
    title_label.config(text="Étape 4 : Lancer la calibration")
    update_instructions("Assurez-vous que vous avez bien modifié et enregistré le fichier `config.toml` avec les paramètres appropriés.")
    update_instructions("Dans le prompt conda déjà actif (vous êtes dans le bon chemin et dans `ipython`), exécutez :")
    update_instructions("```python\nfrom Pose2Sim import Pose2Sim\nPose2Sim.calibration()\n```")
    update_instructions("=============================================================")
    validate_button.config(text="Suivant", command=setup_synchronization)
    validate_button.pack(pady=20)

# intrinsic_var = tk.StringVar()


def setup_synchronization():
    hide_question()
    title_label.config(text="Étape 5 : Choix du type de synchronisation")

    # Ajoutez un label pour la question
    question_label = tk.Label(root, text="Quel type de synchronisation souhaitez-vous effectuer ?", font=("Arial", 12))
    question_label.pack(pady=10)
    question_widgets.append(question_label)

    # Créez les options de synchronisation
    sync_options = [
        ("mouvement (move)", "move"),
        ("son (sound)", "sound"),
        ("manuelle (manual)", "manual"),
    ]
    intrinsic_var.set("")  # Réinitialiser la sélection

    for text, value in sync_options:
        rb = tk.Radiobutton(
            root, text=text, variable=intrinsic_var, value=value, font=("Arial", 10)
        )
        rb.pack(anchor="w", padx=20)
        question_widgets.append(rb)

    # Ajouter le bouton Valider
    validate_button.config(text="Valider", command=handle_sync_choice)
    validate_button.pack(pady=20)
    question_widgets.append(validate_button)


def handle_sync_choice():
    user_response = intrinsic_var.get()
    if not user_response:
        update_instructions("Veuillez sélectionner une option avant de valider.")
        return
    # Masquer les widgets liés à cette question
    for widget in question_widgets:
        widget.pack_forget()
    question_widgets.clear()

    if user_response == "move":
        setup_pose_estimation("move")
    elif user_response == "sound":
        setup_sound_synchronization()
    elif user_response == "manual":
        setup_manual_synchronization()
    
def setup_sound_synchronization():
    """
    Configure l'étape de synchronisation par son et met à jour le fichier config.toml.
    """
    hide_question()
    title_label.config(text="Étape 5 : Synchronisation par Son")
    update_instructions("Configurez les paramètres pour la synchronisation par son :")

    # Afficher les graphes
    label_display_graphs = tk.Label(root, text="Afficher les graphiques de synchronisation :", font=("Arial", 12))
    label_display_graphs.pack(anchor="w", padx=20)
    question_widgets.append(label_display_graphs)

    display_sync_var = tk.StringVar(value="oui")
    for option in ["oui", "non"]:
        rb = tk.Radiobutton(root, text=option, variable=display_sync_var, value=option, font=("Arial", 10))
        rb.pack(anchor="w", padx=40)
        question_widgets.append(rb)

    def save_sound_sync_config():
        """
        Sauvegarde la configuration de la synchronisation par son dans le fichier config.toml.
        """
        # Récupérer les valeurs
        display_sync_plots = "true" if display_sync_var.get() == "oui" else "false"

        # Mettre à jour le fichier config.toml
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["synchronization"],
            key="synchronization_type",
            new_value="sound"
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["synchronization"],
            key="display_sync_plots",
            new_value=display_sync_plots
        )

        update_instructions("Les paramètres de synchronisation par son ont été enregistrés automatiquement dans le fichier config.toml.")
        update_instructions("Dans le prompt `ipython`, exécutez :")
        update_instructions("```python\nfrom Pose2Sim import Pose2Sim\nPose2Sim.synchronization()\n```")
        update_instructions("=============================================================")

        # Masquer les widgets après validation
        for widget in question_widgets:
            widget.pack_forget()
        question_widgets.clear()

        # Passer à la configuration de la pose estimation
        validate_button.config(text="Suivant", command=lambda: setup_pose_estimation("sound"))
        validate_button.pack(pady=20)

    # Bouton Valider
    validate_button.config(text="Valider", command=save_sound_sync_config)
    validate_button.pack(pady=20)
    question_widgets.append(validate_button)

def setup_manual_synchronization():
    """
    Configure l'étape de synchronisation manuelle et met à jour le fichier config.toml.
    """
    hide_question()
    title_label.config(text="Étape 5 : Synchronisation Manuelle")
    update_instructions("Configurez les paramètres pour la synchronisation manuelle :")

    # Afficher les graphes
    label_display_graphs = tk.Label(root, text="Afficher les graphiques de synchronisation :", font=("Arial", 12))
    label_display_graphs.pack(anchor="w", padx=20)
    question_widgets.append(label_display_graphs)

    display_sync_var = tk.StringVar(value="oui")
    for option in ["oui", "non"]:
        rb = tk.Radiobutton(root, text=option, variable=display_sync_var, value=option, font=("Arial", 10))
        rb.pack(anchor="w", padx=40)
        question_widgets.append(rb)

    def save_manual_sync_config():
        """
        Sauvegarde la configuration de la synchronisation manuelle dans le fichier config.toml.
        """
        # Récupérer la valeur choisie
        display_sync_plots = "true" if display_sync_var.get() == "oui" else "false"

        # Mettre à jour le fichier config.toml
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["synchronization"],
            key="synchronization_type",
            new_value="manual"
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["synchronization"],
            key="display_sync_plots",
            new_value=display_sync_plots
        )

        update_instructions("Les paramètres de synchronisation manuelle ont été enregistrés automatiquement dans le fichier config.toml.")
        update_instructions("Dans le prompt `ipython`, exécutez :")
        update_instructions("```python\nfrom Pose2Sim import Pose2Sim\nPose2Sim.synchronization()\n```")
        update_instructions("=============================================================")

        # Masquer les widgets après validation
        for widget in question_widgets:
            widget.pack_forget()
        question_widgets.clear()

        # Passer à la configuration de la pose estimation
        validate_button.config(text="Suivant", command=lambda: setup_pose_estimation("manual"))
        validate_button.pack(pady=20)

    # Bouton Valider
    validate_button.config(text="Valider", command=save_manual_sync_config)
    validate_button.pack(pady=20)
    question_widgets.append(validate_button)



    
def setup_pose_estimation(sync_type):
    hide_question()
    title_label.config(text="Étape 5a : Pose Estimation")
    update_instructions("Configurez les paramètres pour la pose estimation :")

    # Options prédéfinies
    pose_models = ["HALPE_26 (default, body and feet)", "COCO_133 (body, feet, hands)", "COCO_17 (body)"]
    modes = ["lightweight", "balanced", "performance"]
    display_options = ["oui", "non"]

    # Modèle de pose
    label_pose_model = tk.Label(root, text="Modèle de pose :", font=("Arial", 12))
    label_pose_model.pack(anchor="w", padx=20)
    question_widgets.append(label_pose_model)

    pose_model_var = tk.StringVar(value=pose_models[0])
    for model in pose_models:
        rb = tk.Radiobutton(root, text=model, variable=pose_model_var, value=model, font=("Arial", 10))
        rb.pack(anchor="w", padx=40)
        question_widgets.append(rb)

    # Mode
    label_mode = tk.Label(root, text="Mode :", font=("Arial", 12))
    label_mode.pack(anchor="w", padx=20)
    question_widgets.append(label_mode)

    mode_var = tk.StringVar(value=modes[0])
    for mode in modes:
        rb = tk.Radiobutton(root, text=mode, variable=mode_var, value=mode, font=("Arial", 10))
        rb.pack(anchor="w", padx=40)
        question_widgets.append(rb)

    # Fréquence de détection
    label_det_frequency = tk.Label(root, text="Fréquence de détection :", font=("Arial", 12))
    label_det_frequency.pack(anchor="w", padx=20)
    question_widgets.append(label_det_frequency)

    det_frequency_entry = tk.Entry(root, font=("Arial", 10))
    det_frequency_entry.insert(0, "20")
    det_frequency_entry.pack(anchor="w", padx=40)
    question_widgets.append(det_frequency_entry)

    # Afficher les graphes
    label_display_graphs = tk.Label(root, text="Afficher les graphes :", font=("Arial", 12))
    label_display_graphs.pack(anchor="w", padx=20)
    question_widgets.append(label_display_graphs)

    display_var = tk.StringVar(value=display_options[0])
    for option in display_options:
        rb = tk.Radiobutton(root, text=option, variable=display_var, value=option, font=("Arial", 10))
        rb.pack(anchor="w", padx=40)
        question_widgets.append(rb)

    def save_pose_estimation():
        """
        Enregistre les détails de la pose estimation dans le fichier config.toml.
        """
        # Récupération des valeurs des entrées
        pose_model = pose_model_var.get().split()[0]  # Récupère uniquement le code du modèle
        mode = mode_var.get()
        det_frequency = det_frequency_entry.get()
        display_detection = "true" if display_var.get() == "oui" else "false"
    
        update_instructions("Paramètres enregistrés. Mise à jour du fichier de configuration...")
    
        # Mise à jour du fichier config.toml
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["pose"],
            key="pose_model",
            new_value=pose_model
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["pose"],
            key="mode",
            new_value=mode
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["pose"],
            key="det_frequency",
            new_value=int(det_frequency)
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["pose"],
            key="display_detection",
            new_value=display_detection
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["pose"],
            key="overwrite_pose",
            new_value=False
        )
    
        update_instructions("Les paramètres de pose estimation ont été enregistrés automatiquement dans le fichier config.toml.")
        update_instructions("Positionnez-vous dans le dossier du trial à traiter avec :")
        update_instructions("```bash\ncd path/to/trial\n```")
        update_instructions("Dans le prompt `ipython`, exécutez :")
        update_instructions("```python\nfrom Pose2Sim import Pose2Sim\nPose2Sim.poseEstimation()\n```")
        update_instructions("=============================================================")
    
        # Masquer les widgets après validation
        for widget in question_widgets:
            widget.pack_forget()
        question_widgets.clear()
    
        # Passer à la prochaine étape en fonction du type de synchronisation
        if sync_type == "move":
            setup_move_synchronization()
        else:  # sound ou manual
            setup_person_association()

    # Bouton Valider
    validate_button.config(text="Valider", command=save_pose_estimation)
    validate_button.pack(pady=20)
    question_widgets.append(validate_button)

def setup_move_synchronization():
    """
    Configure l'étape de synchronisation par mouvement et met à jour le fichier config.toml.
    """
    hide_question()
    title_label.config(text="Étape 5b : Synchronisation par Mouvement")
    update_instructions("Configurez les paramètres pour la synchronisation par mouvement :")

    # Afficher les graphes
    label_display_graphs = tk.Label(root, text="Afficher les graphiques de synchronisation :", font=("Arial", 12))
    label_display_graphs.pack(anchor="w", padx=20)
    question_widgets.append(label_display_graphs)

    display_sync_var = tk.StringVar(value="oui")
    for option in ["oui", "non"]:
        rb = tk.Radiobutton(root, text=option, variable=display_sync_var, value=option, font=("Arial", 10))
        rb.pack(anchor="w", padx=40)
        question_widgets.append(rb)

    # Points clés à considérer
    label_keypoints = tk.Label(root, text="Points clés à considérer (par ex. RWrist or RAnkle) :", font=("Arial", 12))
    label_keypoints.pack(anchor="w", padx=20)
    question_widgets.append(label_keypoints)

    keypoints_entry = tk.Entry(root, font=("Arial", 10))
    keypoints_entry.insert(0, "RWrist")
    keypoints_entry.pack(anchor="w", padx=40)
    question_widgets.append(keypoints_entry)

    def save_sync_config():
        """
        Sauvegarde la configuration de la synchronisation par mouvement dans le fichier config.toml.
        """
        # Récupérer les valeurs saisies
        display_sync_plots = "true" if display_sync_var.get() == "oui" else "false"
        keypoints_to_consider = [key.strip() for key in keypoints_entry.get().split(",")]

        # Mise à jour du fichier config.toml
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["synchronization"],
            key="synchronization_type",
            new_value="move"
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["synchronization"],
            key="display_sync_plots",
            new_value=display_sync_plots
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["synchronization"],
            key="keypoints_to_consider",
            new_value=keypoints_to_consider
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["synchronization"],
            key="approx_time_maxspeed",
            new_value="auto"
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["synchronization"],
            key="time_range_around_maxspeed",
            new_value=2.0
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["synchronization"],
            key="likelihood_threshold",
            new_value=0.4
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["synchronization"],
            key="filter_cutoff",
            new_value=6
        )
        update_toml_value(
            file_path=CONFIG_PATH,
            section_hierarchy=["synchronization"],
            key="filter_order",
            new_value=4
        )

        update_instructions("Les paramètres de synchronisation par mouvement ont été enregistrés automatiquement dans le fichier config.toml.")
        update_instructions("Dans le prompt `ipython`, exécutez :")
        update_instructions("```python\nfrom Pose2Sim import Pose2Sim\nPose2Sim.synchronization()\n```")
        update_instructions("=============================================================")

        # Masquer les widgets après validation
        for widget in question_widgets:
            widget.pack_forget()
        question_widgets.clear()

        # Passer à l'étape finale
        setup_person_association()

    # Bouton Valider
    validate_button.config(text="Valider", command=save_sync_config)
    validate_button.pack(pady=20)
    question_widgets.append(validate_button)
    
def setup_person_association():
    hide_question()
    title_label.config(text="Étape 6 : Person Association")
    update_instructions("Configurez les paramètres pour l'association des personnes :")

    # Multi Person (true/false)
    label_multi_person = tk.Label(root, text="Multi Person (true/false) :", font=("Arial", 12))
    label_multi_person.pack(anchor="w", padx=20)
    question_widgets.append(label_multi_person)

    multi_person_var = tk.StringVar(value="false")
    for option in ["true", "false"]:
        rb = tk.Radiobutton(root, text=option, variable=multi_person_var, value=option, font=("Arial", 10))
        rb.pack(anchor="w", padx=40)
        question_widgets.append(rb)

    # Itérer sur les paramètres (true/false)
    label_iterative = tk.Label(root, text="Itérer sur différents paramètres (true/false) :", font=("Arial", 12))
    label_iterative.pack(anchor="w", padx=20)
    question_widgets.append(label_iterative)

    iterative_var = tk.StringVar(value="false")
    for option in ["true", "false"]:
        rb = tk.Radiobutton(root, text=option, variable=iterative_var, value=option, font=("Arial", 10))
        rb.pack(anchor="w", padx=40)
        question_widgets.append(rb)

    def handle_next():
        """
        Affiche les champs appropriés en fonction des choix multi_person et iterative.
        """
        # Masquer les widgets actuels
        for widget in question_widgets:
            widget.pack_forget()
        question_widgets.clear()

        # Lire les choix actuels
        iterative = iterative_var.get() == "true"

        # Afficher les champs en fonction du choix de itération
        if not iterative:
            setup_single_value_inputs(multi_person_var.get())
        else:
            setup_range_inputs(multi_person_var.get())

    # Bouton Suivant
    validate_button.config(text="Suivant", command=handle_next)
    validate_button.pack(pady=20)
    question_widgets.append(validate_button)


def setup_single_value_inputs(multi_person):
    """
    Affiche les champs pour les valeurs simples.
    """
    title_label.config(text="Étape 6 : Person Association - Valeurs Simples")
    update_instructions("Configurez les paramètres en saisissant des valeurs simples :")

    # Likelihood Threshold
    label_likelihood = tk.Label(root, text="Likelihood Threshold Association :", font=("Arial", 12))
    label_likelihood.pack(anchor="w", padx=20)
    question_widgets.append(label_likelihood)

    likelihood_entry = tk.Entry(root, font=("Arial", 10))
    likelihood_entry.insert(0, "0.3")
    likelihood_entry.pack(anchor="w", padx=40)
    question_widgets.append(likelihood_entry)

    # Minimum Cameras for Triangulation
    label_min_cameras = tk.Label(root, text="Minimum Cameras for Triangulation :", font=("Arial", 12))
    label_min_cameras.pack(anchor="w", padx=20)
    question_widgets.append(label_min_cameras)

    min_cameras_entry = tk.Entry(root, font=("Arial", 10))
    min_cameras_entry.insert(0, "3")
    min_cameras_entry.pack(anchor="w", padx=40)
    question_widgets.append(min_cameras_entry)

    def save_single_value_inputs():
        likelihood_threshold = float(likelihood_entry.get())
        min_cameras = int(min_cameras_entry.get())

        # Mise à jour du fichier config
        update_toml_value(CONFIG_PATH, ["personAssociation"], "likelihood_threshold_association", likelihood_threshold)
        update_toml_value(CONFIG_PATH, ["triangulation"], "min_cameras_for_triangulation", min_cameras)

        # Passer aux paramètres spécifiques
        if multi_person == "true":
            setup_multi_person_association()
        else:
            setup_single_person_association()

    validate_button.config(text="Valider", command=save_single_value_inputs)
    validate_button.pack(pady=20)
    question_widgets.append(validate_button)

def setup_range_inputs(multi_person):
    """
    Affiche les champs pour les plages de valeurs avec une disposition améliorée.
    """
    title_label.config(text="Étape 6 : Person Association - Plages de Valeurs")
    update_instructions("Configurez les paramètres avec des plages de valeurs :")

    # Conteneur pour organiser les inputs de manière plus claire
    frame = tk.Frame(root)
    frame.pack(anchor="w", padx=20, pady=10)
    question_widgets.append(frame)

    def add_range_inputs(parent_frame, label_text, row):
        """
        Ajoute un label et trois entrées pour une plage de valeurs.
        """
        # Label
        label = tk.Label(parent_frame, text=label_text, font=("Arial", 12))
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)

        # Zones de texte pour début, pas, fin
        entry_start = tk.Entry(parent_frame, font=("Arial", 10), width=10)
        entry_start.grid(row=row, column=1, padx=5, pady=5)

        entry_step = tk.Entry(parent_frame, font=("Arial", 10), width=10)
        entry_step.grid(row=row, column=2, padx=5, pady=5)

        entry_end = tk.Entry(parent_frame, font=("Arial", 10), width=10)
        entry_end.grid(row=row, column=3, padx=5, pady=5)

        return entry_start, entry_step, entry_end

    # Ajout des inputs pour les plages de valeurs
    range_likelihood = add_range_inputs(frame, "Plage de Likelihood Threshold :", 0)
    range_cameras = add_range_inputs(frame, "Plage de Cameras for Triangulation :", 1)

    def save_range_inputs():
        """
        Enregistre les plages de valeurs et met à jour le fichier config.toml.
        """
        try:
            # Récupérer les plages de valeurs
            likelihood_threshold = [
                float(range_likelihood[0].get()),
                float(range_likelihood[1].get()),
                float(range_likelihood[2].get())
            ]
            min_cameras = [
                int(range_cameras[0].get()),
                int(range_cameras[1].get()),
                int(range_cameras[2].get())
            ]

            # Mettre à jour le fichier config.toml
            update_toml_value(CONFIG_PATH, ["personAssociation"], "likelihood_threshold_association", likelihood_threshold)
            update_toml_value(CONFIG_PATH, ["triangulation"], "min_cameras_for_triangulation", min_cameras)

            # Masquer les widgets après validation
            for widget in question_widgets:
                widget.pack_forget()
                widget.destroy()  # Détruire les widgets pour éviter les résidus
            question_widgets.clear()

            # Passer à l'étape spécifique en fonction de multi_person
            if multi_person == "true":
                setup_multi_person_association()
            else:
                setup_single_person_association()

        except ValueError:
            messagebox.showerror("Erreur de saisie", "Veuillez entrer des valeurs valides pour toutes les plages.")

    # Bouton Valider
    validate_button.config(text="Valider", command=save_range_inputs)
    validate_button.pack(pady=20)
    question_widgets.append(validate_button)


def setup_single_person_association():
    title_label.config(text="Étape 6a : Person Association - Single Person")
    update_instructions("Configurez les paramètres pour une seule personne :")

    # Reprojection Error Threshold
    label_reproj_error = tk.Label(root, text="Reprojection Error Threshold Association :", font=("Arial", 12))
    label_reproj_error.pack(anchor="w", padx=20)
    question_widgets.append(label_reproj_error)

    reproj_error_entry = tk.Entry(root, font=("Arial", 10))
    reproj_error_entry.insert(0, "20")
    reproj_error_entry.pack(anchor="w", padx=40)
    question_widgets.append(reproj_error_entry)

    # Tracked Keypoint
    label_tracked_keypoint = tk.Label(root, text="Tracked Keypoint (par ex. ['Neck']) :", font=("Arial", 12))
    label_tracked_keypoint.pack(anchor="w", padx=20)
    question_widgets.append(label_tracked_keypoint)

    tracked_keypoint_entry = tk.Entry(root, font=("Arial", 10))
    tracked_keypoint_entry.insert(0, "['Neck']")
    tracked_keypoint_entry.pack(anchor="w", padx=40)
    question_widgets.append(tracked_keypoint_entry)

    def save_single_person_config():
        reproj_error_threshold = reproj_error_entry.get()
        tracked_keypoint = eval(tracked_keypoint_entry.get())

        # Mettre à jour le fichier config.toml
        update_toml_value(CONFIG_PATH, ["personAssociation", "single_person"], "reproj_error_threshold_association", eval(reproj_error_threshold))
        update_toml_value(CONFIG_PATH, ["personAssociation", "single_person"], "tracked_keypoint", tracked_keypoint)

        update_instructions("Les paramètres pour une seule personne ont été enregistrés automatiquement dans le fichier config.toml.")
        update_instructions("=============================================================")

        # Passer à l'étape finale
        final_step()

    # Bouton pour valider les paramètres single person
    validate_button.config(text="Valider", command=save_single_person_config)
    validate_button.pack(pady=20)
    question_widgets.append(validate_button)

def setup_multi_person_association():
    title_label.config(text="Étape 6b : Person Association - Multi Person")
    update_instructions("Configurez les paramètres pour plusieurs personnes :")

    # Reconstruction Error Threshold
    label_recon_error = tk.Label(root, text="Reconstruction Error Threshold (par ex. 0.1) :", font=("Arial", 12))
    label_recon_error.pack(anchor="w", padx=20)
    question_widgets.append(label_recon_error)

    recon_error_entry = tk.Entry(root, font=("Arial", 10))
    recon_error_entry.insert(0, "0.1")
    recon_error_entry.pack(anchor="w", padx=40)
    question_widgets.append(recon_error_entry)

    # Minimum Affinity
    label_min_affinity = tk.Label(root, text="Minimum Affinity (par ex. 0.2) :", font=("Arial", 12))
    label_min_affinity.pack(anchor="w", padx=20)
    question_widgets.append(label_min_affinity)

    min_affinity_entry = tk.Entry(root, font=("Arial", 10))
    min_affinity_entry.insert(0, "0.2")
    min_affinity_entry.pack(anchor="w", padx=40)
    question_widgets.append(min_affinity_entry)

    def save_multi_person_config():
        recon_error_threshold = float(recon_error_entry.get())
        min_affinity = float(min_affinity_entry.get())

        # Mettre à jour le fichier config.toml
        update_toml_value(CONFIG_PATH, ["personAssociation", "multi_person"], "reconstruction_error_threshold", recon_error_threshold)
        update_toml_value(CONFIG_PATH, ["personAssociation", "multi_person"], "min_affinity", min_affinity)

        update_instructions("Les paramètres pour plusieurs personnes ont été enregistrés automatiquement dans le fichier config.toml.")
        update_instructions("=============================================================")

        # Passer à l'étape finale
        final_step()

    # Bouton pour valider les paramètres multi person
    validate_button.config(text="Valider", command=save_multi_person_config)
    validate_button.pack(pady=20)
    question_widgets.append(validate_button)



def final_step():
    hide_question()
    title_label.config(text="Étape Finale")
    update_instructions("Votre guide est complet.")

   #  # Texte pour la pose estimation
   #  pose_text = ""
   #  if "pose_model" in config_values:
   #      pose_text = f"""
   # [pose]
   # vid_img_extension = 'MP4' # any video or image extension
   # pose_model = '{config_values.get("pose_model", "HALPE_26")}'
   # mode = '{config_values.get("mode", "lightweight")}'
   # det_frequency = {config_values.get("det_frequency", 20)}
   # tracking = false
   # display_detection = true
   # overwrite_pose = false
   # save_video = 'to_video'
   # output_format = 'openpose'
   #      """
    
   #  # Texte pour la synchronisation
   #  sync_text = ""
   #  if "synchronization_type" in config_values:
   #      sync_text = f"""
   # [synchronization]
   # synchronization_type = '{config_values["synchronization_type"]}'
   # display_sync_plots = {config_values.get("display_sync_plots", "true")}
   # keypoints_to_consider = {config_values.get("keypoints_to_consider", "['RWrist']")}
   # approx_time_maxspeed = 'auto'
   # time_range_around_maxspeed = 2.0
   # likelihood_threshold = 0.4
   # filter_cutoff = 6
   # filter_order = 4
   #      """
    
   #  # Rassembler tous les textes
   #  final_config = f"""
   # {pose_text}

   # {sync_text}
   #  """

    # update_instructions("Voici le texte complet à insérer dans votre fichier config.toml :")
    # update_instructions(final_config)

    # Sauvegarder dans un fichier texte
    with open("Pose2Sim_Guide.txt", "w") as file:
        file.write("\n".join(instructions))

    messagebox.showinfo("Guide Terminé", "Le guide personnalisé a été enregistré dans 'Pose2Sim_Guide.txt'.")


# Main Tkinter GUI setup
root = tk.Tk()
root.title("Pose2Sim Interactive Guide")
root.geometry("800x800")

# Title for each major step
title_label = tk.Label(root, font=("Arial", 14, "bold"))
title_label.pack(pady=10)

# Instructions display
instructions_box = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=70, height=20, font=("Arial", 10))
instructions_box.pack(pady=10)
instructions_box.config(state=tk.NORMAL)


# Initialize the guide
initialize_instructions()
for line in instructions:
    instructions_box.insert(tk.END, line + "\n")
instructions_box.config(state=tk.DISABLED)

# Bouton Parcourir
browse_button = tk.Button(root, text="Parcourir le fichier Config.toml", font=("Arial", 12), command=browse_config_file)
browse_button.pack(pady=10)
question_widgets.append(browse_button)

# Bouton Valider
validate_button = tk.Button(root, text="Valider", font=("Arial", 12), command=validate_config_file)
validate_button.pack(pady=10)
question_widgets.append(validate_button)

# Question components
intrinsic_var = tk.StringVar(value="")
question_label = tk.Label(root, font=("Arial", 12))
radios = [tk.Radiobutton(root, variable=intrinsic_var, font=("Arial", 10)) for _ in range(2)]
validate_button = tk.Button(root, font=("Arial", 12), bg="lightblue")

# Start the guide
#setup_classification()

# Run the Tkinter loop
root.mainloop()
