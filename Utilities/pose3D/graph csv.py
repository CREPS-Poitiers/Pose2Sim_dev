import pandas as pd
import matplotlib.pyplot as plt
import sys

# Chemin vers le fichier CSV
csv_file = r"C:\Users\fdelap01\motionagformer\demo\output\squatlab_34\keypoints_3D.csv"  # Remplacez par votre chemin

# Chargement du fichier CSV
try:
    # Tester les séparateurs courants
    data = pd.read_csv(csv_file, sep=None, engine='python', header=None)
    print(f"Fichier chargé avec succès ! {data.shape[0]} lignes, {data.shape[1]} colonnes.")
except Exception as e:
    print("Erreur lors du chargement du fichier :", e)
    sys.exit()

# Afficher les premières lignes du fichier
print("\nAperçu des premières lignes du fichier CSV :")
print(data.head())

# Vérification du nombre de colonnes
if data.shape[1] < 6:
    print("Erreur : Le fichier CSV ne contient pas suffisamment de colonnes (au moins 6 requises).")
    sys.exit()

# Extraire les colonnes X, Y, Z (4ème, 5ème, 6ème colonnes)
x = data.iloc[:, 21]  # 4ème colonne (index 3)
y = data.iloc[:, 22]  # 5ème colonne (index 4)
z = data.iloc[:, 23]  # 6ème colonne (index 5)

# Afficher un aperçu des coordonnées extraites
print("\nAperçu des coordonnées extraites :")
print(f"X : {x.head()}")
print(f"Y : {y.head()}")
print(f"Z : {z.head()}")

# Tracer la trajectoire 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracer la courbe
ax.plot(x, y, z, color='b', marker='o', markersize=1, label="Trajectoire du keypoint")

# Configurer les axes
ax.set_xlabel('X (4ème colonne)')
ax.set_ylabel('Y (5ème colonne)')
ax.set_zlabel('Z (6ème colonne)')
ax.set_title('Trajectoire 3D du keypoint')
ax.legend()

# Afficher le graphique
plt.show()
