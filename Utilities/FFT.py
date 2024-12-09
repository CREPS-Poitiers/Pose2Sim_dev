import os
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from scipy.io import wavfile
from scipy.fft import fft

# Chemin vers le dossier
audio_folder = r"C:\Users\fdelaplace\AppData\Local\anaconda3\envs\Pose2Sim\Lib\site-packages\Pose2Sim\Essai_3x3\Trial_1\videos\audio_files\trimmed_audio"

# Créer l'application PyQtGraph
app = QtWidgets.QApplication([])

# Définir le fond du graphique sur blanc
pg.setConfigOption('background', 'w')  # Fond blanc
pg.setConfigOption('foreground', 'k')  # Texte et lignes en noir

# Fonction pour analyser le spectre fréquentiel d'un fichier audio
def analyze_frequency(file_path):
    # Lire le fichier audio
    samplerate, data = wavfile.read(file_path)
    
    # Prendre qu'un canal
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Nombre d'échantillons
    N = len(data)
    
    # Calcul de la FFT
    freqs = np.fft.fftfreq(N, 1/samplerate)
    fft_values = fft(data)
    
    # Partie positive du spectre de fréquence
    positive_freqs = freqs[:N//2]
    positive_fft_values = np.abs(fft_values[:N//2])
    
    # Limiter les fréquences affichées
    mask = positive_freqs <= 3000
    
    # Créer la fenêtre graphique PyQtGraph
    plot_window = pg.plot()
    plot_window.setTitle(f'Spectre fréquentiel de {os.path.basename(file_path)}')
    plot_window.setLabel('left', 'Amplitude')
    plot_window.setLabel('bottom', 'Fréquence (Hz)')
    
    # Tracer les données
    plot_window.plot(positive_freqs[mask], positive_fft_values[mask])

# Parcourir tous les fichiers audio du dossier
for file_name in os.listdir(audio_folder):
    if file_name.endswith('.wav'):
        file_path = os.path.join(audio_folder, file_name)
        analyze_frequency(file_path)

# Lancer l'application PyQtGraph
app.exec_()
