# Pose2Sim_dev
*
# 1. Télécharger le code
git clone https://github.com/CREPS-Poitiers/Pose2Sim_dev.git
cd Pose2Sim_dev

# 2. Créer et activer l'environnement Conda
conda env create -f environment.yml
conda activate pose2sim-dev

# 3. Installer Pose2Sim
pip install -e .

# 4. Vérifier l'installation
python -c "from Pose2Sim import Pose2Sim; print('Pose2Sim is installed and ready to use!')"
