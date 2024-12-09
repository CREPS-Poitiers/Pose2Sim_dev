#!/usr/bin/env python
# -*- coding: utf-8 -*-



###########################################################################
## PIPELINE MARKERLESS                                                   ##
###########################################################################


## INIT
import shutil
import os
from pathlib import Path
import time
from datetime import datetime
import cv2
from rtmlib import BodyWithFeet, Wholebody, PoseTracker, draw_skeleton
import logging
from skelly_synchronize import skelly_synchronize as sync
from Pose2Sim import Pose2Sim
import ffmpeg



## AUTHORSHIP INFORMATION
__author__ = "Camille ARNAULT"
__credits__ = ["Camille ARNAULT"]
__version__ = "0.1.0"
__maintainer__ = "Camille ARNAULT"
__email__ = "camille.arnault@univ-poitiers.fr"
__status__ = "Development"



# def treatAll():
#     #Création des dossiers de travail
#     classification()
#     #Calibration
#     Pose2Sim.calibration()
#     #Synchronisation des vidéos des essais
#     synchronisation()
#     #Video overlay pour vérif synchro
#     #Estimation de pose 
#     Pose2Sim.poseEstimation()
#     #Association de personnes
#     Pose2Sim.personAssociation()
#     #Triangulation
#     Pose2Sim.triangulation()
#     #Filtrage
#     Pose2Sim.filtering()
    
    
    

def classification() :

    """
    
    ***
    OBJECTIF DE LA FONCTION
    ***
    Classer toutes les vidéos d'une session d'acquisition avec la bonne 
    structure de dossier et les bons noms de dossier et de vidéos.
    
    
    ***
    ARGUMENTS
    ***
    Input :
        * path (optionnel - str) : chemin où toutes les données de la session 
        d'enregistrement on été stockées. par défaut -> chemin de travail.
        
        * videoFormat (optionnel - str): format des fichiers vidéo déposés. 
          par défaut -> .MP4 (format de sortie des goPros)
        
    Output :
        * 
        
    
    ***
    WARNING
    ***
    Attention :
        /1\ Le programme est conçu pour fonctionner avec le setup de gopros du labo qui 
        disposent du GoPro labs i.e d'un nom de fichier associé au numéro de la 
        caméra et à l'heure et la date d'enregistrement. Pour le bon fonctionnement
        du code, ces caractéristiques doivent être conservées.
        Site pour télécharger/mettre à jour le goprolabs : 
            https://gopro.github.io/labs/ (/!\ à reinstaller sur carte SD après chaque maj GoPro)
        Site pour changer le nom : 
            https://gopro.github.io/labs/control/basename/ (code : [yyyymmdd_HHMMSS]-CAMERA00- /!\ cocher la case changement permanent)
        
        /2\ Les expérimentations doivent toujours être faite dans ce même ordre : 
            - calibrations intrinsèques
            - calibrations extrinsèques 
            - acquisitions (xN)
            Si un dossier de calibration existe déjà, il est possible de le coller directement dans le dossier de travail.
            Ce dossier peut ne contenir que la calibration intrinsèque, que la calibration extrinsèque, ou les deux. 
            Dans tous les cas, si un fichier Calib est présent dans le dossier, la calibration sera considérée comme faite, vidéos présentes ou non.
            
        /3\ Mettez toutes vos vidéos dans votre dossier de travail
        
        
    ***
    PISTES D'UPGRADE
    ***
    Vérifier dans le fichier de calib si le nombre de caméras renseignées
    correspond au nombre de caméras trouvées via les noms de fichier
    
    Trouver un moyen d'identifier l'essai pour lequel il manque un ou plusieurs
    fichier quand le nombre de trials calculé n'est pas un int
    
    """

    
    os.system("====================")
    os.system("File classification...")
    os.system("--------------------\n\n")  
    
    
    """
    1 - Initialisation
    Récupération du dossier de travail, des noms des fichiers et du 
    nombre de caméras.
    """
    
    path=None
    videoFormat=".MP4"
    
    #Récup path de travail
    if path==None :
        path=os.getcwd()
    else :
        path = Path(path)
    
    #Récup nom des fichiers vidéo
    filenames =[] 
    for file in os.listdir(path):
        if file.endswith(videoFormat):
            filenames.append(file)
    
    #Forcer le classement des noms de fichiers par date et heure
    filenames=sorted(filenames, key=lambda x: datetime.strptime(x[0:15],"%Y%m%d_%H%M%S"))
    
    #Récupérer le nom et le nombre de caméras
    nbfiles = len(filenames)
    camList = []
    for file in range(0,nbfiles) :
        camList += [filenames[file][filenames[file].find("CAMERA"):filenames[file].find("CAMERA")+8]]
    camNames = list(set(camList))   
    nbcam = len(camNames)
    
    

    
    """
    2 - Calibration path
    Création des dossiers de calibration si inexistants,
    prise en compte du fichier de calibration .toml si ancienne calibration
    à prendre en compte.
    """
    
    #Attribution d'une var aux chemins
    calibFolderPath = os.path.join(path,"calibration")
    calibExtFolderPath = os.path.join(path,"calibration","extrinsics")
    calibIntFolderPath = os.path.join(path,"calibration","intrinsics")
    calibVerifIntFolderPath = os.path.join(path,"intrinsics")
    
    #Si folder non existant, on le créé et on créé les sous dossiers.
    userCalib = False
    userCalibInt=False
    
    if not os.path.exists(calibFolderPath):
        print("Création du dossier calibration avec les sous-dossiers respectifs.")
        os.mkdir(calibFolderPath)
        os.mkdir(calibExtFolderPath)
        if not os.path.exists(calibVerifIntFolderPath):
            os.mkdir(calibIntFolderPath)
            for n in range(1,nbcam+1) :
                if n<10:
                    os.mkdir(os.path.join(calibExtFolderPath,"ext_cam0"+str(n)))
                    os.mkdir(os.path.join(calibIntFolderPath,"int_cam0"+str(n)))
                else :
                    os.mkdir(os.path.join(calibExtFolderPath,"ext_cam"+str(n)))
                    os.mkdir(os.path.join(calibIntFolderPath,"int_cam"+str(n)))
                    
        else :
            userCalibInt=True
            shutil.move(calibVerifIntFolderPath,calibFolderPath)
            for n in range(1,nbcam+1) :
                if n<10:
                    os.mkdir(os.path.join(calibExtFolderPath,"ext_cam0"+str(n)))
                else :
                    os.mkdir(os.path.join(calibExtFolderPath,"ext_cam"+str(n)))

    #Si dossier calib existant vérifier qu'il contient la calibration, si non prévenir l'utilisateur (plus worth de juste déposer les vidéos dans le working directory si calib pas faite du coup)          
    else :
        print("Un dossier calibration a été trouvé.")
        for file in os.listdir(calibFolderPath) : 
            if not any(s in file for s in ["calib", "Calib","Calib_scene"]):
                print("WARNING - Aucun fichier .toml de calibration retrouvé.")
            else :
                userCalib = True
                
                     
    
    """
    3 - Sort trials
    Détermination du nombre d'acquisitions réalisées.
    Si le nombre de fichier n'est pas un multiple du nombre de caméra,
    l'utilisateur est averti et la classification des vidéos n'opère pas.
    Si ok, alors chaque paquet de nbcam fichiers est déplacé dans le dossier 
    Trial_n.
    """
    
    #Récupérer le nombre d'essais. Si pas de fichier calib, on considère +2 paquets de vidéos
    if userCalib == False and userCalibInt == False:
        nbtrials = (nbfiles-nbcam*2)/nbcam
    if userCalib == False and userCalibInt == True:
        nbtrials = (nbfiles-nbcam)/nbcam  
    else :
        nbtrials = (nbfiles)/nbcam
    
    #Si nombre de vidéos cohérent avec nombre de caméras...
    if nbtrials%1 == 0 :
        nbtrials = int(nbtrials)
        
        #Création des dossiers trials et sous-dossiers videos_raw
        for trial in range(1,nbtrials+1):
            if not os.path.exists(os.path.join(path,"Trial_"+str(trial))) : os.mkdir(os.path.join(path,"Trial_"+str(trial)))
            if not os.path.exists(os.path.join(path,"Trial_"+str(trial),"videos_raw")) : os.mkdir(os.path.join(path,"Trial_"+str(trial),"videos_raw"))

        # Si pas de fichier de calib déposée par l'utilisateur...
        if userCalib == False and userCalibInt == False:
            try :
                for acq in range(0,nbtrials+2) :
                    #Classification vidéos calib intrinsèque
                    if acq == 0 :
                        for cam in range(0,nbcam) :
                            if cam<10 :
                                shutil.move(os.path.join(path,filenames[cam]),os.path.join(path,"calibration","intrinsics","int_cam0"+str(cam+1),filenames[cam]))
                            else :
                                shutil.move(os.path.join(path,filenames[cam]),os.path.join(path,"calibration","intrinsics","int_cam"+str(cam+1),filenames[cam]))
                    #Classification vidéos calib extrinsèque
                    if acq == 1 :
                        for cam in range(0,nbcam) :
                            if cam<10 :
                                shutil.move(os.path.join(path,filenames[nbcam*(acq)+cam]),os.path.join(path,"calibration","extrinsics","ext_cam0"+str(cam+1),filenames[nbcam*(acq)+cam]))
                            else : 
                                shutil.move(os.path.join(path,filenames[nbcam*(acq)+cam]),os.path.join(path,"calibration","extrinsics","ext_cam"+str(cam+1),filenames[nbcam*(acq)+cam]))
                    #Classification vidéos essais et copie du fichier de config dans dossier trial
                    if acq not in [0,1] :
                            shutil.copyfile(os.path.join(path,'Config.toml'), os.path.join(path,"Trial_"+str(acq-1),'Config.toml'))
                            for cam in range(0,nbcam) :
                                shutil.move(os.path.join(path,filenames[nbcam*(acq)+cam]),os.path.join(path,"Trial_"+str(acq-1),"videos_raw",filenames[nbcam*(acq)+cam][0:filenames[nbcam*(acq)+cam].find("CAMERA")+8]+".MP4"))          
            except :
                if n == 0 : logging.info("ERROR - unable to transfer intrinsics calibration files.")
                elif n == 1 : logging.info("ERROR - unable to transfer extrinsics calibration files.")
                elif n not in [0,1] : logging.info("ERROR - unable to transfer {filenames[nbcam*(n)+c]} file.")
        
        # Si calib intrinsics deja effectuée et deposée dans un dossier "intrinsics"
        if userCalib == False and userCalibInt == True:
            try :
              
                for acq in range(0,nbtrials+1) :
                    if acq == 0 :
                        for cam in range(0,nbcam) :
                            if cam<10 :
                                shutil.move(os.path.join(path,filenames[nbcam*(acq)+cam]),os.path.join(path,"calibration","extrinsics","ext_cam0"+str(cam+1),filenames[nbcam*(acq)+cam]))
                            else : 
                                shutil.move(os.path.join(path,filenames[nbcam*(acq)+cam]),os.path.join(path,"calibration","extrinsics","ext_cam"+str(cam+1),filenames[nbcam*(acq)+cam]))
                    #Classification vidéos essais et copie du fichier de config dans dossier trial
                    else :
                        shutil.copyfile(os.path.join(path,'Config.toml'), os.path.join(path,"Trial_"+str(acq),'Config.toml'))
                        for cam in range(0,nbcam) :
                            shutil.move(os.path.join(path,filenames[nbcam*(acq)+cam]),os.path.join(path,"Trial_"+str(acq),"videos_raw",filenames[nbcam*(acq)+cam][0:filenames[nbcam*(acq)+cam].find("CAMERA")+8]+".MP4"))          
            except :
                if n == 0 : logging.info("ERROR - unable to transfer intrinsics calibration files.")
                elif n == 1 : logging.info("ERROR - unable to transfer extrinsics calibration files.")
                elif n not in [0,1] : logging.info("ERROR - unable to transfer {filenames[nbcam*(n)+c]} file.")
        
        
        # Si fichier de calib déposée par l'utilisateur...
        else :
            for acq in range(0,nbtrials) :
                #Copie du fichier config dans chaque dossier trial
                try :
                    shutil.copyfile(os.path.join(path,'Config.toml'), os.path.join(path,"Trial_"+str(acq+1),'Config.toml'))
                except :
                    os.system("ERROR - impossible de copier le fichier Config.toml dans les dossiers Trial.")
                #Classification vidéos essais
                for cam in range(0,nbcam) :
                    try :
                        shutil.move(os.path.join(path,filenames[nbcam*(acq)+cam]),os.path.join(path,"Trial_"+str(acq+1),"videos_raw",filenames[nbcam*(acq)+cam][0:filenames[nbcam*(acq)+cam].find("CAMERA")+8]+".MP4"))
                    except :
                        os.system("ERROR - impossible de déplacer et renommer les vidéos dans les dossiers Trial.")
                    
            
    #Avertissement utilisateur si nombre de vidéos non cohérent avec nombre de caméras.
    elif nbtrials%1 != 0 : 
        os.system("ERROR - Number of video files not consistent with number of cameras.")
            
            
    os.system("\n\n--------------------")
    logging.info("f'Dossier de travail : {path}")
    logging.info("f'Nombre de caméras trouvées : {nbcam}")
    logging.info("f'Nombre d'acquisitions classées : {nbtrials}")
    logging.info("\nClassement des enregistrements faits avec succès.")
    os.system("====================")
    




def synchronisation(curTrial=[],mosaicSyncControl=True,plotSyncControl=True) :
    
    """
    
    ***
    OBJECTIF DE LA FONCTION
    ***
    Synchroniser les vidéos à partir de la fonction de synchronisation sonore
    du module pré-existant skelly_synchronize :
    https://github.com/freemocap/skelly_synchronize
    Cette fonction de synchronisation se base sur une cross-correlation du 
    signal de la bande sonore de l'intégralité de la vidéo.
    
        conda install -c conda-forge ffmpeg


    ***
    ARGUMENTS
    ***
    Input :
        * curTrial (optionnel - list of str) : liste d'essais à traiter.
        par défaut tous les essais.
            
        * mosaicSyncControl (optionnel - bool) : Option permettant de générer
        une vidéo mozaïque de toutes les vidéos resynchronisées pour vérifier 
        la synchro.
        par défaut sur None - prends du temps et de l'espace de stockage, à 
        n'utiliser qu'en guise de contrôle sur des essais relativement courts 
        et/ou où la synchro pose question malgré plotSyncControl.
        
        * plotSyncControl (optionnel - bool): Génération d'un plot avec le 
        signal des bandes sonores superposées après la cross-correlation. 
        par défaut sur None.
        
    Output :
        * 
        
    ***
    WARNING
    ***
    Attention :
        /1\ 
        /2\ 
        /3\ 
        
    ***
    PISTES D'UPGRADE
    ***
    Mettre le mosaicmaker dans une fonction à part
    Trouver le moyen de ne pas save les bandes son    
    """
    
    logging.info("====================")
    print("Videos Synchronization...")
    logging.info("--------------------\n\n")  
    
    #Récupération du path de travail
    
    path = os.getcwd()
    folders = os.listdir(path)
    
    #Récupération du nombre et des noms des essais à traiter
    if curTrial == [] :
        nbtrials=0
        trialname = []
        for i in range(0,len(folders)):
            if "Trial" in folders[i]:
                nbtrials += 1
                trialname.append(folders[i])
    else :
        nbtrials = len(curTrial)
        trialname = curTrial
        
    
            
    for trial in range(nbtrials) :
        
        #Récupération des chemins de l'essai à traiter
        raw_video_folder_path = Path(os.path.join(path,trialname[trial],"videos_raw"))
        sync_video_folder_path = Path(os.path.join(path,trialname[trial],"videos"))        
        
        #Si dossier de vidéos synchronisées vide ou inexistant...
        if not os.path.exists(sync_video_folder_path) or os.listdir(sync_video_folder_path)==[] :
            
            #Création du dossier si inexistant
            if not os.path.exists(os.path.join(path,"Trial_"+str(trial+1),"videos")):os.mkdir(os.path.join(path,"Trial_"+str(trial+1),"videos"))
            
            #Fonction de synchronisation
            sync.synchronize_videos_from_audio(raw_video_folder_path=raw_video_folder_path,
                                                synchronized_video_folder_path = sync_video_folder_path,
                                                video_handler="deffcode",
                                                create_debug_plots_bool=plotSyncControl)
        
            # if mosaicSyncControl ==True :
            #     #Récupération des noms des vidéos
            #     syncVideoName =[]
            #     for file in os.listdir(sync_video_folder_path):
            #         if file.endswith(".mp4"):
            #             syncVideoName.append(file)
            #     nbVideos = len(syncVideoName)
                
            #     #Calcul du dimensionnement des vidéos
            #     dimOverlay = 2
            #     while nbVideos/dimOverlay > dimOverlay : dimOverlay += 1
                
            #     #Assemblage de la ligne de commande
            #     mozaicMaker = "ffmpeg"
            #     #Nom des vidéos
            #     for vid in syncVideoName :
            #         mozaicMaker=mozaicMaker+" -i "+str(sync_video_folder_path)+"\\"+vid
            #     #Create overlay base
            #     mozaicMaker = mozaicMaker+' -filter_complex "nullsrc=size=1920x1080 [base];'
                
            #     #Specify output videos dimension
            #     for vid in range(0,nbVideos):
            #         mozaicMaker = mozaicMaker+"["+str(vid)+":v] setpts=PTS-STARTPTS, scale="+str(int(1920/dimOverlay))+"x"+str(int(1080/dimOverlay))+" [v"+str(vid)+"];"
                
            #     #Define vid position
            #     mozaicMaker = mozaicMaker+""
            #     xinc = 1920/dimOverlay
            #     yinc = 1080/dimOverlay
            #     vidInc=0
                
            #     for y in range(0,dimOverlay):
            #         ypos = yinc*y
            #         xpos = 0
            #         for x in range(0,dimOverlay):
            #             if y==0 and x==0 :
            #                 mozaicMaker = mozaicMaker+"[base][v"+str(int(vidInc))+"] overlay=shortest=1:x="+str(int(x*xinc))+":y="+str(int(ypos))+" "
            #                 vidInc += 1
            #             elif y==dimOverlay-1 and x ==dimOverlay-1 :
            #                 mozaicMaker = mozaicMaker+"[tmp"+str(int(vidInc))+"];[tmp"+str(int(vidInc))+"][v"+str(int(vidInc))+"] overlay=shortest=1:x="+str(int(x*xinc))+":y="+str(int(ypos))+'" '
            #                 vidInc += 1
            #             else :
            #                 mozaicMaker = mozaicMaker+"[tmp"+str(int(vidInc))+"];[tmp"+str(int(vidInc))+"][v"+str(int(vidInc))+"] overlay=shortest=1:x="+str(int(x*xinc))+":y="+str(int(ypos))
            #                 vidInc += 1
                            
            #     mozaicMaker = mozaicMaker+"-c:v libx264 "+str(sync_video_folder_path)+"\\"+"SyncVideos.mp4"
            #     os.system(mozaicMaker)
                

            logging.info("Vidéos synchronisées avec succès.")
        else : 
            logging.info("Vidéos déjà synchronisées.")
        

        
    logging.info("\n\n--------------------")
    logging.info("Synchronisation des vidéos faites avec succès.")
    logging.info("====================")
    
       


def synchroMosaique(path=None):
    # Récupérer le chemin du dossier de travail
    if path==None:
        path = os.getcwd()
 
    folders = os.listdir(path)

    # Parcourir les dossiers Trial_X
    for folder in folders:
        if folder.startswith("Trial_") and os.path.isdir(os.path.join(path, folder)):
            trial_folder = os.path.join(path, folder, "videos")
            if os.path.exists(trial_folder):
                # Récupérer les fichiers vidéos .MP4
                video_files = [f for f in os.listdir(trial_folder) if f.endswith(".mp4")]
                nbVideos = len(video_files)
                
                if nbVideos == 0:
                    print(f"Aucune vidéo trouvée dans {trial_folder}")
                    continue
                
                # Calculer les dimensions de la mosaïque
                dimOverlay = 2
                while nbVideos / dimOverlay > dimOverlay:
                    dimOverlay += 1
                
                # Construire la ligne de commande FFmpeg pour créer la mosaïque
                ffmpeg_cmd = "ffmpeg"
                
                # Ajouter chaque vidéo en entrée
                for vid in video_files:
                    ffmpeg_cmd += f" -i {os.path.join(trial_folder, vid)}"
                
                # Créer le filtre pour la mosaïque
                filter_complex = f' -filter_complex "nullsrc=size=1920x1080 [base];'
                
                # Ajouter les vidéos redimensionnées dans le filtre
                for i in range(nbVideos):
                    filter_complex += f"[{i}:v] setpts=PTS-STARTPTS, scale={int(1920/dimOverlay)}x{int(1080/dimOverlay)} [v{i}];"
                
                # Calcul des positions des vidéos dans la mosaïque
                xinc = 1920 // dimOverlay
                yinc = 1080 // dimOverlay
                vidInc = 0
                
                for y in range(dimOverlay):
                    ypos = y * yinc
                    for x in range(dimOverlay):
                        xpos = x * xinc
                        if vidInc < nbVideos:
                            if vidInc == 0:
                                filter_complex += f"[base][v{vidInc}] overlay=shortest=1:x={xpos}:y={ypos} [tmp{vidInc}];"
                            elif vidInc == nbVideos - 1:
                                filter_complex += f"[tmp{vidInc-1}][v{vidInc}] overlay=shortest=1:x={xpos}:y={ypos}\" "
                            else:
                                filter_complex += f"[tmp{vidInc-1}][v{vidInc}] overlay=shortest=1:x={xpos}:y={ypos} [tmp{vidInc}];"
                            vidInc += 1
                
                # Ajouter les autres paramètres à la ligne de commande
                output_path = os.path.join(trial_folder, "SyncVideos.mp4")
                ffmpeg_cmd += filter_complex + f"-c:v libx264 {output_path}"
                
                # Exécuter la commande FFmpeg
                print(f"Création de la mosaïque pour {folder}")
                os.system(ffmpeg_cmd)
            else:
                print(f"Dossier 'videos' introuvable dans {folder}")

# Appeler la fonction
synchroMosaique(r"C:\Users\fdelaplace\AppData\Local\anaconda3\envs\Pose2Sim\Lib\site-packages\Pose2Sim\NewCalib4Pose2Sim")

    
# def synchroMosaique(curTrial=[]) :
#     logging.info("====================")
#     print("Videos Synchronization...")
#     logging.info("--------------------\n\n")  
    
#     #Récupération du path de travail
#     path = os.getcwd()
#     folders = os.listdir(path)
    
#     #Récupération du nombre et des noms des essais à traiter
#     if curTrial == [] :
#         nbtrials=0
#         trialname = []
#         for i in range(0,len(folders)):
#             if "Trial" in folders[i]:
#                 nbtrials += 1
#                 trialname.append(folders[i])
#     else :
#         nbtrials = len(curTrial)
#         trialname = curTrial
        
            
#     for trial in range(nbtrials) :
        
#         #Récupération des chemins de l'essai à traiter
#         raw_video_folder_path = Path(os.path.join(path,trialname[trial],"videos_raw"))
#         sync_video_folder_path = Path(os.path.join(path,trialname[trial],"videos"))  
#         syncVideoName =[]
#         for file in os.listdir(sync_video_folder_path):
#             if file.endswith(".mp4"):
#                 syncVideoName.append(file)
#         nbVideos = len(syncVideoName)
        
#         #Calcul du dimensionnement des vidéos
#         dimOverlay = 2
#         while nbVideos/dimOverlay > dimOverlay : dimOverlay += 1
        
#         #Assemblage de la ligne de commande
#         mozaicMaker = "ffmpeg"
#         #Nom des vidéos
#         for vid in syncVideoName :
#             mozaicMaker=mozaicMaker+" -i "+str(sync_video_folder_path)+"\\"+vid
#         #Create overlay base
#         mozaicMaker = mozaicMaker+' -filter_complex "nullsrc=size=1920x1080 [base];'
        
#         #Specify output videos dimension
#         for vid in range(0,nbVideos):
#             mozaicMaker = mozaicMaker+"["+str(vid)+":v] setpts=PTS-STARTPTS, scale="+str(int(1920/dimOverlay))+"x"+str(int(1080/dimOverlay))+" [v"+str(vid)+"];"
        
#         #Define vid position
#         mozaicMaker = mozaicMaker+""
#         xinc = 1920/dimOverlay
#         yinc = 1080/dimOverlay
#         vidInc=0
        
#         for y in range(0,dimOverlay):
#             ypos = yinc*y
#             xpos = 0
#             for x in range(0,dimOverlay):
#                 if y==0 and x==0 :
#                     mozaicMaker = mozaicMaker+"[base][v"+str(int(vidInc))+"] overlay=shortest=1:x="+str(int(x*xinc))+":y="+str(int(ypos))+" "
#                     vidInc += 1
#                 elif y==dimOverlay-1 and x ==dimOverlay-1 :
#                     mozaicMaker = mozaicMaker+"[tmp"+str(int(vidInc))+"];[tmp"+str(int(vidInc))+"][v"+str(int(vidInc))+"] overlay=shortest=1:x="+str(int(x*xinc))+":y="+str(int(ypos))+'" '
#                     vidInc += 1
#                 else :
#                     mozaicMaker = mozaicMaker+"[tmp"+str(int(vidInc))+"];[tmp"+str(int(vidInc))+"][v"+str(int(vidInc))+"] overlay=shortest=1:x="+str(int(x*xinc))+":y="+str(int(ypos))
#                     vidInc += 1
                    
#         mozaicMaker = mozaicMaker+"-c:v libx264 "+str(sync_video_folder_path)+"\\"+"SyncVideos.mp4"
#         os.system(mozaicMaker)
    
    
    
def synchronisationVerification() :
    
    logging.info("====================")
    print("Synchronization verification...")
    logging.info("====================")  
    
    #Get working directory
    path = os.getcwd()
    folders = os.listdir(path)
    
    #Get number of trials
    nbtrials=0
    trialsName = []
    print(folders)
    for i in range(0,len(folders)):
        if "Trial" in folders[i]: 
            # print(folders[i])
            nbtrials += 1
            trialsName += [folders[i]]
            
    #Get s
    for trial in range(nbtrials) :
        
        #path config
        raw_video_folder_path = Path(os.path.join(path,trialsName[trial],"videos_raw"))
        sync_video_folder_path = Path(os.path.join(path,trialsName[trial],"videos"))        
        
            
        #Récupération des noms des vidéos
        syncVideoName =[]
        for file in os.listdir(sync_video_folder_path):
            if file.endswith(".mp4"):
                syncVideoName.append(file)
        nbVideos = len(syncVideoName)
        
        #Calcul du dimensionnement des vidéos
        dimOverlay = 2
        while nbVideos/dimOverlay > dimOverlay : dimOverlay += 1
        
        #Assemblage de la ligne de commande
        mozaicMaker = "ffmpeg"
        #Nom des vidéos
        for vid in syncVideoName :
            mozaicMaker = mozaicMaker + " -i " + os.path.join(sync_video_folder_path, vid)
        #Create overlay base
        mozaicMaker = mozaicMaker+' -filter_complex "nullsrc=size=1920x1080 [base];'
        
        #Specify output videos dimension
        for vid in range(0,nbVideos):
            mozaicMaker = mozaicMaker+"["+str(vid)+":v] setpts=PTS-STARTPTS, scale="+str(int(1920/dimOverlay))+"x"+str(int(1080/dimOverlay))+" [v"+str(vid)+"];"
        
        #Define vid position
        mozaicMaker = mozaicMaker+""
        xinc = 1920/dimOverlay
        yinc = 1080/dimOverlay
        vidInc=0
        
        for y in range(0,dimOverlay):
            ypos = yinc*y
            xpos = 0
            for x in range(0,dimOverlay):
                if y==0 and x==0 :
                    mozaicMaker = mozaicMaker+"[base][v"+str(int(vidInc))+"] overlay=shortest=1:x="+str(int(x*xinc))+":y="+str(int(ypos))+" "
                    vidInc += 1
                elif y==dimOverlay-1 and x ==dimOverlay-1 :
                    mozaicMaker = mozaicMaker+"[tmp"+str(int(vidInc))+"];[tmp"+str(int(vidInc))+"][v"+str(int(vidInc))+"] overlay=shortest=1:x="+str(int(x*xinc))+":y="+str(int(ypos))+'" '
                    vidInc += 1
                else :
                    mozaicMaker = mozaicMaker+"[tmp"+str(int(vidInc))+"];[tmp"+str(int(vidInc))+"][v"+str(int(vidInc))+"] overlay=shortest=1:x="+str(int(x*xinc))+":y="+str(int(ypos))
                    vidInc += 1
                    
        mozaicMaker = mozaicMaker+"-c:v libx264 "+str(sync_video_folder_path)+"\\"+"SyncVideos.mp4"
        os.system(mozaicMaker)
        
        

    
    
    
    