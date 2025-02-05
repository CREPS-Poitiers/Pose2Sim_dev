#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
###########################################################################
## HUMAN POSE ESTIMATION SCRIPT (WITH DEEPLABCUT INTEGRATION )           ##
###########################################################################

This script estimates pose from videos or image folders and saves the results 
in JSON, video, and/or image formats. It supports real-time visualization, 
various pose models, and customizable modes for inference. 

Key Features:
- **Flexible Model Support**: HALPE_26, COCO_133, COCO_17 are natively supported via RTMLib.
- **Custom Integration with DeepLabCut (DLC)**:
  - Directly integrates with DLC for custom models defined in the `Config.toml` file.
  - Automates environment setup and execution of DLC projects.
  - Handles pose estimation outputs for seamless processing in Pose2Sim.
- **Performance Optimization**:
  - Detection frequency can be customized for faster processing in simple scenarios.
  - Tracking enables consistent person IDs across frames.
- **Output Options**:
  - JSON files in OpenPose format.
  - Annotated videos and/or images showing detected poses.

### DeepLabCut Integration ###
This script includes a newly developed feature for integrating DeepLabCut (DLC):
- The `deeplabcut_env_path`, `config_DLC_project_path`, and `shuffle_number` are configured 
  in the `Config.toml` file.
- Executes DLC pose estimation in a separate environment.
- Converts DLC outputs for compatibility with Pose2Sim for further analysis.

Inputs:
- A `Config.toml` file defining the parameters for pose estimation and DLC integration.
- Video files or folders of images from the specified input directory.

Outputs:
- JSON files with detected keypoints and confidence scores.
- Optionally, annotated videos and/or images showcasing the detected poses.

Authors:
- HunMin Kim, David Pagnon
- Extended by F.Delaplace for DLC integration
- Copyright 2021, Pose2Sim
- Licensed under BSD 3-Clause License

Version: 0.9.4
Maintainer: David Pagnon
Email: contact@david-pagnon.com
"""
## AUTHORSHIP INFORMATION
__author__ = "HunMin Kim, David Pagnon"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["HunMin Kim", "David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## INIT
import os
import glob
import json
import logging
from tqdm import tqdm
import numpy as np
import cv2
import onnxruntime as ort
import subprocess

from rtmlib import PoseTracker, Body, Wholebody, BodyWithFeet, draw_skeleton
from Pose2Sim.common import natural_sort_key


## FUNCTIONS
def save_to_openpose(json_file_path, keypoints, scores):
    '''
    Save the keypoints and scores to a JSON file in the OpenPose format

    INPUTS:
    - json_file_path: Path to save the JSON file
    - keypoints: Detected keypoints
    - scores: Confidence scores for each keypoint

    OUTPUTS:
    - JSON file with the detected keypoints and confidence scores in the OpenPose format
    '''

    # Prepare keypoints with confidence scores for JSON output
    nb_detections = len(keypoints)
    # print('results: ', keypoints, scores)
    detections = []
    for i in range(nb_detections): # nb of detected people
        keypoints_with_confidence_i = []
        for kp, score in zip(keypoints[i], scores[i]):
            keypoints_with_confidence_i.extend([kp[0].item(), kp[1].item(), score.item()])
        detections.append({
                    "person_id": [-1],
                    "pose_keypoints_2d": keypoints_with_confidence_i,
                    "face_keypoints_2d": [],
                    "hand_left_keypoints_2d": [],
                    "hand_right_keypoints_2d": [],
                    "pose_keypoints_3d": [],
                    "face_keypoints_3d": [],
                    "hand_left_keypoints_3d": [],
                    "hand_right_keypoints_3d": []
                })
            
    # Create JSON output structure
    json_output = {"version": 1.3, "people": detections}
    
    # Save JSON output for each frame
    json_output_dir = os.path.abspath(os.path.join(json_file_path, '..'))
    if not os.path.isdir(json_output_dir): os.makedirs(json_output_dir)
    with open(json_file_path, 'w') as json_file:
        json.dump(json_output, json_file)


def process_video(video_path, pose_tracker, tracking, output_format, save_video, save_images, display_detection, frame_range):
    '''
    Estimate pose from a video file
    
    INPUTS:
    - video_path: str. Path to the input video file
    - pose_tracker: PoseTracker. Initialized pose tracker object from RTMLib
    - tracking: bool. Whether to give consistent person ID across frames
    - output_format: str. Output format for the pose estimation results ('openpose', 'mmpose', 'deeplabcut')
    - save_video: bool. Whether to save the output video
    - save_images: bool. Whether to save the output images
    - display_detection: bool. Whether to show real-time visualization
    - frame_range: list. Range of frames to process

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - if save_video: Video file with the detected keypoints and confidence scores drawn on the frames
    - if save_images: Image files with the detected keypoints and confidence scores drawn on the frames
    '''

    try:
        cap = cv2.VideoCapture(video_path)
        cap.read()
        if cap.read()[0] == False:
            raise
    except:
        raise NameError(f"{video_path} is not a video. Images must be put in one subdirectory per camera.")
    
    pose_dir = os.path.abspath(os.path.join(video_path, '..', '..', 'pose'))
    if not os.path.isdir(pose_dir): os.makedirs(pose_dir)
    video_name_wo_ext = os.path.splitext(os.path.basename(video_path))[0]
    json_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_json')
    output_video_path = os.path.join(pose_dir, f'{video_name_wo_ext}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{video_name_wo_ext}_img')
    
    if save_video: # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for the output video
        fps = cap.get(cv2.CAP_PROP_FPS) # Get the frame rate from the raw video
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Get the width and height from the raw video
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H)) # Create the output video file
        
    if display_detection:
        cv2.namedWindow(f"Pose Estimation {os.path.basename(video_path)}", cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)

    frame_idx = 0
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    f_range = [[total_frames] if frame_range==[] else frame_range][0]
    with tqdm(total=total_frames, desc=f'Processing {os.path.basename(video_path)}') as pbar:
        while cap.isOpened():
            # print('\nFrame ', frame_idx)
            success, frame = cap.read()
            if not success:
                break
            
            if frame_idx in range(*f_range):
                # Perform pose estimation on the frame
                keypoints, scores = pose_tracker(frame)

                # Reorder keypoints, scores
                if tracking:
                    max_id = max(pose_tracker.track_ids_last_frame)
                    num_frames, num_points, num_coordinates = keypoints.shape
                    keypoints_filled = np.zeros((max_id+1, num_points, num_coordinates))
                    scores_filled = np.zeros((max_id+1, num_points))
                    keypoints_filled[pose_tracker.track_ids_last_frame] = keypoints
                    scores_filled[pose_tracker.track_ids_last_frame] = scores
                    keypoints = keypoints_filled
                    scores = scores_filled

                # Save to json
                if 'openpose' in output_format:
                    json_file_path = os.path.join(json_output_dir, f'{video_name_wo_ext}_{frame_idx:06d}.json')
                    save_to_openpose(json_file_path, keypoints, scores)

                # Draw skeleton on the frame
                if display_detection or save_video or save_images:
                    img_show = frame.copy()
                    img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1) # maybe change this value if 0.1 is too low
                
                if display_detection:
                    cv2.imshow(f"Pose Estimation {os.path.basename(video_path)}", img_show)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                if save_video:
                    out.write(img_show)

                if save_images:
                    if not os.path.isdir(img_output_dir): os.makedirs(img_output_dir)
                    cv2.imwrite(os.path.join(img_output_dir, f'{video_name_wo_ext}_{frame_idx:06d}.png'), img_show)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    if save_video:
        out.release()
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection:
        cv2.destroyAllWindows()


def process_images(image_folder_path, vid_img_extension, pose_tracker, tracking, output_format, fps, save_video, save_images, display_detection, frame_range):
    '''
    Estimate pose estimation from a folder of images
    
    INPUTS:
    - image_folder_path: str. Path to the input image folder
    - vid_img_extension: str. Extension of the image files
    - pose_tracker: PoseTracker. Initialized pose tracker object from RTMLib
    - tracking: bool. Whether to give consistent person ID across frames
    - output_format: str. Output format for the pose estimation results ('openpose', 'mmpose', 'deeplabcut')
    - save_video: bool. Whether to save the output video
    - save_images: bool. Whether to save the output images
    - display_detection: bool. Whether to show real-time visualization
    - frame_range: list. Range of frames to process

    OUTPUTS:
    - JSON files with the detected keypoints and confidence scores in the OpenPose format
    - if save_video: Video file with the detected keypoints and confidence scores drawn on the frames
    - if save_images: Image files with the detected keypoints and confidence scores drawn on the frames
    '''    

    pose_dir = os.path.abspath(os.path.join(image_folder_path, '..', '..', 'pose'))
    if not os.path.isdir(pose_dir): os.makedirs(pose_dir)
    json_output_dir = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_json')
    output_video_path = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_pose.mp4')
    img_output_dir = os.path.join(pose_dir, f'{os.path.basename(image_folder_path)}_img')

    image_files = glob.glob(os.path.join(image_folder_path, '*'+vid_img_extension))
    sorted(image_files, key=natural_sort_key)

    if save_video: # Set up video writer
        logging.warning('Using default framerate of 60 fps.')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for the output video
        W, H = cv2.imread(image_files[0]).shape[:2][::-1] # Get the width and height from the first image (assuming all images have the same size)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (W, H)) # Create the output video file

    if display_detection:
        cv2.namedWindow(f"Pose Estimation {os.path.basename(image_folder_path)}", cv2.WINDOW_NORMAL)
    
    f_range = [[len(image_files)] if frame_range==[] else frame_range][0]
    for frame_idx, image_file in enumerate(tqdm(image_files, desc=f'\nProcessing {os.path.basename(img_output_dir)}')):
        if frame_idx in range(*f_range):

            try:
                frame = cv2.imread(image_file)
            except:
                raise NameError(f"{image_file} is not an image. Videos must be put in the video directory, not in subdirectories.")
            
            # Perform pose estimation on the image
            keypoints, scores = pose_tracker(frame)

            # Reorder keypoints, scores
            if tracking:
                max_id = max(pose_tracker.track_ids_last_frame)
                num_frames, num_points, num_coordinates = keypoints.shape
                keypoints_filled = np.zeros((max_id+1, num_points, num_coordinates))
                scores_filled = np.zeros((max_id+1, num_points))
                keypoints_filled[pose_tracker.track_ids_last_frame] = keypoints
                scores_filled[pose_tracker.track_ids_last_frame] = scores
                keypoints = keypoints_filled
                scores = scores_filled            
            
            # Extract frame number from the filename
            if 'openpose' in output_format:
                json_file_path = os.path.join(json_output_dir, f"{os.path.splitext(os.path.basename(image_file))[0]}_{frame_idx:06d}.json")
                save_to_openpose(json_file_path, keypoints, scores)

            # Draw skeleton on the image
            if display_detection or save_video or save_images:
                img_show = frame.copy()
                img_show = draw_skeleton(img_show, keypoints, scores, kpt_thr=0.1) # maybe change this value if 0.1 is too low

            if display_detection:
                cv2.imshow(f"Pose Estimation {os.path.basename(image_folder_path)}", img_show)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if save_video:
                out.write(img_show)

            if save_images:
                if not os.path.isdir(img_output_dir): os.makedirs(img_output_dir)
                cv2.imwrite(os.path.join(img_output_dir, f'{os.path.splitext(os.path.basename(image_file))[0]}_{frame_idx:06d}.png'), img_show)

    if save_video:
        logging.info(f"--> Output video saved to {output_video_path}.")
    if save_images:
        logging.info(f"--> Output images saved to {img_output_dir}.")
    if display_detection:
        cv2.destroyAllWindows()


def rtm_estimator(config_dict):
    '''
    Main function to estimate pose from videos or folders of images using the specified pose model.
    The results can be saved as JSON, videos, or images, and optionally displayed in real-time.

    Features:
    - Flexible support for multiple models (HALPE_26, COCO_133, COCO_17, and custom models via DeepLabCut).
    - Customizable detection frequency and consistent person tracking across frames.
    - Integration with DeepLabCut for custom models defined in the `Config.toml` file.
    - GPU support via ONNXRuntime or fallback to CPU using OpenVINO backend.
    
    Inputs:
    - A configuration dictionary (parsed from a `Config.toml` file).
    - Video files or image folders.

    Outputs:
    - Pose estimation results saved in JSON format.
    - Optionally, annotated videos and/or images.
    '''

    # Read the project directory from the configuration file
    project_dir = config_dict['project']['project_dir']  # Directory containing the project files
    
    # Determine session directory based on whether it is a batch or single trial
    session_dir = os.path.realpath(os.path.join(project_dir, '..'))  # For batch processing
    session_dir = session_dir if 'Config.toml' in os.listdir(session_dir) else os.getcwd()  # For single trial

    # Retrieve frame range and synchronization type from the configuration
    frame_range = config_dict.get('project').get('frame_range')  # Frame range to process
    synchronization_type = config_dict.get('synchronization').get('synchronization_type')  # Type of synchronization

    # Set the appropriate video directory based on the synchronization type
    if synchronization_type == 'move':
        video_dir = os.path.join(project_dir, 'videos_raw')  # For motion-based synchronization
    elif synchronization_type in ['sound', 'manual']:
        video_dir = os.path.join(project_dir, 'videos')  # For sound/manual-based synchronization

    # Define the directory where pose estimation results will be saved
    pose_dir = os.path.join(project_dir, 'pose')

    # Retrieve pose estimation parameters from the configuration file
    pose_model = config_dict['pose']['pose_model']  # Pose model to use (e.g., HALPE_26, COCO_133, etc.)
    mode = config_dict['pose']['mode']  # Mode for inference: lightweight, balanced, or performance
    vid_img_extension = config_dict['pose']['vid_img_extension']  # File extension for videos/images
    
    # Define output settings
    output_format = config_dict['pose']['output_format']  # Output format (e.g., 'openpose', 'deeplabcut', etc.)
    save_video = 'to_video' in config_dict['pose']['save_video']  # Whether to save videos with pose estimation
    save_images = 'to_images' in config_dict['pose']['save_video']  # Whether to save images with pose estimation
    display_detection = config_dict['pose']['display_detection']  # Display pose estimation in real-time
    overwrite_pose = config_dict['pose']['overwrite_pose']  # Overwrite existing results if True

    # Detection frequency and tracking options
    det_frequency = config_dict['pose']['det_frequency']  # Run detection every N frames
    tracking = config_dict['pose']['tracking']  # Enable consistent person ID tracking across frames

    # Determine the frame rate for video processing
    video_files = glob.glob(os.path.join(video_dir, '*' + vid_img_extension))  # Get all video files in the directory
    frame_rate = config_dict.get('project').get('frame_rate')  # Frame rate from the configuration
    if frame_rate == 'auto':  # Automatically determine frame rate
        try:
            cap = cv2.VideoCapture(video_files[0])  # Open the first video file
            cap.read()
            if not cap.read()[0]:
                raise ValueError("Failed to read video file.")
        except:
            frame_rate = 60  # Default to 60 FPS if auto-detection fails

    # Check if CUDA is available and set the appropriate backend
    # If CUDAExecutionProvider is available, use GPU with ONNXRuntime backend
    if 'CUDAExecutionProvider' in ort.get_available_providers():
        try:
            import torch
            if torch.cuda.is_available():  # Verify if CUDA is actually accessible
                device = 'cuda'
                backend = 'onnxruntime'
                logging.info(f"\nValid CUDA installation found: using ONNXRuntime backend with GPU.")
        except:
            pass  # Fallback in case of errors during import or detection
    # If MPSExecutionProvider or CoreMLExecutionProvider is available, use GPU with ONNXRuntime
    elif 'MPSExecutionProvider' in ort.get_available_providers() or 'CoreMLExecutionProvider' in ort.get_available_providers():
        device = 'mps'
        backend = 'onnxruntime'
        logging.info(f"\nValid MPS installation found: using ONNXRuntime backend with GPU.")
    # If no GPU is available, fallback to CPU with OpenVINO backend
    else:
        device = 'cpu'
        backend = 'openvino'
        logging.info(f"\nNo valid CUDA installation found: using OpenVINO backend with CPU.")

    # Log the detection frequency configuration
    if det_frequency > 1:
        logging.info(f'Inference run only every {det_frequency} frames. Inbetween, pose estimation tracks previously detected points.')
    elif det_frequency == 1:
        logging.info(f'Inference run on every single frame.')
    else:
        raise ValueError(f"Invalid det_frequency: {det_frequency}. Must be an integer greater or equal to 1.")
    
    # Log tracking option if enabled
    if tracking:
        logging.info(f'Pose estimation will attempt to give consistent person IDs across frames.\n')
        
    # Select the appropriate pose model based on the configuration
    if pose_model.upper() == 'HALPE_26':  # HALPE_26 for body and feet detection
        ModelClass = BodyWithFeet
        logging.info(f"Using HALPE_26 model (body and feet) for pose estimation.")
    elif pose_model.upper() == 'COCO_133':  # COCO_133 for body, feet, hands, and face detection
        ModelClass = Wholebody
        logging.info(f"Using COCO_133 model (body, feet, hands, and face) for pose estimation.")
    elif pose_model.upper() == 'COCO_17':  # COCO_17 for body detection only
        ModelClass = Body
        logging.info(f"Using COCO_17 model (body) for pose estimation.")
    elif pose_model.upper() == 'CUSTOM':  # Custom models like DeepLabCut
        print("Running custom model")
    else:  # Handle invalid model types
        raise ValueError(f"Invalid model_type: {pose_model}. Must be 'HALPE_26', 'COCO_133', or 'COCO_17'. Use another network (MMPose, DeepLabCut, OpenPose, AlphaPose, BlazePose...) and convert the output files if you need another model. See documentation.")
    
    # Log the selected mode of operation (e.g., lightweight, balanced, performance)
    logging.info(f'Mode: {mode}.\n')

    # Print device and backend information for debugging purposes
    print(device, backend)

        
    if pose_model.upper() == 'CUSTOM':
            # Pose2Sim specific operations (if any)
        print("Starting pose estimation with custom deeplabcut...")

        deeplabcut_env_path = config_dict['pose']['deeplabcut_env_path']
        config_DLC_project_path = config_dict['pose']['config_DLC_project_path']
        shuffle_number = config_dict['pose']['shuffle_number']
        


            # Construire le chemin vers deeplabcut_pose2sim.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dlc_script_path = os.path.join(current_dir, "Utilities", "deeplabcut_pose2sim.py")
        
        if not os.path.exists(config_DLC_project_path):
            raise FileNotFoundError(f"Le fichier de configuration {config_DLC_project_path} est introuvable.")
        
        # Command to activate DeepLabcut environment and run the script
        dlc_command = (
            f"conda activate DeepLabcut && python {dlc_script_path} "
            f"--deeplabcut_env_path '{deeplabcut_env_path}' "
            f"--config_DLC_project_path '{config_DLC_project_path}' "
            f"--shuffle_number {shuffle_number} "
            f"--output_folder 'pose-custom' "
            f"{'--display_detection' if display_detection else ''}"  # Ajouter uniquement si True
        )
    
            # Execute the command
        print("Switching to DeepLabcut environment and executing the script...")
        process = subprocess.run(dlc_command, shell=True, capture_output=True, text=True)
    
        # Output from the DeepLabcut script
        print("Output from DeepLabcut:")
        print(process.stdout)
    
        # Error handling
        if process.returncode != 0:
            print("An error occurred while running the DeepLabcut script:")
            print(process.stderr)
            return
    
        # Additional Pose2Sim operations after DeepLabcut completes
        print("Pose estimation completed. Returning to Pose2Sim operations...")
        
    else:
        # Initialize the pose tracker with the selected model class and configuration settings
        pose_tracker = PoseTracker(
            ModelClass,  # The selected model class (e.g., HALPE_26, COCO_133)
            det_frequency=det_frequency,  # Frequency for running detection
            mode=mode,  # Mode of operation (lightweight, balanced, performance)
            backend=backend,  # Backend being used (e.g., ONNXRuntime or OpenVINO)
            device=device,  # Hardware device being used (e.g., CPU, GPU)
            tracking=tracking,  # Whether consistent person IDs are assigned across frames
            to_openpose=False  # Output format flag (not exporting OpenPose format here)
        )

    
        import concurrent.futures
        
        # Logging information for pose estimation
        logging.info('\nEstimating pose...')
        try:
            pose_listdirs_names = next(os.walk(pose_dir))[1]
            os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]
            
            if not overwrite_pose:
                logging.info('Skipping pose estimation as it has already been done. Set overwrite_pose to true in Config.toml if you want to run it again.')
            else:
                logging.info('Overwriting previous pose estimation. Set overwrite_pose to false in Config.toml if you want to keep the previous results.')
                raise  # Force re-estimation of pose
        
        except:
            video_files = glob.glob(os.path.join(video_dir, '*' + vid_img_extension))
            if len(video_files) > 0:  # If video files are found
                logging.info(f'Found video files with extension {vid_img_extension}.')
        
                # Function to process a single video
                def process_single_video(video_path):
                    pose_tracker.reset()
                    process_video(
                        video_path,
                        pose_tracker,
                        tracking,
                        output_format,
                        save_video,
                        save_images,
                        display_detection,
                        frame_range
                    )
        
                # Use ThreadPoolExecutor for parallel processing
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {executor.submit(process_single_video, video): video for video in video_files}
                    
                    for future in concurrent.futures.as_completed(futures):
                        video_path = futures[future]
                        try:
                            future.result()  # Wait for the video processing to complete
                            logging.info(f"Completed pose estimation for {video_path}.")
                        except Exception as exc:
                            logging.error(f"Error processing video {video_path}: {exc}")

            else:
                # If no video files are found, check for image folders instead
                logging.info(f'Found image folders with extension {vid_img_extension}.')
                # Identify all subdirectories in the video directory
                image_folders = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
                for image_folder in image_folders:  # Process each image folder
                    pose_tracker.reset()  # Reset the pose tracker for each folder
                    image_folder_path = os.path.join(video_dir, image_folder)  # Construct the folder path
                    process_images(  # Call the function to process the folder of images
                        image_folder_path,
                        vid_img_extension,
                        pose_tracker,
                        tracking,
                        output_format,
                        frame_rate,
                        save_video,
                        save_images,
                        display_detection,
                        frame_range
                    )
