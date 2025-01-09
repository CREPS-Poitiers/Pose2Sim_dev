# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
#########################################
## VIDEO SYNCHRONIZATION SCRIPT         #
#########################################

Author: F.Delaplace (based on work by David Pagnon and HunMin Kim)
Version: 0.9.4
Creation Date: September 23, 2024

Description:
-------------
This script provides functionality to synchronize videos captured by multiple cameras, specifically for motion analysis purposes. 
It includes several synchronization methods, each tailored for different use cases:

1. **Synchronization by Vertical Acceleration (Default)**:
   - Requires pose estimation data (e.g., JSON files from a pose estimation pipeline) to calculate vertical speeds of keypoints.
   - Analyzes the vertical motion of keypoints across videos to compute temporal offsets.
   - Suitable for scenarios involving distinct, vertical movements (e.g., jumping, walking).

2. **Audio-Based Synchronization**:
   - No prior pose estimation is required.
   - Relies on audio signals embedded within the videos to determine synchronization offsets.
   - Best suited for indoor environments with minimal background noise, but less effective outdoors.

3. **Manual Synchronization**:
   - No pose estimation data is required.
   - Allows the user to manually select a reference frame in one video and align other videos accordingly.
   - Highly reliable and works in all conditions, including complex or noisy environments.

4. **Video Mosaic Creation**:
   - Combines multiple synchronized videos into a single mosaic view for easy comparison.
   - Adds annotations (camera names, offsets) for clarity.

Key Features:
--------------
- Automatic and manual video synchronization options.
- Vertical acceleration synchronization requires pre-computed pose estimation data.
- Audio and manual synchronization methods do not depend on pose estimation, making them versatile.
- Frame-by-frame synchronization with visualization of correlation plots.
- Butterworth filtering for smooth motion analysis.
- Support for multi-camera setups and multiple trials.
- Options to define custom start and end frames for processing.

Intended Use:
-------------
This script is designed for researchers and engineers in fields like biomechanics, sports science, and motion capture, providing robust tools for post-synchronizing videos for accurate analysis.

Dependencies:
-------------
- OpenCV
- NumPy
- pandas
- SciPy
- tqdm
- keyboard
- screeninfo
- ffmpeg
- skelly_synchronize
"""

## AUTHORSHIP INFORMATION
__author__ = "David Pagnon, HunMin Kim"
__copyright__ = "Copyright 2021, Pose2Sim"
__credits__ = ["David Pagnon"]
__license__ = "BSD 3-Clause License"
__version__ = "0.9.4"
__maintainer__ = "David Pagnon"
__email__ = "contact@david-pagnon.com"
__status__ = "Development"


## INIT
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from scipy import interpolate
import json
import os
import glob
import fnmatch
import time
from datetime import datetime
import re
import shutil
from anytree import RenderTree
from anytree.importer import DictImporter
import logging
import toml
import tomlkit
from tqdm import tqdm
from Pose2Sim.common import sort_stringlist_by_last_number
from Pose2Sim.skeletons import *
from pathlib import Path
import ffmpeg
from tqdm import tqdm
import keyboard  # Bibliothèque pour écouter les touches
import screeninfo
import sys
import threading
from queue import Queue

#CHANGER ICI si on veut la skelly de base ou ameliorée avec le filtrage (skelly_synchronize ou skelly_synchronize_dev)
from skelly_synchronize import skelly_synchronize_dev as sync



# FUNCTIONS
def convert_json2pandas(json_files, likelihood_threshold=0.6):
    '''
    Convert a list of JSON files to a pandas DataFrame.

    INPUTS:
    - json_files: list of str. Paths of the the JSON files.
    - likelihood_threshold: float. Drop values if confidence is below likelihood_threshold.
    - frame_range: select files within frame_range.

    OUTPUTS:
    - df_json_coords: dataframe. Extracted coordinates in a pandas dataframe.
    '''

    nb_coord = 25 # int(len(json_data)/3)
    json_coords = []
    for j_p in json_files:
        with open(j_p) as j_f:
            try:
                json_data = json.load(j_f)['people'][0]['pose_keypoints_2d']
                # remove points with low confidence
                json_data = np.array([[json_data[3*i],json_data[3*i+1],json_data[3*i+2]] if json_data[3*i+2]>likelihood_threshold else [0.,0.,0.] for i in range(nb_coord)]).ravel().tolist()
            except:
                # print(f'No person found in {os.path.basename(json_dir)}, frame {i}')
                json_data = [np.nan] * 25*3
        json_coords.append(json_data)
    df_json_coords = pd.DataFrame(json_coords)

    return df_json_coords


def drop_col(df, col_nb):
    '''
    Drops every nth column from a DataFrame.

    INPUTS:
    - df: dataframe. The DataFrame from which columns will be dropped.
    - col_nb: int. The column number to drop.

    OUTPUTS:
    - dataframe: DataFrame with dropped columns.
    '''

    idx_col = list(range(col_nb-1, df.shape[1], col_nb)) 
    df_dropped = df.drop(idx_col, axis=1)
    df_dropped.columns = range(df_dropped.columns.size)
    return df_dropped


def vert_speed(df, axis='y'):
    '''
    Calculate the vertical speed of a DataFrame along a specified axis.

    INPUTS:
    - df: dataframe. DataFrame of 2D coordinates.
    - axis: str. The axis along which to calculate speed. 'x', 'y', or 'z', default is 'y'.

    OUTPUTS:
    - df_vert_speed: DataFrame of vertical speed values.
    '''

    axis_dict = {'x':0, 'y':1, 'z':2}
    df_diff = df.diff()
    df_diff = df_diff.fillna(df_diff.iloc[1]*2)
    df_vert_speed = pd.DataFrame([df_diff.loc[:, 2*k + axis_dict[axis]] for k in range(int(df_diff.shape[1] / 2))]).T # modified ( df_diff.shape[1]*2 to df_diff.shape[1] / 2 )
    df_vert_speed.columns = np.arange(len(df_vert_speed.columns))
    return df_vert_speed


def interpolate_zeros_nans(col, kind):
    '''
    Interpolate missing points (of value nan)

    INPUTS:
    - col: pandas column of coordinates
    - kind: 'linear', 'slinear', 'quadratic', 'cubic'. Default 'cubic'

    OUTPUTS:
    - col_interp: interpolated pandas column
    '''
    
    mask = ~(np.isnan(col) | col.eq(0)) # true where nans or zeros
    idx_good = np.where(mask)[0]
    try: 
        f_interp = interpolate.interp1d(idx_good, col[idx_good], kind=kind, bounds_error=False)
        col_interp = np.where(mask, col, f_interp(col.index))
        return col_interp 
    except:
        # print('No good values to interpolate')
        return col


def time_lagged_cross_corr(camx, camy, lag_range, show=True, ref_cam_id=0, cam_id=1):
    '''
    Compute the time-lagged cross-correlation between two pandas series.

    INPUTS:
    - camx: pandas series. Coordinates of reference camera.
    - camy: pandas series. Coordinates of camera to compare.
    - lag_range: int or list. Range of frames for which to compute cross-correlation.
    - show: bool. If True, display the cross-correlation plot.
    - ref_cam_id: int. The reference camera id.
    - cam_id: int. The camera id to compare.

    OUTPUTS:
    - offset: int. The time offset for which the correlation is highest.
    - max_corr: float. The maximum correlation value.
    '''

    if isinstance(lag_range, int):
        lag_range = [-lag_range, lag_range]

    import hashlib
    print(repr(list(camx)), repr(list(camy)))
    hashlib.md5(pd.util.hash_pandas_object(camx).values).hexdigest()
    hashlib.md5(pd.util.hash_pandas_object(camy).values).hexdigest()

    pearson_r = [camx.corr(camy.shift(lag)) for lag in range(lag_range[0], lag_range[1])]
    offset = int(np.floor(len(pearson_r)/2)-np.argmax(pearson_r))
    if not np.isnan(pearson_r).all():
        max_corr = np.nanmax(pearson_r)

        if show:
            f, ax = plt.subplots(2,1)
            # speed
            camx.plot(ax=ax[0], label = f'Reference: camera #{ref_cam_id}')
            camy.plot(ax=ax[0], label = f'Compared: camera #{cam_id}')
            ax[0].set(xlabel='Frame', ylabel='Speed (px/frame)')
            ax[0].legend()
            # time lagged cross-correlation
            ax[1].plot(list(range(lag_range[0], lag_range[1])), pearson_r)
            ax[1].axvline(np.ceil(len(pearson_r)/2) + lag_range[0],color='k',linestyle='--')
            ax[1].axvline(np.argmax(pearson_r) + lag_range[0],color='r',linestyle='--',label='Peak synchrony')
            plt.annotate(f'Max correlation={np.round(max_corr,2)}', xy=(0.05, 0.9), xycoords='axes fraction')
            ax[1].set(title=f'Offset = {offset} frames', xlabel='Offset (frames)',ylabel='Pearson r')
            
            plt.legend()
            f.tight_layout()
            plt.show()
    else:
        max_corr = 0
        offset = 0
        if show:
            # print('No good values to interpolate')
            pass

    return offset, max_corr


def extract_camera_number(filename):
    """Extract the camera number from the filename, assuming format like CAMERA01.mp4"""
    match = re.search(r"CAMERA(\d+)", filename)
    if match:
        return int(match.group(1))
    return 0  # If impossible to identify a camera number in the name


def synchroMosaique(trial_folder):
    """
    Creates a synchronized video mosaic from multiple camera videos in a given folder.
    """

    # Step 1: Retrieve and sort video files in the folder
    # Collect all .mp4 files in the trial folder
    video_files = [f for f in os.listdir(trial_folder) if f.lower().endswith(".mp4")]
    video_files.sort(key=extract_camera_number)  # Sort videos by their camera number

    nbVideos = len(video_files)  # Count the number of videos

    # If no videos are found, terminate the function
    if nbVideos == 0:
        print(f"No videos found in {trial_folder}")
        return

    # Step 2: Determine the dimensions of the mosaic grid
    # Start with a 2x2 grid and increase grid size until it can accommodate all videos
    dimOverlay = 2
    while nbVideos / dimOverlay > dimOverlay:
        dimOverlay += 1

    # Step 3: Build the FFmpeg command to create the mosaic
    ffmpeg_cmd = "ffmpeg"  # Initialize the FFmpeg command

    # Add each video as an input to the FFmpeg command
    for vid in video_files:
        ffmpeg_cmd += f" -i {os.path.join(trial_folder, vid)}"

    # Define the filter_complex section for FFmpeg
    filter_complex = f' -filter_complex "nullsrc=size=1920x1080 [base];'

    # Resize each video and prepare them for placement in the grid
    for i in range(nbVideos):
        filter_complex += f"[{i}:v] setpts=PTS-STARTPTS, scale={int(1920/dimOverlay)}x{int(1080/dimOverlay)} [v{i}];"

    # Step 4: Position videos within the mosaic grid
    xinc = 1920 // dimOverlay  # Horizontal increment per video
    yinc = 1080 // dimOverlay  # Vertical increment per video
    vidInc = 0  # Video index counter

    # Iterate through the grid rows and columns to place videos
    for y in range(dimOverlay):
        ypos = y * yinc  # Calculate vertical position
        for x in range(dimOverlay):
            xpos = x * xinc  # Calculate horizontal position
            if vidInc < nbVideos:
                if vidInc == 0:
                    # First video is overlaid on the base
                    filter_complex += f"[base][v{vidInc}] overlay=shortest=1:x={xpos}:y={ypos} [tmp{vidInc}];"
                elif vidInc == nbVideos - 1:
                    # Last video completes the filter complex
                    filter_complex += f"[tmp{vidInc-1}][v{vidInc}] overlay=shortest=1:x={xpos}:y={ypos}\" "
                else:
                    # Intermediate videos
                    filter_complex += f"[tmp{vidInc-1}][v{vidInc}] overlay=shortest=1:x={xpos}:y={ypos} [tmp{vidInc}];"
                vidInc += 1

    # Step 5: Save the mosaic video without annotations
    mosaic_path = os.path.join(trial_folder, "debug_files", "SyncVideos.mp4")
    ffmpeg_cmd += filter_complex + f" -c:v libx264 {mosaic_path} -y"

    # Redirect FFmpeg output to suppress logs
    if os.name == 'nt':  # Windows
        ffmpeg_cmd += " > NUL 2>&1"
    else:  # Linux/Mac
        ffmpeg_cmd += " > /dev/null 2>&1"

    # Execute the FFmpeg command to create the mosaic
    print(f"Creating mosaic for {trial_folder}")
    os.system(ffmpeg_cmd)

    # Check if the mosaic video was created successfully
    if not os.path.exists(mosaic_path):
        print(f"Error: The video {mosaic_path} was not created.")
        return

    # Step 6: Add annotations to the mosaic
    output_with_text_path = os.path.join(trial_folder, "debug_files", "SyncVideos_with_text.mp4")
    text_commands = ""  # Initialize annotation commands

    vidInc = 0  # Reset video index counter
    for y in range(dimOverlay):
        ypos = y * yinc + 10  # Vertical position for text
        for x in range(dimOverlay):
            xpos = x * xinc + 30  # Horizontal position for text
            if vidInc < nbVideos:
                # Add "CAMERA X" annotation for each video
                text_commands += f"drawtext=text='CAMERA {vidInc + 1}':fontcolor=black:fontsize=24:x={xpos}:y={ypos},"
                vidInc += 1

    # Add a prompt asking the user to save or delete the mosaic
    text_commands += "drawtext=text='Do you want to keep this mosaic? Type Y to save or N to delete':fontcolor=white:fontsize=24:x=(w-text_w)/2:y=h-th-50"

    # Execute FFmpeg command to add annotations
    ffmpeg_text_cmd = f"ffmpeg -i {mosaic_path} -vf \"{text_commands}\" -c:v libx264 {output_with_text_path} -y"

    # Redirect FFmpeg output to suppress logs
    if os.name == 'nt':  # Windows
        ffmpeg_text_cmd += " > NUL 2>&1"
    else:  # Linux/Mac
        ffmpeg_text_cmd += " > /dev/null 2>&1"

    print(f"Adding text to mosaic for {trial_folder}")
    os.system(ffmpeg_text_cmd)

    # Check if the annotated mosaic was created successfully
    if not os.path.exists(output_with_text_path):
        print(f"Error: The video {output_with_text_path} was not created.")
        return

    # Step 7: Preview and decide to save or delete the mosaic
    os.system(f"start {output_with_text_path}")  # Open the annotated mosaic (Windows only)

    print("Press 'y' to save the video, or 'n' to delete it. (You can press these keys while the video is playing.)")
    while True:
        if keyboard.is_pressed('y'):  # Save the annotated mosaic
            print(f"Video {output_with_text_path} has been saved.")
            print("Closing the video window.")
            os.system("taskkill /im vlc.exe /f")
            os.remove(mosaic_path)  # Delete the unannotated mosaic
            break
        elif keyboard.is_pressed('n'):  # Delete the annotated mosaic
            print("Closing the video window.")
            os.system("taskkill /im vlc.exe /f")
            os.remove(output_with_text_path)  # Delete the annotated mosaic
            os.remove(mosaic_path)  # Delete the unannotated mosaic
            print(f"Video {output_with_text_path} has been deleted.")
            break

        

        
"""
========================================================
____________Functions for manual synchro_______________
========================================================
"""

# Global variables for zoom and pan functionality
zoom_scale = 1.0  # Initial zoom level
pan_x, pan_y = 0, 0  # Initial pan offsets
dragging = False  # Flag to indicate if dragging is in progress
start_x, start_y = 0, 0  # Variables to store the starting point of drag

# Opens a video file and returns the video capture object
def open_video(video_path):
    """
    Opens the video file at the given path and returns a cv2.VideoCapture object.

    Args:
        video_path (str): Path to the video file.

    Returns:
        cv2.VideoCapture: OpenCV video capture object if successful.
        None: If the video file cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)  # Create a video capture object
    if not cap.isOpened():  # Check if the video file is accessible
        print(f"Error: Unable to open the video {video_path}")  # Print error if video cannot be opened
        return None
    return cap  # Return the video capture object


# Retrieves the screen size to enforce maximum window size
def get_screen_size():
    """
    Gets the screen resolution of the primary monitor.

    Returns:
        tuple: Width and height of the screen in pixels.
    """
    screen = screeninfo.get_monitors()[0]  # Access the primary monitor information
    return screen.width, screen.height  # Return screen width and height


# Adjust the window dimensions to fit the screen size
screen_width, screen_height = get_screen_size()  # Global variables for screen resolution


# Finds the video with the fewest frames to use as a reference
def find_reference_video(video_paths):
    """
    Finds the video with the minimum number of frames from a list of video paths.
    This video is considered as the reference for synchronization.

    Args:
        video_paths (list of str): List of paths to video files.

    Returns:
        tuple: Path to the reference video and the total number of frames in it.
    """
    min_frames = float('inf')  # Initialize with a very large number
    reference_path = None  # Initialize the reference path as None

    for video_path in video_paths:
        video = open_video(video_path)  # Open the video file
        if video:  # Proceed if the video file is successfully opened
            total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
            # Update reference if this video has fewer frames
            if total_frames < min_frames:
                min_frames = total_frames
                reference_path = video_path
            video.release()  # Release the video capture object

    return reference_path, min_frames  # Return the reference video path and its frame count


# Callback to handle mouse events for zooming and panning
def mouse_callback(event, x, y, flags, param):
    """
    Handles mouse events for zooming and panning within a video frame.

    Args:
        event (int): The type of mouse event (e.g., button press, release, wheel scroll).
        x (int): The x-coordinate of the mouse pointer.
        y (int): The y-coordinate of the mouse pointer.
        flags (int): Additional flags passed by OpenCV (e.g., scroll direction).
        param: Additional parameters (not used here).

    Modifies:
        Updates global variables `pan_x`, `pan_y`, `dragging`, `start_x`, `start_y`, and `zoom_scale` 
        to track panning and zooming state.
    """
    global pan_x, pan_y, dragging, start_x, start_y, zoom_scale
    if event == cv2.EVENT_LBUTTONDOWN:  # Mouse button pressed
        dragging = True  # Start panning
        start_x, start_y = x, y  # Store starting point for the drag
    elif event == cv2.EVENT_LBUTTONUP:  # Mouse button released
        dragging = False  # Stop panning
    elif event == cv2.EVENT_MOUSEMOVE and dragging:  # Mouse move while dragging
        # Invert the movement for panning effect
        pan_x -= (x - start_x)
        pan_y -= (y - start_y)
        start_x, start_y = x, y  # Update starting point for the next move
    elif event == cv2.EVENT_MOUSEWHEEL:  # Mouse wheel scrolled
        if flags > 0:  # Scroll up to zoom in
            zoom_in(x, y)
        else:  # Scroll down to zoom out
            zoom_out(x, y)


# Zoom in function with dynamic adjustment for the view center
def zoom_in(center_x, center_y):
    """
    Zooms in on the frame while adjusting the view center.

    Args:
        center_x (int): The x-coordinate of the zoom center.
        center_y (int): The y-coordinate of the zoom center.

    Modifies:
        Adjusts `zoom_scale` to increase zoom and re-centers the view.
    """
    global zoom_scale, pan_x, pan_y
    zoom_scale += 0.1  # Increase zoom level
    # Adjust pan to re-center the view based on the zoomed area
    pan_x = int((pan_x + center_x) * 1.1 - center_x)
    pan_y = int((pan_y + center_y) * 1.1 - center_y)


# Zoom out function with limits to prevent excessive zooming out
def zoom_out(center_x, center_y):
    """
    Zooms out on the frame while adjusting the view center.

    Args:
        center_x (int): The x-coordinate of the zoom center.
        center_y (int): The y-coordinate of the zoom center.

    Modifies:
        Adjusts `zoom_scale` to decrease zoom and re-centers the view.
    """
    global zoom_scale, pan_x, pan_y
    zoom_scale = max(zoom_scale - 0.1, 0.1)  # Decrease zoom level but prevent going below 0.1
    # Adjust pan to re-center the view based on the zoomed-out area
    pan_x = int((pan_x + center_x) / 1.1 - center_x)
    pan_y = int((pan_y + center_y) / 1.1 - center_y)


# Preload frames around a specific frame for smoother navigation
def preload_frames(video, center_frame, range_frames=50):
    """
    Preloads frames around a specific frame within a given range for smoother frame navigation.

    Args:
        video (cv2.VideoCapture): The video object to preload frames from.
        center_frame (int): The central frame to preload around.
        range_frames (int): Number of frames to preload before and after the central frame.

    Returns:
        dict: A dictionary where keys are frame numbers and values are the corresponding frames.
    """
    preloaded_frames = {}  # Initialize a dictionary to store preloaded frames
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total number of frames in the video

    # Define the range of frames to preload
    start_frame = max(0, center_frame - range_frames)  # Ensure starting frame is within bounds
    end_frame = min(total_frames, center_frame + range_frames + 1)  # Ensure ending frame is within bounds

    # Iterate through the defined range and preload frames
    for i in range(start_frame, end_frame):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)  # Set the video position to the current frame
        ret, frame = video.read()  # Read the frame
        if ret:  # If frame is read successfully
            preloaded_frames[i] = frame  # Add the frame to the dictionary

    return preloaded_frames  # Return the dictionary of preloaded frames


def navigate_frames(video, window_name="Select Reference Frame", start_frame=0):
    """
    Allows the user to navigate through video frames with zoom and pan capabilities,
    and select a reference frame.

    Args:
        video (cv2.VideoCapture): The video object to navigate.
        window_name (str): Name of the window where the frames are displayed.
        start_frame (int): Frame number to start navigation from.

    Returns:
        tuple: The selected frame (as an image) and its frame number.
    """
    global zoom_scale, pan_x, pan_y
    zoom_scale, pan_x, pan_y = 1.0, 0, 0  # Reset zoom and pan settings
    frame_number = start_frame  # Initialize the starting frame
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video
    fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second of the video
    selected_frame = None  # Variable to store the selected frame

    # Preloading frames in the background for smoother navigation
    preload_lock = threading.Lock()  # Lock to control access to shared resources
    preload_queue = {}  # Dictionary to store preloaded frames
    stop_preloading = threading.Event()  # Event to signal the preloading thread to stop

    def preload_frames():
        """Preloads frames around the current frame for smoother navigation."""
        while not stop_preloading.is_set():
            with preload_lock:
                preload_start = max(0, frame_number - 10)  # Start preloading 10 frames before the current frame
                preload_end = min(total_frames, frame_number + 10)  # End preloading 10 frames after the current frame
                for i in range(preload_start, preload_end):
                    if i not in preload_queue:  # If the frame is not already preloaded
                        video.set(cv2.CAP_PROP_POS_FRAMES, i)  # Set the video to the specific frame
                        ret, frame = video.read()  # Read the frame
                        if ret:  # If the frame is successfully read
                            preload_queue[i] = frame  # Add the frame to the preloading queue
            time.sleep(0.1)  # Pause to avoid excessive CPU usage

    # Start the preloading thread
    preload_thread = threading.Thread(target=preload_frames, daemon=True)
    preload_thread.start()

    # Variables for handling key presses
    right_delay = 0.1   # Delay for step-by-step navigation
    left_delay=0.1     # Delay for step-by-step navigation
    last_right_left_time = 0  # Time of the last key press
    
    last_up_down_time = 0  # Time of the last up/down key press
    up_down_delay = 0.3  # Delay to prevent accidental multiple up/down presses

    # Create the display window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.resizeWindow(window_name, screen_width, screen_height)  # Resize the window to screen dimensions
    cv2.setMouseCallback(window_name, mouse_callback)  # Attach the mouse callback for zoom and pan

    while True:
        # Check if the current frame is preloaded
        with preload_lock:
            frame = preload_queue.get(frame_number, None)  # Get the preloaded frame

        if frame is None:  # If the frame is not preloaded
            print(f"Frame {frame_number} is loading...")  # Notify the user
            time.sleep(0.05)  # Pause to allow the preloading thread to catch up
            continue

        # Calculate the time corresponding to the current frame
        time_in_seconds = frame_number / fps

        # Apply zoom and pan to the frame
        zoomed_frame = apply_zoom_and_pan(
            frame,
            screen_height // 2,
            frame_number=frame_number,
            time_in_seconds=time_in_seconds
        )

        # Display frame number and timestamp on the video
        cv2.putText(zoomed_frame, f"Frame: {frame_number}/{total_frames}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(zoomed_frame, f"Time: {time_in_seconds:.2f}s", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display navigation instructions
        instructions = [
            "Controls:",
            "- Left/Right: Frame -/+",
            "- Up/Down: Jump +/- 100 frames",
            "- [v]: Validate the current frame",
            "- [f]: Enter a frame number",
            "- [q]: Quit"
        ]
        for i, line in enumerate(instructions):
            cv2.putText(zoomed_frame, line, (10, 90 + i * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Show the frame in the window
        cv2.imshow(window_name, zoomed_frame)

        # Handle user input
        key = cv2.waitKey(1)  # Wait for a key press
        current_time = time.time()  # Get the current time

        if keyboard.is_pressed('q'):  # Quit the program
            print("Program stopped by the user.")
            break
        elif keyboard.is_pressed('v'):  # Validate the current frame
            selected_frame = zoomed_frame.copy()  # Save the selected frame
            print(f"Selected frame: {frame_number}")
            break
        elif keyboard.is_pressed('f'):  # Jump to a specific frame
            print("Input mode activated: Enter a frame number.")
            while True:
                try:
                    frame_number = int(input("Enter frame number: "))
                    frame_number = max(0, min(frame_number, total_frames - 1))  # Clamp the frame number
                    print(f"Navigated to frame {frame_number}.")
                    break
                except ValueError:
                    print("Invalid frame number. Please try again.")
        elif keyboard.is_pressed('left'):  # Step back one frame
            if current_time - last_right_left_time > left_delay:
                frame_number = max(0, frame_number - 1)
                last_right_left_time = current_time
        elif keyboard.is_pressed('right'):  # Step forward one frame
            if current_time - last_right_left_time > right_delay:
                frame_number = min(total_frames - 1, frame_number + 1)
                last_right_left_time = current_time
        elif keyboard.is_pressed('up'):  # Jump forward 100 frames
            if current_time - last_up_down_time > up_down_delay:
                frame_number = min(total_frames - 1, frame_number + 100)
                last_up_down_time = current_time
        elif keyboard.is_pressed('down'):  # Jump back 100 frames
            if current_time - last_up_down_time > up_down_delay:
                frame_number = max(0, frame_number - 100)
                last_up_down_time = current_time

    # Stop preloading and clean up resources
    stop_preloading.set()
    preload_thread.join()
    cv2.destroyWindow(window_name)

    return selected_frame, frame_number  # Return the selected frame and frame number


def calculate_min_frames(video_paths, offsets, start_frame, end_frame):
    """
    Calculate the minimum duration (in frames) of all videos after applying the offsets.

    Args:
        video_paths (list): List of paths to the video files.
        offsets (dict): Dictionary containing offsets for each video, where keys are video names and values are offsets.
        start_frame (int): Starting frame for the videos.
        end_frame (int): Ending frame for the videos.

    Returns:
        int: The minimum number of frames available across all videos after accounting for offsets.
    """
    # Calculate the initial duration of the videos based on start and end frames
    min_frames = int(end_frame - start_frame)
    actual_min_frames = min_frames  # Initialize the effective minimum duration

    # Find the most negative offset (largest time shift backward)
    min_offset = min(offsets.values())

    # Iterate over each video to calculate the effective frame duration
    for video_path in video_paths:
        # Extract the base name of the video (without extension) to match the offsets dictionary
        video_name = os.path.basename(video_path).split('.')[0]
        
        # Retrieve the offset for the current video (default to 0 if not found)
        offset = int(offsets.get(video_name, 0))
        
        # Calculate the number of frames to cut from the start based on the most negative offset
        frames_to_cut = abs(min_offset) + offset
        
        # Calculate the effective duration for the video
        effective_duration = min_frames - frames_to_cut
        
        # Update the overall minimum duration across all videos
        actual_min_frames = min(actual_min_frames, effective_duration)

    return actual_min_frames


def synchronize_videos(video_paths, video_save):
    """
    Synchronizes a list of videos by aligning their frames based on a reference video.

    Args:
        video_paths (list): List of paths to the video files to be synchronized.
        video_save (str): Directory where synchronized videos will be saved.

    """
    sync_data = {}  # Dictionary to store synchronization details (offsets and frame counts).

    # Ask the user if they want a custom start and end cut for the videos.
    custom_cut = ask_for_custom_cut()

    # Initialize default start and end frames.
    start_frame, end_frame = 0, None

    # Identify the reference video as the one with the least number of frames.
    reference_path, min_frames = find_reference_video(video_paths)
    min_frames = int(min_frames)  # Ensure the frame count is an integer.
    video_paths.remove(reference_path)  # Remove the reference video from the list.
    video_paths.insert(0, reference_path)  # Add the reference video to the first position.

    if custom_cut:
        # Allow the user to specify a custom range using the reference video.
        reference_video = open_video(video_paths[0])
        if reference_video is not None:
            print(f"Selecting the cutting range for the reference video: {video_paths[0]}")
            start_frame, end_frame = select_start_end_frames(reference_video)
            reference_video.release()
            print(f"Range defined: Start at {start_frame}, End at {end_frame}")
    else:
        # Use the entire video length if no custom range is specified.
        reference_video = open_video(video_paths[0])
        if reference_video is not None:
            end_frame = int(reference_video.get(cv2.CAP_PROP_FRAME_COUNT))
            reference_video.release()

    # Adjust `min_frames` if a custom range is defined.
    if end_frame is not None:
        min_frames = end_frame - start_frame

    offsets = {}  # Dictionary to store offsets for each video.
    max_negative_offset = 0  # Track the largest negative offset.

    # Iterate over the videos to calculate offsets.
    for idx, video_path in enumerate(video_paths):
        video = open_video(video_path)

        if video is None:
            continue

        video_name = os.path.basename(video_path).split('.')[0]

        if idx == 0:
            # The first video (reference video) has an offset of 0.
            print(f"Defining the reference frame for the video: {video_path}")
            reference_window_name = "Reference Video"
            reference_frame, reference_image = navigate_frames(video, reference_window_name)
            offsets[video_name] = 0
            if cv2.getWindowProperty(reference_window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(reference_window_name)
        else:
            # Synchronize the current video by aligning it with the reference frame.
            print(f"Synchronizing video {video_path} with the reference frame.")
            offset = synchronize_with_reference(video, reference_frame, reference_image, video_name)
            offsets[video_name] = offset

            # Update the largest negative offset if necessary.
            if offset < max_negative_offset:
                max_negative_offset = offset

        video.release()

    # Calculate the starting point for all videos based on the largest negative offset.
    cut_start = abs(max_negative_offset)
    print(f"Initial cut (cut_start): {cut_start} frames.")

    # Calculate the minimum effective duration for all videos.
    actual_min_frames = calculate_min_frames(video_paths, offsets, start_frame, end_frame)
    print(f"Final duration after trimming: {actual_min_frames} frames.")

    # Trim and save each video.
    for video_path in video_paths:
        video_name = os.path.basename(video_path).split('.')[0]
        offset = offsets[video_name]
        output_path = os.path.join(video_save, f"sync_{video_name}.MP4")

        # Adjust the starting frame based on the offset.
        start_frame_adjusted = cut_start + offset
        final_frame_count = cut_video(video_path, output_path, start_frame_adjusted, actual_min_frames)

        # Save synchronization details for the current video.
        sync_data[video_name] = {"offset": offset, "final_frame_count": final_frame_count}
        print(f"Synchronized video saved for {video_name} at {output_path}")

    # Save synchronization details to a .toml file.
    if not os.path.exists(os.path.join(video_save, "debug_files")):
        os.mkdir(os.path.join(video_save, "debug_files"))
    toml_path = os.path.join(video_save, "debug_files", "synchronization_debug.toml")
    with open(toml_path, "w") as toml_file:
        toml.dump(sync_data, toml_file)
    print(f"Synchronization offsets and frame counts saved to {toml_path}")

def synchronize_with_reference(video, reference_frame, reference_frame_number, video_name):
    """
    Synchronizes a video with a reference frame by determining the offset in frames.

    Args:
        video: cv2.VideoCapture object for the video to be synchronized.
        reference_frame: The reference frame to compare against.
        reference_frame_number: The frame number of the reference frame.
        video_name: The name of the video being synchronized.

    Returns:
        offset: The frame offset of the video compared to the reference.
    """
    global zoom_scale, pan_x, pan_y
    zoom_scale, pan_x, pan_y = 1.0, 0, 0  # Reset zoom and pan values for comparison

    # Setup windows for displaying reference and comparison frames
    compare_window = "Comparison Frame"
    reference_window = "Reference Frame"

    # Create a window to display the fixed reference frame
    cv2.namedWindow(reference_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(reference_window, screen_width // 2, screen_height // 2)
    cv2.moveWindow(reference_window, 0, 0)

    # Resize the reference frame to fit the screen without distortion
    ref_height, ref_width = reference_frame.shape[:2]
    scaling_factor = min(screen_height // 2 / ref_height, screen_width // 2 / ref_width)
    resized_reference = cv2.resize(
        reference_frame, 
        (int(ref_width * scaling_factor), int(ref_height * scaling_factor)), 
        interpolation=cv2.INTER_LINEAR
    )
    cv2.imshow(reference_window, resized_reference)  # Show the fixed reference frame

    # Create a window for displaying the comparison frame
    cv2.namedWindow(compare_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(compare_window, screen_width // 2, screen_height // 2)
    cv2.moveWindow(compare_window, screen_width // 2, 0)
    cv2.setMouseCallback(compare_window, mouse_callback)  # Attach mouse controls for zoom and pan

    frame_number = reference_frame_number  # Start comparison from the same frame as the reference
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames in the video

    # Setup for preloading frames in the background
    preload_lock = threading.Lock()
    preload_queue = {}
    stop_preloading = threading.Event()

    def preload_frames():
        """
        Preload frames around the current frame to improve responsiveness during navigation.
        """
        while not stop_preloading.is_set():
            with preload_lock:
                preload_start = max(0, frame_number - 10)
                preload_end = min(total_frames, frame_number + 10)
                for i in range(preload_start, preload_end):
                    if i not in preload_queue:
                        video.set(cv2.CAP_PROP_POS_FRAMES, i)
                        ret, frame = video.read()
                        if ret:
                            preload_queue[i] = frame
            threading.Event().wait(0.1)  # Small pause to avoid CPU overload

    preload_thread = threading.Thread(target=preload_frames, daemon=True)
    preload_thread.start()


    # Variables for handling key presses
    right_delay = 0.1   # Delay for step-by-step navigation
    left_delay=0.1     # Delay for step-by-step navigation
    last_right_left_time = 0  # Time of the last key press
    
    # Variables to manage input timing
    last_up_down_time = 0  # Time of the last up/down key press
    up_down_delay = 0.3  # Delay to avoid repeated actions for up/down keys

    while True:
        # Check if the current frame is preloaded
        with preload_lock:
            current_frame = preload_queue.get(frame_number, None)

        if current_frame is None:
            print(f"Frame {frame_number} is loading...")
            time.sleep(0.05)  # Short pause to allow preloading
            continue

        # Calculate the time corresponding to the current frame
        fps = video.get(cv2.CAP_PROP_FPS)
        time_in_seconds = frame_number / fps

        # Apply zoom and pan to the current frame
        zoomed_current = apply_zoom_and_pan(
            current_frame, 
            screen_height // 2, 
            frame_number=frame_number, 
            time_in_seconds=time_in_seconds
        )

        # Display navigation instructions on the comparison frame
        instructions = [
            "Controls:",
            "- Left/Right: Frame -/+",
            "- Up/Down: Jump +/- 100 frames",
            "- [v]: Validate offset",
            "- [f]: Enter a frame number",
            "- [q]: Quit"
        ]
        for i, line in enumerate(instructions):
            cv2.putText(zoomed_current, line, (10, screen_height // 2 - 150 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(compare_window, zoomed_current)  # Show the comparison frame

        key = cv2.waitKey(1) & 0xFF
        current_time = time.time()

        # Handle user inputs for navigation and synchronization
        if keyboard.is_pressed('q'):  # Quit the synchronization process
            print("Program exited by user.")
            offset = frame_number - reference_frame_number
            break
        elif keyboard.is_pressed('v'):  # Validate the current offset
            offset = frame_number - reference_frame_number
            print(f"{video_name}: Offset of {offset} frames from the reference")
            break
        elif keyboard.is_pressed('f'):  # Enter a specific frame number
            try:
                input_frame = int(input("Enter frame number: "))
                frame_number = max(0, min(input_frame, total_frames - 1))  # Clamp to valid range
                print(f"Navigated to frame {frame_number}.")
            except ValueError:
                print("Invalid frame number. Try again.")
        elif keyboard.is_pressed('up'):  # Jump forward by 100 frames
            if current_time - last_up_down_time > up_down_delay:
                frame_number = min(total_frames - 1, frame_number + 100)
                last_up_down_time = current_time
        elif keyboard.is_pressed('down'):  # Jump backward by 100 frames
            if current_time - last_up_down_time > up_down_delay:
                frame_number = max(0, frame_number - 100)
                last_up_down_time = current_time
        elif keyboard.is_pressed('left'):  # Step back one frame
            if current_time - last_right_left_time > left_delay:
                frame_number = max(0, frame_number - 1)
                last_right_left_time = current_time
        elif keyboard.is_pressed('right'):  # Step forward one frame
            if current_time - last_right_left_time > right_delay:
                frame_number = min(total_frames - 1, frame_number + 1)
                last_right_left_time = current_time

        # Handle zoom controls
        if key == ord('+') or key == ord('='):
            zoom_in(screen_width // 4, screen_height // 4)
        elif key == ord('-'):
            zoom_out(screen_width // 4, screen_height // 4)

    # Stop preloading frames
    stop_preloading.set()
    preload_thread.join()

    # Close all windows
    cv2.destroyWindow(reference_window)
    cv2.destroyWindow(compare_window)
    return offset

# Apply zoom and pan transformations to a frame without distorting edges
def apply_zoom_and_pan(frame, target_height, frame_number=None, time_in_seconds=None):
    """
    Applies zoom and pan transformations to a video frame, ensuring the edges are not distorted
    and the frame fits within the target display height.

    Args:
        frame: The input video frame to process.
        target_height: The height of the display window for resizing.
        frame_number: Optional; the current frame number (for overlaying text).
        time_in_seconds: Optional; the current timestamp of the frame in seconds (for overlaying text).

    Returns:
        displayed_frame: The processed frame after zoom and pan adjustments.
    """
    global zoom_scale, pan_x, pan_y

    # Resize the frame based on the zoom scale
    height, width = frame.shape[:2]
    zoomed_width, zoomed_height = int(width * zoom_scale), int(height * zoom_scale)
    resized_frame = cv2.resize(frame, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)

    # Calculate cropping boundaries to ensure panning stays within valid limits
    x_start = max(0, min(pan_x, zoomed_width - width))
    y_start = max(0, min(pan_y, zoomed_height - height))
    x_end = x_start + width
    y_end = y_start + height
    displayed_frame = resized_frame[y_start:y_end, x_start:x_end]

    # Scale the frame to fit within the target height while maintaining aspect ratio
    scaling_factor = target_height / displayed_frame.shape[0]
    final_width = int(displayed_frame.shape[1] * scaling_factor)
    displayed_frame = cv2.resize(displayed_frame, (final_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Optional: Overlay frame number and timestamp (commented out for customization)
    # if frame_number is not None and time_in_seconds is not None:
    #     cv2.putText(displayed_frame, f"Frame: {frame_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    #     cv2.putText(displayed_frame, f"Time: {time_in_seconds:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    return displayed_frame

# Prompt the user to decide if custom start and end frames should be selected
def ask_for_custom_cut():
    """
    Asks the user if they want to define custom start and end frames for video synchronization.

    Returns:
        bool: True if the user chooses to define custom frames, False otherwise.
    """
    response = input("Voulez-vous définir un nouveau début et une nouvelle fin ? (yes/no) : ")
    return response.lower() == "yes"

# Allow the user to select custom start and end frames on a single video
def select_start_end_frames(video):
    """
    Enables the user to manually select the start and end frames for a video.

    Args:
        video: cv2.VideoCapture object representing the video to process.

    Returns:
        tuple: A pair of integers representing the start and end frame numbers.
    """
    print("Sélection de la frame de début.")
    _, start_frame = navigate_frames(video, "Definir Frame Debut")  # Start frame selection

    # Move to the last frame of the video for end frame selection
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    ret, last_frame = video.read()

    if ret:
        print("Sélection de la frame de fin.")
        _, end_frame = navigate_frames(video, "Definir Frame Fin", start_frame=total_frames - 1)  # End frame selection

    return start_frame, end_frame


# Cut a video based on an offset and minimum duration, with a progress bar
def cut_video(video_path, output_path, offset, min_frames):
    """
    Cuts a video starting from a specific frame offset and saves it with a defined duration.

    Args:
        video_path (str): Path to the input video.
        output_path (str): Path where the cut video will be saved.
        offset (int): Number of frames to skip from the start of the video.
        min_frames (int): Total number of frames to include in the output.

    Returns:
        int: The number of frames successfully written to the output video.
    """
    # Open the input video
    video = open_video(video_path)
    if video is None:
        return 0

    # Retrieve video properties
    fps = video.get(cv2.CAP_PROP_FPS)  # Frames per second of the video
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # Frame width
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Frame height

    # Define the video writer to save the output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Set the starting frame based on the offset
    start_frame = max(0, offset)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Initialize the frame counter and progress bar
    frame_count = 0
    with tqdm(total=min_frames, desc=f"Saving {output_path}", unit="frame") as pbar:
        for i in range(min_frames):  # Loop through the minimum required frames
            ret, frame = video.read()  # Read the next frame
            if not ret:
                break  # Exit loop if no more frames are available
            out.write(frame)  # Write the frame to the output video
            frame_count += 1  # Increment the frame counter
            pbar.update(1)  # Update the progress bar
    
    # Release video resources
    video.release()
    out.release()

    return frame_count  # Return the number of frames successfully written

"""
==========================================================================
_____________________________ Main Function _____________________________
==========================================================================

"""

def synchronize_cams_all(level, config_dict):
    '''
    Post-synchronize your cameras in case they are not natively synchronized.

    Post-synchronize cameras using three possible methods:
    1. **Move-based synchronization**: Analyzes keypoints' vertical speeds and finds offsets by maximizing correlation.
    2. **Sound-based synchronization**: Uses audio features to align video streams.
    3. **Manual synchronization**: Requires user input for determining offsets.
    
    Each method outputs synchronized `.json` files or `.mp4` videos as per the chosen type.
    
    INPUTS: 
    - Level (1 for single trial, 2 for multiple trials).
    - `config_dict` with project details and synchronization parameters.
    
    OUTPUTS:
    - Synchronized `.json` files or videos in respective directories.

"""
    '''
    # Retrieve project-related parameters from the configuration file (Config.toml)
    project_dir = config_dict.get('project').get('project_dir')  # Path to the project directory
    pose_dir = os.path.realpath(os.path.join(project_dir, 'pose'))  # Path to the directory containing pose data
    pose_model = config_dict.get('pose').get('pose_model')  # The pose model to use for keypoints extraction
    multi_person = config_dict.get('project').get('multi_person')  # Boolean indicating if multiple people are in the scene
    fps = config_dict.get('project').get('frame_rate')  # Frame rate of the videos, can be 'auto'
    frame_range = config_dict.get('project').get('frame_range')  # Range of frames to analyze
    synchronization_type = config_dict.get('synchronization').get('synchronization_type')  # Type of synchronization (e.g., 'move', 'sound', etc.)
    display_sync_plots = config_dict.get('synchronization').get('display_sync_plots')  # Whether to display synchronization plots
    keypoints_to_consider = config_dict.get('synchronization').get('keypoints_to_consider')  # Specific keypoints to focus on for synchronization
    approx_time_maxspeed = config_dict.get('synchronization').get('approx_time_maxspeed')  # Approximate time for maximum vertical speed
    time_range_around_maxspeed = config_dict.get('synchronization').get('time_range_around_maxspeed')  # Time range around max speed for analysis

    # Filtering parameters for keypoints data
    likelihood_threshold = config_dict.get('synchronization').get('likelihood_threshold')  # Threshold for keypoint likelihood
    filter_cutoff = int(config_dict.get('synchronization').get('filter_cutoff'))  # Cutoff frequency for the filter
    filter_order = int(config_dict.get('synchronization').get('filter_order'))  # Order of the Butterworth filter

    # Determine the video frame rate
    video_dir = os.path.join(project_dir, 'videos_raw')  # Path to the raw video directory
    vid_img_extension = config_dict['pose']['vid_img_extension']  # Extension of video/image files
    video_files = glob.glob(os.path.join(video_dir, '*' + vid_img_extension))  # List of all video/image files

    if fps == 'auto': 
        try:
            # Open the first video file to detect its FPS
            cap = cv2.VideoCapture(video_files[0])
            cap.read()  # Read the first frame to ensure the video is valid
            if cap.read()[0] == False:
                raise ValueError("Unable to read the video.")
            fps = int(cap.get(cv2.CAP_PROP_FPS))  # Retrieve FPS from the video metadata
        except:
            fps = 60  # Default FPS value if auto-detection fails
    lag_range = time_range_around_maxspeed * fps  # Convert the time range around max speed to frames

    if synchronization_type == 'move':
        # Display a warning if the project involves multiple people
        if multi_person:
            logging.warning('\nYou set your project as a multi-person one: make sure you set `approx_time_maxspeed` and `time_range_around_maxspeed` at times where one single person is in the scene, or you may get inaccurate results.')
            do_synchro = input('Do you want to continue? (y/n)')  # Confirm from the user whether to proceed
            if do_synchro.lower() not in ["y", "yes"]:
                logging.warning('Synchronization cancelled.')  # Abort if user declines
                return
            else:
                logging.warning('Synchronization will be attempted.\n')

        # Retrieve the skeleton model for keypoints extraction
        try:  # Attempt to import the model from `skeletons.py`
            model = eval(pose_model)
        except:
            try:  # Attempt to import the model from the Config.toml
                model = DictImporter().import_(config_dict.get('pose').get(pose_model))
                if model.id == 'None':
                    model.id = None
            except:
                raise NameError('Model not found in skeletons.py nor in Config.toml')

        # Extract keypoint IDs and names from the skeleton model
        keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id != None]
        keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id != None]

        # Retrieve a list of JSON directories for each camera
        try:
            pose_listdirs_names = next(os.walk(pose_dir))[1]  # List all subdirectories in the pose directory
            os.listdir(os.path.join(pose_dir, pose_listdirs_names[0]))[0]  # Check if subdirectories contain JSON files
        except:
            raise ValueError(f'No JSON files found in {pose_dir} subdirectories. Make sure you run Pose2Sim.poseEstimation() first.')

        # Sort directories by the numerical suffix in their names
        pose_listdirs_names = sort_stringlist_by_last_number(pose_listdirs_names)
        json_dirs_names = [k for k in pose_listdirs_names if 'json' in k]  # Filter directories containing "json"
        json_dirs = [os.path.join(pose_dir, j_d) for j_d in json_dirs_names]  # Full paths of JSON directories
        json_files_names = [fnmatch.filter(os.listdir(os.path.join(pose_dir, js_dir)), '*.json') for js_dir in json_dirs_names]  # List JSON files in each directory
        json_files_names = [sort_stringlist_by_last_number(j) for j in json_files_names]  # Sort JSON files numerically
        nb_frames_per_cam = [len(fnmatch.filter(os.listdir(os.path.join(json_dir)), '*.json')) for json_dir in json_dirs]  # Count JSON files per camera
        cam_nb = len(json_dirs)  # Number of cameras
        cam_list = list(range(cam_nb))  # List of camera indices

        # Define the frame range for synchronization
        f_range = [[0, min([len(j) for j in json_files_names])] if frame_range == [] else frame_range][0]


        # Determine frames to consider for synchronization
        if isinstance(approx_time_maxspeed, list):  # If specific times are provided for max vertical speed
            approx_frame_maxspeed = [int(fps * t) for t in approx_time_maxspeed]  # Convert times to frame indices
            nb_frames_per_cam = [len(fnmatch.filter(os.listdir(os.path.join(json_dir)), '*.json')) for json_dir in json_dirs]  # Count frames for each camera
            # Define frame ranges around the specified max speed times
            search_around_frames = [
                [int(a - lag_range) if a - lag_range > 0 else 0,  # Start frame (ensuring it is not negative)
                 int(a + lag_range) if a + lag_range < nb_frames_per_cam[i] else nb_frames_per_cam[i] + f_range[0]]  # End frame (ensuring it doesn't exceed available frames)
                for i, a in enumerate(approx_frame_maxspeed)
            ]
            logging.info(f'Synchronization is calculated around the times {approx_time_maxspeed} +/- {time_range_around_maxspeed} s.')
        elif approx_time_maxspeed == 'auto':  # If no specific times, analyze the entire sequence
            # Use the full range of available frames for each camera
            search_around_frames = [[f_range[0], f_range[0] + nb_frames_per_cam[i]] for i in range(cam_nb)]
            logging.info('Synchronization is calculated on the whole sequence. This may take a while.')
        else:
            raise ValueError('approx_time_maxspeed should be a list of floats or "auto"')

        # Log information about the selected keypoints for synchronization
        if keypoints_to_consider == 'right':
            logging.info(f'Keypoints used to compute the best synchronization offset: right side.')
        elif keypoints_to_consider == 'left':
            logging.info(f'Keypoints used to compute the best synchronization offset: left side.')
        elif isinstance(keypoints_to_consider, list):
            logging.info(f'Keypoints used to compute the best synchronization offset: {keypoints_to_consider}.')
        elif keypoints_to_consider == 'all':
            logging.info(f'All keypoints are used to compute the best synchronization offset.')
        logging.info(f'These keypoints are filtered with a Butterworth filter (cut-off frequency: {filter_cutoff} Hz, order: {filter_order}).')
        logging.info(f'They are removed when their likelihood is below {likelihood_threshold}.\n')

        # Extract, interpolate, and filter keypoint coordinates
        logging.info('Synchronizing...')
        df_coords = []  # List to hold dataframes of keypoint coordinates for each camera
        # Define the Butterworth filter coefficients
        b, a = signal.butter(filter_order / 2, filter_cutoff / (fps / 2), 'low', analog=False)

        # Identify JSON files within the specified frame ranges for each camera
        json_files_names_range = [
            [j for j in json_files_cam if int(re.split(r'(\d+)', j)[-2]) in range(*frames_cam)]
            for (json_files_cam, frames_cam) in zip(json_files_names, search_around_frames)
        ]
        json_files_range = [
            [os.path.join(pose_dir, j_dir, j_file) for j_file in json_files_names_range[j]]
            for j, j_dir in enumerate(json_dirs_names)
        ]

        # Validate that there are JSON files within the specified frame ranges
        if np.array([j == [] for j in json_files_names_range]).any():
            raise ValueError(f'No JSON files found within the specified frame range ({frame_range}) at the times {approx_time_maxspeed} +/- {time_range_around_maxspeed} s.')

        # Process keypoint data for each camera
        for i in range(cam_nb):
            # Convert JSON files to pandas DataFrame
            df_coords.append(convert_json2pandas(json_files_range[i], likelihood_threshold=likelihood_threshold))
            df_coords[i] = drop_col(df_coords[i], 3)  # Drop the likelihood column

            # Select keypoints based on the specified criteria
            if keypoints_to_consider == 'right':
                kpt_indices = [i for i, k in zip(keypoints_ids, keypoints_names) if k.startswith('R') or k.startswith('right')]
                kpt_indices = np.sort(np.concatenate([np.array(kpt_indices) * 2, np.array(kpt_indices) * 2 + 1]))
                df_coords[i] = df_coords[i][kpt_indices]
            elif keypoints_to_consider == 'left':
                kpt_indices = [i for i, k in zip(keypoints_ids, keypoints_names) if k.startswith('L') or k.startswith('left')]
                kpt_indices = np.sort(np.concatenate([np.array(kpt_indices) * 2, np.array(kpt_indices) * 2 + 1]))
                df_coords[i] = df_coords[i][kpt_indices]
            elif isinstance(keypoints_to_consider, list):
                kpt_indices = [i for i, k in zip(keypoints_ids, keypoints_names) if k in keypoints_to_consider]
                kpt_indices = np.sort(np.concatenate([np.array(kpt_indices) * 2, np.array(kpt_indices) * 2 + 1]))
                df_coords[i] = df_coords[i][kpt_indices]
            elif keypoints_to_consider == 'all':
                pass
            else:
                raise ValueError(
                    'keypoints_to_consider should be "all", "right", "left", or a list of keypoint names.\n'
                    'If you specified keypoints, make sure that they exist in your pose_model.'
                )

            # Interpolate missing values and apply the Butterworth filter
            df_coords[i] = df_coords[i].apply(interpolate_zeros_nans, axis=0, args=['linear'])  # Linear interpolation for NaNs
            df_coords[i] = df_coords[i].bfill().ffill()  # Backfill and forward-fill any remaining gaps
            df_coords[i] = pd.DataFrame(signal.filtfilt(b, a, df_coords[i], axis=0))  # Apply low-pass filter

        # Compute sum of speeds
        df_speed = []  # List to store vertical speeds for each camera
        sum_speeds = []  # List to store the sum of absolute speeds for each camera
        for i in range(cam_nb):
            # Calculate the vertical speed for the keypoints of the current camera
            df_speed.append(vert_speed(df_coords[i]))
            # Compute the sum of absolute speeds for all keypoints
            sum_speeds.append(abs(df_speed[i]).sum(axis=1))

            # Optional: Uncomment the following lines to set a maximum speed threshold
            # nb_coord = df_speed[i].shape[1]  # Number of coordinates
            # sum_speeds[i][sum_speeds[i] > vmax * nb_coord] = 0  # Zero out values above the threshold

            # Optional: Replace zeros with random values to avoid issues with padding during correlation
            # sum_speeds[i].loc[sum_speeds[i] < 1] = sum_speeds[i].loc[sum_speeds[i] < 1].apply(lambda x: np.random.normal(0, 1))

            # Apply a low-pass Butterworth filter to smooth the speed signals
            sum_speeds[i] = pd.DataFrame(signal.filtfilt(b, a, sum_speeds[i], axis=0)).squeeze()

        # Compute offset for best synchronization:
        # The goal is to find the highest correlation of summed absolute speeds for each camera
        # compared to the reference camera.
        
        # Select the reference camera as the one with the least number of frames
        ref_cam_id = nb_frames_per_cam.index(min(nb_frames_per_cam))
        ref_frame_nb = len(df_coords[ref_cam_id])  # Total number of frames in the reference camera
        lag_range = int(ref_frame_nb / 2)  # Define the maximum range for correlation lag
        cam_list.pop(ref_cam_id)  # Remove the reference camera from the list of cameras to synchronize
        offset = []  # List to store the computed offsets for each camera

        for cam_id in cam_list:
            # Calculate the time-lagged cross-correlation between the reference and current camera
            offset_cam_section, max_corr_cam = time_lagged_cross_corr(
                sum_speeds[ref_cam_id], sum_speeds[cam_id], lag_range,
                show=display_sync_plots, ref_cam_id=ref_cam_id, cam_id=cam_id
            )
            # Adjust the offset based on the search range and frame alignment
            offset_cam = offset_cam_section - (search_around_frames[ref_cam_id][0] - search_around_frames[cam_id][0])

            # Log the results with additional details if a specific time range is used
            if isinstance(approx_time_maxspeed, list):
                logging.info(f'--> Camera {ref_cam_id} and {cam_id}: {offset_cam} frames offset '
                             f'({offset_cam_section} on the selected section), correlation {round(max_corr_cam, 2)}.')
            else:
                logging.info(f'--> Camera {ref_cam_id} and {cam_id}: {offset_cam} frames offset, '
                             f'correlation {round(max_corr_cam, 2)}.')
            offset.append(offset_cam)  # Store the computed offset for the current camera

        # Insert the reference camera offset (always 0) back into the offset list
        offset.insert(ref_cam_id, 0)

        # Rename JSON files according to the computed offsets and copy them to the synchronized folder
        sync_dir = os.path.abspath(os.path.join(pose_dir, '..', 'pose-sync'))
        os.makedirs(sync_dir, exist_ok=True)  # Ensure the output directory exists
        for d, j_dir in enumerate(json_dirs):
            # Create a corresponding directory in the synchronized folder
            os.makedirs(os.path.join(sync_dir, os.path.basename(j_dir)), exist_ok=True)
            for j_file in json_files_names[d]:
                # Adjust the frame index in the file name based on the offset
                j_split = re.split(r'(\d+)', j_file)
                j_split[-2] = f'{int(j_split[-2]) - offset[d]:06d}'
                if int(j_split[-2]) > 0:  # Only copy files with valid frame indices
                    json_offset_name = ''.join(j_split)
                    shutil.copy(os.path.join(pose_dir, os.path.basename(j_dir), j_file),
                                os.path.join(sync_dir, os.path.basename(j_dir), json_offset_name))

        # Uncomment the following section if synchronized video mosaics should be displayed
        # if display_sync_plots == True:
        #     sync_video_folder_path = os.path.join(project_dir, "videos")
        #     synchroMosaique(sync_video_folder_path)
            
        logging.info(f'Synchronized JSON files saved in {sync_dir}.')

    elif synchronization_type == 'sound':
        # Logging the start of the sound-based synchronization process
        logging.info("====================")
        print("Videos Synchronization...")
        logging.info("--------------------\n\n")  

        if level == 1:
            # Retrieve paths for the current trial
            path_folder = os.path.dirname(project_dir)
            
            # Define paths for raw and synchronized video directories
            raw_video_folder_path = Path(os.path.join(project_dir, 'videos_raw'))
            sync_video_folder_path = Path(os.path.join(project_dir, 'videos'))

            # If the synchronized video folder is empty or doesn't exist
            if not os.path.exists(sync_video_folder_path) or os.listdir(sync_video_folder_path) == []:
                
                # Create the synchronized video folder if it doesn't exist
                if not os.path.exists(os.path.join(path_folder, "videos")):
                    os.mkdir(os.path.join(path_folder, "videos"))
                
                # Perform synchronization using audio signals
                sync.synchronize_videos_from_audio(
                    raw_video_folder_path=raw_video_folder_path,
                    synchronized_video_folder_path=sync_video_folder_path,
                    video_handler="deffcode",  # Define the handler for video processing
                    create_debug_plots_bool=display_sync_plots  # Enable or disable debug plots
                )
                # If plots are enabled, create a mosaic of synchronized videos
                if display_sync_plots == True:
                    synchroMosaique(sync_video_folder_path)
        
        if level == 2:
            # Retrieve the general working path
            path_folder = os.path.dirname(project_dir)

            # List all folders in the general path
            folders = os.listdir(path_folder)
            
            # Count the number of trials and get their names
            nbtrials = 0
            trialname = []
            for i in range(len(folders)):
                if "Trial" in folders[i]:
                    nbtrials += 1
                    trialname.append(folders[i])
    
            # Iterate through each trial
            for trial in range(nbtrials):
                
                # Retrieve paths for the current trial
                raw_video_folder_path = Path(os.path.join(path_folder, trialname[trial], "videos_raw"))
                sync_video_folder_path = Path(os.path.join(path_folder, trialname[trial], "videos"))

                # If the synchronized video folder is empty or doesn't exist
                if not os.path.exists(sync_video_folder_path) or os.listdir(sync_video_folder_path) == []:
                    
                    # Create the synchronized video folder if it doesn't exist
                    if not os.path.exists(os.path.join(path_folder, "Trial_" + str(trial + 1), "videos")):
                        os.mkdir(os.path.join(path_folder, "Trial_" + str(trial + 1), "videos"))
                    
                    # Perform synchronization using audio signals
                    sync.synchronize_videos_from_audio(
                        raw_video_folder_path=raw_video_folder_path,
                        synchronized_video_folder_path=sync_video_folder_path,
                        video_handler="deffcode",  # Define the handler for video processing
                        create_debug_plots_bool=display_sync_plots  # Enable or disable debug plots
                    )
                    # If plots are enabled, create a mosaic of synchronized videos
                    if display_sync_plots == True:
                        synchroMosaique(sync_video_folder_path)
        
        # Log a message indicating successful synchronization
        logging.info("Videos successfully synchronized.")

    elif synchronization_type == 'manual':
        # Logging the start of manual synchronization
        logging.info("====================")
        print("Videos Synchronization...")
        logging.info("--------------------\n\n")  

        if level == 1:
            # Retrieve the paths for the current trial
            path_folder = os.path.dirname(project_dir)
            
            # Define paths for raw and synchronized video directories
            raw_video_folder_path = Path(os.path.join(project_dir, 'videos_raw'))
            sync_video_folder_path = Path(os.path.join(project_dir, 'videos'))
            
            # List all MP4 video files in the raw video folder, sorted alphabetically
            video_files = sorted([os.path.join(raw_video_folder_path, f) for f in os.listdir(raw_video_folder_path) if f.lower().endswith(".mp4")])

            # If the synchronized video folder is empty or doesn't exist
            if not os.path.exists(sync_video_folder_path) or os.listdir(sync_video_folder_path) == []:
                
                # Create the synchronized video folder if it doesn't exist
                if not os.path.exists(os.path.join(path_folder, "videos")):
                    os.mkdir(os.path.join(path_folder, "videos"))
                
                # Synchronize the videos manually by calling the `synchronize_videos` function
                synchronize_videos(video_files, sync_video_folder_path)
                # Close any open CV2 windows to clean up
                cv2.destroyAllWindows()
                
                # If display plots are enabled, create a video mosaic for visual confirmation
                if display_sync_plots == True:
                    synchroMosaique(sync_video_folder_path)

        if level == 2:
            # Retrieve the general working path
            path_folder = os.path.dirname(project_dir)

            # List all folders in the working path
            folders = os.listdir(path_folder)
            
            # Identify the number of trials and their names by checking folder names
            nbtrials = 0
            trialname = []
            for i in range(len(folders)):
                if "Trial" in folders[i]:
                    nbtrials += 1
                    trialname.append(folders[i])

            # Process each trial for synchronization
            for trial in range(nbtrials):
                
                # Define the paths for raw and synchronized videos for the current trial
                raw_video_folder_path = Path(os.path.join(path_folder, trialname[trial], "videos_raw"))
                sync_video_folder_path = Path(os.path.join(path_folder, trialname[trial], "videos"))
                
                # List all MP4 video files in the raw video folder, sorted alphabetically
                video_files = sorted([os.path.join(raw_video_folder_path, f) for f in os.listdir(raw_video_folder_path) if f.endswith(".MP4")])

                # If the synchronized video folder is empty or doesn't exist
                if not os.path.exists(sync_video_folder_path) or os.listdir(sync_video_folder_path) == []:
                    
                    # Create the synchronized video folder if it doesn't exist
                    if not os.path.exists(os.path.join(path_folder, "Trial_" + str(trial + 1), "videos")):
                        os.mkdir(os.path.join(path_folder, "Trial_" + str(trial + 1), "videos"))
                    
                    # Synchronize the videos manually by calling the `synchronize_videos` function
                    synchronize_videos(video_files, sync_video_folder_path)
                    # Close any open CV2 windows to clean up
                    cv2.destroyAllWindows()
                    
                    # If display plots are enabled, create a video mosaic for visual confirmation
                    if display_sync_plots == True:
                        synchroMosaique(sync_video_folder_path)
        
        # Log a success message indicating the videos have been synchronized
        logging.info("Videos successfully synchronized.")

    else: 
        # Log a message if videos are already synchronized
        logging.info("Videos already synchronized.")
    
    # Log the completion of the synchronization process
    logging.info("\n\n--------------------")
    logging.info("Video synchronization completed successfully.")
    logging.info("====================")
