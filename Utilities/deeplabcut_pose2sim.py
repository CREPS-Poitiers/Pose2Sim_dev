"""
###########################################################################
## SCRIPT FOR DEEPLABCUT INTEGRATION IN POSE ESTIMATION WORKFLOW        ##
###########################################################################

This script automates the workflow for integrating **DeepLabCut** into a 
pose estimation pipeline, from analyzing videos to organizing results. 
It is designed to simplify the process of extracting pose data from videos 
and converting it into a format compatible with **Pose2Sim**.

### FEATURES ###
1. **Dynamic Configuration:** 
   - Configures paths and parameters via command-line arguments for flexibility.
   - Allows customization for shuffle numbers, output folders, and detection visualization.

2. **Video Analysis:** 
   - Analyzes videos using DeepLabCut and optionally generates labeled videos for visualization.
   - Identifies videos in the `videos` folder and processes them automatically.

3. **Data Conversion:**
   - Converts DeepLabCut `.h5` output files into JSON format, organized per video.
   - Ensures compatibility with Pose2Sim by using a dedicated converter (`DLC_to_OpenPose`).

4. **Post-Processing:**
   - Cleans up unnecessary files, including `.pickle` files.
   - Renames and organizes `.h5` files for clarity and consistency.
   - Moves labeled videos into the specified output folder for better management.

5. **Error Handling:**
   - Verifies the existence of required folders (`videos`) and files.
   - Provides descriptive error messages for easier debugging.

### WORKFLOW ###
1. Parse command-line arguments to configure the script dynamically.
2. Locate videos in the `videos` folder for analysis.
3. Use DeepLabCut to analyze videos and generate pose data (`.h5` files).
4. Convert `.h5` files to JSON format, organizing results in dedicated folders.
5. Cleanup intermediate files (e.g., `.pickle`) and move labeled videos to the output folder.

### USAGE ###
Run this script with the following required and optional arguments:
- `--deeplabcut_env_path`: Path to the DeepLabCut environment.
- `--config_DLC_project_path`: Path to the DeepLabCut project configuration file.
- `--shuffle_number`: (Optional) Shuffle number used during training (default: 1).
- `--output_folder`: (Optional) Output folder for JSON results (default: "pose-custom").
- `--display_detection`: (Optional) Flag to enable labeled video generation for visualization.

### OUTPUTS ###
- JSON files organized into folders per video.
- Optionally labeled videos saved in the output folder.
- Cleaned workspace with unnecessary intermediate files removed.

This script provides a streamlined approach for integrating DeepLabCut 
into a broader pose estimation pipeline, ensuring efficiency and organization.

Authors:
- F.Delaplace for DLC integration in the poseEstimation pipeline
"""




import sys
import os
import glob
import argparse


# Parse command-line arguments to configure the script dynamically.
parser = argparse.ArgumentParser(description="Run DeepLabCut for pose estimation.")
parser.add_argument('--deeplabcut_env_path', type=str, required=True, help="Path to the DeepLabCut environment.")
parser.add_argument('--config_DLC_project_path', type=str, required=True, help="Path to the DeepLabCut project configuration file.")
parser.add_argument('--shuffle_number', type=int, default=1, help="Shuffle number used during DeepLabCut training.")
parser.add_argument('--output_folder', type=str, default="pose-custom", help="Folder to store the generated JSON files.")
parser.add_argument('--display_detection', action='store_true', help="Display detection results during processing.")
args = parser.parse_args()


# Add the DeepLabCut environment path to the system's PATH to ensure the module can be imported.
if args.deeplabcut_env_path not in sys.path:
    sys.path.append(args.deeplabcut_env_path)

# Attempt to import DeepLabCut; handle errors gracefully if it fails.
try:
    import deeplabcut
    print("DeepLabCut successfully imported.")
except ImportError as e:
    print(f"Import error: {e}")
    print("Ensure the path is correct and DeepLabCut is installed.")

# Importing a custom module to convert DeepLabCut output to JSON compatible with Pose2Sim.
import DLC_to_OpenPose

# Normalize and clean paths provided through the command-line arguments.
config_DLC_project_path = os.path.normpath(args.config_DLC_project_path).strip("'\"")
print(f"Cleaned path: {config_DLC_project_path}")

# Assign arguments to variables for easier access throughout the script.
shuffle_number = args.shuffle_number
output_folder = args.output_folder.strip("'\"")
display_detection = args.display_detection


# Main function to handle video analysis and post-processing.
def analyze_and_convert_videos():
    # Step 1: Locate the "videos" folder in the current working directory.
    # Raise an error if the folder is missing.
    current_dir = os.getcwd()
    video_folder = os.path.join(current_dir, "videos")
    if not os.path.exists(video_folder):
        raise FileNotFoundError(f"The 'videos' folder is missing in {current_dir}.")

    # Step 2: Search for MP4 files in the "videos" folder.
    # Raise an error if no video files are found.
    video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
    if not video_files:
        raise FileNotFoundError("No video files found in the 'videos' folder.")

    # Step 3: Analyze videos using DeepLabCut.
    print("Analyzing videos with DeepLabCut...")
    deeplabcut.analyze_videos(config_DLC_project_path, video_files, shuffle=shuffle_number, save_as_csv=False, gputouse=0)
    
    # Optionally display detection results by generating labeled videos.
    if display_detection:
        print("Displaying detection results...")
        deeplabcut.create_labeled_video(config_DLC_project_path, video_files, shuffle=shuffle_number)
        
    # Step 4: Filter and rename `.h5` files for JSON conversion.
    print("Filtering and renaming `.h5` files...")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    h5_files = glob.glob(os.path.join(video_folder, f"*shuffle{shuffle_number}_snapshot_*.h5"))
    for h5_file in h5_files:
        # Rename files by removing unnecessary suffixes for consistency.
        base_name = os.path.basename(h5_file)
        new_name = base_name.split("DLC")[0].rstrip("_") + ".h5"
        new_path = os.path.join(output_folder, new_name)
        os.rename(h5_file, new_path)

        # Step 5: Convert `.h5` files to JSON format for each video.
        video_name = os.path.splitext(new_name)[0]
        json_output_folder = os.path.join(output_folder, f"{video_name}_json")
        if not os.path.exists(json_output_folder):
            os.makedirs(json_output_folder)

        print(f"Converting {new_name} to JSON...")
        DLC_to_OpenPose.DLC_to_OpenPose_func(new_path, json_output_folder)

    # Step 6: Clean up `.pickle` files generated during processing.
    print("Deleting `.pickle` files...")
    pickle_files = glob.glob(os.path.join(video_folder, "*.pickle"))
    for pickle_file in pickle_files:
        os.remove(pickle_file)
        
    # Step 7: Move labeled videos to the output folder.
    print("Moving labeled videos to the output folder...")
    labeled_videos = glob.glob(os.path.join(video_folder, "*labeled*.mp4"))
    for video in labeled_videos:
        file_name = os.path.basename(video)
        destination = os.path.join(current_dir, output_folder, file_name)
        os.rename(video, destination)
        print(f"Moved: {video} -> {destination}")    

    print("Task completed. JSON files are in their respective folders, and `.pickle` files have been deleted.")

# Execute the main function when the script is run directly.
if __name__ == "__main__":
    analyze_and_convert_videos()
