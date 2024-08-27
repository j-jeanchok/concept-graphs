# ===== Generate Dataset with RealSense Camera ===== # 


# ===== Third-party Imports ===== #
import cv2
import numpy as np
from tqdm import trange
import hydra
from omegaconf import DictConfig
import torch

# ===== Local application/library scpecific imports ===== #
from conceptgraph.realsense.realsense import RealSenseApp 
from conceptgraph.utils.general_utils import (
    get_exp_out_path, 
    get_stream_data_out_path, 
    measure_time, 
    save_hydra_config, 
    should_exit_early
)

@hydra.main(
    version_base=None, 
    config_path="../hydra_configs/", 
    config_name="streamlined_detections")

def main(cfg : DictConfig):

    app = RealSenseApp() 

    # Output folder of the detections experiment to use
    det_exp_path = get_exp_out_path(
        dataset_root=cfg.dataset_root, 
        scene_id=cfg.scene_id, 
        exp_suffix=cfg.exp_suffix)

    stream_rgb_path, stream_depth_path, stream_poses_path = get_stream_data_out_path(
        dataset_root=cfg.dataset_root, 
        scene_id=cfg.scene_id,
        make_dir=True)
    
    save_hydra_config(cfg, det_exp_path)

    ## Looping for frames 
    exit_early_flag = False
    # total_frames = 300
    frame_idx = 0

    # for frame_idx in trange(total_frames):
    while True:
        torch.cuda.empty_cache()

        # Check if we should exit early only if the flag hasn't been set yet
        if not exit_early_flag and should_exit_early(cfg.exit_early_file):
            print("Exit early signal detected. Skipping to the final frame...")
            exit_early_flag = True
            break

        # # If exit early flag is set and we're not at the last frame, skip this iteration
        # if exit_early_flag and frame_idx < total_frames - 1:
        #     break

        # Get the frame data
        s_rgb, s_depth, _, s_camera_pose = app.get_frame_data()
        # image_rgb = cv2.cvtColor(s_rgb, cv2.COLOR_BGR2RGB)
        # Show images 
        cv2.namedWindow("RealSense", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("RealSense", np.asanyarray(s_rgb)) 
        cv2.waitKey(1)

        # Save the rgb to the stream folder with an appropriate name
        color_path = stream_rgb_path / f"{frame_idx}.jpg"
        cv2.imwrite(str(color_path), s_rgb)

        # Save depth to the stream folder with an appropriate name
        curr_stream_depth_path = stream_depth_path / f"{frame_idx}.png"
        cv2.imwrite(str(curr_stream_depth_path), s_depth)
        
        # Save the camera pose to the stream folder with an appropriate name 
        curr_stream_pose_path = stream_poses_path / f"{frame_idx}"
        np.save(str(curr_stream_pose_path), s_camera_pose)


        frame_idx += 1

        ## The second condition for exit 
        if frame_idx == 300:
            break 
        

if __name__ == "__main__":
    measure_time(main)()