'''
The script is used to model Grounded SAM detections in 3D, it assumes the tag2text classes are avaialable. It also assumes the dataset has Clip features saved for each object/mask.
'''

# ===== Standard Library Imports ===== # 
import os
import copy
import os
from pathlib import Path
import gzip
import pickle
from collections import Counter 

# ===== Third-party Imports ===== #
from PIL import Image
import numpy as np
import open3d as o3d
import torch
from tqdm import trange

import hydra
from omegaconf import DictConfig

# ===== Local application/library scpecific imports ===== #
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.logging_metrics import DenoisingTracker
from conceptgraph.utils.vis import OnlineObjectRenderer, save_video_from_frames
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses, 
    get_det_out_path, 
    get_exp_out_path, 
    load_saved_hydra_json_config, 
    measure_time, 
    save_hydra_config, 
    save_edge_json,
)

from conceptgraph.slam.slam_classes import MapObjectList, MapEdgeMapping
from conceptgraph.slam.utils import (
    filter_gobs,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    filter_objects,
    merge_objects, 
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_cfg,
    process_edges,
    processing_needed,
    resize_gobs,
    init_process_pcd, 
    get_bounding_box
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
    merge_obj_matches
)
from conceptgraph.utils.vlm import consolidate_captions, get_openai_client


# Disable torch gradient computation
torch.set_grad_enabled(False)

# A logger for this file
@hydra.main(
    version_base=None, 
    config_path="../hydra_configs/", 
    config_name="streamlined_mapping")

def main(cfg : DictConfig):
    cfg = process_cfg(cfg)

        # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
    )

    objects = MapObjectList(device=cfg.device)
    map_edges = MapEdgeMapping(objects)

    openai_client = get_openai_client()

    # For visualization
    if cfg.vis_render:
        view_param = o3d.io.read_pinhole_camera_parameters(
            cfg.render_camera_path)
        obj_renderer = OnlineObjectRenderer(
            view_param=view_param, 
            base_objects=None, 
            gray_map=False) 
        frames = []

    # Output folder for this mapping experiement 
    exp_out_path = get_exp_out_path(
        dataset_root=cfg.dataset_root, 
        scene_id=cfg.scene_id, 
        exp_suffix=cfg.exp_suffix)
    
    # Output folder of the detections experiment to use 
    det_exp_path = get_exp_out_path(
        dataset_root=cfg.dataset_root, 
        scene_id=cfg.scene_id, 
        exp_suffix=cfg.detections_exp_suffix)
    

    # we need to make sure to use the same classes as the ones used in the detections
    detections_exp_cfg = load_saved_hydra_json_config(det_exp_path)
    obj_classes = ObjectClasses(
        classes_file_path=detections_exp_cfg['classes_file'], 
        bg_classes=detections_exp_cfg['bg_classes'], 
        skip_bg=detections_exp_cfg['skip_bg']
    )
    
    # The actual folder with the detections from the detections experiment
    det_exp_pkl_path = get_det_out_path(det_exp_path)

    save_hydra_config(cfg, exp_out_path)
    save_hydra_config(
        detections_exp_cfg, 
        exp_out_path, 
        is_detection_config=True)

    if cfg.save_objects_all_frames:
        obj_all_frames_out_path = exp_out_path / "saved_obj_all_frames" / f"det_{cfg.detections_exp_suffix}"
        os.makedirs(obj_all_frames_out_path, exist_ok=True)

    for frame_idx in trange(len(dataset)):

        # Read info about current frame from dataset
        # color image
        color_path = Path(dataset.color_paths[frame_idx])
        image_original_pil = Image.open(color_path)
        # color and depth tensors, and camera instrinsics matrix
        color_tensor, depth_tensor, intrinsics, *_ = dataset[frame_idx]
        

        # Covert to numpy and do some sanity checks
        depth_tensor = depth_tensor[..., 0]
        depth_array = depth_tensor.cpu().numpy()
        color_np = color_tensor.cpu().numpy() # (H, W, 3)
        image_rgb = (color_np).astype(np.uint8) # (H, W, 3)
        assert image_rgb.max() > 1, "Image is not in range [0, 255]"
        
    
        # Load image detections for the current frame
        raw_gobs = None
        gobs = None # stands for grounded observations
        detections_path = det_exp_pkl_path / (color_path.stem + ".pkl.gz")

        # Load the detections
        color_path = str(color_path)
        detections_path = str(detections_path)
        try:
            with gzip.open(detections_path, "rb") as f:
                raw_gobs = pickle.load(f)
        except FileNotFoundError:
            if frame_idx == len(dataset) - 1:
                break 
            else:
                continue 

        # get pose, this is the untrasformed pose.
        unt_pose = dataset.poses[frame_idx]
        unt_pose = unt_pose.cpu().numpy()

        # Don't apply any transformation otherwise
        adjusted_pose = unt_pose

        # resize the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # filter the observations
        filtered_gobs = filter_gobs(resized_gobs, image_rgb, 
            skip_bg=cfg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=cfg.mask_area_threshold,
            max_bbox_area_ratio=cfg.max_bbox_area_ratio,
            mask_conf_threshold=cfg.mask_conf_threshold,
        )

        gobs = filtered_gobs

        if len(gobs['mask']) == 0: # no detections in this frame
            continue

        # this helps make sure things like pillows on couches are separate objects
        gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth_array,
            masks=gobs['mask'],
            cam_K=intrinsics.cpu().numpy()[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=adjusted_pose,
            min_points_threshold=cfg.min_points_threshold,
            spatial_sim_type=cfg.spatial_sim_type,
            obj_pcd_max_points=cfg.obj_pcd_max_points,
            device=cfg.device,
        )

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=cfg['spatial_sim_type'], 
                    pcd=obj["pcd"],
                )

        detection_list = make_detection_list_from_pcd_and_gobs(
            obj_pcds_and_bboxes, gobs, color_path, obj_classes, frame_idx
        )

        if len(detection_list) == 0: # no detections, skip
            continue

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(objects) == 0:
            objects.extend(detection_list)
            continue 

        ### compute similarities and then merge
        try:
            spatial_sim = compute_spatial_similarities(
                spatial_sim_type=cfg['spatial_sim_type'], 
                detection_list=detection_list, 
                objects=objects,
                downsample_voxel_size=cfg['downsample_voxel_size']
            )

        except ValueError:
            print(f"ValueError Occurs at Frame {frame_idx}")

            # Stop if the error occurs at the final frame
            if frame_idx == len(dataset) - 1:
                break 
            else:
                # Continue next frame execution if not final frame even error occurs. 
                continue

        visual_sim = compute_visual_similarities(detection_list, objects)

        agg_sim = aggregate_similarities(
            match_method=cfg['match_method'], 
            phys_bias=cfg['phys_bias'], 
            spatial_sim=spatial_sim, 
            visual_sim=visual_sim
        )

        # Perform matching of detections to existing objects
        match_indices = match_detections_to_objects(
            agg_sim=agg_sim, 
            detection_threshold=cfg['sim_threshold']  # Use the sim_threshold from the configuration
        )

        # Now merge the detected objects into the existing objects based on the match indices
        objects = merge_obj_matches(
            detection_list=detection_list, 
            objects=objects, 
            match_indices=match_indices,
            downsample_voxel_size=cfg['downsample_voxel_size'], 
            dbscan_remove_noise=cfg['dbscan_remove_noise'], 
            dbscan_eps=cfg['dbscan_eps'], 
            dbscan_min_points=cfg['dbscan_min_points'], 
            spatial_sim_type=cfg['spatial_sim_type'], 
            device=cfg['device']
            # Note: Removed 'match_method' and 'phys_bias' as they do not appear in the provided merge function
        )

        # fix the class names for objects
        # they should be the most popular name, not the first name
        for _, obj in enumerate(objects):
            temp_class_name = obj["class_name"]
            curr_obj_class_id_counter = Counter(obj['class_id'])
            most_common_class_id = curr_obj_class_id_counter.most_common(1)[0][0]
            most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
            if temp_class_name != most_common_class_name:
                obj["class_name"] = most_common_class_name

        map_edges = process_edges(match_indices, gobs, len(objects), objects, map_edges, frame_idx)

        is_final_frame = frame_idx == len(dataset) - 1

        if is_final_frame:
            print("Final frame detected. Performing final post-processing...")

        # Clean up outlier edges
        edges_to_delete = []

        for curr_map_edge in map_edges.edges_by_index.values():
            curr_obj1_idx = curr_map_edge.obj1_idx
            curr_obj2_idx = curr_map_edge.obj2_idx
            curr_first_detected = curr_map_edge.first_detected
            curr_num_det = curr_map_edge.num_detections

            if (frame_idx - curr_first_detected > 5) and curr_num_det < 2:
                edges_to_delete.append((curr_obj1_idx, curr_obj2_idx))

        for edge in edges_to_delete:
            map_edges.delete_edge(edge[0], edge[1])

        ### Perform post-processing periodically if told so
        
        # Denoising 
        if processing_needed(
            cfg["denoise_interval"],
            cfg["run_denoise_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = measure_time(denoise_objects)(
                downsample_voxel_size=cfg['downsample_voxel_size'], 
                dbscan_remove_noise=cfg['dbscan_remove_noise'], 
                dbscan_eps=cfg['dbscan_eps'], 
                dbscan_min_points=cfg['dbscan_min_points'], 
                spatial_sim_type=cfg['spatial_sim_type'], 
                device=cfg['device'], 
                objects=objects
            )

        # Filtering
        if processing_needed(
            cfg["filter_interval"],
            cfg["run_filter_final_frame"],
            frame_idx,
            is_final_frame,
        ):
            objects = filter_objects(
                obj_min_points=cfg['obj_min_points'], 
                obj_min_detections=cfg['obj_min_detections'], 
                objects=objects, 
                map_edges=map_edges
            )

        # Merging
        try:
            if processing_needed(
                cfg["merge_interval"],
                cfg["run_merge_final_frame"],
                frame_idx,
                is_final_frame,
            ):
                objects, map_edges = measure_time(merge_objects)(
                    merge_overlap_thresh=cfg["merge_overlap_thresh"],
                    merge_visual_sim_thresh=cfg["merge_visual_sim_thresh"],
                    merge_text_sim_thresh=cfg["merge_text_sim_thresh"],
                    objects=objects,
                    downsample_voxel_size=cfg["downsample_voxel_size"],
                    dbscan_remove_noise=cfg["dbscan_remove_noise"],
                    dbscan_eps=cfg["dbscan_eps"],
                    dbscan_min_points=cfg["dbscan_min_points"],
                    spatial_sim_type=cfg["spatial_sim_type"],
                    device=cfg["device"],
                    map_edges=map_edges
                )
        except ValueError:
            print("Error occurs, but keep going with the next step")
        finally:
            # Save the objects for the current frame, if needed
            if cfg.save_objects_all_frames:
                # Define the path for saving the current frame's objects
                save_path = obj_all_frames_out_path / f"{frame_idx:06d}.pkl.gz"
                
                # Filter objects based on minimum number of detections and prepare them for saving
                filtered_objects = [
                    obj for obj in objects 
                    if obj['num_detections'] >= cfg.obj_min_detections]
                prepared_objects = prepare_objects_save_vis(
                    MapObjectList(filtered_objects))

                # Create the result dictionary with camera pose and prepared objects
                result = { 
                    "camera_pose": adjusted_pose, 
                    "objects": prepared_objects}
                # Save the result dictionary to a compressed file
                with gzip.open(save_path, 'wb') as f:
                    pickle.dump(result, f)
        try:
            if cfg.save_json: 
                save_edge_json(
                    exp_suffix=cfg.exp_suffix, 
                    exp_out_path = exp_out_path, 
                    objects=objects, 
                    edges=map_edges
                )
        except IndexError:
            continue

        # Render the image with the filtered and colored objects
        if cfg.vis_render:
            # Initialize an empty list for objects meeting the criteria
            filtered_objects = [
                copy.deepcopy(obj) for obj in objects 
                if obj['num_detections'] >= cfg.obj_min_detections and not obj['is_background']
            ]
            objects_vis = MapObjectList(filtered_objects)

            # Apply coloring based on the configuration
            if cfg.class_agnostic:
                objects_vis.color_by_instance()
            else:
                objects_vis.color_by_most_common_classes(obj_classes)

            # Render the image with the filtered and colored objects
            rendered_image, vis = obj_renderer.step(
                image=image_original_pil,
                gt_pose=adjusted_pose,
                new_objects=objects_vis,
                paint_new_objects=False,
                return_vis_handle=cfg.debug_render,
            )

            # If debug mode is enabled, run the visualization
            if cfg.debug_render:
                vis.run()
                del vis  # Delete the visualization handle to free resources

            # Convert the rendered image to uint8 format, if it exists
            if rendered_image is not None:
                rendered_image = (rendered_image * 255).astype(np.uint8)
                frames.append(rendered_image)

            if is_final_frame:
                # Save frames as a mp4 video
                frames = np.stack(frames)
                video_save_path = exp_out_path / (f"s_mapping_{cfg.exp_suffix}.mp4")
                save_video_from_frames(frames, video_save_path, fps=10)
                print("Save video to %s" % video_save_path)
                
    # LOOP OVER -----------------------------------------------------

    # Consolidate captions 
    for object in objects:
        obj_captions = object['captions'][:20]
        consolidated_caption = consolidate_captions(openai_client, obj_captions)
        object['consolidated_caption'] = consolidated_caption

    # Save the pointcloud
    if cfg.save_pcd:
        print("Saving map ...")

        results = {
            'objects': objects.to_serializable(),
            'cfg': cfg,
            'class_names': obj_classes.get_classes_arr(),
            'class_colors': obj_classes.get_class_color_dict_by_index(),
            "edges": map_edges.to_serializable() if map_edges is not None else None 
        }

        pcd_save_path = Path(exp_out_path / f"pcd_{cfg.exp_suffix}.pkl.gz")
        # make the directory if it doesn't exist
        pcd_save_path.parent.mkdir(parents=True, exist_ok=True)

        with gzip.open(str(pcd_save_path), "wb") as f:
            pickle.dump(results, f)
        print(f"Saved point cloud to {pcd_save_path}")

    # Save metadata if all frames are saved
    if cfg.save_objects_all_frames:
        save_meta_path = obj_all_frames_out_path / f"meta.pkl.gz"
        with gzip.open(save_meta_path, "wb") as f:
            pickle.dump({
                'cfg': cfg,
                'class_names': obj_classes.get_classes_arr(),
                'class_colors': obj_classes.get_class_color_dict_by_index(),
            }, f)
            
    tracker = DenoisingTracker()  # Get the singleton instance of DenoisingTracker
    # tracker.generate_report()

if __name__ == "__main__":
    main()
