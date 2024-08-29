# ===== Standard Library Imports ===== # 
from pathlib import Path
import gzip
import pickle

# ===== Third-party Imports ===== #
import cv2
import numpy as np
from tqdm import trange
import hydra
from omegaconf import DictConfig
import torch

from ultralytics import YOLO, SAM
import supervision as sv
import open_clip

# ===== Local application/library scpecific imports ===== #
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import vis_result_fast, save_video_detections
from conceptgraph.utils.general_utils import (
    get_det_out_path, 
    get_exp_out_path, 
    get_vis_out_path, 
    make_vlm_edges_and_captions,
    measure_time, 
    save_hydra_config, 
    ObjectClasses
)
from conceptgraph.utils.model_utils import compute_clip_features_batched 
from conceptgraph.utils.vlm import get_openai_client


@hydra.main(
    version_base=None, 
    config_path="../hydra_configs/", 
    config_name="streamlined_detections")

def main(cfg : DictConfig):

    # Initialize the dataset
    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.desired_height,
        desired_width=cfg.desired_width,
        device="cpu",
        dtype=torch.float,
    )

    # Output folder of the detections experiment to use
    det_exp_path = get_exp_out_path(
        dataset_root=cfg.dataset_root, 
        scene_id=cfg.scene_id, 
        exp_suffix=cfg.exp_suffix)
    det_exp_pkl_path = get_det_out_path(det_exp_path)
    det_exp_vis_path = get_vis_out_path(det_exp_path)

    ## Initialize the detection models
    detection_model = YOLO('yolov8l-world.pt')
    sam_predictor = SAM('mobile_sam.pt') # UltraLytics SAM
    # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(cfg.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    
    # Set the classes for the detection model
    obj_classes = ObjectClasses(
        classes_file_path=cfg.classes_file, 
        bg_classes=cfg.bg_classes, 
        skip_bg=cfg.skip_bg)
    
    detection_model.set_classes(obj_classes.get_classes_arr())

    # Try with OpenAI Client
    openai_client = get_openai_client()
    
    save_hydra_config(cfg, det_exp_path)

    for frame_idx in trange(len(dataset)):

        # Relevant paths and load image
        color_path = Path(dataset.color_paths[frame_idx])
        # opencv can't read Path objects...
        image = cv2.imread(str(color_path)) # This will in BGR color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Do initial object detection
        results = detection_model(image_rgb)

        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detection_class_labels = [
            f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" 
            for class_idx, class_id in enumerate(detection_class_ids)
        ]
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()
        
        # # Get Masks Using UltraLytics SAM
        if xyxy_tensor.numel() != 0:
            sam_out = sam_predictor(image_rgb, bboxes=xyxy_tensor)
            masks_np = sam_out[0].masks.data.cpu().numpy()
        else:
            masks_np = np.empty((0, *image_rgb.shape[:2]), dtype=np.float64)
        
        # Create a detections object that we will save later
        curr_det = sv.Detections(
            xyxy=xyxy_np,
            confidence=confidences,
            class_id=detection_class_ids,
            mask=masks_np,
        )
        
        ## Extract edges and captions from OpenAI API
        labels, edges, _, captions = make_vlm_edges_and_captions(image_rgb, curr_det, obj_classes, detection_class_labels, det_exp_vis_path, color_path, make_edges_flag=True, openai_client=openai_client)
        
        # Compute and save the clip features of detections  
        image_crops, image_feats, text_feats = compute_clip_features_batched(
            image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), cfg.device)


        # Save results
        # Convert the detections to a dict. The elements are in np.array
        results = {
            # add new uuid for each detection 
            "xyxy": curr_det.xyxy,
            "confidence": curr_det.confidence,
            "class_id": curr_det.class_id,
            "mask": curr_det.mask,
            "classes": obj_classes.get_classes_arr(),
            "image_crops": image_crops,
            "image_feats": image_feats,
            "text_feats": text_feats,
            "detection_class_labels": detection_class_labels,
            "labels": labels,
            "edges": edges,
            "captions": captions
        }
        
        # save the detections
        vis_save_path = (det_exp_vis_path / Path(color_path).name).with_suffix(".jpg")

        #Visualize and save the annotated image
        annotated_image, labels = vis_result_fast(image, curr_det, obj_classes.get_classes_arr())
        cv2.imwrite(str(vis_save_path), annotated_image)
        curr_detection_name = (vis_save_path.stem + ".pkl.gz")
        with gzip.open(det_exp_pkl_path / curr_detection_name , "wb") as f:
            pickle.dump(results, f)
        
    if cfg.save_video:
        save_video_detections(det_exp_path)
        

if __name__ == "__main__":
    measure_time(main)()