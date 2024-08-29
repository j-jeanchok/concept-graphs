
**NOTE:** Please check the `README_Origin.md` for the original instruction. This file modifies the original one and adds more detailed explanation to ensure other people can follow more easily.  


# ConceptGraphs: Open-Vocabulary 3D Scene Graphs for Perception and Planning

This repository contains the code for the ConceptGraphs project. ConceptGraphs builds open-vocabulary 3D scenegraphs that enable a broad range of perception and task planning capabilities.

[**Project Page**](https://concept-graphs.github.io/) |
[**Paper**](https://concept-graphs.github.io/assets/pdf/2023-ConceptGraphs.pdf) |
[**ArXiv**](https://arxiv.org/abs/2309.16650) |
[**Video**](https://www.youtube.com/watch?v=mRhNkQwRYnc&feature=youtu.be&ab_channel=AliK)


## 1. Installation 

Local condition: 

```text 
Hardware: 
    CPU Intel Core i7 12th 
    GPU NVIDIA RTX 3070 Ti 8GB   
OS: 
    Ubuntu 22.04
Softwares:
    python 3.10 
    GPU Driver Version 550.90
    CUDA: 12.4
```

We will use python virtual environment and pip instead of Anaconda as used in the original instruction. To create your python environment, run the following commands: 

```bash 
## Create virtual environment 
python3 -m venv conceptgraph 
source conceptgraph/bin/activate 

## Install PyTorch 
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

## Install the Faiss library (CPU version should be fine), this is used for quick indexing of pointclouds for
# duplicate object matching and merging
pip install faiss-cpu==1.8.0.post1

## Install PyTorch3D (stable version)
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

## CUDA Toolkit: This is a problematic step without using conda. Since I have CUDA Toolkit installed before, I am not
# sure if there are any packages required CUDA other than PyTorch, however, if only PyTorch requires, the binary
# PyTorch package installed above has already included its CUDA environment. If required,
# please follow the instruction on [Offical Website][https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04]

## Install the other required libraries (with version used)
pip install dill==0.3.8 distinctipy==1.3.4 h5py==3.11.0 hydra-core==1.3.2 imageio==2.34.2 kornia==0.7.3 natsort==8.4.0 open3d==0.18.0 open_clip_torch==2.26.1 openai==1.37.1 pyliblzfse==0.4.1 pypng==0.20220715.0 rerun-sdk==0.17.0 supervision==0.22.0 tyro==0.8.5 ultralytics==8.2.66 wandb==0.17.5

pip install git+https://github.com/ultralytics/CLIP.git

# You also need to ensure that the installed packages can find the right cuda installation.
# You can do this by setting the CUDA_HOME environment variable.
# You can manually set it to the python environment you are using, or set it to the conda prefix of the environment.
# for me its export CUDA_HOME=/home/kirinhuang/Documents/Code/Python/concept-graphs/conceptgraph
export CUDA_HOME=/path/to/virtual/environment/conceptgraph

## Finally install conceptgraphs
cd /path/to/code/ # wherever you want to install conceptgraphs
# For me, the path will be /home/kirinhuang/Documents/Code/Python/concept-graphs

git clone https://github.com/hntkien/concept-graphs.git
cd concept-graphs
git checkout ali-dev
pip install -e .
```

## 2. Datasets 

### Replica 

Now you will need some data to run the code on, the easiest one to use is the [Replica](https://github.com/facebookresearch/Replica-Dataset). You can install it by using the following commands:

```bash
cd /path/to/data
# you can also download the Replica.zip manually through
# link: https://caiyun.139.com/m/i?1A5Ch5C3abNiL password: v3fY (the zip is split into smaller zips because of the size limitation of caiyun)
wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
unzip Replica.zip
```


## 3. Classes File

The classes file is used as a set of class labels to the object detection model. This is a tradeoff between sacrificing some "open-vocabulary-ness" for more stable detections. We have two classes files avaiable:

```bash
concept-graphs/conceptgraph/scannet200_classes.txt
concept-graphs/conceptgraph/ram_classes_4500.txt
```

The scannet classes are the labels from the [scannet200 dataset](https://rozdavid.github.io/scannet200).
The ram classes are the tags from the [RAM model](https://recognize-anything.github.io/).

Modify the classes file used in `/hydra_configs/classes.yaml`. 


## 4. OpenAI Key 

Please set the OpenAI API Key globally or locally in the virtual environment prior to running the scripts below. Setting in the virtual environment works for me but globally does not. Open `path/to/virtual/env/bin/activate` with any text editor and add the API Key as: 

```bash
export OPENAI_API_KEY="your_key"
```


## 5. Usage 

### 5.1 Setting up your configuration 
We use the [hydra](https://hydra.cc/) package to manage the configuration, so you don't have to give it a bunch of command line arguments, just edit the  entries in the corresponding `.yaml` file in `./conceptgraph/hydra_configs/<script_name>` and run the script.

For example, the configuration file for `./conceptgraph/slam/rerun_realtime_mapping.py` is `./conceptgraph/hydra_configs/rerun_realtime_mapping.yaml`. This configuration file looks like this: 

```yaml
defaults:
  - base
  - base_mapping
  - replica
  - sam
  - classes
  - logging_level
  - _self_

detections_exp_suffix: s_detections_stride_10_run2 # just a convenient name for the detection run
force_detection: !!bool False
save_detections: !!bool True

use_rerun: !!bool True
save_rerun: !!bool True

stride: 10
exp_suffix: r_mapping_stride_10_run2 # just a convenient name for the mapping run
```

All the fundamental `.yaml` files are located in `./conceptgraph/hydra_configs/`. The values are loaded top-down, first from `base.yaml`, then `base_mapping.yaml` then `replica.yaml` and so on. If there is a conflict (i.e. two files are modifying the same config parameter), the values from the earlier file are overwritten. i.e. `replica.yaml` will overwrite any confliting values in `base.yaml` and so on.

Finally `_self_` is loaded, which are the values in `rerun_realtime_mapping.yaml` itself. This is where you can put your own custom values. Also feel free to add your own `.yaml` files to `./conceptgraph/hydra_configs/` and they will be loaded in the same way.

Currently, three datasets were tested: replica, record3d, and realsense. Each dataset configuration file loads two variables called `data_root` and `repo_root` from `./conceptgraph/hydra_configs/base_paths.yaml` file. First, please go to that file and modify the paths according to your setup. In my case, this file will be: 

```yaml
repo_root: /home/kirinhuang/Documents/Code/Python/concept-graphs/concept-graphs
data_root: /media/kirinhuang/SharedOS/my_local_data
```

(I cloned the `concept-graphs` repo into a directory called `concept-graphs`, this is not a typo. Also, all the dataset were stored in a directory called `my_local_data`, you can change the name as you wish)

To run the detection script, you need to edit the paths for the dataset in the `.yaml` file. Here is an example of my `concept-graphs/conceptgraph/hydra_configs/realsense.yaml` file, you need to change these paths to point to where you stored the dataset for RealSense camera:

```yaml
defaults:
  - base_paths
  
dataset_root: ${data_root}/realsense
# (full) /media/kirinhuang/SharedOS/my_local_data/realsense
scene_id: custom_dataset_1 
dataset_config: ${repo_root}/conceptgraph/dataset/dataconfigs/realsense/manipulator_config.yaml
render_camera_path: ${repo_root}/conceptgraph/dataset/dataconfigs/realsense/realsense_camera.json
```


### 5.2 Other configuration files
In the `${repo_root}/conceptgraph/dataset/dataconfigs/` directory, there is one directory containing 2 files for each dataset. The `.yaml` file contains intrinsic parameters of the camera as variables. The `.json` files contains the camera's intrinsics and transformation matrices flattened as column vectors. These parameters are used to load the initial camera model for Open3D. 


### 5.3 Running with Replica dataset 

#### 5.3.1 Node prediction and edge prediction 

First, open `./conceptgraph/hydra_configs/streamlined_detections.yaml` and change the `stride` and `exp_suffix` values as you wish. For instance, `stride = 5` means the program will run prediction on every 5 images in the dataset, i.e., image1 -> image5 -> image10 -> (...). `exp_suffix` determines the directory to store experimental results. Modifying this variable helps differentiate different experiments. Additionally, please specify which dataset you want to use by uncommenting it and commenting out the dataset you do not use. For example, assuming `exp_suffix = s_detections_stride50_yes_bg_44_mr`, here is what the directory tree for `room0` in the `Replica` dataset after running the script:

```bash
${data_root} # This is the dataset root
./Replica # This is the dataset directory
./Replica/room0 # This is the scene_id
./Replica/room0/exps # This parent folder of all the results from conceptgraphs

# This is the folder for the specific run, named according to the exp_suffix
./Replica/room0/exps/s_detections_stride50_yes_bg_44_mr
```

**Note:** For large dataset with small camera transition, setting a higher value for `stride` helps the program run faster without affecting the result much because the difference between each frames is small. However, for fast-moving camera (and real-time run), a small value for `stride` (e.g., 1) will be better. 

To run the script, simply run the following command from the `conceptgraph` directory:

```bash
cd /path/to/code/concept-graphs/conceptgraph/
python /scripts/streamlined_detections.py
```

Note that if you don't have the models installed, it should just automatically download them for you. Here is what the ouput of running the detections script looks like for `room0` in the `Replica` dataset:

```bash
.
./Replica # This is the dataset root
./Replica/room0 # This is the scene_id
./Replica/room0/exps # This parent folder of all the results from conceptgraphs

# This is the folder for the specific run, named according to the exp_suffix
./Replica/room0/exps/s_detections_stride50_yes_bg_44_mr 

# This is where the visualizations are saved, they are images with bounding boxes and masks overlayed
./Replica/room0/exps/s_detections_stride50_yes_bg_44_mr/vis 
./Replica/room0/exps/s_detections_stride50_yes_bg_44_mr/vis/frame000000.jpg 
./Replica/room0/exps/s_detections_stride50_yes_bg_44_mr/vis/frame000050.jpg
./Replica/room0/exps/s_detections_stride50_yes_bg_44_mr/vis/frame000100.jpg
... # rest of the frames
./Replica/room0/exps/s_detections_stride50_yes_bg_44_mr/vis/frame001950.jpg

# This is where the detection results are saved, they are in the form of pkl.gz files 
# that contain a dictionary of the detection results
./Replica/room0/exps/s_detections_stride50_yes_bg_44_mr/detections 
./Replica/room0/exps/s_detections_stride50_yes_bg_44_mr/detections/frame000000.pkl.gz
./Replica/room0/exps/s_detections_stride50_yes_bg_44_mr/detections/frame000050.pkl.gz
./Replica/room0/exps/s_detections_stride50_yes_bg_44_mr/detections/frame000100.pkl.gz
... # rest of the frames
./Replica/room0/exps/s_detections_stride50_yes_bg_44_mr/detections/frame001950.pkl.gz
```


#### 5.3.2 3D Mapping 

Similarly, you need to edit the `streamlined_mapping.yaml` file in the `./conceptgraph/hydra_configs/` directory to point to your paths. Note that you need to tell the mapping script which detection results to use, so you need to set the `detections_exp_suffix` to the `exp_suffix` of the detection run you want to use. So following the example above, you would set 

```yaml
detections_exp_suffix: s_detections_stride50_yes_bg_44_mr
```

in the `streamlined_mapping.yaml` file. Note that the `exp_suffix` in this mapping configuration file can be difined as you like. Once you have set up your mapping configuration, then you can run the mapping script with the following command:

```bash
cd /path/to/code/concept-graphs/conceptgraph/
python slam/streamlined_mapping.py
```

And here is what the output folder looks like for the mapping script:

```bash
.
./Replica # This is the dataset root
./Replica/room0 # This is the scene_id
./Replica/room0/exps # This parent folder of all the results from conceptgraphs

# This is the mapping output folder for the specific run, named according to the exp_suffix
./Replica/room0/exps/s_mapping_yes_bg_multirun_49/
# This is the saved configuration file for the run
./Replica/room0/exps/s_mapping_yes_bg_multirun_49/config_params.json
# We also save the configuration file of the detection run which was used 
./Replica/room0/exps/s_mapping_yes_bg_multirun_49/config_params_detections.json
# The mapping results are saved in a pkl.gz file
./Replica/room0/exps/s_mapping_yes_bg_multirun_49/pcd_s_mapping_yes_bg_multirun_49.pkl.gz
# The video of the mapping process is saved in a mp4 file
./Replica/room0/exps/s_mapping_yes_bg_multirun_49/s_mapping_s_mapping_yes_bg_multirun_49.mp4
# If you set save_objects_all_frames=True, then the object mapping results are saved in a folder
./Replica/room0/exps/s_mapping_yes_bg_multirun_49//saved_obj_all_frames
# In the saved_obj_all_frames folder, there is a folder for each detection run used, and in each of those folders there is a pkl.gz file for each object mapping result
./Replica/room0/exps/s_mapping_yes_bg_multirun_49/saved_obj_all_frames/det_exp_s_detections_stride50_yes_bg_44_mr
000001.pkl.gz
000002.pkl.gz
000003.pkl.gz
...
000039.pkl.gz
meta.pkl.gz
```


#### 5.3.3 Visualization

This script allows you to visualize the map in 3D and query the map objects with text. 

```bash
cd /path/to/code/concept-graphs/conceptgraph/
python scripts/visualize_cfslam_results.py --result_path /path/to/output.pkl.gz
```

So for the above example where the output file is stored in `room0/exps/s_mapping_yes_bg_multirun_49/pcd_s_mapping_yes_bg_multirun_49.pkl.gz` this would look like 

```bash
cd /path/to/code/concept-graphs/conceptgraph/
python scripts/visualize_cfslam_results.py --result_path /path/to/data/root/Replica/room0/exps/s_mapping_yes_bg_multirun_49/pcd_s_mapping_yes_bg_multirun_49.pkl.gz
```

Then in the open3d visualizer window, you can use the following key callbacks to change the visualization. 
* Press `b` to toggle the background point clouds (wall, floor, ceiling, etc.). Only works on the ConceptGraphs-Detect.
* Press `c` to color the point clouds by the object class from the tagging model. Only works on the ConceptGraphs-Detect.
* Press `r` to color the point clouds by RGB. 
* Press `f` and type text in the terminal, and the point cloud will be colored by the CLIP similarity with the input text. 
* Press `i` to color the point clouds by object instance ID. 

**Remember to click on the Open3D GUI before pressing any key**


#### 5.3.4 All in one 

The `rerun_realtime_mapping.py` script runs the detections, builds the scene graph, and vizualizes the results all in one loop. Configuration values are similar to those in the previous two files. This file helps with visualizing the map as it runs detection and mapping, allowing us to see how the whole pipeline works. Other than that, there is no use in running this script due to a high computational resource required. 

**After you have changed the needed configuration values**, you can run a script with a simple command, for example:

```bash
# set up your config first as explained below, then
cd /path/to/code/concept-graphs/conceptgraph/
python slam/rerun_realtime_mapping.py
```

During execution, you can stop the program early instead of running on the whole dataset. Simply open `./conceptgraph/hydra_configs/early_exit.json`, change the value from `false` to `true` and save the file. Note that the first letter is not capitalized. 

**NOTE:** For convinience, the script will also automatically create a symlink `/concept-graphs/latest_pcd_save` -> `Replica/room0/exps/r_mapping_stride_10_run2/pcd_r_mapping_stride_10_run2.pkl.gz` so you can easily access the latest results by using the `latest_pcd_save` path in your argument to the visualization script.

```bash
cd /path/to/code/concept-graphs
python conceptgraph/scripts/visualize_cfslam_results.py \
    --result_path latest_pcd_save 
```

or if you'd like to point it to a specific result, you can just point it to the pkl.gz file directly:

```bash
cd /path/to/code/concept-graphs
python conceptgraph/scripts/visualize_cfslam_results.py \
    --result_path /path/to/data/Replica/room0/exps/r_mapping_stride_10_run2/pcd_r_mapping_stride_10_run2.pkl.gz
```


### 5.4 Running with RealSense Camera 

#### 5.4.1 Camera module 

First, install the `pyrealsense2` package. Note that only version <= 2.53 works with T265. 

```bash
# Install the latest version that still support T265
pip install pyrealsense2==2.53.1.4623
```

`./conceptgraph/realsense/realsense.py` defines a class to implement RealSense D435i and T265 cameras. The D435i can be replaced by any other RealSense stereo cameras, such as D400 series. (L515 (LiDAR) should work as well). For each frame, the module will output a color frame, depth frame, intrinsics matrix, and corrected transformation matrix for two cameras. These four outputs are of NumPy array data type. 


#### 5.4.2 Offline detection

To perform offline prediction, we need to generate a dataset like Replica. Connect the cameras and run the script: 

```bash 
cd /path/to/code/concept-graphs/conceptgraph/
python realsense/dataset_generation.py
```

to generate a dataset in the directory specified in `./conceptgraph/hydra_configs/realsense.yaml`. The maximum frames generated is currently set to 300, but you can adjust according to your preference. You can also stop the program early if needed similar to part 5.3.4. 

After obtaining the dataset, run 

```bash
cd /path/to/code/concept-graphs/conceptgraph/
python /scripts/streamlined_detections.py
```

to perform detection, just as in 5.3.1. 


#### 5.4.3 Online detection

We can also generate the dataset and perform detection simultaneously. In this case, we will run detection for each frame obtained from the camera and save the results to the directory specified in the `realsense.yaml` configuration file. The configuration file used for this script is `streamlined_detections.yaml`. Early exit is also available with the same usage. Connect the cameras and run 

```bash
cd /path/to/code/concept-graphs/conceptgraph/
python realsense/realtime_detections.py
```

**Note:** Due to the sending/receiving request process of OpenAI, the processing time for each frame is long; thus, please avoid moving the camera too fast. Offline is a better choice if you want to have a clearer and more accurate map. 


#### 5.4.4 Mapping

Both offline and online detection scripts generate the scene graph for each frame and stored the results in the corresponding location. Run the command

```bash
cd /path/to/code/concept-graphs/conceptgraph/
python slam/streamlined_mapping.py
```

to obtain the 3D map of the scene based on the detection result. Set `save_json` variable in `./conceptgraph/hydra_configs/base_mapping.yaml` to `True` will log the final scene graph results into a `.json` file which include the objects and their relationships. If, for instance, the `exp_suffix` in the `streamlined_mapping.yaml` is `s_mapping_stride3_1`, then the `.json` file will be saved in `${data_root}/{dataset_name}/{scene_id}/exps/s_mapping_stride3_1/edge_json_s_mapping_stride3_1.json`. 


#### 5.4.5 Visualization 

Run the command:

```bash
cd /path/to/code/concept-graphs/conceptgraph/
python scripts/visualize_cfslam_results.py --result_path /path/to/mapping_output.pkl.gz
```

and perform segmentation/grounding task as in 5.3.3. 


#### 5.4.5 All in one (!!!)

Run the following command:

```bash
cd /path/to/code/concept-graphs/conceptgraph/
python realsense/rs_stream_rerun_realtime_mapping.py
```

to do the same as in 5.3.4. However, due the rendering process, this program requires a large computational resource. 


### Rendering boolean variables

- make_edges: `True` will generate the edge between two objects. `False` otherwise
- save_video: Save a video recording the detection/mapping procedure for each frame. 
- save_objects_all_frames: Save metadata of the objects for each frame 
- vis_render: Open3D video rendering as the program is running. Recommend to be False so as not to consume GPU resource
- use_rerun: Decide whether to visualize the process with rerun.io
