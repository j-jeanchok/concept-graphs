import numpy as np
import pyrealsense2 as rs
import torch
from scipy.spatial.transform import Rotation as R 

# from conceptgraph.utils.geometry import quaternion_to_rotation_matrix

# ===== Convert from Quaternion to Rotation Matrix ===== #
def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion into a rotation matrix.
    
    Parameters:
    - q: A quaternion in the format [x, y, z, w].
    
    Returns:
    - A 3x3 rotation matrix.
    """
    w, x, y, z = q[3], q[0], q[1], q[2]
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])


# ===== RealSenseApp Class ===== #
class RealSenseApp:
    def __init__(self):
        # Create two pipelines: One for T265 and one for D400 
        self.pipeline_t265 = rs.pipeline() 
        self.pipeline_d435 = rs.pipeline() 

        # Configure the T265 pipeline 
        self.config_t265 = rs.config()
        self.config_t265.enable_stream(rs.stream.pose)

        # Configure the D435 pipline 
        self.config_d435 = rs.config()
        self.config_d435.enable_stream(
            rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        self.config_d435.enable_stream(
            rs.stream.depth, 1280, 720, rs.format.z16, 30)

        # Get device product line for setting a supporting resolution
        self.pipeline_t265_wrapper = rs.pipeline_wrapper(self.pipeline_t265) 
        self.pipeline_t265_profile = self.config_t265.resolve(
            self.pipeline_t265_wrapper) 
        self.device_t265 = self.pipeline_t265_profile.get_device() 
        self.device_t265_type = str(self.device_t265.get_info(
            rs.camera_info.product_line))
        
        self.pipeline_d435_wrapper = rs.pipeline_wrapper(self.pipeline_d435) 
        self.pipeline_d435_profile = self.config_d435.resolve(
            self.pipeline_d435_wrapper) 
        self.device_d435 = self.pipeline_d435_profile.get_device() 
        self.device_d435_type = str(self.device_d435.get_info(
            rs.camera_info.product_line))

        # Start streaming
        self.profile_t265 = self.pipeline_t265.start(self.config_t265)
        self.profile_d435 = self.pipeline_d435.start(self.config_d435)
        
        # Turn of auto exposure for depth sensor
        self.profile_d435.get_device().first_depth_sensor().set_option(
            rs.option.enable_auto_exposure, False)

        self.intrinsics = None

    def get_depth_scale(self):
        print(f"Device type: {self.device_d435_type}") 
        depth_sensor = self.profile_d435.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale() 
        print(f"Depth Scale is: {depth_scale}")

    def get_intrinsic_mat_from_intrinsics(self, intrinsics):
        return np.array([
            [intrinsics.fx, 0,             intrinsics.ppx],
            [0,             intrinsics.fy, intrinsics.ppy],
            [0,             0,             1             ]])

    def correct_pose(self, transformation_matrix):
        '''
        This function corrects the pose of the camera by flipping the y and z axes. (This is for Record3D but not sure if needed for RealSense)
        '''
        # Define the transformation matrix P
        P = torch.tensor([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]
        ]).float()

        # Convert transformation_matrix to a tensor for matrix multiplication
        transformation_tensor = torch.from_numpy(transformation_matrix).float()

        # Apply P to transformation_tensor
        final_transformation = P @ transformation_tensor @ P.T
        
        return final_transformation

    def get_frame_data(self):
        # Wait for a new set of frames
        frames_d435 = self.pipeline_d435.wait_for_frames()

        # Get color and depth frames
        color_frame = frames_d435.get_color_frame()
        depth_frame = frames_d435.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None, None
        
        # Convert images to numpy arrays
        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())
        
        # Get intrinsic matrix 
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics 
        intrinsics_matrix = self.get_intrinsic_mat_from_intrinsics(intrinsics)

        # Reformat the intrinsic matrix 
        intrinsics_full = np.eye(4) 
        intrinsics_full[:3, :3] = intrinsics_matrix 

        # Create the extrinsic transformation matrix from the calibration file
        # Translation (in meters)
        translation = np.array([0.009, 0.021, 0.027])

        # Rotation (in radians)
        rpy = np.array([0.000, -0.018, 0.005])
        rotation_matrix = R.from_euler('xyz', rpy).as_matrix()

        # Combine rotation and translation into a single transformation matrix
        T_D435i_to_T265 = np.eye(4)
        T_D435i_to_T265[:3, :3] = rotation_matrix
        T_D435i_to_T265[:3, 3] = translation

        # Example: Inverse transformation (T265 to D435i)
        T_T265_to_D435i = np.linalg.inv(T_D435i_to_T265)

        # Obtain camera pose from t265 
        frames_t265 = self.pipeline_t265.wait_for_frames() 
        pose_frame = frames_t265.get_pose_frame() 

        if pose_frame:
            camera_pose = pose_frame.get_pose_data() 

        quaternion = [camera_pose.rotation.x, camera_pose.rotation.y, camera_pose.rotation.z, camera_pose.rotation.w]

        rotation_matrix = quaternion_to_rotation_matrix(quaternion) 

        transformation_matrix = np.eye(4) 
        transformation_matrix[:3, :3] = rotation_matrix 
        transformation_matrix[:3, 3] = [camera_pose.translation.x, camera_pose.translation.y, camera_pose.translation.z]

        # Transform pose to D435i coordinates
        transformation_matrix = T_T265_to_D435i @ transformation_matrix

        # transformation_matrix = self.correct_pose(
        #     transformation_matrix).numpy()

        return rgb, depth, intrinsics_full, transformation_matrix


# # ===== Example usage ===== #
# import cv2
# # if __name__ == "_main__":
# #     app = RealSenseApp()

# #     app.get_depth_scale() 
# app = RealSenseApp()
# # rgb, depth, intrinsics, transformation = app.get_frame_data()

# for _ in range(3):
#     rgb, depth, intrinsics, transformation = app.get_frame_data()
#     if rgb is not None and depth is not None:
#         cv2.imshow("RGB Image", rgb)
#         cv2.imshow("Depth Image", depth)
#         # Debugging: Check if images are displayed
#         print(intrinsics)
#         print("=" * 20) 
#         print(transformation)
#         app.get_depth_scale()
#         # key = cv2.waitKey(1)
#         # if key & 0xFF == ord('q'):  # Press 'q' to exit
#         #     break
#     else:
#         print("No frames to display.")

# cv2.destroyAllWindows()
