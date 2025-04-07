import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from PIL import Image
import io
import base64

# Assumed Camera Intrinsics (Modify if true values become available)
IMG_WIDTH = 224
IMG_HEIGHT = 224
# Assume focal length is roughly image dimension and principal point is center
FX = 224.0
FY = 224.0
CX = IMG_WIDTH / 2.0
CY = IMG_HEIGHT / 2.0
CAMERA_MATRIX = np.array([
    [FX, 0, CX],
    [0, FY, CY],
    [0, 0, 1]
], dtype=np.float32)

# No distortion assumed
DIST_COEFFS = np.zeros((4, 1)) 

# Length of the axes to draw (adjust for visual scale)
# Increase significantly for better visibility
AXIS_LENGTH = 0.5 # Example: Increased from 0.1 to 0.5

def draw_pose_axes(image_bytes, quaternion, translation):
    """
    Draws 3D coordinate axes on the image based on the predicted pose.

    Args:
        image_bytes: Input image as bytes.
        quaternion: Predicted quaternion [w, x, y, z] or [x, y, z, w].
                    scipy expects [x, y, z, w]. Ensure correct order.
        translation: Predicted translation vector [tx, ty, tz].

    Returns:
        Base64 encoded string of the image with axes drawn, or None on error.
    """
    try:
        # Load image using OpenCV
        img_np = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            print("Error: Could not decode image.")
            return None

        # Resize image if it's not the expected size (important for assumed intrinsics)
        # Use the actual camera intrinsics provided now
        # if img.shape[1] != IMG_WIDTH or img.shape[0] != IMG_HEIGHT:
        #      img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        # Keep original size for projection with correct intrinsics

        # --- Use Provided Camera Intrinsics --- 
        fx = 2988.5795163815555
        fy = 2988.3401159176124
        ccx = 960
        ccy = 600
        camera_matrix = np.array([
            [fx, 0, ccx],
            [0, fy, ccy],
            [0, 0, 1]
        ], dtype=np.float32)

        dist_coeffs = np.array([
            -0.22383016606510672,
            0.51409797089106379,
            -0.00066499611998340662,
            -0.00021404771667484594,
            -0.13124227429077406
        ], dtype=np.float32)
        # --- End Intrinsics ---


        # Ensure inputs are numpy arrays
        translation_vector = np.array(translation, dtype=np.float32).reshape(3, 1)
        quat = np.array(quaternion, dtype=np.float32)

        # Ensure quaternion is in [x, y, z, w] format for SciPy
        # Assuming input is [w, x, y, z] from the model
        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        # If model output is already [x, y, z, w]:
        # quat_xyzw = quat

        # Handle potential zero quaternion
        if np.all(quat_xyzw == 0):
             print("Warning: Zero quaternion received, cannot draw axes.")
             _, buffer = cv2.imencode('.png', img)
             img_base64 = base64.b64encode(buffer).decode('utf-8')
             return img_base64

        # Normalize quaternion
        norm = np.linalg.norm(quat_xyzw)
        if norm < 1e-6:
             print("Warning: Near-zero quaternion norm, cannot draw axes.")
             _, buffer = cv2.imencode('.png', img)
             img_base64 = base64.b64encode(buffer).decode('utf-8')
             return img_base64
        quat_xyzw = quat_xyzw / norm

        # Convert quaternion to rotation vector
        try:
            rotation = R.from_quat(quat_xyzw)
            rotation_vector = rotation.as_rotvec()
        except ValueError as e:
            print(f"Error converting quaternion {quat_xyzw} to rotation: {e}")
            _, buffer = cv2.imencode('.png', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return img_base64

        # Define 3D points for the axes origin and endpoints
        axis_points_3d = np.float32([
            [0, 0, 0],         # Origin
            [AXIS_LENGTH, 0, 0], # X-axis end
            [0, AXIS_LENGTH, 0], # Y-axis end
            [0, 0, AXIS_LENGTH]  # Z-axis end
        ]).reshape(-1, 3)

        # Project 3D points to 2D image plane using actual intrinsics
        image_points_2d, _ = cv2.projectPoints(axis_points_3d,
                                               rotation_vector,
                                               translation_vector,
                                               camera_matrix, # Use actual matrix
                                               dist_coeffs)   # Use actual coeffs

        # Check if projection resulted in valid points (not NaN)
        if np.isnan(image_points_2d).any():
             print("Warning: NaN values occurred during 3D point projection. Cannot draw axes.")
             _, buffer = cv2.imencode('.png', img)
             img_base64 = base64.b64encode(buffer).decode('utf-8')
             return img_base64

        # Draw the axes lines on the image
        try:
            origin = tuple(map(int, image_points_2d[0].ravel()))
            x_axis_end = tuple(map(int, image_points_2d[1].ravel()))
            y_axis_end = tuple(map(int, image_points_2d[2].ravel()))
            z_axis_end = tuple(map(int, image_points_2d[3].ravel()))
        except ValueError:
             print("Warning: Could not convert projected points to integers. Cannot draw axes.")
             _, buffer = cv2.imencode('.png', img)
             img_base64 = base64.b64encode(buffer).decode('utf-8')
             return img_base64

        # Increase line thickness and use arrowed lines
        line_thickness = 3 # Increased thickness
        tip_length = 0.15  # Relative length of the arrowhead

        # Draw lines with arrow heads (BGR colors)
        cv2.arrowedLine(img, origin, x_axis_end, (0, 0, 255), line_thickness, tipLength=tip_length) # X = Red
        cv2.arrowedLine(img, origin, y_axis_end, (0, 255, 0), line_thickness, tipLength=tip_length) # Y = Green
        cv2.arrowedLine(img, origin, z_axis_end, (255, 0, 0), line_thickness, tipLength=tip_length) # Z = Blue


        # Encode the modified image to base64 string
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

    except Exception as e:
        print(f"Error drawing pose axes: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to original image
        try:
             img_np = np.frombuffer(image_bytes, np.uint8)
             img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
             if img is not None:
                 _, buffer = cv2.imencode('.png', img)
                 img_base64 = base64.b64encode(buffer).decode('utf-8')
                 return img_base64
        except Exception as fallback_e:
             print(f"Error during fallback image encoding: {fallback_e}")

        return None # Indicate failure