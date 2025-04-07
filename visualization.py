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
AXIS_LENGTH = 0.1 # Assuming translation units roughly correspond to this scale relative to image content

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
        if img.shape[1] != IMG_WIDTH or img.shape[0] != IMG_HEIGHT:
             img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        # Ensure inputs are numpy arrays
        translation_vector = np.array(translation, dtype=np.float32).reshape(3, 1)
        quat = np.array(quaternion, dtype=np.float32)

        # Ensure quaternion is in [x, y, z, w] format for SciPy
        # Assuming input is [w, x, y, z] - adjust if your model outputs [x, y, z, w]
        # quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]]) 
        # If model output is already [x, y, z, w]:
        quat_xyzw = quat 

        # Handle potential zero quaternion - return original image?
        if np.all(quat_xyzw == 0):
             print("Warning: Zero quaternion received, cannot draw axes.")
             # Encode original image and return
             _, buffer = cv2.imencode('.png', img)
             img_base64 = base64.b64encode(buffer).decode('utf-8')
             return img_base64

        # Normalize quaternion just in case
        norm = np.linalg.norm(quat_xyzw)
        if norm < 1e-6:
             print("Warning: Near-zero quaternion norm, cannot draw axes.")
             _, buffer = cv2.imencode('.png', img)
             img_base64 = base64.b64encode(buffer).decode('utf-8')
             return img_base64
        quat_xyzw = quat_xyzw / norm

        # Convert quaternion to rotation vector (needed for cv2.projectPoints)
        try:
            rotation = R.from_quat(quat_xyzw)
            rotation_vector = rotation.as_rotvec()
        except ValueError as e:
            print(f"Error converting quaternion {quat_xyzw} to rotation: {e}")
            # Return original image on error
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
        # Project 3D points to 2D image plane
        # rotation_vector is applied *first*, then translation_vector
        image_points_2d, _ = cv2.projectPoints(axis_points_3d, 
                                               rotation_vector, 
                                               translation_vector, 
                                               CAMERA_MATRIX, 
                                               DIST_COEFFS)
        # Draw the axes lines on the image
        origin = tuple(map(int, image_points_2d[0].ravel()))
        x_axis_end = tuple(map(int, image_points_2d[1].ravel()))
        y_axis_end = tuple(map(int, image_points_2d[2].ravel()))
        z_axis_end = tuple(map(int, image_points_2d[3].ravel()))

        # Draw thicker lines with higher contrast colors
        line_thickness = 6  # Increased from 4 to 6 for better visibility
        
        # Calculate extended axis endpoints (make lines 200% longer)
        def extend_line(origin, end, factor=2.0):
            # Vector from origin to end
            dx = end[0] - origin[0]
            dy = end[1] - origin[1]
            # Extended endpoint
            extended_x = int(origin[0] + dx * factor)
            extended_y = int(origin[1] + dy * factor)
            return (extended_x, extended_y)
        
        # Get extended endpoints
        x_axis_extended = extend_line(origin, x_axis_end)
        y_axis_extended = extend_line(origin, y_axis_end)
        z_axis_extended = extend_line(origin, z_axis_end)
        
        # Draw lines with brighter colors for better visibility
        # Increased color intensity to maximum (255)
        cv2.line(img, origin, x_axis_extended, (0, 0, 255), line_thickness)  # Red (BGR)
        cv2.line(img, origin, y_axis_extended, (0, 255, 0), line_thickness)  # Green
        cv2.line(img, origin, z_axis_extended, (255, 0, 0), line_thickness)  # Blue
        
        # Add larger arrowheads to make axes more distinguishable
        cv2.arrowedLine(img, origin, x_axis_extended, (0, 0, 255), line_thickness, tipLength=0.6)
        cv2.arrowedLine(img, origin, y_axis_extended, (0, 255, 0), line_thickness, tipLength=0.6)
        cv2.arrowedLine(img, origin, z_axis_extended, (255, 0, 0), line_thickness, tipLength=0.6)
        
        # # Add axis labels at the extended endpoints with larger font
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # font_scale = 1.0
        # font_thickness = 2
        # cv2.putText(img, 'X', x_axis_extended, font, font_scale, (0, 0, 255), font_thickness)
        # cv2.putText(img, 'Y', y_axis_extended, font, font_scale, (0, 255, 0), font_thickness)
        # cv2.putText(img, 'Z', z_axis_extended, font, font_scale, (255, 0, 0), font_thickness)

        # Encode the modified image to base64 string
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64

    except Exception as e:
        print(f"Error drawing pose axes: {e}")
        # Optionally return original image base64 encoded? Or None?
        try:
             # Fallback: encode original image if drawing fails mid-way
             img_np = np.frombuffer(image_bytes, np.uint8)
             img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
             if img is not None:
                 if img.shape[1] != IMG_WIDTH or img.shape[0] != IMG_HEIGHT:
                      img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                 _, buffer = cv2.imencode('.png', img)
                 img_base64 = base64.b64encode(buffer).decode('utf-8')
                 return img_base64
        except Exception as fallback_e:
             print(f"Error during fallback image encoding: {fallback_e}")

        return None # Indicate failure if even fallback fails