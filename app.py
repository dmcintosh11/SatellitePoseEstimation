import os
import io
import base64
import traceback
from contextlib import nullcontext

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import rembg # Added for background removal

# Import pose estimation functions
from inference import predict_pose, loaded_model
from visualization import draw_pose_axes

# Import SF3D components
from stable_fast.sf3d.system import SF3D
from stable_fast.sf3d.utils import get_device, remove_background, resize_foreground

app = Flask(__name__, static_folder='static', template_folder='templates')

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB limit

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Model Loading ---

# Load Pose Estimation Model (already done in inference.py)
if loaded_model is None:
    print("Warning: Pose estimation model failed to load.")

# Load Stable Fast 3D Model
sf3d_model = None
sf3d_device = get_device()
sf3d_model_name = "stabilityai/stable-fast-3d" # Or configure as needed
rembg_session = None # Initialize rembg session

try:
    print(f"Attempting to load Stable Fast 3D model ({sf3d_model_name}) on device: {sf3d_device}...")
    # Ensure device is valid before loading
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
       if sf3d_device != "cpu":
           print(f"Warning: Requested device {sf3d_device} not available, falling back to CPU.")
           sf3d_device = "cpu"

    sf3d_model = SF3D.from_pretrained(
        sf3d_model_name,
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    sf3d_model.to(sf3d_device)
    sf3d_model.eval()
    print("Stable Fast 3D model loaded successfully.")
    # Initialize rembg session only if SF3D loaded
    rembg_session = rembg.new_session()
    print("Rembg session initialized.")

except Exception as e:
    print(f"Error loading Stable Fast 3D model: {e}")
    print("Mesh generation will be unavailable.")
    traceback.print_exc()
    sf3d_model = None # Ensure model is None if loading fails

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Handles image upload, predicts pose, generates 3D mesh, draws axes, and returns results."""
    if loaded_model is None:
         return jsonify({'error': 'Pose estimation model not loaded on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request.'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if file and allowed_file(file.filename):
        img_bytes = file.read()
        mesh_glb_base64 = None
        pose_quaternion = None
        pose_translation = None
        visualization_img_base64 = None

        try:
            # 1. Predict Pose
            print("Predicting pose...")
            pose_quaternion, pose_translation = predict_pose(img_bytes)
            if pose_quaternion is None or pose_translation is None:
                return jsonify({'error': 'Pose prediction failed on server.'}), 500
            print("Pose prediction successful.")

            # 2. Prepare Image for SF3D
            print("Preparing image for 3D mesh generation...")
            try:
                input_image = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                # --- Optional: Background Removal & Resizing (like run.py) ---
                if rembg_session:
                    print("Removing background...")
                    processed_image = remove_background(input_image, rembg_session)
                    print("Resizing foreground...")
                    processed_image = resize_foreground(processed_image, 0.85) # Use desired ratio
                else:
                    print("Rembg session not available, skipping background removal.")
                    processed_image = input_image # Use original if rembg failed
                # -------------------------------------------------------------
                print("Image prepared.")
            except Exception as img_e:
                print(f"Error processing input image: {img_e}")
                return jsonify({'error': f'Failed to process input image: {img_e}'}), 400


            # 3. Run Stable Fast 3D (if loaded)
            if sf3d_model:
                print(f"Running Stable Fast 3D model on device {sf3d_device}...")
                try:
                    # Use autocast for mixed precision if on CUDA
                    autocast_context = torch.autocast(device_type=sf3d_device, dtype=torch.bfloat16) if "cuda" in sf3d_device else nullcontext()

                    with torch.no_grad(), autocast_context:
                        # Assuming run_image accepts a list of PIL Images
                        mesh, _ = sf3d_model.run_image(
                            [processed_image], # Pass image in a list
                            bake_resolution=1024, # Configure as needed
                            remesh="none",        # Configure as needed
                            vertex_count=-1       # Configure as needed
                        )

                    # Assuming run_image returns a list of meshes when batch > 1
                    # If batch is 1 (as here), it might return a single mesh object
                    output_mesh = mesh[0] if isinstance(mesh, list) else mesh

                    print("Stable Fast 3D processing completed.")

                    # Export mesh to bytes
                    print("Exporting mesh to GLB format...")
                    glb_buffer = io.BytesIO()
                    output_mesh.export(glb_buffer, file_type='glb', include_normals=True)
                    glb_buffer.seek(0)
                    mesh_glb_base64 = base64.b64encode(glb_buffer.read()).decode('utf-8')
                    print("Encoded mesh to base64.")

                except Exception as sf3d_e:
                    print(f"Error running Stable Fast 3D model: {sf3d_e}")
                    traceback.print_exc()
                    # Don't return error, just proceed without mesh
                    mesh_glb_base64 = None
                    print("Proceeding without 3D mesh due to error.")
            else:
                print("Stable Fast 3D model not loaded, skipping mesh generation.")


            # 4. Draw visualization axes
            print("Generating pose visualization...")
            try:
                visualization_img_base64 = draw_pose_axes(img_bytes, pose_quaternion, pose_translation)
                if not visualization_img_base64:
                    print("Warning: Failed to draw visualization axes.")
                else:
                    print("Visualization generated.")
            except Exception as vis_e:
                 print(f"Error generating visualization: {vis_e}")
                 visualization_img_base64 = None # Proceed without visualization

            # 5. Prepare response
            response_data = {
                'quaternion': [round(x, 5) for x in pose_quaternion] if pose_quaternion else None,
                'translation': [round(x, 5) for x in pose_translation] if pose_translation else None,
                'mesh_glb_base64': mesh_glb_base64, # Will be None if SF3D failed/skipped
                'visualization_img_base64': visualization_img_base64 # May be None if drawing failed
            }
            return jsonify(response_data)

        except Exception as e:
            print(f"Error processing prediction request: {e}")
            traceback.print_exc() # Print full traceback for debugging
            return jsonify({'error': f'An internal server error occurred: {str(e)[:100]}...'}), 500

    else:
        return jsonify({'error': 'File type not allowed.'}), 400

# Serve static files (CSS, JS)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    # Use environment variables for host and port, useful for deployment
    host = os.environ.get('FLASK_HOST', '0.0.0.0')
    port = int(os.environ.get('FLASK_PORT', 4000))
    # Debug should be False in production
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    print(f"Starting Flask server on {host}:{port} with debug={debug}")
    app.run(host=host, port=port, debug=debug)